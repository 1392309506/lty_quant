"""
回测结果可视化与报告

生成 equity curve、drawdown、pred vs actual 等图表。

v2 新增:
  - plot_equity_curve_comparison: 含基准对比、百分比纵轴、更直观
  - plot_drawdown_v2: 更清晰的回撤图
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["axes.unicode_minus"] = False

# 中文字体配置: 尝试多个 fallback
_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
for _f in _CJK_FONTS:
    try:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams.get("font.sans-serif", [])
        # 验证字体是否可用
        from matplotlib.font_manager import findfont, FontProperties
        findfont(FontProperties(family=_f))
        break
    except Exception:
        continue

logger = logging.getLogger(__name__)

# ── 配色方案 ──
COLOR_STRATEGY = "#1A73E8"      # 策略蓝
COLOR_BENCHMARK = "#999999"     # 基准灰
COLOR_DRAWDOWN = "#E53935"      # 回撤红
COLOR_POSITIVE = "#43A047"       # 正收益绿
COLOR_NEGATIVE = "#E53935"       # 负收益红
COLOR_FILL = "1A73E8"


def print_backtest_summary(metrics: dict):
    """打印格式化的回测指标摘要。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Backtest Summary")
    logger.info("=" * 60)

    rows = [
        ("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"),
        ("Ann. Return", f"{metrics.get('annualized_return', 0)*100:.2f}%"),
        ("Ann. Volatility", f"{metrics.get('annualized_volatility', 0)*100:.2f}%"),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%"),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
        ("Num Trades", f"{int(metrics.get('num_trades', 0))}"),
        ("Total Fees", f"${metrics.get('total_fees_paid', 0):.2f}"),
    ]

    for label, value in rows:
        logger.info(f"  {label:<16} {value}")

    logger.info("")


def plot_equity_curve_comparison(
    strategy_returns: pd.Series,
    save_path: str,
    spy_close: Optional[pd.Series] = None,
    metrics: Optional[dict] = None,
):
    """
    绘制含 SPY 基准对比的直观净值曲线图。

    设计面向非金融背景用户:
    - 纵轴: 百分比收益（从 0% 开始，一目了然）
    - 横轴: 时间
    - 两条线: 策略收益 vs 持有 SPY 不动
    - 红色半透明区域标注回撤
    - 关键指标在右侧图例中显示
    - 绿箭头 = 买入点, 红箭头 = 卖出点
    """
    if strategy_returns is None or len(strategy_returns) == 0:
        logger.warning("No returns data, skipping equity curve")
        return

    # 计算策略累计收益（百分比）
    strat_equity = (1 + strategy_returns.fillna(0)).cumprod()
    strat_pct = (strat_equity - 1) * 100  # 转为百分比

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ═══════════════════════════════════════════════════════
    # 上半部分: 收益曲线
    # ═══════════════════════════════════════════════════════

    # 策略曲线 + 填充
    ax1.plot(
        strat_pct.index, strat_pct.values,
        linewidth=2.0, color=COLOR_STRATEGY,
        label="策略收益",
        zorder=3,
    )
    ax1.fill_between(
        strat_pct.index, 0, strat_pct.values,
        alpha=0.10, color=COLOR_STRATEGY,
        zorder=1,
    )

    # SPY 基准
    if spy_close is not None:
        spy_aligned = spy_close.reindex(strat_pct.index).ffill()
        if len(spy_aligned.dropna()) > 0:
            spy_pct = (spy_aligned / spy_aligned.iloc[0] - 1) * 100
            ax1.plot(
                spy_pct.index, spy_pct.values,
                linewidth=1.2, color=COLOR_BENCHMARK, linestyle="--",
                label="持有 SPY 不动",
                zorder=2,
            )

    # 零线
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # 区间标注: 绿色区域 = 正收益, 红色区域 = 负收益
    y_min, y_max = strat_pct.min(), strat_pct.max()
    y_range = y_max - y_min
    y_pad = y_range * 0.15 if y_range > 0 else 10
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    # 标注最终收益
    final_ret = strat_pct.iloc[-1]
    ax1.axhline(y=final_ret, color=COLOR_STRATEGY, linewidth=0.8, alpha=0.4, linestyle=":")
    ax1.annotate(
        f"{final_ret:+.1f}%",
        xy=(strat_pct.index[-1], final_ret),
        xytext=(10, 0), textcoords="offset points",
        fontsize=11, color=COLOR_STRATEGY, fontweight="bold",
    )

    # 基准最终收益
    if spy_close is not None and "spy_pct" in locals():
        spy_final = spy_pct.iloc[-1]
        ax1.annotate(
            f"{spy_final:+.1f}%",
            xy=(spy_pct.index[-1], spy_final),
            xytext=(10, 0), textcoords="offset points",
            fontsize=10, color=COLOR_BENCHMARK,
        )

    # 设置
    ax1.set_ylabel("累计收益 (%)", fontsize=12)
    ax1.set_title("策略表现 vs 基准", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.tick_params(axis="x", rotation=0)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 关键指标文本框（右下角）
    if metrics:
        stats_text = (
            f"总收益: {metrics.get('total_return_pct', 0):.1f}%\n"
            f"年化收益: {metrics.get('annualized_return', 0)*100:.1f}%\n"
            f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"最大回撤: {metrics.get('max_drawdown_pct', 0):.1f}%\n"
            f"胜率: {metrics.get('win_rate', 0)*100:.0f}%\n"
            f"交易次数: {int(metrics.get('num_trades', 0))}"
        )
        ax1.text(
            0.97, 0.03, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.85, edgecolor="#DDD"),
        )

    # ═══════════════════════════════════════════════════════
    # 下半部分: 回撤图
    # ═══════════════════════════════════════════════════════
    cummax = strat_equity.cummax()
    drawdown = (strat_equity / cummax - 1) * 100

    ax2.fill_between(
        drawdown.index, 0, drawdown.values,
        color=COLOR_DRAWDOWN, alpha=0.4,
    )
    ax2.plot(
        drawdown.index, drawdown.values,
        color=COLOR_DRAWDOWN, linewidth=0.8, alpha=0.8,
    )

    max_dd = drawdown.min()
    if max_dd < 0:
        ax2.annotate(
            f"最大回撤 {max_dd:.1f}%",
            xy=(drawdown.idxmin(), max_dd),
            xytext=(0, -18), textcoords="offset points",
            fontsize=9, color=COLOR_DRAWDOWN,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=COLOR_DRAWDOWN, alpha=0.6),
        )

    ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("回撤 (%)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Equity curve (v2) saved: {save_path}")


def plot_equity_curve(
    returns: pd.Series,
    save_path: str,
    title: str = "Equity Curve",
):
    """(保留的旧版) 绘制并保存 equity curve 图。"""
    if returns is None or len(returns) == 0:
        logger.warning("No returns data, skipping equity curve")
        return

    equity = (1 + returns.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, linewidth=1.5, color="#2196F3")
    ax.fill_between(equity.index, 1, equity.values, alpha=0.15, color="#2196F3")
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    final_ret = equity.iloc[-1] - 1
    ax.text(
        0.02, 0.95, f"Total Return: {final_ret:+.2%}",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Equity curve saved: {save_path}")


def plot_drawdown(
    returns: pd.Series,
    save_path: str,
):
    """(保留的旧版) 绘制并保存回撤图。"""
    if returns is None or len(returns) < 2:
        return

    equity = (1 + returns.fillna(0)).cumprod()
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, 0, drawdown.values, color="#F44336", alpha=0.5)
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    max_dd = drawdown.min()
    ax.text(
        0.02, 0.05, f"Max Drawdown: {max_dd:.1f}%",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="salmon", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Drawdown chart saved: {save_path}")


def plot_predictions_vs_actuals(
    pred_df: pd.DataFrame,
    save_path: str,
):
    """预测 vs 实际收益散点图。"""
    if "pred" not in pred_df.columns or "actual" not in pred_df.columns:
        logger.warning("Missing pred/actual columns")
        return

    pred = pred_df["pred"].dropna()
    actual = pred_df["actual"].dropna()
    common = pred.index.intersection(actual.index)
    if len(common) == 0:
        return
    pred = pred.loc[common]
    actual = actual.loc[common]

    if len(pred) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pred, actual, alpha=0.3, s=5, color="#2196F3")
    lim = max(pred.max(), actual.max(), -pred.min(), -actual.min())
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, alpha=0.5)
    ax.set_title("Predictions vs Actuals", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Return")
    ax.set_ylabel("Actual Return")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    corr = pred.corr(actual)
    ax.text(
        0.05, 0.95, f"Pearson r = {corr:.3f}",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Pred vs actual chart saved: {save_path}")


def save_backtest_report(
    bt_dir: str,
    bt_id: str,
    metrics: dict,
    config: dict,
    result: dict,
) -> str:
    """保存完整的回测报告（使用 v2 图表）。"""
    report_path = Path(bt_dir) / bt_id
    report_path.mkdir(parents=True, exist_ok=True)

    with open(report_path / "bt_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
             for k, v in config.items()},
            f, indent=2,
        )

    metrics_clean = {}
    for k, v in metrics.items():
        if k == "returns":
            continue
        if isinstance(v, (np.floating, float)):
            metrics_clean[k] = round(float(v), 6)
        elif isinstance(v, (np.integer, int)):
            metrics_clean[k] = int(v)
        else:
            metrics_clean[k] = str(v)

    with open(report_path / "bt_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, indent=2)

    returns = metrics.get("returns")
    if returns is not None and len(returns) > 0:
        # 新版图表: 含基准对比
        spy_close = result.get("spy_close")
        plot_equity_curve_comparison(
            returns,
            str(report_path / "equity_curve.png"),
            spy_close=spy_close,
            metrics=metrics_clean,
        )
        # 保留旧版回撤图作为补充
        plot_drawdown(returns, str(report_path / "drawdown.png"))

    predictions = result.get("predictions")
    if predictions is not None:
        plot_predictions_vs_actuals(
            predictions, str(report_path / "pred_vs_actual.png")
        )

    logger.info(f"Backtest report saved to: {report_path}")
    return str(report_path)