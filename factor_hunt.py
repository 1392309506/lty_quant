#!/usr/bin/env python3
"""
factor_hunt.py — 单因子探索与回测

用法:
  python factor_hunt.py <FACTOR_NAME>   # 测试单个因子
  python factor_hunt.py --list          # 列出可用因子
  python factor_hunt.py --all           # 批量测试所有新因子 (逐个)

输出:
  experiments/<exp_id>/ — 实验目录
  - meta.json: 因子元信息
  - ic_daily.csv: 每日 IC 序列
  - layer_returns.csv: 分层收益
  - summary.json: 综合结果与有效判定

有效因子判定 (factor_validity.md):
  1. |Rank IC| > 0.02
  2. |ICIR| > 0.3
  3. Layer monotonicity: top > bottom
  4. Layer spread Sharpe > 0.3 (年化)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# GBK 编码兼容
import io
if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("factor_hunt")

# ── 项目导入 ──
from src.experiment import generate_experiment_id
from src.data.fetcher import (
    fetch_all_data,
    extract_close_matrix,
    extract_volume_matrix,
)
from src.data.label import compute_forward_returns

# ── 新因子注册表 ──
from src.factors.new_factors import NEW_FACTOR_REGISTRY, NEW_FACTOR_NAMES

# ── 常量 ──
EXPERIMENTS_ROOT = Path(__file__).resolve().parent / "experiments"
EXPERIMENTS_ROOT.mkdir(exist_ok=True)

IC_THRESHOLD = 0.02
ICIR_THRESHOLD = 0.15
LAYER_SPREAD_SHARPE_THRESHOLD = 0.3
FORWARD_PERIOD = 10  # default forward return period

# ── 数据加载 ──


def load_data():
    """加载 OHLCV 数据"""
    logger.info("📥 加载市场数据...")
    df = fetch_all_data(force_refresh=False)
    if df.empty:
        logger.error("❌ 无数据，先运行 data_fetcher.py")
        sys.exit(1)

    closes = extract_close_matrix(df)
    volumes = extract_volume_matrix(df)
    opens = df.xs("Open", axis=1, level=1).sort_index(axis=1).sort_index()
    highs = df.xs("High", axis=1, level=1).sort_index(axis=1).sort_index()
    lows = df.xs("Low", axis=1, level=1).sort_index(axis=1).sort_index()

    logger.info(
        f"   数据: {closes.shape[0]} 天 × {closes.shape[1]} 个标的"
    )

    # 计算远期收益
    forward_ret = compute_forward_returns(closes, periods=[FORWARD_PERIOD])
    fwd_col = f"forward_return_{FORWARD_PERIOD}"

    labels = {}
    for ticker in closes.columns:
        try:
            labels[ticker] = forward_ret[(ticker, fwd_col)]
        except KeyError:
            continue

    label_df = pd.DataFrame(labels, index=closes.index)
    return closes, volumes, opens, highs, lows, label_df

# ── 因子计算 ──


def compute_single_factor(
    name: str,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    opens: pd.DataFrame = None,
) -> pd.DataFrame:
    """计算单个因子 (全部标的)，返回 (Date × ticker) DataFrame"""
    info = NEW_FACTOR_REGISTRY.get(name)
    if info is None:
        raise ValueError(f"未知因子: {name}，可用: {', '.join(NEW_FACTOR_NAMES)}")

    func = info["func"]
    params = info["params"]
    arg_map = info["args"]

    logger.info(f"🔬 计算因子 [{name}]...")

    result = {}
    for ticker in closes.columns:
        kwargs = {}
        for arg_name, src in arg_map.items():
            if src == "close":
                kwargs[arg_name] = closes[ticker].dropna()
            elif src == "volume":
                kwargs[arg_name] = volumes[ticker].dropna()
            elif src == "high":
                kwargs[arg_name] = highs[ticker].dropna()
            elif src == "low":
                kwargs[arg_name] = lows[ticker].dropna()
            elif src == "open":
                kwargs[arg_name] = opens[ticker].dropna() if opens is not None else closes[ticker].dropna()
        kwargs.update(params)
        try:
            series = func(**kwargs)
            result[ticker] = series
        except Exception as e:
            logger.warning(f"   [{ticker}] 计算失败: {e}")
            continue

    df = pd.DataFrame(result, index=closes.index)
    logger.info(f"   因子形状: {df.shape}, 缺失: {df.isna().sum().sum():,}")
    return df

# ── 有效性评估 ──


def evaluate_factor(name: str, factor_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
    """
    综合评估因子。

    Returns:
        dict: {ic_mean, ic_std, icir, ic_series, layer_returns, monotonic, verdict}
    """
    logger.info("📊 评估因子表现...")

    # 对齐日期
    common_dates = factor_df.index.intersection(label_df.index)
    f = factor_df.loc[common_dates]
    l = label_df.loc[common_dates]

    # 每日 Rank IC (Spearman)
    ic_values = []
    for date in f.index:
        f_today = f.loc[date].dropna()
        l_today = l.loc[date].dropna()
        common_tickers = f_today.index.intersection(l_today.index)
        if len(common_tickers) < 5:
            continue
        f_vals = f_today[common_tickers]
        l_vals = l_today[common_tickers]
        if f_vals.nunique() < 2 or l_vals.nunique() < 2:
            continue
        rho, _ = spearmanr(f_vals, l_vals)
        if not np.isnan(rho):
            ic_values.append({"Date": date, "IC": rho})

    ic_df = pd.DataFrame(ic_values)
    if ic_df.empty:
        logger.warning("⚠️  IC 序列为空 (数据不足)")
        return {"valid": False, "reason": "IC 序列为空"}

    ic_mean = ic_df["IC"].mean()
    ic_std = ic_df["IC"].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # IC 正值比例
    hit_rate = (ic_df["IC"] > 0).mean()

    logger.info(
        f"   Rank IC: {ic_mean:.4f} ± {ic_std:.4f}, "
        f"ICIR: {icir:.3f}, HitRate: {hit_rate:.1%}"
    )

    # ── 分层组合分析 ──
    # 每日期按因子值分为 5 组 (quintile)，记录 Q5(high) 和 Q1(low) 的日收益
    layer_returns = []
    daily_spreads = []  # (Date, spread_return)

    for date in f.index:
        f_today = f.loc[date].dropna()
        l_today = l.loc[date].dropna()
        common_tickers = f_today.index.intersection(l_today.index)
        if len(common_tickers) < 10:
            continue
        f_vals = f_today[common_tickers]
        l_vals = l_today[common_tickers]

        # 分为 5 层 (处理重复 bin edges)
        try:
            quintiles = pd.qcut(f_vals, q=5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"], duplicates="drop")
        except ValueError:
            quintiles = pd.qcut(f_vals.rank(method="first"), q=5,
                                labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
        for q in ["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"]:
            mask = quintiles == q
            if mask.sum() > 0:
                layer_returns.append({
                    "Date": date, "Layer": q, "return": l_vals[mask].mean(),
                })

        # 记录当日的 spread 收益 (Q5 - Q1)
        q5_mask = quintiles == "Q5(high)"
        q1_mask = quintiles == "Q1(low)"
        if q5_mask.sum() > 0 and q1_mask.sum() > 0:
            spread_ret = l_vals[q5_mask].mean() - l_vals[q1_mask].mean()
            daily_spreads.append({"Date": date, "spread_return": spread_ret})

    layer_df = pd.DataFrame(layer_returns)

    # 聚合: 每层的平均收益
    layer_avg = layer_df.groupby("Layer")["return"].agg(["mean", "std", "count"])
    layer_avg = layer_avg.sort_index()

    # Spread 时间序列 (Q5 - Q1 日收益)
    spread_series = pd.DataFrame(daily_spreads).set_index("Date")["spread_return"]
    if len(spread_series) > 20:
        spread_mean = float(spread_series.mean())
        spread_std_daily = float(spread_series.std())
        spread_sharpe = (spread_mean / spread_std_daily * np.sqrt(252)) if spread_std_daily > 1e-10 else 0.0

        # 单调性检查: Q5 > Q4 > ... > Q1 (在平均收益上单调递减或递增)
        # 简化: Q5 > Q1 且 Q4 > Q2 且中间未严重断裂
        monotonic_up = (
            layer_avg.loc["Q5(high)", "mean"] > layer_avg.loc["Q4", "mean"]
            > layer_avg.loc["Q3", "mean"]
            > layer_avg.loc["Q2", "mean"]
            > layer_avg.loc["Q1(low)", "mean"]
        ) if all(q in layer_avg.index for q in ["Q5(high)", "Q4", "Q3", "Q2", "Q1(low)"]) else False

        monotonic_down = (
            layer_avg.loc["Q5(high)", "mean"] < layer_avg.loc["Q4", "mean"]
            < layer_avg.loc["Q3", "mean"]
            < layer_avg.loc["Q2", "mean"]
            < layer_avg.loc["Q1(low)", "mean"]
        ) if all(q in layer_avg.index for q in ["Q5(high)", "Q4", "Q3", "Q2", "Q1(low)"]) else False

        monotonic = monotonic_up or monotonic_down
        top_bottom_ok = layer_avg.loc["Q5(high)", "mean"] > layer_avg.loc["Q1(low)", "mean"]

        logger.info(
            f"   Layer Spread (Q5-Q1): {spread_mean:.4f}, "
            f"Sharpe: {spread_sharpe:.2f}, "
            f"Monotonic: {monotonic}"
        )
    else:
        spread_mean = 0.0
        spread_sharpe = 0.0
        monotonic = False
        top_bottom_ok = False
        logger.warning("Cannot compute Layer Spread (Q5/Q1 data missing)")

    # ── 有效性判定 ──
    checks = {
        "abs_ic_ok": bool(abs(ic_mean) > IC_THRESHOLD),
        "abs_icir_ok": bool(abs(icir) > ICIR_THRESHOLD),
        "top_bottom_ok": bool(top_bottom_ok),
        "spread_sharpe_ok": bool(spread_sharpe > LAYER_SPREAD_SHARPE_THRESHOLD),
    }
    valid = bool(all(checks.values()))

    summary = {
        "factor_name": name,
        "category": NEW_FACTOR_REGISTRY[name]["category"],
        "description": NEW_FACTOR_REGISTRY[name]["description"],
        "n_samples": int(len(ic_df)),
        "ic_mean": round(float(ic_mean), 6),
        "ic_std": round(float(ic_std), 6),
        "icir": round(float(icir), 4),
        "hit_rate": round(float(hit_rate), 4),
        "layer_spread_mean": round(float(spread_mean), 6),
        "layer_spread_sharpe": round(float(spread_sharpe), 4),
        "monotonic": bool(monotonic),
        "checks": checks,
        "valid": valid,
    }

    if valid:
        logger.info(f"✅ 因子 [{name}] 有效!")
    else:
        logger.info(f"❌ 因子 [{name}] 未通过有效性检查")

    return summary, ic_df, layer_df


def save_factor_experiment(
    name: str,
    summary: dict,
    ic_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> str:
    """保存因子实验结果"""
    exp_id = generate_experiment_id()
    exp_path = EXPERIMENTS_ROOT / exp_id
    exp_path.mkdir(parents=True, exist_ok=True)

    # meta.json
    meta = {
        "factor_name": name,
        "category": NEW_FACTOR_REGISTRY[name]["category"],
        "description": NEW_FACTOR_REGISTRY[name]["description"],
        "params": NEW_FACTOR_REGISTRY[name]["params"],
        "args": NEW_FACTOR_REGISTRY[name]["args"],
        "exp_id": exp_id,
    }
    with open(exp_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ic_daily.csv
    ic_df.to_csv(exp_path / "ic_daily.csv", index=False)

    # layer_returns.csv
    layer_df.to_csv(exp_path / "layer_returns.csv", index=False)

    # summary.json
    with open(exp_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # factor_values.parquet (可选择: 节省空间)
    factor_df.to_parquet(exp_path / "factor_values.parquet")

    logger.info(f"💾 实验已保存: {exp_path}")
    return str(exp_path)


def prune_experiments(max_keep: int = 10):
    """删除最旧的实验，保留最近 max_keep 个"""
    exp_dirs = sorted(
        [d for d in EXPERIMENTS_ROOT.iterdir() if d.is_dir() and d.name.startswith("2026")],
        key=lambda d: d.name,
    )
    if len(exp_dirs) <= max_keep:
        return
    to_delete = exp_dirs[: len(exp_dirs) - max_keep]
    for d in to_delete:
        import shutil
        shutil.rmtree(d)
        logger.info(f"🗑️  删除旧实验: {d.name}")


def test_single_factor(name: str) -> tuple:
    """测试单个因子，返回 (summary, exp_path)"""
    closes, volumes, opens, highs, lows, labels = load_data()

    factor_df = compute_single_factor(name, closes, volumes, highs, lows, opens)
    summary, ic_df, layer_df = evaluate_factor(name, factor_df, labels)

    exp_path = save_factor_experiment(name, summary, ic_df, layer_df, factor_df)
    prune_experiments(10)
    return summary, exp_path


def list_factors():
    """列出所有可用因子"""
    print(f"\n{'=' * 70}")
    print(f"{'可用新因子':^70}")
    print(f"{'=' * 70}")
    for cat in ["技术面", "成交量", "统计", "横截面"]:
        print(f"\n【{cat}】")
        for name in sorted(NEW_FACTOR_NAMES):
            info = NEW_FACTOR_REGISTRY[name]
            if info["category"] == cat:
                print(f"  {name:<25s} — {info['description']}")


if __name__ == "__main__":
    # Parse --period argument
    period_idx = next((i for i, a in enumerate(sys.argv) if a == "--period"), None)
    if period_idx is not None and period_idx + 1 < len(sys.argv):
        FORWARD_PERIOD = int(sys.argv[period_idx + 1])
        # Remove period args so they don't interfere
        sys.argv = [a for i, a in enumerate(sys.argv) if i not in (period_idx, period_idx + 1)]

    if "--list" in sys.argv:
        list_factors()
        sys.exit(0)

    targets = []
    if "--all" in sys.argv:
        targets = NEW_FACTOR_NAMES
    else:
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                continue
            if arg in NEW_FACTOR_REGISTRY:
                targets.append(arg)
            else:
                logger.error(f"Unknown factor: {arg}, use --list")
                sys.exit(1)

    if not targets:
        print("Usage: python factor_hunt.py <FACTOR_NAME> [FACTOR_NAME ...]")
        print("      python factor_hunt.py --all")
        print("      python factor_hunt.py --list")
        print("      python factor_hunt.py BB_WIDTH --period 5")
        sys.exit(1)

    results = []
    for fname in targets:
        logger.info(f"\n{'=' * 60}")
        logger.info("Testing factor: %s", fname)
        logger.info(f"{'=' * 60}")
        try:
            summary, exp_path = test_single_factor(fname)
            results.append((fname, summary["valid"], summary))
        except Exception as e:
            logger.error(f"Testing [{fname}] failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((fname, False, {"error": str(e)}))

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"{'因子探索汇总':^60}")
    print(f"{'=' * 60}")
    print(f"{'Factor':<25} {'IC':>8} {'ICIR':>8} {'LayerSharpe':>12} {'Status':>6}")
    print("-" * 60)
    for fname, valid, s in results:
        ic = s.get("ic_mean", 0)
        icir = s.get("icir", 0)
        sharpe = s.get("layer_spread_sharpe", 0)
        status = "[OK] " if valid else "[FAIL]"
        print(f"{fname:<25} {ic:>8.4f} {icir:>8.3f} {sharpe:>12.2f} {status:>6}")
    print(f"{'=' * 60}")