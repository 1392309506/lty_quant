"""
new_factors.py — 新因子库 (拓展因子池)

每个因子函数接受 close / high / low / volume 等 pd.Series，
返回 pd.Series（索引为日期）。

所有实现遵循 CONSTITUTION.md 约束：
- 只使用 OHLCV 数据，不依赖外部数据源
- 纯函数，无状态
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# 第一梯队: 技术面拓展
# ═══════════════════════════════════════════════════════════════════


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD 线: 快EMA - 慢EMA

    公式: EMA(close, 12) - EMA(close, 26)
    含义: > 0 → 短期动量强于长期; < 0 → 弱
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def macd_signal_diff(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD 与信号线的差值 (柱状图高度)

    公式: MACD - EMA(MACD, 9)
    含义: > 0 → 加速上涨; < 0 → 减速
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    威廉 %R — 超买超卖指标

    公式: (HH_{window} - close) / (HH_{window} - LL_{window}) * -100
    含义: < -80 → 超卖; > -20 → 超买
    """
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    denominator = (hh - ll).replace(0, np.nan)
    return (-100 * (hh - close) / denominator).clip(-100, 0)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    商品通道指数 (Commodity Channel Index)

    公式: (TP - SMA(TP)) / (0.015 * mean(|TP - SMA(TP)|))
    TP = (H + L + C) / 3
    含义: > 100 → 超买; < -100 → 超卖
    """
    tp = (high + low + close) / 3
    sma = tp.rolling(window, min_periods=window).mean()
    mad = (tp - sma).abs().rolling(window, min_periods=window).mean()
    mad = mad.replace(0, np.nan)
    return ((tp - sma) / (0.015 * mad)).clip(-300, 300)


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """
    变动率 (Rate of Change)

    公式: (close_t / close_{t-window} - 1)
    含义: 正 → 上涨动能在; 负 → 下跌动能在
    与 MOMO_XX 类似但计算方法不同 (MOMO 用的是 pct_change)
    用 10 和 20 日比较以区分短期/中期
    """
    return (close / close.shift(window) - 1).clip(-1, 5)


def trix(close: pd.Series, window: int = 15) -> pd.Series:
    """
    三重平滑 EMA (Triple-smoothed EMA)

    公式: triple_ema = EMA(EMA(EMA(close, n), n), n)
    trix = (triple_ema_t / triple_ema_{t-1} - 1) * 100
    含义: 趋势方向与加速度，对噪音不敏感
    """
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change().clip(-0.1, 0.1)


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """
    效率比 (Kaufman Efficiency Ratio)

    公式: |close_t - close_{t-window}| / sum(|close_i - close_{i-1}| for i in window)
    含义: 1.0 → 完美趋势; 0.0 → 完全随机
    """
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window, min_periods=window).sum()
    path_length = path_length.replace(0, np.nan)
    return (net_change / path_length).clip(0, 1)


# ═══════════════════════════════════════════════════════════════════
# 第二梯队: 成交量相关因子
# ═══════════════════════════════════════════════════════════════════


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    资金流向指数 (Money Flow Index)

    公式: TP = (H + L + C)/3
    MF = TP * Volume
    根据 TP 是否高于前一日 TP，分正/负资金流
    MFI = 100 - 100 / (1 + 正MF/负MF)
    含义: > 80 → 超买; < 20 → 超卖
    """
    tp = (high + low + close) / 3
    mf = tp * volume

    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)

    pos_sum = pos_mf.rolling(window, min_periods=window).sum()
    neg_sum = neg_mf.rolling(window, min_periods=window).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi_val = 100 - (100 / (1 + mfr))
    return mfi_val.clip(0, 100)


def vwap_distance(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    价格与 VWAP 的偏离度

    公式: VWAP = sum(TP * Volume) / sum(Volume) over window
    distance = (close - VWAP) / close
    含义: > 0 → 价格在 VWAP 之上 (强势)
    """
    tp = (high + low + close) / 3
    vwap_num = (tp * volume).rolling(window, min_periods=window).sum()
    vwap_den = volume.rolling(window, min_periods=window).sum()
    vwap_den = vwap_den.replace(0, np.nan)
    vwap = vwap_num / vwap_den
    return ((close - vwap) / close).clip(-0.1, 0.1)


def chaikin_mf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    佳庆资金流 (Chaikin Money Flow)

    公式: MFM = [(C - L) - (H - C)] / (H - L)
    CMF = sum(MFM * Volume) / sum(Volume) over window
    含义: > 0 → 买入压力; < 0 → 卖出压力
    """
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    cmf_num = (mfm * volume).rolling(window, min_periods=window).sum()
    cmf_den = volume.rolling(window, min_periods=window).sum()
    cmf_den = cmf_den.replace(0, np.nan)
    return (cmf_num / cmf_den).clip(-1, 1)


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    量价趋势 (Volume Price Trend)

    公式: VPT_t = VPT_{t-1} + volume_t * (close_t - close_{t-1}) / close_{t-1}
    含义: 上升 → 量价配合; 下降 → 背离
    返回 20 日变化率 (标准化)
    """
    daily_ret = close.pct_change()
    vpt = (volume * daily_ret).cumsum()
    vpt_change = vpt.diff(20)
    return vpt_change / vpt_change.abs().rolling(20).mean().replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════
# 第三梯队: 统计与高阶因子
# ═══════════════════════════════════════════════════════════════════


def skew_20(close: pd.Series, window: int = 20) -> pd.Series:
    """20 日收益率偏度 — 衡量尾部方向"""
    ret = close.pct_change()
    return ret.rolling(window, min_periods=window).skew().clip(-3, 3)


def kurt_20(close: pd.Series, window: int = 20) -> pd.Series:
    """20 日收益率峰度 (超额) — 衡量尾部厚度"""
    ret = close.pct_change()
    return ret.rolling(window, min_periods=window).kurt().clip(-1, 10)


def serial_corr(close: pd.Series, window: int = 5, lag: int = 1) -> pd.Series:
    """
    收益率自相关

    公式: rolling corr(ret_t, ret_{t-lag})
    含义: > 0 → 趋势延续; < 0 → 反转
    """
    ret = close.pct_change()
    return ret.rolling(window, min_periods=window).corr(ret.shift(lag)).clip(-1, 1)


def bb_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    布林带宽度归一化

    公式: (bb_upper - bb_lower) / bb_middle
    含义: 宽 → 高波动/突破; 窄 → 低波动/盘整 (压缩)
    """
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return (2 * num_std * std / ma).clip(0, 1)


def ulcer_index(close: pd.Series, window: int = 14) -> pd.Series:
    """
    溃疡指数 (Ulcer Index) — 下行风险度量

    公式: UI = sqrt(mean(((close - rolling_max) / rolling_max)^2))
    含义: 高 → 下行风险大 (负向因子)
    """
    rolling_max = close.rolling(window, min_periods=window).max()
    pct_dd = ((close - rolling_max) / rolling_max) ** 2
    return np.sqrt(pct_dd.rolling(window, min_periods=window).mean()).clip(0, 0.3)


def max_dd_60(close: pd.Series, window: int = 60) -> pd.Series:
    """
    60 日最大回撤

    公式: min((close - rolling_max) / rolling_max)
    含义: 近期最大回撤深度 (负值)
    """
    rolling_max = close.rolling(window, min_periods=window).max()
    dd = (close - rolling_max) / rolling_max
    return dd.rolling(window, min_periods=window).min().clip(-1, 0)


# ═══════════════════════════════════════════════════════════════════
# 第四梯队: 横截面因子 (在 prepare_training_data 中已有 rank/zscore)
# 这里添加上一交易日反转
# ═══════════════════════════════════════════════════════════════════


def price_reversal_1(close: pd.Series) -> pd.Series:
    """
    隔夜/昨日反转

    公式: -close.pct_change(1)
    含义: 信号方向取反。负值 → 昨日涨 → 今日可能跌 (均值回归)
    如果直接使用，IC 应为负值；取负号后 IC 应为正值
    """
    # 这里不取负号，交给使用者决定是否取反
    return close.pct_change(1)


# ═══════════════════════════════════════════════════════════════════
# 第五梯队: 进阶技术因子
# ═══════════════════════════════════════════════════════════════════


def price_to_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """
    价格与均线比: close / SMA(close, N)

    公式: close / rolling_mean(close, N)
    含义: > 1 → 价格在均线上方; < 1 → 下方
    """
    sma = close.rolling(window, min_periods=window).mean()
    return (close / sma).clip(0.5, 2.0)


def high_low_ratio(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    归一化振幅比: rolling_mean(H-L) / close

    公式: mean(H - L over window) / close
    含义: 波动活性因子 — 高值意味着近期活跃
    """
    hl_range = (high - low).rolling(window, min_periods=window).mean()
    return (hl_range / close).clip(0, 0.3)


def overnight_gap(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """
    隔夜跳空: (open_t - close_{t-1}) / close_{t-1}

    公式: open_t / prev_close - 1
    含义: > 0 → 正向跳空(利多); < 0 → 负向跳空(利空)
    跳空后的延续/反转效应是经典异象
    """
    prev_close = close.shift(1)
    return ((open_price - prev_close) / prev_close).clip(-0.1, 0.1)


def beta_relative(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """
    与基准的相关性(近似 Beta): 60 日滚动 Beta 相对于 SPY

    公式: cov(ret_stock, ret_spy) / var(ret_spy)
    含义: 高 beta → 高系统性风险暴露
    """
    ret_stock = close.pct_change()
    ret_bmk = benchmark.pct_change()
    cov = ret_stock.rolling(window, min_periods=window).cov(ret_bmk)
    var_bmk = ret_bmk.rolling(window, min_periods=window).var()
    var_bmk = var_bmk.replace(0, np.nan)
    return (cov / var_bmk).clip(-2, 5)


def volume_shock(volume: pd.Series, window: int = 5, baseline_window: int = 60) -> pd.Series:
    """
    成交量冲击: 5 日均量 / 60 日均量

    公式: avg(volume_{t-5:t}) / avg(volume_{t-60:t})
    含义: > 1.5 → 局部放量; < 0.5 → 局部缩量
    与 vol_ma_ratio 的区别: 用短/长均量比，更敏感
    """
    short_ma = volume.rolling(window, min_periods=window).mean()
    long_ma = volume.rolling(baseline_window, min_periods=baseline_window).mean()
    long_ma = long_ma.replace(0, np.nan)
    return (short_ma / long_ma).clip(0, 10)


def price_acceleration(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """
    价格加速度: 短期动量 - 长期动量

    公式: momo(fast) - momo(slow)
    含义: > 0 → 加速上涨; < 0 → 减速/反转
    与 MOM_RATIO 不同: 加法而非除法，避免分母问题
    """
    return (close.pct_change(fast) - close.pct_change(slow)).clip(-0.5, 0.5)


def irs_relative_strength(close: pd.Series, benchmark: pd.Series, window: int = 20) -> pd.Series:
    """
    相对强弱: stock_return / benchmark_return over window

    公式: (close_t / close_{t-window}) / (bmk_t / bmk_{t-window})
    含义: > 1 → 跑赢基准; < 1 → 跑输
    """
    stock_ret = close / close.shift(window)
    bmk_ret = benchmark / benchmark.shift(window)
    bmk_ret = bmk_ret.replace(0, np.nan)
    return (stock_ret / bmk_ret).clip(0, 5)


def price_position_in_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    价格在 N 日区间中的位置: (close - low_N) / (high_N - low_N)

    公式: (close - rolling_min_low) / (rolling_max_high - rolling_min_low)
    含义: 0 → 在区间底部; 1 → 在区间顶部
    与 BB_POS 的区别: 用实际高低点而非标准差
    """
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    denominator = (hh - ll).replace(0, np.nan)
    return ((close - ll) / denominator).clip(0, 1)


def range_expansion(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    """
    价格范围扩张率: 当前范围 / 过去 N 天平均范围 - 1

    公式: (H-L)_t / rolling_mean(H-L, window)_t - 1
    含义: > 0 → 波动扩张; < 0 → 波动收缩
    与 BB_WIDTH 的区别: 单日而非 N 日平均
    """
    curr_range = high - low
    avg_range = curr_range.rolling(window, min_periods=window).mean()
    avg_range = avg_range.replace(0, np.nan)
    return (curr_range / avg_range - 1).clip(-1, 10)


def rolling_sharpe(close: pd.Series, window: int = 60) -> pd.Series:
    """
    滚动夏普比: 滚动收益均值 / 滚动标准差

    公式: mean(ret, window) / std(ret, window) * sqrt(252)
    含义: 高 → 近期风险调整收益好 (动量延续)
    Edge: 当 std → 0 时返回 0
    """
    ret = close.pct_change()
    mean_ret = ret.rolling(window, min_periods=window).mean()
    std_ret = ret.rolling(window, min_periods=window).std()
    std_ret = std_ret.replace(0, np.nan)
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe.clip(-5, 5)


def cross_sectional_momentum(close: pd.Series, window: int = 60) -> pd.Series:
    """
    横截面动量排名代理: N 日收益率

    公式: close_t / close_{t-window} - 1
    含义: 与 MOMO_60 计算相同，但在这里作为单独的横截面特征使用
    在 ML 模型中，排名版本 (MOMO_60_rank) 已经被 add_cross_sectional_features 生成
    这个因子是原始版本，用于直接测试
    """
    return close.pct_change(periods=window).clip(-1, 5)


def trend_intensity(close: pd.Series, window: int = 20) -> pd.Series:
    """
    趋势强度: 价格移动方向的一致性与幅度

    公式: mean(sign(diff(close))) * |mean(diff(close))| / std(diff(close))
    含义: 高正值 → 强上涨趋势; 高负值 → 强下跌趋势; 接近 0 → 盘整
    这是 ADX 的简化版本
    """
    ret = close.diff()
    direction = np.sign(ret).rolling(window, min_periods=window).mean()
    magnitude = ret.rolling(window, min_periods=window).mean().abs()
    vol = ret.rolling(window, min_periods=window).std()
    vol = vol.replace(0, np.nan)
    intensity = direction * magnitude / vol
    return intensity.clip(-1, 1) * 100


def streak_length(close: pd.Series, window: int = 20) -> pd.Series:
    """
    连涨连跌天数: 过去 N 天内连续同向变动的最大天数

    公式: 计算连续正/负收益的最大天数
    含义: 连涨天数长 → 超买风险; 连跌天数长 → 超卖反弹
    """
    ret = close.diff()
    pos = (ret > 0).astype(int)
    neg = (ret < 0).astype(int)
    # 滚动最大连涨/连跌天数
    pos_streak = pos * (pos.groupby((pos != pos.shift()).cumsum()).cumsum())
    neg_streak = neg * (neg.groupby((neg != neg.shift()).cumsum()).cumsum())
    # 取最大值：正数=连涨，负数=连跌
    max_pos = pos_streak.rolling(window, min_periods=1).max()
    max_neg = neg_streak.rolling(window, min_periods=1).max()
    return max_pos.where(max_pos >= max_neg, -max_neg).clip(-20, 20)


def volume_conviction(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    成交量确认: 带成交量的价格变动

    公式: sign(ret) * volume / avg_volume
    含义: 高正值 → 放量上涨 (强势); 高负值 → 放量下跌 (弱势)
    """
    ret = close.pct_change()
    avg_vol = volume.rolling(window, min_periods=window).mean()
    avg_vol = avg_vol.replace(0, np.nan)
    conviction = np.sign(ret) * (volume / avg_vol)
    return conviction.clip(-10, 10)


# ═══════════════════════════════════════════════════════════════════
# 注册表 — 所有新因子的元信息
# ═══════════════════════════════════════════════════════════════════

NEW_FACTOR_REGISTRY = {
    "MACD": {
        "func": macd,
        "args": {"close": "close"},
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "category": "技术面",
        "description": "MACD 线: 快EMA - 慢EMA",
    },
    "MACD_SIGNAL_DIFF": {
        "func": macd_signal_diff,
        "args": {"close": "close"},
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "category": "技术面",
        "description": "MACD 柱状图: MACD - 信号线",
    },
    "WILLIAMS_R": {
        "func": williams_r,
        "args": {"high": "high", "low": "low", "close": "close"},
        "params": {"window": 14},
        "category": "技术面",
        "description": "威廉 %R: 超买超卖",
    },
    "CCI_20": {
        "func": cci,
        "args": {"high": "high", "low": "low", "close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "商品通道指数",
    },
    "ROC_10": {
        "func": roc,
        "args": {"close": "close"},
        "params": {"window": 10},
        "category": "技术面",
        "description": "10日变动率",
    },
    "ROC_20": {
        "func": roc,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "20日变动率",
    },
    "TRIX": {
        "func": trix,
        "args": {"close": "close"},
        "params": {"window": 15},
        "category": "技术面",
        "description": "三重平滑EMA",
    },
    "EFFICIENCY_RATIO": {
        "func": efficiency_ratio,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "Kaufman效率比",
    },
    "MFI_14": {
        "func": mfi,
        "args": {"high": "high", "low": "low", "close": "close", "volume": "volume"},
        "params": {"window": 14},
        "category": "成交量",
        "description": "资金流向指数",
    },
    "VWAP_DISTANCE": {
        "func": vwap_distance,
        "args": {"close": "close", "high": "high", "low": "low", "volume": "volume"},
        "params": {"window": 20},
        "category": "成交量",
        "description": "价格偏离VWAP",
    },
    "CHAIKIN_MF": {
        "func": chaikin_mf,
        "args": {"high": "high", "low": "low", "close": "close", "volume": "volume"},
        "params": {"window": 20},
        "category": "成交量",
        "description": "佳庆资金流",
    },
    "VOLUME_PRICE_TREND": {
        "func": volume_price_trend,
        "args": {"close": "close", "volume": "volume"},
        "params": {},
        "category": "成交量",
        "description": "量价趋势",
    },
    "SKEW_20": {
        "func": skew_20,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "统计",
        "description": "20日收益率偏度",
    },
    "KURT_20": {
        "func": kurt_20,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "统计",
        "description": "20日收益率超额峰度",
    },
    "SERIAL_CORR_5": {
        "func": serial_corr,
        "args": {"close": "close"},
        "params": {"window": 5, "lag": 1},
        "category": "统计",
        "description": "5日收益率自相关",
    },
    "SERIAL_CORR_10": {
        "func": serial_corr,
        "args": {"close": "close"},
        "params": {"window": 10, "lag": 1},
        "category": "统计",
        "description": "10日收益率自相关",
    },
    "BB_WIDTH": {
        "func": bb_width,
        "args": {"close": "close"},
        "params": {"window": 20, "num_std": 2.0},
        "category": "统计",
        "description": "布林带宽度归一化",
    },
    "ULCER_INDEX": {
        "func": ulcer_index,
        "args": {"close": "close"},
        "params": {"window": 14},
        "category": "统计",
        "description": "溃疡指数(下行风险)",
    },
    "MAX_DD_60": {
        "func": max_dd_60,
        "args": {"close": "close"},
        "params": {"window": 60},
        "category": "统计",
        "description": "60日最大回撤",
    },
    "PRICE_REVERSAL_1": {
        "func": price_reversal_1,
        "args": {"close": "close"},
        "params": {},
        "category": "横截面",
        "description": "昨日收益率(取负号后为反转因子)",
    },
    # ── 第五梯队 ──
    "PRICE_TO_SMA_20": {
        "func": price_to_sma,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "价格/20日均线",
    },
    "PRICE_TO_SMA_50": {
        "func": price_to_sma,
        "args": {"close": "close"},
        "params": {"window": 50},
        "category": "技术面",
        "description": "价格/50日均线",
    },
    "HIGH_LOW_RATIO": {
        "func": high_low_ratio,
        "args": {"high": "high", "low": "low", "close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "归一化振幅比",
    },
    "OVERNIGHT_GAP": {
        "func": overnight_gap,
        "args": {"open_price": "open", "close": "close"},
        "params": {},
        "category": "技术面",
        "description": "隔夜跳空幅度",
    },
    "VOLUME_SHOCK": {
        "func": volume_shock,
        "args": {"volume": "volume"},
        "params": {"window": 5, "baseline_window": 60},
        "category": "成交量",
        "description": "成交量冲击(5/60日均量比)",
    },
    "PRICE_ACCELERATION": {
        "func": price_acceleration,
        "args": {"close": "close"},
        "params": {"fast": 10, "slow": 30},
        "category": "技术面",
        "description": "价格加速度(10日-30日动量差)",
    },
    # ── 第六梯队: 价格位置与范围 ──
    "PRICE_IN_RANGE_20": {
        "func": price_position_in_range,
        "args": {"high": "high", "low": "low", "close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "价格在20日区间中的位置",
    },
    "RANGE_EXPANSION": {
        "func": range_expansion,
        "args": {"high": "high", "low": "low", "close": "close"},
        "params": {"window": 10},
        "category": "技术面",
        "description": "价格范围扩张率",
    },
    # ── 第七梯队: 风险调整后动量 ──
    "ROLLING_SHARPE_60": {
        "func": rolling_sharpe,
        "args": {"close": "close"},
        "params": {"window": 60},
        "category": "统计",
        "description": "60日滚动夏普比",
    },
    "CROSS_MOMENTUM_60": {
        "func": cross_sectional_momentum,
        "args": {"close": "close"},
        "params": {"window": 60},
        "category": "技术面",
        "description": "60日收益(横截面动量)",
    },
    "TREND_INTENSITY": {
        "func": trend_intensity,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "趋势强度(简化ADX)",
    },
    "STREAK_LENGTH": {
        "func": streak_length,
        "args": {"close": "close"},
        "params": {"window": 20},
        "category": "技术面",
        "description": "连涨连跌天数",
    },
    "VOLUME_CONVICTION": {
        "func": volume_conviction,
        "args": {"close": "close", "volume": "volume"},
        "params": {"window": 20},
        "category": "成交量",
        "description": "成交量确认信号",
    },
}

NEW_FACTOR_NAMES = sorted(NEW_FACTOR_REGISTRY.keys())