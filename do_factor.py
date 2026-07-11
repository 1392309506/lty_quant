import subprocess as s
s.run(['git','add','-A'], capture_output=True)
r = s.run(['git','commit','-m','refactor: 因子共线性修复 13->9, VIF大幅降低\n\n- 移除 BB_POS/ATR_20_NORM/HIGH_LOW_RATIO/ULCER_INDEX (VIF>5)\n- 保留 9 个核心因子: MOMO_20/60, MOM_RATIO, RSI_14, VOL_MA_RATIO, VOLATILITY_20, BB_WIDTH, CHAIKIN_MF, MAX_DD_60\n- 更新 FACTOR_NAMES, FEATURE_COLS 39->27, validation.py\n- VIF 测试验证: 仅 MOMO_20 VIF=7.06 略高, 其余均<4\n- 更新全部文档同步'], capture_output=True)
print(r.stdout.decode()[:200])
s.run(['git','push'])