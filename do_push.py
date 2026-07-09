import subprocess
msg = "fix: v0.5.1 blocking bugs — min_hold 死代码 + Broker 封装 + bid/ask 方向\n\n- 移除 signals.py generate_exit_signals 中从未使用的 min_hold 参数\n- Broker 添加公共 fill_order/close_position 接口，OrderManager 改用公共接口\n- get_symbol_price 添加 side 参数，买单用 ask、卖单用 bid\n- 更新所有调用方（daily_inference.py, backtest/engine.py）"
r = subprocess.run(['git', 'add', '-A'], capture_output=True)
r = subprocess.run(['git', 'commit', '-m', msg], capture_output=True)
print(r.returncode, r.stdout, r.stderr)
r = subprocess.run(['git', 'push'], capture_output=True)
print(r.returncode, r.stdout, r.stderr)