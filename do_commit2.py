import subprocess
msg = "feat: v0.5.2 — 模拟盘启动脚本 scripts/run_simulation.py\n\n- 新增 scripts/run_simulation.py — 自动交易循环\n  ├─ daily_inference 信号生成（支持 --no-inference）\n  ├─ Broker(simulate=True) 执行开/平仓\n  ├─ RiskManager 风控检查（杠杆/止损/每日次数）\n  ├─ OrderManager 记录审计日志\n  ├─ 实时行情定价（从缓存 parquet 读取最新价）\n  └─ 状态持久化（simulation_state.json 恢复）\n- Broker 新增 _sim_latest_prices 缓存，解决模拟模式定价问题\n- risk.py 添加浮点比较 epsilon，防止边界误拒"
r = subprocess.run(['git', 'add', '-A'], capture_output=True)
r = subprocess.run(['git', 'commit', '-m', msg], capture_output=True)
print(r.returncode, r.stdout.decode(), r.stderr.decode())
r = subprocess.run(['git', 'push'], capture_output=True)
print(r.returncode, r.stdout.decode(), r.stderr.decode())