import subprocess as s
r = s.run(['git', 'add', '-A'], capture_output=True)
r = s.run(['git', 'commit', '-m', 'docs: 全面重写使用指南.md — 分类命令、删除Jupyter、补充模拟盘'], capture_output=True)
print(r.stdout.decode()[:200])
r = s.run(['git', 'push'], capture_output=True)
print(r.stdout.decode()[:200])