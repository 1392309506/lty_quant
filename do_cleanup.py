import subprocess
r = subprocess.run(['git', 'add', '-A'], capture_output=True)
r = subprocess.run(['git', 'commit', '-m', 'docs: TO_DO_LIST.md 标记 v0.5.1/v0.5.2 已完成\n\n- 删除残留的临时脚本 do_commit2.py'], capture_output=True)
print(r.returncode, r.stdout.decode()[:200], r.stderr.decode()[:200])
r = subprocess.run(['git', 'push'], capture_output=True)
print(r.returncode, r.stdout.decode()[:200], r.stderr.decode()[:200])