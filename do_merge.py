import subprocess as s
s.run(['git','add','-A'], capture_output=True)
r = s.run(['git','commit','-m','docs: merge 当前记录 进 审查报告，删除旧文件，更新引用'], capture_output=True)
print(r.stdout.decode()[:200])
s.run(['git','push'])