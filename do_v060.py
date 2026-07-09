import subprocess as s
r = s.run(['git','add','-A'], capture_output=True)
r = s.run(['git','commit','-m','docs: v0.6.0 文档同步 — TO_DO_LIST/CHANGELOG/README/CONSTITUTION/版本号'], capture_output=True)
print(r.stdout.decode()[:300])
r = s.run(['git','push'], capture_output=True)
print(r.stdout.decode()[:300])