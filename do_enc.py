import subprocess as s
r = s.run(['git', 'add', '-A'], capture_output=True)
r = s.run(['git', 'commit', '-m', 'fix: Windows GBK emoji auto-compat via src/__init__.py'], capture_output=True)
print(r.stdout.decode()[:300])
r = s.run(['git', 'push'], capture_output=True)
print(r.stdout.decode()[:300])