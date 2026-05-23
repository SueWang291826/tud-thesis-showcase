"""Check replan counts in all CSV files."""
from pathlib import Path
import os, sys
os.chdir(r'E:\TUD-Thesis\station\experiment')
sys.path.insert(0, '.')

out_base = Path('outputs/step5_simulation/capacity_sweep')

print('All dynamic summary.csv files:')
for p in sorted(out_base.rglob('summary.csv')):
    txt = p.read_text(encoding='utf-8').strip()
    if not txt:
        continue
    lines = txt.split('\n')
    if len(lines) < 2:
        continue
    h = lines[0].split(',')
    v = lines[1].split(',')
    d = dict(zip(h, v))
    mode = d.get('routing_mode', '')
    if mode == 'dynamic':
        n = d.get('n_agents', '?')
        arr = d.get('arrive_rate', '?')
        rp = d.get('total_replans', '?')
        mtime = p.stat().st_mtime
        import datetime
        mt = datetime.datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
        print(f'  [{mt}] {p.parent.name}/summary.csv  N={n}  arrive={arr}  total_replans={rp}')
