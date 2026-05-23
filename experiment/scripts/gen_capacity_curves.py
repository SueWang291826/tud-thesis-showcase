"""Generate capacity_curves.png from all completed summary.csv files."""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config
from src.viz_thesis import fig_capacity_curves

cfg = load_config("config/experiment_config.yaml")
T_s = cfg["simulation"]["T_s"]

sweep_dir = Path("outputs/step5_simulation/capacity_sweep")
results = []

for csv_path in sorted(sweep_dir.rglob("summary.csv")):
    txt = csv_path.read_text(encoding="utf-8").strip()
    if not txt:
        continue
    lines = txt.split("\n")
    if len(lines) < 2:
        continue
    header = lines[0].split(",")
    vals   = lines[1].split(",")
    d = dict(zip(header, vals))

    n = int(float(d.get("n_agents", 0)))
    mode = d.get("routing_mode", "static")
    if n == 0:
        continue

    r = {
        "n_agents": n,
        "label": mode,
        "T_s": T_s,
        "arrive_rate":     float(d.get("arrive_rate", 0)),
        "mean_travel_time": float(d.get("mean_travel_time", 0)),
        "travel_times":    [],  # empty; fig will use mean_travel_time fallback
        "total_replans":   int(float(d.get("total_replans", 0))),
        "replan_events":   [],  # empty; fig will use total_replans fallback
    }
    results.append(r)
    print(
        f"  {mode:8s}  N={n:5d}: arrive={r['arrive_rate']*100:.1f}%"
        f"  mean_tt={r['mean_travel_time']:.1f}s  replans={r['total_replans']}"
    )

print(f"\nTotal data points: {len(results)}")
print("Generating capacity_curves.png ...")

fig_dir = sweep_dir / "figures"
fig_dir.mkdir(exist_ok=True)
fig_capacity_curves(results, fig_dir, cfg=cfg)
print("Done!")
