"""Dynamic capacity sweep: N=1000, 1500, 2000 with optimized congestion computation."""
import sys, pickle, csv, os
from pathlib import Path

# Ensure working directory = experiment root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
from src.utils import load_config
from src.routing import define_semantic_regions, sample_agents
from src.simulation import run_simulation
from src.viz_thesis import fig_capacity_curves

cfg = load_config("config/experiment_config.yaml")
with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    G = pickle.load(f)

regions = define_semantic_regions(G, cfg)
T_s = cfg["simulation"]["T_s"]
out_base = Path("outputs/step5_simulation/capacity_sweep")

DYNAMIC_NS = [1000, 1500, 2000]
dynamic_results = []

for n in DYNAMIC_NS:
    print(f"[dynamic] N={n} ...", flush=True)
    cfg_run = cfg.copy()
    cfg_run["simulation"] = dict(cfg["simulation"])
    cfg_run["simulation"]["n_agents"] = n
    cfg_run["simulation"]["seed"] = cfg["simulation"]["seed"] + 1000
    # ── Speed-up params for capacity sweep ───────────────────────────────────
    # Recompute congestion every 15 steps (7.5s at dt=0.5s)
    cfg_run["simulation"]["congestion_recompute_every"] = 15
    # Smaller BFS horizon: 6 hops instead of 12 (4× faster BFS)
    cfg_run["simulation"]["congestion_max_hops"] = 6
    # Only replan when blocked for >=6s (reduces Dijkstra calls)
    cfg_run["simulation"]["replan_wait_threshold_s"] = 6.0
    # Cap Dijkstra calls per timestep to 4 (weight-dict baked path = ~20ms/call)
    # 4 replans × 1200 steps × 20ms = ~96s pathfinding per N
    cfg_run["simulation"]["max_replans_per_step"] = 4

    agents = sample_agents(
        regions=regions, flows=cfg_run["simulation"]["flows"],
        n_agents=n, T=T_s, seed=cfg_run["simulation"]["seed"],
        walking_speed=cfg["simulation"]["walking_speed_ms"],
        elderly_ratio=cfg["simulation"].get("elderly_ratio", 0.2),
    )
    scratch = out_base / f"N{n}_dynamic"
    scratch.mkdir(parents=True, exist_ok=True)

    r = run_simulation(G, agents, cfg_run, out_dir=scratch,
                       routing_mode="dynamic", label=f"dynamic_N{n}",
                       write_traj=False)
    r["n_agents"] = n
    r["label"] = "dynamic"
    r["T_s"] = T_s

    arr = r.get("arrive_rate", 0) * 100
    tts = r.get("travel_times", [])
    mtt = sum(tts) / len(tts) if tts else float("nan")
    rp = len(r.get("replan_events", []))
    print(f"  -> arrived={arr:.1f}%  mean_tt={mtt:.1f}s  replans={rp}", flush=True)
    dynamic_results.append(r)

print("Dynamic sweep DONE", flush=True)

# Now load static results + combine and generate capacity_curves.png
print("\nLoading all summary.csv files ...", flush=True)
all_results = []

for csv_path in sorted(out_base.rglob("summary.csv")):
    txt = csv_path.read_text(encoding="utf-8").strip()
    if not txt:
        continue
    lines = txt.split("\n")
    if len(lines) < 2:
        continue
    header = lines[0].split(",")
    vals = lines[1].split(",")
    d = dict(zip(header, vals))
    n_csv = int(float(d.get("n_agents", 0)))
    mode = d.get("routing_mode", "static")
    if n_csv == 0:
        continue
    # Skip dynamic N that we just computed (will use in-memory result)
    if mode == "dynamic" and n_csv in [r["n_agents"] for r in dynamic_results]:
        continue
    r_csv = {
        "n_agents": n_csv,
        "label": mode,
        "T_s": T_s,
        "arrive_rate": float(d.get("arrive_rate", 0)),
        "mean_travel_time": float(d.get("mean_travel_time", 0)),
        "travel_times": [],
        "total_replans": int(float(d.get("total_replans", 0))),
        "replan_events": [],
    }
    all_results.append(r_csv)
    print(f"  CSV: {mode:8s} N={n_csv:5d}  arrive={r_csv['arrive_rate']*100:.1f}%")

# Add in-memory dynamic results (have full travel_times and replan_events lists)
for r in dynamic_results:
    all_results.append(r)
    arr = r.get("arrive_rate", 0) * 100
    tts = r.get("travel_times", [])
    mtt = sum(tts) / len(tts) if tts else float("nan")
    print(f"  MEM: dynamic   N={r['n_agents']:5d}  arrive={arr:.1f}%  mean_tt={mtt:.1f}s")

print(f"\nTotal data points: {len(all_results)}")
print("Generating capacity_curves.png ...", flush=True)
fig_dir = out_base / "figures"
fig_dir.mkdir(exist_ok=True)
fig_capacity_curves(all_results, fig_dir, cfg=cfg)
print("All done!")
