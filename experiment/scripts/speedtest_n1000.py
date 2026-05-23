"""Speed test for N=1000 dynamic with all optimizations."""
import sys, pickle, time
from pathlib import Path; import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent))
import os; os.chdir(Path(__file__).parent.parent)
from src.utils import load_config
from src.routing import define_semantic_regions, sample_agents
from src.simulation import run_simulation

cfg = load_config("config/experiment_config.yaml")
G = pickle.load(open("outputs/step3_graph/navigation_graph.gpickle","rb"))
regions = define_semantic_regions(G, cfg)
T_s = cfg["simulation"]["T_s"]

cfg_fast = cfg.copy()
cfg_fast["simulation"] = dict(cfg["simulation"])
cfg_fast["simulation"]["n_agents"] = 1000
cfg_fast["simulation"]["seed"] = cfg["simulation"]["seed"] + 1000
cfg_fast["simulation"]["congestion_recompute_every"] = 15
cfg_fast["simulation"]["congestion_max_hops"] = 6
cfg_fast["simulation"]["replan_wait_threshold_s"] = 6.0
cfg_fast["simulation"]["max_replans_per_step"] = 4

agents = sample_agents(regions=regions, flows=cfg_fast["simulation"]["flows"],
    n_agents=1000, T=T_s, seed=cfg_fast["simulation"]["seed"],
    walking_speed=cfg["simulation"]["walking_speed_ms"],
    elderly_ratio=cfg["simulation"].get("elderly_ratio", 0.2))

tmp = Path(tempfile.mkdtemp())
print("N=1000 dynamic fast test ...", flush=True)
t0 = time.perf_counter()
r = run_simulation(G, agents, cfg_fast, out_dir=tmp, routing_mode="dynamic", label="test", write_traj=False)
elapsed = time.perf_counter() - t0
tts = r.get("travel_times", [])
mtt = sum(tts)/len(tts) if tts else 0
print(f"Done {elapsed:.0f}s: arrive={r['arrive_rate']*100:.1f}%  mean_tt={mtt:.0f}s  replans={len(r.get('replan_events',[]))}")
