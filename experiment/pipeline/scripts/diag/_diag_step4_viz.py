"""Debug each step4 viz function individually."""
import sys, traceback, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.routing import define_semantic_regions, sample_agents

cfg = load_config(str(ROOT / "config" / "experiment_config.yaml"))
out_dir = ROOT / "outputs" / "step4_routing"

print("Loading graph...")
with open(ROOT / "outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    G = pickle.load(f)
print("Extracting geometry...")
products = load_preprocessing_products(cfg)
geometries, all_connectors, _ = extract_all_levels(cfg, products)

regions = define_semantic_regions(G, cfg)
sim_cfg = cfg["simulation"]
agents = sample_agents(
    regions=regions, flows=sim_cfg["flows"],
    n_agents=sim_cfg["n_agents"], T=sim_cfg["T_s"], seed=sim_cfg["seed"],
    walking_speed=sim_cfg["walking_speed_ms"],
    elderly_ratio=sim_cfg.get("elderly_ratio", 0.0),
)
print(f"Agents: {len(agents)}, Regions: {list(regions.keys())}")

fig_dir = out_dir / "figures"
elev = {lvl: lc["elevation_m"] for lvl, lc in cfg["station"]["levels"].items()
        if lc.get("is_walkable", False)}
html_dir = out_dir / "interactive"

from src.viz import fig_semantic_regions_map, fig_agent_overview, fig_example_paths
from src.viz_interactive import fig_interactive_agent_flow

for step, fn, args in [
    ("semantic_regions_map", fig_semantic_regions_map, (G, regions, geometries, fig_dir, cfg)),
    ("agent_overview", fig_agent_overview, (agents, fig_dir, cfg)),
    ("example_paths", fig_example_paths, (G, agents, geometries, fig_dir, cfg)),
    ("interactive_agent_flow", fig_interactive_agent_flow,
     (G, regions, agents, geometries, elev, html_dir, cfg)),
]:
    print(f"\n--- {step} ---")
    try:
        fn(*args)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

print("\nALL DONE")
