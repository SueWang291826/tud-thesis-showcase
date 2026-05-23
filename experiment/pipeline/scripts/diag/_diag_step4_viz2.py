"""Minimal debug: test viz functions from saved outputs only (no IFC)."""
import sys, traceback, pickle, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config

cfg = load_config(str(ROOT / "config" / "experiment_config.yaml"))
step4_dir = ROOT / "outputs" / "step4_routing"

print("Loading graph...", flush=True)
with open(ROOT / "outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    G = pickle.load(f)
print(f"Graph: {G.number_of_nodes()} nodes", flush=True)

# Load saved agents from step4 output
agents_path = step4_dir / "agents.json"
regions_path = step4_dir / "semantic_regions.json"
if agents_path.exists():
    with open(agents_path) as f:
        agents = json.load(f)
    print(f"Loaded {len(agents)} agents", flush=True)
else:
    print("No agents.json found!", flush=True)
    sys.exit(1)

if regions_path.exists():
    with open(regions_path) as f:
        regions = json.load(f)
    print(f"Regions: {list(regions.keys())}", flush=True)
else:
    print("No semantic_regions.json found!", flush=True)
    sys.exit(1)

# Fake geometries with empty floors
from shapely.geometry import box
geometries = {
    "F1": {"floor": box(0, 0, 160, 25)},
    "F3": {"floor": box(0, 0, 155, 22)},
    "F4": {"floor": box(0, 0, 158, 26)},
}

fig_dir = step4_dir / "figures"
elev = {lvl: lc["elevation_m"] for lvl, lc in cfg["station"]["levels"].items()
        if lc.get("is_walkable", False)}
html_dir = step4_dir / "interactive"

from src.viz import fig_semantic_regions_map, fig_agent_overview, fig_example_paths
from src.viz_interactive import fig_interactive_agent_flow

for step, fn, args in [
    ("semantic_regions_map", fig_semantic_regions_map, (G, regions, geometries, fig_dir, cfg)),
    ("agent_overview", fig_agent_overview, (agents, fig_dir, cfg)),
    ("example_paths", fig_example_paths, (G, agents, geometries, fig_dir, cfg)),
    ("interactive_agent_flow", fig_interactive_agent_flow,
     (G, regions, agents, geometries, elev, html_dir, cfg)),
]:
    print(f"\n--- {step} ---", flush=True)
    sys.stdout.flush()
    try:
        fn(*args)
        print("  OK", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

print("\nALL DONE", flush=True)
