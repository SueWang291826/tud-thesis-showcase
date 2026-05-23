"""
Step 4: Routing —Per-entrance Bidirectional Route Map
=======================================================

Computes exactly 2 directed shortest paths per entrance (10 paths total):

  —Inbound  (ENTRANCE →nearest PSD):
        entrance node →inbound fare-gate (unpaid→paid) →down escalator →PSD
  —Outbound (same PSD →ENTRANCE):
        PSD →up escalator →outbound fare-gate (paid→unpaid) →entrance node

The directed_weight function in src/routing.py enforces:
  - Fare gates: one-way (inbound gate = unpaid→paid, outbound gate = paid→unpaid)
  - Escalators: one-way per the right-hand rule
      left  pair (F1@x≈4.9):  y≈.9  unit goes UP,   y≈2.5 unit goes DOWN
      right pair (F1@x≈18.5): y≈2.5 unit goes UP,   y≈.9  unit goes DOWN
      F3→F4 single units:       both go UP (outbound); inbound uses stairs

Outputs (all in outputs/step4_routing/):
    figures/entrance_route_map.png           —static matplotlib per-entrance path map
    figures/semantic_regions_map_F1.png      —high-resolution F1 plan view
    figures/semantic_regions_map_F3.png      —high-resolution F3 plan view
    figures/semantic_regions_map_F4.png      —high-resolution F4 plan view
    figures/semantic_regions_map_3d.png      —3D overview of OD regions
  interactive/interactive_entrance_routes.html  —interactive 3D Plotly map
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.routing import (
    define_semantic_regions,
    patch_escalator_directions,
    find_entrance_paths,
    save_routing_outputs,
    sample_agents,
    find_path,
)


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step4"])

    print("=" * 60)
    print("STEP 4 —Routing: per-entrance bidirectional routes")
    print("=" * 60)

    # ---- Load graph from Step 3 ------------------------------------------
    graph_path = ROOT / cfg["output"]["step_dirs"]["step3"] / "navigation_graph.gpickle"
    print(f"  Loading graph from {graph_path} ...")
    with open(graph_path, "rb") as _f:
        G = pickle.load(_f)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ---- Patch escalator directions (right-hand rule) --------------------
    n_patched = patch_escalator_directions(G)
    print(f"  Escalator direction patch: {n_patched} edges updated")

    # ---- Semantic regions ------------------------------------------------
    regions = define_semantic_regions(G, cfg)
    for name, nodes in regions.items():
        print(f"  Region '{name}': {len(nodes)} nodes")

    entrance_nodes = regions.get("ENTRANCE", [])
    platform_nodes = regions.get("PLATFORM", [])

    # ---- Per-entrance path computation -----------------------------------
    print("\n  Computing per-entrance bidirectional routes ...")
    ep_list = find_entrance_paths(G, entrance_nodes, platform_nodes)

    for ep in ep_list:
        ename = ep["entrance_name"].replace("entrance_", "Gate ")
        in_hops  = len(ep["inbound_path"])
        out_hops = len(ep["outbound_path"])
        print(
            f"    {ename} ({ep['level']}): "
            f"inbound {in_hops} nodes / {ep['inbound_cost']:.1f}s  |  "
            f"outbound {out_hops} nodes / {ep['outbound_cost']:.1f}s"
        )

    # ---- Load geometries for visualisation -------------------------------
    products = load_preprocessing_products(cfg)
    geometries, all_connectors, _ = extract_all_levels(cfg, products)

    # ---- Also sample agents (for compatibility with Step 5) --------------
    sim_cfg = cfg["simulation"]
    agents = sample_agents(
        regions=regions,
        flows=sim_cfg.get("flows", {"ENTRANCE->PLATFORM": 0.5, "PLATFORM->EXIT": 0.5}),
        n_agents=sim_cfg["n_agents"],
        T=sim_cfg["T_s"],
        seed=sim_cfg["seed"],
        walking_speed=sim_cfg["walking_speed_ms"],
        elderly_ratio=sim_cfg.get("elderly_ratio", 0.0),
    )
    print(f"\n  Agents sampled: {len(agents)}")
    save_routing_outputs(G, regions, agents, out_dir)

    # ---- Static visualisations -------------------------------------------
    print("\n  Generating static visualisations ...")
    from src.viz import fig_semantic_regions_map, fig_entrance_route_map

    fig_dir = out_dir / "figures"
    fig_semantic_regions_map(G, regions, geometries, fig_dir, cfg)
    fig_entrance_route_map(G, ep_list, geometries, fig_dir, cfg)
    print("  5 static figure files saved")

    # ---- Interactive visualisation ---------------------------------------
    print("\n  Generating interactive visualisation ...")
    from src.viz_interactive import fig_interactive_entrance_routes

    elevations = {
        lvl: lc["elevation_m"]
        for lvl, lc in cfg["station"]["levels"].items()
        if lc.get("is_walkable", False)
    }
    html_dir = out_dir / "interactive"
    p = fig_interactive_entrance_routes(G, ep_list, geometries, elevations, html_dir, cfg)
    print(f"  [html] {p.name}")

    print(f"\n[Step 4] Outputs →{out_dir}")
    print(f"         {len(ep_list)} entrances × 2 paths = {len(ep_list)*2} routes computed")
    return cfg, G, geometries, all_connectors, regions, agents


if __name__ == "__main__":
    main()
