"""
Step 5: Simulation
===================

Run ABM simulation(s) —static and/or dynamic routing.
Loads the pre-built navigation graph from Step 3 gpickle.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.routing import define_semantic_regions, sample_agents
from src.simulation import run_simulation


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir_base = Path(ROOT / cfg["output"]["step_dirs"]["step5"])

    print("=" * 60)
    print("STEP 5 —Simulation")
    print("=" * 60)

    # Load pre-built graph from step3 (skip 50s IFC re-extraction)
    graph_path = ROOT / cfg["output"]["step_dirs"]["step3"] / "navigation_graph.gpickle"
    print(f"  Loading graph from {graph_path} ...")
    with open(graph_path, "rb") as _f:
        G = pickle.load(_f)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Extract geometries (needed for visualisation)
    products = load_preprocessing_products(cfg)
    geometries, all_connectors, _ = extract_all_levels(cfg, products)

    regions = define_semantic_regions(G, cfg)

    sim_cfg = cfg["simulation"]
    agents = sample_agents(
        regions=regions,
        flows=sim_cfg.get("flows", {"ENTRANCE->PLATFORM": 1.0}),
        n_agents=sim_cfg["n_agents"],
        T=sim_cfg["T_s"],
        seed=sim_cfg["seed"],
        walking_speed=sim_cfg["walking_speed_ms"],
        elderly_ratio=sim_cfg.get("elderly_ratio", 0.0),
    )

    results = []

    # --- Scenario A: Static routing ---
    print("\n--- Scenario A: Static Routing ---")
    result_static = run_simulation(G, agents, cfg,
                                    out_dir=out_dir_base / "static",
                                    routing_mode="static",
                                    label="static")
    result_static["label"] = "static"
    results.append(result_static)

    arrive_rate = result_static.get("arrive_rate", 0)
    mean_tt = (sum(result_static["travel_times"]) / len(result_static["travel_times"])
               if result_static["travel_times"] else 0)
    print(f"  Arrival rate: {arrive_rate:.1%}")
    print(f"  Mean travel time: {mean_tt:.1f}s")

    # --- Scenario B: Dynamic routing ---
    print("\n--- Scenario B: Dynamic Routing ---")
    result_dynamic = run_simulation(G, agents, cfg,
                                     out_dir=out_dir_base / "dynamic",
                                     routing_mode="dynamic",
                                     label="dynamic")
    result_dynamic["label"] = "dynamic"
    results.append(result_dynamic)

    arrive_rate_d = result_dynamic.get("arrive_rate", 0)
    mean_tt_d = (sum(result_dynamic["travel_times"]) / len(result_dynamic["travel_times"])
                 if result_dynamic["travel_times"] else 0)
    print(f"  Arrival rate: {arrive_rate_d:.1%}")
    print(f"  Mean travel time: {mean_tt_d:.1f}s")

    # --- Save ---
    out_dir_base.mkdir(parents=True, exist_ok=True)
    from src.utils import dump_json
    for r in results:
        label = r.get("label", "unknown")
        dump_json(out_dir_base / f"result_{label}.json", {
            k: v for k, v in r.items()
            if k not in ("trajectory_frames",)  # frames are large, save separately
        })

    # --- Visualization ---
    print("\n  Generating visualisations ...")
    from src.viz import (
        fig_arrival_curve, fig_queue_over_time,
        fig_comparison_bar, fig_elderly_vs_normal,
        fig_replan_timeline, fig_route_flow_diff,
    )
    from src.evaluation import compute_scenario_metrics

    fig_dir = out_dir_base / "figures"
    fig_arrival_curve(results, fig_dir, cfg)
    fig_queue_over_time(results, fig_dir, cfg)

    # Quick metrics for comparison charts
    metrics_list = [compute_scenario_metrics(r) for r in results]
    fig_comparison_bar(metrics_list, fig_dir, cfg)
    fig_elderly_vs_normal(metrics_list, fig_dir, cfg)

    # Replanning analysis (Step 5 new figures)
    result_static  = next((r for r in results if r.get("label") == "static"),  {})
    result_dynamic = next((r for r in results if r.get("label") == "dynamic"), {})
    fig_replan_timeline(result_dynamic, fig_dir, cfg)
    fig_route_flow_diff(G, result_static, result_dynamic, geometries, fig_dir, cfg)
    print("  6 static figures saved")

    # --- Interactive: animated agent simulation --------------------------
    print("\n  Generating interactive animations ...")
    from src.viz_interactive import (
        fig_interactive_simulation_animation,
        fig_3d_simulation_animation,
        fig_interactive_route_diff,
    )

    elevations = {
        lvl: lc["elevation_m"]
        for lvl, lc in cfg["station"]["levels"].items()
        if lc.get("is_walkable", False)
    }
    html_dir = out_dir_base / "interactive"

    for scenario_label in ("static", "dynamic"):
        traj_file = out_dir_base / scenario_label / "traj_agents.jsonl"
        if not traj_file.exists():
            continue
        # 2-D top-down animation (kept for backward compatibility)
        fig_interactive_simulation_animation(
            traj_path=traj_file,
            geometries=geometries,
            elevations=elevations,
            out_dir=html_dir,
            dt_frame=2.0,
            label=scenario_label,
        )
        # 3-D axonometric animation (rotatable, three floors separated)
        fig_3d_simulation_animation(
            traj_path=traj_file,
            geometries=geometries,
            elevations=elevations,
            all_connectors=all_connectors,
            cfg=cfg,
            out_dir=html_dir,
            G=G,
            dt_frame=2.0,
            label=scenario_label,
        )

    # 3-D interactive route flow difference (dynamic vs static)
    if result_static and result_dynamic:
        fig_interactive_route_diff(
            G=G,
            static_result=result_static,
            dynamic_result=result_dynamic,
            geometries=geometries,
            elevations=elevations,
            all_connectors=all_connectors,
            cfg=cfg,
            out_dir=html_dir,
        )

    print(f"\n[Step 5] Outputs →{out_dir_base}")
    return cfg, G, geometries, all_connectors, regions, results


if __name__ == "__main__":
    main()
