"""
Run Full Pipeline
==================

Orchestrates all 7 steps end-to-end.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config path/to/config.yaml
    python scripts/run_pipeline.py --steps 0,1,2,3   # run only selected steps
    python scripts/run_pipeline.py --skip-viz          # skip visualization
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, dump_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "config" / "experiment_config.yaml"),
                        help="Path to experiment config YAML")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated step numbers to run (e.g. '0,1,2,3'). "
                             "Default: all steps.")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualisation generation")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(","))
    else:
        steps_to_run = {0, 1, 2, 3, 4, 5, 6}

    timings = {}
    pipeline_start = time.time()

    print("=" * 60)
    print("=  Multi-Level Indoor Navigation Experiment Pipeline  =")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Steps : {sorted(steps_to_run)}")
    print()

    # ----------------------------------------------------------------
    # Step 0: Load preprocessing products
    # ----------------------------------------------------------------
    if 0 in steps_to_run:
        t0 = time.time()
        from src.data_loader import (
            load_preprocessing_products,
            filter_obstacles_for_navigation,
            filter_connectors_for_navigation,
            save_step0_outputs,
        )

        print("=" * 60)
        print("STEP 0 ->Load Preprocessing Products")
        print("=" * 60)

        products = load_preprocessing_products(cfg)
        nav_obs = filter_obstacles_for_navigation(products["obstacle_df"])
        nav_conn = filter_connectors_for_navigation(products["connector_df"])
        out0 = Path(ROOT / cfg["output"]["step_dirs"]["step0"])
        save_step0_outputs(products, nav_obs, nav_conn, out0)

        print(f"  retained={len(products['retained_df'])}, "
              f"nav_obs={len(nav_obs)}, nav_conn={len(nav_conn)}")

        if not args.skip_viz:
            from src.viz import fig_data_overview
            fig_data_overview(products, nav_obs, nav_conn, out0 / "figures", cfg)

        timings["step0"] = time.time() - t0
    else:
        # Must still load data for downstream steps
        from src.data_loader import (
            load_preprocessing_products,
            filter_obstacles_for_navigation,
            filter_connectors_for_navigation,
        )
        products = load_preprocessing_products(cfg)
        nav_obs = filter_obstacles_for_navigation(products["obstacle_df"])
        nav_conn = filter_connectors_for_navigation(products["connector_df"])

    # ----------------------------------------------------------------
    # Step 1: Geometry extraction
    # ----------------------------------------------------------------
    if 1 in steps_to_run:
        t0 = time.time()
        from src.geometry_extractor import extract_all_levels, save_geometry_outputs

        print("\n" + "=" * 60)
        print("STEP 1 ->Geometry Extraction")
        print("=" * 60)

        geometries, all_connectors, control_points = extract_all_levels(cfg, products)
        out1 = Path(ROOT / cfg["output"]["step_dirs"]["step1"])
        save_geometry_outputs(geometries, all_connectors, out1, control_points)

        for lvl, g in sorted(geometries.items()):
            area = g["walkable"].area if g["walkable"] and not g["walkable"].is_empty else 0
            print(f"  {lvl}: walkable={area:.0f}m\u00b2")

        if not args.skip_viz:
            from src.viz import fig_all_levels_geometry, fig_level_area_breakdown
            fig_all_levels_geometry(geometries, out1 / "figures", cfg)
            fig_level_area_breakdown(geometries, out1 / "figures", cfg)

        timings["step1"] = time.time() - t0
    else:
        from src.geometry_extractor import extract_all_levels
        geometries, all_connectors, control_points = extract_all_levels(cfg, products)

    # ----------------------------------------------------------------
    # Step 2: Node sampling
    # ----------------------------------------------------------------
    if 2 in steps_to_run:
        t0 = time.time()
        from src.node_sampler import sample_all_levels, save_sampling_outputs

        print("\n" + "=" * 60)
        print("STEP 2 ->Node Sampling")
        print("=" * 60)

        level_nodes = sample_all_levels(geometries, cfg)
        out2 = Path(ROOT / cfg["output"]["step_dirs"]["step2"])
        save_sampling_outputs(level_nodes, out2)

        total = sum(len(v) for v in level_nodes.values())
        for lvl, nodes in sorted(level_nodes.items()):
            print(f"  {lvl}: {len(nodes)} nodes")
        print(f"  Total: {total}")

        if not args.skip_viz:
            from src.viz import fig_all_levels_nodes, fig_node_density_per_level
            nodes_for_viz = {}
            for lvl, data in level_nodes.items():
                nv = data.get("nodes_valid", data) if isinstance(data, dict) else data
                if isinstance(nv, list) and nv and isinstance(nv[0], dict):
                    nodes_for_viz[lvl] = [(n["x"], n["y"]) for n in nv]
                else:
                    nodes_for_viz[lvl] = nv if isinstance(nv, list) else []
            fig_all_levels_nodes(geometries, nodes_for_viz, out2 / "figures", cfg)
            fig_node_density_per_level(nodes_for_viz, geometries, out2 / "figures", cfg)

        timings["step2"] = time.time() - t0
    else:
        from src.node_sampler import sample_all_levels
        level_nodes = sample_all_levels(geometries, cfg)

    # ----------------------------------------------------------------
    # Step 3: Graph construction
    # ----------------------------------------------------------------
    if 3 in steps_to_run:
        t0 = time.time()
        from src.node_sampler import voxelize_connectors
        from src.graph_builder import build_navigation_graph, save_graph_outputs

        print("\n" + "=" * 60)
        print("STEP 3 ->Graph Construction")
        print("=" * 60)

        # Voxelise connector anchor nodes and merge into level_nodes
        conn_nodes = voxelize_connectors(all_connectors, geometries, cfg)
        for lk, cns in conn_nodes.items():
            if lk in level_nodes:
                level_nodes[lk]["nodes_valid"].extend(cns)
                level_nodes[lk]["nodes_all"].extend(cns)
                level_nodes[lk]["n_valid"] = len(level_nodes[lk]["nodes_valid"])
                level_nodes[lk]["n_total"] = len(level_nodes[lk]["nodes_all"])

        G = build_navigation_graph(geometries, level_nodes, all_connectors, cfg)
        out3 = Path(ROOT / cfg["output"]["step_dirs"]["step3"])
        save_graph_outputs(G, all_connectors, out3)

        import networkx as nx
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Connected: {nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)}")

        if not args.skip_viz:
            from src.viz import (
                fig_graph_per_level, fig_graph_cross_level_edges,
                fig_connectors_overview, fig_graph_degree_distribution,
                fig_multilevel_isometric,
            )
            nodes_for_viz = {}
            for lvl, data in level_nodes.items():
                nv = data.get("nodes_valid", data) if isinstance(data, dict) else data
                if isinstance(nv, list) and nv and isinstance(nv[0], dict):
                    nodes_for_viz[lvl] = [(n["x"], n["y"]) for n in nv]
                else:
                    nodes_for_viz[lvl] = nv if isinstance(nv, list) else []
            elevations = {}
            for lk, lc in cfg["station"]["levels"].items():
                if lc.get("is_walkable", False) or lc.get("role") == "connector_pass":
                    elevations[lk] = lc["elevation_m"]
            fd = out3 / "figures"
            fig_graph_per_level(G, geometries, fd, cfg)
            fig_graph_cross_level_edges(G, fd, cfg)
            fig_connectors_overview(all_connectors, geometries, fd, cfg)
            fig_graph_degree_distribution(G, fd, cfg)
            fig_multilevel_isometric(geometries, nodes_for_viz, elevations, fd, cfg)

        timings["step3"] = time.time() - t0
    else:
        from src.node_sampler import voxelize_connectors
        from src.graph_builder import build_navigation_graph
        conn_nodes = voxelize_connectors(all_connectors, geometries, cfg)
        for lk, cns in conn_nodes.items():
            if lk in level_nodes:
                level_nodes[lk]["nodes_valid"].extend(cns)
                level_nodes[lk]["nodes_all"].extend(cns)
                level_nodes[lk]["n_valid"] = len(level_nodes[lk]["nodes_valid"])
                level_nodes[lk]["n_total"] = len(level_nodes[lk]["nodes_all"])
        G = build_navigation_graph(geometries, level_nodes, all_connectors, cfg)

    # ----------------------------------------------------------------
    # Step 4: Routing & OD setup
    # ----------------------------------------------------------------
    if 4 in steps_to_run:
        t0 = time.time()
        from src.routing import define_semantic_regions, sample_agents, save_routing_outputs

        print("\n" + "=" * 60)
        print("STEP 4 ->Routing & OD Setup")
        print("=" * 60)

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
        out4 = Path(ROOT / cfg["output"]["step_dirs"]["step4"])
        save_routing_outputs(G, regions, agents, out4)

        for name, nodes in regions.items():
            print(f"  Region '{name}': {len(nodes)} nodes")
        print(f"  Agents: {len(agents)}")

        if not args.skip_viz:
            from src.viz import fig_semantic_regions_map, fig_agent_overview, fig_example_paths
            fd = out4 / "figures"
            fig_semantic_regions_map(G, regions, geometries, fd, cfg)
            fig_agent_overview(agents, fd, cfg)
            fig_example_paths(G, agents, geometries, fd, cfg)

        timings["step4"] = time.time() - t0
    else:
        from src.routing import define_semantic_regions, sample_agents
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

    # ----------------------------------------------------------------
    # Step 5: Simulation
    # ----------------------------------------------------------------
    scenario_results = []
    if 5 in steps_to_run:
        t0 = time.time()
        from src.simulation import run_simulation

        print("\n" + "=" * 60)
        print("STEP 5 ->Simulation")
        print("=" * 60)

        # Static
        print("\n--- Scenario A: Static Routing ---")
        cfg_static = {**cfg, "simulation": {**cfg["simulation"], "routing_mode": "static"}}
        r_static = run_simulation(G, agents, cfg_static,
                                  out_dir=Path(ROOT / cfg["output"]["step_dirs"]["step5"]) / "static",
                                  label="static")
        r_static["label"] = "static"
        scenario_results.append(r_static)

        # Dynamic
        print("\n--- Scenario B: Dynamic Routing ---")
        cfg_dynamic = {**cfg, "simulation": {**cfg["simulation"], "routing_mode": "dynamic"}}
        r_dynamic = run_simulation(G, agents, cfg_dynamic,
                                   out_dir=Path(ROOT / cfg["output"]["step_dirs"]["step5"]) / "dynamic",
                                   label="dynamic")
        r_dynamic["label"] = "dynamic"
        scenario_results.append(r_dynamic)

        out5 = Path(ROOT / cfg["output"]["step_dirs"]["step5"])
        out5.mkdir(parents=True, exist_ok=True)
        for r in scenario_results:
            dump_json(out5 / f"result_{r['label']}.json", {
                k: v for k, v in r.items() if k != "trajectory_frames"
            })

        if not args.skip_viz:
            from src.viz import (
                fig_travel_time_distribution, fig_stair_queue_timeline,
                fig_arrival_curve, fig_comparison_bar, fig_elderly_vs_normal,
            )
            from src.evaluation import compute_scenario_metrics as _cmp
            fd = out5 / "figures"
            fig_travel_time_distribution(scenario_results, fd, cfg)
            fig_stair_queue_timeline(scenario_results, fd, cfg)
            fig_arrival_curve(scenario_results, fd, cfg)
            _ml = [_cmp(r) for r in scenario_results]
            fig_comparison_bar(_ml, fd, cfg)
            fig_elderly_vs_normal(_ml, fd, cfg)

        timings["step5"] = time.time() - t0

    # ----------------------------------------------------------------
    # Step 6: Evaluation
    # ----------------------------------------------------------------
    if 6 in steps_to_run:
        t0 = time.time()
        from src.evaluation import (
            compute_scenario_metrics,
            compare_scenarios,
            graph_topology_metrics,
            save_evaluation_outputs,
        )

        print("\n" + "=" * 60)
        print("STEP 6 ->Evaluation")
        print("=" * 60)

        # Load results if step 5 was skipped
        if not scenario_results:
            step5_dir = Path(ROOT / cfg["output"]["step_dirs"]["step5"])
            from src.utils import load_json
            for rf in sorted(step5_dir.glob("result_*.json")):
                scenario_results.append(load_json(rf))

        metrics_list = [compute_scenario_metrics(r) for r in scenario_results]
        graph_metrics = graph_topology_metrics(G)

        eval_dir = Path(ROOT / cfg["output"]["step_dirs"]["step6"])
        save_evaluation_outputs(scenario_results, graph_metrics, eval_dir)

        for m in metrics_list:
            print(f"  [{m['label']}] arrive={m['arrive_rate']:.1%}, "
                  f"mean_tt={m['mean_travel_time']:.1f}s")

        comp = compare_scenarios(metrics_list)
        if comp.get("comparisons"):
            for c in comp["comparisons"]:
                pct = c.get("mean_travel_time_pct_change", 0)
                print(f"  delta mean_tt: {pct:+.1f}%")

        # Visualisations
        if not args.skip_viz:
            print("\n  Generating evaluation visualisations ...")
            from src.viz import (
                fig_comparison_bar, fig_arrival_curve, fig_queue_over_time,
                fig_elderly_vs_normal, fig_connector_utilisation,
            )
            from src.evaluation import connector_utilisation as _cu

            fd = eval_dir / "figures"
            fig_comparison_bar(metrics_list, fd, cfg)
            fig_arrival_curve(scenario_results, fd, cfg)
            fig_queue_over_time(scenario_results, fd, cfg)
            fig_elderly_vs_normal(metrics_list, fd, cfg)

            util_list = []
            for r in scenario_results:
                u = _cu(r)
                u["label"] = r.get("label", "?")
                util_list.append(u)
            fig_connector_utilisation(util_list, fd, cfg)
            print("  5 evaluation figures generated")

        timings["step6"] = time.time() - t0

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for step, dt in sorted(timings.items()):
        print(f"  {step}: {dt:.1f}s")
    print(f"  Total: {total_time:.1f}s")

    # Save timing report
    out_base = Path(ROOT / cfg["output"]["base_dir"])
    out_base.mkdir(parents=True, exist_ok=True)
    dump_json(out_base / "pipeline_timing.json", {
        **timings,
        "total_s": total_time,
    })


if __name__ == "__main__":
    main()