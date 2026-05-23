"""
Step 5 — Extended Experiments
================================

Three thesis scenarios (per supervisor feedback) + wheelchair accessibility.

Scenario A  Individual routing verification
           Small set of agents, diverse types (normal / elderly),
           show that routing constraints are correct.

Scenario B  Many-agent static simulation  (200 agents, no replanning)
           Establish baseline. Produce heat maps and space-use figures.

Scenario C  Many-agent congestion-aware rerouting  (200 agents, dynamic)
           Compare against B. Show rerouting benefit with stats + diff maps.

Wheelchair  Accessibility path comparison
           Same base graph, stairs/escalators disabled for wheelchair agents.
           Show path difference with fig_wheelchair_path_comparison.

Output tree
-----------
outputs/step5_simulation/
├── scenA_individual/       figures for Scenario A
├── scenB_static/           figures for Scenario B
├── scenC_dynamic/          figures for Scenario C
├── wheelchair/             accessibility path comparison
└── comparison/             cross-scenario stats, box-plots, connector bars
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
    define_semantic_regions, sample_agents,
    find_path, directed_weight, penalised_weight,
)
from src.simulation import run_simulation
from src.viz_thesis import (
    fig_node_heatmap_per_level,
    fig_small_multiples_time,
    fig_individual_paths,
    fig_stats_comparison,
    fig_travel_time_box,
    fig_connector_load_bars,
    fig_edge_throughput_heatmap,
    fig_wheelchair_path_comparison,
    fig_flow_diff_two_scenarios,
    fig_capacity_curves,
    generate_evaluation_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_graph(cfg: dict) -> object:
    import networkx as nx  # noqa: F401
    graph_path = ROOT / cfg["output"]["step_dirs"]["step3"] / "navigation_graph.gpickle"
    print(f"  Loading graph from {graph_path} …")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def _save_result_json(result: dict, path: Path) -> None:
    from src.utils import dump_json
    dump_json(path, {k: v for k, v in result.items()
                     if k not in ("trajectory_frames",)})


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A — Individual routing verification (small N)
# ─────────────────────────────────────────────────────────────────────────────

def run_scenA(cfg: dict, G, geometries: dict, regions: dict, out_base: Path) -> dict:
    """Run scenario A: 20 agents (mixed types), static routing, verify paths."""
    print("\n" + "═" * 60)
    print("SCENARIO A — Individual Routing Verification (N=20)")
    print("═" * 60)

    out_dir = out_base / "scenA_individual"
    fig_dir = out_dir / "figures"

    # Small agent set — all four entrances represented
    cfg_a = cfg.copy()
    cfg_a["simulation"] = dict(cfg["simulation"])
    cfg_a["simulation"]["n_agents"] = 20
    cfg_a["simulation"]["elderly_ratio"] = 0.3  # 30% elderly to show heterogeneity
    cfg_a["simulation"]["flows"] = {"ENTRANCE->PLATFORM": 0.7, "PLATFORM->EXIT": 0.3}

    agents = sample_agents(
        regions=regions,
        flows=cfg_a["simulation"]["flows"],
        n_agents=cfg_a["simulation"]["n_agents"],
        T=cfg["simulation"]["T_s"],
        seed=cfg["simulation"]["seed"],
        walking_speed=cfg["simulation"]["walking_speed_ms"],
        elderly_ratio=cfg_a["simulation"]["elderly_ratio"],
    )

    result = run_simulation(G, agents, cfg_a,
                            out_dir=out_dir, routing_mode="static", label="scenA")
    result["label"] = "scenA_individual"
    _save_result_json(result, out_dir / "result_scenA.json")

    # ── Visualisations ──────────────────────────────────────────────────────
    traj_file = out_dir / "traj_agents.jsonl"

    # 1. Individual path traces (primary evidence paths are correct)
    fig_individual_paths(traj_file, G, geometries, fig_dir,
                         label="Scenario A — Individual (N=20)",
                         n_agents=8, seed=42, cfg=cfg)

    # 2. Small multiples with OD colours (3 time snapshots sufficient for 20 agents)
    fig_small_multiples_time(traj_file, G, geometries, fig_dir,
                             label="Scenario A — Individual",
                             n_snapshots=4, cfg=cfg)

    # 3. Node heat map
    fig_node_heatmap_per_level(traj_file, G, geometries, fig_dir,
                               label="scenA", cfg=cfg)

    print(f"  → Scenario A outputs: {out_dir}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B — Many-agent static baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_scenB(cfg: dict, G, geometries: dict, regions: dict, out_base: Path) -> dict:
    """Run scenario B: 200 agents, static routing (baseline)."""
    print("\n" + "═" * 60)
    print("SCENARIO B — Many-Agent Static Simulation (N=200)")
    print("═" * 60)

    out_dir = out_base / "scenB_static"
    fig_dir = out_dir / "figures"

    agents = sample_agents(
        regions=regions,
        flows=cfg["simulation"]["flows"],
        n_agents=cfg["simulation"]["n_agents"],
        T=cfg["simulation"]["T_s"],
        seed=cfg["simulation"]["seed"],
        walking_speed=cfg["simulation"]["walking_speed_ms"],
        elderly_ratio=cfg["simulation"].get("elderly_ratio", 0.2),
    )

    result = run_simulation(G, agents, cfg,
                            out_dir=out_dir, routing_mode="static", label="scenB_static")
    result["label"] = "scenB_static"
    _save_result_json(result, out_dir / "result_scenB.json")

    traj_file = out_dir / "traj_agents.jsonl"

    # 1. Node visit frequency heat map — shows where people accumulate
    fig_node_heatmap_per_level(traj_file, G, geometries, fig_dir,
                               label="Scenario B — Static", cfg=cfg)

    # 2. Edge throughput heat map — shows which corridors / connectors are used
    fig_edge_throughput_heatmap(result, G, geometries, fig_dir,
                                label="scenB_static", cfg=cfg)

    # 3. Small multiples (5 time snapshots)
    fig_small_multiples_time(traj_file, G, geometries, fig_dir,
                             label="Scenario B — Static",
                             n_snapshots=5, cfg=cfg)

    # 4. Individual paths (6 agents for illustration)
    fig_individual_paths(traj_file, G, geometries, fig_dir,
                         label="Scenario B — Static",
                         n_agents=6, seed=0, cfg=cfg)

    print(f"  → Scenario B outputs: {out_dir}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C — Many-agent congestion-aware rerouting
# ─────────────────────────────────────────────────────────────────────────────

def run_scenC(cfg: dict, G, geometries: dict, regions: dict, out_base: Path) -> dict:
    """Run scenario C: 200 agents, dynamic congestion-aware routing."""
    print("\n" + "═" * 60)
    print("SCENARIO C — Many-Agent Dynamic (Congestion-Aware) Simulation (N=200)")
    print("═" * 60)

    out_dir = out_base / "scenC_dynamic"
    fig_dir = out_dir / "figures"

    agents = sample_agents(
        regions=regions,
        flows=cfg["simulation"]["flows"],
        n_agents=cfg["simulation"]["n_agents"],
        T=cfg["simulation"]["T_s"],
        seed=cfg["simulation"]["seed"],
        walking_speed=cfg["simulation"]["walking_speed_ms"],
        elderly_ratio=cfg["simulation"].get("elderly_ratio", 0.2),
    )

    result = run_simulation(G, agents, cfg,
                            out_dir=out_dir, routing_mode="dynamic", label="scenC_dynamic")
    result["label"] = "scenC_dynamic"
    _save_result_json(result, out_dir / "result_scenC.json")

    traj_file = out_dir / "traj_agents.jsonl"

    # 1. Node heat map
    fig_node_heatmap_per_level(traj_file, G, geometries, fig_dir,
                               label="Scenario C — Dynamic", cfg=cfg)

    # 2. Edge throughput heat map
    fig_edge_throughput_heatmap(result, G, geometries, fig_dir,
                                label="scenC_dynamic", cfg=cfg)

    # 3. Small multiples
    fig_small_multiples_time(traj_file, G, geometries, fig_dir,
                             label="Scenario C — Dynamic",
                             n_snapshots=5, cfg=cfg)

    # 4. Replan timeline (from existing viz.py)
    from src.viz import fig_replan_timeline
    fig_replan_timeline(result, fig_dir, cfg)

    print(f"  → Scenario C outputs: {out_dir}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Wheelchair accessibility comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_wheelchair(cfg: dict, G, geometries: dict, regions: dict, out_base: Path) -> None:
    """Build a wheelchair-accessible subgraph and compare paths."""
    print("\n" + "═" * 60)
    print("WHEELCHAIR — Accessibility Path Comparison")
    print("═" * 60)
    import networkx as nx
    import random

    out_dir = out_base / "wheelchair"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Build wheelchair subgraph: disable stairs & escalators ──────────────
    G_wc = G.copy()
    disallowed = {"stair", "escalator"}
    edges_to_remove = [
        (u, v) for u, v, d in G_wc.edges(data=True)
        if any(x in d.get("edge_type", "") for x in disallowed)
    ]
    G_wc.remove_edges_from(edges_to_remove)
    print(f"  Wheelchair graph: removed {len(edges_to_remove)} stair/escalator edges")

    wfn = directed_weight(G)
    wfn_wc = directed_weight(G_wc)

    # Pick 3 representative entrance → platform pairs
    entrance_nodes = regions.get("ENTRANCE", [])
    platform_nodes = regions.get("PLATFORM", [])
    rng = random.Random(42)
    pairs = []
    for _ in range(3):
        if entrance_nodes and platform_nodes:
            o = rng.choice(entrance_nodes)
            d = rng.choice(platform_nodes)
            pairs.append((o, d))

    saved = []
    for i, (origin, dest) in enumerate(pairs):
        normal_path = find_path(G, origin, dest, wfn)
        wc_path = find_path(G_wc, origin, dest, wfn_wc)

        # Wheelchair path length == 1 → no cross-level accessible route found
        if len(wc_path) <= 1:
            print(f"  Pair {i+1}: ⚠️  No wheelchair-accessible cross-level route "
                  f"(elevator only; disabled if disconnected). "
                  f"normal path = {len(normal_path)} nodes")

        o_grp = G.nodes.get(origin, {}).get("entrance_group", origin[:12])
        d_str = dest[:12]

        fig_wheelchair_path_comparison(
            G, normal_path, wc_path, geometries,
            out_dir=fig_dir,
            origin_label=f"Entrance {o_grp}",
            dest_label=f"PSD {d_str}",
            cfg=cfg,
        )
        # Save as numbered figures if multiple pairs (replace() overwrites existing)
        src = fig_dir / "wheelchair_path_comparison.png"
        dst = fig_dir / f"wheelchair_path_comparison_{i+1}.png"
        if src.exists():
            src.replace(dst)
        saved.append(dst)

        print(f"  Pair {i+1}: normal={len(normal_path)} nodes, "
              f"wheelchair={len(wc_path)} nodes")

    print(f"  → Wheelchair outputs: {fig_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Capacity / system-boundary sweep
# ─────────────────────────────────────────────────────────────────────────────

# Agent-count ladder: sparse at low end, dense around inflection, then push hard
SWEEP_NS_STATIC  = [100, 200, 300, 500, 750, 1000, 1500, 2000]
SWEEP_NS_DYNAMIC = [200, 500, 1000, 1500, 2000]


def run_capacity_sweep(cfg: dict, G, regions: dict, out_base: Path) -> list[dict]:
    """Sweep agent count from 100 to 2000 to find throughput saturation.

    Runs *static* routing for all N values and *dynamic* routing for a subset.
    Trajectories are NOT written to disk (``write_traj=False``) to save space
    and time — only aggregate statistics are recorded.

    Returns a flat list of result dicts, each augmented with ``n_agents``,
    ``label``, and ``T_s``.
    """
    print("\n" + "═" * 60)
    print("CAPACITY SWEEP — System Boundary Analysis")
    print(f"  Static  N = {SWEEP_NS_STATIC}")
    print(f"  Dynamic N = {SWEEP_NS_DYNAMIC}")
    print("═" * 60)

    T_s = cfg["simulation"]["T_s"]
    out_dir = out_base / "capacity_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    def _single_run(n: int, mode: str, run_label: str, seed_offset: int = 0,
                    extra_sim_cfg: dict | None = None) -> dict:
        cfg_run = cfg.copy()
        cfg_run["simulation"] = dict(cfg["simulation"])
        cfg_run["simulation"]["n_agents"] = n
        cfg_run["simulation"]["seed"] = cfg["simulation"]["seed"] + seed_offset
        if extra_sim_cfg:
            cfg_run["simulation"].update(extra_sim_cfg)

        agents = sample_agents(
            regions=regions,
            flows=cfg_run["simulation"]["flows"],
            n_agents=n,
            T=T_s,
            seed=cfg_run["simulation"]["seed"],
            walking_speed=cfg["simulation"]["walking_speed_ms"],
            elderly_ratio=cfg["simulation"].get("elderly_ratio", 0.2),
        )

        # Use a minimal scratch dir — no traj written
        scratch_dir = out_dir / f"N{n}_{mode}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        r = run_simulation(G, agents, cfg_run,
                           out_dir=scratch_dir,
                           routing_mode=mode,
                           label=run_label,
                           write_traj=False)
        r["n_agents"]  = n
        r["label"]     = mode
        r["T_s"]       = T_s
        return r

    # ── Static sweep ─────────────────────────────────────────────────────────
    for n in SWEEP_NS_STATIC:
        print(f"  [static] N={n:4d} ...", end=" ", flush=True)
        r = _single_run(n, "static", f"static_N{n}")
        arr_pct = r.get("arrive_rate", 0) * 100
        tts = r.get("travel_times", [])
        mean_tt_s = sum(tts) / len(tts) if tts else float("nan")
        print(f"arrived={arr_pct:.1f}%  mean_tt={mean_tt_s:.1f}s")
        all_results.append(r)

    # ── Dynamic sweep ─────────────────────────────────────────────────────────
    for n in SWEEP_NS_DYNAMIC:
        print(f"  [dynmic] N={n:4d} ...", end=" ", flush=True)
        # For large-N sweep runs, throttle congestion recomputation to every 2 steps
        extra = {"congestion_recompute_every": 2} if n >= 500 else None
        r = _single_run(n, "dynamic", f"dynamic_N{n}", seed_offset=1000, extra_sim_cfg=extra)
        arr_pct = r.get("arrive_rate", 0) * 100
        tts = r.get("travel_times", [])
        mean_tt_s = sum(tts) / len(tts) if tts else float("nan")
        replan_n = len(r.get("replan_events", []))
        print(f"arrived={arr_pct:.1f}%  mean_tt={mean_tt_s:.1f}s  replans={replan_n}")
        all_results.append(r)

    # ── Output figure ─────────────────────────────────────────────────────────
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_capacity_curves(all_results, fig_dir, cfg=cfg)
    print(f"  → Capacity sweep outputs: {out_dir}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-scenario comparison figures
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(results_bc: list[dict], G, geometries: dict,
                   out_base: Path, cfg: dict,
                   result_a: dict | None = None) -> None:
    """Generate figures comparing Scenario B (static) vs C (dynamic)."""
    print("\n" + "═" * 60)
    print("COMPARISON — B (static) vs C (dynamic)")
    print("═" * 60)

    out_dir = out_base / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stats comparison (all results including scenA for reference)
    fig_stats_comparison(results_bc, out_dir, cfg=cfg)

    # Travel time box-plots
    fig_travel_time_box(results_bc, out_dir, cfg=cfg)

    # Connector load bars
    fig_connector_load_bars(results_bc, G, out_dir, cfg=cfg)

    # Flow diff between B (static) and C (dynamic)
    if len(results_bc) >= 2:
        r_b = next((r for r in results_bc if "static" in r.get("label", "")), results_bc[0])
        r_c = next((r for r in results_bc if "dynamic" in r.get("label", "")), results_bc[-1])
        fig_flow_diff_two_scenarios(
            r_b, r_c, G, geometries, out_dir,
            label_a="static", label_b="dynamic", cfg=cfg,
        )

    # Evaluation summary (Markdown + condensed figure)
    all_results = ([result_a] if result_a else []) + results_bc
    eval_dir = out_base / "evaluation"
    generate_evaluation_report(all_results, eval_dir, cfg=cfg)

    print(f"  → Comparison outputs: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str | None = None) -> None:
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_base = Path(ROOT / cfg["output"]["step_dirs"]["step5"])
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 5 — Extended Experiments (Thesis)")
    print("=" * 60)

    # ── Shared resources ────────────────────────────────────────────────────
    G = _load_graph(cfg)
    products = load_preprocessing_products(cfg)
    geometries, all_connectors, _ = extract_all_levels(cfg, products)
    regions = define_semantic_regions(G, cfg)

    print(f"\n  Entrance nodes : {len(regions.get('ENTRANCE', []))}")
    print(f"  Platform nodes : {len(regions.get('PLATFORM', []))}")

    # ── Run experiments ──────────────────────────────────────────────────────
    result_a = run_scenA(cfg, G, geometries, regions, out_base)
    result_b = run_scenB(cfg, G, geometries, regions, out_base)
    result_c = run_scenC(cfg, G, geometries, regions, out_base)

    # ── Accessibility ────────────────────────────────────────────────────────
    run_wheelchair(cfg, G, geometries, regions, out_base)

    # ── Cross-scenario comparison ────────────────────────────────────────────
    run_comparison([result_b, result_c], G, geometries, out_base, cfg,
                   result_a=result_a)

    # ── Capacity / system-boundary sweep ────────────────────────────────────
    run_capacity_sweep(cfg, G, regions, out_base)

    print("\n" + "=" * 60)
    print(f"All step-5 experiments complete → {out_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
