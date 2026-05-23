"""
Step 3: Graph Construction  (v3 —KD-tree + ABM-ready)
========================================================

Build per-level floor graphs (KD-tree accelerated), merge
connector-anchor nodes, add typed vertical connectors and
door toggle-edges with ABM state metadata, prune to largest
connected component.

Changes from v2:
* KD-tree spatial index for floor-graph & snap
* ABM-ready edge attributes on PSD doors & elevator
* F2 excluded from visualization (no walkable graph)
* Connector overview & 2.5D isometric draw connector geometry
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.node_sampler import sample_all_levels, voxelize_connectors
from src.graph_builder import build_navigation_graph, save_graph_outputs


def _merge_connector_nodes(
    level_nodes: dict,
    conn_nodes: dict[str, list[dict]],
) -> dict:
    """Inject connector anchor nodes into per-level node sets."""
    for lk, cns in conn_nodes.items():
        if lk in level_nodes:
            level_nodes[lk]["nodes_valid"].extend(cns)
            level_nodes[lk]["nodes_all"].extend(cns)
            level_nodes[lk]["n_valid"] = len(level_nodes[lk]["nodes_valid"])
            level_nodes[lk]["n_total"] = len(level_nodes[lk]["nodes_all"])
    return level_nodes


def _collect_elevator_door_cfg(cfg: dict) -> dict[str, list[dict]]:
    """Collect elevator door configs from all levels (for viz)."""
    elev_doors: dict[str, list[dict]] = {}
    for lvl, lcfg in cfg["station"]["levels"].items():
        dd = lcfg.get("dynamic_doors", {})
        eds = dd.get("elevator_doors", [])
        if eds:
            elev_doors[lvl] = eds
    return elev_doors


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step3"])

    print("=" * 60)
    print("STEP 3 —Graph Construction (v3 —KD-tree + ABM-ready)")
    print("=" * 60)

    # ---- upstream ----
    products = load_preprocessing_products(cfg)
    geometries, all_connectors, control_points = extract_all_levels(cfg, products)
    level_nodes = sample_all_levels(geometries, cfg)

    # ---- connector anchor voxelisation ----
    print("\n  Voxelising connectors ...")
    conn_nodes = voxelize_connectors(all_connectors, geometries, cfg)
    total_conn = sum(len(v) for v in conn_nodes.values())
    print(f"  Total connector anchor nodes: {total_conn}")

    # ---- merge connector anchors into node sets ----
    level_nodes = _merge_connector_nodes(level_nodes, conn_nodes)
    for lk, data in sorted(level_nodes.items()):
        print(f"  {lk}: {data['n_valid']:,} nodes (merged)")

    # ---- build graph ----
    print("\n  Building navigation graph ...")
    G = build_navigation_graph(geometries, level_nodes, all_connectors, cfg)
    print(f"\n  Graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    print(f"  Connected: {__import__('networkx').is_weakly_connected(G)}")

    save_graph_outputs(G, all_connectors, out_dir)
    print(f"\n  Outputs →{out_dir}")

    # --- Visualization ---
    print("\n  Generating visualisations ...")
    from src.viz import (
        fig_connectors_overview,
        fig_graph_per_level,
        fig_graph_cross_level_edges,
        fig_graph_degree_distribution,
        fig_multilevel_isometric,
    )
    from src.viz_interactive import (
        fig_interactive_station,
        fig_interactive_graph,
        fig_interactive_cross_section,
    )

    fig_dir = out_dir / "figures"
    html_dir = out_dir / "interactive"

    # nodes_for_viz: dict[str, list[tuple[x, y]]] for isometric plot
    nodes_for_viz = {}
    for lvl, data in level_nodes.items():
        nv = data.get("nodes_valid", data) if isinstance(data, dict) else data
        if isinstance(nv, list) and nv and isinstance(nv[0], dict):
            nodes_for_viz[lvl] = [(n["x"], n["y"]) for n in nv]
        else:
            nodes_for_viz[lvl] = nv if isinstance(nv, list) else []

    # Elevations: only walkable levels (exclude F2)
    elevations = {}
    for lvl, lc in cfg["station"]["levels"].items():
        if lc.get("is_walkable", False):
            elevations[lvl] = lc["elevation_m"]

    # Elevator door config for connector overview
    elev_door_cfg = _collect_elevator_door_cfg(cfg)

    # 1. Connector overview (new: accepts all_connectors list)
    fig_connectors_overview(
        all_connectors, geometries, fig_dir, cfg,
        door_cfg=elev_door_cfg,
    )
    # 2. Per-level graph
    fig_graph_per_level(G, geometries, fig_dir, cfg)
    # 3. All connector edges (with footprints)
    fig_graph_cross_level_edges(G, fig_dir, cfg,
                                all_connectors=all_connectors)
    # 4. Degree distribution
    fig_graph_degree_distribution(G, fig_dir, cfg)
    # 5. 2.5D isometric with 3D connector geometry
    fig_multilevel_isometric(
        geometries, nodes_for_viz, elevations, fig_dir, cfg,
        all_connectors=all_connectors,
    )
    print("  5 static figures saved")

    # --- Interactive HTML (Plotly) ---
    print("\n  Generating interactive visualisations ...")
    p = fig_interactive_station(
        geometries, nodes_for_viz, elevations, all_connectors,
        html_dir, cfg, G=G,
    )
    print(f"  [html] {p.name}")
    p = fig_interactive_graph(
        G, all_connectors, elevations, html_dir, cfg,
    )
    print(f"  [html] {p.name}")
    p = fig_interactive_cross_section(
        geometries, all_connectors, elevations, html_dir, cfg,
    )
    print(f"  [html] {p.name}")
    print("  3 interactive HTML saved")

    return cfg, geometries, level_nodes, all_connectors, G


if __name__ == "__main__":
    main()
