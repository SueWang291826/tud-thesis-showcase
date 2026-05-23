"""Step 2: Node Sampling (v2 —human-scale aware)
==================================================

Grid sampling with obstacle clearance filtering, connector exclusion,
and control-point (fare gate / security scanner) awareness.

Design notes
------------
* Grid resolution = 0.5 m  ≈human shoulder width.
* min_clearance   = 0.25 m ≈agent body radius.
* Fare gates (~1.3 × 0.4 m) and railings (~12 × 0.25 m) are thin but
  physically impassable —they are already subtracted from the walkable
  polygon (Step 1) and the clearance check ensures no node sits too
  close to their boundary.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.node_sampler import (
    sample_all_levels, save_sampling_outputs, voxelize_connectors,
)


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step2"])

    print("=" * 60)
    print("STEP 2 —Node Sampling (v2)")
    print("=" * 60)

    # ---- upstream ----
    products = load_preprocessing_products(cfg)
    geometries, all_connectors, control_points = extract_all_levels(cfg, products)

    scfg = cfg["sampling"]
    print(f"  Grid resolution : {scfg['grid_resolution_m']} m")
    print(f"  Min clearance   : {scfg['min_clearance_m']} m  "
          f"(agent radius = {cfg['simulation']['agent_radius_m']} m)")

    # ---- sample ----
    level_nodes = sample_all_levels(geometries, cfg)

    total = sum(v["n_valid"] for v in level_nodes.values())
    for lvl, data in sorted(level_nodes.items()):
        ratio = data['n_valid'] / data['n_total'] * 100 if data['n_total'] else 0
        print(f"  {lvl}: {data['n_valid']:,} usable / {data['n_total']:,} total "
              f"({ratio:.1f}%)")
    print(f"  Total: {total:,} usable nodes")

    save_sampling_outputs(level_nodes, out_dir)
    print(f"\n  Outputs →{out_dir}")

    # ---- connector voxelisation ----
    print("\n  Voxelising connectors ...")
    conn_nodes = voxelize_connectors(all_connectors, geometries, cfg)
    total_conn = sum(len(v) for v in conn_nodes.values())
    print(f"  Total connector anchor nodes: {total_conn}")

    # ---- visualisation ----
    print("\n  Generating visualisations ...")
    from src.viz import (
        fig_all_levels_nodes, fig_node_density_per_level,
        fig_connector_voxels,
    )

    nodes_for_viz = {}
    for lvl, data in level_nodes.items():
        nv = data.get("nodes_valid", data) if isinstance(data, dict) else data
        if isinstance(nv, list) and nv and isinstance(nv[0], dict):
            nodes_for_viz[lvl] = [(n["x"], n["y"]) for n in nv]
        else:
            nodes_for_viz[lvl] = nv if isinstance(nv, list) else []

    fig_dir = out_dir / "figures"
    fig_all_levels_nodes(geometries, nodes_for_viz, fig_dir, cfg)
    fig_node_density_per_level(nodes_for_viz, geometries, fig_dir, cfg)
    fig_connector_voxels(geometries, conn_nodes, fig_dir, cfg)
    print("  3 figures saved")

    return cfg, geometries, level_nodes, all_connectors, control_points


if __name__ == "__main__":
    main()
