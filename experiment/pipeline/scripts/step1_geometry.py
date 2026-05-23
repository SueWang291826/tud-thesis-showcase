"""
Step 1: Geometry Extraction  (v2 - IFC-based floor + typed connectors)
======================================================================

Extracts floor polygons from raw IFC IfcSlab elements, obstacles from
bbox CSV, and connectors (escalators / elevators / stair flights) from
IFC. Saves GeoJSON outputs per level.

Outputs (all in outputs/step1_geometry/):
    <level_key>/floor.geojson
    <level_key>/obstacles.geojson
    <level_key>/connectors.geojson
    <level_key>/control_points.geojson
    <level_key>/walkable.geojson
    geometry_summary.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels, save_geometry_outputs


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step1"])

    print("=" * 60)
    print("STEP 1 -- Geometry Extraction (v2)")
    print("=" * 60)

    products = load_preprocessing_products(cfg)
    geometries, all_connectors, control_points = extract_all_levels(cfg, products)

    print("\n  Per-level summary:")
    for lvl, g in sorted(geometries.items()):
        floor = g.get("floor")
        floor_area = floor.area if floor is not None else 0.0
        n_obs  = len(g.get("obstacles", []))
        n_conn = len(g.get("connectors", []))
        n_cp   = len(g.get("control_points", []))
        walkable = g.get("walkable")
        walkable_area = walkable.area if walkable is not None else 0.0
        print(f"  {lvl}: floor={floor_area:,.0f}m2, obstacles={n_obs}, "
              f"connectors={n_conn}, control_pts={n_cp}, "
              f"walkable={walkable_area:,.0f}m2")

    print(f"\n  Total connectors: {sum(1 for c in all_connectors)}")
    print(f"\n  Total control points: {sum(1 for cp in control_points)}")

    save_geometry_outputs(geometries, all_connectors, out_dir, control_points)
    print(f"\n  Outputs -> {out_dir}")

    # Visualisation
    print("\n  Generating visualisations ...")
    from src.viz import (
        fig_all_levels_geometry,
        fig_level_area_breakdown,
        fig_connectors_on_floors,
    )

    fig_dir = out_dir / "figures"
    fig_all_levels_geometry(geometries, fig_dir, cfg)
    fig_level_area_breakdown(geometries, fig_dir, cfg)
    fig_connectors_on_floors(geometries, all_connectors, fig_dir, cfg)
    print("  3 figures saved")

    return cfg, geometries, all_connectors, control_points


if __name__ == "__main__":
    main()
