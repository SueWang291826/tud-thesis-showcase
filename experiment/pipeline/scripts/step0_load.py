"""
Step 0: Load and Validate Data
===============================

Reads the v2/v3 preprocessing CSV products (retained elements,
connectors, obstacles, bounding boxes) and resolves IFC file paths.
Filters connectors and obstacles to navigation-relevant subsets.

Outputs (all in outputs/step0_data/):
    data_summary.json      -- element counts + available IFC paths
    level_counts.json      -- per-level retained/obstacle/connector counts
    nav_obstacles.csv      -- navigation-relevant obstacle subset
    nav_connectors.csv     -- navigation-relevant connector subset
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import (
    load_preprocessing_products,
    filter_obstacles_for_navigation,
    filter_connectors_for_navigation,
    save_step0_outputs,
)


def main(config_path: str | None = None):
    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step0"])

    print("=" * 60)
    print("STEP 0 -- Data Loading & Validation")
    print("=" * 60)

    products = load_preprocessing_products(cfg)
    nav_obstacles  = filter_obstacles_for_navigation(products["obstacle_df"])
    nav_connectors = filter_connectors_for_navigation(products["connector_df"])

    print(f"  Retained elements : {len(products['retained_df']):,}")
    print(f"  All obstacles     : {len(products['obstacle_df']):,}")
    print(f"  Nav obstacles     : {len(nav_obstacles):,}")
    print(f"  All connectors    : {len(products['connector_df']):,}")
    print(f"  Nav connectors    : {len(nav_connectors):,}")
    print(f"  IFC subsets       : {list(products['ifc_paths'].keys())}")

    levels = products["levels"]
    for lvl, info in levels.items():
        print(f"    {lvl}: {info.get('name_en', '')} ({info.get('name_cn', '')})")

    save_step0_outputs(products, nav_obstacles, nav_connectors, out_dir)
    print(f"\n  Outputs -> {out_dir}")

    # Visualisation
    print("\n  Generating visualisations ...")
    from src.viz import fig_data_overview, fig_level_area_breakdown

    fig_dir = out_dir / "figures"
    fig_data_overview(products, nav_obstacles, nav_connectors, fig_dir, cfg)
    fig_level_area_breakdown(products["retained_df"], fig_dir, cfg)
    print("  2 figures saved")

    return cfg, products, nav_obstacles, nav_connectors


if __name__ == "__main__":
    main()
