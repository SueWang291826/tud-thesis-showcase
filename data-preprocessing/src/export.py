"""
Intermediate Data Export Module.

Exports cleaned, structured intermediate data that downstream stages can
consume without re-reading raw IFC files.

Export formats:
- CSV: Universal, human-readable, Excel-compatible (UTF-8 BOM)
- JSON: Structured metadata, summaries, and mappings
- (GeoJSON: reserved for spatial footprints if needed later)

Design rationale:
- CSV is chosen as the primary tabular format for maximum interoperability.
- JSON is used for hierarchical and metadata structures.
- Pickle/parquet are avoided at this stage to maintain transparency and
  reproducibility without binary dependencies.
- GeoJSON export is reserved for when 2D footprints are extracted.

All exported files use deterministic naming and are organized by storey and category.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import save_json, save_dataframe

logger = logging.getLogger(__name__)


def export_intermediate_data(
    classified_dfs: Dict[str, pd.DataFrame],
    storey_mapping: Dict[str, Any],
    proxy_dfs: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Export clean intermediate data for downstream consumption.

    Args:
        classified_dfs: Per-file classified element DataFrames.
        storey_mapping: Storey mapping results.
        proxy_dfs: Per-file proxy inventory DataFrames.
        config: Pipeline configuration.
        output_dir: Base export directory.

    Returns:
        Export manifest dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting intermediate data")

    manifest = {"exports": []}

    # ---- 1. Combined classified inventory ----
    combined = pd.concat(
        [df.assign(source_file=label) for label, df in classified_dfs.items()],
        ignore_index=True,
    )
    path = output_dir / "all_elements_classified.csv"
    save_dataframe(combined, path)
    manifest["exports"].append({
        "name": "all_elements_classified",
        "path": str(path),
        "rows": len(combined),
        "description": "Complete inventory of all IFC product elements with semantic classification"
    })
    logger.info(f"  Exported all_elements_classified.csv ({len(combined)} rows)")

    # ---- 2. Storey-wise exports ----
    storey_dir = output_dir / "by_storey"
    storey_dir.mkdir(exist_ok=True)

    storey_names = combined["storey_name"].unique()
    for sname in sorted(storey_names):
        safe_name = sname.replace(" ", "_").replace("/", "_")
        storey_df = combined[combined["storey_name"] == sname]
        path = storey_dir / f"elements_{safe_name}.csv"
        save_dataframe(storey_df, path)
        manifest["exports"].append({
            "name": f"storey_{safe_name}",
            "path": str(path),
            "rows": len(storey_df),
            "description": f"Elements on storey: {sname}"
        })

    logger.info(f"  Exported {len(storey_names)} storey-wise element tables")

    # ---- 3. Category-wise exports ----
    category_dir = output_dir / "by_category"
    category_dir.mkdir(exist_ok=True)

    categories = combined["category"].unique()
    for cat in sorted(categories):
        cat_df = combined[combined["category"] == cat]
        path = category_dir / f"elements_{cat}.csv"
        save_dataframe(cat_df, path)
        manifest["exports"].append({
            "name": f"category_{cat}",
            "path": str(path),
            "rows": len(cat_df),
            "description": f"Elements classified as: {cat}"
        })

    logger.info(f"  Exported {len(categories)} category-wise element tables")

    # ---- 4. Walkable level exports (platform + concourse) ----
    walkable_levels = storey_mapping.get("walkable_levels", [])
    if walkable_levels:
        walkable_df = combined[combined["storey_name"].isin(walkable_levels)]
        path = output_dir / "walkable_level_elements.csv"
        save_dataframe(walkable_df, path)
        manifest["exports"].append({
            "name": "walkable_level_elements",
            "path": str(path),
            "rows": len(walkable_df),
            "description": f"All elements on walkable public levels: {walkable_levels}"
        })
        logger.info(f"  Exported walkable level elements: {len(walkable_df)} rows")

    # ---- 5. Connector candidates ----
    connector_df = combined[combined["category"] == "vertical_connector"]
    if len(connector_df) > 0:
        path = output_dir / "connector_candidates.csv"
        save_dataframe(connector_df, path)
        manifest["exports"].append({
            "name": "connector_candidates",
            "path": str(path),
            "rows": len(connector_df),
            "description": "Vertical connector candidates (stairs, ramps, elevators, escalators)"
        })
        logger.info(f"  Exported connector candidates: {len(connector_df)} rows")

    # ---- 6. Obstacle candidates ----
    obstacle_df = combined[combined["category"] == "obstacle"]
    if len(obstacle_df) > 0:
        path = output_dir / "obstacle_candidates.csv"
        save_dataframe(obstacle_df, path)
        manifest["exports"].append({
            "name": "obstacle_candidates",
            "path": str(path),
            "rows": len(obstacle_df),
            "description": "Obstacle candidates (walls, columns, etc.)"
        })
        logger.info(f"  Exported obstacle candidates: {len(obstacle_df)} rows")

    # ---- 7. Walkable support candidates ----
    walkable_support_df = combined[combined["category"] == "walkable_support"]
    if len(walkable_support_df) > 0:
        path = output_dir / "walkable_support_candidates.csv"
        save_dataframe(walkable_support_df, path)
        manifest["exports"].append({
            "name": "walkable_support_candidates",
            "path": str(path),
            "rows": len(walkable_support_df),
            "description": "Walkable support candidates (floor slabs, platforms)"
        })

    # ---- 8. Combined proxy inventory ----
    if proxy_dfs:
        combined_proxy = pd.concat(
            [df.assign(source_file=label) for label, df in proxy_dfs.items()],
            ignore_index=True,
        )
        path = output_dir / "proxy_inventory_combined.csv"
        save_dataframe(combined_proxy, path)
        manifest["exports"].append({
            "name": "proxy_inventory_combined",
            "path": str(path),
            "rows": len(combined_proxy),
            "description": "Combined proxy inventory with inferred categories"
        })

    # ---- 9. Storey mapping ----
    save_json(storey_mapping, output_dir / "storey_mapping.json")
    manifest["exports"].append({
        "name": "storey_mapping",
        "path": str(output_dir / "storey_mapping.json"),
        "rows": None,
        "description": "Cross-file storey mapping with functional roles"
    })

    # ---- Save export manifest ----
    save_json(manifest, output_dir / "export_manifest.json")
    logger.info(f"  Export manifest: {len(manifest['exports'])} items saved")

    return manifest
