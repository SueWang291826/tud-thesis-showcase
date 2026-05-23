"""
Step 0: Data Loader
====================

Load preprocessed CSV products from v2/v3 pipeline and resolve IFC paths.
This module bridges the preprocessing pipeline outputs to the experiment
framework, avoiding redundant IFC parsing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_config, dump_json


def _resolve_path(base: Path, rel: str) -> Path:
    """Resolve a relative path against a base directory."""
    p = base / rel
    if p.exists():
        return p
    raise FileNotFoundError(f"Expected file not found: {p}")


def load_preprocessing_products(config: dict) -> dict[str, Any]:
    """Load all v2/v3 CSV products into DataFrames.

    Returns a dict with keys:
        - retained_df: All retained elements (8,119 rows)
        - barrier_df: Barrier objects
        - connector_df: Validated connectors with subtypes
        - obstacle_df: Recalibrated obstacles with subcategories
        - bbox_df: Bounding box samples in metres
        - ifc_paths: Resolved IFC subset paths
        - ifc_raw_paths: Resolved raw IFC paths
        - levels: Level configuration from config
    """
    data_cfg = config["data"]
    # Resolve relative to experiment root (parent of src/)
    _experiment_root = Path(__file__).resolve().parent.parent
    prep_root = (_experiment_root / data_cfg["preprocessing_root"]).resolve()

    # Load CSVs
    retained_df = pd.read_csv(_resolve_path(prep_root, data_cfg["retained_csv"]))
    barrier_df = pd.read_csv(_resolve_path(prep_root, data_cfg["barrier_csv"]))
    connector_df = pd.read_csv(_resolve_path(prep_root, data_cfg["connector_csv"]))
    obstacle_df = pd.read_csv(_resolve_path(prep_root, data_cfg["obstacle_csv"]))
    bbox_df = pd.read_csv(_resolve_path(prep_root, data_cfg["bbox_csv"]))

    # Resolve IFC paths
    ifc_paths = {}
    for key, rel in data_cfg["ifc_subsets"].items():
        ifc_paths[key] = _resolve_path(prep_root, rel)
    
    ifc_raw_paths = {}
    for key, rel in data_cfg["ifc_raw"].items():
        p = (prep_root / rel).resolve()
        if p.exists():
            ifc_raw_paths[key] = p

    # Merge bbox data onto obstacle/connector for geometry
    # (bbox_df has min_x, max_x, min_y, max_y, min_z, max_z per guid)
    if "min_x" not in obstacle_df.columns and "guid" in obstacle_df.columns:
        bbox_cols = ["guid", "source_file", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
        avail = [c for c in bbox_cols if c in bbox_df.columns]
        obstacle_df = obstacle_df.merge(
            bbox_df[avail].drop_duplicates(subset=["guid", "source_file"]),
            on=["guid", "source_file"],
            how="left",
        )

    if "min_x" not in connector_df.columns and "guid" in connector_df.columns:
        bbox_cols = ["guid", "source_file", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
        avail = [c for c in bbox_cols if c in bbox_df.columns]
        connector_df = connector_df.merge(
            bbox_df[avail].drop_duplicates(subset=["guid", "source_file"]),
            on=["guid", "source_file"],
            how="left",
        )

    levels = config["station"]["levels"]

    return {
        "retained_df": retained_df,
        "barrier_df": barrier_df,
        "connector_df": connector_df,
        "obstacle_df": obstacle_df,
        "bbox_df": bbox_df,
        "ifc_paths": ifc_paths,
        "ifc_raw_paths": ifc_raw_paths,
        "levels": levels,
    }


def filter_obstacles_for_navigation(obstacle_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only obstacles relevant for navigation (from v3 recalibration).

    Keeps: obstacle_floor_intrusive, obstacle_barrier_relevant,
           obstacle_clearance_relevant.
    Drops: obstacle_skin_panel, obstacle_uncertain, obstacle_small_irrelevant.
    """
    keep_cats = [
        "obstacle_floor_intrusive",
        "obstacle_barrier_relevant",
        "obstacle_clearance_relevant",
    ]
    return obstacle_df[obstacle_df["obstacle_subcat"].isin(keep_cats)].copy()


def filter_connectors_for_navigation(connector_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only navigation-relevant connectors (exclude F2 technical doors).

    Keeps: stair, stair_flight, escalator, elevator.
    Drops: f2_technical_door (reclassified as non-connector).
    """
    drop_subtypes = ["f2_technical_door"]
    return connector_df[~connector_df["connector_subtype"].isin(drop_subtypes)].copy()


def get_level_elements(
    df: pd.DataFrame,
    level_key: str,
    levels: dict,
) -> pd.DataFrame:
    """Filter DataFrame to elements belonging to a specific station level.
    
    Matches on storey_name column using the level's Chinese name pattern.
    """
    level_info = levels[level_key]
    name_cn = level_info["name_cn"]
    # storey_name format: "F1 站台层", "F2 设备层", etc.
    pattern = f"{level_key} {name_cn}"
    return df[df["storey_name"] == pattern].copy()


def save_step0_outputs(
    data: dict,
    nav_obstacles: pd.DataFrame,
    nav_connectors: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    """Save Step 0 summary to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_retained": len(data["retained_df"]),
        "total_obstacles": len(data["obstacle_df"]),
        "total_connectors": len(data["connector_df"]),
        "total_bbox": len(data["bbox_df"]),
        "nav_obstacles_keep": len(nav_obstacles),
        "nav_connectors_keep": len(nav_connectors),
        "ifc_subsets_available": list(data["ifc_paths"].keys()),
        "ifc_raw_available": list(data["ifc_raw_paths"].keys()),
        "levels": {k: v for k, v in data["levels"].items()},
    }
    dump_json(out_dir / "data_summary.json", summary)

    # Save filtered navigation subsets
    nav_obstacles.to_csv(out_dir / "nav_obstacles.csv", index=False)
    nav_connectors.to_csv(out_dir / "nav_connectors.csv", index=False)

    # Per-level element counts
    level_counts = {}
    for level_key in data["levels"]:
        level_info = data["levels"][level_key]
        pattern = f"{level_key} {level_info['name_cn']}"
        n_ret = len(data["retained_df"][data["retained_df"]["storey_name"] == pattern])
        n_obs = len(data["obstacle_df"][data["obstacle_df"]["storey_name"] == pattern])
        n_con = len(data["connector_df"][data["connector_df"]["storey_name"] == pattern])
        level_counts[level_key] = {
            "retained": n_ret,
            "obstacles": n_obs,
            "connectors": n_con,
        }
    dump_json(out_dir / "level_counts.json", level_counts)

    print(f"[Step 0] Data loaded: {summary['total_retained']} retained, "
          f"{summary['total_obstacles']} obstacles, {summary['total_connectors']} connectors")
