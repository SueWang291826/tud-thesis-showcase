"""
Unit Normalization Module (v2).

Converts all experiment-facing data products to use metres as the
canonical length unit.

Key facts:
- Source IFC files use MILLI METRE (IFC2X3 with MILLI prefix).
- ifcopenshell.geom.create_shape returns vertices in METRES (SI).
  This was verified empirically: element 止步块400x400 → bbox dx=0.4m.
- Therefore bbox data from v1 geometry_checks and proxy_audit is
  ALREADY in metres and needs no conversion.
- Storey elevations in v1 config are in millimetres → must be ÷ 1000.
- Any raw IFC coordinate accessed directly (not via geom) would be in mm.

This module:
1. Generates a metre-normalised storey reference table.
2. Adds unit metadata columns to exported DataFrames.
3. Ensures all downstream v2 exports carry explicit unit annotations.
4. Documents the unit provenance chain for thesis reproducibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import save_json, save_dataframe

logger = logging.getLogger(__name__)

MM_TO_M = 0.001


def build_metre_storey_table(config: Dict[str, Any]) -> pd.DataFrame:
    """Build a storey reference table with elevations in metres.

    Args:
        config: v1 pipeline config containing storeys section.

    Returns:
        DataFrame with storey metadata in metres.
    """
    storeys = config.get("storeys", {})
    records = []
    for floor_id, ref in sorted(storeys.items()):
        elev_mm = ref.get("elevation_mm", 0)
        records.append({
            "floor_id": floor_id,
            "name_cn": ref.get("name_cn", ""),
            "name_en": ref.get("name_en", ""),
            "storey_name": f"{floor_id} {ref.get('name_cn', '')}",
            "elevation_mm": elev_mm,
            "elevation_m": round(elev_mm * MM_TO_M, 4),
            "is_public": ref.get("is_public", False),
            "role": ref.get("role", "unknown"),
            "unit": "metre",
        })
    return pd.DataFrame(records)


def normalize_bbox_columns(
    df: pd.DataFrame,
    bbox_prefix: str = "bbox_",
    source_unit: str = "metre",
) -> pd.DataFrame:
    """Ensure bbox columns have explicit unit annotation.

    If bbox data comes from ifcopenshell.geom (source_unit=metre), just adds
    a unit column.  If source_unit=millimetre, divides by 1000.

    Args:
        df: DataFrame with bbox columns.
        bbox_prefix: Column name prefix for bbox fields.
        source_unit: Unit of the incoming bbox data.

    Returns:
        DataFrame with unit-annotated bbox columns.
    """
    df = df.copy()
    bbox_cols = [c for c in df.columns if c.startswith(bbox_prefix)]

    if source_unit == "millimetre":
        for col in bbox_cols:
            if df[col].dtype in ("float64", "float32", "int64"):
                df[col] = df[col] * MM_TO_M
        logger.info(f"  Converted {len(bbox_cols)} bbox columns from mm to m")
    elif source_unit == "metre":
        logger.info(f"  BBox columns already in metres ({len(bbox_cols)} columns)")

    df["bbox_unit"] = "metre"
    return df


def normalize_classified_elements(
    classified_dfs: Dict[str, pd.DataFrame],
    storey_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine and unit-annotate classified element tables.

    Adds storey elevation in metres and unit metadata.

    Args:
        classified_dfs: Per-file classified DataFrames from v1.
        storey_table: Storey reference table in metres.

    Returns:
        Combined metre-normalised DataFrame.
    """
    combined = pd.concat(
        [df.assign(source_file=label) for label, df in classified_dfs.items()],
        ignore_index=True,
    )

    # Merge storey elevation in metres
    storey_elev = storey_table[["storey_name", "elevation_m"]].drop_duplicates()
    combined = combined.merge(storey_elev, on="storey_name", how="left")

    combined["coordinate_unit"] = "metre"
    combined["elevation_unit"] = "metre"

    return combined


def normalize_bbox_table(
    bbox_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine and unit-annotate bbox sample tables.

    BBox data from ifcopenshell.geom is already in metres.
    """
    if not bbox_dfs:
        return pd.DataFrame()

    combined = pd.concat(
        [df.assign(source_file=label) for label, df in bbox_dfs.items()],
        ignore_index=True,
    )
    combined = normalize_bbox_columns(combined, source_unit="metre")
    return combined


def normalize_proxy_table(
    proxy_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine and unit-annotate proxy inventory tables.

    BBox data from ifcopenshell.geom is already in metres.
    """
    if not proxy_dfs:
        return pd.DataFrame()

    combined = pd.concat(
        [df.assign(source_file=label) for label, df in proxy_dfs.items()],
        ignore_index=True,
    )
    combined = normalize_bbox_columns(combined, source_unit="metre")
    return combined


def generate_unit_provenance(
    config: Dict[str, Any],
    v2_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a unit provenance document for thesis reproducibility.

    Returns:
        Dictionary documenting the unit conversion chain.
    """
    return {
        "source_ifc_schema": config.get("ifc", {}).get("expected_schema", "IFC2X3"),
        "source_ifc_length_unit": "MILLI METRE",
        "source_ifc_unit_factor": "1 IFC unit = 1 mm = 0.001 m",
        "ifcopenshell_geom_output_unit": "metre (SI default)",
        "verification": (
            "Empirically verified: element '止步块400x400' (400mm tile) "
            "yields bbox dx=0.4 in ifcopenshell.geom output, confirming metres."
        ),
        "downstream_canonical_unit": "metre",
        "storey_elevation_conversion": "v1 config mm → v2 outputs m (÷1000)",
        "bbox_conversion": "none needed — ifcopenshell.geom already outputs metres",
        "exported_ifc_unit": (
            "Filtered IFC files retain original mm unit system. "
            "IFC geometry is NOT rescaled to avoid corruption. "
            "Only downstream CSV/JSON exports use metres."
        ),
        "columns_in_metres": [
            "elevation_m",
            "bbox_min_x", "bbox_max_x", "bbox_min_y", "bbox_max_y",
            "bbox_min_z", "bbox_max_z", "bbox_dx", "bbox_dy", "bbox_dz",
        ],
    }


def run_unit_normalization(
    classified_dfs: Dict[str, pd.DataFrame],
    bbox_dfs: Dict[str, pd.DataFrame],
    proxy_dfs: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    v2_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the full unit normalization step.

    Args:
        classified_dfs: v1 classification results.
        bbox_dfs: v1 bbox sample tables.
        proxy_dfs: v1 proxy inventories.
        config: v1 pipeline config.
        v2_config: v2 pipeline config.
        output_dir: Output directory for normalised products.

    Returns:
        Dictionary of normalised DataFrames and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Unit Normalization (v2)")
    logger.info("=" * 60)

    # 1. Storey reference in metres
    storey_table = build_metre_storey_table(config)
    save_dataframe(storey_table, output_dir / "storey_reference_metres.csv")
    logger.info(f"  Storey reference: {len(storey_table)} storeys, elevations in metres")

    # 2. Normalised classified elements
    norm_elements = normalize_classified_elements(classified_dfs, storey_table)
    save_dataframe(norm_elements, output_dir / "elements_classified_metres.csv")
    logger.info(f"  Normalised elements: {len(norm_elements)} rows")

    # 3. Normalised bbox table
    norm_bbox = normalize_bbox_table(bbox_dfs)
    if len(norm_bbox) > 0:
        save_dataframe(norm_bbox, output_dir / "bbox_samples_metres.csv")
        logger.info(f"  Normalised bbox samples: {len(norm_bbox)} rows")

    # 4. Normalised proxy table
    norm_proxy = normalize_proxy_table(proxy_dfs)
    if len(norm_proxy) > 0:
        save_dataframe(norm_proxy, output_dir / "proxy_inventory_metres.csv")
        logger.info(f"  Normalised proxy inventory: {len(norm_proxy)} rows")

    # 5. Unit provenance document
    provenance = generate_unit_provenance(config, v2_config)
    save_json(provenance, output_dir / "unit_provenance.json")
    logger.info("  Unit provenance document saved")

    return {
        "storey_table": storey_table,
        "norm_elements": norm_elements,
        "norm_bbox": norm_bbox,
        "norm_proxy": norm_proxy,
        "provenance": provenance,
    }
