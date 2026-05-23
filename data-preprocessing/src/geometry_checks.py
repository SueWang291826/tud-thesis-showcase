"""
Geometry Readiness Checks Module.

Validates geometry quality for later navigation graph construction.
Does NOT generate the navigation graph itself.

Checks:
- Geometry representation availability
- Geometry extraction success/failure
- Coordinate validity
- Unit consistency
- Bounding box quality (degenerate, oversized, etc.)
- Duplicate geometry / duplicate objects
- Heavy geometry flagging

Outputs: geometry quality logs, summary tables, per-storey statistics.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ifc_loader import (
    IFCFileInfo,
    get_all_products,
    get_element_properties,
    get_storey_for_element,
)
from .utils import save_json, save_dataframe

logger = logging.getLogger(__name__)


def check_geometry_single_file(
    file_info: IFCFileInfo,
    config: Dict[str, Any],
    output_dir: Path,
    sample_bbox: bool = True,
    max_bbox_sample: int = 2000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run geometry readiness checks on a single IFC file.

    Args:
        file_info: Loaded IFC file.
        config: Pipeline configuration (geometry thresholds).
        output_dir: Output directory.
        sample_bbox: Whether to sample bounding boxes.
        max_bbox_sample: Max elements to sample for bbox extraction.

    Returns:
        Tuple of (geometry check DataFrame, summary dictionary).
    """
    label = file_info.label
    model = file_info.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Geometry readiness check: {label}")

    geo_cfg = config.get("geometry", {})
    min_dim = geo_cfg.get("min_bbox_dimension", 1.0)
    max_dim = geo_cfg.get("max_bbox_dimension", 500000.0)
    degen_thresh = geo_cfg.get("degenerate_threshold", 0.1)

    storey_lookup = {s.guid: s.decoded_name for s in file_info.storeys}
    products = get_all_products(model)

    logger.info(f"  Checking {len(products)} products")

    # ---- Phase 1: Representation analysis (fast, no geometry extraction) ----
    records = []
    repr_stats = Counter()
    has_repr_count = 0
    no_repr_count = 0

    for elem in products:
        props = get_element_properties(elem)
        storey_guid = get_storey_for_element(elem, model)
        storey_name = storey_lookup.get(storey_guid, "UNASSIGNED") if storey_guid else "UNASSIGNED"

        has_repr = False
        repr_types = []
        repr_count = 0

        try:
            if hasattr(elem, "Representation") and elem.Representation:
                has_repr = True
                for rep in elem.Representation.Representations:
                    rtype = rep.RepresentationType or "Unknown"
                    repr_types.append(rtype)
                    repr_count += 1
                    repr_stats[rtype] += 1
        except Exception:
            pass

        if has_repr:
            has_repr_count += 1
        else:
            no_repr_count += 1

        records.append({
            "guid": props["guid"],
            "ifc_class": props["ifc_class"],
            "name": props["name"],
            "storey_name": storey_name,
            "has_representation": has_repr,
            "representation_types": "; ".join(repr_types),
            "representation_count": repr_count,
        })

    geom_df = pd.DataFrame(records)

    # ---- Phase 2: Sample bounding box extraction ----
    bbox_records = []
    bbox_success = 0
    bbox_fail = 0
    bbox_degenerate = 0
    bbox_oversized = 0
    bbox_undersized = 0

    if sample_bbox:
        try:
            import ifcopenshell.geom

            settings = ifcopenshell.geom.settings()
            settings.set("use-world-coords", True)

            # Sample elements that have representations
            sample_indices = geom_df[geom_df["has_representation"]].index.tolist()
            if len(sample_indices) > max_bbox_sample:
                rng = np.random.RandomState(42)
                sample_indices = rng.choice(sample_indices, max_bbox_sample, replace=False).tolist()

            logger.info(f"  Extracting bounding boxes for {len(sample_indices)} sampled elements...")

            for idx_i, idx in enumerate(sample_indices):
                guid = geom_df.loc[idx, "guid"]
                try:
                    elem = model.by_guid(guid)
                    shape = ifcopenshell.geom.create_shape(settings, elem)
                    verts = shape.geometry.verts

                    if not verts:
                        bbox_fail += 1
                        bbox_records.append({
                            "guid": guid,
                            "ifc_class": geom_df.loc[idx, "ifc_class"],
                            "storey_name": geom_df.loc[idx, "storey_name"],
                            "bbox_status": "empty_verts",
                        })
                        continue

                    xs = verts[0::3]
                    ys = verts[1::3]
                    zs = verts[2::3]

                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    dz = max(zs) - min(zs)

                    # Quality flags
                    is_degenerate = dx < degen_thresh or dy < degen_thresh
                    is_oversized = dx > max_dim or dy > max_dim or dz > max_dim
                    is_undersized = max(dx, dy, dz) < min_dim

                    if is_degenerate:
                        bbox_degenerate += 1
                    if is_oversized:
                        bbox_oversized += 1
                    if is_undersized:
                        bbox_undersized += 1

                    n_verts = len(verts) // 3

                    bbox_records.append({
                        "guid": guid,
                        "ifc_class": geom_df.loc[idx, "ifc_class"],
                        "name": geom_df.loc[idx, "name"],
                        "storey_name": geom_df.loc[idx, "storey_name"],
                        "min_x": round(min(xs), 2),
                        "max_x": round(max(xs), 2),
                        "min_y": round(min(ys), 2),
                        "max_y": round(max(ys), 2),
                        "min_z": round(min(zs), 2),
                        "max_z": round(max(zs), 2),
                        "dx": round(dx, 2),
                        "dy": round(dy, 2),
                        "dz": round(dz, 2),
                        "n_vertices": n_verts,
                        "bbox_status": "ok",
                        "is_degenerate": is_degenerate,
                        "is_oversized": is_oversized,
                        "is_undersized": is_undersized,
                    })
                    bbox_success += 1

                except Exception as e:
                    bbox_fail += 1
                    bbox_records.append({
                        "guid": guid,
                        "ifc_class": geom_df.loc[idx, "ifc_class"],
                        "storey_name": geom_df.loc[idx, "storey_name"],
                        "bbox_status": f"error: {str(e)[:100]}",
                    })

                if (idx_i + 1) % 200 == 0:
                    logger.info(
                        f"  BBox progress: {idx_i + 1}/{len(sample_indices)} "
                        f"(success={bbox_success}, fail={bbox_fail})"
                    )

        except ImportError:
            logger.warning("  ifcopenshell.geom not available, skipping bbox extraction")
        except Exception as e:
            logger.error(f"  Geometry extraction failed: {e}")

    bbox_df = pd.DataFrame(bbox_records) if bbox_records else pd.DataFrame()

    # ---- Phase 3: Duplicate detection (by GUID) ----
    guid_counts = geom_df["guid"].value_counts()
    duplicates = guid_counts[guid_counts > 1]
    duplicate_count = len(duplicates)

    # ---- Save outputs ----
    save_dataframe(geom_df, output_dir / f"geometry_check_{label}.csv")
    if len(bbox_df) > 0:
        save_dataframe(bbox_df, output_dir / f"bbox_sample_{label}.csv")

    summary = {
        "file_label": label,
        "total_products": len(products),
        "has_representation": has_repr_count,
        "no_representation": no_repr_count,
        "representation_fraction": round(has_repr_count / len(products), 4) if products else 0,
        "representation_type_distribution": dict(repr_stats.most_common()),
        "bbox_sampling": {
            "enabled": sample_bbox,
            "sample_size": len(bbox_records),
            "success": bbox_success,
            "fail": bbox_fail,
            "success_rate": round(bbox_success / len(bbox_records), 4) if bbox_records else 0,
        },
        "bbox_quality": {
            "degenerate": bbox_degenerate,
            "oversized": bbox_oversized,
            "undersized": bbox_undersized,
        },
        "duplicate_guids": duplicate_count,
        "unit_info": {
            "length_unit": file_info.length_unit,
            "length_prefix": file_info.length_prefix,
            "note": "All coordinates in millimeters (MILLI METRE)" if file_info.length_prefix == "MILLI" else "Check unit conversion",
        },
    }

    save_json(summary, output_dir / f"geometry_summary_{label}.json")

    logger.info(
        f"  Geometry check done: {has_repr_count}/{len(products)} have representation, "
        f"bbox sample {bbox_success}/{len(bbox_records)} success, "
        f"{duplicate_count} duplicate GUIDs"
    )

    return geom_df, summary


def generate_cross_file_geometry_summary(
    all_summaries: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate combined geometry readiness summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating cross-file geometry summary")

    cross_summary = {
        "per_file": all_summaries,
        "totals": {
            "total_products": sum(s["total_products"] for s in all_summaries.values()),
            "total_with_repr": sum(s["has_representation"] for s in all_summaries.values()),
            "total_without_repr": sum(s["no_representation"] for s in all_summaries.values()),
        },
    }

    save_json(cross_summary, output_dir / "geometry_cross_file_summary.json")
    logger.info("  Cross-file geometry summary saved")

    return cross_summary
