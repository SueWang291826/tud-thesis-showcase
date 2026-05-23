"""
Proxy Audit Module.

Dedicated analysis of IfcBuildingElementProxy and similar ambiguous elements.
This is a critical experimental component because:
- Many metro station elements are modelled as proxies
- Proxy handling may become a thesis contribution
- Proxies need individual analysis by name, type, geometry, and storey

Outputs:
- Detailed proxy inventory (CSV) with editable category field
- Proxy statistics and distributions
- Support for re-ingesting manually edited proxy labels
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ifc_loader import (
    IFCFileInfo,
    get_all_products,
    get_element_properties,
    get_storey_for_element,
)
from .semantic_classifier import SemanticClassifier
from .utils import save_json, save_dataframe

logger = logging.getLogger(__name__)


def _get_element_bbox(element, settings=None) -> Optional[Dict[str, float]]:
    """Attempt to extract bounding box for an element.

    Uses ifcopenshell.geom if available, falls back to None.

    Returns:
        Dictionary with min/max coordinates or None if extraction fails.
    """
    try:
        import ifcopenshell.geom

        if settings is None:
            settings = ifcopenshell.geom.settings()
            settings.set("use-world-coords", True)

        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = shape.geometry.verts
        if not verts:
            return None

        # Vertices are flat: [x0, y0, z0, x1, y1, z1, ...]
        xs = verts[0::3]
        ys = verts[1::3]
        zs = verts[2::3]

        return {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
            "dx": max(xs) - min(xs),
            "dy": max(ys) - min(ys),
            "dz": max(zs) - min(zs),
        }
    except Exception:
        return None


def audit_proxies_single_file(
    file_info: IFCFileInfo,
    classifier: SemanticClassifier,
    output_dir: Path,
    extract_bbox: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform detailed proxy audit for a single IFC file.

    Args:
        file_info: Loaded IFC file.
        classifier: Semantic classifier for candidate category inference.
        output_dir: Output directory.
        extract_bbox: Whether to attempt bounding box extraction.

    Returns:
        Tuple of (proxy DataFrame, summary dictionary).
    """
    label = file_info.label
    model = file_info.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Proxy audit: {label}")

    storey_lookup = {s.guid: s.decoded_name for s in file_info.storeys}
    products = get_all_products(model)

    # Filter to proxy elements
    proxy_classes = {"IfcBuildingElementProxy"}
    proxies = [p for p in products if p.is_a() in proxy_classes]

    logger.info(f"  Found {len(proxies)} proxy elements out of {len(products)} products")

    # Optionally set up geometry settings once
    geom_settings = None
    if extract_bbox:
        try:
            import ifcopenshell.geom
            geom_settings = ifcopenshell.geom.settings()
            geom_settings.set("use-world-coords", True)
        except Exception as e:
            logger.warning(f"  Cannot initialize geometry: {e}. Skipping bbox extraction.")
            extract_bbox = False

    records = []
    bbox_success = 0
    bbox_fail = 0

    for i, elem in enumerate(proxies):
        props = get_element_properties(elem)
        storey_guid = get_storey_for_element(elem, model)
        storey_name = storey_lookup.get(storey_guid, "UNASSIGNED") if storey_guid else "UNASSIGNED"

        # Classify
        category, rule_source, note = classifier.classify(
            ifc_class=props["ifc_class"],
            name=props["name"],
            object_type=props["object_type"],
            predefined_type=props["predefined_type"],
        )

        # Detect geometry representation availability
        has_representation = False
        repr_types = []
        try:
            if hasattr(elem, "Representation") and elem.Representation:
                has_representation = True
                for rep in elem.Representation.Representations:
                    repr_types.append(rep.RepresentationType or "Unknown")
        except Exception:
            pass

        record = {
            "guid": props["guid"],
            "name": props["name"],
            "object_type": props["object_type"],
            "predefined_type": props["predefined_type"],
            "description": props["description"],
            "tag": props["tag"],
            "storey_name": storey_name,
            "has_representation": has_representation,
            "representation_types": "; ".join(repr_types) if repr_types else "",
            "inferred_category": category,
            "rule_source": rule_source,
            "rule_note": note,
            # Editable field for manual review
            "reviewed_category": "",
            "reviewer_notes": "",
        }

        # Bounding box
        if extract_bbox and has_representation:
            bbox = _get_element_bbox(elem, geom_settings)
            if bbox:
                record.update({
                    f"bbox_{k}": round(v, 2) for k, v in bbox.items()
                })
                bbox_success += 1
            else:
                bbox_fail += 1
        elif extract_bbox:
            bbox_fail += 1

        records.append(record)

        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{len(proxies)} proxies...")

    proxy_df = pd.DataFrame(records)

    # ---- Save proxy inventory ----
    save_dataframe(proxy_df, output_dir / f"proxy_inventory_{label}.csv")
    logger.info(f"  Saved proxy inventory: {len(proxy_df)} records")

    # ---- Statistics ----
    inferred_dist = proxy_df["inferred_category"].value_counts().to_dict() if len(proxy_df) > 0 else {}
    storey_dist = proxy_df["storey_name"].value_counts().to_dict() if len(proxy_df) > 0 else {}
    name_dist = proxy_df["name"].value_counts().to_dict() if len(proxy_df) > 0 else {}
    type_dist = proxy_df["object_type"].value_counts().to_dict() if len(proxy_df) > 0 else {}

    summary = {
        "file_label": label,
        "total_proxies": len(proxy_df),
        "total_products": len(products),
        "proxy_fraction": round(len(proxy_df) / len(products), 4) if products else 0,
        "bbox_extraction": {
            "attempted": extract_bbox,
            "success": bbox_success,
            "fail": bbox_fail,
        },
        "inferred_category_distribution": inferred_dist,
        "storey_distribution": storey_dist,
        "top_names": dict(list(sorted(name_dist.items(), key=lambda x: -x[1]))[:30]),
        "top_object_types": dict(list(sorted(type_dist.items(), key=lambda x: -x[1]))[:30]),
        "has_representation_count": int(proxy_df["has_representation"].sum()) if len(proxy_df) > 0 else 0,
    }

    save_json(summary, output_dir / f"proxy_summary_{label}.json")
    logger.info(
        f"  Proxy categories: "
        + ", ".join(f"{k}={v}" for k, v in sorted(inferred_dist.items(), key=lambda x: -x[1]))
    )

    return proxy_df, summary


def generate_cross_file_proxy_summary(
    all_proxy_dfs: Dict[str, pd.DataFrame],
    all_summaries: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate combined proxy analysis across all files.

    Args:
        all_proxy_dfs: Per-file proxy DataFrames.
        all_summaries: Per-file proxy summaries.
        output_dir: Output directory.

    Returns:
        Cross-file proxy summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating cross-file proxy summary")

    # Combine all proxies
    combined = pd.concat(
        [df.assign(file_label=label) for label, df in all_proxy_dfs.items()],
        ignore_index=True,
    )
    save_dataframe(combined, output_dir / "proxy_inventory_combined.csv")

    # Cross-file statistics
    total_proxies = sum(s["total_proxies"] for s in all_summaries.values())
    total_products = sum(s["total_products"] for s in all_summaries.values())

    cross_summary = {
        "total_proxies": total_proxies,
        "total_products": total_products,
        "overall_proxy_fraction": round(total_proxies / total_products, 4) if total_products else 0,
        "per_file": {
            label: {
                "proxies": s["total_proxies"],
                "fraction": s["proxy_fraction"],
            }
            for label, s in all_summaries.items()
        },
        "combined_category_distribution": combined["inferred_category"].value_counts().to_dict()
        if len(combined) > 0 else {},
        "combined_storey_distribution": combined["storey_name"].value_counts().to_dict()
        if len(combined) > 0 else {},
    }

    save_json(cross_summary, output_dir / "proxy_cross_file_summary.json")
    logger.info(f"  Combined proxy analysis: {total_proxies} proxies across {len(all_proxy_dfs)} files")

    return cross_summary


def reingest_proxy_labels(
    proxy_csv_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Re-ingest a manually edited proxy inventory CSV.

    After manual review, the 'reviewed_category' column may have been filled.
    This function loads the edited file and produces updated statistics.

    Args:
        proxy_csv_path: Path to the edited proxy CSV.
        output_dir: Output directory for updated results.

    Returns:
        Updated proxy DataFrame with final categories.
    """
    logger.info(f"Re-ingesting proxy labels from: {proxy_csv_path}")

    df = pd.read_csv(proxy_csv_path, encoding="utf-8-sig")

    # Merge: use reviewed_category if filled, else keep inferred_category
    if "reviewed_category" in df.columns and "inferred_category" in df.columns:
        df["final_category"] = df.apply(
            lambda row: row["reviewed_category"]
            if pd.notna(row["reviewed_category"]) and str(row["reviewed_category"]).strip()
            else row["inferred_category"],
            axis=1,
        )
    else:
        logger.warning("Expected columns 'reviewed_category' and 'inferred_category' not found.")
        return df

    # Updated statistics
    final_dist = df["final_category"].value_counts().to_dict()
    reviewed_count = df[
        df["reviewed_category"].notna() & (df["reviewed_category"].str.strip() != "")
    ].shape[0]

    summary = {
        "total_proxies": len(df),
        "manually_reviewed": reviewed_count,
        "review_fraction": round(reviewed_count / len(df), 4) if len(df) else 0,
        "final_category_distribution": final_dist,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "proxy_reingest_summary.json")
    save_dataframe(df, output_dir / "proxy_final.csv")

    logger.info(
        f"  Re-ingested: {reviewed_count}/{len(df)} manually reviewed, "
        f"final categories: {final_dist}"
    )

    return df
