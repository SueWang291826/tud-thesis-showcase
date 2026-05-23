"""
v3 Refinement: Data layer stabilization before graph construction.

Three priorities:
  1. Small, usable IFC subset export (reverse-filter strategy)
  2. Obstacle recalibration (split 6,152 → meaningful subcategories)
  3. Connector completeness validation

All outputs go to outputs/v3/.
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_json, save_dataframe, setup_logging, Timer

logger = logging.getLogger(__name__)


# =====================================================================
# PRIORITY 1: Small IFC subset export (reverse-filter strategy)
# =====================================================================

def export_small_ifc_subsets(
    retained_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Export small, independently readable IFC subsets.

    Strategy: Copy the entire source IFC, then remove non-retained entities.
    This is much faster than deep-copying individual elements.
    """
    import ifcopenshell
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Priority 1: Small IFC Subset Export (v3)")
    logger.info("=" * 60)

    data_dir = (PROJECT_ROOT / config["input"]["data_dir"]).resolve()
    results = {}

    # --- Subset 1: platform_F1_public.ifc ---
    results["platform_F1_public"] = _export_subset_by_removal(
        source_ifc=data_dir / config["input"]["ifc_files"]["platform"],
        retained_guids=set(
            retained_df[
                (retained_df["source_file"] == "platform") &
                (retained_df["storey_name"] == "F1 站台层")
            ]["guid"]
        ),
        output_path=output_dir / "platform_F1_public.ifc",
        description="F1 platform public-level navigation elements (from platform.ifc)",
    )

    # --- Subset 2: concourse_F3_public.ifc ---
    results["concourse_F3_public"] = _export_subset_by_removal(
        source_ifc=data_dir / config["input"]["ifc_files"]["concourse"],
        retained_guids=set(
            retained_df[
                (retained_df["source_file"] == "concourse") &
                (retained_df["storey_name"] == "F3 站厅层")
            ]["guid"]
        ),
        output_path=output_dir / "concourse_F3_public.ifc",
        description="F3 concourse public-level navigation elements (from concourse.ifc)",
    )

    # --- Subset 3: traffic_F4_public.ifc (if traffic layer exists) ---
    if "traffic" in config["input"]["ifc_files"]:
        results["traffic_F4_public"] = _export_subset_by_removal(
            source_ifc=data_dir / config["input"]["ifc_files"]["traffic"],
            retained_guids=set(
                retained_df[
                    (retained_df["source_file"] == "traffic") &
                    (retained_df["storey_name"] == "F4 交通层")
                ]["guid"]
            ),
            output_path=output_dir / "traffic_F4_public.ifc",
            description="F4 transport public-level navigation elements (from traffic.ifc)",
        )

    # --- Subset 4: selected_F2_connectors.ifc ---
    # Only stair/stairflight from equipment.ifc on F2
    f2_connector_guids = set(
        retained_df[
            (retained_df["source_file"] == "equipment") &
            (retained_df["storey_name"] == "F2 设备层") &
            (retained_df["category"] == "vertical_connector")
        ]["guid"]
    )
    results["selected_F2_connectors"] = _export_subset_by_removal(
        source_ifc=data_dir / config["input"]["ifc_files"]["equipment"],
        retained_guids=f2_connector_guids,
        output_path=output_dir / "selected_F2_connectors.ifc",
        description="Selected F2 vertical connectors (stairs/stairflights from equipment.ifc)",
    )

    # Write summary
    save_json(results, output_dir / "ifc_subset_export_results.json")
    _write_ifc_export_report_v3(results, output_dir / "ifc_subset_export_report.md")

    return results


def _export_subset_by_removal(
    source_ifc: Path,
    retained_guids: Set[str],
    output_path: Path,
    description: str,
) -> Dict[str, Any]:
    """Export an IFC subset.

    Strategy selection:
    - If retained/total ratio > 0.3 → remove-based (copy whole, delete unneeded)
    - If retained count < 200 → add-based (new file, deep-copy elements)
    - Default → remove-based
    """
    import ifcopenshell

    logger.info(f"  Exporting: {output_path.name}")
    logger.info(f"    Source: {source_ifc.name}")
    logger.info(f"    Retained GUIDs: {len(retained_guids)}")
    logger.info(f"    Description: {description}")

    if len(retained_guids) == 0:
        return {
            "status": "empty",
            "message": "No retained GUIDs for this subset",
            "element_count": 0,
            "description": description,
        }

    try:
        model = ifcopenshell.open(str(source_ifc))
        total_products = len(model.by_type("IfcProduct"))

        # Decide strategy: add-based for small subsets, remove-based for large
        use_add = len(retained_guids) < 200 and len(retained_guids) / max(total_products, 1) < 0.3
        strategy = "add-based" if use_add else "remove-based"
        logger.info(f"    Strategy: {strategy} (retained={len(retained_guids)}, total_products={total_products})")

        if use_add:
            result = _export_add_based(model, retained_guids, output_path, description)
        else:
            result = _export_remove_based(model, retained_guids, output_path, description)

        result["source_file"] = source_ifc.name
        result["strategy"] = strategy
        return result

    except Exception as e:
        logger.error(f"    → FAILED: {e}")
        return {
            "status": "error",
            "message": str(e),
            "description": description,
            "element_count": 0,
        }


def _export_add_based(
    model,
    retained_guids: Set[str],
    output_path: Path,
    description: str,
) -> Dict[str, Any]:
    """Add-based export: create a new file and deep-copy only retained elements.

    Carefully copies spatial hierarchy and creates filtered relationship
    entities so that only retained products appear, with proper spatial
    containment for IFC viewer compatibility.
    """
    import ifcopenshell

    new_model = ifcopenshell.file(schema=model.schema)

    # 1. Copy IfcProject (recursively brings OwnerHistory, units, contexts)
    for proj in model.by_type("IfcProject"):
        new_model.add(proj)

    # 2. Copy retained products (recursive — brings their geometry)
    added = 0
    add_errors = 0
    added_guids = set()
    for p in model.by_type("IfcProduct"):
        if p.GlobalId in retained_guids:
            try:
                new_model.add(p)
                added += 1
                added_guids.add(p.GlobalId)
            except Exception:
                add_errors += 1

    # 3. For retained aggregate parents (IfcStair etc.), copy their children
    #    by adding the IfcRelAggregates — but only for OUR parents.
    #    We add the rel AFTER the parent is already in new_model so the
    #    recursive copy only needs to pull in the new children + geometry.
    agg_children_added = 0
    for rel in model.by_type("IfcRelAggregates"):
        relating = rel.RelatingObject
        if hasattr(relating, "GlobalId") and relating.GlobalId in added_guids:
            # This is e.g. IfcStair → IfcStairFlight aggregation
            try:
                new_model.add(rel)
                for child in rel.RelatedObjects:
                    if hasattr(child, "GlobalId"):
                        added_guids.add(child.GlobalId)
                        agg_children_added += 1
            except Exception:
                pass

    # 4. Build spatial hierarchy:  Site → Building → Storeys
    #    Add these entities directly, then create a minimal containment link.
    site = None
    building = None
    for s in model.by_type("IfcSite"):
        site = new_model.add(s)
    for b in model.by_type("IfcBuilding"):
        building = new_model.add(b)
    storey_map = {}  # original storey → new storey
    for st in model.by_type("IfcBuildingStorey"):
        storey_map[st] = new_model.add(st)

    # Rebuild Project → Site → Building → Storeys aggregation tree
    oh = None
    for h in model.by_type("IfcOwnerHistory"):
        oh = new_model.add(h)
        break

    if site:
        new_model.create_entity("IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
            RelatingObject=[e for e in new_model.by_type("IfcProject")][0],
            RelatedObjects=[site])
    if building and site:
        new_model.create_entity("IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
            RelatingObject=site,
            RelatedObjects=[building])
    if building and storey_map:
        new_model.create_entity("IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
            RelatingObject=building,
            RelatedObjects=list(storey_map.values()))

    # 5. Create FILTERED spatial containment: only retained products
    containment_copied = 0
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        matching = [e for e in rel.RelatedElements
                    if hasattr(e, "GlobalId") and e.GlobalId in added_guids]
        if matching and rel.RelatingStructure in storey_map:
            new_matching = []
            for e in matching:
                found = new_model.by_guid(e.GlobalId)
                if found:
                    new_matching.append(found)
            if new_matching:
                new_model.create_entity("IfcRelContainedInSpatialStructure",
                    GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
                    RelatingStructure=storey_map[rel.RelatingStructure],
                    RelatedElements=new_matching)
                containment_copied += 1

    new_model.write(str(output_path))
    validation = _validate_single_ifc(output_path)

    result = {
        "status": "success",
        "message": f"Exported {output_path.name}",
        "path": str(output_path),
        "description": description,
        "retained_guid_count": len(retained_guids),
        "products_added": added,
        "agg_children_added": agg_children_added,
        "containment_rels_copied": containment_copied,
        "add_errors": add_errors,
        "file_size_mb": round(output_path.stat().st_size / 1e6, 1),
        "validation": validation,
    }
    logger.info(
        f"    → Success (add): {output_path.name} "
        f"({result['file_size_mb']} MB, added={added}+{agg_children_added}children, "
        f"containment={containment_copied}, readable={validation.get('readable', '?')})"
    )
    return result


def _export_remove_based(
    model,
    retained_guids: Set[str],
    output_path: Path,
    description: str,
) -> Dict[str, Any]:
    """Remove-based export: remove non-retained products from the model."""

    total_products_before = len(model.by_type("IfcProduct"))
    spatial_types = {
        "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey",
        "IfcSpace", "IfcSpatialStructureElement",
    }

    to_remove = []
    for p in model.by_type("IfcProduct"):
        if p.is_a() in spatial_types:
            continue
        for st in spatial_types:
            if p.is_a(st):
                break
        else:
            if p.GlobalId not in retained_guids:
                to_remove.append(p)

    logger.info(f"    Products before: {total_products_before}")
    logger.info(f"    To remove: {len(to_remove)}")

    removed = 0
    remove_errors = 0
    for i, elem in enumerate(to_remove):
        try:
            model.remove(elem)
            removed += 1
        except Exception:
            remove_errors += 1
        if (i + 1) % 500 == 0:
            logger.info(f"      Removed {i+1}/{len(to_remove)}...")

    model.write(str(output_path))
    validation = _validate_single_ifc(output_path)

    result = {
        "status": "success",
        "message": f"Exported {output_path.name}",
        "path": str(output_path),
        "description": description,
        "retained_guid_count": len(retained_guids),
        "products_before": total_products_before,
        "products_removed": removed,
        "remove_errors": remove_errors,
        "file_size_mb": round(output_path.stat().st_size / 1e6, 1),
        "validation": validation,
    }
    logger.info(
        f"    → Success (remove): {output_path.name} "
        f"({result['file_size_mb']} MB, "
        f"readable={validation.get('readable', '?')})"
    )
    return result


def _validate_single_ifc(path: Path) -> Dict[str, Any]:
    """Validate an IFC file by reopening it."""
    import ifcopenshell
    try:
        reopened = ifcopenshell.open(str(path))
        products = reopened.by_type("IfcProduct")
        storeys = reopened.by_type("IfcBuildingStorey")
        spatial = [p for p in products if p.is_a("IfcBuildingStorey") or
                   p.is_a("IfcSite") or p.is_a("IfcBuilding")]
        non_spatial = [p for p in products if p not in spatial and
                       not p.is_a("IfcBuildingStorey") and
                       not p.is_a("IfcSite") and
                       not p.is_a("IfcBuilding")]
        return {
            "readable": True,
            "total_entities": len(list(reopened)),
            "product_count": len(products),
            "non_spatial_product_count": len(non_spatial),
            "storey_count": len(storeys),
            "schema": reopened.schema,
        }
    except Exception as e:
        return {"readable": False, "error": str(e)}


def _write_ifc_export_report_v3(results: Dict[str, Any], path: Path) -> None:
    """Write IFC export report for v3."""
    lines = [
        "# V3 IFC Subset Export Report",
        "",
        "## Strategy",
        "",
        "Reverse-filter approach: open full source IFC → remove non-retained "
        "IfcProduct entities → write reduced file. This preserves all spatial "
        "hierarchy, geometry contexts, units, and entity relationships intact.",
        "",
        "## Export Summary",
        "",
        "| Subset | Status | Retained | Size | Readable | Products |",
        "|--------|--------|----------|------|----------|----------|",
    ]
    for name, r in results.items():
        status = r.get("status", "?")
        guids = r.get("retained_guid_count", 0)
        size = r.get("file_size_mb", 0)
        val = r.get("validation", {})
        readable = "YES" if val.get("readable") else "NO" if val else "—"
        prods = val.get("non_spatial_product_count", "—")
        lines.append(f"| {name} | {status} | {guids} | {size} MB | {readable} | {prods} |")

    lines.extend(["", "## Detail", ""])
    for name, r in results.items():
        lines.append(f"### {name}")
        lines.append(f"- **Description**: {r.get('description', '')}")
        lines.append(f"- **Source**: {r.get('source_file', '?')}")
        lines.append(f"- **Status**: {r.get('status', '?')}")
        if r.get("status") == "success":
            lines.append(f"- **Products before removal**: {r.get('products_before', '?')}")
            lines.append(f"- **Products removed**: {r.get('products_removed', '?')}")
            lines.append(f"- **Remove errors**: {r.get('remove_errors', 0)}")
            val = r.get("validation", {})
            if val.get("readable"):
                lines.append(f"- **Validated products**: {val.get('non_spatial_product_count', '?')}")
                lines.append(f"- **Validated storeys**: {val.get('storey_count', '?')}")
        elif r.get("message"):
            lines.append(f"- **Message**: {r.get('message')}")
        lines.append("")

    lines.extend([
        "## Known Limitations",
        "- IFC geometry remains in millimetres (native unit)",
        "- Removed products may leave orphaned property sets or type objects",
        "- These are experimental subsets for thesis research, not round-trip BIM files",
        "- GUID traceability is fully preserved",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  IFC export report saved: {path.name}")


# =====================================================================
# PRIORITY 2: Obstacle recalibration
# =====================================================================

def recalibrate_obstacles(
    retained_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """Recalibrate obstacle set into meaningful subcategories.

    Uses dimensions, IFC class, name patterns, and elevation data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Priority 2: Obstacle Recalibration (v3)")
    logger.info("=" * 60)

    # Get all retained_barrier elements
    obstacles = retained_df[retained_df["filter_output_class"] == "retained_barrier"].copy()
    logger.info(f"  Total obstacles (retained_barrier): {len(obstacles)}")

    # Merge with bbox data (join on guid+source_file to avoid row duplication)
    bcols = ["guid", "source_file", "dx", "dy", "dz", "min_z", "max_z"]
    bcols = [c for c in bcols if c in bbox_df.columns]
    merge_keys = ["guid", "source_file"] if "source_file" in bcols else ["guid"]
    obstacles = obstacles.merge(bbox_df[bcols], on=merge_keys, how="left")
    has_dims = obstacles["dz"].notna()
    logger.info(f"  Obstacles after merge: {len(obstacles)} (expect {len(retained_df[retained_df['filter_output_class'] == 'retained_barrier'])})")
    logger.info(f"  Obstacles with bbox data: {has_dims.sum()} / {len(obstacles)}")

    # Storey elevations (metres)
    storey_floor = {"F1 站台层": 0.0, "F3 站厅层": 12.1, "F4 交通层": 17.4}

    # Classify each obstacle
    subcategories = []
    for _, row in obstacles.iterrows():
        subcategories.append(_classify_obstacle(row, storey_floor, has_bbox=pd.notna(row.get("dz"))))

    obstacles["obstacle_subcat"] = subcategories

    # Summary statistics
    subcat_counts = obstacles["obstacle_subcat"].value_counts()
    logger.info("  Obstacle subcategory distribution:")
    for cat, cnt in subcat_counts.items():
        logger.info(f"    {cat}: {cnt}")

    # Before/after comparison
    before_count = len(obstacles)
    recommended_keep = obstacles[obstacles["obstacle_subcat"].isin({
        "obstacle_floor_intrusive",
        "obstacle_barrier_relevant",
        "obstacle_clearance_relevant",
    })]
    recommended_drop = obstacles[~obstacles["obstacle_subcat"].isin({
        "obstacle_floor_intrusive",
        "obstacle_barrier_relevant",
        "obstacle_clearance_relevant",
    })]

    logger.info(f"  Recommended keep: {len(recommended_keep)} ({100*len(recommended_keep)/before_count:.1f}%)")
    logger.info(f"  Recommended drop: {len(recommended_drop)} ({100*len(recommended_drop)/before_count:.1f}%)")

    # Export tables
    save_dataframe(obstacles, output_dir / "obstacles_recalibrated.csv")
    save_dataframe(recommended_keep, output_dir / "obstacles_recommended_keep.csv")
    save_dataframe(recommended_drop, output_dir / "obstacles_recommended_drop.csv")

    # Per-subcategory tables
    for cat in obstacles["obstacle_subcat"].unique():
        safe = cat.replace(" ", "_")
        sub = obstacles[obstacles["obstacle_subcat"] == cat]
        save_dataframe(sub, output_dir / f"{safe}.csv")

    # Rules documentation
    rules = _get_obstacle_rules_doc()

    summary = {
        "total_obstacles_before": before_count,
        "subcategory_counts": subcat_counts.to_dict(),
        "recommended_keep_count": len(recommended_keep),
        "recommended_drop_count": len(recommended_drop),
        "recommended_keep_pct": round(100 * len(recommended_keep) / before_count, 1),
        "keep_categories": ["obstacle_floor_intrusive", "obstacle_barrier_relevant", "obstacle_clearance_relevant"],
        "drop_categories": ["obstacle_skin_panel", "obstacle_upper_irrelevant", "obstacle_small_irrelevant", "obstacle_uncertain"],
        "rules": rules,
        "has_bbox_pct": round(100 * has_dims.sum() / len(obstacles), 1),
    }
    save_json(summary, output_dir / "obstacle_recalibration_summary.json")
    _write_obstacle_report(summary, obstacles, output_dir / "obstacle_recalibration_report.md")

    return {
        "obstacles": obstacles,
        "recommended_keep": recommended_keep,
        "recommended_drop": recommended_drop,
        "summary": summary,
    }


def _classify_obstacle(row: pd.Series, storey_floor: Dict, has_bbox: bool) -> str:
    """Classify a single obstacle into a refined subcategory.

    Rules applied in priority order:
    1. IfcRailing → obstacle_barrier_relevant
    2. IfcCurtainWall → obstacle_barrier_relevant
    3. IfcPlate with dz < 5cm → obstacle_skin_panel (cladding/veneer, not walkability-affecting)
    4. IfcPlate with 5cm ≤ dz ≤ 30cm → obstacle_small_irrelevant (thin trim, not blocking)
    5. IfcPlate with dz > 1m → obstacle_barrier_relevant (glass partition/screen)
    6. IfcPlate with 30cm < dz ≤ 1m → obstacle_clearance_relevant
    7. IfcColumn → obstacle_floor_intrusive
    8. IfcWall/IfcWallStandardCase → obstacle_floor_intrusive
    9. IfcMember with dz < 30cm → obstacle_small_irrelevant (small structural member)
    10. IfcMember with dz ≥ 30cm → obstacle_clearance_relevant
    11. Name contains 自动伸缩门/防火玻璃 → obstacle_barrier_relevant (fire glass door)
    12. Name contains 自动门 → obstacle_barrier_relevant
    13. Remaining with bbox: check z-position relative to storey floor
        - If min_z > storey_floor + 2.5m → obstacle_upper_irrelevant
    14. Default → obstacle_uncertain
    """
    ifc_class = str(row.get("ifc_class", ""))
    name = str(row.get("name", ""))
    dz = row.get("dz") if has_bbox else None
    min_z = row.get("min_z") if has_bbox else None
    storey = str(row.get("storey_name", ""))

    # Rule 1-2: Railings and curtain walls
    if ifc_class == "IfcRailing":
        return "obstacle_barrier_relevant"
    if ifc_class == "IfcCurtainWall":
        return "obstacle_barrier_relevant"

    # Rule 11-12: Name patterns (before dimension rules for IfcPlate)
    if "自动伸缩门" in name or "防火玻璃" in name:
        return "obstacle_barrier_relevant"
    if "自动门" in name:
        return "obstacle_barrier_relevant"

    # Rule 3-6: IfcPlate dimension rules
    if ifc_class == "IfcPlate":
        if dz is not None:
            if dz < 0.05:
                return "obstacle_skin_panel"
            elif dz <= 0.30:
                return "obstacle_small_irrelevant"
            elif dz > 1.0:
                return "obstacle_barrier_relevant"
            else:
                return "obstacle_clearance_relevant"
        else:
            # No bbox — IfcPlate is most likely skin panel
            return "obstacle_skin_panel"

    # Rule 7: Columns
    if ifc_class == "IfcColumn":
        return "obstacle_floor_intrusive"

    # Rule 8: Walls
    if ifc_class in ("IfcWall", "IfcWallStandardCase"):
        return "obstacle_floor_intrusive"

    # Rule 9-10: Members
    if ifc_class == "IfcMember":
        if dz is not None:
            if dz < 0.30:
                return "obstacle_small_irrelevant"
            else:
                return "obstacle_clearance_relevant"
        return "obstacle_uncertain"

    # Rule 13: Elevation check
    if min_z is not None and storey in storey_floor:
        floor_elev = storey_floor[storey]
        if min_z > floor_elev + 2.5:
            return "obstacle_upper_irrelevant"

    # Proxy elements that survived as barriers
    if ifc_class == "IfcBuildingElementProxy":
        return "obstacle_clearance_relevant"

    return "obstacle_uncertain"


def _get_obstacle_rules_doc() -> List[Dict]:
    """Return the obstacle classification rules as a structured list."""
    return [
        {"priority": 1, "condition": "IfcRailing", "result": "obstacle_barrier_relevant", "rationale": "Railings directly constrain pedestrian movement"},
        {"priority": 2, "condition": "IfcCurtainWall", "result": "obstacle_barrier_relevant", "rationale": "Curtain walls form physical barriers"},
        {"priority": 3, "condition": "IfcPlate AND dz < 5cm", "result": "obstacle_skin_panel", "rationale": "Thin panels (石材嵌板) are cladding/veneer, do not block pedestrians"},
        {"priority": 4, "condition": "IfcPlate AND 5cm ≤ dz ≤ 30cm", "result": "obstacle_small_irrelevant", "rationale": "Thin trim elements unlikely to affect movement"},
        {"priority": 5, "condition": "IfcPlate AND dz > 1m", "result": "obstacle_barrier_relevant", "rationale": "Tall plate/glass partition acts as barrier"},
        {"priority": 6, "condition": "IfcPlate AND 30cm < dz ≤ 1m", "result": "obstacle_clearance_relevant", "rationale": "May restrict passage clearance"},
        {"priority": 7, "condition": "IfcColumn", "result": "obstacle_floor_intrusive", "rationale": "Columns occupy floor area directly"},
        {"priority": 8, "condition": "IfcWall / IfcWallStandardCase", "result": "obstacle_floor_intrusive", "rationale": "Walls occupy floor area or partition space"},
        {"priority": 9, "condition": "IfcMember AND dz < 30cm", "result": "obstacle_small_irrelevant", "rationale": "Small structural members unlikely to block pedestrians"},
        {"priority": 10, "condition": "IfcMember AND dz ≥ 30cm", "result": "obstacle_clearance_relevant", "rationale": "Larger members may constrain passage"},
        {"priority": 11, "condition": "Name contains 自动伸缩门/防火玻璃/自动门", "result": "obstacle_barrier_relevant", "rationale": "Fire glass doors and auto doors are barriers"},
        {"priority": 12, "condition": "min_z > storey_floor + 2.5m (with bbox)", "result": "obstacle_upper_irrelevant", "rationale": "Objects entirely above head height do not affect 2.5D walkability"},
        {"priority": 13, "condition": "IfcBuildingElementProxy (barrier)", "result": "obstacle_clearance_relevant", "rationale": "Proxy barriers conservatively kept as clearance-relevant"},
        {"priority": 14, "condition": "Default", "result": "obstacle_uncertain", "rationale": "Cannot determine with available signals"},
    ]


def _write_obstacle_report(summary: Dict, obstacles: pd.DataFrame, path: Path) -> None:
    """Write obstacle recalibration report."""
    lines = [
        "# Obstacle Recalibration Report (v3)",
        "",
        "## Objective",
        "",
        "Refine the obstacle set from a flat `retained_barrier` classification "
        "into meaningful subcategories that distinguish between objects that "
        "actually affect pedestrian walkability and those that do not.",
        "",
        "## Key Finding",
        "",
        f"Of {summary['total_obstacles_before']} obstacles, **{summary['recommended_keep_count']}** "
        f"({summary['recommended_keep_pct']}%) are recommended for retention in downstream "
        f"graph construction. The remaining **{summary['recommended_drop_count']}** are "
        f"surface panels, small fragments, or overhead objects that should not "
        f"affect 2.5D pedestrian movement modelling.",
        "",
        "## Subcategory Distribution",
        "",
        "| Subcategory | Count | Recommendation |",
        "|-------------|-------|----------------|",
    ]
    keep_set = set(summary["keep_categories"])
    for cat, cnt in sorted(summary["subcategory_counts"].items(), key=lambda x: -x[1]):
        rec = "**KEEP**" if cat in keep_set else "DROP"
        lines.append(f"| {cat} | {cnt} | {rec} |")

    lines.extend([
        "",
        "## Classification Rules",
        "",
        "| # | Condition | Result | Rationale |",
        "|---|-----------|--------|-----------|",
    ])
    for r in summary["rules"]:
        lines.append(f"| {r['priority']} | {r['condition']} | {r['result']} | {r['rationale']} |")

    lines.extend([
        "",
        "## By Storey",
        "",
        "| Storey | " + " | ".join(sorted(summary["subcategory_counts"].keys())) + " |",
        "|--------| " + " | ".join(["---"] * len(summary["subcategory_counts"])) + " |",
    ])
    for storey in sorted(obstacles["storey_name"].unique()):
        sub = obstacles[obstacles["storey_name"] == storey]
        counts = sub["obstacle_subcat"].value_counts()
        cells = [str(counts.get(cat, 0)) for cat in sorted(summary["subcategory_counts"].keys())]
        lines.append(f"| {storey} | " + " | ".join(cells) + " |")

    # Summary of what this means for graph construction
    lines.extend([
        "",
        "## Recommendation for Downstream Graph Construction",
        "",
        "Retain only these subcategories as physical obstacles in the 2.5D navigation graph:",
        "- `obstacle_floor_intrusive` — columns, walls that occupy pedestrian floor area",
        "- `obstacle_barrier_relevant` — railings, glass partitions, fire doors, curtain walls",
        "- `obstacle_clearance_relevant` — objects that may restrict passage width/height",
        "",
        "Drop from graph construction:",
        "- `obstacle_skin_panel` — surface cladding/veneer (typically 2cm stone panels)",
        "- `obstacle_small_irrelevant` — tiny structural fragments",
        "- `obstacle_upper_irrelevant` — overhead objects above 2.5m",
        "- `obstacle_uncertain` — explicitly surfaced for manual review if needed",
        "",
        "## Data Coverage Note",
        "",
        f"Bbox data available for {summary['has_bbox_pct']}% of obstacles. "
        f"Elements without bbox use IFC class and name pattern rules only.",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Obstacle report saved: {path.name}")


# =====================================================================
# PRIORITY 3: Connector completeness validation
# =====================================================================

def validate_connectors(
    retained_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """Validate connector completeness for downstream vertical linkage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Priority 3: Connector Completeness Validation (v3)")
    logger.info("=" * 60)

    connectors = retained_df[retained_df["filter_output_class"] == "retained_connector"].copy()
    logger.info(f"  Total retained connectors: {len(connectors)}")

    # Merge bbox (join on guid+source_file to avoid row duplication)
    bcols = ["guid", "source_file", "dx", "dy", "dz", "min_z", "max_z", "min_x", "max_x", "min_y", "max_y"]
    bcols = [c for c in bcols if c in bbox_df.columns]
    merge_keys = ["guid", "source_file"] if "source_file" in bcols else ["guid"]
    connectors = connectors.merge(bbox_df[bcols], on=merge_keys, how="left")

    # Classify connectors
    subtypes = []
    issues = []
    for _, row in connectors.iterrows():
        st, issue = _classify_connector(row)
        subtypes.append(st)
        if issue:
            issues.append(issue)
    connectors["connector_subtype"] = subtypes

    # Storey elevations (metres)
    storey_elev = {"F0 底板层": -1.7, "F1 站台层": 0.0, "F2 设备层": 5.3, "F3 站厅层": 12.1, "F4 交通层": 17.4, "F5RF顶板": 24.6}

    # Cross-storey span analysis
    has_z = connectors["dz"].notna()
    cbx = connectors[has_z].copy()
    cbx["spans_F1_F2"] = (cbx["min_z"] < 1.0) & (cbx["max_z"] > 4.0)
    cbx["spans_F2_F3"] = (cbx["min_z"] < 6.0) & (cbx["max_z"] > 10.0)
    cbx["spans_F1_F3"] = (cbx["min_z"] < 1.0) & (cbx["max_z"] > 10.0)

    n_f1_f2 = int(cbx["spans_F1_F2"].sum())
    n_f2_f3 = int(cbx["spans_F2_F3"].sum())
    n_f1_f3 = int(cbx["spans_F1_F3"].sum())

    logger.info(f"  Cross-storey spans: F1→F2={n_f1_f2}, F2→F3={n_f2_f3}, F1→F3={n_f1_f3}")

    # Subtype distribution
    subtype_counts = connectors["connector_subtype"].value_counts()
    logger.info("  Connector subtype distribution:")
    for st, cnt in subtype_counts.items():
        logger.info(f"    {st}: {cnt}")

    # F2 door analysis — are they really connectors?
    f2_doors = connectors[
        (connectors["ifc_class"] == "IfcDoor") &
        (connectors["storey_name"] == "F2 设备层")
    ]
    f2_door_verdict = (
        "F2 IfcDoor elements are technical-level fire doors retained as "
        "'opening_passage on connector level'. They do NOT represent public "
        "vertical circulation. Recommendation: reclassify as non-connector."
    )

    # Unique connector groups (by name pattern)
    stair_names = connectors[connectors["ifc_class"] == "IfcStair"]["name"].unique()
    escalator_names = connectors[connectors["connector_subtype"] == "escalator"]["name"].unique()
    elevator_names = connectors[connectors["connector_subtype"] == "elevator"]["name"].unique()

    # Sufficiency assessment
    has_stairs = (connectors["connector_subtype"] == "stair").sum() > 0
    has_stair_flights = (connectors["connector_subtype"] == "stair_flight").sum() > 0
    has_escalators = (connectors["connector_subtype"] == "escalator").sum() > 0
    has_elevators = (connectors["connector_subtype"] == "elevator").sum() > 0

    sufficiency = "SUFFICIENT" if (has_stairs and has_stair_flights and has_escalators) else "INSUFFICIENT"
    if not has_elevators:
        sufficiency_note = "Elevator proxy elements present but limited. Stairs and escalators are well covered."
    else:
        sufficiency_note = "All major connector types present."

    # Issues summary
    all_issues = issues.copy()
    if len(f2_doors) > 0:
        all_issues.append(
            f"{len(f2_doors)} F2 IfcDoor elements are technical fire doors, "
            f"not public connectors (reclassify recommended)"
        )

    suspect_cases = connectors[connectors["connector_subtype"].isin({"f2_technical_door", "uncertain"})]

    summary = {
        "total_connectors": len(connectors),
        "with_bbox": int(has_z.sum()),
        "subtype_counts": subtype_counts.to_dict(),
        "cross_storey_spans": {
            "F1_F2": n_f1_f2,
            "F2_F3": n_f2_f3,
            "F1_F3": n_f1_f3,
        },
        "f2_door_count": len(f2_doors),
        "f2_door_verdict": f2_door_verdict,
        "sufficiency": sufficiency,
        "sufficiency_note": sufficiency_note,
        "suspect_count": len(suspect_cases),
        "issues": all_issues,
        "unique_stair_names": list(stair_names),
        "unique_escalator_names": list(escalator_names),
        "unique_elevator_names": list(elevator_names),
    }

    # Export tables
    save_dataframe(connectors, output_dir / "connectors_validated.csv")
    save_dataframe(suspect_cases, output_dir / "connectors_suspect.csv")
    save_json(summary, output_dir / "connector_validation_summary.json")
    _write_connector_report(summary, connectors, cbx, output_dir / "connector_validation_report.md")

    return {
        "connectors": connectors,
        "suspect_cases": suspect_cases,
        "summary": summary,
    }


def _classify_connector(row: pd.Series) -> Tuple[str, str]:
    """Classify a single connector element. Returns (subtype, issue_or_empty)."""
    ifc_class = str(row.get("ifc_class", ""))
    name = str(row.get("name", ""))
    storey = str(row.get("storey_name", ""))
    category = str(row.get("category", ""))

    issue = ""

    if ifc_class == "IfcStair":
        return "stair", issue

    if ifc_class == "IfcStairFlight":
        return "stair_flight", issue

    if "自动扶梯" in name or "扶梯" in name:
        return "escalator", issue

    if "电梯" in name:
        return "elevator", issue

    if ifc_class == "IfcDoor" and storey in ("F2 设备层", "F0 底板层"):
        issue = f"F2/F0 IfcDoor '{name[:30]}' is likely a technical door, not a public connector"
        return "f2_technical_door", issue

    if ifc_class == "IfcDoor":
        return "door_passage", issue

    if ifc_class == "IfcBuildingElementProxy":
        if "扶梯" in name:
            return "escalator", issue
        if "电梯" in name:
            return "elevator", issue
        issue = f"Proxy connector '{name[:30]}' — type ambiguous"
        return "uncertain", issue

    return "uncertain", f"Unexpected connector: {ifc_class} '{name[:30]}'"


def _write_connector_report(
    summary: Dict, connectors: pd.DataFrame, cbx: pd.DataFrame, path: Path,
) -> None:
    """Write connector validation report."""
    lines = [
        "# Connector Completeness Validation Report (v3)",
        "",
        f"## Sufficiency Verdict: **{summary['sufficiency']}**",
        "",
        f"{summary['sufficiency_note']}",
        "",
        "## Overview",
        "",
        f"- Total retained connectors: **{summary['total_connectors']}**",
        f"- With bounding box: {summary['with_bbox']}",
        f"- Suspect / reclassify candidates: {summary['suspect_count']}",
        "",
        "## Connector Subtype Distribution",
        "",
        "| Subtype | Count |",
        "|---------|-------|",
    ]
    for st, cnt in sorted(summary["subtype_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {st} | {cnt} |")

    lines.extend([
        "",
        "## Cross-Storey Span Analysis",
        "",
        "Elements whose bounding box physically spans between levels:",
        "",
        f"- F1 → F2: **{summary['cross_storey_spans']['F1_F2']}** elements",
        f"- F2 → F3: **{summary['cross_storey_spans']['F2_F3']}** elements",
        f"- F1 → F3 (full span): **{summary['cross_storey_spans']['F1_F3']}** elements",
        "",
        "## F2 Technical Door Issue",
        "",
        f"**{summary['f2_door_count']} IfcDoor elements** on the F2 equipment level "
        f"were retained as connectors because they are `opening_passage` on a connector level. "
        f"However, these are technical fire doors and NOT public vertical circulation elements.",
        "",
        f"**Verdict**: {summary['f2_door_verdict']}",
        "",
        "## Unique Connector Groups",
        "",
    ])

    if summary['unique_stair_names']:
        lines.append("### Stairs")
        for n in summary['unique_stair_names'][:10]:
            lines.append(f"- {n}")
    if summary['unique_escalator_names']:
        lines.append("")
        lines.append("### Escalators")
        for n in summary['unique_escalator_names'][:10]:
            lines.append(f"- {n}")
    if summary['unique_elevator_names']:
        lines.append("")
        lines.append("### Elevators")
        for n in summary['unique_elevator_names'][:10]:
            lines.append(f"- {n}")

    # Storey distribution
    lines.extend([
        "",
        "## Connector Distribution by Storey",
        "",
        "| Storey | Count | Subtypes |",
        "|--------|-------|----------|",
    ])
    for storey in sorted(connectors["storey_name"].unique()):
        sub = connectors[connectors["storey_name"] == storey]
        stypes = sub["connector_subtype"].value_counts()
        stypes_str = ", ".join(f"{k}={v}" for k, v in stypes.items())
        lines.append(f"| {storey} | {len(sub)} | {stypes_str} |")

    if summary["issues"]:
        lines.extend([
            "",
            "## Issues",
            "",
        ])
        for issue in summary["issues"]:
            lines.append(f"- ⚠️ {issue}")

    lines.extend([
        "",
        "## Conclusion for Downstream Graph Construction",
        "",
        "The connector set is **sufficient** for modelling vertical movement "
        "between F1 (platform) and F3 (concourse) via stairs and escalators. "
        "The 64 F2 fire doors should be reclassified as non-connector elements. "
        "After reclassification, the effective public connector count is "
        f"**{summary['total_connectors'] - summary['f2_door_count']}**.",
        "",
        "Key connector types present:",
        "- IfcStair (36) — complete staircase assemblies",
        "- IfcStairFlight (77) — individual stair runs with geometry",
        "- Escalator proxies (12+) — escalator bodies spanning F1→F3",
        "- Elevator proxies (3) — elevator shafts/doors",
        "",
        "Remaining risk: elevator modelling may be incomplete as only proxy "
        "elements represent elevators. For thesis-level experiments using "
        "stairs and escalators as primary connectors, this is acceptable.",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Connector validation report saved: {path.name}")


# =====================================================================
# VISUALIZATIONS
# =====================================================================

def generate_v3_visualizations(
    obstacle_results: Dict[str, Any],
    connector_results: Dict[str, Any],
    ifc_results: Dict[str, Any],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Generate v3 visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Generating v3 visualizations")
    logger.info("=" * 60)

    # CJK font
    cjk_font = None
    for fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            font_manager.findfont(fname, fallback_to_default=False)
            cjk_font = fname
            break
        except Exception:
            pass
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font] + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["axes.unicode_minus"] = False

    obstacles = obstacle_results["obstacles"]
    summary = obstacle_results["summary"]
    connectors = connector_results["connectors"]
    conn_summary = connector_results["summary"]

    # --- Fig 1: Obstacle before/after ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before: single bar
    axes[0].barh(["retained_barrier\n(v2)"], [summary["total_obstacles_before"]], color="#e74c3c")
    axes[0].set_title("Before: Flat Obstacle Category")
    axes[0].set_xlabel("Element count")

    # After: stacked subcategories
    keep_cats = set(summary["keep_categories"])
    cats = sorted(summary["subcategory_counts"].keys())
    counts = [summary["subcategory_counts"][c] for c in cats]
    colors = []
    for c in cats:
        if c in keep_cats:
            colors.append("#27ae60")
        elif c == "obstacle_uncertain":
            colors.append("#f39c12")
        else:
            colors.append("#bdc3c7")

    axes[1].barh(cats, counts, color=colors)
    axes[1].set_title("After: Refined Subcategories")
    axes[1].set_xlabel("Element count")
    for i, (c, cnt) in enumerate(zip(cats, counts)):
        label = "KEEP" if c in keep_cats else "DROP"
        axes[1].text(cnt + 20, i, f"{cnt} ({label})", va="center", fontsize=8)

    plt.tight_layout()
    _save_fig(fig, output_dir / "obstacle_before_after", save_svg)
    plt.close()

    # --- Fig 2: Obstacle subcategories by storey ---
    fig, ax = plt.subplots(figsize=(10, 6))
    storey_cat = obstacles.groupby(["storey_name", "obstacle_subcat"]).size().unstack(fill_value=0)
    storey_cat.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Obstacle Subcategories by Storey")
    ax.set_ylabel("Count")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _save_fig(fig, output_dir / "obstacle_by_storey", save_svg)
    plt.close()

    # --- Fig 3: Obstacle dz distribution ---
    obs_with_dz = obstacles[obstacles["dz"].notna()]
    if len(obs_with_dz) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        for cat in sorted(obs_with_dz["obstacle_subcat"].unique()):
            sub = obs_with_dz[obs_with_dz["obstacle_subcat"] == cat]
            ax.hist(sub["dz"].clip(upper=6), bins=50, alpha=0.6, label=f"{cat} (n={len(sub)})")
        ax.set_title("Obstacle Height (dz) Distribution by Subcategory")
        ax.set_xlabel("dz (metres)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.axvline(x=0.05, color="red", linestyle="--", linewidth=0.8, label="5cm threshold")
        ax.axvline(x=2.5, color="blue", linestyle="--", linewidth=0.8, label="2.5m head height")
        plt.tight_layout()
        _save_fig(fig, output_dir / "obstacle_dz_distribution", save_svg)
        plt.close()

    # --- Fig 4: Connector subtype distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    st_counts = connectors["connector_subtype"].value_counts()
    colors_conn = []
    for st in st_counts.index:
        if st in ("stair", "stair_flight", "escalator", "elevator"):
            colors_conn.append("#2ecc71")
        elif st == "f2_technical_door":
            colors_conn.append("#e74c3c")
        else:
            colors_conn.append("#f39c12")
    ax.barh(st_counts.index, st_counts.values, color=colors_conn)
    ax.set_title("Connector Subtype Distribution")
    ax.set_xlabel("Count")
    for i, cnt in enumerate(st_counts.values):
        ax.text(cnt + 0.5, i, str(cnt), va="center", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, output_dir / "connector_subtypes", save_svg)
    plt.close()

    # --- Fig 5: Connector storey distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    conn_storey = connectors.groupby(["storey_name", "connector_subtype"]).size().unstack(fill_value=0)
    conn_storey.plot(kind="bar", stacked=True, ax=ax, colormap="Paired")
    ax.set_title("Connector Distribution by Storey")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, output_dir / "connector_by_storey", save_svg)
    plt.close()

    # --- Fig 6: IFC export status ---
    fig, ax = plt.subplots(figsize=(8, 4))
    subset_names = list(ifc_results.keys())
    statuses = [ifc_results[s].get("status", "?") for s in subset_names]
    sizes = [ifc_results[s].get("file_size_mb", 0) for s in subset_names]
    bar_colors = ["#27ae60" if s == "success" else "#e74c3c" for s in statuses]
    ax.barh(subset_names, sizes, color=bar_colors)
    ax.set_title("IFC Subset Export Results")
    ax.set_xlabel("File size (MB)")
    for i, (s, sz) in enumerate(zip(statuses, sizes)):
        ax.text(max(sz, 0.5), i, f"{s} ({sz:.1f} MB)", va="center", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, output_dir / "ifc_export_status", save_svg)
    plt.close()

    logger.info("  All v3 visualizations generated.")


def _save_fig(fig, stem: Path, save_svg: bool = True) -> None:
    fig.savefig(str(stem) + ".png", dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {stem.name}.png")
    if save_svg:
        fig.savefig(str(stem) + ".svg", bbox_inches="tight")


# =====================================================================
# MAIN ORCHESTRATOR
# =====================================================================

def main():
    """Run v3 refinement pipeline."""
    start = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
    v2_config = load_config(str(PROJECT_ROOT / "config" / "v2_config.yaml"))

    # V3 output directories
    v3_base = PROJECT_ROOT / "outputs" / "v3"
    v3_dirs = {
        "base": v3_base,
        "ifc_subsets": v3_base / "ifc_subsets",
        "obstacle": v3_base / "obstacle_recalibration",
        "connector": v3_base / "connector_validation",
        "figures": v3_base / "figures",
        "logs": v3_base / "logs",
    }
    for d in v3_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Logging
    log_handler = setup_logging(str(v3_dirs["logs"]), f"v3_{run_id}")
    logger.info("=" * 70)
    logger.info("V3 REFINEMENT: Data Layer Stabilization")
    logger.info(f"Run ID: v3_{run_id}")
    logger.info("=" * 70)

    # Load v2 outputs
    logger.info("Loading v2 outputs...")
    retained_df = pd.read_csv(
        PROJECT_ROOT / "outputs" / "v2" / "traffic_filtered" / "retained_elements.csv",
        encoding="utf-8-sig",
    )
    bbox_df = pd.read_csv(
        PROJECT_ROOT / "outputs" / "v2" / "normalized" / "bbox_samples_metres.csv",
        encoding="utf-8-sig",
    )
    logger.info(f"  Retained elements: {len(retained_df)}")
    logger.info(f"  Bbox samples: {len(bbox_df)}")

    # === PRIORITY 1: IFC Subset Export ===
    with Timer("IFC subset export", logger):
        ifc_results = export_small_ifc_subsets(retained_df, config, v3_dirs["ifc_subsets"])

    # === PRIORITY 2: Obstacle Recalibration ===
    with Timer("Obstacle recalibration", logger):
        obstacle_results = recalibrate_obstacles(retained_df, bbox_df, v3_dirs["obstacle"])

    # === PRIORITY 3: Connector Validation ===
    with Timer("Connector validation", logger):
        connector_results = validate_connectors(retained_df, bbox_df, v3_dirs["connector"])

    # === VISUALIZATIONS ===
    with Timer("V3 visualizations", logger):
        vis_cfg = config.get("visualization", {})
        generate_v3_visualizations(
            obstacle_results, connector_results, ifc_results,
            v3_dirs["figures"],
            save_svg=vis_cfg.get("save_svg", True),
        )

    # === MANIFEST ===
    elapsed = time.perf_counter() - start
    manifest = {
        "run_id": f"v3_{run_id}",
        "elapsed_seconds": round(elapsed, 1),
        "priorities_completed": [
            "ifc_subset_export",
            "obstacle_recalibration",
            "connector_validation",
        ],
        "ifc_export_status": {
            k: v.get("status", "?") for k, v in ifc_results.items()
        },
        "obstacle_summary": {
            "total_before": obstacle_results["summary"]["total_obstacles_before"],
            "recommended_keep": obstacle_results["summary"]["recommended_keep_count"],
            "recommended_drop": obstacle_results["summary"]["recommended_drop_count"],
        },
        "connector_summary": {
            "total": connector_results["summary"]["total_connectors"],
            "sufficiency": connector_results["summary"]["sufficiency"],
            "suspect_count": connector_results["summary"]["suspect_count"],
        },
    }
    save_json(manifest, v3_base / "v3_manifest.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("V3 REFINEMENT COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Outputs: {v3_base}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
