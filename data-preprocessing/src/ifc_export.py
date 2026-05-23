"""
Filtered IFC Subset Export Module (v2).

Generates navigation-oriented IFC subset files by copying retained
elements from the source IFC models to new IFC files.

Strategy:
  1. Create a fresh IFC file with the same schema (IFC2X3).
  2. Copy essential project hierarchy (IfcProject, IfcSite, IfcBuilding,
     relevant IfcBuildingStorey instances).
  3. Copy retained IfcProduct elements by deep-traversal of their entity
     dependency graph (geometry representations, materials, styles, etc.).
  4. Re-create spatial containment relationships for copied elements.

Limitations (documented):
  - IFC unit system is preserved as-is (millimetres). We do NOT modify
    IFC geometry coordinates to metres to avoid corruption risk.
  - Some inverse relationships (e.g. IfcRelDefinesByProperties shared
    across many elements) may be duplicated or lost if the referenced
    elements were filtered out.
  - Property sets may be incomplete if they reference dropped elements.
  - The exported IFC is intended as an experimental subset, not a
    round-trip-safe BIM deliverable.

Subsets generated:
  - platform_public.ifc   — F1 public-level elements
  - concourse_public.ifc  — F3 public-level elements
  - vertical_connectors.ifc — cross-storey connector elements
  - navigation_merged.ifc — combined navigation subset
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ifcopenshell

from .utils import save_json, save_dataframe

logger = logging.getLogger(__name__)


def _collect_element_dependencies(
    element,
    model: ifcopenshell.file,
    visited: Set[int],
    max_depth: int = 15,
    _depth: int = 0,
) -> List:
    """Recursively collect all entities that an element depends on.

    Traverses forward references (attributes that point to other entities)
    to build the complete dependency closure needed to copy an element
    to a new file.

    Args:
        element: The IFC entity to collect dependencies for.
        model: Source IFC model.
        visited: Set of already-visited entity ids (to avoid cycles).
        max_depth: Maximum recursion depth.
        _depth: Current recursion depth.

    Returns:
        List of dependent IFC entities (excluding the element itself).
    """
    if _depth > max_depth:
        return []
    eid = element.id()
    if eid in visited:
        return []
    visited.add(eid)

    deps = []
    try:
        for attr_idx in range(len(element)):
            attr_val = element[attr_idx]
            if isinstance(attr_val, ifcopenshell.entity_instance):
                deps.append(attr_val)
                deps.extend(
                    _collect_element_dependencies(
                        attr_val, model, visited, max_depth, _depth + 1
                    )
                )
            elif isinstance(attr_val, (tuple, list)):
                for item in attr_val:
                    if isinstance(item, ifcopenshell.entity_instance):
                        deps.append(item)
                        deps.extend(
                            _collect_element_dependencies(
                                item, model, visited, max_depth, _depth + 1
                            )
                        )
    except Exception:
        pass

    return deps


def _copy_entity_to_file(
    entity,
    target_file: ifcopenshell.file,
    copied_map: Dict[int, Any],
) -> Any:
    """Deep-copy a single IFC entity to a target file.

    Uses a mapping to avoid duplicating already-copied entities.

    Args:
        entity: Source IFC entity.
        target_file: Target IFC file.
        copied_map: Mapping of source entity id → target entity.

    Returns:
        The copied entity in the target file.
    """
    eid = entity.id()
    if eid in copied_map:
        return copied_map[eid]

    try:
        new_entity = target_file.add(entity)
        copied_map[eid] = new_entity
        return new_entity
    except Exception:
        copied_map[eid] = None
        return None


def export_filtered_ifc(
    retained_df,
    file_infos: Dict[str, Any],
    v2_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Export filtered IFC subset files.

    Args:
        retained_df: DataFrame of retained elements (with guid, source_file,
                     storey_name, filter_output_class columns).
        file_infos: Dictionary of label → IFCFileInfo from v1.
        v2_config: v2 configuration.
        output_dir: Output directory for IFC files.

    Returns:
        Export results dictionary with success/failure status per subset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Filtered IFC Subset Export (v2)")
    logger.info("=" * 60)

    ifc_cfg = v2_config.get("ifc_export", {})
    if not ifc_cfg.get("enabled", True):
        logger.info("  IFC export disabled in config — skipping")
        return {"status": "disabled"}

    subsets_cfg = ifc_cfg.get("subsets", {})
    results = {}

    for subset_name, subset_cfg in subsets_cfg.items():
        logger.info(f"  Exporting subset: {subset_name}")
        try:
            result = _export_single_subset(
                subset_name=subset_name,
                subset_cfg=subset_cfg,
                retained_df=retained_df,
                file_infos=file_infos,
                output_dir=output_dir,
            )
            results[subset_name] = result
            logger.info(f"    → {result.get('status', 'unknown')}: {result.get('message', '')}")
        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            results[subset_name] = {
                "status": "error",
                "message": str(e),
                "element_count": 0,
            }

    # Validation: try to reopen
    if ifc_cfg.get("validate_reopen", True):
        _validate_exported_ifcs(results, output_dir)

    # Save export report
    save_json(results, output_dir / "ifc_export_results.json")
    _write_ifc_export_report(results, output_dir / "ifc_export_report.md")

    return results


def _export_single_subset(
    subset_name: str,
    subset_cfg: Dict[str, Any],
    retained_df,
    file_infos: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Export a single IFC subset file.

    Args:
        subset_name: Name of the subset (e.g., "platform_public").
        subset_cfg: Subset configuration from v2_config.
        retained_df: Retained elements DataFrame.
        file_infos: IFCFileInfo dictionary.
        output_dir: Output directory.

    Returns:
        Result dictionary.
    """
    import pandas as pd

    # Filter elements for this subset
    target_storeys = set(subset_cfg.get("storeys", []))
    target_categories = set(subset_cfg.get("categories", []))
    include_connectors = subset_cfg.get("include_connectors", False)

    mask = pd.Series(False, index=retained_df.index)

    if target_storeys:
        mask |= retained_df["storey_name"].isin(target_storeys)

    if target_categories:
        mask |= retained_df["category"].isin(target_categories)

    if include_connectors:
        mask |= (retained_df["filter_output_class"] == "retained_connector")

    subset_elements = retained_df[mask]

    if len(subset_elements) == 0:
        return {
            "status": "empty",
            "message": "No elements matched subset criteria",
            "element_count": 0,
        }

    # For very large subsets, export GUID list only (IFC deep-copy is too slow)
    MAX_IFC_EXPORT_ELEMENTS = 2000
    if len(subset_elements) > MAX_IFC_EXPORT_ELEMENTS:
        logger.info(
            f"    Subset has {len(subset_elements)} elements (>{MAX_IFC_EXPORT_ELEMENTS}). "
            f"Exporting GUID list only (IFC deep-copy would be too slow)."
        )
        guid_list_path = output_dir / f"{subset_name}_guids.csv"
        save_dataframe(subset_elements[["guid", "source_file", "storey_name", "category", "ifc_class", "name"]],
                       guid_list_path)
        return {
            "status": "guid_list_only",
            "message": f"Exported GUID list ({len(subset_elements)} elements) — too large for IFC deep-copy",
            "path": str(guid_list_path),
            "element_count": len(subset_elements),
            "copy_errors": 0,
            "description": subset_cfg.get("description", ""),
        }

    # Group elements by source file
    elements_by_file = {}
    for _, row in subset_elements.iterrows():
        source = row["source_file"]
        if source not in elements_by_file:
            elements_by_file[source] = []
        elements_by_file[source].append(row["guid"])

    # Pick the primary source file (the one with most elements)
    primary_source = max(elements_by_file, key=lambda k: len(elements_by_file[k]))
    primary_model = file_infos[primary_source].model

    # Create new IFC file based on the primary source's schema
    target_path = output_dir / f"{subset_name}.ifc"
    target_file = ifcopenshell.file(schema=primary_model.schema)

    copied_map = {}  # source entity id → target entity
    copied_guids = set()
    copy_errors = 0

    # Copy project hierarchy from primary source
    _copy_project_hierarchy(primary_model, target_file, copied_map)

    # Copy elements from each source
    for source_label, guids in elements_by_file.items():
        if source_label not in file_infos:
            logger.warning(f"    Source file not found: {source_label}")
            continue

        model = file_infos[source_label].model
        logger.info(f"    Copying {len(guids)} elements from {source_label}...")

        for i, guid in enumerate(guids, 1):
            try:
                elem = model.by_guid(guid)
                _copy_element_with_dependencies(
                    elem, model, target_file, copied_map
                )
                copied_guids.add(guid)
            except Exception as e:
                copy_errors += 1
                if copy_errors <= 5:
                    logger.warning(f"    Could not copy element {guid}: {e}")
            if i % 200 == 0:
                logger.info(f"      Progress: {i}/{len(guids)} elements copied")

    # Write the file
    target_file.write(str(target_path))

    return {
        "status": "success",
        "message": f"Exported {len(copied_guids)} elements to {target_path.name}",
        "path": str(target_path),
        "element_count": len(copied_guids),
        "copy_errors": copy_errors,
        "file_size_bytes": target_path.stat().st_size if target_path.exists() else 0,
        "description": subset_cfg.get("description", ""),
    }


def _copy_project_hierarchy(
    source: ifcopenshell.file,
    target: ifcopenshell.file,
    copied_map: Dict[int, Any],
) -> None:
    """Copy the essential project hierarchy from source to target.

    Copies: IfcProject, IfcSite, IfcBuilding, all IfcBuildingStorey,
    plus their relationships and unit assignments.
    """
    # Copy IfcOwnerHistory (often referenced by many entities)
    for oh in source.by_type("IfcOwnerHistory"):
        _copy_entity_to_file(oh, target, copied_map)

    # Copy units
    for ua in source.by_type("IfcUnitAssignment"):
        _copy_entity_to_file(ua, target, copied_map)

    # Copy project
    for proj in source.by_type("IfcProject"):
        _copy_entity_to_file(proj, target, copied_map)

    # Copy site
    for site in source.by_type("IfcSite"):
        _copy_entity_to_file(site, target, copied_map)

    # Copy building
    for bldg in source.by_type("IfcBuilding"):
        _copy_entity_to_file(bldg, target, copied_map)

    # Copy all storeys
    for storey in source.by_type("IfcBuildingStorey"):
        _copy_entity_to_file(storey, target, copied_map)

    # Copy spatial hierarchy relationships
    for rel in source.by_type("IfcRelAggregates"):
        try:
            relating = rel.RelatingObject
            if relating.is_a() in (
                "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey"
            ):
                _copy_entity_to_file(rel, target, copied_map)
        except Exception:
            pass

    # Copy geometric representation context
    for ctx in source.by_type("IfcGeometricRepresentationContext"):
        _copy_entity_to_file(ctx, target, copied_map)


def _copy_element_with_dependencies(
    element,
    source: ifcopenshell.file,
    target: ifcopenshell.file,
    copied_map: Dict[int, Any],
) -> None:
    """Copy an element and its dependency closure to target file."""
    # The ifcopenshell.file.add() method handles deep copying of
    # referenced entities automatically in recent versions.
    # We just need to add the element directly.
    _copy_entity_to_file(element, target, copied_map)

    # Also copy spatial containment relationships for this element
    try:
        if hasattr(element, "ContainedInStructure"):
            for rel in element.ContainedInStructure:
                _copy_entity_to_file(rel, target, copied_map)
    except Exception:
        pass

    # Copy aggregate relationships (element is a part of something)
    try:
        if hasattr(element, "Decomposes"):
            for rel in element.Decomposes:
                _copy_entity_to_file(rel, target, copied_map)
    except Exception:
        pass

    # Copy property sets
    try:
        if hasattr(element, "IsDefinedBy"):
            for rel in element.IsDefinedBy:
                _copy_entity_to_file(rel, target, copied_map)
    except Exception:
        pass


def _validate_exported_ifcs(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Try to reopen exported IFC files to validate them."""
    logger.info("  Validating exported IFC files...")

    for subset_name, result in results.items():
        if result.get("status") != "success":
            continue

        ifc_path = Path(result.get("path", ""))
        if not ifc_path.exists():
            result["validation"] = {"readable": False, "error": "File not found"}
            continue

        try:
            reopened = ifcopenshell.open(str(ifc_path))
            products = reopened.by_type("IfcProduct")
            storeys = reopened.by_type("IfcBuildingStorey")
            result["validation"] = {
                "readable": True,
                "reopened_entity_count": len(list(reopened)),
                "reopened_product_count": len(products),
                "reopened_storey_count": len(storeys),
                "schema": reopened.schema,
            }
            logger.info(
                f"    {subset_name}: ✓ readable, "
                f"{len(products)} products, {len(storeys)} storeys"
            )
        except Exception as e:
            result["validation"] = {
                "readable": False,
                "error": str(e),
            }
            logger.warning(f"    {subset_name}: ✗ validation failed: {e}")


def _write_ifc_export_report(results: Dict[str, Any], path: Path) -> None:
    """Write human-readable IFC export report."""
    lines = []
    lines.append("# Filtered IFC Export Report (v2)")
    lines.append("")
    lines.append("## Export Summary")
    lines.append("")
    lines.append("| Subset | Status | Elements | Size | Readable |")
    lines.append("|--------|--------|----------|------|----------|")

    for name, r in results.items():
        status = r.get("status", "?")
        elems = r.get("element_count", 0)
        size_mb = r.get("file_size_bytes", 0) / 1e6
        val = r.get("validation", {})
        readable = "✓" if val.get("readable") else "✗" if val else "—"
        lines.append(f"| {name} | {status} | {elems} | {size_mb:.1f} MB | {readable} |")

    lines.append("")

    # Detailed per-subset
    for name, r in results.items():
        lines.append(f"## {name}")
        lines.append(f"- **Description**: {r.get('description', '')}")
        lines.append(f"- **Status**: {r.get('status', '')}")
        lines.append(f"- **Message**: {r.get('message', '')}")
        lines.append(f"- **Elements copied**: {r.get('element_count', 0)}")
        lines.append(f"- **Copy errors**: {r.get('copy_errors', 0)}")

        val = r.get("validation", {})
        if val:
            lines.append(f"- **Readable**: {val.get('readable', False)}")
            if val.get("readable"):
                lines.append(f"- **Reopened entities**: {val.get('reopened_entity_count', 0)}")
                lines.append(f"- **Reopened products**: {val.get('reopened_product_count', 0)}")
            if val.get("error"):
                lines.append(f"- **Validation error**: {val.get('error', '')}")
        lines.append("")

    # Documented limitations
    lines.append("## Known Limitations")
    lines.append("- IFC geometry remains in millimetres (native unit). Not rescaled.")
    lines.append("- Downstream CSV/JSON exports use metres; IFC files use mm.")
    lines.append("- Some inverse relationships may be lost for filtered-out elements.")
    lines.append("- Property sets referencing dropped elements may be incomplete.")
    lines.append("- These are experimental subsets, not round-trip BIM deliverables.")
    lines.append("- GUID traceability is preserved (same GlobalIds as source).")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  IFC export report saved: {path.name}")
