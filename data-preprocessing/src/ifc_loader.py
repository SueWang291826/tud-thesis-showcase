"""
IFC File Loader Module.

Provides safe, logged loading of IFC files using ifcopenshell.
Extracts basic metadata (schema, project hierarchy, storey definitions)
immediately upon loading, before any heavy processing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ifcopenshell

from .utils import decode_ifc_x2, safe_ifc_str

logger = logging.getLogger(__name__)


@dataclass
class StoreyInfo:
    """Parsed storey information from an IFC file."""
    guid: str
    raw_name: str
    decoded_name: str
    elevation_mm: float
    long_name: str = ""


@dataclass
class IFCFileInfo:
    """Container for loaded IFC file metadata and model reference."""
    label: str                     # Human label: platform / equipment / concourse
    filepath: Path
    model: ifcopenshell.file       # The loaded ifcopenshell model
    schema: str = ""
    project_name: str = ""
    project_guid: str = ""
    building_name: str = ""
    building_guid: str = ""
    site_name: str = ""
    storeys: List[StoreyInfo] = field(default_factory=list)
    entity_count: int = 0
    length_unit: str = ""
    length_prefix: str = ""


def load_ifc_file(filepath: Path, label: str) -> IFCFileInfo:
    """Load a single IFC file and extract basic metadata.

    Args:
        filepath: Path to the IFC file.
        label: Human-readable label (e.g. 'platform').

    Returns:
        IFCFileInfo with model reference and parsed metadata.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If IFC parsing fails.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"IFC file not found: {filepath}")

    logger.info(f"Loading IFC file: {filepath.name} (label={label})")
    try:
        model = ifcopenshell.open(str(filepath))
    except Exception as e:
        raise RuntimeError(f"Failed to parse IFC file {filepath}: {e}") from e

    info = IFCFileInfo(label=label, filepath=filepath, model=model)

    # Schema
    info.schema = model.schema

    # Project
    projects = model.by_type("IfcProject")
    if projects:
        proj = projects[0]
        info.project_name = safe_ifc_str(proj.Name)
        info.project_guid = proj.GlobalId

    # Site
    sites = model.by_type("IfcSite")
    if sites:
        info.site_name = safe_ifc_str(sites[0].Name)

    # Building
    buildings = model.by_type("IfcBuilding")
    if buildings:
        bldg = buildings[0]
        info.building_name = safe_ifc_str(bldg.Name)
        info.building_guid = bldg.GlobalId

    # Storeys
    storeys = model.by_type("IfcBuildingStorey")
    for s in storeys:
        si = StoreyInfo(
            guid=s.GlobalId,
            raw_name=str(s.Name) if s.Name else "",
            decoded_name=safe_ifc_str(s.Name),
            elevation_mm=float(s.Elevation) if s.Elevation is not None else 0.0,
            long_name=safe_ifc_str(s.LongName) if hasattr(s, "LongName") else "",
        )
        info.storeys.append(si)
    info.storeys.sort(key=lambda x: x.elevation_mm)

    # Unit detection
    _detect_length_unit(model, info)

    # Total entity count
    info.entity_count = len(list(model))

    logger.info(
        f"  Loaded {info.entity_count} entities, schema={info.schema}, "
        f"{len(info.storeys)} storeys"
    )

    return info


def _detect_length_unit(model: ifcopenshell.file, info: IFCFileInfo) -> None:
    """Detect the length unit from the IFC model."""
    try:
        units = model.by_type("IfcUnitAssignment")
        if units:
            for unit in units[0].Units:
                if hasattr(unit, "UnitType") and unit.UnitType == "LENGTHUNIT":
                    info.length_unit = str(unit.Name) if hasattr(unit, "Name") else ""
                    if hasattr(unit, "Prefix") and unit.Prefix:
                        info.length_prefix = str(unit.Prefix)
                    break
    except Exception as e:
        logger.warning(f"  Could not detect length unit: {e}")


def get_storey_for_element(
    element,
    model: ifcopenshell.file,
    _depth: int = 0,
) -> Optional[str]:
    """Determine which IfcBuildingStorey an element belongs to.

    Resolution order:
      1. Direct spatial containment (IfcRelContainedInSpatialStructure).
      2. Walk up the IfcRelAggregates (Decomposes) chain recursively.
         Many Revit exports nest sub-components (IfcPlate inside
         IfcCurtainWall, IfcStairFlight inside IfcStair, etc.) without
         giving sub-components their own spatial containment.  The
         ancestor that IS spatially contained carries the storey info.
      3. Give up after 10 hops (safety guard against malformed graphs).

    Args:
        element: An IFC product element.
        model: The ifcopenshell model.
        _depth: Internal recursion depth counter (do not set manually).

    Returns:
        The GlobalId of the containing storey, or None.
    """
    MAX_DEPTH = 10
    if _depth > MAX_DEPTH:
        return None

    # Method 1: Direct spatial containment
    try:
        if hasattr(element, "ContainedInStructure"):
            for rel in element.ContainedInStructure:
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    return structure.GlobalId
    except Exception:
        pass

    # Method 2: Walk up the Decomposes chain
    try:
        if hasattr(element, "Decomposes"):
            for rel in element.Decomposes:
                parent = rel.RelatingObject
                if parent.is_a("IfcBuildingStorey"):
                    return parent.GlobalId
                # Recurse: ask the parent for its storey
                result = get_storey_for_element(parent, model, _depth + 1)
                if result is not None:
                    return result
    except Exception:
        pass

    return None


def get_all_products(model: ifcopenshell.file) -> list:
    """Get all IfcProduct instances (physical elements) from the model.

    Excludes spatial structure elements (Site, Building, Storey, Space).
    """
    products = model.by_type("IfcProduct")
    spatial_types = {"IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace", "IfcProject"}
    return [p for p in products if p.is_a() not in spatial_types]


def get_element_properties(element) -> Dict[str, Any]:
    """Extract commonly needed properties from an IFC element.

    Returns a flat dictionary with decoded string values.
    """
    props = {
        "guid": element.GlobalId if hasattr(element, "GlobalId") else "",
        "ifc_class": element.is_a(),
        "name": safe_ifc_str(element.Name) if hasattr(element, "Name") and element.Name else "",
        "object_type": safe_ifc_str(element.ObjectType) if hasattr(element, "ObjectType") and element.ObjectType else "",
        "description": safe_ifc_str(element.Description) if hasattr(element, "Description") and element.Description else "",
        "tag": str(element.Tag) if hasattr(element, "Tag") and element.Tag else "",
    }

    # PredefinedType (available on many IFC element types)
    predefined_type = ""
    if hasattr(element, "PredefinedType") and element.PredefinedType:
        predefined_type = str(element.PredefinedType)
    props["predefined_type"] = predefined_type

    return props
