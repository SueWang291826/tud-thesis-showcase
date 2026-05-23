"""
Step 1 - Geometry Extractor  (v2 rewrite)
==========================================

**Floor polygons**
    Extracted from *raw* IFC files via ``IfcSlab`` -> ``ifcopenshell.geom``
    + z-level filtering.  Coordinates come out in **metres** because
    ``ifcopenshell`` auto-converts from the IFC's mm unit.

**Obstacle polygons**
    From preprocessing bbox CSV - fast and already calibrated.

**Connector footprints** (mixed sources)
    Stair flights  - bbox CSV  (75 elements with valid bboxes)
    Escalators     - raw IFC   (IfcBuildingElementProxy, name contains "自动扶梯")
    Elevators      - raw IFC   (IfcBuildingElementProxy, name contains "电梯")

Each connector carries its own z-range so downstream code can derive
which levels it connects.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union

from src.utils import (
    flatten_polygons, polygon_from_bbox, dump_json, write_geojson,
    polygon_feature, point_feature, safe_name,
)


# ---------------------------------------------------------------------------
#  Low-level IFC -> Shapely helpers  (with caching)
# ---------------------------------------------------------------------------

_GEOM_SETTINGS = None          # lazy singleton

def _ifc_geom_settings():
    """Create and cache ifcopenshell.geom settings (USE_WORLD_COORDS)."""
    global _GEOM_SETTINGS
    if _GEOM_SETTINGS is None:
        import ifcopenshell.geom
        s = ifcopenshell.geom.settings()
        s.set(s.USE_WORLD_COORDS, True)
        _GEOM_SETTINGS = s
    return _GEOM_SETTINGS


# ---- Shared shape cache (key = element.id()) ----
_shape_cache: dict[int, tuple | None] = {}   # id -> (verts, faces) or None


def _get_shape_data(element, settings) -> tuple | None:
    """Return (verts, faces) for *element*, using a global LRU cache."""
    eid = element.id()
    if eid in _shape_cache:
        return _shape_cache[eid]
    try:
        import ifcopenshell.geom
        shape = ifcopenshell.geom.create_shape(settings, element)
        v, f = shape.geometry.verts, shape.geometry.faces
        result = (v, f) if v else None
    except Exception:
        result = None
    _shape_cache[eid] = result
    return result


def _verts_to_bbox(verts) -> dict:
    """Compute bbox from flat vertex list using numpy (fast)."""
    arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return dict(min_x=lo[0], max_x=hi[0],
                min_y=lo[1], max_y=hi[1],
                min_z=lo[2], max_z=hi[2])


def _element_xy_polygon(element, settings) -> Polygon | None:
    """2-D XY footprint from an IFC element via its triangulated mesh."""
    sd = _get_shape_data(element, settings)
    if sd is None:
        return None
    verts, faces = sd
    if not faces:
        return None

    arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    pts_xy = arr[:, :2]  # Nx2
    face_arr = np.asarray(faces, dtype=np.intp).reshape(-1, 3)

    triangles = []
    for ia, ib, ic in face_arr:
        if max(ia, ib, ic) >= len(pts_xy):
            continue
        tri = Polygon([pts_xy[ia], pts_xy[ib], pts_xy[ic]])
        if tri.is_valid and tri.area > 1e-6:
            triangles.append(tri)
    if not triangles:
        return None
    return unary_union(triangles).buffer(0)


def _element_z_range(element, settings) -> tuple[float, float] | None:
    """Return (z_min, z_max) in metres for an IFC element."""
    sd = _get_shape_data(element, settings)
    if sd is None:
        return None
    arr = np.asarray(sd[0], dtype=np.float64).reshape(-1, 3)
    return (float(arr[:, 2].min()), float(arr[:, 2].max()))


def _element_full_bbox(element, settings) -> dict | None:
    """Return full {min_x,max_x,...,z_max} dict for an IFC element."""
    sd = _get_shape_data(element, settings)
    if sd is None:
        return None
    return _verts_to_bbox(sd[0])


# ---- IFC model cache ----
_model_cache: dict[str, Any] = {}


def _open_ifc(path: Path):
    """Open an IFC file, returning a cached model if already loaded."""
    key = str(path)
    if key not in _model_cache:
        import ifcopenshell
        _model_cache[key] = ifcopenshell.open(key)
    return _model_cache[key]


def clear_ifc_caches():
    """Release all cached IFC models and shape data (call after extraction)."""
    _shape_cache.clear()
    _model_cache.clear()


# ---------------------------------------------------------------------------
#  Batch IFC element extraction  (single-pass optimisation)
# ---------------------------------------------------------------------------

def _batch_extract_proxy_data(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
) -> dict[str, list[dict]]:
    """Single-pass extraction of ALL IfcBuildingElementProxy data.

    Opens each IFC file once, iterates IfcBuildingElementProxy once,
    and classifies each element into categories based on its name.
    All bbox/footprint geometry is computed here to avoid repeated
    ``create_shape`` calls.

    Returns dict with keys: 'escalators', 'elevators', 'fare_gates',
    'security_scanners', 'blind_paths', plus per-element data dicts.
    """
    settings = _ifc_geom_settings()
    seen_guids: dict[str, set[str]] = defaultdict(set)

    # Output containers
    escalators: list[dict] = []
    elevators: list[dict] = []
    fare_gates: list[dict] = []
    security_scanners: list[dict] = []
    blind_paths: list[dict] = []
    _blind_deferred: list[tuple] = []

    for src_label, ifc_path in raw_ifc_paths.items():
        ifc_path = Path(ifc_path)
        if not ifc_path.exists():
            continue
        model = _open_ifc(ifc_path)

        for elem in model.by_type("IfcBuildingElementProxy"):
            name = elem.Name or ""
            guid = elem.GlobalId

            # --- classify ---
            cat = None
            if "扶梯" in name or "escalator" in name.lower():
                cat = "escalator"
            elif ("电梯" in name or "elevator" in name.lower() or "lift" in name.lower()) \
                    and "电梯门" not in name and "封口" not in name:
                cat = "elevator"
            elif "闸机" in name:
                cat = "fare_gate"
            elif "安检" in name:
                cat = "security_scanner"
            elif "止步块" in name:
                cat = "blind_warning"
            elif "行步块" in name:
                cat = "blind_guide"
            else:
                continue

            if guid in seen_guids[cat]:
                continue
            seen_guids[cat].add(guid)

            # --- geometry (computed once!) ---
            bb = _element_full_bbox(elem, settings)
            if bb is None:
                continue

            z_mid = (bb["min_z"] + bb["max_z"]) / 2
            level = min(levels.items(),
                        key=lambda x: abs(x[1]["elevation_m"] - z_mid))[0]

            # --- dispatch ---
            if cat == "escalator":
                fp = _element_xy_polygon(elem, settings)
                bl, tl = _infer_level_pair(bb["min_z"], bb["max_z"], levels)

                # Physical landing positions from vertex z-distribution.
                # Bottom 15% of vertices → centroid = entry at lower level;
                # Top 15% of vertices → centroid = exit at upper level.
                sd = _get_shape_data(elem, settings)
                if sd is not None:
                    _va = np.asarray(sd[0], dtype=np.float64).reshape(-1, 3)
                    _zmin, _zmax = _va[:, 2].min(), _va[:, 2].max()
                    _dz = max(_zmax - _zmin, 0.1)
                    _bot_m = _va[:, 2] < (_zmin + 0.15 * _dz)
                    _top_m = _va[:, 2] > (_zmax - 0.15 * _dz)
                    _bot_xy = [float(_va[_bot_m, 0].mean()),
                               float(_va[_bot_m, 1].mean())]
                    _top_xy = [float(_va[_top_m, 0].mean()),
                               float(_va[_top_m, 1].mean())]
                else:
                    _bot_xy = [(bb["min_x"]+bb["max_x"])/2,
                               (bb["min_y"]+bb["max_y"])/2]
                    _top_xy = _bot_xy[:]

                escalators.append({
                    "id": f"esc_{guid}", "type": "escalator",
                    "guid": guid, "name": name,
                    "footprint": fp or box(bb["min_x"], bb["min_y"],
                                           bb["max_x"], bb["max_y"]),
                    "z_min": bb["min_z"], "z_max": bb["max_z"],
                    "bottom_level": bl, "top_level": tl,
                    "bottom_xy": _bot_xy,
                    "top_xy":    _top_xy,
                    "direction": "up", "source": src_label, **bb,
                })

            elif cat == "elevator":
                fp = _element_xy_polygon(elem, settings)
                elevators.append({
                    "_guid": guid, "_name": name, "_bb": bb,
                    "_fp": fp, "_src": src_label, "_level": level,
                })

            elif cat == "fare_gate":
                fp = box(bb["min_x"], bb["min_y"], bb["max_x"], bb["max_y"])
                fare_gates.append({
                    "id": f"gate_{guid[:8]}", "type": "fare_gate",
                    "role": "fare_gate", "guid": guid, "name": name,
                    "footprint": fp,
                    "z_min": bb["min_z"], "z_max": bb["max_z"],
                    "level": level, "source": src_label, **bb,
                })

            elif cat == "security_scanner":
                fp = box(bb["min_x"], bb["min_y"], bb["max_x"], bb["max_y"])
                security_scanners.append({
                    "id": f"scan_{guid[:8]}", "type": "security_scanner",
                    "role": "security_scanner", "guid": guid, "name": name,
                    "footprint": fp,
                    "z_min": bb["min_z"], "z_max": bb["max_z"],
                    "level": level, "source": src_label, **bb,
                })

            elif cat in ("blind_warning", "blind_guide"):
                # Fast path: use IFC placement instead of create_shape
                bp_cat = "warning" if cat == "blind_warning" else "guide"
                _blind_deferred.append((elem, guid, name, bp_cat, src_label))

    # --- Fast blind-path extraction via placement matrix ---
    # Avoids 1000+ create_shape calls (~100s → <1s)
    if _blind_deferred:
        import ifcopenshell.util.placement as _ifc_plc
        import ifcopenshell.util.unit as _ifc_unit
        # Detect length unit scale (mm→m = 0.001)
        _unit_scales: dict[int, float] = {}
        # Map IFC source labels to level keys
        _src_to_level = {"platform": "F1", "concourse": "F3", "traffic": "F4"}
        for elem, guid, name, bp_cat, src_label in _blind_deferred:
            model_file = elem.wrapped_data.file
            mid = id(model_file)
            if mid not in _unit_scales:
                try:
                    _unit_scales[mid] = _ifc_unit.calculate_unit_scale(model_file)
                except Exception:
                    _unit_scales[mid] = 0.001  # fallback: assume mm
            scale = _unit_scales[mid]
            try:
                mat = _ifc_plc.get_local_placement(elem.ObjectPlacement)
                x = float(mat[0][3]) * scale
                y = float(mat[1][3]) * scale
                z = float(mat[2][3]) * scale
            except Exception:
                continue
            # Determine level from source file label (reliable) or z (fallback)
            level = _src_to_level.get(src_label)
            if level is None:
                level = min(levels.items(),
                            key=lambda kv: abs(kv[1]["elevation_m"] - z))[0]
            # Small default footprint (≈0.3m typically for tactile blocks)
            half = 0.15
            blind_paths.append({
                "id": f"tact_{guid[:8]}", "category": bp_cat,
                "guid": guid, "name": name, "level": level,
                "footprint": box(x - half, y - half, x + half, y + half),
                "source": src_label,
                "min_x": x - half, "max_x": x + half,
                "min_y": y - half, "max_y": y + half,
                "min_z": z, "max_z": z + 0.05,
            })

    return {
        "escalators": escalators,
        "elevators": elevators,
        "fare_gates": fare_gates,
        "security_scanners": security_scanners,
        "blind_paths": blind_paths,
    }


def _bbox_rectangle(row: pd.Series) -> Polygon | None:
    """Polygon from DataFrame row with min_x / max_x / min_y / max_y."""
    try:
        minx = float(row["min_x"])
        maxx = float(row["max_x"])
        miny = float(row["min_y"])
        maxy = float(row["max_y"])
        if math.isnan(minx) or (maxx - minx) < 0.01 or (maxy - miny) < 0.01:
            return None
        return box(minx, miny, maxx, maxy)
    except (KeyError, ValueError, TypeError):
        return None


# ====================================================================
#  Floor polygon extraction  (from raw IFC -> IfcSlab)
# ====================================================================

def extract_floor_from_raw_ifc(
    ifc_path: Path | str,
    level_elevation: float,
    z_tolerance: float = 2.0,
    min_slab_area: float = 10.0,
) -> Polygon | None:
    """Extract the walkable floor polygon for one level.

    Strategy
    --------
    1. Open raw IFC.
    2. Iterate all ``IfcSlab`` elements.
    3. For each slab, compute vertices via ``ifcopenshell.geom``.
    4. Keep slabs whose **max_z** (top surface) falls within
       ``[level_elevation - z_tolerance, level_elevation + z_tolerance]``.
    5. Discard slabs with 2-D area < *min_slab_area* (filters out stair
       landings approx 2 m2).
    6. Return the union of XY footprints of all remaining slabs.

    Returns ``None`` when no suitable slabs are found.
    """
    ifc_path = Path(ifc_path)
    if not ifc_path.exists():
        print(f"    [WARN] IFC not found: {ifc_path}")
        return None

    model = _open_ifc(ifc_path)
    settings = _ifc_geom_settings()

    slab_polys: list[Polygon] = []
    z_lo = level_elevation - z_tolerance
    z_hi = level_elevation + z_tolerance

    for slab in model.by_type("IfcSlab"):
        zr = _element_z_range(slab, settings)
        if zr is None:
            continue
        z_min, z_max = zr

        # Keep slab if its top surface is near the target level
        if not (z_lo <= z_max <= z_hi):
            continue

        poly = _element_xy_polygon(slab, settings)
        if poly is None or poly.is_empty or poly.area < min_slab_area:
            continue

        slab_polys.append(poly)

    if not slab_polys:
        return None

    floor = unary_union(slab_polys).buffer(0)
    return floor


# ====================================================================
#  Connector extraction from IFC  (escalators + elevators)
# ====================================================================

def _infer_level_pair(
    z_min: float,
    z_max: float,
    levels: dict,
) -> tuple[str, str]:
    """Return (bottom_level, top_level) for a connector spanning z_min->z_max."""
    elevs = sorted(
        [(k, v["elevation_m"]) for k, v in levels.items()],
        key=lambda x: x[1],
    )
    bottom = min(elevs, key=lambda x: abs(x[1] - z_min))[0]
    top    = min(elevs, key=lambda x: abs(x[1] - z_max))[0]
    if bottom == top:
        # push top to next level above bottom
        for lk, le in elevs:
            if le > levels[bottom]["elevation_m"]:
                top = lk
                break
    return bottom, top


def extract_escalators_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    _batch: dict | None = None,
) -> list[dict]:
    """Extract all unique escalators across all raw IFC files.

    If *_batch* is provided (from ``_batch_extract_proxy_data``), uses
    pre-computed data instead of re-opening IFC files.
    """
    if _batch is not None:
        return _batch["escalators"]

    # Fallback: standalone extraction (should not be reached in pipeline)
    batch = _batch_extract_proxy_data(raw_ifc_paths, levels)
    return batch["escalators"]


def extract_elevators_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    override_levels: list[str] | None = None,
    _batch: dict | None = None,
) -> list[dict]:
    """Extract elevator elements from raw IFC files.

    If *_batch* is provided, uses pre-computed data from
    ``_batch_extract_proxy_data`` instead of re-opening IFC files.
    """
    raw_elevators = (_batch or _batch_extract_proxy_data(raw_ifc_paths, levels))["elevators"]

    results: list[dict] = []
    for raw in raw_elevators:
        bb = raw["_bb"]
        fp = raw["_fp"]
        guid = raw["_guid"]
        name = raw["_name"]

        if override_levels:
            connected = list(override_levels)
        else:
            connected = [
                lk for lk, lv in sorted(levels.items(),
                                         key=lambda x: x[1]["elevation_m"])
                if bb["min_z"] - 1.0 <= lv["elevation_m"] <= bb["max_z"] + 1.0
            ]

        results.append({
            "id": f"elev_{guid[:8]}",
            "type": "elevator",
            "guid": guid,
            "name": name,
            "footprint": fp or box(bb["min_x"], bb["min_y"],
                                   bb["max_x"], bb["max_y"]),
            "z_min": bb["min_z"],
            "z_max": bb["max_z"],
            "connected_levels": connected,
            "xy": [(bb["min_x"] + bb["max_x"]) / 2,
                   (bb["min_y"] + bb["max_y"]) / 2],
            "source": raw["_src"],
            **bb,
        })

    return results


def extract_stair_flights_from_bbox(
    connector_df: pd.DataFrame,
    levels: dict,
    config: dict | None = None,
) -> list[dict]:
    """Build **stair-chain** dicts from the preprocessing CSV bbox data.

    Processing pipeline:
    1. Filter to ``stair_flight`` rows with valid bounding-box coords.
    2. **Deduplicate** rows that appear in multiple IFC subsets
       (same element-name + run-number + bbox → keep one).
    3. **Group** runs by IFC element-ID into chains (one chain per
       physical staircase structure).
    4. **Filter** blacklisted elements (config ``connectors.stair.element_blacklist``).
    5. **Detect landings** at transitions between consecutive runs.
    6. Compute per-chain z-range, connected walkable levels, and
       per-level anchor positions.

    Returns list of *stair_chain* connector dicts (one per chain),
    NOT individual flight dicts.
    """
    import re
    from src.data_loader import filter_connectors_for_navigation

    nav = filter_connectors_for_navigation(connector_df)
    flights = nav[nav["connector_subtype"] == "stair_flight"].copy()

    required = ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]
    if not all(c in flights.columns for c in required):
        return []

    flights = flights.dropna(subset=required)
    if flights.empty:
        return []

    # ---- 1. Parse element-ID and run-number from the name field ----
    #  Name format: "现场浇注楼梯:楼梯:797162 Run 3"
    _run_re = re.compile(r":(\d+)\s+Run\s+(\d+)$")

    raw_runs: list[dict] = []
    for _, row in flights.iterrows():
        name = str(row.get("name", ""))
        m = _run_re.search(name)
        if not m:
            continue
        elem_id = m.group(1)
        run_num = int(m.group(2))
        minx, maxx = float(row["min_x"]), float(row["max_x"])
        miny, maxy = float(row["min_y"]), float(row["max_y"])
        raw_runs.append({
            "element_id": elem_id,
            "run_num": run_num,
            "name": name,
            "guid": str(row.get("guid", "")),
            "z_min": float(row["min_z"]),
            "z_max": float(row["max_z"]),
            "min_x": minx, "max_x": maxx,
            "min_y": miny, "max_y": maxy,
        })

    # ---- 2. Deduplicate (same element + run + bbox) ----
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in raw_runs:
        key = (r["element_id"], r["run_num"],
               round(r["min_x"], 1), round(r["min_y"], 1),
               round(r["max_x"], 1), round(r["max_y"], 1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    print(f"      (stair dedup: {len(raw_runs)} rows → {len(deduped)} unique)")

    # ---- 3. Blacklist filter ----
    blacklist: set[str] = set()
    if config:
        bl_list = (config.get("connectors", {})
                         .get("stair", {})
                         .get("element_blacklist", []))
        blacklist = {str(e) for e in bl_list}

    if blacklist:
        before = len(deduped)
        deduped = [r for r in deduped if r["element_id"] not in blacklist]
        print(f"      (stair blacklist: removed {before - len(deduped)} "
              f"runs, {len(deduped)} remain)")

    # ---- 4. Group by element-ID into chains ----
    return _group_stair_chains(deduped, levels)


def _group_stair_chains(
    runs: list[dict],
    levels: dict,
    z_tolerance: float = 1.5,
) -> list[dict]:
    """Group deduplicated stair runs into chains, one per IFC element.

    Each chain carries:
    * Footprint union (for exclusion zones).
    * Connected walkable levels (from overall z-range).
    * Per-level anchor coordinates (entry/exit centroids).
    * Landing positions between consecutive runs.
    * Full run list (for Step 3 chain-node interpolation).

    Parameters
    ----------
    z_tolerance : float
        Extra margin (metres) when matching z-range to level elevations.
        Accounts for landing/slab thickness gaps between stair top and
        floor surface.
    """
    from collections import defaultdict

    elevs = sorted(
        [(k, v["elevation_m"]) for k, v in levels.items()],
        key=lambda x: x[1],
    )
    walkable_elevs = [
        (k, v["elevation_m"]) for k, v in levels.items()
        if v.get("is_walkable", False)
    ]
    walkable_elevs.sort(key=lambda x: x[1])

    # Group runs by element
    by_elem: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_elem[r["element_id"]].append(r)

    chains: list[dict] = []

    for elem_id, elem_runs in by_elem.items():
        # Sort by z_min so runs form bottom→top chain
        elem_runs.sort(key=lambda r: r["z_min"])

        # Overall z-range of the chain
        z_min = min(r["z_min"] for r in elem_runs)
        z_max = max(r["z_max"] for r in elem_runs)

        # Connected walkable levels (within z-range ± tolerance)
        connected: list[str] = []
        for lk, le in walkable_elevs:
            if z_min - z_tolerance <= le <= z_max + z_tolerance:
                connected.append(lk)

        # Require at least 2 connected walkable levels for public stair
        if len(connected) < 2:
            print(f"      (stair chain {elem_id}: z=[{z_min:.1f},{z_max:.1f}] "
                  f"connects {connected} — skipping, <2 walkable levels)")
            continue

        # ---- Per-level anchor positions ----
        # For each connected level, find the run whose z-range includes
        # (or is nearest to) the level's elevation.  The anchor is placed
        # at the **outward edge** of the footprint (plus a small offset)
        # so it lands just outside the exclusion zone, in walkable floor
        # area.  This ensures short snap distances to nearby floor nodes.
        _ANCHOR_OFFSET = 0.5          # metres beyond footprint boundary
        all_run_min_x = min(r["min_x"] for r in elem_runs)
        all_run_max_x = max(r["max_x"] for r in elem_runs)
        fp_cx = (all_run_min_x + all_run_max_x) / 2

        level_anchors: dict[str, dict] = {}
        for lk in connected:
            le = levels[lk]["elevation_m"]
            # Score each run: prefer the run whose z_min or z_max is
            # closest to the level elevation
            best_run = min(elem_runs,
                           key=lambda r: min(abs(r["z_min"] - le),
                                             abs(r["z_max"] - le)))
            run_cx = (best_run["min_x"] + best_run["max_x"]) / 2
            cy = (best_run["min_y"] + best_run["max_y"]) / 2
            # Push anchor to outward edge of the footprint
            if run_cx >= fp_cx:
                cx = all_run_max_x + _ANCHOR_OFFSET
            else:
                cx = all_run_min_x - _ANCHOR_OFFSET
            level_anchors[lk] = {"x": round(cx, 3), "y": round(cy, 3)}

        # ---- Landings (flat platforms between consecutive runs) ----
        landings: list[dict] = []
        for i in range(len(elem_runs) - 1):
            r1 = elem_runs[i]       # lower run
            r2 = elem_runs[i + 1]   # upper run
            # Landing z = midpoint of gap between run tops/bottoms
            lz = (r1["z_max"] + r2["z_min"]) / 2
            # Landing x,y = midpoint in the gap between runs
            lx = ((r1["min_x"] + r1["max_x"]) / 2 +
                  (r2["min_x"] + r2["max_x"]) / 2) / 2
            ly = ((r1["min_y"] + r1["max_y"]) / 2 +
                  (r2["min_y"] + r2["max_y"]) / 2) / 2
            landings.append({
                "z": round(lz, 3),
                "x": round(lx, 3),
                "y": round(ly, 3),
                "between_runs": (i, i + 1),
            })

        # ---- Footprint = union of all run bounding boxes ----
        run_boxes = [box(r["min_x"], r["min_y"], r["max_x"], r["max_y"])
                     for r in elem_runs]
        footprint = unary_union(run_boxes).buffer(0)

        guid = elem_runs[0].get("guid", "")

        chains.append({
            "id": f"stair_{elem_id}",
            "type": "stair_chain",
            "element_id": elem_id,
            "guid": guid,
            "name": f"stair_chain:{elem_id} ({len(elem_runs)} runs)",
            "footprint": footprint,
            "z_min": z_min,
            "z_max": z_max,
            "connected_levels": connected,
            "level_anchors": level_anchors,
            "landings": landings,
            "runs": elem_runs,
            "n_runs": len(elem_runs),
            "direction": "bidirectional",
            "source": "bbox_csv",
        })

    return chains


# ====================================================================
#  Fare gate & security scanner extraction (from IFC)
# ====================================================================

def extract_fare_gates_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    _batch: dict | None = None,
) -> list[dict]:
    """Extract fare gate (闸机验票门) elements from raw IFC files.

    Uses *_batch* from ``_batch_extract_proxy_data`` when available.
    """
    return (_batch or _batch_extract_proxy_data(raw_ifc_paths, levels))["fare_gates"]


def extract_security_scanners_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    _batch: dict | None = None,
) -> list[dict]:
    """Extract security scanner (安检机) elements from raw IFC files.

    Uses *_batch* from ``_batch_extract_proxy_data`` when available.
    """
    return (_batch or _batch_extract_proxy_data(raw_ifc_paths, levels))["security_scanners"]


# ====================================================================
#  Railing extraction (IFC fallback for railings missing bbox)
# ====================================================================

def extract_railings_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    existing_guids: set[str] | None = None,
) -> list[dict]:
    """Extract railing (栏杆扶手) footprints from raw IFC IfcRailing elements.

    Only extracts railings whose *guid* is NOT in *existing_guids* (to
    avoid double-counting elements already captured via the bbox CSV).
    Uses cached IFC models for performance.
    """
    settings = _ifc_geom_settings()
    seen: set[str] = set()
    results: list[dict] = []
    skip = existing_guids or set()

    for src_label, ifc_path in raw_ifc_paths.items():
        ifc_path = Path(ifc_path)
        if not ifc_path.exists():
            continue
        model = _open_ifc(ifc_path)

        for elem in model.by_type("IfcRailing"):
            guid = elem.GlobalId
            if guid in seen or guid in skip:
                continue
            seen.add(guid)

            bb = _element_full_bbox(elem, settings)
            if bb is None:
                continue

            fp = box(bb["min_x"], bb["min_y"], bb["max_x"], bb["max_y"])
            if fp.area < 0.01:
                continue

            z_mid = (bb["min_z"] + bb["max_z"]) / 2
            level = min(
                levels.items(),
                key=lambda x: abs(x[1]["elevation_m"] - z_mid),
            )[0]

            results.append({
                "guid": guid,
                "name": elem.Name or "",
                "level": level,
                "footprint": fp,
                **bb,
            })

    return results


# ====================================================================
#  Obstacle extraction  (from bbox CSV - fast, pre-calibrated)
# ====================================================================

# ---- Blind path (盲道) extraction from IFC ----

def extract_blind_paths_from_ifc(
    raw_ifc_paths: dict[str, Path],
    levels: dict,
    _batch: dict | None = None,
) -> list[dict]:
    """Extract tactile paving elements (盲道) from raw IFC files.

    Uses *_batch* from ``_batch_extract_proxy_data`` when available.
    """
    return (_batch or _batch_extract_proxy_data(raw_ifc_paths, levels))["blind_paths"]

def obstacles_from_bbox(
    obstacle_df: pd.DataFrame,
    level_key: str,
    levels: dict,
) -> list[Polygon]:
    """Build obstacle polygons from preprocessing bbox data.

    Applies z-height filtering: only keeps elements whose z-range
    overlaps [level_elev - 0.5, level_elev + level_height].  This
    prevents deep-underground slabs, above-ceiling ducts, and
    cross-level structural elements from appearing as obstacles.
    """
    from src.data_loader import get_level_elements, filter_obstacles_for_navigation

    level_obs = get_level_elements(obstacle_df, level_key, levels)
    nav_obs = filter_obstacles_for_navigation(level_obs)

    # ---------- z-height filter ----------
    elev = levels[level_key]["elevation_m"]
    z_floor = elev - 0.5          # half a metre below slab top
    z_ceil  = elev + 5.0          # generous ceiling height

    # Determine which column name to use for min_z / max_z
    # (may be suffixed _x / _y after pandas merge)
    min_z_col = ("min_z_x" if "min_z_x" in nav_obs.columns
                 else "min_z" if "min_z" in nav_obs.columns else None)
    max_z_col = ("max_z_x" if "max_z_x" in nav_obs.columns
                 else "max_z" if "max_z" in nav_obs.columns else None)

    polys = []
    for _, row in nav_obs.iterrows():
        # Apply z filter when z-data is available
        if min_z_col and max_z_col:
            z_lo = row.get(min_z_col)
            z_hi = row.get(max_z_col)
            if pd.notna(z_lo) and pd.notna(z_hi):
                if z_hi < z_floor or z_lo > z_ceil:
                    continue   # element entirely outside level's z-band

        poly = _bbox_rectangle(row)
        if poly is not None:
            polys.append(poly)
    return polys


# ====================================================================
#  Wall extraction for room exclusion  (from retained CSV + bbox CSV)
# ====================================================================

def walls_as_obstacles(
    retained_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    level_key: str,
    levels: dict,
    raw_ifc_path: Path | str | None = None,
    forbidden_zone_bounds: list[list[float]] | None = None,
) -> list[Polygon]:
    """Extract wall obstacle polygons from preprocessing data.

    For orthogonal walls (min bbox dimension ≤ ``DIAG_THRESH``) the fast
    axis-aligned bounding-box rectangle is used.  For **diagonal** walls
    (both bbox dimensions large) the actual IFC triangulated polygon is
    looked up via GUID so the footprint is not inflated.

    Walls whose bbox lies entirely inside a *forbidden zone* always use
    the cheap bbox (the zone blocks the area anyway).
    """
    from src.data_loader import get_level_elements

    level_df = get_level_elements(retained_df, level_key, levels)

    # Filter to wall IFC classes
    wall_classes = {"IfcWall", "IfcWallStandardCase", "IfcCurtainWall"}
    walls = level_df[level_df["ifc_class"].isin(wall_classes)].copy()

    if walls.empty:
        return []

    # Merge bbox if not already present
    if "min_x" not in walls.columns:
        bcols = [c for c in ["guid", "source_file",
                             "min_x", "max_x", "min_y", "max_y",
                             "min_z", "max_z"]
                 if c in bbox_df.columns]
        mk = ["guid", "source_file"] if "source_file" in bcols else ["guid"]
        walls = walls.merge(
            bbox_df[bcols].drop_duplicates(subset=mk), on=mk, how="left",
        )

    # ---------- z-height filter ----------
    elev = levels[level_key]["elevation_m"]
    z_floor = elev - 0.5
    z_ceil  = elev + 5.5

    min_z_col = ("min_z_x" if "min_z_x" in walls.columns
                 else "min_z" if "min_z" in walls.columns else None)
    max_z_col = ("max_z_x" if "max_z_x" in walls.columns
                 else "max_z" if "max_z" in walls.columns else None)

    # Pre-build forbidden-zone union for quick "skip IFC" check
    fz_union = None
    if forbidden_zone_bounds:
        fz_polys = [box(b[0], b[1], b[2], b[3]) for b in forbidden_zone_bounds]
        fz_union = unary_union(fz_polys).buffer(0)

    DIAG_THRESH = 1.5   # min bbox dim must exceed this
    DIAG_AREA   = 5.0   # bbox area must exceed this

    import re
    _re_thickness = re.compile(r"(\d{3,4})")
    from shapely.geometry import LineString

    polys: list[Polygon] = []
    n_diag = 0
    for _, row in walls.iterrows():
        if min_z_col and max_z_col:
            z_lo = row.get(min_z_col)
            z_hi = row.get(max_z_col)
            if pd.notna(z_lo) and pd.notna(z_hi):
                if z_hi < z_floor or z_lo > z_ceil:
                    continue

        try:
            dx = float(row["max_x"]) - float(row["min_x"])
            dy = float(row["max_y"]) - float(row["min_y"])
        except (KeyError, ValueError, TypeError):
            dx = dy = 0.0

        is_diag_candidate = (
            dx > DIAG_THRESH and dy > DIAG_THRESH
            and dx * dy > DIAG_AREA
        )

        # Skip wall entirely if its bbox is inside a forbidden zone.
        # The zone already blocks the area; keeping the wall as a bbox
        # rectangle creates unwanted remnant strips after walkable-passage
        # carving (e.g. thin "vertical walls" that don't exist in reality).
        if fz_union is not None:
            try:
                _bx = (float(row["min_x"]), float(row["min_y"]),
                        float(row["max_x"]), float(row["max_y"]))
                if not any(pd.isna(v) for v in _bx):
                    wall_box = box(*_bx)
                    if fz_union.contains(wall_box):
                        continue
            except (KeyError, ValueError, TypeError):
                pass

        if is_diag_candidate:
            # Approximate diagonal wall as buffered diagonal lines (X-cross)
            # Parse wall thickness from name  (e.g. "隔墙200" → 0.2 m)
            name = str(row.get("name", ""))
            m = _re_thickness.search(name)
            thickness = int(m.group(1)) / 1000.0 if m else 0.3
            thickness = max(0.1, min(thickness, 1.0))
            minx, miny = float(row["min_x"]), float(row["min_y"])
            maxx, maxy = float(row["max_x"]), float(row["max_y"])
            # Two possible diagonals → union (X shape, much smaller than bbox)
            line1 = LineString([(minx, miny), (maxx, maxy)])
            line2 = LineString([(minx, maxy), (maxx, miny)])
            buf1 = line1.buffer(thickness / 2, cap_style=2)  # flat cap
            buf2 = line2.buffer(thickness / 2, cap_style=2)
            approx = buf1.union(buf2)
            if approx.is_valid and not approx.is_empty:
                polys.append(approx)
                n_diag += 1
                continue

        poly = _bbox_rectangle(row)
        if poly is not None:
            polys.append(poly)

    if n_diag:
        print(f"    (diagonal walls resolved via geometric approx: {n_diag})")
    return polys


# ====================================================================
#  Manual spatial overrides
# ====================================================================

def _apply_manual_overrides(
    level_key: str,
    config: dict,
    floor: Polygon | MultiPolygon,
    obstacles: list[Polygon],
) -> tuple[any, list[Polygon], dict]:
    """Apply manual spatial overrides defined in experiment_config.yaml.

    Processing order:
    1. **floor_patches** – union additional rectangles into floor.
    2. **obstacle_removals** – clip existing obstacles against removal zones.
    3. **forbidden_zones** – add rectangles to obstacles (private rooms etc.).
    4. **track_zones / entrances** – stored as metadata (not geometry changes).

    Returns
    -------
    floor : Polygon/MultiPolygon (possibly enlarged by patches)
    obstacles : list[Polygon] (after removals + forbidden additions)
    metadata : dict  with keys track_zones, entrances, forbidden_zone_polys
    """
    overrides = config["station"].get("manual_overrides", {}).get(level_key, {})
    metadata: dict = {"track_zones": [], "entrances": [], "forbidden_zone_polys": []}

    if not overrides:
        return floor, obstacles, metadata

    # 1. Floor patches
    floor_patches = overrides.get("floor_patches", [])
    for fp in floor_patches:
        b = fp["bounds"]
        patch = box(b[0], b[1], b[2], b[3])
        floor = unary_union([floor, patch]).buffer(0)
        print(f"    [override] floor_patch '{fp['name']}' added "
              f"({b[2]-b[0]:.1f}×{b[3]-b[1]:.1f}m)")

    # 2. Obstacle removals (clip obstacles against removal zones)
    removals = overrides.get("obstacle_removals", [])
    for rem in removals:
        b = rem["bounds"]
        removal_poly = box(b[0], b[1], b[2], b[3])
        n_before = len(obstacles)
        new_obs = []
        for obs in obstacles:
            if obs.intersects(removal_poly):
                clipped = obs.difference(removal_poly)
                if not clipped.is_empty and clipped.area > 0.01:
                    new_obs.append(clipped)
            else:
                new_obs.append(obs)
        removed_count = n_before - len(new_obs)
        obstacles = new_obs
        print(f"    [override] obstacle_removal '{rem['name']}': "
              f"clipped/removed {removed_count} obstacles → {len(obstacles)}")

    # 3. Forbidden zones
    forbidden = overrides.get("forbidden_zones", [])
    forbidden_polys = []
    for fz in forbidden:
        b = fz["bounds"]
        poly = box(b[0], b[1], b[2], b[3])
        forbidden_polys.append(poly)
        obstacles.append(poly)
    if forbidden:
        print(f"    [override] {len(forbidden)} forbidden zones added "
              f"→ {len(obstacles)} total obstacles")
    metadata["forbidden_zone_polys"] = forbidden_polys

    # 4. Walkable passages — carve out obstacles inside forbidden zones
    #    (e.g. passage under an elevated staircase that is itself forbidden)
    passages = overrides.get("walkable_passages", [])
    for pas in passages:
        b = pas["bounds"]
        passage_poly = box(b[0], b[1], b[2], b[3])
        n_before = len(obstacles)
        new_obs = []
        for obs in obstacles:
            if obs.intersects(passage_poly):
                clipped = obs.difference(passage_poly)
                if not clipped.is_empty and clipped.area > 0.01:
                    for part in flatten_polygons(clipped):
                        new_obs.append(part)
            else:
                new_obs.append(obs)
        obstacles = new_obs
        print(f"    [override] walkable_passage '{pas['name']}' carved: "
              f"{n_before} -> {len(obstacles)} obstacles")

    # 5. Track zones (metadata only — kept walkable for simulation)
    track_zones = overrides.get("track_zones", [])
    for tz in track_zones:
        b = tz["bounds"]
        metadata["track_zones"].append({
            "name": tz["name"],
            "polygon": box(b[0], b[1], b[2], b[3]),
        })
    if track_zones:
        print(f"    [override] {len(track_zones)} track zones registered")

    # 6. Entrances (metadata only)
    entrances = overrides.get("entrances", [])
    for ent in entrances:
        b = ent["bounds"]
        metadata["entrances"].append({
            "name": ent["name"],
            "polygon": box(b[0], b[1], b[2], b[3]),
        })
    if entrances:
        print(f"    [override] {len(entrances)} entrances registered")

    # 7. Dynamic doors (platform screen doors + elevator doors)
    #    Added as OBSTACLES (default state = closed).  Metadata records
    #    each individual door segment so the ABM can toggle graph-edges
    #    without mutating obstacle geometry.
    dynamic_doors_cfg = overrides.get("dynamic_doors", {})
    all_dynamic_doors: list[dict] = []

    # --- Platform screen doors ---
    for psd in dynamic_doors_cfg.get("platform_screen_doors", []):
        b = psd["barrier_bounds"]
        barrier_poly = box(b[0], b[1], b[2], b[3])
        obstacles.append(barrier_poly)           # full barrier = obstacle

        # Generate door x-positions from parametric definition
        door_xs: list[float] = [psd["first_door_x"]]
        for i in range(1, psd["door_count"]):
            door_xs.append(
                psd["second_door_x"] + (i - 1) * psd["door_interval"]
            )

        w = psd["door_width"]
        y1, y2 = b[1], b[3]
        y_center = round((y1 + y2) / 2, 3)

        for i, dx in enumerate(door_xs):
            all_dynamic_doors.append({
                "id": f"{psd['name']}_D{i+1:02d}",
                "type": "platform_screen_door",
                "side": psd.get("side", ""),
                "group": psd["name"],
                "bounds": [round(dx, 2), y1, round(dx + w, 2), y2],
                "center_x": round(dx + w / 2, 2),
                "center_y": y_center,
                "default_state": "closed",
            })

        print(f"    [override] PSD barrier '{psd['name']}': "
              f"wall {b[2]-b[0]:.0f}m + {len(door_xs)} door segments")

    # --- Elevator doors ---
    #   Shaft walls become a static obstacle (the whole shaft box).
    #   A door opening is carved on one face of the shaft.  The door
    #   metadata is recorded for ABM toggle-edges, same as PSD doors.
    #   Interior capacity nodes are emitted separately by node_sampler.
    for ed in dynamic_doors_cfg.get("elevator_doors", []):
        # Only activate on levels the elevator serves
        elev_levels = ed.get("levels", [])
        if level_key not in elev_levels:
            continue

        sb = ed["shaft_bounds"]       # [min_x, min_y, max_x, max_y]
        shaft_poly = box(sb[0], sb[1], sb[2], sb[3])

        face = ed.get("door_face", "south")
        dw = ed.get("door_width", 1.2)
        dcx = ed.get("door_center_x", (sb[0] + sb[2]) / 2)
        capacity = ed.get("capacity", 20)

        # Compute door opening bounds on the specified face
        half_w = dw / 2
        if face == "south":
            door_bounds = [dcx - half_w, sb[1], dcx + half_w, sb[1] + 0.1]
            door_cy = sb[1]
        elif face == "north":
            door_bounds = [dcx - half_w, sb[3] - 0.1, dcx + half_w, sb[3]]
            door_cy = sb[3]
        elif face == "west":
            dcy = ed.get("door_center_y", (sb[1] + sb[3]) / 2)
            door_bounds = [sb[0], dcy - half_w, sb[0] + 0.1, dcy + half_w]
            door_cy = dcy
        else:   # east
            dcy = ed.get("door_center_y", (sb[1] + sb[3]) / 2)
            door_bounds = [sb[2] - 0.1, dcy - half_w, sb[2], dcy + half_w]
            door_cy = dcy

        # Carve door opening from shaft to make the barrier
        door_opening = box(*door_bounds)
        barrier = shaft_poly.difference(door_opening)
        if not barrier.is_empty:
            obstacles.append(barrier)

        all_dynamic_doors.append({
            "id": ed["name"],
            "type": "elevator_door",
            "face": face,
            "shaft_bounds": sb,
            "door_bounds": [round(v, 3) for v in door_bounds],
            "bounds": [round(v, 3) for v in door_bounds],
            "center_x": round(dcx, 3),
            "center_y": round(door_cy, 3),
            "capacity": capacity,
            "levels": elev_levels,
            "default_state": "closed",
        })
        print(f"    [override] Elevator door '{ed['name']}' on {level_key}: "
              f"shaft {sb[2]-sb[0]:.1f}×{sb[3]-sb[1]:.1f}m, "
              f"door {face} w={dw}m, cap={capacity}")

    metadata["dynamic_doors"] = all_dynamic_doors
    if all_dynamic_doors:
        print(f"    [override] {len(all_dynamic_doors)} dynamic doors total")

    # 8. Fare gate cluster barriers (闸机组)
    #    Replace individual IFC gate footprints with one solid wall per cluster,
    #    then record evenly-spaced passage positions for node/edge generation.
    metadata["fare_gate_passages"] = []
    metadata["fare_gate_groups"] = []   # wall polygon + direction per barrier group
    for gate_group in dynamic_doors_cfg.get("fare_gate_barriers", []):
        b = gate_group["barrier_bounds"]
        wall = box(b[0], b[1], b[2], b[3])
        obstacles.append(wall)
        metadata["fare_gate_groups"].append({
            "name": gate_group["name"],
            "direction": gate_group.get("direction", "inbound"),
            "barrier_bounds": b,
            "wall_polygon": wall,
        })

        gate_axis = gate_group.get("gate_axis", "y")
        passage_count = gate_group.get("passage_count", 5)
        margin = 0.4  # metres from barrier end to first/last passage

        if gate_axis == "y":
            span = b[3] - b[1] - 2 * margin
            step = span / (passage_count - 1) if passage_count > 1 else 0
            for i in range(passage_count):
                py = b[1] + margin + i * step
                metadata["fare_gate_passages"].append({
                    "id": f"fg_{gate_group['name']}_P{i+1:02d}",
                    "group": gate_group["name"],
                    "direction": gate_group.get("direction", "inbound"),
                    "paid_side": gate_group.get("paid_side", "east"),
                    "gate_axis": gate_axis,
                    "barrier_bounds": b,
                    "center_x": (b[0] + b[2]) / 2,
                    "center_y": round(py, 3),
                })
        else:  # x-axis gates
            span = b[2] - b[0] - 2 * margin
            step = span / (passage_count - 1) if passage_count > 1 else 0
            for i in range(passage_count):
                px = b[0] + margin + i * step
                metadata["fare_gate_passages"].append({
                    "id": f"fg_{gate_group['name']}_P{i+1:02d}",
                    "group": gate_group["name"],
                    "direction": gate_group.get("direction", "inbound"),
                    "paid_side": gate_group.get("paid_side", "north"),
                    "gate_axis": gate_axis,
                    "barrier_bounds": b,
                    "center_x": round(px, 3),
                    "center_y": (b[1] + b[3]) / 2,
                })

    n_fg_groups = len(dynamic_doors_cfg.get("fare_gate_barriers", []))
    if n_fg_groups:
        print(f"    [override] {n_fg_groups} fare gate barriers, "
              f"{len(metadata['fare_gate_passages'])} passages")

    # 9. Security scanner cluster barriers (安检机组)
    metadata["scanner_passages"] = []
    for sc_group in dynamic_doors_cfg.get("security_scanner_barriers", []):
        b = sc_group["barrier_bounds"]
        wall = box(b[0], b[1], b[2], b[3])
        obstacles.append(wall)

        scanner_axis = sc_group.get("scanner_axis", "y")
        passage_count = sc_group.get("passage_count", 3)
        margin = 0.3

        if scanner_axis == "y":
            span = b[3] - b[1] - 2 * margin
            step = span / (passage_count - 1) if passage_count > 1 else 0
            for i in range(passage_count):
                py = b[1] + margin + i * step
                metadata["scanner_passages"].append({
                    "id": f"sc_{sc_group['name']}_P{i+1:02d}",
                    "group": sc_group["name"],
                    "scanner_axis": scanner_axis,
                    "approach_side": sc_group.get("approach_side", "west"),
                    "barrier_bounds": b,
                    "center_x": (b[0] + b[2]) / 2,
                    "center_y": round(py, 3),
                })
        else:
            span = b[2] - b[0] - 2 * margin
            step = span / (passage_count - 1) if passage_count > 1 else 0
            for i in range(passage_count):
                px = b[0] + margin + i * step
                metadata["scanner_passages"].append({
                    "id": f"sc_{sc_group['name']}_P{i+1:02d}",
                    "group": sc_group["name"],
                    "scanner_axis": scanner_axis,
                    "approach_side": sc_group.get("approach_side", "south"),
                    "barrier_bounds": b,
                    "center_x": round(px, 3),
                    "center_y": (b[1] + b[3]) / 2,
                })

    n_sc_groups = len(dynamic_doors_cfg.get("security_scanner_barriers", []))
    if n_sc_groups:
        print(f"    [override] {n_sc_groups} scanner barriers, "
              f"{len(metadata['scanner_passages'])} passages")

    return floor, obstacles, metadata


# ====================================================================
#  Level-geometry assembly
# ====================================================================

def _floor_fallback_from_elements(
    retained_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    level_key: str,
    levels: dict,
) -> Polygon | None:
    """Fallback: bounding rectangle of all retained elements on a level."""
    from src.data_loader import get_level_elements

    level_df = get_level_elements(retained_df, level_key, levels)
    # Merge bbox if not present
    if "min_x" not in level_df.columns:
        bcols = [c for c in ["guid", "source_file", "min_x", "max_x",
                              "min_y", "max_y"] if c in bbox_df.columns]
        mk = ["guid", "source_file"] if "source_file" in bcols else ["guid"]
        level_df = level_df.merge(
            bbox_df[bcols].drop_duplicates(subset=mk), on=mk, how="left",
        )

    valid = level_df.dropna(subset=["min_x", "max_x", "min_y", "max_y"])
    if valid.empty:
        return None

    return box(
        float(valid["min_x"].min()),
        float(valid["min_y"].min()),
        float(valid["max_x"].max()),
        float(valid["max_y"].max()),
    )


def extract_level_geometry(
    level_key: str,
    config: dict,
    data: dict,
    all_connectors: list[dict],
    control_points: list[dict] | None = None,
    ifc_railings: list[dict] | None = None,
) -> dict:
    """Assemble floor + obstacles + walls + railings + control_points + connectors.

    Parameters
    ----------
    all_connectors : list[dict]
        Master list produced by ``extract_all_connectors``.
    control_points : list[dict] | None
        Fare gates + security scanners (their footprints become obstacles).
    ifc_railings : list[dict] | None
        Railings extracted from IFC (bbox-missing ones only).
    """
    levels = config["station"]["levels"]
    level_info = levels[level_key]
    elev = level_info["elevation_m"]

    print(f"  [{level_key}] Extracting geometry (elev={elev}m) ...")

    # ---- Floor ----
    floor = None
    # Determine which raw IFC maps to this level
    level_raw_map = config["station"].get("level_raw_ifc", {})
    raw_ifc_key = level_raw_map.get(level_key)
    raw_ifc_paths = data.get("ifc_raw_paths", {})

    if raw_ifc_key and raw_ifc_key in raw_ifc_paths:
        raw_path = raw_ifc_paths[raw_ifc_key]
        floor = extract_floor_from_raw_ifc(raw_path, elev)
        if floor is not None:
            from src.utils import flatten_polygons as _fp
            print(f"    Floor from IFC: {floor.area:.0f} m2"
                  f"  ({len(_fp(floor))} polygon(s))")

    if floor is None:
        # Fallback: bounding rectangle of retained elements
        floor = _floor_fallback_from_elements(
            data["retained_df"], data["bbox_df"], level_key, levels,
        )
        if floor is not None:
            print(f"    Floor FALLBACK (bbox envelope): {floor.area:.0f} m2")
        else:
            floor = box(0, 0, 150, 25)
            print(f"    Floor ABSOLUTE FALLBACK: 150x25 m")

    # ---- Obstacles (from preprocessing CSV) ----
    obstacles = obstacles_from_bbox(data["obstacle_df"], level_key, levels)
    print(f"    Obstacles (CSV): {len(obstacles)}")

    # ---- Walls as obstacles (for room exclusion) ----
    # Pass raw IFC path for diagonal-wall polygon resolution
    # and forbidden zone bounds so walls inside zones skip IFC lookup
    raw_wall_path = raw_ifc_paths.get(raw_ifc_key) if raw_ifc_key else None
    fz_bounds = [
        fz["bounds"] for fz in
        config["station"].get("manual_overrides", {}).get(level_key, {}).get("forbidden_zones", [])
    ]
    wall_obs = walls_as_obstacles(
        data["retained_df"], data["bbox_df"], level_key, levels,
        raw_ifc_path=raw_wall_path,
        forbidden_zone_bounds=fz_bounds or None,
    )
    if wall_obs:
        obstacles.extend(wall_obs)
        print(f"    Wall obstacles: {len(wall_obs)}  (total: {len(obstacles)})")

    # ---- Railings from IFC (bbox-missing fallback) ----
    level_rails = [
        r for r in (ifc_railings or [])
        if r.get("level") == level_key
    ]
    if level_rails:
        for r in level_rails:
            fp = r.get("footprint")
            if fp is not None and not fp.is_empty:
                obstacles.append(fp)
        print(f"    Railing obstacles (IFC): {len(level_rails)}  (total: {len(obstacles)})")

    # ---- Control points as obstacles ----
    #    When a level has fare_gate_barriers / security_scanner_barriers
    #    defined in dynamic_doors, the gate/scanner IFC footprints are
    #    replaced by the solid barrier wall (added in Section 8/9 below).
    #    Individual CP obstacles are therefore skipped for those types.
    level_overrides_dd = (
        config["station"].get("manual_overrides", {})
        .get(level_key, {}).get("dynamic_doors", {})
    )
    _has_fg_barriers = bool(level_overrides_dd.get("fare_gate_barriers", []))
    _has_sc_barriers = bool(level_overrides_dd.get("security_scanner_barriers", []))

    level_cps = [
        cp for cp in (control_points or [])
        if cp.get("level") == level_key
    ]
    cp_obstacle_polys = []
    cp_deferred = 0
    for cp in level_cps:
        cp_type = cp.get("type", "")
        if _has_fg_barriers and cp_type == "fare_gate":
            cp_deferred += 1
            continue   # handled by solid barrier wall (Section 8)
        if _has_sc_barriers and cp_type == "security_scanner":
            cp_deferred += 1
            continue   # handled by solid barrier wall (Section 9)
        fp = cp.get("footprint")
        if fp is not None and not fp.is_empty:
            cp_obstacle_polys.append(fp)
    if cp_obstacle_polys:
        obstacles.extend(cp_obstacle_polys)
        print(f"    Control-point obstacles (gates/scanners): "
              f"{len(cp_obstacle_polys)}  (total: {len(obstacles)})")
    if cp_deferred:
        print(f"    [{cp_deferred} gate/scanner CPs deferred to barrier builder]")

    # ---- Apply manual spatial overrides ----
    floor, obstacles, override_meta = _apply_manual_overrides(
        level_key, config, floor, obstacles,
    )

    obs_union = unary_union(obstacles).buffer(0) if obstacles else Polygon()

    # ---- Connectors that touch this level ----
    level_connectors = [
        c for c in all_connectors
        if _connector_touches_level(c, level_key)
    ]
    print(f"    Connectors touching {level_key}: {len(level_connectors)}")

    # ---- Filter ALL connectors inside forbidden zones ----
    #  Any connector whose footprint overlaps >50 % with the union of
    #  forbidden zones is removed, regardless of type.
    forbidden_polys = override_meta.get("forbidden_zone_polys", [])
    if forbidden_polys:
        fz_union = unary_union(forbidden_polys).buffer(0)
        filtered_connectors: list[dict] = []
        for c in level_connectors:
            fp = c.get("footprint")
            if fp is not None and not fp.is_empty:
                overlap_area = fp.intersection(fz_union).area
                if overlap_area > 0.5 * fp.area:   # >50 % inside → skip
                    print(f"      [skip] connector {c['id']} "
                          f"({c['type']}, {c.get('name','')[:35]}) "
                          f"in forbidden zone")
                    continue
            filtered_connectors.append(c)
        n_removed = len(level_connectors) - len(filtered_connectors)
        if n_removed:
            print(f"    Connectors after forbidden-zone filter: "
                  f"{len(filtered_connectors)} (removed {n_removed})")
        level_connectors = filtered_connectors

    # ---- Walkable area = floor - obstacles ----
    walkable = floor.difference(obs_union) if not obs_union.is_empty else floor
    if walkable.is_empty:
        walkable = floor

    return {
        "floor": floor,
        "obstacles": obstacles,
        "obstacle_union": obs_union,
        "connectors": level_connectors,
        "control_points": level_cps,
        "walkable": walkable,
        "bbox": list(floor.bounds),
        "level_key": level_key,
        "elevation_m": elev,
        "track_zones": override_meta.get("track_zones", []),
        "entrances": override_meta.get("entrances", []),
        "forbidden_zone_polys": override_meta.get("forbidden_zone_polys", []),
        "dynamic_doors": override_meta.get("dynamic_doors", []),
        "fare_gate_passages": override_meta.get("fare_gate_passages", []),
        "fare_gate_groups": override_meta.get("fare_gate_groups", []),
        "scanner_passages": override_meta.get("scanner_passages", []),
    }


def _connector_touches_level(c: dict, level_key: str) -> bool:
    """True if the connector is relevant to *level_key*."""
    if c["type"] in ("elevator", "stair_chain"):
        return level_key in c.get("connected_levels", [])
    return (c.get("bottom_level") == level_key or
            c.get("top_level") == level_key)


# ====================================================================
#  Top-level orchestrator
# ====================================================================

def extract_all_connectors(config: dict, data: dict) -> tuple[list[dict], list[dict]]:
    """Extract ALL connectors + control points (fare gates, scanners).

    Should be called once before per-level geometry extraction so
    that every level can reference the same master lists.

    Returns
    -------
    connectors : list[dict]
        Escalators + elevators + stair flights.
    control_points : list[dict]
        Fare gates + security scanners (physical obstacles with routing role).
    """
    levels = config["station"]["levels"]
    raw_ifc_paths = data.get("ifc_raw_paths", {})

    # --- Batch-extract all IfcBuildingElementProxy in one pass ---
    import time as _time
    _t0 = _time.perf_counter()
    batch = _batch_extract_proxy_data(raw_ifc_paths, levels)
    print(f"\n  Batch IFC proxy extraction: {_time.perf_counter()-_t0:.1f}s "
          f"(esc={len(batch['escalators'])}, elev={len(batch['elevators'])}, "
          f"gate={len(batch['fare_gates'])}, scan={len(batch['security_scanners'])}, "
          f"blind={len(batch['blind_paths'])})")

    print("\n  Extracting connectors ...")

    # 1. Escalators from batch
    escalators = extract_escalators_from_ifc(raw_ifc_paths, levels, _batch=batch)
    print(f"    Escalators from IFC: {len(escalators)}")

    # 2. Elevators from batch (with optional manual override)
    elevator_override = config["station"].get("elevator_override", {})
    override_levels = elevator_override.get("connected_levels")
    elevators = extract_elevators_from_ifc(raw_ifc_paths, levels, override_levels, _batch=batch)
    print(f"    Elevators from IFC: {len(elevators)}")

    # 3. Stair flights from bbox CSV (deduplicated + chained)
    flights = extract_stair_flights_from_bbox(data["connector_df"], levels, config)
    print(f"    Stair chains from bbox: {len(flights)}")

    all_conn = escalators + elevators + flights

    # Summary
    for c in all_conn:
        if c["type"] == "elevator":
            lvls = ",".join(c.get("connected_levels", []))
            print(f"      {c['id']}  {c['name'][:30]}  levels=[{lvls}]"
                  f"  z=[{c['z_min']:.1f},{c['z_max']:.1f}]")
        elif c["type"] == "stair_chain":
            lvls = ",".join(c.get("connected_levels", []))
            anchors = c.get("level_anchors", {})
            anchor_str = "  ".join(f"{lk}=({a['x']:.1f},{a['y']:.1f})"
                                   for lk, a in anchors.items())
            print(f"      {c['id']}  {c['type']}  elem={c['element_id']}  "
                  f"levels=[{lvls}]  {c['n_runs']} runs  "
                  f"z=[{c['z_min']:.1f},{c['z_max']:.1f}]  "
                  f"landings={len(c['landings'])}  anchors: {anchor_str}")
        else:
            print(f"      {c['id']}  {c['type']}  {c['name'][:30]}"
                  f"  {c.get('bottom_level','?')}->{c.get('top_level','?')}"
                  f"  z=[{c['z_min']:.1f},{c['z_max']:.1f}]")

    # ---- Control points (fare gates + security scanners) ----
    print("\n  Extracting control points (fare gates + security scanners) ...")

    fare_gates = extract_fare_gates_from_ifc(raw_ifc_paths, levels, _batch=batch)
    print(f"    Fare gates from IFC: {len(fare_gates)}")

    scanners = extract_security_scanners_from_ifc(raw_ifc_paths, levels, _batch=batch)
    print(f"    Security scanners from IFC: {len(scanners)}")

    control_points = fare_gates + scanners

    for cp in control_points[:5]:
        print(f"      {cp['id']}  {cp['type']}  {cp['name'][:30]}"
              f"  level={cp['level']}  z=[{cp['z_min']:.1f},{cp['z_max']:.1f}]")
    if len(control_points) > 5:
        print(f"      ... ({len(control_points) - 5} more)")

    return all_conn, control_points, batch


def extract_all_levels(
    config: dict,
    data: dict,
    *,
    use_cache: bool = True,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """Extract geometry for all walkable levels + global connectors.

    Results are pickled after the first run so subsequent calls (e.g.
    Step 2, Step 3 running in the same or a new process) reload in ~0.5 s
    instead of re-parsing IFC files for ~60 s.  The cache is invalidated
    automatically whenever any source IFC file changes on disk or the
    station/sampling config keys change.

    Returns
    -------
    all_geometry : dict[str, dict]
        Keyed by level_key.  Each value has *floor*, *obstacles*,
        *walkable*, *connectors*, *control_points*, *bbox*, etc.
    all_connectors : list[dict]
        Master connector list (escalators + elevators + stair flights).
    control_points : list[dict]
        Fare gates + security scanners with routing roles.
    """
    # ------------------------------------------------------------------
    # Cache lookup
    # ------------------------------------------------------------------
    _CACHE_VERSION = "v4"
    cache_file: Path | None = None
    try:
        step1_dir = Path(config["output"]["step_dirs"]["step1"])
        cache_file = step1_dir / "_geometry_cache.pkl"
    except (KeyError, TypeError):
        use_cache = False

    def _cache_key() -> dict:
        raw_paths = data.get("ifc_raw_paths", {})
        mtimes: dict[str, float] = {}
        for k, v in raw_paths.items():
            p = Path(v)
            if p.exists():
                mtimes[k] = p.stat().st_mtime
        cfg_src = (
            str(config.get("station", {}))
            + str(config.get("sampling", {}))
        )
        return {
            "version": _CACHE_VERSION,
            "mtimes": mtimes,
            "cfg_hash": hashlib.md5(cfg_src.encode()).hexdigest()[:8],
        }

    if use_cache and cache_file is not None and cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("key") == _cache_key():
                print("  [CACHE] Step1 geometry reloaded from cache (~0.5 s)")
                return (
                    cached["all_geometry"],
                    cached["all_connectors"],
                    cached["control_points"],
                )
        except Exception:
            pass  # corrupt or version-mismatch → fall through to full extraction

    # ------------------------------------------------------------------
    # Full extraction (original logic, unchanged)
    # ------------------------------------------------------------------
    result = _extract_all_levels_impl(config, data)

    # Persist cache
    if use_cache and cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as fh:
                pickle.dump(
                    {
                        "key": _cache_key(),
                        "all_geometry": result[0],
                        "all_connectors": result[1],
                        "control_points": result[2],
                    },
                    fh,
                    protocol=4,
                )
            print("  [CACHE] Step1 geometry cached for future runs")
        except Exception as exc:
            print(f"  [CACHE] Warning: could not write cache: {exc}")

    return result


def _extract_all_levels_impl(
    config: dict,
    data: dict,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    levels = config["station"]["levels"]

    # Step A: connectors + control points first (global, cross-level)
    all_connectors, control_points, batch = extract_all_connectors(config, data)

    # Step A2: railings from IFC (fill gaps where bbox CSV is missing)
    raw_ifc_paths = data.get("ifc_raw_paths", {})
    # Collect GUIDs of railings already covered by obstacle bbox CSV
    obs_df = data["obstacle_df"]
    existing_rail_guids: set[str] = set()
    if "ifc_class" in obs_df.columns and "guid" in obs_df.columns:
        bbox_df = data["bbox_df"]
        rail_obs = obs_df[obs_df["ifc_class"] == "IfcRailing"]
        if "min_x" not in rail_obs.columns:
            rail_obs = rail_obs.merge(
                bbox_df[["guid", "source_file", "min_x"]].drop_duplicates(
                    subset=["guid", "source_file"]),
                on=["guid", "source_file"], how="left",
            )
        has_bbox = rail_obs.dropna(subset=["min_x"])
        existing_rail_guids = set(has_bbox["guid"].unique())

    ifc_railings = extract_railings_from_ifc(
        raw_ifc_paths, levels, existing_guids=existing_rail_guids,
    )
    if ifc_railings:
        print(f"\n  Railings from IFC (bbox-missing): {len(ifc_railings)}")

    # Step A3: blind paths (盲道) from batch
    blind_paths = extract_blind_paths_from_ifc(raw_ifc_paths, levels, _batch=batch)
    if blind_paths:
        from collections import Counter as _Cnt
        bp_lvl = _Cnt(bp["level"] for bp in blind_paths)
        bp_cat = _Cnt(bp["category"] for bp in blind_paths)
        print(f"\n  Blind paths from IFC: {len(blind_paths)}"
              f"  ({dict(bp_cat)}, per-level: {dict(bp_lvl)})")

    # Step B: per-level geometry
    all_geometry: dict[str, dict] = {}

    for level_key, level_info in levels.items():
        if level_info.get("is_walkable", False):
            all_geometry[level_key] = extract_level_geometry(
                level_key, config, data, all_connectors,
                control_points, ifc_railings,
            )
            # Attach blind paths for this level
            all_geometry[level_key]["blind_paths"] = [
                bp for bp in blind_paths if bp["level"] == level_key
            ]

    # F2: connector-pass geometry (non-walkable level)
    if "F2" in levels and not levels["F2"].get("is_walkable", False):
        f2_conns = [c for c in all_connectors
                    if _connector_passes_through_f2(c, levels)]
        f2_floor = _build_f2_connector_envelope(f2_conns, config)
        f2_bbox = list(f2_floor.bounds) if (f2_floor and not f2_floor.is_empty) else [0, 0, 0, 0]
        all_geometry["F2"] = {
            "floor": f2_floor,
            "obstacles": [],
            "obstacle_union": Polygon(),
            "connectors": f2_conns,
            "control_points": [],
            "walkable": f2_floor,
            "bbox": f2_bbox,
            "level_key": "F2",
            "elevation_m": levels["F2"]["elevation_m"],
            "track_zones": [],
            "entrances": [],
            "forbidden_zone_polys": [],
            "dynamic_doors": [],
            "fare_gate_passages": [],
            "fare_gate_groups": [],
            "scanner_passages": [],
            "blind_paths": [],
        }
        area = f2_floor.area if (f2_floor and not f2_floor.is_empty) else 0.0
        print(f"  [F2] Connector-pass geometry: {len(f2_conns)} pass-through elements, "
              f"envelope={area:.1f} m2")

    # Release cached IFC models and shape data to free memory
    clear_ifc_caches()

    return all_geometry, all_connectors, control_points


def _connector_passes_through_f2(c: dict, levels: dict) -> bool:
    """True if the connector's z-range passes through F2 (5.3 m)."""
    f2_elev = levels.get("F2", {}).get("elevation_m", 5.3)
    return c.get("z_min", 99) <= f2_elev <= c.get("z_max", -99)


def _build_f2_connector_envelope(conns: list[dict], config: dict) -> Polygon | MultiPolygon:
    """Build a coarse F2 geometry envelope from connector footprints.

    F2 is non-walkable by design, but a footprint envelope improves
    diagnostics and visualization consistency.
    """
    if not conns:
        return Polygon()

    polys: list[Polygon | MultiPolygon] = []
    for c in conns:
        fp = c.get("footprint")
        if fp is None or fp.is_empty:
            continue
        polys.append(fp)

    if not polys:
        return Polygon()

    pad = (config.get("geometry", {}).get("f2_connector_envelope_pad_m", 0.6))
    try:
        pad = float(pad)
    except (TypeError, ValueError):
        pad = 0.6
    pad = max(0.0, pad)

    env = unary_union(polys)
    if pad > 0:
        env = env.buffer(pad)
    return env.buffer(0)


# ====================================================================
#  Persistence
# ====================================================================

def save_geometry_outputs(
    all_geometry: dict,
    all_connectors: list[dict],
    out_dir: str | Path,
    control_points: list[dict] | None = None,
) -> None:
    """Write GeoJSON + summary JSON to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for level_key, geom in all_geometry.items():
        ldir = out_dir / level_key
        ldir.mkdir(parents=True, exist_ok=True)

        # Floor
        if geom["floor"] is not None:
            feats = [polygon_feature(p, {"level": level_key, "type": "floor"})
                     for p in flatten_polygons(geom["floor"])]
            write_geojson(ldir / "floor.geojson", feats)

        # Obstacles
        obs_feats = []
        for i, poly in enumerate(geom["obstacles"]):
            for p in flatten_polygons(poly):
                obs_feats.append(polygon_feature(
                    p, {"level": level_key, "type": "obstacle", "idx": i}))
        write_geojson(ldir / "obstacles.geojson", obs_feats)

        # Connectors on this level
        conn_feats = []
        for c in geom["connectors"]:
            fp = c.get("footprint")
            if fp is None or fp.is_empty:
                continue
            props = {k: v for k, v in c.items()
                     if k != "footprint" and not isinstance(v, (Polygon, MultiPolygon))}
            for p in flatten_polygons(fp):
                conn_feats.append(polygon_feature(p, props))
        write_geojson(ldir / "connectors.geojson", conn_feats)

        # Control points on this level
        cp_feats = []
        for cp in geom.get("control_points", []):
            fp = cp.get("footprint")
            if fp is None or fp.is_empty:
                continue
            props = {k: v for k, v in cp.items()
                     if k != "footprint" and not isinstance(v, (Polygon, MultiPolygon))}
            for p in flatten_polygons(fp):
                cp_feats.append(polygon_feature(p, props))
        write_geojson(ldir / "control_points.geojson", cp_feats)

        # Walkable
        if geom["walkable"] is not None:
            feats = [polygon_feature(p, {"level": level_key, "type": "walkable"})
                     for p in flatten_polygons(geom["walkable"])]
            write_geojson(ldir / "walkable.geojson", feats)

        floor_area = geom["floor"].area if geom["floor"] else 0
        walk_area  = geom["walkable"].area if geom["walkable"] else 0
        n_cp = len(geom.get("control_points", []))
        n_doors = len(geom.get("dynamic_doors", []))
        summary[level_key] = {
            "floor_area_m2": round(floor_area, 1),
            "n_obstacles": len(geom["obstacles"]),
            "n_connectors": len(geom["connectors"]),
            "n_control_points": n_cp,
            "n_dynamic_doors": n_doors,
            "walkable_area_m2": round(walk_area, 1),
            "elevation_m": geom["elevation_m"],
            "bbox": [round(v, 2) for v in geom["bbox"]],
        }

    # Global connector manifest
    conn_manifest = []
    for c in all_connectors:
        entry = {k: v for k, v in c.items()
                 if k != "footprint" and not isinstance(v, (Polygon, MultiPolygon))}
        if c.get("footprint") is not None:
            entry["footprint_area_m2"] = round(c["footprint"].area, 2)
        conn_manifest.append(entry)

    # Global control point manifest
    cp_manifest = []
    for cp in (control_points or []):
        entry = {k: v for k, v in cp.items()
                 if k != "footprint" and not isinstance(v, (Polygon, MultiPolygon))}
        if cp.get("footprint") is not None:
            entry["footprint_area_m2"] = round(cp["footprint"].area, 2)
        cp_manifest.append(entry)

    # Dynamic doors manifest  (all levels combined)
    door_manifest = []
    for level_key, geom in all_geometry.items():
        for dd in geom.get("dynamic_doors", []):
            entry = {k: v for k, v in dd.items()
                     if not isinstance(v, (Polygon, MultiPolygon))}
            entry["level"] = level_key
            door_manifest.append(entry)

    dump_json(out_dir / "geometry_summary.json", summary)
    dump_json(out_dir / "connectors_manifest.json", conn_manifest)
    dump_json(out_dir / "control_points_manifest.json", cp_manifest)
    if door_manifest:
        dump_json(out_dir / "dynamic_doors_manifest.json", door_manifest)

    print(f"\n  [Step 1] Saved geometry for {len(all_geometry)} levels, "
          f"{len(all_connectors)} connectors, "
          f"{len(cp_manifest)} control points, "
          f"{len(door_manifest)} dynamic doors -> {out_dir}")
