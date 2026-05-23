"""
Step 2: Node Sampler  (v2 – human-scale aware)
================================================

Grid-based walkable node sampling with clearance filtering.

Design choices
--------------
* **Grid resolution ≈ 0.5 m** matches human shoulder width (~0.45 m).
  This ensures the grid can resolve passages as narrow as one person
  (e.g. fare-gate slot width 0.43 m still gets at least one column of
  nodes, though clearance will shrink near-edge nodes' usability).

* **min_clearance = 0.25 m** equals the agent body radius.
  Nodes closer than one body radius to any obstacle edge are tagged
  unusable – an agent centred there would clip the obstacle.

* **Connector exclusion zone** prevents floor-grid nodes from
  overlapping stair/escalator footprints (those areas will be populated
  with dedicated connector-chain nodes in Step 3).

* **Control-point buffer** adds a small margin around fare gates and
  security scanners so that floor nodes don't sit *on* the device
  polygon boundary (which is already subtracted from walkable area in
  Step 1, but numerical noise can leave borderline nodes).

* **Door nodes** – for each dynamic door (platform screen door,
  elevator door) two forced nodes are injected: one on the *platform*
  side and one on the *track / shaft* side.  These guarantee that the
  graph (Step 3) can create toggle-edges through the barrier.

For each walkable level (F1, F3, F4):
1. Overlay regular grid on walkable polygon
2. Filter out points inside stair/escalator exclusion zones
3. Compute clearance distance to nearest obstacle polygon
4. Mark nodes as usable if clearance >= min_clearance
5. Inject door-adjacent nodes for dynamic doors

Produces per-level CSV and GeoJSON node files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapely                       # shapely ≥ 2.0 vectorised API
from shapely.geometry import Point
from shapely.ops import unary_union

from src.utils import (
    flatten_polygons, write_geojson, point_feature,
    dump_json,
)


# ====================================================================
#  Single-level sampling  (vectorised  – shapely ≥ 2.0 + numpy)
# ====================================================================

def sample_level_nodes(
    level_key: str,
    elevation_m: float,
    floor_geom,
    obstacle_union,
    grid_res: float,
    min_clearance: float,
    exclude_geom=None,
) -> tuple[list[dict], list[dict]]:
    """Sample grid nodes on a level's walkable polygon.

    Uses **vectorised** shapely 2.0 operations to avoid per-point
    Python overhead:
    * ``np.meshgrid`` for grid generation
    * ``shapely.contains_xy`` for bulk containment
    * ``shapely.distance`` for bulk clearance
    """
    polygons = flatten_polygons(floor_geom)
    if not polygons:
        return [], []

    minx, miny, maxx, maxy = floor_geom.bounds

    # 1. Grid as numpy arrays  (replaces iter_grid_points loop)
    nx = int(np.floor((maxx - minx) / grid_res)) + 1
    ny = int(np.floor((maxy - miny) / grid_res)) + 1
    xs = minx + np.arange(nx) * grid_res
    ys = miny + np.arange(ny) * grid_res
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()

    # 2. Vectorised containment → boolean mask
    mask = shapely.contains_xy(floor_geom, gx, gy)

    # 3. Exclude connector zones
    if exclude_geom is not None and not exclude_geom.is_empty:
        mask &= ~shapely.contains_xy(exclude_geom, gx, gy)

    # Keep only inside points
    gx = gx[mask]
    gy = gy[mask]
    n = len(gx)

    # 4. Clearance via vectorised distance
    has_obs = obstacle_union is not None and not obstacle_union.is_empty
    if has_obs and n > 0:
        pts = shapely.points(gx, gy)                 # ndarray of Point
        clearances = shapely.distance(pts, obstacle_union)
    else:
        clearances = np.full(n, 999.0)

    usable_mask = clearances >= min_clearance

    # 5. Pre-round for output
    gx_r = np.round(gx, 3)
    gy_r = np.round(gy, 3)
    cl_r = np.round(clearances, 4)
    z = float(elevation_m)

    # 6. Build output dicts
    nodes_all: list[dict] = []
    nodes_valid: list[dict] = []
    for i in range(n):
        node = {
            "id": f"{level_key}_n_{i}",
            "level": level_key,
            "x": float(gx_r[i]),
            "y": float(gy_r[i]),
            "z": z,
            "clearance": float(cl_r[i]),
            "usable": bool(usable_mask[i]),
            "node_type": "floor",
            # Blind-path tags are injected later by _tag_blind_path_nodes().
            "is_blind_path": False,
            "blind_category": "",
            "surface_type": "normal",
        }
        nodes_all.append(node)
        if usable_mask[i]:
            nodes_valid.append(node)

    return nodes_valid, nodes_all


def _tag_blind_path_nodes(
    nodes_all: list[dict],
    blind_paths: list[dict],
    grid_res: float = 0.5,
    blind_path_cfg: dict | None = None,
) -> dict[str, int]:
    """Tag floor nodes that lie on tactile blind-path surfaces.

    Node type remains ``floor`` for compatibility with Step 3 builder.
    Distinction is carried by:
      - ``is_blind_path`` (bool)
      - ``blind_category`` ("guide" | "warning" | "")
      - ``surface_type`` ("blind_guide" | "blind_warning" | "normal")

    To avoid fragmented / noisy tags from tiny IFC tactile blocks, guide
    footprints are expanded and morphologically bridged before sampling.
    Very small isolated guide clusters are removed.
    """
    if not nodes_all or not blind_paths:
        return {"guide": 0, "warning": 0, "total": 0}

    floor_nodes = [n for n in nodes_all if n.get("node_type") == "floor"]
    if not floor_nodes:
        return {"guide": 0, "warning": 0, "total": 0}

    xs = np.array([n["x"] for n in floor_nodes], dtype=np.float64)
    ys = np.array([n["y"] for n in floor_nodes], dtype=np.float64)

    guide_polys = [bp["footprint"] for bp in blind_paths
                   if bp.get("category") == "guide" and bp.get("footprint") is not None]
    warning_polys = [bp["footprint"] for bp in blind_paths
                     if bp.get("category") == "warning" and bp.get("footprint") is not None]

    guide_mask = np.zeros(len(floor_nodes), dtype=bool)
    warning_mask = np.zeros(len(floor_nodes), dtype=bool)

    pts = shapely.points(xs, ys)
    cfg = blind_path_cfg or {}
    guide_expand   = cfg.get("guide_expand_m",   max(0.15, 0.30 * grid_res))
    guide_bridge   = cfg.get("guide_bridge_m",   max(2.0,  4.0  * grid_res))
    warning_expand = cfg.get("warning_expand_m", max(0.10, 0.25 * grid_res))
    capture_tol    = cfg.get("capture_tol_m",    max(0.20, 0.40 * grid_res))

    if guide_polys:
        guide_union = unary_union(guide_polys)
        if guide_union is not None and not guide_union.is_empty:
            guide_geom = guide_union.buffer(guide_expand, join_style=2, cap_style=2)
            # Closing: connect near-adjacent tactile blocks into continuous strips.
            guide_geom = guide_geom.buffer(guide_bridge, join_style=2, cap_style=2)
            guide_geom = guide_geom.buffer(-guide_bridge, join_style=2, cap_style=2)
            if guide_geom is None or guide_geom.is_empty:
                guide_geom = guide_union.buffer(guide_expand, join_style=2, cap_style=2)
            if guide_geom is not None and not guide_geom.is_empty:
                guide_mask = shapely.distance(pts, guide_geom) <= capture_tol

    if warning_polys:
        warning_union = unary_union(warning_polys)
        if warning_union is not None and not warning_union.is_empty:
            warning_geom = warning_union.buffer(warning_expand, join_style=2, cap_style=2)
            if warning_geom is not None and not warning_geom.is_empty:
                warning_mask = shapely.distance(pts, warning_geom) <= capture_tol

    # Remove tiny isolated guide components on the sampling grid.
    guide_only = np.where(guide_mask & ~warning_mask)[0]
    if len(guide_only) > 0:
        min_comp_size = (blind_path_cfg or {}).get("min_component_nodes", 1)
        x0 = float(xs.min())
        y0 = float(ys.min())
        cell_to_idx: dict[tuple[int, int], int] = {}
        for idx in guide_only:
            gx = int(round((float(xs[idx]) - x0) / grid_res))
            gy = int(round((float(ys[idx]) - y0) / grid_res))
            cell_to_idx[(gx, gy)] = int(idx)

        visited: set[int] = set()
        keep_idx: set[int] = set()
        nbs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for idx in guide_only:
            i = int(idx)
            if i in visited:
                continue
            gx = int(round((float(xs[i]) - x0) / grid_res))
            gy = int(round((float(ys[i]) - y0) / grid_res))

            stack = [(gx, gy)]
            comp: list[int] = []
            while stack:
                cx, cy = stack.pop()
                j = cell_to_idx.get((cx, cy))
                if j is None or j in visited:
                    continue
                visited.add(j)
                comp.append(j)
                for dx, dy in nbs:
                    stack.append((cx + dx, cy + dy))

            if len(comp) >= min_comp_size:
                keep_idx.update(comp)

        new_guide_mask = np.zeros_like(guide_mask)
        if keep_idx:
            new_guide_mask[list(keep_idx)] = True
        guide_mask = new_guide_mask

    # Warning has higher priority if overlap happens.
    for i, n in enumerate(floor_nodes):
        if warning_mask[i]:
            n["is_blind_path"] = True
            n["blind_category"] = "warning"
            n["surface_type"] = "blind_warning"
        elif guide_mask[i]:
            n["is_blind_path"] = True
            n["blind_category"] = "guide"
            n["surface_type"] = "blind_guide"

    n_warning = int(np.count_nonzero(warning_mask))
    # guide-only count (exclude warning overlap)
    n_guide = int(np.count_nonzero(guide_mask & ~warning_mask))
    return {
        "guide": n_guide,
        "warning": n_warning,
        "total": n_guide + n_warning,
    }


# ====================================================================
#  Dynamic-door node injection
# ====================================================================

def _generate_door_nodes(
    dynamic_doors: list[dict],
    level_key: str,
    elevation_m: float,
    grid_res: float = 0.5,
    offset: float = 0.35,
) -> list[dict]:
    """Create platform-side and track-side nodes for each dynamic door.

    For platform screen doors the *platform* node sits inside the
    station proper and the *track* node sits on the track side.
    Both are placed ``offset`` metres away from the barrier edge so
    that the graph builder can create a short toggle-edge through
    the barrier.

    For **elevator doors** the approach is different:
    * One *entry* node outside the door face (in the walkable corridor).
    * Multiple *interior* nodes inside the shaft, arranged in a small
      grid matching the elevator's standing capacity.  These nodes
      carry ``node_type="elevator_interior"`` and ``capacity`` metadata
      so that the ABM can enforce boarding limits.

    Parameters
    ----------
    offset : float
        Distance from barrier edge to door node (default 0.35 m ≈
        slightly larger than agent body radius 0.25 m).
    """
    door_nodes: list[dict] = []
    z = float(elevation_m)

    for dd in dynamic_doors:
        dtype = dd.get("type", "")

        # ============================================================
        #  Elevator door → entry node + interior capacity nodes
        # ============================================================
        if dtype == "elevator_door":
            door_nodes.extend(
                _generate_elevator_nodes(dd, level_key, z, offset)
            )
            continue

        # ============================================================
        #  Platform screen door (PSD)
        # ============================================================
        b = dd["bounds"]       # [x1, y1, x2, y2]
        x1, x2 = float(b[0]), float(b[2])
        y1, y2 = b[1], b[3]
        side = dd.get("side", "")

        if side == "south":
            # South barrier: platform is NORTH, track is SOUTH
            platform_y = round(y2 + offset, 3)
            track_y    = round(y1 - offset, 3)
        elif side == "north":
            # North barrier: platform is SOUTH, track is NORTH
            platform_y = round(y1 - offset, 3)
            track_y    = round(y2 + offset, 3)
        else:
            cy = (y1 + y2) / 2
            platform_y = round(cy - offset, 3)
            track_y    = round(cy + offset, 3)

        # Width-aware PSD nodes: sample multiple points along each door segment.
        # Use roughly one sample per floor-grid cell so the door has explicit width.
        width = max(0.05, x2 - x1)
        n_pts = max(1, int(np.floor(width / max(0.1, grid_res))) + 1)
        if n_pts == 1:
            xs = [dd["center_x"]]
        else:
            step = width / (n_pts - 1)
            xs = [x1 + i * step for i in range(n_pts)]

        for j, sx in enumerate(xs):
            door_seg_id = f"{dd['id']}_w{j:02d}"
            door_nodes.append({
                "id": f"door_{door_seg_id}_P",
                "level": level_key,
                "x": round(sx, 3),
                "y": platform_y,
                "z": z,
                "clearance": offset,
                "usable": True,
                "node_type": "door_platform",
                "door_id": door_seg_id,
                "door_group": dd["id"],
                "door_type": dd["type"],
            })
            door_nodes.append({
                "id": f"door_{door_seg_id}_T",
                "level": level_key,
                "x": round(sx, 3),
                "y": track_y,
                "z": z,
                "clearance": offset,
                "usable": True,
                "node_type": "door_track",
                "door_id": door_seg_id,
                "door_group": dd["id"],
                "door_type": dd["type"],
            })

    return door_nodes


def _generate_elevator_nodes(
    dd: dict,
    level_key: str,
    z: float,
    offset: float,
) -> list[dict]:
    """Generate entry + interior nodes for one elevator door on one level.

    Interior layout:
    * Grid spacing derived from capacity and shaft area.
    * Standing density ≈ capacity / shaft_area (typ. 4-7 pax/m²).
    * Grid resolves to ~0.5 m spacing for comfortable packing.
    * Each interior node carries ``capacity`` = per-node share
      (total capacity divided evenly among interior nodes for ABM
      crowd-distribution).

    The entry node sits ``offset`` metres outside the door face,
    in the walkable corridor, so the graph can create a short
    toggle-edge from corridor → shaft interior.
    """
    import math

    nodes: list[dict] = []
    door_id = dd["id"]
    face = dd.get("face", "south")
    sb = dd.get("shaft_bounds", dd.get("bounds", [0, 0, 0, 0]))
    capacity = dd.get("capacity", 20)
    cx = dd["center_x"]

    sx1, sy1, sx2, sy2 = sb
    shaft_w = sx2 - sx1     # width  (x)
    shaft_h = sy2 - sy1     # depth  (y)

    # ---- Entry node (outside door face) ----
    if face == "south":
        entry_x, entry_y = cx, round(sy1 - offset, 3)
    elif face == "north":
        entry_x, entry_y = cx, round(sy2 + offset, 3)
    elif face == "west":
        entry_x, entry_y = round(sx1 - offset, 3), dd.get("center_y", (sy1 + sy2) / 2)
    else:  # east
        entry_x, entry_y = round(sx2 + offset, 3), dd.get("center_y", (sy1 + sy2) / 2)

    nodes.append({
        "id": f"elev_{door_id}_{level_key}_entry",
        "level": level_key,
        "x": round(entry_x, 3),
        "y": round(entry_y, 3),
        "z": z,
        "clearance": offset,
        "usable": True,
        "node_type": "elevator_entry",
        "door_id": door_id,
        "door_type": "elevator_door",
        "capacity": capacity,
    })

    # ---- Interior capacity nodes (grid inside shaft) ----
    # Target grid spacing: ~0.5 m (shoulder width), with wall margin 0.2 m
    margin = 0.2
    inner_w = shaft_w - 2 * margin
    inner_h = shaft_h - 2 * margin
    grid_sp = 0.5

    if inner_w < 0.3 or inner_h < 0.3:
        # Shaft too small for interior grid — single centre node
        nodes.append({
            "id": f"elev_{door_id}_{level_key}_int_0",
            "level": level_key,
            "x": round((sx1 + sx2) / 2, 3),
            "y": round((sy1 + sy2) / 2, 3),
            "z": z,
            "clearance": margin,
            "usable": True,
            "node_type": "elevator_interior",
            "door_id": door_id,
            "door_type": "elevator_door",
            "capacity": capacity,
        })
        return nodes

    nx = max(1, int(inner_w / grid_sp) + 1)
    ny = max(1, int(inner_h / grid_sp) + 1)
    n_interior = nx * ny
    cap_per_node = math.ceil(capacity / n_interior)

    idx = 0
    for ix in range(nx):
        px = sx1 + margin + (inner_w * ix / max(nx - 1, 1) if nx > 1 else inner_w / 2)
        for iy in range(ny):
            py = sy1 + margin + (inner_h * iy / max(ny - 1, 1) if ny > 1 else inner_h / 2)
            nodes.append({
                "id": f"elev_{door_id}_{level_key}_int_{idx}",
                "level": level_key,
                "x": round(px, 3),
                "y": round(py, 3),
                "z": z,
                "clearance": margin,
                "usable": True,
                "node_type": "elevator_interior",
                "door_id": door_id,
                "door_type": "elevator_door",
                "capacity": cap_per_node,
            })
            idx += 1

    return nodes


# ====================================================================
#  Fare gate passage node generator
# ====================================================================

def _generate_fare_gate_nodes(
    fg_passages: list[dict],
    level_key: str,
    elevation_m: float,
) -> list[dict]:
    """Create a single gate node per passage, placed at the barrier centre.

    Node type encodes direction directly:
    - ``fare_gate_entry`` – inbound (进站): unpaid_floor → node → paid_floor
    - ``fare_gate_exit``  – outbound (出站): paid_floor → node → unpaid_floor

    The node stores ``barrier_bounds``, ``gate_axis``, and ``paid_side`` so
    that ``_connect_fare_gate_nodes`` can wire directed edges to the correct
    floor nodes on each side without relying on node position alone.
    """
    nodes: list[dict] = []
    for passage in fg_passages:
        direction = passage.get("direction", "inbound")
        node_type = "fare_gate_entry" if direction == "inbound" else "fare_gate_exit"
        nodes.append({
            "id": passage["id"],
            "level": level_key,
            "x": passage["center_x"],
            "y": passage["center_y"],
            "z": elevation_m,
            "node_type": node_type,
            "passage_id": passage["id"],
            "gate_group": passage["group"],
            "direction": direction,
            "barrier_bounds": passage["barrier_bounds"],
            "gate_axis": passage.get("gate_axis", "y"),
            "paid_side": passage.get("paid_side", "east"),
        })
    return nodes


# ====================================================================
#  Security scanner passage node generator
# ====================================================================

def _generate_scanner_nodes(
    sc_passages: list[dict],
    level_key: str,
    elevation_m: float,
    offset: float = 0.45,
) -> list[dict]:
    """Create paired approach/exit nodes flanking each scanner passage."""
    nodes: list[dict] = []
    for passage in sc_passages:
        b = passage["barrier_bounds"]
        approach_side = passage.get("approach_side", "west")
        scanner_axis = passage.get("scanner_axis", "y")
        cx = passage["center_x"]
        cy = passage["center_y"]

        if scanner_axis == "y":
            if approach_side == "west":
                approach_x = b[0] - offset
                exit_x     = b[2] + offset
            else:
                approach_x = b[2] + offset
                exit_x     = b[0] - offset
            nodes.append({
                "id": f"{passage['id']}_approach",
                "level": level_key, "x": round(approach_x, 3),
                "y": round(cy, 3), "z": elevation_m,
                "node_type": "scanner_approach",
                "passage_id": passage["id"],
                "scanner_group": passage["group"],
            })
            nodes.append({
                "id": f"{passage['id']}_exit",
                "level": level_key, "x": round(exit_x, 3),
                "y": round(cy, 3), "z": elevation_m,
                "node_type": "scanner_exit",
                "passage_id": passage["id"],
                "scanner_group": passage["group"],
            })
        else:
            if approach_side == "south":
                approach_y = b[1] - offset
                exit_y     = b[3] + offset
            else:
                approach_y = b[3] + offset
                exit_y     = b[1] - offset
            nodes.append({
                "id": f"{passage['id']}_approach",
                "level": level_key, "x": round(cx, 3),
                "y": round(approach_y, 3), "z": elevation_m,
                "node_type": "scanner_approach",
                "passage_id": passage["id"],
                "scanner_group": passage["group"],
            })
            nodes.append({
                "id": f"{passage['id']}_exit",
                "level": level_key, "x": round(cx, 3),
                "y": round(exit_y, 3), "z": elevation_m,
                "node_type": "scanner_exit",
                "passage_id": passage["id"],
                "scanner_group": passage["group"],
            })
    return nodes


# ====================================================================
#  Exclusion zone builder
# ====================================================================

def build_exclusion_zone(
    connectors: list[dict],
    control_points: list[dict] | None = None,
    connector_buffer: float = 0.3,
    cp_buffer: float = 0.15,
) -> any:
    """Build exclusion polygon from connector + control-point footprints.

    Floor-grid nodes inside this zone are excluded:
    * **Connectors** (stairs, escalators, elevators) – they'll get
      dedicated chain nodes in Step 3.
    * **Control points** (fare gates, scanners) – small extra buffer
      to keep grid nodes off the device boundary.
    """
    polys = []
    for c in connectors:
        fp = c.get("footprint")
        if fp is not None and not fp.is_empty:
            polys.append(fp.buffer(connector_buffer))

    for cp in (control_points or []):
        fp = cp.get("footprint")
        if fp is not None and not fp.is_empty:
            polys.append(fp.buffer(cp_buffer))

    if polys:
        return unary_union(polys).buffer(0)
    return None


# ====================================================================
#  Connector anchor-node generation  (voxelisation)
# ====================================================================

def voxelize_connectors(
    all_connectors: list[dict],
    all_geometry: dict,
    config: dict,
) -> dict[str, list[dict]]:
    """Generate anchor nodes at each connector's entry/exit on each level.

    Node placement strategy by connector type:

    * **stair_chain** – uses the chain's ``level_anchors`` dict that
      stores the precise entry/exit centroid per connected walkable level
      (computed during chain grouping in Step 1).

    * **escalator** – centroid of the footprint on each served level.

    * **elevator** – **skipped** here.  Elevator doors are handled as
      dynamic doors (similar to PSD) via ``_generate_door_nodes``.

    Returns
    -------
    dict[str, list[dict]]
        Keyed by level_key, each value is a list of connector-node dicts.
    """
    from shapely.geometry import Point

    conn_nodes: dict[str, list[dict]] = {}
    node_id_counter = 0
    seen_anchors: set[tuple] = set()   # (connector_id, level, round_x, round_y)

    for c in all_connectors:
        ctype = c["type"]

        # Elevator → handled as dynamic door, skip here
        if ctype == "elevator":
            continue

        fp = c.get("footprint")
        if fp is None or fp.is_empty:
            continue

        # Determine served levels
        if ctype == "stair_chain":
            served = c.get("connected_levels", [])
        else:
            served = [lk for lk in [c.get("bottom_level"), c.get("top_level")]
                      if lk is not None]

        for lk in served:
            geom = all_geometry.get(lk)
            if geom is None or geom.get("walkable") is None:
                continue

            z = geom["elevation_m"]

            # --- Determine anchor x, y ---
            # Anchors are placed just **outside** the connector footprint
            # so they land in walkable floor area (not inside the
            # exclusion zone).  Step interpolation still uses the
            # original IFC / run positions stored in the connector dict.
            _ANCHOR_OFFSET = 0.5  # metres beyond footprint boundary

            if ctype == "stair_chain":
                # Already pushed to footprint edge in geometry_extractor
                anchor = c.get("level_anchors", {}).get(lk)
                if anchor is None:
                    continue
                cx, cy = anchor["x"], anchor["y"]
            elif ctype == "escalator":
                # Use IFC vertex-derived physical landing positions,
                # then project to just outside the footprint boundary.
                bl = c.get("bottom_level")
                if lk == bl:
                    orig_x, orig_y = c["bottom_xy"]
                else:
                    orig_x, orig_y = c["top_xy"]
                # Push to outward edge of footprint
                fx0, fy0, fx1, fy1 = fp.bounds
                fp_cx = (fx0 + fx1) / 2
                if orig_x >= fp_cx:
                    cx = fx1 + _ANCHOR_OFFSET
                else:
                    cx = fx0 - _ANCHOR_OFFSET
                cy = orig_y
            else:
                # generic connector: centroid of footprint
                cx, cy = fp.centroid.x, fp.centroid.y

            # Deduplicate: same connector on same level at same position
            dedup_key = (c["id"], lk, round(cx, 1), round(cy, 1))
            if dedup_key in seen_anchors:
                continue
            seen_anchors.add(dedup_key)

            node = {
                "id": f"conn_{node_id_counter}",
                "level": lk,
                "x": round(cx, 3),
                "y": round(cy, 3),
                "z": z,
                "node_type": ctype,
                "connector_id": c["id"],
                "connector_name": c.get("name", ""),
            }
            node_id_counter += 1
            conn_nodes.setdefault(lk, []).append(node)

    for lk, nodes in conn_nodes.items():
        n_stair = sum(1 for n in nodes if n["node_type"] == "stair_chain")
        n_esc = sum(1 for n in nodes if n["node_type"] == "escalator")
        print(f"    {lk}: {len(nodes)} connector nodes "
              f"(ST:{n_stair} ES:{n_esc})")

    return conn_nodes


# ====================================================================
#  Multi-level orchestrator
# ====================================================================

def sample_all_levels(
    all_geometry: dict,
    config: dict,
) -> dict[str, dict]:
    """Sample nodes for all walkable levels.

    Returns dict keyed by level_key with:
        - nodes_valid: list[dict]
        - nodes_all: list[dict]
        - n_valid: int
        - n_total: int
    """
    sampling_cfg = config["sampling"]
    grid_res = sampling_cfg["grid_resolution_m"]
    min_clearance = sampling_cfg["min_clearance_m"]
    exclude_stairs = sampling_cfg.get("exclude_stair_footprints", True)
    conn_buffer = sampling_cfg.get("exclude_connector_buffer_m", 0.3)
    cp_buffer = sampling_cfg.get("control_point_buffer_m", 0.15)

    all_nodes: dict[str, dict] = {}

    for level_key, geom in all_geometry.items():
        if geom["walkable"] is None:
            continue  # Skip non-walkable levels (F2)

        print(f"  Sampling {level_key} ...")

        # Build exclusion zone from connectors + control points
        exclude = None
        if exclude_stairs and geom.get("connectors"):
            exclude = build_exclusion_zone(
                geom["connectors"],
                geom.get("control_points"),
                connector_buffer=conn_buffer,
                cp_buffer=cp_buffer,
            )
            # Subtract walkable_passage areas from exclusion zone.
            # Some connectors have oversized footprints (e.g. full escalator
            # span) that overlap legitimate walkable corridors.  Manual
            # walkable_passages already remove obstacle obstructions; we
            # also need to let grid nodes appear there by carving the same
            # region out of the connector exclusion zone.
            overrides = (
                config["station"].get("manual_overrides", {})
                .get(level_key, {})
            )
            passages = overrides.get("walkable_passages", [])
            if passages and exclude is not None and not exclude.is_empty:
                from shapely.geometry import box as _box
                from shapely.ops import unary_union as _uu
                exempt_polys = [_box(*p["bounds"]) for p in passages]
                if exempt_polys:
                    exempt = _uu(exempt_polys)
                    exclude = exclude.difference(exempt)
                    if exclude.is_empty:
                        exclude = None

        nodes_valid, nodes_all = sample_level_nodes(
            level_key=level_key,
            elevation_m=geom["elevation_m"],
            floor_geom=geom["walkable"],
            obstacle_union=geom["obstacle_union"],
            grid_res=grid_res,
            min_clearance=min_clearance,
            exclude_geom=exclude,
        )

        # --- Tag blind-path nodes (tactile paving) on floor grid ---
        blind_paths = geom.get("blind_paths", [])
        if blind_paths:
            bp_stats = _tag_blind_path_nodes(
                nodes_all,
                blind_paths,
                grid_res=grid_res,
                blind_path_cfg=config.get("blind_path"),
            )
            if bp_stats["total"] > 0:
                print("    + blind-path floor nodes "
                      f"(guide:{bp_stats['guide']}, warning:{bp_stats['warning']}, "
                      f"total:{bp_stats['total']})")

        # --- Inject dynamic-door nodes ---
        dynamic_doors = geom.get("dynamic_doors", [])
        if dynamic_doors:
            door_nodes = _generate_door_nodes(
                dynamic_doors, level_key, geom["elevation_m"],
                grid_res=grid_res,
            )
            nodes_valid.extend(door_nodes)
            nodes_all.extend(door_nodes)
            n_psd = sum(1 for d in door_nodes
                        if d.get("door_type") == "platform_screen_door")
            n_elv_entry = sum(1 for d in door_nodes
                              if d.get("node_type") == "elevator_entry")
            n_elv_int = sum(1 for d in door_nodes
                            if d.get("node_type") == "elevator_interior")
            parts = [f"PSD:{n_psd}"]
            if n_elv_entry:
                parts.append(f"ELV_entry:{n_elv_entry}")
            if n_elv_int:
                parts.append(f"ELV_int:{n_elv_int}")
            print(f"    + {len(door_nodes)} door nodes "
                  f"({', '.join(parts)})")

        # --- Inject fare gate passage nodes ---
        fg_passages = geom.get("fare_gate_passages", [])
        if fg_passages:
            fg_nodes = _generate_fare_gate_nodes(
                fg_passages, level_key, geom["elevation_m"])
            nodes_valid.extend(fg_nodes)
            nodes_all.extend(fg_nodes)
            print(f"    + {len(fg_nodes)} fare gate passage nodes "
                  f"({len(fg_passages)} passages)")

        all_nodes[level_key] = {
            "nodes_valid": nodes_valid,
            "nodes_all": nodes_all,
            "n_valid": len(nodes_valid),
            "n_total": len(nodes_all),
        }
        print(f"    {level_key}: {len(nodes_valid):,} usable / "
              f"{len(nodes_all):,} total nodes")

    return all_nodes


# ====================================================================
#  Persistence
# ====================================================================

def save_sampling_outputs(all_nodes: dict, out_dir: str | Path) -> None:
    """Save sampling outputs to CSV and GeoJSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for level_key, data in all_nodes.items():
        csv_path = out_dir / f"nodes_{level_key}.csv"
        pd.DataFrame(data["nodes_valid"]).to_csv(csv_path, index=False)

        feats = [point_feature(n["x"], n["y"], n) for n in data["nodes_all"]]
        write_geojson(out_dir / f"nodes_{level_key}_all.geojson", feats)

        feats_valid = [point_feature(n["x"], n["y"], n)
                       for n in data["nodes_valid"]]
        write_geojson(out_dir / f"nodes_{level_key}.geojson", feats_valid)

        summary[level_key] = {
            "n_valid": data["n_valid"],
            "n_total": data["n_total"],
            "ratio": round(data["n_valid"] / data["n_total"], 4)
                     if data["n_total"] > 0 else 0,
        }

    dump_json(out_dir / "sampling_summary.json", summary)
    total_valid = sum(d["n_valid"] for d in all_nodes.values())
    print(f"[Step 2] Sampling complete: {total_valid:,} total usable nodes")
