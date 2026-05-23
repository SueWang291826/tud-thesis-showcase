"""
Step 3 · Graph Builder  (v3 — KD-tree + ABM-ready)
====================================================

Changes from v2
~~~~~~~~~~~~~~~~
* **KD-tree spatial index** (``scipy.spatial.cKDTree``) replaces the
  hand-rolled grid-cell hash for floor-graph neighbour queries *and*
  anchor-to-floor snapping.   Build time drops from O(N·C_cell) to
  O(N·log N) and the constant factor is much smaller thanks to
  contiguous NumPy arrays.

* **ABM-ready edge metadata** — every PSD door, elevator door, and
  elevator transport edge now carries ``state``, ``open_duration_s``,
  ``close_duration_s``, ``queue_capacity`` so the Agent-Based Model
  can toggle edge passability at runtime without re-building the graph.

* **F2 excluded** — the equipment level has ``is_walkable: false``
  in config; the builder never creates floor nodes or edges there.
  Elevator transport edges cross directly from F1 → F3 (which happens
  to pass through F2 in the physical station but there is no walkable
  surface to model).

Input
-----
*  ``all_geometry``   – per-level Shapely geometry    (Step 1)
*  ``all_nodes``      – per-level node lists          (Step 2)
*  ``all_connectors`` – typed connector list           (Step 1)

Output
------
A single ``nx.Graph`` with typed nodes and edges for multi-level
pathfinding (floor, stair, escalator, elevator, door).

Node attributes:
  x, y, z, level, node_type, connector_id, door_id, capacity

Edge attributes:
  length_2d, length_3d, travel_time, edge_type, connector_id,
  capacity, direction, toggleable, state, open_duration_s,
  close_duration_s, queue_capacity
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from src.utils import (
    euclidean_2d, euclidean_3d,
    dump_json, write_geojson, line_feature, point_feature,
)


# ====================================================================
#  A. Floor graph (intra-level) — KD-tree accelerated
# ====================================================================

def build_floor_graph(
    g: nx.Graph,
    nodes: list[dict],
    obstacle_union,
    grid_res: float,
    connectivity: int = 8,
    los_check: bool = True,
    walking_speed: float = 1.2,
) -> int:
    """Add nodes and 8/4-connected grid edges for one level.

    Uses ``scipy.spatial.cKDTree`` for O(N log N) neighbour search
    instead of a hand-rolled grid-cell hash.

    Returns number of edges added.
    """
    if not nodes:
        return 0

    for n in nodes:
        g.add_node(n["id"], **n)

    # Build KD-tree over (x, y) coordinates
    ids = [n["id"] for n in nodes]
    coords = np.array([(n["x"], n["y"]) for n in nodes], dtype=np.float64)
    tree = cKDTree(coords)

    max_dist = (grid_res * math.sqrt(2) * 1.05
                if connectivity == 8 else grid_res * 1.05)

    # Query all pairs within max_dist (returns sparse distance matrix)
    pairs = tree.query_pairs(r=max_dist, output_type="ndarray")

    node_by_id = {n["id"]: n for n in nodes}
    n_edges = 0

    for i, j in pairs:
        nid_a, nid_b = ids[i], ids[j]
        na, nb = node_by_id[nid_a], node_by_id[nid_b]

        dx = abs(na["x"] - nb["x"])
        dy = abs(na["y"] - nb["y"])

        # 4-connectivity: skip diagonals
        if connectivity == 4 and dx > 0.01 and dy > 0.01:
            continue

        if (los_check and obstacle_union is not None
                and not obstacle_union.is_empty):
            line = LineString([(na["x"], na["y"]), (nb["x"], nb["y"])])
            if line.intersects(obstacle_union):
                continue

        d2d = math.hypot(dx, dy)
        d3d = euclidean_3d(
            (na["x"], na["y"], na["z"]),
            (nb["x"], nb["y"], nb["z"]),
        )
        for _src, _dst in ((nid_a, nid_b), (nid_b, nid_a)):
            g.add_edge(
                _src, _dst,
                length_2d=d2d, length_3d=d3d,
                travel_time=d3d / walking_speed,
                edge_type="floor",
                level=na["level"],
            )
        n_edges += 2

    return n_edges


# ====================================================================
#  B. Snap isolated special nodes to floor — KD-tree
# ====================================================================

def _snap_isolated_nodes_to_floor(
    g: nx.Graph,
    config: dict,
) -> int:
    """Connect isolated non-floor nodes to nearest floor node via KD-tree.

    After the grid-edge phase, connector-anchor and door nodes that
    sit inside exclusion zones (no grid neighbours within max_dist)
    are stranded.  This function adds a single 'anchor_snap' edge
    from each such node to the closest floor node on the same level.

    Returns number of snap edges added.
    """
    ws = config["simulation"]["walking_speed_ms"]

    # Pre-index floor nodes by level with KD-tree
    floor_by_level: dict[str, tuple[list[str], np.ndarray]] = {}
    for lk in {attr["level"] for _, attr in g.nodes(data=True)}:
        fids, fcoords = [], []
        for nid, attr in g.nodes(data=True):
            if attr.get("node_type") == "floor" and attr["level"] == lk:
                fids.append(nid)
                fcoords.append((attr["x"], attr["y"]))
        if fids:
            floor_by_level[lk] = (fids, cKDTree(np.array(fcoords)))

    snap_types = {
        "stair_chain", "escalator",
        "elevator_entry",
        "door_platform", "door_track",
    }

    n_snapped = 0
    for nid, attr in list(g.nodes(data=True)):
        nt = attr.get("node_type", "")
        lk = attr.get("level", "")
        if nt not in snap_types or lk not in floor_by_level:
            continue

        # Already has at least one floor neighbour → skip
        if any(g.nodes[nb].get("node_type") == "floor"
               for nb in g.neighbors(nid)):
            continue

        ax, ay, az = attr["x"], attr["y"], attr["z"]
        fids, ftree = floor_by_level[lk]
        dist, idx = ftree.query([ax, ay], k=1)

        best_fnid = fids[idx]
        fn = g.nodes[best_fnid]
        d3d = euclidean_3d((ax, ay, az), (fn["x"], fn["y"], fn["z"]))
        cid = attr.get("connector_id", attr.get("door_id", ""))
        for _src, _dst in ((nid, best_fnid), (best_fnid, nid)):
            g.add_edge(
                _src, _dst,
                length_2d=dist, length_3d=d3d,
                travel_time=d3d / ws,
                edge_type="anchor_snap",
                connector_id=cid,
            )
        n_snapped += 1

    return n_snapped


# ====================================================================
#  C. Node indexes (built once after floor-graph phase)
# ====================================================================

def _build_connector_anchor_index(g: nx.Graph) -> dict[tuple, str]:
    """(connector_id, level) → node_id  for stair/escalator anchors."""
    idx: dict[tuple, str] = {}
    for nid, attr in g.nodes(data=True):
        if attr.get("node_type") in ("stair_chain", "escalator"):
            idx[(attr.get("connector_id"), attr.get("level"))] = nid
    return idx


def _build_elevator_node_index(g: nx.Graph) -> dict[tuple, list[str]]:
    """(door_id, level) → [node_ids]  for elevator entry+interior."""
    idx: dict[tuple, list[str]] = defaultdict(list)
    for nid, attr in g.nodes(data=True):
        if attr.get("node_type") in ("elevator_entry", "elevator_interior"):
            idx[(attr.get("door_id"), attr.get("level"))].append(nid)
    return idx


def _build_door_pair_index(g: nx.Graph) -> dict[str, dict[str, str]]:
    """door_id → {"platform": nid, "track": nid}  for PSD doors."""
    idx: dict[str, dict[str, str]] = defaultdict(dict)
    for nid, attr in g.nodes(data=True):
        nt = attr.get("node_type", "")
        if nt == "door_platform":
            idx[attr["door_id"]]["platform"] = nid
        elif nt == "door_track":
            idx[attr["door_id"]]["track"] = nid
    return idx


# ====================================================================
#  C. Stair-chain construction
# ====================================================================

def _build_stair_waypoints(
    bot_anchor: dict,
    top_anchor: dict,
    bot_z: float,
    top_z: float,
    runs: list[dict],
    landings: list[dict],
) -> list[tuple[float, float, float]]:
    """3D waypoint list from bottom anchor → runs/landings → top anchor."""
    pts: list[tuple[float, float, float]] = [
        (bot_anchor["x"], bot_anchor["y"], bot_z),
    ]

    for i, run in enumerate(runs):
        rx = (run["min_x"] + run["max_x"]) / 2
        ry = (run["min_y"] + run["max_y"]) / 2
        pts.append((rx, ry, run["z_min"]))
        pts.append((rx, ry, run["z_max"]))

        for landing in landings:
            if landing.get("between_runs") == (i, i + 1):
                pts.append((landing["x"], landing["y"], landing["z"]))
                break

    pts.append((top_anchor["x"], top_anchor["y"], top_z))

    # Deduplicate consecutive near-identical points
    clean = [pts[0]]
    for p in pts[1:]:
        if euclidean_3d(clean[-1], p) > 0.01:
            clean.append(p)
    return clean


def _interpolate_chain_nodes(
    waypoints: list[tuple],
    chain_id: str,
    dz_step: float,
    pair_label: str,
    runs: list[dict] | None = None,
    step_capacity: int = 2,
) -> list[dict]:
    """Per-run step-platform nodes at *dz_step* intervals.

    Each node represents a single **step tread** (horizontal platform)
    at a specific elevation inside the stair.  Nodes carry the platform
    bounding box (``platform_bbox``) for ABM occupancy checks and for
    the 3-D visualisation.

    Parameters
    ----------
    runs : list[dict], optional
        The run dicts with ``min_x, max_x, min_y, max_y, z_min, z_max``.
        When provided, step nodes are generated *per run* and receive
        accurate (x, y) coordinates interpolated along the run's
        horizontal extent.  Omit to fall back to waypoint interpolation.
    step_capacity : int
        Max agents that can occupy one step simultaneously (ABM).
    """
    # ------------------------------------------------------------------
    # Per-run decomposition (preferred when run geometry is available)
    # ------------------------------------------------------------------
    if runs:
        nodes: list[dict] = []
        # ascent direction
        first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
        last_xc  = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
        asc_x = first_xc < last_xc          # x increases with z

        global_step = 0
        for ri, run in enumerate(runs):
            x0, x1 = run["min_x"], run["max_x"]
            y0, y1 = run["min_y"], run["max_y"]
            zlo, zhi = run["z_min"], run["z_max"]
            dz = zhi - zlo
            n_steps = max(1, int(round(dz / dz_step)))
            run_len_x = x1 - x0
            ycen = (y0 + y1) / 2

            for k in range(n_steps):
                frac = (k + 0.5) / n_steps   # centre of the k-th step
                z_k = zlo + frac * dz
                # x interpolated along the run's horizontal span
                if asc_x:
                    x_k = x0 + frac * run_len_x
                else:
                    x_k = x1 - frac * run_len_x
                # step tread depth in x direction
                tread_dx = run_len_x / n_steps
                nodes.append({
                    "id": f"{chain_id}_{pair_label}_r{ri}_s{k}",
                    "x": round(x_k, 4),
                    "y": round(ycen, 4),
                    "z": round(z_k, 4),
                    "node_type": "stair_step",
                    "level": "STAIR",
                    "connector_id": chain_id,
                    "connector_type": "stair",
                    "run_index": ri,
                    "step_index": global_step,
                    "step_capacity": step_capacity,
                    "platform_bbox": {
                        "x0": round(x_k - tread_dx / 2, 4),
                        "x1": round(x_k + tread_dx / 2, 4),
                        "y0": round(y0, 4),
                        "y1": round(y1, 4),
                        "z":  round(z_k, 4),
                    },
                })
                global_step += 1
        return nodes

    # ------------------------------------------------------------------
    # Fallback: uniform waypoint interpolation (no run geometry)
    # ------------------------------------------------------------------
    z_range = abs(waypoints[-1][2] - waypoints[0][2])
    n_steps = max(1, int(math.ceil(z_range / dz_step)))

    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(cum[-1] + euclidean_3d(waypoints[i - 1], waypoints[i]))
    total = cum[-1]
    if total < 0.01:
        return []

    nodes = []
    for k in range(1, n_steps):
        t = k / n_steps
        target = t * total
        x, y, z = waypoints[-1]
        for j in range(1, len(cum)):
            if cum[j] >= target - 1e-9:
                denom = cum[j] - cum[j - 1]
                frac = (target - cum[j - 1]) / denom if denom > 0 else 0
                x = waypoints[j - 1][0] + (waypoints[j][0] - waypoints[j - 1][0]) * frac
                y = waypoints[j - 1][1] + (waypoints[j][1] - waypoints[j - 1][1]) * frac
                z = waypoints[j - 1][2] + (waypoints[j][2] - waypoints[j - 1][2]) * frac
                break
        nodes.append({
            "id": f"{chain_id}_{pair_label}_{k}",
            "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
            "node_type": "stair_step",
            "level": "STAIR",
            "connector_id": chain_id,
            "connector_type": "stair",
            "step_index": k - 1,
            "step_capacity": step_capacity,
        })
    return nodes


def add_stair_chains(
    g: nx.Graph,
    stair_chains: list[dict],
    anchor_index: dict[tuple, str],
    levels_cfg: dict,
    config: dict,
) -> int:
    """Create intermediate chain nodes and stair edges.

    For each consecutive pair of connected levels, builds a chain from
    bottom anchor → run waypoints → top anchor.

    Returns total number of stair edges added.
    """
    stair_cfg = config["connectors"]["stair"]
    dz_step = stair_cfg["dz_step_m"]
    speed_factor = stair_cfg["speed_factor"]
    ws = config["simulation"]["walking_speed_ms"]
    capacity = stair_cfg["capacity"]
    step_cap = stair_cfg.get("step_capacity", 2)

    n_edges = 0

    for chain in stair_chains:
        cid = chain["id"]
        connected = chain["connected_levels"]
        anchors = chain.get("level_anchors", {})
        runs = chain.get("runs", [])
        landings = chain.get("landings", [])

        if len(connected) < 2:
            continue

        # Sort connected levels by elevation
        lp = sorted(
            [(lk, levels_cfg[lk]["elevation_m"]) for lk in connected],
            key=lambda x: x[1],
        )

        for idx in range(len(lp) - 1):
            bot_lk, bot_z = lp[idx]
            top_lk, top_z = lp[idx + 1]

            bot_nid = anchor_index.get((cid, bot_lk))
            top_nid = anchor_index.get((cid, top_lk))
            if bot_nid is None or top_nid is None:
                print(f"  WARN: missing anchor for {cid} "
                      f"({bot_lk}={bot_nid}, {top_lk}={top_nid})")
                continue

            # Level anchor coordinates
            ba = anchors.get(
                bot_lk,
                {"x": g.nodes[bot_nid]["x"], "y": g.nodes[bot_nid]["y"]},
            )
            ta = anchors.get(
                top_lk,
                {"x": g.nodes[top_nid]["x"], "y": g.nodes[top_nid]["y"]},
            )

            # Filter runs that fall within this level pair (±1.5 m tolerance)
            pair_runs = [
                r for r in runs
                if r["z_min"] >= bot_z - 1.5 and r["z_max"] <= top_z + 1.5
            ]
            if not pair_runs:
                pair_runs = runs
            pair_runs.sort(key=lambda r: r["z_min"])

            waypoints = _build_stair_waypoints(
                ba, ta, bot_z, top_z, pair_runs, landings,
            )
            chain_nodes = _interpolate_chain_nodes(
                waypoints, cid, dz_step, f"{bot_lk}_{top_lk}",
                runs=pair_runs, step_capacity=step_cap,
            )
            for cn in chain_nodes:
                g.add_node(cn["id"], **cn)

            # Full chain: anchor_bot → intermediates → anchor_top
            full = ([bot_nid]
                    + [cn["id"] for cn in chain_nodes]
                    + [top_nid])

            for a, b in zip(full[:-1], full[1:]):
                na = g.nodes[a]
                nb = g.nodes[b]
                d3d = euclidean_3d(
                    (na["x"], na["y"], na["z"]),
                    (nb["x"], nb["y"], nb["z"]),
                )
                d2d = euclidean_2d(
                    (na["x"], na["y"]), (nb["x"], nb["y"]),
                )
                for _src, _dst in ((a, b), (b, a)):
                    g.add_edge(
                        _src, _dst,
                        length_2d=d2d, length_3d=d3d,
                        travel_time=d3d / (ws * speed_factor),
                        edge_type="stair",
                        connector_id=cid,
                        capacity=capacity,
                        step_capacity=step_cap,
                    )
                    n_edges += 1

    return n_edges


# ====================================================================
#  D. Escalator links
# ====================================================================

def add_escalator_links(
    g: nx.Graph,
    escalators: list[dict],
    anchor_index: dict[tuple, str],
    levels_cfg: dict,
    config: dict,
) -> int:
    """Add escalator step-platform nodes and edges between anchor nodes.

    Each escalator is decomposed into *N* horizontal step nodes (same
    scheme as stairs) so the ABM can model per-step occupancy and the
    visualisation shows the true 3-D step cascade.

    Returns number of escalator edges added.
    """
    esc_cfg = config["connectors"]["escalator"]
    belt_speed = esc_cfg["speed_ms"]
    cap = esc_cfg["capacity"]
    dz_step = esc_cfg.get("dz_step_m", 0.40)  # escalator step ~0.4 m
    step_cap = esc_cfg.get("step_capacity", 2)

    n_edges = 0

    for esc in escalators:
        eid = esc["id"]
        bl = esc.get("bottom_level")
        tl = esc.get("top_level")
        direction = esc.get("direction", "up")

        bn = anchor_index.get((eid, bl))
        tn = anchor_index.get((eid, tl))
        if bn is None or tn is None:
            print(f"  WARN: missing anchor for {eid} "
                  f"({bl}={bn}, {tl}={tn})")
            continue

        na_d = g.nodes[bn]
        nb_d = g.nodes[tn]

        # Use level elevations for z (anchor z may be identical)
        z0 = levels_cfg.get(bl, {}).get("elevation_m", na_d["z"])
        z1 = levels_cfg.get(tl, {}).get("elevation_m", nb_d["z"])

        # ---- Generate intermediate step nodes ----
        # Use the original IFC vertex-derived physical landing positions
        # for step interpolation.  Anchor *nodes* may have been offset
        # to outside the footprint for better floor-snap, but steps
        # must still be positioned inside the physical escalator.
        x_bot = esc.get("bottom_xy", [na_d["x"]])[0]
        x_top = esc.get("top_xy", [nb_d["x"]])[0]
        y0 = esc.get("min_y", na_d["y"])
        y1 = esc.get("max_y", nb_d["y"])
        ycen = (y0 + y1) / 2

        dz = abs(z1 - z0)
        n_steps = max(1, int(round(dz / dz_step)))
        run_len_x = x_top - x_bot  # signed: positive if ascending in x

        step_nodes: list[dict] = []
        for k in range(n_steps):
            frac = (k + 0.5) / n_steps
            z_k = z0 + frac * (z1 - z0)
            # Linear interpolation from IFC bottom to top landing
            x_k = x_bot + frac * run_len_x
            tread_dx = abs(run_len_x) / n_steps
            nid = f"{eid}_step_{k}"
            node = {
                "id": nid,
                "x": round(x_k, 4),
                "y": round(ycen, 4),
                "z": round(z_k, 4),
                "node_type": "escalator_step",
                "level": "ESCALATOR",
                "connector_id": eid,
                "connector_type": "escalator",
                "run_index": 0,
                "step_index": k,
                "step_capacity": step_cap,
                "platform_bbox": {
                    "x0": round(x_k - tread_dx / 2, 4),
                    "x1": round(x_k + tread_dx / 2, 4),
                    "y0": round(y0, 4),
                    "y1": round(y1, 4),
                    "z":  round(z_k, 4),
                },
            }
            g.add_node(nid, **node)
            step_nodes.append(node)

        # ---- Chain: anchor_bot → step nodes → anchor_top ----
        full = ([bn] + [sn["id"] for sn in step_nodes] + [tn])

        for a, b in zip(full[:-1], full[1:]):
            na = g.nodes[a]
            nb = g.nodes[b]
            az = na.get("z", z0)
            bz = nb.get("z", z1)
            d3d = euclidean_3d((na["x"], na["y"], az),
                               (nb["x"], nb["y"], bz))
            d2d = euclidean_2d((na["x"], na["y"]),
                               (nb["x"], nb["y"]))
            tt = d3d / belt_speed if belt_speed > 0 else d3d
            for _src, _dst in ((a, b), (b, a)):
                g.add_edge(
                    _src, _dst,
                    length_2d=d2d, length_3d=d3d,
                    travel_time=tt,
                    edge_type="escalator",
                    connector_id=eid,
                    direction=direction,
                    capacity=cap,
                    step_capacity=step_cap,
                )
                n_edges += 1

    return n_edges


# ====================================================================
#  E. Elevator transport
# ====================================================================

def add_elevator_edges(
    g: nx.Graph,
    elev_index: dict[tuple, list[str]],
    levels_cfg: dict,
    config: dict,
) -> int:
    """Create elevator door, interior-mesh, and cross-level transport edges.

    Edge categories created (all ABM-ready):

    * ``elevator_door``     – entry ↔ interior  (same level, toggleable)
      Carries ``state="closed"`` by default.  The ABM opens the door
      when the car arrives and ``dwell_time_s`` has not yet elapsed,
      then closes it again.

    * ``elevator_interior`` – interior ↔ interior  (same level, shuffle)
      Always traversable — agents redistribute inside the car.

    * ``elevator``          – interior(L_a) ↔ interior(L_b)  (transport)
      Carries ``travel_time = dwell + travel``.  ``state`` starts as
      ``"idle"``: the ABM moves it to ``"boarding"`` / ``"travelling"``
      / ``"alighting"`` as the car cycles.

    Returns total number of elevator edges added.
    """
    elev_cfg = config["connectors"]["elevator"]
    dwell = elev_cfg["dwell_time_s"]
    travel = elev_cfg["travel_time_s"]
    ws = config["simulation"]["walking_speed_ms"]
    cap = elev_cfg["capacity_batch"]

    # Group by door_id → {level → {entry: [...], interior: [...]}}
    by_door: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"entry": [], "interior": []}),
    )
    for (did, lk), nids in elev_index.items():
        for nid in nids:
            nt = g.nodes[nid].get("node_type")
            if nt == "elevator_entry":
                by_door[did][lk]["entry"].append(nid)
            elif nt == "elevator_interior":
                by_door[did][lk]["interior"].append(nid)

    n_edges = 0

    for did, per_level in by_door.items():

        # --- a) Door edges: entry ↔ each interior (same level) ---
        for lk, grp in per_level.items():
            for entry_nid in grp["entry"]:
                en = g.nodes[entry_nid]
                for int_nid in grp["interior"]:
                    inn = g.nodes[int_nid]
                    d3d = euclidean_3d(
                        (en["x"], en["y"], en["z"]),
                        (inn["x"], inn["y"], inn["z"]),
                    )
                    d2d = euclidean_2d(
                        (en["x"], en["y"]), (inn["x"], inn["y"]))
                    for _s, _d in ((entry_nid, int_nid), (int_nid, entry_nid)):
                        g.add_edge(
                            _s, _d,
                            length_2d=d2d, length_3d=d3d,
                            travel_time=d3d / ws,
                            edge_type="elevator_door",
                            door_id=did,
                            toggleable=True,
                            state="closed",
                            dwell_time_s=dwell,
                            capacity=cap,
                            queue_capacity=cap,
                        )
                        n_edges += 1

            # --- b) Interior mesh (same level) ---
            ints = grp["interior"]
            for i in range(len(ints)):
                for j in range(i + 1, len(ints)):
                    ni = g.nodes[ints[i]]
                    nj = g.nodes[ints[j]]
                    d3d = euclidean_3d(
                        (ni["x"], ni["y"], ni["z"]),
                        (nj["x"], nj["y"], nj["z"]),
                    )
                    d2d = euclidean_2d(
                        (ni["x"], ni["y"]), (nj["x"], nj["y"]))
                    for _s, _d in ((ints[i], ints[j]), (ints[j], ints[i])):
                        g.add_edge(
                            _s, _d,
                            length_2d=d2d, length_3d=d3d,
                            travel_time=d3d / ws,
                            edge_type="elevator_interior",
                            door_id=did,
                        )
                        n_edges += 1

        # --- c) Cross-level transport edges ---
        lvl_list = sorted(
            per_level.keys(),
            key=lambda lk: levels_cfg[lk]["elevation_m"],
        )
        for li in range(len(lvl_list)):
            for lj in range(li + 1, len(lvl_list)):
                lk_a, lk_b = lvl_list[li], lvl_list[lj]
                z_diff = abs(
                    levels_cfg[lk_a]["elevation_m"]
                    - levels_cfg[lk_b]["elevation_m"]
                )
                tt = dwell + travel
                for a_nid in per_level[lk_a]["interior"]:
                    for b_nid in per_level[lk_b]["interior"]:
                        for _s, _d in ((a_nid, b_nid), (b_nid, a_nid)):
                            g.add_edge(
                                _s, _d,
                                length_2d=0.0, length_3d=z_diff,
                                travel_time=tt,
                                edge_type="elevator",
                                door_id=did,
                                capacity=cap,
                                state="idle",
                                dwell_time_s=dwell,
                                travel_time_s=travel,
                            )
                            n_edges += 1

    return n_edges


# ====================================================================
#  F. PSD door toggle-edges  (ABM-ready)
# ====================================================================

# ====================================================================
#  Fare gate passage edges
# ====================================================================

def _connect_fare_gate_nodes(
    g: nx.DiGraph,
    config: dict,
) -> int:
    """Wire each single-node gate to the nearest floor on each side.

    Gate nodes (``fare_gate_entry`` / ``fare_gate_exit``) sit inside the
    barrier wall polygon and are unreachable via floor-graph edges
    (LOS-blocked).  This function adds two **directed** edges per gate::

        fare_gate_entry : floor_unpaid → gate → floor_paid
        fare_gate_exit  : floor_paid   → gate → floor_unpaid

    The correct floor side is selected by filtering floor nodes that lie
    fully outside the wall on the appropriate side, then choosing the
    closest by 2-D Euclidean distance.

    Returns total number of edges added (2 per gate node).
    """
    ws = config["simulation"]["walking_speed_ms"]

    # Build per-level floor-node list once
    floor_by_level: dict[str, list[tuple]] = defaultdict(list)
    for nid, attr in g.nodes(data=True):
        if attr.get("node_type") == "floor":
            floor_by_level[attr["level"]].append(
                (nid, attr["x"], attr["y"], attr["z"]))

    def _nearest(candidates, qx, qy):
        if not candidates:
            return None, None, None, None
        best = min(candidates, key=lambda c: math.hypot(c[1] - qx, c[2] - qy))
        return best  # (nid, x, y, z)

    n_edges = 0
    for nid, attr in g.nodes(data=True):
        nt = attr.get("node_type")
        if nt not in ("fare_gate_entry", "fare_gate_exit"):
            continue

        lk        = attr["level"]
        cx, cy    = attr["x"], attr["y"]
        az        = attr["z"]
        b         = attr["barrier_bounds"]   # [minx, miny, maxx, maxy]
        gate_axis = attr.get("gate_axis", "y")
        paid_side = attr.get("paid_side", "east")
        direction = attr["direction"]

        all_floor = floor_by_level.get(lk, [])

        # Split floor nodes into paid / unpaid sides based on barrier geometry
        if gate_axis == "y":          # barrier is a vertical wall (runs along y)
            if paid_side == "east":
                unpaid = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fx < b[0]]
                paid   = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fx > b[2]]
            else:  # paid side = west
                paid   = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fx < b[0]]
                unpaid = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fx > b[2]]
        else:  # gate_axis == "x": barrier is a horizontal wall (runs along x)
            if paid_side == "north":
                unpaid = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fy < b[1]]
                paid   = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fy > b[3]]
            else:  # paid side = south
                paid   = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fy < b[1]]
                unpaid = [(fn, fx, fy, fz) for fn, fx, fy, fz in all_floor
                          if fy > b[3]]

        u_nid, ux, uy, uz = _nearest(unpaid, cx, cy)
        p_nid, px, py, pz = _nearest(paid,   cx, cy)
        if u_nid is None or p_nid is None:
            continue

        d_u   = math.hypot(ux - cx, uy - cy)
        d_p   = math.hypot(px - cx, py - cy)
        d3d_u = euclidean_3d((cx, cy, az), (ux, uy, uz))
        d3d_p = euclidean_3d((cx, cy, az), (px, py, pz))

        base = dict(
            edge_type="fare_gate",
            gate_group=attr.get("gate_group", ""),
            passage_id=attr.get("passage_id", ""),
            direction=direction,
            throughput_s=2.0,
            gate_penalty_s=3.0,
            queue_capacity=15,
        )

        if nt == "fare_gate_entry":   # unpaid floor → gate → paid floor
            g.add_edge(u_nid, nid,
                       length_2d=d_u, length_3d=d3d_u,
                       travel_time=d3d_u / ws, **base)
            g.add_edge(nid, p_nid,
                       length_2d=d_p, length_3d=d3d_p,
                       travel_time=d3d_p / ws, **base)
        else:                          # fare_gate_exit: paid floor → gate → unpaid floor
            g.add_edge(p_nid, nid,
                       length_2d=d_p, length_3d=d3d_p,
                       travel_time=d3d_p / ws, **base)
            g.add_edge(nid, u_nid,
                       length_2d=d_u, length_3d=d3d_u,
                       travel_time=d3d_u / ws, **base)
        n_edges += 2

    return n_edges


# ====================================================================
#  Security scanner passage edges
# ====================================================================

def _build_scanner_pair_index(g: nx.Graph) -> dict[str, dict[str, str]]:
    """Return {passage_id: {"approach": nid, "exit": nid}} for scanner nodes."""
    idx: dict[str, dict[str, str]] = {}
    for nid, attr in g.nodes(data=True):
        nt = attr.get("node_type")
        if nt not in ("scanner_approach", "scanner_exit"):
            continue
        pid = attr["passage_id"]
        if pid not in idx:
            idx[pid] = {}
        role = "approach" if nt == "scanner_approach" else "exit"
        idx[pid][role] = nid
    return idx


def add_scanner_edges(
    g: nx.Graph,
    pair_idx: dict[str, dict[str, str]],
    config: dict,
) -> int:
    """Connect approach/exit node pairs for each scanner passage.

    Each edge carries:
    - ``edge_type``         : ``"security_scanner"``
    - ``scanner_penalty_s`` : fixed delay (6 s)
    - ``throughput_s``      : seconds per pedestrian (4 s)

    Returns number of edges added.
    """
    ws = config["simulation"]["walking_speed_ms"]
    n_edges = 0

    for pid, pair in pair_idx.items():
        approach_nid = pair.get("approach")
        exit_nid     = pair.get("exit")
        if approach_nid is None or exit_nid is None:
            continue
        aa = g.nodes[approach_nid]
        ea = g.nodes[exit_nid]
        dist = euclidean_2d((aa["x"], aa["y"]), (ea["x"], ea["y"]))
        # One-way directed edge: approach → exit (inbound security check only).
        # Exiting passengers cannot use the scanner passage in reverse.
        g.add_edge(
            approach_nid, exit_nid,
            length_2d=dist, length_3d=dist,
            travel_time=dist / ws + 6.0,
            edge_type="security_scanner",
            passage_id=pid,
            scanner_group=aa.get("scanner_group", ""),
            scanner_penalty_s=6.0,
            throughput_s=4.0,
            queue_capacity=8,
        )
        n_edges += 1

    return n_edges


# ====================================================================
#  PSD door toggle-edges
# ====================================================================

def add_door_toggle_edges(
    g: nx.Graph,
    door_index: dict[str, dict[str, str]],
    config: dict,
) -> int:
    """Add toggle-edges for platform screen doors (platform ↔ track).

    ABM attributes on every PSD edge
    ----------------------------------
    * ``state``           : ``"closed"`` default — train not docked
    * ``toggleable``      : ``True`` — ABM can flip open/closed
    * ``open_duration_s`` : how long doors stay open per train arrival
    * ``close_duration_s``: time to close after open duration expires
    * ``queue_capacity``  : max queueing agents per door segment

    The simulation opens all doors on one side simultaneously when a
    train event fires, then closes them again after *open_duration_s*.

    Returns number of door edges added.
    """
    ws = config["simulation"]["walking_speed_ms"]
    n_edges = 0

    for did, pair in door_index.items():
        pn = pair.get("platform")
        tn = pair.get("track")
        if pn is None or tn is None:
            continue
        np_ = g.nodes[pn]
        nt_ = g.nodes[tn]
        d2d = euclidean_2d(
            (np_["x"], np_["y"]), (nt_["x"], nt_["y"]))
        for _s, _d in ((pn, tn), (tn, pn)):
            g.add_edge(
                _s, _d,
                length_2d=d2d, length_3d=d2d,
                travel_time=d2d / ws,
                edge_type="psd_door",
                door_id=did,
                toggleable=True,
                state="closed",
                open_duration_s=30.0,
                close_duration_s=3.0,
                queue_capacity=6,
            )
            n_edges += 1

    return n_edges


# ====================================================================
#  F-bis. Entrance / exit nodes
# ====================================================================

def _add_entrance_nodes(
    g: nx.Graph,
    all_geometry: dict,
    config: dict,
) -> int:
    """Create width-aware entrance nodes from real entrance spans.

    For each configured entrance rectangle, all floor nodes that:
      1) fall inside the rectangle, and
      2) lie close to the level floor boundary
    are promoted to ``node_type='entrance'``.

    This models entrance gates as a *segment with width* rather than a
    single centroid point. One node per entrance is flagged with
    ``entrance_primary=True`` for cleaner labels in visualisations.

    Fallback: if no boundary node is found, create one centroid entrance
    node and connect it to the nearest floor node (legacy behaviour).

    Returns number of entrance nodes created/promoted.
    """
    ws = config["simulation"]["walking_speed_ms"]
    levels_cfg = config["station"]["levels"]

    # Pre-index floor nodes by level with KD-tree
    floor_by_level: dict[str, tuple[list[str], np.ndarray]] = {}
    for lk in {attr["level"] for _, attr in g.nodes(data=True)}:
        fids, fcoords = [], []
        for nid, attr in g.nodes(data=True):
            if attr.get("node_type") == "floor" and attr["level"] == lk:
                fids.append(nid)
                fcoords.append((attr["x"], attr["y"]))
        if fids:
            floor_by_level[lk] = (fids, cKDTree(np.array(fcoords)))

    n_added = 0
    grid_res = float(config.get("sampling", {}).get("grid_resolution_m", 0.5))
    boundary_tol = max(0.25, grid_res * 0.65)

    for lk, geom in all_geometry.items():
        entrances = geom.get("entrances", [])
        if not entrances:
            continue
        z = levels_cfg.get(lk, {}).get("elevation_m",
                                       geom.get("elevation_m", 0))
        floor = geom.get("floor")
        floor_boundary = floor.boundary if floor is not None and not floor.is_empty else None

        for i, ent in enumerate(entrances):
            poly = ent["polygon"]
            poly_boundary = poly.boundary
            cx = (poly.bounds[0] + poly.bounds[2]) / 2
            cy = (poly.bounds[1] + poly.bounds[3]) / 2
            name = ent.get("name", f"entrance_{lk}_{i}")

            # 1) Promote a SINGLE edge-row of floor nodes inside the entrance span.
            promoted: list[str] = []
            if lk in floor_by_level and floor_boundary is not None:
                fids, _ = floor_by_level[lk]
                minx, miny, maxx, maxy = poly.bounds

                inside_ids: list[str] = []
                for fid in fids:
                    fa = g.nodes[fid]
                    pt = Point(fa["x"], fa["y"])
                    if poly.buffer(1e-6).contains(pt):
                        inside_ids.append(fid)

                # Split into the 4 rectangle-edge rows.
                side_rows: dict[str, list[str]] = {
                    "left": [], "right": [], "bottom": [], "top": [],
                }
                for fid in inside_ids:
                    fa = g.nodes[fid]
                    x, y = float(fa["x"]), float(fa["y"])
                    if abs(x - minx) <= boundary_tol:
                        side_rows["left"].append(fid)
                    if abs(x - maxx) <= boundary_tol:
                        side_rows["right"].append(fid)
                    if abs(y - miny) <= boundary_tol:
                        side_rows["bottom"].append(fid)
                    if abs(y - maxy) <= boundary_tol:
                        side_rows["top"].append(fid)

                # Pick the MOST OUTER side (closest to station floor boundary).
                available_sides = [k for k, v in side_rows.items() if v]
                chosen_ids: list[str] = []
                chosen_side: str | None = None
                if available_sides:
                    def _side_score(side_key: str) -> tuple[float, float]:
                        ids = side_rows[side_key]
                        dsum = 0.0
                        for sid in ids:
                            sa = g.nodes[sid]
                            dsum += floor_boundary.distance(Point(sa["x"], sa["y"]))
                        davg = dsum / max(1, len(ids))
                        return (davg, -float(len(ids)))

                    # For F3 entrances, prioritize the *true outer* side:
                    # count floor nodes in a strip OUTSIDE each side, then pick
                    # the side with the fewest outside nodes.
                    if lk == "F3":
                        pad = max(0.8, 2.0 * grid_res)
                        eps = 1e-6

                        def _outside_count(side_key: str) -> int:
                            c = 0
                            for fid2 in fids:
                                fa2 = g.nodes[fid2]
                                x2 = float(fa2["x"])
                                y2 = float(fa2["y"])
                                if side_key == "left":
                                    if (minx - pad) <= x2 < (minx - eps) and (miny - eps) <= y2 <= (maxy + eps):
                                        c += 1
                                elif side_key == "right":
                                    if (maxx + eps) < x2 <= (maxx + pad) and (miny - eps) <= y2 <= (maxy + eps):
                                        c += 1
                                elif side_key == "bottom":
                                    if (miny - pad) <= y2 < (miny - eps) and (minx - eps) <= x2 <= (maxx + eps):
                                        c += 1
                                elif side_key == "top":
                                    if (maxy + eps) < y2 <= (maxy + pad) and (minx - eps) <= x2 <= (maxx + eps):
                                        c += 1
                            return c

                        best_side = min(
                            available_sides,
                            key=lambda s: (_outside_count(s), _side_score(s)),
                        )
                    else:
                        best_side = min(available_sides, key=_side_score)
                    chosen_side = best_side
                    chosen_ids = side_rows[best_side]
                elif inside_ids:
                    # Degenerate fallback: choose the thinnest row nearest to any
                    # entrance-rectangle edge (still one-row behavior).
                    edge_d = []
                    for fid in inside_ids:
                        fa = g.nodes[fid]
                        d = poly_boundary.distance(Point(fa["x"], fa["y"]))
                        edge_d.append((d, fid))
                    dmin = min(d for d, _ in edge_d)
                    chosen_ids = [fid for d, fid in edge_d if d <= dmin + 1e-6]
                    chosen_side = None

                # Create dedicated entrance nodes projected to the selected edge
                # line, then connect each to its backing floor node.
                for j, fid in enumerate(chosen_ids):
                    fa = g.nodes[fid]
                    x0, y0 = float(fa["x"]), float(fa["y"])

                    if chosen_side == "left":
                        ex, ey = minx, y0
                    elif chosen_side == "right":
                        ex, ey = maxx, y0
                    elif chosen_side == "bottom":
                        ex, ey = x0, miny
                    elif chosen_side == "top":
                        ex, ey = x0, maxy
                    else:
                        # Fallback if side is unknown: keep nearest-point geometry.
                        ex, ey = x0, y0

                    enid = f"ent_{lk}_{name}_{j:03d}"
                    g.add_node(
                        enid,
                        id=enid,
                        x=round(ex, 3), y=round(ey, 3), z=z,
                        level=lk,
                        node_type="entrance",
                        entrance_name=name,
                        entrance_group=name,
                        entrance_primary=False,
                        capacity=100,
                    )

                    d2d = euclidean_2d((ex, ey), (x0, y0))
                    d3d = euclidean_3d((ex, ey, z), (x0, y0, fa["z"]))
                    for _s, _d in ((enid, fid), (fid, enid)):
                        g.add_edge(
                            _s, _d,
                            length_2d=d2d,
                            length_3d=d3d,
                            travel_time=d3d / ws,
                            edge_type="entrance",
                            entrance_name=name,
                        )
                    promoted.append(enid)

            if promoted:
                # Mark one representative for labels / legends
                primary = min(
                    promoted,
                    key=lambda fid: (g.nodes[fid]["x"] - cx) ** 2 + (g.nodes[fid]["y"] - cy) ** 2,
                )
                g.nodes[primary]["entrance_primary"] = True
                n_added += len(promoted)
                continue

            # 2) Fallback (legacy): create one centroid entrance node + link to floor
            nid = f"ent_{lk}_{name}"
            g.add_node(
                nid,
                id=nid,
                x=round(cx, 3), y=round(cy, 3), z=z,
                level=lk,
                node_type="entrance",
                entrance_name=name,
                entrance_group=name,
                entrance_primary=True,
                capacity=100,
            )

            if lk in floor_by_level:
                fids, ftree = floor_by_level[lk]
                dist, idx = ftree.query([cx, cy], k=1)
                fn = g.nodes[fids[idx]]
                d3d = euclidean_3d((cx, cy, z), (fn["x"], fn["y"], fn["z"]))
                for _s, _d in ((nid, fids[idx]), (fids[idx], nid)):
                    g.add_edge(
                        _s, _d,
                        length_2d=dist, length_3d=d3d,
                        travel_time=d3d / ws,
                        edge_type="entrance",
                        entrance_name=name,
                    )
            n_added += 1

    return n_added


# ====================================================================
#  G-extra. Blind path chain edges + dangling-stub pruning
# ====================================================================

def _add_blind_path_chain_edges(g: nx.Graph, config: dict) -> int:
    """Add direct edges between adjacent guide-blind-path nodes.

    After grid floor-graph construction blind-guide node pairs that are
    within ``blind_path_link_factor × grid_res`` of each other but have
    no floor edge (e.g. because LOS failed through a thin obstacle
    footprint) are stitched together.  This guarantees the tactile strip
    is a connected subgraph even when the LOS check would otherwise sever it.

    Returns number of edges added.
    """
    grid_res = config["sampling"]["grid_resolution_m"]
    factor   = config["graph"].get("blind_path_link_factor", 1.5)
    ws       = config["simulation"]["walking_speed_ms"]
    max_link = factor * grid_res

    by_level: dict[str, list] = defaultdict(list)
    for nid, attr in g.nodes(data=True):
        if attr.get("blind_category") == "guide":
            by_level[attr.get("level", "")].append(
                (nid, attr["x"], attr["y"])
            )

    n_added = 0
    for level_nodes in by_level.values():
        if len(level_nodes) < 2:
            continue
        coords = np.array([(x, y) for _, x, y in level_nodes], dtype=np.float64)
        tree = cKDTree(coords)
        for i, j in tree.query_pairs(r=max_link):
            a, b = level_nodes[i][0], level_nodes[j][0]
            na, nb = g.nodes[a], g.nodes[b]
            d2 = euclidean_2d((na["x"], na["y"]), (nb["x"], nb["y"]))
            d3 = euclidean_3d((na["x"], na["y"], na["z"]),
                               (nb["x"], nb["y"], nb["z"]))
            for _s, _d in ((a, b), (b, a)):
                if not g.has_edge(_s, _d):
                    g.add_edge(_s, _d,
                               length_2d=d2, length_3d=d3,
                               travel_time=d3 / ws,
                               edge_type="blind_path")
                    n_added += 1
    return n_added


def _prune_dangling_stubs(g: nx.Graph, config: dict) -> int:
    """Iteratively remove degree-1 floor nodes that are not entrance/exit nodes.

    A "dangling stub" is a floor node with exactly one neighbour that is
    not reachable from any other path — these arise at narrow peninsulas
    in the floor polygon or from isolated artefacts.  We keep degree-1
    nodes that carry ``is_entrance=True`` so actual entry points survive.

    Returns total number of nodes removed.
    """
    removed = 0
    changed = True
    while changed:
        changed = False
        for nid in list(g.nodes()):
            if nid not in g:
                continue
            attr = g.nodes[nid]
            # For DiGraph: count unique adjacent nodes (predecessors + successors)
            unique_adj = set(g.predecessors(nid)) | set(g.successors(nid))
            if (attr.get("node_type") == "floor"
                    and len(unique_adj) == 1
                    and not attr.get("is_entrance", False)):
                g.remove_node(nid)
                removed += 1
                changed = True
    return removed


# ====================================================================
#  G. Main orchestrator
# ====================================================================

def build_navigation_graph(
    all_geometry: dict,
    all_nodes: dict,
    all_connectors: list[dict],
    config: dict,
) -> nx.Graph:
    """Build the unified 2.5D navigation graph.

    Parameters
    ----------
    all_geometry : dict[str, dict]
        Per-level geometry (floor, obstacles, obstacle_union, …).
    all_nodes : dict[str, dict]
        Per-level node data with ``nodes_valid`` list (floor,
        connector-anchor, door, and elevator nodes merged).
    all_connectors : list[dict]
        Typed connector list from Step 1 (stair_chain, escalator,
        elevator).
    config : dict
        Full experiment configuration.
    """
    graph_cfg = config["graph"]
    sim_cfg = config["simulation"]
    levels_cfg = config["station"]["levels"]

    g = nx.DiGraph()

    # ------- A. Floor graphs per level ---------------------------------
    total_floor_edges = 0
    for level_key, node_data in all_nodes.items():
        nodes = node_data.get("nodes_valid", [])
        if not nodes:
            continue
        geom = all_geometry.get(level_key, {})
        obs_union = geom.get("obstacle_union")

        n_edges = build_floor_graph(
            g, nodes, obs_union,
            grid_res=config["sampling"]["grid_resolution_m"],
            connectivity=graph_cfg["neighbor_connectivity"],
            los_check=graph_cfg["los_check"],
            walking_speed=sim_cfg["walking_speed_ms"],
        )
        total_floor_edges += n_edges
        print(f"  {level_key}: {len(nodes):,} nodes, "
              f"{n_edges:,} floor edges")

    # ------- Snap isolated special nodes to floor -----------------------
    n_snap = _snap_isolated_nodes_to_floor(g, config)
    if n_snap:
        print(f"  Anchor → floor snaps: {n_snap}")

    # ------- Build lookup indexes -------------------------------------
    anchor_idx = _build_connector_anchor_index(g)
    elev_idx = _build_elevator_node_index(g)
    door_idx = _build_door_pair_index(g)

    # ------- B. Stair chains ------------------------------------------
    stairs = [c for c in all_connectors if c["type"] == "stair_chain"]
    n_stair = add_stair_chains(
        g, stairs, anchor_idx, levels_cfg, config)
    print(f"  Stair chains: {len(stairs)} chains, {n_stair} edges")

    # ------- C. Escalator links ---------------------------------------
    escalators = [c for c in all_connectors if c["type"] == "escalator"]
    n_esc = add_escalator_links(
        g, escalators, anchor_idx, levels_cfg, config)
    print(f"  Escalators: {len(escalators)} units, {n_esc} edges")

    # ------- D. Elevator transport ------------------------------------
    n_elev = add_elevator_edges(
        g, elev_idx, levels_cfg, config)
    print(f"  Elevator: {n_elev} edges "
          f"(door + interior + transport)")

    # ------- E. PSD door toggle-edges ---------------------------------
    n_door = add_door_toggle_edges(g, door_idx, config)
    print(f"  PSD doors: {n_door} toggle-edges")

    # ------- F. Fare gate passage edges (single-node directed model) ---
    n_fg = _connect_fare_gate_nodes(g, config)
    n_fg_gates = sum(
        1 for _, a in g.nodes(data=True)
        if a.get("node_type") in ("fare_gate_entry", "fare_gate_exit")
    )
    print(f"  Fare gates: {n_fg} passage-edges ({n_fg_gates} gate nodes)")

    # ------- G. Entrance nodes ----------------------------------------
    n_ent = _add_entrance_nodes(g, all_geometry, config)
    if n_ent:
        print(f"  Entrances: {n_ent} entrance nodes")

    # ------- I. Blind path chain edges --------------------------------
    n_blind_edges = _add_blind_path_chain_edges(g, config)
    if n_blind_edges:
        print(f"  Blind path: {n_blind_edges} chain edges added")

    # ------- J. Remove dangling floor stubs ---------------------------
    if graph_cfg.get("prune_dangling_stubs", True):
        n_stubs = _prune_dangling_stubs(g, config)
        if n_stubs:
            print(f"  Pruned {n_stubs} dangling floor stubs")

    # ------- H. Prune to largest connected component ------------------
    if graph_cfg.get("prune_to_largest_cc", True):
        ccs = list(nx.weakly_connected_components(g))
        if len(ccs) > 1:
            largest = max(ccs, key=len)
            removed = g.number_of_nodes() - len(largest)
            g = g.subgraph(largest).copy()
            print(f"  Pruned {removed} nodes from "
                  f"{len(ccs) - 1} small components")

    print(f"\n  [Step 3] Graph: {g.number_of_nodes():,} nodes, "
          f"{g.number_of_edges():,} edges")
    return g


# ====================================================================
#  H. Save outputs
# ====================================================================

def save_graph_outputs(
    g: nx.Graph,
    all_connectors: list[dict],
    out_dir: str | Path,
) -> None:
    """Save graph to gpickle and export GeoJSON summaries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- graph pickle --
    with (out_dir / "navigation_graph.gpickle").open("wb") as f:
        pickle.dump(g, f)

    # -- statistics --
    node_types: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    for _, attr in g.nodes(data=True):
        node_types[attr.get("node_type", "unknown")] += 1
        level_counts[attr.get("level", "unknown")] += 1

    edge_types: dict[str, int] = defaultdict(int)
    for _, _, attr in g.edges(data=True):
        edge_types[attr.get("edge_type", "unknown")] += 1

    conn_summary = {
        "stair_chains": len([c for c in all_connectors
                             if c["type"] == "stair_chain"]),
        "escalators": len([c for c in all_connectors
                           if c["type"] == "escalator"]),
        "elevators": len([c for c in all_connectors
                          if c["type"] == "elevator"]),
    }

    summary = {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "node_types": dict(node_types),
        "level_node_counts": dict(level_counts),
        "edge_types": dict(edge_types),
        "connectors": conn_summary,
        "is_connected": nx.is_weakly_connected(g) if g.is_directed() else nx.is_connected(g),
    }
    dump_json(out_dir / "graph_summary.json", summary)

    # -- nodes GeoJSON --
    node_feats = []
    for nid, attr in g.nodes(data=True):
        props = {"id": nid}
        props.update({k: v for k, v in attr.items()
                      if isinstance(v, (str, int, float, bool))})
        node_feats.append(point_feature(attr["x"], attr["y"], props))
    write_geojson(out_dir / "nodes_all.geojson", node_feats)

    # -- edges GeoJSON (one file per edge_type) --
    for etype in edge_types:
        feats = []
        for u, v, attr in g.edges(data=True):
            if attr.get("edge_type") != etype:
                continue
            nu, nv = g.nodes[u], g.nodes[v]
            props = {"u": u, "v": v}
            props.update({k: val for k, val in attr.items()
                          if isinstance(val, (str, int, float, bool))})
            feats.append(line_feature(
                [(nu["x"], nu["y"]), (nv["x"], nv["y"])], props))
        write_geojson(out_dir / f"edges_{etype}.geojson", feats)
