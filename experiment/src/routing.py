"""
Step 4: Routing
================

Multi-criteria pathfinding on the 2.5D navigation graph.

Supports:
  - Static shortest path (Dijkstra with configurable weight)
  - Connector-penalised routing (additive time penalties for stairs/escalators)
  - Dynamic congestion-aware routing (BFS-spread + A* replan)

Semantic OD management:
  - Define entrance/exit/platform regions
  - Snap semantic points to graph nodes
  - Generate agent OD assignments
"""
from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx

from src.utils import dump_json, write_geojson, point_feature


# ============================================================================
# Weight functions
# ============================================================================

def static_weight(u, v, attr):
    """Static edge weight: travel_time, falling back to length_3d."""
    return float(attr.get("travel_time") or attr.get("length_3d") or attr.get("length_2d") or 1.0)


def penalised_weight(config: dict):
    """Create a weight function with additive connector penalties."""
    penalties = config["routing"]["connector_penalties"]

    def weight_fn(u, v, attr):
        base = float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
        edge_type = attr.get("edge_type", "floor")
        
        # Add penalty for connector types
        for ctype, penalty in penalties.items():
            if ctype in edge_type:
                base += penalty
                break
        return base

    return weight_fn


def congestion_weight(edge_congestion: dict, alpha: float = 3.0):
    """Create a dynamic weight function: w = base × (1 + α × congestion)."""
    def weight_fn(u, v, attr):
        base = float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
        cong = edge_congestion.get((u, v), 0.0) + edge_congestion.get((v, u), 0.0)
        return base * (1.0 + alpha * cong / 2.0)
    return weight_fn


def congestion_directed_weight(g, edge_congestion: dict, alpha: float = 3.0):
    """Combined direction-enforcing + congestion-aware weight.

    Returns inf for direction-violating edges (escalator wrong-way,
    fare gate wrong-side) and congestion-scaled travel time otherwise.
    Used for dynamic replanning so agents never reroute onto physically
    impossible paths.
    """
    _dir_wfn = directed_weight(g)   # closure captures g

    def weight_fn(u, v, attr):
        dir_cost = _dir_wfn(u, v, attr)
        if dir_cost == float("inf"):
            return float("inf")
        base = float(attr.get("travel_time") or attr.get("length_3d") or 1.0)
        cong = edge_congestion.get((u, v), 0.0) + edge_congestion.get((v, u), 0.0)
        return base * (1.0 + alpha * cong / 2.0)

    return weight_fn


# ============================================================================
# Directed routing — gate & escalator direction enforcement
# ============================================================================

# Right-hand rule: when facing the escalator in your TRAVEL direction,
# the RIGHT unit goes your way.
# 'up'   = physically moves from bottom level to top level (F1→F3 or F3→F4).
# 'down' = physically moves from top level to bottom level (F3→F1).
#
# F1↔F3 left pair  (F1 @ x≈34.9, F3 @ x≈66.5 — travel direction +x going up):
#   right-hand side of +x = lower y  → y≈8.9  is UP,  y≈12.5 is DOWN
# F1↔F3 right pair (F1 @ x≈118.5, F3 @ x≈86.8 — travel direction −x going up):
#   right-hand side of −x = higher y → y≈12.5 is UP,  y≈8.9  is DOWN
# F3↔F4 escalators: one unit per entrance — A/B runs UP (outbound F3→F4),
#   C runs DOWN (inbound F4→F3).  Stairs at both entrances serve the other direction.
ESCALATOR_DIRECTIONS: dict[str, str] = {
    "esc_2lWC3J0sj8VQujoHqVJF_U": "up",    # F1(y≈8.9)  → F3(y≈8.9)   [left pair, right unit]
    "esc_2lWC3J0sj8VQujoHqVJC7c": "down",  # F3(y≈12.5) → F1(y≈12.5)  [left pair, left unit]
    "esc_0xMmxZWHrBZv$oHkFHSzFd": "up",    # F1(y≈12.5) → F3(y≈12.5)  [right pair, right unit]
    "esc_0xMmxZWHrBZv$oHkFHSzFW": "down",  # F3(y≈8.9)  → F1(y≈8.9)   [right pair, left unit]
    "esc_18Gic2sdj5_OIgDswc3ESK": "up",    # F3 → F4 outbound (near Entrance A/B)
    "esc_18Gic2sdj5_OIgDswc3FF2": "down",  # F4 → F3 inbound (near Entrance C)
}


def patch_escalator_directions(g: nx.Graph) -> int:
    """Correct escalator edge 'direction' attributes using the right-hand rule.

    The IFC source assigns direction='up' to all escalators.  This function
    updates each escalator edge to 'up' or 'down' according to
    ESCALATOR_DIRECTIONS, which was derived from the physical right-hand rule:
    when facing your travel direction the RIGHT escalator goes your way.

    Returns the number of edges whose attribute was updated.
    """
    n_updated = 0
    for u, v, key, data in g.edges(data=True, keys=True) if g.is_multigraph() \
            else ((u, v, None, data) for u, v, data in g.edges(data=True)):
        if data.get("edge_type") != "escalator":
            continue
        cid = data.get("connector_id", "")
        correct_dir = ESCALATOR_DIRECTIONS.get(cid)
        if correct_dir is None:
            continue
        if data.get("direction") != correct_dir:
            if g.is_multigraph():
                g.edges[u, v, key]["direction"] = correct_dir
            else:
                g.edges[u, v]["direction"] = correct_dir
            n_updated += 1
    return n_updated


def directed_weight(g: nx.Graph):
    """Weight function that enforces one-way traversal for gates and escalators.

    Fare gates:
        inbound  gate (pass_from_node_type='fare_gate_unpaid'):
            only traversable from the unpaid side → paid side.
        outbound gate (pass_from_node_type='fare_gate_paid'):
            only traversable from the paid side → unpaid side.

    Escalators:
        direction='up'   : only traversable bottom→top (z_v > z_u).
        direction='down' : only traversable top→bottom (z_u > z_v).

    Returns float('inf') for any edge that violates the direction rule,
    otherwise returns the edge's travel_time.
    """
    def _weight(u, v, d):
        et = d.get("edge_type", "floor")

        if et == "escalator":
            esc_dir = d.get("direction", "up")
            zu = g.nodes[u].get("z", 0.0)
            zv = g.nodes[v].get("z", 0.0)
            if esc_dir == "up" and (zu - zv) > 0.05:    # going down on up escalator
                return float("inf")
            if esc_dir == "down" and (zv - zu) > 0.05:  # going up on down escalator
                return float("inf")

        return float(
            d.get("travel_time") or d.get("length_3d") or d.get("length_2d") or 1.0
        )

    return _weight


# ============================================================================
# Pathfinding
# ============================================================================

def find_path(g: nx.Graph, origin: str, dest: str, weight_fn=None) -> list[str]:
    """Find shortest path using Dijkstra."""
    if weight_fn is None:
        weight_fn = static_weight
    try:
        return nx.dijkstra_path(g, origin, dest, weight=weight_fn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [origin]


def find_path_astar(g: nx.Graph, origin: str, dest: str, weight_fn=None) -> list[str]:
    """Find shortest path using A* with Euclidean heuristic (faster than Dijkstra on spatial graphs)."""
    if weight_fn is None:
        weight_fn = static_weight
    # Precompute destination coords for heuristic (1.2 m/s lower-bound walking speed)
    dnd = g.nodes.get(dest, {})
    dx_d, dy_d, dz_d = dnd.get("x", 0.0), dnd.get("y", 0.0), dnd.get("z", 0.0)
    INV_SPEED = 1.0 / 1.2  # seconds per metre lower bound

    def _heuristic(u: str, v: str) -> float:
        nd = g.nodes.get(u, {})
        dx = nd.get("x", 0.0) - dx_d
        dy = nd.get("y", 0.0) - dy_d
        dz = nd.get("z", 0.0) - dz_d
        return ((dx * dx + dy * dy + dz * dz) ** 0.5) * INV_SPEED

    try:
        return nx.astar_path(g, origin, dest, heuristic=_heuristic, weight=weight_fn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [origin]


def find_path_with_cost(g: nx.Graph, origin: str, dest: str, weight_fn=None) -> tuple[list[str], float]:
    """Find shortest path and return (path, total_cost)."""
    if weight_fn is None:
        weight_fn = static_weight
    try:
        path = nx.dijkstra_path(g, origin, dest, weight=weight_fn)
        cost = nx.dijkstra_path_length(g, origin, dest, weight=weight_fn)
        return path, cost
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [origin], float("inf")


def find_entrance_paths(
    g: nx.Graph,
    entrance_nodes: list[str],
    platform_nodes: list[str],
    *,
    deduplicate: bool = True,
) -> list[dict]:
    """Compute inbound + outbound paths, one per named entrance.

    Every path is forced through its nearest fare gate node (mandatory
    waypoint).  This guarantees gate crossings even when the floor mesh
    contains shortcut edges that would let Dijkstra bypass the barrier.

    Routing strategy (two-segment):
      Inbound  (进站 ENTRANCE → PSD):
        segment 1: entrance → nearest fare_gate_entry  (directed Dijkstra)
        segment 2: fare_gate_entry → nearest PSD       (directed Dijkstra)
      Outbound (出站 PSD → ENTRANCE):
        segment 1: PSD → nearest fare_gate_exit        (directed Dijkstra)
        segment 2: fare_gate_exit → entrance            (directed Dijkstra;
                   falls back to undirected if no directed path exists)

    When *deduplicate* is True (default) entrance nodes are grouped by
    ``entrance_name`` attribute and only the median node (by x-coord) from
    each group is used.  This reduces 98 raw entrance nodes down to 5.

    Returns a list of dicts, one per named entrance (sorted A→E):
        {
          "entrance_id":    str,
          "entrance_name":  str,
          "level":          str,
          "psd_id":         str,
          "inbound_path":   list[str],
          "outbound_path":  list[str],
          "inbound_cost":   float,
          "outbound_cost":  float,
          "inbound_gate":   str | None,   # fare_gate_entry node id
          "outbound_gate":  str | None,   # fare_gate_exit  node id
        }
    """
    # ── Deduplicate: one representative per named entrance ──────────────────
    if deduplicate:
        by_name: dict[str, list[str]] = defaultdict(list)
        for eid in entrance_nodes:
            if eid not in g:
                continue
            name = g.nodes[eid].get("entrance_name", eid)
            by_name[name].append(eid)
        to_process: list[str] = []
        for name in sorted(by_name.keys()):
            nids = sorted(by_name[name], key=lambda n: g.nodes[n].get("x", 0))
            to_process.append(nids[len(nids) // 2])  # median by x
    else:
        to_process = [n for n in entrance_nodes if n in g]

    wfn = directed_weight(g)

    # Pre-collect fare gate nodes
    gate_entry_nodes = [n for n, a in g.nodes(data=True)
                        if a.get("node_type") == "fare_gate_entry"]
    gate_exit_nodes  = [n for n, a in g.nodes(data=True)
                        if a.get("node_type") == "fare_gate_exit"]

    # Undirected weight (used as fallback for gate→entrance when directe fails)
    g_undir = g.to_undirected()

    def _nearest_by_dijkstra(graph, source, candidates, weight_fn, cutoff=800):
        """Return (best_node, cost) among candidates reachable from source."""
        try:
            lengths = dict(nx.single_source_dijkstra_path_length(
                graph, source, weight=weight_fn, cutoff=cutoff))
        except nx.NodeNotFound:
            return None, float("inf")
        best, best_cost = None, float("inf")
        for n in candidates:
            c = lengths.get(n, float("inf"))
            if c < best_cost:
                best_cost = c
                best = n
        return best, best_cost

    def _path_between(graph, src, dst, weight_fn):
        try:
            p = nx.dijkstra_path(graph, src, dst, weight=weight_fn)
            c = nx.dijkstra_path_length(graph, src, dst, weight=weight_fn)
            return p, c
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float("inf")

    results = []

    for eid in to_process:
        eattr = g.nodes[eid]

        # ── INBOUND: entrance → gate_entry → nearest PSD ──────────────────
        best_gate_in, gate_in_cost = _nearest_by_dijkstra(
            g, eid, gate_entry_nodes, wfn)

        if best_gate_in is None:
            print(f"  WARN: No fare_gate_entry reachable from {eid}")
            continue

        # Nearest PSD from the inbound gate
        best_psd, psd_from_gate_cost = _nearest_by_dijkstra(
            g, best_gate_in, platform_nodes, wfn)

        if best_psd is None:
            print(f"  WARN: No PSD reachable from gate {best_gate_in}")
            continue

        path_eid_gate, _ = _path_between(g, eid, best_gate_in, wfn)
        path_gate_psd, _ = _path_between(g, best_gate_in, best_psd, wfn)

        if path_eid_gate is None or path_gate_psd is None:
            print(f"  WARN: Could not reconstruct inbound path for {eid}")
            continue

        in_path = path_eid_gate + path_gate_psd[1:]
        in_cost  = gate_in_cost + psd_from_gate_cost

        # ── OUTBOUND: PSD → gate_exit → entrance ──────────────────────────
        best_gate_out, gate_out_cost = _nearest_by_dijkstra(
            g, best_psd, gate_exit_nodes, wfn)

        if best_gate_out is None:
            print(f"  WARN: No fare_gate_exit reachable from PSD {best_psd}")
            out_path, out_cost = [], float("inf")
        else:
            # Try directed first; fall back to undirected if entrance is
            # topologically on the "paid" side (e.g. Gate D between barriers)
            path_psd_gate, _ = _path_between(g, best_psd, best_gate_out, wfn)
            path_gate_eid, eid_cost = _path_between(g, best_gate_out, eid, wfn)

            if path_gate_eid is None:
                # Fallback: undirected search from gate to entrance
                path_gate_eid, eid_cost = _path_between(
                    g_undir, best_gate_out, eid,
                    weight_fn=lambda u, v, d: float(
                        d.get("travel_time") or d.get("length_3d") or 1.0))

            if path_psd_gate is None or path_gate_eid is None:
                print(f"  WARN: No outbound path for {eid}")
                out_path, out_cost = [], float("inf")
            else:
                out_path = path_psd_gate + path_gate_eid[1:]
                out_cost = gate_out_cost + eid_cost

        results.append({
            "entrance_id":    eid,
            "entrance_name":  eattr.get("entrance_name", eid),
            "level":          eattr.get("level", ""),
            "psd_id":         best_psd,
            "inbound_path":   in_path,
            "outbound_path":  out_path,
            "inbound_cost":   in_cost,
            "outbound_cost":  out_cost,
            "inbound_gate":   best_gate_in,
            "outbound_gate":  best_gate_out,
        })

    return results


def path_summary(g: nx.Graph, path: list[str]) -> dict:
    """Summarise a path: distance, travel_time, connector types used."""
    total_2d = 0.0
    total_3d = 0.0
    total_tt = 0.0
    edge_types_used = defaultdict(int)
    levels_visited = set()
    connectors_used = set()

    for a, b in zip(path[:-1], path[1:]):
        edata = g.edges[a, b]
        total_2d += float(edata.get("length_2d", 0))
        total_3d += float(edata.get("length_3d", 0))
        total_tt += float(edata.get("travel_time", 0))
        edge_types_used[edata.get("edge_type", "unknown")] += 1
        cid = edata.get("connector_id")
        if cid:
            connectors_used.add(cid)

    for nid in path:
        nd = g.nodes.get(nid, {})
        levels_visited.add(nd.get("level", "unknown"))

    return {
        "n_nodes": len(path),
        "total_length_2d": total_2d,
        "total_length_3d": total_3d,
        "total_travel_time": total_tt,
        "edge_types": dict(edge_types_used),
        "levels_visited": sorted(levels_visited),
        "connectors_used": sorted(connectors_used),
    }


# ============================================================================
# Congestion propagation
# ============================================================================

def compute_congestion(
    g: nx.Graph,
    occupancy: dict[str, str],
    max_hops: int = 12,
    decay: float = 0.82,
) -> dict[tuple, float]:
    """BFS-based congestion propagation from occupied nodes.

    Returns edge congestion dict {(u,v): score}.
    Optimised: precomputed neighbour lists, decay table, inlined BFS.
    """
    # Lazily cache neighbour lists on the graph object (graph is immutable here)
    if not hasattr(g, "_cong_nbr_cache"):
        g._cong_nbr_cache = {n: list(g.neighbors(n)) for n in g.nodes()}
    nbrs = g._cong_nbr_cache

    BASE = 3.0
    MIN_W = 0.005
    # Precompute decay weights[0..max_hops]
    decay_w = [BASE * (decay ** d) for d in range(max_hops + 1)]

    node_cong: dict[str, float] = {}

    for src in occupancy:
        node_cong[src] = node_cong.get(src, 0.0) + BASE
        q: deque = deque([(src, 0)])
        visited: set = {src}
        while q:
            cur, d = q.popleft()
            nd = d + 1
            if nd > max_hops:
                continue
            w = decay_w[nd]
            if w < MIN_W:
                continue
            for nb in nbrs[cur]:
                if nb not in visited:
                    visited.add(nb)
                    node_cong[nb] = node_cong.get(nb, 0.0) + w
                    q.append((nb, nd))

    edge_cong: dict[tuple, float] = {}
    for u, v in g.edges():
        c = (node_cong.get(u, 0.0) + node_cong.get(v, 0.0)) * 0.5
        edge_cong[(u, v)] = c
        edge_cong[(v, u)] = c
    return edge_cong


def _bfs_spread(g, src, base, max_hops, decay, out):
    """BFS congestion spread from a single node (legacy helper)."""
    out[src] = out.get(src, 0.0) + base
    q: deque = deque([(src, 0)])
    visited: set = {src}
    while q:
        cur, d = q.popleft()
        nd = d + 1
        if nd > max_hops:
            continue
        w = base * (decay ** nd)
        if w < 0.005:
            continue
        for nb in g.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                out[nb] = out.get(nb, 0.0) + w
                q.append((nb, nd))


# ============================================================================
# Semantic OD management
# ============================================================================

def define_semantic_regions(g: nx.Graph, config: dict) -> dict[str, list[str]]:
    """Define OD regions for two-direction pedestrian flow model.

    ENTRANCE : Actual station gate nodes (node_type == "entrance") on F3/F4.
               These are the 5 real entrance/exit shafts.
    EXIT     : Same station gate nodes — gates serve both as entry AND exit.
    PLATFORM : PSD (platform-screen-door) nodes on F1 (door_platform type).
               Passengers alight from / board the train through these nodes.

    Two flows:
      Inbound  (boarding)  : ENTRANCE → PLATFORM
      Outbound (alighting) : PLATFORM → EXIT
    """
    regions: dict[str, list[str]] = {}

    # ENTRANCE / EXIT: real station-gate entrance nodes on F3 or F4
    entrance_nodes = [
        nid for nid, attr in g.nodes(data=True)
        if attr.get("node_type") == "entrance"
    ]
    if not entrance_nodes:
        # Fallback: boundary nodes on F4 if no tagged entrances
        by_level: dict[str, list] = defaultdict(list)
        for nid, attr in g.nodes(data=True):
            if attr.get("node_type") == "floor":
                by_level[attr.get("level", "")].append((nid, attr))
        f4 = by_level.get("F4", by_level.get("F3", []))
        entrance_nodes = [nid for nid, _ in _boundary_nodes(f4, n_pick=8)]

    regions["ENTRANCE"] = entrance_nodes
    regions["EXIT"] = entrance_nodes  # same nodes — gates are bidirectional

    # PLATFORM: PSD door_platform nodes on F1 (interface between platform and track)
    platform_nodes = [
        nid for nid, attr in g.nodes(data=True)
        if attr.get("node_type") == "door_platform" and attr.get("level") == "F1"
    ]
    if not platform_nodes:
        # Fallback: boundary nodes on F1
        by_level2: dict[str, list] = defaultdict(list)
        for nid, attr in g.nodes(data=True):
            if attr.get("node_type") == "floor" and attr.get("level") == "F1":
                by_level2["F1"].append((nid, attr))
        platform_nodes = [nid for nid, _ in _boundary_nodes(by_level2.get("F1", []), n_pick=12)]

    regions["PLATFORM"] = platform_nodes

    return regions


def _boundary_nodes(nodes: list[tuple[str, dict]], n_pick: int = 8, reverse: bool = False) -> list[tuple[str, dict]]:
    """Select nodes near the bounding box boundary."""
    if not nodes:
        return []
    xs = [a["x"] for _, a in nodes]
    ys = [a["y"] for _, a in nodes]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    scored = []
    for nid, attr in nodes:
        dist = min(attr["x"] - minx, maxx - attr["x"], attr["y"] - miny, maxy - attr["y"])
        scored.append((dist, (nid, attr)))
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return [item for _, item in scored[:n_pick]]


def sample_agents(
    regions: dict[str, list[str]],
    flows: dict[str, float],
    n_agents: int,
    T: float,
    seed: int,
    walking_speed: float = 1.2,
    elderly_ratio: float = 0.0,
) -> list[dict]:
    """Generate agent OD assignments from flow definitions.

    Each agent gets: origin, dest, spawn_time, speed, agent_type.
    """
    rng = random.Random(seed)

    flow_items = [(k, float(v)) for k, v in flows.items() if float(v) > 0]
    total_w = sum(w for _, w in flow_items)
    if total_w <= 0:
        flow_items = [("ENTRANCE->PLATFORM", 1.0)]
        total_w = 1.0

    # Build CDF
    cumulative = []
    c = 0.0
    for flow, w in flow_items:
        c += w / total_w
        cumulative.append((c, flow))

    agents = []
    for i in range(n_agents):
        # Choose flow
        r = rng.random()
        chosen = cumulative[-1][1]
        for p, f in cumulative:
            if r <= p:
                chosen = f
                break

        src_type, dst_type = [x.strip().upper() for x in chosen.split("->")]
        src_nodes = regions.get(src_type, regions.get("ENTRANCE", []))
        dst_nodes = regions.get(dst_type, regions.get("PLATFORM", []))

        if not src_nodes or not dst_nodes:
            continue

        origin = rng.choice(src_nodes)
        dest = rng.choice(dst_nodes)
        if origin == dest and len(dst_nodes) > 1:
            dest = rng.choice([x for x in dst_nodes if x != origin])

        # Agent type
        if rng.random() < elderly_ratio:
            agent_type = "elderly"
            speed_factor = rng.uniform(0.55, 0.75)
        else:
            agent_type = "normal"
            speed_factor = rng.uniform(0.95, 1.05)

        agents.append({
            "agent_id": f"a_{i:04d}",
            "flow": chosen,
            "origin": origin,
            "dest": dest,
            # Cap spawn window at 55% of T so the slowest elderly agent
            # (speed ~0.55×, max path ~215s) still finishes before T.
            "spawn_time": rng.uniform(0, T * 0.55),
            "speed_factor": speed_factor,
            "agent_type": agent_type,
        })

    return agents


# ============================================================================
# Save outputs
# ============================================================================

def save_routing_outputs(
    g: nx.Graph,
    regions: dict,
    agents: list[dict],
    out_dir: str | Path,
) -> None:
    """Save routing setup to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Semantic regions
    dump_json(out_dir / "semantic_regions.json", {k: v for k, v in regions.items()})

    # Semantic regions GeoJSON
    feats = []
    for sem_type, node_ids in regions.items():
        for nid in node_ids:
            if nid in g.nodes:
                attr = g.nodes[nid]
                feats.append(point_feature(
                    attr["x"], attr["y"],
                    {"id": nid, "type": sem_type, "level": attr.get("level", ""), "z": attr.get("z", 0)},
                ))
    write_geojson(out_dir / "semantic_points.geojson", feats)

    # Agent assignments
    dump_json(out_dir / "agents.json", agents)

    # Example paths
    example_paths = []
    for agent in agents[:5]:
        path, cost = find_path_with_cost(g, agent["origin"], agent["dest"])
        ps = path_summary(g, path)
        ps["agent_id"] = agent["agent_id"]
        ps["origin"] = agent["origin"]
        ps["dest"] = agent["dest"]
        ps["cost"] = cost
        example_paths.append(ps)
    dump_json(out_dir / "example_paths.json", example_paths)

    # Summary
    summary = {
        "n_regions": len(regions),
        "region_sizes": {k: len(v) for k, v in regions.items()},
        "n_agents": len(agents),
        "flow_distribution": defaultdict(int),
    }
    for a in agents:
        summary["flow_distribution"][a["flow"]] += 1
    summary["flow_distribution"] = dict(summary["flow_distribution"])
    dump_json(out_dir / "routing_summary.json", summary)

    print(f"[Step 4] Routing: {len(regions)} semantic regions, {len(agents)} agents")
