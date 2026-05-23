"""
Step 5: ABM Simulation
=======================

Enhanced Agent-Based Model with typed connector semantics.

Features:
  - Heterogeneous agents (normal + elderly with different speeds)
  - Static routing (Dijkstra shortest path, fixed at spawn)
  - Dynamic routing (periodic congestion-aware replanning)
  - Typed connector capacity gates (stairs, escalators, elevators)
  - Per-tick trajectory and occupancy recording

Outputs:
  - traj_agents.jsonl: per-tick agent positions
  - summary.csv: aggregate metrics
  - Travel time distributions
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import networkx as nx

from src.routing import (find_path, static_weight, congestion_weight,
                         compute_congestion, congestion_directed_weight as _cdw,
                         find_path_astar as _find_path_fast)


RoutingMode = Literal["static", "dynamic"]


def run_simulation(
    g: nx.Graph,
    agents: list[dict],
    config: dict,
    out_dir: str | Path,
    routing_mode: RoutingMode = "static",
    label: str = "",
    write_traj: bool = True,
) -> dict:
    """Run the ABM simulation.

    Parameters
    ----------
    g : nx.Graph
        The 2.5D navigation graph.
    agents : list[dict]
        Agent definitions (from routing.sample_agents).
    config : dict
        Experiment config.
    out_dir : path-like
        Output directory.
    routing_mode : "static" or "dynamic"
        Whether agents replan based on congestion.
    label : str
        Label for this simulation run (e.g. "baseline", "dynamic").

    Returns
    -------
    dict with travel_times, wait_times, stair_queue, edge_throughput, etc.
    """
    sim_cfg = config["simulation"]
    dt = sim_cfg["dt_s"]
    T = sim_cfg["T_s"]
    seed = sim_cfg["seed"]
    walking_speed = sim_cfg["walking_speed_ms"]
    stair_cap = config["connectors"]["stair"]["capacity"]
    esc_cap = config["connectors"]["escalator"]["capacity"]
    replan_interval = sim_cfg.get("replan_interval_s", 5.0)
    replan_wait_thresh = sim_cfg.get("replan_wait_threshold_s", 2.0)
    congestion_alpha = sim_cfg.get("congestion_alpha", 2.0)
    # When False, agents only replan when actively waiting (not on a timer).
    # This models realistic behaviour: pedestrians deviate only when blocked.
    replan_timer_enabled = sim_cfg.get("replan_timer_enabled", False)
    # Throttle: recompute congestion every K steps (1 = every step, 2 = every 1s at dt=0.5s)
    congestion_every_k = sim_cfg.get("congestion_recompute_every", 1)
    # Cap: at most this many Dijkstra replans per timestep (prevents O(N) per step for large N)
    max_replans_per_step = sim_cfg.get("max_replans_per_step", 9999)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    agents_by_id = {a["agent_id"]: dict(a) for a in agents}

    # Identify connector node sets for capacity gating
    # node_types: stair anchor='stair_chain', stair step='stair_step'
    #             esc anchor='escalator', esc step='escalator_step'
    #             elevator='elevator_entry'/'elevator_interior'
    stair_nodes = {n for n, d in g.nodes(data=True)
                   if d.get("node_type") in ("stair_chain", "stair_step")}
    esc_nodes = {n for n, d in g.nodes(data=True)
                 if d.get("node_type") in ("escalator", "escalator_step")}
    elev_nodes = {n for n, d in g.nodes(data=True)
                  if d.get("node_type") in ("elevator_entry", "elevator_interior")}

    # Map each escalator node → its connector_id for per-unit capacity gating
    esc_node_cid: dict[str, str] = {
        n: d["connector_id"]
        for n, d in g.nodes(data=True)
        if d.get("node_type") in ("escalator", "escalator_step")
        and d.get("connector_id")
    }

    # Apply correct escalator directions (IFC default is 'up' for all; patch overrides
    # based on ESCALATOR_DIRECTIONS which encodes right-hand-rule + inbound/outbound roles).
    from src.routing import directed_weight as _directed_weight, patch_escalator_directions as _patch_esc
    _patch_esc(g)
    _dir_wfn = _directed_weight(g)
    paths: dict[str, list[str]] = {}
    for a in agents:
        paths[a["agent_id"]] = find_path(g, a["origin"], a["dest"], _dir_wfn)

    occupancy: dict[str, str] = {}  # node_id -> agent_id
    active: set[str] = set()
    arrived: set[str] = set()

    state = {
        aid: {
            "path_idx": 0,
            "wait_time": 0.0,
            "total_wait": 0.0,
            "arrive_time": None,
            "started": False,
            "last_replan_t": -999.0,
            "replan_count": 0,
        }
        for aid in agents_by_id
    }

    # Output streams
    traj_fp = (out_dir / "traj_agents.jsonl").open("w", encoding="utf-8") if write_traj else None

    # Write agent metadata (origin / dest / type) for downstream visualisation
    if write_traj:
        import json as _json
        _meta = {
            a["agent_id"]: {
                "origin": a["origin"],
                "dest": a["dest"],
                "agent_type": a.get("agent_type", "normal"),
                "spawn_time": a.get("spawn_time", 0.0),
            }
            for a in agents
        }
        with (out_dir / "agent_meta.json").open("w", encoding="utf-8") as _mf:
            _mf.write(_json.dumps(_meta, ensure_ascii=False))

    # Accumulators
    edge_throughput: dict[tuple, int] = defaultdict(int)
    stair_queue_over_time: list[tuple[float, int]] = []
    replan_events: list[dict] = []
    _cached_econg: dict = {}
    _cached_wdict: dict = {}   # pre-baked weight dict for fast Dijkstra
    _cong_step_counter: int = 0

    # Time steps
    times = []
    t = 0.0
    while t <= T + 1e-9:
        times.append(round(t, 3))
        t += dt

    for t in times:
        # --- Spawn ---
        for aid, a in agents_by_id.items():
            if state[aid]["started"] or aid in arrived:
                continue
            if a["spawn_time"] <= t:
                start = paths[aid][0]
                if start not in occupancy:
                    occupancy[start] = aid
                    state[aid]["started"] = True
                    active.add(aid)

        # --- Dynamic replanning ---
        if routing_mode == "dynamic":
            _cong_step_counter += 1
            if _cong_step_counter >= congestion_every_k:
                _cong_step_counter = 0
                _cached_econg = compute_congestion(g, occupancy,
                                                   max_hops=sim_cfg.get("congestion_max_hops", 12),
                                                   decay=sim_cfg.get("congestion_decay", 0.82))
                # Bake the weight function into a dict for O(1) lookups in Dijkstra
                _wfn_tmp = _cdw(g, _cached_econg, alpha=congestion_alpha)
                _cached_wdict = {}
                for _u, _v, _d in g.edges(data=True):
                    _cached_wdict[(_u, _v)] = _wfn_tmp(_u, _v, _d)
                    _cached_wdict[(_v, _u)] = _wfn_tmp(_v, _u, _d)
            # Use pre-baked dict via lambda (2x faster than function weight per Dijkstra call)
            _wlambda = _cached_wdict.__getitem__ if _cached_wdict else None
            wfunc = (lambda u, v, d, _wd=_cached_wdict: _wd.get((u, v), 1.0)) if _cached_wdict else _cdw(g, _cached_econg, alpha=congestion_alpha)

            _replans_this_step = 0
            for aid in list(active):
                if _replans_this_step >= max_replans_per_step:
                    break
                a = agents_by_id[aid]
                if a.get("agent_type") == "elderly":
                    continue  # Elderly never replan

                st = state[aid]
                since = t - st["last_replan_t"]
                should_replan = False

                if replan_timer_enabled and since >= replan_interval:
                    should_replan = True
                if st["wait_time"] >= replan_wait_thresh:
                    should_replan = True

                if should_replan and st["path_idx"] < len(paths[aid]) - 1:
                    cur = paths[aid][st["path_idx"]]
                    new_path = _find_path_fast(g, cur, a["dest"], wfunc)
                    if len(new_path) > 1:
                        paths[aid] = new_path
                        st["path_idx"] = 0
                        st["last_replan_t"] = t
                        st["replan_count"] += 1
                        st["wait_time"] = 0.0
                        replan_events.append({
                            "t": t, "agent_id": aid, "new_path_len": len(new_path),
                        })
                        _replans_this_step += 1

        # --- Movement ---
        queue_near_stairs = 0
        moves: list[tuple[str, str, str]] = []

        # Pre-count connector occupancy ONCE per timestep (avoids O(N²) per-agent recount)
        _stair_occ = sum(1 for n in occupancy if n in stair_nodes)
        # Per-unit escalator occupancy: connector_id → agent count
        _esc_cid_occ: dict[str, int] = {}
        for _n in occupancy:
            _cid = esc_node_cid.get(_n)
            if _cid:
                _esc_cid_occ[_cid] = _esc_cid_occ.get(_cid, 0) + 1

        for aid in list(active):
            p = paths[aid]
            st = state[aid]
            a = agents_by_id[aid]
            idx = st["path_idx"]
            cur = p[idx]

            if idx >= len(p) - 1:
                st["arrive_time"] = t
                arrived.add(aid)
                active.remove(aid)
                if occupancy.get(cur) == aid:
                    del occupancy[cur]
                continue

            nxt = p[idx + 1]

            # Capacity gate: stairs
            if nxt in stair_nodes and cur not in stair_nodes:
                if _stair_occ >= stair_cap:
                    st["wait_time"] += dt
                    st["total_wait"] += dt
                    queue_near_stairs += 1
                    continue

            # Capacity gate: escalators (per-unit, keyed by connector_id)
            if nxt in esc_nodes and cur not in esc_nodes:
                _nxt_cid = esc_node_cid.get(nxt, "")
                if _esc_cid_occ.get(_nxt_cid, 0) >= esc_cap:
                    st["wait_time"] += dt
                    st["total_wait"] += dt
                    continue

            # Speed-based movement probability
            speed = a.get("speed_factor", 1.0)
            move_prob = min(1.0, speed)

            if nxt not in occupancy:
                if rng.random() < move_prob:
                    moves.append((aid, cur, nxt))
                else:
                    st["wait_time"] += dt
                    st["total_wait"] += dt
            else:
                st["wait_time"] += dt
                st["total_wait"] += dt
                # Deadlock breaker
                if rng.random() < 0.15:
                    moves.append((aid, cur, nxt))
                if cur in stair_nodes or nxt in stair_nodes:
                    queue_near_stairs += 1

        # Execute moves
        for aid, cur, nxt in moves:
            if occupancy.get(cur) == aid:
                del occupancy[cur]
            occupancy[nxt] = aid
            state[aid]["path_idx"] += 1
            state[aid]["wait_time"] = 0.0
            edge_throughput[tuple(sorted((cur, nxt)))] += 1

        stair_queue_over_time.append((t, queue_near_stairs))

        # Write trajectories
        if traj_fp is not None:
            for nid, aid in occupancy.items():
                nd = g.nodes[nid]
                traj_fp.write(json.dumps({
                    "t": t, "agent_id": aid, "node_id": nid,
                    "x": nd["x"], "y": nd["y"], "z": nd["z"],
                }, ensure_ascii=False) + "\n")

    if traj_fp is not None:
        traj_fp.close()

    # --- Statistics ---
    travel_times, wait_times = [], []
    elderly_tt, normal_tt = [], []
    elderly_wt, normal_wt = [], []

    for aid, a in agents_by_id.items():
        st = state[aid]
        w = st["total_wait"]
        wait_times.append(w)
        if a.get("agent_type") == "elderly":
            elderly_wt.append(w)
        else:
            normal_wt.append(w)
        if st["arrive_time"] is not None:
            tt = st["arrive_time"] - a["spawn_time"]
            travel_times.append(tt)
            if a.get("agent_type") == "elderly":
                elderly_tt.append(tt)
            else:
                normal_tt.append(tt)

    n_agents = len(agents_by_id)
    n_arrived = sum(1 for s in state.values() if s["arrive_time"] is not None)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "routing_mode", "arrive_rate", "mean_travel_time",
            "mean_travel_elderly", "mean_travel_normal",
            "mean_wait_time", "max_queue", "n_agents", "total_replans",
        ])
        w.writeheader()
        w.writerow({
            "label": label,
            "routing_mode": routing_mode,
            "arrive_rate": n_arrived / n_agents if n_agents else 0,
            "mean_travel_time": sum(travel_times) / len(travel_times) if travel_times else 0,
            "mean_travel_elderly": sum(elderly_tt) / len(elderly_tt) if elderly_tt else 0,
            "mean_travel_normal": sum(normal_tt) / len(normal_tt) if normal_tt else 0,
            "mean_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0,
            "max_queue": max((q for _, q in stair_queue_over_time), default=0),
            "n_agents": n_agents,
            "total_replans": sum(s["replan_count"] for s in state.values()),
        })

    result = {
        "travel_times": travel_times,
        "wait_times": wait_times,
        "elderly_travel_times": elderly_tt,
        "normal_travel_times": normal_tt,
        "elderly_wait_times": elderly_wt,
        "normal_wait_times": normal_wt,
        "stair_queue_over_time": stair_queue_over_time,
        "edge_throughput": {f"{u}|{v}": c for (u, v), c in edge_throughput.items()},
        "replan_events": replan_events,
        "routing_mode": routing_mode,
        "arrive_rate": n_arrived / n_agents if n_agents else 0,
        "label": label,
    }

    print(f"[Step 5] Simulation '{label}': {n_arrived}/{n_agents} arrived, "
          f"mean_tt={sum(travel_times) / len(travel_times):.1f}s" if travel_times else "")
    return result
