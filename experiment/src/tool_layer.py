"""
Tool Layer — Python SDK  (Phase 3)
====================================

StationToolLayer wraps the existing World Model (graph, routing, simulation,
evaluation) into eight callable, typed tools.  It owns:
  - Lazy-loading and caching of the navigation graph
  - Semantic identifier resolution (entrance:A, platform, F3_paid …)
  - Thin adapters for all eight core capabilities

Eight tools:
  query_environment()    — level structure, node/edge counts
  query_connectors()     — stair / escalator / elevator / gate status
  query_bottlenecks()    — congestion hot-spots from a simulation result
  plan_route()           — Dijkstra with pluggable strategy
  replan_route()         — congestion-aware replanning
  simulate_scenario()    — ABM run returning key metrics
  compare_strategies()   — two-scenario comparison with deltas
  explain_decision()     — template-driven natural-language explanation
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

from src.tool_schemas import (
    BottleneckEdge,
    BottleneckReport,
    ConnectorQueryResponse,
    ConnectorStatus,
    DataNotReadyError,
    DecisionExplanation,
    EvidenceItem,
    EnvironmentSnapshot,
    InvalidLevelError,
    InvalidNodeError,
    LevelSummary,
    MetricDelta,
    NoPathError,
    RoutePlan,
    RouteSegment,
    ScenarioComparison,
    SimulationResult,
    ToolResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONNECTOR_EDGE_TYPES = {"stair", "escalator", "elevator", "fare_gate"}

_METRIC_LOWER_BETTER = {
    "mean_travel_time",
    "median_travel_time",
    "p95_travel_time",
    "max_travel_time",
    "mean_wait_time",
    "max_queue_near_stairs",
    "mean_elderly_travel",
}

_LEVEL_ALIASES: dict[str, str] = {
    "f1": "F1", "platform": "F1", "站台": "F1",
    "f3": "F3", "concourse": "F3", "站厅": "F3",
    "f4": "F4", "transport": "F4", "交通": "F4",
}


class _WorldModelCache:
    """Holds lazily-loaded shared resources."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._g: Optional[nx.Graph] = None
        self._config: Optional[dict] = None
        self._regions: Optional[dict] = None
        self.graph_hash: str = ""

    # ── graph ──────────────────────────────────────────────────────────────

    def graph(self) -> nx.Graph:
        if self._g is not None:
            return self._g
        path = self.base_dir / "outputs/step3_graph/navigation_graph.gpickle"
        if not path.exists():
            raise DataNotReadyError(
                "navigation_graph.gpickle not found — run scripts/step3_graph.py first"
            )
        with open(path, "rb") as fh:
            raw = fh.read()
        self.graph_hash = hashlib.md5(raw[:8192]).hexdigest()[:8]
        self._g = pickle.loads(raw)
        assert self._g is not None
        # Apply correct escalator directions
        from src.routing import patch_escalator_directions
        patch_escalator_directions(self._g)
        return self._g

    # ── config ─────────────────────────────────────────────────────────────

    def config(self) -> dict:
        if self._config is not None:
            return self._config
        config_path = self.base_dir / "config/experiment_config.yaml"
        from src.utils import load_config
        self._config = load_config(config_path)
        return self._config

    # ── semantic regions ───────────────────────────────────────────────────

    def regions(self) -> dict:
        if self._regions is not None:
            return self._regions
        reg_path = self.base_dir / "outputs/step4_routing/semantic_regions.json"
        if reg_path.exists():
            with open(reg_path, encoding="utf-8") as fh:
                self._regions = json.load(fh)
        else:
            # Compute on demand
            from src.routing import define_semantic_regions
            raw = define_semantic_regions(self.graph(), self.config())
            self._regions = {k: list(v) for k, v in raw.items()}
        assert self._regions is not None
        return self._regions


# ---------------------------------------------------------------------------
# Main SDK class
# ---------------------------------------------------------------------------

class StationToolLayer:
    """Unified tool interface for the metro-station World Model."""

    def __init__(self, base_dir: Optional[str | Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self._cache = _WorldModelCache(Path(base_dir))
        self._last_sim_result: Optional[dict] = None

    # ── Semantic resolver ──────────────────────────────────────────────────

    def _resolve(self, identifier: str) -> str:
        """Map a semantic string or node ID to a concrete graph node ID.

        Supported patterns (case-insensitive):
          <exact node id>        — returned as-is
          entrance:A … entrance:E — any entrance node in that group
          platform / F1          — random PLATFORM region node (door_platform)
          concourse / F3         — random floor node on F3
          transport / F4         — random floor node on F4
          region:<REGION_KEY>    — random node from that semantic region

        Graph facts:
          - entrance_group attr = "entrance_A" … "entrance_E"
          - A/B/C on F4, D/E on F3
          - PLATFORM region = door_platform nodes on F1
        """
        g = self._cache.graph()

        # Exact node ID
        if identifier in g:
            return identifier

        raw = identifier.strip().lower()

        # entrance:<label>  →  match entrance_group == "entrance_<label>"
        if raw.startswith("entrance:"):
            label = identifier.split(":", 1)[1].upper()
            group_key = f"entrance_{label}"
            candidates = [
                n for n, d in g.nodes(data=True)
                if d.get("node_type") == "entrance"
                and d.get("entrance_group", "").lower() == group_key.lower()
            ]
            if candidates:
                return random.choice(candidates)
            # Fallback: any entrance node
            any_entrance = [
                n for n, d in g.nodes(data=True)
                if d.get("node_type") == "entrance"
            ]
            if any_entrance:
                return random.choice(any_entrance)
            raise InvalidNodeError(f"entrance:{label}")

        # region:<KEY>
        if raw.startswith("region:"):
            key = identifier.split(":", 1)[1].upper()
            nodes = self._cache.regions().get(key, [])
            if nodes:
                return random.choice(nodes)
            raise InvalidNodeError(f"region:{key}")

        # Semantic shortcuts via regions dict
        regions = self._cache.regions()
        direct_region: dict[str, str] = {
            "platform": "PLATFORM",
            "psd": "PLATFORM",
            "entrance": "ENTRANCE",
            "exit": "EXIT",
        }
        if raw in direct_region:
            nodes = regions.get(direct_region[raw], [])
            if nodes:
                return random.choice(nodes)

        # Level alias → random floor node on that level
        level_key = _LEVEL_ALIASES.get(raw)
        if level_key:
            # Prefer entrance nodes on that level, else any floor node
            for ntype in ("entrance", "floor"):
                candidates = [
                    n for n, d in g.nodes(data=True)
                    if d.get("level") == level_key and d.get("node_type") == ntype
                ]
                if candidates:
                    return random.choice(candidates)

        # Semantic vector fallback — find closest node by description similarity
        try:
            from src.node_vector_index import NodeVectorIndex
            idx = NodeVectorIndex.get_instance(g)
            candidates = idx.search(identifier, k=3)
            if candidates:
                return candidates[0]
        except Exception:
            pass  # index not built yet or module unavailable

        raise InvalidNodeError(identifier)

    # ──────────────────────────────────────────────────────────────────────
    # Tool 1: query_environment
    # ──────────────────────────────────────────────────────────────────────

    def query_environment(self, level: Optional[str] = None) -> EnvironmentSnapshot:
        """Return environment overview: level structure, node/edge counts."""
        g = self._cache.graph()
        cfg = self._cache.config()
        levels_cfg: dict = cfg.get("station", {}).get("levels", {})

        node_types: Counter = Counter()
        edge_types: Counter = Counter()
        level_node_counts: Counter = Counter()
        level_edge_counts: Counter = Counter()
        blind_count = 0

        lvl_filter = level.upper() if level else None

        for _, attr in g.nodes(data=True):
            lv = attr.get("level", "")
            if lvl_filter and lv != lvl_filter:
                continue
            node_types[attr.get("node_type", "unknown")] += 1
            level_node_counts[lv] += 1
            if attr.get("is_blind_path"):
                blind_count += 1

        for u, _, attr in g.edges(data=True):
            u_lv = g.nodes.get(u, {}).get("level", "")
            if lvl_filter and u_lv != lvl_filter:
                continue
            edge_types[attr.get("edge_type", "unknown")] += 1
            level_edge_counts[u_lv] += 1

        summaries: list[LevelSummary] = []
        for lk, lcfg in levels_cfg.items():
            if lvl_filter and lk != lvl_filter:
                continue
            if not lcfg.get("is_walkable", False):
                continue
            summaries.append(LevelSummary(
                level=lk,
                name_en=lcfg.get("name_en", lk),
                elevation_m=lcfg.get("elevation_m", 0.0),
                n_nodes=level_node_counts.get(lk, 0),
                n_edges=level_edge_counts.get(lk, 0),
                is_walkable=True,
                role=lcfg.get("role", ""),
            ))

        total_n = level_node_counts.get(lvl_filter, g.number_of_nodes()) if lvl_filter else g.number_of_nodes()
        total_e = sum(level_edge_counts[lk] for lk in (level_edge_counts if not lvl_filter else [lvl_filter]))

        return EnvironmentSnapshot(
            ok=True,
            graph_hash=self._cache.graph_hash,
            levels=summaries,
            total_nodes=total_n,
            total_edges=total_e,
            blind_path_nodes=blind_count,
            edge_type_counts=dict(edge_types),
            node_type_counts=dict(node_types),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 2: query_connectors
    # ──────────────────────────────────────────────────────────────────────

    def query_connectors(
        self,
        connector_type: Optional[str] = None,
        level: Optional[str] = None,
    ) -> ConnectorQueryResponse:
        """Return status of vertical/control connectors in the graph."""
        g = self._cache.graph()
        cfg = self._cache.config()
        conn_cfg: dict = cfg.get("connectors", {})

        target_types = _CONNECTOR_EDGE_TYPES
        if connector_type:
            ct = connector_type.lower()
            # Normalise aliases
            ct = {"gate": "fare_gate", "gates": "fare_gate",
                  "stairs": "stair", "escalators": "escalator",
                  "elevators": "elevator"}.get(ct, ct)
            target_types = {ct}

        lvl_filter = level.upper() if level else None

        seen: dict[str, ConnectorStatus] = {}
        for u, v, attr in g.edges(data=True):
            et = attr.get("edge_type", "")
            if et not in target_types:
                continue
            cid = attr.get("connector_id") or f"__auto_{u}_{v}"

            u_attr = g.nodes.get(u, {})
            v_attr = g.nodes.get(v, {})
            u_lv = u_attr.get("level", "")
            v_lv = v_attr.get("level", "")

            if lvl_filter and lvl_filter not in (u_lv, v_lv):
                continue

            if cid in seen:
                continue

            # Capacity from edge or config fallback
            cap: int = int(attr.get("capacity") or 0)
            if cap == 0:
                if et == "stair":
                    cap = int(conn_cfg.get("stair", {}).get("capacity", 4))
                elif et == "escalator":
                    cap = int(conn_cfg.get("escalator", {}).get("capacity", 8))
                elif et == "elevator":
                    cap = int(conn_cfg.get("elevator", {}).get("capacity_batch", 20))
                elif et == "fare_gate":
                    cap = int(conn_cfg.get("fare_gate", {}).get("queue_capacity", 15))

            direction = str(attr.get("direction", "bidirectional"))
            state = str(attr.get("state", "open"))

            ax = u_attr.get("x")
            ay = u_attr.get("y")
            bx = v_attr.get("x")
            by_ = v_attr.get("y")

            seen[cid] = ConnectorStatus(
                connector_id=cid,
                connector_type=et,
                from_level=u_lv,
                to_level=v_lv,
                direction=direction,
                capacity=cap,
                state=state,
                anchor_from=(ax, ay) if ax is not None else None,
                anchor_to=(bx, by_) if bx is not None else None,
            )

        connectors = list(seen.values())
        return ConnectorQueryResponse(
            ok=True,
            graph_hash=self._cache.graph_hash,
            connectors=connectors,
            total=len(connectors),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 3: query_bottlenecks
    # ──────────────────────────────────────────────────────────────────────

    def query_bottlenecks(
        self,
        sim_result: Optional[dict] = None,
        percentile: int = 90,
    ) -> BottleneckReport:
        """Identify high-throughput / congested edges from a simulation result.

        If no result is provided, falls back to graph degree-centrality of
        connector nodes (topology-only estimate).
        """
        g = self._cache.graph()
        result = sim_result or self._last_sim_result

        if result is None:
            # Topology fallback: report connectors with highest degree
            top: list[BottleneckEdge] = []
            for u, v, attr in g.edges(data=True):
                if attr.get("edge_type") in _CONNECTOR_EDGE_TYPES:
                    top.append(BottleneckEdge(
                        u=u, v=v,
                        edge_type=attr.get("edge_type", ""),
                        throughput=0,
                        congestion_score=float(g.degree(u) + g.degree(v)),
                        level=g.nodes.get(u, {}).get("level", ""),
                        connector_id=attr.get("connector_id"),
                    ))
            top.sort(key=lambda x: -x.congestion_score)
            return BottleneckReport(
                ok=True,
                graph_hash=self._cache.graph_hash,
                top_bottlenecks=top[:10],
                threshold_percentile=percentile,
                total_edges_analysed=g.number_of_edges(),
                sim_result_id="topology_estimate",
            )

        throughput: dict[str, int] = result.get("edge_throughput", {})
        if not throughput:
            return BottleneckReport(
                ok=True,
                graph_hash=self._cache.graph_hash,
                sim_result_id=result.get("label", ""),
            )

        values = sorted(throughput.values())
        threshold = values[max(0, int(len(values) * percentile / 100) - 1)]
        max_val = values[-1] if values else 1

        hot: list[BottleneckEdge] = []
        for edge_key, count in throughput.items():
            if count < threshold:
                continue
            parts = edge_key.split("|", 1)
            if len(parts) != 2:
                continue
            u, v = parts
            edata = g.get_edge_data(u, v) or {}
            u_attr = g.nodes.get(u, {})
            hot.append(BottleneckEdge(
                u=u, v=v,
                edge_type=edata.get("edge_type", "unknown"),
                throughput=count,
                congestion_score=count / max_val,
                level=u_attr.get("level", ""),
                connector_id=edata.get("connector_id"),
            ))

        hot.sort(key=lambda x: -x.throughput)
        sq = result.get("stair_queue_over_time", [])
        max_q = max((q for _, q in sq), default=0.0)

        return BottleneckReport(
            ok=True,
            graph_hash=self._cache.graph_hash,
            top_bottlenecks=hot[:10],
            threshold_percentile=percentile,
            total_edges_analysed=len(throughput),
            max_queue_near_stairs=float(max_q),
            sim_result_id=result.get("label", ""),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 4: plan_route
    # ──────────────────────────────────────────────────────────────────────

    def plan_route(
        self,
        origin: str,
        destination: str,
        strategy: str = "directed",
    ) -> RoutePlan:
        """Plan a route using one of four strategies.

        strategy options:
          "directed"   — direction-enforcing (escalator / gate one-way rules)
          "static"     — pure travel-time shortest path
          "penalised"  — adds time penalties for slower connectors (stairs, etc.)
          "accessible" — alias for "penalised"; avoids escalators
        """
        g = self._cache.graph()
        cfg = self._cache.config()

        try:
            o_node = self._resolve(origin)
            d_node = self._resolve(destination)
        except InvalidNodeError as exc:
            return RoutePlan(
                ok=False, error=str(exc), error_code=exc.code,
                graph_hash=self._cache.graph_hash,
                origin=origin, destination=destination, strategy=strategy,
            )

        from src.routing import (
            directed_weight, penalised_weight, static_weight,
            find_path_with_cost, path_summary,
        )

        strat = strategy.lower()
        if strat in ("directed", "default"):
            wfn = directed_weight(g)
        elif strat in ("penalised", "accessible", "elderly"):
            wfn = penalised_weight(cfg)
        else:
            wfn = static_weight

        path, cost = find_path_with_cost(g, o_node, d_node, wfn)
        if cost == float("inf") or len(path) <= 1:
            return RoutePlan(
                ok=False, error=f"No path from '{origin}' to '{destination}'",
                error_code="no_path",
                graph_hash=self._cache.graph_hash,
                origin=origin, destination=destination, strategy=strategy,
            )

        summary = path_summary(g, path)

        segments: list[RouteSegment] = []
        for a, b in zip(path[:-1], path[1:]):
            edata = g.get_edge_data(a, b) or {}
            a_attr = g.nodes.get(a, {})
            segments.append(RouteSegment(
                from_node=a,
                to_node=b,
                edge_type=edata.get("edge_type", "floor"),
                distance_m=float(edata.get("length_3d") or edata.get("length_2d") or 0.0),
                travel_time_s=float(edata.get("travel_time") or 0.0),
                level=a_attr.get("level", ""),
                connector_id=edata.get("connector_id"),
                direction=edata.get("direction"),
            ))

        has_escalator = any(s.edge_type == "escalator" for s in segments)

        return RoutePlan(
            ok=True,
            graph_hash=self._cache.graph_hash,
            origin=origin,
            destination=destination,
            strategy=strategy,
            path=path,
            segments=segments,
            total_distance_m=round(summary["total_length_3d"], 2),
            total_travel_time_s=round(summary["total_travel_time"], 1),
            levels_traversed=summary["levels_visited"],
            connectors_used=summary["connectors_used"],
            is_accessible=not has_escalator,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 5: replan_route
    # ──────────────────────────────────────────────────────────────────────

    def replan_route(
        self,
        origin: str,
        destination: str,
        occupancy: Optional[dict[str, str]] = None,
        alpha: float = 3.0,
    ) -> RoutePlan:
        """Congestion-aware replanning.

        occupancy: {node_id: agent_id} mapping of current agent positions.
        If None, falls back to plan_route("directed").
        """
        g = self._cache.graph()

        if not occupancy:
            return self.plan_route(origin, destination, strategy="directed")

        from src.routing import (
            compute_congestion, congestion_directed_weight,
            find_path_with_cost, path_summary,
            directed_weight,
        )

        try:
            o_node = self._resolve(origin)
            d_node = self._resolve(destination)
        except InvalidNodeError as exc:
            return RoutePlan(
                ok=False, error=str(exc), error_code=exc.code,
                graph_hash=self._cache.graph_hash,
                origin=origin, destination=destination, strategy="dynamic",
            )

        edge_cong = compute_congestion(g, occupancy)
        try:
            wfn = congestion_directed_weight(g, edge_cong, alpha=alpha)
        except Exception:
            wfn = directed_weight(g)

        path, cost = find_path_with_cost(g, o_node, d_node, wfn)
        if cost == float("inf") or len(path) <= 1:
            # Fallback to static
            from src.routing import static_weight
            path, cost = find_path_with_cost(g, o_node, d_node, static_weight)

        if cost == float("inf") or len(path) <= 1:
            return RoutePlan(
                ok=False, error=f"No path from '{origin}' to '{destination}'",
                error_code="no_path",
                graph_hash=self._cache.graph_hash,
                origin=origin, destination=destination, strategy="dynamic",
            )

        summary = path_summary(g, path)
        segments: list[RouteSegment] = []
        for a, b in zip(path[:-1], path[1:]):
            edata = g.get_edge_data(a, b) or {}
            a_attr = g.nodes.get(a, {})
            segments.append(RouteSegment(
                from_node=a, to_node=b,
                edge_type=edata.get("edge_type", "floor"),
                distance_m=float(edata.get("length_3d") or edata.get("length_2d") or 0.0),
                travel_time_s=float(edata.get("travel_time") or 0.0),
                level=a_attr.get("level", ""),
                connector_id=edata.get("connector_id"),
                direction=edata.get("direction"),
            ))

        return RoutePlan(
            ok=True,
            graph_hash=self._cache.graph_hash,
            origin=origin, destination=destination,
            strategy="dynamic",
            path=path, segments=segments,
            total_distance_m=round(summary["total_length_3d"], 2),
            total_travel_time_s=round(summary["total_travel_time"], 1),
            levels_traversed=summary["levels_visited"],
            connectors_used=summary["connectors_used"],
            is_accessible=not any(s.edge_type == "escalator" for s in segments),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 6: simulate_scenario
    # ──────────────────────────────────────────────────────────────────────

    def simulate_scenario(
        self,
        n_agents: int = 200,
        routing_mode: str = "static",
        label: str = "",
        flows: Optional[dict] = None,
        elderly_ratio: float = 0.1,
        seed: int = 42,
    ) -> SimulationResult:
        """Run an ABM scenario and return summary metrics."""
        g = self._cache.graph()
        cfg = self._cache.config()
        regions = self._cache.regions()

        sim_cfg = dict(cfg.get("simulation", {}))
        sim_cfg["seed"] = seed

        if flows is None:
            flows = sim_cfg.get("default_flows") or {
                "ENTRANCE->PLATFORM": 0.7,
                "PLATFORM->EXIT": 0.3,
            }

        try:
            from src.routing import sample_agents
            agents = sample_agents(
                regions=regions,
                flows=flows,
                n_agents=n_agents,
                T=sim_cfg.get("T_s", 300),
                seed=seed,
                walking_speed=sim_cfg.get("walking_speed_ms", 1.2),
                elderly_ratio=elderly_ratio,
            )
        except Exception as exc:
            return SimulationResult(
                ok=False, error=f"Agent generation failed: {exc}",
                error_code="agent_gen_failed",
                graph_hash=self._cache.graph_hash,
            )

        run_label = label or routing_mode
        out_dir = (
            self._cache.base_dir
            / "outputs/step5_simulation/agent_tool"
            / run_label
        )

        try:
            from src.simulation import run_simulation
            from typing import cast as _cast, Literal as _Literal
            _mode = _cast(_Literal["static", "dynamic"], routing_mode)
            result = run_simulation(g, agents, cfg, out_dir, _mode, run_label)
        except Exception as exc:
            return SimulationResult(
                ok=False, error=f"Simulation failed: {exc}",
                error_code="sim_failed",
                graph_hash=self._cache.graph_hash,
            )

        self._last_sim_result = result

        from src.evaluation import compute_scenario_metrics
        m = compute_scenario_metrics(result)

        sim_id = hashlib.md5(
            f"{run_label}{n_agents}{routing_mode}{time.time()}".encode()
        ).hexdigest()[:8]

        return SimulationResult(
            ok=True,
            graph_hash=self._cache.graph_hash,
            label=run_label,
            routing_mode=routing_mode,
            sim_result_id=sim_id,
            out_dir=str(out_dir),
            n_agents=m["n_agents"],
            n_arrived=m["n_arrived"],
            arrive_rate=m["arrive_rate"],
            mean_travel_time_s=round(m["mean_travel_time"], 2),
            median_travel_time_s=round(m["median_travel_time"], 2),
            p95_travel_time_s=round(m["p95_travel_time"], 2),
            max_travel_time_s=round(m["max_travel_time"], 2),
            mean_wait_time_s=round(m["mean_wait_time"], 2),
            max_queue_near_stairs=m["max_queue_near_stairs"],
            mean_elderly_travel_s=round(m["mean_elderly_travel"], 2),
            mean_normal_travel_s=round(m.get("mean_normal_travel", 0.0), 2),
            n_elderly=m["n_elderly"],
            n_normal=m.get("n_normal", n_agents - m["n_elderly"]),
            total_replans=m["total_replans"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 7: compare_strategies
    # ──────────────────────────────────────────────────────────────────────

    def compare_strategies(
        self,
        scenario_a: Optional[SimulationResult] = None,
        scenario_b: Optional[SimulationResult] = None,
        n_agents: int = 200,
    ) -> ScenarioComparison:
        """Compare two routing strategies (defaults: static vs. dynamic).

        If pre-run SimulationResult objects are provided they are used directly;
        otherwise fresh simulations are run.
        """

        def _to_metrics(r: SimulationResult) -> dict:
            return {
                "label": r.label,
                "routing_mode": r.routing_mode,
                "n_agents": r.n_agents,
                "n_arrived": r.n_arrived,
                "arrive_rate": r.arrive_rate,
                "mean_travel_time": r.mean_travel_time_s,
                "median_travel_time": r.median_travel_time_s,
                "p95_travel_time": r.p95_travel_time_s,
                "max_travel_time": r.max_travel_time_s,
                "mean_wait_time": r.mean_wait_time_s,
                "max_queue_near_stairs": r.max_queue_near_stairs,
                "mean_elderly_travel": r.mean_elderly_travel_s,
                "total_replans": r.total_replans,
            }

        if scenario_a is None:
            scenario_a = self.simulate_scenario(
                n_agents=n_agents, routing_mode="static", label="static"
            )
        if not scenario_a.ok:
            return ScenarioComparison(
                ok=False, error=scenario_a.error, error_code=scenario_a.error_code,
                graph_hash=self._cache.graph_hash,
            )

        if scenario_b is None:
            scenario_b = self.simulate_scenario(
                n_agents=n_agents, routing_mode="dynamic", label="dynamic"
            )
        if not scenario_b.ok:
            return ScenarioComparison(
                ok=False, error=scenario_b.error, error_code=scenario_b.error_code,
                graph_hash=self._cache.graph_hash,
            )

        ma = _to_metrics(scenario_a)
        mb = _to_metrics(scenario_b)

        from src.evaluation import compare_scenarios
        raw = compare_scenarios([ma, mb])
        comps = raw.get("comparisons", [{}])
        comp = comps[0] if comps else {}

        _compare_keys = [
            "mean_travel_time", "p95_travel_time", "arrive_rate",
            "max_queue_near_stairs", "mean_wait_time", "mean_elderly_travel",
        ]
        deltas: list[MetricDelta] = []
        for key in _compare_keys:
            bv = comp.get(f"{key}_baseline", 0.0)
            sv = comp.get(f"{key}_scenario", 0.0)
            delta = comp.get(f"{key}_delta", 0.0)
            pct = comp.get(f"{key}_pct_change", 0.0)
            lower_is_better = key in _METRIC_LOWER_BETTER
            better = (delta < 0) if lower_is_better else (delta > 0)
            deltas.append(MetricDelta(
                metric=key,
                baseline_value=round(bv, 2),
                scenario_value=round(sv, 2),
                delta=round(delta, 2),
                pct_change=round(pct, 1),
                better=better,
            ))

        # Auto-generate summary sentence
        tt_delta = next((d for d in deltas if d.metric == "mean_travel_time"), None)
        if tt_delta:
            direction = "improves" if tt_delta.better else "worsens"
            summary = (
                f"'{scenario_b.label}' {direction} mean travel time by "
                f"{abs(tt_delta.pct_change):.1f}% vs '{scenario_a.label}' "
                f"({tt_delta.baseline_value:.1f}s → {tt_delta.scenario_value:.1f}s)."
            )
        else:
            summary = (
                f"'{scenario_b.label}' vs '{scenario_a.label}': "
                f"arrive rate {mb['arrive_rate']:.1%} vs {ma['arrive_rate']:.1%}."
            )

        return ScenarioComparison(
            ok=True,
            graph_hash=self._cache.graph_hash,
            baseline_label=scenario_a.label,
            scenario_label=scenario_b.label,
            deltas=deltas,
            summary_sentence=summary,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tool 8: explain_decision
    # ──────────────────────────────────────────────────────────────────────

    def explain_decision(
        self,
        route: Optional[RoutePlan] = None,
        comparison: Optional[ScenarioComparison] = None,
        context: str = "",
    ) -> DecisionExplanation:
        """Generate a template-driven natural-language explanation.

        Accepts a RoutePlan, a ScenarioComparison, or both.
        Returns structured reasoning steps + evidence + recommendation.
        """
        steps: list[str] = []
        evidence: list[EvidenceItem] = []
        conclusion = ""
        recommendation = ""

        if context:
            steps.append(f"Context: {context}")

        if route and route.ok:
            steps.append(
                f"Route analysed: {route.origin} → {route.destination} "
                f"(strategy: {route.strategy})"
            )
            steps.append(
                f"Distance: {route.total_distance_m:.0f} m  |  "
                f"Travel time: {route.total_travel_time_s:.0f} s"
            )
            levels = " → ".join(route.levels_traversed)
            steps.append(f"Levels traversed: {levels}")
            if route.connectors_used:
                conns = ", ".join(route.connectors_used)
                steps.append(f"Connectors used: {conns}")

            evidence += [
                EvidenceItem("route", "total_distance_m",
                             round(route.total_distance_m, 1), "m"),
                EvidenceItem("route", "total_travel_time_s",
                             round(route.total_travel_time_s, 0), "s"),
                EvidenceItem("route", "n_path_nodes", len(route.path), "nodes"),
            ]
            accessibility = (
                "accessible (no escalators)" if route.is_accessible
                else "includes escalators"
            )
            conclusion = (
                f"Route from {route.origin} to {route.destination}: "
                f"{route.total_distance_m:.0f} m, "
                f"{route.total_travel_time_s:.0f} s, {accessibility}."
            )
            if route.strategy in ("directed", "static"):
                recommendation = (
                    "This is the default time-optimal path with direction constraints enforced."
                )
            elif route.strategy in ("penalised", "accessible", "elderly"):
                recommendation = (
                    "This route applies penalties to stairs and escalators — "
                    "preferred for elderly or mobility-impaired passengers."
                )
            elif route.strategy == "dynamic":
                recommendation = (
                    "This route was computed with live congestion avoidance — "
                    "recalculate regularly as conditions change."
                )

        if comparison and comparison.ok:
            steps.append(
                f"Strategies compared: '{comparison.baseline_label}' vs "
                f"'{comparison.scenario_label}'"
            )
            improved = [d for d in comparison.deltas if d.better]
            worsened = [d for d in comparison.deltas if not d.better]
            if improved:
                names = ", ".join(d.metric.replace("_", " ") for d in improved[:3])
                steps.append(f"  Improvements in: {names}")
            if worsened:
                names = ", ".join(d.metric.replace("_", " ") for d in worsened[:2])
                steps.append(f"  Trade-offs in: {names}")

            for d in comparison.deltas[:4]:
                evidence.append(EvidenceItem(
                    "comparison", d.metric,
                    [d.baseline_value, d.scenario_value],
                    "s" if "time" in d.metric else "",
                ))

            conclusion = comparison.summary_sentence or conclusion
            recommendation = (
                f"Use '{comparison.scenario_label}' when "
                + (
                    "congestion is high and dynamic replanning is feasible."
                    if "dynamic" in comparison.scenario_label
                    else "the scenario conditions match operational needs."
                )
            )

        if not conclusion:
            conclusion = "No specific route or comparison was provided for explanation."
            recommendation = "Provide a RoutePlan or ScenarioComparison for a detailed analysis."

        return DecisionExplanation(
            ok=True,
            graph_hash=self._cache.graph_hash,
            conclusion=conclusion,
            reasoning_steps=steps,
            evidence=evidence,
            recommendation=recommendation,
        )
