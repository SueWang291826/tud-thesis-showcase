"""
Tool Layer — Unified Data Schemas  (Phase 2)
=============================================

Defines the request / response contracts shared by the Python SDK
(tool_layer.py), the REST API (tool_api.py), and the Agent orchestrator
(agent_orchestrator.py).

All response objects inherit from ToolResponse and carry:
  schema_version  — contract version for forward-compatibility
  graph_hash      — short MD5 of the loaded graph file (reproducibility)
  ok / error      — standard success / failure envelope
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_SCHEMA_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ToolError(Exception):
    """Base class for tool-layer errors."""
    code: str = "tool_error"

    def __init__(self, message: str, code: str = "tool_error"):
        super().__init__(message)
        self.code = code


class NoPathError(ToolError):
    def __init__(self, origin: str, dest: str):
        super().__init__(
            f"No navigable path from '{origin}' to '{dest}'",
            "no_path",
        )


class InvalidNodeError(ToolError):
    def __init__(self, node_id: str):
        super().__init__(f"Node not found in graph: '{node_id}'", "invalid_node")


class InvalidLevelError(ToolError):
    def __init__(self, level: str):
        super().__init__(f"Level not recognised: '{level}'", "invalid_level")


class DataNotReadyError(ToolError):
    def __init__(self, what: str = ""):
        msg = f"Required data not ready{': ' + what if what else ''}"
        super().__init__(msg, "data_not_ready")


class SimulationTimeoutError(ToolError):
    def __init__(self):
        super().__init__("Simulation exceeded the allowed time budget", "sim_timeout")


# ─────────────────────────────────────────────────────────────────────────────
# Base response envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResponse:
    ok: bool = True
    error: Optional[str] = None
    error_code: Optional[str] = None
    schema_version: str = TOOL_SCHEMA_VERSION
    graph_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_error(cls, exc: Exception) -> "ToolResponse":
        code = getattr(exc, "code", "tool_error")
        return cls(ok=False, error=str(exc), error_code=code)


# ─────────────────────────────────────────────────────────────────────────────
# query_environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelSummary:
    level: str
    name_en: str
    elevation_m: float
    n_nodes: int
    n_edges: int
    is_walkable: bool
    role: str


@dataclass
class EnvironmentSnapshot(ToolResponse):
    levels: list = field(default_factory=list)       # list[LevelSummary]
    total_nodes: int = 0
    total_edges: int = 0
    blind_path_nodes: int = 0
    edge_type_counts: dict = field(default_factory=dict)
    node_type_counts: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# query_connectors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConnectorStatus:
    connector_id: str
    connector_type: str        # stair | escalator | elevator | fare_gate
    from_level: str
    to_level: str
    direction: str             # up | down | bidirectional | inbound | outbound
    capacity: int
    state: str                 # open | closed
    anchor_from: Optional[tuple] = None
    anchor_to: Optional[tuple] = None


@dataclass
class ConnectorQueryResponse(ToolResponse):
    connectors: list = field(default_factory=list)   # list[ConnectorStatus]
    total: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# query_bottlenecks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckEdge:
    u: str
    v: str
    edge_type: str
    throughput: int
    congestion_score: float
    level: str
    connector_id: Optional[str] = None


@dataclass
class BottleneckReport(ToolResponse):
    top_bottlenecks: list = field(default_factory=list)   # list[BottleneckEdge]
    threshold_percentile: int = 90
    total_edges_analysed: int = 0
    max_queue_near_stairs: float = 0.0
    sim_result_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# plan_route / replan_route
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteSegment:
    from_node: str
    to_node: str
    edge_type: str
    distance_m: float
    travel_time_s: float
    level: str
    connector_id: Optional[str] = None
    direction: Optional[str] = None


@dataclass
class RoutePlan(ToolResponse):
    origin: str = ""
    destination: str = ""
    strategy: str = "directed"
    path: list = field(default_factory=list)          # list[str] node IDs
    segments: list = field(default_factory=list)      # list[RouteSegment]
    total_distance_m: float = 0.0
    total_travel_time_s: float = 0.0
    levels_traversed: list = field(default_factory=list)
    connectors_used: list = field(default_factory=list)
    is_accessible: bool = True                        # True when no escalators


# ─────────────────────────────────────────────────────────────────────────────
# simulate_scenario
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult(ToolResponse):
    label: str = ""
    routing_mode: str = "static"
    n_agents: int = 0
    n_arrived: int = 0
    arrive_rate: float = 0.0
    mean_travel_time_s: float = 0.0
    median_travel_time_s: float = 0.0
    p95_travel_time_s: float = 0.0
    max_travel_time_s: float = 0.0
    mean_wait_time_s: float = 0.0
    max_queue_near_stairs: float = 0.0
    mean_elderly_travel_s: float = 0.0
    mean_normal_travel_s: float = 0.0
    n_elderly: int = 0
    n_normal: int = 0
    total_replans: int = 0
    sim_result_id: Optional[str] = None
    out_dir: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# compare_strategies
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricDelta:
    metric: str
    baseline_value: float
    scenario_value: float
    delta: float
    pct_change: float
    better: bool     # True when scenario is an improvement for this metric


@dataclass
class ScenarioComparison(ToolResponse):
    baseline_label: str = ""
    scenario_label: str = ""
    deltas: list = field(default_factory=list)    # list[MetricDelta]
    summary_sentence: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# explain_decision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    kind: str       # "metric" | "route" | "comparison"
    label: str
    value: Any
    unit: str = ""


@dataclass
class DecisionExplanation(ToolResponse):
    conclusion: str = ""
    reasoning_steps: list = field(default_factory=list)   # list[str]
    evidence: list = field(default_factory=list)          # list[EvidenceItem]
    recommendation: str = ""
