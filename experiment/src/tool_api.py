"""
Tool API — FastAPI REST Layer  (Phase 4)
==========================================

Exposes all 8 tool methods as JSON endpoints, plus:
  POST /api/chat          — natural language → Agent Orchestrator → tool calls
  GET  /                  — serve Web Chat UI (web/index.html)

Endpoints:
  POST /tools/query_environment
  POST /tools/query_connectors
  POST /tools/query_bottlenecks
  POST /tools/plan_route
  POST /tools/replan_route
  POST /tools/simulate_scenario
  POST /tools/compare_strategies
  POST /tools/explain_decision

  POST /api/chat
  GET  /
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Deferred imports (allows uvicorn reload without heavy startup cost)
_tool_layer: Any = None
_node_data_cache: Any = None

BASE_DIR = Path(__file__).parent.parent  # experiment/


def _get_tool_layer() -> Any:
    global _tool_layer
    if _tool_layer is None:
        from src.tool_layer import StationToolLayer  # type: ignore[import]
        _tool_layer = StationToolLayer(base_dir=BASE_DIR)
    return _tool_layer


def _get_node_data() -> dict:
    """Build (once) node scatter data for the 3-D base map."""
    global _node_data_cache
    if _node_data_cache is not None:
        return _node_data_cache

    from collections import defaultdict

    tl = _get_tool_layer()
    g = tl._cache.graph()
    cfg = tl._cache.config()
    elevations: dict[str, float] = {
        k: float(v.get("elevation_m", 0.0))
        for k, v in cfg.get("station", {}).get("levels", {}).items()
    }
    Z_SCALE = 3.0
    LEVEL_COLORS = {"F1": "#2196F3", "F3": "#4CAF50", "F4": "#FF9800"}
    LEVEL_NAMES = {
        "F1": "Platform (F1)", "F3": "Concourse (F3)", "F4": "Transport (F4)"
    }

    floor_x: dict = defaultdict(list)
    floor_y: dict = defaultdict(list)
    floor_z: dict = defaultdict(list)
    ent_x, ent_y, ent_z, ent_lbl = [], [], [], []
    psd_x, psd_y, psd_z = [], [], []
    # Connector representative points (one per connector_id)
    conn_seen: set = set()
    stair_x, stair_y, stair_z, stair_lbl = [], [], [], []
    esc_x, esc_y, esc_z, esc_lbl = [], [], [], []
    elev_x, elev_y, elev_z, elev_lbl = [], [], [], []

    # Elevation midpoints for connectors spanning levels
    elev_f1 = elevations.get("F1", 0.0)
    elev_f3 = elevations.get("F3", 12.1)
    elev_f4 = elevations.get("F4", 17.4)
    _stair_z_mid = round((elev_f1 + elev_f3) / 2 * Z_SCALE, 2)
    _esc_z_mid   = round((elev_f1 + elev_f3) / 2 * Z_SCALE, 2)

    for nid, nd in g.nodes(data=True):
        nx_ = nd.get("x")
        ny_ = nd.get("y")
        if nx_ is None or ny_ is None:
            continue
        lv = nd.get("level", "")
        nt = nd.get("node_type", "floor")
        ct = nd.get("connector_type", "")
        z = round(elevations.get(lv, 0.0) * Z_SCALE, 2)
        fx, fy = round(float(nx_), 2), round(float(ny_), 2)

        if nt == "entrance":
            ent_x.append(fx); ent_y.append(fy); ent_z.append(z)
            grp = nd.get("entrance_group", nid)
            ent_lbl.append(grp.replace("entrance_", "Gate ").upper())
        elif nt == "door_platform":
            psd_x.append(fx); psd_y.append(fy); psd_z.append(z)
        elif ct == "stair" or nt == "stair_step":
            cid = nd.get("connector_id", str(nid))
            if cid not in conn_seen:
                conn_seen.add(cid)
                stair_x.append(fx); stair_y.append(fy)
                stair_z.append(_stair_z_mid)
                stair_lbl.append(f"Stair {cid[-8:]}")
        elif ct == "escalator" or nt == "escalator":
            cid = nd.get("connector_id", str(nid))
            if cid not in conn_seen:
                conn_seen.add(cid)
                esc_x.append(fx); esc_y.append(fy)
                esc_z.append(round(elevations.get(lv, elev_f1) * Z_SCALE, 2))
                esc_lbl.append(f"Escalator {cid[-8:]}")
        elif nt in ("elevator_entry", "elevator_interior"):
            cid = nd.get("connector_id", str(nid))
            if cid not in conn_seen:
                conn_seen.add(cid)
                elev_x.append(fx); elev_y.append(fy)
                elev_z.append(round(elevations.get(lv, elev_f1) * Z_SCALE, 2))
                elev_lbl.append(f"Elevator {cid[-8:]}")
        elif lv in LEVEL_COLORS:
            floor_x[lv].append(fx)
            floor_y[lv].append(fy)
            floor_z[lv].append(z)

    layers = []
    for lv in ("F4", "F3", "F1"):
        if floor_x[lv]:
            layers.append({
                "name": LEVEL_NAMES.get(lv, lv),
                "color": LEVEL_COLORS[lv],
                "x": floor_x[lv], "y": floor_y[lv], "z": floor_z[lv],
                "symbol": "circle", "size": 1.5, "opacity": 0.3,
            })
    if stair_x:
        layers.append({
            "name": "Stairs",
            "color": "#CE93D8", "x": stair_x, "y": stair_y, "z": stair_z,
            "labels": stair_lbl, "symbol": "square", "size": 6, "opacity": 0.85,
        })
    if esc_x:
        layers.append({
            "name": "Escalators",
            "color": "#80CBC4", "x": esc_x, "y": esc_y, "z": esc_z,
            "labels": esc_lbl, "symbol": "diamond", "size": 6, "opacity": 0.85,
        })
    if elev_x:
        layers.append({
            "name": "Elevators",
            "color": "#FFD54F", "x": elev_x, "y": elev_y, "z": elev_z,
            "labels": elev_lbl, "symbol": "circle", "size": 8, "opacity": 1.0,
        })
    if ent_x:
        layers.append({
            "name": "Entrances",
            "color": "#00BCD4",
            "x": ent_x, "y": ent_y, "z": ent_z,
            "labels": ent_lbl,
            "symbol": "diamond", "size": 8, "opacity": 1.0,
        })
    if psd_x:
        layers.append({
            "name": "Platform Doors",
            "color": "#FF6F00",
            "x": psd_x, "y": psd_y, "z": psd_z,
            "symbol": "square", "size": 4, "opacity": 0.8,
        })

    _node_data_cache = {"elevations": elevations, "z_scale": Z_SCALE, "layers": layers}
    return _node_data_cache


def _route_to_3d(route_result: Any, tl: Any) -> Optional[dict]:
    """Extract 3-D coordinates from a RoutePlan for the map overlay."""
    if not route_result or not getattr(route_result, "ok", False):
        return None
    path: list = getattr(route_result, "path", [])
    if not path:
        return None
    g = tl._cache.graph()
    cfg = tl._cache.config()
    elevations: dict[str, float] = {
        k: float(v.get("elevation_m", 0.0))
        for k, v in cfg.get("station", {}).get("levels", {}).items()
    }
    Z_SCALE = 3.0
    elev_f1 = elevations.get("F1", 0.0)
    elev_f3 = elevations.get("F3", 12.1)
    _stair_z_mid = round((elev_f1 + elev_f3) / 2 * Z_SCALE, 2)

    xs, ys, zs, labels = [], [], [], []
    # Connector waypoints along the route
    conn_wx, conn_wy, conn_wz, conn_wlbl, conn_wclr, conn_wsym = [], [], [], [], [], []
    _CONN_COLORS = {
        "stair": "#CE93D8", "stair_step": "#CE93D8",
        "escalator": "#80CBC4",
        "elevator_entry": "#FFD54F", "elevator_interior": "#FFD54F",
    }
    _CONN_SYM = {
        "stair": "square", "stair_step": "square",
        "escalator": "diamond",
        "elevator_entry": "circle", "elevator_interior": "circle",
    }
    seen_conn = set()

    for nid in path:
        nd = g.nodes.get(nid, {})
        x, y = nd.get("x"), nd.get("y")
        if x is None or y is None:
            continue
        lv = nd.get("level", "F1")
        nt = nd.get("node_type", "floor")
        ct = nd.get("connector_type", "")

        # z for stair nodes (level=STAIR) use midpoint
        if lv == "STAIR" or nt == "stair_step":
            z = _stair_z_mid
        else:
            z = round(elevations.get(lv, 0.0) * Z_SCALE, 2)

        zs.append(z)
        xs.append(round(float(x), 2))
        ys.append(round(float(y), 2))
        labels.append(f"{str(nid)[:24]} ({lv})")

        # Mark connector waypoints (one per connector_id)
        eff_type = ct or nt
        if eff_type in _CONN_COLORS:
            cid = nd.get("connector_id", str(nid))
            if cid not in seen_conn:
                seen_conn.add(cid)
                conn_wx.append(round(float(x), 2))
                conn_wy.append(round(float(y), 2))
                conn_wz.append(z)
                type_name = {"stair": "Stair", "stair_step": "Stair",
                             "escalator": "Escalator",
                             "elevator_entry": "Elevator", "elevator_interior": "Elevator"}.get(eff_type, eff_type)
                conn_wlbl.append(f"{type_name} {cid[-8:]}")
                conn_wclr.append(_CONN_COLORS[eff_type])
                conn_wsym.append(_CONN_SYM[eff_type])

    if not xs:
        return None

    result = {
        "x": xs, "y": ys, "z": zs, "labels": labels,
        "origin": route_result.origin,
        "destination": route_result.destination,
        "strategy": route_result.strategy,
        "total_distance_m": route_result.total_distance_m,
        "total_travel_time_s": route_result.total_travel_time_s,
        "is_accessible": route_result.is_accessible,
    }
    if conn_wx:
        result["connectors"] = {
            "x": conn_wx, "y": conn_wy, "z": conn_wz,
            "labels": conn_wlbl, "colors": conn_wclr, "symbols": conn_wsym,
        }
    return result


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Station Agent Tool API",
    version="1.0.0",
    description="Three-layer Agent Architecture — REST Tool Interface",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files only if the directory exists
_web_dir = BASE_DIR / "agent" / "web"
if _web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class QueryEnvironmentRequest(BaseModel):
    level: Optional[str] = Field(None, json_schema_extra={"example": "F3"})


class QueryConnectorsRequest(BaseModel):
    connector_type: Optional[str] = Field(None, json_schema_extra={"example": "escalator"})
    level: Optional[str] = Field(None, json_schema_extra={"example": "F3"})


class QueryBottlenecksRequest(BaseModel):
    percentile: int = Field(90, ge=50, le=99)


class PlanRouteRequest(BaseModel):
    origin: str = Field(..., json_schema_extra={"example": "entrance:A"})
    destination: str = Field(..., json_schema_extra={"example": "platform"})
    strategy: str = Field("directed", json_schema_extra={"example": "directed"})


class ReplanRouteRequest(BaseModel):
    origin: str
    destination: str
    occupancy: Optional[dict[str, str]] = None
    alpha: float = Field(3.0, gt=0.0)


class SimulateScenarioRequest(BaseModel):
    n_agents: int = Field(200, ge=1, le=5000)
    routing_mode: str = Field("static", json_schema_extra={"example": "static"})
    label: str = Field("", json_schema_extra={"example": "test_run"})
    flows: Optional[dict] = None
    elderly_ratio: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = Field(42)


class CompareStrategiesRequest(BaseModel):
    n_agents: int = Field(200, ge=1, le=5000)


class ExplainDecisionRequest(BaseModel):
    route_origin: Optional[str] = None
    route_destination: Optional[str] = None
    route_strategy: str = "directed"
    run_comparison: bool = False
    comparison_n_agents: int = 100
    context: str = ""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    model_provider: str = "deepseek"  # deepseek | gpt-4o-mini | gpt-4o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonify(obj) -> dict:
    """Convert a dataclass / Pydantic model to a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


# ---------------------------------------------------------------------------
# Tool endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui():
    html_path = BASE_DIR / "agent" / "web" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "Station Agent API is running. See /docs for endpoints."})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/map/nodes", include_in_schema=False)
async def api_map_nodes():
    """Return node scatter layers for the 3-D base map (cached after first call)."""
    from fastapi.concurrency import run_in_threadpool
    data = await run_in_threadpool(_get_node_data)
    return JSONResponse(data)

@app.post("/tools/query_environment")
async def api_query_environment(req: QueryEnvironmentRequest):
    tl = _get_tool_layer()
    result = tl.query_environment(level=req.level)
    return _jsonify(result)


@app.post("/tools/query_connectors")
async def api_query_connectors(req: QueryConnectorsRequest):
    tl = _get_tool_layer()
    result = tl.query_connectors(
        connector_type=req.connector_type,
        level=req.level,
    )
    return _jsonify(result)


@app.post("/tools/query_bottlenecks")
async def api_query_bottlenecks(req: QueryBottlenecksRequest):
    tl = _get_tool_layer()
    result = tl.query_bottlenecks(percentile=req.percentile)
    return _jsonify(result)


@app.post("/tools/plan_route")
async def api_plan_route(req: PlanRouteRequest):
    tl = _get_tool_layer()
    result = tl.plan_route(
        origin=req.origin,
        destination=req.destination,
        strategy=req.strategy,
    )
    return _jsonify(result)


@app.post("/tools/replan_route")
async def api_replan_route(req: ReplanRouteRequest):
    tl = _get_tool_layer()
    result = tl.replan_route(
        origin=req.origin,
        destination=req.destination,
        occupancy=req.occupancy,
        alpha=req.alpha,
    )
    return _jsonify(result)


@app.post("/tools/simulate_scenario")
async def api_simulate_scenario(req: SimulateScenarioRequest):
    tl = _get_tool_layer()
    result = tl.simulate_scenario(
        n_agents=req.n_agents,
        routing_mode=req.routing_mode,
        label=req.label,
        flows=req.flows,
        elderly_ratio=req.elderly_ratio,
        seed=req.seed,
    )
    return _jsonify(result)


@app.post("/tools/compare_strategies")
async def api_compare_strategies(req: CompareStrategiesRequest):
    tl = _get_tool_layer()
    result = tl.compare_strategies(n_agents=req.n_agents)
    return _jsonify(result)


@app.post("/tools/explain_decision")
async def api_explain_decision(req: ExplainDecisionRequest):
    tl = _get_tool_layer()

    route = None
    if req.route_origin and req.route_destination:
        route = tl.plan_route(req.route_origin, req.route_destination, req.route_strategy)

    comparison = None
    if req.run_comparison:
        comparison = tl.compare_strategies(n_agents=req.comparison_n_agents)

    result = tl.explain_decision(route=route, comparison=comparison, context=req.context)
    return _jsonify(result)


# ---------------------------------------------------------------------------
# Chat endpoint (Agent Orchestrator)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Natural language interface — routes message to Agent Orchestrator."""
    from fastapi.concurrency import run_in_threadpool
    from src.agent_orchestrator import AgentOrchestrator
    tl = _get_tool_layer()

    def _handle():
        orch = AgentOrchestrator(tl, provider=req.model_provider)
        reply = orch.handle(req.message)
        route_3d = _route_to_3d(orch.last_route, tl) if orch.last_route else None
        llm_func = getattr(orch, 'last_func', None)
        llm_mode = getattr(orch, 'last_mode', 'llm')
        return reply, route_3d, llm_func, llm_mode, req.model_provider

    reply, route_3d, llm_func, llm_mode, llm_provider = await run_in_threadpool(_handle)
    return {"reply": reply, "session_id": req.session_id, "route_3d": route_3d, "llm_func": llm_func, "llm_mode": llm_mode, "llm_provider": llm_provider}
