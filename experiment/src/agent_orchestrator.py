"""
Agent Orchestrator  (Phase 5b — DeepSeek LLM)
==============================================

Uses DeepSeek's OpenAI-compatible Function-Calling API to understand
natural language (Chinese & English) and dispatch to the correct
StationToolLayer tool.

Setup — set your API key before starting the server:
  PowerShell:  $env:DEEPSEEK_API_KEY = "sk-..."
  or create a file  experiment/.env  containing:
    DEEPSEEK_API_KEY=sk-...
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

# Load .env file if present (requires python-dotenv, silently ignored otherwise)
try:
    from dotenv import load_dotenv
    import pathlib as _pl
    _ENV_PATH = _pl.Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
except ImportError:
    pass

import re
from openai import OpenAI


# ---------------------------------------------------------------------------
# Regex fallback classifier (used when DeepSeek API is unavailable)
# ---------------------------------------------------------------------------

_RULES: list[tuple[list[str], str]] = [
    (["route|path|way|get to|navigate|from.*to|怎么走|路线|导航|前往"], "navigate"),
    (["replan|replanning|congestion.*route|avoid.*crowd|绕路|换路|实时.*路线"], "navigate_replan"),
    (["escalator|elevator|lift|stair|gate|door|扶梯|电梯|楼梯|闸机|出入口"], "query_connectors"),
    (["bottleneck|congest|crowd|hot.?spot|jam|拥堵|热点|瓶颈|排队"], "query_bottlenecks"),
    (["level|floor|structure|layout|overview|station.*info|层|楼层|站台|站厅|结构|概览"], "query_environment"),
    (["simulat|模拟|仿真|run.*scenario|场景"], "simulate_scenario"),
    (["compar|static.*dynamic|dynamic.*static|策略.*比较|比较.*策略|优劣"], "compare_strategies"),
    (["facility|toilet|restroom|wifi|schedule|emergency|时刻|运营|厕所|洗手间|设施|末班|开放|服务|紧急|痴散"], "query_knowledge"),
    (["explain|why|reason|recommend|分析|解释|建议|为什么|最优"], "explain_decision"),
]
_LEVEL_RE = re.compile(r"\b(f1|f3|f4|platform|concourse|站台|站厅)\b", re.I)
_ENT_RE   = re.compile(r"entrance[:\s]*([a-e])|入口[:\s]*([a-e])", re.I)
_N_RE     = re.compile(r"(\d{2,4})\s*(agent|person|passenger|人)", re.I)
_CONN_MAP = {"扶梯": "escalator", "电梯": "elevator", "楼梯": "stair", "闸机": "fare_gate", "闸门": "fare_gate"}


def _regex_classify(message: str) -> tuple[str, dict]:
    """Fallback: regex → (func_name, args)."""
    msg = message.lower()
    func = "query_environment"  # default
    for patterns, name in _RULES:
        for pat in patterns:
            if re.search(pat, msg):
                func = name
                break
        else:
            continue
        break

    args: dict = {}
    # level
    lm = _LEVEL_RE.search(message)
    if lm:
        raw = lm.group(1).lower()
        lv = {"f1": "F1", "platform": "F1", "站台": "F1",
              "f3": "F3", "concourse": "F3", "站厅": "F3", "f4": "F4"}.get(raw)
        if lv:
            args["level"] = lv
    # entrance
    em = _ENT_RE.search(message)
    entrance_label = ((em.group(1) or em.group(2)).upper() if em else "A")
    # navigate args
    if func in ("navigate", "navigate_replan"):
        args["origin"] = f"entrance:{entrance_label}"
        args["destination"] = "platform"
        if re.search(r"exit|出口|离开|离站", message, re.I):
            args["origin"], args["destination"] = "platform", f"entrance:{entrance_label}"
        if func == "navigate_replan":
            args["replan"] = True
        func = "navigate"
    # connector type
    if func == "query_connectors":
        for cn, ct in {
            "escalator": "escalator", "elevator": "elevator",
            "stair": "stair", "fare_gate": "fare_gate", "gate": "fare_gate",
            **_CONN_MAP,
        }.items():
            if re.search(cn, message, re.I):
                args["connector_type"] = ct
                break
    # n_agents
    if func in ("simulate_scenario", "compare_strategies"):
        nm = _N_RE.search(message)
        args["n_agents"] = min(max(int(nm.group(1)), 10), 2000) if nm else 200
        if func == "simulate_scenario":
            args["routing_mode"] = "dynamic" if re.search(r"dynamic|动态", message, re.I) else "static"
    return func, args


# ---------------------------------------------------------------------------
# Multi-provider LLM client
# ---------------------------------------------------------------------------

_PROVIDERS: dict = {
    "deepseek": {
        "env_key":  "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model":    "deepseek-chat",
        "display":  "DeepSeek",
    },
    "gpt-4o-mini": {
        "env_key":  "OPENAI_API_KEY",
        "base_url": None,
        "model":    "gpt-4o-mini",
        "display":  "GPT-4o-mini",
    },
    "gpt-4o": {
        "env_key":  "OPENAI_API_KEY",
        "base_url": None,
        "model":    "gpt-4o",
        "display":  "GPT-4o",
    },
}
_DEFAULT_PROVIDER = "deepseek"

# Per-provider client cache (lazy init)
_clients: dict[str, OpenAI] = {}


def _get_client(provider: str = _DEFAULT_PROVIDER) -> tuple[OpenAI, str]:
    """Return (OpenAI-compatible client, model_name) for the given provider key."""
    cfg = _PROVIDERS.get(provider) or _PROVIDERS[_DEFAULT_PROVIDER]
    if provider not in _clients:
        api_key = os.environ.get(cfg["env_key"], "")
        if not api_key:
            raise RuntimeError(
                f"{cfg['env_key']} is not set.\n"
                f"Add {cfg['env_key']}=sk-... to experiment/.env"
            )
        kwargs: dict = {"api_key": api_key}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        _clients[provider] = OpenAI(**kwargs)
    return _clients[provider], cfg["model"]


# ---------------------------------------------------------------------------
# Function / Tool schemas for DeepSeek function calling
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Plan a navigation route inside the metro station. "
                "Entrances A/B/C are on level F4; D is on F3. Platform is on F1. "
                "Use replan=true when the user mentions congestion or wants to avoid crowds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Start location: 'entrance:A', 'entrance:B', 'entrance:C' (on F4) or 'entrance:D' (on F3), or 'platform'.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "End location: 'entrance:A'–'entrance:D' or 'platform'.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["directed", "static", "penalised", "accessible", "dynamic"],
                        "description": "Routing strategy (default: directed).",
                    },
                    "replan": {
                        "type": "boolean",
                        "description": "Replan with congestion avoidance (default: false).",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_environment",
            "description": "Query station structure, node/edge statistics, level information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["F1", "F3", "F4"],
                        "description": "Filter to a specific level (omit for all levels).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_connectors",
            "description": "Query vertical connectors: escalators, elevators, stairs, fare gates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connector_type": {
                        "type": "string",
                        "enum": ["escalator", "elevator", "stair", "fare_gate"],
                    },
                    "level": {
                        "type": "string",
                        "enum": ["F1", "F3", "F4"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_bottlenecks",
            "description": "Identify congestion hotspots and bottleneck edges in the station.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percentile": {
                        "type": "integer",
                        "description": "Congestion percentile threshold 50–99 (default 90).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_scenario",
            "description": "Run a crowd/agent simulation inside the station.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_agents": {
                        "type": "integer",
                        "description": "Number of passengers to simulate (default 200).",
                    },
                    "routing_mode": {
                        "type": "string",
                        "enum": ["static", "dynamic"],
                        "description": "Routing mode (default: static).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": "Compare static vs dynamic routing strategies for a given passenger load.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_agents": {
                        "type": "integer",
                        "description": "Number of agents for comparison (default 100).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_decision",
            "description": "Explain why a particular route or routing strategy was chosen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The user's question or context string.",
                    },
                    "with_route": {
                        "type": "boolean",
                        "description": "Plan a route first before explaining (default: false).",
                    },
                    "entrance": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Entrance to use when planning contextual route (default: A).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": (
                "Answer questions about station facilities, accessibility features, "
                "operating hours, schedule, emergency procedures, WiFi, toilets, "
                "lost & found, or any general station FAQ. "
                "Use when the question does NOT fit navigate/environment/connectors/simulation/comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question to search in the station knowledge base.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are an intelligent navigation assistant for a metro station. "
    "Station layout: F1=Platform(站台层), F3=Concourse(站厅层), F4=Transport hub(交通层). "
    "There are 4 active entrances: A, B, C are on F4 (street level); D is on F3 (concourse level). "
    "Simulation benchmark (200 agents): static routing → 98.0% arrival / 184 s avg; "
    "dynamic (congestion-aware) routing → 98.0% arrival / 193.7 s avg, 78% of agents replanned. "
    "Always respond by calling one of the provided functions — never reply as plain text. "
    "If the user's origin/destination is ambiguous, default to entrance:A → platform. "
    "Detect Chinese keywords: 路线/导航/前往=navigate, 结构/层/概览=query_environment, "
    "扶梯/电梯/楼梯/闸机=query_connectors, 拥堵/热点/瓶颈=query_bottlenecks, "
    "模拟/仿真=simulate_scenario, 比较/策略=compare_strategies, 解释/建议/为什么=explain_decision, "
    "设施/时刻表/厠所/无障碍/服务/FAQ/运营/紧急疑散=query_knowledge."
)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(message: str, provider: str = _DEFAULT_PROVIDER) -> tuple[str, dict]:
    """Send message to given LLM provider, return (function_name, kwargs_dict)."""
    client, model = _get_client(provider)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tools=_TOOLS,
        tool_choice="required",
        temperature=0.0,
    )
    choice = resp.choices[0]
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        return tc.function.name, json.loads(tc.function.arguments or "{}")
    raise ValueError(
        f"DeepSeek returned no function call (finish_reason={choice.finish_reason})"
    )


# ---------------------------------------------------------------------------
# Response formatters
# ---------------------------------------------------------------------------

def _fmt_route(result) -> str:
    if not result.ok:
        return f"Route planning failed: {result.error}"
    lines = [
        f"**Route Plan** (strategy: {result.strategy})",
        f"- Origin: `{result.origin}` → Destination: `{result.destination}`",
        f"- Distance: **{result.total_distance_m:.0f} m**  |  "
        f"Travel time: **{result.total_travel_time_s:.0f} s**",
        f"- Levels: {' → '.join(result.levels_traversed)}",
    ]
    if result.connectors_used:
        lines.append(f"- Connectors used: {', '.join(result.connectors_used)}")
    acc = "✅ Accessible (no escalators)" if result.is_accessible else "⚠ Includes escalators"
    lines.append(f"- Accessibility: {acc}")
    return "\n".join(lines)


def _fmt_env(result) -> str:
    if not result.ok:
        return f"Environment query failed: {result.error}"
    lines = [f"**Station Environment** (graph hash `{result.graph_hash}`)",
             f"- Total nodes: {result.total_nodes}  |  Total edges: {result.total_edges}",
             f"- Tactile path nodes: {result.blind_path_nodes}"]
    for lv in result.levels:
        lines.append(
            f"- {lv.level} ({lv.name_en})  "
            f"nodes: {lv.n_nodes}  edges: {lv.n_edges}  elev: {lv.elevation_m} m"
        )
    return "\n".join(lines)


def _fmt_connectors(result) -> str:
    if not result.ok:
        return f"Connector query failed: {result.error}"
    if not result.connectors:
        return "No connectors found matching the criteria."
    lines = [f"**Connector Status** ({result.total} total)"]
    for c in result.connectors[:12]:
        lines.append(
            f"- `{c.connector_id}` [{c.connector_type}] "
            f"{c.from_level}→{c.to_level}  "
            f"capacity: {c.capacity}  direction: {c.direction}  state: {c.state}"
        )
    if result.total > 12:
        lines.append(f"  … ({result.total - 12} more — use /tools/query_connectors for full list)")
    return "\n".join(lines)


def _fmt_bottleneck(result) -> str:
    if not result.ok:
        return f"Bottleneck analysis failed: {result.error}"
    if not result.top_bottlenecks:
        return "No significant bottlenecks detected."
    lines = [f"**Congestion Hot-spots** (top {len(result.top_bottlenecks)} edges)"]
    for b in result.top_bottlenecks:
        cid = f"[{b.connector_id}]" if b.connector_id else ""
        lines.append(
            f"- {b.u[:18]}…→{b.v[:18]}…  "
            f"type: {b.edge_type}{cid}  "
            f"throughput: {b.throughput}  score: {b.congestion_score:.2f}  level: {b.level}"
        )
    if result.max_queue_near_stairs:
        lines.append(f"- Max queue near stairs: **{result.max_queue_near_stairs:.1f} agents**")
    return "\n".join(lines)


def _fmt_sim(result) -> str:
    if not result.ok:
        return f"Simulation failed: {result.error}"
    lines = [
        f"**Simulation Result** ({result.label}, {result.routing_mode} routing)",
        f"- Agents: {result.n_agents}  Arrived: {result.n_arrived} ({result.arrive_rate:.1%})",
        f"- Mean travel time: {result.mean_travel_time_s:.1f} s  "
        f"P95: {result.p95_travel_time_s:.1f} s  Max: {result.max_travel_time_s:.1f} s",
        f"- Mean wait time: {result.mean_wait_time_s:.1f} s  "
        f"Peak stair queue: {result.max_queue_near_stairs:.1f}",
        f"- Elderly agents: {result.n_elderly}, mean travel: {result.mean_elderly_travel_s:.1f} s",
        f"- Replanning events: {result.total_replans}",
    ]
    return "\n".join(lines)


def _fmt_compare(result) -> str:
    if not result.ok:
        return f"Comparison failed: {result.error}"
    lines = [
        f"**Strategy Comparison**: `{result.baseline_label}` vs `{result.scenario_label}`",
        result.summary_sentence or "",
    ]
    for d in result.deltas:
        icon = "✅" if d.better else "⚠"
        lines.append(
            f"{icon} {d.metric.replace('_', ' ')} | "
            f"{d.baseline_value} → {d.scenario_value} | "
            f"Δ {d.delta:+.2f} ({d.pct_change:+.1f}%)"
        )
    return "\n".join(lines)


def _fmt_explain(result) -> str:
    if not result.ok:
        return f"Explanation failed: {result.error}"
    lines = ["**Decision Analysis**"]
    for step in result.reasoning_steps:
        lines.append(f"1. {step}")
    lines.append(f"\n**Conclusion**: {result.conclusion}")
    lines.append(f"\n**Recommendation**: {result.recommendation}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """Routes user messages via DeepSeek LLM, with regex fallback."""

    def __init__(self, tool_layer: Any, provider: str = "deepseek"):
        self._tl = tool_layer
        self._provider = provider
        self.last_route: Any = None
        self.last_func: Optional[str] = None
        self.last_mode: str = "llm"  # "llm" or "regex"

    def handle(self, message: str) -> str:
        self.last_route = None
        self.last_mode = "llm"
        # Explicit regex-only mode (no LLM call)
        if self._provider == "regex":
            self.last_mode = "regex"
            func_name, args = _regex_classify(message)
            return self._dispatch(func_name, args, message)
        try:
            func_name, args = _call_llm(message, self._provider)
            return self._dispatch(func_name, args, message)
        except Exception as llm_exc:  # noqa: BLE001
            # Graceful fallback to regex classifier
            self.last_mode = "regex"
            try:
                func_name, args = _regex_classify(message)
                return self._dispatch(func_name, args, message)
            except Exception as exc:  # noqa: BLE001
                return (
                    f"⚠ Agent error: {llm_exc}\n\n"
                    "Examples you can try:\n"
                    "- *Route from entrance B to platform*\n"
                    "- *Show escalators on F3*\n"
                    "- *Station overview*"
                )

    def _dispatch(self, func_name: str, args: dict, raw_message: str) -> str:
        self.last_func = func_name  # expose to tool_api for frontend display
        tl = self._tl

        if func_name == "navigate":
            origin = args.get("origin", "entrance:A")
            dest = args.get("destination", "platform")
            strategy = args.get("strategy", "directed")
            replan = args.get("replan", False)
            if replan:
                result = tl.replan_route(origin, dest)
            else:
                result = tl.plan_route(origin, dest, strategy)
            self.last_route = result
            return _fmt_route(result)

        if func_name == "query_environment":
            result = tl.query_environment(level=args.get("level"))
            return _fmt_env(result)

        if func_name == "query_connectors":
            result = tl.query_connectors(
                connector_type=args.get("connector_type"),
                level=args.get("level"),
            )
            return _fmt_connectors(result)

        if func_name == "query_bottlenecks":
            result = tl.query_bottlenecks(percentile=args.get("percentile", 90))
            return _fmt_bottleneck(result)

        if func_name == "simulate_scenario":
            result = tl.simulate_scenario(
                n_agents=args.get("n_agents", 200),
                routing_mode=args.get("routing_mode", "static"),
            )
            return _fmt_sim(result)

        if func_name == "compare_strategies":
            result = tl.compare_strategies(n_agents=args.get("n_agents", 100))
            return _fmt_compare(result)

        if func_name == "explain_decision":
            route = None
            if args.get("with_route", False):
                entrance = args.get("entrance", "A")
                route = tl.plan_route(f"entrance:{entrance}", "platform")
            result = tl.explain_decision(
                route=route, context=args.get("context", raw_message)
            )
            return _fmt_explain(result)

        if func_name == "query_knowledge":
            return self._rag_answer(args.get("query", raw_message))

        return f"Unknown function returned by LLM: {func_name}"

    def _rag_answer(self, query: str) -> str:
        """Retrieve from station knowledge base, then generate answer via LLM."""
        try:
            from src.rag_retriever import StationRAG
            chunks = StationRAG.get_instance().query(query, n_results=3)
        except Exception as e:
            return f"Knowledge base unavailable: {e}"

        if not chunks:
            return "I couldn't find relevant information in the station knowledge base."

        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
        try:
            client, model = _get_client(self._provider)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful metro station assistant. "
                            "Answer the user's question based ONLY on the provided context passages. "
                            "Be concise and friendly. If the context does not contain the answer, say so clearly."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content or "No answer generated."
        except Exception:
            return (
                f"**Station Knowledge** (top {len(chunks)} passages):\n\n"
                + "\n\n---\n\n".join(chunks)
            )
