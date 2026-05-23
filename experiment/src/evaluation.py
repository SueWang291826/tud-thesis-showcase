"""
Step 6: Evaluation
===================

Experiment metrics, comparison analysis, and thesis-quality output.

Computes:
  - Per-scenario summary statistics
  - Cross-scenario comparison tables
  - Improvement percentages
  - Connector utilisation metrics
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.utils import dump_json


def compute_scenario_metrics(result: dict) -> dict:
    """Compute summary metrics from a simulation result."""
    tt = result.get("travel_times", [])
    wt = result.get("wait_times", [])
    ett = result.get("elderly_travel_times", [])
    ntt = result.get("normal_travel_times", [])
    sq = result.get("stair_queue_over_time", [])

    mean_tt = sum(tt) / len(tt) if tt else 0.0
    median_tt = sorted(tt)[len(tt) // 2] if tt else 0.0
    p95_tt = sorted(tt)[int(len(tt) * 0.95)] if len(tt) > 1 else mean_tt
    max_tt = max(tt) if tt else 0.0
    
    mean_wt = sum(wt) / len(wt) if wt else 0.0
    max_queue = max((q for _, q in sq), default=0)

    return {
        "label": result.get("label", ""),
        "routing_mode": result.get("routing_mode", ""),
        "n_agents": len(wt),
        "n_arrived": len(tt),
        "arrive_rate": result.get("arrive_rate", 0.0),
        "mean_travel_time": mean_tt,
        "median_travel_time": median_tt,
        "p95_travel_time": p95_tt,
        "max_travel_time": max_tt,
        "mean_wait_time": mean_wt,
        "max_queue_near_stairs": max_queue,
        "mean_elderly_travel": sum(ett) / len(ett) if ett else 0.0,
        "mean_normal_travel": sum(ntt) / len(ntt) if ntt else 0.0,
        "n_elderly": len(ett),
        "n_normal": len(ntt),
        "total_replans": sum(1 for _ in result.get("replan_events", [])),
    }


def compare_scenarios(metrics_list: list[dict]) -> dict:
    """Compare multiple scenario metrics.
    
    Returns comparison dict with per-metric changes and improvement percentages.
    """
    if len(metrics_list) < 2:
        return {"scenarios": metrics_list}

    baseline = metrics_list[0]
    comparisons = []
    
    for m in metrics_list[1:]:
        comp = {
            "baseline": baseline["label"],
            "scenario": m["label"],
        }
        # Compute deltas
        for key in ["mean_travel_time", "median_travel_time", "p95_travel_time",
                     "mean_wait_time", "max_queue_near_stairs", "arrive_rate"]:
            bv = baseline.get(key, 0)
            sv = m.get(key, 0)
            comp[f"{key}_baseline"] = bv
            comp[f"{key}_scenario"] = sv
            comp[f"{key}_delta"] = sv - bv
            if bv > 0:
                comp[f"{key}_pct_change"] = (sv - bv) / bv * 100
            else:
                comp[f"{key}_pct_change"] = 0.0
        comparisons.append(comp)

    return {
        "scenarios": metrics_list,
        "comparisons": comparisons,
    }


def graph_topology_metrics(g: nx.Graph) -> dict:
    """Compute graph-level metrics for the navigation graph."""
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    level_counts = defaultdict(int)

    for _, attr in g.nodes(data=True):
        node_types[attr.get("node_type", "unknown")] += 1
        level_counts[attr.get("level", "unknown")] += 1

    for _, _, attr in g.edges(data=True):
        edge_types[attr.get("edge_type", "unknown")] += 1

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "is_connected": nx.is_weakly_connected(g) if g.is_directed() else nx.is_connected(g),
        "n_components": nx.number_weakly_connected_components(g) if g.is_directed() else nx.number_connected_components(g),
        "diameter": (nx.diameter(g.to_undirected()) if (nx.is_weakly_connected(g) if g.is_directed() else nx.is_connected(g)) and g.number_of_nodes() < 10000 else -1),
        "avg_degree": sum(d for _, d in g.degree()) / g.number_of_nodes() if g.number_of_nodes() > 0 else 0,
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "level_counts": dict(level_counts),
    }


def connector_utilisation(result: dict) -> dict:
    """Compute connector utilisation from edge throughput."""
    throughput = result.get("edge_throughput", {})
    
    stair_total = 0
    escalator_total = 0
    elevator_total = 0
    floor_total = 0

    for edge_key, count in throughput.items():
        ek = edge_key.lower()
        # Node-ID prefixes: stair anchors/steps contain "stair_",
        # escalator anchors/steps contain "esc_", elevators contain "elev_".
        if "stair_" in ek:
            stair_total += count
        elif "esc_" in ek or "escalator" in ek:
            escalator_total += count
        elif "elev_" in ek or "elevator" in ek:
            elevator_total += count
        else:
            floor_total += count

    total = stair_total + escalator_total + elevator_total + floor_total
    return {
        "stair_throughput": stair_total,
        "escalator_throughput": escalator_total,
        "elevator_throughput": elevator_total,
        "floor_throughput": floor_total,
        "total_throughput": total,
        "stair_share": stair_total / total if total > 0 else 0,
        "escalator_share": escalator_total / total if total > 0 else 0,
    }


# ============================================================================
# Save outputs
# ============================================================================

def save_evaluation_outputs(
    scenario_results: list[dict],
    graph_metrics: dict,
    out_dir: str | Path,
) -> None:
    """Save all evaluation outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-scenario metrics
    all_metrics = []
    for result in scenario_results:
        metrics = compute_scenario_metrics(result)
        all_metrics.append(metrics)
        dump_json(out_dir / f"metrics_{metrics['label']}.json", metrics)

    # Comparison
    comparison = compare_scenarios(all_metrics)
    dump_json(out_dir / "comparison.json", comparison)

    # Graph topology
    dump_json(out_dir / "graph_topology.json", graph_metrics)

    # Connector utilisation per scenario
    for result in scenario_results:
        util = connector_utilisation(result)
        dump_json(out_dir / f"connector_util_{result.get('label', 'unknown')}.json", util)

    # Summary table (markdown)
    md_lines = ["# Experiment Results\n"]
    md_lines.append("| Metric | " + " | ".join(m["label"] for m in all_metrics) + " |")
    md_lines.append("|--------|" + "|".join(["--------"] * len(all_metrics)) + "|")
    
    for key in ["arrive_rate", "mean_travel_time", "median_travel_time",
                 "p95_travel_time", "mean_wait_time", "max_queue_near_stairs"]:
        values = [f"{m.get(key, 0):.2f}" for m in all_metrics]
        md_lines.append(f"| {key} | " + " | ".join(values) + " |")

    (out_dir / "results_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[Step 6] Evaluation: {len(all_metrics)} scenarios compared")
