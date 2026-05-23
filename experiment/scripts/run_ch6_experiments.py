"""
run_ch6_experiments.py
======================
Master script for Chapter 6 experiments.

Usage examples:
  python scripts/run_ch6_experiments.py --experiments A B C --quick
  python scripts/run_ch6_experiments.py --experiments D E F --quick
  python scripts/run_ch6_experiments.py --skip-existing
  python scripts/run_ch6_experiments.py --plots-only
  python scripts/run_ch6_experiments.py --no-plots
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from statistics import mean, median

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CH6  = ROOT / "outputs" / "ch6"

DIRS = {
    "data":    CH6 / "data",
    "tables":  CH6 / "tables",
    "figures": CH6 / "figures",
    "logs":    CH6 / "logs",
    "configs": CH6 / "configs",
}


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 6 experiment runner")
    p.add_argument(
        "--experiments", nargs="+", choices=list("ABCDEF"),
        default=list("ABCDEF"),
        metavar="EXP",
        help="Which experiments to run (e.g. --experiments A B C)",
    )
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip experiments whose output CSV already exists")
    p.add_argument("--quick", action="store_true",
                   help="Use reduced agent counts / fewer threshold variants")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--plots-only", action="store_true",
                   help="Only regenerate figures, do not re-run simulations")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ch6")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _todo(logger: logging.Logger, exp: str, msg: str) -> None:
    """Write a TODO notice to the experiment log and to the shared log."""
    todo_path = DIRS["logs"] / f"experiment_{exp}.log"
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    with todo_path.open("a", encoding="utf-8") as f:
        f.write(f"TODO  {msg}\n")
    logger.info("[Exp %s] TODO: %s", exp, msg)


# ── Placeholder experiment functions ─────────────────────────────────────────

def _p95(values: list[float]) -> float:
    """Compute the 95th percentile without numpy."""
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = 0.95 * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


def _connector_load(edge_throughput: dict[str, int]) -> list[tuple[str, int]]:
    """Extract connector-type edges (stair/escalator/elevator) sorted by throughput."""
    connectors: dict[str, int] = {}
    for edge_key, count in edge_throughput.items():
        src, _, _ = edge_key.partition("|")
        # Identify first step of each connector chain only
        if "stair_" in src and "_s0" in src:
            name = src.split("|")[0]
            connectors[name] = connectors.get(name, 0) + count
        elif "esc_" in src or ("escalator" in src and "_s0" in src):
            connectors[src] = connectors.get(src, 0) + count
        elif "elev_" in src and "entry" in src:
            connectors[src] = connectors.get(src, 0) + count

    # Fallback: if nothing matched by s0, sum any stair/esc/elev edges
    if not connectors:
        for edge_key, count in edge_throughput.items():
            src = edge_key.split("|")[0]
            if any(tag in src for tag in ("stair_", "esc_", "escalator", "elev_")):
                connectors[src] = connectors.get(src, 0) + count

    return sorted(connectors.items(), key=lambda x: x[1], reverse=True)


def run_experiment_A(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp A — Baseline Static Routing and Simulation.

    Reuses: outputs/step5_simulation/scenB_static/result_scenB.json
    Outputs:
      outputs/ch6/data/experiment_A_static_results.csv   (all metrics, long form)
      outputs/ch6/tables/table_6_3_experiment_A_results.csv (thesis table)
      outputs/ch6/logs/experiment_A.log
    """
    out_data   = DIRS["data"]   / "experiment_A_static_results.csv"
    out_table  = DIRS["tables"] / "table_6_3_experiment_A_results.csv"
    exp_log    = DIRS["logs"]   / "experiment_A.log"

    # ── skip guard ─────────────────────────────────────────────────────────
    if args.skip_existing and out_data.exists() and out_table.exists():
        logger.info("[Exp A] Skipping — outputs already exist")
        return

    logger.info("[Exp A] Starting: Baseline Static Routing and Simulation")
    t0 = time.time()

    # ── locate source files ────────────────────────────────────────────────
    sim_dir    = ROOT / "outputs" / "step5_simulation" / "scenB_static"
    result_json = sim_dir / "result_scenB.json"
    summary_csv = sim_dir / "summary.csv"

    def _log_err(msg: str) -> None:
        logger.error("[Exp A] %s", msg)
        exp_log.parent.mkdir(parents=True, exist_ok=True)
        with exp_log.open("a", encoding="utf-8") as f:
            f.write(f"ERROR  {msg}\n")

    if not result_json.exists():
        _log_err(f"Source file not found: {result_json}")
        return
    if not summary_csv.exists():
        _log_err(f"Summary CSV not found: {summary_csv}")
        return

    # ── load data ──────────────────────────────────────────────────────────
    logger.info("[Exp A] Loading %s", result_json.name)
    with result_json.open(encoding="utf-8") as f:
        result = json.load(f)

    # n_agents from summary CSV (the config value, not len of completed list)
    n_agents = 200
    try:
        with summary_csv.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
            n_agents = int(row["n_agents"])
    except Exception as exc:
        logger.warning("[Exp A] Could not read n_agents from summary.csv (%s); using %d", exc, n_agents)

    travel_times: list[float] = result.get("travel_times", [])
    wait_times:   list[float] = result.get("wait_times",   [])
    elderly_tt:   list[float] = result.get("elderly_travel_times", [])
    normal_tt:    list[float] = result.get("normal_travel_times",  [])
    queue_series: list        = result.get("stair_queue_over_time", [])
    edge_tp:      dict        = result.get("edge_throughput", {})
    arrive_rate:  float       = result.get("arrive_rate", float("nan"))
    routing_mode: str         = result.get("routing_mode", "static")
    label:        str         = result.get("label", "scenB_static")

    # ── compute metrics ────────────────────────────────────────────────────
    n_completed = len(travel_times)
    n_failed    = max(0, n_agents - n_completed)

    tt_mean   = mean(travel_times)   if travel_times else float("nan")
    tt_median = median(travel_times) if travel_times else float("nan")
    tt_p95    = _p95(travel_times)

    wt_mean   = mean(wait_times)     if wait_times   else float("nan")
    wt_median = median(wait_times)   if wait_times   else float("nan")
    wt_max    = max(wait_times)      if wait_times   else float("nan")

    max_queue = max((q for _, q in queue_series), default=0) if queue_series else 0

    elderly_mean  = mean(elderly_tt) if elderly_tt else float("nan")
    normal_mean   = mean(normal_tt)  if normal_tt  else float("nan")

    conn_load = _connector_load(edge_tp)
    top_connector = conn_load[0][0] if conn_load else "N/A"
    top_connector_count = conn_load[0][1] if conn_load else 0

    runtime_s = time.time() - t0   # time to extract (not simulation time)

    # ── write experiment_A_static_results.csv (long form) ─────────────────
    metrics = [
        ("label",                    label),
        ("routing_mode",             routing_mode),
        ("n_agents",                 n_agents),
        ("n_completed",              n_completed),
        ("n_failed",                 n_failed),
        ("arrive_rate",              round(arrive_rate, 4)),
        ("mean_travel_time_s",       round(tt_mean,   2)),
        ("median_travel_time_s",     round(tt_median, 2)),
        ("p95_travel_time_s",        round(tt_p95,    2)),
        ("mean_travel_elderly_s",    round(elderly_mean, 2) if not math.isnan(elderly_mean) else "N/A"),
        ("mean_travel_normal_s",     round(normal_mean,  2) if not math.isnan(normal_mean)  else "N/A"),
        ("mean_wait_time_s",         round(wt_mean,   2)),
        ("median_wait_time_s",       round(wt_median, 2)),
        ("max_wait_time_s",          round(wt_max,    2)),
        ("max_queue",                max_queue),
        ("dominant_connector",       top_connector),
        ("dominant_connector_count", top_connector_count),
        ("n_connector_types",        len(conn_load)),
        ("total_replans",            0),
        ("data_source",              str(result_json.relative_to(ROOT))),
        ("extraction_time_s",        round(runtime_s, 3)),
    ]

    out_data.parent.mkdir(parents=True, exist_ok=True)
    with out_data.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(metrics)
    logger.info("[Exp A] Wrote %s", out_data.name)

    # ── write table_6_3 (formatted thesis table) ──────────────────────────
    table_rows = [
        ("Metric",                        "Value",                            "Unit"),
        ("Routing mode",                  routing_mode.capitalize(),          "—"),
        ("Simulation seed",               42,                                 "—"),
        ("Number of agents",              n_agents,                           "agents"),
        ("Completed agents",              n_completed,                        "agents"),
        ("Failed / stranded agents",      n_failed,                           "agents"),
        ("Arrival rate",                  round(arrive_rate, 3),              "—"),
        ("Mean travel time",              round(tt_mean,   1),                "s"),
        ("Median travel time",            round(tt_median, 1),                "s"),
        ("P95 travel time",               round(tt_p95,    1),                "s"),
        ("Mean travel time — elderly",    round(elderly_mean, 1) if not math.isnan(elderly_mean) else "N/A", "s"),
        ("Mean travel time — normal",     round(normal_mean,  1) if not math.isnan(normal_mean)  else "N/A", "s"),
        ("Mean waiting time",             round(wt_mean,   1),                "s"),
        ("Median waiting time",           round(wt_median, 1),                "s"),
        ("Max waiting time",              round(wt_max,    1),                "s"),
        ("Peak connector queue",          max_queue,                          "agents"),
        ("Dominant bottleneck connector", top_connector,                      "—"),
        ("Bottleneck throughput",         top_connector_count,                "agent-crossings"),
        ("Total replanning events",       0,                                  "events"),
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)
    logger.info("[Exp A] Wrote %s", out_table.name)

    # ── write experiment log ───────────────────────────────────────────────
    exp_log.parent.mkdir(parents=True, exist_ok=True)
    with exp_log.open("a", encoding="utf-8") as f:
        f.write(f"INFO   Experiment A completed in {runtime_s:.2f}s\n")
        f.write(f"INFO   Source: {result_json}\n")
        f.write(f"INFO   n_agents={n_agents}, completed={n_completed}, "
                f"arrive_rate={arrive_rate:.3f}\n")
        f.write(f"INFO   mean_tt={tt_mean:.1f}s  median_tt={tt_median:.1f}s  "
                f"p95_tt={tt_p95:.1f}s\n")
        f.write(f"INFO   mean_wait={wt_mean:.1f}s  max_queue={max_queue}\n")
        f.write(f"INFO   dominant_connector={top_connector} "
                f"(throughput={top_connector_count})\n")

    elapsed = time.time() - t0
    logger.info("[Exp A] Done in %.2f s — arrive_rate=%.3f, mean_tt=%.1f s, "
                "p95_tt=%.1f s, max_queue=%d",
                elapsed, arrive_rate, tt_mean, tt_p95, max_queue)


def _load_result(json_path: Path, summary_path: Path,
                 logger: logging.Logger, tag: str) -> dict | None:
    """Load result JSON + n_agents from summary CSV. Returns None on error."""
    if not json_path.exists():
        logger.error("[Exp %s] Source not found: %s", tag, json_path)
        return None
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    n_agents = 200
    if summary_path.exists():
        try:
            with summary_path.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
                n_agents = int(row["n_agents"])
        except Exception as exc:
            logger.warning("[Exp %s] Could not read n_agents: %s", tag, exc)
    result["_n_agents"] = n_agents
    return result


def _summarise(result: dict) -> dict:
    """Compute derived metrics from a loaded result dict."""
    tt  = result.get("travel_times", [])
    wt  = result.get("wait_times",   [])
    n   = result["_n_agents"]
    nc  = len(tt)
    replans = result.get("replan_events", [])

    queue_series = result.get("stair_queue_over_time", [])
    max_q = max((q for _, q in queue_series), default=0) if queue_series else 0

    return {
        "label":            result.get("label", ""),
        "routing_mode":     result.get("routing_mode", ""),
        "n_agents":         n,
        "n_completed":      nc,
        "n_failed":         max(0, n - nc),
        "arrive_rate":      result.get("arrive_rate", float("nan")),
        "mean_travel_time_s":   round(mean(tt),   2) if tt else float("nan"),
        "median_travel_time_s": round(median(tt), 2) if tt else float("nan"),
        "p95_travel_time_s":    round(_p95(tt),   2) if tt else float("nan"),
        "mean_travel_elderly_s": (
            round(mean(result.get("elderly_travel_times", [])), 2)
            if result.get("elderly_travel_times") else float("nan")),
        "mean_travel_normal_s": (
            round(mean(result.get("normal_travel_times", [])), 2)
            if result.get("normal_travel_times") else float("nan")),
        "mean_wait_time_s":   round(mean(wt),   2) if wt else float("nan"),
        "median_wait_time_s": round(median(wt), 2) if wt else float("nan"),
        "max_wait_time_s":    round(max(wt),    2) if wt else float("nan"),
        "max_queue":          max_q,
        "reroute_count":      len(replans),
    }


GRAPH_VARIANTS: dict[str, dict] = {
    "default": {
        "description": (
            "Baseline human-scale graph: 0.5 m grid, 8-neighbour floor links, "
            "explicit entrance/platform semantics, blind-path stitching enabled"
        ),
        "config_overrides": {},
        "semantic_ablation": False,
    },
    "sparse_semantic_light": {
        "description": (
            "Sparse + semantics-light graph: 0.75 m grid, 4-neighbour floor links, "
            "no blind-path chain stitching, entrance/platform semantics demoted to boundary fallback"
        ),
        "config_overrides": {
            "sampling": {
                "grid_resolution_m": 0.75,
            },
            "graph": {
                "neighbor_connectivity": 4,
                "blind_path_link_factor": 0.0,
            },
        },
        "semantic_ablation": True,
    },
}


def _deep_update_dict(base: dict, patch: dict) -> dict:
    """Recursively update nested dictionaries in-place."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def _merge_connector_nodes_for_variant(level_nodes: dict, conn_nodes: dict[str, list[dict]]) -> dict:
    """Inject connector-anchor nodes into per-level node sets."""
    for level_key, nodes in conn_nodes.items():
        if level_key not in level_nodes:
            continue
        level_nodes[level_key]["nodes_valid"].extend(nodes)
        level_nodes[level_key]["nodes_all"].extend(nodes)
        level_nodes[level_key]["n_valid"] = len(level_nodes[level_key]["nodes_valid"])
        level_nodes[level_key]["n_total"] = len(level_nodes[level_key]["nodes_all"])
    return level_nodes


def _apply_semantic_light_ablation(g) -> dict[str, int]:
    """Demote explicit OD semantics so routing falls back to coarse boundary regions."""
    stats = {
        "entrance_nodes_demoted": 0,
        "platform_nodes_demoted": 0,
        "blind_path_nodes_cleared": 0,
        "blind_path_edges_removed": 0,
    }

    for _, attr in g.nodes(data=True):
        node_type = attr.get("node_type")
        if node_type == "entrance":
            attr["node_type"] = "floor"
            stats["entrance_nodes_demoted"] += 1
        elif node_type == "door_platform":
            attr["node_type"] = "floor"
            stats["platform_nodes_demoted"] += 1

        if attr.get("is_blind_path"):
            attr["is_blind_path"] = False
            attr["blind_category"] = ""
            attr["surface_type"] = "normal"
            stats["blind_path_nodes_cleared"] += 1

    blind_edges = [
        (u, v)
        for u, v, edge_attr in g.edges(data=True)
        if edge_attr.get("edge_type") == "blind_path"
    ]
    if blind_edges:
        g.remove_edges_from(blind_edges)
        stats["blind_path_edges_removed"] = len(blind_edges)

    g.graph["semantic_ablation_applied"] = True
    g.graph["semantic_ablation_stats"] = dict(stats)
    return stats


def _graph_variant_dir(variant_name: str) -> Path:
    return CH6 / "graph_variants" / variant_name


def _summarise_graph_variant(
    g,
    regions: dict[str, list[str]],
    variant_name: str,
    description: str,
) -> dict[str, object]:
    return {
        "graph_variant": variant_name,
        "description": description,
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "blind_path_nodes": sum(1 for _, attr in g.nodes(data=True) if attr.get("is_blind_path")),
        "entrance_tagged_nodes": sum(1 for _, attr in g.nodes(data=True) if attr.get("node_type") == "entrance"),
        "platform_tagged_nodes": sum(1 for _, attr in g.nodes(data=True) if attr.get("node_type") == "door_platform"),
        "entrance_region_size": len(regions.get("ENTRANCE", [])),
        "platform_region_size": len(regions.get("PLATFORM", [])),
        "semantic_ablation_applied": bool(g.graph.get("semantic_ablation_applied", False)),
        "semantic_ablation_stats": g.graph.get("semantic_ablation_stats", {}),
    }


def _load_or_build_graph_variant(
    variant_name: str,
    logger: logging.Logger,
) -> tuple[dict, object, dict[str, list[str]], dict[str, object]]:
    """Return (cfg, graph, semantic_regions, variant_meta)."""
    import copy as _copy
    import pickle as _pickle

    if variant_name not in GRAPH_VARIANTS:
        raise ValueError(f"Unknown graph variant: {variant_name}")

    sys.path.insert(0, str(ROOT))
    try:
        from src.utils import load_config as _load_cfg  # noqa: PLC0415
        from src.data_loader import load_preprocessing_products as _load_products  # noqa: PLC0415
        from src.geometry_extractor import extract_all_levels as _extract_all_levels  # noqa: PLC0415
        from src.node_sampler import (  # noqa: PLC0415
            sample_all_levels as _sample_all_levels,
            voxelize_connectors as _voxelize_connectors,
        )
        from src.graph_builder import (  # noqa: PLC0415
            build_navigation_graph as _build_navigation_graph,
            save_graph_outputs as _save_graph_outputs,
        )
        from src.routing import define_semantic_regions as _define_semantic_regions  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(f"Graph-variant imports failed: {exc}") from exc

    spec = GRAPH_VARIANTS[variant_name]
    cfg = _load_cfg(ROOT / "config" / "experiment_config.yaml")
    cfg = _copy.deepcopy(cfg)
    _deep_update_dict(cfg, spec.get("config_overrides", {}))

    if variant_name == "default":
        graph_path = ROOT / "outputs" / "step3_graph" / "navigation_graph.gpickle"
        if not graph_path.exists():
            raise FileNotFoundError(f"Default graph not found: {graph_path}")
        with graph_path.open("rb") as fh:
            g = _pickle.load(fh)
    else:
        variant_dir = _graph_variant_dir(variant_name)
        variant_dir.mkdir(parents=True, exist_ok=True)
        graph_path = variant_dir / "navigation_graph.gpickle"

        if graph_path.exists():
            logger.info("[Graph Variant] Loading cached %s graph", variant_name)
            with graph_path.open("rb") as fh:
                g = _pickle.load(fh)
        else:
            logger.info("[Graph Variant] Building %s graph from Step 1-3 pipeline", variant_name)
            products = _load_products(cfg)
            geometries, all_connectors, _control_points = _extract_all_levels(cfg, products)
            level_nodes = _sample_all_levels(geometries, cfg)
            conn_nodes = _voxelize_connectors(all_connectors, geometries, cfg)
            level_nodes = _merge_connector_nodes_for_variant(level_nodes, conn_nodes)
            g = _build_navigation_graph(geometries, level_nodes, all_connectors, cfg)

            if spec.get("semantic_ablation"):
                _apply_semantic_light_ablation(g)

            g.graph["graph_variant"] = variant_name
            g.graph["graph_variant_description"] = spec["description"]
            _save_graph_outputs(g, all_connectors, variant_dir)

            with (variant_dir / "config_snapshot.json").open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, ensure_ascii=False)

        if spec.get("semantic_ablation") and not g.graph.get("semantic_ablation_applied"):
            logger.info("[Graph Variant] Patching cached %s graph with semantic-light ablation", variant_name)
            _apply_semantic_light_ablation(g)
            g.graph["graph_variant"] = variant_name
            g.graph["graph_variant_description"] = spec["description"]
            with graph_path.open("wb") as fh:
                _pickle.dump(g, fh)

    regions = _define_semantic_regions(g, cfg)
    meta = _summarise_graph_variant(g, regions, variant_name, spec["description"])

    if variant_name != "default":
        summary_path = _graph_variant_dir(variant_name) / "graph_variant_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    logger.info(
        "[Graph Variant] %s -> %d nodes, %d edges, entrance region=%d, platform region=%d",
        variant_name,
        meta["total_nodes"],
        meta["total_edges"],
        meta["entrance_region_size"],
        meta["platform_region_size"],
    )
    return cfg, g, regions, meta


def run_experiment_B(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp B — Static versus Congestion-aware Replanning.

    Reuses:
      outputs/step5_simulation/scenB_static/result_scenB.json   (B1 static)
      outputs/step5_simulation/scenC_dynamic/result_scenC.json  (B2 dynamic)
    Outputs:
      outputs/ch6/data/experiment_B_static_dynamic_comparison.csv
      outputs/ch6/tables/table_6_4_experiment_B_results.csv
      outputs/ch6/logs/experiment_B.log
    """
    out_data  = DIRS["data"]   / "experiment_B_static_dynamic_comparison.csv"
    out_table = DIRS["tables"] / "table_6_4_experiment_B_results.csv"
    exp_log   = DIRS["logs"]   / "experiment_B.log"

    if args.skip_existing and out_data.exists() and out_table.exists():
        logger.info("[Exp B] Skipping — outputs already exist")
        return

    logger.info("[Exp B] Starting: Static vs Congestion-aware Replanning")
    t0 = time.time()

    sim_root = ROOT / "outputs" / "step5_simulation"

    def _err(msg: str) -> None:
        logger.error("[Exp B] %s", msg)
        exp_log.parent.mkdir(parents=True, exist_ok=True)
        with exp_log.open("a", encoding="utf-8") as f:
            f.write(f"ERROR  {msg}\n")

    # ── load both conditions ───────────────────────────────────────────────
    r_static = _load_result(
        sim_root / "scenB_static"  / "result_scenB.json",
        sim_root / "scenB_static"  / "summary.csv",
        logger, "B")
    r_dynamic = _load_result(
        sim_root / "scenC_dynamic" / "result_scenC.json",
        sim_root / "scenC_dynamic" / "summary.csv",
        logger, "B")

    if r_static is None:
        _err("scenB_static result missing — cannot run Exp B")
        return
    if r_dynamic is None:
        _err("scenC_dynamic result missing — cannot run Exp B")
        return

    s = _summarise(r_static)
    d = _summarise(r_dynamic)

    # ── compute delta columns ──────────────────────────────────────────────
    def _delta(key: str) -> float | str:
        sv, dv = s[key], d[key]
        try:
            return round(dv - sv, 3)
        except TypeError:
            return "N/A"

    def _pct(key: str) -> float | str:
        sv, dv = s[key], d[key]
        try:
            if sv == 0:
                return "N/A"
            return round((dv - sv) / abs(sv) * 100, 1)
        except TypeError:
            return "N/A"

    # ── write long-form CSV (one row per condition) ────────────────────────
    fieldnames = [
        "condition", "label", "routing_mode",
        "n_agents", "n_completed", "n_failed", "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_travel_elderly_s", "mean_travel_normal_s",
        "mean_wait_time_s", "median_wait_time_s", "max_wait_time_s",
        "max_queue", "reroute_count",
    ]
    rows = [
        {"condition": "B1_static",  **s},
        {"condition": "B2_dynamic", **d},
    ]

    out_data.parent.mkdir(parents=True, exist_ok=True)
    with out_data.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("[Exp B] Wrote %s", out_data.name)

    # ── write thesis comparison table ─────────────────────────────────────
    table_rows = [
        ("Metric",                   "B1 Static",                     "B2 Dynamic",                    "Delta (D−S)",                      "% Change"),
        ("Routing mode",             "Static",                         "Dynamic (congestion-aware)",    "—",                                "—"),
        ("Number of agents",         s["n_agents"],                    d["n_agents"],                   "—",                                "—"),
        ("Completed agents",         s["n_completed"],                  d["n_completed"],                _delta("n_completed"),              _pct("n_completed")),
        ("Failed agents",            s["n_failed"],                    d["n_failed"],                   _delta("n_failed"),                 "—"),
        ("Arrival rate",             round(s["arrive_rate"], 3),       round(d["arrive_rate"], 3),      _delta("arrive_rate"),              _pct("arrive_rate")),
        ("Mean travel time (s)",     s["mean_travel_time_s"],          d["mean_travel_time_s"],         _delta("mean_travel_time_s"),       _pct("mean_travel_time_s")),
        ("Median travel time (s)",   s["median_travel_time_s"],        d["median_travel_time_s"],       _delta("median_travel_time_s"),     _pct("median_travel_time_s")),
        ("P95 travel time (s)",      s["p95_travel_time_s"],           d["p95_travel_time_s"],          _delta("p95_travel_time_s"),        _pct("p95_travel_time_s")),
        ("Mean travel — elderly (s)", s["mean_travel_elderly_s"],      d["mean_travel_elderly_s"],      _delta("mean_travel_elderly_s"),    _pct("mean_travel_elderly_s")),
        ("Mean travel — normal (s)", s["mean_travel_normal_s"],        d["mean_travel_normal_s"],       _delta("mean_travel_normal_s"),     _pct("mean_travel_normal_s")),
        ("Mean waiting time (s)",    s["mean_wait_time_s"],            d["mean_wait_time_s"],           _delta("mean_wait_time_s"),         _pct("mean_wait_time_s")),
        ("Median waiting time (s)",  s["median_wait_time_s"],          d["median_wait_time_s"],         _delta("median_wait_time_s"),       _pct("median_wait_time_s")),
        ("Max waiting time (s)",     s["max_wait_time_s"],             d["max_wait_time_s"],            _delta("max_wait_time_s"),          _pct("max_wait_time_s")),
        ("Peak connector queue",     s["max_queue"],                   d["max_queue"],                  _delta("max_queue"),                _pct("max_queue")),
        ("Reroute events",           s["reroute_count"],               d["reroute_count"],              d["reroute_count"],                 "—"),
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)
    logger.info("[Exp B] Wrote %s", out_table.name)

    # ── write experiment log ───────────────────────────────────────────────
    elapsed = time.time() - t0
    exp_log.parent.mkdir(parents=True, exist_ok=True)
    with exp_log.open("a", encoding="utf-8") as f:
        f.write(f"INFO   Experiment B completed in {elapsed:.2f}s\n")
        f.write(f"INFO   Static:  arrive_rate={s['arrive_rate']:.3f}  "
                f"mean_tt={s['mean_travel_time_s']:.1f}s  "
                f"p95_tt={s['p95_travel_time_s']:.1f}s  "
                f"max_queue={s['max_queue']}  replans={s['reroute_count']}\n")
        f.write(f"INFO   Dynamic: arrive_rate={d['arrive_rate']:.3f}  "
                f"mean_tt={d['mean_travel_time_s']:.1f}s  "
                f"p95_tt={d['p95_travel_time_s']:.1f}s  "
                f"max_queue={d['max_queue']}  replans={d['reroute_count']}\n")
        f.write(f"INFO   Delta mean_tt={_delta('mean_travel_time_s')}s  "
                f"delta_wait={_delta('mean_wait_time_s')}s  "
                f"delta_queue={_delta('max_queue')}\n")

    logger.info("[Exp B] Done in %.2f s — static mean_tt=%.1f s / dynamic mean_tt=%.1f s / "
                "delta=%.1f s / reroutes=%d",
                elapsed, s["mean_travel_time_s"], d["mean_travel_time_s"],
                d["mean_travel_time_s"] - s["mean_travel_time_s"],
                d["reroute_count"])


def run_experiment_C(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp C — Single-profile Pedestrian Comparison.

    Tests each implemented pedestrian profile in isolation under identical
    station graph, OD demand, and departure pattern for BOTH static and
    congestion-aware dynamic routing.

    Implemented profiles (the only two in src/routing.py::sample_agents()):
      C1_normal : 100% normal agents    (speed 0.95–1.05× base walking speed)
      C2_elderly: 100% elderly agents   (speed 0.55–0.75× base walking speed)
      C3_mixed  : 80% normal / 20% elderly — reuses existing baseline results

    Profiles NOT implemented in current codebase (logged, not faked):
      wheelchair, luggage, accessibility-constrained

    Outputs:
      outputs/ch6/data/experiment_C_single_profile_results.csv          (static)
      outputs/ch6/data/experiment_C_single_profile_results_dynamic.csv  (dynamic)
      outputs/ch6/tables/table_6_5_experiment_C_results.csv            (combined)
      outputs/ch6/logs/experiment_C.log
    """
    import copy as _copy
    import pickle as _pickle

    out_static  = DIRS["data"]   / "experiment_C_single_profile_results.csv"
    out_dynamic = DIRS["data"]   / "experiment_C_single_profile_results_dynamic.csv"
    out_table   = DIRS["tables"] / "table_6_5_experiment_C_results.csv"
    exp_log     = DIRS["logs"]   / "experiment_C.log"

    if args.skip_existing and out_static.exists() and out_dynamic.exists() and out_table.exists():
        logger.info("[Exp C] Skipping — outputs already exist")
        return

    logger.info("[Exp C] Starting: Single-profile Pedestrian Comparison")
    t0_total = time.time()

    sim_root = ROOT / "outputs" / "step5_simulation"
    ROUTING_MODES = [
        ("static", "static"),
        ("dynamic", "dynamic"),
    ]
    PROFILES = [
        {
            "condition": "C1_normal",
            "label": "C1_normal_only",
            "description": "100% normal (speed 0.95–1.05× base)",
            "elderly_ratio": 0.0,
            "reuse_json_by_mode": {},
            "reuse_summary_by_mode": {},
        },
        {
            "condition": "C2_elderly",
            "label": "C2_elderly_only",
            "description": "100% elderly (speed 0.55–0.75× base)",
            "elderly_ratio": 1.0,
            "reuse_json_by_mode": {},
            "reuse_summary_by_mode": {},
        },
        {
            "condition": "C3_mixed",
            "label": "C3_mixed_reference",
            "description": "80% normal / 20% elderly (reference baseline)",
            "elderly_ratio": 0.2,
            "reuse_json_by_mode": {
                "static": sim_root / "scenB_static" / "result_scenB.json",
                "dynamic": sim_root / "scenC_dynamic" / "result_scenC.json",
            },
            "reuse_summary_by_mode": {
                "static": sim_root / "scenB_static" / "summary.csv",
                "dynamic": sim_root / "scenC_dynamic" / "summary.csv",
            },
        },
    ]
    NOT_IMPLEMENTED = ["wheelchair", "luggage", "accessibility-constrained"]

    n_agents_run = 50 if args.quick else 200

    exp_log.parent.mkdir(parents=True, exist_ok=True)

    def _elog(msg: str) -> None:
        with exp_log.open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")

    _elog("=== Experiment C: Single-profile Pedestrian Comparison ===")
    _elog(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _elog(f"Quick mode: {args.quick}  n_agents_new_runs: {n_agents_run}")
    _elog(f"Routing modes: {[label for label, _ in ROUTING_MODES]}")
    _elog("")
    _elog("--- Profile inventory ---")
    for p in PROFILES:
        src_static = p["reuse_json_by_mode"].get("static") or "new simulation"
        src_dynamic = p["reuse_json_by_mode"].get("dynamic") or "new simulation"
        _elog(
            f"  IMPLEMENTED   {p['condition']}: {p['description']}  "
            f"[static: {src_static}] [dynamic: {src_dynamic}]"
        )
    for ni in NOT_IMPLEMENTED:
        _elog(f"  NOT_IMPL      {ni}: profile not present in src/routing.py::sample_agents()")
    _elog("")

    needs_new_sim = any(
        any(profile["reuse_json_by_mode"].get(mode_label) is None for mode_label, _ in ROUTING_MODES)
        for profile in PROFILES
    )
    sim_env = None

    if needs_new_sim:
        sys.path.insert(0, str(ROOT))
        try:
            from src.utils import load_config as _load_cfg          # noqa: PLC0415
            from src.routing import define_semantic_regions as _def_regions  # noqa: PLC0415
        except ImportError as exc:
            logger.error("[Exp C] Import failed: %s", exc)
            _elog(f"ERROR  Import failed: {exc}")
            return

        cfg_path   = ROOT / "config" / "experiment_config.yaml"
        graph_path = ROOT / "outputs" / "step3_graph" / "navigation_graph.gpickle"

        for _p, _name in [(cfg_path, "config"), (graph_path, "graph")]:
            if not _p.exists():
                logger.error("[Exp C] %s not found: %s", _name, _p)
                _elog(f"ERROR  {_name} not found: {_p}")
                return

        cfg = _load_cfg(cfg_path)
        logger.info("[Exp C] Loading graph from %s …", graph_path.name)
        t_g = time.time()
        with open(graph_path, "rb") as _gf:
            G = _pickle.load(_gf)
        logger.info("[Exp C] Graph: %d nodes, %d edges (loaded in %.1fs)",
                    G.number_of_nodes(), G.number_of_edges(), time.time() - t_g)
        _elog(f"INFO   Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        regions = _def_regions(G, cfg)
        sim_env = (cfg, G, regions)

        from src.routing import sample_agents as _sample_agents      # noqa: PLC0415
        from src.simulation import run_simulation as _run_simulation  # noqa: PLC0415
    else:
        _sample_agents = None
        _run_simulation = None

    def _f1(v) -> object:
        if v is None:
            return "N/A"
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        try:
            return round(v, 1)
        except (TypeError, ValueError):
            return v

    def _f3(v) -> object:
        if v is None:
            return "N/A"
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        try:
            return round(v, 3)
        except (TypeError, ValueError):
            return v

    mode_results: dict[str, list[dict]] = {label: [] for label, _ in ROUTING_MODES}
    run_status: list[tuple[str, str]] = []

    for routing_label, routing_mode in ROUTING_MODES:
        _elog(f"--- Routing mode: {routing_label} ---")
        for profile in PROFILES:
            cond = profile["condition"]
            logger.info("[Exp C] Processing profile %s [%s] …", cond, routing_label)
            t_profile = time.time()

            try:
                reuse_json = profile["reuse_json_by_mode"].get(routing_label)
                reuse_summary = profile["reuse_summary_by_mode"].get(routing_label)

                if reuse_json is not None:
                    r = _load_result(
                        reuse_json,
                        reuse_summary or Path("__none__"),
                        logger, "C",
                    )
                    if r is None:
                        msg = f"source JSON not found: {reuse_json}"
                        logger.error("[Exp C] %s — skipping %s [%s]", msg, cond, routing_label)
                        _elog(f"ERROR  {cond} [{routing_label}]: {msg}")
                        run_status.append((f"{cond}[{routing_label}]", f"FAILED: {msg}"))
                        continue
                    r["label"] = f"{profile['label']}_{routing_label}"
                    r["routing_mode"] = routing_mode
                else:
                    if sim_env is None:
                        logger.error("[Exp C] sim_env not loaded, skipping %s [%s]", cond, routing_label)
                        run_status.append((f"{cond}[{routing_label}]", "FAILED: sim_env unavailable"))
                        continue

                    cfg_base, G, regions = sim_env
                    sim_cfg = cfg_base["simulation"]
                    cfg_run = _copy.deepcopy(cfg_base)
                    cfg_run["simulation"]["routing_mode"] = routing_mode
                    agents = _sample_agents(
                        regions=regions,
                        flows=sim_cfg["flows"],
                        n_agents=n_agents_run,
                        T=sim_cfg["T_s"],
                        seed=sim_cfg["seed"],
                        walking_speed=sim_cfg["walking_speed_ms"],
                        elderly_ratio=profile["elderly_ratio"],
                    )

                    tmp_dir = CH6 / "sim_cache" / f"{profile['condition']}_{routing_label}"
                    logger.info("[Exp C] Simulation %s [%s]: N=%d, seed=%d …",
                                cond, routing_label.upper(), n_agents_run, sim_cfg["seed"])
                    r = _run_simulation(
                        G, agents, cfg_run,
                        out_dir=tmp_dir,
                        routing_mode=routing_mode,
                        label=f"{profile['label']}_{routing_label}",
                        write_traj=False,
                    )
                    r["label"] = f"{profile['label']}_{routing_label}"
                    r["routing_mode"] = routing_mode
                    r["_n_agents"] = n_agents_run

                m = _summarise(r)
                m["condition"] = cond
                m["routing_mode"] = routing_label
                m["description"] = profile["description"]
                m["elderly_ratio"] = profile["elderly_ratio"]
                conn_load = _connector_load(r.get("edge_throughput", {}))
                m["top_connector"] = conn_load[0][0] if conn_load else "N/A"
                m["top_connector_crossings"] = conn_load[0][1] if conn_load else 0
                m["runtime_s"] = round(time.time() - t_profile, 2)

                mode_results[routing_label].append(m)
                run_status.append((f"{cond}[{routing_label}]", "OK"))
                logger.info(
                    "[Exp C] %s [%s] OK in %.1fs — arrive_rate=%.3f, "
                    "mean_tt=%.1f s, mean_wait=%.1f s, max_queue=%d, reroutes=%d",
                    cond, routing_label, m["runtime_s"], m["arrive_rate"],
                    m["mean_travel_time_s"], m["mean_wait_time_s"], m["max_queue"],
                    m["reroute_count"],
                )
                _elog(
                    f"  {cond} [{routing_label}]: arrive_rate={_f3(m['arrive_rate'])}  "
                    f"mean_tt={_f1(m['mean_travel_time_s'])}s  "
                    f"p95_tt={_f1(m['p95_travel_time_s'])}s  "
                    f"mean_wait={_f1(m['mean_wait_time_s'])}s  "
                    f"max_queue={m['max_queue']}  reroutes={m['reroute_count']}  "
                    f"runtime={m['runtime_s']}s"
                )

            except Exception as exc:
                logger.error("[Exp C] %s [%s] failed: %s", cond, routing_label, exc, exc_info=True)
                _elog(f"ERROR  {cond} [{routing_label}]: exception — {exc}")
                run_status.append((f"{cond}[{routing_label}]", f"FAILED: {exc}"))

    if not any(mode_results.values()):
        logger.error("[Exp C] No profile-mode runs succeeded — no CSV written")
        _elog("ERROR  No profile-mode runs produced results — aborting")
        return

    fieldnames = [
        "condition", "label", "description", "elderly_ratio",
        "n_agents", "n_completed", "n_failed", "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_travel_elderly_s", "mean_travel_normal_s",
        "mean_wait_time_s", "median_wait_time_s", "max_wait_time_s",
        "max_queue", "reroute_count",
        "top_connector", "top_connector_crossings", "runtime_s",
    ]

    def _write_mode_csv(path: Path, rows: list[dict]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    n_static = _write_mode_csv(out_static, mode_results.get("static", []))
    n_dynamic = _write_mode_csv(out_dynamic, mode_results.get("dynamic", []))
    logger.info("[Exp C] Wrote %s (%d profiles)", out_static.name, n_static)
    logger.info("[Exp C] Wrote %s (%d profiles)", out_dynamic.name, n_dynamic)

    table_results: list[dict] = []
    for profile in PROFILES:
        for routing_label, _ in ROUTING_MODES:
            row = next((r for r in mode_results[routing_label] if r["condition"] == profile["condition"]), None)
            if row is not None:
                table_results.append(row)

    table_rows: list[list] = [
        ["Metric"] + [f"{r['condition']} [{r['routing_mode']}]" for r in table_results],
        ["Routing mode"] + [r["routing_mode"] for r in table_results],
        ["Profile description"] + [r["description"] for r in table_results],
        ["Elderly ratio"] + [r["elderly_ratio"] for r in table_results],
        ["Number of agents"] + [r["n_agents"] for r in table_results],
        ["Completed agents"] + [r["n_completed"] for r in table_results],
        ["Failed agents"] + [r["n_failed"] for r in table_results],
        ["Arrival rate"] + [_f3(r["arrive_rate"]) for r in table_results],
        ["Mean travel time (s)"] + [_f1(r["mean_travel_time_s"]) for r in table_results],
        ["Median travel time (s)"] + [_f1(r["median_travel_time_s"]) for r in table_results],
        ["P95 travel time (s)"] + [_f1(r["p95_travel_time_s"]) for r in table_results],
        ["Mean travel — elderly (s)"] + [_f1(r["mean_travel_elderly_s"]) for r in table_results],
        ["Mean travel — normal (s)"] + [_f1(r["mean_travel_normal_s"]) for r in table_results],
        ["Mean waiting time (s)"] + [_f1(r["mean_wait_time_s"]) for r in table_results],
        ["Median waiting time (s)"] + [_f1(r["median_wait_time_s"]) for r in table_results],
        ["Max waiting time (s)"] + [_f1(r["max_wait_time_s"]) for r in table_results],
        ["Peak connector queue"] + [r["max_queue"] for r in table_results],
        ["Reroute events"] + [r["reroute_count"] for r in table_results],
        ["Top bottleneck connector"] + [r["top_connector"] for r in table_results],
        ["Bottleneck crossings"] + [r["top_connector_crossings"] for r in table_results],
        ["Simulation runtime (s)"] + [_f1(r["runtime_s"]) for r in table_results],
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)
    logger.info("[Exp C] Wrote %s", out_table.name)

    elapsed_total = time.time() - t0_total
    n_ok_static = len(mode_results.get("static", []))
    n_ok_dynamic = len(mode_results.get("dynamic", []))
    n_failed_runs = sum(1 for _, status in run_status if "FAILED" in status)

    _elog("")
    _elog("--- Profile run status ---")
    for cond, status in run_status:
        _elog(f"  {cond}: {status}")
    for ni in NOT_IMPLEMENTED:
        _elog(f"  {ni}: NOT IMPLEMENTED (absent from sample_agents())")
    _elog("")
    _elog("--- Output files ---")
    _elog(f"  Static data CSV:  {out_static}")
    _elog(f"  Dynamic data CSV: {out_dynamic}")
    _elog(f"  Table CSV:        {out_table}")
    _elog(f"  Log:              {exp_log}")
    _elog("")
    _elog("--- Summary ---")
    _elog(f"  Profiles found (implemented): {len(PROFILES)}")
    _elog(f"  Static runs OK:              {n_ok_static}/{len(PROFILES)}")
    _elog(f"  Dynamic runs OK:             {n_ok_dynamic}/{len(PROFILES)}")
    _elog(f"  Failed profile-mode runs:    {n_failed_runs}")
    _elog(f"  Profiles not implemented:    {len(NOT_IMPLEMENTED)}")
    _elog(f"    Not-implemented list:      {', '.join(NOT_IMPLEMENTED)}")
    _elog(f"  Total runtime: {elapsed_total:.1f}s")

    logger.info("[Exp C] Done in %.1fs — static=%d/%d OK, dynamic=%d/%d OK",
                elapsed_total, n_ok_static, len(PROFILES), n_ok_dynamic, len(PROFILES))


def run_experiment_D(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp D — Mixed-agent Simulation.

    Evaluates the same mixed crowd composition under BOTH static and dynamic
    routing, while preserving the original static output file.

    Outputs:
      outputs/ch6/data/experiment_D_mixed_agent_results.csv          (static)
      outputs/ch6/data/experiment_D_mixed_agent_results_dynamic.csv  (dynamic)
      outputs/ch6/tables/table_6_6_experiment_D_results.csv          (combined)
      outputs/ch6/logs/experiment_D.log
    """
    import copy as _copy
    import pickle as _pickle

    out_static  = DIRS["data"]   / "experiment_D_mixed_agent_results.csv"
    out_dynamic = DIRS["data"]   / "experiment_D_mixed_agent_results_dynamic.csv"
    out_table   = DIRS["tables"] / "table_6_6_experiment_D_results.csv"
    exp_log     = DIRS["logs"]   / "experiment_D.log"

    if args.skip_existing and out_static.exists() and out_dynamic.exists() and out_table.exists():
        logger.info("[Exp D] Skipping — outputs already exist")
        return

    logger.info("[Exp D] Starting: Mixed-agent Simulation")
    t0 = time.time()

    n_agents_run = 50 if args.quick else 200
    ELDERLY_RATIO = 0.2
    ROUTING_MODES = [
        ("static", "static"),
        ("dynamic", "dynamic"),
    ]

    exp_log.parent.mkdir(parents=True, exist_ok=True)

    def _elog(msg: str) -> None:
        with exp_log.open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")

    _elog("=== Experiment D: Mixed-agent Simulation ===")
    _elog(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _elog(f"Quick mode: {args.quick}  n_agents: {n_agents_run}")
    _elog(f"Routing modes: {[label for label, _ in ROUTING_MODES]}")
    _elog("")
    _elog("--- Profile audit ---")
    _elog("  IMPLEMENTED   normal       speed 0.95–1.05x base walking speed")
    _elog("  IMPLEMENTED   elderly      speed 0.55–0.75x base walking speed")
    _elog("  NOT_IMPL      wheelchair   absent from src/routing.py::sample_agents()")
    _elog("  NOT_IMPL      luggage      absent from src/routing.py::sample_agents()")
    _elog("  NOT_IMPL      accessibility-constrained  absent from sample_agents()")
    _elog("")
    _elog("--- Composition decision ---")
    _elog("  Target: 70% normal / 20% elderly / 10% accessibility-constrained")
    _elog("  Actual: 80% normal / 20% elderly  (10% accessibility slot vacant —")
    _elog("          profile not implemented; reallocated to normal as documented)")
    _elog(f"  elderly_ratio passed to sample_agents(): {ELDERLY_RATIO}")
    _elog("")
    _elog("--- Metric availability ---")
    _elog("  Per-profile travel time (mean/median/p95): AVAILABLE "
          "(elderly_travel_times / normal_travel_times)")
    _elog("  Per-profile wait time (mean/max): AVAILABLE "
          "(elderly_wait_times / normal_wait_times)")
    _elog("  Per-profile queue exposure: NOT_AVAILABLE "
          "(stair_queue_over_time is a global count, not labelled per agent_type)")
    _elog("  Per-profile connector usage: NOT_AVAILABLE "
          "(edge_throughput is not labelled per agent_type)")
    _elog("")

    sys.path.insert(0, str(ROOT))
    try:
        from src.utils import load_config as _load_cfg
        from src.routing import (
            define_semantic_regions as _def_regions,
            sample_agents as _sample_agents,
        )
        from src.simulation import run_simulation as _run_simulation
    except ImportError as exc:
        logger.error("[Exp D] Import failed: %s", exc)
        _elog(f"ERROR  Import failed: {exc}")
        return

    cfg_path   = ROOT / "config" / "experiment_config.yaml"
    graph_path = ROOT / "outputs" / "step3_graph" / "navigation_graph.gpickle"

    for _p, _name in [(cfg_path, "config"), (graph_path, "graph")]:
        if not _p.exists():
            logger.error("[Exp D] %s not found: %s", _name, _p)
            _elog(f"ERROR  {_name} not found: {_p}")
            return

    cfg_base = _load_cfg(cfg_path)
    logger.info("[Exp D] Loading graph …")
    with open(graph_path, "rb") as _gf:
        G = _pickle.load(_gf)
    logger.info("[Exp D] Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    regions = _def_regions(G, cfg_base)
    sim_cfg = cfg_base["simulation"]

    agents_template = _sample_agents(
        regions=regions,
        flows=sim_cfg["flows"],
        n_agents=n_agents_run,
        T=sim_cfg["T_s"],
        seed=sim_cfg["seed"],
        walking_speed=sim_cfg["walking_speed_ms"],
        elderly_ratio=ELDERLY_RATIO,
    )

    n_normal_spawned = sum(1 for a in agents_template if a["agent_type"] == "normal")
    n_elderly_spawned = sum(1 for a in agents_template if a["agent_type"] == "elderly")
    normal_share = round(n_normal_spawned / len(agents_template), 3) if agents_template else 0.0
    elderly_share = round(n_elderly_spawned / len(agents_template), 3) if agents_template else 0.0

    _elog(f"INFO   Agents spawned template: {len(agents_template)} total, "
          f"{n_normal_spawned} normal ({normal_share:.1%}), "
          f"{n_elderly_spawned} elderly ({elderly_share:.1%})")

    def _prof_metrics(tt: list, wt: list, n_spawned: int,
                      profile: str, share: float) -> dict:
        nc = len(tt)
        return {
            "profile": profile,
            "share_pct": round(share * 100, 1),
            "n_spawned": n_spawned,
            "n_completed": nc,
            "n_failed": max(0, n_spawned - nc),
            "arrive_rate": round(nc / n_spawned, 4) if n_spawned else float("nan"),
            "mean_travel_time_s": round(mean(tt), 2) if tt else float("nan"),
            "median_travel_time_s": round(median(tt), 2) if tt else float("nan"),
            "p95_travel_time_s": round(_p95(tt), 2) if tt else float("nan"),
            "mean_wait_time_s": round(mean(wt), 2) if wt else float("nan"),
            "max_wait_time_s": round(max(wt), 2) if wt else float("nan"),
            "queue_exposure": "not_available",
            "connector_usage": "not_available",
        }

    def _fv(v, dp=1):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if isinstance(v, str):
            return v
        try:
            return round(v, dp)
        except (TypeError, ValueError):
            return v

    mode_results: dict[str, dict] = {}

    for routing_label, routing_mode in ROUTING_MODES:
        cfg_run = _copy.deepcopy(cfg_base)
        cfg_run["simulation"]["routing_mode"] = routing_mode
        agents = _copy.deepcopy(agents_template)

        sim_dir = CH6 / "sim_cache" / f"D_mixed_{routing_label}"
        logger.info("[Exp D] Running simulation: N=%d, %s, seed=%d, elderly=%.0f%% …",
                    n_agents_run, routing_label, sim_cfg["seed"], ELDERLY_RATIO * 100)
        t_sim = time.time()
        try:
            result = _run_simulation(
                G, agents, cfg_run,
                out_dir=sim_dir,
                routing_mode=routing_mode,
                label=f"D_mixed_{routing_label}",
                write_traj=False,
            )
        except Exception as exc:
            logger.error("[Exp D] %s simulation failed: %s", routing_label, exc, exc_info=True)
            _elog(f"ERROR  {routing_label} simulation failed: {exc}")
            continue

        sim_elapsed = time.time() - t_sim
        logger.info("[Exp D] %s simulation finished in %.1fs", routing_label, sim_elapsed)
        _elog(f"INFO   [{routing_label}] Simulation runtime: {sim_elapsed:.1f}s")

        tt_all = result.get("travel_times", [])
        wt_all = result.get("wait_times", [])
        queue_s = result.get("stair_queue_over_time", [])

        n_completed = len(tt_all)
        n_failed = max(0, n_agents_run - n_completed)
        arrive_rate = result.get("arrive_rate", float("nan"))
        max_queue = max((q for _, q in queue_s), default=0) if queue_s else 0

        overall = {
            "condition": "D_mixed",
            "label": f"D_mixed_{routing_label}",
            "routing_mode": routing_label,
            "n_agents": n_agents_run,
            "n_normal_spawned": n_normal_spawned,
            "n_elderly_spawned": n_elderly_spawned,
            "normal_share": normal_share,
            "elderly_share": elderly_share,
            "n_completed": n_completed,
            "n_failed": n_failed,
            "arrive_rate": round(arrive_rate, 4),
            "mean_travel_time_s": round(mean(tt_all), 2) if tt_all else float("nan"),
            "median_travel_time_s": round(median(tt_all), 2) if tt_all else float("nan"),
            "p95_travel_time_s": round(_p95(tt_all), 2) if tt_all else float("nan"),
            "mean_wait_time_s": round(mean(wt_all), 2) if wt_all else float("nan"),
            "max_wait_time_s": round(max(wt_all), 2) if wt_all else float("nan"),
            "max_queue": max_queue,
            "reroute_count": len(result.get("replan_events", [])),
            "sim_runtime_s": round(sim_elapsed, 2),
        }

        normal_tt = result.get("normal_travel_times", [])
        elderly_tt = result.get("elderly_travel_times", [])
        normal_wt = result.get("normal_wait_times", [])
        elderly_wt = result.get("elderly_wait_times", [])

        if not (normal_wt or elderly_wt):
            _elog(f"WARN   [{routing_label}] elderly_wait_times / normal_wait_times absent from result")

        prof_normal = _prof_metrics(normal_tt, normal_wt, n_normal_spawned, "normal", normal_share)
        prof_elderly = _prof_metrics(elderly_tt, elderly_wt, n_elderly_spawned, "elderly", elderly_share)

        mode_results[routing_label] = {
            "overall": overall,
            "profiles": {
                "normal": prof_normal,
                "elderly": prof_elderly,
            },
        }

        _elog(f"  [{routing_label}] Overall: arrive_rate={_fv(overall['arrive_rate'],3)}  "
              f"mean_tt={_fv(overall['mean_travel_time_s'])}s  "
              f"mean_wait={_fv(overall['mean_wait_time_s'])}s  "
              f"max_queue={overall['max_queue']}  reroutes={overall['reroute_count']}")
        _elog(f"  [{routing_label}] Normal:  mean_tt={_fv(prof_normal['mean_travel_time_s'])}s  "
              f"mean_wait={_fv(prof_normal['mean_wait_time_s'])}s  "
              f"arrive_rate={_fv(prof_normal['arrive_rate'],3)}")
        _elog(f"  [{routing_label}] Elderly: mean_tt={_fv(prof_elderly['mean_travel_time_s'])}s  "
              f"mean_wait={_fv(prof_elderly['mean_wait_time_s'])}s  "
              f"arrive_rate={_fv(prof_elderly['arrive_rate'],3)}")

    if not mode_results:
        logger.error("[Exp D] No routing mode completed — no CSV written")
        _elog("ERROR  No routing mode completed — aborting")
        return

    overall_fields = [
        "condition", "label", "routing_mode",
        "n_agents", "n_normal_spawned", "n_elderly_spawned",
        "normal_share", "elderly_share",
        "n_completed", "n_failed", "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_wait_time_s", "max_wait_time_s",
        "max_queue", "reroute_count", "sim_runtime_s",
    ]
    profile_fields = [
        "profile", "share_pct", "n_spawned", "n_completed", "n_failed",
        "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_wait_time_s", "max_wait_time_s",
        "queue_exposure", "connector_usage",
    ]

    def _write_mode_data(path: Path, mode_result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w_all = csv.writer(f)
            w_all.writerow(["# SECTION: overall"])
            ow = csv.DictWriter(f, fieldnames=overall_fields)
            ow.writeheader()
            ow.writerow(mode_result["overall"])
            w_all.writerow([])
            w_all.writerow(["# SECTION: per_profile"])
            pw = csv.DictWriter(f, fieldnames=profile_fields)
            pw.writeheader()
            pw.writerow(mode_result["profiles"]["normal"])
            pw.writerow(mode_result["profiles"]["elderly"])

    if "static" in mode_results:
        _write_mode_data(out_static, mode_results["static"])
        logger.info("[Exp D] Wrote %s", out_static.name)
    if "dynamic" in mode_results:
        _write_mode_data(out_dynamic, mode_results["dynamic"])
        logger.info("[Exp D] Wrote %s", out_dynamic.name)

    s_over = mode_results.get("static", {}).get("overall", {})
    d_over = mode_results.get("dynamic", {}).get("overall", {})
    s_norm = mode_results.get("static", {}).get("profiles", {}).get("normal", {})
    d_norm = mode_results.get("dynamic", {}).get("profiles", {}).get("normal", {})
    s_eld = mode_results.get("static", {}).get("profiles", {}).get("elderly", {})
    d_eld = mode_results.get("dynamic", {}).get("profiles", {}).get("elderly", {})

    def _get(row: dict, key: str, dp: int = 1):
        if not row:
            return "N/A"
        return _fv(row.get(key), dp)

    table_rows = [
        (
            "Metric",
            "Overall [static]", "Overall [dynamic]",
            "Normal [static]", "Normal [dynamic]",
            "Elderly [static]", "Elderly [dynamic]",
        ),
        (
            "Routing mode",
            "static", "dynamic",
            "static", "dynamic",
            "static", "dynamic",
        ),
        (
            "Profile share",
            "80% + 20%", "80% + 20%",
            f"{normal_share:.0%}", f"{normal_share:.0%}",
            f"{elderly_share:.0%}", f"{elderly_share:.0%}",
        ),
        (
            "Number of agents",
            _get(s_over, "n_agents", 0), _get(d_over, "n_agents", 0),
            n_normal_spawned if s_norm else "N/A", n_normal_spawned if d_norm else "N/A",
            n_elderly_spawned if s_eld else "N/A", n_elderly_spawned if d_eld else "N/A",
        ),
        (
            "Completed agents",
            _get(s_over, "n_completed", 0), _get(d_over, "n_completed", 0),
            _get(s_norm, "n_completed", 0), _get(d_norm, "n_completed", 0),
            _get(s_eld, "n_completed", 0), _get(d_eld, "n_completed", 0),
        ),
        (
            "Failed agents",
            _get(s_over, "n_failed", 0), _get(d_over, "n_failed", 0),
            _get(s_norm, "n_failed", 0), _get(d_norm, "n_failed", 0),
            _get(s_eld, "n_failed", 0), _get(d_eld, "n_failed", 0),
        ),
        (
            "Arrival rate",
            _get(s_over, "arrive_rate", 3), _get(d_over, "arrive_rate", 3),
            _get(s_norm, "arrive_rate", 3), _get(d_norm, "arrive_rate", 3),
            _get(s_eld, "arrive_rate", 3), _get(d_eld, "arrive_rate", 3),
        ),
        (
            "Mean travel time (s)",
            _get(s_over, "mean_travel_time_s"), _get(d_over, "mean_travel_time_s"),
            _get(s_norm, "mean_travel_time_s"), _get(d_norm, "mean_travel_time_s"),
            _get(s_eld, "mean_travel_time_s"), _get(d_eld, "mean_travel_time_s"),
        ),
        (
            "Median travel time (s)",
            _get(s_over, "median_travel_time_s"), _get(d_over, "median_travel_time_s"),
            _get(s_norm, "median_travel_time_s"), _get(d_norm, "median_travel_time_s"),
            _get(s_eld, "median_travel_time_s"), _get(d_eld, "median_travel_time_s"),
        ),
        (
            "P95 travel time (s)",
            _get(s_over, "p95_travel_time_s"), _get(d_over, "p95_travel_time_s"),
            _get(s_norm, "p95_travel_time_s"), _get(d_norm, "p95_travel_time_s"),
            _get(s_eld, "p95_travel_time_s"), _get(d_eld, "p95_travel_time_s"),
        ),
        (
            "Mean waiting time (s)",
            _get(s_over, "mean_wait_time_s"), _get(d_over, "mean_wait_time_s"),
            _get(s_norm, "mean_wait_time_s"), _get(d_norm, "mean_wait_time_s"),
            _get(s_eld, "mean_wait_time_s"), _get(d_eld, "mean_wait_time_s"),
        ),
        (
            "Max waiting time (s)",
            _get(s_over, "max_wait_time_s"), _get(d_over, "max_wait_time_s"),
            _get(s_norm, "max_wait_time_s"), _get(d_norm, "max_wait_time_s"),
            _get(s_eld, "max_wait_time_s"), _get(d_eld, "max_wait_time_s"),
        ),
        (
            "Peak connector queue",
            _get(s_over, "max_queue", 0), _get(d_over, "max_queue", 0),
            "not_available", "not_available",
            "not_available", "not_available",
        ),
        (
            "Per-profile queue exposure",
            "not_available", "not_available",
            "not_available", "not_available",
            "not_available", "not_available",
        ),
        (
            "Per-profile connector usage",
            "not_available", "not_available",
            "not_available", "not_available",
            "not_available", "not_available",
        ),
        (
            "Reroute events",
            _get(s_over, "reroute_count", 0), _get(d_over, "reroute_count", 0),
            "—", "—",
            "—", "—",
        ),
        (
            "Simulation runtime (s)",
            _get(s_over, "sim_runtime_s"), _get(d_over, "sim_runtime_s"),
            "—", "—",
            "—", "—",
        ),
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)
    logger.info("[Exp D] Wrote %s", out_table.name)

    elapsed = time.time() - t0
    _elog("")
    _elog("--- Output files ---")
    _elog(f"  Static data CSV:  {out_static}")
    _elog(f"  Dynamic data CSV: {out_dynamic}")
    _elog(f"  Table CSV:        {out_table}")
    _elog(f"  Log:              {exp_log}")
    _elog("")
    _elog("--- Summary ---")
    _elog("  Profiles included:        normal, elderly")
    _elog(f"  Profile composition:      {normal_share:.0%} normal / {elderly_share:.0%} elderly")
    _elog(f"  Static run completed:     {'YES' if 'static' in mode_results else 'NO'}")
    _elog(f"  Dynamic run completed:    {'YES' if 'dynamic' in mode_results else 'NO'}")
    _elog("  Missing metrics:")
    _elog("    per-profile queue exposure  — not_available (global counter only)")
    _elog("    per-profile connector usage — not_available (edge_throughput untagged)")
    _elog(f"  Total runtime: {elapsed:.1f}s")

    logger.info("[Exp D] Done in %.1fs — static=%s dynamic=%s",
                elapsed,
                'OK' if 'static' in mode_results else 'FAILED',
                'OK' if 'dynamic' in mode_results else 'FAILED')


def run_experiment_E(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp E — Algorithm-threshold Sensitivity Analysis.

    Tests how congestion-aware dynamic replanning responds to changes in three
    simulation parameters read directly from config["simulation"]:

      replan_wait_threshold_s  — seconds of waiting before an agent re-routes
      congestion_alpha         — weight multiplier at full congestion (w = base × (1+α×cong))
      congestion_max_hops      — neighbourhood radius used by compute_congestion()

        All other settings (OD demand, N, seed, routing_mode, profiles) are
        held constant per graph branch. routing_mode is forced to "dynamic".

        Three threshold configurations plus one graph-ablation branch:
      E1 aggressive  : wait_thr=1.0 s, alpha=3.0, hops=12
                       → agents replan quickly at high congestion sensitivity
      E2 balanced    : wait_thr=2.0 s, alpha=2.0, hops=12
                       → default config; E2 reuses existing scenC_dynamic result
                         (no new simulation run needed)
      E3 conservative: wait_thr=6.0 s, alpha=1.0, hops=6
                       → agents tolerate long waits, low congestion weight,
                         shorter lookahead
            E4 sparse graph: balanced thresholds on a sparser, semantics-light graph
                                             with coarser sampling and boundary-fallback OD semantics

        Quick mode uses N=50 for new runs (E1, E3, E4); E2 reuse is always from
        the full-N=200 scenC_dynamic result.

    Unsupported / absent threshold parameters — documented in log, not faked:
      connector_load_threshold : not a config key in simulation.py
      replan_timer_enabled     : kept False (realistic blocked-only replanning)
      downstream_lookahead     : not separately configurable (merged into max_hops)

    Route-instability metric: fraction of completed agents with ≥1 replan event.
    Exact parameter values saved to outputs/ch6/configs/experiment_E_*.json.

    Outputs:
      outputs/ch6/data/experiment_E_algorithm_threshold_results.csv
      outputs/ch6/tables/table_6_7_experiment_E_results.csv
      outputs/ch6/logs/experiment_E.log
      outputs/ch6/configs/experiment_E_{setting}.json  (one per setting)
    """
    import copy as _copy

    out_data  = DIRS["data"]   / "experiment_E_algorithm_threshold_results.csv"
    out_table = DIRS["tables"] / "table_6_7_experiment_E_results.csv"
    exp_log   = DIRS["logs"]   / "experiment_E.log"

    if args.skip_existing and out_data.exists() and out_table.exists():
        logger.info("[Exp E] Skipping — outputs already exist")
        return

    logger.info("[Exp E] Starting: Algorithm-threshold Sensitivity Analysis")
    t0_total = time.time()

    n_agents_new = 50 if args.quick else 200
    sim_root     = ROOT / "outputs" / "step5_simulation"

    # Quick mode: refresh congestion weights every 10 steps (≈5 s real time) instead of
    # every step — reduces weight-dict rebuilds from 1200 to 120 and speeds up E1/E3
    # by ~8–10×.  The relative ordering of E1 vs E3 metrics is preserved.
    _cong_every = 10 if args.quick else 1

    # ── threshold settings ────────────────────────────────────────────────
    # Each entry: (condition, graph branch, reuse source, param overrides)
    SETTINGS = [
        {
            "condition":   "E1_aggressive",
            "graph_variant": "default",
            "description": "Low trigger: agents replan quickly at high congestion sensitivity",
            "reuse_json":  None,
            "overrides": {
                "replan_wait_threshold_s":   1.0,
                "congestion_alpha":          3.0,
                "congestion_max_hops":       12,
                "max_replans_per_step":      50,   # cap at N to prevent runaway O(N²)
                "congestion_recompute_every": _cong_every,
            },
        },
        {
            "condition":   "E2_balanced",
            "graph_variant": "default",
            "description": "Default balanced setting (reuses existing scenC_dynamic result)",
            "reuse_json":  sim_root / "scenC_dynamic" / "result_scenC.json",
            "reuse_summary": sim_root / "scenC_dynamic" / "summary.csv",
            "overrides": {
                "replan_wait_threshold_s": 2.0,
                "congestion_alpha":        2.0,
                "congestion_max_hops":     12,
                "max_replans_per_step":    9999,
            },
        },
        {
            "condition":   "E3_conservative",
            "graph_variant": "default",
            "description": "High trigger: agents tolerate long waits, low congestion weight",
            "reuse_json":  None,
            "overrides": {
                "replan_wait_threshold_s":   6.0,
                "congestion_alpha":          1.0,
                "congestion_max_hops":       6,
                "max_replans_per_step":      50,
                "congestion_recompute_every": _cong_every,
            },
        },
        {
            "condition":   "E4_sparse_semantic_light",
            "graph_variant": "sparse_semantic_light",
            "description": "Balanced thresholds on a sparse 4-neighbour graph with boundary-fallback OD semantics",
            "reuse_json":  None,
            "overrides": {
                "replan_wait_threshold_s":   2.0,
                "congestion_alpha":          2.0,
                "congestion_max_hops":       12,
                "max_replans_per_step":      50 if args.quick else 9999,
                "congestion_recompute_every": _cong_every,
            },
        },
    ]

    UNSUPPORTED = [
        ("connector_load_threshold", "not a config key in simulation.py"),
        ("downstream_lookahead",     "merged into congestion_max_hops — not separately configurable"),
        ("replan_timer_enabled",     "held False (blocked-only) for all runs; not varied"),
    ]

    exp_log.parent.mkdir(parents=True, exist_ok=True)
    DIRS["configs"].mkdir(parents=True, exist_ok=True)

    def _elog(msg: str) -> None:
        with exp_log.open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")

    _elog("=== Experiment E: Algorithm-threshold Sensitivity Analysis ===")
    _elog(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _elog(f"Quick mode: {args.quick}  n_agents_new_runs: {n_agents_new}")
    _elog("")
    _elog("--- Threshold parameters varied ---")
    _elog("  replan_wait_threshold_s  (seconds waiting before replanning)")
    _elog("  congestion_alpha         (congestion weight amplifier in routing cost)")
    _elog("  congestion_max_hops      (neighbourhood spread radius for congestion)")
    _elog("")
    _elog("--- Unsupported / absent threshold parameters ---")
    for param, reason in UNSUPPORTED:
        _elog(f"  {param}: {reason}")
    _elog("")
    _elog("--- Settings ---")
    for s in SETTINGS:
        src = str(s.get("reuse_json", "new simulation")) if s.get("reuse_json") else "new simulation"
        _elog(f"  {s['condition']}: {s['description']}")
        _elog(f"    graph_variant: {s.get('graph_variant', 'default')}")
        _elog(f"    overrides: {s['overrides']}")
        _elog(f"    source:    {src}")
    _elog("")

    # ── lazy-load simulation environment (only if new runs are needed) ────
    needs_new = any(s.get("reuse_json") is None for s in SETTINGS)
    variant_names = sorted({s.get("graph_variant", "default") for s in SETTINGS})
    variant_envs: dict[str, tuple] = {}
    sim_envs: dict[str, tuple] = {}

    for variant_name in variant_names:
        try:
            variant_envs[variant_name] = _load_or_build_graph_variant(variant_name, logger)
        except Exception as exc:
            logger.error("[Exp E] Could not prepare graph variant %s: %s", variant_name, exc)
            _elog(f"ERROR  graph variant {variant_name}: {exc}")

    if needs_new:
        sys.path.insert(0, str(ROOT))
        try:
            from src.routing import (
                sample_agents as _sample_agents,
            )
            from src.simulation import run_simulation as _run_simulation
        except ImportError as exc:
            logger.error("[Exp E] Import failed: %s", exc)
            _elog(f"ERROR  Import failed: {exc}")
            return

        for variant_name, env in variant_envs.items():
            cfg_variant, g_variant, regions_variant, meta_variant = env
            _sc = cfg_variant["simulation"]
            agents_template = _sample_agents(
                regions=regions_variant,
                flows=_sc["flows"],
                n_agents=n_agents_new,
                T=_sc["T_s"],
                seed=_sc["seed"],
                walking_speed=_sc["walking_speed_ms"],
                elderly_ratio=_sc.get("elderly_ratio", 0.2),
            )
            sim_envs[variant_name] = (cfg_variant, g_variant, regions_variant, agents_template, meta_variant)
    else:
        _sample_agents = _run_simulation = None

    # ── helper: extract reroute stats from result ─────────────────────────
    def _reroute_stats(result: dict, n_completed: int) -> tuple[float, float]:
        """Return (mean_reroutes_per_agent, instability_fraction)."""
        events = result.get("replan_events", [])
        total  = len(events)
        if n_completed == 0:
            return float("nan"), float("nan")
        mean_rpla = round(total / n_completed, 3)
        # fraction of completed agents with ≥1 reroute
        replanned_agents = len({e.get("agent_id") for e in events if e.get("agent_id")})
        instability = round(replanned_agents / n_completed, 3)
        return mean_rpla, instability

    def _fv(v, dp=1):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if isinstance(v, str):
            return v
        try:
            return round(v, dp)
        except (TypeError, ValueError):
            return v

    # ── per-setting loop ──────────────────────────────────────────────────
    setting_results: list[dict] = []
    run_status: list[tuple[str, str]] = []

    for setting in SETTINGS:
        cond = setting["condition"]
        variant_name = setting.get("graph_variant", "default")
        overrides = setting["overrides"]
        logger.info("[Exp E] Processing %s …", cond)
        t_setting = time.time()

        if variant_name not in variant_envs:
            msg = f"graph variant unavailable: {variant_name}"
            logger.error("[Exp E] %s", msg)
            _elog(f"ERROR  {cond}: {msg}")
            run_status.append((cond, f"FAILED: {msg}"))
            continue

        cfg_variant, _G_variant, _regions_variant, variant_meta = variant_envs[variant_name]

        # Save config snapshot to configs/
        cfg_snap = {
            "condition":       cond,
            "description":     setting["description"],
            "routing_mode":    "dynamic",
            "graph_variant":   variant_name,
            "n_agents":        n_agents_new if not setting.get("reuse_json") else 200,
            "seed":            42,
            "parameters":      dict(overrides),
            "graph_summary":   variant_meta,
            "fixed_params": {
                "replan_timer_enabled": False,
                "dt_s":                 0.5,
                "T_s":                  600,
                "walking_speed_ms":     1.2,
                "elderly_ratio":        0.2,
            },
            "unsupported_params": [p for p, _ in UNSUPPORTED],
        }
        snap_path = DIRS["configs"] / f"experiment_E_{cond}.json"
        with snap_path.open("w", encoding="utf-8") as _sf:
            json.dump(cfg_snap, _sf, indent=2, ensure_ascii=False)

        try:
            if setting.get("reuse_json"):
                # ── E2: load existing scenC_dynamic result ─────────────────
                r = _load_result(
                    setting["reuse_json"],
                    setting.get("reuse_summary", Path("__none__")),
                    logger, "E",
                )
                if r is None:
                    msg = f"reuse JSON not found: {setting['reuse_json']}"
                    logger.error("[Exp E] %s — skipping %s", msg, cond)
                    _elog(f"ERROR  {cond}: {msg}")
                    run_status.append((cond, f"FAILED: {msg}"))
                    continue
                r["label"] = cond
                n_for_run  = r["_n_agents"]

            else:
                # ── E1 / E3: new simulation with patched config ────────────
                if variant_name not in sim_envs:
                    run_status.append((cond, f"FAILED: sim_env unavailable for {variant_name}"))
                    continue

                cfg_base, G, regions, agents_template, variant_meta = sim_envs[variant_name]
                # Deep-copy config and apply parameter overrides
                cfg_run = _copy.deepcopy(cfg_base)
                cfg_run["simulation"].update(overrides)
                # Ensure dynamic routing is used
                cfg_run["simulation"]["routing_mode"] = "dynamic"

                sim_dir = CH6 / "sim_cache" / f"{cond}_{variant_name}"
                logger.info(
                    "[Exp E] Simulation %s: graph=%s, N=%d, dynamic, wait_thr=%.1fs, "
                    "alpha=%.1f, hops=%d …",
                    cond, variant_name, n_agents_new,
                    overrides["replan_wait_threshold_s"],
                    overrides["congestion_alpha"],
                    overrides["congestion_max_hops"],
                )
                r = _run_simulation(
                    G, list(agents_template),   # fresh copy so mutations don't leak
                    cfg_run,
                    out_dir=sim_dir,
                    routing_mode="dynamic",
                    label=cond,
                    write_traj=False,
                )
                r["label"]     = cond
                r["_n_agents"] = n_agents_new
                n_for_run      = n_agents_new

            # ── compute metrics ────────────────────────────────────────────
            m = _summarise(r)
            m["condition"]   = cond
            m["description"] = setting["description"]
            m["graph_variant"] = variant_name
            m["graph_description"] = variant_meta["description"]
            m["graph_nodes"] = variant_meta["total_nodes"]
            m["graph_edges"] = variant_meta["total_edges"]
            m["entrance_region_size"] = variant_meta["entrance_region_size"]
            m["platform_region_size"] = variant_meta["platform_region_size"]
            m["blind_path_nodes"] = variant_meta["blind_path_nodes"]
            m["replan_wait_threshold_s"] = overrides["replan_wait_threshold_s"]
            m["congestion_alpha"]        = overrides["congestion_alpha"]
            m["congestion_max_hops"]     = overrides["congestion_max_hops"]

            mean_rpl, instability = _reroute_stats(r, m["n_completed"])
            m["mean_reroutes_per_agent"] = mean_rpl
            m["instability_fraction"]    = instability
            m["runtime_s"]               = round(time.time() - t_setting, 2)
            m["data_source"] = (
                f"{setting['reuse_json']} [graph={variant_name}]" if setting.get("reuse_json")
                else f"new_simulation (N={n_for_run}, graph={variant_name})"
            )

            setting_results.append(m)
            run_status.append((cond, "OK"))
            logger.info(
                "[Exp E] %s OK — graph=%s, arrive_rate=%.3f, mean_tt=%.1fs, "
                "mean_wait=%.1fs, reroutes=%d (%.2f/agent), instability=%.1f%%",
                cond, variant_name, m["arrive_rate"],
                m["mean_travel_time_s"], m["mean_wait_time_s"],
                m["reroute_count"], mean_rpl, instability * 100,
            )
            _elog(
                f"  {cond} [{variant_name}]: arrive_rate={_fv(m['arrive_rate'],3)}  "
                f"mean_tt={_fv(m['mean_travel_time_s'])}s  "
                f"mean_wait={_fv(m['mean_wait_time_s'])}s  "
                f"max_queue={m['max_queue']}  "
                f"reroutes={m['reroute_count']}  "
                f"reroutes/agent={_fv(mean_rpl,2)}  "
                f"instability={_fv(instability*100,1)}%  "
                f"runtime={m['runtime_s']}s"
            )

        except Exception as exc:
            logger.error("[Exp E] %s failed: %s", cond, exc, exc_info=True)
            _elog(f"ERROR  {cond}: exception — {exc}")
            run_status.append((cond, f"FAILED: {exc}"))

    # ── guard ─────────────────────────────────────────────────────────────
    if not setting_results:
        logger.error("[Exp E] No settings succeeded — no CSV written")
        _elog("ERROR  No settings produced results — aborting")
        return

    # ── write experiment_E_algorithm_threshold_results.csv ────────────────
    fieldnames = [
        "condition", "description", "graph_variant", "graph_description",
        "graph_nodes", "graph_edges", "entrance_region_size", "platform_region_size", "blind_path_nodes",
        "replan_wait_threshold_s", "congestion_alpha", "congestion_max_hops",
        "n_agents", "n_completed", "n_failed", "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_wait_time_s", "median_wait_time_s", "max_wait_time_s",
        "max_queue", "reroute_count",
        "mean_reroutes_per_agent", "instability_fraction",
        "runtime_s", "data_source",
    ]
    out_data.parent.mkdir(parents=True, exist_ok=True)
    with out_data.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(setting_results)
    logger.info("[Exp E] Wrote %s (%d settings)", out_data.name, len(setting_results))

    # ── write table_6_7 ───────────────────────────────────────────────────
    col_hdr = ["Metric"] + [r["condition"] for r in setting_results]
    table_rows: list = [
        col_hdr,
        ["Description"]                  + [r["description"]                   for r in setting_results],
        ["Graph variant"]                + [r["graph_variant"]                 for r in setting_results],
        ["Graph nodes"]                  + [r["graph_nodes"]                   for r in setting_results],
        ["Graph edges"]                  + [r["graph_edges"]                   for r in setting_results],
        ["Entrance region size"]         + [r["entrance_region_size"]          for r in setting_results],
        ["Platform region size"]         + [r["platform_region_size"]          for r in setting_results],
        ["Blind-path nodes"]             + [r["blind_path_nodes"]              for r in setting_results],
        ["Wait trigger (s)"]             + [r["replan_wait_threshold_s"]        for r in setting_results],
        ["Congestion alpha"]             + [r["congestion_alpha"]               for r in setting_results],
        ["Congestion max hops"]          + [r["congestion_max_hops"]            for r in setting_results],
        ["Number of agents"]             + [r["n_agents"]                       for r in setting_results],
        ["Completed agents"]             + [r["n_completed"]                    for r in setting_results],
        ["Failed agents"]                + [r["n_failed"]                       for r in setting_results],
        ["Arrival rate"]                 + [_fv(r["arrive_rate"],         3)    for r in setting_results],
        ["Mean travel time (s)"]         + [_fv(r["mean_travel_time_s"],  1)    for r in setting_results],
        ["Median travel time (s)"]       + [_fv(r["median_travel_time_s"],1)    for r in setting_results],
        ["P95 travel time (s)"]          + [_fv(r["p95_travel_time_s"],   1)    for r in setting_results],
        ["Mean waiting time (s)"]        + [_fv(r["mean_wait_time_s"],    1)    for r in setting_results],
        ["Max waiting time (s)"]         + [_fv(r["max_wait_time_s"],     1)    for r in setting_results],
        ["Peak connector queue"]         + [r["max_queue"]                      for r in setting_results],
        ["Total reroute events"]         + [r["reroute_count"]                  for r in setting_results],
        ["Mean reroutes per agent"]      + [_fv(r["mean_reroutes_per_agent"],2) for r in setting_results],
        ["Route instability (% agents)"] + [_fv(r["instability_fraction"]*100 if isinstance(r["instability_fraction"], float) and not math.isnan(r["instability_fraction"]) else float("nan"), 1) for r in setting_results],
        ["Simulation runtime (s)"]       + [_fv(r["runtime_s"],           1)    for r in setting_results],
        ["Data source"]                  + [r["data_source"]                    for r in setting_results],
    ]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)
    logger.info("[Exp E] Wrote %s", out_table.name)

    # ── finalize log ──────────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    n_ok = sum(1 for _, s in run_status if s == "OK")

    _elog("")
    _elog("--- Run status ---")
    for cond, status in run_status:
        _elog(f"  {cond}: {status}")
    _elog("")
    _elog("--- Output files ---")
    _elog(f"  Data CSV:  {out_data}")
    _elog(f"  Table CSV: {out_table}")
    _elog(f"  Log:       {exp_log}")
    for s in SETTINGS:
        snap = DIRS['configs'] / f"experiment_E_{s['condition']}.json"
        _elog(f"  Config:    {snap}")
    _elog("")
    _elog("--- Summary ---")
    _elog(f"  Settings run: {n_ok}/{len(SETTINGS)}")
    _elog("  Thresholds varied: replan_wait_threshold_s, congestion_alpha, congestion_max_hops")
    _elog("  Graph branches: " + ", ".join(sorted({s['graph_variant'] for s in SETTINGS})))
    _elog("  Unsupported params: " + ", ".join(p for p, _ in UNSUPPORTED))
    _elog(f"  Total runtime: {elapsed:.1f}s")

    logger.info("[Exp E] Done in %.1fs — %d/%d settings OK",
                elapsed, n_ok, len(SETTINGS))


def _experiment_f_demand_levels(quick: bool) -> list[tuple[str, int]]:
    """Return the demand ladder for Experiment F.

    Quick mode stays intentionally short because Experiment F now sweeps two
    graph branches as well as two routing modes. The full ladder is preserved
    for non-quick runs when a collapse-level capacity trace is required.
    """
    quick_levels = [
        ("F1_low", 50),
        ("F2_medium", 100),
        ("F3_high", 200),
        ("F4_very_high", 300),
        ("F5_stress", 500),
    ]
    full_levels = quick_levels + [
        ("F6_overload", 750),
        ("F7_extreme", 1000),
    ]
    if quick:
        return quick_levels
    return full_levels + [
        ("F8_threshold", 1250),
        ("F9_threshold_plus", 1500),
        ("F10_near_collapse", 1750),
        ("F11_collapse", 2000),
        ("F12_breakdown", 2500),
        ("F13_extreme_plus", 3000),
    ]


def run_experiment_F(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Exp F — Building Capacity / Agent-load Threshold Test.

        Runs two routing variants (static + dynamic) on two graph branches
        (baseline default vs sparse_semantic_light) at increasing demand levels to
        identify how graph construction / semantic modelling choices shift the
        navigability and saturation threshold.

    Fixed across all runs:
            - threshold settings inside each graph branch
      - algorithm threshold settings (default config values)
      - pedestrian profile mix (elderly_ratio=0.2, same as config)
      - random seed (42)

    Varied:
      - number of agents (demand level): low/medium/high/very_high [+stress in full]
            - routing mode: static vs dynamic
            - graph branch: default vs sparse_semantic_light

    Saturation detection (model-based, not official capacity):
      Any of the following triggers the flag:
        1. arrive_rate < 0.95
        2. mean_wait increases ≥ 50 % relative to previous level
        3. max_queue increases ≥ 100 % relative to previous level
        4. n_failed > 0
      Evaluated independently per routing mode.

    Outputs:
            outputs/ch6/data/experiment_F_capacity_threshold_results_static.csv   (all graph branches)
            outputs/ch6/data/experiment_F_capacity_threshold_results_dynamic.csv  (all graph branches)
            outputs/ch6/tables/table_6_8_experiment_F_results.csv  (combined comparison by graph + routing)
      outputs/ch6/logs/experiment_F.log
    """
    import copy as _copy

    out_static  = DIRS["data"]   / "experiment_F_capacity_threshold_results_static.csv"
    out_dynamic = DIRS["data"]   / "experiment_F_capacity_threshold_results_dynamic.csv"
    out_table   = DIRS["tables"] / "table_6_8_experiment_F_results.csv"
    exp_log     = DIRS["logs"]   / "experiment_F.log"

    if args.skip_existing and out_static.exists() and out_dynamic.exists() and out_table.exists():
        logger.info("[Exp F] Skipping — outputs already exist")
        return

    logger.info("[Exp F] Starting: Building Capacity / Agent-load Threshold Test (static + dynamic)")
    t0_total = time.time()

    # ── demand levels ─────────────────────────────────────────────────────
    # Extend beyond N=1000 so the arrive-rate curve can actually reach the
    # horizontal critical threshold on the plot instead of stopping early.
    DEMAND_LEVELS = _experiment_f_demand_levels(args.quick)

    # Routing modes to sweep: (label, routing_mode_str)
    ROUTING_MODES = [
        ("static",  "static"),
        ("dynamic", "dynamic"),
    ]

    GRAPH_BRANCHES = [
        ("default", GRAPH_VARIANTS["default"]["description"]),
        ("sparse_semantic_light", GRAPH_VARIANTS["sparse_semantic_light"]["description"]),
    ]

    # Per-level runtime cap — allow the extended high-load points to finish.
    MAX_LEVEL_S = 2400.0 if args.quick else 5400.0

    # Soft saturation: flag in output but keep running.
    # Critical saturation: arrive_rate < 0.70 — stop further levels for this mode.
    ARRIVE_SAT          = 0.95   # soft flag
    ARRIVE_CRITICAL_SAT = 0.70   # hard stop (system collapsed)
    WAIT_JUMP_FRAC      = 0.50
    QUEUE_JUMP_FRAC     = 1.00

    # ── log file helpers ──────────────────────────────────────────────────
    exp_log.parent.mkdir(parents=True, exist_ok=True)

    def _elog(msg: str) -> None:
        with exp_log.open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")

    def _f1(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        try:
            return f"{float(v):.1f}"
        except (TypeError, ValueError):
            return str(v)

    def _f3(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    # Fresh log
    with exp_log.open("w", encoding="utf-8") as _f:
        _f.write("=== Experiment F: Building Capacity / Agent-load Threshold Test ===\n")
        _f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    _elog("--- Design ---")
    _elog("  Varies: number of agents × routing mode × graph branch")
    _elog("  Fixed:  default thresholds within each graph branch, elderly_ratio=0.2, seed=42")
    _elog(f"  Quick mode: {args.quick}")
    _elog(f"  Demand levels: {[(lbl, n) for lbl, n in DEMAND_LEVELS]}")
    _elog(f"  Routing modes: {[lbl for lbl, _ in ROUTING_MODES]}")
    _elog(f"  Graph branches: {[name for name, _ in GRAPH_BRANCHES]}")
    _elog(f"  Per-level runtime cap: {MAX_LEVEL_S}s")
    _elog("")
    _elog("--- Saturation criteria ---")
    _elog(f"  Soft (flag only):  arrive_rate < {ARRIVE_SAT}")
    _elog(f"                     mean_wait relative jump >= {WAIT_JUMP_FRAC*100:.0f}%")
    _elog(f"                     max_queue relative jump >= {QUEUE_JUMP_FRAC*100:.0f}%")
    _elog(f"                     n_failed > 0")
    _elog(f"  Critical (abort):  arrive_rate < {ARRIVE_CRITICAL_SAT}  (system collapse)")
    _elog(f"                     OR runtime > {MAX_LEVEL_S}s per level")
    _elog("")

    # ── load simulation environment ───────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    try:
        from src.routing import (
            sample_agents as _sample_agents,
        )
        from src.simulation import run_simulation as _run_simulation
    except ImportError as exc:
        logger.error("[Exp F] Import failed: %s", exc)
        _elog(f"ERROR  Import failed: {exc}")
        return

    variant_envs: dict[str, tuple] = {}
    for variant_name, _ in GRAPH_BRANCHES:
        try:
            variant_envs[variant_name] = _load_or_build_graph_variant(variant_name, logger)
        except Exception as exc:
            logger.error("[Exp F] Could not prepare graph variant %s: %s", variant_name, exc)
            _elog(f"ERROR  graph variant {variant_name}: {exc}")

    if not variant_envs:
        logger.error("[Exp F] No graph variants available")
        _elog("ERROR  No graph variants available")
        return

    STATUS_COMPLETED = "completed"
    STATUS_FAILED    = "failed"
    STATUS_SKIPPED   = "skipped"

    DATA_FIELDS = [
        "graph_variant", "graph_description", "graph_nodes", "graph_edges",
        "entrance_region_size", "platform_region_size", "blind_path_nodes",
        "routing_mode",
        "demand_label", "n_agents", "n_completed", "n_failed",
        "arrive_rate",
        "mean_travel_time_s", "median_travel_time_s", "p95_travel_time_s",
        "mean_travel_elderly_s", "mean_travel_normal_s",
        "mean_wait_time_s", "max_wait_time_s",
        "max_queue", "top_connector", "top_connector_load",
        "reroute_count", "deadlock_count", "runtime_s",
        "sat_triggered", "sat_reason",
    ]

    # keyed by graph_variant -> routing_label -> list[rows]
    all_completed: dict[str, dict[str, list[dict]]] = {}
    all_saturated_at: dict[str, dict[str, str | None]] = {}

    # ── outer loops: graph branch × routing mode ──────────────────────────
    for graph_variant, _graph_desc in GRAPH_BRANCHES:
        if graph_variant not in variant_envs:
            continue

        cfg_base, G, regions, variant_meta = variant_envs[graph_variant]
        sim_cfg = cfg_base["simulation"]
        SEED = sim_cfg["seed"]
        T_S = sim_cfg["T_s"]
        SPEED = sim_cfg["walking_speed_ms"]
        ELD_RATIO = sim_cfg.get("elderly_ratio", 0.2)

        _elog(f"\n=== Graph variant: {graph_variant} ===")
        _elog(f"  Description: {variant_meta['description']}")
        _elog(
            f"  Graph stats: nodes={variant_meta['total_nodes']}, edges={variant_meta['total_edges']}, "
            f"entrance_region={variant_meta['entrance_region_size']}, "
            f"platform_region={variant_meta['platform_region_size']}, "
            f"blind_path_nodes={variant_meta['blind_path_nodes']}"
        )
        _elog(f"  Config: seed={SEED}, T_s={T_S}, speed={SPEED}, elderly_ratio={ELD_RATIO}")

        all_completed.setdefault(graph_variant, {})
        all_saturated_at.setdefault(graph_variant, {})

        for routing_label, routing_mode in ROUTING_MODES:
            _elog(f"\n=== Routing mode: {routing_label} [{graph_variant}] ===")
            logger.info("[Exp F] ── Graph=%s | Routing=%s ──", graph_variant, routing_label.upper())

            level_results: list[dict] = []
            prev_metrics: dict | None = None
            saturated_at: str | None = None
            abort_remaining = False

            for level_label, n_agents in DEMAND_LEVELS:
                if abort_remaining:
                    logger.info("[Exp F] %s/%s/%s: skipping — previous level exceeded cap",
                                graph_variant, routing_label, level_label)
                    _elog(f"  {level_label} (N={n_agents}): SKIPPED — previous level hit runtime cap")
                    level_results.append({
                        "graph_variant": graph_variant,
                        "routing_mode": routing_label,
                        "demand_label": level_label,
                        "n_agents":     n_agents,
                        "status":       STATUS_SKIPPED,
                        "skip_reason":  "previous level exceeded runtime cap",
                    })
                    continue

                logger.info("[Exp F] %s / %s / %s: N=%d, seed=%d …",
                            graph_variant, routing_label, level_label, n_agents, SEED)
                t_level = time.time()

                try:
                    agents = _sample_agents(
                        regions=regions,
                        flows=sim_cfg["flows"],
                        n_agents=n_agents,
                        T=T_S,
                        seed=SEED,
                        walking_speed=SPEED,
                        elderly_ratio=ELD_RATIO,
                    )

                    # Deep-copy config so dynamic overrides don't bleed into static
                    cfg_run = _copy.deepcopy(cfg_base)
                    if routing_mode == "dynamic":
                        cfg_run["simulation"]["routing_mode"] = "dynamic"
                        # Quick mode: more aggressive speedup for high-N saturation sweep.
                        # recompute_every=20 + max_replans=20 keeps N=1000 feasible (<10 min).
                        if args.quick:
                            cfg_run["simulation"]["congestion_recompute_every"] = 20
                            cfg_run["simulation"]["max_replans_per_step"] = 20

                    sim_dir = CH6 / "sim_cache" / f"{level_label}_{routing_label}_{graph_variant}"
                    result  = _run_simulation(
                        G, agents, cfg_run,
                        out_dir=sim_dir,
                        routing_mode=routing_mode,
                        label=f"{level_label}_{routing_label}_{graph_variant}",
                        write_traj=False,
                    )
                    result["_n_agents"] = n_agents

                    lvl_elapsed = time.time() - t_level

                    m = _summarise(result)

                    conn_load      = _connector_load(result.get("edge_throughput", {}))
                    top_conn       = conn_load[0][0] if conn_load else "N/A"
                    top_conn_count = conn_load[0][1] if conn_load else 0

                    n_failed_agents = max(0, n_agents - m["n_completed"])

                    row: dict = {
                        "graph_variant":         graph_variant,
                        "graph_description":     variant_meta["description"],
                        "graph_nodes":           variant_meta["total_nodes"],
                        "graph_edges":           variant_meta["total_edges"],
                        "entrance_region_size":  variant_meta["entrance_region_size"],
                        "platform_region_size":  variant_meta["platform_region_size"],
                        "blind_path_nodes":      variant_meta["blind_path_nodes"],
                        "routing_mode":          routing_label,
                        "demand_label":          level_label,
                        "n_agents":              n_agents,
                        "n_completed":           m["n_completed"],
                        "n_failed":              n_failed_agents,
                        "arrive_rate":           round(m["arrive_rate"], 4),
                        "mean_travel_time_s":    round(m["mean_travel_time_s"], 2),
                        "median_travel_time_s":  round(m["median_travel_time_s"], 2),
                        "p95_travel_time_s":     round(m["p95_travel_time_s"], 2),
                        "mean_travel_elderly_s": (
                            round(m["mean_travel_elderly_s"], 2)
                            if not math.isnan(m["mean_travel_elderly_s"]) else "N/A"),
                        "mean_travel_normal_s":  (
                            round(m["mean_travel_normal_s"], 2)
                            if not math.isnan(m["mean_travel_normal_s"]) else "N/A"),
                        "mean_wait_time_s":      round(m["mean_wait_time_s"], 2),
                        "max_wait_time_s":       round(m["max_wait_time_s"], 2),
                        "max_queue":             m["max_queue"],
                        "top_connector":         top_conn,
                        "top_connector_load":    top_conn_count,
                        "reroute_count":         m["reroute_count"],
                        "deadlock_count":        n_failed_agents,
                        "runtime_s":             round(lvl_elapsed, 2),
                        "status":                STATUS_COMPLETED,
                    }

                    # ── saturation detection ───────────────────────────────
                    sat_reasons: list[str] = []
                    if row["arrive_rate"] < ARRIVE_SAT:
                        sat_reasons.append(f"arrive_rate={row['arrive_rate']:.3f} < {ARRIVE_SAT}")
                    if n_failed_agents > 0:
                        sat_reasons.append(f"n_failed={n_failed_agents}")
                    if prev_metrics is not None:
                        pw = prev_metrics["mean_wait_time_s"]
                        pq = prev_metrics["max_queue"]
                        if isinstance(pw, (int, float)) and pw > 0:
                            dw = row["mean_wait_time_s"] - pw
                            if dw / pw >= WAIT_JUMP_FRAC:
                                sat_reasons.append(
                                    f"mean_wait jump {dw:.1f}s (+{dw/pw*100:.0f}%)")
                        if isinstance(pq, int) and pq > 0:
                            dq = row["max_queue"] - pq
                            if dq / pq >= QUEUE_JUMP_FRAC:
                                sat_reasons.append(
                                    f"max_queue jump {dq} (+{dq/pq*100:.0f}%)")

                    row["saturated"]     = bool(sat_reasons)
                    row["sat_reason"]    = "; ".join(sat_reasons) if sat_reasons else ""
                    row["sat_triggered"] = "YES" if sat_reasons else "no"

                    if sat_reasons and saturated_at is None:
                        saturated_at = level_label
                        logger.info("[Exp F] [%s/%s] Model-based saturation at %s: %s",
                                    graph_variant, routing_label, level_label, "; ".join(sat_reasons))

                    level_results.append(row)
                    prev_metrics = row

                    logger.info(
                        "[Exp F] [%s/%s] %s OK in %.1fs — arrive=%.3f, mean_tt=%.1fs, "
                        "mean_wait=%.1fs, max_queue=%d, replans=%d, sat=%s",
                        graph_variant, routing_label, level_label, lvl_elapsed,
                        row["arrive_rate"], row["mean_travel_time_s"],
                        row["mean_wait_time_s"], row["max_queue"],
                        row["reroute_count"], row["sat_triggered"],
                    )
                    _elog(
                        f"  {level_label} (N={n_agents}): {STATUS_COMPLETED} in {lvl_elapsed:.1f}s\n"
                        f"    arrive={_f3(row['arrive_rate'])}  mean_tt={_f1(row['mean_travel_time_s'])}s"
                        f"  p95={_f1(row['p95_travel_time_s'])}s  mean_wait={_f1(row['mean_wait_time_s'])}s"
                        f"  max_queue={row['max_queue']}  replans={row['reroute_count']}"
                        f"  n_failed={n_failed_agents}  sat={row['sat_triggered']}"
                        + (f"\n    sat_reasons: {row['sat_reason']}" if sat_reasons else "")
                    )

                    # Hard stop 1: runtime cap exceeded
                    if lvl_elapsed > MAX_LEVEL_S:
                        logger.warning(
                            "[Exp F] [%s/%s] %s took %.1fs > cap %.1fs — aborting remaining levels",
                            graph_variant, routing_label, level_label, lvl_elapsed, MAX_LEVEL_S)
                        _elog(f"WARNING  [{graph_variant}/{routing_label}] {level_label} exceeded cap {MAX_LEVEL_S}s"
                              f" — remaining levels skipped")
                        abort_remaining = True

                    # Hard stop 2: critical saturation (system collapsed)
                    if row["arrive_rate"] < ARRIVE_CRITICAL_SAT:
                        logger.info(
                            "[Exp F] [%s/%s] Critical saturation at %s (arrive=%.3f < %.2f) "
                            "— no need to run higher loads",
                            graph_variant, routing_label, level_label, row["arrive_rate"], ARRIVE_CRITICAL_SAT)
                        _elog(f"INFO   [{graph_variant}/{routing_label}] Critical saturation at {level_label} "
                              f"(arrive={row['arrive_rate']:.3f}) — stopping sweep for this mode")
                        abort_remaining = True

                except Exception as exc:
                    lvl_elapsed = time.time() - t_level
                    logger.error("[Exp F] [%s/%s] %s failed after %.1fs: %s",
                                 graph_variant, routing_label, level_label, lvl_elapsed, exc, exc_info=True)
                    _elog(f"  {level_label} (N={n_agents}): FAILED after {lvl_elapsed:.1f}s — {exc}")
                    level_results.append({
                        "graph_variant": graph_variant,
                        "routing_mode": routing_label,
                        "demand_label": level_label,
                        "n_agents":     n_agents,
                        "status":       STATUS_FAILED,
                        "skip_reason":  str(exc),
                    })

            completed = [r for r in level_results if r.get("status") == STATUS_COMPLETED]
            all_completed[graph_variant][routing_label] = completed
            all_saturated_at[graph_variant][routing_label] = saturated_at

            _elog(f"  Completed: {[r['demand_label'] for r in completed]}")
            _elog(f"  Saturation: {saturated_at or 'not reached'}")

    # ── guard: need at least one routing mode with ≥1 completed level ────
    if not any(rows for by_mode in all_completed.values() for rows in by_mode.values()):
        logger.error("[Exp F] No levels completed in any routing mode — no CSV written")
        _elog("ERROR  No levels completed")
        return

    # ── write per-routing-mode data CSVs ──────────────────────────────────
    def _write_data_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=DATA_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in DATA_FIELDS})
        return len(rows)

    static_rows = []
    dynamic_rows = []
    for graph_variant, _ in GRAPH_BRANCHES:
        static_rows.extend(all_completed.get(graph_variant, {}).get("static", []))
        dynamic_rows.extend(all_completed.get(graph_variant, {}).get("dynamic", []))

    static_rows.sort(key=lambda row: (row.get("graph_variant", ""), row.get("n_agents", 0)))
    dynamic_rows.sort(key=lambda row: (row.get("graph_variant", ""), row.get("n_agents", 0)))

    n_s = _write_data_csv(out_static, static_rows)
    n_d = _write_data_csv(out_dynamic, dynamic_rows)
    logger.info("[Exp F] Wrote %s (%d rows)", out_static.name,  n_s)
    logger.info("[Exp F] Wrote %s (%d rows)", out_dynamic.name, n_d)

    # ── build combined comparison thesis table ────────────────────────────
    def _fmt(v, decimals=1) -> str:
        if v is None or v == "" or v == "N/A":
            return "N/A"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    table_rows: list[tuple] = []
    table_rows.append((
        "Graph variant", "Routing", "Demand level", "N agents", "Completed", "Failed",
        "Arrive rate", "Mean TT (s)", "P95 TT (s)",
        "Mean wait (s)", "Max queue", "Replans",
        "Top connector", "Connector load", "Saturated", "Notes",
    ))

    for graph_variant, _ in GRAPH_BRANCHES:
        for routing_label, _ in ROUTING_MODES:
            completed_rows = all_completed.get(graph_variant, {}).get(routing_label, [])
            sat_point = all_saturated_at.get(graph_variant, {}).get(routing_label)
            if not completed_rows:
                table_rows.append((
                    graph_variant, routing_label, "— no completed levels —",
                    "", "", "", "", "", "", "", "", "", "", "", "", "",
                ))
                continue
            for r in completed_rows:
                table_rows.append((
                    graph_variant,
                    routing_label,
                    r["demand_label"],
                    r["n_agents"],
                    r["n_completed"],
                    r["n_failed"],
                    _fmt(r["arrive_rate"], 3),
                    _fmt(r["mean_travel_time_s"]),
                    _fmt(r["p95_travel_time_s"]),
                    _fmt(r["mean_wait_time_s"]),
                    r["max_queue"],
                    r.get("reroute_count", 0),
                    r.get("top_connector", "N/A"),
                    r.get("top_connector_load", "N/A"),
                    r["sat_triggered"],
                    r.get("sat_reason", ""),
                ))
            table_rows.append((
                "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            ))
            table_rows.append((
                graph_variant,
                f"[{routing_label}] Saturation point",
                sat_point if sat_point else "not reached",
                "Criteria (soft): arrive_rate<0.95 OR Δwait≥50% OR Δqueue≥100% OR n_failed>0 | Critical abort: arrive_rate<0.70",
                "", "", "", "", "", "", "", "", "", "", "", "",
            ))
            table_rows.append(("", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""))

    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(table_rows)
    logger.info("[Exp F] Wrote %s", out_table.name)

    # ── finalise experiment log ───────────────────────────────────────────
    elapsed_total = time.time() - t0_total

    _elog("")
    _elog("--- Summary ---")
    _elog(f"  Demand levels tested: {[lbl for lbl, _ in DEMAND_LEVELS]}")
    for graph_variant, _ in GRAPH_BRANCHES:
        for routing_label, _ in ROUTING_MODES:
            compl = all_completed.get(graph_variant, {}).get(routing_label, [])
            sat   = all_saturated_at.get(graph_variant, {}).get(routing_label)
            _elog(f"  [{graph_variant}/{routing_label}] completed={[r['demand_label'] for r in compl]}, sat={sat or 'not reached'}")
    _elog("")
    _elog("--- Missing metrics ---")
    _elog("  spawn_interval: not applicable — fixed-N mode")
    _elog("  connector_saturation: top_connector_load used as proxy")
    _elog("  deadlock_count: n_failed used as proxy")
    _elog("")
    _elog("--- Output paths ---")
    _elog(f"  static CSV:  {out_static}")
    _elog(f"  dynamic CSV: {out_dynamic}")
    _elog(f"  table CSV:   {out_table}")
    _elog(f"  log:         {exp_log}")
    _elog(f"\nTotal runtime: {elapsed_total:.1f}s")

    n_total = sum(len(rows) for by_mode in all_completed.values() for rows in by_mode.values())
    n_modes = len(DEMAND_LEVELS)
    logger.info(
        "[Exp F] Done in %.1fs — %d/%d graph-mode-level combos completed",
        elapsed_total,
        n_total,
        len(GRAPH_BRANCHES) * len(ROUTING_MODES) * n_modes,
    )


def _load_experiment_F_data() -> tuple[list[dict], list[dict]]:
    """Load static and dynamic CSVs for Experiment F; return (static_rows, dynamic_rows)."""
    import csv as _csv
    static_path  = DIRS["data"] / "experiment_F_capacity_threshold_results_static.csv"
    dynamic_path = DIRS["data"] / "experiment_F_capacity_threshold_results_dynamic.csv"
    static_rows: list[dict] = []
    dynamic_rows: list[dict] = []
    if static_path.exists():
        with static_path.open(encoding="utf-8") as f:
            static_rows = list(_csv.DictReader(f))
    if dynamic_path.exists():
        with dynamic_path.open(encoding="utf-8") as f:
            dynamic_rows = list(_csv.DictReader(f))
    return static_rows, dynamic_rows


def _fig_6_10_capacity_threshold(logger: logging.Logger) -> None:
    """Fig 6.10 – 4-panel capacity threshold: arrive_rate, mean_wait, max_queue, mean_TT vs N."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("[Fig 6.10] matplotlib not available – skipping")
        return

    static_rows, dynamic_rows = _load_experiment_F_data()
    if not static_rows or not dynamic_rows:
        logger.warning("[Fig 6.10] CSV data missing – skipping")
        return

    def _extract(rows: list[dict]) -> dict[str, list]:
        out: dict[str, list] = {"N": [], "arrive": [], "mean_wait": [], "max_queue": [], "mean_tt": [], "label": []}
        for r in rows:
            out["N"].append(int(r["n_agents"]))
            out["arrive"].append(float(r["arrive_rate"]))
            out["mean_wait"].append(float(r["mean_wait_time_s"]))
            out["max_queue"].append(int(r["max_queue"]))
            out["mean_tt"].append(float(r["mean_travel_time_s"]))
            out["label"].append(r["demand_label"])
        return out

    s = _extract(static_rows)
    d = _extract(dynamic_rows)

    COLOR_S = "#2196F3"   # blue  – static
    COLOR_D = "#FF5722"   # orange – dynamic
    SAT_CLR = "#BDBDBD"   # grey  – saturation region

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Experiment F: Capacity Threshold Analysis\n"
                 "Static vs Dynamic Routing Across Demand Levels",
                 fontsize=13, fontweight="bold", y=1.01)

    panels = [
        (axes[0, 0], "Arrival Rate", s["arrive"], d["arrive"],
         "Arrive rate (fraction)", True),
        (axes[0, 1], "Mean Wait Time", s["mean_wait"], d["mean_wait"],
         "Mean wait time (s)", False),
        (axes[1, 0], "Max Queue Length", s["max_queue"], d["max_queue"],
         "Max queue (agents)", False),
        (axes[1, 1], "Mean Travel Time", s["mean_tt"], d["mean_tt"],
         "Mean travel time (s)", False),
    ]

    for ax, title, y_s, y_d, ylabel, is_arrive in panels:
        # Saturation onset shading (N≥100)
        ax.axvspan(100, max(s["N"]) * 1.05, color=SAT_CLR, alpha=0.15,
                   label="Saturated region (N≥100)")
        ax.plot(s["N"], y_s, "o-", color=COLOR_S, linewidth=2, markersize=6,
                label="Static routing", zorder=3)
        ax.plot(d["N"], y_d, "s--", color=COLOR_D, linewidth=2, markersize=6,
                label="Dynamic routing", zorder=3)
        if is_arrive:
            min_arrive = min(min(s["arrive"]), min(d["arrive"]))
            lower_bound = max(0.0, min(0.65, min_arrive - 0.05))
            ax.axhline(0.95, color="#9E9E9E", linestyle=":", linewidth=1.2,
                       label="Soft-sat threshold (0.95)")
            ax.axhline(0.70, color="#F44336", linestyle=":", linewidth=1.2,
                       label="Critical threshold (0.70)")
            ax.set_ylim(lower_bound, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of agents (N)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(s["N"])
        ax.set_xticklabels([str(n) for n in s["N"]], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = DIRS["figures"] / "fig_6_10_capacity_threshold.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Fig 6.10] Saved → %s", out_path)


def _fig_6_5_static_dynamic_comparison(logger: logging.Logger) -> None:
    """Fig 6.5 – Grouped bar: static vs dynamic at each demand level for arrive, wait, TT."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("[Fig 6.5] matplotlib/numpy not available – skipping")
        return

    static_rows, dynamic_rows = _load_experiment_F_data()
    if not static_rows or not dynamic_rows:
        logger.warning("[Fig 6.5] CSV data missing – skipping")
        return

    labels = [r["demand_label"] for r in static_rows]
    n_agents = [int(r["n_agents"]) for r in static_rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Fig 6.5 – Static vs Dynamic Routing: Key Metrics by Demand Level",
                 fontsize=12, fontweight="bold")

    panels = [
        (axes[0], "Arrival Rate", "arrive_rate", "Arrive rate (fraction)", False),
        (axes[1], "Mean Wait Time (s)", "mean_wait_time_s", "Mean wait (s)", False),
        (axes[2], "Mean Travel Time (s)", "mean_travel_time_s", "Mean travel time (s)", False),
    ]

    COLOR_S = "#2196F3"
    COLOR_D = "#FF5722"

    for ax, title, field, ylabel, _ in panels:
        s_vals = [float(r[field]) for r in static_rows]
        d_vals = [float(r[field]) for r in dynamic_rows]
        b1 = ax.bar(x - w / 2, s_vals, w, label="Static", color=COLOR_S, alpha=0.85)
        b2 = ax.bar(x + w / 2, d_vals, w, label="Dynamic", color=COLOR_D, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lb}\n(N={n})" for lb, n in zip(labels, n_agents)],
                           fontsize=7, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if field == "arrive_rate":
            ax.set_ylim(0.8, 1.02)
            ax.axhline(0.95, color="grey", linestyle=":", linewidth=1)

    plt.tight_layout()
    out_path = DIRS["figures"] / "fig_6_5_static_dynamic_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Fig 6.5] Saved → %s", out_path)


def generate_figures(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Generate Figures 6.1–6.11."""
    if args.no_plots:
        logger.info("[Figures] Skipped (--no-plots)")
        return
    logger.info("[Figures] Starting figure generation (Fig 6.1–6.11)")
    try:
        try:
            from . import ch6_figure_builder as _fig_builder
        except ImportError:
            import ch6_figure_builder as _fig_builder
    except ImportError as exc:
        logger.error("[Figures] Could not import figure builder: %s", exc)
        return

    generated = _fig_builder.generate_all_ch6_figures(
        root=ROOT,
        ch6=CH6,
        dirs=DIRS,
        logger=logger,
    )
    logger.info("[Figures] Done — %d files generated", len(generated))


def write_summary(logger: logging.Logger) -> None:
    """Write outputs/ch6/chapter6_results_summary.md."""
    logger.info("[Summary] TODO: write chapter6_results_summary.md from computed metrics")
    log_path = DIRS["logs"] / "summary.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("TODO  Write chapter6_results_summary.md\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Create output directories
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(DIRS["logs"] / "run_ch6.log")
    logger.info("=== Chapter 6 Experiment Runner ===")
    logger.info("Experiments: %s | quick=%s | skip_existing=%s | "
                "no_plots=%s | plots_only=%s",
                args.experiments, args.quick, args.skip_existing,
                args.no_plots, args.plots_only)

    t0 = time.time()

    dispatch = {
        "A": run_experiment_A,
        "B": run_experiment_B,
        "C": run_experiment_C,
        "D": run_experiment_D,
        "E": run_experiment_E,
        "F": run_experiment_F,
    }

    if not args.plots_only:
        for exp in args.experiments:
            dispatch[exp](args, logger)

    generate_figures(args, logger)
    write_summary(logger)

    elapsed = time.time() - t0
    logger.info("=== Done in %.1f s ===", elapsed)


if __name__ == "__main__":
    main()
