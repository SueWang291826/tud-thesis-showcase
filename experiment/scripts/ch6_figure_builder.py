from __future__ import annotations

import csv
import json
import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATIC_COLOR = "#1f77b4"
DYNAMIC_COLOR = "#d95f02"
LEVEL_COLORS = {
    "F1": "#2a9d8f",
    "F3": "#457b9d",
    "F4": "#e76f51",
    "STAIR": "#6d597a",
    "ESCALATOR": "#e9c46a",
}
EDGE_TYPE_COLORS = {
    "floor": "#c7c7c7",
    "stair": "#e76f51",
    "escalator": "#2a9d8f",
    "elevator": "#457b9d",
    "fare_gate": "#f4a261",
    "entrance": "#264653",
    "psd_door": "#7f7f7f",
    "anchor_snap": "#9d4edd",
}
PROFILE_COLORS = {
    "C1_normal": "#2a9d8f",
    "C2_elderly": "#bc6c25",
    "C3_mixed": "#6c757d",
    "normal": "#2a9d8f",
    "elderly": "#bc6c25",
}
THRESHOLD_COLORS = {
    "E1_aggressive": "#d62828",
    "E2_balanced": "#457b9d",
    "E3_conservative": "#2a9d8f",
    "E4_sparse_semantic_light": "#bc6c25",
}
GRAPH_VARIANT_COLORS = {
    "default": "#457b9d",
    "sparse_semantic_light": "#bc6c25",
}
GRAPH_VARIANT_LABELS = {
    "default": "default graph",
    "sparse_semantic_light": "sparse semantics-light graph",
}
GRAPH_VARIANT_SHORT = {
    "default": "default",
    "sparse_semantic_light": "sparse",
}


def generate_all_ch6_figures(
    root: Path,
    ch6: Path,
    dirs: dict[str, Path],
    logger: logging.Logger,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfb",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })

    graph_variants = _load_graph_variants(root, logger)
    graph = graph_variants.get("default")
    graph_artifacts = _prepare_graph_artifacts(graph) if graph is not None else None
    graph_variant_artifacts = {
        variant_name: _prepare_graph_artifacts(variant_graph)
        for variant_name, variant_graph in graph_variants.items()
    }
    datasets = _load_datasets(root, dirs, logger)
    datasets["graph_variants"] = graph_variants
    datasets["graph_variant_artifacts"] = graph_variant_artifacts

    figure_builders = [
        ("fig_6_1_pipeline_outputs", _fig_6_1_pipeline_outputs),
        ("fig_6_2_graph_overview", _fig_6_2_graph_overview),
        ("fig_6_3_graph_statistics", _fig_6_3_graph_statistics),
        ("fig_6_4_baseline_static_heatmap", _fig_6_4_baseline_static_heatmap),
        ("fig_6_5_static_dynamic_comparison", _fig_6_5_static_dynamic_comparison),
        ("fig_6_6_route_redistribution", _fig_6_6_route_redistribution),
        ("fig_6_7_profile_comparison", _fig_6_7_profile_comparison),
        ("fig_6_8_mixed_agent_results", _fig_6_8_mixed_agent_results),
        ("fig_6_9_algorithm_threshold_sensitivity", _fig_6_9_algorithm_threshold_sensitivity),
        ("fig_6_10_capacity_threshold", _fig_6_10_capacity_threshold),
        ("fig_6_11_overall_findings_summary", _fig_6_11_overall_findings_summary),
        ("fig_6_12_baseline_bottlenecks", _fig_6_12_baseline_bottlenecks),
        ("fig_6_13_capacity_breakdown", _fig_6_13_capacity_breakdown),
    ]

    generated: list[Path] = []
    for stem, builder in figure_builders:
        try:
            out_paths = builder(
                root=root,
                dirs=dirs,
                datasets=datasets,
                graph=graph,
                graph_artifacts=graph_artifacts,
                plt=plt,
                np=np,
                line_collection_cls=LineCollection,
                logger=logger,
            )
            generated.extend(out_paths)
        except Exception as exc:  # pragma: no cover - visualization guard
            logger.exception("[Figures] %s failed: %s", stem, exc)

    manifest = dirs["figures"] / "figure_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["figure_file"])
        for path in generated:
            writer.writerow([str(path)])
    logger.info("[Figures] Generated %d files across %d figure groups", len(generated), len(figure_builders))
    logger.info("[Figures] Manifest saved -> %s", manifest)
    return generated


def _load_datasets(root: Path, dirs: dict[str, Path], logger: logging.Logger) -> dict[str, Any]:
    step5 = root / "outputs" / "step5_simulation"
    data = {
        "A_metrics": _read_metric_csv(dirs["data"] / "experiment_A_static_results.csv"),
        "A_result": _read_json(step5 / "scenB_static" / "result_scenB.json"),
        "A_traj": _read_jsonl_points(step5 / "scenB_static" / "traj_agents.jsonl"),
        "B_rows": _read_csv_dicts(dirs["data"] / "experiment_B_static_dynamic_comparison.csv"),
        "B_static_result": _read_json(step5 / "scenB_static" / "result_scenB.json"),
        "B_dynamic_result": _read_json(step5 / "scenC_dynamic" / "result_scenC.json"),
        "B_static_traj": _read_jsonl_points(step5 / "scenB_static" / "traj_agents.jsonl"),
        "B_dynamic_traj": _read_jsonl_points(step5 / "scenC_dynamic" / "traj_agents.jsonl"),
        "C_static": _read_csv_dicts(dirs["data"] / "experiment_C_single_profile_results.csv"),
        "C_dynamic": _read_csv_dicts(dirs["data"] / "experiment_C_single_profile_results_dynamic.csv"),
        "D_static": _read_sectioned_csv(dirs["data"] / "experiment_D_mixed_agent_results.csv"),
        "D_dynamic": _read_sectioned_csv(dirs["data"] / "experiment_D_mixed_agent_results_dynamic.csv"),
        "E_rows": _read_csv_dicts(dirs["data"] / "experiment_E_algorithm_threshold_results.csv"),
        "F_static": _read_csv_dicts(dirs["data"] / "experiment_F_capacity_threshold_results_static.csv"),
        "F_dynamic": _read_csv_dicts(dirs["data"] / "experiment_F_capacity_threshold_results_dynamic.csv"),
    }
    logger.info("[Figures] Loaded datasets for experiments A-F")
    return data


def _graph_variant_path(root: Path, variant_name: str) -> Path:
    if variant_name == "default":
        return root / "outputs" / "step3_graph" / "navigation_graph.gpickle"
    return root / "outputs" / "ch6" / "graph_variants" / variant_name / "navigation_graph.gpickle"


def _load_graph_variants(root: Path, logger: logging.Logger) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant_name in ["default", "sparse_semantic_light"]:
        path = _graph_variant_path(root, variant_name)
        if not path.exists():
            logger.warning("[Figures] Graph variant missing, skipping: %s", path)
            continue
        with path.open("rb") as fh:
            variants[variant_name] = pickle.load(fh)
    return variants


def _prepare_graph_artifacts(graph) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    segments_by_type: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    segments_by_level: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    edge_lookup: dict[str, tuple[list[tuple[float, float]], str, str]] = {}
    x_vals: list[float] = []
    y_vals: list[float] = []

    for node_id, attrs in graph.nodes(data=True):
        if "x" not in attrs or "y" not in attrs:
            continue
        x = float(attrs["x"])
        y = float(attrs["y"])
        x_vals.append(x)
        y_vals.append(y)
        nodes.append({
            "id": node_id,
            "x": x,
            "y": y,
            "level": attrs.get("level", "?"),
            "node_type": attrs.get("node_type", "?"),
            "is_blind_path": bool(attrs.get("is_blind_path", False)),
            "usable": bool(attrs.get("usable", False)),
            "clearance": _as_float(attrs.get("clearance")),
            "surface_type": attrs.get("surface_type", "unknown"),
        })

    for u, v, attrs in graph.edges(data=True):
        if u not in graph.nodes or v not in graph.nodes:
            continue
        nu = graph.nodes[u]
        nv = graph.nodes[v]
        if "x" not in nu or "x" not in nv or "y" not in nu or "y" not in nv:
            continue
        seg = [(float(nu["x"]), float(nu["y"])), (float(nv["x"]), float(nv["y"]))]
        edge_type = attrs.get("edge_type", "other")
        level = attrs.get("level", "?")
        segments_by_type[edge_type].append(seg)
        segments_by_level[level].append(seg)
        edge_lookup[f"{u}|{v}"] = (seg, edge_type, level)
        edge_lookup[f"{v}|{u}"] = (seg, edge_type, level)

    return {
        "nodes": nodes,
        "segments_by_type": dict(segments_by_type),
        "segments_by_level": dict(segments_by_level),
        "edge_lookup": edge_lookup,
        "bounds": (min(x_vals), max(x_vals), min(y_vals), max(y_vals)) if x_vals and y_vals else (0.0, 1.0, 0.0, 1.0),
        "node_level_counts": Counter(n["level"] for n in nodes),
        "node_type_counts": Counter(n["node_type"] for n in nodes),
    }


def _read_metric_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = list(csv.reader(path.open(encoding="utf-8")))
    return {row[0]: row[1] for row in rows[1:] if len(row) >= 2}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _group_rows_by_graph_variant(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("graph_variant") or "default"].append(row)
    for variant_rows in grouped.values():
        variant_rows.sort(key=lambda row: _as_float(row.get("n_agents")))
    return dict(grouped)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl_points(path: Path, stride: int = 3) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    points: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if stride > 1 and idx % stride != 0:
                continue
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            x = row.get("x")
            y = row.get("y")
            if x is None or y is None:
                continue
            points.append((float(x), float(y)))
    return points


def _read_sectioned_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}
    sections: dict[str, list[dict[str, str]]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("# SECTION:"):
            idx += 1
            continue
        name = line.split(":", 1)[1].strip()
        idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            break
        header = next(csv.reader([lines[idx]]))
        idx += 1
        rows: list[dict[str, str]] = []
        while idx < len(lines):
            raw = lines[idx]
            if not raw.strip():
                idx += 1
                break
            if raw.strip().startswith("# SECTION:"):
                break
            values = next(csv.reader([raw]))
            rows.append({header[i]: values[i] if i < len(values) else "" for i in range(len(header))})
            idx += 1
        sections[name] = rows
    return sections


def _save_figure(fig, out_path: Path, dpi: int = 180) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.canvas.draw_idle()
    fig.clf()
    return [png_path, svg_path]


def _sort_levels(levels: list[str]) -> list[str]:
    def _key(value: str) -> tuple[int, str]:
        if value.startswith("F") and value[1:].isdigit():
            return (0, f"{int(value[1:]):04d}")
        return (1, value)
    return sorted(levels, key=_key)


def _as_float(value: Any) -> float:
    try:
        if value in (None, "", "N/A", "nan"):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _draw_base_plan(ax, graph_artifacts, line_collection_cls, edge_alpha: float = 0.12) -> None:
    floor_segments = graph_artifacts["segments_by_type"].get("floor", [])
    if floor_segments:
        lc = line_collection_cls(floor_segments, colors="#b8b8b8", linewidths=0.25, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)
    x_min, x_max, y_min, y_max = graph_artifacts["bounds"]
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_edge_overlay(ax, graph_artifacts, throughput: dict[str, float], np, line_collection_cls,
                       cmap: str, title: str, top_n: int = 700, diverging: bool = False) -> Any:
    _draw_base_plan(ax, graph_artifacts, line_collection_cls, edge_alpha=0.08)
    items: list[tuple[list[tuple[float, float]], float]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for key, value in throughput.items():
        info = graph_artifacts["edge_lookup"].get(key)
        if info is None:
            continue
        seg = info[0]
        seg_key = tuple(seg)
        if seg_key in seen:
            continue
        seen.add(seg_key)
        items.append((seg, float(value)))
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    items = items[:top_n]
    if not items:
        ax.set_title(title)
        return None
    segments = [item[0] for item in items]
    values = np.array([item[1] for item in items], dtype=float)
    width_scale = np.abs(values)
    width_scale = width_scale / max(width_scale.max(), 1.0)
    lc = line_collection_cls(segments, cmap=cmap, linewidths=0.8 + 3.0 * width_scale, alpha=0.92, zorder=3)
    lc.set_array(values)
    if diverging:
        vmax = max(float(np.max(np.abs(values))), 1.0)
        lc.set_clim(-vmax, vmax)
    ax.add_collection(lc)
    ax.set_title(title)
    return lc


def _top_connector_edges(graph_artifacts, throughput: dict[str, int], limit: int = 10) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for key, count in throughput.items():
        info = graph_artifacts["edge_lookup"].get(key)
        if info is None:
            continue
        _, edge_type, _ = info
        if edge_type == "floor":
            continue
        items.append((_short_edge_label(key), int(count)))
    items.sort(key=lambda item: item[1], reverse=True)
    dedup: list[tuple[str, int]] = []
    seen: set[str] = set()
    for label, count in items:
        if label in seen:
            continue
        seen.add(label)
        dedup.append((label, count))
        if len(dedup) >= limit:
            break
    return dedup


def _short_edge_label(edge_key: str) -> str:
    parts = edge_key.split("|")
    for part in parts:
        if any(token in part for token in ("stair", "elev", "escalator", "gate", "entrance")):
            return part[:36]
    return edge_key[:36]


def _fig_6_1_pipeline_outputs(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    if graph_artifacts is None:
        return []
    nodes = graph_artifacts["nodes"]
    levels = _sort_levels(list(graph_artifacts["node_level_counts"].keys()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.1 - Processed station graph outputs", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for level in levels:
        subset = [n for n in nodes if n["level"] == level and n["node_type"] == "floor"]
        if not subset:
            continue
        ax.scatter([n["x"] for n in subset], [n["y"] for n in subset], s=3, alpha=0.35,
                   label=level, color=LEVEL_COLORS.get(level, "#808080"))
    ax.set_title("Walkable nodes by level")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    _draw_base_plan(ax, graph_artifacts, line_collection_cls)
    important_types = {
        "entrance": "#264653",
        "door_platform": "#457b9d",
        "fare_gate_entry": "#f4a261",
        "fare_gate_exit": "#f4a261",
        "stair_step": "#e76f51",
        "escalator_step": "#2a9d8f",
        "elevator_interior": "#6d597a",
    }
    for node_type, color in important_types.items():
        subset = [n for n in nodes if n["node_type"] == node_type]
        if not subset:
            continue
        ax.scatter([n["x"] for n in subset], [n["y"] for n in subset], s=10, alpha=0.8, color=color, label=node_type)
    ax.set_title("Extracted connectors, gates, and anchors")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1, 0]
    blind_nodes = [n for n in nodes if n["is_blind_path"]]
    normal_nodes = [n for n in nodes if not n["is_blind_path"] and n["node_type"] == "floor"]
    ax.scatter([n["x"] for n in normal_nodes], [n["y"] for n in normal_nodes], s=2, alpha=0.16, color="#cfcfcf", label="regular floor")
    if blind_nodes:
        ax.scatter([n["x"] for n in blind_nodes], [n["y"] for n in blind_nodes], s=10, alpha=0.9, color="#c1121f", label="blind-path nodes")
    ax.set_title("Accessibility-tagged blind path nodes")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    clearances = [n["clearance"] for n in nodes if not math.isnan(n["clearance"])]
    usable = sum(1 for n in nodes if n["usable"])
    unusable = len(nodes) - usable
    ax.hist(clearances, bins=30, color="#457b9d", alpha=0.8)
    ax.axvline(float(np.median(clearances)), color="#d95f02", linestyle=":", linewidth=1.5, label=f"median = {float(np.median(clearances)):.2f} m")
    ax.set_title("Clearance distribution across sampled nodes")
    ax.set_xlabel("clearance (m)")
    ax.set_ylabel("node count")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f"usable: {usable}\nunusable: {unusable}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_1_pipeline_outputs")


def _fig_6_2_graph_overview(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    variant_artifacts = datasets.get("graph_variant_artifacts", {})
    variants = [variant for variant in ["default", "sparse_semantic_light"] if variant in variant_artifacts]
    if not variants and graph_artifacts is not None:
        variant_artifacts = {"default": graph_artifacts}
        variants = ["default"]
    if not variants:
        return []
    out_paths: list[Path] = []
    for variant_name in variants:
        artifacts = variant_artifacts[variant_name]
        x_min, x_max, y_min, y_max = artifacts["bounds"]
        for level in ["F1", "F3", "F4"]:
            fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
            segments = artifacts["segments_by_level"].get(level, [])
            if segments:
                lc = line_collection_cls(
                    segments,
                    colors=LEVEL_COLORS.get(level, "#999999"),
                    linewidths=0.28,
                    alpha=0.4,
                )
                ax.add_collection(lc)
            subset = [n for n in artifacts["nodes"] if n["level"] == level and n["node_type"] == "floor"]
            ax.scatter(
                [n["x"] for n in subset],
                [n["y"] for n in subset],
                s=3,
                alpha=0.18,
                color=LEVEL_COLORS.get(level, "#999999"),
            )
            connector_subset = [n for n in artifacts["nodes"] if n["level"] == level and n["node_type"] != "floor"]
            ax.scatter(
                [n["x"] for n in connector_subset],
                [n["y"] for n in connector_subset],
                s=16,
                alpha=0.9,
                color="#111111",
            )
            ax.set_title(
                f"Fig 6.2 - {GRAPH_VARIANT_LABELS.get(variant_name, variant_name.replace('_', ' '))} | {level}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_aspect("equal")
            ax.set_xlim(x_min - 2, x_max + 2)
            ax.set_ylim(y_min - 2, y_max + 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.015,
                0.05,
                f"variant: {GRAPH_VARIANT_SHORT.get(variant_name, variant_name)}\nlevel: {level}\n"
                f"nodes: {sum(1 for n in artifacts['nodes'] if n['level'] == level)}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
            )
            plt.tight_layout()
            out_paths.extend(
                _save_figure(
                    fig,
                    dirs["figures"] / f"fig_6_2_graph_overview_{variant_name}_{level}",
                    dpi=360,
                )
            )
    return out_paths


def _fig_6_3_graph_statistics(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    variant_graphs = datasets.get("graph_variants", {})
    variant_artifacts = datasets.get("graph_variant_artifacts", {})
    variants = [variant for variant in ["default", "sparse_semantic_light"] if variant in variant_graphs and variant in variant_artifacts]
    if not variants and graph is not None and graph_artifacts is not None:
        variant_graphs = {"default": graph}
        variant_artifacts = {"default": graph_artifacts}
        variants = ["default"]
    if not variants:
        return []

    summaries: dict[str, dict[str, Any]] = {}
    for variant_name in variants:
        variant_graph = variant_graphs[variant_name]
        artifacts = variant_artifacts[variant_name]
        usable_count = sum(1 for node in artifacts["nodes"] if node["usable"])
        summaries[variant_name] = {
            "total_nodes": variant_graph.number_of_nodes(),
            "total_edges": variant_graph.number_of_edges(),
            "level_counts": artifacts["node_level_counts"],
            "edge_type_counts": Counter(d.get("edge_type", "other") for _, _, d in variant_graph.edges(data=True)),
            "blind_path_nodes": sum(1 for node in artifacts["nodes"] if node["is_blind_path"]),
            "usable_nodes": usable_count,
            "non_usable_nodes": len(artifacts["nodes"]) - usable_count,
            "entrance_tagged_nodes": sum(1 for _, attr in variant_graph.nodes(data=True) if attr.get("node_type") == "entrance"),
            "platform_tagged_nodes": sum(1 for _, attr in variant_graph.nodes(data=True) if attr.get("node_type") == "door_platform"),
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.3 - Topology comparison between default and sparse graph variants", fontsize=14, fontweight="bold")

    x = np.arange(len(variants))
    width = 0.35
    axes[0, 0].bar(x - width / 2, [summaries[v]["total_nodes"] / 1000.0 for v in variants], width,
                   color="#457b9d", alpha=0.9, label="nodes")
    axes[0, 0].bar(x + width / 2, [summaries[v]["total_edges"] / 1000.0 for v in variants], width,
                   color="#bc6c25", alpha=0.9, label="edges")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([GRAPH_VARIANT_SHORT.get(v, v) for v in variants])
    axes[0, 0].set_title("Total topology size")
    axes[0, 0].set_ylabel("count (thousands)")
    axes[0, 0].legend(fontsize=8)

    levels = _sort_levels(sorted({level for variant_name in variants for level in summaries[variant_name]["level_counts"].keys()}))
    x_levels = np.arange(len(levels))
    variant_width = 0.8 / max(len(variants), 1)
    for idx, variant_name in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2) * variant_width
        axes[0, 1].bar(
            x_levels + offset,
            [summaries[variant_name]["level_counts"].get(level, 0) for level in levels],
            variant_width,
            color=GRAPH_VARIANT_COLORS.get(variant_name, "#999999"),
            alpha=0.88,
            label=GRAPH_VARIANT_SHORT.get(variant_name, variant_name),
        )
    axes[0, 1].set_xticks(x_levels)
    axes[0, 1].set_xticklabels(levels)
    axes[0, 1].set_title("Node counts by level")
    axes[0, 1].set_ylabel("nodes")
    axes[0, 1].legend(fontsize=8)

    edge_order = sorted(
        {edge_type for variant_name in variants for edge_type in summaries[variant_name]["edge_type_counts"].keys()},
        key=lambda key: sum(summaries[variant_name]["edge_type_counts"].get(key, 0) for variant_name in variants),
        reverse=True,
    )[:6]
    x_edge = np.arange(len(edge_order))
    for idx, variant_name in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2) * variant_width
        axes[1, 0].bar(
            x_edge + offset,
            [summaries[variant_name]["edge_type_counts"].get(edge_type, 0) for edge_type in edge_order],
            variant_width,
            color=GRAPH_VARIANT_COLORS.get(variant_name, "#999999"),
            alpha=0.88,
            label=GRAPH_VARIANT_SHORT.get(variant_name, variant_name),
        )
    axes[1, 0].set_xticks(x_edge)
    axes[1, 0].set_xticklabels(edge_order, rotation=30, ha="right")
    axes[1, 0].set_title("Edge counts by edge type")
    axes[1, 0].set_ylabel("edges")
    axes[1, 0].legend(fontsize=8)

    semantic_fields = [
        ("blind_path_nodes", "blind-path"),
        ("entrance_tagged_nodes", "entrance-tagged"),
        ("platform_tagged_nodes", "platform-tagged"),
        ("usable_nodes", "usable"),
    ]
    x_sem = np.arange(len(semantic_fields))
    for idx, variant_name in enumerate(variants):
        offset = (idx - (len(variants) - 1) / 2) * variant_width
        axes[1, 1].bar(
            x_sem + offset,
            [summaries[variant_name][field] for field, _ in semantic_fields],
            variant_width,
            color=GRAPH_VARIANT_COLORS.get(variant_name, "#999999"),
            alpha=0.88,
            label=GRAPH_VARIANT_SHORT.get(variant_name, variant_name),
        )
    axes[1, 1].set_xticks(x_sem)
    axes[1, 1].set_xticklabels([label for _, label in semantic_fields], rotation=20, ha="right")
    axes[1, 1].set_title("Semantic and accessibility tags")
    axes[1, 1].set_ylabel("node count")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_3_graph_statistics")


def _fig_6_4_baseline_static_heatmap(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    result = datasets.get("A_result", {})
    traj = datasets.get("A_traj", [])
    if not result or not traj or graph_artifacts is None:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.4 - Baseline static routing occupancy and bottlenecks", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    _draw_base_plan(ax, graph_artifacts, line_collection_cls)
    xs = [pt[0] for pt in traj]
    ys = [pt[1] for pt in traj]
    hb = ax.hexbin(xs, ys, gridsize=65, cmap="YlOrRd", mincnt=1, linewidths=0.0, alpha=0.9)
    fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02, label="trajectory samples")
    ax.set_title("Static trajectory density")

    ax = axes[0, 1]
    queue = result.get("stair_queue_over_time", [])
    if queue:
        q_t = [row[0] for row in queue]
        q_v = [row[1] for row in queue]
        ax.plot(q_t, q_v, color=STATIC_COLOR, linewidth=2)
        ax.fill_between(q_t, q_v, color=STATIC_COLOR, alpha=0.18)
    ax.set_title("Connector queue over time")
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("queued agents")

    ax = axes[1, 0]
    tt = result.get("travel_times", [])
    wt = [value for value in result.get("wait_times", []) if value and value > 0]
    if tt:
        ax.hist(tt, bins=24, color=STATIC_COLOR, alpha=0.72, label="travel time")
    if wt:
        ax.hist(wt, bins=24, color="#e76f51", alpha=0.62, label="wait time")
    ax.set_title("Travel and wait time distributions")
    ax.set_xlabel("seconds")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    top_connectors = _top_connector_edges(graph_artifacts, result.get("edge_throughput", {}), limit=10)
    labels = [item[0] for item in top_connectors][::-1]
    values = [item[1] for item in top_connectors][::-1]
    ax.barh(labels, values, color="#6d597a", alpha=0.9)
    ax.set_title("Top connector-edge throughputs")
    ax.set_xlabel("agent crossings")

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_4_baseline_static_heatmap")


def _fig_6_5_static_dynamic_comparison(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    rows = datasets.get("B_rows", [])
    if len(rows) < 2:
        return []
    static = next((row for row in rows if row.get("routing_mode") == "static"), rows[0])
    dynamic = next((row for row in rows if row.get("routing_mode") == "dynamic"), rows[-1])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Fig 6.5 - Static vs dynamic routing at baseline demand", fontsize=14, fontweight="bold")

    metrics = [
        ("arrive_rate", "Arrival rate", "fraction"),
        ("mean_travel_time_s", "Mean travel time", "s"),
        ("mean_wait_time_s", "Mean wait time", "s"),
        ("max_queue", "Peak queue", "agents"),
        ("p95_travel_time_s", "P95 travel time", "s"),
        ("reroute_count", "Reroute events", "events"),
    ]
    for ax, (field, title, unit) in zip(axes.flat, metrics):
        s_val = _as_float(static.get(field))
        d_val = _as_float(dynamic.get(field))
        values = [s_val, d_val]
        ax.bar(["static", "dynamic"], values, color=[STATIC_COLOR, DYNAMIC_COLOR], alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(unit)
        delta = d_val - s_val
        pct = 0.0 if s_val == 0.0 or math.isnan(s_val) else (delta / s_val) * 100.0
        ax.text(0.5, 0.97, f"delta = {delta:+.2f}\n{pct:+.1f}%", transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_5_static_dynamic_comparison")


def _fig_6_6_route_redistribution(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    static_result = datasets.get("B_static_result", {})
    dynamic_result = datasets.get("B_dynamic_result", {})
    if not static_result or not dynamic_result or graph_artifacts is None:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Fig 6.6 - Route redistribution under congestion-aware replanning", fontsize=14, fontweight="bold")

    static_lc = _plot_edge_overlay(
        axes[0], graph_artifacts, static_result.get("edge_throughput", {}), np, line_collection_cls,
        cmap="YlOrRd", title="Static edge throughput",
    )
    dynamic_lc = _plot_edge_overlay(
        axes[1], graph_artifacts, dynamic_result.get("edge_throughput", {}), np, line_collection_cls,
        cmap="YlOrRd", title="Dynamic edge throughput",
    )
    diff: dict[str, float] = {}
    keys = set(static_result.get("edge_throughput", {}).keys()) | set(dynamic_result.get("edge_throughput", {}).keys())
    for key in keys:
        diff[key] = float(dynamic_result.get("edge_throughput", {}).get(key, 0)) - float(static_result.get("edge_throughput", {}).get(key, 0))
    diff_lc = _plot_edge_overlay(
        axes[2], graph_artifacts, diff, np, line_collection_cls,
        cmap="coolwarm", title="Dynamic - static edge load", diverging=True,
    )

    if static_lc is not None:
        fig.colorbar(static_lc, ax=axes[0], fraction=0.04, pad=0.01, label="crossings")
    if dynamic_lc is not None:
        fig.colorbar(dynamic_lc, ax=axes[1], fraction=0.04, pad=0.01, label="crossings")
    if diff_lc is not None:
        fig.colorbar(diff_lc, ax=axes[2], fraction=0.04, pad=0.01, label="delta crossings")

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_6_route_redistribution")


def _fig_6_7_profile_comparison(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    static_rows = datasets.get("C_static", [])
    dynamic_rows = datasets.get("C_dynamic", [])
    if not static_rows or not dynamic_rows:
        return []
    profiles = ["C1_normal", "C2_elderly", "C3_mixed"]
    label_map = {
        "C1_normal": "normal",
        "C2_elderly": "elderly",
        "C3_mixed": "mixed",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.7 - Profile comparison under static and dynamic routing", fontsize=14, fontweight="bold")

    x = np.arange(len(profiles))
    width = 0.35
    panels = [
        (axes[0, 0], "arrive_rate", "Arrival rate"),
        (axes[0, 1], "mean_travel_time_s", "Mean travel time (s)"),
        (axes[1, 0], "mean_wait_time_s", "Mean wait time (s)"),
        (axes[1, 1], "reroute_count", "Reroute events"),
    ]
    for ax, field, title in panels:
        s_vals = [_as_float(next(row[field] for row in static_rows if row["condition"] == profile)) for profile in profiles]
        d_vals = [_as_float(next(row[field] for row in dynamic_rows if row["condition"] == profile)) for profile in profiles]
        ax.bar(x - width / 2, s_vals, width, color=STATIC_COLOR, label="static")
        ax.bar(x + width / 2, d_vals, width, color=DYNAMIC_COLOR, label="dynamic")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{label_map[profile]}\n(Ns={next(row['n_agents'] for row in static_rows if row['condition'] == profile)}, Nd={next(row['n_agents'] for row in dynamic_rows if row['condition'] == profile)})"
            for profile in profiles
        ], fontsize=8)
        ax.legend(fontsize=8)
    axes[1, 1].text(0.02, 0.02,
                    "Note: C3 in quick mode reuses the N=200 baseline results,\nwhile C1/C2 quick simulations run at N=50.",
                    transform=axes[1, 1].transAxes, ha="left", va="bottom", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_7_profile_comparison")


def _fig_6_8_mixed_agent_results(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    d_static = datasets.get("D_static", {})
    d_dynamic = datasets.get("D_dynamic", {})
    if not d_static or not d_dynamic:
        return []

    s_over = (d_static.get("overall") or [{}])[0]
    d_over = (d_dynamic.get("overall") or [{}])[0]
    s_profiles = {row["profile"]: row for row in d_static.get("per_profile", [])}
    d_profiles = {row["profile"]: row for row in d_dynamic.get("per_profile", [])}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.8 - Mixed-agent crowd results", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    metrics = ["mean_travel_time_s", "mean_wait_time_s", "max_queue", "reroute_count"]
    labels = ["travel", "wait", "queue", "reroutes"]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, [_as_float(s_over.get(metric)) for metric in metrics], width, color=STATIC_COLOR, label="static")
    ax.bar(x + width / 2, [_as_float(d_over.get(metric)) for metric in metrics], width, color=DYNAMIC_COLOR, label="dynamic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Overall crowd metrics")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    prof_names = ["normal", "elderly"]
    x = np.arange(len(prof_names))
    ax.bar(x - width / 2, [_as_float(s_profiles[p].get("mean_travel_time_s")) for p in prof_names], width, color=STATIC_COLOR, label="static")
    ax.bar(x + width / 2, [_as_float(d_profiles[p].get("mean_travel_time_s")) for p in prof_names], width, color=DYNAMIC_COLOR, label="dynamic")
    ax.set_xticks(x)
    ax.set_xticklabels(prof_names)
    ax.set_title("Per-profile mean travel time")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x - width / 2, [_as_float(s_profiles[p].get("mean_wait_time_s")) for p in prof_names], width, color=STATIC_COLOR, label="static")
    ax.bar(x + width / 2, [_as_float(d_profiles[p].get("mean_wait_time_s")) for p in prof_names], width, color=DYNAMIC_COLOR, label="dynamic")
    ax.set_xticks(x)
    ax.set_xticklabels(prof_names)
    ax.set_title("Per-profile mean wait time")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    static_reduction = [
        _as_float(d_profiles[p].get("mean_wait_time_s")) - _as_float(s_profiles[p].get("mean_wait_time_s"))
        for p in prof_names
    ]
    ax.barh(prof_names, static_reduction, color=[PROFILE_COLORS[p] for p in prof_names])
    ax.axvline(0.0, color="#666666", linewidth=1)
    ax.set_title("Dynamic minus static mean wait")
    ax.set_xlabel("seconds (negative is better)")
    share_text = (
        f"composition: {float(s_over.get('normal_share', 0)):.0%} normal / {float(s_over.get('elderly_share', 0)):.0%} elderly\n"
        f"arrival: static={_as_float(s_over.get('arrive_rate')):.3f}, dynamic={_as_float(d_over.get('arrive_rate')):.3f}"
    )
    ax.text(0.98, 0.05, share_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_8_mixed_agent_results")


def _fig_6_9_algorithm_threshold_sensitivity(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    rows = datasets.get("E_rows", [])
    if not rows:
        return []
    rows = sorted(
        rows,
        key=lambda row: (
            0 if (row.get("graph_variant") or "default") == "default" else 1,
            _as_float(row.get("replan_wait_threshold_s")),
            row.get("condition", ""),
        ),
    )
    labels = []
    for row in rows:
        variant = row.get("graph_variant") or "default"
        if variant == "default":
            labels.append(f"{row['condition']}\nN={row['n_agents']}")
        else:
            labels.append(f"{row['condition']}\n{GRAPH_VARIANT_SHORT.get(variant, variant)}\nN={row['n_agents']}")
    colors = [THRESHOLD_COLORS.get(row["condition"], "#999999") for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Fig 6.9 - Threshold and graph sensitivity under dynamic routing", fontsize=14, fontweight="bold")
    metrics = [
        ("arrive_rate", "Arrival rate", "fraction"),
        ("mean_travel_time_s", "Mean travel time", "s"),
        ("mean_wait_time_s", "Mean wait time", "s"),
        ("max_queue", "Peak queue", "agents"),
        ("reroute_count", "Reroute events", "events"),
        ("instability_fraction", "Route instability", "fraction"),
    ]
    for ax, (field, title, ylabel) in zip(axes.flat, metrics):
        values = [_as_float(row.get(field)) for row in rows]
        ax.bar(
            labels,
            values,
            color=colors,
            alpha=0.88,
            edgecolor=[GRAPH_VARIANT_COLORS.get(row.get("graph_variant") or "default", "#444444") for row in rows],
            linewidth=1.4,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
    note_lines = []
    if any(row.get("condition") == "E2_balanced" for row in rows):
        note_lines.append("E2 reuses the full N=200 baseline dynamic result.")
    if any((row.get("graph_variant") or "default") != "default" for row in rows):
        note_lines.append("E4 switches to a sparse semantics-light graph branch.")
    if any((row.get("condition") in {"E1_aggressive", "E3_conservative", "E4_sparse_semantic_light"}) for row in rows):
        note_lines.append("Quick mode runs new simulations at N=50 for non-reuse branches.")
    axes[1, 2].text(0.02, 0.03,
                    "\n".join(note_lines),
                    transform=axes[1, 2].transAxes, ha="left", va="bottom", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_9_algorithm_threshold_sensitivity")


def _fig_6_10_capacity_threshold(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    static_rows = datasets.get("F_static", [])
    dynamic_rows = datasets.get("F_dynamic", [])
    if not static_rows or not dynamic_rows:
        return []

    def _extract(rows: list[dict[str, str]]) -> dict[str, list[Any]]:
        return {
            "N": [int(_as_float(row.get("n_agents"))) for row in rows],
            "arrive": [_as_float(row.get("arrive_rate")) for row in rows],
            "wait": [_as_float(row.get("mean_wait_time_s")) for row in rows],
            "queue": [_as_float(row.get("max_queue")) for row in rows],
            "tt": [_as_float(row.get("mean_travel_time_s")) for row in rows],
        }

    static_by_variant = _group_rows_by_graph_variant(static_rows)
    dynamic_by_variant = _group_rows_by_graph_variant(dynamic_rows)
    variants = [variant for variant in ["default", "sparse_semantic_light"] if variant in static_by_variant or variant in dynamic_by_variant]
    if not variants:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Fig 6.10 - Capacity curves across graph variants", fontsize=14, fontweight="bold")
    panels = [
        (axes[0, 0], "arrive", "Arrival rate", True),
        (axes[0, 1], "wait", "Mean wait time (s)", False),
        (axes[1, 0], "queue", "Peak queue (agents)", False),
        (axes[1, 1], "tt", "Mean travel time (s)", False),
    ]
    max_n = max(int(_as_float(row.get("n_agents"))) for row in static_rows + dynamic_rows)
    for ax, field, title, is_arrive in panels:
        for variant in variants:
            color = GRAPH_VARIANT_COLORS.get(variant, "#666666")
            label_base = GRAPH_VARIANT_LABELS.get(variant, variant.replace("_", " "))
            s_rows = static_by_variant.get(variant, [])
            d_rows = dynamic_by_variant.get(variant, [])
            if s_rows:
                s = _extract(s_rows)
                ax.plot(s["N"], s[field], marker="o", linewidth=2.2, color=color, label=f"{label_base} | static")
            if d_rows:
                d = _extract(d_rows)
                ax.plot(d["N"], d[field], marker="s", linewidth=2.2, linestyle="--", color=color, label=f"{label_base} | dynamic")
        if max_n > 1000:
            ax.axvspan(1000, max_n, color="#d9d9d9", alpha=0.14)
        ax.set_title(title)
        ax.set_xlabel("number of agents")
        ax.legend(fontsize=8)
        if is_arrive:
            min_arrive = min(_as_float(row.get("arrive_rate")) for row in static_rows + dynamic_rows)
            ax.axhline(0.95, color="#999999", linestyle=":", linewidth=1.3)
            ax.axhline(0.70, color="#c1121f", linestyle=":", linewidth=1.3)
            ax.set_ylim(max(0.0, min(0.62, min_arrive - 0.05)), 1.02)
    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_10_capacity_threshold")


def _fig_6_11_overall_findings_summary(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Fig 6.11 - Cross-experiment summary dashboard", fontsize=14, fontweight="bold")

    a_metrics = datasets.get("A_metrics", {})
    axes[0, 0].bar(
        ["arrive", "travel", "wait", "queue"],
        [
            _as_float(a_metrics.get("arrive_rate")),
            _as_float(a_metrics.get("mean_travel_time_s")),
            _as_float(a_metrics.get("mean_wait_time_s")),
            _as_float(a_metrics.get("max_queue")),
        ],
        color=["#2a9d8f", STATIC_COLOR, "#e76f51", "#6d597a"],
    )
    axes[0, 0].set_title("A: baseline static reference")

    b_rows = datasets.get("B_rows", [])
    if len(b_rows) >= 2:
        static = next((row for row in b_rows if row.get("routing_mode") == "static"), b_rows[0])
        dynamic = next((row for row in b_rows if row.get("routing_mode") == "dynamic"), b_rows[-1])
        deltas = [
            _as_float(dynamic.get("arrive_rate")) - _as_float(static.get("arrive_rate")),
            _as_float(dynamic.get("mean_travel_time_s")) - _as_float(static.get("mean_travel_time_s")),
            _as_float(dynamic.get("mean_wait_time_s")) - _as_float(static.get("mean_wait_time_s")),
            _as_float(dynamic.get("max_queue")) - _as_float(static.get("max_queue")),
        ]
        axes[0, 1].barh(["arrive", "travel", "wait", "queue"], deltas, color=[DYNAMIC_COLOR if value >= 0 else "#2a9d8f" for value in deltas])
        axes[0, 1].axvline(0.0, color="#666666", linewidth=1)
    axes[0, 1].set_title("B: dynamic minus static delta")

    c_static = datasets.get("C_static", [])
    c_dynamic = datasets.get("C_dynamic", [])
    c_profiles = ["C1_normal", "C2_elderly", "C3_mixed"]
    axes[0, 2].plot(range(len(c_profiles)), [_as_float(next(row["mean_wait_time_s"] for row in c_static if row["condition"] == p)) for p in c_profiles],
                    marker="o", color=STATIC_COLOR, label="static")
    axes[0, 2].plot(range(len(c_profiles)), [_as_float(next(row["mean_wait_time_s"] for row in c_dynamic if row["condition"] == p)) for p in c_profiles],
                    marker="s", linestyle="--", color=DYNAMIC_COLOR, label="dynamic")
    axes[0, 2].set_xticks(range(len(c_profiles)))
    axes[0, 2].set_xticklabels(["normal", "elderly", "mixed"])
    axes[0, 2].set_title("C: profile wait comparison")
    axes[0, 2].legend(fontsize=8)

    d_static = datasets.get("D_static", {})
    d_dynamic = datasets.get("D_dynamic", {})
    if d_static and d_dynamic:
        s_norm = {row["profile"]: row for row in d_static.get("per_profile", [])}
        d_norm = {row["profile"]: row for row in d_dynamic.get("per_profile", [])}
        axes[1, 0].bar(["normal", "elderly"],
                       [
                           _as_float(d_norm["normal"].get("mean_wait_time_s")) - _as_float(s_norm["normal"].get("mean_wait_time_s")),
                           _as_float(d_norm["elderly"].get("mean_wait_time_s")) - _as_float(s_norm["elderly"].get("mean_wait_time_s")),
                       ],
                       color=[PROFILE_COLORS["normal"], PROFILE_COLORS["elderly"]])
        axes[1, 0].axhline(0.0, color="#666666", linewidth=1)
    axes[1, 0].set_title("D: wait reduction by profile")

    e_rows = datasets.get("E_rows", [])
    if e_rows:
        for row in e_rows:
            variant = row.get("graph_variant") or "default"
            axes[1, 1].scatter(
                _as_float(row.get("reroute_count")),
                _as_float(row.get("mean_wait_time_s")),
                s=120 + 300 * _as_float(row.get("instability_fraction")),
                c=THRESHOLD_COLORS.get(row["condition"], "#777777"),
                alpha=0.9,
                edgecolors=GRAPH_VARIANT_COLORS.get(variant, "#444444"),
                linewidths=1.4,
            )
            label = row["condition"] if variant == "default" else f"{row['condition']} ({GRAPH_VARIANT_SHORT.get(variant, variant)})"
            axes[1, 1].text(_as_float(row.get("reroute_count")), _as_float(row.get("mean_wait_time_s")), label, fontsize=8)
    axes[1, 1].set_title("E: wait vs reroutes across graph branches")
    axes[1, 1].set_xlabel("reroute events")
    axes[1, 1].set_ylabel("mean wait (s)")

    f_static = datasets.get("F_static", [])
    f_dynamic = datasets.get("F_dynamic", [])
    if f_static and f_dynamic:
        f_static_by_variant = _group_rows_by_graph_variant(f_static)
        f_dynamic_by_variant = _group_rows_by_graph_variant(f_dynamic)
        for variant in ["default", "sparse_semantic_light"]:
            color = GRAPH_VARIANT_COLORS.get(variant, "#666666")
            label_base = GRAPH_VARIANT_LABELS.get(variant, variant.replace("_", " "))
            if variant in f_static_by_variant:
                rows = f_static_by_variant[variant]
                axes[1, 2].plot([int(_as_float(row.get("n_agents"))) for row in rows], [_as_float(row.get("arrive_rate")) for row in rows],
                                color=color, marker="o", label=f"{label_base} | static")
            if variant in f_dynamic_by_variant:
                rows = f_dynamic_by_variant[variant]
                axes[1, 2].plot([int(_as_float(row.get("n_agents"))) for row in rows], [_as_float(row.get("arrive_rate")) for row in rows],
                                color=color, marker="s", linestyle="--", label=f"{label_base} | dynamic")
        axes[1, 2].axhline(0.70, color="#c1121f", linestyle=":", linewidth=1.3)
    axes[1, 2].set_title("F: capacity curve by graph branch")
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_11_overall_findings_summary")


def _fig_6_12_baseline_bottlenecks(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    static_result = datasets.get("B_static_result", {})
    dynamic_result = datasets.get("B_dynamic_result", {})
    if not static_result or not dynamic_result or graph_artifacts is None:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Fig 6.12 - Bottleneck connector comparison", fontsize=14, fontweight="bold")
    for ax, title, result, color in [
        (axes[0], "Static bottlenecks", static_result, STATIC_COLOR),
        (axes[1], "Dynamic bottlenecks", dynamic_result, DYNAMIC_COLOR),
    ]:
        top = _top_connector_edges(graph_artifacts, result.get("edge_throughput", {}), limit=12)
        labels = [item[0] for item in top][::-1]
        values = [item[1] for item in top][::-1]
        ax.barh(labels, values, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("connector crossings")
    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_12_baseline_bottlenecks")


def _fig_6_13_capacity_breakdown(root, dirs, datasets, graph, graph_artifacts, plt, np, line_collection_cls, logger):
    static_rows = datasets.get("F_static", [])
    dynamic_rows = datasets.get("F_dynamic", [])
    if not static_rows or not dynamic_rows:
        return []
    static_by_variant = _group_rows_by_graph_variant(static_rows)
    dynamic_by_variant = _group_rows_by_graph_variant(dynamic_rows)
    variants = [variant for variant in ["default", "sparse_semantic_light"] if variant in static_by_variant or variant in dynamic_by_variant]
    if not variants:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 6.13 - Capacity breakdown beyond the threshold by graph branch", fontsize=14, fontweight="bold")

    def _plot_pair(ax, field: str, title: str, ylabel: str) -> None:
        for variant in variants:
            color = GRAPH_VARIANT_COLORS.get(variant, "#666666")
            label_base = GRAPH_VARIANT_LABELS.get(variant, variant.replace("_", " "))
            s_rows = static_by_variant.get(variant, [])
            d_rows = dynamic_by_variant.get(variant, [])
            if s_rows:
                s_n = [int(_as_float(row.get("n_agents"))) for row in s_rows]
                s_v = [_as_float(row.get(field)) for row in s_rows]
                ax.plot(s_n, s_v, marker="o", color=color, linewidth=2, label=f"{label_base} | static")
            if d_rows:
                d_n = [int(_as_float(row.get("n_agents"))) for row in d_rows]
                d_v = [_as_float(row.get(field)) for row in d_rows]
                ax.plot(d_n, d_v, marker="s", color=color, linewidth=2, linestyle="--", label=f"{label_base} | dynamic")
        ax.set_title(title)
        ax.set_xlabel("number of agents")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    _plot_pair(axes[0, 0], "n_failed", "Failed agents vs load", "failed agents")
    _plot_pair(axes[0, 1], "reroute_count", "Reroute count vs load", "reroutes")
    _plot_pair(axes[1, 0], "top_connector_load", "Top connector load vs demand", "crossings")
    _plot_pair(axes[1, 1], "runtime_s", "Runtime vs demand", "seconds")

    plt.tight_layout()
    return _save_figure(fig, dirs["figures"] / "fig_6_13_capacity_breakdown")
