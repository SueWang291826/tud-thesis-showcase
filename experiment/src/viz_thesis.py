"""
viz_thesis.py
=============

Thesis-quality visualisation functions for Step-5 experiments.

Design principles (per supervisor feedback):
  1. No "all agents at once" global animation -- too cluttered.
  2. Heat maps for aggregated space use / congestion.
  3. Small multiples (time × floor) for agent distribution over time.
  4. Individual path traces for handful of selected agents.
  5. Statistical charts for rerouting effectiveness.
  6. Agent colour = OD group (origin entrance), NOT floor.

Public API
----------
  fig_node_heatmap_per_level     - Aggregated node visit frequency heat map
  fig_space_use_heatmap          - KDE smooth density over floor polygon
  fig_small_multiples_time       - Grid of snapshots: floors × time-slices
  fig_individual_paths           - Highlight 4-6 selected agent trajectories
  fig_od_group_paths             - Path traces coloured by origin entrance
  fig_stats_comparison           - Bar / box-plot comparison across scenarios
  fig_connector_load_bars        - Per-connector throughput bar chart
  fig_travel_time_box            - Box-plot travel times by agent type × scenario
  fig_wheelchair_path_comparison - Wheelchair vs normal path on same floor plan
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from shapely.geometry import MultiPolygon, Polygon

from src.utils import setup_matplotlib_font, flatten_polygons

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

# Entrance→colour mapping (A=red, B=orange, C=green, D=purple, unknown=grey)
OD_PALETTE = {
    "A": "#E53935",
    "B": "#FB8C00",
    "C": "#43A047",
    "D": "#8E24AA",
    "unknown": "#78909C",
}

LEVEL_FILL = {"F1": "#E3F2FD", "F3": "#E8F5E9", "F4": "#FFF3E0"}
LEVEL_EDGE_C = {"F1": "#90CAF9", "F3": "#A5D6A7", "F4": "#FFCC80"}
WALKABLE_LEVELS = ("F1", "F3", "F4")

DPI = 180


def _edir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_floor(ax, geo: dict, level: str, alpha: float = 0.35):
    """Draw floor polygon background on ax."""
    floor = geo.get("floor")
    if floor is None or floor.is_empty:
        return
    polys = floor.geoms if isinstance(floor, MultiPolygon) else [floor]
    for poly in polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys,
                color=LEVEL_FILL.get(level, "#F5F5F5"),
                alpha=alpha,
                edgecolor=LEVEL_EDGE_C.get(level, "#BDBDBD"),
                linewidth=0.5, zorder=0)


def _load_traj(traj_path: Path) -> list[dict]:
    """Load traj_agents.jsonl into list of dicts."""
    rows = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _entrance_group(origin_node: str, g: nx.Graph) -> str:
    """Return entrance group letter (A/B/C/D) from an origin node id."""
    if origin_node not in g:
        return "unknown"
    eg = g.nodes[origin_node].get("entrance_group", "")
    if eg:
        # eg may be "entrance_D" or just "D"
        letter = eg.split("_")[-1].upper()
        if letter in ("A", "B", "C", "D"):
            return letter
    # Fallback: parse from node_id string like "ent_F3_entrance_A_000"
    for letter in ("A", "B", "C", "D"):
        if f"entrance_{letter}" in origin_node.lower():
            return letter
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Node visit frequency heat map (aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def fig_node_heatmap_per_level(
    traj_path: str | Path,
    g: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    label: str = "",
    *,
    cfg: dict | None = None,
) -> Path:
    """Heat map of node visit counts for each walking level.

    Each node is coloured by total visit-ticks across the simulation.
    Reveals congestion hotspots and underused corridors.
    """
    setup_matplotlib_font()
    od = _edir(out_dir)
    rows = _load_traj(Path(traj_path))

    # Count visits per node
    visit_count: dict[str, int] = defaultdict(int)
    for r in rows:
        visit_count[r["node_id"]] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI)

    for ax, lvl in zip(axes, WALKABLE_LEVELS):
        _plot_floor(ax, geometries.get(lvl, {}), lvl)

        xs, ys, cs = [], [], []
        for nid, cnt in visit_count.items():
            if nid not in g:
                continue
            nd = g.nodes[nid]
            if nd.get("level") != lvl:
                continue
            xs.append(nd["x"])
            ys.append(nd["y"])
            cs.append(cnt)

        if xs:
            vmax = max(cs) or 1
            sc = ax.scatter(xs, ys, c=cs, cmap="YlOrRd",
                            vmin=0, vmax=vmax,
                            s=3, alpha=0.85, edgecolors="none", zorder=5)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                         label="Visit ticks")

        ax.set_aspect("equal")
        ax.set_title(f"{lvl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)

    scenario_str = f" — {label}" if label else ""
    fig.suptitle(f"Node Visit Frequency Heat Map{scenario_str}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = od / f"heatmap_node_visits{'_' + label if label else ''}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] heatmap_node_visits → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 2. Small multiples: agent distribution at T snapshots × 3 floors
# ─────────────────────────────────────────────────────────────────────────────

def fig_small_multiples_time(
    traj_path: str | Path,
    g: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    label: str = "",
    n_snapshots: int = 5,
    *,
    cfg: dict | None = None,
) -> Path:
    """N-column × 3-row grid: each column = one time snapshot, each row = floor.

    Agents are coloured by OD group (origin entrance: A/B/C/D).
    """
    setup_matplotlib_font()
    od = _edir(out_dir)
    rows = _load_traj(Path(traj_path))

    if not rows:
        print("[viz_thesis] small_multiples: empty trajectory, skipping.")
        return od / "small_multiples_empty.png"

    T_max = max(r["t"] for r in rows)
    snap_times = np.linspace(0, T_max, n_snapshots + 1)[1:]  # exclude t=0

    # Build per-time-step lookup: t -> {node_id: agent_id}
    by_t: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_t[r["t"]].append(r)

    # Collect available times
    avail_times = sorted(by_t.keys())

    def _nearest_t(target):
        return min(avail_times, key=lambda x: abs(x - target))

    snap_actual = [_nearest_t(st) for st in snap_times]

    # Build agent → OD group map
    # Priority: agent_meta.json (origin node) > first-seen node in trajectory
    agent_group: dict[str, str] = {}
    meta_path = Path(traj_path).parent / "agent_meta.json"
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        for aid, info in meta.items():
            agent_group[aid] = _entrance_group(info["origin"], g)
    else:
        agent_first: dict[str, dict] = {}
        for r in sorted(rows, key=lambda x: x["t"]):
            if r["agent_id"] not in agent_first:
                agent_first[r["agent_id"]] = r
        for aid, r in agent_first.items():
            agent_group[aid] = _entrance_group(r["node_id"], g)

    nrows, ncols = 3, n_snapshots
    fig = plt.figure(figsize=(ncols * 5, nrows * 4), dpi=DPI)
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.35, wspace=0.15)

    for col, (t_snap, t_actual) in enumerate(zip(snap_times, snap_actual)):
        snap_rows = by_t[t_actual]
        # node_id → agent_id for this snapshot
        node_agent = {r["node_id"]: r["agent_id"] for r in snap_rows}

        for row, lvl in enumerate(WALKABLE_LEVELS):
            ax = fig.add_subplot(gs[row, col])
            _plot_floor(ax, geometries.get(lvl, {}), lvl)

            xs, ys, colours = [], [], []
            for nid, aid in node_agent.items():
                if nid not in g:
                    continue
                nd = g.nodes[nid]
                if nd.get("level") != lvl:
                    continue
                xs.append(nd["x"])
                ys.append(nd["y"])
                grp = agent_group.get(aid, "unknown")
                colours.append(OD_PALETTE.get(grp, "#78909C"))

            if xs:
                ax.scatter(xs, ys, c=colours, s=12, alpha=0.85,
                           edgecolors="none", zorder=5, linewidths=0)

            ax.set_aspect("equal")
            ax.tick_params(labelsize=5.5, left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            if col == 0:
                ax.set_ylabel(lvl, fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(f"t = {t_actual:.0f} s", fontsize=9, fontweight="bold")

    # Legend for OD groups
    handles = [mpatches.Patch(color=c, label=f"Entrance {g}")
               for g, c in OD_PALETTE.items() if g != "unknown"]
    fig.legend(handles=handles, title="Origin entrance",
               loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    scenario_str = f" — {label}" if label else ""
    fig.suptitle(
        f"Agent Distribution Over Time × Floor{scenario_str}\n"
        "Colour = origin entrance group",
        fontsize=13, fontweight="bold", y=1.01,
    )
    p = od / f"small_multiples{'_' + label if label else ''}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] small_multiples → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 3. Individual agent path traces (few agents, full trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def fig_individual_paths(
    traj_path: str | Path,
    g: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    label: str = "",
    n_agents: int = 6,
    seed: int = 42,
    *,
    cfg: dict | None = None,
) -> Path:
    """Show full trajectories of N selected agents, one subplot per floor.

    Each agent has a fixed colour maintained across all floor subplots so the
    viewer can follow a single person through multiple levels.
    """
    setup_matplotlib_font()
    od = _edir(out_dir)
    rows = _load_traj(Path(traj_path))
    if not rows:
        return od / "individual_paths_empty.png"

    # Build per-agent ordered trajectory
    traj_by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["t"]):
        traj_by_agent[r["agent_id"]].append(r)

    rng = np.random.default_rng(seed)
    # Prefer agents that visited multiple floors (more interesting)
    multi = [aid for aid, pts in traj_by_agent.items()
             if len({g.nodes[p["node_id"]].get("level") for p in pts
                     if p["node_id"] in g}) > 1]
    pool = multi if len(multi) >= n_agents else list(traj_by_agent.keys())
    chosen = list(rng.choice(pool, size=min(n_agents, len(pool)), replace=False))

    cmap_agents = plt.cm.get_cmap("tab10", len(chosen))
    agent_colours = {aid: cmap_agents(i) for i, aid in enumerate(chosen)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI)

    for ax, lvl in zip(axes, WALKABLE_LEVELS):
        _plot_floor(ax, geometries.get(lvl, {}), lvl)

        for aid in chosen:
            pts = traj_by_agent[aid]
            # Filter for this level
            xs = [p["x"] for p in pts
                  if p["node_id"] in g and g.nodes[p["node_id"]].get("level") == lvl]
            ys = [p["y"] for p in pts
                  if p["node_id"] in g and g.nodes[p["node_id"]].get("level") == lvl]
            if len(xs) < 2:
                continue
            colour = agent_colours[aid]
            ax.plot(xs, ys, color=colour, linewidth=1.4, alpha=0.82, zorder=4)
            # Start dot
            ax.scatter([xs[0]], [ys[0]], color=colour, s=45, zorder=6,
                       edgecolors="white", linewidths=0.8, marker="o")
            # End dot
            ax.scatter([xs[-1]], [ys[-1]], color=colour, s=45, zorder=6,
                       edgecolors="white", linewidths=0.8, marker="s")

        ax.set_aspect("equal")
        ax.set_title(f"{lvl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

    handles = [mpatches.Patch(color=agent_colours[a], label=a[:8])
               for a in chosen]
    fig.legend(handles=handles, title="Agent ID (● start  ■ end)",
               loc="lower center", ncol=min(6, len(chosen)), fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    scenario_str = f" — {label}" if label else ""
    fig.suptitle(
        f"Individual Agent Trajectories{scenario_str}\n"
        f"{len(chosen)} selected agents; ● = start,  ■ = arrival",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = od / f"individual_paths{'_' + label if label else ''}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] individual_paths → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical comparison across scenarios
# ─────────────────────────────────────────────────────────────────────────────

def fig_stats_comparison(
    results: list[dict],
    out_dir: str | Path,
    *,
    cfg: dict | None = None,
) -> Path:
    """2 × 2 panel: arrival rate, mean TT, wait time, replan count.

    Each scenario gets a distinct colour.  When exactly 2 scenarios are
    provided the second panel annotates the percentage change vs the first.
    """
    setup_matplotlib_font()
    od = _edir(out_dir)

    # Short display labels
    raw_labels = [r.get("label", f"S{i}") for i, r in enumerate(results)]
    labels = [lbl.replace("scenB_static", "B – Static")
                  .replace("scenC_dynamic", "C – Dynamic")
                  .replace("scenA_individual", "A – Individual")
              for lbl in raw_labels]

    arrive_rates = [r.get("arrive_rate", 0) * 100 for r in results]
    mean_tts = [
        (sum(r["travel_times"]) / len(r["travel_times"]))
        if r.get("travel_times") else 0.0
        for r in results
    ]
    mean_waits = [
        (sum(r["wait_times"]) / len(r["wait_times"]))
        if r.get("wait_times") else 0.0
        for r in results
    ]
    replan_counts = [len(r.get("replan_events", [])) for r in results]

    # Per-scenario palette
    palette = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A", "#F57F17"]
    bar_colours = [palette[i % len(palette)] for i in range(len(results))]
    bar_kwargs = dict(edgecolor="white", alpha=0.88)

    def _pct_delta(vals):
        """Return delta annotation string (vs first value) for each bar."""
        if len(vals) < 2:
            return [""] * len(vals)
        base = vals[0]
        out = [""]
        for v in vals[1:]:
            if base != 0:
                d = (v - base) / abs(base) * 100
                sign = "+" if d >= 0 else ""
                out.append(f"({sign}{d:.1f}%)")
            else:
                out.append("")
        return out

    def _draw_bar(ax, vals, title, ylabel, fmt="{:.1f}", show_delta=True):
        bars = ax.bar(labels, vals, color=bar_colours, **bar_kwargs)
        deltas = _pct_delta(vals) if show_delta else [""] * len(vals)
        for bar, val, dlt in zip(bars, vals, deltas):
            y_top = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_top + max(vals) * 0.01,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
            if dlt:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        y_top + max(vals) * 0.07,
                        dlt, ha="center", va="bottom",
                        fontsize=7.5, color="#555")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.22 if any(v > 0 for v in vals) else 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=DPI)
    _draw_bar(axes[0, 0], arrive_rates, "Arrival Rate (%)", "%", "{:.1f}%")
    _draw_bar(axes[0, 1], mean_tts, "Mean Travel Time (s)", "seconds", "{:.1f}")
    _draw_bar(axes[1, 0], mean_waits, "Mean Wait Time (s)", "seconds", "{:.1f}")
    _draw_bar(axes[1, 1], replan_counts, "Total Replan Events", "count", "{:.0f}",
              show_delta=False)

    # Arrival rate reference line at 100 %
    axes[0, 0].axhline(100, color="#9E9E9E", linestyle=":", linewidth=1.2)

    # Legend swatch per scenario
    handles = [mpatches.Patch(color=bar_colours[i], label=labels[i])
               for i in range(len(results))]
    fig.legend(handles=handles, loc="lower center", ncol=len(results),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Scenario Comparison — Aggregate Statistics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = od / "stats_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] stats_comparison → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 5. Travel time box-plots by agent type × scenario
# ─────────────────────────────────────────────────────────────────────────────

def fig_travel_time_box(
    results: list[dict],
    out_dir: str | Path,
    *,
    cfg: dict | None = None,
) -> Path:
    """Box-plot of travel times: grouped by scenario, split by agent type."""
    setup_matplotlib_font()
    od = _edir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    positions, data_series, tick_labels, colours_list = [], [], [], []
    gap = 3
    pos = 1

    for r in results:
        lbl = r.get("label", "?")
        _DLBL = {
            "scenA_individual": "A – Individual", "scenA": "A – Individual",
            "scenB_static": "B – Static",         "scenB": "B – Static",
            "scenC_dynamic": "C – Dynamic",        "scenC": "C – Dynamic",
        }
        disp_lbl = _DLBL.get(lbl, lbl)
        normal_tt = r.get("normal_travel_times", []) or r.get("travel_times", [])
        elderly_tt = r.get("elderly_travel_times", [])

        if normal_tt:
            data_series.append(normal_tt)
            positions.append(pos)
            tick_labels.append(f"{disp_lbl}\nnormal")
            colours_list.append("#1565C0")
            pos += 1

        if elderly_tt:
            data_series.append(elderly_tt)
            positions.append(pos)
            tick_labels.append(f"{disp_lbl}\nelderly")
            colours_list.append("#E53935")
            pos += 1

        pos += gap  # gap between scenarios

    if not data_series:
        ax.text(0.5, 0.5, "No travel time data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        bp = ax.boxplot(data_series, positions=positions, widths=0.7,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(bp["boxes"], colours_list):
            patch.set_facecolor(col)
            patch.set_alpha(0.82)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8.5)

    ax.set_ylabel("Travel time (s)", fontsize=10)
    ax.set_title("Travel Time Distribution by Scenario × Agent Type",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    legend_elems = [
        mpatches.Patch(color="#1565C0", alpha=0.82, label="Normal agents"),
        mpatches.Patch(color="#E53935", alpha=0.82, label="Elderly agents"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    fig.tight_layout()
    p = od / "travel_time_boxplot.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] travel_time_boxplot → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-connector throughput bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_connector_load_bars(
    results: list[dict],
    g: nx.Graph,
    out_dir: str | Path,
    *,
    cfg: dict | None = None,
) -> Path:
    """Grouped bar chart: throughput per connector type per scenario."""
    setup_matplotlib_font()
    od = _edir(out_dir)

    # Identify connector edges by edge_type
    conn_types = ("stair", "escalator", "elevator")

    # Build a set of edge-keys per connector type
    type_edges: dict[str, set[str]] = {ct: set() for ct in conn_types}
    for u, v, d in g.edges(data=True):
        et = d.get("edge_type", "")
        for ct in conn_types:
            if ct in et:
                key = f"{min(u,v)}|{max(u,v)}"
                type_edges[ct].add(key)
                break

    scenario_labels = [r.get("label", f"S{i}") for i, r in enumerate(results)]
    _DISPLAY = {
        "scenA_individual": "A – Individual", "scenA": "A – Individual",
        "scenB_static": "B – Static",         "scenB": "B – Static",
        "scenC_dynamic": "C – Dynamic",        "scenC": "C – Dynamic",
    }
    display_labels = [_DISPLAY.get(lbl, lbl) for lbl in scenario_labels]
    x = np.arange(len(conn_types))
    width = 0.8 / max(len(results), 1)
    palette = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    for i, r in enumerate(results):
        et = r.get("edge_throughput", {})
        totals = []
        for ct in conn_types:
            total = sum(et.get(k, 0) for k in type_edges[ct])
            totals.append(total)
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, totals, width=width * 0.9,
                      label=display_labels[i],
                      color=palette[i % len(palette)],
                      edgecolor="white", alpha=0.88)
        for bar, val in zip(bars, totals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(int(val)), ha="center", va="bottom",
                        fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ct.capitalize() for ct in conn_types], fontsize=10)
    ax.set_ylabel("Total edge traversals", fontsize=10)
    ax.set_title("Connector Utilisation by Scenario", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = od / "connector_load_bars.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] connector_load_bars → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge throughput heat map (aggregated, per level)
# ─────────────────────────────────────────────────────────────────────────────

def fig_edge_throughput_heatmap(
    result: dict,
    g: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    label: str = "",
    *,
    cfg: dict | None = None,
) -> Path:
    """Draw edges coloured by traversal count — thesis-quality edge heat map.

    Wider, redder edges = more agents passed through.
    """
    setup_matplotlib_font()
    od = _edir(out_dir)
    et = result.get("edge_throughput", {})

    # Parse edge keys
    edge_vals: dict[tuple[str, str], int] = {}
    for key, val in et.items():
        parts = key.split("|", 1)
        if len(parts) == 2:
            edge_vals[(parts[0], parts[1])] = val

    vmax = max(edge_vals.values(), default=1)
    cmap = plt.cm.get_cmap("YlOrRd")
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI)

    for ax, lvl in zip(axes, WALKABLE_LEVELS):
        _plot_floor(ax, geometries.get(lvl, {}), lvl)

        segments, colours, widths = [], [], []
        for (u, v), cnt in edge_vals.items():
            u_nd = g.nodes.get(u, {})
            v_nd = g.nodes.get(v, {})
            if u_nd.get("level") != lvl and v_nd.get("level") != lvl:
                continue
            x0, y0 = u_nd.get("x"), u_nd.get("y")
            x1, y1 = v_nd.get("x"), v_nd.get("y")
            if None in (x0, y0, x1, y1):
                continue
            segments.append([(x0, y0), (x1, y1)])
            rgba = cmap(norm(cnt))
            colours.append(rgba)
            widths.append(max(0.5, min(4.5, 0.5 + cnt * 0.08)))

        if segments:
            lc = mcoll.LineCollection(segments, colors=colours,
                                      linewidths=widths, alpha=0.85, zorder=5)
            ax.add_collection(lc)
            ax.autoscale_view()

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="Traversals")

        ax.set_aspect("equal")
        ax.set_title(f"{lvl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

    scenario_str = f" — {label}" if label else ""
    fig.suptitle(f"Edge Throughput Heat Map{scenario_str}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = od / f"edge_throughput_heatmap{'_' + label if label else ''}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] edge_throughput_heatmap → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 8. Wheelchair vs normal path comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_wheelchair_path_comparison(
    g: nx.Graph,
    normal_path: list[str],
    wheelchair_path: list[str],
    geometries: dict,
    out_dir: str | Path,
    origin_label: str = "",
    dest_label: str = "",
    *,
    cfg: dict | None = None,
) -> Path:
    """Side-by-side path traces: normal routing vs wheelchair-accessible routing.

    Shows per-floor connector usage differences.
    """
    setup_matplotlib_font()
    od = _edir(out_dir)

    def _path_xy_by_level(path: list[str]) -> dict[str, tuple[list, list]]:
        by_lvl: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for nid in path:
            if nid not in g:
                continue
            nd = g.nodes[nid]
            lvl = nd.get("level", "")
            if lvl in WALKABLE_LEVELS:
                by_lvl[lvl][0].append(nd["x"])
                by_lvl[lvl][1].append(nd["y"])
        return dict(by_lvl)

    normal_xy = _path_xy_by_level(normal_path)
    wc_xy = _path_xy_by_level(wheelchair_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=DPI)
    titles = ["Normal routing", "Wheelchair routing"]
    paths_data = [normal_xy, wc_xy]
    colours = ["#1565C0", "#E53935"]

    for row, (path_xy, title, colour) in enumerate(zip(paths_data, titles, colours)):
        for col, lvl in enumerate(WALKABLE_LEVELS):
            ax = axes[row, col]
            _plot_floor(ax, geometries.get(lvl, {}), lvl)

            xs, ys = path_xy.get(lvl, ([], []))
            if len(xs) >= 2:
                ax.plot(xs, ys, color=colour, linewidth=2.2, alpha=0.88, zorder=5)
                ax.scatter([xs[0]], [ys[0]], color=colour, s=60, zorder=7,
                           edgecolors="white", linewidths=1.0, marker="o")
                ax.scatter([xs[-1]], [ys[-1]], color=colour, s=60, zorder=7,
                           edgecolors="white", linewidths=1.0, marker="s")

            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{title}\n{lvl}", fontsize=9, fontweight="bold")
            else:
                ax.set_title(lvl, fontsize=9, fontweight="bold")

    # Path summaries
    def _summarise(path: list[str]) -> str:
        if not path:
            return "No path"
        edge_types: dict[str, int] = defaultdict(int)
        total_len = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if g.has_edge(u, v):
                d = g.edges[u, v]
                edge_types[d.get("edge_type", "floor")] += 1
                total_len += float(d.get("length_3d", 0))
        connector_str = ", ".join(f"{k}×{v}" for k, v in edge_types.items()
                                  if k != "floor")
        return f"{len(path)} nodes, {total_len:.1f} m, connectors: {connector_str or 'none'}"

    fig.text(0.01, 0.02,
             f"Normal:      {_summarise(normal_path)}\n"
             f"Wheelchair: {_summarise(wheelchair_path)}",
             fontsize=8.5, color="#333", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.8))

    od_str = f"{origin_label} → {dest_label}" if origin_label else ""
    fig.suptitle(
        f"Path Comparison: Normal vs Wheelchair Routing  {od_str}\n"
        "● start   ■ end",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = od / "wheelchair_path_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] wheelchair_path_comparison → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 9. Flow-difference map between two scenarios (reuse-friendly)
# ─────────────────────────────────────────────────────────────────────────────

def fig_flow_diff_two_scenarios(
    result_a: dict,
    result_b: dict,
    g: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    *,
    cfg: dict | None = None,
) -> Path:
    """Diverging edge throughput diff: B minus A per level."""
    setup_matplotlib_font()
    od = _edir(out_dir)

    et_a = result_a.get("edge_throughput", {})
    et_b = result_b.get("edge_throughput", {})
    all_keys = set(et_a) | set(et_b)
    diff = {k: et_b.get(k, 0) - et_a.get(k, 0) for k in all_keys}

    all_diffs = list(diff.values())
    if not all_diffs:
        print("[viz_thesis] flow_diff: no edge throughput data — skipping figure")
        return od / "flow_diff_skipped.txt"
    vmax = max(1, max(abs(v) for v in all_diffs))
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#1565C0", "#90CAF9", "#E0E0E0", "#EF9A9A", "#B71C1C"], N=256)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI,
                               constrained_layout=True)

    for ax, lvl in zip(axes, WALKABLE_LEVELS):
        _plot_floor(ax, geometries.get(lvl, {}), lvl)
        segments, colours, widths = [], [], []

        for edge_key, dv in diff.items():
            u, v = edge_key.split("|", 1)
            u_nd = g.nodes.get(u, {})
            v_nd = g.nodes.get(v, {})
            if u_nd.get("level") != lvl and v_nd.get("level") != lvl:
                continue
            x0, y0 = u_nd.get("x"), u_nd.get("y")
            x1, y1 = v_nd.get("x"), v_nd.get("y")
            if None in (x0, y0, x1, y1):
                continue
            segments.append([(x0, y0), (x1, y1)])
            colours.append(cmap(norm(dv)))
            widths.append(max(0.4, min(4.0, 0.4 + abs(dv) * 0.08)))

        if segments:
            lc = mcoll.LineCollection(segments, colors=colours,
                                      linewidths=widths, alpha=0.80, zorder=5)
            ax.add_collection(lc)
            ax.autoscale_view()

        ax.set_aspect("equal")
        ax.set_title(f"{lvl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.02)
    cbar.set_label(f"Δ traversals  ({label_b} − {label_a})", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Route Flow Difference: {label_b} \u2212 {label_a}\n"
        f"(+) Red = more traffic in {label_b}  \u00b7  (\u2212) Blue = more traffic in {label_a}",
        fontsize=12, fontweight="bold",
    )
    fname = f"flow_diff_{label_a}_vs_{label_b}.png"
    p = od / fname
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] flow_diff → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 10. Capacity / system-boundary sweep curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_capacity_curves(
    sweep_results: list[dict],
    out_dir: str | Path,
    *,
    cfg: dict | None = None,
) -> Path:
    """Four-panel figure summarising a capacity sweep over agent count N.

    Panels
    ------
    (0,0) System throughput (agents/min) vs N — saturation curve
    (0,1) Arrival rate (%) vs N — shows saturation / collapse
    (1,0) Mean travel time (s) vs N — congestion growth
    (1,1) Rerouting events (dynamic only) vs N — planning overhead

    Saturation threshold (90 % arrival rate) is annotated as a vertical
    dashed line and shaded region on panels (0,0) and (0,1).
    """
    setup_matplotlib_font()
    od = _edir(out_dir)

    from collections import defaultdict as _dd

    by_label: dict[str, list[dict]] = _dd(list)
    for r in sweep_results:
        by_label[r.get("label", "run")].append(r)

    # Consistent palette: static = blue, dynamic = red
    label_colours = {"static": "#1565C0", "dynamic": "#E53935"}
    label_markers  = {"static": "o",       "dynamic": "s"}
    SAT_THRESH = 90.0          # arrival-rate saturation threshold (%)
    SATURATION_COLOUR = "#FF6F00"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

    # Track saturation N for annotation
    sat_n: dict[str, float | None] = {}

    for lbl, runs in by_label.items():
        runs_sorted = sorted(runs, key=lambda r: r["n_agents"])
        ns   = [r["n_agents"] for r in runs_sorted]
        T_sv = [r.get("T_s", (cfg["simulation"]["T_s"] if cfg else 600))
                for r in runs_sorted]

        arrive_pct  = [r.get("arrive_rate", 0) * 100 for r in runs_sorted]
        throughputs = [
            r.get("arrive_rate", 0) * r["n_agents"] / (t / 60.0)
            for r, t in zip(runs_sorted, T_sv)
        ]
        mean_tts = [
            # Prefer full list; fall back to precomputed mean_travel_time key
            (sum(r["travel_times"]) / len(r["travel_times"]))
            if r.get("travel_times")
            else r.get("mean_travel_time", float("nan"))
            for r in runs_sorted
        ]
        replan_counts = [
            # Prefer event list if non-empty; fall back to total_replans from summary
            len(r["replan_events"]) if r.get("replan_events")
            else r.get("total_replans", 0)
            for r in runs_sorted
        ]

        col = label_colours.get(lbl, "#43A047")
        mk  = label_markers.get(lbl, "D")
        kw  = dict(color=col, marker=mk, linewidth=2, markersize=6, alpha=0.9)

        # Find saturation N (first N where arrival_rate < threshold)
        _sat = None
        for n, ap in zip(ns, arrive_pct):
            if ap < SAT_THRESH:
                _sat = n
                break
        sat_n[lbl] = _sat

        axes[0, 0].plot(ns, throughputs, label=lbl.capitalize(), **kw)
        axes[0, 1].plot(ns, arrive_pct,  label=lbl.capitalize(), **kw)
        axes[1, 0].plot(ns, mean_tts,    label=lbl.capitalize(), **kw)

        # Panel (1,1): replan events only for dynamic; skip for static
        if lbl == "dynamic" and any(rc > 0 for rc in replan_counts):
            axes[1, 1].plot(ns, replan_counts, label="Replan events", **kw)
        elif lbl == "static":
            # Plot a flat zero reference for static
            axes[1, 1].plot(ns, [0] * len(ns), color=col, marker=mk,
                            linewidth=1.2, markersize=4, alpha=0.5,
                            linestyle="--", label="Static (0 replans)")

    # ── Saturation annotations ────────────────────────────────────────────
    for ax_idx, ax in enumerate([axes[0, 0], axes[0, 1]]):
        for lbl, sn in sat_n.items():
            if sn is not None:
                col = label_colours.get(lbl, SATURATION_COLOUR)
                ax.axvline(sn, color=col, linestyle="--", linewidth=1.4,
                           alpha=0.6, zorder=3)
                ax.axvspan(sn, ax.get_xlim()[1] if ax.get_xlim()[1] > sn else sn + 200,
                           alpha=0.06, color=col, zorder=2)
                if ax_idx == 1:
                    ax.text(sn, SAT_THRESH + 1,
                            f"  {lbl}\n  sat. N={sn}",
                            fontsize=7, color=col, va="bottom")

    # ── Panel styling ─────────────────────────────────────────────────────
    panel_cfg = [
        (axes[0, 0], "System Throughput vs Agent Load",
         "Input agent count N", "Throughput (agents / min)"),
        (axes[0, 1], "Arrival Rate vs Agent Load (Saturation Curve)",
         "Input agent count N", "Arrival rate (%)"),
        (axes[1, 0], "Mean Travel Time vs Agent Load",
         "Input agent count N", "Mean travel time (s)"),
        (axes[1, 1], "Dynamic Rerouting Events vs Agent Load",
         "Input agent count N", "Replan events (cumulative)"),
    ]
    for ax, title, xlabel, ylabel in panel_cfg:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

    # Reference lines
    axes[0, 1].axhline(100, color="#9E9E9E", linestyle=":", linewidth=1.2)
    axes[0, 1].axhline(SAT_THRESH, color=SATURATION_COLOUR, linestyle=":",
                       linewidth=1.0, alpha=0.7,
                       label=f"{SAT_THRESH:.0f}% threshold")
    axes[0, 1].set_ylim(50, 108)
    axes[0, 1].legend(fontsize=8)

    fig.suptitle(
        "Capacity Analysis — System Boundary Sweep\n"
        "Throughput, arrival rate, travel time and rerouting overhead vs input load",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = od / "capacity_curves.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_thesis] capacity_curves → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 11. Evaluation summary — Markdown table + condensed comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    results: list[dict],
    out_dir: str | Path,
    *,
    capacity_results: list[dict] | None = None,
    cfg: dict | None = None,
) -> tuple[Path, Path]:
    """Write a Markdown evaluation table and a condensed comparison figure.

    Parameters
    ----------
    results:
        List of per-scenario result dicts (scenA, B, C).  Each must have
        ``label``, ``arrive_rate``, ``travel_times``, ``wait_times``,
        ``elderly_travel_times``, ``normal_travel_times``, ``replan_events``,
        ``edge_throughput``.
    out_dir:
        Output directory for the Markdown file and figure.
    capacity_results:
        Optional flat list from the capacity sweep (for saturation summary).

    Returns
    -------
    (markdown_path, figure_path)
    """
    setup_matplotlib_font()
    od = _edir(out_dir)

    # ── Build per-scenario metrics dict ──────────────────────────────────────
    def _metrics(r: dict) -> dict:
        tt = r.get("travel_times", [])
        wt = r.get("wait_times", [])
        ett = r.get("elderly_travel_times", [])
        ntt = r.get("normal_travel_times", [])
        rp = r.get("replan_events", [])
        arrive = r.get("arrive_rate", 0) * 100
        mean_tt = sum(tt) / len(tt) if tt else float("nan")
        mean_wt = sum(wt) / len(wt) if wt else float("nan")
        mean_ett = sum(ett) / len(ett) if ett else float("nan")
        mean_ntt = sum(ntt) / len(ntt) if ntt else float("nan")
        # connector breakdown from edge_throughput
        # Keys are "nodeA|nodeB"; node IDs embed connector type names
        et = r.get("edge_throughput", {})
        stair_use = float(sum(v for k, v in et.items()
                              if "stair" in k))
        esc_use   = float(sum(v for k, v in et.items()
                              if "|esc_" in k or k.startswith("esc_")))
        elev_use  = float(sum(v for k, v in et.items()
                              if "elev" in k or "lift" in k))
        return dict(
            arrive_pct=arrive,
            mean_tt=mean_tt,
            mean_wt=mean_wt,
            mean_ett=mean_ett,
            mean_ntt=mean_ntt,
            replans=len(rp),
            stair_use=stair_use,
            esc_use=esc_use,
            elev_use=elev_use,
            n_agents=r.get("n_agents", len(tt) + (r.get("n_agents", 0) - len(tt))),
        )

    metric_rows = [(r.get("label", f"S{i}"), _metrics(r))
                   for i, r in enumerate(results)]

    # ── Markdown report ───────────────────────────────────────────────────────
    _DLBL = {
        "scenA_individual": "A – Individual", "scenA": "A – Individual",
        "scenB_static": "B – Static",         "scenB": "B – Static",
        "scenC_dynamic": "C – Dynamic",        "scenC": "C – Dynamic",
    }
    display_labels = [(_DLBL.get(lbl, lbl), m) for lbl, m in metric_rows]

    lines = [
        "# Evaluation Summary\n",
        "## Scenario Metrics\n",
        "| Metric | " + " | ".join(lbl for lbl, _ in display_labels) + " |",
        "|--------|" + "|".join(["-------"] * len(display_labels)) + "|",
    ]

    def _fmt(v):
        if isinstance(v, float):
            return "—" if v != v else f"{v:.1f}"  # NaN check
        return str(v)

    metric_labels = [
        ("arrive_pct",  "Arrival rate (%)"),
        ("mean_tt",     "Mean travel time (s)"),
        ("mean_wt",     "Mean wait time (s)"),
        ("mean_ntt",    "Mean TT — Normal (s)"),
        ("mean_ett",    "Mean TT — Elderly (s)"),
        ("replans",     "Total replan events"),
        ("stair_use",   "Stair traversals"),
        ("esc_use",     "Escalator traversals"),
        ("elev_use",    "Elevator traversals"),
    ]
    for key, label in metric_labels:
        row_vals = [_fmt(m[key]) for _, m in display_labels]
        lines.append(f"| {label} | " + " | ".join(row_vals) + " |")

    # B vs C delta block
    if len(metric_rows) >= 2:
        b_m = next((m for lbl, m in metric_rows if "static" in lbl or lbl == "B"), None)
        c_m = next((m for lbl, m in metric_rows if "dynamic" in lbl or lbl == "C"), None)
        if b_m and c_m:
            lines += [
                "\n## B (Static) vs C (Dynamic) — Key Deltas\n",
                "| Metric | B | C | Δ (C − B) | Δ% |",
                "|--------|---|---|-----------|-----|",
            ]
            for key, label in [
                ("arrive_pct",  "Arrival rate (%)"),
                ("mean_tt",     "Mean travel time (s)"),
                ("mean_wt",     "Mean wait time (s)"),
                ("stair_use",   "Stair traversals"),
                ("esc_use",     "Escalator traversals"),
            ]:
                bv, cv = b_m[key], c_m[key]
                if isinstance(bv, float) and bv == bv and cv == cv:
                    delta = cv - bv
                    pct = (delta / abs(bv) * 100) if bv != 0 else float("nan")
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"| {label} | {_fmt(bv)} | {_fmt(cv)} |"
                        f" {sign}{_fmt(delta)} | {sign}{_fmt(pct)}% |"
                    )
                else:
                    lines.append(f"| {label} | {_fmt(bv)} | {_fmt(cv)} | — | — |")

    # Capacity saturation summary
    if capacity_results:
        lines += ["\n## Capacity Sweep — Saturation Summary\n",
                  "| Mode | N | Arrival (%) | Mean TT (s) | Replans |",
                  "|------|---|-------------|-------------|---------|"]
        for r in sorted(capacity_results, key=lambda r: (r.get("label", ""), r["n_agents"])):
            lbl = r.get("label", "?")
            n   = r["n_agents"]
            ap  = r.get("arrive_rate", 0) * 100
            tt  = r.get("travel_times", [])
            mtt = sum(tt) / len(tt) if tt else float("nan")
            rp  = len(r.get("replan_events", []))
            lines.append(f"| {lbl} | {n} | {ap:.1f}% | {_fmt(mtt)} | {rp} |")

    md_path = od / "evaluation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eval] Markdown report → {md_path}")

    # ── Condensed comparison figure (3-panel) ────────────────────────────────
    if len(metric_rows) < 2:
        return md_path, md_path

    # Use display labels for the figure (already computed above)
    scen_labels = [lbl for lbl, _ in display_labels]
    palette = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A"]
    bar_colours = [palette[i % len(palette)] for i in range(len(display_labels))]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    def _hbar(ax, vals, labels, title, xlabel, colours, fmt="{:.1f}"):
        ys = range(len(labels))
        bars = ax.barh(list(ys), vals, color=colours, edgecolor="white",
                       alpha=0.88, height=0.55)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.2 if any(v > 0 for v in vals) else 1)

    arrive_vals = [m["arrive_pct"] for _, m in display_labels]
    tt_vals     = [m["mean_tt"] for _, m in display_labels]
    wt_vals     = [m["mean_wt"] for _, m in display_labels]

    _hbar(ax1, arrive_vals, scen_labels, "Arrival Rate", "(%)", bar_colours, "{:.1f}%")
    _hbar(ax2, tt_vals,     scen_labels, "Mean Travel Time", "(s)", bar_colours)
    _hbar(ax3, wt_vals,     scen_labels, "Mean Wait Time", "(s)", bar_colours)
    ax1.axvline(100, color="#9E9E9E", linestyle=":", linewidth=1.2)

    fig.suptitle("Evaluation Summary — Scenario Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_path = od / "evaluation_summary.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Summary figure → {fig_path}")
    return md_path, fig_path
