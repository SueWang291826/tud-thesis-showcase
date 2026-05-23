"""
Visualization Module
=====================

Thesis-quality matplotlib visualizations for multi-level indoor navigation.

Figure groups:
  A. Per-level geometry & nodes
  B. Unified 2.5D multi-level view
  C. Connector details
  D. Graph topology
  E. Simulation frames & GIF
  F. Evaluation comparison charts
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon, MultiPolygon

from src.utils import setup_matplotlib_font, flatten_polygons, save_gif

# ============================================================================
# Defaults & colour palettes
# ============================================================================

LEVEL_COLOURS = {
    "F1": "#2196F3",   # blue
    "F2": "#9E9E9E",   # grey
    "F3": "#4CAF50",   # green
    "F4": "#FF9800",   # orange
}

CONNECTOR_COLOURS = {
    "stair": "#795548",
    "escalator": "#E91E63",
    "elevator": "#9C27B0",
    "elevator_door": "#AB47BC",
    "elevator_interior": "#CE93D8",
    "psd_door": "#FF9800",
    "anchor_snap": "#4CAF50",
    "fare_gate": "#607D8B",
    "floor": "#BDBDBD",
}

EDGE_TYPE_STYLES = {
    "floor":              {"color": "#BDBDBD", "linewidth": 0.3, "alpha": 0.4},
    "stair":              {"color": "#795548", "linewidth": 1.5, "alpha": 0.9},
    "escalator":          {"color": "#E91E63", "linewidth": 1.8, "alpha": 0.9},
    "elevator":           {"color": "#9C27B0", "linewidth": 2.0, "alpha": 0.9},
    "elevator_door":      {"color": "#AB47BC", "linewidth": 1.2, "alpha": 0.8},
    "elevator_interior":  {"color": "#CE93D8", "linewidth": 0.8, "alpha": 0.6},
    "psd_door":           {"color": "#FF9800", "linewidth": 1.0, "alpha": 0.8},
    "anchor_snap":        {"color": "#4CAF50", "linewidth": 1.0, "alpha": 0.7},
}


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_viz_params(cfg: dict) -> dict:
    v = cfg.get("visualization", {})
    return {
        "dpi": v.get("dpi", 180),
        "figsize_single": v.get("figsize_single", [10, 8]),
        "figsize_comparison": v.get("figsize_comparison", [14, 6]),
        "gif_fps": v.get("gif_fps", 4),
        "gif_dt_frame_s": v.get("gif_dt_frame_s", 2.0),
    }


# ============================================================================
# A. Per-level geometry plots
# ============================================================================

def _plot_polygon(ax, poly: Polygon, **kwargs):
    """Plot a Shapely Polygon on a matplotlib Axes."""
    if poly is None or poly.is_empty:
        return
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, **kwargs)
    for interior in poly.interiors:
        ix, iy = interior.xy
        ax.fill(ix, iy, color="white", alpha=0.9)


def _plot_multipolygon(ax, geom, **kwargs):
    """Plot a Polygon or MultiPolygon."""
    for p in flatten_polygons(geom):
        _plot_polygon(ax, p, **kwargs)


def plot_level_geometry(
    ax,
    level_id: str,
    floor_poly,
    obstacle_polys: list,
    walkable_poly=None,
    title: str | None = None,
    forbidden_zones: list | None = None,
    track_zones: list | None = None,
    entrances: list | None = None,
    blind_paths: list | None = None,
):
    """Plot floor, obstacles, walkable, forbidden zones, tracks, entrances, blind paths."""
    if floor_poly and not floor_poly.is_empty:
        _plot_multipolygon(ax, floor_poly, color="#E3F2FD", alpha=0.5,
                           edgecolor="#1565C0", linewidth=0.8, label="Floor")

    for obs in (obstacle_polys or []):
        _plot_multipolygon(ax, obs, color="#EF5350", alpha=0.55, edgecolor="#B71C1C", linewidth=0.4)

    if walkable_poly and not walkable_poly.is_empty:
        _plot_multipolygon(ax, walkable_poly, color="#C8E6C9", alpha=0.25,
                           edgecolor="#2E7D32", linewidth=0.5, linestyle="--")

    # Forbidden zones (hatched)
    for fz in (forbidden_zones or []):
        _plot_multipolygon(ax, fz, color="#FF1744", alpha=0.18,
                           edgecolor="#D50000", linewidth=1.0, linestyle="--")

    # Track zones (yellow band)
    for tz in (track_zones or []):
        poly = tz.get("polygon") if isinstance(tz, dict) else tz
        if poly and not poly.is_empty:
            _plot_multipolygon(ax, poly, color="#FFD600", alpha=0.15,
                               edgecolor="#F57F17", linewidth=1.0, linestyle=":")

    # Entrances (green)
    for ent in (entrances or []):
        poly = ent.get("polygon") if isinstance(ent, dict) else ent
        if poly and not poly.is_empty:
            _plot_multipolygon(ax, poly, color="#00E676", alpha=0.6,
                               edgecolor="#1B5E20", linewidth=1.5)

    # Blind paths (tactile paving)
    for bp in (blind_paths or []):
        fp = bp.get("footprint") if isinstance(bp, dict) else bp
        if fp and not fp.is_empty:
            cat = bp.get("category", "guide") if isinstance(bp, dict) else "guide"
            if cat == "warning":
                _plot_multipolygon(ax, fp, color="#FF6F00", alpha=0.85,
                                   edgecolor="#E65100", linewidth=0.3)
            else:  # guide
                _plot_multipolygon(ax, fp, color="#FFAB00", alpha=0.75,
                                   edgecolor="#FF8F00", linewidth=0.3)

    ax.set_aspect("equal")
    ax.set_title(title or f"Level {level_id}", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)


def plot_level_nodes(
    ax,
    nodes: list[tuple[float, float]],
    level_id: str,
    color: str | None = None,
    s: float = 1.0,
):
    """Scatter plot of sampled nodes for one level."""
    if not nodes:
        return
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    c = color or LEVEL_COLOURS.get(level_id, "#666")
    ax.scatter(xs, ys, c=c, s=s, marker=".", alpha=0.6, zorder=5)


def fig_all_levels_geometry(
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Generate a multi-panel figure showing geometry for all levels.
    
    Parameters
    ----------
    geometries : dict
        level_id -> dict with floor, obstacles, obstacle_union, walkable, bbox
    """
    vp = _get_viz_params(cfg)
    levels = sorted(geometries.keys())
    n = len(levels)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), dpi=vp["dpi"])
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, lvl in enumerate(levels):
        g = geometries[lvl]
        ax = axes[i]
        plot_level_geometry(
            ax, lvl,
            floor_poly=g.get("floor"),
            obstacle_polys=g.get("obstacles", []),
            walkable_poly=g.get("walkable"),
            title=f"{lvl}",
            forbidden_zones=g.get("forbidden_zone_polys"),
            track_zones=g.get("track_zones"),
            entrances=g.get("entrances"),
            blind_paths=g.get("blind_paths"),
        )

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Level Geometry — Floor / Obstacles / Walkable", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = _ensure_dir(out_dir) / "all_levels_geometry.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_all_levels_nodes(
    geometries: dict,
    level_nodes: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Multi-panel per-level geometry with sampled nodes overlay."""
    vp = _get_viz_params(cfg)
    levels = sorted(geometries.keys())
    n = len(levels)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), dpi=vp["dpi"])
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, lvl in enumerate(levels):
        g = geometries[lvl]
        ax = axes[i]
        plot_level_geometry(ax, lvl, g.get("floor"), g.get("obstacles", []),
                            g.get("walkable"),
                            forbidden_zones=g.get("forbidden_zone_polys"),
                            track_zones=g.get("track_zones"),
                            entrances=g.get("entrances"),
                            blind_paths=g.get("blind_paths"))
        nodes = level_nodes.get(lvl, [])
        plot_level_nodes(ax, nodes, lvl, s=0.8)
        ax.set_title(f"{lvl}  ({len(nodes)} nodes)", fontsize=11, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Sampled Nodes per Level", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = _ensure_dir(out_dir) / "all_levels_nodes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# B. Unified 2.5D isometric view  (v4 — full 3-D connector geometry)
# ============================================================================

# ---- 3-D geometry helpers ------------------------------------------------

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """'#RRGGBB' → (r, g, b) in [0, 1]."""
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _darken(hex_color: str, factor: float = 0.55) -> str:
    """Return a darker shade for edge highlights."""
    r, g, b = _hex_to_rgb(hex_color)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor * 255), int(g * factor * 255), int(b * factor * 255))


def _inclined_slab(top4, thickness: float) -> list[list[tuple]]:
    """6 quad-faces of a slab given its 4 top-surface vertices (CCW)."""
    bot4 = [(x, y, z - thickness) for x, y, z in top4]
    t, b = top4, bot4
    return [
        list(t), list(b),
        [t[0], t[1], b[1], b[0]],
        [t[1], t[2], b[2], b[1]],
        [t[2], t[3], b[3], b[2]],
        [t[3], t[0], b[0], b[3]],
    ]


def _box_faces(lo, hi) -> list[list[tuple]]:
    """6 quad-faces of an axis-aligned box from two corners."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _draw_stair_chain_3d(ax, conn, elevations, z_scale, slab_h,
                         dz_step: float = 0.18):
    """Render each *run* of a stair chain as per-step horizontal platforms."""
    runs = conn.get("runs", [])
    if not runs:
        return
    color = CONNECTOR_COLOURS["stair"]
    edge_c = _darken(color)

    first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
    last_xc  = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
    asc_x = first_xc < last_xc

    for run in runs:
        x0, x1 = run["min_x"], run["max_x"]
        y0, y1 = run["min_y"], run["max_y"]
        zlo = run["z_min"] * z_scale
        zhi = run["z_max"] * z_scale
        dz = abs(zhi - zlo)
        n_steps = max(1, int(round(dz / (dz_step * z_scale))))
        run_dx = x1 - x0
        step_dx = abs(run_dx) / n_steps

        for k in range(n_steps):
            frac = (k + 0.5) / n_steps
            z_k = zlo + frac * (zhi - zlo)
            if asc_x:
                cx = x0 + frac * run_dx
            else:
                cx = x1 - frac * abs(run_dx)
            sx0, sx1 = cx - step_dx / 2, cx + step_dx / 2
            # brightness gradient
            brightness = 1.0 - 0.3 * frac
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            step_color = "#{:02x}{:02x}{:02x}".format(
                int(r * brightness), int(g * brightness), int(b * brightness))
            top = [(sx0, y0, z_k), (sx0, y1, z_k),
                   (sx1, y1, z_k), (sx1, y0, z_k)]
            faces = _inclined_slab(top, slab_h * 0.4)
            ax.add_collection3d(Poly3DCollection(
                faces, alpha=0.55, facecolor=step_color,
                edgecolor=edge_c, linewidths=0.3, zorder=5))

    # landing platforms
    hw = (runs[0]["max_y"] - runs[0]["min_y"]) / 2
    hl = 1.5
    for ld in conn.get("landings", []):
        lx, ly, lz = ld["x"], ld["y"], ld["z"] * z_scale
        top = [(lx - hl, ly - hw, lz), (lx - hl, ly + hw, lz),
               (lx + hl, ly + hw, lz), (lx + hl, ly - hw, lz)]
        ax.add_collection3d(Poly3DCollection(
            _inclined_slab(top, slab_h * 0.5),
            alpha=0.40, facecolor=color,
            edgecolor=edge_c, linewidths=0.3, zorder=5))


def _draw_escalator_3d(ax, conn, elevations, z_scale, slab_h,
                       dz_step: float = 0.40):
    """Render an escalator as per-step horizontal platforms between its two levels.

    Parameters
    ----------
    dz_step : float
        Escalator step riser height in metres (unscaled).
    """
    color = CONNECTOR_COLOURS["escalator"]
    edge_c = _darken(color)

    bot_lv = conn.get("bottom_level")
    top_lv = conn.get("top_level")
    if not (bot_lv and top_lv):
        return
    if bot_lv not in elevations or top_lv not in elevations:
        return

    zlo = elevations[bot_lv] * z_scale
    zhi = elevations[top_lv] * z_scale

    # Use physical landing positions if available
    bot_xy = conn.get("bottom_xy")
    top_xy = conn.get("top_xy")
    if bot_xy and top_xy:
        asc_x = bot_xy[0] < top_xy[0]
        x0 = min(bot_xy[0], top_xy[0])
        x1 = max(bot_xy[0], top_xy[0])
    else:
        x0 = conn.get("min_x", 0)
        x1 = conn.get("max_x", 0)
    y0 = conn.get("min_y", 0)
    y1 = conn.get("max_y", 0)
    if x1 - x0 < 0.1:
        return

    dz = abs(zhi - zlo)
    n_steps = max(1, int(round(dz / (dz_step * z_scale))))
    run_dx = x1 - x0
    step_dx = abs(run_dx) / n_steps

    for k in range(n_steps):
        frac = (k + 0.5) / n_steps
        z_k = zlo + frac * (zhi - zlo)
        if asc_x:
            cx = x0 + frac * run_dx
        else:
            cx = x1 - frac * abs(run_dx)
        sx0, sx1 = cx - step_dx / 2, cx + step_dx / 2
        brightness = 1.0 - 0.3 * frac
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        step_color = "#{:02x}{:02x}{:02x}".format(
            int(r * brightness), int(g * brightness), int(b * brightness))
        top = [(sx0, y0, z_k), (sx0, y1, z_k),
               (sx1, y1, z_k), (sx1, y0, z_k)]
        faces = _inclined_slab(top, slab_h * 0.3)
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.45, facecolor=step_color,
            edgecolor=edge_c, linewidths=0.3, zorder=5))


def _draw_elevator_3d(ax, conn, elevations, z_scale):
    """Render an elevator shaft as a vertical box between served levels."""
    color = CONNECTOR_COLOURS["elevator"]
    edge_c = _darken(color)

    served = [lk for lk in conn.get("connected_levels", [])
              if lk in elevations]
    if len(served) < 2:
        return
    zs = sorted([elevations[lk] * z_scale for lk in served])

    fp = conn.get("footprint")
    if not (fp and not fp.is_empty):
        return
    x0, y0, x1, y1 = fp.bounds

    # Shaft body
    faces = _box_faces((x0, y0, zs[0]), (x1, y1, zs[-1]))
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=0.35, facecolor=color,
        edgecolor=edge_c, linewidths=0.5, zorder=6))

    # Horizontal floor indicators at each served level
    for lk in served:
        ze = elevations[lk] * z_scale
        quad = [(x0, y0, ze), (x1, y0, ze), (x1, y1, ze), (x0, y1, ze)]
        ax.add_collection3d(Poly3DCollection(
            [quad], alpha=0.60, facecolor=color,
            edgecolor=edge_c, linewidths=0.6, zorder=7))


# ---- main isometric figure -----------------------------------------------

def fig_multilevel_isometric(
    geometries: dict,
    level_nodes: dict,
    elevations: dict,
    out_dir: str | Path,
    cfg: dict,
    z_scale: float = 3.0,
    all_connectors: list[dict] | None = None,
) -> Path:
    """
    2.5D stacked view — each level at its real elevation (×z_scale),
    with **3-D connector geometry** (inclined stair slabs, escalator
    ramps, elevator shafts) drawn between the floor planes.
    """
    vp = _get_viz_params(cfg)
    fig = plt.figure(figsize=(16, 11), dpi=vp["dpi"])
    ax = fig.add_subplot(111, projection="3d")

    levels = sorted(elevations.keys(), key=lambda k: elevations[k])
    slab_h = 0.35 * z_scale          # visual thickness of inclined slabs

    # ---- Pre-compute stair ascent directions for escalator matching --
    # For each stair chain record: level_pair → (centroid_x, asc_x)
    _stair_dirs: list[tuple[tuple[str, str], float, bool]] = []
    if all_connectors:
        for c in all_connectors:
            if c["type"] != "stair_chain":
                continue
            runs = c.get("runs", [])
            if len(runs) < 2:
                continue
            first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
            last_xc  = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
            asc_x = first_xc < last_xc     # x increases with z
            lvls = tuple(sorted(c.get("connected_levels", [])))
            stair_cx = sum(r["min_x"] + r["max_x"]
                           for r in runs) / (2 * len(runs))
            _stair_dirs.append((lvls, stair_cx, asc_x))

    def _escalator_asc_x(esc_conn: dict) -> bool:
        """Return ascent direction for an escalator by matching the nearest
        stair chain that serves the same level pair."""
        bl = esc_conn.get("bottom_level", "")
        tl = esc_conn.get("top_level", "")
        pair = tuple(sorted([bl, tl]))
        exc = (esc_conn.get("min_x", 0) + esc_conn.get("max_x", 0)) / 2
        best_dist, best_asc = float("inf"), True
        for (sp, sx, sa) in _stair_dirs:
            if sp == pair:
                d = abs(sx - exc)
                if d < best_dist:
                    best_dist, best_asc = d, sa
        return best_asc

    # ---- Floor outlines + filled surfaces + nodes --------------------
    for lvl in levels:
        z = elevations[lvl] * z_scale
        g = geometries.get(lvl, {})
        floor = g.get("floor")
        colour = LEVEL_COLOURS.get(lvl, "#999")

        if floor and not floor.is_empty:
            for poly in flatten_polygons(floor):
                xs, ys = poly.exterior.xy
                zs = [z] * len(xs)
                ax.plot(xs, ys, zs, color=colour, linewidth=0.9, alpha=0.7)
                # Light fill so levels are visually distinct planes
                ax.add_collection3d(Poly3DCollection(
                    [list(zip(xs, ys, zs))],
                    alpha=0.07, facecolor=colour))

        nodes = level_nodes.get(lvl, [])
        if nodes:
            nxs = [n[0] for n in nodes]
            nys = [n[1] for n in nodes]
            nzs = [z] * len(nodes)
            ax.scatter(nxs, nys, nzs, c=colour, s=0.5, alpha=0.35)

    # ---- 3-D connector geometry --------------------------------------
    if all_connectors:
        for c in all_connectors:
            ctype = c["type"]
            if ctype == "stair_chain":
                _draw_stair_chain_3d(ax, c, elevations, z_scale, slab_h)
            elif ctype == "escalator":
                _draw_escalator_3d(ax, c, elevations, z_scale, slab_h)
            elif ctype == "elevator":
                _draw_elevator_3d(ax, c, elevations, z_scale)

    # ---- Axes & labels -----------------------------------------------
    ax.set_xlabel("X (m)", fontsize=9, labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=9, labelpad=8)
    ax.set_zlabel("Elevation (×{:.0f})".format(z_scale), fontsize=9,
                  labelpad=6)
    ax.set_title("Multi-Level 2.5D Station View",
                 fontsize=13, fontweight="bold")
    ax.view_init(elev=28, azim=-55)       # good default isometric angle

    # ---- Legend — levels + connector types ----------------------------
    handles = [mpatches.Patch(color=LEVEL_COLOURS.get(l, "#999"), label=l)
               for l in levels]
    if all_connectors:
        seen: set[str] = set()
        for c in all_connectors:
            ct = c["type"].replace("_chain", "")
            if ct not in seen:
                handles.append(mpatches.Patch(
                    color=CONNECTOR_COLOURS.get(ct, "#333"),
                    label=ct.capitalize()))
                seen.add(ct)
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.85)

    out = _ensure_dir(out_dir) / "multilevel_isometric.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# C. Connector visualizations  (v3 — accepts all_connectors list)
# ============================================================================

def fig_connectors_overview(
    all_connectors: list[dict],
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
    *,
    door_cfg: dict | None = None,
) -> Path:
    """Plot connectors grouped by type on their respective floor plans.

    Parameters
    ----------
    all_connectors : list[dict]
        Flat list of connectors from ``extract_all_connectors``.
        Each dict has ``type`` ∈ {stair_chain, escalator, elevator}
        and ``footprint`` (Shapely geometry).
    door_cfg : dict, optional
        Dynamic-door config from YAML (``station.levels.F1.dynamic_doors``
        etc.) for PSD barriers and elevator doors.
    """
    from shapely.geometry import box as shp_box

    vp = _get_viz_params(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=vp["dpi"])

    type_map = {
        "stair_chain": ("Stairs", CONNECTOR_COLOURS["stair"]),
        "escalator": ("Escalators", CONNECTOR_COLOURS["escalator"]),
        "elevator": ("Elevators", CONNECTOR_COLOURS["elevator"]),
    }

    # Group connectors by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in all_connectors:
        by_type[c["type"]].append(c)

    for ax, (ctype, (title, colour)) in zip(axes, type_map.items()):
        conns = by_type.get(ctype, [])

        # Light background geometry (skip F2)
        for lvl, g in geometries.items():
            floor = g.get("floor")
            if floor and not floor.is_empty:
                _plot_multipolygon(ax, floor, color="#F5F5F5", alpha=0.3,
                                   edgecolor="#BDBDBD", linewidth=0.3)

        # Plot connector footprints
        for c in conns:
            fp = c.get("footprint")
            if fp is not None and not fp.is_empty:
                _plot_multipolygon(ax, fp, color=colour, alpha=0.6,
                                   edgecolor="black", linewidth=0.8)
                # Label
                cx, cy = fp.centroid.x, fp.centroid.y
                label = c.get("id", "")[:12]
                ax.annotate(label, (cx, cy), fontsize=5, ha="center",
                            va="center", color="white", fontweight="bold")

            # Stair chain anchor markers
            if ctype == "stair_chain":
                for lk, anchor in c.get("level_anchors", {}).items():
                    ax.plot(anchor["x"], anchor["y"], marker="^",
                            markersize=5, color=colour, markeredgecolor="k",
                            markeredgewidth=0.3, zorder=10)

        # Add elevator shaft from dynamic door config
        if ctype == "elevator" and door_cfg:
            for lvl_cfg in door_cfg.values():
                for ed in lvl_cfg:
                    sb = ed.get("shaft_bounds")
                    if sb and len(sb) == 4:
                        shaft_poly = shp_box(*sb)
                        _plot_polygon(ax, shaft_poly, color=colour, alpha=0.5,
                                      edgecolor="black", linewidth=1.0)
                        ax.annotate(ed.get("name", "elev")[:10],
                                    (shaft_poly.centroid.x, shaft_poly.centroid.y),
                                    fontsize=5, ha="center", color="white",
                                    fontweight="bold")

        # Add PSD barriers
        if ctype == "escalator" and door_cfg:
            pass  # PSD not plotted in escalator panel

        ax.set_aspect("equal")
        ax.set_title(f"{title} ({len(conns)})", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle("Connector Overview", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _ensure_dir(out_dir) / "connectors_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# D. Graph topology
# ============================================================================

def fig_graph_per_level(
    G: nx.Graph,
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Per-level graph edges overlaid on floor plan."""
    vp = _get_viz_params(cfg)
    levels = sorted(geometries.keys())
    n = len(levels)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), dpi=vp["dpi"])
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        g = geometries[lvl]
        plot_level_geometry(ax, lvl, g.get("floor"), g.get("obstacles", []))

        # Collect edges for this level
        level_nodes = {n for n, d in G.nodes(data=True) if d.get("level") == lvl}
        edge_count = 0
        for u, v, d in G.edges(data=True):
            if u in level_nodes and v in level_nodes:
                ux, uy = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
                vx, vy = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
                etype = d.get("edge_type", "floor")
                style = EDGE_TYPE_STYLES.get(etype, EDGE_TYPE_STYLES["floor"])
                ax.plot([ux, vx], [uy, vy], **style, zorder=3)
                edge_count += 1

        ax.set_title(f"{lvl} — {len(level_nodes)} nodes, {edge_count} edges",
                      fontsize=10, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Navigation Graph per Level", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = _ensure_dir(out_dir) / "graph_per_level.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_graph_cross_level_edges(
    G: nx.Graph,
    out_dir: str | Path,
    cfg: dict,
    all_connectors: list[dict] | None = None,
) -> Path:
    """Plot ALL connector edges (not just cross-level) with footprints.

    Includes: stair, escalator, elevator, elevator_door, elevator_interior,
    psd_door, anchor_snap — i.e. every edge whose ``edge_type`` ≠ ``floor``.
    This gives a complete picture of how the levels are linked.
    """
    from shapely.geometry import box as shp_box

    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=vp["dpi"])

    # Batch-scatter nodes by level
    from collections import defaultdict as _dd
    pts_by_lvl: dict[str, tuple[list, list]] = _dd(lambda: ([], []))
    for _, d in G.nodes(data=True):
        lvl = d.get("level", "?")
        pts_by_lvl[lvl][0].append(d.get("x", 0))
        pts_by_lvl[lvl][1].append(d.get("y", 0))
    for lvl, (xs, ys) in pts_by_lvl.items():
        ax.scatter(xs, ys, c=LEVEL_COLOURS.get(lvl, "#ccc"),
                   s=0.3, alpha=0.15, linewidths=0)

    # ---- Connector footprints as background reference ----
    if all_connectors:
        for c in all_connectors:
            fp = c.get("footprint")
            ctype = c["type"].replace("_chain", "")
            colour = CONNECTOR_COLOURS.get(ctype, "#333")
            if fp and not fp.is_empty:
                _plot_multipolygon(ax, fp, color=colour, alpha=0.25,
                                   edgecolor=colour, linewidth=1.0)

    # ---- ALL non-floor edges ----
    edge_counts: dict[str, int] = defaultdict(int)
    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type", "floor")
        if etype == "floor":
            continue
        ux, uy = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
        vx, vy = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
        style = EDGE_TYPE_STYLES.get(etype, EDGE_TYPE_STYLES.get("floor", {}))
        lw = style.get("linewidth", 1.0) * 1.2
        ax.plot([ux, vx], [uy, vy], color=style.get("color", "#333"),
                linewidth=lw, alpha=style.get("alpha", 0.8), zorder=5)
        edge_counts[etype] += 1

    # Legend with counts
    handles = []
    for etype, count in sorted(edge_counts.items()):
        c = CONNECTOR_COLOURS.get(etype, "#333")
        handles.append(mpatches.Patch(color=c, label=f"{etype} ({count})"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    total_conn = sum(edge_counts.values())
    ax.set_title(f"Connector Edges — {total_conn} non-floor edges",
                 fontsize=12, fontweight="bold")

    out = _ensure_dir(out_dir) / "graph_cross_level_edges.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# E. Simulation frames & GIF
# ============================================================================

def _draw_sim_frame(
    ax,
    G: nx.Graph,
    geometries: dict,
    agents_state: list[dict],
    time_s: float,
    level_id: str | None = None,
):
    """Draw a single simulation frame on the given axes.
    
    Parameters
    ----------
    agents_state : list[dict]
        Each dict has keys: agent_id, x, y, level, status, agent_type
    level_id : str or None
        If None, plot all levels flattened; otherwise filter by level.
    """
    # Background geometry
    for lvl, g in geometries.items():
        if level_id and lvl != level_id:
            continue
        plot_level_geometry(ax, lvl, g.get("floor"), g.get("obstacles", []))

    # Agents
    for a in agents_state:
        if level_id and a.get("level") != level_id:
            continue
        colour = "#E91E63" if a.get("agent_type") == "elderly" else "#2196F3"
        marker = "o" if a.get("status") == "moving" else "x"
        alpha = 0.8 if a.get("status") == "moving" else 0.4
        ax.scatter(a["x"], a["y"], c=colour, s=8, marker=marker, alpha=alpha, zorder=10)

    ax.set_aspect("equal")
    title_lvl = level_id if level_id else "All Levels"
    ax.set_title(f"t = {time_s:.1f}s — {title_lvl}", fontsize=10)


def render_simulation_frame(
    G: nx.Graph,
    geometries: dict,
    agents_state: list[dict],
    time_s: float,
    out_path: str | Path,
    cfg: dict,
    level_id: str | None = None,
) -> Path:
    """Render and save a single simulation frame."""
    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=vp["figsize_single"], dpi=vp["dpi"])
    _draw_sim_frame(ax, G, geometries, agents_state, time_s, level_id)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_simulation_gif(
    G: nx.Graph,
    geometries: dict,
    trajectory_frames: list[dict],
    out_dir: str | Path,
    cfg: dict,
    label: str = "simulation",
    level_id: str | None = None,
) -> Path:
    """Generate GIF from a list of trajectory frames.
    
    Parameters
    ----------
    trajectory_frames : list[dict]
        Each dict has keys: time_s, agents (list of agent state dicts)
    """
    vp = _get_viz_params(cfg)
    frames_dir = _ensure_dir(Path(out_dir) / f"gif_frames_{label}")
    frame_paths = []

    for idx, frame in enumerate(trajectory_frames):
        fp = frames_dir / f"frame_{idx:05d}.png"
        render_simulation_frame(
            G, geometries,
            frame["agents"], frame["time_s"],
            fp, cfg, level_id,
        )
        frame_paths.append(fp)

    gif_path = _ensure_dir(out_dir) / f"{label}.gif"
    save_gif(frame_paths, gif_path, fps=vp["gif_fps"])
    return gif_path


def fig_simulation_snapshots(
    G: nx.Graph,
    geometries: dict,
    trajectory_frames: list[dict],
    out_dir: str | Path,
    cfg: dict,
    label: str = "snapshots",
    n_snapshots: int = 6,
) -> Path:
    """Panel of N equally-spaced snapshots from a simulation."""
    vp = _get_viz_params(cfg)
    n = min(n_snapshots, len(trajectory_frames))
    if n == 0:
        return Path(out_dir) / f"{label}_snapshots.png"

    indices = [int(i * (len(trajectory_frames) - 1) / max(n - 1, 1)) for i in range(n)]
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), dpi=vp["dpi"])
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for k, idx in enumerate(indices):
        frame = trajectory_frames[idx]
        _draw_sim_frame(axes_flat[k], G, geometries, frame["agents"], frame["time_s"])

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Simulation Snapshots — {label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _ensure_dir(out_dir) / f"{label}_snapshots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# F. Evaluation comparison charts
# ============================================================================

def fig_comparison_bar(
    metrics_list: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Side-by-side bar chart comparing key metrics across scenarios."""
    vp = _get_viz_params(cfg)
    labels = [m.get("label", f"S{i}") for i, m in enumerate(metrics_list)]
    keys = ["mean_travel_time", "median_travel_time", "p95_travel_time", "mean_wait_time"]
    display = ["Mean TT (s)", "Median TT (s)", "P95 TT (s)", "Mean Wait (s)"]

    n_groups = len(keys)
    n_bars = len(metrics_list)
    x = np.arange(n_groups)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=vp["figsize_comparison"], dpi=vp["dpi"])

    for i, m in enumerate(metrics_list):
        vals = [m.get(k, 0) for k in keys]
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m.get("label", ""), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=9)
    ax.set_ylabel("Seconds")
    ax.set_title("Scenario Comparison — Travel & Wait Times", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    out = _ensure_dir(out_dir) / "comparison_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_arrival_curve(
    scenario_results: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Cumulative arrival curves for multiple scenarios."""
    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=vp["figsize_single"], dpi=vp["dpi"])

    for result in scenario_results:
        tt = sorted(result.get("travel_times", []))
        if not tt:
            continue
        cum = np.arange(1, len(tt) + 1)
        label = result.get("label", "")
        ax.plot(tt, cum, linewidth=1.5, alpha=0.85, label=label)

    ax.set_xlabel("Travel Time (s)")
    ax.set_ylabel("Cumulative Arrivals")
    ax.set_title("Cumulative Arrival Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    out = _ensure_dir(out_dir) / "arrival_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_queue_over_time(
    scenario_results: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Stair/connector queue lengths over time."""
    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=vp["figsize_single"], dpi=vp["dpi"])

    for result in scenario_results:
        sq = result.get("stair_queue_over_time", [])
        if not sq:
            continue
        ts = [t for t, _ in sq]
        qs = [q for _, q in sq]
        label = result.get("label", "")
        ax.plot(ts, qs, linewidth=1.2, alpha=0.8, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Length")
    ax.set_title("Connector Queue Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    out = _ensure_dir(out_dir) / "queue_over_time.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_connector_utilisation(
    util_list: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Stacked bar chart showing connector share per scenario."""
    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=vp["figsize_comparison"], dpi=vp["dpi"])

    labels = [u.get("label", f"S{i}") for i, u in enumerate(util_list)]
    stair_vals  = [u.get("stair_share", 0)     for u in util_list]
    esc_vals    = [u.get("escalator_share", 0)  for u in util_list]
    elev_share  = [1 - s - e for s, e in zip(stair_vals, esc_vals)]  # elevator + floor

    x = np.arange(len(labels))
    ax.bar(x, stair_vals, 0.5, label="Stairs",
           color=CONNECTOR_COLOURS["stair"], alpha=0.85)
    ax.bar(x, esc_vals, 0.5, bottom=stair_vals, label="Escalators",
           color=CONNECTOR_COLOURS["escalator"], alpha=0.85)
    ax.bar(x, elev_share, 0.5,
           bottom=[s + e for s, e in zip(stair_vals, esc_vals)],
           label="Elevator+Floor", color=CONNECTOR_COLOURS["elevator"], alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Share")
    ax.set_title("Connector Utilisation by Scenario", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    out = _ensure_dir(out_dir) / "connector_utilisation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_elderly_vs_normal(
    metrics_list: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Grouped bar chart: elderly vs normal travel times per scenario."""
    vp = _get_viz_params(cfg)
    fig, ax = plt.subplots(figsize=vp["figsize_comparison"], dpi=vp["dpi"])

    labels = [m.get("label", f"S{i}") for i, m in enumerate(metrics_list)]
    elderly = [m.get("mean_elderly_travel", 0) for m in metrics_list]
    normal  = [m.get("mean_normal_travel", 0)  for m in metrics_list]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, normal,  w, label="Normal",  color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, elderly, w, label="Elderly", color="#E91E63", alpha=0.85)

    for i in range(len(labels)):
        ax.text(x[i] - w / 2, normal[i] + 0.5, f"{normal[i]:.1f}",
                ha="center", fontsize=7)
        ax.text(x[i] + w / 2, elderly[i] + 0.5, f"{elderly[i]:.1f}",
                ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Travel Time (s)")
    ax.set_title("Elderly vs Normal Agent Travel Times", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    out = _ensure_dir(out_dir) / "elderly_vs_normal.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# G. Step 0 — Data overview charts
# ============================================================================

def fig_data_overview(
    products: dict,
    nav_obs,
    nav_conn,
    out_dir: str | Path,
    cfg: dict,
) -> list[Path]:
    """Generate data overview figures for Step 0.

    Three panels:
      1. Retained element counts by storey
      2. Connector subtype distribution (full vs nav-filtered)
      3. Obstacle subcategory distribution (full vs nav-filtered)
    """
    import pandas as pd
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)
    outputs = []

    # --- 1. Element counts by storey ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=vp["dpi"])
    ret_df = products["retained_df"]
    if "storey_name" in ret_df.columns:
        counts = ret_df["storey_name"].value_counts().sort_index()
        colours = []
        for sn in counts.index:
            lk = sn.split()[0] if isinstance(sn, str) else "?"
            colours.append(LEVEL_COLOURS.get(lk, "#999"))
        bars = ax.barh(counts.index.astype(str), counts.values, color=colours, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8)
    ax.set_xlabel("Element Count")
    ax.set_title("Retained Elements by Storey", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    p = od / "data_elements_by_storey.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # --- 2. Connector subtype distribution ---
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 5), dpi=vp["dpi"])
    conn_df = products["connector_df"]

    # Full
    ax = axes2[0]
    if "connector_subtype" in conn_df.columns:
        cc = conn_df["connector_subtype"].value_counts()
        cc.plot.barh(ax=ax, color=[CONNECTOR_COLOURS.get(k, "#777") for k in cc.index])
        for i, (idx, val) in enumerate(cc.items()):
            ax.text(val + 0.5, i, str(val), va="center", fontsize=8)
    ax.set_title("All Connectors", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count")

    # Nav-filtered
    ax = axes2[1]
    if "connector_subtype" in nav_conn.columns:
        cc2 = nav_conn["connector_subtype"].value_counts()
        cc2.plot.barh(ax=ax, color=[CONNECTOR_COLOURS.get(k, "#777") for k in cc2.index])
        for i, (idx, val) in enumerate(cc2.items()):
            ax.text(val + 0.5, i, str(val), va="center", fontsize=8)
    ax.set_title("Nav-Filtered Connectors", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count")

    fig.suptitle("Connector Subtype Distribution", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = od / "data_connector_subtypes.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    # --- 3. Obstacle subcategory distribution ---
    fig, axes3 = plt.subplots(1, 2, figsize=(12, 5), dpi=vp["dpi"])
    obs_df = products["obstacle_df"]

    subcat_colors = {
        "obstacle_floor_intrusive": "#D32F2F",
        "obstacle_barrier_relevant": "#F57C00",
        "obstacle_clearance_relevant": "#FBC02D",
        "obstacle_skin_panel": "#90CAF9",
        "obstacle_uncertain": "#CE93D8",
        "obstacle_small_irrelevant": "#A5D6A7",
    }

    ax = axes3[0]
    if "obstacle_subcat" in obs_df.columns:
        oc = obs_df["obstacle_subcat"].value_counts()
        ax.barh(oc.index.astype(str), oc.values,
                color=[subcat_colors.get(k, "#999") for k in oc.index])
        for i, (idx, val) in enumerate(oc.items()):
            ax.text(val + 5, i, str(val), va="center", fontsize=7)
    ax.set_title("All Obstacles", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count")

    ax = axes3[1]
    if "obstacle_subcat" in nav_obs.columns:
        oc2 = nav_obs["obstacle_subcat"].value_counts()
        ax.barh(oc2.index.astype(str), oc2.values,
                color=[subcat_colors.get(k, "#999") for k in oc2.index])
        for i, (idx, val) in enumerate(oc2.items()):
            ax.text(val + 5, i, str(val), va="center", fontsize=7)
    ax.set_title("Nav-Filtered (KEEP)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count")

    fig.suptitle("Obstacle Subcategory Distribution", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = od / "data_obstacle_subcats.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p)

    print(f"[viz] Step 0: {len(outputs)} figures → {od}")
    return outputs


# ============================================================================
# H. Step 1 — Level area breakdown
# ============================================================================

def fig_level_area_breakdown(
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Bar chart comparing floor area vs walkable area per level."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    levels = sorted(geometries.keys())
    floor_areas = []
    walkable_areas = []
    for lvl in levels:
        g = geometries[lvl]
        fa = g["floor"].area if g["floor"] and not g["floor"].is_empty else 0
        wa = g["walkable"].area if g["walkable"] and not g["walkable"].is_empty else 0
        floor_areas.append(fa)
        walkable_areas.append(wa)

    x = np.arange(len(levels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=vp["dpi"])
    b1 = ax.bar(x - w / 2, floor_areas, w, label="Floor Area", color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x + w / 2, walkable_areas, w, label="Walkable Area", color="#66BB6A", edgecolor="white")

    for bar, val in zip(b1, floor_areas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", fontsize=8)
    for bar, val in zip(b2, walkable_areas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("Area (m²)")
    ax.set_title("Floor Area vs Walkable Area per Level", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    p = od / "level_area_breakdown.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 1: level_area_breakdown → {p}")
    return p


# ============================================================================
# H-bis. Step 1 — Connectors on floors  (NEW)
# ============================================================================

def fig_connectors_on_floors(
    geometries: dict,
    all_connectors: list[dict],
    out_dir: str | Path,
    cfg: dict,
    control_points: list[dict] | None = None,
) -> Path:
    """Per-level floor plan with typed connectors + control points overlaid.

    One subplot per walkable level (F1, F3, F4) + F2 connector-only.
    Connectors are colour-coded by type: stair_flight / escalator / elevator.
    Control points: fare_gate (orange) / security_scanner (cyan).
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    levels = sorted(geometries.keys())
    n = len(levels)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5),
                             dpi=vp["dpi"])
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten().tolist()

    conn_colours = {
        "stair_flight": "#795548",
        "escalator": "#E91E63",
        "elevator": "#9C27B0",
    }
    cp_colours = {
        "fare_gate": "#FF9800",
        "security_scanner": "#00BCD4",
    }

    for i, lvl in enumerate(levels):
        ax = axes[i]
        g = geometries[lvl]

        # Floor background
        floor = g.get("floor")
        if floor and not floor.is_empty:
            _plot_multipolygon(ax, floor, color="#E3F2FD", alpha=0.5,
                               edgecolor="#1565C0", linewidth=0.8)

        # Obstacles (regular only, exclude control-point footprints)
        n_cp_on_level = len(g.get("control_points", []))
        n_total_obs = len(g.get("obstacles", []))
        # control point obstacles are appended at the END of obstacles list
        n_regular_obs = n_total_obs - n_cp_on_level
        for obs in g.get("obstacles", [])[:n_regular_obs]:
            _plot_multipolygon(ax, obs, color="#EF5350", alpha=0.4,
                               edgecolor="#B71C1C", linewidth=0.3)

        # Forbidden zones (hatched red overlay)
        for fz in g.get("forbidden_zone_polys", []):
            _plot_multipolygon(ax, fz, color="#FF1744", alpha=0.15,
                               edgecolor="#D50000", linewidth=0.8, linestyle="--")

        # Track zones (yellow band)
        for tz in g.get("track_zones", []):
            poly = tz.get("polygon") if isinstance(tz, dict) else tz
            if poly and not poly.is_empty:
                _plot_multipolygon(ax, poly, color="#FFD600", alpha=0.12,
                                   edgecolor="#F57F17", linewidth=0.8, linestyle=":")

        # Entrances (green)
        for ent in g.get("entrances", []):
            poly = ent.get("polygon") if isinstance(ent, dict) else ent
            if poly and not poly.is_empty:
                _plot_multipolygon(ax, poly, color="#00E676", alpha=0.5,
                                   edgecolor="#1B5E20", linewidth=1.2)

        # Control points (skip IFC fare gate footprints when barrier groups define
        # solid walls — those are drawn separately below with direction colours)
        has_fg_groups = bool(g.get("fare_gate_groups"))
        for cp in g.get("control_points", []):
            fp = cp.get("footprint")
            if fp is None or fp.is_empty:
                continue
            ctype = cp.get("type", "unknown")
            if has_fg_groups and ctype == "fare_gate":
                continue  # drawn via fare_gate_groups below
            colour = cp_colours.get(ctype, "#607D8B")
            _plot_multipolygon(ax, fp, color=colour, alpha=0.7,
                               edgecolor="black", linewidth=0.4)
            cx, cy = fp.centroid.x, fp.centroid.y
            label_txt = "FG" if ctype == "fare_gate" else "SC"
            ax.text(cx, cy, label_txt, fontsize=3.5, ha="center", va="center",
                    color="white", fontweight="bold")

        # Fare gate barrier groups — inbound in green, outbound in red
        _fg_dir_colours = {"inbound": "#43A047", "outbound": "#E53935"}
        for fgg in g.get("fare_gate_groups", []):
            wall_poly = fgg.get("wall_polygon")
            if wall_poly is None or wall_poly.is_empty:
                continue
            fdir = fgg.get("direction", "inbound")
            fcol = _fg_dir_colours.get(fdir, "#FF9800")
            _plot_multipolygon(ax, wall_poly, color=fcol, alpha=0.85,
                               edgecolor="black", linewidth=0.6)
            cx, cy = wall_poly.centroid.x, wall_poly.centroid.y
            flbl = "IN" if fdir == "inbound" else "OUT"
            ax.text(cx, cy, flbl, fontsize=5, ha="center", va="center",
                    color="white", fontweight="bold")

        # Connectors touching this level
        for c in g.get("connectors", []):
            fp = c.get("footprint")
            if fp is None or fp.is_empty:
                continue
            ctype = c.get("type", "unknown")
            colour = conn_colours.get(ctype, "#607D8B")
            _plot_multipolygon(ax, fp, color=colour, alpha=0.6,
                               edgecolor="black", linewidth=0.5)
            # Label
            cx, cy = fp.centroid.x, fp.centroid.y
            label_txt = ctype[0].upper()  # S / E / e
            if ctype == "elevator":
                label_txt = "EL"
            elif ctype == "escalator":
                label_txt = "ES"
            elif ctype == "stair_flight":
                label_txt = "ST"
            ax.text(cx, cy, label_txt, fontsize=5, ha="center", va="center",
                    color="white", fontweight="bold")

        # Title
        n_esc = sum(1 for c in g.get("connectors", []) if c.get("type") == "escalator")
        n_elv = sum(1 for c in g.get("connectors", []) if c.get("type") == "elevator")
        n_stf = sum(1 for c in g.get("connectors", []) if c.get("type") == "stair_flight")
        n_fg_in  = sum(1 for fgg in g.get("fare_gate_groups", []) if fgg.get("direction") == "inbound")
        n_fg_out = sum(1 for fgg in g.get("fare_gate_groups", []) if fgg.get("direction") == "outbound")
        n_fg  = sum(1 for c in g.get("control_points", []) if c.get("type") == "fare_gate")
        n_sc  = sum(1 for c in g.get("control_points", []) if c.get("type") == "security_scanner")
        title_parts = [f"ST:{n_stf}", f"ES:{n_esc}", f"EL:{n_elv}"]
        if n_fg_in or n_fg_out:
            title_parts.append(f"IN:{n_fg_in} OUT:{n_fg_out}")
        elif n_fg:
            title_parts.append(f"FG:{n_fg}")
        if n_sc:
            title_parts.append(f"SC:{n_sc}")
        ax.set_title(
            f"{lvl}  ({' '.join(title_parts)})",
            fontsize=10, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#795548",
               markersize=10, label="Stair flight"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E91E63",
               markersize=10, label="Escalator"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9C27B0",
               markersize=10, label="Elevator"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#43A047",
               markersize=10, label="Inbound gate (unpaid→paid)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
               markersize=10, label="Outbound gate (paid→unpaid)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#00BCD4",
               markersize=10, label="Security scanner"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=8,
               frameon=False)

    fig.suptitle("Connectors & Control Points on Floor Plans",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    p = od / "connectors_on_floors.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 1: connectors_on_floors → {p}")
    return p


def fig_connector_voxels(
    geometries: dict,
    connector_nodes: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Per-level floor plan with connector anchor nodes plotted by type.

    Stairs = brown, Escalators = pink, Elevators = purple.
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    levels = sorted(k for k, g in geometries.items()
                    if g.get("walkable") is not None)
    n = len(levels)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5),
                             dpi=vp["dpi"])
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten().tolist()

    type_colours = {
        "stair_flight": ("#795548", "^"),
        "escalator": ("#E91E63", "D"),
        "elevator": ("#9C27B0", "s"),
    }

    for i, lvl in enumerate(levels):
        ax = axes[i]
        g = geometries[lvl]

        # Floor background
        if g.get("floor") and not g["floor"].is_empty:
            _plot_multipolygon(ax, g["floor"], color="#E3F2FD", alpha=0.4,
                               edgecolor="#1565C0", linewidth=0.6)
        if g.get("walkable") and not g["walkable"].is_empty:
            _plot_multipolygon(ax, g["walkable"], color="#C8E6C9", alpha=0.2,
                               edgecolor="#2E7D32", linewidth=0.4, linestyle="--")

        # Connector footprints (faint)
        for c in g.get("connectors", []):
            fp = c.get("footprint")
            if fp is None or fp.is_empty:
                continue
            ctype = c.get("type", "unknown")
            colour = type_colours.get(ctype, ("#607D8B", "o"))[0]
            _plot_multipolygon(ax, fp, color=colour, alpha=0.25,
                               edgecolor=colour, linewidth=0.8)

        # Connector anchor nodes
        nodes = connector_nodes.get(lvl, [])
        for ntype, (colour, marker) in type_colours.items():
            xs = [n["x"] for n in nodes if n["node_type"] == ntype]
            ys = [n["y"] for n in nodes if n["node_type"] == ntype]
            if xs:
                ax.scatter(xs, ys, c=colour, s=30, marker=marker,
                           edgecolors="black", linewidths=0.5, zorder=10,
                           alpha=0.9)

        n_nodes = len(nodes)
        ax.set_title(f"{lvl}  ({n_nodes} connector nodes)",
                     fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#795548",
               markersize=8, label="Stair anchor"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E91E63",
               markersize=8, label="Escalator anchor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9C27B0",
               markersize=8, label="Elevator anchor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle("Connector Voxelisation — Anchor Nodes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    p = od / "connector_voxels.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] connector_voxels → {p}")
    return p


# ============================================================================
# I. Step 2 — Node density histogram
# ============================================================================

def fig_node_density_per_level(
    level_nodes: dict,
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Node count bar chart + density (nodes/m²) per level."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    levels = sorted(level_nodes.keys())
    n_nodes = []
    densities = []
    for lvl in levels:
        nn = len(level_nodes[lvl])
        n_nodes.append(nn)
        g = geometries.get(lvl, {})
        wa = g["walkable"].area if g.get("walkable") and not g["walkable"].is_empty else 1.0
        densities.append(nn / wa)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=vp["dpi"])
    colours = [LEVEL_COLOURS.get(l, "#999") for l in levels]

    # Bar: node counts
    bars = ax1.bar(levels, n_nodes, color=colours, edgecolor="white")
    for bar, val in zip(bars, n_nodes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Number of Nodes")
    ax1.set_title("Sampled Nodes per Level", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Bar: density
    bars2 = ax2.bar(levels, densities, color=colours, edgecolor="white")
    for bar, val in zip(bars2, densities):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontsize=9)
    ax2.set_ylabel("Nodes / m²")
    ax2.set_title("Node Density (nodes per m²)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Sampling Statistics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = od / "node_density_per_level.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 2: node_density_per_level → {p}")
    return p


# ============================================================================
# J. Step 3 — Graph degree distribution
# ============================================================================

def fig_graph_degree_distribution(
    G: nx.Graph,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Degree distribution histogram + edge type breakdown."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=vp["dpi"])

    # Degree histogram
    degrees = [d for _, d in G.degree()]
    ax1.hist(degrees, bins=range(0, max(degrees) + 2), color="#1976D2",
             edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Degree Distribution (mean={np.mean(degrees):.1f})",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Edge type counts
    etype_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        etype_counts[d.get("edge_type", "floor")] += 1
    etypes = sorted(etype_counts.keys())
    evalues = [etype_counts[e] for e in etypes]
    ecolors = [EDGE_TYPE_STYLES.get(e, {}).get("color", "#999") for e in etypes]
    bars = ax2.barh(etypes, evalues, color=ecolors, edgecolor="white")
    for bar, val in zip(bars, evalues):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9)
    ax2.set_xlabel("Edge Count")
    ax2.set_title("Edge Type Breakdown", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Graph Topology — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = od / "graph_degree_distribution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 3: graph_degree_distribution → {p}")
    return p


# ============================================================================
# K. Step 4 — Semantic regions & paths
# ============================================================================

def fig_semantic_regions_map(
    G: nx.Graph,
    regions: dict[str, list[str]],
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Plot semantic OD regions with two-direction flow model.

    Shows:
      ▲ ENTRANCE/EXIT (cyan)  — 5 station gate nodes on F3/F4
      ● PLATFORM      (blue)  — PSD door_platform nodes on F1
    Flow arrows:
      Inbound  ENTRANCE → PLATFORM
      Outbound PLATFORM → EXIT
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)
    high_dpi = max(int(vp["dpi"]), 360)

    # Two directions mapped to their colours
    REGION_COLOURS = {
        "ENTRANCE": "#00ACC1",   # cyan  — station gate (entering)
        "EXIT":     "#00ACC1",   # same  — same gate used as exit
        "PLATFORM": "#1565C0",   # dark blue — train door
    }
    REGION_LABELS = {
        "ENTRANCE": "Entrance\n(F3/F4 station gates)",
        "EXIT":     "Exit\n(F3/F4 station gates)",
        "PLATFORM": "PSD Door\n(F1 train doors)",
    }
    level_fill = {"F1": "#CFE3F8", "F3": "#F8E0BB", "F4": "#D7EED7"}
    level_edge = {"F1": "#5C88B0", "F3": "#C68C38", "F4": "#6CA06C"}
    level_nodes = {"F1": "#9EB6C7", "F3": "#C6B293", "F4": "#A6C5A6"}

    elevations = {
        lvl: meta.get("elevation_m", 0.0)
        for lvl, meta in cfg.get("station", {}).get("levels", {}).items()
        if meta.get("is_walkable", False)
    }
    levels = [lvl for lvl in ["F1", "F3", "F4"] if lvl in geometries]
    legacy_path = od / "semantic_regions_map.png"
    if legacy_path.exists():
        try:
            legacy_path.unlink()
        except OSError:
            pass

    def _node_records(node_ids: list[str], level_filter: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for nid in node_ids:
            if nid not in G:
                continue
            d = G.nodes[nid]
            if level_filter and d.get("level") != level_filter:
                continue
            if "x" not in d or "y" not in d:
                continue
            out.append({
                "id": nid,
                "x": float(d["x"]),
                "y": float(d["y"]),
                "z": float(elevations.get(d.get("level"), 0.0)),
                "level": d.get("level"),
                "data": d,
            })
        return out

    def _centroid(records: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
        if not records:
            return None, None, None
        xs = [r["x"] for r in records]
        ys = [r["y"] for r in records]
        zs = [r["z"] for r in records]
        return sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)

    def _draw_floor_background_2d(ax, focus_level: str) -> None:
        floor = geometries.get(focus_level, {}).get("floor")
        if floor and not floor.is_empty:
            _plot_multipolygon(
                ax,
                floor,
                color=level_fill.get(focus_level, "#F5F5F5"),
                alpha=0.54,
                edgecolor=level_edge.get(focus_level, "#9E9E9E"),
                linewidth=1.05,
            )
            try:
                cx, cy = floor.centroid.x, floor.centroid.y
                ax.text(cx, cy, focus_level, fontsize=11,
                        color=level_edge.get(focus_level, "#7A7A7A"),
                        ha="center", va="center", alpha=0.75, fontweight="bold")
            except Exception:
                pass

        bg_xs, bg_ys = [], []
        for _, d in G.nodes(data=True):
            if d.get("level") != focus_level:
                continue
            if d.get("node_type") not in ("floor", "door_platform", "entrance"):
                continue
            if "x" not in d or "y" not in d:
                continue
            bg_xs.append(float(d["x"]))
            bg_ys.append(float(d["y"]))
        if bg_xs:
            ax.scatter(
                bg_xs,
                bg_ys,
                c=level_nodes.get(focus_level, "#BDBDBD"),
                s=4.2,
                alpha=0.42,
                zorder=1,
            )

    def _annotate_primary_entrances(ax, records: list[dict[str, Any]], colour: str) -> None:
        for rec in records:
            d = rec["data"]
            if not d.get("entrance_primary", False):
                continue
            ename = d.get("entrance_name", rec["id"]).replace("entrance_", "Gate ")
            ax.annotate(
                ename,
                (rec["x"], rec["y"]),
                fontsize=8,
                ha="center",
                va="bottom",
                color=colour,
                xytext=(0, 8),
                textcoords="offset points",
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 0.3},
            )

    out_paths: list[Path] = []

    gate_records = _node_records(sorted(set(regions.get("ENTRANCE", [])) | set(regions.get("EXIT", []))))
    platform_records = _node_records(regions.get("PLATFORM", []))

    for focus_level in levels:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=high_dpi)
        _draw_floor_background_2d(ax, focus_level)

        legend_title = "OD Regions"
        if focus_level == "F1":
            level_records = [r for r in platform_records if r["level"] == focus_level]
            if level_records:
                ax.scatter(
                    [r["x"] for r in level_records],
                    [r["y"] for r in level_records],
                    c=REGION_COLOURS["PLATFORM"],
                    s=150,
                    marker="o",
                    edgecolor="#0D47A1",
                    linewidth=1.15,
                    zorder=10,
                    label=f"{REGION_LABELS['PLATFORM']} (n={len(level_records)})",
                )
            subtitle = "Plan View — Platform spawn / destination region"
        else:
            level_records = [r for r in gate_records if r["level"] == focus_level]
            if level_records:
                ax.scatter(
                    [r["x"] for r in level_records],
                    [r["y"] for r in level_records],
                    c=REGION_COLOURS["ENTRANCE"],
                    s=145,
                    marker="^",
                    edgecolor="#006064",
                    linewidth=1.15,
                    zorder=10,
                    label=f"Entrance / Exit\n(Bidirectional gates) (n={len(level_records)})",
                )
                _annotate_primary_entrances(ax, level_records, REGION_COLOURS["ENTRANCE"])
            subtitle = "Plan View — Bidirectional gate spawn / destination region"

        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper right", title=legend_title, title_fontsize=10)
        ax.set_title(
            f"Pedestrian Spawn Regions — {focus_level}\n{subtitle}",
            fontsize=14,
            fontweight="bold",
        )
        ax.tick_params(labelsize=9)
        fig.tight_layout()

        p_level = od / f"semantic_regions_map_{focus_level}.png"
        fig.savefig(p_level, bbox_inches="tight", dpi=high_dpi)
        plt.close(fig)
        out_paths.append(p_level)

    # ---- 3D overview -----------------------------------------------------
    z_scale = 2.4
    fig3d = plt.figure(figsize=(15, 10), dpi=high_dpi)
    ax3d = fig3d.add_subplot(111, projection="3d")

    x_all, y_all = [], []
    for lvl in levels:
        z = elevations.get(lvl, 0.0) * z_scale
        floor = geometries.get(lvl, {}).get("floor")
        if floor and not floor.is_empty:
            for poly in flatten_polygons(floor):
                xs, ys = poly.exterior.xy
                zs = [z] * len(xs)
                x_all.extend(xs)
                y_all.extend(ys)
                ax3d.plot(xs, ys, zs, color=LEVEL_COLOURS.get(lvl, "#999999"), linewidth=0.9, alpha=0.65)
                ax3d.add_collection3d(
                    Poly3DCollection(
                        [list(zip(xs, ys, zs))],
                        alpha=0.12,
                        facecolor=level_fill.get(lvl, "#F5F5F5"),
                        edgecolor="#CFCFCF",
                        linewidths=0.4,
                    )
                )
                try:
                    cx, cy = poly.centroid.x, poly.centroid.y
                    ax3d.text(cx, cy, z + 0.8, lvl, fontsize=10, color="#7A7A7A")
                except Exception:
                    pass

        bg_xs, bg_ys, bg_zs = [], [], []
        for _, d in G.nodes(data=True):
            if d.get("level") != lvl:
                continue
            if d.get("node_type") not in ("floor", "door_platform", "entrance"):
                continue
            if "x" not in d or "y" not in d:
                continue
            bg_xs.append(float(d["x"]))
            bg_ys.append(float(d["y"]))
            bg_zs.append(z)
        if bg_xs:
            ax3d.scatter(bg_xs, bg_ys, bg_zs, c="#D9D9D9", s=0.6, alpha=0.18, depthshade=False)

    if platform_records:
        ax3d.scatter(
            [r["x"] for r in platform_records],
            [r["y"] for r in platform_records],
            [r["z"] * z_scale for r in platform_records],
            c=REGION_COLOURS["PLATFORM"],
            s=42,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            depthshade=False,
        )

    if gate_records:
        ax3d.scatter(
            [r["x"] for r in gate_records],
            [r["y"] for r in gate_records],
            [r["z"] * z_scale for r in gate_records],
            c=REGION_COLOURS["ENTRANCE"],
            s=54,
            marker="^",
            edgecolors="white",
            linewidths=0.5,
            depthshade=False,
        )

    ent_cx, ent_cy, ent_cz = _centroid(_node_records(regions.get("ENTRANCE", [])))
    plt_cx, plt_cy, plt_cz = _centroid(platform_records)
    ex_cx, ex_cy, ex_cz = _centroid(_node_records(regions.get("EXIT", [])))

    if ent_cx is not None and plt_cx is not None:
        ax3d.quiver(
            ent_cx, ent_cy, ent_cz * z_scale,
            plt_cx - ent_cx, plt_cy - ent_cy, (plt_cz - ent_cz) * z_scale,
            color="#FF6F00", linewidth=2.0, arrow_length_ratio=0.08,
        )

    if plt_cx is not None and ex_cx is not None:
        ax3d.quiver(
            plt_cx, plt_cy, plt_cz * z_scale,
            ex_cx - plt_cx, ex_cy - plt_cy, (ex_cz - plt_cz) * z_scale,
            color="#FB8C00", linewidth=2.0, arrow_length_ratio=0.08,
        )

    legend_handles = [
        mlines.Line2D([], [], color=REGION_COLOURS["ENTRANCE"], marker="^", linestyle="None",
                      markersize=10, markeredgecolor="white", label=f"Entrance / Exit gates (n={len(gate_records)})"),
        mlines.Line2D([], [], color=REGION_COLOURS["PLATFORM"], marker="o", linestyle="None",
                      markersize=10, markeredgecolor="white", label=f"PSD doors (n={len(platform_records)})"),
    ]
    ax3d.legend(handles=legend_handles, loc="upper right", fontsize=9, title="OD Regions", title_fontsize=10)
    ax3d.set_title(
        "Pedestrian Spawn Regions — 3D Overview\nInbound: ENT → PLT  |  Outbound: PLT → EXIT",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax3d.text2D(
        0.02,
        0.86,
        "Orange arrows show bidirectional OD flow between gate regions\nand platform doors across the stacked floors.",
        transform=ax3d.transAxes,
        fontsize=10,
        color="#E65100",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    ax3d.set_xlabel("X (m)", fontsize=10)
    ax3d.set_ylabel("Y (m)", fontsize=10)
    ax3d.set_zlabel(f"Elevation (×{z_scale:.1f})", fontsize=10)
    ax3d.view_init(elev=24, azim=-58)
    if x_all and y_all:
        x_span = max(x_all) - min(x_all)
        y_span = max(y_all) - min(y_all)
        z_values = [elevations.get(lvl, 0.0) * z_scale for lvl in levels]
        z_span = (max(z_values) - min(z_values)) if z_values else 1.0
        try:
            ax3d.set_box_aspect((max(x_span, 1.0), max(y_span, 1.0), max(z_span * 2.2, 1.0)))
        except Exception:
            pass

    p_3d = od / "semantic_regions_map_3d.png"
    fig3d.savefig(p_3d, bbox_inches="tight", dpi=high_dpi)
    plt.close(fig3d)
    out_paths.append(p_3d)

    print("[viz] Step 4: semantic_regions_map split outputs:")
    for path in out_paths:
        print(f"  - {path}")
    return p_3d


# Distinct colours for five entrances
_ENTRANCE_PALETTE = [
    "#1565C0",  # Entrance A — deep blue
    "#6A1B9A",  # Entrance B — deep purple
    "#2E7D32",  # Entrance C — deep green
    "#E65100",  # Entrance D — deep orange
    "#AD1457",  # Entrance E — deep pink
]


def fig_entrance_route_map(
    G: nx.Graph,
    entrance_paths: list[dict],
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Two-panel route map: left = inbound (Entrance→PSD), right = outbound (PSD→Entrance).

    Each panel shows the 5 entrance routes with per-entrance colours.
    Star markers highlight where the fare gate node appears in each path.
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    level_fill = {"F1": "#E3F2FD", "F3": "#FFF3E0", "F4": "#E8F5E9"}

    fig, axes = plt.subplots(1, 2,
                             figsize=(vp["figsize_single"][0] * 2,
                                      vp["figsize_single"][1]),
                             dpi=vp["dpi"])

    panels = [
        (axes[0], "inbound_path",  "inbound_cost",  "inbound_gate",
         "Inbound  (Entrance → PSD)"),
        (axes[1], "outbound_path", "outbound_cost", "outbound_gate",
         "Outbound (PSD → Entrance)"),
    ]

    for ax, path_key, cost_key, gate_key, title in panels:
        # Background floors
        for lvl, gd in geometries.items():
            floor = gd.get("floor")
            if floor and not floor.is_empty:
                _plot_multipolygon(ax, floor,
                                   color=level_fill.get(lvl, "#F5F5F5"), alpha=0.30,
                                   edgecolor="#BDBDBD", linewidth=0.5)
                try:
                    cx, cy = floor.centroid.x, floor.centroid.y
                    ax.text(cx, cy, lvl, fontsize=9, color="#9E9E9E",
                            ha="center", va="center", alpha=0.6,
                            fontweight="bold")
                except Exception:
                    pass

        def _path_xy(path):
            xs, ys = [], []
            for nid in path:
                if nid not in G:
                    continue
                d = G.nodes[nid]
                xs.append(d.get("x", 0))
                ys.append(d.get("y", 0))
            return xs, ys

        for i, ep in enumerate(entrance_paths):
            colour = _ENTRANCE_PALETTE[i % len(_ENTRANCE_PALETTE)]
            ename  = ep["entrance_name"].replace("entrance_", "Gate ")
            cost   = ep.get(cost_key, 0.0)
            path   = ep.get(path_key, [])

            xs, ys = _path_xy(path)
            if len(xs) >= 2:
                ax.plot(xs, ys, "-", color=colour, linewidth=2.0, alpha=0.88,
                        zorder=6, label=f"{ename} ({cost:.0f}s)")
                ax.annotate("", xy=(xs[-1], ys[-1]),
                            xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle="-|>", color=colour,
                                            lw=1.8, mutation_scale=14), zorder=8)

            # Fare gate node marker (★)
            gnid = ep.get(gate_key)
            if gnid and gnid in G:
                ga = G.nodes[gnid]
                ax.scatter([ga["x"]], [ga["y"]], c=colour, s=220,
                           marker="*", edgecolors="white", linewidth=0.8,
                           zorder=14)

            # Entrance marker (▲)
            eid = ep["entrance_id"]
            if eid in G:
                ea = G.nodes[eid]
                ax.scatter([ea["x"]], [ea["y"]], c=colour, s=130, marker="^",
                           edgecolors="white", linewidth=1.0, zorder=12)
                ax.annotate(ename, (ea["x"], ea["y"]),
                            fontsize=7, color=colour, fontweight="bold",
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom")

            # PSD marker (■)
            pid = ep["psd_id"]
            if pid in G:
                pa = G.nodes[pid]
                ax.scatter([pa["x"]], [pa["y"]], c=colour, s=70, marker="s",
                           edgecolors="white", linewidth=0.8, zorder=11)

        import matplotlib.lines as mlines
        sym_legend = [
            mlines.Line2D([], [], color="#555", marker="*", linestyle="None",
                          markersize=9, label="Fare Gate"),
            mlines.Line2D([], [], color="#555", marker="^", linestyle="None",
                          markersize=8, label="Entrance"),
            mlines.Line2D([], [], color="#555", marker="s", linestyle="None",
                          markersize=7, label="PSD Door"),
        ]
        ax.legend(handles=sym_legend, fontsize=7, loc="upper right",
                  title="Symbols", title_fontsize=7.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(labelsize=7)

    # Bottom entrance colour legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=_ENTRANCE_PALETTE[i % len(_ENTRANCE_PALETTE)],
                       label=ep["entrance_name"].replace("entrance_", "Gate "))
        for i, ep in enumerate(entrance_paths)
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(entrance_paths), fontsize=8,
               title="Entrances", title_fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    p = od / "entrance_route_map.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 4: entrance_route_map → {p}")
    return p


def fig_agent_overview(
    agents: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Agent overview: spawn time histogram + flow breakdown pie chart."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=vp["dpi"])

    # 1. Spawn time histogram
    spawns = [a["spawn_time"] for a in agents]
    ax1.hist(spawns, bins=20, color="#42A5F5", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Spawn Time (s)")
    ax1.set_ylabel("Count")
    ax1.set_title("Agent Spawn Times", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Flow distribution
    flow_counts = defaultdict(int)
    for a in agents:
        flow_counts[a["flow"]] += 1
    labels = list(flow_counts.keys())
    sizes = list(flow_counts.values())
    ax2.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 8})
    ax2.set_title("Flow Distribution", fontsize=11, fontweight="bold")

    # 3. Agent type breakdown
    type_counts = defaultdict(int)
    for a in agents:
        type_counts[a.get("agent_type", "normal")] += 1
    tc_labels = list(type_counts.keys())
    tc_values = list(type_counts.values())
    type_colours = {"normal": "#2196F3", "elderly": "#E91E63"}
    ax3.bar(tc_labels, tc_values,
            color=[type_colours.get(t, "#999") for t in tc_labels],
            edgecolor="white")
    for i, (lbl, val) in enumerate(zip(tc_labels, tc_values)):
        ax3.text(i, val + 1, str(val), ha="center", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Count")
    ax3.set_title("Agent Type Breakdown", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Agent Setup — {len(agents)} agents", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = od / "agent_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 4: agent_overview → {p}")
    return p


def fig_example_paths(
    G: nx.Graph,
    agents: list[dict],
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
    n_paths: int = 5,
) -> Path:
    """Plot example shortest paths on the station plan."""
    from src.routing import find_path
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=vp["dpi"])

    # Background floors
    for lvl, g in geometries.items():
        floor = g.get("floor")
        if floor and not floor.is_empty:
            _plot_multipolygon(ax, floor, color="#F5F5F5", alpha=0.3,
                               edgecolor="#BDBDBD", linewidth=0.3)

    # All nodes as tiny grey
    for nid, d in G.nodes(data=True):
        ax.scatter(d.get("x", 0), d.get("y", 0), c="#E0E0E0", s=0.15, zorder=1)

    path_colours = ["#E91E63", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
                    "#009688", "#F44336", "#3F51B5"]
    n_show = min(n_paths, len(agents))
    for i, agent in enumerate(agents[:n_show]):
        path = find_path(G, agent["origin"], agent["dest"])
        if len(path) < 2:
            continue
        xs = [G.nodes[n]["x"] for n in path]
        ys = [G.nodes[n]["y"] for n in path]
        colour = path_colours[i % len(path_colours)]
        ax.plot(xs, ys, color=colour, linewidth=1.8, alpha=0.8, zorder=5,
                label=f"{agent['agent_id']}: {agent['flow']} ({len(path)} hops)")
        # Origin & dest markers
        ax.scatter(xs[0], ys[0], c=colour, s=60, marker="o", edgecolor="black",
                   linewidth=0.8, zorder=10)
        ax.scatter(xs[-1], ys[-1], c=colour, s=60, marker="*", edgecolor="black",
                   linewidth=0.8, zorder=10)

    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Example Shortest Paths ({n_show} agents)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=7)

    p = od / "example_paths.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 4: example_paths → {p}")
    return p


# ============================================================================
# L. Step 5 — Travel-time distribution
# ============================================================================

def fig_travel_time_distribution(
    scenario_results: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Histogram of travel times per scenario, overlaid."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=vp["figsize_single"], dpi=vp["dpi"])

    for result in scenario_results:
        tt = result.get("travel_times", [])
        if not tt:
            continue
        label = result.get("label", "?")
        ax.hist(tt, bins=20, alpha=0.55, label=f"{label} (n={len(tt)})", edgecolor="white")

    ax.set_xlabel("Travel Time (s)")
    ax.set_ylabel("Agent Count")
    ax.set_title("Travel Time Distribution by Scenario", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    p = od / "travel_time_distribution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 5: travel_time_distribution → {p}")
    return p


def fig_stair_queue_timeline(
    scenario_results: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Stair queue length over simulation time per scenario."""
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=vp["figsize_single"], dpi=vp["dpi"])

    for result in scenario_results:
        sq = result.get("stair_queue_over_time", [])
        if not sq:
            continue
        ts = [t for t, _ in sq]
        qs = [q for _, q in sq]
        label = result.get("label", "?")
        ax.plot(ts, qs, linewidth=1.0, alpha=0.8, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Length")
    ax.set_title("Stair Queue Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    p = od / "stair_queue_timeline.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 5: stair_queue_timeline → {p}")
    return p


# ============================================================================
# Master visualisation runner
# ============================================================================

def run_all_viz(
    geometries: dict,
    level_nodes: dict,
    elevations: dict,
    connector_groups: dict,
    G: nx.Graph,
    scenario_results: list[dict],
    metrics_list: list[dict],
    out_dir: str | Path,
    cfg: dict,
) -> list[Path]:
    """Generate all thesis figures. Returns list of output paths."""
    setup_matplotlib_font()
    od = _ensure_dir(out_dir)
    outputs = []

    print("[viz] Generating level geometry ...")
    outputs.append(fig_all_levels_geometry(geometries, od, cfg))

    print("[viz] Generating level nodes ...")
    outputs.append(fig_all_levels_nodes(geometries, level_nodes, od, cfg))

    print("[viz] Generating isometric view ...")
    outputs.append(fig_multilevel_isometric(geometries, level_nodes, elevations, od, cfg))

    print("[viz] Generating connector overview ...")
    outputs.append(fig_connectors_overview(connector_groups, geometries, od, cfg))

    print("[viz] Generating graph per level ...")
    outputs.append(fig_graph_per_level(G, geometries, od, cfg))

    print("[viz] Generating cross-level edges ...")
    outputs.append(fig_graph_cross_level_edges(G, od, cfg))

    if scenario_results:
        print("[viz] Generating comparison charts ...")
        outputs.append(fig_comparison_bar(metrics_list, od, cfg))
        outputs.append(fig_arrival_curve(scenario_results, od, cfg))
        outputs.append(fig_queue_over_time(scenario_results, od, cfg))
        outputs.append(fig_elderly_vs_normal(metrics_list, od, cfg))

    print(f"[viz] Done — {len(outputs)} figures saved to {od}")
    return outputs


# ============================================================================
# M. Step 5 — Replanning analysis figures
# ============================================================================

def fig_replan_timeline(
    dynamic_result: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Replan event timeline for the dynamic routing scenario.

    Top panel  — bar chart: replan events per 30-second bin.
    Bottom panel — line chart: cumulative unique agents that have replanned.
    Annotated with total events and fraction of replanning agents.
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    events = dynamic_result.get("replan_events", [])
    T_s = cfg.get("simulation", {}).get("T_s", 600)
    n_agents = cfg.get("simulation", {}).get("n_agents", 200)
    bin_w = 30  # seconds per bin

    if not events:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=vp["dpi"])
        ax.text(0.5, 0.5, "No replan events (static scenario)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        p = od / "replan_timeline.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        return p

    # ------ data prep ------
    times = [float(e["t"]) for e in events]
    agents = [e["agent_id"] for e in events]

    bins = np.arange(0, T_s + bin_w, bin_w)
    counts, _ = np.histogram(times, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # cumulative unique agents
    seen: set[str] = set()
    cum_agents: list[int] = []
    cum_times:  list[float] = []
    for e in sorted(events, key=lambda x: float(x["t"])):
        seen.add(e["agent_id"])
        cum_times.append(float(e["t"]))
        cum_agents.append(len(seen))

    # ------ figure ------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=vp["dpi"],
                                   gridspec_kw={"height_ratios": [1.6, 1]})

    # Top: bar chart of event counts
    bars = ax1.bar(bin_centers, counts, width=bin_w * 0.85,
                   color="#EF5350", edgecolor="white", alpha=0.88)
    peak_idx = int(np.argmax(counts))
    ax1.annotate(f"peak {counts[peak_idx]}",
                 xy=(bin_centers[peak_idx], counts[peak_idx]),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=8, color="#B71C1C", fontweight="bold")
    ax1.set_ylabel("Replan events / 30 s window")
    ax1.set_xlim(0, T_s)
    ax1.set_title(
        f"Replanning Activity — Dynamic Routing\n"
        f"Total events: {len(events):,}  ·  "
        f"Agents that replanned: {len(set(agents))}/{n_agents} "
        f"({len(set(agents))/n_agents:.0%})",
        fontsize=11, fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: cumulative unique agents
    ax2.plot(cum_times, cum_agents, color="#1565C0", linewidth=1.8, alpha=0.9)
    ax2.axhline(n_agents, color="#9E9E9E", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_ylabel("Unique agents replanned")
    ax2.set_xlim(0, T_s)
    ax2.set_ylim(0, n_agents * 1.05)
    ax2.text(T_s * 0.98, n_agents + 2, f"total agents: {n_agents}",
             ha="right", fontsize=7.5, color="#9E9E9E")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    p = od / "replan_timeline.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 5: replan_timeline → {p}")
    return p


def fig_route_flow_diff(
    G: nx.DiGraph,
    static_result: dict,
    dynamic_result: dict,
    geometries: dict,
    out_dir: str | Path,
    cfg: dict,
) -> Path:
    """Per-level edge-flow difference heatmap: dynamic minus static throughput.

    Three subplots (F1 / F3 / F4).  Each edge with a non-zero difference is
    drawn as a coloured line using a diverging Red-Blue palette:
      🔴 Red  — more pedestrians used this edge in the *dynamic* scenario
      🔵 Blue — more pedestrians used this edge in the *static* scenario
    Grey edges have the same throughput in both scenarios.

    The figure reveals which corridors are load-shifted by congestion-aware
    replanning, making the route redistribution immediately visible.
    """
    setup_matplotlib_font()
    vp = _get_viz_params(cfg)
    od = _ensure_dir(out_dir)

    et_s = static_result.get("edge_throughput", {})
    et_d = dynamic_result.get("edge_throughput", {})
    all_edge_keys = set(et_s) | set(et_d)

    diff: dict[str, int] = {
        k: et_d.get(k, 0) - et_s.get(k, 0) for k in all_edge_keys
    }

    LEVELS = ("F1", "F3", "F4")
    LEVEL_FILL  = {"F1": "#E3F2FD", "F3": "#E8F5E9", "F4": "#FFF3E0"}
    LEVEL_EDGE  = {"F1": "#90CAF9", "F3": "#A5D6A7", "F4": "#FFCC80"}

    all_diffs = list(diff.values())
    vmax = max(1, max(abs(v) for v in all_diffs))

    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "flow_diff",
        ["#1565C0", "#64B5F6", "#E0E0E0", "#EF9A9A", "#B71C1C"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    def _node_xy(nid):
        if nid not in G:
            return None, None
        d = G.nodes[nid]
        return d.get("x", None), d.get("y", None)

    def _node_level(nid):
        if nid not in G:
            return None
        return G.nodes[nid].get("level", None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=vp["dpi"])

    for ax, lvl in zip(axes, LEVELS):
        # Floor background
        geo = geometries.get(lvl, {})
        floor = geo.get("floor")
        if floor and not floor.is_empty:
            _plot_multipolygon(ax, floor,
                               color=LEVEL_FILL.get(lvl, "#F5F5F5"), alpha=0.4,
                               edgecolor=LEVEL_EDGE.get(lvl, "#BDBDBD"), linewidth=0.6)

        segments, colors, widths = [], [], []

        for edge_key, dv in diff.items():
            u, v = edge_key.split("|", 1)
            # Include edge if either endpoint belongs to this level
            u_lvl = _node_level(u)
            v_lvl = _node_level(v)
            if u_lvl != lvl and v_lvl != lvl:
                continue
            x0, y0 = _node_xy(u)
            x1, y1 = _node_xy(v)
            if x0 is None or x1 is None:
                continue
            segments.append([(x0, y0), (x1, y1)])
            rgba = cmap(norm(dv))
            colors.append(rgba)
            # Width proportional to |diff|, min 0.4
            widths.append(max(0.4, min(4.0, 0.4 + abs(dv) * 0.08)))

        if segments:
            lc = mcoll.LineCollection(
                segments, colors=colors, linewidths=widths,
                alpha=0.80, zorder=5,
            )
            ax.add_collection(lc)
            ax.autoscale_view()

        ax.set_aspect("equal")
        ax.set_title(f"{lvl}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Δ throughput  (dynamic − static)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Annotation: top increased / decreased edges
    top_inc = sorted(diff.items(), key=lambda x: -x[1])[:3]
    top_dec = sorted(diff.items(), key=lambda x: x[1])[:3]
    note = (
        "Top +Δ (more dynamic traffic):  "
        + "  |  ".join(f"{k.split('|')[0].split('_')[0]}…+{v}" for k, v in top_inc)
        + "\nTop −Δ (more static traffic):    "
        + "  |  ".join(f"{k.split('|')[0].split('_')[0]}…{v}" for k, v in top_dec)
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#555",
             style="italic", wrap=True)

    fig.suptitle(
        "Route Flow Difference: Dynamic − Static Routing\n"
        "🔴 Red = more traffic in dynamic  ·  🔵 Blue = more traffic in static",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    p = od / "route_flow_diff.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Step 5: route_flow_diff → {p}")
    return p
