"""
Interactive Visualization Module (Plotly)
==========================================

Browser-based 3D interactive visualizations for the multi-level
station navigation graph.  Each ``fig_*`` function writes a
self-contained ``.html`` file that can be opened in any browser.

Design goals
------------
* Toggle individual floors, connector types, node layers on/off
* Mouse rotate / zoom / pan for large-footprint buildings
* Hover-over metadata (connector ID, edge type, elevation …)
* Exported as a single portable HTML per figure
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import networkx as nx

# Plotly is an optional heavy dependency — fail loudly at import time
try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "plotly is required for interactive visualizations.  "
        "Install with:  pip install plotly") from exc

from src.utils import flatten_polygons

# ---- colour palettes (keep in-sync with viz.py) -------------------------

LEVEL_RGBA = {
    "F1": "rgba(33,150,243,{a})",
    "F3": "rgba(76,175,80,{a})",
    "F4": "rgba(255,152,0,{a})",
}
LEVEL_HEX = {"F1": "#2196F3", "F3": "#4CAF50", "F4": "#FF9800"}
CONN_HEX = {
    "stair": "#795548", "stair_chain": "#795548",
    "escalator": "#E91E63",
    "elevator": "#9C27B0",
    "elevator_door": "#AB47BC",
    "elevator_interior": "#CE93D8",
    "psd_door": "#FF9800",
    "anchor_snap": "#4CAF50",
}


def _ensure_dir(p: str | Path) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =========================================================================
# 1. Interactive 3-D multi-level station map
# =========================================================================

def _polygon_boundary_trace(
    poly, z: float, color: str, name: str,
    legendgroup: str, showlegend: bool,
    width: float = 2.0,
) -> go.Scatter3d:
    """Create a Scatter3d trace for a polygon outline at elevation *z*."""
    xs, ys = poly.exterior.xy
    xs, ys = list(xs), list(ys)
    return go.Scatter3d(
        x=xs, y=ys, z=[z] * len(xs),
        mode="lines",
        line=dict(color=color, width=width),
        name=name, legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="text",
        hovertext=f"{name}  z={z:.1f}m",
    )


def _mesh_from_quad(corners, color, opacity, name, legendgroup,
                    showlegend=False, htext=""):
    """Create a Mesh3d from 4+ coplanar points (convex polygon).

    *corners* is a list of (x, y, z) tuples.
    """
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    n = len(corners)
    # fan triangulation from vertex 0
    ii, jj, kk = [], [], []
    for t in range(1, n - 1):
        ii.append(0); jj.append(t); kk.append(t + 1)
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=ii, j=jj, k=kk,
        color=color, opacity=opacity,
        name=name, legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="text", hovertext=htext,
        flatshading=True,
    )


def _box_wireframe(lo, hi, color, name, legendgroup,
                   showlegend=False, width=2) -> go.Scatter3d:
    """Wireframe of an axis-aligned box as a single Scatter3d trace."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    # 12 edges, drawn as a single path with None separators
    edges = [
        (x0,y0,z0),(x1,y0,z0), None, (x1,y0,z0),(x1,y1,z0), None,
        (x1,y1,z0),(x0,y1,z0), None, (x0,y1,z0),(x0,y0,z0), None,
        (x0,y0,z1),(x1,y0,z1), None, (x1,y0,z1),(x1,y1,z1), None,
        (x1,y1,z1),(x0,y1,z1), None, (x0,y1,z1),(x0,y0,z1), None,
        (x0,y0,z0),(x0,y0,z1), None, (x1,y0,z0),(x1,y0,z1), None,
        (x1,y1,z0),(x1,y1,z1), None, (x0,y1,z0),(x0,y1,z1), None,
    ]
    xs, ys, zs = [], [], []
    for e in edges:
        if e is None:
            xs.append(None); ys.append(None); zs.append(None)
        else:
            xs.append(e[0]); ys.append(e[1]); zs.append(e[2])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name, legendgroup=legendgroup,
        showlegend=showlegend, hoverinfo="skip",
    )


def _inclined_quad_mesh(x0, y0, x1, y1, zlo, zhi, asc_x: bool,
                        color, opacity, name, legendgroup,
                        showlegend=False, htext=""):
    """Single inclined rectangle between two elevations."""
    if asc_x:
        verts = [(x0, y0, zlo), (x0, y1, zlo),
                 (x1, y1, zhi), (x1, y0, zhi)]
    else:
        verts = [(x1, y0, zlo), (x1, y1, zlo),
                 (x0, y1, zhi), (x0, y0, zhi)]
    return _mesh_from_quad(verts, color, opacity, name, legendgroup,
                           showlegend, htext)


def _stepped_quads_3d(
    x0, y0, x1, y1, zlo, zhi, asc_x: bool, dz_step: float,
    color, opacity, name, legendgroup,
    connector_id: str = "", run_index: int = 0,
    showlegend: bool = False,
) -> list:
    """Return a list of Mesh3d traces — one horizontal quad per step tread.

    Each step is a thin horizontal rectangle at its z elevation.
    A colour gradient from *zlo* to *zhi* adds depth cue.
    """
    import plotly.express as px  # for sample_colorscale

    dz = abs(zhi - zlo)
    n_steps = max(1, int(round(dz / dz_step)))
    run_dx = x1 - x0
    step_dx = abs(run_dx) / n_steps
    traces = []

    for k in range(n_steps):
        frac = (k + 0.5) / n_steps
        z_k = zlo + frac * (zhi - zlo)
        if asc_x:
            cx = x0 + frac * run_dx
        else:
            cx = x1 - frac * abs(run_dx)
        sx0, sx1 = cx - step_dx / 2, cx + step_dx / 2

        # colour gradient (darken with elevation)
        brightness = 1.0 - 0.35 * frac
        r, g, b = _hex_to_rgb(color)
        step_color = f"rgb({int(r*brightness)},{int(g*brightness)},{int(b*brightness)})"

        verts = [
            (sx0, y0, z_k), (sx0, y1, z_k),
            (sx1, y1, z_k), (sx1, y0, z_k),
        ]
        htext = (f"{connector_id} run {run_index} step {k}/{n_steps}"
                 f"  z={z_k:.2f}m")
        traces.append(
            _mesh_from_quad(
                verts, step_color, opacity, name, legendgroup,
                showlegend=(showlegend and k == 0), htext=htext,
            )
        )
    return traces


def _stepped_profile_2d(
    x0, x1, zlo, zhi, asc_x: bool, dz_step: float,
    color, opacity, name, legendgroup,
    connector_id: str = "",
) -> list:
    """Return Scatter traces for sawtooth step profile in a cross-section.

    Each step is drawn as a small notched rectangle (tread + riser).
    """
    dz = abs(zhi - zlo)
    n_steps = max(1, int(round(dz / dz_step)))
    run_dx = x1 - x0
    step_dx = abs(run_dx) / n_steps
    step_dz = dz / n_steps

    xs_path, ys_path = [], []
    for k in range(n_steps):
        frac_lo = k / n_steps
        frac_hi = (k + 1) / n_steps
        z_lo_k = zlo + frac_lo * dz
        z_hi_k = zlo + frac_hi * dz
        if asc_x:
            xl = x0 + frac_lo * run_dx
            xr = x0 + frac_hi * run_dx
        else:
            xl = x1 - frac_lo * abs(run_dx)
            xr = x1 - frac_hi * abs(run_dx)
        # tread then riser
        xs_path += [xl, xr, xr]
        ys_path += [z_hi_k, z_hi_k, z_lo_k] if not asc_x else [z_lo_k, z_lo_k, z_hi_k]

    # For ascending-x: each step starts low-left, goes right, then riser up
    # Rebuild with clear staircase profile
    xs_path, ys_path = [], []
    for k in range(n_steps):
        z_k = zlo + k * step_dz
        z_k1 = zlo + (k + 1) * step_dz
        if asc_x:
            xl = x0 + k * step_dx
            xr = x0 + (k + 1) * step_dx
            # tread (horizontal)
            xs_path += [xl, xr]
            ys_path += [z_k, z_k]
            # riser (vertical)
            xs_path += [xr, xr]
            ys_path += [z_k, z_k1]
        else:
            xl = x1 - k * step_dx
            xr = x1 - (k + 1) * step_dx
            xs_path += [xl, xr]
            ys_path += [z_k, z_k]
            xs_path += [xr, xr]
            ys_path += [z_k, z_k1]
    # final top tread
    if asc_x:
        xs_path.append(x0 + n_steps * step_dx)
    else:
        xs_path.append(x1 - n_steps * step_dx)
    ys_path.append(zhi)

    return [go.Scatter(
        x=xs_path, y=ys_path,
        mode="lines",
        line=dict(color=color, width=1.5),
        opacity=opacity,
        name=name, legendgroup=legendgroup,
        showlegend=False,
        hoverinfo="text",
        hovertext=f"{connector_id}  z=[{zlo:.1f},{zhi:.1f}]  {n_steps} steps",
    )]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex colour to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ---- stair ascent direction (reuse logic from viz.py) --------------------

def _precompute_stair_dirs(all_connectors):
    dirs = []
    for c in all_connectors:
        if c["type"] != "stair_chain":
            continue
        runs = c.get("runs", [])
        if len(runs) < 2:
            continue
        first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
        last_xc = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
        asc_x = first_xc < last_xc
        pair = tuple(sorted(c.get("connected_levels", [])))
        stair_cx = sum(r["min_x"] + r["max_x"] for r in runs) / (2 * len(runs))
        dirs.append((pair, stair_cx, asc_x))
    return dirs


def _escalator_asc_x_from_dirs(esc, stair_dirs):
    bl = esc.get("bottom_level", "")
    tl = esc.get("top_level", "")
    pair = tuple(sorted([bl, tl]))
    exc = (esc.get("min_x", 0) + esc.get("max_x", 0)) / 2
    best_dist, best = float("inf"), True
    for sp, sx, sa in stair_dirs:
        if sp == pair:
            d = abs(sx - exc)
            if d < best_dist:
                best_dist, best = d, sa
    return best


# =========================================================================

def fig_interactive_station(
    geometries: dict,
    level_nodes: dict,
    elevations: dict,
    all_connectors: list[dict],
    out_dir: str | Path,
    cfg: dict,
    *,
    G: nx.Graph | None = None,
    z_scale: float = 3.0,
) -> Path:
    """
    Interactive 3-D station map (Plotly).

    Features
    --------
    * Toggle each floor (F1/F3/F4), nodes, connectors independently
    * Hover for metadata
    * Mouse rotate / zoom

    Returns the path to the saved HTML file.
    """
    fig = go.Figure()
    levels = sorted(elevations.keys(), key=lambda k: elevations[k])
    stair_dirs = _precompute_stair_dirs(all_connectors)

    # ---- Floor outlines and filled surfaces ----------------------------
    for lvl in levels:
        z = elevations[lvl] * z_scale
        g = geometries.get(lvl, {})
        floor = g.get("floor")
        color = LEVEL_HEX.get(lvl, "#999")
        fill_rgba = LEVEL_RGBA.get(lvl, "rgba(150,150,150,{a})").format(a=0.10)

        first_poly = True
        if floor and not floor.is_empty:
            for poly in flatten_polygons(floor):
                # Outline
                fig.add_trace(_polygon_boundary_trace(
                    poly, z, color, lvl,
                    legendgroup=lvl, showlegend=first_poly, width=2.5))
                # Filled mesh
                xs, ys = poly.exterior.xy
                verts = list(zip(xs, ys, [z] * len(xs)))
                fig.add_trace(_mesh_from_quad(
                    verts, fill_rgba, 0.12,
                    f"{lvl} floor", legendgroup=lvl,
                    htext=f"{lvl}  elev={elevations[lvl]:.1f}m"))
                first_poly = False

    # ---- Nodes per level -----------------------------------------------
    for lvl in levels:
        z = elevations[lvl] * z_scale
        nodes = level_nodes.get(lvl, [])
        if not nodes:
            continue
        nxs = [n[0] for n in nodes]
        nys = [n[1] for n in nodes]
        color = LEVEL_HEX.get(lvl, "#999")
        fig.add_trace(go.Scatter3d(
            x=nxs, y=nys, z=[z] * len(nodes),
            mode="markers",
            marker=dict(size=1.2, color=color, opacity=0.35),
            name=f"{lvl} nodes ({len(nodes)})",
            legendgroup=f"{lvl}_nodes",
            hoverinfo="text",
            hovertext=[f"{lvl} ({x:.1f}, {y:.1f})" for x, y in zip(nxs, nys)],
        ))

    # ---- Stair chains (per-step 3D quads) ----------------------------------
    stair_first = True
    stair_dz = cfg.get("connectors", {}).get("stair", {}).get("dz_step_m", 0.18)
    for c in all_connectors:
        if c["type"] != "stair_chain":
            continue
        runs = c.get("runs", [])
        if not runs:
            continue
        cid = c.get("id", "stair")
        color = CONN_HEX["stair"]

        first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
        last_xc = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
        asc_x = first_xc < last_xc

        for ri, run in enumerate(runs):
            x0, x1 = run["min_x"], run["max_x"]
            y0, y1 = run["min_y"], run["max_y"]
            zlo = run["z_min"] * z_scale
            zhi = run["z_max"] * z_scale
            for tr in _stepped_quads_3d(
                x0, y0, x1, y1, zlo, zhi, asc_x, stair_dz * z_scale,
                color, 0.65, "Stair", "conn_stair",
                connector_id=cid, run_index=ri,
                showlegend=(stair_first and ri == 0),
            ):
                fig.add_trace(tr)
        stair_first = False

        # Landings
        hw = (runs[0]["max_y"] - runs[0]["min_y"]) / 2
        hl = 1.5
        for ld in c.get("landings", []):
            lx, ly, lz = ld["x"], ld["y"], ld["z"] * z_scale
            verts = [(lx - hl, ly - hw, lz), (lx - hl, ly + hw, lz),
                     (lx + hl, ly + hw, lz), (lx + hl, ly - hw, lz)]
            fig.add_trace(_mesh_from_quad(
                verts, color, 0.45, "Stair landing",
                legendgroup="conn_stair",
                htext=f"{cid} landing z={ld['z']:.1f}"))

    # ---- Escalators (per-step 3D quads) ------------------------------------
    esc_first = True
    esc_dz = cfg.get("connectors", {}).get("escalator", {}).get("dz_step_m", 0.40)
    for c in all_connectors:
        if c["type"] != "escalator":
            continue
        color = CONN_HEX["escalator"]
        cid = c.get("id", "esc")
        bot_lv = c.get("bottom_level")
        top_lv = c.get("top_level")
        if not (bot_lv and top_lv):
            continue
        if bot_lv not in elevations or top_lv not in elevations:
            continue
        zlo = elevations[bot_lv] * z_scale
        zhi = elevations[top_lv] * z_scale
        # Use physical landing positions (bottom_xy / top_xy)
        bot_xy = c.get("bottom_xy")
        top_xy = c.get("top_xy")
        if bot_xy and top_xy:
            asc_x = bot_xy[0] < top_xy[0]
            x_lo = min(bot_xy[0], top_xy[0])
            x_hi = max(bot_xy[0], top_xy[0])
        else:
            x_lo = c.get("min_x", 0)
            x_hi = c.get("max_x", 0)
            asc_x = _escalator_asc_x_from_dirs(c, stair_dirs)
        y0 = c.get("min_y", 0)
        y1 = c.get("max_y", 0)
        if x_hi - x_lo < 0.1:
            continue
        for tr in _stepped_quads_3d(
            x_lo, y0, x_hi, y1, zlo, zhi, asc_x, esc_dz * z_scale,
            color, 0.60, "Escalator", "conn_escalator",
            connector_id=cid,
            showlegend=esc_first,
        ):
            fig.add_trace(tr)
        esc_first = False

    # ---- Elevators -------------------------------------------------------
    elev_first = True
    for c in all_connectors:
        if c["type"] != "elevator":
            continue
        color = CONN_HEX["elevator"]
        cid = c.get("id", "elev")
        served = [lk for lk in c.get("connected_levels", [])
                  if lk in elevations]
        if len(served) < 2:
            continue
        zs = sorted([elevations[lk] * z_scale for lk in served])
        fp = c.get("footprint")
        if not (fp and not fp.is_empty):
            continue
        x0, y0, x1, y1 = fp.bounds

        # Wireframe box
        fig.add_trace(_box_wireframe(
            (x0, y0, zs[0]), (x1, y1, zs[-1]),
            color, "Elevator", legendgroup="conn_elevator",
            showlegend=elev_first, width=3))

        # Floor planes at each served level
        for lk in served:
            ze = elevations[lk] * z_scale
            verts = [(x0, y0, ze), (x1, y0, ze),
                     (x1, y1, ze), (x0, y1, ze)]
            fig.add_trace(_mesh_from_quad(
                verts, color, 0.55, f"Elevator {lk}",
                legendgroup="conn_elevator",
                htext=f"{cid} @ {lk}"))
        elev_first = False

    # ---- Entrance / PSD spawn layers (if graph is provided) ---------------
    if G is not None:
        # Blind-path floor nodes (tactile paving): overlay for visibility
        blind_guide = [(n, d) for n, d in G.nodes(data=True)
                       if d.get("node_type") == "floor" and d.get("blind_category") == "guide"]
        blind_warning = [(n, d) for n, d in G.nodes(data=True)
                         if d.get("node_type") == "floor" and d.get("blind_category") == "warning"]

        if blind_guide:
            fig.add_trace(go.Scatter3d(
                x=[d["x"] for _, d in blind_guide],
                y=[d["y"] for _, d in blind_guide],
                z=[elevations.get(d.get("level"), d.get("z", 0)) * z_scale
                   for _, d in blind_guide],
                mode="markers",
                marker=dict(size=3.2, color="#FBC02D", symbol="square", opacity=0.95),
                name=f"Blind Guide Nodes ({len(blind_guide)})",
                legendgroup="blind_nodes",
                hoverinfo="text",
                hovertext=[f"Blind guide node ({d.get('level','')})" for _, d in blind_guide],
            ))

        if blind_warning:
            fig.add_trace(go.Scatter3d(
                x=[d["x"] for _, d in blind_warning],
                y=[d["y"] for _, d in blind_warning],
                z=[elevations.get(d.get("level"), d.get("z", 0)) * z_scale
                   for _, d in blind_warning],
                mode="markers",
                marker=dict(size=4.0, color="#D32F2F", symbol="x", opacity=0.95),
                name=f"Blind Warning Nodes ({len(blind_warning)})",
                legendgroup="blind_nodes",
                hoverinfo="text",
                hovertext=[f"Blind warning node ({d.get('level','')})" for _, d in blind_warning],
            ))

        ent_nodes = [(n, d) for n, d in G.nodes(data=True)
                     if d.get("node_type") == "entrance"]
        if ent_nodes:
            fig.add_trace(go.Scatter3d(
                x=[d["x"] for _, d in ent_nodes],
                y=[d["y"] for _, d in ent_nodes],
                z=[elevations.get(d.get("level"), d.get("z", 0)) * z_scale
                   for _, d in ent_nodes],
                mode="markers+text",
                marker=dict(size=9, color="#00BCD4", symbol="diamond",
                            opacity=1.0, line=dict(color="white", width=1.5)),
                text=[
                    d.get("entrance_name", "").replace("entrance_", "Gate ")
                    if d.get("entrance_primary", False) else ""
                    for _, d in ent_nodes
                ],
                textposition="top center",
                textfont=dict(size=10, color="#006064"),
                name=f"Entrances ({len(ent_nodes)} gates)",
                legendgroup="ent_layer",
                hoverinfo="text",
                hovertext=[f"Entrance: {d.get('entrance_name','')}  "
                           f"({d['x']:.1f},{d['y']:.1f})  {d.get('level','')}<br>"
                           f"Inbound spawn / Outbound destination"
                           for _, d in ent_nodes],
            ))
        psd_nodes = [(n, d) for n, d in G.nodes(data=True)
                     if d.get("node_type") == "door_platform"
                     and d.get("level") == "F1"]
        if psd_nodes:
            fig.add_trace(go.Scatter3d(
                x=[d["x"] for _, d in psd_nodes],
                y=[d["y"] for _, d in psd_nodes],
                z=[elevations.get("F1", 0) * z_scale] * len(psd_nodes),
                mode="markers",
                marker=dict(size=4, color="#FF6F00", symbol="square",
                            opacity=0.80, line=dict(color="white", width=0.4)),
                name=f"Platform Screen Doors ({len(psd_nodes)} PSD)",
                legendgroup="psd_layer",
                hoverinfo="text",
                hovertext=[f"PSD: {nid}  ({d['x']:.1f},{d['y']:.1f}) F1<br>"
                           f"Outbound spawn / Inbound destination"
                           for nid, d in psd_nodes],
            ))

    # ---- Layout ----------------------------------------------------------
    fig.update_layout(
        title=dict(text="Interactive 3D Station Map",
                   font=dict(size=18)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (×{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.7),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1500, height=900,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11),
            itemsizing="constant",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out = _ensure_dir(out_dir) / "interactive_station_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# =========================================================================
# 2. Interactive graph topology explorer
# =========================================================================

def fig_interactive_graph(
    G: nx.Graph,
    all_connectors: list[dict],
    elevations: dict,
    out_dir: str | Path,
    cfg: dict,
    *,
    z_scale: float = 3.0,
    floor_edge_sample: float = 0.05,
) -> Path:
    """
    Interactive graph explorer (Plotly).

    Shows nodes coloured by level, edges by type.
    Floor edges are sub-sampled for performance.

    Parameters
    ----------
    floor_edge_sample : float
        Fraction of floor edges to display (0.05 = 5 %).
    """
    fig = go.Figure()

    # ---- Nodes by level ---
    by_level: dict[str, list] = defaultdict(list)
    for n, d in G.nodes(data=True):
        by_level[d.get("level", "?")].append((n, d))

    for lvl in sorted(by_level.keys()):
        nds = by_level[lvl]
        color = LEVEL_HEX.get(lvl, "#999")
        xs = [d.get("x", 0) for _, d in nds]
        ys = [d.get("y", 0) for _, d in nds]
        # Step nodes carry their own z; floor nodes use level elevation
        if lvl in ("STAIR", "ESCALATOR"):
            zs = [d.get("z", 0) * z_scale for _, d in nds]
            color = CONN_HEX.get("stair" if lvl == "STAIR" else "escalator", "#999")
        else:
            zz = elevations.get(lvl, 0) * z_scale
            zs = [zz] * len(nds)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=1.5, color=color, opacity=0.4),
            name=f"{lvl} ({len(nds)} nodes)",
            legendgroup=f"nodes_{lvl}",
            hoverinfo="text",
            hovertext=[f"n={nid}  {lvl}  ({d.get('x',0):.1f},{d.get('y',0):.1f})"
                       for nid, d in nds],
        ))

    # ---- Entrance nodes (larger markers, cyan) ---
    blind_guide = [(n, d) for n, d in G.nodes(data=True)
                   if d.get("node_type") == "floor" and d.get("blind_category") == "guide"]
    blind_warning = [(n, d) for n, d in G.nodes(data=True)
                     if d.get("node_type") == "floor" and d.get("blind_category") == "warning"]

    if blind_guide:
        fig.add_trace(go.Scatter3d(
            x=[d["x"] for _, d in blind_guide],
            y=[d["y"] for _, d in blind_guide],
            z=[elevations.get(d.get("level"), d.get("z", 0)) * z_scale for _, d in blind_guide],
            mode="markers",
            marker=dict(size=3.0, color="#FBC02D", symbol="square", opacity=0.95),
            name=f"Blind Guide ({len(blind_guide)})",
            legendgroup="blind",
            hoverinfo="text",
            hovertext=[f"Blind guide: {nid}" for nid, _ in blind_guide],
        ))

    if blind_warning:
        fig.add_trace(go.Scatter3d(
            x=[d["x"] for _, d in blind_warning],
            y=[d["y"] for _, d in blind_warning],
            z=[elevations.get(d.get("level"), d.get("z", 0)) * z_scale for _, d in blind_warning],
            mode="markers",
            marker=dict(size=3.6, color="#D32F2F", symbol="x", opacity=0.95),
            name=f"Blind Warning ({len(blind_warning)})",
            legendgroup="blind",
            hoverinfo="text",
            hovertext=[f"Blind warning: {nid}" for nid, _ in blind_warning],
        ))

    ent_nodes = [(n, d) for n, d in G.nodes(data=True)
                 if d.get("node_type") == "entrance"]
    if ent_nodes:
        exs = [d["x"] for _, d in ent_nodes]
        eys = [d["y"] for _, d in ent_nodes]
        ezs = [elevations.get(d.get("level"), d.get("z", 0)) * z_scale
               for _, d in ent_nodes]
        fig.add_trace(go.Scatter3d(
            x=exs, y=eys, z=ezs,
            mode="markers+text",
            marker=dict(size=8, color="#00BCD4", symbol="diamond",
                        opacity=0.95, line=dict(color="white", width=1)),
            text=[
                d.get("entrance_name", "").replace("entrance_", "Gate ")
                if d.get("entrance_primary", False) else ""
                for _, d in ent_nodes
            ],
            textposition="top center",
            textfont=dict(size=10, color="#006064"),
            name=f"Entrances ({len(ent_nodes)})",
            legendgroup="entrances",
            hoverinfo="text",
            hovertext=[f"Entrance: {d.get('entrance_name','')}  "
                       f"({d['x']:.1f},{d['y']:.1f})  {d.get('level','')}<br>"
                       f"<b>Inbound spawn / Outbound destination</b>"
                       for _, d in ent_nodes],
        ))

    # ---- Platform PSD door_platform nodes (spawn for outbound direction) ---
    psd_nodes = [(n, d) for n, d in G.nodes(data=True)
                 if d.get("node_type") == "door_platform" and d.get("level") == "F1"]
    if psd_nodes:
        pxs = [d["x"] for _, d in psd_nodes]
        pys = [d["y"] for _, d in psd_nodes]
        pzs = [elevations.get("F1", 0) * z_scale for _ in psd_nodes]
        fig.add_trace(go.Scatter3d(
            x=pxs, y=pys, z=pzs,
            mode="markers",
            marker=dict(size=4, color="#FF6F00", symbol="square",
                        opacity=0.85, line=dict(color="white", width=0.5)),
            name=f"Platform Screen Doors ({len(psd_nodes)})",
            legendgroup="psd",
            hoverinfo="text",
            hovertext=[f"PSD: {nid}<br>({d['x']:.1f},{d['y']:.1f}) F1<br>"
                       f"<b>Outbound spawn / Inbound destination</b>"
                       for nid, d in psd_nodes],
        ))

    # ---- Edges by type ---
    edge_by_type: dict[str, list] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        edge_by_type[d.get("edge_type", "floor")].append((u, v, d))

    rng = np.random.default_rng(42)

    edge_colors = {
        "floor": "#BDBDBD", "stair": "#795548", "escalator": "#E91E63",
        "elevator": "#9C27B0", "elevator_door": "#AB47BC",
        "elevator_interior": "#CE93D8", "psd_door": "#FF9800",
        "anchor_snap": "#4CAF50", "entrance": "#00BCD4",
    }

    for etype, edges in edge_by_type.items():
        # Sub-sample floor edges
        if etype == "floor":
            k = max(1, int(len(edges) * floor_edge_sample))
            idx = rng.choice(len(edges), size=k, replace=False)
            edges = [edges[i] for i in idx]

        color = edge_colors.get(etype, "#666")
        xs, ys, zs = [], [], []
        htexts = []
        for u, v, d in edges:
            ud, vd = G.nodes[u], G.nodes[v]
            ux, uy = ud.get("x", 0), ud.get("y", 0)
            vx, vy = vd.get("x", 0), vd.get("y", 0)
            # For step nodes (STAIR / ESCALATOR level), use their actual z
            ulvl = ud.get("level", "")
            vlvl = vd.get("level", "")
            uz = (ud["z"] * z_scale if ulvl in ("STAIR", "ESCALATOR")
                  else elevations.get(ulvl, 0) * z_scale)
            vz = (vd["z"] * z_scale if vlvl in ("STAIR", "ESCALATOR")
                  else elevations.get(vlvl, 0) * z_scale)
            xs += [ux, vx, None]
            ys += [uy, vy, None]
            zs += [uz, vz, None]
        total = len(edge_by_type.get(etype, []))
        shown = len(edges)
        label = f"{etype} ({shown}" + (f"/{total}" if shown < total else "") + ")"
        width = 1.0 if etype == "floor" else 3.0
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            name=label, legendgroup=f"edge_{etype}",
            hoverinfo="skip",
            opacity=0.4 if etype == "floor" else 0.85,
        ))

    fig.update_layout(
        title=dict(text="Interactive Graph Explorer", font=dict(size=18)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (×{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.7),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1500, height=900,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out = _ensure_dir(out_dir) / "interactive_graph.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# =========================================================================
# 3. Interactive cross-section
# =========================================================================

def fig_interactive_cross_section(
    geometries: dict,
    all_connectors: list[dict],
    elevations: dict,
    out_dir: str | Path,
    cfg: dict,
    *,
    z_scale: float = 1.0,
) -> Path:
    """
    Interactive XZ cross-section along the station longitudinal axis.

    * Y-slider to move the cut plane
    * Floors shown as horizontal bands
    * Connectors as inclined / vertical shapes
    """
    fig = go.Figure()
    levels = sorted(elevations.keys(), key=lambda k: elevations[k])
    stair_dirs = _precompute_stair_dirs(all_connectors)

    # ---- Floor slabs (XZ rectangles) ---
    for lvl in levels:
        g = geometries.get(lvl, {})
        floor = g.get("floor")
        if not (floor and not floor.is_empty):
            continue
        elev = elevations[lvl]
        x0, _, x1, _ = floor.bounds
        color = LEVEL_HEX.get(lvl, "#999")
        slab = 0.6
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[elev - slab / 2, elev - slab / 2,
               elev + slab / 2, elev + slab / 2, elev - slab / 2],
            fill="toself",
            fillcolor=LEVEL_RGBA.get(lvl, "rgba(150,150,150,{a})").format(a=0.3),
            line=dict(color=color, width=2),
            name=lvl,
            legendgroup=lvl,
            hoverinfo="text",
            hovertext=f"{lvl}  elev={elev:.1f}m  x=[{x0:.0f},{x1:.0f}]",
        ))
        # Label
        fig.add_annotation(
            x=x0 - 3, y=elev,
            text=f"<b>{lvl}</b>", showarrow=False,
            font=dict(size=14, color=color),
        )

    # ---- Stair chains (stepped profile) ---
    stair_dz = cfg.get("connectors", {}).get("stair", {}).get("dz_step_m", 0.18)
    for c in all_connectors:
        if c["type"] != "stair_chain":
            continue
        runs = c.get("runs", [])
        if not runs:
            continue
        cid = c.get("id", "stair")
        color = CONN_HEX["stair"]
        first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
        last_xc = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
        asc_x = first_xc < last_xc

        for run in runs:
            x0, x1 = run["min_x"], run["max_x"]
            zlo, zhi = run["z_min"], run["z_max"]
            for tr in _stepped_profile_2d(
                x0, x1, zlo, zhi, asc_x, stair_dz,
                color, 0.7, "Stair", "stair",
                connector_id=cid,
            ):
                fig.add_trace(tr)

    # ---- Escalators (stepped profile) ---
    esc_dz = cfg.get("connectors", {}).get("escalator", {}).get("dz_step_m", 0.40)
    for c in all_connectors:
        if c["type"] != "escalator":
            continue
        cid = c.get("id", "esc")
        color = CONN_HEX["escalator"]
        bot_lv = c.get("bottom_level")
        top_lv = c.get("top_level")
        if not (bot_lv and top_lv):
            continue
        if bot_lv not in elevations or top_lv not in elevations:
            continue
        zlo = elevations[bot_lv]
        zhi = elevations[top_lv]
        # Use physical landings if available
        bot_xy = c.get("bottom_xy")
        top_xy = c.get("top_xy")
        if bot_xy and top_xy:
            asc_x = bot_xy[0] < top_xy[0]
            x0 = min(bot_xy[0], top_xy[0])
            x1 = max(bot_xy[0], top_xy[0])
        else:
            x0 = c.get("min_x", 0)
            x1 = c.get("max_x", 0)
            asc_x = _escalator_asc_x_from_dirs(c, stair_dirs)
        for tr in _stepped_profile_2d(
            x0, x1, zlo, zhi, asc_x, esc_dz,
            color, 0.7, "Escalator", "escalator",
            connector_id=cid,
        ):
            fig.add_trace(tr)

    # ---- Elevators ---
    for c in all_connectors:
        if c["type"] != "elevator":
            continue
        cid = c.get("id", "elev")
        color = CONN_HEX["elevator"]
        served = [lk for lk in c.get("connected_levels", [])
                  if lk in elevations]
        if len(served) < 2:
            continue
        zs_vals = sorted([elevations[lk] for lk in served])
        fp = c.get("footprint")
        if not (fp and not fp.is_empty):
            continue
        x0, _, x1, _ = fp.bounds
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[zs_vals[0], zs_vals[0], zs_vals[-1], zs_vals[-1], zs_vals[0]],
            fill="toself",
            fillcolor=CONN_HEX["elevator"],
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.35,
            name="Elevator", legendgroup="elevator",
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{cid}  {', '.join(served)}",
        ))

    # ---- Singleton legend entries ---
    for ctype, lbl in [("stair", "Stair"), ("escalator", "Escalator"),
                       ("elevator", "Elevator")]:
        has = any(c["type"] in (ctype, ctype + "_chain") for c in all_connectors)
        if has:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=CONN_HEX[ctype], size=10),
                name=lbl, legendgroup=ctype, showlegend=True,
            ))

    fig.update_layout(
        title=dict(text="Longitudinal Cross-Section (XZ)",
                   font=dict(size=16)),
        xaxis_title="X (m)",
        yaxis_title="Elevation (m)",
        width=1400, height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        hovermode="closest",
    )

    out = _ensure_dir(out_dir) / "interactive_cross_section.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# =========================================================================
# 4. Interactive agent-flow map (Step 4)
# =========================================================================

def fig_interactive_agent_flow(
    G: nx.Graph,
    regions: dict[str, list[str]],
    agents: list[dict],
    geometries: dict,
    elevations: dict,
    out_dir: str | Path,
    cfg: dict,
    *,
    z_scale: float = 3.0,
    n_sample_paths: int = 10,
) -> Path:
    """Interactive 3D visualization of the two-direction pedestrian flow.

    Shows:
    - Floor outlines (F1/F3/F4)
    - ENTRANCE nodes (cyan ♦) — station gate spawn points for inbound flow
    - PLATFORM nodes (orange ■) — PSD door spawn points for outbound flow
    - Sample shortest paths for each direction
    - Toggle each direction on/off via legend

    Output: interactive_agent_flow.html
    """
    from src.routing import find_path
    import random as _random

    fig = go.Figure()

    # ---- Floor outlines --------------------------------------------------
    for lvl in sorted(elevations.keys(), key=lambda k: elevations[k]):
        z = elevations[lvl] * z_scale
        g = geometries.get(lvl, {})
        floor = g.get("floor")
        if not floor or floor.is_empty:
            continue
        color = LEVEL_HEX.get(lvl, "#999")
        first = True
        for poly in flatten_polygons(floor):
            fig.add_trace(_polygon_boundary_trace(
                poly, z, color, f"Floor {lvl}",
                legendgroup=f"floor_{lvl}", showlegend=first, width=2.0))
            first = False

    # ---- ENTRANCE nodes (5 station gates) --------------------------------
    ent_ids = regions.get("ENTRANCE", [])
    if ent_ids:
        exs = [G.nodes[n]["x"] for n in ent_ids if n in G]
        eys = [G.nodes[n]["y"] for n in ent_ids if n in G]
        ezs = [elevations.get(G.nodes[n].get("level"), 0) * z_scale
               for n in ent_ids if n in G]
        labels = [G.nodes[n].get("entrance_name", n).replace("entrance_", "Gate ")
                  for n in ent_ids if n in G]
        fig.add_trace(go.Scatter3d(
            x=exs, y=eys, z=ezs,
            mode="markers+text",
            marker=dict(size=10, color="#00ACC1", symbol="diamond",
                        opacity=1.0, line=dict(color="white", width=1.5)),
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#006064"),
            name=f"Entrances ({len(exs)} gates, F3/F4)",
            legendgroup="entrance_spawn",
            hoverinfo="text",
            hovertext=[f"Entrance: {lbl}<br>Inbound spawn / Outbound destination<br>"
                       f"Spawn for: ENTRANCE→PLATFORM"
                       for lbl in labels],
        ))

    # ---- PLATFORM nodes (PSD door_platform on F1) -----------------------
    plt_ids = regions.get("PLATFORM", [])
    if plt_ids:
        pxs = [G.nodes[n]["x"] for n in plt_ids if n in G]
        pys = [G.nodes[n]["y"] for n in plt_ids if n in G]
        pzs = [elevations.get("F1", 0) * z_scale] * len(pxs)
        fig.add_trace(go.Scatter3d(
            x=pxs, y=pys, z=pzs,
            mode="markers",
            marker=dict(size=5, color="#FF6F00", symbol="square",
                        opacity=0.9, line=dict(color="white", width=0.5)),
            name=f"Platform Screen Doors ({len(pxs)} PSD, F1)",
            legendgroup="platform_spawn",
            hoverinfo="text",
            hovertext=[f"PSD: {nid}<br>Outbound spawn / Inbound destination<br>"
                       f"Spawn for: PLATFORM→EXIT"
                       for nid in [n for n in plt_ids if n in G]],
        ))

    # ---- Sample paths -- Inbound: ENTRANCE -> PLATFORM ----------------------
    flow_colours = {"ENTRANCE->PLATFORM": "#E91E63", "PLATFORM->EXIT": "#4CAF50"}
    flow_labels  = {"ENTRANCE->PLATFORM": "Inbound ENT\u2192PLT",
                    "PLATFORM->EXIT":     "Outbound PLT\u2192EXIT"}

    rng = _random.Random(cfg.get("simulation", {}).get("seed", 42))
    shown: dict[str, int] = {"ENTRANCE->PLATFORM": 0, "PLATFORM->EXIT": 0}
    quota = n_sample_paths // 2 or 1

    path_xs: dict[str, list] = {k: [] for k in flow_colours}
    path_ys: dict[str, list] = {k: [] for k in flow_colours}
    path_zs: dict[str, list] = {k: [] for k in flow_colours}

    for agent in rng.sample(agents, min(len(agents), n_sample_paths * 4)):
        flow = agent.get("flow", "")
        if flow not in shown or shown[flow] >= quota:
            continue
        path = find_path(G, agent["origin"], agent["dest"])
        if len(path) < 2:
            continue
        for nd in path:
            if nd not in G:
                continue
            d = G.nodes[nd]
            lvl = d.get("level", "")
            path_xs[flow].append(d.get("x", 0))
            path_ys[flow].append(d.get("y", 0))
            z_nd = (d.get("z", 0) * z_scale
                    if lvl in ("STAIR", "ESCALATOR")
                    else elevations.get(lvl, 0) * z_scale)
            path_zs[flow].append(z_nd)
        path_xs[flow].append(None)
        path_ys[flow].append(None)
        path_zs[flow].append(None)
        shown[flow] += 1

    for flow, colour in flow_colours.items():
        if not path_xs[flow]:
            continue
        fig.add_trace(go.Scatter3d(
            x=path_xs[flow], y=path_ys[flow], z=path_zs[flow],
            mode="lines",
            line=dict(color=colour, width=3),
            opacity=0.75,
            name=f"{flow_labels[flow]} ({shown[flow]} paths)",
            legendgroup=f"path_{flow}",
            hoverinfo="skip",
        ))

    # ---- Layout ----------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="Pedestrian Flow Regions · Bidirectional Interactive Flow<br>"
                 "<sub>Entrance gates → Platform PSD (Inbound)　|　Platform PSD → Entrance gates (Outbound)</sub>",
            font=dict(size=16)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (×{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.7),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        width=1500, height=900,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.88)",
            font=dict(size=12),
            title=dict(text="Layers (click to toggle)", font=dict(size=12)),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
    )

    out = _ensure_dir(out_dir) / "interactive_agent_flow.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# Distinct palette for up to 5 entrances
_ENT_PALETTE = [
    "#1565C0",  # A — blue
    "#6A1B9A",  # B — purple
    "#2E7D32",  # C — green
    "#E65100",  # D — orange
    "#AD1457",  # E — pink
]


def fig_interactive_entrance_routes(
    G: nx.Graph,
    entrance_paths: list[dict],
    geometries: dict,
    elevations: dict,
    out_dir: str | Path,
    cfg: dict,
    *,
    z_scale: float = 3.0,
) -> Path:
    """Interactive 3D visualisation of per-entrance bidirectional routes.

    Layout:
      • Each entrance has its own colour (A-E).
      • Inbound  (entrance→PSD) — solid thick line, legend: "Inbound Gate X".
      • Outbound (PSD→entrance) — dashed thinner line, legend: "Outbound Gate X".
      • Fare gate nodes: green diamond = entry gate, red diamond = exit gate.
      • Dropdown filter buttons: All / Inbound Only / Outbound Only.

    Output: interactive_entrance_routes.html
    """
    fig = go.Figure()

    # ---- Floor outlines --------------------------------------------------
    for lvl in sorted(elevations.keys(), key=lambda k: elevations[k]):
        z_lv = elevations[lvl] * z_scale
        gd = geometries.get(lvl, {})
        floor = gd.get("floor")
        if not floor or floor.is_empty:
            continue
        color = LEVEL_HEX.get(lvl, "#999")
        first = True
        for poly in flatten_polygons(floor):
            fig.add_trace(_polygon_boundary_trace(
                poly, z_lv, color, f"Floor {lvl}",
                legendgroup=f"floor_{lvl}", showlegend=first, width=1.5))
            first = False

    # ---- Per-entrance routes ---------------------------------------------
    def _node_xyz(nid):
        if nid not in G:
            return None, None, None
        d = G.nodes[nid]
        lvl = d.get("level", "")
        zv = d.get("z", 0) * z_scale if lvl in ("STAIR", "ESCALATOR") \
            else elevations.get(lvl, 0) * z_scale
        return d.get("x", 0), d.get("y", 0), zv

    def _path_coords(path):
        xs, ys, zs = [], [], []
        for nid in path:
            x, y, z = _node_xyz(nid)
            if x is None:
                continue
            xs.append(x); ys.append(y); zs.append(z)
        return xs, ys, zs

    # Track trace indices for dropdown visibility
    floor_trace_count = sum(
        len(list(flatten_polygons(geometries.get(lv, {}).get("floor") or __import__("shapely.geometry", fromlist=["MultiPolygon"]).MultiPolygon())))
        for lv in elevations
        if geometries.get(lv, {}).get("floor") and not geometries[lv]["floor"].is_empty
    )

    inbound_indices:  list[int] = []
    outbound_indices: list[int] = []

    for i, ep in enumerate(entrance_paths):
        colour = _ENT_PALETTE[i % len(_ENT_PALETTE)]
        ename  = ep["entrance_name"].replace("entrance_", "Gate ")

        # --- Inbound path ---
        ix, iy, iz = _path_coords(ep["inbound_path"])
        if ix:
            idx = len(fig.data)
            fig.add_trace(go.Scatter3d(
                x=ix, y=iy, z=iz,
                mode="lines",
                line=dict(color=colour, width=5),
                opacity=0.90,
                name=f"Inbound {ename} ({ep['inbound_cost']:.0f}s)",
                legendgroup="inbound",
                legendgrouptitle=dict(text="Inbound Routes") if i == 0 else dict(),
                hoverinfo="skip",
                visible=True,
            ))
            inbound_indices.append(idx)

        # --- Outbound path ---
        ox, oy, oz = _path_coords(ep["outbound_path"])
        if ox:
            idx = len(fig.data)
            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz,
                mode="lines",
                line=dict(color=colour, width=3, dash="dash"),
                opacity=0.80,
                name=f"Outbound {ename} ({ep['outbound_cost']:.0f}s)",
                legendgroup="outbound",
                legendgrouptitle=dict(text="Outbound Routes") if i == 0 else dict(),
                hoverinfo="skip",
                visible=True,
            ))
            outbound_indices.append(idx)

        # --- Entrance marker ---
        eid = ep["entrance_id"]
        ex, ey, ez = _node_xyz(eid)
        if ex is not None:
            fig.add_trace(go.Scatter3d(
                x=[ex], y=[ey], z=[ez],
                mode="markers+text",
                marker=dict(size=12, color=colour, symbol="diamond",
                            opacity=1.0, line=dict(color="white", width=1.5)),
                text=[ename],
                textposition="top center",
                textfont=dict(size=11, color=colour),
                name=f"Entrance {ename}",
                legendgroup="entrances",
                legendgrouptitle=dict(text="Entrances") if i == 0 else dict(),
                showlegend=True,
                hoverinfo="text",
                hovertext=[f"{ename} (Level: {ep['level']})<br>"
                           f"Inbound: {ep['inbound_cost']:.0f}s | "
                           f"Outbound: {ep['outbound_cost']:.0f}s"],
            ))

        # --- PSD marker ---
        pid = ep["psd_id"]
        px, py, pz = _node_xyz(pid)
        if px is not None:
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode="markers",
                marker=dict(size=7, color=colour, symbol="square",
                            opacity=0.85, line=dict(color="white", width=1.0)),
                showlegend=False,
                hoverinfo="text",
                hovertext=[f"PSD: {pid}"],
            ))

        # --- Fare gate markers ---
        for gate_key, gate_color, gate_label, ggroup_title in [
            ("inbound_gate",  "#2E7D32", "Entry Gate", "Entry Gates (Inbound)"),
            ("outbound_gate", "#C62828", "Exit Gate",  "Exit Gates (Outbound)"),
        ]:
            gnid = ep.get(gate_key)
            if not gnid or gnid not in G:
                continue
            gx, gy, gz = _node_xyz(gnid)
            if gx is None:
                continue
            ga = G.nodes[gnid]
            show_in_legend = (i == 0)
            fig.add_trace(go.Scatter3d(
                x=[gx], y=[gy], z=[gz],
                mode="markers",
                marker=dict(size=10, color=gate_color, symbol="diamond",
                            opacity=1.0, line=dict(color="white", width=1.5)),
                name=ggroup_title,
                legendgroup=gate_key,
                legendgrouptitle=dict(text=ggroup_title) if show_in_legend
                    else dict(),
                showlegend=show_in_legend,
                hoverinfo="text",
                hovertext=[f"{gate_label}: {gnid}<br>"
                           f"({ga.get('x',0):.1f}, {ga.get('y',0):.1f})<br>"
                           f"Entrance: {ename}"],
            ))

    # ---- Dropdown buttons: All / Inbound Only / Outbound Only ------------
    n_total = len(fig.data)

    in_set  = set(inbound_indices)
    out_set = set(outbound_indices)

    def _visibility(show_in: bool, show_out: bool):
        v = []
        for idx in range(n_total):
            if idx in in_set:
                v.append(show_in)
            elif idx in out_set:
                v.append(show_out)
            else:
                v.append(True)
        return v

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.01, y=1.09,
            bgcolor="white",
            bordercolor="#aaa",
            font=dict(size=12),
            buttons=[
                dict(label="All",
                     method="update",
                     args=[{"visible": _visibility(True, True)}]),
                dict(label="Inbound Only",
                     method="update",
                     args=[{"visible": _visibility(True, False)}]),
                dict(label="Outbound Only",
                     method="update",
                     args=[{"visible": _visibility(False, True)}]),
            ],
            showactive=True,
            active=0,
        )],
    )

    # ---- Layout ----------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="Bidirectional Entrance Routes<br>"
                 "<sub>Solid = Inbound (Entrance\u2192PSD)  |  Dashed = Outbound (PSD\u2192Entrance)"
                 "  |  \u25c6 Green = Entry Gate  \u25c6 Red = Exit Gate</sub>",
            font=dict(size=15)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (\xd7{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.7),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        width=1500, height=900,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.88)",
            font=dict(size=12),
            groupclick="toggleitem",
        ),
        margin=dict(l=10, r=10, t=100, b=10),
    )

    out = _ensure_dir(out_dir) / "interactive_entrance_routes.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# =========================================================================
# 6. Interactive ABM simulation animation (2-D top-down, time slider)
# =========================================================================

def fig_interactive_simulation_animation(
    traj_path: str | Path,
    geometries: dict,
    elevations: dict,
    out_dir: str | Path,
    *,
    dt_frame: float = 2.0,
    label: str = "static",
) -> Path:
    """Animated 2-D top-down visualization of ABM agent positions over time.

    Reads the ``traj_agents.jsonl`` produced by ``run_simulation()`` and
    generates a self-contained Plotly HTML with a time-slider + Play/Pause
    control showing all agents moving through the station at each time step.

    Color code:
      🔵 Blue  — F1 Platform level  (z ≈ 0 m)
      🟢 Green — F3 Concourse level (z ≈ 12 m)
      🟠 Orange— F4 Transport Hub   (z ≈ 17 m)

    Parameters
    ----------
    traj_path : path-like
        Path to ``traj_agents.jsonl`` written by :func:`run_simulation`.
    geometries : dict
        Per-level geometry dicts (must contain ``"floor"`` polygon).
    elevations : dict
        ``{level_name: elevation_m}`` mapping, e.g. ``{"F1": 0.0, "F3": 12.1}``.
    out_dir : path-like
        Output directory.  The HTML is written as
        ``interactive_sim_{label}.html``.
    dt_frame : float
        Seconds between successive animation frames (downsampling).
    label : str
        Scenario tag used in the file name and figure title.

    Returns
    -------
    Path
        Absolute path to the written ``.html`` file.
    """
    import json
    from collections import defaultdict

    traj_path = Path(traj_path)
    out_dir_p = _ensure_dir(out_dir)

    # ------------------------------------------------------------------ load
    raw: dict[float, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            t_key = round(float(rec["t"]), 2)
            raw[t_key][rec["agent_id"]] = (
                float(rec["x"]), float(rec["y"]), float(rec["z"])
            )

    if not raw:
        # Nothing to animate — write empty placeholder
        out = out_dir_p / f"interactive_sim_{label}.html"
        out.write_text("<p>No trajectory data found.</p>")
        return out

    # --------------------------------------------------------- downsample time
    sorted_ts = sorted(raw.keys())
    keep_ts: list[float] = []
    next_keep = sorted_ts[0]
    for t in sorted_ts:
        if t >= next_keep - 1e-6:
            keep_ts.append(t)
            next_keep = t + dt_frame

    # -------------------------------- map z elevation → floor label / colour
    # Build a fast z→level lookup by rounding elevation to nearest 0.5m bucket
    _level_colors = {"F1": "#2196F3", "F3": "#4CAF50", "F4": "#FF9800"}
    _sorted_levels = sorted(elevations.items(), key=lambda kv: kv[1])  # asc z

    def _z_to_level(z: float) -> str:
        """Return floor label for a given agent z coordinate."""
        best, best_d = "F1", 1e9
        for lvl, elev in elevations.items():
            d = abs(elev - z)
            if d < best_d:
                best_d = d
                best = lvl
        return best

    # ----------------------------------------- static background: floor plans
    bg_traces: list[go.BaseTraceType] = []
    floor_shown: set[str] = set()
    for lvl in ("F4", "F3", "F1"):
        geo = geometries.get(lvl, {})
        floor_poly = geo.get("floor")
        if floor_poly is None or floor_poly.is_empty:
            continue
        color = _level_colors.get(lvl, "#999")
        for poly in flatten_polygons(floor_poly):
            xs, ys = poly.exterior.xy
            bg_traces.append(go.Scatter(
                x=list(xs), y=list(ys),
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.35,
                name=f"Floor {lvl}",
                legendgroup=f"bg_floor_{lvl}",
                showlegend=(lvl not in floor_shown),
                hoverinfo="skip",
            ))
            floor_shown.add(lvl)

    n_bg = len(bg_traces)
    agent_trace_indices = [n_bg, n_bg + 1, n_bg + 2]  # F4, F3, F1

    # -------------------------------------------- helper: build agent traces
    def _agent_traces(t_key: float) -> list[go.Scatter]:
        frame_data = raw.get(t_key, {})
        buckets: dict[str, tuple[list, list, list]] = {
            lvl: ([], [], []) for lvl in ("F4", "F3", "F1")
        }
        for aid, (x, y, z) in frame_data.items():
            lvl = _z_to_level(z)
            if lvl not in buckets:
                lvl = "F1"
            buckets[lvl][0].append(x)
            buckets[lvl][1].append(y)
            buckets[lvl][2].append(aid)

        traces = []
        for lvl in ("F4", "F3", "F1"):
            xs, ys, aids = buckets[lvl]
            color = _level_colors[lvl]
            count = len(xs)
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.85,
                    line=dict(width=0),
                ),
                name=f"{lvl} ({count})",
                legendgroup=f"agents_{lvl}",
                showlegend=True,
                hovertext=[f"{a}  {lvl}" for a in aids],
                hoverinfo="text",
            ))
        return traces

    # ---------------------------------------------------- initial agent traces
    init_agent_traces = _agent_traces(keep_ts[0])

    # ------------------------------------------------------ animation frames
    frames = [
        go.Frame(
            data=_agent_traces(t),
            traces=agent_trace_indices,
            name=str(t),
        )
        for t in keep_ts
    ]

    # ----------------------------------------------------------------- figure
    fig = go.Figure(
        data=bg_traces + init_agent_traces,
        frames=frames,
    )

    # Slider steps
    slider_steps = [
        {
            "args": [
                [str(t)],
                {"frame": {"duration": 120, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 60}},
            ],
            "label": f"{t:.0f}s",
            "method": "animate",
        }
        for t in keep_ts
    ]

    n_frames = len(keep_ts)
    total_t = keep_ts[-1] if keep_ts else 0

    fig.update_layout(
        title=dict(
            text=(
                f"ABM Agent Animation — {label.title()} Routing<br>"
                "<sub>"
                "🔵 F1 Platform · "
                "🟢 F3 Concourse · "
                "🟠 F4 Transport Hub  |  "
                f"200 agents · T={total_t:.0f} s · dt={dt_frame:.0f} s/frame"
                "</sub>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(title="X (m)", scaleanchor="y", scaleratio=1,
                   showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#e8e8e8"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        width=1400,
        height=750,
        margin=dict(l=60, r=60, t=110, b=130),
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=12),
            title=dict(text="Layer (click to toggle)", font=dict(size=12)),
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 120, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 60}}],
                    "label": "▶  Play",
                    "method": "animate",
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "⏸  Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.05,
            "y": 0.0,
            "xanchor": "right",
            "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": slider_steps,
            "x": 0.07,
            "len": 0.9,
            "y": 0.0,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "Time: ",
                "suffix": " s",
                "visible": True,
                "xanchor": "center",
                "font": {"size": 14, "color": "#333"},
            },
            "transition": {"duration": 60},
            "pad": {"b": 10, "t": 50},
        }],
    )

    out = out_dir_p / f"interactive_sim_{label}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  [html] {out.name}  ({n_frames} frames, {total_t:.0f}s)")
    return out


# =========================================================================
# 7. Interactive 3-D axonometric ABM simulation animation (rotatable)
# =========================================================================

def fig_3d_simulation_animation(
    traj_path: str | Path,
    geometries: dict,
    elevations: dict,
    all_connectors: list[dict],
    cfg: dict,
    out_dir: str | Path,
    *,
    G=None,
    dt_frame: float = 2.0,
    label: str = "static",
    z_scale: float = 3.0,
) -> Path:
    """Interactive 3-D axonometric animated ABM simulation.

    Three floor planes shown at their scaled elevations (×z_scale) with
    stair / escalator / elevator connector shapes bridging the levels.
    Each floor's plan outlines and key elements (stair openings, elevator
    shafts, entrances, PSD gates) are visible as markers on the floor surface.

    Agents are animated as 3-D dots at their real (x, y, z×z_scale) positions.
    The scene can be rotated freely with the mouse; use the time slider or
    Play/Pause buttons to control the animation.

    Parameters
    ----------
    traj_path      : path to ``traj_agents.jsonl``
    geometries     : per-level geometry dicts (need ``"floor"`` polygon)
    elevations     : ``{level: elevation_m}`` mapping
    all_connectors : parsed connector list from ``extract_all_levels``
    cfg            : full experiment config dict
    out_dir        : output directory
    G              : (optional) navigation graph for entrance/PSD markers
    dt_frame       : seconds between animation frames (downsampling)
    label          : scenario tag used in filename & title
    z_scale        : vertical exaggeration factor

    Returns
    -------
    Path to the written ``.html`` file.
    """
    import json
    from collections import defaultdict

    traj_path  = Path(traj_path)
    out_dir_p  = _ensure_dir(out_dir)

    # ---- Load trajectory -------------------------------------------------
    raw: dict[float, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            t_key = round(float(rec["t"]), 2)
            raw[t_key][rec["agent_id"]] = (
                float(rec["x"]), float(rec["y"]), float(rec["z"])
            )

    if not raw:
        out = out_dir_p / f"interactive_3d_sim_{label}.html"
        out.write_text("<p>No trajectory data found.</p>")
        return out

    # ---- Downsample time steps -------------------------------------------
    sorted_ts = sorted(raw.keys())
    keep_ts: list[float] = []
    next_keep = sorted_ts[0]
    for t in sorted_ts:
        if t >= next_keep - 1e-6:
            keep_ts.append(t)
            next_keep = t + dt_frame

    # ---- Floor label lookup from z-coordinate ----------------------------
    def _z_to_level(z: float) -> str:
        best, best_d = "F1", 1e9
        for lvl, elev in elevations.items():
            d = abs(elev - z)
            if d < best_d:
                best_d, best = d, lvl
        return best

    stair_dirs = _precompute_stair_dirs(all_connectors)
    levels = sorted(elevations.keys(), key=lambda k: elevations[k])

    _lv_colors = {"F1": "#2196F3", "F3": "#4CAF50", "F4": "#FF9800"}
    _lv_fill   = {
        "F1": "rgba(33,150,243,0.10)",
        "F3": "rgba(76,175,80,0.10)",
        "F4": "rgba(255,152,0,0.10)",
    }

    bg_traces: list = []

    # =========================================================
    # LAYER 1 — Floor outlines + semi-transparent slab meshes
    # =========================================================
    for lvl in levels:
        z_lv = elevations[lvl] * z_scale
        geo  = geometries.get(lvl, {})
        floor = geo.get("floor")
        if not floor or floor.is_empty:
            continue
        color = _lv_colors.get(lvl, "#999")
        fill  = _lv_fill.get(lvl, "rgba(150,150,150,0.08)")
        first = True
        for poly in flatten_polygons(floor):
            bg_traces.append(_polygon_boundary_trace(
                poly, z_lv, color, lvl,
                legendgroup=f"floor_{lvl}", showlegend=first, width=2.5))
            xs, ys = poly.exterior.xy
            verts = list(zip(list(xs), list(ys), [z_lv] * len(xs)))
            bg_traces.append(_mesh_from_quad(
                verts, fill, 0.12,
                f"{lvl} slab", legendgroup=f"floor_{lvl}",
                htext=f"{lvl}  elev={elevations[lvl]:.1f}m"))
            first = False

    # =========================================================
    # LAYER 2 — Element markers ON each floor plane
    # (stair openings ■, escalator landings ◆, elevator shafts ●)
    # =========================================================
    stair_pts:  dict[str, tuple] = defaultdict(lambda: ([], [], []))
    esc_pts:    dict[str, tuple] = defaultdict(lambda: ([], [], []))
    elev_pts:   dict[str, tuple] = defaultdict(lambda: ([], [], []))

    # Stair chain openings (bottom and top of each run)
    for c in all_connectors:
        if c["type"] != "stair_chain":
            continue
        cid = c.get("id", "stair")
        for run in c.get("runs", []):
            cx = (run["min_x"] + run["max_x"]) / 2
            cy = (run["min_y"] + run["max_y"]) / 2
            for z_open, label_txt in [(run["z_min"], "bottom"), (run["z_max"], "top")]:
                lvl_k = _z_to_level(z_open)
                stair_pts[lvl_k][0].append(cx)
                stair_pts[lvl_k][1].append(cy)
                stair_pts[lvl_k][2].append(
                    f"Stair {cid} {label_txt} opening<br>"
                    f"z={z_open:.1f}m  ({lvl_k})")

    # Escalator bottom / top landings
    for c in all_connectors:
        if c["type"] != "escalator":
            continue
        cid = c.get("id", "esc")
        for lv_key, xy_key in [("bottom_level", "bottom_xy"),
                                ("top_level",    "top_xy")]:
            lv  = c.get(lv_key, "")
            bxy = c.get(xy_key)
            if lv in elevations and bxy:
                esc_pts[lv][0].append(bxy[0])
                esc_pts[lv][1].append(bxy[1])
                esc_pts[lv][2].append(f"Escalator {cid}<br>{lv_key.split('_')[0]} ({lv})")

    # Elevator shaft footprints per served level
    for c in all_connectors:
        if c["type"] != "elevator":
            continue
        cid = c.get("id", "elev")
        fp  = c.get("footprint")
        if not (fp and not fp.is_empty):
            continue
        cx, cy = fp.centroid.x, fp.centroid.y
        for lk in c.get("connected_levels", []):
            if lk in elevations:
                elev_pts[lk][0].append(cx)
                elev_pts[lk][1].append(cy)
                elev_pts[lk][2].append(f"Elevator {cid} @ {lk}")

    # Assemble combined marker traces (one trace per element type for legend clarity)
    def _collect_markers(pts_dict):
        xs, ys, zs, txts = [], [], [], []
        for lvl, (lxs, lys, ltxts) in pts_dict.items():
            z_lv = elevations.get(lvl, 0) * z_scale + 0.25  # hover above slab
            xs.extend(lxs); ys.extend(lys)
            zs.extend([z_lv] * len(lxs))
            txts.extend(ltxts)
        return xs, ys, zs, txts

    sx, sy, sz, sh = _collect_markers(stair_pts)
    if sx:
        bg_traces.append(go.Scatter3d(
            x=sx, y=sy, z=sz, mode="markers",
            marker=dict(size=10, color=CONN_HEX["stair"], symbol="square",
                        opacity=0.90, line=dict(color="white", width=1.2)),
            name="Stair openings", legendgroup="elem_stair",
            hovertext=sh, hoverinfo="text",
        ))

    ex, ey, ez_e, eh = _collect_markers(esc_pts)
    if ex:
        bg_traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez_e, mode="markers",
            marker=dict(size=10, color=CONN_HEX["escalator"], symbol="diamond",
                        opacity=0.90, line=dict(color="white", width=1.2)),
            name="Escalator landings", legendgroup="elem_esc",
            hovertext=eh, hoverinfo="text",
        ))

    lvx, lvy, lvz, lvh = _collect_markers(elev_pts)
    if lvx:
        bg_traces.append(go.Scatter3d(
            x=lvx, y=lvy, z=lvz, mode="markers",
            marker=dict(size=13, color=CONN_HEX["elevator"], symbol="circle",
                        opacity=0.90, line=dict(color="white", width=1.5)),
            name="Elevator shafts", legendgroup="elem_elev",
            hovertext=lvh, hoverinfo="text",
        ))

    # =========================================================
    # LAYER 3 — 3-D connector shapes bridging the levels
    # =========================================================
    stair_dz_p = cfg.get("connectors", {}).get("stair", {}).get("dz_step_m", 0.18)
    s3d_first  = True
    for c in all_connectors:
        if c["type"] != "stair_chain":
            continue
        runs = c.get("runs", [])
        if not runs:
            continue
        cid   = c.get("id", "stair")
        color = CONN_HEX["stair"]
        first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
        last_xc  = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
        asc_x    = first_xc < last_xc
        for ri, run in enumerate(runs):
            for tr in _stepped_quads_3d(
                run["min_x"], run["min_y"], run["max_x"], run["max_y"],
                run["z_min"] * z_scale, run["z_max"] * z_scale,
                asc_x, stair_dz_p * z_scale,
                color, 0.70, "Stair", "conn_stair",
                connector_id=cid, run_index=ri,
                showlegend=(s3d_first and ri == 0),
            ):
                bg_traces.append(tr)
        hw = (runs[0]["max_y"] - runs[0]["min_y"]) / 2
        for ld in c.get("landings", []):
            lx, ly, lz = ld["x"], ld["y"], ld["z"] * z_scale
            verts = [(lx - 1.5, ly - hw, lz), (lx - 1.5, ly + hw, lz),
                     (lx + 1.5, ly + hw, lz), (lx + 1.5, ly - hw, lz)]
            bg_traces.append(_mesh_from_quad(
                verts, color, 0.50, "Stair landing", legendgroup="conn_stair",
                htext=f"{cid} landing z={ld['z']:.1f}"))
        s3d_first = False

    esc_dz_p  = cfg.get("connectors", {}).get("escalator", {}).get("dz_step_m", 0.40)
    e3d_first = True
    for c in all_connectors:
        if c["type"] != "escalator":
            continue
        color   = CONN_HEX["escalator"]
        cid     = c.get("id", "esc")
        bot_lv  = c.get("bottom_level")
        top_lv  = c.get("top_level")
        if not (bot_lv and top_lv):
            continue
        if bot_lv not in elevations or top_lv not in elevations:
            continue
        zlo     = elevations[bot_lv] * z_scale
        zhi     = elevations[top_lv] * z_scale
        bot_xy  = c.get("bottom_xy")
        top_xy  = c.get("top_xy")
        if bot_xy and top_xy:
            asc_x = bot_xy[0] < top_xy[0]
            x_lo  = min(bot_xy[0], top_xy[0])
            x_hi  = max(bot_xy[0], top_xy[0])
        else:
            x_lo  = c.get("min_x", 0)
            x_hi  = c.get("max_x", 0)
            asc_x = _escalator_asc_x_from_dirs(c, stair_dirs)
        y0_e = c.get("min_y", 0)
        y1_e = c.get("max_y", 0)
        if x_hi - x_lo < 0.1:
            continue
        for tr in _stepped_quads_3d(
            x_lo, y0_e, x_hi, y1_e, zlo, zhi, asc_x, esc_dz_p * z_scale,
            color, 0.65, "Escalator", "conn_escalator",
            connector_id=cid, showlegend=e3d_first,
        ):
            bg_traces.append(tr)
        e3d_first = False

    lv3d_first = True
    for c in all_connectors:
        if c["type"] != "elevator":
            continue
        color  = CONN_HEX["elevator"]
        cid    = c.get("id", "elev")
        served = [lk for lk in c.get("connected_levels", []) if lk in elevations]
        if len(served) < 2:
            continue
        zs_e   = sorted([elevations[lk] * z_scale for lk in served])
        fp     = c.get("footprint")
        if not (fp and not fp.is_empty):
            continue
        x0e, y0e, x1e, y1e = fp.bounds
        bg_traces.append(_box_wireframe(
            (x0e, y0e, zs_e[0]), (x1e, y1e, zs_e[-1]),
            color, "Elevator shaft", legendgroup="conn_elevator",
            showlegend=lv3d_first, width=3))
        for lk in served:
            ze = elevations[lk] * z_scale
            verts = [(x0e, y0e, ze), (x1e, y0e, ze),
                     (x1e, y1e, ze), (x0e, y1e, ze)]
            bg_traces.append(_mesh_from_quad(
                verts, color, 0.50, f"Elev {lk}",
                legendgroup="conn_elevator", htext=f"{cid} @ {lk}"))
        lv3d_first = False

    # =========================================================
    # LAYER 4 — Entrance gates & Platform Screen Doors
    # =========================================================
    if G is not None:
        ent_nodes = [(n, d) for n, d in G.nodes(data=True)
                     if d.get("node_type") == "entrance"]
        if ent_nodes:
            bg_traces.append(go.Scatter3d(
                x=[d["x"] for _, d in ent_nodes],
                y=[d["y"] for _, d in ent_nodes],
                z=[elevations.get(d.get("level"), 0) * z_scale
                   for _, d in ent_nodes],
                mode="markers+text",
                marker=dict(size=12, color="#00BCD4", symbol="diamond",
                            opacity=1.0, line=dict(color="white", width=2.0)),
                text=[
                    d.get("entrance_name", "").replace("entrance_", "Gate ")
                    if d.get("entrance_primary", False) else ""
                    for _, d in ent_nodes
                ],
                textposition="top center",
                textfont=dict(size=11, color="#006064"),
                name="Entrance Gates",
                legendgroup="ent_layer",
                hoverinfo="text",
                hovertext=[
                    f"{d.get('entrance_name','')} @ {d.get('level','')}"
                    for _, d in ent_nodes
                ],
            ))

        psd_nodes = [(n, d) for n, d in G.nodes(data=True)
                     if d.get("node_type") == "door_platform"
                     and d.get("level") == "F1"]
        if psd_nodes:
            bg_traces.append(go.Scatter3d(
                x=[d["x"] for _, d in psd_nodes],
                y=[d["y"] for _, d in psd_nodes],
                z=[elevations.get("F1", 0) * z_scale] * len(psd_nodes),
                mode="markers",
                marker=dict(size=5, color="#FF6F00", symbol="square",
                            opacity=0.85, line=dict(color="white", width=0.5)),
                name=f"Platform Screen Doors ({len(psd_nodes)})",
                legendgroup="psd_layer",
                hoverinfo="text",
                hovertext=[f"PSD: {nid} @ F1" for nid, _ in psd_nodes],
            ))

    # =========================================================
    # ANIMATED TRACES — 3 Scatter3d traces (F4 / F3 / F1 agents)
    # positioned at indices:  n_bg+0, n_bg+1, n_bg+2
    # =========================================================
    n_bg       = len(bg_traces)
    agent_idx  = [n_bg, n_bg + 1, n_bg + 2]
    _floor_ord = ("F4", "F3", "F1")

    def _agent_scatter3d(t_key: float) -> list:
        frame_data = raw.get(t_key, {})
        buckets: dict[str, list] = {lv: ([], [], [], []) for lv in _floor_ord}
        for aid, (x, y, z) in frame_data.items():
            lvl = _z_to_level(z)
            if lvl not in buckets:
                lvl = "F1"
            ax, ay, az, aa = buckets[lvl]
            ax.append(x); ay.append(y)
            az.append(z * z_scale)   # real z → smooth stair transitions
            aa.append(aid)
        traces = []
        for lv in _floor_ord:
            xs, ys, zs, aids = buckets[lv]
            color = _lv_colors.get(lv, "#999")
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=4.5,
                    color=color,
                    opacity=0.92,
                    line=dict(color="rgba(255,255,255,0.4)", width=0.5),
                ),
                name=f"Agents {lv} ({len(xs)})",
                legendgroup=f"agents_{lv}",
                showlegend=True,
                hovertext=[f"{a} @ {lv}" for a in aids],
                hoverinfo="text",
            ))
        return traces

    init_agents = _agent_scatter3d(keep_ts[0])

    frames = [
        go.Frame(
            data=_agent_scatter3d(t),
            traces=agent_idx,
            name=str(t),
        )
        for t in keep_ts
    ]

    # =========================================================
    # LAYOUT
    # =========================================================
    fig = go.Figure(data=bg_traces + init_agents, frames=frames)

    slider_steps = [
        {
            "args": [
                [str(t)],
                {"frame": {"duration": 100, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 50}},
            ],
            "label": f"{t:.0f}s",
            "method": "animate",
        }
        for t in keep_ts
    ]

    n_frames = len(keep_ts)
    total_t  = keep_ts[-1] if keep_ts else 0

    fig.update_layout(
        title=dict(
            text=(
                f"3D Axonometric ABM Simulation — {label.title()} Routing<br>"
                "<sub>"
                "🟠 F4 Transport Hub  ·  🟢 F3 Concourse  ·  🔵 F1 Platform  |  "
                "■ Stair opening  ◆ Escalator landing  ● Elevator shaft  ◆ Gate  |  "
                f"200 agents · T={total_t:.0f}s · drag to rotate · scroll to zoom"
                "</sub>"
            ),
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (×{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.80),
            camera=dict(
                eye=dict(x=1.5, y=-1.8, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            xaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5),
            yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5),
            zaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5),
        ),
        width=1500,
        height=900,
        margin=dict(l=10, r=10, t=110, b=140),
        paper_bgcolor="#ffffff",
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=11),
            title=dict(text="Layer (click to toggle)", font=dict(size=11)),
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 50},
                    }],
                    "label": "▶  Play",
                    "method": "animate",
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    "label": "⏸  Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.05,
            "y": 0.0,
            "xanchor": "right",
            "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "steps": slider_steps,
            "x": 0.07,
            "len": 0.90,
            "y": 0.0,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "Time: ",
                "suffix": " s",
                "visible": True,
                "xanchor": "center",
                "font": {"size": 14, "color": "#333"},
            },
            "transition": {"duration": 50},
            "pad": {"b": 10, "t": 50},
        }],
    )

    out = out_dir_p / f"interactive_3d_sim_{label}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  [3d-html] {out.name}  ({n_frames} frames, {total_t:.0f}s)")
    return out


# =========================================================================
# 8. Interactive 3-D route flow difference map (dynamic − static)
# =========================================================================

def fig_interactive_route_diff(
    G: nx.DiGraph,
    static_result: dict,
    dynamic_result: dict,
    geometries: dict,
    elevations: dict,
    all_connectors: list[dict],
    cfg: dict,
    out_dir: str | Path,
    *,
    z_scale: float = 3.0,
    min_abs_diff: int = 3,
) -> Path:
    """Interactive 3-D visualisation of edge flow differences between routing modes.

    Edges are coloured by ``Δ = dynamic_throughput − static_throughput``:
      🔴 Red   — edge carried *more* pedestrians under dynamic (congestion-aware) routing
      🔵 Blue  — edge carried *more* pedestrians under static routing
      ⬜ Grey  — negligible difference (|Δ| < ``min_abs_diff``)

    Floor outlines and connector shapes (stairs / escalators / elevators) are
    shown as static background context on the three separated floor planes.

    Interactive controls:
      * Drag to rotate  ·  Scroll to zoom  ·  Click legend to toggle layers
      * Dropdown: All edges / Increased in Dynamic / Decreased in Dynamic

    Parameters
    ----------
    G               : navigation graph
    static_result   : ``result_static.json`` dict (must contain ``edge_throughput``)
    dynamic_result  : ``result_dynamic.json`` dict
    geometries      : per-level geometry dicts
    elevations      : ``{level: elevation_m}``
    all_connectors  : parsed connector list
    cfg             : experiment config dict
    out_dir         : output directory
    z_scale         : vertical exaggeration
    min_abs_diff    : hide edges with |Δ| below this count

    Returns
    -------
    Path to ``interactive_route_diff.html``
    """
    out_dir_p = _ensure_dir(out_dir)

    et_s = static_result.get("edge_throughput", {})
    et_d = dynamic_result.get("edge_throughput", {})
    replan_evts = dynamic_result.get("replan_events", [])
    n_agents    = cfg.get("simulation", {}).get("n_agents", 200)
    t_s         = cfg.get("simulation", {}).get("T_s", 600)

    all_edge_keys = set(et_s) | set(et_d)
    diff: dict[str, int] = {
        k: et_d.get(k, 0) - et_s.get(k, 0) for k in all_edge_keys
    }

    # Node position & level lookup
    def _node_xyz(nid):
        if nid not in G:
            return None, None, None
        d = G.nodes[nid]
        lvl = d.get("level", "")
        z = (d.get("z", 0) * z_scale if lvl in ("STAIR", "ESCALATOR")
             else elevations.get(lvl, 0) * z_scale)
        return d.get("x"), d.get("y"), z

    stair_dirs = _precompute_stair_dirs(all_connectors)
    levels = sorted(elevations.keys(), key=lambda k: elevations[k])

    # ---- diverging colour bins ----
    # Map diff value → hex colour using 5-class diverging scale
    _RED5   = ["#B71C1C", "#E57373", "#FFCDD2", "#EF9A9A", "#EF5350"]
    _BLUE5  = ["#0D47A1", "#64B5F6", "#BBDEFB", "#90CAF9", "#42A5F5"]
    _GREY   = "#BDBDBD"

    all_vals = [v for v in diff.values() if abs(v) >= min_abs_diff]
    if all_vals:
        vmax = max(abs(v) for v in all_vals)
    else:
        vmax = 1

    def _diff_color(dv: int) -> str:
        if abs(dv) < min_abs_diff:
            return _GREY
        frac = min(1.0, abs(dv) / vmax)
        idx  = min(4, int(frac * 5))
        return _RED5[idx] if dv > 0 else _BLUE5[idx]

    # =====================================================
    # Background — floor outlines + connector shapes
    # =====================================================
    fig = go.Figure()

    # Floor outlines
    for lvl in levels:
        z_lv = elevations[lvl] * z_scale
        geo  = geometries.get(lvl, {})
        floor = geo.get("floor")
        if not floor or floor.is_empty:
            continue
        color = LEVEL_HEX.get(lvl, "#999")
        first = True
        for poly in flatten_polygons(floor):
            fig.add_trace(_polygon_boundary_trace(
                poly, z_lv, color, lvl,
                legendgroup=f"floor_{lvl}", showlegend=first, width=2.0))
            xs, ys = poly.exterior.xy
            verts = list(zip(list(xs), list(ys), [z_lv] * len(xs)))
            fig.add_trace(_mesh_from_quad(
                verts, LEVEL_RGBA.get(lvl, "rgba(150,150,150,{a})").format(a=0.08),
                0.10, f"{lvl} slab", legendgroup=f"floor_{lvl}",
                htext=f"{lvl}  z={elevations[lvl]:.1f}m"))
            first = False

    # Connector shapes (stairs / escalators / elevators) — same as 3D animation
    stair_dz_p  = cfg.get("connectors", {}).get("stair",     {}).get("dz_step_m", 0.18)
    esc_dz_p    = cfg.get("connectors", {}).get("escalator", {}).get("dz_step_m", 0.40)
    s3_first, e3_first, l3_first = True, True, True

    for c in all_connectors:
        if c["type"] == "stair_chain":
            runs = c.get("runs", []); cid = c.get("id", "stair")
            if not runs: continue
            color = CONN_HEX["stair"]
            first_xc = (runs[0]["min_x"] + runs[0]["max_x"]) / 2
            last_xc  = (runs[-1]["min_x"] + runs[-1]["max_x"]) / 2
            asc_x    = first_xc < last_xc
            for ri, run in enumerate(runs):
                for tr in _stepped_quads_3d(
                    run["min_x"], run["min_y"], run["max_x"], run["max_y"],
                    run["z_min"] * z_scale, run["z_max"] * z_scale,
                    asc_x, stair_dz_p * z_scale,
                    color, 0.60, "Stair", "conn_stair",
                    connector_id=cid, run_index=ri,
                    showlegend=(s3_first and ri == 0),
                ):
                    fig.add_trace(tr)
            s3_first = False

        elif c["type"] == "escalator":
            cid = c.get("id", "esc"); color = CONN_HEX["escalator"]
            bl, tl = c.get("bottom_level"), c.get("top_level")
            if not (bl and tl and bl in elevations and tl in elevations): continue
            zlo, zhi = elevations[bl] * z_scale, elevations[tl] * z_scale
            bxy, txy = c.get("bottom_xy"), c.get("top_xy")
            if bxy and txy:
                asc_x = bxy[0] < txy[0]; x_lo = min(bxy[0], txy[0]); x_hi = max(bxy[0], txy[0])
            else:
                x_lo = c.get("min_x", 0); x_hi = c.get("max_x", 0)
                asc_x = _escalator_asc_x_from_dirs(c, stair_dirs)
            y0e, y1e = c.get("min_y", 0), c.get("max_y", 0)
            if x_hi - x_lo < 0.1: continue
            for tr in _stepped_quads_3d(
                x_lo, y0e, x_hi, y1e, zlo, zhi, asc_x, esc_dz_p * z_scale,
                color, 0.60, "Escalator", "conn_escalator",
                connector_id=cid, showlegend=e3_first,
            ):
                fig.add_trace(tr)
            e3_first = False

        elif c["type"] == "elevator":
            cid = c.get("id", "elev"); color = CONN_HEX["elevator"]
            served = [lk for lk in c.get("connected_levels", []) if lk in elevations]
            if len(served) < 2: continue
            zs_e = sorted([elevations[lk] * z_scale for lk in served])
            fp   = c.get("footprint")
            if not (fp and not fp.is_empty): continue
            x0e, y0e, x1e, y1e = fp.bounds
            fig.add_trace(_box_wireframe(
                (x0e, y0e, zs_e[0]), (x1e, y1e, zs_e[-1]),
                color, "Elevator shaft", legendgroup="conn_elevator",
                showlegend=l3_first, width=3))
            l3_first = False

    # =====================================================
    # Edge flow difference traces — grouped by bucket
    # =====================================================
    # 5 buckets: strong_inc, weak_inc, neutral, weak_dec, strong_dec
    threshold_strong = max(1, int(vmax * 0.5))

    buckets = {
        "strong_inc":  {"label": f"Δ > +{threshold_strong} (dynamic)", "color": "#B71C1C", "w": 3.5},
        "weak_inc":    {"label": f"Δ +1…+{threshold_strong}",           "color": "#EF5350", "w": 2.0},
        "neutral":     {"label": f"|Δ| < {min_abs_diff} (same)",        "color": "#BDBDBD", "w": 0.6},
        "weak_dec":    {"label": f"Δ -{threshold_strong}…-1",           "color": "#42A5F5", "w": 2.0},
        "strong_dec":  {"label": f"Δ < -{threshold_strong} (static)",   "color": "#0D47A1", "w": 3.5},
    }

    edge_segs: dict[str, tuple] = {k: ([], [], [], []) for k in buckets}

    for edge_key, dv in diff.items():
        u, v = edge_key.split("|", 1)
        x0, y0, z0 = _node_xyz(u)
        x1, y1, z1 = _node_xyz(v)
        if x0 is None or x1 is None:
            continue
        if dv > threshold_strong:
            bk = "strong_inc"
        elif dv >= min_abs_diff:
            bk = "weak_inc"
        elif dv <= -threshold_strong:
            bk = "strong_dec"
        elif dv <= -min_abs_diff:
            bk = "weak_dec"
        else:
            bk = "neutral"
        xs, ys, zs, htxts = edge_segs[bk]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]
        htxts += [f"{edge_key}<br>Δ={dv:+d} (dyn={et_d.get(edge_key,0)}, sta={et_s.get(edge_key,0)})", None, None]

    inc_traces, dec_traces, neu_traces = [], [], []
    for bk, (xs, ys, zs, htxts) in edge_segs.items():
        if not xs:
            continue
        info = buckets[bk]
        visible = bk != "neutral"   # hide neutral by default to reduce clutter
        idx = len(fig.data)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=info["color"], width=info["w"]),
            opacity=0.80 if bk != "neutral" else 0.25,
            name=info["label"],
            legendgroup=f"diff_{bk}",
            hoverinfo="text",
            hovertext=htxts,
            visible=visible,
        ))
        if "inc" in bk:
            inc_traces.append(idx)
        elif "dec" in bk:
            dec_traces.append(idx)
        else:
            neu_traces.append(idx)

    # =====================================================
    # Replan event summary annotation (text)
    # =====================================================
    unique_replanners = len({e["agent_id"] for e in replan_evts})
    summary_txt = (
        f"Dynamic − Static routing comparison<br>"
        f"Agents that replanned: {unique_replanners}/{n_agents} ({unique_replanners/n_agents:.0%})<br>"
        f"Total replan events: {len(replan_evts):,} over T={t_s}s"
    )

    # =====================================================
    # Dropdown: All / Increased / Decreased / Neutral
    # =====================================================
    n_total = len(fig.data)
    all_vis  = [True] * n_total
    inc_vis  = [True if i in set(inc_traces) else False for i in range(n_total)]
    dec_vis  = [True if i in set(dec_traces) else False for i in range(n_total)]

    fig.update_layout(
        title=dict(
            text=(
                "Route Flow Difference: Dynamic − Static Routing<br>"
                "<sub>"
                "🔴 Red = more traffic in dynamic (replanned routes)  ·  "
                "🔵 Blue = more traffic in static  ·  "
                "Drag to rotate · Scroll to zoom"
                "</sub>"
            ),
            font=dict(size=14),
        ),
        annotations=[dict(
            text=summary_txt,
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            showarrow=False,
            font=dict(size=11, color="#333"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            align="left",
        )],
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.01, y=1.10,
            bgcolor="white",
            bordercolor="#aaa",
            font=dict(size=12),
            buttons=[
                dict(label="All Changed Edges",
                     method="update",
                     args=[{"visible": all_vis}]),
                dict(label="▲ More in Dynamic (Red)",
                     method="update",
                     args=[{"visible": inc_vis}]),
                dict(label="▼ More in Static (Blue)",
                     method="update",
                     args=[{"visible": dec_vis}]),
            ],
            showactive=True,
            active=0,
        )],
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title=f"Elevation (×{z_scale:.0f})",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.5, z=0.80),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
        ),
        width=1500, height=900,
        margin=dict(l=10, r=10, t=120, b=10),
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=11),
            title=dict(text="Flow difference (click to toggle)", font=dict(size=11)),
        ),
    )

    out = out_dir_p / "interactive_route_diff.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    n_changed = sum(1 for v in diff.values() if abs(v) >= min_abs_diff)
    print(f"  [html] interactive_route_diff.html  ({n_changed} changed edges)")
    return out
