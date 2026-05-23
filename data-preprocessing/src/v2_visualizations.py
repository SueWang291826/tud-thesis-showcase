"""
V2 Visualization Module.

Generates presentation-ready visualizations for the v2 experimental
subset pipeline:

1. Before/after proxy disambiguation
2. Before/after cross-storey filtering
3. Retained vs dropped element composition
4. Traffic-relevant connector distribution
5. Public-layer object composition after filtering
6. Spatial previews of final retained subsets (F1, F3)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent colour scheme
DECISION_COLORS = {
    "keep_traffic_relevant": "#27ae60",
    "keep_barrier_relevant": "#2ecc71",
    "drop_traffic_irrelevant": "#e74c3c",
    "uncertain_low_priority": "#f39c12",
    "uncertain_high_priority": "#e67e22",
}

FILTER_COLORS = {
    "retained_walkable": "#2ecc71",
    "retained_connector": "#3498db",
    "retained_barrier": "#9b59b6",
    "dropped_technical": "#e74c3c",
    "dropped_overhead": "#c0392b",
    "dropped_non_public": "#95a5a6",
    "dropped_uncertain": "#f39c12",
}

FILE_COLORS = {
    "platform": "#3498db",
    "equipment": "#e74c3c",
    "concourse": "#2ecc71",
}


def _setup_style():
    """Apply CJK-compatible Matplotlib style."""
    import matplotlib.font_manager as fm
    cjk_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "STSong",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    found_font = None
    available = {f.name for f in fm.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available:
            found_font = font
            break

    base = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
    if found_font:
        base["font.sans-serif"] = [found_font, "DejaVu Sans"]
        base["axes.unicode_minus"] = False
    plt.rcParams.update(base)


def _save_fig(fig: plt.Figure, path: Path, save_svg: bool = True):
    """Save figure in PNG and optionally SVG."""
    fig.savefig(path, bbox_inches="tight")
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 1. Proxy disambiguation: before vs after
# ==============================================================================

def plot_proxy_before_after(
    proxy_resolved: pd.DataFrame,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Side-by-side comparison of v1 uncertain proxies vs v2 resolved."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(proxy_resolved) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before: v1 inferred categories
    v1_dist = proxy_resolved["v1_inferred_category"].value_counts()
    ax = axes[0]
    v1_colors = ["#e67e22" if c == "uncertain" else "#3498db" for c in v1_dist.index]
    v1_dist.plot(kind="bar", ax=ax, color=v1_colors, edgecolor="white")
    ax.set_title("Before (v1): Proxy Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)

    # After: v2 decisions
    v2_dist = proxy_resolved["v2_decision"].value_counts()
    ax = axes[1]
    v2_colors = [DECISION_COLORS.get(c, "#95a5a6") for c in v2_dist.index]
    v2_dist.plot(kind="bar", ax=ax, color=v2_colors, edgecolor="white")
    ax.set_title("After (v2): Proxy Decisions")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Proxy Disambiguation: Before vs After", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_fig(fig, output_dir / "proxy_before_after.png", save_svg)
    logger.info("  Saved: proxy_before_after.png")


def plot_proxy_subcategories(
    proxy_resolved: pd.DataFrame,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Horizontal bar chart of v2 proxy sub-categories."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(proxy_resolved) == 0:
        return

    sub_dist = proxy_resolved["v2_sub_category"].value_counts()
    sub_dist = sub_dist[sub_dist.index != ""]  # Remove empty

    if len(sub_dist) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub_dist) * 0.4)))
    colors = []
    for cat in sub_dist.index:
        dec = proxy_resolved[proxy_resolved["v2_sub_category"] == cat]["v2_decision"].mode()
        if len(dec) > 0:
            colors.append(DECISION_COLORS.get(dec.iloc[0], "#95a5a6"))
        else:
            colors.append("#95a5a6")

    sub_dist.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_ylabel("Sub-Category")
    ax.set_title("V2 Proxy Sub-Category Distribution")
    ax.invert_yaxis()

    _save_fig(fig, output_dir / "proxy_subcategories.png", save_svg)
    logger.info("  Saved: proxy_subcategories.png")


# ==============================================================================
# 2. Traffic filter: retained vs dropped
# ==============================================================================

def plot_filter_retained_dropped(
    all_filtered: pd.DataFrame,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Stacked bar: retained vs dropped by storey."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(all_filtered) == 0:
        return

    storeys = sorted(all_filtered["storey_name"].unique())
    retained_counts = []
    dropped_counts = []

    for s in storeys:
        sdf = all_filtered[all_filtered["storey_name"] == s]
        retained_counts.append(len(sdf[sdf["filter_decision"] == "keep"]))
        dropped_counts.append(len(sdf[sdf["filter_decision"] == "drop"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(storeys))
    ax.bar(x, retained_counts, label="Retained", color="#27ae60", edgecolor="white")
    ax.bar(x, dropped_counts, bottom=retained_counts, label="Dropped",
           color="#e74c3c", edgecolor="white")

    ax.set_xlabel("Storey")
    ax.set_ylabel("Element Count")
    ax.set_title("Traffic Filter: Retained vs Dropped by Storey")
    ax.set_xticks(x)
    ax.set_xticklabels(storeys, rotation=30, ha="right")
    ax.legend()

    # Add percentage labels
    for i, (r, d) in enumerate(zip(retained_counts, dropped_counts)):
        total = r + d
        if total > 0:
            ax.text(i, total + total * 0.01, f"{r/total:.0%}", ha="center", fontsize=9)

    _save_fig(fig, output_dir / "filter_retained_dropped.png", save_svg)
    logger.info("  Saved: filter_retained_dropped.png")


def plot_filter_output_classes(
    all_filtered: pd.DataFrame,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Horizontal bar of filter output classification counts."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(all_filtered) == 0:
        return

    dist = all_filtered["filter_output_class"].value_counts()
    colors = [FILTER_COLORS.get(c, "#95a5a6") for c in dist.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(dist) * 0.5)))
    dist.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_ylabel("Output Class")
    ax.set_title("Traffic Filter Output Classification")
    ax.invert_yaxis()

    _save_fig(fig, output_dir / "filter_output_classes.png", save_svg)
    logger.info("  Saved: filter_output_classes.png")


# ==============================================================================
# 3. Retained element composition on public levels
# ==============================================================================

def plot_public_level_composition(
    retained_df: pd.DataFrame,
    public_levels: List[str],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Stacked bar: semantic category composition per public level after filtering."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(retained_df) == 0:
        return

    from .visualization import COLORS as CAT_COLORS

    focus = retained_df[retained_df["storey_name"].isin(public_levels)]
    if len(focus) == 0:
        return

    categories = sorted(focus["category"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(public_levels))
    bottom = np.zeros(len(public_levels))

    for cat in categories:
        counts = [
            len(focus[(focus["storey_name"] == s) & (focus["category"] == cat)])
            for s in public_levels
        ]
        ax.bar(
            x, counts, bottom=bottom,
            label=cat, color=CAT_COLORS.get(cat, "#95a5a6"),
            edgecolor="white", linewidth=0.5,
        )
        bottom += np.array(counts)

    ax.set_xlabel("Public Level")
    ax.set_ylabel("Retained Element Count")
    ax.set_title("Post-Filter Public Level Composition")
    ax.set_xticks(x)
    ax.set_xticklabels(public_levels, rotation=15)
    ax.legend(loc="upper right", fontsize=8)

    _save_fig(fig, output_dir / "public_level_composition.png", save_svg)
    logger.info("  Saved: public_level_composition.png")


# ==============================================================================
# 4. Connector distribution
# ==============================================================================

def plot_connector_distribution(
    retained_df: pd.DataFrame,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Bar chart of retained vertical connector elements by storey and type."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    connectors = retained_df[
        retained_df["filter_output_class"] == "retained_connector"
    ]
    if len(connectors) == 0:
        logger.info("  No connectors to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By storey
    ax = axes[0]
    storey_dist = connectors["storey_name"].value_counts()
    storey_dist.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Storey")
    ax.set_ylabel("Count")
    ax.set_title("Retained Connectors by Storey")
    ax.tick_params(axis="x", rotation=30)

    # By IFC class
    ax = axes[1]
    class_dist = connectors["ifc_class"].value_counts().head(10)
    class_dist.plot(kind="bar", ax=ax, color="#2ecc71", edgecolor="white")
    ax.set_xlabel("IFC Class")
    ax.set_ylabel("Count")
    ax.set_title("Retained Connectors by IFC Class")
    ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Traffic-Relevant Connector Distribution", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_fig(fig, output_dir / "connector_distribution.png", save_svg)
    logger.info("  Saved: connector_distribution.png")


# ==============================================================================
# 5. Spatial previews of retained subsets
# ==============================================================================

def plot_retained_spatial_preview(
    retained_df: pd.DataFrame,
    bbox_dfs: Dict[str, pd.DataFrame],
    storey_name: str,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Plan-view bbox preview of retained elements on a specific storey.

    Uses bbox data from v1 geometry checks (already in metres).
    """
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    storey_retained = retained_df[retained_df["storey_name"] == storey_name]
    if len(storey_retained) == 0:
        return

    # Collect bbox data for retained elements
    retained_guids = set(storey_retained["guid"])

    all_bbox = pd.concat(bbox_dfs.values(), ignore_index=True) if bbox_dfs else pd.DataFrame()
    if len(all_bbox) == 0 or "min_x" not in all_bbox.columns:
        logger.info(f"  No bbox data for spatial preview of {storey_name}")
        return

    plot_data = all_bbox[all_bbox["guid"].isin(retained_guids)].copy()
    if len(plot_data) == 0:
        logger.info(f"  No bbox matches for retained elements on {storey_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Merge category info
    cat_map = dict(zip(storey_retained["guid"], storey_retained["category"]))

    from .visualization import COLORS as CAT_COLORS

    for _, row in plot_data.iterrows():
        if pd.isna(row.get("min_x")) or pd.isna(row.get("min_y")):
            continue

        cat = cat_map.get(row["guid"], "unknown")
        color = CAT_COLORS.get(cat, "#95a5a6")
        alpha = 0.6

        x = row["min_x"]
        y = row["min_y"]
        w = row.get("dx", row.get("max_x", x) - x)
        h = row.get("dy", row.get("max_y", y) - y)

        if w > 0 and h > 0:
            rect = plt.Rectangle((x, y), w, h,
                                 facecolor=color, edgecolor="black",
                                 linewidth=0.3, alpha=alpha)
            ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")

    safe_name = storey_name.replace(" ", "_")
    ax.set_title(f"Retained Subset Spatial Preview: {storey_name}")

    # Legend
    from matplotlib.patches import Patch
    legend_cats = sorted(set(cat_map.values()))
    legend_handles = [
        Patch(facecolor=CAT_COLORS.get(c, "#95a5a6"), label=c, alpha=0.6)
        for c in legend_cats
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    _save_fig(fig, output_dir / f"retained_spatial_{safe_name}.png", save_svg)
    logger.info(f"  Saved: retained_spatial_{safe_name}.png")


# ==============================================================================
# Master function
# ==============================================================================

def generate_v2_visualizations(
    proxy_resolved: pd.DataFrame,
    all_filtered: pd.DataFrame,
    retained_df: pd.DataFrame,
    bbox_dfs: Dict[str, pd.DataFrame],
    public_levels: List[str],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Generate all v2 visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Generating v2 visualizations")
    logger.info("=" * 60)

    # 1. Proxy before/after
    plot_proxy_before_after(proxy_resolved, output_dir, save_svg)
    plot_proxy_subcategories(proxy_resolved, output_dir, save_svg)

    # 2. Traffic filter results
    plot_filter_retained_dropped(all_filtered, output_dir, save_svg)
    plot_filter_output_classes(all_filtered, output_dir, save_svg)

    # 3. Public level composition
    plot_public_level_composition(retained_df, public_levels, output_dir, save_svg)

    # 4. Connector distribution
    plot_connector_distribution(retained_df, output_dir, save_svg)

    # 5. Spatial previews for public levels
    for level in public_levels:
        plot_retained_spatial_preview(
            retained_df, bbox_dfs, level, output_dir, save_svg
        )

    logger.info("All v2 visualizations generated.")
