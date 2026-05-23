"""
Visualization Module for IFC Preprocessing.

Generates publication-quality static figures for:
- IFC audit overview (entity counts, storey distributions)
- Storey mapping cross-file comparisons
- Semantic classification distributions
- Proxy analysis visualizations
- Geometry readiness summaries
- Lightweight spatial previews (plan-view bounding boxes)

All figures are saved to disk in PNG (and optionally SVG).
Designed for thesis progress meetings and supervisor presentations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- Shared styling ----
COLORS = {
    "walkable_support": "#2ecc71",
    "obstacle": "#e74c3c",
    "vertical_connector": "#3498db",
    "opening_passage": "#f39c12",
    "railing_barrier": "#9b59b6",
    "structural": "#7f8c8d",
    "ceiling_roof": "#1abc9c",
    "ignorable": "#bdc3c7",
    "uncertain": "#e67e22",
}

FILE_COLORS = {
    "platform": "#3498db",
    "equipment": "#e74c3c",
    "concourse": "#2ecc71",
}


def _setup_style():
    """Apply consistent Matplotlib style with CJK font support."""
    # Try to find a font that supports Chinese characters
    import matplotlib.font_manager as fm
    cjk_fonts = [
        "Microsoft YaHei",    # Windows
        "SimHei",             # Windows
        "SimSun",             # Windows
        "STSong",             # macOS
        "Noto Sans CJK SC",  # Linux
        "WenQuanYi Micro Hei",  # Linux
    ]
    found_font = None
    available = {f.name for f in fm.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available:
            found_font = font
            break

    base_config = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }

    if found_font:
        base_config["font.sans-serif"] = [found_font, "DejaVu Sans"]
        base_config["axes.unicode_minus"] = False
        logger.info(f"  Using CJK font: {found_font}")

    plt.rcParams.update(base_config)


def _save_fig(fig: plt.Figure, path: Path, save_svg: bool = True):
    """Save figure in PNG and optionally SVG."""
    fig.savefig(path, bbox_inches="tight")
    if save_svg:
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# A. IFC Audit Visualizations
# ==============================================================================

def plot_entity_counts_by_class(
    audits: Dict[str, Dict[str, Any]],
    output_dir: Path,
    top_n: int = 20,
    save_svg: bool = True,
) -> None:
    """Bar chart of top entity counts by IFC class, per file."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect top classes across all files
    all_classes = {}
    for audit in audits.values():
        for cls, count in audit["class_counts"].items():
            all_classes[cls] = all_classes.get(cls, 0) + count
    top_classes = sorted(all_classes, key=all_classes.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top_classes))
    width = 0.25
    labels = list(audits.keys())

    for i, (label, audit) in enumerate(audits.items()):
        counts = [audit["class_counts"].get(cls, 0) for cls in top_classes]
        ax.bar(
            x + i * width, counts, width,
            label=label, color=FILE_COLORS.get(label, f"C{i}"),
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("IFC Class")
    ax.set_ylabel("Count")
    ax.set_title("Entity Counts by IFC Class (Top {})".format(top_n))
    ax.set_xticks(x + width)
    ax.set_xticklabels(top_classes, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "entity_counts_by_class.png", save_svg)
    logger.info("  Saved: entity_counts_by_class.png")


def plot_entity_counts_by_storey(
    audits: Dict[str, Dict[str, Any]],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Bar chart of entity counts by storey, per file."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all storey names across files
    all_storeys = set()
    for audit in audits.values():
        for s in audit["storeys"]:
            if s["storey_name"] != "UNASSIGNED":
                all_storeys.add(s["storey_name"])
    storey_order = sorted(all_storeys)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(storey_order))
    width = 0.25
    labels = list(audits.keys())

    for i, (label, audit) in enumerate(audits.items()):
        counts = []
        for sname in storey_order:
            c = 0
            for s in audit["storeys"]:
                if s["storey_name"] == sname:
                    c = s["element_count"]
                    break
            counts.append(c)
        ax.bar(
            x + i * width, counts, width,
            label=label, color=FILE_COLORS.get(label, f"C{i}"),
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Storey")
    ax.set_ylabel("Element Count")
    ax.set_title("Element Distribution by Storey Across Files")
    ax.set_xticks(x + width)
    ax.set_xticklabels(storey_order, rotation=30, ha="right")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "entity_counts_by_storey.png", save_svg)
    logger.info("  Saved: entity_counts_by_storey.png")


def plot_proxy_proportion(
    audits: Dict[str, Dict[str, Any]],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Stacked bar showing proxy vs non-proxy proportion per file."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(audits.keys())
    proxy_counts = [audits[l]["proxy_count"] for l in labels]
    non_proxy = [audits[l]["total_products"] - audits[l]["proxy_count"] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, non_proxy, label="Non-Proxy", color="#3498db", edgecolor="white")
    ax.bar(x, proxy_counts, bottom=non_proxy, label="IfcBuildingElementProxy", color="#e74c3c", edgecolor="white")

    ax.set_xlabel("IFC File")
    ax.set_ylabel("Product Count")
    ax.set_title("Proxy vs Non-Proxy Elements per File")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add percentage labels
    for i, (p, np_) in enumerate(zip(proxy_counts, non_proxy)):
        total = p + np_
        if total > 0:
            ax.text(i, total + total * 0.01, f"{p/total:.0%} proxy", ha="center", fontsize=9)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "proxy_proportion.png", save_svg)
    logger.info("  Saved: proxy_proportion.png")


# ==============================================================================
# B. Storey Mapping Visualizations
# ==============================================================================

def plot_storey_file_heatmap(
    storey_mapping: Dict[str, Any],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Heatmap of element counts: storeys × files."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = storey_mapping.get("storey_file_matrix", {})
    if not matrix:
        logger.warning("  No storey-file matrix available for heatmap")
        return

    storeys = sorted(matrix.keys())
    files = sorted(set(k for row in matrix.values() for k in row.keys()))

    data = np.zeros((len(storeys), len(files)))
    for i, s in enumerate(storeys):
        for j, f in enumerate(files):
            data[i, j] = matrix[s].get(f, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=30, ha="right")
    ax.set_yticks(range(len(storeys)))
    ax.set_yticklabels(storeys)
    ax.set_xlabel("IFC File")
    ax.set_ylabel("Storey")
    ax.set_title("Element Count: Storey × File")

    # Add text annotations
    for i in range(len(storeys)):
        for j in range(len(files)):
            val = int(data[i, j])
            color = "white" if val > data.max() * 0.5 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Element Count")

    _save_fig(fig, output_dir / "storey_file_heatmap.png", save_svg)
    logger.info("  Saved: storey_file_heatmap.png")


def plot_storey_dominance(
    storey_mapping: Dict[str, Any],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Horizontal stacked bar: storey element composition across files."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = storey_mapping.get("storey_file_matrix", {})
    if not matrix:
        return

    storeys = sorted(matrix.keys())
    files = sorted(set(k for row in matrix.values() for k in row.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(storeys))

    left = np.zeros(len(storeys))
    for f_idx, fname in enumerate(files):
        vals = [matrix[s].get(fname, 0) for s in storeys]
        ax.barh(
            y, vals, left=left,
            label=fname, color=FILE_COLORS.get(fname, f"C{f_idx}"),
            edgecolor="white", linewidth=0.5,
        )
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(storeys)
    ax.set_xlabel("Element Count")
    ax.set_title("Storey Element Composition by Source File")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "storey_dominance.png", save_svg)
    logger.info("  Saved: storey_dominance.png")


# ==============================================================================
# C. Semantic Classification Visualizations
# ==============================================================================

def plot_semantic_category_distribution(
    classified_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Bar chart of semantic category distribution per file and combined."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [df.assign(file_label=label) for label, df in classified_dfs.items()],
        ignore_index=True,
    )

    categories = sorted(combined["category"].unique())
    labels = list(classified_dfs.keys())

    # Per-file grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.25

    for i, label in enumerate(labels):
        df = classified_dfs[label]
        counts = [len(df[df["category"] == c]) for c in categories]
        bars = ax.bar(
            x + i * width, counts, width,
            label=label, color=FILE_COLORS.get(label, f"C{i}"),
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Semantic Category")
    ax.set_ylabel("Element Count")
    ax.set_title("Semantic Classification Distribution by File")
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "semantic_category_distribution.png", save_svg)
    logger.info("  Saved: semantic_category_distribution.png")

    # Combined pie chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    cat_counts = combined["category"].value_counts()
    colors = [COLORS.get(c, "#95a5a6") for c in cat_counts.index]

    wedges, texts, autotexts = ax2.pie(
        cat_counts.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        pctdistance=0.8,
        startangle=90,
    )
    ax2.legend(
        wedges, [f"{c} ({v:,})" for c, v in zip(cat_counts.index, cat_counts.values)],
        title="Category",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
    )
    ax2.set_title("Combined Semantic Category Distribution")

    _save_fig(fig2, output_dir / "semantic_category_pie.png", save_svg)
    logger.info("  Saved: semantic_category_pie.png")


def plot_semantic_by_storey(
    classified_dfs: Dict[str, pd.DataFrame],
    storey_mapping: Dict[str, Any],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Stacked bar: semantic categories per storey for walkable levels."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [df.assign(file_label=label) for label, df in classified_dfs.items()],
        ignore_index=True,
    )

    walkable = storey_mapping.get("walkable_levels", [])
    if not walkable:
        # Use storeys with most elements
        storey_counts = combined["storey_name"].value_counts()
        walkable = storey_counts.head(3).index.tolist()

    focus_df = combined[combined["storey_name"].isin(walkable)]
    if len(focus_df) == 0:
        logger.warning("  No elements on walkable levels for storey-category plot")
        return

    categories = sorted(focus_df["category"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(walkable))
    bottom = np.zeros(len(walkable))

    for cat in categories:
        counts = [len(focus_df[(focus_df["storey_name"] == s) & (focus_df["category"] == cat)]) for s in walkable]
        ax.bar(
            x, counts, bottom=bottom,
            label=cat, color=COLORS.get(cat, "#95a5a6"),
            edgecolor="white", linewidth=0.5,
        )
        bottom += np.array(counts)

    ax.set_xlabel("Storey")
    ax.set_ylabel("Element Count")
    ax.set_title("Semantic Categories on Key Storeys")
    ax.set_xticks(x)
    ax.set_xticklabels(walkable, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "semantic_by_storey.png", save_svg)
    logger.info("  Saved: semantic_by_storey.png")


# ==============================================================================
# D. Proxy Analysis Visualizations
# ==============================================================================

def plot_proxy_by_storey(
    proxy_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Bar chart of proxy counts by storey across files."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [df.assign(file_label=label) for label, df in proxy_dfs.items()],
        ignore_index=True,
    )

    if len(combined) == 0:
        logger.info("  No proxies to visualize")
        return

    storey_counts = combined.groupby(["storey_name", "file_label"]).size().reset_index(name="count")
    pivot = storey_counts.pivot_table(index="storey_name", columns="file_label", values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, color=[FILE_COLORS.get(c, f"C{i}") for i, c in enumerate(pivot.columns)],
               edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Storey")
    ax.set_ylabel("Proxy Count")
    ax.set_title("Proxy Elements by Storey and File")
    ax.legend(title="File")
    plt.xticks(rotation=30, ha="right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "proxy_by_storey.png", save_svg)
    logger.info("  Saved: proxy_by_storey.png")


def plot_proxy_categories(
    proxy_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Bar chart of proxy inferred category distribution."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [df.assign(file_label=label) for label, df in proxy_dfs.items()],
        ignore_index=True,
    )

    if len(combined) == 0 or "inferred_category" not in combined.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    cat_counts = combined["inferred_category"].value_counts()
    colors = [COLORS.get(c, "#95a5a6") for c in cat_counts.index]
    cat_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Inferred Category")
    ax.set_ylabel("Count")
    ax.set_title("Proxy Inferred Category Distribution (Combined)")
    plt.xticks(rotation=35, ha="right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "proxy_categories.png", save_svg)
    logger.info("  Saved: proxy_categories.png")


def plot_proxy_bbox_distribution(
    proxy_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Histogram of proxy bounding box sizes if available."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(list(proxy_dfs.values()), ignore_index=True)

    if "bbox_dx" not in combined.columns:
        logger.info("  No bbox data available for proxy bbox distribution plot")
        return

    has_bbox = combined.dropna(subset=["bbox_dx"])
    if len(has_bbox) < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, dim, label in zip(axes, ["bbox_dx", "bbox_dy", "bbox_dz"], ["Width (X)", "Depth (Y)", "Height (Z)"]):
        data = has_bbox[dim].dropna()
        if len(data) > 0:
            ax.hist(data, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
            ax.set_xlabel(f"{label} (mm)")
            ax.set_ylabel("Count")
            ax.set_title(f"Proxy {label}")

    fig.suptitle("Proxy Bounding Box Size Distributions", fontsize=12)
    plt.tight_layout()

    _save_fig(fig, output_dir / "proxy_bbox_distribution.png", save_svg)
    logger.info("  Saved: proxy_bbox_distribution.png")


# ==============================================================================
# E. Geometry Readiness Visualizations
# ==============================================================================

def plot_geometry_readiness(
    geom_summaries: Dict[str, Dict[str, Any]],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Plot geometry extraction success/failure per file."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(geom_summaries.keys())
    has_repr = [geom_summaries[l]["has_representation"] for l in labels]
    no_repr = [geom_summaries[l]["no_representation"] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, has_repr, label="Has Geometry", color="#2ecc71", edgecolor="white")
    ax.bar(x, no_repr, bottom=has_repr, label="No Geometry", color="#e74c3c", edgecolor="white")

    ax.set_xlabel("IFC File")
    ax.set_ylabel("Product Count")
    ax.set_title("Geometry Representation Availability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i in range(len(labels)):
        total = has_repr[i] + no_repr[i]
        if total > 0:
            ax.text(i, total + 50, f"{has_repr[i]/total:.0%}", ha="center", fontsize=9)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig(fig, output_dir / "geometry_readiness.png", save_svg)
    logger.info("  Saved: geometry_readiness.png")


def plot_bbox_size_distribution(
    bbox_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Histogram of bounding box sizes from geometry checks."""
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(list(bbox_dfs.values()), ignore_index=True)

    if "dx" not in combined.columns or len(combined) == 0:
        logger.info("  No bbox data for size distribution plot")
        return

    has_data = combined.dropna(subset=["dx"])
    if len(has_data) < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, dim, label in zip(axes, ["dx", "dy", "dz"], ["Width (X)", "Depth (Y)", "Height (Z)"]):
        data = has_data[dim].clip(upper=has_data[dim].quantile(0.95))
        ax.hist(data, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        ax.set_xlabel(f"{label} (mm)")
        ax.set_ylabel("Count")
        ax.set_title(f"BBox {label}")

    fig.suptitle("Bounding Box Size Distributions (95th percentile clip)", fontsize=12)
    plt.tight_layout()

    _save_fig(fig, output_dir / "bbox_size_distribution.png", save_svg)
    logger.info("  Saved: bbox_size_distribution.png")


# ==============================================================================
# F. Lightweight Spatial Previews
# ==============================================================================

def plot_spatial_preview(
    bbox_df: pd.DataFrame,
    classified_df: pd.DataFrame,
    storey_name: str,
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Generate a 2D plan-view spatial preview for a storey.

    Uses bounding box data merged with classification.
    Shows elements as colored rectangles in plan view.
    """
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bbox_df is None or len(bbox_df) == 0:
        return

    # Filter to storey
    if "storey_name" in bbox_df.columns:
        storey_bbox = bbox_df[bbox_df["storey_name"] == storey_name].copy()
    else:
        return

    if len(storey_bbox) == 0:
        logger.info(f"  No bbox data for storey {storey_name}")
        return

    # Merge with classification
    if "guid" in storey_bbox.columns and "guid" in classified_df.columns:
        storey_bbox = storey_bbox.merge(
            classified_df[["guid", "category"]],
            on="guid",
            how="left",
            suffixes=("", "_cls"),
        )

    required_cols = ["min_x", "max_x", "min_y", "max_y"]
    if not all(c in storey_bbox.columns for c in required_cols):
        return

    valid = storey_bbox.dropna(subset=required_cols)
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    for _, row in valid.iterrows():
        cat = row.get("category", "uncertain")
        color = COLORS.get(cat, "#95a5a6")
        alpha = 0.6 if cat != "ignorable" else 0.15

        x = row["min_x"]
        y = row["min_y"]
        w = row["max_x"] - row["min_x"]
        h = row["max_y"] - row["min_y"]

        rect = plt.Rectangle(
            (x, y), w, h,
            linewidth=0.3, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

    ax.set_xlim(valid["min_x"].min() - 1000, valid["max_x"].max() + 1000)
    ax.set_ylim(valid["min_y"].min() - 1000, valid["max_y"].max() + 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Plan View Preview: {storey_name}\n({len(valid)} elements with bbox data)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS.get(c, "#95a5a6"), label=c, alpha=0.7)
        for c in sorted(valid["category"].unique()) if c in COLORS
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    safe_name = storey_name.replace(" ", "_").replace("/", "_")
    _save_fig(fig, output_dir / f"spatial_preview_{safe_name}.png", save_svg)
    logger.info(f"  Saved: spatial_preview_{safe_name}.png")


def generate_all_spatial_previews(
    bbox_dfs: Dict[str, pd.DataFrame],
    classified_dfs: Dict[str, pd.DataFrame],
    storey_mapping: Dict[str, Any],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Generate spatial previews for key storeys."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine data
    combined_bbox = pd.concat(list(bbox_dfs.values()), ignore_index=True) if bbox_dfs else pd.DataFrame()
    combined_classified = pd.concat(list(classified_dfs.values()), ignore_index=True) if classified_dfs else pd.DataFrame()

    if len(combined_bbox) == 0:
        logger.warning("  No bbox data available for spatial previews")
        return

    # Determine key storeys from mapping
    key_storeys = storey_mapping.get("walkable_levels", [])
    # Also include storeys with significant element counts
    all_storeys = combined_bbox["storey_name"].value_counts()
    for sname in all_storeys.index[:5]:  # Top 5 by count
        if sname not in key_storeys and sname != "UNASSIGNED":
            key_storeys.append(sname)

    for sname in key_storeys:
        plot_spatial_preview(
            combined_bbox, combined_classified, sname,
            output_dir, save_svg=save_svg,
        )


# ==============================================================================
# Master visualization function
# ==============================================================================

def generate_all_visualizations(
    audits: Dict[str, Dict[str, Any]],
    storey_mapping: Dict[str, Any],
    classified_dfs: Dict[str, pd.DataFrame],
    proxy_dfs: Dict[str, pd.DataFrame],
    geom_summaries: Dict[str, Dict[str, Any]],
    bbox_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    save_svg: bool = True,
) -> None:
    """Generate all preprocessing visualizations.

    Args:
        audits: Per-file audit results.
        storey_mapping: Storey mapping results.
        classified_dfs: Per-file classified DataFrames.
        proxy_dfs: Per-file proxy DataFrames.
        geom_summaries: Per-file geometry summaries.
        bbox_dfs: Per-file bbox DataFrames.
        output_dir: Base figures directory.
        save_svg: Whether to also save SVG format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Generating all visualizations")
    logger.info("=" * 60)

    # A. Audit visualizations
    audit_dir = output_dir / "audit"
    plot_entity_counts_by_class(audits, audit_dir, save_svg=save_svg)
    plot_entity_counts_by_storey(audits, audit_dir, save_svg=save_svg)
    plot_proxy_proportion(audits, audit_dir, save_svg=save_svg)

    # B. Storey mapping visualizations
    storey_dir = output_dir / "storey_mapping"
    plot_storey_file_heatmap(storey_mapping, storey_dir, save_svg=save_svg)
    plot_storey_dominance(storey_mapping, storey_dir, save_svg=save_svg)

    # C. Semantic classification visualizations
    semantic_dir = output_dir / "semantic"
    if classified_dfs:
        plot_semantic_category_distribution(classified_dfs, semantic_dir, save_svg=save_svg)
        plot_semantic_by_storey(classified_dfs, storey_mapping, semantic_dir, save_svg=save_svg)

    # D. Proxy visualizations
    proxy_fig_dir = output_dir / "proxy"
    if proxy_dfs:
        plot_proxy_by_storey(proxy_dfs, proxy_fig_dir, save_svg=save_svg)
        plot_proxy_categories(proxy_dfs, proxy_fig_dir, save_svg=save_svg)
        plot_proxy_bbox_distribution(proxy_dfs, proxy_fig_dir, save_svg=save_svg)

    # E. Geometry readiness visualizations
    geom_fig_dir = output_dir / "geometry"
    if geom_summaries:
        plot_geometry_readiness(geom_summaries, geom_fig_dir, save_svg=save_svg)
    if bbox_dfs:
        plot_bbox_size_distribution(bbox_dfs, geom_fig_dir, save_svg=save_svg)

    # F. Spatial previews
    spatial_dir = output_dir / "spatial_preview"
    if bbox_dfs and classified_dfs:
        generate_all_spatial_previews(
            bbox_dfs, classified_dfs, storey_mapping,
            spatial_dir, save_svg=save_svg,
        )

    logger.info("All visualizations generated.")
