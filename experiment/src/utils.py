"""
Shared utility functions.
=========================

Geometry helpers, GeoJSON I/O, file helpers, matplotlib font config.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml
from shapely.geometry import MultiPolygon, Polygon, Point, LineString, box


# ============================================================================
# Config
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# Geometry helpers
# ============================================================================

def flatten_polygons(geom) -> list[Polygon]:
    """Extract a flat list of Polygons from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(flatten_polygons(g))
        return out
    return []


def polygon_from_bbox(bbox: list[float]) -> Polygon:
    """Create a Shapely box from [minx, miny, maxx, maxy]."""
    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def iter_grid_points(bounds: tuple[float, ...], resolution: float) -> Iterator[tuple[float, float]]:
    """Yield (x, y) grid points within bounding box."""
    minx, miny, maxx, maxy = bounds
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            yield (x, y)
            y += resolution
        x += resolution


def euclidean_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def euclidean_3d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def safe_name(s: str) -> str:
    """Convert a storey name to a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_")


# ============================================================================
# GeoJSON I/O
# ============================================================================

def write_geojson(path: str | Path, features: list[dict]) -> None:
    """Write a GeoJSON FeatureCollection."""
    fc = {"type": "FeatureCollection", "features": features}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)


def point_feature(x: float, y: float, props: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": props,
    }


def line_feature(coords: list[tuple], props: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[c[0], c[1]] for c in coords]},
        "properties": props,
    }


def polygon_feature(geom: Polygon, props: dict) -> dict:
    """Convert a Shapely Polygon to a GeoJSON Feature."""
    coords = [list(geom.exterior.coords)]
    for ring in geom.interiors:
        coords.append(list(ring.coords))
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": props,
    }


def dump_json(path: str | Path, obj: Any) -> None:
    """Write any JSON-serialisable object."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# GIF assembly
# ============================================================================

def save_gif(frame_paths: list[str | Path], out_path: str | Path, fps: float = 4.0) -> None:
    """Assemble PNG frames into an animated GIF."""
    from PIL import Image
    if not frame_paths:
        return
    frames = [Image.open(str(p)) for p in frame_paths]
    duration = int(1000 / fps)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


# ============================================================================
# Matplotlib CJK font
# ============================================================================

def setup_matplotlib_font() -> None:
    """Configure matplotlib for CJK text rendering."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    available = [n for n in candidates if n in installed]
    if available:
        plt.rcParams["font.sans-serif"] = available + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
