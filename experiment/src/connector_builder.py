"""
Typed Connector Builder
========================

Model vertical connectors (stairs, escalators, elevators) and passage
constraints (fare gates) as typed, parameterised graph elements.

Each connector type has:
  - Geometry: footprint polygon + z-range
  - Parameters: capacity, speed, directionality
  - Graph representation: chain nodes, directed/undirected edges

Connector Type Semantics
------------------------
- **Stair**: Bidirectional chain of intermediate nodes at dz intervals.
  Capacity-gated (max N agents simultaneously). Speed factor < 1.
- **Escalator**: Unidirectional (up or down). Fixed belt speed.
  Modelled as a linear chain with forced direction.
- **Elevator**: Batch transport. No intermediate nodes — modelled as
  a single weighted edge with dwell_time + travel_time.
- **Fare Gate**: Internal passage constraint on F3. Modelled as a
  single edge with throughput delay.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from src.utils import euclidean_2d, euclidean_3d


# ============================================================================
# Connector grouping
# ============================================================================

def group_connectors_by_type(connector_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group connectors by subtype for separate modelling."""
    from src.data_loader import filter_connectors_for_navigation
    nav = filter_connectors_for_navigation(connector_df)
    
    groups = {}
    for subtype, group in nav.groupby("connector_subtype"):
        groups[subtype] = group.copy()
    return groups


def identify_stair_groups(
    connector_df: pd.DataFrame,
    levels: dict,
) -> list[dict]:
    """Identify stair groups connecting adjacent levels.

    Groups stair + stair_flight elements by spatial proximity
    to form logical stair units. Each stair unit connects two levels.
    
    Returns list of stair definitions with:
        - id, bottom_level, top_level
        - bottom_xy, top_xy, z_min, z_max
        - component_guids
    """
    from src.data_loader import filter_connectors_for_navigation
    nav = filter_connectors_for_navigation(connector_df)
    stairs_df = nav[nav["connector_subtype"].isin(["stair", "stair_flight"])].copy()
    
    if stairs_df.empty:
        return []

    # Sort levels by elevation
    level_elevations = sorted(
        [(k, v["elevation_m"]) for k, v in levels.items()],
        key=lambda x: x[1],
    )

    stair_groups = []
    
    # Group by spatial proximity (cluster by centroid)
    if "min_x" in stairs_df.columns:
        stairs_df["cx"] = (stairs_df["min_x"] + stairs_df["max_x"]) / 2
        stairs_df["cy"] = (stairs_df["min_y"] + stairs_df["max_y"]) / 2
        stairs_df["cz"] = (stairs_df["min_z"] + stairs_df["max_z"]) / 2
    else:
        return []

    # Simple clustering: group elements within 3m XY distance
    assigned = set()
    cluster_id = 0
    
    for idx, row in stairs_df.iterrows():
        if idx in assigned:
            continue
        cluster = [idx]
        assigned.add(idx)
        cx, cy = row["cx"], row["cy"]
        
        for idx2, row2 in stairs_df.iterrows():
            if idx2 in assigned:
                continue
            d = math.sqrt((cx - row2["cx"]) ** 2 + (cy - row2["cy"]) ** 2)
            if d < 3.0:
                cluster.append(idx2)
                assigned.add(idx2)
        
        cluster_df = stairs_df.loc[cluster]
        z_min = cluster_df["min_z"].min()
        z_max = cluster_df["max_z"].max()
        
        # Determine which levels this stair connects
        bottom_level = None
        top_level = None
        for lk, elev in level_elevations:
            if abs(elev - z_min) < 3.0:
                bottom_level = lk
            if abs(elev - z_max) < 3.0:
                top_level = lk
        
        if bottom_level is None:
            bottom_level = level_elevations[0][0]
        if top_level is None:
            top_level = level_elevations[-1][0]
        if bottom_level == top_level and len(level_elevations) > 1:
            # Find the next level up
            for lk, elev in level_elevations:
                if elev > levels[bottom_level]["elevation_m"]:
                    top_level = lk
                    break

        stair_groups.append({
            "id": f"stair_{cluster_id}",
            "connector_type": "stair",
            "bottom_level": bottom_level,
            "top_level": top_level,
            "bottom_xy": [float(cluster_df["cx"].mean()), float(cluster_df["cy"].mean())],
            "top_xy": [float(cluster_df["cx"].mean()) + 0.5, float(cluster_df["cy"].mean()) + 0.5],
            "z_min": float(z_min),
            "z_max": float(z_max),
            "component_guids": list(cluster_df["guid"]),
            "n_components": len(cluster),
        })
        cluster_id += 1

    return stair_groups


def identify_escalator_groups(
    connector_df: pd.DataFrame,
    levels: dict,
) -> list[dict]:
    """Identify escalator units connecting levels.
    
    Each escalator is directional (up or down).
    """
    from src.data_loader import filter_connectors_for_navigation
    nav = filter_connectors_for_navigation(connector_df)
    esc_df = nav[nav["connector_subtype"] == "escalator"].copy()
    
    if esc_df.empty:
        return []

    level_elevations = sorted(
        [(k, v["elevation_m"]) for k, v in levels.items()],
        key=lambda x: x[1],
    )

    escalators = []
    for idx, (_, row) in enumerate(esc_df.iterrows()):
        cx = (float(row.get("min_x", 0)) + float(row.get("max_x", 0))) / 2
        cy = (float(row.get("min_y", 0)) + float(row.get("max_y", 0))) / 2
        z_min = float(row.get("min_z", 0))
        z_max = float(row.get("max_z", 0))

        # Determine levels
        bottom_level = min(level_elevations, key=lambda x: abs(x[1] - z_min))[0]
        top_level = min(level_elevations, key=lambda x: abs(x[1] - z_max))[0]
        if bottom_level == top_level:
            for lk, elev in level_elevations:
                if elev > levels[bottom_level]["elevation_m"]:
                    top_level = lk
                    break

        # Infer direction from name (heuristic)
        name = str(row.get("name", "")).lower()
        direction = "up" if "上" in name or "up" in name else "down" if "下" in name or "down" in name else "up"

        escalators.append({
            "id": f"escalator_{idx}",
            "connector_type": "escalator",
            "bottom_level": bottom_level,
            "top_level": top_level,
            "bottom_xy": [cx - 0.3, cy],
            "top_xy": [cx + 0.3, cy],
            "z_min": z_min,
            "z_max": z_max,
            "direction": direction,
            "guid": row.get("guid", ""),
        })

    return escalators


def identify_elevator_groups(
    connector_df: pd.DataFrame,
    levels: dict,
) -> list[dict]:
    """Identify elevator shafts connecting multiple levels."""
    from src.data_loader import filter_connectors_for_navigation
    nav = filter_connectors_for_navigation(connector_df)
    elev_df = nav[nav["connector_subtype"] == "elevator"].copy()
    
    if elev_df.empty:
        return []

    level_elevations = sorted(
        [(k, v["elevation_m"]) for k, v in levels.items()],
        key=lambda x: x[1],
    )

    elevators = []
    for idx, (_, row) in enumerate(elev_df.iterrows()):
        cx = (float(row.get("min_x", 0)) + float(row.get("max_x", 0))) / 2
        cy = (float(row.get("min_y", 0)) + float(row.get("max_y", 0))) / 2
        z_min = float(row.get("min_z", 0))
        z_max = float(row.get("max_z", 0))

        # Elevator connects all levels within its z-range
        connected_levels = [
            lk for lk, elev in level_elevations
            if z_min - 1.0 <= elev <= z_max + 1.0
        ]

        elevators.append({
            "id": f"elevator_{idx}",
            "connector_type": "elevator",
            "connected_levels": connected_levels,
            "xy": [cx, cy],
            "z_min": z_min,
            "z_max": z_max,
            "guid": row.get("guid", ""),
        })

    return elevators


def build_all_connectors(
    connector_df: pd.DataFrame,
    levels: dict,
    config: dict,
) -> dict:
    """Build all typed connector definitions.

    Returns dict with:
        - stairs: list of stair group definitions
        - escalators: list of escalator definitions  
        - elevators: list of elevator definitions
        - summary: counts per type
    """
    stairs = identify_stair_groups(connector_df, levels)
    escalators = identify_escalator_groups(connector_df, levels)
    elevators = identify_elevator_groups(connector_df, levels)

    summary = {
        "n_stair_groups": len(stairs),
        "n_escalators": len(escalators),
        "n_elevators": len(elevators),
        "stair_connections": [(s["bottom_level"], s["top_level"]) for s in stairs],
        "escalator_connections": [(e["bottom_level"], e["top_level"]) for e in escalators],
        "elevator_connected_levels": [e["connected_levels"] for e in elevators],
    }

    print(f"  Connectors: {len(stairs)} stair groups, "
          f"{len(escalators)} escalators, {len(elevators)} elevators")

    return {
        "stairs": stairs,
        "escalators": escalators,
        "elevators": elevators,
        "summary": summary,
    }
