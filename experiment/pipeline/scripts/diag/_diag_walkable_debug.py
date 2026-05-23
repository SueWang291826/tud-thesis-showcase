"""Direct debug: is point (46.5, 10) in F3 walkable polygon?"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from shapely.geometry import Point

config = load_config(str(ROOT / "config" / "experiment_config.yaml"))
data = load_preprocessing_products(config)

all_geometry, all_connectors, control_points = extract_all_levels(config, data)

geom = all_geometry["F3"]
walkable = geom["walkable"]
obs_union = geom["obstacle_union"]
floor = geom["floor"]

# Test points
test_pts = [
    (46.5, 10.0, "east of scanner, mid-height"),
    (46.5, 8.5, "east of scanner, near south"),
    (48.0, 10.0, "corridor mid"),
    (50.0, 10.0, "corridor center"),
    (50.0, 13.5, "corridor north"),
    (50.0, 14.5, "corridor far north"),
    (43.5, 10.0, "scanner approach (west)"),
    (70.0, 10.0, "central paid zone"),
]

print("=== F3 walkable polygon test ===")
print(f"Floor bounds: {[round(v,1) for v in floor.bounds]}")
print(f"Walkable bounds: {[round(v,1) for v in walkable.bounds]}")
print(f"Walkable area: {walkable.area:.1f} m2")

for x, y, label in test_pts:
    pt = Point(x, y)
    in_floor = floor.contains(pt)
    in_walkable = walkable.contains(pt)
    obs_dist = pt.distance(obs_union) if not obs_union.is_empty else 999.0
    print(f"  ({x:.1f},{y:.1f}) {label:40s}: "
          f"in_floor={in_floor}, in_walkable={in_walkable}, "
          f"obs_dist={obs_dist:.3f}m")

print()
# At what y values does walkable cover x=48?
print("=== Y-scan at x=48.0 ===")
for y in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5]:
    pt = Point(48.0, y)
    in_w = walkable.contains(pt)
    d = pt.distance(obs_union) if not obs_union.is_empty else 999.0
    print(f"  y={y:.1f}: in_walkable={in_w}, obs_dist={d:.3f}m ({'OK' if d>=0.25 else 'BLOCKED'})")

print("\nDone.")

from src.geometry_extractor import extract_all_levels
from src.config_loader import load_config
from src.data_loader import load_data
from shapely.geometry import Point

config = load_config("config/experiment_config.yaml")
data = load_data(config)

all_geometry, all_connectors, control_points = extract_all_levels(config, data)

geom = all_geometry["F3"]
walkable = geom["walkable"]
obs_union = geom["obstacle_union"]
floor = geom["floor"]

# Test points
test_pts = [
    (46.5, 10.0, "east of scanner, mid-height"),
    (46.5, 8.5, "east of scanner, near south"),
    (48.0, 10.0, "corridor mid"),
    (50.0, 10.0, "corridor center"),
    (50.0, 13.5, "corridor north"),
    (50.0, 14.5, "corridor far north"),
    (43.5, 10.0, "scanner approach (west)"),
    (70.0, 10.0, "central paid zone"),
]

print("=== F3 walkable polygon test ===")
print(f"Floor bounds: {[round(v,1) for v in floor.bounds]}")
print(f"Walkable bounds: {[round(v,1) for v in walkable.bounds]}")
print(f"Walkable area: {walkable.area:.1f} m²")

for x, y, label in test_pts:
    pt = Point(x, y)
    in_floor = floor.contains(pt)
    in_walkable = walkable.contains(pt)
    obs_dist = pt.distance(obs_union) if not obs_union.is_empty else 999.0
    print(f"  ({x:.1f},{y:.1f}) {label:40s}: "
          f"in_floor={in_floor}, in_walkable={in_walkable}, "
          f"obs_dist={obs_dist:.3f}m")

print()
# At what y values does walkable cover x=48?
print("=== Y-scan at x=48.0 ===")
for y in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5]:
    pt = Point(48.0, y)
    in_w = walkable.contains(pt)
    d = pt.distance(obs_union) if not obs_union.is_empty else 999.0
    print(f"  y={y:.1f}: in_walkable={in_w}, obs_dist={d:.3f}m ({'OK' if d>=0.25 else 'BLOCKED'})")

print("\nDone.")
