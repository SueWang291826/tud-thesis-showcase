"""Check F3 connector footprints and exclusion zones."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from shapely.ops import unary_union
from shapely.geometry import box

config = load_config(str(ROOT / "config" / "experiment_config.yaml"))
data = load_preprocessing_products(config)

all_geometry, all_connectors, control_points = extract_all_levels(config, data)

f3_geom = all_geometry["F3"]
f3_connectors = f3_geom["connectors"]

print(f"F3 connectors: {len(f3_connectors)}")

# Build exclusion polygon from connector footprints
excl_buf = config["sampling"]["exclude_connector_buffer_m"]
excl_polys = []
for c in f3_connectors:
    fp = c.get("footprint")
    if fp is not None and not fp.is_empty:
        excl_polys.append(fp.buffer(excl_buf))

if excl_polys:
    exclude_geom = unary_union(excl_polys)
else:
    from shapely.geometry import Polygon
    exclude_geom = Polygon()

print(f"Exclusion zone total area: {exclude_geom.area:.1f} m2")
print(f"Exclusion zone bounds: {[round(v,1) for v in exclude_geom.bounds]}")

# Check exclusion in problem zones
problem_zones = [
    ("SW_left_unpaid_corridor", 46, 7.5, 56, 14.7),
    ("SW_scanner_approach", 42, 7.5, 46, 14),
    ("SE_right_unpaid_corridor", 95, 7.5, 120, 14.7),
]

for name, x0, y0, x1, y1 in problem_zones:
    region = box(x0, y0, x1, y1)
    excl_overlap = exclude_geom.intersection(region)
    pct = excl_overlap.area / region.area * 100
    print(f"\nZone {name} (x={x0}-{x1}, y={y0}-{y1}):")
    print(f"  Exclusion overlap: {excl_overlap.area:.1f}m2 / {region.area:.1f}m2 ({pct:.0f}%)")

# Detail: which connectors overlap with left unpaid corridor?
print("\n=== Connectors overlapping left unpaid corridor (x=46-56, y=7.5-14.7) ===")
corridor = box(46, 7.5, 56, 14.7)
for c in f3_connectors:
    fp = c.get("footprint")
    if fp is None or fp.is_empty:
        continue
    fp_buf = fp.buffer(excl_buf)
    if fp_buf.intersects(corridor):
        b = [round(v, 1) for v in fp.bounds]
        bb = [round(v, 1) for v in fp_buf.bounds]
        ovl = fp_buf.intersection(corridor).area
        print(f"  {c['id']:30s} type={c['type']:12s} footprint={b}")
        print(f"    buffered={bb}  overlap_in_corridor={ovl:.1f}m2")

print("\nDone.")
