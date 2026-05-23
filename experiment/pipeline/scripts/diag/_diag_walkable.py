"""Check F3 walkable polygon coverage in problem zones."""
import json, pathlib
from shapely.geometry import shape

p = pathlib.Path("outputs/step1_geometry/F3/walkable.geojson")
data = json.loads(p.read_text())

# Union all walkable polygons
from shapely.ops import unary_union
geoms = [shape(f["geometry"]) for f in data["features"] if f.get("geometry")]
walkable = unary_union(geoms)
print(f"F3 walkable area: {walkable.area:.1f} m²")

zones = [
    ("SW_paid_corridor (x=46.5-54, y=7.5-14.5)", 46.5, 7.5, 54, 14.5),
    ("SW_connector_approach (x=26-43, y=5-7)", 26, 5, 43, 7),
    ("SW_connector_landing (x=26-43, y=0-5)", 26, 0, 43, 5),
    ("SE_paid_corridor (x=99-120, y=7.5-14.5)", 99, 7.5, 120, 14.5),
    ("F3_entrance_D (x=62-80, y=19-22)", 62, 19, 80, 22),
    ("F3_entrance_E (x=112-122, y=19-22)", 112, 19, 122, 22),
]

from shapely.geometry import box
for name, x0, y0, x1, y1 in zones:
    region = box(x0, y0, x1, y1)
    intersection = walkable.intersection(region)
    pct = intersection.area / region.area * 100
    print(f"  {name}: walkable={intersection.area:.1f}m² / {region.area:.1f}m² ({pct:.0f}%)")

print("\nDone.")
