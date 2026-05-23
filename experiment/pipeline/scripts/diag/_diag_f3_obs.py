"""Inspect F3 obstacles in problem zones from GeoJSON."""
import json, pathlib

p = pathlib.Path("outputs/step1_geometry/F3/obstacles.geojson")
data = json.loads(p.read_text())
features = data.get("features", [])
print(f"F3 obstacles total: {len(features)}")

def bbox_of(feature):
    geom = feature.get("geometry", {})
    coords = []
    gt = geom.get("type", "")
    if gt == "Polygon":
        for pt in geom["coordinates"][0]:
            coords.append(pt)
    elif gt == "MultiPolygon":
        for poly in geom["coordinates"]:
            for pt in poly[0]:
                coords.append(pt)
    elif gt == "LineString":
        coords = geom["coordinates"]
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))

zones = [
    ("SW_paid_corridor (x=42-60, y=7-15)", 42, 7, 60, 15),
    ("SE_paid_corridor (x=95-120, y=5-15)", 95, 5, 120, 15),
    ("SW_connector_approach (x=26-44, y=5-15)", 26, 5, 44, 15),
    ("F3_entrance_D_area (x=60-85, y=18-23)", 60, 18, 85, 23),
    ("F3_entrance_E_area (x=108-125, y=17-23)", 108, 17, 125, 23),
]

for zone_name, x0, y0, x1, y1 in zones:
    zone_obs = []
    for feat in features:
        b = bbox_of(feat)
        if b is None:
            continue
        ox0, oy0, ox1, oy1 = b
        if ox0 < x1 and ox1 > x0 and oy0 < y1 and oy1 > y0:
            zone_obs.append((b, feat.get("properties", {})))
    print(f"\nZone {zone_name}: {len(zone_obs)} obstacles")
    for b, props in sorted(zone_obs, key=lambda x: x[0][0])[:20]:
        b2 = [round(v, 2) for v in b]
        name = str(props.get("Name", props.get("name", "?")))[:50]
        print(f"  {name:50s}  bbox={b2}")

print("\nDone.")
