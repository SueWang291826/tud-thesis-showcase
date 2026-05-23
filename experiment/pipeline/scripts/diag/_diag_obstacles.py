"""Inspect F3 obstacles in problem zones."""
import json, pathlib

p = pathlib.Path("outputs/step1_geometry/F3_geometry.json")
if not p.exists():
    print("step1 geometry output not found, running step1...")
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/step1_geometry.py"], check=True)
    
data = json.loads(p.read_text())
obs = data.get("obstacles", [])
print(f"F3 obstacles total: {len(obs)}")

# Problem zones to inspect
zones = [
    ("SW_paid_corridor", 42, 7, 60, 15),
    ("SE_paid_corridor", 95, 5, 120, 15),
    ("F3_entrance_D_area", 60, 18, 85, 23),
    ("F3_entrance_E_area", 108, 17, 125, 23),
    ("F4_right_corner", 119, 7, 134, 23),  # on F3 if F3 has that area
]

for zone_name, x0, y0, x1, y1 in zones:
    zone_obs = []
    for o in obs:
        b = o.get("bounds", [])
        if len(b) == 4:
            ox0, oy0, ox1, oy1 = b
            if ox0 < x1 and ox1 > x0 and oy0 < y1 and oy1 > y0:
                zone_obs.append(o)
    print(f"\nZone {zone_name} (x={x0}-{x1}, y={y0}-{y1}): {len(zone_obs)} obstacles")
    for o in sorted(zone_obs, key=lambda x: x.get("bounds", [0])[0]):
        b = [round(v, 2) for v in o["bounds"]]
        name = o.get("name", "?")[:45]
        area = o.get("area_m2", 0)
        print(f"  {name:45s}  bounds={b}  area={area:.2f}m2")

print("\nDone.")
