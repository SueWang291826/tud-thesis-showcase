"""Diagnose connector anchors & stair landings."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["MPLBACKEND"] = "Agg"

from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
import json

cfg = load_config("config/experiment_config.yaml")
data = load_preprocessing_products(cfg)
geometries, all_connectors, control_points = extract_all_levels(cfg, data)

print("\n" + "="*70)
print("ALL CONNECTORS SUMMARY")
print("="*70)
for c in all_connectors:
    fp = c.get("footprint")
    fp_str = f"area={fp.area:.2f}" if fp and not fp.is_empty else "NO footprint"
    if c["type"] == "elevator":
        lvls = ",".join(c.get("connected_levels", []))
        print(f"  {c['id']:<20s} {c['type']:<15s} levels=[{lvls}]  "
              f"z=[{c['z_min']:.1f},{c['z_max']:.1f}]  {fp_str}  {c['name'][:40]}")
    else:
        bl = c.get("bottom_level", "?")
        tl = c.get("top_level", "?")
        print(f"  {c['id']:<20s} {c['type']:<15s} {bl}->{tl}  "
              f"z=[{c['z_min']:.1f},{c['z_max']:.1f}]  {fp_str}  {c['name'][:40]}")

# Check which connectors appear on each level
print("\n" + "="*70)
print("CONNECTORS PER LEVEL (after forbidden zone filter)")
print("="*70)
for lvl in ["F1", "F3", "F4"]:
    geom = geometries[lvl]
    conns = geom["connectors"]
    print(f"\n--- {lvl}: {len(conns)} connectors ---")
    for c in conns:
        fp = c.get("footprint")
        if fp and not fp.is_empty:
            cx, cy = fp.centroid.x, fp.centroid.y
            bds = fp.bounds
            print(f"  {c['id']:<20s} {c['type']:<15s} centroid=({cx:.1f},{cy:.1f})  "
                  f"bounds=[{bds[0]:.1f},{bds[1]:.1f},{bds[2]:.1f},{bds[3]:.1f}]  "
                  f"{c['name'][:35]}")
        else:
            print(f"  {c['id']:<20s} {c['type']:<15s} NO footprint  {c['name'][:35]}")

# Check stair flights specifically near the escalator area
print("\n" + "="*70)
print("STAIR FLIGHTS WITH Z-RANGE DETAILS")
print("="*70)
stairs = [c for c in all_connectors if c["type"] == "stair_flight"]
# Group by name (IFC element)
from collections import defaultdict
by_name = defaultdict(list)
for s in stairs:
    by_name[s["name"][:40]].append(s)

for name, flights in sorted(by_name.items()):
    print(f"\n  {name}:")
    for f in flights:
        fp = f.get("footprint")
        if fp and not fp.is_empty:
            bds = fp.bounds
            print(f"    {f['id']:<20s} {f.get('bottom_level','?')}->{f.get('top_level','?')}  "
                  f"z=[{f['z_min']:.1f},{f['z_max']:.1f}]  "
                  f"bounds=[{bds[0]:.1f},{bds[1]:.1f},{bds[2]:.1f},{bds[3]:.1f}]")
        else:
            print(f"    {f['id']:<20s} {f.get('bottom_level','?')}->{f.get('top_level','?')}  "
                  f"z=[{f['z_min']:.1f},{f['z_max']:.1f}]  NO footprint")

# Elevator details
print("\n" + "="*70)
print("ELEVATOR DETAILS")
print("="*70)
elevators = [c for c in all_connectors if c["type"] == "elevator"]
for e in elevators:
    print(f"  {e['id']}")
    for k, v in e.items():
        if k == "footprint":
            if v and not v.is_empty:
                print(f"    footprint: bounds={[round(x,2) for x in v.bounds]}, area={v.area:.2f}")
            else:
                print(f"    footprint: NONE")
        else:
            print(f"    {k}: {v}")

# Escalator details
print("\n" + "="*70)
print("ESCALATOR DETAILS")
print("="*70)
escs = [c for c in all_connectors if c["type"] == "escalator"]
for e in escs:
    fp = e.get("footprint")
    if fp and not fp.is_empty:
        bds = fp.bounds
        print(f"  {e['id']:<20s} {e.get('bottom_level','?')}->{e.get('top_level','?')}  "
              f"z=[{e['z_min']:.1f},{e['z_max']:.1f}]  "
              f"bounds=[{bds[0]:.1f},{bds[1]:.1f},{bds[2]:.1f},{bds[3]:.1f}]  "
              f"{e['name'][:40]}")
    else:
        print(f"  {e['id']:<20s} NO footprint  {e['name'][:40]}")
