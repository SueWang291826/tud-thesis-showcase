"""Debug: check raw IFC coordinates WITHOUT dividing by 1000."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ifcopenshell
import ifcopenshell.geom

RAW_PLATFORM = Path(r"e:\TUD-Thesis\station\data0\站台层.ifc")
model = ifcopenshell.open(str(RAW_PLATFORM))

# Check project units
for unit_assign in model.by_type("IfcUnitAssignment"):
    for u in unit_assign.Units:
        if hasattr(u, 'UnitType') and u.UnitType == 'LENGTHUNIT':
            prefix = getattr(u, 'Prefix', None)
            name = getattr(u, 'Name', None)
            print(f"  Length unit: Prefix={prefix}, Name={name}")

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

# Check a few IfcSlab - show RAW vertex range (no /1000)
slabs = model.by_type("IfcSlab")
print(f"\n--- IfcSlab (first 5, RAW coords) ---")
for slab in slabs[:5]:
    name = slab.Name or ""
    try:
        shape = ifcopenshell.geom.create_shape(settings, slab)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            print(f"  '{name}': x=[{min(xs):.3f},{max(xs):.3f}] "
                  f"y=[{min(ys):.3f},{max(ys):.3f}] "
                  f"z=[{min(zs):.3f},{max(zs):.3f}] "
                  f"n_verts={len(xs)}")
    except Exception as e:
        print(f"  '{name}': FAIL - {e}")

# Check the slab that should be the big platform floor
print(f"\n--- Large platform slab (站台地板) ---")
for slab in slabs:
    name = slab.Name or ""
    if "站台地板" in name:
        try:
            shape = ifcopenshell.geom.create_shape(settings, slab)
            verts = shape.geometry.verts
            if verts:
                xs = [verts[i] for i in range(0, len(verts), 3)]
                ys = [verts[i+1] for i in range(0, len(verts), 3)]
                zs = [verts[i+2] for i in range(0, len(verts), 3)]
                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                print(f"  '{name}': x=[{min(xs):.3f},{max(xs):.3f}] (dx={dx:.3f}) "
                      f"y=[{min(ys):.3f},{max(ys):.3f}] (dy={dy:.3f}) "
                      f"z=[{min(zs):.3f},{max(zs):.3f}] "
                      f"n_verts={len(xs)}")
        except Exception as e:
            print(f"  '{name}': FAIL - {e}")

# Also check an element that HAS bbox in CSV for reference
# Let's check an IfcPlate (many on F1 with bbox)
print(f"\n--- Sample IfcPlate (should have known bbox) ---")
plates = model.by_type("IfcPlate")
for plate in plates[:3]:
    name = plate.Name or ""
    try:
        shape = ifcopenshell.geom.create_shape(settings, plate)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            print(f"  '{name}': x=[{min(xs):.3f},{max(xs):.3f}] "
                  f"y=[{min(ys):.3f},{max(ys):.3f}] "
                  f"z=[{min(zs):.3f},{max(zs):.3f}] "
                  f"n_verts={len(xs)}")
    except Exception as e:
        print(f"  '{name}': FAIL - {e}")

# Also check escalator and stair flight for calibration
print(f"\n--- Sample escalator ---")
for p in model.by_type("IfcBuildingElementProxy"):
    if "扶梯" in (p.Name or ""):
        try:
            shape = ifcopenshell.geom.create_shape(settings, p)
            verts = shape.geometry.verts
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            print(f"  '{p.Name}': x=[{min(xs):.3f},{max(xs):.3f}] "
                  f"y=[{min(ys):.3f},{max(ys):.3f}] "
                  f"z=[{min(zs):.3f},{max(zs):.3f}]")
        except:
            pass
        break

print(f"\n--- Sample IfcStairFlight ---")
for sf in model.by_type("IfcStairFlight")[:3]:
    try:
        shape = ifcopenshell.geom.create_shape(settings, sf)
        verts = shape.geometry.verts
        xs = [verts[i] for i in range(0, len(verts), 3)]
        ys = [verts[i+1] for i in range(0, len(verts), 3)]
        zs = [verts[i+2] for i in range(0, len(verts), 3)]
        print(f"  '{sf.Name}': x=[{min(xs):.3f},{max(xs):.3f}] "
              f"y=[{min(ys):.3f},{max(ys):.3f}] "
              f"z=[{min(zs):.3f},{max(zs):.3f}]")
    except:
        pass

print("\nDone!")
