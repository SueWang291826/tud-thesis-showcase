"""Quick check: F3 concourse slabs + elevator coords from raw IFC, NO /1000."""
import ifcopenshell, ifcopenshell.geom

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

# --- F3 concourse slabs ---
print("=== F3 Concourse IfcSlab (large ones) ===")
m3 = ifcopenshell.open(r"e:\TUD-Thesis\station\data0\站厅层.ifc")
for slab in m3.by_type("IfcSlab"):
    name = slab.Name or ""
    try:
        shape = ifcopenshell.geom.create_shape(settings, slab)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            area = dx * dy
            if area > 50:
                print(f"  '{name}': x=[{min(xs):.1f},{max(xs):.1f}] "
                      f"y=[{min(ys):.1f},{max(ys):.1f}] "
                      f"z=[{min(zs):.1f},{max(zs):.1f}] "
                      f"area~{area:.0f}m²")
    except:
        pass

# --- F4 traffic slabs ---
print("\n=== F4 Traffic IfcSlab (large ones) ===")
m4 = ifcopenshell.open(r"e:\TUD-Thesis\station\data0\交通层.ifc")
for slab in m4.by_type("IfcSlab"):
    name = slab.Name or ""
    try:
        shape = ifcopenshell.geom.create_shape(settings, slab)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            area = dx * dy
            if area > 50:
                print(f"  '{name}': x=[{min(xs):.1f},{max(xs):.1f}] "
                      f"y=[{min(ys):.1f},{max(ys):.1f}] "
                      f"z=[{min(zs):.1f},{max(zs):.1f}] "
                      f"area~{area:.0f}m²")
    except:
        pass

# --- Elevator (from platform IFC, no /1000 this time) ---
print("\n=== Elevator geometry (raw meters) ===")
m1 = ifcopenshell.open(r"e:\TUD-Thesis\station\data0\站台层.ifc")
for p in m1.by_type("IfcBuildingElementProxy"):
    if "电梯" in (p.Name or ""):
        try:
            shape = ifcopenshell.geom.create_shape(settings, p)
            verts = shape.geometry.verts
            if verts:
                xs = [verts[i] for i in range(0, len(verts), 3)]
                ys = [verts[i+1] for i in range(0, len(verts), 3)]
                zs = [verts[i+2] for i in range(0, len(verts), 3)]
                print(f"  '{p.Name}' ({p.GlobalId[:8]}): "
                      f"x=[{min(xs):.2f},{max(xs):.2f}] "
                      f"y=[{min(ys):.2f},{max(ys):.2f}] "
                      f"z=[{min(zs):.2f},{max(zs):.2f}] "
                      f"n_verts={len(xs)}")
        except Exception as e:
            print(f"  '{p.Name}': FAIL - {e}")

# --- Escalator list (all, from all files) ---
print("\n=== All escalators (unique) ===")
seen = set()
for label, path in [
    ("platform", r"e:\TUD-Thesis\station\data0\站台层.ifc"),
    ("concourse", r"e:\TUD-Thesis\station\data0\站厅层.ifc"),
    ("traffic", r"e:\TUD-Thesis\station\data0\交通层.ifc"),
]:
    m = ifcopenshell.open(path) if label != "platform" else m1
    if label == "concourse": m = m3
    if label == "traffic": m = m4
    for p in m.by_type("IfcBuildingElementProxy"):
        if "扶梯" in (p.Name or "") and p.GlobalId not in seen:
            seen.add(p.GlobalId)
            try:
                shape = ifcopenshell.geom.create_shape(settings, p)
                verts = shape.geometry.verts
                if verts:
                    xs = [verts[i] for i in range(0, len(verts), 3)]
                    ys = [verts[i+1] for i in range(0, len(verts), 3)]
                    zs = [verts[i+2] for i in range(0, len(verts), 3)]
                    print(f"  [{label}] '{p.Name}' ({p.GlobalId[:8]}): "
                          f"x=[{min(xs):.1f},{max(xs):.1f}] "
                          f"y=[{min(ys):.1f},{max(ys):.1f}] "
                          f"z=[{min(zs):.1f},{max(zs):.1f}]")
            except:
                pass

print("\nDone!")
