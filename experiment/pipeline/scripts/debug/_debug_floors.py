"""
Debug: investigate floor slab extraction from raw IFC files.

Goal: determine if we can extract proper floor polygons from IFC directly,
even for F1/F3 whose IfcSlab elements have no bbox in the CSV.

Also explore: alpha shape / convex hull of all level elements as alternative.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import ifcopenshell
import ifcopenshell.geom

# ---- Config ----
DATA_ROOT = ROOT.parent / "data-preprocessing"
RAW_IFC = {
    "platform": DATA_ROOT / ".." / "data0" / "站台层.ifc",
    "concourse": DATA_ROOT / ".." / "data0" / "站厅层.ifc",
    "traffic": DATA_ROOT / ".." / "data0" / "交通层.ifc",
}
SUBSET_IFC = {
    "F1": DATA_ROOT / "outputs" / "v3" / "ifc_subsets" / "platform_F1_public.ifc",
    "F3": DATA_ROOT / "outputs" / "v3" / "ifc_subsets" / "concourse_F3_public.ifc",
    "F4": DATA_ROOT / "outputs" / "v3" / "ifc_subsets" / "traffic_F4_public.ifc",
}
BBOX_CSV = DATA_ROOT / "outputs" / "v2" / "normalized" / "bbox_samples_metres.csv"

bbox_df = pd.read_csv(BBOX_CSV)

# =================================================================
# Approach 1: Try extracting IfcSlab geometry from RAW IFC for F1
# =================================================================
print("=" * 70)
print("APPROACH 1: Extract IfcSlab from RAW platform IFC")
print("=" * 70)

model = ifcopenshell.open(str(RAW_IFC["platform"]))
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

slabs = model.by_type("IfcSlab")
print(f"  IfcSlab count in raw platform IFC: {len(slabs)}")

for slab in slabs[:10]:
    name = slab.Name or ""
    has_rep = slab.Representation is not None
    try:
        shape = ifcopenshell.geom.create_shape(settings, slab)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            # Convert from mm to m
            xs_m = [x/1000 for x in xs]
            ys_m = [y/1000 for y in ys]
            zs_m = [z/1000 for z in zs]
            area_xy = (max(xs_m)-min(xs_m)) * (max(ys_m)-min(ys_m))
            print(f"  Slab '{name}' ({slab.GlobalId[:8]}): "
                  f"x=[{min(xs_m):.1f},{max(xs_m):.1f}] "
                  f"y=[{min(ys_m):.1f},{max(ys_m):.1f}] "
                  f"z=[{min(zs_m):.1f},{max(zs_m):.1f}] "
                  f"bbox_area~{area_xy:.0f}m² verts={len(xs)}")
        else:
            print(f"  Slab '{name}': empty verts")
    except Exception as e:
        print(f"  Slab '{name}': FAIL - {e}")

# =================================================================
# Approach 2: Convex hull of all retained elements per level
# =================================================================
print("\n" + "=" * 70)
print("APPROACH 2: Convex hull from retained element bboxes")
print("=" * 70)

retained_df = pd.read_csv(DATA_ROOT / "outputs" / "v2" / "traffic_filtered" / "retained_elements.csv")
merged = retained_df.merge(
    bbox_df[["guid", "source_file", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]].drop_duplicates(subset=["guid", "source_file"]),
    on=["guid", "source_file"],
    how="left",
)

for storey in ["F1 站台层", "F3 站厅层", "F4 交通层"]:
    level = merged[merged["storey_name"] == storey]
    has_bbox = level["min_x"].notna()
    n_total = len(level)
    n_bbox = int(has_bbox.sum())
    if n_bbox > 0:
        minx = level.loc[has_bbox, "min_x"].min()
        maxx = level.loc[has_bbox, "max_x"].max()
        miny = level.loc[has_bbox, "min_y"].min()
        maxy = level.loc[has_bbox, "max_y"].max()
        area = (maxx - minx) * (maxy - miny)
        print(f"  {storey}: {n_bbox}/{n_total} have bbox → "
              f"x=[{minx:.1f},{maxx:.1f}] y=[{miny:.1f},{maxy:.1f}] "
              f"area~{area:.0f}m²")
    else:
        print(f"  {storey}: {n_bbox}/{n_total} have bbox → NO DATA")

# =================================================================
# Approach 3: Try IfcSpace from IFC subset files
# =================================================================
print("\n" + "=" * 70)
print("APPROACH 3: IfcSpace from subset IFC files")
print("=" * 70)

for level_key, ifc_path in SUBSET_IFC.items():
    if not ifc_path.exists():
        print(f"  {level_key}: file not found")
        continue
    model = ifcopenshell.open(str(ifc_path))
    spaces = model.by_type("IfcSpace")
    print(f"  {level_key}: {len(spaces)} IfcSpace elements")
    for sp in spaces[:5]:
        print(f"    Space: {sp.Name or 'unnamed'}, LongName={sp.LongName or ''}")

# =================================================================
# Approach 4: Try extracting from raw IFC - all IfcSlab with large area
# =================================================================
print("\n" + "=" * 70)
print("APPROACH 4: Large IfcSlab from RAW concourse IFC (F3)")
print("=" * 70)

model3 = ifcopenshell.open(str(RAW_IFC["concourse"]))
slabs3 = model3.by_type("IfcSlab")
print(f"  IfcSlab count in raw concourse IFC: {len(slabs3)}")

for slab in slabs3:
    name = slab.Name or ""
    try:
        shape = ifcopenshell.geom.create_shape(settings, slab)
        verts = shape.geometry.verts
        if verts:
            xs = [verts[i]/1000 for i in range(0, len(verts), 3)]
            ys = [verts[i+1]/1000 for i in range(0, len(verts), 3)]
            zs = [verts[i+2]/1000 for i in range(0, len(verts), 3)]
            area_xy = (max(xs)-min(xs)) * (max(ys)-min(ys))
            if area_xy > 50:  # Only show large slabs
                print(f"  Slab '{name}' ({slab.GlobalId[:8]}): "
                      f"x=[{min(xs):.1f},{max(xs):.1f}] "
                      f"y=[{min(ys):.1f},{max(ys):.1f}] "
                      f"z=[{min(zs):.1f},{max(zs):.1f}] "
                      f"bbox_area~{area_xy:.0f}m² verts={len(xs)}")
    except Exception as e:
        pass  # Skip failures silently for batch

# =================================================================
# Approach 5: Extract elevator geometry from raw IFC
# =================================================================
print("\n" + "=" * 70)
print("APPROACH 5: Elevator geometry from RAW IFC")
print("=" * 70)

for src_name, ifc_path in RAW_IFC.items():
    model = ifcopenshell.open(str(ifc_path))
    for product in model.by_type("IfcProduct"):
        pname = product.Name or ""
        if "电梯" in pname:
            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
                verts = shape.geometry.verts
                if verts:
                    xs = [verts[i]/1000 for i in range(0, len(verts), 3)]
                    ys = [verts[i+1]/1000 for i in range(0, len(verts), 3)]
                    zs = [verts[i+2]/1000 for i in range(0, len(verts), 3)]
                    print(f"  [{src_name}] '{pname}' ({product.is_a()}): "
                          f"x=[{min(xs):.1f},{max(xs):.1f}] "
                          f"y=[{min(ys):.1f},{max(ys):.1f}] "
                          f"z=[{min(zs):.1f},{max(zs):.1f}] "
                          f"verts={len(xs)}")
                else:
                    print(f"  [{src_name}] '{pname}': empty verts")
            except Exception as e:
                print(f"  [{src_name}] '{pname}': FAIL - {type(e).__name__}: {e}")

print("\nDone!")
