"""Debug: get bbox coords for fare gates and security scanners."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import ifcopenshell, ifcopenshell.geom

DATA_ROOT = ROOT.parent / "data-preprocessing"
BBOX_CSV  = DATA_ROOT / "outputs" / "v2" / "normalized" / "bbox_samples_metres.csv"
RET_CSV   = DATA_ROOT / "outputs" / "v2" / "traffic_filtered" / "retained_elements.csv"

bbox_df = pd.read_csv(BBOX_CSV)
ret_df  = pd.read_csv(RET_CSV)

# Merge bbox
merged = ret_df.merge(
    bbox_df[["guid", "source_file", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z", "dx", "dy", "dz"]].drop_duplicates(subset=["guid", "source_file"]),
    on=["guid", "source_file"], how="left"
)

# ---- Fare gates ----
gates = merged[merged["name"].fillna("").str.contains("闸机")]
print("=== FARE GATES (闸机) ===")
print(f"Total: {len(gates)}, with bbox: {gates['min_x'].notna().sum()}")
if gates['min_x'].notna().any():
    g = gates[gates['min_x'].notna()]
    print(f"  X range: [{g['min_x'].min():.1f}, {g['max_x'].max():.1f}]")
    print(f"  Y range: [{g['min_y'].min():.1f}, {g['max_y'].max():.1f}]")
    print(f"  Z range: [{g['min_z'].min():.1f}, {g['max_z'].max():.1f}]")
    print(f"  Typical size: dx={g['dx'].median():.2f}, dy={g['dy'].median():.2f}, dz={g['dz'].median():.2f}")
    # Show a few
    print(g[["guid", "name", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]].head(5).to_string(index=False))

# Try IFC extraction for gates without bbox
gates_no_bbox = gates[gates['min_x'].isna()]
print(f"\n  Gates WITHOUT bbox: {len(gates_no_bbox)}")

# ---- Security scanners ----
scanners = merged[merged["name"].fillna("").str.contains("安检")]
print(f"\n=== SECURITY SCANNERS (安检机) ===")
print(f"Total: {len(scanners)}, with bbox: {scanners['min_x'].notna().sum()}")
if scanners['min_x'].notna().any():
    s = scanners[scanners['min_x'].notna()]
    print(f"  X range: [{s['min_x'].min():.1f}, {s['max_x'].max():.1f}]")
    print(f"  Y range: [{s['min_y'].min():.1f}, {s['max_y'].max():.1f}]")
    print(f"  Z range: [{s['min_z'].min():.1f}, {s['max_z'].max():.1f}]")
    print(f"  Typical size: dx={s['dx'].median():.2f}, dy={s['dy'].median():.2f}, dz={s['dz'].median():.2f}")
    print(s[["guid", "name", "min_x", "max_x", "min_y", "max_y"]].to_string(index=False))

# Try IFC direct for those without bbox
scanners_no_bbox = scanners[scanners['min_x'].isna()]
print(f"\n  Scanners WITHOUT bbox: {len(scanners_no_bbox)}")

# ---- Extract from raw IFC for those without bbox ----
print("\n=== IFC DIRECT EXTRACTION ===")
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

model = ifcopenshell.open(str(ROOT.parent / "data0" / "站厅层.ifc"))
for product in model.by_type("IfcBuildingElementProxy"):
    pname = product.Name or ""
    if "闸机" in pname or "安检" in pname:
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = shape.geometry.verts
            if verts:
                xs = [verts[i] for i in range(0, len(verts), 3)]
                ys = [verts[i+1] for i in range(0, len(verts), 3)]
                zs = [verts[i+2] for i in range(0, len(verts), 3)]
                print(f"  '{pname}' ({product.GlobalId[:8]}): "
                      f"x=[{min(xs):.1f},{max(xs):.1f}] "
                      f"y=[{min(ys):.1f},{max(ys):.1f}] "
                      f"z=[{min(zs):.1f},{max(zs):.1f}] "
                      f"dx={max(xs)-min(xs):.2f} dy={max(ys)-min(ys):.2f}")
        except Exception as e:
            print(f"  '{pname}': FAIL - {e}")

# ---- Check what obstacle_subcat gates/scanners have ----
obs_df = pd.read_csv(DATA_ROOT / "outputs" / "v3" / "obstacle_recalibration" / "obstacles_recalibrated.csv")
print("\n=== Obstacle subcategory of gates/scanners ===")
for kw in ["闸机", "安检"]:
    mask = obs_df["name"].fillna("").str.contains(kw)
    if mask.any():
        hits = obs_df[mask]
        print(f"  '{kw}' in obstacles: {len(hits)}")
        print(hits["obstacle_subcat"].value_counts().to_string())
    else:
        print(f"  '{kw}' NOT found in obstacles CSV")

# Check in connector CSV
conn_df = pd.read_csv(DATA_ROOT / "outputs" / "v3" / "connector_validation" / "connectors_validated.csv")
for kw in ["闸机", "安检"]:
    mask = conn_df["name"].fillna("").str.contains(kw)
    if mask.any():
        hits = conn_df[mask]
        print(f"\n  '{kw}' in connectors: {len(hits)}")
        print(hits["connector_subtype"].value_counts().to_string())

print("\nDone!")
