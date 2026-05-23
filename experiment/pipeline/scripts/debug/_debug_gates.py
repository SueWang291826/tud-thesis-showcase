"""
Debug: find fare gates (闸机), security scanners (安检机), and enclosed rooms
in the raw IFC files and preprocessing CSV data.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/debug → experiment
sys.path.insert(0, str(ROOT))

import pandas as pd

DATA_ROOT = ROOT.parent / "data-preprocessing"
BBOX_CSV  = DATA_ROOT / "outputs" / "v2" / "normalized" / "bbox_samples_metres.csv"
RET_CSV   = DATA_ROOT / "outputs" / "v2" / "traffic_filtered" / "retained_elements.csv"
OBS_CSV   = DATA_ROOT / "outputs" / "v3" / "obstacle_recalibration" / "obstacles_recalibrated.csv"
CONN_CSV  = DATA_ROOT / "outputs" / "v3" / "connector_validation" / "connectors_validated.csv"

bbox_df = pd.read_csv(BBOX_CSV)
ret_df  = pd.read_csv(RET_CSV)
obs_df  = pd.read_csv(OBS_CSV)
conn_df = pd.read_csv(CONN_CSV)

# ---- 1. Search for fare gate / security scanner keywords ----
keywords = ["闸机", "安检", "检票", "gate", "turnstile", "scanner", "security", "fare"]

print("=" * 70)
print("1. KEYWORD SEARCH in retained elements")
print("=" * 70)

for kw in keywords:
    mask = ret_df["name"].fillna("").str.contains(kw, case=False)
    if mask.any():
        hits = ret_df[mask]
        print(f"\n  '{kw}': {len(hits)} hits")
        print(hits[["guid", "name", "ifc_class", "storey_name"]].head(20).to_string(index=False))

# Also search in object_type column if it exists
if "object_type" in ret_df.columns:
    print("\n--- object_type column ---")
    for kw in keywords:
        mask = ret_df["object_type"].fillna("").str.contains(kw, case=False)
        if mask.any():
            hits = ret_df[mask]
            print(f"\n  '{kw}' (object_type): {len(hits)} hits")
            print(hits[["guid", "name", "ifc_class", "storey_name", "object_type"]].head(10).to_string(index=False))

# ---- 2. Search for room-related keywords ----
print("\n" + "=" * 70)
print("2. ROOM / ENCLOSED SPACE keywords")
print("=" * 70)

room_kw = ["房间", "房", "厕所", "卫生间", "办公", "机房", "配电", "值班", "警务",
           "room", "toilet", "wc", "office", "control"]

for kw in room_kw:
    mask = ret_df["name"].fillna("").str.contains(kw, case=False)
    if mask.any():
        hits = ret_df[mask]
        print(f"\n  '{kw}': {len(hits)} hits")
        print(hits[["guid", "name", "ifc_class", "storey_name"]].head(10).to_string(index=False))

# ---- 3. Check IfcDoor data (doors define room boundaries) ----
print("\n" + "=" * 70)
print("3. IfcDoor summary per storey")
print("=" * 70)

doors = ret_df[ret_df["ifc_class"] == "IfcDoor"]
print(f"  Total IfcDoor: {len(doors)}")
door_by_storey = doors.groupby("storey_name").size()
print(door_by_storey.to_string())

# Sample door names
print("\n  Sample door names:")
print(doors["name"].value_counts().head(20).to_string())

# ---- 4. Check IfcWall and IfcCurtainWall ----
print("\n" + "=" * 70)
print("4. Wall types per storey")
print("=" * 70)

for cls in ["IfcWall", "IfcWallStandardCase", "IfcCurtainWall"]:
    walls = ret_df[ret_df["ifc_class"] == cls]
    if len(walls) > 0:
        print(f"\n  {cls}: {len(walls)}")
        w_by_storey = walls.groupby("storey_name").size()
        print("    " + w_by_storey.to_string().replace("\n", "\n    "))

# ---- 5. Search for fare gates / security in IFC directly ----
print("\n" + "=" * 70)
print("5. Direct IFC search for 闸机/安检/检票")
print("=" * 70)

import ifcopenshell

for label, path in [
    ("concourse", str(ROOT.parent / "data0" / "站厅层.ifc")),
    ("platform", str(ROOT.parent / "data0" / "站台层.ifc")),
    ("traffic", str(ROOT.parent / "data0" / "交通层.ifc")),
]:
    model = ifcopenshell.open(path)
    found = []
    for product in model.by_type("IfcProduct"):
        pname = (product.Name or "")
        ptype = (product.ObjectType or "") if hasattr(product, "ObjectType") else ""
        combined = pname + " " + ptype
        if any(kw in combined for kw in ["闸机", "安检", "检票", "gate", "turnstile"]):
            found.append((product.GlobalId[:8], product.is_a(), pname[:50], ptype[:30]))
    
    print(f"\n  [{label}] {len(found)} matches:")
    for guid, cls, nm, ot in found[:20]:
        print(f"    {guid} {cls:30s} {nm:50s} {ot}")

# ---- 6. IfcSpace in raw IFC ----
print("\n" + "=" * 70)
print("6. IfcSpace / IfcZone in raw IFC")
print("=" * 70)

for label, path in [
    ("concourse", str(ROOT.parent / "data0" / "站厅层.ifc")),
    ("platform", str(ROOT.parent / "data0" / "站台层.ifc")),
]:
    model = ifcopenshell.open(path)
    for stype in ["IfcSpace", "IfcZone"]:
        items = model.by_type(stype)
        print(f"  [{label}] {stype}: {len(items)}")
        for item in items[:5]:
            print(f"    {item.Name or 'unnamed'}, LongName={item.LongName or ''}")

print("\nDone!")
