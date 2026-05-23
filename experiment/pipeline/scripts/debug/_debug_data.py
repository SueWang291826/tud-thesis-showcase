"""Debug script: examine raw data for connectors and floors."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

prep = ROOT.parent / "data-preprocessing"

conn = pd.read_csv(prep / "outputs/v3/connector_validation/connectors_validated.csv")
bbox = pd.read_csv(prep / "outputs/v2/normalized/bbox_samples_metres.csv")
retained = pd.read_csv(prep / "outputs/v2/traffic_filtered/retained_elements.csv")

print("=" * 60)
print("ESCALATOR DETAILS")
print("=" * 60)
esc = conn[conn["connector_subtype"] == "escalator"]
esc_guids = esc["guid"].unique()
print(f"Unique escalator GUIDs: {len(esc_guids)}")
for g in esc_guids:
    rows = bbox[bbox["guid"] == g]
    name = esc[esc["guid"] == g].iloc[0]["name"]
    if len(rows) > 0:
        r = rows.iloc[0]
        print(f"  {name}")
        print(f"    bbox: x=[{r['min_x']:.1f}, {r['max_x']:.1f}] y=[{r['min_y']:.1f}, {r['max_y']:.1f}] z=[{r['min_z']:.1f}, {r['max_z']:.1f}]")
    else:
        print(f"  {name}  -- NO BBOX")

print()
print("=" * 60)
print("ELEVATOR DETAILS")
print("=" * 60)
elev = conn[conn["connector_subtype"] == "elevator"]
elev_guids = elev["guid"].unique()
print(f"Unique elevator GUIDs: {len(elev_guids)}")
for g in elev_guids:
    rows = bbox[bbox["guid"] == g]
    name = elev[elev["guid"] == g].iloc[0]["name"]
    if len(rows) > 0:
        r = rows.iloc[0]
        print(f"  {name}")
        print(f"    bbox: x=[{r['min_x']:.1f}, {r['max_x']:.1f}] y=[{r['min_y']:.1f}, {r['max_y']:.1f}] z=[{r['min_z']:.1f}, {r['max_z']:.1f}]")
    else:
        print(f"  {name}  -- NO BBOX")

print()
print("=" * 60)
print("FLOORS (IfcSlab) IN RETAINED DATA")
print("=" * 60)
slabs = retained[retained["ifc_class"] == "IfcSlab"]
print(f"Total IfcSlab: {len(slabs)}")
for st in sorted(slabs["storey_name"].unique()):
    sub = slabs[slabs["storey_name"] == st]
    print(f"  {st}: {len(sub)}")
    for _, r in sub.head(3).iterrows():
        print(f"    name={r['name'][:50]}")

# Check what IfcSlab looks like in bbox
slab_guids = slabs["guid"].unique()
slab_bbox = bbox[bbox["guid"].isin(slab_guids)]
print(f"\nSlabs with bbox: {len(slab_bbox)}")
for st in sorted(slabs["storey_name"].unique()):
    sub_slabs = slabs[slabs["storey_name"] == st]
    sub_bbox = slab_bbox[slab_bbox["guid"].isin(sub_slabs["guid"])]
    if len(sub_bbox) > 0:
        area_sum = ((sub_bbox["max_x"] - sub_bbox["min_x"]) * (sub_bbox["max_y"] - sub_bbox["min_y"])).sum()
        print(f"  {st}: {len(sub_bbox)} bbox slabs, total XY area ~ {area_sum:.0f} m²")

print()
print("=" * 60)
print("UNIQUE IFC CLASSES IN RETAINED BY STOREY")
print("=" * 60)
for st in sorted(retained["storey_name"].unique()):
    sub = retained[retained["storey_name"] == st]
    classes = sub["ifc_class"].value_counts()
    print(f"\n  {st} ({len(sub)} total):")
    for cls, cnt in classes.items():
        print(f"    {cls}: {cnt}")

print()
print("=" * 60)
print("STAIR DETAILS - sample per storey")
print("=" * 60)
stairs = conn[conn["connector_subtype"].isin(["stair"])]
for st in sorted(stairs["storey_name"].unique()):
    sub = stairs[stairs["storey_name"] == st]
    print(f"\n  {st}: {len(sub)} stairs")
    for _, r in sub.head(3).iterrows():
        g = r["guid"]
        bb = bbox[bbox["guid"] == g]
        if len(bb) > 0:
            b = bb.iloc[0]
            print(f"    {r['name'][:50]}")
            print(f"      z=[{b['min_z']:.1f}, {b['max_z']:.1f}]  src={r['source_file']}")
        else:
            print(f"    {r['name'][:50]}  -- NO BBOX")
