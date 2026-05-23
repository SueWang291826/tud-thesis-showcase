"""Debug: stair flights with bbox, elevator search."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
prep = ROOT.parent / "data-preprocessing"

conn = pd.read_csv(prep / "outputs/v3/connector_validation/connectors_validated.csv")
bbox = pd.read_csv(prep / "outputs/v2/normalized/bbox_samples_metres.csv")
ret = pd.read_csv(prep / "outputs/v2/traffic_filtered/retained_elements.csv")

# Stair flights WITH bbox
flights = conn[conn["connector_subtype"] == "stair_flight"]
print("=== STAIR FLIGHTS with bbox ===")
for st in sorted(flights["storey_name"].unique()):
    sub = flights[flights["storey_name"] == st]
    has_bbox = sum(1 for _, r in sub.iterrows() if len(bbox[bbox["guid"] == r["guid"]]) > 0)
    print(f"  {st}: {len(sub)} flights, {has_bbox} with bbox")

# Sample stair flight bbox
print("\nSample stair flight bboxes (F1):")
f1_flights = flights[flights["storey_name"].str.contains("F1")]
for _, r in f1_flights.head(5).iterrows():
    bb = bbox[bbox["guid"] == r["guid"]]
    if len(bb) > 0:
        b = bb.iloc[0]
        print(f"  {r['name'][:40]}  z=[{b['min_z']:.2f},{b['max_z']:.2f}]  dx={b['max_x']-b['min_x']:.1f}  dy={b['max_y']-b['min_y']:.1f}")
    else:
        print(f"  {r['name'][:40]}  NO BBOX")

# Check elevator elements in retained by keyword
print("\n=== Searching for elevator elements ===")
for kw in ["电梯", "elevator", "lift"]:
    matches = ret[ret["name"].str.contains(kw, na=False, case=False)]
    if len(matches) > 0:
        print(f"\n  Keyword '{kw}': {len(matches)} matches")
        for _, r in matches.iterrows():
            bb = bbox[bbox["guid"] == r["guid"]]
            if len(bb) > 0:
                b = bb.iloc[0]
                print(f"    {r['name'][:50]} | {r['ifc_class']} | {r['storey_name']}")
                print(f"      bbox: x=[{b['min_x']:.1f},{b['max_x']:.1f}] y=[{b['min_y']:.1f},{b['max_y']:.1f}] z=[{b['min_z']:.1f},{b['max_z']:.1f}]")
            else:
                print(f"    {r['name'][:50]} | {r['ifc_class']} | {r['storey_name']}  NO BBOX")

# Check for escalator elements in retained
print("\n=== Searching for escalator elements ===")
for kw in ["自动扶梯", "扶梯", "escalator"]:
    matches = ret[ret["name"].str.contains(kw, na=False, case=False)]
    if len(matches) > 0:
        print(f"\n  Keyword '{kw}': {len(matches)} matches")
        for _, r in matches.head(8).iterrows():
            bb = bbox[bbox["guid"] == r["guid"]]
            if len(bb) > 0:
                b = bb.iloc[0]
                print(f"    {r['name'][:50]} | {r['ifc_class']} | {r['storey_name']}")
                print(f"      bbox: z=[{b['min_z']:.1f},{b['max_z']:.1f}]")
            else:
                print(f"    {r['name'][:50]} | {r['ifc_class']} | {r['storey_name']}  NO BBOX")

# F4 slab details
print("\n=== F4 SLAB DETAILS ===")
f4_slabs = ret[(ret["storey_name"].str.contains("F4")) & (ret["ifc_class"] == "IfcSlab")]
for _, r in f4_slabs.iterrows():
    bb = bbox[bbox["guid"] == r["guid"]]
    if len(bb) > 0:
        b = bb.iloc[0]
        area = (b["max_x"] - b["min_x"]) * (b["max_y"] - b["min_y"])
        print(f"  {r['name'][:50]}")
        print(f"    bbox: x=[{b['min_x']:.1f},{b['max_x']:.1f}] y=[{b['min_y']:.1f},{b['max_y']:.1f}] z=[{b['min_z']:.1f},{b['max_z']:.1f}] area={area:.0f}m²")
