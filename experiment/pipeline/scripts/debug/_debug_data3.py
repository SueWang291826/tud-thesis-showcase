"""Debug: F1/F3 slab details and IFC direct inspection."""
import pandas as pd
from pathlib import Path

prep = Path("E:/TUD-Thesis/station/data-preprocessing")
bbox = pd.read_csv(prep / "outputs/v2/normalized/bbox_samples_metres.csv")
ret = pd.read_csv(prep / "outputs/v2/traffic_filtered/retained_elements.csv")

# F1 slabs
f1s = ret[(ret["storey_name"].str.contains("F1")) & (ret["ifc_class"] == "IfcSlab")]
print("=== F1 SLABS ===")
for _, r in f1s.iterrows():
    bb = bbox[bbox["guid"] == r["guid"]]
    if len(bb) > 0:
        b = bb.iloc[0]
        area = (b["max_x"] - b["min_x"]) * (b["max_y"] - b["min_y"])
        print(f"  {r['name'][:45]} area={area:.0f}m2 z=[{b['min_z']:.2f},{b['max_z']:.2f}]")
    else:
        print(f"  {r['name'][:45]} NO BBOX")

print()
# F3 slabs
f3s = ret[(ret["storey_name"].str.contains("F3")) & (ret["ifc_class"] == "IfcSlab")]
print("=== F3 SLABS ===")
for _, r in f3s.iterrows():
    bb = bbox[bbox["guid"] == r["guid"]]
    if len(bb) > 0:
        b = bb.iloc[0]
        area = (b["max_x"] - b["min_x"]) * (b["max_y"] - b["min_y"])
        print(f"  {r['name'][:45]} area={area:.0f}m2 z=[{b['min_z']:.2f},{b['max_z']:.2f}]")
    else:
        print(f"  {r['name'][:45]} NO BBOX")

# Now try to open IFC directly for elevator
print()
print("=== IFC direct inspection for elevator ===")
try:
    import ifcopenshell
    ifc = ifcopenshell.open(str(prep / "../data0/站台层.ifc"))
    
    # Search for elevator
    for el in ifc.by_type("IfcBuildingElementProxy"):
        name = el.Name or ""
        if "电梯" in name or "elevator" in name.lower():
            print(f"  Found: {name} | GlobalId={el.GlobalId}")
            # Try to get placement/shape
            if hasattr(el, "ObjectPlacement"):
                print(f"    Has ObjectPlacement")
            if hasattr(el, "Representation"):
                print(f"    Has Representation: {el.Representation is not None}")
except Exception as e:
    print(f"  Error: {e}")

# Check how many unique escalator elements we really have (by guid)
conn = pd.read_csv(prep / "outputs/v3/connector_validation/connectors_validated.csv")
esc = conn[conn["connector_subtype"] == "escalator"]
print()
print("=== Escalator unique GUIDs ===")
print(f"  Total rows: {len(esc)}, unique GUIDs: {esc['guid'].nunique()}")
for g in esc["guid"].unique():
    sub = esc[esc["guid"] == g]
    sources = sub["source_file"].unique()
    print(f"  {g[:30]}: appears in {list(sources)}")

# Escalator z-range interpretation
print()
print("=== Escalator level connections ===")
# z=[-1.2, 13.1] for 6.2m type → spans F1(0m) through F2(5.3m) to F3(12.1m)
# z=[10.9, 18.4] for 5.3m type → spans F3(12.1m) to F4(17.4m)
for g in esc["guid"].unique():
    bb = bbox[bbox["guid"] == g]
    if len(bb) > 0:
        b = bb.iloc[0]
        name = esc[esc["guid"] == g].iloc[0]["name"]
        print(f"  {name[:40]}: z=[{b['min_z']:.1f},{b['max_z']:.1f}]")
        if b["min_z"] < 1 and b["max_z"] > 12:
            print(f"    => connects F1→F3 (through F2)")
        elif b["min_z"] > 10 and b["max_z"] > 17:
            print(f"    => connects F3→F4")
