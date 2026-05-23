"""Check railing obstacle coverage."""
import pandas as pd
from pathlib import Path

prep = Path(__file__).resolve().parent.parent.parent.parent / "data-preprocessing"
obs = pd.read_csv(prep / "outputs/v3/obstacle_recalibration/obstacles_recalibrated.csv")
bbox = pd.read_csv(prep / "outputs/v2/normalized/bbox_samples_metres.csv")

if "min_x" not in obs.columns:
    obs = obs.merge(
        bbox[["guid","source_file","min_x","max_x","min_y","max_y","min_z","max_z"]]
        .drop_duplicates(subset=["guid","source_file"]),
        on=["guid","source_file"], how="left",
    )

rail_obs = obs[obs["ifc_class"] == "IfcRailing"]
print(f"IfcRailing in obstacle CSV: {len(rail_obs)}")
has_bbox = rail_obs.dropna(subset=["min_x"])
print(f"  with bbox: {len(has_bbox)}")
print(f"  per storey: {rail_obs['storey_name'].value_counts().to_dict()}")

keep = ["obstacle_floor_intrusive", "obstacle_barrier_relevant", "obstacle_clearance_relevant"]
kept = rail_obs[rail_obs["obstacle_subcat"].isin(keep)]
print(f"  kept by filter: {len(kept)}")
no_bbox = kept[kept["min_x"].isna()]
print(f"  kept BUT no bbox: {len(no_bbox)}")
if not no_bbox.empty:
    print(f"  Storeys missing bbox: {no_bbox['storey_name'].value_counts().to_dict()}")
    print(no_bbox[["guid", "name", "storey_name"]].to_string())
