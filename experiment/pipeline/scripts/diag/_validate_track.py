"""Verify F1 track zones are properly forbidden."""
import pickle, networkx as nx

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    g = pickle.load(f)

f1_floor = [(n, d) for n, d in g.nodes(data=True)
            if d.get("level") == "F1" and d.get("node_type") == "floor"]

south_track = [(n, d) for n, d in f1_floor if d["y"] < 3.5]
north_track = [(n, d) for n, d in f1_floor if d["y"] > 17.8]
platform    = [(n, d) for n, d in f1_floor if 3.5 <= d["y"] <= 17.8]

print(f"F1 floor nodes: {len(f1_floor)} total")
print(f"  south track zone (y<3.5)  : {len(south_track)}")
print(f"  north track zone (y>17.8) : {len(north_track)}")
print(f"  platform zone             : {len(platform)}")

if south_track:
    print("  WARN: unexpected nodes in south track:")
    for n, d in south_track[:5]:
        print(f"    ({d['x']:.2f},{d['y']:.2f})")

if north_track:
    print("  WARN: unexpected nodes in north track:")
    for n, d in north_track[:5]:
        print(f"    ({d['x']:.2f},{d['y']:.2f})")

print("\nDone.")
