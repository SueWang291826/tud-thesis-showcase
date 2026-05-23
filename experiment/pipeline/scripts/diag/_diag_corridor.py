"""Detailed corridor node check after rebuild."""
import pickle, networkx as nx, numpy as np

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    g = pickle.load(f)

# Fine-grained check of F3 left unpaid corridor
f3_floor = [(d["x"], d["y"]) for _, d in g.nodes(data=True)
            if d.get("level") == "F3" and d.get("node_type") == "floor"]
f3a = np.array(f3_floor) if f3_floor else np.empty((0,2))

print("=== F3 left unpaid corridor detail (x=44-57, y=7.5-15) ===")
for x0, x1 in [(44, 46.1), (46.1, 49), (49, 52), (52, 55), (55, 57)]:
    for y0, y1 in [(7.5, 10), (10, 12.5), (12.5, 15)]:
        mask = (f3a[:,0]>=x0) & (f3a[:,0]<x1) & (f3a[:,1]>=y0) & (f3a[:,1]<y1)
        cnt = mask.sum()
        label = "OK" if cnt > 5 else ("SPARSE" if cnt > 0 else "EMPTY")
        print(f"  x={x0:.1f}-{x1:.1f}, y={y0:.1f}-{y1:.1f}: {cnt:3d} nodes  [{label}]")

print()
print("=== F3 right unpaid corridor detail (x=95-121, y=7.5-15) ===")
for x0, x1 in [(95, 100), (100, 105), (105, 112), (112, 118), (118, 121)]:
    for y0, y1 in [(7.5, 10), (10, 12.5), (12.5, 15)]:
        mask = (f3a[:,0]>=x0) & (f3a[:,0]<x1) & (f3a[:,1]>=y0) & (f3a[:,1]<y1)
        cnt = mask.sum()
        label = "OK" if cnt > 5 else ("SPARSE" if cnt > 0 else "EMPTY")
        print(f"  x={x0:.1f}-{x1:.1f}, y={y0:.1f}-{y1:.1f}: {cnt:3d} nodes  [{label}]")

print()
print("=== F4 SE corner detail (x=110-134, y=6-14) ===")
f4_floor = [(d["x"], d["y"]) for _, d in g.nodes(data=True)
            if d.get("level") == "F4" and d.get("node_type") == "floor"]
f4a = np.array(f4_floor) if f4_floor else np.empty((0,2))
for x0, x1 in [(110, 118), (118, 122), (122, 128), (128, 131)]:
    for y0, y1 in [(6, 9), (9, 11), (11, 14)]:
        mask = (f4a[:,0]>=x0) & (f4a[:,0]<x1) & (f4a[:,1]>=y0) & (f4a[:,1]<y1)
        cnt = mask.sum()
        label = "OK" if cnt > 5 else ("SPARSE" if cnt > 0 else "EMPTY")
        print(f"  x={x0:.1f}-{x1:.1f}, y={y0:.1f}-{y1:.1f}: {cnt:3d} nodes  [{label}]")

print("\nDone.")
