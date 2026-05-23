"""Diagnostic: identify missing node areas and wrong forbidden zones."""
import pickle, networkx as nx, numpy as np

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    g = pickle.load(f)

# ── F3 node density heatmap in key zones ──
f3_floor = [(d["x"], d["y"]) for _, d in g.nodes(data=True)
            if d.get("level") == "F1" and d.get("node_type") == "floor"]
f3_pts = np.array(f3_floor) if f3_floor else np.empty((0,2))

print("=== F1 platform floor node gaps ===")
# Check x=57-85, y=3.5-17.8 in 2m bands
for y0 in np.arange(3.5, 18.0, 2.0):
    y1 = y0 + 2.0
    for x0 in [57, 65, 69, 75, 80]:
        x1 = x0 + 8
        mask = (f3_pts[:,0]>=x0) & (f3_pts[:,0]<x1) & (f3_pts[:,1]>=y0) & (f3_pts[:,1]<y1)
        cnt = mask.sum()
        if cnt < 10:
            print(f"  F1 x={x0:.0f}-{x1:.0f}, y={y0:.1f}-{y1:.1f}: {cnt} nodes (SPARSE)")

print()
print("=== F3 floor node density by zone ===")
f3fl = [(d["x"], d["y"]) for _, d in g.nodes(data=True)
        if d.get("level") == "F3" and d.get("node_type") == "floor"]
f3a = np.array(f3fl) if f3fl else np.empty((0,2))

zones = [
    ("SW_conn_paid", 42.6, 7.5, 57.0, 14.7),
    ("between_L_gates", 42.6, 0.75, 55.42, 7.5),
    ("central_paid", 57.0, 0.0, 95.0, 22.0),
    ("SE_conn_paid", 95.0, 5.0, 119.0, 14.7),
    ("SE_conn_south", 95.0, 0.0, 119.0, 5.0),
    ("F3_entrance_D", 60.0, 19.0, 82.0, 23.0),
    ("F3_entrance_E", 110.0, 17.0, 125.0, 23.0),
]
for name, x0, y0, x1, y1 in zones:
    mask = (f3a[:,0]>=x0) & (f3a[:,0]<x1) & (f3a[:,1]>=y0) & (f3a[:,1]<y1)
    print(f"  {name:30s}: {mask.sum():4d} nodes  (area={(x1-x0)*(y1-y0):.0f} m²)")

print()
print("=== F4 floor node density - right corner ===")
f4fl = [(d["x"], d["y"]) for _, d in g.nodes(data=True)
        if d.get("level") == "F4" and d.get("node_type") == "floor"]
f4a = np.array(f4fl) if f4fl else np.empty((0,2))
zones4 = [
    ("F4_right_upper", 120.0, 7.0, 131.0, 23.0),
    ("F4_right_lower", 120.0, -3.0, 134.0, 7.0),
    ("F4_SE_landing",  110.0, -0.5, 120.0, 7.0),
]
for name, x0, y0, x1, y1 in zones4:
    mask = (f4a[:,0]>=x0) & (f4a[:,0]<x1) & (f4a[:,1]>=y0) & (f4a[:,1]<y1)
    print(f"  {name:30s}: {mask.sum():4d} nodes  (area={(x1-x0)*(y1-y0):.0f} m²)")

print()
print("=== Entrance nodes ===")
ent = [(n, d) for n, d in g.nodes(data=True) if d.get("node_type") == "entrance"]
for n, d in sorted(ent, key=lambda x: x[0]):
    print(f"  {n}: level={d['level']}  ({d['x']:.1f},{d['y']:.1f})")

print("\nDone.")
