"""Diagnose F3 connector chain nodes and blank areas around connectors."""
import pickle, networkx as nx

G = pickle.load(open("outputs/step3_graph/navigation_graph.gpickle", "rb"))

# F3 connector chain nodes — level attr is "ESCALATOR"/"STAIR" but z≈12.1 for F3
conn_f3 = [(n, G.nodes[n]) for n, d in G.nodes(data=True)
           if d.get("node_type") in ("escalator", "stair")
           and 11.5 <= d.get("z", 0) <= 13.0]
print(f"=== F3-level connector chain nodes z=11.5-13 ({len(conn_f3)}) ===")
for n, d in sorted(conn_f3, key=lambda x: x[1]["x"]):
    nbrs = dict()
    for v in G.neighbors(n):
        nt = G.nodes[v].get("node_type", "?")
        nbrs[nt] = nbrs.get(nt, 0) + 1
    print(f"  {n}: ({d['x']:.1f},{d['y']:.1f}) type={d['node_type']} deg={G.degree(n)} nbr_types={nbrs}")

# Check anchor_snap edges (indicates connector node snapped far to find floor)
print("\n=== Anchor snap edges ===")
for u, v, data in G.edges(data=True):
    if data.get("edge_type") == "anchor_snap":
        du, dv = G.nodes[u], G.nodes[v]
        dist = data.get("length_2d", 0)
        print(f"  {u}({du.get('node_type','?')}) <-> {v}({dv.get('node_type','?')}) dist={dist:.2f}m")

# F3 floor nodes near left escalator area
print("\nF3 floor nodes in x=45-68, y=7-15:")
f3_mid = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n, d in G.nodes(data=True)
          if d.get("level") == "F3" and d.get("node_type") == "floor"
          and 45 <= d["x"] <= 68 and 7 <= d["y"] <= 15]
print(f"  count={len(f3_mid)}")
for p in sorted(f3_mid)[:15]:
    print(f"  {p}")

# F3 floor nodes near right escalator area
print("\nF3 floor nodes in x=95-120, y=7-15:")
f3_right_mid = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n, d in G.nodes(data=True)
                if d.get("level") == "F3" and d.get("node_type") == "floor"
                and 95 <= d["x"] <= 120 and 7 <= d["y"] <= 15]
print(f"  count={len(f3_right_mid)}")
for p in sorted(f3_right_mid)[:15]:
    print(f"  {p}")

# Check LEFT side: what is the closest escalator/stair at F3 to left scanner exit?
# Left scanner exits at x≈46.5, y=8.33/10.33/12.33
print("\nClosest connector nodes (z≈12.1) to left scanner exits (x=46.5, y=8-13):")
sc_exits = [(46.5, 8.33), (46.5, 10.33), (46.5, 12.33)]
for sx, sy in sc_exits:
    closest = min(conn_f3, key=lambda nd: ((nd[1]["x"]-sx)**2 + (nd[1]["y"]-sy)**2)**0.5)
    dist = ((closest[1]["x"]-sx)**2 + (closest[1]["y"]-sy)**2)**0.5
    print(f"  ({sx},{sy}) -> {closest[0]} at ({closest[1]['x']:.1f},{closest[1]['y']:.1f}) z={closest[1].get('z',0):.1f} dist={dist:.1f}m")
