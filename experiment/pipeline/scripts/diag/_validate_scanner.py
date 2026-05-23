"""Validate scanner node connectivity and gate direction enforcement."""
import pickle, networkx as nx

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    g = pickle.load(f)

sc_nodes = [(n, d) for n, d in g.nodes(data=True)
            if d.get("node_type") in ("scanner_approach", "scanner_exit")]
fg_nodes = [(n, d) for n, d in g.nodes(data=True)
            if d.get("node_type") in ("fare_gate_entry", "fare_gate_exit")]

print("=== Scanner nodes ===")
for n, d in sorted(sc_nodes, key=lambda x: x[0]):
    print(f"  {n}: ({d['x']:.2f},{d['y']:.2f}) type={d['node_type']}")

print("\n=== Fare gate nodes ===")
for n, d in sorted(fg_nodes, key=lambda x: x[0]):
    print(f"  {n}: ({d['x']:.2f},{d['y']:.2f}) type={d['node_type']} dir={d.get('direction','-')}")

# Check scanner edge neighbours
print("\n=== Scanner edge neighbours ===")
for n, d in sc_nodes:
    nbrs = list(g.neighbors(n))
    etypes = [g[n][nb].get("edge_type", "?") for nb in nbrs]
    print(f"  {n}: {len(nbrs)} neighbours, etypes={set(etypes)}")

# Verify F1 is reachable from scanner_exit
left_exits = [n for n, d in sc_nodes if "left" in n and d.get("node_type") == "scanner_exit"]
f1_nodes = [n for n, d in g.nodes(data=True) if d.get("level") == "F1" and d.get("node_type") == "floor"]
if left_exits and f1_nodes:
    src = left_exits[0]
    tgt = f1_nodes[len(f1_nodes) // 2]  # middle F1 node
    tgt_attr = g.nodes[tgt]
    try:
        path = nx.shortest_path(g, src, tgt, weight="travel_time")
        etypes = set(g[u][v].get("edge_type", "?") for u, v in zip(path[:-1], path[1:]))
        print(f"\nPath: scanner_exit {src} → F1 ({tgt_attr['x']:.1f},{tgt_attr['y']:.1f})")
        print(f"  {len(path)} nodes, edge types: {sorted(etypes)}")
        checkpoints = [(n, g.nodes[n]) for n in path
                       if g.nodes[n].get("node_type", "") in
                       ("fare_gate_entry", "fare_gate_exit")]
        if checkpoints:
            print("  Checkpoints in path:")
            for cn, cd in checkpoints:
                print(f"    {cd['node_type']:20s} ({cd['x']:.2f},{cd['y']:.2f})")
        else:
            print("  No checkpoints (scanner→escalator→F1 direct)")
    except nx.NetworkXNoPath:
        print(f"\nNO PATH from {src} to F1!")

print("\nDone.")
