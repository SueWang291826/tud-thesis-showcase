"""Quick path validation: entrance → F1 platform, checking gate/scanner nodes."""
import pickle, networkx as nx
from pathlib import Path

graph_path = Path("outputs/step3_graph/navigation_graph.gpickle")
with graph_path.open("rb") as f:
    g = pickle.load(f)

print(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges, Connected={nx.is_weakly_connected(g) if g.is_directed() else nx.is_connected(g)}")

# Entrances
entrance_nodes = [(n, d) for n, d in g.nodes(data=True) if d.get("node_type") == "entrance"]
print("\nEntrances:")
for nid, d in entrance_nodes:
    print(f"  {nid}  ({d['x']:.1f}, {d['y']:.1f})  level={d['level']}")

# F1 floor node near centre
f1_floor = [(n, d) for n, d in g.nodes(data=True)
            if d.get("node_type") == "floor" and d.get("level") == "F1"]
target_node, target_attr = min(f1_floor, key=lambda nd: (nd[1]["x"] - 80) ** 2 + (nd[1]["y"] - 10) ** 2)
print(f"\nTarget F1 node: {target_node} at ({target_attr['x']:.1f}, {target_attr['y']:.1f})")

# Shortest path from each entrance to F1
for src_nid, src_d in entrance_nodes:
    try:
        path = nx.shortest_path(g, src_nid, target_node, weight="travel_time")
    except nx.NetworkXNoPath:
        print(f"\nEntrance {src_nid}: NO PATH")
        continue

    crossed_types = set()
    for u, v in zip(path[:-1], path[1:]):
        crossed_types.add(g[u][v].get("edge_type", "?"))

    print(f"\nEntrance {src_nid} ({src_d['x']:.1f},{src_d['y']:.1f}) → F1 target")
    print(f"  Path: {len(path)} nodes, edge types: {sorted(crossed_types)}")

    checkpoints = []
    for n in path:
        nt = g.nodes[n].get("node_type", "")
        if nt in ("fare_gate_entry", "fare_gate_exit"):
            nd = g.nodes[n]
            checkpoints.append(
                f"    {nt:20s} ({nd['x']:.2f}, {nd['y']:.2f})  dir={nd.get('direction', '-')}  passage={nd.get('passage_id', nd.get('passage_id', '-'))}"
            )
    if checkpoints:
        print("  Checkpoints:")
        for cp in checkpoints:
            print(cp)
    else:
        print("  WARNING: No gate/scanner checkpoints crossed!")

print("\nDone.")
