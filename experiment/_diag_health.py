"""Quick graph health check after track zone changes."""
import pickle, networkx as nx, collections

G = pickle.load(open("outputs/step3_graph/navigation_graph.gpickle", "rb"))
print(f"Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
print(f"Connected: {nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)}")

# Count by level and type
by_level = collections.Counter()
by_type = collections.Counter()
for n, d in G.nodes(data=True):
    by_level[d.get("level", "?")] += 1
    by_type[d.get("node_type", "?")] += 1

print("\nBy level:", dict(sorted(by_level.items())))
print("By type:", dict(sorted(by_type.items())))

# Check F1 y range
f1_ys = [d["y"] for n, d in G.nodes(data=True)
         if d.get("level") == "F1" and d.get("node_type") == "floor"]
if f1_ys:
    print(f"\nF1 floor y range: {min(f1_ys):.2f} to {max(f1_ys):.2f}")
    in_track_south = sum(1 for y in f1_ys if y < 3.5)
    in_track_north = sum(1 for y in f1_ys if y > 17.8)
    print(f"  In south track (y<3.5): {in_track_south}")
    print(f"  In north track (y>17.8): {in_track_north}")
    print(f"  Platform (y=3.5-17.8): {len(f1_ys) - in_track_south - in_track_north}")

# Check F4 presence
f4_nodes = [n for n, d in G.nodes(data=True) if d.get("level") == "F4"]
print(f"\nF4 nodes in graph: {len(f4_nodes)}")

# Connectors in graph
conn_types = collections.Counter()
for n, d in G.nodes(data=True):
    if d.get("node_type") in ("escalator", "stair", "elevator"):
        z = d.get("z", 0)
        conn_types[f"{d['node_type']}@z={z:.0f}"] += 1
print("\nConnector chain nodes by type+z (top10):")
for k, v in sorted(conn_types.most_common(10)):
    print(f"  {k}: {v}")

# Key paths
entrances = {n: G.nodes[n] for n, d in G.nodes(data=True) if d.get("node_type") == "entrance"}
print(f"\nEntrances: {list(entrances.keys())}")
f1_floor = [n for n, d in G.nodes(data=True)
            if d.get("level") == "F1" and d.get("node_type") == "floor"]
if f1_floor and entrances:
    tgt = f1_floor[len(f1_floor)//2]
    for ename, edata in list(entrances.items())[:2]:
        try:
            p = nx.shortest_path(G, ename, tgt, weight="travel_time")
            cost = nx.shortest_path_length(G, ename, tgt, weight="travel_time")
            crossed = [G[u][v].get("edge_type","") for u,v in zip(p[:-1],p[1:])
                      if G[u][v].get("edge_type","") in ("fare_gate","security_scanner")]
            print(f"  {ename} -> F1: {len(p)} nodes, {cost:.1f}s, gates/scan={crossed}")
        except Exception as e:
            print(f"  {ename} -> F1: FAILED {e}")
