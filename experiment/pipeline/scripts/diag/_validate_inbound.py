"""Check left scanner approach connectivity and full inbound flow."""
import pickle, networkx as nx

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    g = pickle.load(f)

# Left scanner approach neighbour check
left_approaches = [n for n, d in g.nodes(data=True)
                   if "left_scanner" in n and d.get("node_type") == "scanner_approach"]
print("Left scanner approach neighbours:")
for n in left_approaches[:2]:
    nbrs = list(g.neighbors(n))
    nd = g.nodes[n]
    print(f"  {n} ({nd['x']:.2f},{nd['y']:.2f}): {len(nbrs)} neighbours")
    for nb in nbrs:
        nbd = g.nodes[nb]
        et = g[n][nb].get("edge_type")
        if et != "security_scanner":
            print(f"    -> {nbd.get('node_type')} ({nbd['x']:.2f},{nbd['y']:.2f}) edge={et}")

# Full inbound flow: entrance_A -> scanner_exit
ent_a = "ent_F4_entrance_A"
sc_exit = "sc_left_scanner_P01_exit"
try:
    path = nx.shortest_path(g, ent_a, sc_exit, weight="travel_time")
    etypes = set(g[u][v].get("edge_type") for u, v in zip(path[:-1], path[1:]))
    print(f"\nentrance_A -> left_scanner_exit: {len(path)} nodes, etypes={sorted(etypes)}")
    for nd_id in path:
        nt = g.nodes[nd_id].get("node_type", "")
        if any(x in nt for x in ("gate", "scanner", "entrance", "escalator", "stair_chain")):
            nd = g.nodes[nd_id]
            print(f"  {nt:22s} ({nd['x']:.2f},{nd['y']:.2f})")
except Exception as e:
    print(f"Error: {e}")

print("\nDone.")
