"""Diagnose Gate C and D routing through fare gates."""
import pickle, json
from pathlib import Path
import networkx as nx

GRAPH = Path("outputs/step3_graph/navigation_graph.gpickle")
with open(GRAPH, "rb") as f:
    G = pickle.load(f)

# --- 1. Fare gate nodes ---
gates_entry = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_entry"]
gates_exit  = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_exit"]

print(f"\n=== fare_gate_entry nodes: {len(gates_entry)} ===")
for n, d in sorted(gates_entry, key=lambda x: x[1].get("x", 0)):
    print(f"  {n}  level={d.get('level')}  x={d.get('x', 0):.1f}  y={d.get('y', 0):.1f}")

print(f"\n=== fare_gate_exit nodes: {len(gates_exit)} ===")
for n, d in sorted(gates_exit, key=lambda x: x[1].get("x", 0)):
    print(f"  {n}  level={d.get('level')}  x={d.get('x', 0):.1f}  y={d.get('y', 0):.1f}")

# --- 2. F3 connectivity: can fare_gate_entry reach F3? ---
entry_nodes = [n for n, d in gates_entry]
exit_nodes  = [n for n, d in gates_exit]
print("\n=== Edges FROM fare_gate_entry nodes ===")
for n in entry_nodes:
    succs = list(G.successors(n))
    preds = list(G.predecessors(n))
    print(f"  {n}: predecessors={preds[:5]}, successors={succs[:5]}")

print("\n=== Edges TO fare_gate_exit nodes ===")
for n in exit_nodes:
    succs = list(G.successors(n))
    preds = list(G.predecessors(n))
    print(f"  {n}: predecessors={preds[:5]}, successors={succs[:5]}")

# --- 3. Are Gate C/D entrances reachable from gate entry nodes? ---
entrance_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "entrance"]
from collections import defaultdict
by_name = defaultdict(list)
for n, d in entrance_nodes:
    by_name[d.get("entrance_name", "")].append((n, d))

print("\n=== Check reachability: entry gate -> Gate C/D entrances ===")
for name in sorted(by_name.keys()):
    nodes_in_group = [n for n, d in by_name[name]]
    x_vals = sorted([d.get("x", 0) for _, d in by_name[name]])
    lvl = by_name[name][0][1].get("level", "")
    print(f"\n  Entrance '{name}' ({lvl}): {len(nodes_in_group)} nodes, x=[{x_vals[0]:.1f},{x_vals[-1]:.1f}]")
    # pick median
    nodes_sorted = sorted(nodes_in_group, key=lambda n: G.nodes[n].get("x", 0))
    rep = nodes_sorted[len(nodes_sorted)//2]
    print(f"    representative: {rep}  x={G.nodes[rep].get('x',0):.1f}")
    # check if any entry gate can reach this rep
    for g_entry in entry_nodes:
        try:
            path = nx.dijkstra_path(G, rep, g_entry)
            print(f"    {rep} -> {g_entry}: path exists, len={len(path)}")
        except nx.NetworkXNoPath:
            print(f"    {rep} -> {g_entry}: NO PATH")
        except Exception as e:
            print(f"    {rep} -> {g_entry}: error {e}")
