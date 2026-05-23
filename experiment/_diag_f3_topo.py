"""Diagnose why Gate D path bypasses fare gates on F3.

After fare_gate_entry x=56 and x=103, Gate D (x=72) might be
in the *paid zone* - or there's a floor path that bypasses.
We need to check:
  1. The staircase node connecting F3 to F1 used by Gate D path
  2. Whether that staircase is in the paid or unpaid zone
  3. Whether there is open floor connectivity across the gate barrier
"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import networkx as nx
from collections import defaultdict
from routing import find_entrance_paths, directed_weight

with open("outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    G = pickle.load(f)

entrance_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entrance"]
platform_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "door_platform"]
paths = find_entrance_paths(G, entrance_nodes, platform_nodes, deduplicate=True)

gate_entry_set = {n for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_entry"}
gate_exit_set  = {n for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_exit"}

for ep in paths:
    name = ep["entrance_name"].replace("entrance_", "Gate ")
    if name not in ("Gate C", "Gate D", "Gate E"):
        continue
    print(f"\n{'='*60}")
    print(f"{name}  ({ep['level']})")
    print(f"{'='*60}")

    for direction, path, label_gate in [
        ("INBOUND",  ep["inbound_path"],  gate_entry_set),
        ("OUTBOUND", ep["outbound_path"], gate_exit_set),
    ]:
        gates_in_path = [n for n in path if n in label_gate]
        print(f"\n  {direction} ({len(path)} nodes, {ep['inbound_cost'] if direction=='INBOUND' else ep['outbound_cost']:.1f}s)  gates={gates_in_path}")

        # Find the stair/escalator transition nodes (F3 level connections)
        print("  Stair/escalator transitions:")
        for i, n in enumerate(path):
            nt = G.nodes[n].get("node_type", "")
            lvl = G.nodes[n].get("level", "")
            if nt in ("stair_step", "escalator_step", "stair_chain"):
                d = G.nodes[n]
                print(f"    [{i}] {n}  type={nt}  level={lvl}  x={d.get('x',0):.1f} y={d.get('y',0):.1f} z={d.get('z',0):.1f}")

        # Print the full F3 segment of the path
        f3_seg = [(i, n) for i, n in enumerate(path) if G.nodes[n].get("level") == "F3"]
        print(f"  F3 segment ({len(f3_seg)} nodes, idx {f3_seg[0][0] if f3_seg else '?'} - {f3_seg[-1][0] if f3_seg else '?'}):")
        # sample every 5th
        shown = f3_seg[::max(1, len(f3_seg)//20)]
        for i, n in shown:
            d = G.nodes[n]
            nt = d.get("node_type","floor")
            print(f"    [{i}] {n}  x={d.get('x',0):.1f} y={d.get('y',0):.1f}  {nt}")

# Check: is there a direct floor-path from Gate D area on F3 to the staircase
# that bypasses the y≈16-20 fare-gate band?
print("\n\n=== Checking F3 floor connectivity around x=72, y=10-22 ===")
# Find all F3 floor nodes in that x range
band_nodes = [n for n, d in G.nodes(data=True)
              if d.get("level") == "F3"
              and 65 <= d.get("x", 0) <= 80
              and 8 <= d.get("y", 0) <= 22]
print(f"F3 floor nodes in x=[65,80], y=[8,22]: {len(band_nodes)}")
ys = sorted(set(round(G.nodes[n].get("y", 0), 1) for n in band_nodes))
print(f"  y values: {ys}")

# Check if there's a path from y>17 to y<12 without passing a fare gate
high_y = [n for n in band_nodes if G.nodes[n].get("y", 0) > 17]
low_y  = [n for n in band_nodes if G.nodes[n].get("y", 0) < 12]
print(f"  high_y nodes (y>17): {len(high_y)}, low_y nodes (y<12): {len(low_y)}")

if high_y and low_y:
    # Try to find a direct path
    src = high_y[len(high_y)//2]
    dst = low_y[len(low_y)//2]
    sd = G.nodes[src]
    dd = G.nodes[dst]
    print(f"  Testing {src} (x={sd.get('x',0):.1f} y={sd.get('y',0):.1f}) → {dst} (x={dd.get('x',0):.1f} y={dd.get('y',0):.1f})")
    try:
        sp = nx.shortest_path(G, src, dst)
        gates_in = [n for n in sp if n in gate_entry_set | gate_exit_set]
        print(f"  Path length: {len(sp)}, gates encountered: {gates_in}")
        # show nodes around gate transition or full if short
        for i, n in enumerate(sp):
            d = G.nodes[n]
            nt = d.get("node_type", "floor")
            extra = ""
            if n in gate_entry_set:
                extra = " <<< ENTRY GATE"
            elif n in gate_exit_set:
                extra = " <<< EXIT GATE"
            print(f"    [{i}] {n}  x={d.get('x',0):.1f} y={d.get('y',0):.1f}  {nt}{extra}")
    except nx.NetworkXNoPath:
        print("  No path found")
