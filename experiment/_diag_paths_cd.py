"""Check actual computed paths for all entrances - are fare gates in path?"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import networkx as nx
from collections import defaultdict
from routing import find_entrance_paths, directed_weight

GRAPH = Path("outputs/step3_graph/navigation_graph.gpickle")
with open(GRAPH, "rb") as f:
    G = pickle.load(f)

entrance_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entrance"]
platform_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "door_platform"]

print(f"Entrances: {len(entrance_nodes)}, PSDs: {len(platform_nodes)}")

paths = find_entrance_paths(G, entrance_nodes, platform_nodes, deduplicate=True)

gate_entry_set = {n for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_entry"}
gate_exit_set  = {n for n, d in G.nodes(data=True) if d.get("node_type") == "fare_gate_exit"}

print()
for ep in paths:
    name = ep["entrance_name"].replace("entrance_", "Gate ")
    lvl  = ep["level"]
    eid  = ep["entrance_id"]
    ex   = G.nodes[eid].get("x", 0)

    # scan inbound path for gate nodes
    in_gates  = [n for n in ep["inbound_path"]  if n in gate_entry_set]
    out_gates = [n for n in ep["outbound_path"] if n in gate_exit_set]

    # level sequence for inbound  
    in_levels  = []
    for n in ep["inbound_path"]:
        lv = G.nodes[n].get("level","?")
        if not in_levels or in_levels[-1] != lv:
            in_levels.append(lv)
    out_levels = []
    for n in ep["outbound_path"]:
        lv = G.nodes[n].get("level","?")
        if not out_levels or out_levels[-1] != lv:
            out_levels.append(lv)

    print(f"{name} ({lvl}, x={ex:.1f}):")
    print(f"  inbound  ({ep['inbound_cost']:.1f}s, {len(ep['inbound_path'])} nodes)  gates={in_gates}  levels: {in_levels}")
    print(f"  outbound ({ep['outbound_cost']:.1f}s, {len(ep['outbound_path'])} nodes)  gates={out_gates}  levels: {out_levels}")

    # If no gate found, show where path is near F3 fare gate x coords
    if not in_gates or not out_gates:
        print(f"  *** BYPASS DETECTED ***")
        # show first 20 nodes around F3
        f3nodes = [(i, n) for i, n in enumerate(ep["inbound_path"])
                   if G.nodes[n].get("level") == "F3"]
        if f3nodes:
            seg = f3nodes[:8]
            for i, n in seg:
                d = G.nodes[n]
                print(f"    inbound[{i}] {n}  x={d.get('x',0):.1f} y={d.get('y',0):.1f}  type={d.get('node_type','')}")
