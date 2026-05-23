"""Diagnostic: inspect entrance and platform node types."""
import pickle, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "outputs/step3_graph/navigation_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Entrance nodes
ent = [(nid, d) for nid, d in G.nodes(data=True) if d.get("node_type") == "entrance"]
print(f"Entrance nodes: {len(ent)}")
for nid, d in ent:
    print(f"  {nid}: level={d.get('level')} x={d.get('x',0):.1f} y={d.get('y',0):.1f} name={d.get('entrance_name','?')}")

# F1 node types
f1_types = collections.Counter(d.get("node_type") for _, d in G.nodes(data=True) if d.get("level") == "F1")
print("\nF1 node types:", dict(f1_types))

# PSD door nodes on F1
psd = [(nid, d) for nid, d in G.nodes(data=True) if d.get("node_type") == "psd_door" and d.get("level") == "F1"]
print(f"\nPSD door nodes F1: {len(psd)}")
for nid, d in psd[:8]:
    print(f"  {nid}: x={d.get('x',0):.1f} y={d.get('y',0):.1f}")

# What node types adjacent to track zones on F1?
track_adj = [(nid, d) for nid, d in G.nodes(data=True)
             if d.get("level") == "F1" and d.get("node_type") in ("psd_door", "floor")
             and d.get("y", 99) < 2.0]
print(f"\nF1 nodes y<2 (track side): {len(track_adj)}")
for nid, d in track_adj[:8]:
    print(f"  {nid}: type={d.get('node_type')} x={d.get('x',0):.1f} y={d.get('y',0):.1f}")
