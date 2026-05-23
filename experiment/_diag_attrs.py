"""Check what attributes entrance nodes actually have."""
import pickle, sys
sys.path.insert(0, 'src')
import networkx as nx

with open('outputs/step3_graph/navigation_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Sample 5 nodes from each node_type
by_type = {}
for n, a in G.nodes(data=True):
    t = a.get('node_type', 'NONE')
    by_type.setdefault(t, []).append((n, a))

print("Node types and counts:")
for t in sorted(by_type, key=lambda k: -len(by_type[k])):
    print(f"  {t}: {len(by_type[t])}")

print()
print("Sample entrance-type node attributes:")
for t, nodes in by_type.items():
    if 'entrance' in t.lower() or 'gate' in t.lower():
        n, a = nodes[0]
        print(f"\nType={t}  node={n}")
        for k, v in sorted(a.items()):
            print(f"    {k} = {v!r}")
