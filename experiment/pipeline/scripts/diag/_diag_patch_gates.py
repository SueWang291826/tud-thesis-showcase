"""Check F1 right-bottom patch effect and gate edge attributes."""
import pickle

with open('outputs/step3_graph/navigation_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

f1 = [(nid, d) for nid, d in G.nodes(data=True) if d.get('level') == 'F1']

# Check x=135+ area after patch
new_nodes = [(nid, d) for nid, d in f1 if 135 <= d.get('x', 0) < 137 and 3 < d.get('y', 0) < 8]
print(f'F1 x=135-137, y=3-8: {len(new_nodes)} nodes')
for _, d in new_nodes:
    print(f'  ({d["x"]:.2f}, {d["y"]:.2f})')

print()
for xstart in range(129, 138):
    band = [(nid, d) for nid, d in f1 if xstart <= d.get('x', 0) < xstart + 1 and 4 <= d.get('y', 0) < 7]
    bar = '#' * len(band)
    print(f'  x={xstart}-{xstart+1}, y=4-7: {len(band):3d}  {bar}')

# Check gate edge pass_from_node_type
print()
fg_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'fare_gate']
print('Gate edge attributes (first 4 edges):')
for u, v, d in fg_edges[:4]:
    pft = d.get('pass_from_node_type', 'MISSING')
    dr = d.get('direction', '?')
    grp = d.get('gate_group', '?')
    print(f'  {u} <-> {v}')
    print(f'    direction={dr}  pass_from_node_type={pft}  group={grp}')
print(f'Total fare gate edges: {len(fg_edges)}')
