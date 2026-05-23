"""Diagnose F1 right-bottom corner gap and fare gate nodes."""
import pickle, networkx as nx

with open('outputs/step3_graph/navigation_graph.gpickle','rb') as f:
    G = pickle.load(f)

print('Graph type:', type(G).__name__)

# F1 nodes
f1_nodes = [(nid, d) for nid,d in G.nodes(data=True) if d.get('level')=='F1']

# Right-bottom corner: x>110, y<8 (south of center, high x)
rb = [(nid,d) for nid,d in f1_nodes if d.get('x',0)>110 and d.get('y',0)<8]
print(f'\nF1 right-bottom (x>110, y<8): {len(rb)} nodes')
if rb:
    xs = [d.get('x',0) for _,d in rb]
    ys = [d.get('y',0) for _,d in rb]
    print(f'  x: {min(xs):.1f} – {max(xs):.1f}')
    print(f'  y: {min(ys):.1f} – {max(ys):.1f}')

# Very right corner x>125
rb2 = [(nid,d) for nid,d in f1_nodes if d.get('x',0)>125 and d.get('y',0)<8]
print(f'F1 very right (x>125, y<8): {len(rb2)} nodes')

# Scan full x range of F1 bottom strip y=3.5-7
print('\nF1 bottom strip (y=3.5-7) node density by x-band:')
bottom = [(nid,d) for nid,d in f1_nodes if 3.5 < d.get('y',0) < 7.0]
for xstart in range(20, 140, 10):
    band = [d for _,d in bottom if xstart <= d.get('x',0) < xstart+10]
    bar = '#' * (len(band)//2)
    print(f'  x={xstart:3d}-{xstart+10}: {len(band):3d} nodes  {bar}')

# Fare gate nodes
print('\nFare gate nodes:')
fg = [(nid,d) for nid,d in G.nodes(data=True) if 'fare_gate' in str(d.get('node_type',''))]
for nid,d in sorted(fg, key=lambda x: x[0]):
    nt = d.get('node_type','')
    dr = d.get('direction','?')
    grp = d.get('gate_group','?')
    x,y = d.get('x',0), d.get('y',0)
    print(f'  {nid:45s}  type={nt:20s} dir={dr:8s} grp={grp}  ({x:.2f},{y:.2f})')

# Fare gate edges
print('\nFare gate edges:')
fg_edges = [(u,v,d) for u,v,d in G.edges(data=True) if d.get('edge_type')=='fare_gate']
for u,v,d in fg_edges:
    print(f'  {u} <-> {v}  dir={d.get("direction")} grp={d.get("gate_group")}')
print(f'\nTotal fare gate edges: {len(fg_edges)}')
