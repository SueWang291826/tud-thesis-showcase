"""Smoke test: verify both fixes work end-to-end."""
import pickle, sys
sys.path.insert(0, '.')
from src.routing import ESCALATOR_DIRECTIONS, directed_weight, patch_escalator_directions

G = pickle.load(open('outputs/step3_graph/navigation_graph.gpickle', 'rb'))

# Before patch: all edges are 'up'
before = {cid: None for cid in ESCALATOR_DIRECTIONS}
for u, v, d in G.edges(data=True):
    cid = d.get('connector_id', '')
    if cid in before:
        before[cid] = d.get('direction')

print('Before patch:')
for cid, d in before.items():
    print(f'  {cid[:30]}: {d}')

n = patch_escalator_directions(G)
print(f'\npatch_escalator_directions updated {n} edges')

# Check per-unit escalator capacity setup
esc_node_cid = {
    n: d['connector_id']
    for n, d in G.nodes(data=True)
    if d.get('node_type') in ('escalator', 'escalator_step') and d.get('connector_id')
}
print(f'\nesc_node_cid: {len(esc_node_cid)} escalator nodes mapped to connector_ids')
from collections import Counter
counts = Counter(esc_node_cid.values())
for cid, cnt in sorted(counts.items()):
    dir_now = ESCALATOR_DIRECTIONS.get(cid, '?')
    print(f'  {cid[:35]}: {cnt} nodes, dir={dir_now}')

# Verify F4->F3 downward is now traversable via directed_weight
wfn = directed_weight(G)
found_down = False
for u, v, d in G.edges(data=True):
    cid = d.get('connector_id', '')
    if cid == 'esc_18Gic2sdj5_OIgDswc3FF2':
        z_u = G.nodes[u].get('z', 0)
        z_v = G.nodes[v].get('z', 0)
        w = wfn(u, v, d)
        if z_u > z_v + 0.05:  # downward edge
            if w < float('inf'):
                found_down = True
            print(f'  F4->F3 downward edge weight={w:.2f}  z_u={z_u:.2f} z_v={z_v:.2f}')
            break
print(f'F4->F3 inbound escalator traversable: {found_down}')
