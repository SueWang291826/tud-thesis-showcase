"""Verify escalator direction patching after the routing.py fix."""
import pickle, sys
sys.path.insert(0, '.')
from src.routing import patch_escalator_directions, ESCALATOR_DIRECTIONS

G = pickle.load(open('outputs/step3_graph/navigation_graph.gpickle', 'rb'))

TARGET = {'esc_18Gic2sdj5_OIgDswc3ESK', 'esc_18Gic2sdj5_OIgDswc3FF2'}
seen = set()
for u, v, d in G.edges(data=True):
    cid = d.get('connector_id', '')
    if cid in TARGET and cid not in seen:
        print(f'BEFORE: {cid} dir={d.get("direction", "?")}')
        seen.add(cid)

n = patch_escalator_directions(G)
print(f'Patched {n} edges')

seen.clear()
for u, v, d in G.edges(data=True):
    cid = d.get('connector_id', '')
    if cid in TARGET and cid not in seen:
        print(f'AFTER:  {cid} dir={d.get("direction", "?")}')
        seen.add(cid)

print('\nESCALATOR_DIRECTIONS table:')
for k, v in ESCALATOR_DIRECTIONS.items():
    if k in TARGET:
        print(f'  {k}: {v}')
