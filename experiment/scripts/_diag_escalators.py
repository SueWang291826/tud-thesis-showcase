"""Diagnose all escalator connectors in the navigation graph."""
import pickle, sys
sys.path.insert(0, '.')

G = pickle.load(open('outputs/step3_graph/navigation_graph.gpickle', 'rb'))

esc_ids = {}
for u, v, d in G.edges(data=True):
    if d.get('edge_type') == 'escalator':
        cid = d.get('connector_id', '?')
        z_u = G.nodes[u].get('z', 0)
        z_v = G.nodes[v].get('z', 0)
        if cid not in esc_ids:
            esc_ids[cid] = {'u_z': round(z_u, 1), 'v_z': round(z_v, 1),
                            'dir': d.get('direction', '?'), 'edges': 0,
                            'node_types': set()}
        esc_ids[cid]['edges'] += 1
        esc_ids[cid]['node_types'].add(G.nodes[u].get('node_type', '?'))
        esc_ids[cid]['node_types'].add(G.nodes[v].get('node_type', '?'))

print('All escalator connector IDs:')
for cid, info in sorted(esc_ids.items()):
    dz = abs(info['u_z'] - info['v_z'])
    level = 'F1-F3' if dz > 3 else 'F3-F4'
    print(f"  {cid}: dir={info['dir']} z={info['u_z']}..{info['v_z']} ({level}) edges={info['edges']}")

# Also count nodes by connector_id  
print()
print('Escalator node counts by connector_id:')
node_by_cid = {}
for n, d in G.nodes(data=True):
    nt = d.get('node_type', '')
    if nt in ('escalator', 'escalator_step'):
        cid = d.get('connector_id', '?')
        node_by_cid.setdefault(cid, []).append(n)

for cid, nodes in sorted(node_by_cid.items()):
    print(f"  {cid}: {len(nodes)} nodes")
