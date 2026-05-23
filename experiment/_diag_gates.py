"""Diagnose fare gate nodes and whether Gate C/D routes pass through them."""
import pickle, sys, networkx as nx
sys.path.insert(0, 'src')
from routing import find_entrance_paths, directed_weight

with open('outputs/step3_graph/navigation_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# ── 1. All fare gate nodes ──────────────────────────────────────────────────
gates = [(n, a) for n, a in G.nodes(data=True)
         if 'fare_gate' in a.get('node_type', '')]
print(f"Fare gate nodes total: {len(gates)}")
for n, a in sorted(gates, key=lambda x: (x[1].get('level',''), x[1].get('x',0))):
    print(f"  {n}  type={a['node_type']}  level={a.get('level')}  "
          f"x={a.get('x',0):.1f}  y={a.get('y',0):.1f}")

# ── 2. Entrance nodes by name ───────────────────────────────────────────────
print()
ent_nodes = [n for n, a in G.nodes(data=True) if a.get('region') == 'ENTRANCE']
by_name = {}
for n in ent_nodes:
    nm = G.nodes[n].get('entrance_name', '?')
    by_name.setdefault(nm, []).append(n)
for nm in sorted(by_name):
    nodes = by_name[nm]
    xs = sorted(G.nodes[n].get('x', 0) for n in nodes)
    lvl = G.nodes[nodes[0]].get('level','?')
    print(f"  {nm}  level={lvl}  n={len(nodes)}  x=[{xs[0]:.1f},{xs[-1]:.1f}]")

# ── 3. Compute 5 entrance paths and check gate presence ────────────────────
print()
plt_nodes = [n for n, a in G.nodes(data=True) if a.get('region') == 'PLATFORM']
eps = find_entrance_paths(G, ent_nodes, plt_nodes)

gate_ids = {n for n, a in G.nodes(data=True) if 'fare_gate' in a.get('node_type', '')}

for ep in eps:
    ename = ep['entrance_name'].replace('entrance_', 'Gate ')
    # Check inbound
    ib_gates = [n for n in ep['inbound_path'] if n in gate_ids]
    ob_gates = [n for n in ep['outbound_path'] if n in gate_ids]
    print(f"{ename}:")
    print(f"  inbound_gate field  = {ep['inbound_gate']}")
    print(f"  fare gates in path  = {ib_gates}")
    print(f"  outbound_gate field = {ep['outbound_gate']}")
    print(f"  fare gates in path  = {ob_gates}")

# ── 4. Connectivity check: can Gate C entrance reach fare gates? ─────────────
print()
# find gate C rep node
gate_c_nodes = by_name.get('entrance_C', [])
if gate_c_nodes:
    xs = sorted(gate_c_nodes, key=lambda n: G.nodes[n].get('x', 0))
    rep_c = xs[len(xs)//2]
    wfn = directed_weight(G)
    try:
        lengths = nx.single_source_dijkstra_path_length(G, rep_c, weight=wfn)
    except Exception as e:
        lengths = {}
        print(f"Dijkstra error: {e}")
    print(f"Gate C rep node: {rep_c}  level={G.nodes[rep_c].get('level')}  x={G.nodes[rep_c].get('x',0):.1f}")
    print("Reachable fare gate nodes from Gate C:")
    for gn in sorted(gate_ids):
        cost = lengths.get(gn, float('inf'))
        ga = G.nodes[gn]
        print(f"  {gn}  type={ga['node_type']}  cost={cost:.1f}  level={ga.get('level')}  x={ga.get('x',0):.1f}")
