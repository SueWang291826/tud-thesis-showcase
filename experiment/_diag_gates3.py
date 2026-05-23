"""
Check right gate paid_side in graph, then verify Gate C/D path gate crossings.
Output to file for reliable reading.
"""
import pickle, sys, math
sys.path.insert(0, 'src')
import networkx as nx
from routing import find_entrance_paths, directed_weight
from collections import defaultdict

out_lines = []
def p(s=""):
    out_lines.append(str(s))

with open('outputs/step3_graph/navigation_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# 1. Right gate paid_side values
p("=== Right gate node paid_side ===")
for n, a in G.nodes(data=True):
    if 'right' in n and 'fare_gate' in a.get('node_type',''):
        p(f"  {n}  paid_side={a.get('paid_side')}  type={a['node_type']}  x={a.get('x',0):.1f}")

# 2. Check edges on right inbound gate
p()
p("=== Edges on fg_right_inbound_P03 ===")
target = 'fg_right_inbound_P03'
if target in G:
    p(f"in-edges (→ gate):")
    for u, _, d in G.in_edges(target, data=True):
        ua = G.nodes[u]
        p(f"  {u} (x={ua.get('x',0):.1f}, type={ua.get('node_type')}) --{d.get('edge_type')}--> {target}")
    p(f"out-edges (gate →):")
    for _, v, d in G.out_edges(target, data=True):
        va = G.nodes[v]
        p(f"  {target} --{d.get('edge_type')}--> {v} (x={va.get('x',0):.1f}, type={va.get('node_type')})")

# 3. Get representative entrance nodes
ent_nodes = [n for n, a in G.nodes(data=True) if a.get('node_type') == 'entrance']
by_name = defaultdict(list)
for n in ent_nodes:
    by_name[G.nodes[n].get('entrance_name','?')].append(n)

reps = {}
for nm in sorted(by_name):
    nodes = sorted(by_name[nm], key=lambda n: G.nodes[n].get('x', 0))
    reps[nm] = nodes[len(nodes)//2]

p()
p("=== Representative entrance nodes ===")
for nm, n in reps.items():
    a = G.nodes[n]
    p(f"  {nm}: {n}  level={a.get('level')}  x={a.get('x',0):.1f}  y={a.get('y',0):.1f}")

# 4. Check which floor nodes are at x>104 on F3 (unpaid side of right gate)
p()
p("=== F3 floor nodes east of right inbound gate (x>104) - should be 'unpaid' ===")
east_f3 = [(n, a) for n, a in G.nodes(data=True)
           if a.get('node_type') == 'floor' and a.get('level') == 'F3'
           and a.get('x', 0) > 104]
p(f"  Count: {len(east_f3)}")
if east_f3:
    xs = sorted(a.get('x',0) for _,a in east_f3)
    p(f"  x range: [{xs[0]:.1f}, {xs[-1]:.1f}]")

# 5. Check Gate D connectivity to left inbound gate
p()
p("=== Gate D (x=65-79) → can it reach fg_left_inbound? ===")
gate_d_rep = reps.get('entrance_D')
if gate_d_rep:
    # Check if there's an edge path from Gate D rep to left inbound gates
    left_entry = [n for n, a in G.nodes(data=True)
                  if a.get('node_type') == 'fare_gate_entry' and 'left' in n]
    wfn = directed_weight(G)
    try:
        lengths = dict(nx.single_source_dijkstra_path_length(G, gate_d_rep, weight=wfn, cutoff=500))
        for gn in left_entry:
            cost = lengths.get(gn, float('inf'))
            p(f"  Gate D -> {gn}: cost={cost:.1f}s")
    except Exception as e:
        p(f"  ERROR: {e}")

# 6. Check full paths for Gate C and D
p()
p("=== Full path gate node check (Gate C and D) ===")
plt_nodes = [n for n, a in G.nodes(data=True)
             if a.get('node_type') == 'door_platform' and a.get('level') == 'F1']
eps = find_entrance_paths(G, ent_nodes, plt_nodes)
gate_ids = {n for n, a in G.nodes(data=True) if 'fare_gate' in a.get('node_type', '')}

for ep in eps:
    ename = ep['entrance_name'].replace('entrance_', 'Gate ')
    ib_gates = [n for n in ep['inbound_path'] if n in gate_ids]
    ob_gates = [n for n in ep['outbound_path'] if n in gate_ids]
    p(f"{ename}: inbound_gate={ep['inbound_gate']}  path_gates={ib_gates}")
    p(f"        outbound_gate={ep['outbound_gate']}  path_gates={ob_gates}")

with open('_diag_gates_result.txt', 'w') as f:
    f.write('\n'.join(out_lines))

print("Done - see _diag_gates_result.txt")
