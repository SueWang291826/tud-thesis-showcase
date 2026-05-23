"""Quick gate diagnostic - no routing, just topology check."""
import pickle, sys
sys.path.insert(0, 'src')
import networkx as nx

with open('outputs/step3_graph/navigation_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

gate_ids = {n for n, a in G.nodes(data=True) if 'fare_gate' in a.get('node_type', '')}
ent_nodes = [n for n, a in G.nodes(data=True) if a.get('region') == 'ENTRANCE']

# Group entrances
by_name = {}
for n in ent_nodes:
    nm = G.nodes[n].get('entrance_name', '?')
    by_name.setdefault(nm, []).append(n)

# Pick representative per entrance (median by x)
reps = {}
for nm in sorted(by_name):
    nodes = sorted(by_name[nm], key=lambda n: G.nodes[n].get('x', 0))
    reps[nm] = nodes[len(nodes)//2]
    lvl = G.nodes[reps[nm]].get('level','?')
    print(f"{nm}: rep={reps[nm]}  level={lvl}  x={G.nodes[reps[nm]].get('x',0):.1f}  y={G.nodes[reps[nm]].get('y',0):.1f}")

print()
# Check direct edges: can any entrance directly reach a fare gate?
for nm, rep in reps.items():
    # BFS up to depth 2000 but stop early
    reachable_gates = []
    try:
        for n in nx.bfs_tree(G, rep).nodes():
            if n in gate_ids:
                ga = G.nodes[n]
                reachable_gates.append(f"{n}(type={ga['node_type']},x={ga.get('x',0):.1f})")
    except Exception as e:
        reachable_gates = [f"ERROR: {e}"]
    print(f"{nm} -> gates reachable: {reachable_gates[:6]}")

print()
# Check: what nodes are DIRECT neighbours of gate C rep node?
for nm in ['entrance_C', 'entrance_D']:
    rep = reps.get(nm)
    if not rep:
        continue
    print(f"\n{nm} rep={rep} neighbours (out-edges):")
    for _, nbr, d in G.out_edges(rep, data=True):
        print(f"  -> {nbr}  edge_type={d.get('edge_type')}  region={G.nodes[nbr].get('region')}  node_type={G.nodes[nbr].get('node_type')}")
