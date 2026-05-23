"""Regenerate Step 3 interactive HTML files only.



Loads the saved navigation graph from gpickle and re-extracts

geometry (needed for floor outlines), then writes the 3 interactive

HTML files.  Skips graph construction (saves ~10s).

"""

import pickle

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(ROOT))



from src.utils import load_config

from src.data_loader import load_preprocessing_products

from src.geometry_extractor import extract_all_levels

from src.viz_interactive import (

    fig_interactive_station,

    fig_interactive_graph,

    fig_interactive_cross_section,

)





def main():

    cfg = load_config(str(ROOT / "config" / "experiment_config.yaml"))

    out_dir = Path(ROOT / cfg["output"]["step_dirs"]["step3"])

    html_dir = out_dir / "interactive"



    print("Regenerating Step 3 interactive HTML...")



    # Load graph

    graph_path = out_dir / "navigation_graph.gpickle"

    print(f"  Loading graph: {graph_path}")

    with open(graph_path, "rb") as f:

        G = pickle.load(f)

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")



    # Extract geometries (needed for floor outlines)

    products = load_preprocessing_products(cfg)

    geometries, all_connectors, _ = extract_all_levels(cfg, products)



    # Level nodes for station 3D viz

    nodes_for_viz = {}

    import networkx as nx

    by_level = {}

    for nid, d in G.nodes(data=True):

        lvl = d.get("level", "")

        if lvl and d.get("node_type") == "floor":

            by_level.setdefault(lvl, []).append((d.get("x", 0), d.get("y", 0)))

    nodes_for_viz = by_level



    elevations = {

        lvl: lc["elevation_m"]

        for lvl, lc in cfg["station"]["levels"].items()

        if lc.get("is_walkable", False)

    }



    print("\n  Writing interactive HTML files ...")

    p = fig_interactive_station(

        geometries, nodes_for_viz, elevations, all_connectors,

        html_dir, cfg, G=G,

    )

    print(f"  [html] {p.name}")



    p = fig_interactive_graph(G, all_connectors, elevations, html_dir, cfg)

    print(f"  [html] {p.name}")



    p = fig_interactive_cross_section(

        geometries, all_connectors, elevations, html_dir, cfg)

    print(f"  [html] {p.name}")



    print(f"\n  Done! →{html_dir}")





if __name__ == "__main__":

    main()

