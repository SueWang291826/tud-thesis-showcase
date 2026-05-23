"""

Step 6: Evaluation

===================



Compute metrics, compare scenarios, generate thesis figures.

"""

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(ROOT))



from src.utils import load_config

from src.evaluation import (

    compute_scenario_metrics,

    compare_scenarios,

    graph_topology_metrics,

    connector_utilisation,

    save_evaluation_outputs,

)





def main(config_path: str | None = None):

    """Run evaluation.

    

    This script is typically called after step5 has completed and

    result JSON files exist in the step5 output directory.

    It can also receive scenario results directly via run_pipeline.

    """

    cfg_path = config_path or str(ROOT / "config" / "experiment_config.yaml")

    cfg = load_config(cfg_path)



    print("=" * 60)

    print("STEP 6 —Evaluation")

    print("=" * 60)



    eval_dir = Path(ROOT / cfg["output"]["step_dirs"]["step6"])



    # Load scenario results from step5 outputs

    step5_dir = Path(ROOT / cfg["output"]["step_dirs"]["step5"])

    from src.utils import load_json



    result_files = sorted(step5_dir.glob("result_*.json"))

    if not result_files:

        print("  No simulation results found in step5 outputs.")

        print("  Run step5_simulate.py first, or use run_pipeline.py.")

        return



    scenario_results = []

    for rf in result_files:

        print(f"  Loading {rf.name}")

        scenario_results.append(load_json(rf))



    # Compute metrics

    metrics_list = [compute_scenario_metrics(r) for r in scenario_results]

    for m in metrics_list:

        print(f"\n  [{m['label']}] arrive={m['arrive_rate']:.1%}, "

              f"mean_tt={m['mean_travel_time']:.1f}s, "

              f"p95_tt={m['p95_travel_time']:.1f}s")



    # Comparison

    comp = compare_scenarios(metrics_list)

    if comp.get("comparisons"):

        for c in comp["comparisons"]:

            pct = c.get("mean_travel_time_pct_change", 0)

            print(f"  {c['baseline']} →{c['scenario']}: "

                  f"mean_tt change = {pct:+.1f}%")



    # Try loading graph for topology metrics

    graph_metrics = {}

    graph_path = Path(ROOT / cfg["output"]["step_dirs"]["step3"]) / "navigation_graph.gpickle"

    G = None

    if graph_path.exists():

        import pickle

        with open(graph_path, "rb") as f:

            G = pickle.load(f)

        graph_metrics = graph_topology_metrics(G)

        print(f"\n  Graph: {graph_metrics['total_nodes']} nodes, "

              f"{graph_metrics['total_edges']} edges, "

              f"connected={graph_metrics['is_connected']}")



    save_evaluation_outputs(scenario_results, graph_metrics, eval_dir)



    # --- Visualization ---

    print("\n  Generating visualisations ...")

    from src.viz import (

        fig_comparison_bar, fig_arrival_curve, fig_queue_over_time,

        fig_elderly_vs_normal, fig_connector_utilisation,

    )



    fig_dir = eval_dir / "figures"

    fig_comparison_bar(metrics_list, fig_dir, cfg)

    fig_arrival_curve(scenario_results, fig_dir, cfg)

    fig_queue_over_time(scenario_results, fig_dir, cfg)

    fig_elderly_vs_normal(metrics_list, fig_dir, cfg)



    # Connector utilisation (if graph available)

    if G is not None:

        util_list = []

        for r in scenario_results:

            util = connector_utilisation(r)

            util["label"] = r.get("label", "?")

            util_list.append(util)

        fig_connector_utilisation(util_list, fig_dir, cfg)

        print("  5 figures saved")

    else:

        # Still compute utilisation from results alone

        util_list = []

        for r in scenario_results:

            util = connector_utilisation(r)

            util["label"] = r.get("label", "?")

            util_list.append(util)

        fig_connector_utilisation(util_list, fig_dir, cfg)

        print("  5 figures saved")



    print(f"\n  Outputs →{eval_dir}")





if __name__ == "__main__":

    main()

