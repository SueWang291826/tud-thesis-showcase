"""
Re-run only the wheelchair + comparison + evaluation sections.
ScenA/B/C already completed successfully.
"""
import sys, pickle, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from pipeline.scripts.step5_experiments import (
    _load_graph, run_wheelchair, run_comparison
)
from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.routing import define_semantic_regions
from src.viz_thesis import generate_evaluation_report

cfg = load_config(str(ROOT / "config" / "experiment_config.yaml"))
out_base = Path(ROOT / cfg["output"]["step_dirs"]["step5"])

G = _load_graph(cfg)
products = load_preprocessing_products(cfg)
geometries, all_connectors, _ = extract_all_levels(cfg, products)
regions = define_semantic_regions(G, cfg)

# Load results from saved JSON (includes edge_throughput)
def _load_result_from_json(json_path, label, T_s):
    import json
    d = json.loads(json_path.read_text(encoding="utf-8"))
    # edge_throughput keys are stored as "u|v" strings; convert back to tuple keys
    et_raw = d.get("edge_throughput", {})
    et = {}
    for k, v in et_raw.items():
        parts = k.split("|", 1)
        if len(parts) == 2:
            et[k] = v  # keep as string key (fig_flow_diff uses string keys)
    d["edge_throughput"] = et
    d["label"] = label
    d["T_s"] = T_s
    d["replan_events"] = d.get("replan_events", [])
    d["travel_times"] = d.get("travel_times", [])
    tts = d["travel_times"]
    d["mean_travel_time"] = sum(tts) / len(tts) if tts else 0.0
    d["total_replans"] = len(d["replan_events"])
    return d

T_s = cfg["simulation"]["T_s"]
result_b = _load_result_from_json(out_base / "scenB_static" / "result_scenB.json", "scenB_static", T_s)
result_c = _load_result_from_json(out_base / "scenC_dynamic" / "result_scenC.json", "scenC_dynamic", T_s)
result_a = _load_result_from_json(out_base / "scenA_individual" / "result_scenA.json", "scenA_individual", T_s)

print(f"ScenA: {result_a}")
print(f"ScenB: arrive={result_b['arrive_rate']*100:.1f}%  tt={result_b['mean_travel_time']:.1f}s")
print(f"ScenC: arrive={result_c['arrive_rate']*100:.1f}%  tt={result_c['mean_travel_time']:.1f}s  replans={result_c['total_replans']}")

# Wheelchair (fixed: uses replace() now)
run_wheelchair(cfg, G, geometries, regions, out_base)

# Comparison + evaluation
run_comparison([result_b, result_c], G, geometries, out_base, cfg, result_a=result_a)

all_results = [result_a, result_b, result_c]
eval_dir = out_base / "evaluation"
generate_evaluation_report(all_results, eval_dir, cfg=cfg)

print("\nDone: wheelchair + comparison + evaluation complete.")
