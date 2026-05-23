"""
Re-run all scenarios A/B/C + wheelchair + comparison + evaluation with fixed simulation.

Skips the slow full capacity sweep — that is handled separately by:
  - scripts/run_dynamic_sweep.py       (N=1000/1500/2000 dynamic)
  - scripts/rerun_dynamic_small.py     (N=200/500 dynamic)

Patches applied in this run:
  1. Per-unit escalator capacity (previously global)
  2. F4→F3 inbound escalator now routed via esc_18Gic2sdj5_OIgDswc3FF2 (direction=down)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Make sure we run from experiment root
import os
os.chdir(ROOT)

from pipeline.scripts.step5_experiments import (
    _load_graph, run_scenA, run_scenB, run_scenC,
    run_wheelchair, run_comparison
)
from src.utils import load_config
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels
from src.routing import define_semantic_regions
from src.viz_thesis import generate_evaluation_report

cfg_path = ROOT / "config" / "experiment_config.yaml"
cfg = load_config(str(cfg_path))
out_base = Path(ROOT / cfg["output"]["step_dirs"]["step5"])
out_base.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RERUN ALL SCENARIOS (Fixed simulation: per-unit esc cap + F4→F3 esc inbound)")
print("=" * 60)

G = _load_graph(cfg)
products = load_preprocessing_products(cfg)
geometries, all_connectors, _ = extract_all_levels(cfg, products)
regions = define_semantic_regions(G, cfg)

print(f"\n  Entrance nodes : {len(regions.get('ENTRANCE', []))}")
print(f"  Platform nodes : {len(regions.get('PLATFORM', []))}")

result_a = run_scenA(cfg, G, geometries, regions, out_base)
result_b = run_scenB(cfg, G, geometries, regions, out_base)
result_c = run_scenC(cfg, G, geometries, regions, out_base)

run_wheelchair(cfg, G, geometries, regions, out_base)

run_comparison([result_b, result_c], G, geometries, out_base, cfg, result_a=result_a)

# Also regenerate figures from all-scenario list (including A)
all_results = [result_a, result_b, result_c]
eval_dir = out_base / "evaluation"
generate_evaluation_report(all_results, eval_dir, cfg=cfg)

print("\n" + "=" * 60)
print(f"All scenarios complete → {out_base}")
print("=" * 60)
