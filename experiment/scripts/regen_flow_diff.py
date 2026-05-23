"""Regenerate both flow_diff figures with improved subtitles."""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config
from src.viz_thesis import fig_flow_diff_two_scenarios
from src.data_loader import load_preprocessing_products
from src.geometry_extractor import extract_all_levels

cfg = load_config("config/experiment_config.yaml")
G = pickle.load(open("outputs/step3_graph/navigation_graph.gpickle", "rb"))
products = load_preprocessing_products(cfg)
geometries, _, _ = extract_all_levels(cfg, products)

res_a = json.loads(Path("outputs/step5_simulation/scenA_individual/result_scenA.json").read_text(encoding="utf-8"))
res_b = json.loads(Path("outputs/step5_simulation/scenB_static/result_scenB.json").read_text(encoding="utf-8"))
res_c = json.loads(Path("outputs/step5_simulation/scenC_dynamic/result_scenC.json").read_text(encoding="utf-8"))

out = Path("outputs/step5_simulation/comparison")

# A vs B (individual vs static)
fig_flow_diff_two_scenarios(res_a, res_b, G, geometries, out,
                             label_a="A", label_b="B", cfg=cfg)

# B vs C (static vs dynamic)
fig_flow_diff_two_scenarios(res_b, res_c, G, geometries, out,
                             label_a="static", label_b="dynamic", cfg=cfg)
print("done")
