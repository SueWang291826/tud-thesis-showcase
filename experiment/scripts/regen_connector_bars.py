"""Regenerate connector_load_bars.png with improved legend labels."""
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config
from src.viz_thesis import fig_connector_load_bars

cfg = load_config("config/experiment_config.yaml")
G = pickle.load(open("outputs/step3_graph/navigation_graph.gpickle", "rb"))

res_b = json.loads(Path("outputs/step5_simulation/scenB_static/result_scenB.json").read_text(encoding="utf-8"))
res_c = json.loads(Path("outputs/step5_simulation/scenC_dynamic/result_scenC.json").read_text(encoding="utf-8"))
res_b["label"] = "scenB_static"
res_c["label"] = "scenC_dynamic"

out = Path("outputs/step5_simulation/comparison")
fig_connector_load_bars([res_b, res_c], G, out, cfg=cfg)
print("done")
