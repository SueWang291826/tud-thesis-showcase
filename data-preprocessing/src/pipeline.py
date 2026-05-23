"""
IFC Preprocessing Pipeline - Main Entry Point.

This is the single entry point for the entire preprocessing pipeline.
It orchestrates all modules in a deterministic sequence:

Phase 1 (v1) — Audit & Classification:
  1. Load IFC files
  2. Audit each file
  3. Storey mapping
  4. Semantic classification
  5. Proxy audit
  6. Geometry readiness checks
  7. Export intermediate data
  8. Generate visualizations
  9. Save run manifest

Phase 2 (v2) — Experimental Subset Generation:
  10. Unit normalization (→ metres)
  11. Automatic proxy disambiguation
  12. Cross-storey traffic filtering
  13. Filtered IFC subset export
  14. V2 visualizations
  15. V2 validation report

Usage:
    cd data-preprocessing/
    python -m src.pipeline

All v1 outputs go to outputs/; v2 outputs go to outputs/v2/.
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    setup_logging,
    ensure_output_dirs,
    create_run_manifest,
    save_run_manifest,
    Timer,
    save_json,
)
from src.ifc_loader import IFCFileInfo, load_ifc_file
from src.audit import audit_single_file, generate_cross_file_audit
from src.storey_mapping import generate_storey_mapping
from src.semantic_classifier import (
    SemanticClassifier,
    classify_file_elements,
    generate_cross_file_classification_summary,
)
from src.proxy_audit import (
    audit_proxies_single_file,
    generate_cross_file_proxy_summary,
)
from src.geometry_checks import (
    check_geometry_single_file,
    generate_cross_file_geometry_summary,
)
from src.export import export_intermediate_data
from src.visualization import generate_all_visualizations

# V2 modules
from src.unit_normalizer import run_unit_normalization
from src.proxy_disambiguator import run_proxy_disambiguation
from src.traffic_filter import run_traffic_filter
from src.ifc_export import export_filtered_ifc
from src.v2_visualizations import generate_v2_visualizations
from src.validation import generate_validation_report


def main():
    """Run the full preprocessing pipeline."""
    pipeline_start = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ---- Configuration ----
    config_path = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    policy_path = PROJECT_ROOT / "config" / "semantic_policy.yaml"

    config = load_config(str(config_path))
    output_dirs = ensure_output_dirs(config, PROJECT_ROOT)

    # V2 configuration
    v2_config_path = PROJECT_ROOT / "config" / "v2_config.yaml"
    v2_config = load_config(str(v2_config_path))

    # Ensure v2 output directories exist
    v2_output_cfg = v2_config.get("v2_output", {})
    v2_dirs = {}
    for key, rel_path in v2_output_cfg.items():
        abs_path = PROJECT_ROOT / rel_path
        abs_path.mkdir(parents=True, exist_ok=True)
        v2_dirs[key] = abs_path

    # ---- Logging ----
    logger = setup_logging(str(output_dirs["logs_dir"]), run_id)
    logger.info("=" * 70)
    logger.info("IFC PREPROCESSING PIPELINE")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("=" * 70)

    # ---- Resolve IFC file paths ----
    data_dir = (PROJECT_ROOT / config["input"]["data_dir"]).resolve()
    ifc_files: Dict[str, Path] = {}
    for label, filename in config["input"]["ifc_files"].items():
        fpath = data_dir / filename
        if not fpath.exists():
            logger.error(f"IFC file not found: {fpath}")
            sys.exit(1)
        ifc_files[label] = fpath
        logger.info(f"  Input: {label} -> {fpath} ({fpath.stat().st_size / 1e6:.1f} MB)")

    # ---- Run manifest (before processing starts, with file hashes) ----
    logger.info("Computing input file hashes for reproducibility...")
    manifest = create_run_manifest(config, PROJECT_ROOT, ifc_files, run_id)

    # ==================================================================
    # STEP 1: Load IFC files
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 1: Loading IFC files")
    logger.info("=" * 50)

    file_infos: Dict[str, IFCFileInfo] = {}
    for label, fpath in ifc_files.items():
        with Timer(f"Loading {label}", logger):
            file_infos[label] = load_ifc_file(fpath, label)

    # ==================================================================
    # STEP 2: Audit each file
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 2: Auditing IFC files")
    logger.info("=" * 50)

    audits: Dict[str, dict] = {}
    for label, fi in file_infos.items():
        with Timer(f"Auditing {label}", logger):
            audits[label] = audit_single_file(fi, output_dirs["audit_dir"])

    # Cross-file audit
    with Timer("Cross-file audit", logger):
        cross_audit = generate_cross_file_audit(audits, output_dirs["audit_dir"])

    # ==================================================================
    # STEP 3: Storey mapping
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 3: Storey mapping")
    logger.info("=" * 50)

    with Timer("Storey mapping", logger):
        storey_mapping = generate_storey_mapping(
            audits, config, output_dirs["storey_dir"]
        )

    # ==================================================================
    # STEP 4: Semantic classification
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 4: Semantic classification")
    logger.info("=" * 50)

    classifier = SemanticClassifier(str(policy_path))

    classified_dfs: Dict[str, "pd.DataFrame"] = {}
    for label, fi in file_infos.items():
        with Timer(f"Classifying {label}", logger):
            classified_dfs[label] = classify_file_elements(
                fi, classifier, output_dirs["semantic_dir"]
            )

    with Timer("Cross-file classification summary", logger):
        cross_classification = generate_cross_file_classification_summary(
            classified_dfs, output_dirs["semantic_dir"]
        )

    # ==================================================================
    # STEP 5: Proxy audit
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 5: Proxy audit")
    logger.info("=" * 50)

    proxy_dfs: Dict[str, "pd.DataFrame"] = {}
    proxy_summaries: Dict[str, dict] = {}
    for label, fi in file_infos.items():
        with Timer(f"Proxy audit {label}", logger):
            pdf, psummary = audit_proxies_single_file(
                fi, classifier, output_dirs["proxy_dir"],
                extract_bbox=config["processing"].get("extract_geometry", True),
            )
            proxy_dfs[label] = pdf
            proxy_summaries[label] = psummary

    with Timer("Cross-file proxy summary", logger):
        cross_proxy = generate_cross_file_proxy_summary(
            proxy_dfs, proxy_summaries, output_dirs["proxy_dir"]
        )

    # ==================================================================
    # STEP 6: Geometry readiness checks
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 6: Geometry readiness checks")
    logger.info("=" * 50)

    geom_dfs: Dict[str, "pd.DataFrame"] = {}
    geom_summaries: Dict[str, dict] = {}
    bbox_dfs: Dict[str, "pd.DataFrame"] = {}

    for label, fi in file_infos.items():
        with Timer(f"Geometry check {label}", logger):
            gdf, gsummary = check_geometry_single_file(
                fi, config, output_dirs["geometry_dir"],
                sample_bbox=True,
                max_bbox_sample=2000,
            )
            geom_dfs[label] = gdf
            geom_summaries[label] = gsummary

            # Load bbox sample if exists
            bbox_path = output_dirs["geometry_dir"] / f"bbox_sample_{label}.csv"
            if bbox_path.exists():
                import pandas as pd
                bbox_dfs[label] = pd.read_csv(bbox_path, encoding="utf-8-sig")

    with Timer("Cross-file geometry summary", logger):
        cross_geom = generate_cross_file_geometry_summary(
            geom_summaries, output_dirs["geometry_dir"]
        )

    # ==================================================================
    # STEP 7: Export intermediate data
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 7: Exporting intermediate data")
    logger.info("=" * 50)

    with Timer("Intermediate export", logger):
        export_manifest = export_intermediate_data(
            classified_dfs, storey_mapping, proxy_dfs,
            config, output_dirs["exports_dir"],
        )

    # ==================================================================
    # STEP 8: Generate visualizations
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 8: Generating visualizations")
    logger.info("=" * 50)

    with Timer("All visualizations", logger):
        vis_cfg = config.get("visualization", {})
        generate_all_visualizations(
            audits=audits,
            storey_mapping=storey_mapping,
            classified_dfs=classified_dfs,
            proxy_dfs=proxy_dfs,
            geom_summaries=geom_summaries,
            bbox_dfs=bbox_dfs,
            output_dir=output_dirs["figures_dir"],
            save_svg=vis_cfg.get("save_svg", True),
        )

    # ==================================================================
    # STEP 9: Save run manifest (v1)
    # ==================================================================
    v1_elapsed = time.perf_counter() - pipeline_start
    manifest["v1_elapsed_seconds"] = round(v1_elapsed, 1)
    manifest["steps_completed"] = [
        "ifc_loading", "audit", "storey_mapping", "semantic_classification",
        "proxy_audit", "geometry_checks", "intermediate_export", "visualization",
    ]
    save_run_manifest(manifest, PROJECT_ROOT / "outputs" / "run_manifest.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"V1 PIPELINE COMPLETE — {v1_elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("Starting V2: Experimental Subset Generation ...")
    logger.info("")

    # ==================================================================
    # STEP 10: Unit normalization (v2)
    # ==================================================================
    logger.info("=" * 50)
    logger.info("STEP 10: Unit normalization → metres (v2)")
    logger.info("=" * 50)

    with Timer("Unit normalization", logger):
        unit_results = run_unit_normalization(
            classified_dfs=classified_dfs,
            bbox_dfs=bbox_dfs,
            proxy_dfs=proxy_dfs,
            config=config,
            v2_config=v2_config,
            output_dir=v2_dirs["normalized_dir"],
        )
    norm_elements = unit_results["norm_elements"]

    # ==================================================================
    # STEP 11: Automatic proxy disambiguation (v2)
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 11: Automatic proxy disambiguation (v2)")
    logger.info("=" * 50)

    with Timer("Proxy disambiguation", logger):
        proxy_results = run_proxy_disambiguation(
            proxy_dfs=proxy_dfs,
            v2_config=v2_config,
            output_dir=v2_dirs["proxy_resolved_dir"],
        )
    proxy_resolved = proxy_results["combined"]

    # ==================================================================
    # STEP 12: Cross-storey traffic filtering (v2)
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 12: Cross-storey traffic filtering (v2)")
    logger.info("=" * 50)

    with Timer("Traffic filtering", logger):
        filter_results = run_traffic_filter(
            norm_elements=norm_elements,
            proxy_resolved=proxy_resolved,
            v2_config=v2_config,
            output_dir=v2_dirs["traffic_filtered_dir"],
        )
    retained_df = filter_results["retained"]
    all_filtered = filter_results["all_filtered"]

    # ==================================================================
    # STEP 13: Filtered IFC subset export (v2)
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 13: Filtered IFC subset export (v2)")
    logger.info("=" * 50)

    with Timer("IFC subset export", logger):
        ifc_results = export_filtered_ifc(
            retained_df=retained_df,
            file_infos=file_infos,
            v2_config=v2_config,
            output_dir=v2_dirs["ifc_subsets_dir"],
        )

    # ==================================================================
    # STEP 14: V2 visualizations
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 14: V2 visualizations")
    logger.info("=" * 50)

    with Timer("V2 visualizations", logger):
        tf_cfg = v2_config.get("traffic_filter", {})
        public_levels = tf_cfg.get("public_walkable_levels", [])
        vis_cfg = config.get("visualization", {})
        generate_v2_visualizations(
            proxy_resolved=proxy_resolved,
            all_filtered=all_filtered,
            retained_df=retained_df,
            bbox_dfs=bbox_dfs,
            public_levels=public_levels,
            output_dir=v2_dirs["figures_v2_dir"],
            save_svg=vis_cfg.get("save_svg", True),
        )

    # ==================================================================
    # STEP 15: Validation report (v2)
    # ==================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 15: Validation report (v2)")
    logger.info("=" * 50)

    with Timer("Validation report", logger):
        validation_report = generate_validation_report(
            unit_results=unit_results,
            proxy_results=proxy_results,
            filter_results=filter_results,
            ifc_results=ifc_results,
            v2_config=v2_config,
            output_dir=v2_dirs["validation_dir"],
        )

    # ==================================================================
    # Final manifest update
    # ==================================================================
    pipeline_elapsed = time.perf_counter() - pipeline_start
    manifest["pipeline_elapsed_seconds"] = round(pipeline_elapsed, 1)
    manifest["v2_elapsed_seconds"] = round(pipeline_elapsed - v1_elapsed, 1)
    manifest["steps_completed"].extend([
        "unit_normalization", "proxy_disambiguation", "traffic_filter",
        "ifc_subset_export", "v2_visualizations", "v2_validation",
    ])
    manifest["v2_validation_status"] = validation_report["overall"]["status"]
    save_run_manifest(manifest, PROJECT_ROOT / "outputs" / "run_manifest.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("FULL PIPELINE COMPLETE (v1 + v2)")
    logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f}min)")
    logger.info(f"  v1: {v1_elapsed:.1f}s  |  v2: {pipeline_elapsed - v1_elapsed:.1f}s")
    logger.info(f"V2 validation: {validation_report['overall']['status']}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"V1 outputs: {output_dirs['base_dir']}")
    logger.info(f"V2 outputs: {v2_dirs['base_dir']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
