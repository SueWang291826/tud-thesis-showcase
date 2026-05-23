# IFC Preprocessing Pipeline for Metro Station Indoor Navigation

## Overview

This pipeline performs **thesis-grade preprocessing** of IFC (Industry Foundation Classes) BIM files from a metro station model, producing clean intermediate data and audit artifacts for downstream indoor navigation research.

**Target domain**: Indoor pedestrian navigation in metro stations.
**Focus levels**: Platform (站台层) and Concourse (站厅层).
**IFC Schema**: IFC2X3 (Revit 2026 export).

## Preprocessing Objectives

1. **Audit** each IFC file for schema, hierarchy, entity types, anomalies
2. **Map storeys** across files to determine functional roles
3. **Classify elements** semantically for navigation relevance
4. **Audit proxy elements** (IfcBuildingElementProxy) in detail
5. **Check geometry readiness** for later graph construction
6. **Export clean intermediate data** in research-friendly formats
7. **Generate visualizations** for inspection and presentation

## Project Structure

```
data-preprocessing/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config/
│   ├── pipeline_config.yaml         # Main configuration
│   └── semantic_policy.yaml         # Semantic filtering rules
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # Main entry point (orchestrator)
│   ├── ifc_loader.py                # IFC file loading & metadata extraction
│   ├── audit.py                     # IFC audit & entity analysis
│   ├── storey_mapping.py            # Cross-file storey mapping
│   ├── semantic_classifier.py       # Rule-based semantic classification
│   ├── proxy_audit.py               # Proxy element analysis
│   ├── geometry_checks.py           # Geometry readiness validation
│   ├── export.py                    # Intermediate data export
│   ├── visualization.py             # Visualization generation
│   └── utils.py                     # Shared utilities
└── outputs/                         # Generated at runtime
    ├── audit/                       # Per-file and cross-file audit reports
    ├── storey_mapping/              # Storey mapping artifacts
    ├── semantic/                    # Semantic classification results
    ├── proxy/                       # Proxy inventory and analysis
    ├── geometry/                    # Geometry readiness checks
    ├── exports/                     # Clean intermediate data for downstream
    ├── figures/                     # All visualizations (PNG + SVG)
    ├── logs/                        # Structured pipeline logs
    └── run_manifest.json            # Run metadata with file hashes
```

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `ifcopenshell`, `pandas`, `matplotlib`, `numpy`, `shapely`, `pyyaml`

### Run the full pipeline

```bash
cd data-preprocessing
python -m src.pipeline
```

The pipeline runs all steps sequentially and saves all outputs to `outputs/`.

### Configuration

Edit `config/pipeline_config.yaml` to adjust:
- Input file paths
- Geometry check parameters
- Visualization settings
- Processing limits

Edit `config/semantic_policy.yaml` to modify:
- Class-based classification rules
- Name-pattern rules (regex, bilingual CN/EN)
- PredefinedType rules

## Output Guide

### Audit (`outputs/audit/`)
- `audit_<file>.json` - Machine-readable audit per file
- `audit_<file>.md` - Human-readable audit report
- `products_<file>.csv` - Full element inventory
- `cross_file_summary.md` - Combined comparison

### Storey Mapping (`outputs/storey_mapping/`)
- `storey_mapping.json` - Complete mapping with functional roles
- `storey_mapping.md` - Narrative explanation
- `storey_file_matrix.csv` - Element count matrix

### Semantic Classification (`outputs/semantic/`)
- `classified_<file>.csv` - Full classification per file
- `category_distribution_<file>.csv` - Category counts
- `cross_file_category_comparison.csv` - Combined comparison

### Proxy Audit (`outputs/proxy/`)
- `proxy_inventory_<file>.csv` - Detailed proxy inventory with editable `reviewed_category` column
- `proxy_summary_<file>.json` - Proxy statistics
- `proxy_inventory_combined.csv` - Combined proxy inventory

### Geometry (`outputs/geometry/`)
- `geometry_check_<file>.csv` - Representation analysis
- `bbox_sample_<file>.csv` - Sampled bounding boxes
- `geometry_summary_<file>.json` - Readiness summary

### Exports (`outputs/exports/`)
- `all_elements_classified.csv` - Complete classified inventory
- `by_storey/` - Per-storey element tables
- `by_category/` - Per-category element tables
- `walkable_level_elements.csv` - Platform + Concourse elements
- `connector_candidates.csv` - Vertical connector candidates
- `obstacle_candidates.csv` - Obstacle candidates

### Figures (`outputs/figures/`)
- `audit/` - Entity count charts, proxy proportion
- `storey_mapping/` - Heatmap, dominance charts
- `semantic/` - Category distributions, pie charts
- `proxy/` - Proxy by storey, categories, bbox distributions
- `geometry/` - Readiness charts, bbox distributions
- `spatial_preview/` - Plan-view previews per storey

## Key Assumptions (Explicit)

1. IFC files use IFC2X3 schema with millimeter units
2. IfcSpace is NOT available (excluded from Revit export)
3. IfcProxy is excluded; IfcBuildingElementProxy is present
4. IfcTransportElement is excluded; vertical connectors may be proxies
5. Platform (F1, 0mm) and Concourse (F3, 12100mm) are public walkable levels
6. Equipment level (F2, 5300mm) is NOT a public walkable level
7. All three files share the same project/building/storey hierarchy
8. Storey assignment uses IfcRelContainedInSpatialStructure

## Proxy Review Workflow

The proxy inventory CSV files contain an empty `reviewed_category` column.
To manually review proxies:

1. Open `outputs/proxy/proxy_inventory_<file>.csv` in Excel
2. Fill in `reviewed_category` where you disagree with `inferred_category`
3. Add notes in `reviewer_notes` column
4. Re-run the reingest function to update statistics

## What This Pipeline Does NOT Do

- Build the navigation graph (2.5D or otherwise)
- Implement pedestrian simulation
- Extract precise walkable surface geometry
- Generate routing paths

These are downstream tasks that consume the clean intermediate outputs.

## Reproducibility

Each run generates:
- A unique run ID (UTC timestamp)
- SHA-256 hashes of all input files
- A complete run manifest (`outputs/run_manifest.json`)
- Structured logs with timestamps
