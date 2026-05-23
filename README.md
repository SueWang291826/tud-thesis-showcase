# TUD Station Thesis Project

This repository contains the full MSc thesis project for multi-level indoor navigation in an urban rail transit station. It combines IFC preprocessing, navigation graph construction, routing and simulation experiments, and a public-facing static showcase website.

## Repository Scope

- data-preprocessing: thesis-grade IFC preprocessing pipeline for audit, semantic classification, geometry checks, and clean intermediate exports
- experiment: the main experiment framework for geometry extraction, 2.5D graph building, routing, simulation, evaluation, and the LLM navigation agent
- data0: original IFC inputs tracked with Git LFS
- site: static showcase website for GitHub Pages deployment

## Core Workflow

1. Preprocess BIM data from the IFC station models in data-preprocessing
2. Build the navigation-ready geometry, graph, routes, and simulations in experiment
3. Publish results and interactive visualizations through the site website

## Main Project Components

### 1. IFC preprocessing

The preprocessing pipeline audits IFC files, aligns storeys across files, classifies elements by navigation relevance, checks geometry readiness, and exports structured datasets for downstream research.

Key outputs include:

- storey mappings and semantic inventories
- obstacle and connector candidate tables
- geometry readiness reports
- inspection figures and run manifests

See [data-preprocessing/README.md](data-preprocessing/README.md) for pipeline details.

### 2. Navigation and simulation experiments

The experiment framework models a four-level station with typed vertical connectors and a unified 2.5D navigation graph. It supports routing, dynamic simulation, evaluation, and thesis figure generation.

Key features include:

- multi-level station representation with public and connector-only floors
- typed stairs, escalators, elevators, and fare-gate constraints
- routing and agent-based simulation workflows
- evaluation outputs and interactive visualizations
- LLM navigation agent components with tool calling and retrieval support

See [experiment/README.md](experiment/README.md) for experiment details.

## Static Showcase Site

The repository includes a static project website in [site/README.md](site/README.md). GitHub Pages is configured through a workflow that deploys the contents of the site directory directly, so the published site is available from the repository Pages URL without the extra /site path.

Current Pages URL:

- https://suewang291826.github.io/tud-thesis-showcase/

## Data and Storage Notes

- The IFC source files in data0 are stored with Git LFS because they exceed normal GitHub file size limits.
- Runtime outputs, local environments, archives, and private .env files are intentionally excluded from version control.

## Quick Start

### Preprocessing

```bash
cd data-preprocessing
pip install -r requirements.txt
python -m src.pipeline
```

### Experiments

```bash
cd experiment
pip install -r requirements.txt
python pipeline/scripts/run_pipeline.py
```

### Local website preview

```bash
cd site
python -m http.server 8080
```

Then open http://localhost:8080.

## Repository Structure

```text
.
├── data-preprocessing/
├── data0/
├── experiment/
├── site/
└── outputs/                # local generated outputs, not tracked
```

## Thesis Context

This project supports the thesis LLM Navigation Agent for Urban Rail Transit Stations at TU Delft. The repository is organized to keep research code, data processing, experimental evaluation, and public presentation in one place while separating generated artifacts from source materials.