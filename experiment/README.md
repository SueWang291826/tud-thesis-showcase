# Multi-Level Indoor Navigation Experiment Framework

## Overview

Thesis-grade, reproducible experiment framework for multi-level metro station
indoor navigation. Upgrades from demo_v3 prototype to a formal 4-level 2.5D
navigation system with typed connectors, real obstacle geometry, and
configurable routing/simulation.

## Station Levels

| Level | Chinese | English     | Elev (m) | Public | Role                              |
|-------|---------|-------------|----------|--------|-----------------------------------|
| F0    | 底板层  | Base Slab   | -1.7     | No     | structural_base – excluded        |
| F1    | 站台层  | Platform    |  0.0     | **Yes**| platform – passenger boarding     |
| F2    | 设备层  | Equipment   |  5.3     | No     | connector-only – stair/escalator  |
| F3    | 站厅层  | Concourse   | 12.1     | **Yes**| concourse – fare gates, services  |
| F4    | 交通层  | Transport   | 17.4     | **Yes**| surface / street-level access     |
| F5RF  | 顶板    | Roof Slab   | 24.6     | No     | structural_roof – excluded        |

- **F2 is NOT a public walkable floor.** It exists only for connector continuity
  (stair/escalator geometry passes through F2).

## Pipeline Steps

```
Step 0: load_data        – Load v2/v3 CSV products, IFC subsets
Step 1: extract_geometry  – IFC → floor polygons, obstacles, connectors per level
Step 2: sample_nodes      – Grid sampling + clearance filtering per walkable level
Step 3: build_graph       – 2.5D unified navigation graph (F1+F3+F4 + typed connectors)
Step 4: routing           – Multi-criteria pathfinding with connector costs
Step 5: simulate          – ABM with heterogeneous agents
Step 6: evaluate          – Metrics, comparisons, thesis figures
```

## Connector Types

| Subtype       | Count | Connects       | Modelling                         |
|---------------|-------|----------------|-----------------------------------|
| stair         | 38    | F1↔F2, F2↔F3   | Chain nodes, capacity-gated       |
| stair_flight  | 81    | (children)     | Merged into parent stair chains   |
| escalator     | 16    | F1↔F3          | Directional, fixed speed 0.5 m/s  |
| elevator      | 3     | F1↔F3          | Batch capacity, fixed dwell time  |
| fare_gate     | —     | F3 internal    | Passage constraint, queue delay   |

## Obstacle Categories (from v3 recalibration)

| Subcategory                 | Count | Action | Modelling                       |
|-----------------------------|-------|--------|---------------------------------|
| obstacle_floor_intrusive    | 1081  | KEEP   | Columns, pillars, kiosks        |
| obstacle_barrier_relevant   | 151   | KEEP   | Railings, barriers              |
| obstacle_clearance_relevant | 112   | KEEP   | Wall segments, structural       |
| obstacle_skin_panel         | 4939  | DROP   | Decorative façade panels        |
| obstacle_uncertain          | 230   | DROP   | Ambiguous – conservative drop   |
| obstacle_small_irrelevant   | 132   | DROP   | Tiny fittings                   |

## Directory Structure

```
experiment/
├── config/
│   └── experiment_config.yaml    # All parameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Step 0: load CSV + IFC
│   ├── geometry_extractor.py     # Step 1: IFC → Shapely polygons
│   ├── node_sampler.py           # Step 2: grid sampling + clearance
│   ├── connector_builder.py      # Typed connector modelling
│   ├── graph_builder.py          # Step 3: 2.5D graph construction
│   ├── routing.py                # Step 4: pathfinding algorithms
│   ├── simulation.py             # Step 5: ABM engine
│   ├── evaluation.py             # Step 6: metrics & analysis
│   ├── viz.py                    # All visualization functions
│   └── utils.py                  # Shared utilities
├── scripts/
│   ├── run_pipeline.py           # End-to-end pipeline runner
│   ├── step0_load.py
│   ├── step1_extract.py
│   ├── step2_sample.py
│   ├── step3_graph.py
│   ├── step4_route.py
│   ├── step5_simulate.py
│   └── step6_evaluate.py
├── outputs/                      # Generated outputs (gitignored)
├── requirements.txt
└── README.md
```

## Differences from demo_v3

| Aspect                | demo_v3                         | This Framework                      |
|-----------------------|---------------------------------|-------------------------------------|
| Levels                | 2 (hardcoded lowest two)        | 4 (F1, F2-connector, F3, F4)       |
| Obstacles             | Walls only                      | Columns + barriers + walls          |
| Connectors            | Stairs only (1 type)            | Stairs, escalators, elevators, gates|
| Graph                 | Single undirected                | Directed multigraph (capacity/dir)  |
| Routing               | Static Dijkstra only            | Static + dynamic + multi-criteria   |
| Simulation            | Single occupancy grid           | Enhanced ABM with typed connectors  |
| Data source           | Single IFC file                 | 4 IFC files, preprocessed CSV       |
| Reproducibility       | config.json only                | Full YAML + manifest + seed control |

## Requirements

```
ifcopenshell>=0.7
shapely>=2.0
networkx>=3.2
matplotlib>=3.8
numpy>=1.24
pandas>=2.0
pyyaml>=6.0
```
