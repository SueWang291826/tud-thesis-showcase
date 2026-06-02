# Interactive BIM-to-Navigation Digital Twin Demonstrator

This directory contains an isolated Vite + TypeScript + Three.js demonstrator for thesis-defense use. It does not overwrite the root thesis pipeline, the existing site directory, or research code outside this folder.

## What this app shows

The demonstrator walks through seven deterministic stages:

1. BIM model
2. BIM-derived surface-sampled point cloud
3. Movement-relevant geometry
4. 2.5D navigation graph
5. Snapped route
6. Simulation
7. Integrated overview

Optional MediaPipe gesture control can be enabled only after clicking Start Gesture Control. If local MediaPipe assets are missing or webcam permission is denied, the app shows a warning and keeps keyboard and mouse controls active.
If webcam permission is denied, the gesture panel falls back to a swipe simulator: short drags trigger stage commands, and hold-then-drag grabs the BIM model for direct rotation.

## Repository inventory used by this demo

Authoritative thesis inputs and outputs are read from the existing repository structure:

- IFC inputs: ../../data0/设备层.ifc, ../../data0/站台层.ifc, ../../data0/站厅层.ifc, ../../data0/交通层.ifc
- Graph source: ../../experiment/outputs/step3_graph/nodes_all.geojson and edges_*.geojson
- Graph summary: ../../experiment/outputs/step3_graph/graph_summary.json
- Semantic points: ../../experiment/outputs/step4_routing/semantic_points.geojson
- Route examples: ../../experiment/outputs/step4_routing/example_paths.json
- Simulation outputs: ../../experiment/outputs/step5_simulation/dynamic and ../../experiment/outputs/step5_simulation/static

The adapter script converts those assets into frontend-facing schemas under public/data:

- public/data/navigation_graph.json
- public/data/semantic_anchors.json
- public/data/routes.json
- public/data/simulation.json

## Commands

From inside this directory:

```bash
npm install
npm run dev
npm run build
npm run preview
npm run export:station-bim
```

Helper commands:

```bash
python scripts/prepare_thesis_demo_data.py
python scripts/validate_demo_assets.py
python scripts/create_placeholder_assets.py --force
```

Convenience npm scripts:

```bash
npm run prepare:data
npm run validate:assets
npm run placeholder:assets
npm run export:station-bim
```

## Asset replacement

### BIM model

If you have a GLB export, place it at:

- public/models/station_bim.glb

To generate that GLB directly from the repository IFC stack with the locally installed ifcopenshell package:

```bash
npm run export:station-bim
```

If the file is missing, Stage 1 shows a placeholder massing derived from graph extents.

### Point cloud

If you have a PLY file sampled from the BIM mesh, place it at:

- public/pointcloud/station_surface_sampled.ply

If the file is missing, Stage 2 shows a placeholder point cloud derived from navigation nodes.

### MediaPipe gesture assets

If you want local gesture recognition, place the assets at:

- public/mediapipe/gesture_recognizer.task
- public/mediapipe/wasm/

If these assets are absent, the demo stays usable with keyboard and mouse controls only.

## IFC to GLB conversion

IfcOpenShell exposes IfcConvert for offline format conversion.

PowerShell:

```powershell
./scripts/convert_ifc_to_glb.ps1 ../../data0/站厅层.ifc ./public/models/station_bim.glb
```

Bash:

```bash
./scripts/convert_ifc_to_glb.sh ../../data0/站厅层.ifc ./public/models/station_bim.glb
```

## Surface sampling

Install the optional Python helpers:

```bash
pip install -r requirements-demo.txt
```

Then sample a mesh to an ASCII PLY point cloud:

```bash
python scripts/sample_mesh_to_pointcloud.py ./public/models/station_bim.glb ./public/pointcloud/station_surface_sampled.ply --samples 50000 --seed 42
```

## Configuration

All app-facing paths and defaults live in public/config/demo-config.json.

Configurable items include:

- asset paths
- placeholder toggles
- camera defaults
- visual colors and transition timing
- graph and movement render step
- MediaPipe local asset paths and recognition thresholds

If your GLB or PLY uses a different origin or orientation, edit the transform blocks for bimModel or pointCloud in public/config/demo-config.json.

## Placeholder mode

The app is designed to stay usable even when defense-time assets are incomplete.

- Missing GLB: Stage 1 uses placeholder BIM massing.
- Missing PLY: Stage 2 uses a placeholder point cloud.
- Missing movement geometry asset: Stage 3 derives fallback geometry from walkable graph nodes.
- Missing simulation file: Stage 6 uses an illustrative fallback and labels it as such.
- Missing MediaPipe assets or denied webcam permission: a warning is shown and keyboard or mouse controls continue to work.

To generate placeholder JSON assets explicitly:

```bash
python scripts/create_placeholder_assets.py --force
```

## Offline notes

- The app has no backend and does not require network requests beyond local static asset loading.
- The demonstrator can run entirely from local files served by Vite preview or dev server.
- MediaPipe is optional and should be provided locally if gesture control is required offline.
- The point cloud should be presented as an explanatory visualization derived from BIM, not as measured reality capture.

## Required point-cloud disclaimer

Use this exact wording in thesis-defense narration and do not replace it with measured-data language:

> This point cloud is a BIM-derived, surface-sampled visualization generated from IFC/mesh geometry. It is not measured LiDAR, photogrammetry, or any other sensor-captured reality data.

## Keyboard controls

- 1-7: jump to stages
- ArrowLeft and ArrowRight: previous or next stage
- Space: play or pause simulation
- R: reset camera
- A: toggle auto-rotate
- G: toggle gesture control
- H: toggle help overlay
- F: toggle fullscreen
- Esc: close overlays

## Manual test checklist

- Run npm install
- Run npm run build
- Run npm run dev
- Run npm run preview
- Confirm placeholder mode appears when public/models/station_bim.glb is absent
- Confirm placeholder mode appears when public/pointcloud/station_surface_sampled.ply is absent
- Confirm navigation_graph.json, semantic_anchors.json, routes.json, and simulation.json are present in public/data
- Confirm stage switching works with buttons, number keys, and arrow keys
- Confirm OrbitControls, reset view, fullscreen, and help overlay work
- Confirm Stage 2 visibly shows the BIM-derived point-cloud disclaimer
- Confirm route selection changes the highlighted snapped route
- Confirm simulation selection changes the loaded scenario
- Confirm missing MediaPipe assets or denied webcam permission only produce warnings
- Confirm graph metrics render from data and missing values show Not provided

## Notes on generated data

The current adapter has already been run against the available thesis outputs in this repository. That generated:

- navigation graph with 18454 nodes and 134858 edges
- 5 route examples
- 2 simulation scenarios

Re-run python scripts/prepare_thesis_demo_data.py whenever the authoritative thesis outputs change.
