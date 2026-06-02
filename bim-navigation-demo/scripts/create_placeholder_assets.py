from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXACT_POINT_CLOUD_DISCLAIMER = (
    'This point cloud is a BIM-derived, surface-sampled visualization generated from IFC/mesh geometry. '
    'It is not measured LiDAR, photogrammetry, or any other sensor-captured reality data.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create placeholder JSON assets for the BIM navigation demo.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'public' / 'data',
        help='Target directory for placeholder JSON assets.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files.',
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any, force: bool) -> None:
    if path.exists() and not force:
        print(f'Skipping existing file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'Wrote {path}')


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    graph = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/create_placeholder_assets.py',
            'sourcePaths': ['placeholder://graph'],
        },
        'nodes': [
            {'id': 'F1_a', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'level': 'F1', 'nodeType': 'floor', 'usable': True, 'clearance': None, 'blindCategory': '', 'surfaceType': 'normal'},
            {'id': 'F1_b', 'x': 12.0, 'y': 0.0, 'z': 0.0, 'level': 'F1', 'nodeType': 'platform', 'usable': True, 'clearance': None, 'blindCategory': '', 'surfaceType': 'normal'},
            {'id': 'F3_a', 'x': 12.0, 'y': 12.0, 'z': 6.0, 'level': 'F3', 'nodeType': 'floor', 'usable': True, 'clearance': None, 'blindCategory': '', 'surfaceType': 'normal'},
            {'id': 'F4_a', 'x': 12.0, 'y': 24.0, 'z': 12.0, 'level': 'F4', 'nodeType': 'entrance', 'usable': True, 'clearance': None, 'blindCategory': '', 'surfaceType': 'normal'}
        ],
        'edges': [
            {'id': 'e1', 'source': 'F1_a', 'target': 'F1_b', 'length2d': 12.0, 'length3d': 12.0, 'travelTime': 10.0, 'edgeType': 'floor', 'level': 'F1'},
            {'id': 'e2', 'source': 'F1_b', 'target': 'F3_a', 'length2d': 12.0, 'length3d': 13.4, 'travelTime': 18.0, 'edgeType': 'stair', 'level': 'STAIR'},
            {'id': 'e3', 'source': 'F3_a', 'target': 'F4_a', 'length2d': 12.0, 'length3d': 13.4, 'travelTime': 18.0, 'edgeType': 'entrance', 'level': 'F4'}
        ],
        'levels': ['F1', 'F3', 'F4'],
        'bounds': {
            'min': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'max': {'x': 12.0, 'y': 24.0, 'z': 12.0}
        },
        'metrics': {
            'total_nodes': 4,
            'total_edges': 3,
            'is_connected': True,
            'note': 'Placeholder graph created for demonstrator fallback mode.'
        }
    }

    anchors = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/create_placeholder_assets.py',
            'sourcePaths': ['placeholder://anchors'],
        },
        'anchors': [
            {'id': 'anchor_platform', 'type': 'PLATFORM', 'level': 'F1', 'x': 12.0, 'y': 0.0, 'z': 0.0},
            {'id': 'anchor_exit', 'type': 'EXIT', 'level': 'F4', 'x': 12.0, 'y': 24.0, 'z': 12.0}
        ],
        'counts': {'EXIT': 1, 'PLATFORM': 1}
    }

    routes = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/create_placeholder_assets.py',
            'sourcePaths': ['placeholder://routes'],
        },
        'defaultRouteId': 'route_01',
        'routes': [
            {
                'id': 'route_01',
                'label': 'Placeholder platform to exit',
                'originNodeId': 'F1_a',
                'destinationNodeId': 'F4_a',
                'sourceKind': 'derived-dijkstra',
                'snappedNodeIds': ['F1_a', 'F1_b', 'F3_a', 'F4_a'],
                'polyline': [
                    {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    {'x': 12.0, 'y': 0.0, 'z': 0.0},
                    {'x': 12.0, 'y': 12.0, 'z': 6.0},
                    {'x': 12.0, 'y': 24.0, 'z': 12.0}
                ],
                'metrics': {
                    'totalTravelTime': 46.0,
                    'totalLength2d': 36.0,
                    'levelsVisited': ['F1', 'F3', 'F4'],
                    'disclaimer': EXACT_POINT_CLOUD_DISCLAIMER
                }
            }
        ]
    }

    simulation = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/create_placeholder_assets.py',
            'sourcePaths': ['placeholder://simulation'],
        },
        'defaultScenarioId': 'illustrative_fallback',
        'scenarios': [
            {
                'id': 'illustrative_fallback',
                'label': 'Illustrative fallback simulation',
                'kind': 'illustrative',
                'routingMode': 'illustrative',
                'summary': {
                    'mean_travel_time': None,
                    'mean_wait_time': None,
                    'max_queue': None,
                    'n_agents': 4,
                    'total_replans': None
                },
                'frames': [
                    {'t': 0.0, 'agents': [{'id': 'p1', 'x': 0.0, 'y': 0.0, 'z': 0.0}]},
                    {'t': 1.0, 'agents': [{'id': 'p1', 'x': 12.0, 'y': 0.0, 'z': 0.0}, {'id': 'p2', 'x': 0.0, 'y': 0.0, 'z': 0.0}]},
                    {'t': 2.0, 'agents': [{'id': 'p1', 'x': 12.0, 'y': 12.0, 'z': 6.0}, {'id': 'p2', 'x': 12.0, 'y': 0.0, 'z': 0.0}]},
                    {'t': 3.0, 'agents': [{'id': 'p1', 'x': 12.0, 'y': 24.0, 'z': 12.0}, {'id': 'p2', 'x': 12.0, 'y': 12.0, 'z': 6.0}]} 
                ],
                'timeline': {'start': 0.0, 'end': 3.0, 'step': 1.0}
            }
        ]
    }

    movement = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/create_placeholder_assets.py',
            'sourcePaths': ['placeholder://movement'],
        },
        'surfaces': [
            {
                'id': 'movement_f1',
                'level': 'F1',
                'label': 'Placeholder walkable strip',
                'source': 'asset',
                'points': [
                    {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    {'x': 6.0, 'y': 0.0, 'z': 0.0},
                    {'x': 12.0, 'y': 0.0, 'z': 0.0}
                ]
            }
        ]
    }

    write_json(output_dir / 'navigation_graph.json', graph, args.force)
    write_json(output_dir / 'semantic_anchors.json', anchors, args.force)
    write_json(output_dir / 'routes.json', routes, args.force)
    write_json(output_dir / 'simulation.json', simulation, args.force)
    write_json(output_dir / 'movement_geometry.json', movement, args.force)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
