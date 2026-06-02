from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert thesis experiment outputs into frontend demo JSON assets.',
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Repository root containing experiment/ and bim-navigation-demo/.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for demo data. Defaults to bim-navigation-demo/public/data.',
    )
    parser.add_argument(
        '--max-routes',
        type=int,
        default=5,
        help='Maximum number of example routes to export.',
    )
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Only keep every Nth simulation frame to reduce payload size.',
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def level_sort_key(level: str) -> tuple[int, str]:
    if level.startswith('F') and level[1:].isdigit():
        return (int(level[1:]), level)
    return (math.inf, level)


def relative_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def compute_bounds(nodes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    min_x = min(node['x'] for node in nodes)
    min_y = min(node['y'] for node in nodes)
    min_z = min(node['z'] for node in nodes)
    max_x = max(node['x'] for node in nodes)
    max_y = max(node['y'] for node in nodes)
    max_z = max(node['z'] for node in nodes)
    return {
        'min': {'x': min_x, 'y': min_y, 'z': min_z},
        'max': {'x': max_x, 'y': max_y, 'z': max_z},
    }


def coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            if '.' in stripped or 'e' in stripped.lower():
                return float(stripped)
            return int(stripped)
        except ValueError:
            return value
    return value


def load_graph(step3_dir: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    nodes_path = step3_dir / 'nodes_all.geojson'
    graph_summary_path = step3_dir / 'graph_summary.json'
    edge_paths = sorted(step3_dir.glob('edges_*.geojson'))

    node_features = read_json(nodes_path)['features']
    nodes: list[dict[str, Any]] = []
    node_lookup: dict[str, dict[str, Any]] = {}
    levels: set[str] = set()

    for feature in node_features:
        properties = feature.get('properties', {})
        coordinates = feature.get('geometry', {}).get('coordinates', [])
        node = {
            'id': properties['id'],
            'x': float(properties.get('x', coordinates[0])),
            'y': float(properties.get('y', coordinates[1])),
            'z': float(properties.get('z', 0.0)),
            'level': properties.get('level', 'UNKNOWN'),
            'nodeType': properties.get('node_type', 'unknown'),
            'usable': bool(properties.get('usable', True)),
            'clearance': properties.get('clearance'),
            'blindCategory': properties.get('blind_category', ''),
            'surfaceType': properties.get('surface_type', ''),
        }
        nodes.append(node)
        node_lookup[node['id']] = node
        levels.add(node['level'])

    edges: list[dict[str, Any]] = []
    source_paths = [relative_path(nodes_path, repo_root)]

    for edge_path in edge_paths:
        payload = read_json(edge_path)
        source_paths.append(relative_path(edge_path, repo_root))
        for index, feature in enumerate(payload.get('features', [])):
            properties = feature.get('properties', {})
            edges.append(
                {
                    'id': f"{edge_path.stem}_{index}",
                    'source': properties['u'],
                    'target': properties['v'],
                    'length2d': float(properties.get('length_2d', 0.0)),
                    'length3d': float(properties.get('length_3d', 0.0)),
                    'travelTime': float(properties.get('travel_time', 0.0)),
                    'edgeType': properties.get('edge_type', edge_path.stem.removeprefix('edges_')),
                    'level': properties.get('level', 'UNKNOWN'),
                }
            )

    graph_summary = read_json(graph_summary_path) if graph_summary_path.exists() else {}
    if graph_summary_path.exists():
        source_paths.append(relative_path(graph_summary_path, repo_root))

    metrics = {
        **graph_summary,
        'average_degree': round((2 * len(edges) / len(nodes)) if nodes else 0.0, 3),
        'levels': sorted(levels, key=level_sort_key),
    }

    graph = {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/prepare_thesis_demo_data.py',
            'sourcePaths': source_paths,
        },
        'nodes': nodes,
        'edges': edges,
        'levels': sorted(levels, key=level_sort_key),
        'bounds': compute_bounds(nodes),
        'metrics': metrics,
    }
    return graph, node_lookup, source_paths


def load_semantic_anchors(step4_dir: Path, repo_root: Path) -> dict[str, Any]:
    semantic_points_path = step4_dir / 'semantic_points.geojson'
    payload = read_json(semantic_points_path)
    anchors: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for feature in payload.get('features', []):
        properties = feature.get('properties', {})
        coordinates = feature.get('geometry', {}).get('coordinates', [])
        anchor = {
            'id': properties['id'],
            'type': properties.get('type', 'UNKNOWN'),
            'level': properties.get('level', 'UNKNOWN'),
            'x': float(coordinates[0]),
            'y': float(coordinates[1]),
            'z': float(properties.get('z', 0.0)),
        }
        anchors.append(anchor)
        counts[anchor['type']] += 1

    return {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/prepare_thesis_demo_data.py',
            'sourcePaths': [relative_path(semantic_points_path, repo_root)],
        },
        'anchors': anchors,
        'counts': dict(sorted(counts.items())),
    }


def extract_existing_path_ids(record: dict[str, Any]) -> list[str] | None:
    candidate_keys = ('node_ids', 'path_node_ids', 'path_nodes', 'node_sequence', 'nodes')
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return value
    return None


def build_adjacency(graph: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in graph['edges']:
        adjacency[edge['source']].append((edge['target'], float(edge['travelTime'])))
        adjacency[edge['target']].append((edge['source'], float(edge['travelTime'])))
    return adjacency


def shortest_path(
    adjacency: dict[str, list[tuple[str, float]]],
    start: str,
    goal: str,
) -> list[str] | None:
    import heapq

    queue: list[tuple[float, str]] = [(0.0, start)]
    distances: dict[str, float] = {start: 0.0}
    previous: dict[str, str] = {}
    visited: set[str] = set()

    while queue:
        current_cost, node_id = heapq.heappop(queue)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id == goal:
            break

        for neighbor_id, weight in adjacency.get(node_id, []):
            tentative = current_cost + weight
            if tentative < distances.get(neighbor_id, math.inf):
                distances[neighbor_id] = tentative
                previous[neighbor_id] = node_id
                heapq.heappush(queue, (tentative, neighbor_id))

    if goal not in distances:
        return None

    path = [goal]
    cursor = goal
    while cursor != start:
        cursor = previous[cursor]
        path.append(cursor)
    path.reverse()
    return path


def summarize_route(
    node_ids: list[str],
    node_lookup: dict[str, dict[str, Any]],
    edge_lookup: dict[frozenset[str], dict[str, Any]],
) -> dict[str, Any]:
    total_length_2d = 0.0
    total_length_3d = 0.0
    total_travel_time = 0.0
    edge_types: dict[str, int] = defaultdict(int)
    levels_visited: set[str] = set()

    for left, right in zip(node_ids, node_ids[1:]):
        edge = edge_lookup.get(frozenset((left, right)))
        if not edge:
            continue
        total_length_2d += float(edge['length2d'])
        total_length_3d += float(edge['length3d'])
        total_travel_time += float(edge['travelTime'])
        edge_types[edge['edgeType']] += 1
        levels_visited.add(edge['level'])

    polyline = [
        {
            'x': float(node_lookup[node_id]['x']),
            'y': float(node_lookup[node_id]['y']),
            'z': float(node_lookup[node_id]['z']),
        }
        for node_id in node_ids
        if node_id in node_lookup
    ]

    return {
        'snappedNodeIds': node_ids,
        'polyline': polyline,
        'metrics': {
            'totalTravelTime': total_travel_time,
            'totalLength2d': total_length_2d,
            'totalLength3d': total_length_3d,
            'edgeTypes': dict(sorted(edge_types.items())),
            'levelsVisited': sorted(levels_visited, key=level_sort_key),
        },
    }


def load_routes(
    step4_dir: Path,
    graph: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
    repo_root: Path,
    max_routes: int,
) -> dict[str, Any]:
    example_paths_path = step4_dir / 'example_paths.json'
    payload = read_json(example_paths_path) if example_paths_path.exists() else []
    adjacency = build_adjacency(graph)
    edge_lookup = {
        frozenset((edge['source'], edge['target'])): edge
        for edge in graph['edges']
    }

    routes: list[dict[str, Any]] = []
    for index, record in enumerate(payload[:max_routes], start=1):
        origin = record.get('origin')
        destination = record.get('dest') or record.get('destination')
        if not origin or not destination:
            continue
        if origin not in node_lookup or destination not in node_lookup:
            continue

        provided_path = extract_existing_path_ids(record)
        if provided_path and all(node_id in node_lookup for node_id in provided_path):
            node_ids = provided_path
            source_kind = 'provided'
        else:
            node_ids = shortest_path(adjacency, origin, destination)
            source_kind = 'derived-dijkstra'
        if not node_ids:
            continue

        route_summary = summarize_route(node_ids, node_lookup, edge_lookup)
        metrics = {
            key: value
            for key, value in record.items()
            if key not in {'origin', 'dest', 'destination', 'node_ids', 'path_node_ids', 'path_nodes', 'node_sequence', 'nodes'}
        }
        metrics = {key: coerce_number(value) for key, value in metrics.items()}
        metrics.update(route_summary['metrics'])

        routes.append(
            {
                'id': f'route_{index:02d}',
                'label': f"{origin} to {destination}",
                'originNodeId': origin,
                'destinationNodeId': destination,
                'sourceKind': source_kind,
                'snappedNodeIds': route_summary['snappedNodeIds'],
                'polyline': route_summary['polyline'],
                'metrics': metrics,
            }
        )

    if not routes and graph['nodes']:
        first = graph['nodes'][0]['id']
        last = graph['nodes'][-1]['id']
        node_ids = shortest_path(adjacency, first, last) or [first, last]
        route_summary = summarize_route(node_ids, node_lookup, edge_lookup)
        routes.append(
            {
                'id': 'route_01',
                'label': f'{first} to {last}',
                'originNodeId': first,
                'destinationNodeId': last,
                'sourceKind': 'derived-dijkstra',
                'snappedNodeIds': route_summary['snappedNodeIds'],
                'polyline': route_summary['polyline'],
                'metrics': route_summary['metrics'],
            }
        )

    return {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/prepare_thesis_demo_data.py',
            'sourcePaths': [relative_path(example_paths_path, repo_root)] if example_paths_path.exists() else [],
        },
        'routes': routes,
        'defaultRouteId': routes[0]['id'] if routes else None,
    }


def read_summary(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            return {}
        return {key: coerce_number(value) for key, value in row.items()}


def read_traj_frames(path: Path, frame_step: int) -> list[dict[str, Any]]:
    frames_by_t: dict[float, list[dict[str, Any]]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            t = float(record['t'])
            frames_by_t[t].append(
                {
                    'id': record['agent_id'],
                    'x': float(record['x']),
                    'y': float(record['y']),
                    'z': float(record['z']),
                }
            )

    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(sorted(frames_by_t)):
        if index % max(1, frame_step) != 0:
            continue
        frames.append({'t': timestamp, 'agents': frames_by_t[timestamp]})
    return frames


def fallback_simulation(routes: dict[str, Any]) -> dict[str, Any]:
    polyline = routes['routes'][0]['polyline'] if routes['routes'] else []
    frames: list[dict[str, Any]] = []
    frame_count = 24
    for frame_index in range(frame_count):
        ratio = frame_index / max(1, frame_count - 1)
        agents: list[dict[str, Any]] = []
        for agent_index in range(6):
            shifted = max(0.0, min(1.0, ratio - agent_index * 0.08))
            point_index = min(len(polyline) - 1, round(shifted * max(0, len(polyline) - 1))) if polyline else 0
            point = polyline[point_index] if polyline else {'x': 0.0, 'y': 0.0, 'z': 0.0}
            agents.append(
                {
                    'id': f'illustrative_{agent_index + 1}',
                    'x': point['x'],
                    'y': point['y'],
                    'z': point['z'],
                }
            )
        frames.append({'t': float(frame_index), 'agents': agents})

    return {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/prepare_thesis_demo_data.py',
            'sourcePaths': [],
        },
        'scenarios': [
            {
                'id': 'illustrative_fallback',
                'label': 'Illustrative fallback simulation',
                'kind': 'illustrative',
                'routingMode': 'illustrative',
                'summary': {
                    'label': 'illustrative',
                    'mean_travel_time': None,
                    'mean_wait_time': None,
                    'max_queue': None,
                    'n_agents': 6,
                    'total_replans': None,
                },
                'frames': frames,
                'timeline': {
                    'start': frames[0]['t'],
                    'end': frames[-1]['t'],
                    'step': 1.0,
                },
            }
        ],
        'defaultScenarioId': 'illustrative_fallback',
    }


def load_simulation(
    step5_dir: Path,
    repo_root: Path,
    frame_step: int,
    routes: dict[str, Any],
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for scenario_dir in (step5_dir / 'dynamic', step5_dir / 'static'):
        summary_path = scenario_dir / 'summary.csv'
        traj_path = scenario_dir / 'traj_agents.jsonl'
        if not summary_path.exists() or not traj_path.exists():
            continue

        summary = read_summary(summary_path)
        frames = read_traj_frames(traj_path, frame_step)
        if not frames:
            continue

        source_paths.extend(
            [relative_path(summary_path, repo_root), relative_path(traj_path, repo_root)]
        )
        step = frames[1]['t'] - frames[0]['t'] if len(frames) > 1 else 1.0
        scenario_id = str(summary.get('label') or scenario_dir.name)
        scenarios.append(
            {
                'id': scenario_id,
                'label': scenario_id.replace('_', ' ').title(),
                'kind': 'loaded',
                'routingMode': str(summary.get('routing_mode', scenario_dir.name)),
                'summary': summary,
                'frames': frames,
                'timeline': {
                    'start': frames[0]['t'],
                    'end': frames[-1]['t'],
                    'step': step,
                },
            }
        )

    if not scenarios:
        return fallback_simulation(routes)

    default_scenario = next(
        (scenario['id'] for scenario in scenarios if str(scenario['routingMode']).lower() == 'dynamic'),
        scenarios[0]['id'],
    )
    return {
        'meta': {
            'generatedAt': now_iso(),
            'generator': 'scripts/prepare_thesis_demo_data.py',
            'sourcePaths': source_paths,
        },
        'scenarios': scenarios,
        'defaultScenarioId': default_scenario,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or (repo_root / 'bim-navigation-demo' / 'public' / 'data')).resolve()

    step3_dir = repo_root / 'experiment' / 'outputs' / 'step3_graph'
    step4_dir = repo_root / 'experiment' / 'outputs' / 'step4_routing'
    step5_dir = repo_root / 'experiment' / 'outputs' / 'step5_simulation'

    graph, node_lookup, _ = load_graph(step3_dir, repo_root)
    anchors = load_semantic_anchors(step4_dir, repo_root)
    routes = load_routes(step4_dir, graph, node_lookup, repo_root, args.max_routes)
    simulation = load_simulation(step5_dir, repo_root, args.frame_step, routes)

    write_json(output_dir / 'navigation_graph.json', graph, indent=None)
    write_json(output_dir / 'semantic_anchors.json', anchors, indent=2)
    write_json(output_dir / 'routes.json', routes, indent=2)
    write_json(output_dir / 'simulation.json', simulation, indent=2)

    print(f'Wrote navigation graph to {output_dir / "navigation_graph.json"}')
    print(f'Wrote semantic anchors to {output_dir / "semantic_anchors.json"}')
    print(f'Wrote routes to {output_dir / "routes.json"}')
    print(f'Wrote simulation to {output_dir / "simulation.json"}')
    print(f'Graph nodes: {len(graph["nodes"])} | edges: {len(graph["edges"])}')
    print(f'Routes: {len(routes["routes"])} | simulation scenarios: {len(simulation["scenarios"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
