from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXACT_POINT_CLOUD_DISCLAIMER = (
    'This point cloud is a BIM-derived, surface-sampled visualization generated from IFC/mesh geometry. '
    'It is not measured LiDAR, photogrammetry, or any other sensor-captured reality data.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate config and public assets for the BIM navigation demo.',
    )
    parser.add_argument(
        '--demo-root',
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help='Path to the bim-navigation-demo directory.',
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def resolve_public_path(demo_root: Path, asset_path: str) -> Path:
    return demo_root / 'public' / Path(*asset_path.lstrip('/').split('/'))


def main() -> int:
    args = parse_args()
    demo_root = args.demo_root.resolve()
    config_path = demo_root / 'public' / 'config' / 'demo-config.json'
    errors: list[str] = []
    warnings: list[str] = []

    if not config_path.exists():
        print(f'ERROR: missing config file {config_path}')
        return 1

    config = read_json(config_path)
    if config.get('app', {}).get('pointCloudDisclaimer') != EXACT_POINT_CLOUD_DISCLAIMER:
        errors.append('Point cloud disclaimer does not match the required wording.')

    required_json = {
        'navigation_graph.json': ['nodes', 'edges'],
        'semantic_anchors.json': ['anchors'],
        'routes.json': ['routes'],
    }
    optional_json = {
        'simulation.json': ['scenarios'],
        'movement_geometry.json': ['surfaces'],
    }

    data_dir = demo_root / 'public' / 'data'
    for file_name, keys in required_json.items():
        path = data_dir / file_name
        if not path.exists():
            errors.append(f'Missing required data file: {path}')
            continue
        payload = read_json(path)
        missing_keys = [key for key in keys if key not in payload]
        if missing_keys:
            errors.append(f'{path} is missing keys: {", ".join(missing_keys)}')

    for file_name, keys in optional_json.items():
        path = data_dir / file_name
        if not path.exists():
            warnings.append(f'Optional file missing: {path}')
            continue
        payload = read_json(path)
        missing_keys = [key for key in keys if key not in payload]
        if missing_keys:
            warnings.append(f'{path} is missing keys: {", ".join(missing_keys)}')

    asset_checks = [
        ('GLB model', config['assets']['bimModel']['path'], True),
        ('PLY point cloud', config['assets']['pointCloud']['path'], True),
        ('MediaPipe model asset', config['assets']['mediapipe']['modelAssetPath'], True),
    ]

    for label, asset_path, optional in asset_checks:
        path = resolve_public_path(demo_root, asset_path)
        if not path.exists():
            message = f'{label} not found at {path}'
            if optional:
                warnings.append(message)
            else:
                errors.append(message)

    mediapipe_wasm_dir = resolve_public_path(demo_root, config['assets']['mediapipe']['wasmPath'])
    if not mediapipe_wasm_dir.exists() or not any(mediapipe_wasm_dir.iterdir()):
        warnings.append(f'MediaPipe wasm directory is empty: {mediapipe_wasm_dir}')

    if errors:
        print('Validation failed with errors:')
        for error in errors:
            print(f'  - {error}')
    else:
        print('Validation passed without hard errors.')

    if warnings:
        print('Warnings:')
        for warning in warnings:
            print(f'  - {warning}')

    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
