from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sample a mesh surface and write an ASCII PLY point cloud.',
    )
    parser.add_argument('input_mesh', type=Path, help='Input mesh path such as GLB, GLTF, OBJ, or PLY.')
    parser.add_argument('output_ply', type=Path, help='Output ASCII PLY path.')
    parser.add_argument('--samples', type=int, default=50000, help='Number of surface samples to generate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic sampling.')
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force='scene')
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not meshes:
            raise ValueError(f'No mesh geometry found in {path}.')
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f'Unsupported mesh type loaded from {path}: {type(loaded)!r}')


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii', newline='\n') as handle:
        handle.write('ply\n')
        handle.write('format ascii 1.0\n')
        handle.write(f'element vertex {len(points)}\n')
        handle.write('property float x\n')
        handle.write('property float y\n')
        handle.write('property float z\n')
        handle.write('end_header\n')
        for x_value, y_value, z_value in points:
            handle.write(f'{x_value:.6f} {y_value:.6f} {z_value:.6f}\n')


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    mesh = load_mesh(args.input_mesh)
    points, _ = trimesh.sample.sample_surface(mesh, args.samples)
    write_ascii_ply(args.output_ply, points)
    print(f'Sampled {len(points)} points from {args.input_mesh} -> {args.output_ply}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
