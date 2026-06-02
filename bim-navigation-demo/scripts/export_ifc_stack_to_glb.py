from __future__ import annotations

import argparse
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom as geom


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Export one or more IFC files into a single GLB for the BIM navigation demo.',
  )
  parser.add_argument(
    'inputs',
    nargs='+',
    help='Input IFC files to append into the output GLB in the given order.',
  )
  parser.add_argument(
    '--output',
    required=True,
    help='Target GLB path.',
  )
  return parser.parse_args()


def build_geometry_settings() -> geom.settings:
  settings = geom.settings()
  settings.set('use-world-coords', True)
  settings.set('apply-default-materials', True)
  return settings


def build_serializer_settings() -> geom.serializer_settings:
  settings = geom.serializer_settings()
  settings.set('use-element-guids', True)
  settings.set('use-element-types', True)
  settings.set('y-up', True)
  return settings


def export_ifc_stack(input_paths: list[Path], output_path: Path) -> int:
  geometry_settings = build_geometry_settings()
  serializer_settings = build_serializer_settings()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  serializer = geom.serializers.gltf(str(output_path), geometry_settings, serializer_settings)
  serializer.setUnitNameAndMagnitude('METER', 1.0)
  serializer.writeHeader()

  total_shapes = 0
  for input_path in input_paths:
    model = ifcopenshell.open(str(input_path))
    file_shapes = 0
    for shape in geom.iterate(geometry_settings, model):
      serializer.write(shape)
      file_shapes += 1

    total_shapes += file_shapes
    print(f'Appended {file_shapes} products from {input_path}')

  serializer.finalize()
  print(f'Wrote GLB to {output_path} with {total_shapes} products from {len(input_paths)} IFC files.')
  return total_shapes


def main() -> int:
  args = parse_args()
  input_paths = [Path(item).resolve() for item in args.inputs]
  output_path = Path(args.output).resolve()

  missing = [path for path in input_paths if not path.exists()]
  if missing:
    for path in missing:
      print(f'Missing IFC input: {path}')
    return 1

  export_ifc_stack(input_paths, output_path)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())