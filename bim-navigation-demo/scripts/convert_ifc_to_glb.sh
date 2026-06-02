#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 INPUT.ifc OUTPUT.glb" >&2
  exit 1
fi

if ! command -v IfcConvert >/dev/null 2>&1; then
  echo "IfcConvert was not found on PATH. Install IfcOpenShell before running this script." >&2
  exit 1
fi

input_ifc="$1"
output_glb="$2"

if [ ! -f "$input_ifc" ]; then
  echo "Input IFC not found: $input_ifc" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_glb")"
IfcConvert "$input_ifc" "$output_glb"
echo "Wrote GLB asset to $output_glb"
