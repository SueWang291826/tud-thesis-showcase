"""
Storey Mapping Module.

Determines cross-file storey correspondences:
- Which storeys dominate each IFC file
- Maps file labels to functional roles (platform, equipment, concourse)
- Identifies overlaps and inconsistencies
- Produces reproducible mapping artifacts

Outputs: storey_mapping.json, storey_mapping.md, statistics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import save_json

logger = logging.getLogger(__name__)


def generate_storey_mapping(
    audits: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a complete storey mapping from audit results.

    Determines which IFC file predominantly contains which storey,
    and maps each storey to its functional role for navigation.

    Args:
        audits: Per-file audit results.
        config: Pipeline configuration (contains storey reference info).
        output_dir: Directory for storey mapping outputs.

    Returns:
        Storey mapping dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating storey mapping")

    storey_ref = config.get("storeys", {})
    file_labels = list(audits.keys())

    # ---- Build storey-file matrix ----
    # For each storey, count elements per file
    storey_names = set()
    for audit in audits.values():
        for s in audit["storeys"]:
            if s["storey_name"] != "UNASSIGNED":
                storey_names.add(s["storey_name"])

    storey_file_matrix = {}
    for sname in sorted(storey_names):
        storey_file_matrix[sname] = {}
        for label in file_labels:
            count = 0
            for s in audits[label]["storeys"]:
                if s["storey_name"] == sname:
                    count = s["element_count"]
                    break
            storey_file_matrix[sname][label] = count

    # ---- Determine dominant file for each storey ----
    storey_dominant_file = {}
    for sname, file_counts in storey_file_matrix.items():
        total = sum(file_counts.values())
        if total > 0:
            dominant = max(file_counts, key=file_counts.get)
            dominant_count = file_counts[dominant]
            storey_dominant_file[sname] = {
                "dominant_file": dominant,
                "dominant_count": dominant_count,
                "total_count": total,
                "dominance_ratio": round(dominant_count / total, 4) if total else 0,
                "distribution": file_counts,
            }
        else:
            storey_dominant_file[sname] = {
                "dominant_file": None,
                "dominant_count": 0,
                "total_count": 0,
                "dominance_ratio": 0,
                "distribution": file_counts,
            }

    # ---- Determine dominant storey for each file ----
    file_dominant_storey = {}
    for label in file_labels:
        storey_counts = {}
        for sname, file_counts in storey_file_matrix.items():
            storey_counts[sname] = file_counts.get(label, 0)
        total = sum(storey_counts.values())
        if total > 0:
            dominant = max(storey_counts, key=storey_counts.get)
            file_dominant_storey[label] = {
                "dominant_storey": dominant,
                "dominant_count": storey_counts[dominant],
                "total_elements": total,
                "dominance_ratio": round(storey_counts[dominant] / total, 4),
                "storey_distribution": storey_counts,
            }
        else:
            file_dominant_storey[label] = {
                "dominant_storey": None,
                "dominant_count": 0,
                "total_elements": 0,
                "dominance_ratio": 0,
                "storey_distribution": storey_counts,
            }

    # ---- Map to functional roles ----
    # Use config reference and dominant storey analysis
    functional_mapping = {}
    for sname in sorted(storey_names):
        # Try to match to reference storeys
        role = "unknown"
        is_public = False
        elevation = None

        for floor_id, ref in storey_ref.items():
            ref_cn = ref.get("name_cn", "")
            ref_en = ref.get("name_en", "")
            # Match by Chinese name contained in storey name
            if ref_cn and ref_cn in sname:
                role = ref.get("role", "unknown")
                is_public = ref.get("is_public", False)
                elevation = ref.get("elevation_mm")
                break

        functional_mapping[sname] = {
            "role": role,
            "is_public": is_public,
            "elevation_mm": elevation,
            "is_walkable_level": role in ("platform", "concourse"),
            "dominant_file": storey_dominant_file[sname]["dominant_file"],
        }

    # ---- Identify cross-file overlaps ----
    overlaps = []
    for sname, info in storey_dominant_file.items():
        dist = info["distribution"]
        non_zero = {k: v for k, v in dist.items() if v > 0}
        if len(non_zero) > 1:
            overlaps.append({
                "storey": sname,
                "files_with_elements": non_zero,
                "dominant_file": info["dominant_file"],
                "dominance_ratio": info["dominance_ratio"],
            })

    # ---- Assemble full mapping ----
    mapping = {
        "storey_file_matrix": storey_file_matrix,
        "storey_dominant_file": storey_dominant_file,
        "file_dominant_storey": file_dominant_storey,
        "functional_mapping": functional_mapping,
        "cross_file_overlaps": overlaps,
        "walkable_levels": [
            sname for sname, fm in functional_mapping.items() if fm["is_walkable_level"]
        ],
        "public_levels": [
            sname for sname, fm in functional_mapping.items() if fm["is_public"]
        ],
        "assumptions": [
            "Storey assignment is based on IfcRelContainedInSpatialStructure relationships.",
            "Functional role mapping uses Chinese storey names matched to config reference.",
            "Platform (站台层) and Concourse (站厅层) are the primary walkable public levels.",
            "Equipment level (设备层) is NOT a public walkable level.",
            "Cross-file overlaps indicate shared elements (e.g., vertical connectors spanning multiple levels).",
        ],
    }

    # ---- Save outputs ----
    save_json(mapping, output_dir / "storey_mapping.json")
    _write_storey_mapping_markdown(mapping, config, output_dir / "storey_mapping.md")

    # Save storey-file matrix as CSV
    matrix_rows = []
    for sname, file_counts in storey_file_matrix.items():
        row = {"storey": sname}
        row.update(file_counts)
        row["total"] = sum(file_counts.values())
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(output_dir / "storey_file_matrix.csv", index=False, encoding="utf-8-sig")

    logger.info(f"  Storey mapping saved to {output_dir}")
    logger.info(f"  Walkable levels identified: {mapping['walkable_levels']}")
    logger.info(f"  Cross-file overlaps found: {len(overlaps)}")

    return mapping


def _write_storey_mapping_markdown(
    mapping: Dict[str, Any],
    config: Dict[str, Any],
    path: Path,
) -> None:
    """Write storey mapping results as a Markdown report."""
    lines = []
    lines.append("# Storey Mapping Report")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("This mapping is derived by counting IFC product elements assigned to each")
    lines.append("IfcBuildingStorey across the three IFC files. The dominant file for a storey")
    lines.append("is the one containing the most elements for that storey.")
    lines.append("")

    lines.append("## Assumptions")
    for a in mapping["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")

    # File → dominant storey
    lines.append("## File → Dominant Storey")
    lines.append("| File | Dominant Storey | Elements | Ratio |")
    lines.append("|------|----------------|----------|-------|")
    for label, info in mapping["file_dominant_storey"].items():
        lines.append(
            f"| {label} | {info['dominant_storey']} | "
            f"{info['dominant_count']:,} | {info['dominance_ratio']:.1%} |"
        )
    lines.append("")

    # Storey → dominant file
    lines.append("## Storey → Dominant File")
    lines.append("| Storey | Dominant File | Elements | Total | Ratio |")
    lines.append("|--------|-------------|----------|-------|-------|")
    for sname, info in mapping["storey_dominant_file"].items():
        lines.append(
            f"| {sname} | {info['dominant_file']} | "
            f"{info['dominant_count']:,} | {info['total_count']:,} | "
            f"{info['dominance_ratio']:.1%} |"
        )
    lines.append("")

    # Functional roles
    lines.append("## Functional Role Mapping")
    lines.append("| Storey | Role | Public | Walkable | Elevation (mm) |")
    lines.append("|--------|------|--------|----------|---------------|")
    for sname, fm in mapping["functional_mapping"].items():
        elev = f"{fm['elevation_mm']:.0f}" if fm["elevation_mm"] is not None else "?"
        lines.append(
            f"| {sname} | {fm['role']} | "
            f"{'Yes' if fm['is_public'] else 'No'} | "
            f"{'Yes' if fm['is_walkable_level'] else 'No'} | {elev} |"
        )
    lines.append("")

    # Cross-file overlaps
    lines.append("## Cross-File Overlaps")
    if mapping["cross_file_overlaps"]:
        for ov in mapping["cross_file_overlaps"]:
            lines.append(f"- **{ov['storey']}**: present in {ov['files_with_elements']}, "
                         f"dominant in `{ov['dominant_file']}` ({ov['dominance_ratio']:.0%})")
    else:
        lines.append("- No cross-file overlaps detected.")
    lines.append("")

    lines.append("## Key Conclusions")
    lines.append(f"- **Walkable public levels**: {', '.join(mapping['walkable_levels'])}")
    lines.append(f"- **Public levels**: {', '.join(mapping['public_levels'])}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
