"""
IFC Audit Module.

Performs comprehensive auditing of each IFC file:
- Schema and project hierarchy
- Entity counts by IFC class
- Entity counts by storey
- Proxy statistics
- Navigation-relevant element statistics (doors, stairs, slabs, walls, etc.)
- Bounding box estimation
- Anomaly detection

Outputs: JSON summaries, CSV tables, Markdown reports.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ifc_loader import IFCFileInfo, get_all_products, get_element_properties, get_storey_for_element
from .utils import save_json, save_dataframe, safe_ifc_str

logger = logging.getLogger(__name__)

# IFC classes relevant for indoor navigation
NAVIGATION_RELEVANT_CLASSES = [
    "IfcSlab", "IfcWall", "IfcWallStandardCase", "IfcColumn",
    "IfcDoor", "IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight",
    "IfcRailing", "IfcCurtainWall", "IfcPlate", "IfcBeam",
    "IfcCovering", "IfcBuildingElementProxy", "IfcMember",
    "IfcFooting", "IfcRoof", "IfcWindow",
]


def audit_single_file(file_info: IFCFileInfo, output_dir: Path) -> Dict[str, Any]:
    """Perform a comprehensive audit of a single IFC file.

    Args:
        file_info: Loaded IFC file information.
        output_dir: Directory for audit outputs.

    Returns:
        Audit summary dictionary.
    """
    label = file_info.label
    model = file_info.model
    logger.info(f"Auditing IFC file: {label} ({file_info.filepath.name})")

    file_output_dir = output_dir / label
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Build storey GUID -> decoded name lookup
    storey_lookup = {s.guid: s.decoded_name for s in file_info.storeys}

    # ---- 1. Entity counts by IFC class ----
    class_counts = Counter()
    all_entities = list(model)
    for entity in all_entities:
        class_counts[entity.is_a()] += 1

    class_counts_sorted = dict(
        sorted(class_counts.items(), key=lambda x: -x[1])
    )

    # ---- 2. Get all products and their properties ----
    products = get_all_products(model)
    logger.info(f"  Found {len(products)} product entities (excluding spatial structure)")

    product_records = []
    storey_counts = Counter()
    storey_class_counts = defaultdict(Counter)
    unassigned_count = 0

    for elem in products:
        props = get_element_properties(elem)
        storey_guid = get_storey_for_element(elem, model)
        storey_name = storey_lookup.get(storey_guid, "UNASSIGNED") if storey_guid else "UNASSIGNED"

        if storey_name == "UNASSIGNED":
            unassigned_count += 1

        props["storey_guid"] = storey_guid or ""
        props["storey_name"] = storey_name
        product_records.append(props)

        storey_counts[storey_name] += 1
        storey_class_counts[storey_name][props["ifc_class"]] += 1

    products_df = pd.DataFrame(product_records)

    # ---- 3. Navigation-relevant statistics ----
    nav_stats = {}
    for cls in NAVIGATION_RELEVANT_CLASSES:
        count = class_counts.get(cls, 0)
        nav_stats[cls] = count

    # ---- 4. Proxy statistics ----
    proxy_count = class_counts.get("IfcBuildingElementProxy", 0)
    proxy_fraction = proxy_count / len(products) if products else 0.0

    proxy_names = Counter()
    proxy_types = Counter()
    if proxy_count > 0:
        proxy_df = products_df[products_df["ifc_class"] == "IfcBuildingElementProxy"]
        proxy_names = Counter(proxy_df["name"].values)
        proxy_types = Counter(proxy_df["object_type"].values)

    # ---- 5. Storey summary ----
    storey_summary = []
    for s in file_info.storeys:
        count = storey_counts.get(s.decoded_name, 0)
        storey_summary.append({
            "storey_name": s.decoded_name,
            "guid": s.guid,
            "elevation_mm": s.elevation_mm,
            "element_count": count,
            "fraction": count / len(products) if products else 0.0,
        })
    # Add unassigned
    storey_summary.append({
        "storey_name": "UNASSIGNED",
        "guid": "",
        "elevation_mm": None,
        "element_count": unassigned_count,
        "fraction": unassigned_count / len(products) if products else 0.0,
    })

    # ---- 6. Anomaly detection ----
    anomalies = []

    if unassigned_count > 0:
        anomalies.append(
            f"{unassigned_count} elements ({unassigned_count/len(products)*100:.1f}%) "
            f"have no storey assignment"
        )

    if proxy_fraction > 0.3:
        anomalies.append(
            f"High proxy fraction: {proxy_count} proxies "
            f"({proxy_fraction*100:.1f}% of products)"
        )

    # Check for empty storeys in this file
    for s in storey_summary:
        if s["storey_name"] != "UNASSIGNED" and s["element_count"] == 0:
            anomalies.append(f"Storey '{s['storey_name']}' has 0 elements in this file")

    # Check for missing navigation elements
    if nav_stats.get("IfcSlab", 0) == 0:
        anomalies.append("No IfcSlab found - floor detection may be difficult")
    if nav_stats.get("IfcDoor", 0) == 0:
        anomalies.append("No IfcDoor found")
    if nav_stats.get("IfcStair", 0) == 0 and nav_stats.get("IfcStairFlight", 0) == 0:
        anomalies.append("No stairs found - vertical connections may be in proxies")

    # ---- Build audit summary ----
    audit = {
        "file_label": label,
        "filename": file_info.filepath.name,
        "schema": file_info.schema,
        "project_name": file_info.project_name,
        "project_guid": file_info.project_guid,
        "building_name": file_info.building_name,
        "building_guid": file_info.building_guid,
        "length_unit": file_info.length_unit,
        "length_prefix": file_info.length_prefix,
        "total_entities": file_info.entity_count,
        "total_products": len(products),
        "storey_count": len(file_info.storeys),
        "storeys": storey_summary,
        "class_counts": class_counts_sorted,
        "nav_relevant_stats": nav_stats,
        "proxy_count": proxy_count,
        "proxy_fraction": round(proxy_fraction, 4),
        "proxy_top_names": dict(proxy_names.most_common(30)),
        "proxy_top_types": dict(proxy_types.most_common(30)),
        "unassigned_elements": unassigned_count,
        "anomalies": anomalies,
        "storey_class_detail": {
            k: dict(v.most_common()) for k, v in storey_class_counts.items()
        },
    }

    # ---- Save outputs ----
    # JSON audit
    save_json(audit, file_output_dir / f"audit_{label}.json")
    logger.info(f"  Saved audit JSON: audit_{label}.json")

    # CSV - full product inventory
    save_dataframe(products_df, file_output_dir / f"products_{label}.csv")
    logger.info(f"  Saved product inventory: products_{label}.csv ({len(products_df)} rows)")

    # CSV - class counts
    class_df = pd.DataFrame(
        list(class_counts_sorted.items()),
        columns=["ifc_class", "count"],
    )
    save_dataframe(class_df, file_output_dir / f"class_counts_{label}.csv")

    # CSV - storey summary
    storey_df = pd.DataFrame(storey_summary)
    save_dataframe(storey_df, file_output_dir / f"storey_summary_{label}.csv")

    # Markdown audit report
    _write_audit_markdown(audit, file_output_dir / f"audit_{label}.md")
    logger.info(f"  Saved audit report: audit_{label}.md")

    return audit


def _write_audit_markdown(audit: Dict[str, Any], path: Path) -> None:
    """Write a human-readable Markdown audit report."""
    lines = []
    lines.append(f"# IFC Audit Report: {audit['file_label']}")
    lines.append(f"")
    lines.append(f"**File**: {audit['filename']}  ")
    lines.append(f"**Schema**: {audit['schema']}  ")
    lines.append(f"**Project**: {audit['project_name']} (`{audit['project_guid']}`)  ")
    lines.append(f"**Building**: {audit['building_name']} (`{audit['building_guid']}`)  ")
    lines.append(f"**Length Unit**: {audit['length_prefix']} {audit['length_unit']}  ")
    lines.append(f"**Total Entities**: {audit['total_entities']:,}  ")
    lines.append(f"**Total Products**: {audit['total_products']:,}  ")
    lines.append(f"")

    # Anomalies
    lines.append(f"## Anomalies")
    if audit["anomalies"]:
        for a in audit["anomalies"]:
            lines.append(f"- ⚠️ {a}")
    else:
        lines.append(f"- No anomalies detected.")
    lines.append(f"")

    # Storey summary
    lines.append(f"## Storey Summary")
    lines.append(f"| Storey | Elevation (mm) | Elements | Fraction |")
    lines.append(f"|--------|---------------|----------|----------|")
    for s in audit["storeys"]:
        elev = f"{s['elevation_mm']:.0f}" if s["elevation_mm"] is not None else "N/A"
        lines.append(
            f"| {s['storey_name']} | {elev} | {s['element_count']:,} | "
            f"{s['fraction']:.1%} |"
        )
    lines.append(f"")

    # Navigation-relevant statistics
    lines.append(f"## Navigation-Relevant Elements")
    lines.append(f"| IFC Class | Count |")
    lines.append(f"|-----------|-------|")
    for cls, count in sorted(audit["nav_relevant_stats"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cls} | {count:,} |")
    lines.append(f"")

    # Proxy info
    lines.append(f"## Proxy Analysis Summary")
    lines.append(f"- **Proxy Count**: {audit['proxy_count']:,}")
    lines.append(f"- **Proxy Fraction**: {audit['proxy_fraction']:.1%}")
    lines.append(f"")
    if audit["proxy_top_names"]:
        lines.append(f"### Top Proxy Names")
        lines.append(f"| Name | Count |")
        lines.append(f"|------|-------|")
        for name, count in sorted(
            audit["proxy_top_names"].items(), key=lambda x: -x[1]
        )[:20]:
            lines.append(f"| {name} | {count:,} |")
    lines.append(f"")

    # Top IFC classes
    lines.append(f"## Entity Counts by IFC Class (Top 30)")
    lines.append(f"| IFC Class | Count |")
    lines.append(f"|-----------|-------|")
    for cls, count in list(audit["class_counts"].items())[:30]:
        lines.append(f"| {cls} | {count:,} |")
    lines.append(f"")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_cross_file_audit(
    audits: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a combined cross-file audit summary.

    Args:
        audits: Dictionary of label -> audit results.
        output_dir: Directory for output.

    Returns:
        Cross-file summary dictionary.
    """
    logger.info("Generating cross-file audit summary")

    # Collect all IFC classes across files
    all_classes = set()
    for audit in audits.values():
        all_classes.update(audit["class_counts"].keys())

    # Build comparison table
    comparison_rows = []
    for cls in sorted(all_classes):
        row = {"ifc_class": cls}
        for label, audit in audits.items():
            row[f"count_{label}"] = audit["class_counts"].get(cls, 0)
        row["total"] = sum(
            audit["class_counts"].get(cls, 0) for audit in audits.values()
        )
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("total", ascending=False)
    save_dataframe(comparison_df, output_dir / "cross_file_class_comparison.csv")

    # Storey comparison across files
    storey_comparison = []
    all_storey_names = set()
    for audit in audits.values():
        for s in audit["storeys"]:
            all_storey_names.add(s["storey_name"])

    for sname in sorted(all_storey_names):
        row = {"storey_name": sname}
        for label, audit in audits.items():
            count = 0
            for s in audit["storeys"]:
                if s["storey_name"] == sname:
                    count = s["element_count"]
                    break
            row[f"count_{label}"] = count
        row["total"] = sum(v for k, v in row.items() if k.startswith("count_"))
        storey_comparison.append(row)

    storey_comp_df = pd.DataFrame(storey_comparison)
    save_dataframe(storey_comp_df, output_dir / "cross_file_storey_comparison.csv")

    # Cross-file summary
    summary = {
        "files_audited": list(audits.keys()),
        "total_products": {
            label: audit["total_products"] for label, audit in audits.items()
        },
        "total_entities": {
            label: audit["total_entities"] for label, audit in audits.items()
        },
        "proxy_counts": {
            label: audit["proxy_count"] for label, audit in audits.items()
        },
        "proxy_fractions": {
            label: audit["proxy_fraction"] for label, audit in audits.items()
        },
        "schemas": {
            label: audit["schema"] for label, audit in audits.items()
        },
        "storey_comparison": storey_comparison,
        "all_anomalies": {
            label: audit["anomalies"] for label, audit in audits.items()
        },
    }

    save_json(summary, output_dir / "cross_file_summary.json")

    # Markdown cross-file report
    _write_cross_file_markdown(summary, audits, output_dir / "cross_file_summary.md")
    logger.info("  Saved cross-file audit summary")

    return summary


def _write_cross_file_markdown(
    summary: Dict[str, Any],
    audits: Dict[str, Dict[str, Any]],
    path: Path,
) -> None:
    """Write a cross-file audit summary as Markdown."""
    lines = []
    lines.append("# Cross-File IFC Audit Summary")
    lines.append("")
    lines.append("## File Overview")
    lines.append("| File | Schema | Products | Entities | Proxies | Proxy % |")
    lines.append("|------|--------|----------|----------|---------|---------|")
    for label in summary["files_audited"]:
        lines.append(
            f"| {label} | {summary['schemas'][label]} | "
            f"{summary['total_products'][label]:,} | "
            f"{summary['total_entities'][label]:,} | "
            f"{summary['proxy_counts'][label]:,} | "
            f"{summary['proxy_fractions'][label]:.1%} |"
        )
    lines.append("")

    lines.append("## Storey Element Distribution Across Files")
    lines.append("| Storey | " + " | ".join(summary["files_audited"]) + " | Total |")
    lines.append("|--------|" + "|".join(["-------"] * (len(summary["files_audited"]) + 1)) + "|")
    for row in summary["storey_comparison"]:
        vals = " | ".join(
            f"{row.get(f'count_{lbl}', 0):,}" for lbl in summary["files_audited"]
        )
        lines.append(f"| {row['storey_name']} | {vals} | {row['total']:,} |")
    lines.append("")

    lines.append("## All Anomalies")
    for label, anomalies in summary["all_anomalies"].items():
        lines.append(f"### {label}")
        if anomalies:
            for a in anomalies:
                lines.append(f"- ⚠️ {a}")
        else:
            lines.append("- No anomalies.")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
