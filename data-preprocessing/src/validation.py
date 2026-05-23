"""
Validation Module (v2).

Generates comprehensive validation summaries for all v2 processing stages:
1. Unit normalization consistency
2. Proxy uncertainty reduction metrics
3. Cross-storey filtering effectiveness
4. Filtered IFC export success/failure
5. Overall pipeline integrity checks
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import save_json

logger = logging.getLogger(__name__)


def generate_validation_report(
    unit_results: Dict[str, Any],
    proxy_results: Dict[str, Any],
    filter_results: Dict[str, Any],
    ifc_results: Dict[str, Any],
    v2_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a comprehensive v2 validation report.

    Args:
        unit_results: Unit normalization results.
        proxy_results: Proxy disambiguation results.
        filter_results: Traffic filter results.
        ifc_results: IFC export results.
        v2_config: v2 configuration.
        output_dir: Output directory.

    Returns:
        Validation summary dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Generating Validation Report (v2)")
    logger.info("=" * 60)

    report = {
        "unit_normalization": _validate_units(unit_results),
        "proxy_disambiguation": _validate_proxy(proxy_results),
        "traffic_filter": _validate_filter(filter_results),
        "ifc_export": _validate_ifc(ifc_results),
        "overall": {},
    }

    # Overall assessment
    issues = []
    for section, data in report.items():
        if section == "overall":
            continue
        if isinstance(data, dict):
            section_issues = data.get("issues", [])
            issues.extend([f"[{section}] {i}" for i in section_issues])

    report["overall"] = {
        "total_issues": len(issues),
        "all_issues": issues,
        "status": "PASS" if len(issues) == 0 else "WARN" if len(issues) < 5 else "FAIL",
    }

    save_json(report, output_dir / "validation_report.json")
    _write_validation_markdown(report, output_dir / "validation_report.md")

    logger.info(f"  Validation status: {report['overall']['status']}")
    logger.info(f"  Issues found: {len(issues)}")

    return report


def _validate_units(unit_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate unit normalization."""
    issues = []
    checks = {}

    provenance = unit_results.get("provenance", {})
    checks["canonical_unit"] = provenance.get("downstream_canonical_unit", "?")
    checks["ifcopenshell_geom_unit"] = provenance.get("ifcopenshell_geom_output_unit", "?")

    if checks["canonical_unit"] != "metre":
        issues.append("Canonical unit is not 'metre'")

    # Check storey table
    storey_table = unit_results.get("storey_table")
    if storey_table is not None and len(storey_table) > 0:
        checks["storey_count"] = len(storey_table)
        checks["has_elevation_m"] = "elevation_m" in storey_table.columns
        if not checks["has_elevation_m"]:
            issues.append("Storey table missing elevation_m column")
    else:
        issues.append("Storey table is empty or missing")

    # Check normalised elements
    norm_elements = unit_results.get("norm_elements")
    if norm_elements is not None and len(norm_elements) > 0:
        checks["element_count"] = len(norm_elements)
        checks["has_elevation_m_col"] = "elevation_m" in norm_elements.columns
        checks["has_unit_col"] = "coordinate_unit" in norm_elements.columns
    else:
        issues.append("Normalised elements table is empty")

    return {"checks": checks, "issues": issues, "status": "PASS" if not issues else "WARN"}


def _validate_proxy(proxy_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate proxy disambiguation."""
    issues = []
    checks = {}

    summary = proxy_results.get("summary", {})
    total = summary.get("total_proxies", 0)
    checks["total_proxies"] = total

    res = summary.get("resolution_summary", {})
    checks["resolution_rate_pct"] = res.get("resolution_rate_pct", 0)
    checks["remaining_uncertain"] = res.get("total_uncertain", 0)
    checks["resolved_keep"] = res.get("keep_traffic", 0)
    checks["resolved_drop"] = res.get("drop_irrelevant", 0)

    if total > 0 and checks["resolution_rate_pct"] < 50:
        issues.append(
            f"Low proxy resolution rate: {checks['resolution_rate_pct']}% "
            f"({checks['remaining_uncertain']} still uncertain)"
        )

    if checks["remaining_uncertain"] > total * 0.3:
        issues.append(
            f"High remaining uncertainty: {checks['remaining_uncertain']}/{total} proxies"
        )

    return {"checks": checks, "issues": issues, "status": "PASS" if not issues else "WARN"}


def _validate_filter(filter_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate traffic filtering."""
    issues = []
    checks = {}

    summary = filter_results.get("summary", {})
    total = summary.get("total_elements", 0)
    retained = summary.get("retained_count", 0)
    dropped = summary.get("dropped_count", 0)

    checks["total_elements"] = total
    checks["retained"] = retained
    checks["dropped"] = dropped
    checks["retention_rate_pct"] = summary.get("retention_rate_pct", 0)

    if total > 0:
        if retained == 0:
            issues.append("CRITICAL: No elements retained after filtering!")
        if checks["retention_rate_pct"] > 95:
            issues.append(
                f"Suspiciously high retention rate ({checks['retention_rate_pct']}%) "
                f"— filter may not be working"
            )

    # Check retained has public level elements
    retained_by_storey = summary.get("retained_by_storey", {})
    checks["retained_F1"] = retained_by_storey.get("F1 站台层", 0)
    checks["retained_F3"] = retained_by_storey.get("F3 站厅层", 0)

    if checks["retained_F1"] == 0:
        issues.append("No elements retained on F1 platform level")
    if checks["retained_F3"] == 0:
        issues.append("No elements retained on F3 concourse level")

    return {"checks": checks, "issues": issues, "status": "PASS" if not issues else "WARN"}


def _validate_ifc(ifc_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate IFC export results."""
    issues = []
    checks = {}

    if ifc_results.get("status") == "disabled":
        return {"checks": {"enabled": False}, "issues": [], "status": "SKIP"}

    for subset_name, result in ifc_results.items():
        if not isinstance(result, dict):
            continue

        status = result.get("status", "unknown")
        checks[subset_name] = {
            "status": status,
            "elements": result.get("element_count", 0),
            "errors": result.get("copy_errors", 0),
        }

        val = result.get("validation", {})
        if val:
            checks[subset_name]["readable"] = val.get("readable", False)
            if not val.get("readable", False) and status == "success":
                issues.append(f"Subset '{subset_name}' exported but failed validation")
        elif status == "success":
            issues.append(f"Subset '{subset_name}' not validated")

        if status == "error":
            issues.append(f"Subset '{subset_name}' export failed: {result.get('message', '')}")

    return {"checks": checks, "issues": issues, "status": "PASS" if not issues else "WARN"}


def _write_validation_markdown(report: Dict[str, Any], path: Path) -> None:
    """Write validation report as Markdown."""
    lines = []
    lines.append("# V2 Pipeline Validation Report")
    lines.append("")

    overall = report.get("overall", {})
    status = overall.get("status", "UNKNOWN")
    status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "❓")

    lines.append(f"## Overall Status: {status_emoji} {status}")
    lines.append(f"- **Issues found**: {overall.get('total_issues', 0)}")
    lines.append("")

    if overall.get("all_issues"):
        lines.append("### All Issues")
        for issue in overall["all_issues"]:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")

    # Section details
    for section in ["unit_normalization", "proxy_disambiguation", "traffic_filter", "ifc_export"]:
        data = report.get(section, {})
        sec_status = data.get("status", "?")
        lines.append(f"## {section.replace('_', ' ').title()}: {sec_status}")

        checks = data.get("checks", {})
        if checks:
            lines.append("| Check | Value |")
            lines.append("|-------|-------|")
            for k, v in checks.items():
                if isinstance(v, dict):
                    v_str = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                else:
                    v_str = str(v)
                lines.append(f"| {k} | {v_str} |")

        sec_issues = data.get("issues", [])
        if sec_issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in sec_issues:
                lines.append(f"- ⚠️ {issue}")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Validation report saved: {path.name}")
