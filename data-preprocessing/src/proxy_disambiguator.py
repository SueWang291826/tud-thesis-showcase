"""
Automatic Proxy Disambiguation Module (v2).

Replaces the manual proxy review workflow with a multi-signal automatic
classification system.  Each uncertain proxy is evaluated using:

1. Name token matching (highest priority — Chinese/English patterns)
2. Name + dimension combined rules (higher confidence)
3. Dimension-only heuristics (lower confidence, for remaining unknowns)
4. Default fallback

The output is a decision table with:
- decision: keep_traffic_relevant / keep_barrier_relevant /
            drop_traffic_irrelevant / uncertain_low_priority /
            uncertain_high_priority
- sub_category: more specific functional label
- confidence: [0-1] score reflecting rule quality
- rules_fired: which rules matched (for auditability)

Design philosophy:
- Conservative but practical — prefer a clear decision over "uncertain"
- Transparent — every decision is traceable to a named rule
- Editable — rules live in YAML, not hardcoded
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import load_config, save_json, save_dataframe

logger = logging.getLogger(__name__)

# Valid proxy decisions
VALID_DECISIONS = {
    "keep_traffic_relevant",
    "keep_barrier_relevant",
    "drop_traffic_irrelevant",
    "uncertain_low_priority",
    "uncertain_high_priority",
}


class ProxyDisambiguator:
    """Multi-signal automatic proxy classifier."""

    def __init__(self, rules_path: str):
        """Load disambiguation rules from YAML.

        Args:
            rules_path: Path to proxy_disambiguation_rules.yaml.
        """
        self.rules = load_config(rules_path)
        self._compile_name_rules()
        self._compile_name_dim_rules()
        self._parse_dim_rules()
        self.default = self.rules.get("default", {
            "decision": "uncertain_low_priority",
            "confidence": 0.20,
            "note": "No rule matched",
        })

        logger.info(
            f"ProxyDisambiguator loaded: "
            f"{len(self._name_rules)} name rules, "
            f"{len(self._name_dim_rules)} name+dim rules, "
            f"{len(self._dim_rules)} dim-only rules"
        )

    def _compile_name_rules(self):
        """Pre-compile name regex patterns."""
        self._name_rules = []
        for rule in self.rules.get("name_rules", []):
            try:
                pattern = re.compile(rule["pattern"], re.IGNORECASE)
                self._name_rules.append({
                    "pattern": pattern,
                    "pattern_str": rule["pattern"],
                    "decision": rule["decision"],
                    "sub_category": rule.get("sub_category", ""),
                    "confidence": rule.get("confidence", 0.8),
                    "note": rule.get("note", ""),
                })
            except re.error as e:
                logger.warning(f"Invalid regex in disambiguation rules: {rule['pattern']}: {e}")

    def _compile_name_dim_rules(self):
        """Pre-compile combined name+dimension rules."""
        self._name_dim_rules = []
        for rule in self.rules.get("name_and_dim_rules", []):
            try:
                pattern = re.compile(rule["pattern"], re.IGNORECASE)
                self._name_dim_rules.append({
                    "pattern": pattern,
                    "pattern_str": rule["pattern"],
                    "dim_check": rule.get("dimension_check", {}),
                    "decision": rule["decision"],
                    "sub_category": rule.get("sub_category", ""),
                    "confidence": rule.get("confidence", 0.9),
                    "note": rule.get("note", ""),
                })
            except re.error as e:
                logger.warning(f"Invalid regex in name+dim rules: {rule['pattern']}: {e}")

    def _parse_dim_rules(self):
        """Parse dimension-only heuristic rules."""
        self._dim_rules = []
        for rule in self.rules.get("dimension_rules", []):
            self._dim_rules.append({
                "condition": rule.get("condition", ""),
                "dim_check": rule.get("dimension_check", {}),
                "decision": rule["decision"],
                "sub_category": rule.get("sub_category", ""),
                "confidence": rule.get("confidence", 0.4),
                "note": rule.get("note", ""),
            })

    def _check_dimensions(self, dim_check: Dict, dx: float, dy: float, dz: float) -> bool:
        """Evaluate a dimension constraint dictionary against bbox values.

        All dimension values are in metres.
        """
        if "dx_max" in dim_check and dx > dim_check["dx_max"]:
            return False
        if "dy_max" in dim_check and dy > dim_check["dy_max"]:
            return False
        if "dz_max" in dim_check and dz > dim_check["dz_max"]:
            return False
        if "dz_min" in dim_check and dz < dim_check["dz_min"]:
            return False
        if "footprint_min" in dim_check:
            if min(dx, dy) < dim_check["footprint_min"]:
                return False
        if "footprint_max" in dim_check:
            if max(dx, dy) > dim_check["footprint_max"]:
                return False
        return True

    def disambiguate(
        self,
        name: str,
        object_type: str,
        dx: Optional[float],
        dy: Optional[float],
        dz: Optional[float],
        storey_name: str = "",
    ) -> Dict[str, Any]:
        """Classify a single proxy element.

        Args:
            name: Element name (decoded).
            object_type: Object type string.
            dx, dy, dz: BBox dimensions in metres (None if unavailable).
            storey_name: Storey assignment.

        Returns:
            Decision dictionary with keys:
            decision, sub_category, confidence, rules_fired, note
        """
        rules_fired = []
        has_dims = dx is not None and dy is not None and dz is not None

        # Text targets: combine name + object_type
        text_targets = [t for t in [name, object_type] if t]

        # --- Priority 1: Name + dimension combined rules ---
        if has_dims:
            for rule in self._name_dim_rules:
                for text in text_targets:
                    if rule["pattern"].search(text):
                        if self._check_dimensions(rule["dim_check"], dx, dy, dz):
                            rules_fired.append(
                                f"name_dim:{rule['pattern_str']}+dims"
                            )
                            return {
                                "decision": rule["decision"],
                                "sub_category": rule["sub_category"],
                                "confidence": rule["confidence"],
                                "rules_fired": "; ".join(rules_fired),
                                "note": rule["note"],
                            }

        # --- Priority 2: Name-only rules ---
        for rule in self._name_rules:
            for text in text_targets:
                if rule["pattern"].search(text):
                    rules_fired.append(f"name:{rule['pattern_str']}")
                    return {
                        "decision": rule["decision"],
                        "sub_category": rule["sub_category"],
                        "confidence": rule["confidence"],
                        "rules_fired": "; ".join(rules_fired),
                        "note": rule["note"],
                    }

        # --- Priority 3: Dimension-only heuristics ---
        if has_dims:
            for rule in self._dim_rules:
                if self._check_dimensions(rule["dim_check"], dx, dy, dz):
                    rules_fired.append(f"dim:{rule['condition']}")
                    return {
                        "decision": rule["decision"],
                        "sub_category": rule["sub_category"],
                        "confidence": rule["confidence"],
                        "rules_fired": "; ".join(rules_fired),
                        "note": rule["note"],
                    }

        # --- Default fallback ---
        return {
            "decision": self.default["decision"],
            "sub_category": "",
            "confidence": self.default["confidence"],
            "rules_fired": "default_fallback",
            "note": self.default["note"],
        }


def run_proxy_disambiguation(
    proxy_dfs: Dict[str, pd.DataFrame],
    v2_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute automatic proxy disambiguation on all proxy inventories.

    Args:
        proxy_dfs: Per-file proxy inventory DataFrames from v1.
        v2_config: v2 pipeline configuration.
        output_dir: Output directory for disambiguation results.

    Returns:
        Dictionary with resolved proxy DataFrames and summary statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Automatic Proxy Disambiguation (v2)")
    logger.info("=" * 60)

    # Load rules
    rules_path = v2_config.get("proxy_disambiguation", {}).get(
        "rules_file", "config/proxy_disambiguation_rules.yaml"
    )
    # Resolve relative to project root
    from pathlib import Path as P
    project_root = P(__file__).resolve().parent.parent
    rules_full = project_root / rules_path
    disambiguator = ProxyDisambiguator(str(rules_full))

    # Process all proxy inventories
    resolved_dfs = {}
    all_records = []

    for label, pdf in proxy_dfs.items():
        logger.info(f"  Disambiguating proxies for: {label} ({len(pdf)} proxies)")

        records = []
        for _, row in pdf.iterrows():
            # Get bbox dimensions (already in metres from ifcopenshell.geom)
            dx = row.get("bbox_dx") if pd.notna(row.get("bbox_dx")) else None
            dy = row.get("bbox_dy") if pd.notna(row.get("bbox_dy")) else None
            dz = row.get("bbox_dz") if pd.notna(row.get("bbox_dz")) else None

            result = disambiguator.disambiguate(
                name=str(row.get("name", "")),
                object_type=str(row.get("object_type", "")),
                dx=dx, dy=dy, dz=dz,
                storey_name=str(row.get("storey_name", "")),
            )

            record = {
                "guid": row["guid"],
                "name": row.get("name", ""),
                "object_type": row.get("object_type", ""),
                "storey_name": row.get("storey_name", ""),
                "source_file": label,
                "v1_inferred_category": row.get("inferred_category", ""),
                "v2_decision": result["decision"],
                "v2_sub_category": result["sub_category"],
                "v2_confidence": result["confidence"],
                "v2_rules_fired": result["rules_fired"],
                "v2_note": result["note"],
                "bbox_dx_m": dx,
                "bbox_dy_m": dy,
                "bbox_dz_m": dz,
                "bbox_unit": "metre",
            }
            records.append(record)

        rdf = pd.DataFrame(records)
        resolved_dfs[label] = rdf
        all_records.extend(records)

        # Per-file save
        save_dataframe(rdf, output_dir / f"proxy_resolved_{label}.csv")

        # Log decision distribution
        if len(rdf) > 0:
            dist = rdf["v2_decision"].value_counts()
            logger.info(f"    Decisions: {dict(dist)}")

    # Combined table
    combined = pd.DataFrame(all_records)
    save_dataframe(combined, output_dir / "proxy_resolved_combined.csv")

    # Summary statistics
    summary = _compute_disambiguation_summary(combined)
    save_json(summary, output_dir / "proxy_disambiguation_summary.json")
    _write_disambiguation_report(combined, summary, output_dir / "proxy_disambiguation_report.md")

    logger.info(f"  Total proxies processed: {len(combined)}")
    if len(combined) > 0:
        resolved_pct = (
            combined["v2_decision"].isin(
                {"keep_traffic_relevant", "keep_barrier_relevant", "drop_traffic_irrelevant"}
            ).sum() / len(combined) * 100
        )
        uncertain_pct = 100 - resolved_pct
        logger.info(f"  Resolved (high confidence): {resolved_pct:.1f}%")
        logger.info(f"  Remaining uncertain: {uncertain_pct:.1f}%")

    return {
        "resolved_dfs": resolved_dfs,
        "combined": combined,
        "summary": summary,
    }


def _compute_disambiguation_summary(combined: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for proxy disambiguation."""
    if len(combined) == 0:
        return {"total": 0}

    decision_dist = combined["v2_decision"].value_counts().to_dict()
    sub_cat_dist = combined["v2_sub_category"].value_counts().to_dict()
    conf_stats = {
        "mean": round(combined["v2_confidence"].mean(), 3),
        "median": round(combined["v2_confidence"].median(), 3),
        "min": round(combined["v2_confidence"].min(), 3),
        "max": round(combined["v2_confidence"].max(), 3),
    }

    # Before vs after comparison
    v1_uncertain = (combined["v1_inferred_category"] == "uncertain").sum()
    v2_keep = combined["v2_decision"].isin(
        {"keep_traffic_relevant", "keep_barrier_relevant"}
    ).sum()
    v2_drop = (combined["v2_decision"] == "drop_traffic_irrelevant").sum()
    v2_uncertain_hi = (combined["v2_decision"] == "uncertain_high_priority").sum()
    v2_uncertain_lo = (combined["v2_decision"] == "uncertain_low_priority").sum()

    return {
        "total_proxies": len(combined),
        "v1_uncertain_count": int(v1_uncertain),
        "v2_decision_distribution": decision_dist,
        "v2_sub_category_distribution": sub_cat_dist,
        "v2_confidence_stats": conf_stats,
        "resolution_summary": {
            "keep_traffic": int(v2_keep),
            "drop_irrelevant": int(v2_drop),
            "uncertain_high": int(v2_uncertain_hi),
            "uncertain_low": int(v2_uncertain_lo),
            "total_resolved": int(v2_keep + v2_drop),
            "total_uncertain": int(v2_uncertain_hi + v2_uncertain_lo),
            "resolution_rate_pct": round(
                (v2_keep + v2_drop) / len(combined) * 100, 1
            ) if len(combined) > 0 else 0,
        },
        "by_file": {
            f"{file}|{decision}": int(count)
            for (file, decision), count in
            combined.groupby("source_file")["v2_decision"].value_counts().items()
        },
    }


def _write_disambiguation_report(
    combined: pd.DataFrame,
    summary: Dict[str, Any],
    path: Path,
) -> None:
    """Write a human-readable Markdown disambiguation report."""
    lines = []
    lines.append("# Proxy Disambiguation Report (v2)")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Total proxies**: {summary.get('total_proxies', 0)}")
    lines.append(f"- **v1 uncertain**: {summary.get('v1_uncertain_count', 0)}")
    res = summary.get("resolution_summary", {})
    lines.append(f"- **v2 resolved**: {res.get('total_resolved', 0)} "
                 f"({res.get('resolution_rate_pct', 0)}%)")
    lines.append(f"  - Keep (traffic-relevant): {res.get('keep_traffic', 0)}")
    lines.append(f"  - Drop (irrelevant): {res.get('drop_irrelevant', 0)}")
    lines.append(f"- **v2 remaining uncertain**: {res.get('total_uncertain', 0)}")
    lines.append(f"  - High priority: {res.get('uncertain_high', 0)}")
    lines.append(f"  - Low priority: {res.get('uncertain_low', 0)}")
    lines.append("")

    # Confidence statistics
    conf = summary.get("v2_confidence_stats", {})
    lines.append("## Confidence Statistics")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in conf.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Decision distribution
    lines.append("## Decision Distribution")
    lines.append("| Decision | Count |")
    lines.append("|----------|-------|")
    for dec, count in sorted(
        summary.get("v2_decision_distribution", {}).items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"| {dec} | {count} |")
    lines.append("")

    # Sub-category distribution
    lines.append("## Sub-Category Distribution")
    lines.append("| Sub-Category | Count |")
    lines.append("|-------------|-------|")
    for cat, count in sorted(
        summary.get("v2_sub_category_distribution", {}).items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # Sample decision table
    if len(combined) > 0:
        lines.append("## Decision Sample (first 30)")
        lines.append("| GUID | Name | Decision | SubCat | Conf | Rule |")
        lines.append("|------|------|----------|--------|------|------|")
        for _, row in combined.head(30).iterrows():
            guid_short = str(row["guid"])[:12]
            name_short = str(row["name"])[:30]
            lines.append(
                f"| {guid_short}… | {name_short} | {row['v2_decision']} | "
                f"{row['v2_sub_category']} | {row['v2_confidence']:.2f} | "
                f"{row['v2_rules_fired']} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Disambiguation report saved: {path.name}")
