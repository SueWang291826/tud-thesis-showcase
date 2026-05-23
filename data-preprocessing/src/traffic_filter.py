"""
Cross-Storey Traffic Relevance Filter (v2).

Applies a strict keep/drop policy to retain only elements that are
relevant to pedestrian navigation.  Elements are evaluated based on:

1. Storey location (public walkable vs technical)
2. Semantic category (v1 classification)
3. Proxy resolution (v2 disambiguation decisions)
4. IFC class (MEP class blacklist)
5. Name patterns (MEP/technical token detection)

Output classifications:
  retained_walkable  — element on public level relevant to navigation
  retained_connector — vertical/horizontal connector between zones
  retained_barrier   — obstacle or barrier affecting movement
  dropped_technical  — MEP/technical component
  dropped_overhead   — overhead structure not relevant to floor nav
  dropped_non_public — on non-public level with no traffic role
  dropped_uncertain  — unresolved uncertain proxy (low priority)
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .utils import load_config, save_json, save_dataframe

logger = logging.getLogger(__name__)


def run_traffic_filter(
    norm_elements: pd.DataFrame,
    proxy_resolved: pd.DataFrame,
    v2_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Apply traffic-relevance filtering to the normalised element set.

    Args:
        norm_elements: Metre-normalised classified element table.
        proxy_resolved: v2 proxy disambiguation results.
        v2_config: v2 configuration.
        output_dir: Output directory.

    Returns:
        Dictionary with filtered DataFrames and statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Cross-Storey Traffic Relevance Filter (v2)")
    logger.info("=" * 60)

    # Load filter policy
    project_root = Path(__file__).resolve().parent.parent
    policy_path = project_root / v2_config.get("traffic_filter", {}).get(
        "policy_file", "config/traffic_filter_policy.yaml"
    )
    policy = load_config(str(policy_path))

    # Configuration
    tf_cfg = v2_config.get("traffic_filter", {})
    public_levels = set(tf_cfg.get("public_walkable_levels", []))
    connector_levels = set(tf_cfg.get("connector_source_levels", []))
    always_keep_cats = set(tf_cfg.get("always_keep_categories", []))
    always_drop_cats = set(tf_cfg.get("always_drop_categories", []))

    # Policy details
    drop_classes = set(policy.get("drop_ifc_classes", []))
    overhead_classes = set(policy.get("overhead_structural_classes", []))
    drop_patterns = _compile_drop_patterns(policy.get("drop_name_patterns", []))

    # Build proxy decision lookup (guid → v2_decision)
    proxy_lookup = {}
    if len(proxy_resolved) > 0:
        for _, row in proxy_resolved.iterrows():
            proxy_lookup[row["guid"]] = {
                "decision": row["v2_decision"],
                "sub_category": row.get("v2_sub_category", ""),
                "confidence": row.get("v2_confidence", 0),
            }

    logger.info(f"  Total elements to filter: {len(norm_elements)}")
    logger.info(f"  Public levels: {public_levels}")
    logger.info(f"  Proxy decisions available: {len(proxy_lookup)}")

    # Apply filter to each element
    decisions = []
    for _, row in norm_elements.iterrows():
        decision = _classify_element(
            row=row,
            public_levels=public_levels,
            connector_levels=connector_levels,
            always_keep_cats=always_keep_cats,
            always_drop_cats=always_drop_cats,
            drop_classes=drop_classes,
            overhead_classes=overhead_classes,
            drop_patterns=drop_patterns,
            proxy_lookup=proxy_lookup,
        )
        decisions.append(decision)

    # Attach decisions to DataFrame
    result_df = norm_elements.copy()
    result_df["filter_decision"] = [d["decision"] for d in decisions]
    result_df["filter_reason"] = [d["reason"] for d in decisions]
    result_df["filter_output_class"] = [d["output_class"] for d in decisions]

    # Split into retained vs dropped
    retained_mask = result_df["filter_decision"] == "keep"
    dropped_mask = result_df["filter_decision"] == "drop"
    retained_df = result_df[retained_mask].copy()
    dropped_df = result_df[dropped_mask].copy()

    # Save outputs
    save_dataframe(result_df, output_dir / "all_elements_filtered.csv")
    save_dataframe(retained_df, output_dir / "retained_elements.csv")
    save_dataframe(dropped_df, output_dir / "dropped_elements.csv")

    # Category-specific exports
    _export_filtered_subsets(retained_df, output_dir)

    # Summary statistics
    summary = _compute_filter_summary(result_df, retained_df, dropped_df)
    save_json(summary, output_dir / "traffic_filter_summary.json")
    _write_filter_report(summary, result_df, output_dir / "traffic_filter_report.md")

    logger.info(f"  Retained: {len(retained_df)} elements")
    logger.info(f"  Dropped: {len(dropped_df)} elements")
    logger.info(
        f"  Retention rate: {len(retained_df)/len(result_df)*100:.1f}%"
        if len(result_df) > 0 else "  No elements"
    )

    return {
        "all_filtered": result_df,
        "retained": retained_df,
        "dropped": dropped_df,
        "summary": summary,
    }


def _compile_drop_patterns(patterns: List[Dict]) -> List[Dict]:
    """Compile regex drop patterns from policy."""
    compiled = []
    for p in patterns:
        try:
            compiled.append({
                "pattern": re.compile(p["pattern"], re.IGNORECASE),
                "reason": p.get("reason", "name pattern match"),
            })
        except re.error as e:
            logger.warning(f"Invalid drop pattern: {p['pattern']}: {e}")
    return compiled


def _classify_element(
    row: pd.Series,
    public_levels: Set[str],
    connector_levels: Set[str],
    always_keep_cats: Set[str],
    always_drop_cats: Set[str],
    drop_classes: Set[str],
    overhead_classes: Set[str],
    drop_patterns: List[Dict],
    proxy_lookup: Dict[str, Dict],
) -> Dict[str, str]:
    """Classify a single element as keep/drop with reason.

    Returns:
        Dictionary with decision, reason, output_class.
    """
    ifc_class = str(row.get("ifc_class", ""))
    category = str(row.get("category", ""))
    storey = str(row.get("storey_name", ""))
    name = str(row.get("name", ""))
    guid = str(row.get("guid", ""))
    on_public = storey in public_levels
    on_connector_level = storey in connector_levels

    # --- Rule 1: Always-drop IFC classes (MEP) ---
    if ifc_class in drop_classes:
        return {
            "decision": "drop",
            "reason": f"IFC class {ifc_class} is in MEP drop list",
            "output_class": "dropped_technical",
        }

    # --- Rule 2: Name-based MEP detection ---
    for pat in drop_patterns:
        if pat["pattern"].search(name):
            return {
                "decision": "drop",
                "reason": pat["reason"],
                "output_class": "dropped_technical",
            }

    # --- Rule 3: Always-drop categories ---
    if category in always_drop_cats:
        return {
            "decision": "drop",
            "reason": f"Category '{category}' is always dropped",
            "output_class": "dropped_overhead" if category == "ceiling_roof" else "dropped_technical",
        }

    # --- Rule 4: Elements on public walkable levels ---
    if on_public:
        # Overhead structural on public level → drop
        if ifc_class in overhead_classes and category == "structural":
            return {
                "decision": "drop",
                "reason": f"Overhead structural element ({ifc_class}) on public level",
                "output_class": "dropped_overhead",
            }

        # Always-keep categories on public level → keep
        if category in always_keep_cats:
            output_cls = _category_to_output_class(category)
            return {
                "decision": "keep",
                "reason": f"Category '{category}' on public level {storey}",
                "output_class": output_cls,
            }

        # Structural on public level → keep as barrier (columns, etc.)
        if category == "structural":
            return {
                "decision": "keep",
                "reason": f"Structural element on public level (potential obstacle)",
                "output_class": "retained_barrier",
            }

        # Uncertain elements on public level: check proxy resolution
        if category == "uncertain":
            return _resolve_uncertain(guid, storey, proxy_lookup, on_public=True)

        # Anything else on public level — default keep
        return {
            "decision": "keep",
            "reason": f"Element on public level {storey}",
            "output_class": "retained_walkable",
        }

    # --- Rule 5: Elements on connector source levels (F0, F2) ---
    if on_connector_level:
        # Vertical connectors → keep
        if category == "vertical_connector":
            return {
                "decision": "keep",
                "reason": f"Vertical connector on level {storey}",
                "output_class": "retained_connector",
            }

        # Opening passages that might serve as circulation transfer
        if category == "opening_passage":
            return {
                "decision": "keep",
                "reason": f"Opening/passage on connector level {storey}",
                "output_class": "retained_connector",
            }

        # Railings that are part of stair assemblies (check name)
        if category == "railing_barrier":
            if re.search(r"(?i)(stair|楼梯|ramp|坡道)", name):
                return {
                    "decision": "keep",
                    "reason": "Railing associated with stairs/ramps on connector level",
                    "output_class": "retained_connector",
                }

        # Uncertain proxies: check disambiguation
        if category == "uncertain":
            proxy_info = proxy_lookup.get(guid, {})
            if proxy_info.get("decision") in ("keep_traffic_relevant", "keep_barrier_relevant"):
                sub = proxy_info.get("sub_category", "")
                if sub in ("vertical_connector", "escalator", "elevator", "tactile_paving"):
                    return {
                        "decision": "keep",
                        "reason": f"Proxy resolved as traffic-relevant ({sub}) on connector level",
                        "output_class": "retained_connector",
                    }

        # Everything else on non-public levels → drop
        return {
            "decision": "drop",
            "reason": f"Non-public level {storey} — not a traffic connector",
            "output_class": "dropped_non_public",
        }

    # --- Rule 6: UNASSIGNED or other levels ---
    if storey == "UNASSIGNED":
        # IfcOpeningElement unassigned → drop (known residual)
        if ifc_class == "IfcOpeningElement":
            return {
                "decision": "drop",
                "reason": "Unassigned IfcOpeningElement (boolean void)",
                "output_class": "dropped_technical",
            }
        # Vertical connectors even if unassigned → keep
        if category == "vertical_connector":
            return {
                "decision": "keep",
                "reason": "Vertical connector (unassigned storey)",
                "output_class": "retained_connector",
            }
        return {
            "decision": "drop",
            "reason": "Unassigned storey — cannot determine traffic relevance",
            "output_class": "dropped_non_public",
        }

    # --- Rule 7: All other levels (F4, F5 etc.) ---
    if category == "vertical_connector":
        return {
            "decision": "keep",
            "reason": f"Vertical connector on level {storey}",
            "output_class": "retained_connector",
        }

    return {
        "decision": "drop",
        "reason": f"Non-public level {storey} — not traffic-relevant",
        "output_class": "dropped_non_public",
    }


def _resolve_uncertain(
    guid: str,
    storey: str,
    proxy_lookup: Dict[str, Dict],
    on_public: bool,
) -> Dict[str, str]:
    """Resolve uncertain elements using proxy disambiguation results."""
    proxy_info = proxy_lookup.get(guid, {})

    if not proxy_info:
        if on_public:
            # Uncertain on public level without proxy info → keep conservatively
            return {
                "decision": "keep",
                "reason": "Uncertain on public level — kept conservatively",
                "output_class": "retained_walkable",
            }
        return {
            "decision": "drop",
            "reason": "Uncertain without proxy resolution",
            "output_class": "dropped_uncertain",
        }

    v2_dec = proxy_info.get("decision", "")

    if v2_dec == "keep_traffic_relevant":
        return {
            "decision": "keep",
            "reason": f"Proxy resolved: keep_traffic_relevant ({proxy_info.get('sub_category', '')})",
            "output_class": "retained_walkable",
        }
    elif v2_dec == "keep_barrier_relevant":
        return {
            "decision": "keep",
            "reason": f"Proxy resolved: keep_barrier_relevant ({proxy_info.get('sub_category', '')})",
            "output_class": "retained_barrier",
        }
    elif v2_dec == "drop_traffic_irrelevant":
        return {
            "decision": "drop",
            "reason": f"Proxy resolved: drop_traffic_irrelevant ({proxy_info.get('sub_category', '')})",
            "output_class": "dropped_technical",
        }
    elif v2_dec == "uncertain_high_priority" and on_public:
        return {
            "decision": "keep",
            "reason": "Proxy uncertain_high_priority on public level — kept conservatively",
            "output_class": "retained_walkable",
        }
    else:
        if on_public:
            return {
                "decision": "keep",
                "reason": f"Proxy uncertain on public level — kept conservatively",
                "output_class": "retained_walkable",
            }
        return {
            "decision": "drop",
            "reason": f"Proxy {v2_dec} on non-public level",
            "output_class": "dropped_uncertain",
        }


def _category_to_output_class(category: str) -> str:
    """Map semantic category to filter output class."""
    mapping = {
        "walkable_support": "retained_walkable",
        "obstacle": "retained_barrier",
        "vertical_connector": "retained_connector",
        "opening_passage": "retained_walkable",
        "railing_barrier": "retained_barrier",
        "structural": "retained_barrier",
    }
    return mapping.get(category, "retained_walkable")


def _export_filtered_subsets(retained_df: pd.DataFrame, output_dir: Path) -> None:
    """Export subcategory tables from retained elements."""
    subsets = {
        "walkable_objects": retained_df[
            retained_df["filter_output_class"] == "retained_walkable"
        ],
        "connector_objects": retained_df[
            retained_df["filter_output_class"] == "retained_connector"
        ],
        "barrier_objects": retained_df[
            retained_df["filter_output_class"] == "retained_barrier"
        ],
    }

    for name, df in subsets.items():
        if len(df) > 0:
            save_dataframe(df, output_dir / f"{name}.csv")
            logger.info(f"    {name}: {len(df)} elements")

    # Per-storey retained
    for storey in sorted(retained_df["storey_name"].unique()):
        safe = storey.replace(" ", "_").replace("/", "_")
        sdf = retained_df[retained_df["storey_name"] == storey]
        save_dataframe(sdf, output_dir / f"retained_{safe}.csv")


def _compute_filter_summary(
    all_df: pd.DataFrame,
    retained: pd.DataFrame,
    dropped: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute filtering statistics."""
    total = len(all_df)
    return {
        "total_elements": total,
        "retained_count": len(retained),
        "dropped_count": len(dropped),
        "retention_rate_pct": round(len(retained) / total * 100, 1) if total else 0,
        "retained_by_output_class": retained["filter_output_class"].value_counts().to_dict() if len(retained) > 0 else {},
        "dropped_by_output_class": dropped["filter_output_class"].value_counts().to_dict() if len(dropped) > 0 else {},
        "retained_by_storey": retained["storey_name"].value_counts().to_dict() if len(retained) > 0 else {},
        "dropped_by_storey": dropped["storey_name"].value_counts().to_dict() if len(dropped) > 0 else {},
        "retained_by_category": retained["category"].value_counts().to_dict() if len(retained) > 0 else {},
        "dropped_by_category": dropped["category"].value_counts().to_dict() if len(dropped) > 0 else {},
        "retained_by_ifc_class": retained["ifc_class"].value_counts().to_dict() if len(retained) > 0 else {},
    }


def _write_filter_report(
    summary: Dict[str, Any],
    all_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write a human-readable Markdown traffic filter report."""
    lines = []
    lines.append("# Traffic Relevance Filter Report (v2)")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Total elements**: {summary['total_elements']}")
    lines.append(f"- **Retained**: {summary['retained_count']} ({summary['retention_rate_pct']}%)")
    lines.append(f"- **Dropped**: {summary['dropped_count']}")
    lines.append("")

    lines.append("## Retained by Output Class")
    lines.append("| Output Class | Count |")
    lines.append("|-------------|-------|")
    for cls, count in sorted(
        summary.get("retained_by_output_class", {}).items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"| {cls} | {count} |")
    lines.append("")

    lines.append("## Dropped by Reason Class")
    lines.append("| Reason Class | Count |")
    lines.append("|-------------|-------|")
    for cls, count in sorted(
        summary.get("dropped_by_output_class", {}).items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"| {cls} | {count} |")
    lines.append("")

    lines.append("## Retained by Storey")
    lines.append("| Storey | Retained |")
    lines.append("|--------|----------|")
    for s, count in sorted(summary.get("retained_by_storey", {}).items()):
        lines.append(f"| {s} | {count} |")
    lines.append("")

    lines.append("## Retained by Semantic Category")
    lines.append("| Category | Retained |")
    lines.append("|----------|----------|")
    for cat, count in sorted(
        summary.get("retained_by_category", {}).items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # Top drop reasons
    if len(all_df) > 0:
        dropped = all_df[all_df["filter_decision"] == "drop"]
        if len(dropped) > 0:
            lines.append("## Top Drop Reasons")
            lines.append("| Reason | Count |")
            lines.append("|--------|-------|")
            for reason, count in dropped["filter_reason"].value_counts().head(20).items():
                lines.append(f"| {reason} | {count} |")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Traffic filter report saved: {path.name}")
