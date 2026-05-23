"""
Semantic Classifier Module.

Applies the semantic filtering policy to classify IFC elements into
functional categories for indoor navigation preprocessing.

Classification priority:
1. IFC class-based rules
2. Name pattern-based rules (especially important for proxies)
3. PredefinedType-based rules
4. Default: uncertain

Outputs: classified element tables, category distributions, per-storey summaries.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ifc_loader import IFCFileInfo, get_all_products, get_element_properties, get_storey_for_element
from .utils import load_semantic_policy, save_json, save_dataframe

logger = logging.getLogger(__name__)

# Valid semantic categories
VALID_CATEGORIES = {
    "walkable_support",
    "obstacle",
    "vertical_connector",
    "opening_passage",
    "railing_barrier",
    "structural",
    "ceiling_roof",
    "ignorable",
    "uncertain",
}


class SemanticClassifier:
    """Rule-based semantic classifier for IFC elements.

    Loads rules from a YAML policy file and applies them to classify
    each element into a navigation-relevant category.
    """

    def __init__(self, policy_path: str):
        """Initialize classifier with a policy file.

        Args:
            policy_path: Path to the semantic_policy.yaml file.
        """
        self.policy = load_semantic_policy(policy_path)
        self.class_rules = self.policy.get("class_rules", {})
        self.name_rules = self.policy.get("name_rules", [])
        self.predefined_type_rules = self.policy.get("predefined_type_rules", {})

        # Pre-compile name regex patterns
        self._compiled_name_rules = []
        for rule in self.name_rules:
            try:
                pattern = re.compile(rule["pattern"], re.IGNORECASE)
                self._compiled_name_rules.append({
                    "pattern": pattern,
                    "category": rule["category"],
                    "note": rule.get("note", ""),
                })
            except re.error as e:
                logger.warning(f"Invalid regex in name_rules: {rule['pattern']}: {e}")

        logger.info(
            f"Semantic classifier loaded: "
            f"{len(self.class_rules)} class rules, "
            f"{len(self._compiled_name_rules)} name rules, "
            f"{len(self.predefined_type_rules)} predefined type rules"
        )

    def classify(
        self,
        ifc_class: str,
        name: str = "",
        object_type: str = "",
        predefined_type: str = "",
    ) -> Tuple[str, str, str]:
        """Classify an element based on its attributes.

        Args:
            ifc_class: The IFC class name (e.g., 'IfcWall').
            name: Decoded element name.
            object_type: Decoded object type.
            predefined_type: Predefined type string.

        Returns:
            Tuple of (category, rule_source, note):
            - category: One of VALID_CATEGORIES
            - rule_source: Which rule matched ('class', 'name', 'predefined_type', 'default')
            - note: Explanation of why this category was assigned
        """
        # Step 1: Class-based rule
        if ifc_class in self.class_rules:
            rule = self.class_rules[ifc_class]
            category = rule.get("category", "uncertain")

            # Special case: if class says "uncertain", try name/type rules first
            if category != "uncertain":
                return category, "class_rule", rule.get("note", "")

        # Step 2: Name pattern rules (critical for proxies and uncertain classes)
        # Check element name
        for nrule in self._compiled_name_rules:
            if name and nrule["pattern"].search(name):
                return nrule["category"], "name_rule", nrule["note"]
            if object_type and nrule["pattern"].search(object_type):
                return nrule["category"], "name_rule (object_type)", nrule["note"]

        # Step 3: PredefinedType rules
        if predefined_type and predefined_type in self.predefined_type_rules:
            rule = self.predefined_type_rules[predefined_type]
            return rule["category"], "predefined_type_rule", rule.get("note", "")

        # Step 4: If we had a class rule that said uncertain, return that
        if ifc_class in self.class_rules:
            rule = self.class_rules[ifc_class]
            return rule.get("category", "uncertain"), "class_rule (fallback)", rule.get("note", "")

        # Step 5: Default - unknown class
        return "uncertain", "default", f"No rule matched for class {ifc_class}"


def classify_file_elements(
    file_info: IFCFileInfo,
    classifier: SemanticClassifier,
    output_dir: Path,
) -> pd.DataFrame:
    """Classify all product elements in an IFC file.

    Args:
        file_info: Loaded IFC file information.
        classifier: The semantic classifier.
        output_dir: Output directory for classification results.

    Returns:
        DataFrame with classification results.
    """
    label = file_info.label
    model = file_info.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Classifying elements in: {label}")

    storey_lookup = {s.guid: s.decoded_name for s in file_info.storeys}
    products = get_all_products(model)

    records = []
    for elem in products:
        props = get_element_properties(elem)
        storey_guid = get_storey_for_element(elem, model)
        storey_name = storey_lookup.get(storey_guid, "UNASSIGNED") if storey_guid else "UNASSIGNED"

        category, rule_source, note = classifier.classify(
            ifc_class=props["ifc_class"],
            name=props["name"],
            object_type=props["object_type"],
            predefined_type=props["predefined_type"],
        )

        records.append({
            "guid": props["guid"],
            "ifc_class": props["ifc_class"],
            "name": props["name"],
            "object_type": props["object_type"],
            "predefined_type": props["predefined_type"],
            "storey_name": storey_name,
            "category": category,
            "rule_source": rule_source,
            "rule_note": note,
        })

    df = pd.DataFrame(records)

    # Save full classification
    save_dataframe(df, output_dir / f"classified_{label}.csv")

    # Category distribution
    cat_dist = df["category"].value_counts().to_dict()
    cat_dist_df = pd.DataFrame(
        list(cat_dist.items()), columns=["category", "count"]
    ).sort_values("count", ascending=False)
    save_dataframe(cat_dist_df, output_dir / f"category_distribution_{label}.csv")

    # Per-storey category distribution
    storey_cat = df.groupby(["storey_name", "category"]).size().reset_index(name="count")
    save_dataframe(storey_cat, output_dir / f"storey_category_{label}.csv")

    # Rule source distribution
    rule_dist = df["rule_source"].value_counts().to_dict()

    # Summary
    summary = {
        "file_label": label,
        "total_classified": len(df),
        "category_distribution": cat_dist,
        "rule_source_distribution": rule_dist,
        "uncertain_count": int(df[df["category"] == "uncertain"].shape[0]),
        "uncertain_fraction": round(
            df[df["category"] == "uncertain"].shape[0] / len(df), 4
        ) if len(df) > 0 else 0,
    }
    save_json(summary, output_dir / f"classification_summary_{label}.json")

    logger.info(
        f"  Classified {len(df)} elements: "
        + ", ".join(f"{k}={v}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1]))
    )
    logger.info(f"  Uncertain: {summary['uncertain_count']} ({summary['uncertain_fraction']:.1%})")

    return df


def generate_cross_file_classification_summary(
    all_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a combined classification summary across all files.

    Args:
        all_dfs: Dictionary of label -> classified DataFrame.
        output_dir: Output directory.

    Returns:
        Cross-file classification summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating cross-file classification summary")

    # Combine all files
    combined = pd.concat(
        [df.assign(file_label=label) for label, df in all_dfs.items()],
        ignore_index=True,
    )

    # Cross-file category comparison
    cross_cat = combined.groupby(["file_label", "category"]).size().reset_index(name="count")
    pivot = cross_cat.pivot_table(
        index="category", columns="file_label", values="count", fill_value=0
    ).astype(int)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.to_csv(output_dir / "cross_file_category_comparison.csv", encoding="utf-8-sig")

    # Combined category distribution
    total_dist = combined["category"].value_counts().to_dict()

    summary = {
        "total_elements": len(combined),
        "category_distribution": total_dist,
        "per_file": {
            label: df["category"].value_counts().to_dict()
            for label, df in all_dfs.items()
        },
    }
    save_json(summary, output_dir / "cross_file_classification_summary.json")

    logger.info(f"  Combined classification: {len(combined)} elements across {len(all_dfs)} files")

    return summary
