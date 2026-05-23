"""
Shared utility functions for the IFC preprocessing pipeline.

Provides:
- Configuration loading (YAML)
- Logging setup
- Path management
- IFC string decoding (X2 encoding)
- File hashing for reproducibility
- Timer utilities
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ==============================================================================
# Configuration
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_semantic_policy(policy_path: str) -> Dict[str, Any]:
    """Load the semantic filtering policy from YAML.

    Args:
        policy_path: Path to the semantic policy YAML file.

    Returns:
        Policy dictionary.
    """
    return load_config(policy_path)


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(log_dir: str, run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Set up structured logging to both file and console.

    Args:
        log_dir: Directory for log files.
        run_id: Unique run identifier for the log filename.
        level: Logging level.

    Returns:
        Configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"preprocessing_{run_id}.log"

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # File handler - detailed
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_fmt)
    root_logger.addHandler(fh)

    # Console handler - concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(console_fmt)
    root_logger.addHandler(ch)

    return root_logger


# ==============================================================================
# Path Management
# ==============================================================================

def ensure_output_dirs(config: Dict[str, Any], project_root: Path) -> Dict[str, Path]:
    """Create all output directories specified in config.

    Args:
        config: Pipeline configuration dictionary.
        project_root: Root path of the preprocessing project.

    Returns:
        Dictionary mapping directory keys to absolute paths.
    """
    output_cfg = config["output"]
    dirs = {}
    for key, rel_path in output_cfg.items():
        abs_path = project_root / rel_path
        abs_path.mkdir(parents=True, exist_ok=True)
        dirs[key] = abs_path
    return dirs


# ==============================================================================
# IFC String Decoding
# ==============================================================================

def decode_ifc_x2(text: str) -> str:
    r"""Decode IFC \\X2\\ encoded Unicode strings.

    In IFC files, Unicode characters are encoded as \\X2\\XXXX\\X0\\
    where XXXX is the hex code point.

    Args:
        text: Raw IFC string possibly containing X2 encoding.

    Returns:
        Decoded Unicode string.
    """
    if text is None:
        return ""
    if "\\X2\\" not in text and "\\X0\\" not in text:
        return text

    def _replace(match: re.Match) -> str:
        hex_str = match.group(1)
        chars = []
        for i in range(0, len(hex_str), 4):
            code_point = int(hex_str[i : i + 4], 16)
            chars.append(chr(code_point))
        return "".join(chars)

    return re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", _replace, text)


def safe_ifc_str(value: Any) -> str:
    """Safely convert an IFC attribute value to a decoded string.

    Handles None, X2 encoding, and other edge cases.
    """
    if value is None:
        return ""
    s = str(value)
    return decode_ifc_x2(s)


# ==============================================================================
# File Hashing
# ==============================================================================

def file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        filepath: Path to the file.
        chunk_size: Read buffer size.

    Returns:
        Hex digest string.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ==============================================================================
# Run Manifest
# ==============================================================================

def create_run_manifest(
    config: Dict[str, Any],
    project_root: Path,
    ifc_files: Dict[str, Path],
    run_id: str,
) -> Dict[str, Any]:
    """Create a run manifest documenting input files and run parameters.

    Args:
        config: Pipeline configuration.
        project_root: Root path.
        ifc_files: Dictionary of label -> Path for IFC files.
        run_id: Unique run identifier.

    Returns:
        Manifest dictionary.
    """
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "project_root": str(project_root),
        "config_snapshot": config,
        "input_files": {},
    }
    for label, fpath in ifc_files.items():
        manifest["input_files"][label] = {
            "path": str(fpath),
            "size_bytes": fpath.stat().st_size,
            "sha256": file_sha256(str(fpath)),
        }
    return manifest


def save_run_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Save run manifest to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


# ==============================================================================
# Timer Context Manager
# ==============================================================================

class Timer:
    """Simple timer for measuring execution duration."""

    def __init__(self, description: str = "", logger: Optional[logging.Logger] = None):
        self.description = description
        self.logger = logger
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.logger and self.description:
            self.logger.info(f"[START] {self.description}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.logger and self.description:
            self.logger.info(
                f"[DONE]  {self.description} ({self.elapsed:.1f}s)"
            )


# ==============================================================================
# JSON / CSV helpers
# ==============================================================================

def save_json(data: Any, path: Path, ensure_ascii: bool = False) -> None:
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii, default=str)


def save_dataframe(df, path: Path, index: bool = False) -> None:
    """Save a pandas DataFrame to CSV with UTF-8 BOM for Excel compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
