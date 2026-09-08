"""Provenance helpers for canonical ML artefacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _repo_root(path: Path) -> Path:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path if path.is_dir() else path.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return Path(root)
    except Exception:
        return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _script_path(root: Path) -> Path | None:
    candidate = Path(sys.argv[0])
    if candidate.exists():
        return candidate.resolve()
    return None


def _git_blob_sha(path: Path | None, root: Path) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "hash-object", str(path)],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "xgboost",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def provenance_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    if path.suffix:
        return path.with_suffix(".provenance.json")
    return path.with_name(f"{path.name}.provenance.json")


def write_manifest(
    out_path: str | Path,
    *,
    input_paths: Iterable[str | Path] = (),
    script_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a sibling provenance manifest for an output artefact."""
    output = Path(out_path)
    root = _repo_root(output)
    script = Path(script_path).resolve() if script_path is not None else _script_path(root)

    inputs = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        inputs.append(
            {
                "path": _relative(path, root),
                "exists": path.exists(),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": {
            "path": _relative(output, root),
            "exists": output.exists(),
            "sha256": _sha256(output),
        },
        "script": {
            "path": _relative(script, root) if script else None,
            "git_blob_sha": _git_blob_sha(script, root),
            "sha256": _sha256(script) if script else None,
        },
        "inputs": inputs,
        "environment": {
            "python": sys.version,
            "packages": _package_versions(),
            "cwd": str(Path.cwd()),
            "command_line": sys.argv,
            "EVENTS_SINGLE_SOURCE": os.environ.get("EVENTS_SINGLE_SOURCE"),
            "PRS_NOWCAST_ALLOW_LLM_CALLS": os.environ.get("PRS_NOWCAST_ALLOW_LLM_CALLS"),
        },
    }
    if extra:
        manifest["extra"] = extra

    manifest_path = provenance_path(output)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
