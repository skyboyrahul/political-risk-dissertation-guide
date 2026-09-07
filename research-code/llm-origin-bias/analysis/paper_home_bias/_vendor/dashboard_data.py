"""Small standalone subset of the monolith dashboard data service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
SIGNALS_DIR = ROOT / "signals"
MODELS_CONFIG_PATH = ROOT / "config" / "models.yaml"

BATCH_TO_BASE: dict[str, str] = {
    "gpt54_batch": "gpt54",
    "xai_grok41fast_batch": "xai_grok41fast",
}

DASHBOARD_MODELS: list[str] = [
    "deepseek_deepseekv32",
    "minimax_m27",
    "xai_grok41fast",
    "gpt54",
]

COMPONENT_MAX: dict[str, int] = {
    "government_stability": 12,
    "socioeconomic_conditions": 12,
    "investment_profile": 12,
    "internal_conflict": 12,
    "external_conflict": 12,
    "corruption": 6,
    "military_in_politics": 6,
    "religious_tensions": 6,
    "law_and_order": 6,
    "ethnic_tensions": 6,
    "democratic_accountability": 6,
    "bureaucracy_quality": 4,
}

COMPONENT_LABELS: dict[str, str] = {
    "government_stability": "Government Stability",
    "socioeconomic_conditions": "Socioeconomic Conditions",
    "investment_profile": "Investment Profile",
    "internal_conflict": "Internal Conflict",
    "external_conflict": "External Conflict",
    "corruption": "Corruption",
    "military_in_politics": "Military in Politics",
    "religious_tensions": "Religious Tensions",
    "law_and_order": "Law and Order",
    "ethnic_tensions": "Ethnic Tensions",
    "democratic_accountability": "Democratic Accountability",
    "bureaucracy_quality": "Bureaucracy Quality",
}
COMPONENT_ORDER = list(COMPONENT_MAX.keys())
COMPONENT_CSV_FIELDS: dict[str, str] = {
    component: COMPONENT_LABELS[component] for component in COMPONENT_ORDER
}


def format_score(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    rounded = round(float(value), 1)
    return int(rounded) if rounded.is_integer() else rounded


def _known_model_tags() -> list[str]:
    if not MODELS_CONFIG_PATH.exists():
        return DASHBOARD_MODELS
    payload = yaml.safe_load(MODELS_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    rows = payload.get("models", []) or []
    tags = [str(row.get("tag", "")).strip() for row in rows if isinstance(row, dict)]
    return [tag for tag in tags if tag]


def _resolve_model_dir(country_dir: Path, model_tag: str) -> Path | None:
    primary = country_dir / model_tag
    if primary.is_dir():
        return primary
    for batch_tag, base_tag in BATCH_TO_BASE.items():
        if base_tag == model_tag and (country_dir / batch_tag).is_dir():
            return country_dir / batch_tag
    return None


def _candidate_model_dirs(country_dir: Path, model_tag: str) -> list[Path]:
    candidates: list[Path] = []
    for batch_tag, base_tag in BATCH_TO_BASE.items():
        if base_tag == model_tag and (country_dir / batch_tag).is_dir():
            candidates.append(country_dir / batch_tag)
    primary = country_dir / model_tag
    if primary.is_dir():
        candidates.append(primary)
    return candidates


def models_with_signals(country: str) -> list[str]:
    country_dir = SIGNALS_DIR / country
    if not country_dir.exists():
        return []
    batch_tags = set(BATCH_TO_BASE)
    available: list[str] = []
    for model in _known_model_tags():
        if model in batch_tags:
            continue
        model_dir = _resolve_model_dir(country_dir, model)
        if model_dir is not None and any(path.is_file() for path in model_dir.glob("*/*.json")):
            available.append(model)
    return available


def scan_signals(country: str, models: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{model: {YYYY-MM: signal-details}}`` for the copied signal tree."""
    active_models = models if models is not None else models_with_signals(country)
    result: dict[str, dict[str, Any]] = {model: {} for model in active_models}

    country_dir = SIGNALS_DIR / country
    for model in active_models:
        model_dirs = _candidate_model_dirs(country_dir, model) if country_dir.exists() else []
        for sig_file in sorted(path for model_dir in model_dirs for path in model_dir.glob("*/*.json")):
            try:
                parts = sig_file.stem.split("_")
                year, month = int(parts[0]), int(parts[1])
                key = f"{year}-{month:02d}"
                payload = json.loads(sig_file.read_text(encoding="utf-8"))
                components: dict[str, float | int | None] = {}
                total = 0.0
                valid = True
                for component, max_value in COMPONENT_MAX.items():
                    value = payload.get(component)
                    components[component] = value
                    if not isinstance(value, (int, float)):
                        valid = False
                        continue
                    total += float(value)
                    if not 0 <= value <= max_value:
                        valid = False
                result[model][key] = {
                    "status": "valid" if valid else "invalid",
                    "total_prs": format_score(total),
                    "event_source": payload.get("event_source", "wikipedia"),
                    "mtime": sig_file.stat().st_mtime,
                    "mtime_iso": datetime.fromtimestamp(
                        sig_file.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "path": sig_file.relative_to(ROOT).as_posix(),
                    "rationale": payload.get("rationale", ""),
                    "components": components,
                }
            except Exception:
                continue
    return result

