"""Recompute the Chapter 5 five-country repeat-generation results.

This module is deliberately analysis-only. It compares the 720 frozen repeat
responses from the original study with their corresponding canonical originals.
Those empirical inputs are excluded from this public source edition. It never calls an LLM.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT
    / "results"
    / "external_evidence"
    / "monthly-llm-risk-signals"
    / "experiments"
    / "self_consistency"
)
REGEN_ROOT = EVIDENCE_ROOT / "regens"
RESULTS_ROOT = EVIDENCE_ROOT / "results"
BUNDLE_MANIFEST = EVIDENCE_ROOT / "REGEN_BUNDLE_MANIFEST.json"

COUNTRIES = ("south_africa", "china", "united_states", "russia", "brazil")
MODELS = ("deepseek_deepseekv32", "minimax_m27", "xai_grok41fast", "gpt54")
YEARS = (2015, 2018, 2021)
MONTHS = tuple(range(1, 13))
COMPONENTS = (
    "government_stability",
    "socioeconomic_conditions",
    "investment_profile",
    "internal_conflict",
    "external_conflict",
    "corruption",
    "military_in_politics",
    "religious_tensions",
    "law_and_order",
    "ethnic_tensions",
    "democratic_accountability",
    "bureaucracy_quality",
)


@dataclass(frozen=True)
class Pair:
    country: str
    model: str
    year: int
    month: int
    original: dict[str, Any]
    regenerated: dict[str, Any]

    @property
    def route_confounded(self) -> bool:
        """Whether the original and repeat used different serving routes."""
        return self.country == "brazil" and self.model == "xai_grok41fast"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _signal_path(country: str, model: str, year: int, month: int) -> Path:
    return REPO_ROOT / "signals" / country / model / str(year) / f"{year}_{month:02d}.json"


def _regen_path(country: str, model: str, year: int, month: int) -> Path:
    return REGEN_ROOT / country / model / str(year) / f"{year}_{month:02d}.json"


def load_pairs() -> list[Pair]:
    """Load the locked five-country grid, failing on any missing cell."""
    pairs: list[Pair] = []
    for country in COUNTRIES:
        for model in MODELS:
            for year in YEARS:
                for month in MONTHS:
                    original_path = _signal_path(country, model, year, month)
                    regen_path = _regen_path(country, model, year, month)
                    if not original_path.is_file() or not regen_path.is_file():
                        raise FileNotFoundError(
                            f"missing repeat-generation pair: {country}/{model}/{year}-{month:02d}"
                        )
                    pairs.append(
                        Pair(
                            country=country,
                            model=model,
                            year=year,
                            month=month,
                            original=json.loads(original_path.read_text(encoding="utf-8")),
                            regenerated=json.loads(regen_path.read_text(encoding="utf-8")),
                        )
                    )
    return pairs


def composite(payload: dict[str, Any]) -> float:
    """Return the 0--100 composite as the sum of the twelve components."""
    return float(sum(float(payload[component]) for component in COMPONENTS))


def icc_3_1(original: Iterable[float], regenerated: Iterable[float]) -> float | None:
    """Return ICC(3,1), the two-way mixed-effects single-measure coefficient."""
    left = list(original)
    right = list(regenerated)
    if len(left) != len(right) or len(left) < 2:
        return None
    values = np.asarray(list(zip(left, right)), dtype=float)
    if np.allclose(values, values[0, 0]):
        return None

    n_targets, n_raters = values.shape
    grand_mean = float(values.mean())
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ss_rows = n_raters * float(np.sum((row_means - grand_mean) ** 2))
    ss_error = float(
        np.sum((values - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)
    )
    ms_rows = ss_rows / (n_targets - 1)
    ms_error = ss_error / ((n_targets - 1) * (n_raters - 1))
    denominator = ms_rows + ((n_raters - 1) * ms_error)
    if abs(denominator) < 1e-12:
        return None
    return (ms_rows - ms_error) / denominator


def metrics(value_pairs: Iterable[tuple[float, float]]) -> dict[str, Any]:
    """Compute the exact agreement statistics stored by the source analysis."""
    pairs = list(value_pairs)
    original = [float(left) for left, _ in pairs]
    regenerated = [float(right) for _, right in pairs]
    absolute = [abs(left - right) for left, right in pairs]
    correlation = None
    if len(original) >= 2 and len(set(original)) > 1 and len(set(regenerated)) > 1:
        correlation = float(pearsonr(original, regenerated).statistic)
    return {
        "count": len(pairs),
        "mad": _round(sum(absolute) / len(absolute)),
        "exact_match_pct": _round(100.0 * sum(value == 0 for value in absolute) / len(absolute), 2),
        "max_abs_diff": _round(max(absolute)),
        "pearson_r": _round(correlation),
        "icc_3_1": _round(icc_3_1(original, regenerated)),
        "mean_original": _round(sum(original) / len(original)),
        "mean_regenerated": _round(sum(regenerated) / len(regenerated)),
    }


def _group_metrics(pairs: Iterable[Pair]) -> dict[str, dict[str, Any]]:
    rows = list(pairs)
    component_pairs = [
        (float(pair.original[component]), float(pair.regenerated[component]))
        for pair in rows
        for component in COMPONENTS
    ]
    composite_pairs = [(composite(pair.original), composite(pair.regenerated)) for pair in rows]
    return {
        "all_components": metrics(component_pairs),
        "composite": metrics(composite_pairs),
    }


def recompute() -> dict[str, Any]:
    """Recompute the overall, per-model and per-country numerical contracts."""
    pairs = load_pairs()
    return {
        "selection": {
            "countries": list(COUNTRIES),
            "models": list(MODELS),
            "years": list(YEARS),
        },
        "counts": {
            "expected_signal_pairs": 720,
            "completed_signal_pairs": len(pairs),
            "completed_component_pairs": len(pairs) * len(COMPONENTS),
            "missing_pairs": 0,
            "route_confounded_pairs": sum(pair.route_confounded for pair in pairs),
        },
        "overall": _group_metrics(pairs),
        "per_model": {
            model: _group_metrics(pair for pair in pairs if pair.model == model) for model in MODELS
        },
        "per_country": {
            country: _group_metrics(pair for pair in pairs if pair.country == country)
            for country in COUNTRIES
        },
    }


def verify_against_frozen() -> dict[str, Any]:
    """Recompute and compare every thesis-facing aggregate with frozen evidence."""
    actual = recompute()
    expected_summary = json.loads((RESULTS_ROOT / "summary.json").read_text(encoding="utf-8"))
    expected_models = json.loads((RESULTS_ROOT / "per_model.json").read_text(encoding="utf-8"))
    expected_countries = json.loads((RESULTS_ROOT / "per_country.json").read_text(encoding="utf-8"))
    manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))

    if actual["selection"] != expected_summary["selection"]:
        raise AssertionError("five-country selection differs from frozen summary")
    for key in ("expected_signal_pairs", "completed_signal_pairs", "completed_component_pairs", "missing_pairs"):
        if actual["counts"][key] != expected_summary["counts"][key]:
            raise AssertionError(f"count mismatch for {key}")
    if actual["counts"]["route_confounded_pairs"] != 36:
        raise AssertionError("expected 36 Brazil/Grok route-confounded pairs")
    if actual["overall"] != expected_summary["overall"]:
        raise AssertionError("overall repeat-generation metrics differ from frozen summary")
    for model in MODELS:
        if actual["per_model"][model] != expected_models["models"][model]["overall"]:
            raise AssertionError(f"per-model metrics differ for {model}")
    for country in COUNTRIES:
        if actual["per_country"][country] != expected_countries["countries"][country]["overall"]:
            raise AssertionError(f"per-country metrics differ for {country}")
    if manifest["file_count"] != 720 or len(manifest["files"]) != 720:
        raise AssertionError("repeat-generation bundle manifest is not a complete 720-file grid")
    return actual
