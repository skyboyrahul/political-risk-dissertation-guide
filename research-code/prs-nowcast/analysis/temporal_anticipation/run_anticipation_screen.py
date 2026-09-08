#!/usr/bin/env python3
"""Temporal-anticipation screen on the canonical monthly LLM signal panel.

This adapts the annual closed-book leakage screen in
``llm-divergence/scripts/audits/a2_temporal_leakage_v2.py`` to the monthly,
evidence-fed panel held in ``signals/``.  For each (model, shock event) pair it
asks whether the model's monthly composite drifted away from its own recent
baseline during the twelve months immediately before a known shock.

Two properties of this panel differ from the original screen and change how the
output must be read.

1.  The metric is the ICRG-style composite, the plain sum of the twelve
    component scores, matching ``pipeline/gdi_features.py``.  Higher composite
    means *lower* risk, so an anticipated shock appears as a **negative**
    z-score.  The original screen used a break probability, where anticipation
    appeared as a positive z.  Signs are therefore inverted relative to the
    source design.
2.  Every model saw the same Wikipedia evidence bundle for the same month, so a
    raw elevated anticipation score is not by itself evidence of parametric
    leakage: it may simply mean the contemporaneous evidence was already
    alarming.  The cross-model differential (a model's z minus the four-model
    mean z for the same event) is the statistic that isolates model-specific
    behaviour on shared evidence.

Run from the repository root::

    python analysis/temporal_anticipation/run_anticipation_screen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_DIR = REPO_ROOT / "signals"
OUT_DIR = REPO_ROOT / "results" / "methods_evidence" / "temporal_anticipation"

# Component list and plain-sum composite convention copied from
# pipeline/gdi_features.py.
COMPONENTS = [
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
]

MODELS = [
    "deepseek_deepseekv32",
    "gpt54",
    "minimax_m27",
    "xai_grok41fast",
]

PANEL_FIRST_MONTH = (2005, 1)
PANEL_LAST_MONTH = (2024, 12)

PRE_SHOCK_MONTHS = 12
BASELINE_MONTHS = 36
CLIP_LIMIT = 10.0
STD_FLOOR = 1e-6
STD_DDOF = 1

PLACEBO_SHOCK_MONTH = (2017, 6)

# The ten TEST_CASES of the source screen, with shock months supplied at monthly
# resolution.  ``canonical_country`` is the signals/ directory name, or None if
# the country is outside the canonical 25 and the case must be dropped.
EVENT_CASES = [
    {"event": "COVID-19", "country": "China", "canonical_country": "china",
     "shock_year": 2020, "shock_month": 1},
    {"event": "Russia-Ukraine War", "country": "Ukraine", "canonical_country": "ukraine",
     "shock_year": 2022, "shock_month": 2},
    {"event": "Russia-Ukraine War", "country": "Russia", "canonical_country": "russia",
     "shock_year": 2022, "shock_month": 2},
    {"event": "Arab Spring", "country": "Egypt", "canonical_country": "egypt",
     "shock_year": 2011, "shock_month": 1},
    {"event": "Arab Spring", "country": "Tunisia", "canonical_country": None,
     "shock_year": 2011, "shock_month": 1},
    {"event": "Arab Spring", "country": "Libya", "canonical_country": None,
     "shock_year": 2011, "shock_month": 2},
    {"event": "Brexit referendum", "country": "United Kingdom",
     "canonical_country": "united_kingdom", "shock_year": 2016, "shock_month": 6},
    {"event": "Bolsonaro election", "country": "Brazil", "canonical_country": "brazil",
     "shock_year": 2018, "shock_month": 10},
    {"event": "Venezuela crisis", "country": "Venezuela", "canonical_country": "venezuela",
     "shock_year": 2017, "shock_month": 3},
    {"event": "Myanmar coup", "country": "Myanmar", "canonical_country": None,
     "shock_year": 2021, "shock_month": 2},
]


def month_index(year: int, month: int) -> int:
    """Map a calendar month to a dense integer index."""
    return year * 12 + (month - 1)


def index_to_label(index: int) -> str:
    year, month = divmod(index, 12)
    return f"{year}-{month + 1:02d}"


def load_composites(signals_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """Load monthly composites as ``{country: {model: {month_index: composite}}}``."""
    panel: dict[str, dict[str, dict[int, float]]] = {}

    for country_dir in sorted(p for p in signals_dir.iterdir() if p.is_dir()):
        country = country_dir.name
        by_model: dict[str, dict[int, float]] = {}

        for model_dir in sorted(p for p in country_dir.iterdir() if p.is_dir()):
            months: dict[int, float] = {}
            for sig_file in sorted(model_dir.rglob("*.json")):
                payload = json.loads(sig_file.read_text(encoding="utf-8"))
                total = 0.0
                for comp in COMPONENTS:
                    value = payload[comp]
                    if not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Non-numeric component {comp} in {sig_file}"
                        )
                    total += float(value)
                year = int(payload["year"])
                month = int(payload["month"])
                months[month_index(year, month)] = total
            if months:
                by_model[model_dir.name] = months

        if by_model:
            panel[country] = by_model

    return panel


def pooled_std(series: dict[int, float]) -> float:
    """Sample standard deviation of a model-country composite over the panel."""
    lo = month_index(*PANEL_FIRST_MONTH)
    hi = month_index(*PANEL_LAST_MONTH)
    values = [v for idx, v in sorted(series.items()) if lo <= idx <= hi]
    return float(np.std(np.asarray(values, dtype=float), ddof=STD_DDOF))


def compute_case(
    series: dict[int, float],
    shock_idx: int,
) -> dict[str, Any]:
    """Compute the anticipation score for one model-country-shock triple."""
    pre_start = shock_idx - PRE_SHOCK_MONTHS
    pre_end = shock_idx - 1
    base_start = pre_start - BASELINE_MONTHS
    base_end = pre_start - 1

    baseline_idx = list(range(base_start, base_end + 1))
    pre_idx = list(range(pre_start, pre_end + 1))

    missing = [i for i in baseline_idx + pre_idx if i not in series]
    if missing:
        raise KeyError(
            f"missing months {[index_to_label(i) for i in missing]}"
        )

    baseline_values = np.asarray([series[i] for i in baseline_idx], dtype=float)
    pre_values = np.asarray([series[i] for i in pre_idx], dtype=float)

    baseline_mean = float(baseline_values.mean())
    baseline_std_raw = float(baseline_values.std(ddof=STD_DDOF))

    fallback_fired = baseline_std_raw < STD_FLOOR
    baseline_std = pooled_std(series) if fallback_fired else baseline_std_raw

    monthly_z = [(float(v) - baseline_mean) / baseline_std for v in pre_values]
    raw_score = float(np.mean(monthly_z))
    score = float(np.clip(raw_score, -CLIP_LIMIT, CLIP_LIMIT))

    return {
        "shock_month": index_to_label(shock_idx),
        "pre_shock_window": [index_to_label(pre_start), index_to_label(pre_end)],
        "baseline_window": [index_to_label(base_start), index_to_label(base_end)],
        "baseline_mean": baseline_mean,
        "baseline_std_raw": baseline_std_raw,
        "baseline_std_used": baseline_std,
        "baseline_std_fallback": fallback_fired,
        "pre_shock_mean_composite": float(pre_values.mean()),
        "monthly_z": monthly_z,
        "anticipation_z_unclipped": raw_score,
        "anticipation_z": score,
        "clipped": raw_score != score,
    }


def resolve_cases(panel: dict[str, dict[str, dict[int, float]]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Split the source test cases into kept events, dropped events and placebos."""
    canonical_countries = sorted(panel)
    panel_start = month_index(*PANEL_FIRST_MONTH)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for case in EVENT_CASES:
        country = case["canonical_country"]
        shock_idx = month_index(case["shock_year"], case["shock_month"])
        base_start = shock_idx - PRE_SHOCK_MONTHS - BASELINE_MONTHS

        if country is None or country not in canonical_countries:
            dropped.append({
                "event": case["event"],
                "country": case["country"],
                "shock_month": index_to_label(shock_idx),
                "reason": "country absent from the canonical 25-country panel",
            })
            continue
        if base_start < panel_start:
            dropped.append({
                "event": case["event"],
                "country": case["country"],
                "shock_month": index_to_label(shock_idx),
                "reason": (
                    "baseline window would start at "
                    f"{index_to_label(base_start)}, before the panel start "
                    f"{index_to_label(panel_start)}"
                ),
            })
            continue

        kept.append({
            "case_type": "event",
            "case_id": f"{case['event']} / {case['country']}",
            "event": case["event"],
            "country": case["country"],
            "canonical_country": country,
            "shock_idx": shock_idx,
        })

    event_countries = {c["canonical_country"] for c in kept}
    placebo_idx = month_index(*PLACEBO_SHOCK_MONTH)
    placebos = [
        {
            "case_type": "placebo",
            "case_id": f"PLACEBO / {country}",
            "event": "PLACEBO",
            "country": country,
            "canonical_country": country,
            "shock_idx": placebo_idx,
        }
        for country in canonical_countries
        if country not in event_countries
    ]

    return kept, dropped, placebos


def welch(events: list[float], placebos: list[float]) -> dict[str, Any]:
    result = stats.ttest_ind(events, placebos, equal_var=False)
    return {
        "test": "Welch two-sample t-test (events vs placebos)",
        "n_events": len(events),
        "n_placebos": len(placebos),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": float(result.df),
    }


def build_spec(
    run_date: str,
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    placebos: list[dict[str, Any]],
    fallbacks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_date": run_date,
        "screen": "temporal anticipation on the canonical monthly LLM signal panel",
        "source_design": (
            "llm-divergence/scripts/audits/a2_temporal_leakage_v2.py, "
            "llm-divergence/docs/A2_METHODOLOGY.md"
        ),
        "signals_dir": str(SIGNALS_DIR.relative_to(REPO_ROOT)),
        "models": list(MODELS),
        "n_countries": 25,
        "panel_window": [
            f"{PANEL_FIRST_MONTH[0]}-{PANEL_FIRST_MONTH[1]:02d}",
            f"{PANEL_LAST_MONTH[0]}-{PANEL_LAST_MONTH[1]:02d}",
        ],
        "metric": "composite = plain sum of the 12 ICRG-style component scores",
        "metric_source": "pipeline/gdi_features.py (_load_signals plain sum)",
        "sign_convention": (
            "Higher composite means lower risk, so anticipated deterioration "
            "appears as a NEGATIVE z-score. This inverts the sign convention of "
            "the source annual screen, which used a break probability."
        ),
        "pre_shock_window_months": PRE_SHOCK_MONTHS,
        "pre_shock_window_rule": "the 12 calendar months strictly before the shock month",
        "baseline_window_months": BASELINE_MONTHS,
        "baseline_window_rule": (
            "the 36 calendar months strictly before the pre-shock window"
        ),
        "z_rule": (
            "z_t = (composite_t - baseline_mean) / baseline_std for each of the 12 "
            "pre-shock months; anticipation score = mean of the 12 z_t values"
        ),
        "std_ddof": STD_DDOF,
        "clip": [-CLIP_LIMIT, CLIP_LIMIT],
        "degenerate_std_rule": (
            f"if baseline std < {STD_FLOOR:g}, substitute the model's pooled monthly "
            "composite std for that country over 2005-2024"
        ),
        "degenerate_std_firings": fallbacks,
        "placebo_rule": (
            "every canonical country with no kept event, evaluated at a pseudo-shock "
            "month of "
            f"{PLACEBO_SHOCK_MONTH[0]}-{PLACEBO_SHOCK_MONTH[1]:02d}"
        ),
        "placebo_shock_month_rationale": (
            "an arbitrary mid-panel month chosen for the absence of a global shock; "
            "it sits between the 2015-2016 refugee and Brexit period and the 2018 "
            "trade-war period, and its 2013-06 to 2017-05 evaluation span contains no "
            "event from the kept list"
        ),
        "placebo_countries": [c["canonical_country"] for c in placebos],
        "kept_events": [
            {
                "event": c["event"],
                "country": c["country"],
                "canonical_country": c["canonical_country"],
                "shock_month": index_to_label(c["shock_idx"]),
            }
            for c in kept
        ],
        "dropped_events": dropped,
        "tests": [
            "per model: Welch two-sample t-test of event scores against placebo scores",
            (
                "per event: cross-model differential, a model's z minus the "
                "four-model mean z for that event"
            ),
        ],
        "differential_interpretation": (
            "All four models received the same Wikipedia evidence bundle for the same "
            "month, so a raw anticipation score confounds parametric foreknowledge "
            "with legitimate evidence-driven early warning. The differential removes "
            "the event-common component and is the leakage-shaped statistic on shared "
            "evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal-anticipation screen on the monthly LLM signal panel."
    )
    parser.add_argument(
        "--date",
        default="2026-07-29",
        help="Run date recorded in the spec block (default: 2026-07-29).",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=SIGNALS_DIR,
        help="Path to the signals directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for the JSON and CSV artefacts.",
    )
    args = parser.parse_args()

    panel = load_composites(args.signals_dir)
    observed_models = sorted({m for by_model in panel.values() for m in by_model})
    if observed_models != MODELS:
        raise SystemExit(
            f"model roster mismatch: expected {MODELS}, found {observed_models}"
        )

    kept, dropped, placebos = resolve_cases(panel)

    fallbacks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for case in kept + placebos:
        for model in MODELS:
            series = panel[case["canonical_country"]][model]
            detail = compute_case(series, case["shock_idx"])
            if detail["baseline_std_fallback"]:
                fallbacks.append({
                    "model": model,
                    "case_id": case["case_id"],
                    "baseline_std_raw": detail["baseline_std_raw"],
                    "pooled_std_used": detail["baseline_std_used"],
                })
            rows.append({
                "model": model,
                "case_type": case["case_type"],
                "case_id": case["case_id"],
                "event": case["event"],
                "country": case["country"],
                "canonical_country": case["canonical_country"],
                **detail,
            })

    frame = pd.DataFrame(rows)

    # Cross-model differential, events only.
    event_frame = frame[frame["case_type"] == "event"]
    event_ids = list(dict.fromkeys(c["case_id"] for c in kept))
    event_mean_z = {
        case_id: float(
            event_frame.loc[event_frame["case_id"] == case_id, "anticipation_z"].mean()
        )
        for case_id in event_ids
    }
    differentials = {
        model: {
            case_id: float(
                event_frame.loc[
                    (event_frame["model"] == model)
                    & (event_frame["case_id"] == case_id),
                    "anticipation_z",
                ].iloc[0]
            )
            - event_mean_z[case_id]
            for case_id in event_ids
        }
        for model in MODELS
    }

    # Placebo differential, reported as the null reference for the event
    # differential: a model that simply runs hotter than its peers everywhere
    # would show a non-zero event differential without any event-specific
    # behaviour.
    placebo_frame = frame[frame["case_type"] == "placebo"]
    placebo_ids = [c["case_id"] for c in placebos]
    placebo_mean_z = {
        case_id: float(
            placebo_frame.loc[
                placebo_frame["case_id"] == case_id, "anticipation_z"
            ].mean()
        )
        for case_id in placebo_ids
    }
    placebo_differentials = {
        model: {
            case_id: float(
                placebo_frame.loc[
                    (placebo_frame["model"] == model)
                    & (placebo_frame["case_id"] == case_id),
                    "anticipation_z",
                ].iloc[0]
            )
            - placebo_mean_z[case_id]
            for case_id in placebo_ids
        }
        for model in MODELS
    }

    frame["differential_z"] = [
        differentials[r["model"]][r["case_id"]]
        if r["case_type"] == "event"
        else placebo_differentials[r["model"]][r["case_id"]]
        for _, r in frame.iterrows()
    ]

    per_model: dict[str, Any] = {}
    for model in MODELS:
        events = [
            float(v)
            for v in frame.loc[
                (frame["model"] == model) & (frame["case_type"] == "event"),
                "anticipation_z",
            ]
        ]
        placebo_scores = [
            float(v)
            for v in frame.loc[
                (frame["model"] == model) & (frame["case_type"] == "placebo"),
                "anticipation_z",
            ]
        ]
        event_diffs = [differentials[model][cid] for cid in event_ids]
        placebo_diffs = [placebo_differentials[model][cid] for cid in placebo_ids]
        per_model[model] = {
            "events_mean_z": float(np.mean(events)),
            "events_std_z": float(np.std(events, ddof=STD_DDOF)),
            "events_n": len(events),
            "placebo_mean_z": float(np.mean(placebo_scores)),
            "placebo_std_z": float(np.std(placebo_scores, ddof=STD_DDOF)),
            "placebo_n": len(placebo_scores),
            "welch_events_vs_placebos": welch(events, placebo_scores),
            "mean_differential_events": float(np.mean(event_diffs)),
            "std_differential_events": float(np.std(event_diffs, ddof=STD_DDOF)),
            "mean_differential_placebos": float(np.mean(placebo_diffs)),
        }

    per_event_table = {
        case_id: {
            **{model: differentials[model][case_id] + event_mean_z[case_id]
               for model in MODELS},
            "four_model_mean_z": event_mean_z[case_id],
        }
        for case_id in event_ids
    }

    summary = {
        "spec": build_spec(args.date, kept, dropped, placebos, fallbacks),
        "per_model": per_model,
        "per_event_z": per_event_table,
        "per_event_differential": {
            model: differentials[model] for model in MODELS
        },
        "per_placebo_z": {
            case_id: {
                model: float(
                    placebo_frame.loc[
                        (placebo_frame["model"] == model)
                        & (placebo_frame["case_id"] == case_id),
                        "anticipation_z",
                    ].iloc[0]
                )
                for model in MODELS
            }
            for case_id in placebo_ids
        },
        "cases": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "anticipation_screen.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    csv_columns = [
        "model",
        "case_type",
        "case_id",
        "event",
        "country",
        "canonical_country",
        "shock_month",
        "baseline_mean",
        "baseline_std_used",
        "baseline_std_fallback",
        "pre_shock_mean_composite",
        "anticipation_z",
        "clipped",
        "differential_z",
    ]
    csv_frame = frame[csv_columns].sort_values(
        ["case_type", "case_id", "model"], kind="mergesort"
    )
    csv_frame.to_csv(args.output_dir / "anticipation_scores.csv", index=False)

    print(f"Kept events: {len(kept)}; dropped: {len(dropped)}; placebos: {len(placebos)}")
    print(f"Degenerate-std fallbacks fired: {len(fallbacks)}")
    print()
    print(f"{'model':24} {'events':>9} {'placebo':>9} {'t':>8} {'p':>8} {'mean diff':>10}")
    for model in MODELS:
        block = per_model[model]
        print(
            f"{model:24} {block['events_mean_z']:9.4f} "
            f"{block['placebo_mean_z']:9.4f} "
            f"{block['welch_events_vs_placebos']['t_statistic']:8.4f} "
            f"{block['welch_events_vs_placebos']['p_value']:8.4f} "
            f"{block['mean_differential_events']:10.4f}"
        )
    print()
    print(f"Wrote {args.output_dir / 'anticipation_screen.json'}")
    print(f"Wrote {args.output_dir / 'anticipation_scores.csv'}")


if __name__ == "__main__":
    main()
