#!/usr/bin/env python3
"""Relate Wikipedia coverage density to canonical raw country-level MAE.

This analysis deliberately uses raw composite MAE from the retained country
difficulty artefact. It does not use nowcast skill gain relative to persistence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from scipy.stats import t as student_t


REPO = Path(__file__).resolve().parents[2]
DIFFICULTY_PATH = (
    REPO
    / "results"
    / "canonical"
    / "analysis"
    / "country_difficulty"
    / "country_difficulty_decomposition.json"
)
EVENT_REFERENCE_PATH = REPO / "data" / "external" / "event_coverage_panel_2026-05-23.json"
OUTPUT_DIR = REPO / "results" / "methods_evidence" / "coverage_difficulty"
OUTPUT_JSON = OUTPUT_DIR / "coverage_difficulty.json"
OUTPUT_CSV = OUTPUT_DIR / "coverage_difficulty.csv"
OUTPUT_README = OUTPUT_DIR / "README.md"

YEAR_MIN = 2005
YEAR_MAX = 2024
PAGEVIEW_START = "2015070100"
PAGEVIEW_END = "2024123100"
PAGEVIEW_ENDPOINT = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/monthly/{start}/{end}"
)
USER_AGENT = "prs-nowcast-coverage-difficulty/1.0 (academic research)"
SEED = 42
BOOTSTRAP_DRAWS = 20_000


def _default_input_root() -> Path:
    candidates = [
        REPO.parent / "monthly-llm-risk-signals" / "data" / "input",
        REPO.parents[1] / "repos" / "monthly-llm-risk-signals" / "data" / "input",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_file(input_root: Path, country: str, year: int, month: int, source: str) -> Path:
    filename = f"{country}_{year}_{month:02d}_{source}_filtered.json"
    flat = input_root / country / filename
    if flat.exists():
        return flat
    return input_root / country / str(year) / filename


def _load_event_list(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of event objects: {path}")
    return payload


def read_single_source_event_coverage(
    input_root: Path,
    countries: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Count the Portal-first bundles used by EVENTS_SINGLE_SOURCE=1.

    Portal is selected when its file exists. Year-in-Country is used only when
    the Portal file is absent. Empty selected files remain empty cells.
    """

    rows: dict[str, dict[str, Any]] = {}
    manifest = hashlib.sha256()
    for country in sorted(countries):
        total_events = 0
        empty_cells = 0
        portal_cells = 0
        yic_cells = 0
        selected_files = 0
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            for month in range(1, 13):
                portal = _event_file(input_root, country, year, month, "portal")
                yic = _event_file(input_root, country, year, month, "yic")
                selected: Path | None
                if portal.exists():
                    selected = portal
                    portal_cells += 1
                elif yic.exists():
                    selected = yic
                    yic_cells += 1
                else:
                    selected = None

                if selected is None:
                    empty_cells += 1
                    manifest.update(f"{country}/{year}/{month:02d}:missing\n".encode())
                    continue

                selected_files += 1
                raw = selected.read_bytes()
                events = _load_event_list(selected)
                event_count = len(events)
                total_events += event_count
                if event_count == 0:
                    empty_cells += 1
                relative = selected.relative_to(input_root)
                manifest.update(str(relative).encode())
                manifest.update(b"\0")
                manifest.update(hashlib.sha256(raw).digest())
                manifest.update(b"\n")

        rows[country] = {
            "event_count": total_events,
            "event_cells": 240,
            "event_selected_files": selected_files,
            "event_empty_cells": empty_cells,
            "event_portal_cells": portal_cells,
            "event_yic_fallback_cells": yic_cells,
            "event_mean_per_cell": total_events / 240.0,
        }
    return rows, manifest.hexdigest()


def reconcile_event_counts(
    observed: dict[str, dict[str, Any]],
    reference_path: Path = EVENT_REFERENCE_PATH,
) -> dict[str, Any]:
    reference = _read_json(reference_path)
    expected = {str(row["country"]): row for row in reference["per_country"]}
    missing = sorted(set(expected) ^ set(observed))
    mismatches: list[dict[str, Any]] = []
    for country in sorted(set(expected) & set(observed)):
        actual_count = int(observed[country]["event_count"])
        expected_count = int(expected[country]["total_events_sent"])
        actual_empty = int(observed[country]["event_empty_cells"])
        expected_empty = int(expected[country]["empty_cells"])
        if actual_count != expected_count or actual_empty != expected_empty:
            mismatches.append(
                {
                    "country": country,
                    "observed_event_count": actual_count,
                    "reference_event_count": expected_count,
                    "observed_empty_cells": actual_empty,
                    "reference_empty_cells": expected_empty,
                }
            )
    return {
        "status": "CONFIRMED" if not missing and not mismatches else "BLOCKED",
        "reference": str(reference_path.relative_to(REPO)),
        "missing_or_extra_countries": missing,
        "mismatches": mismatches,
        "observed_total_events": sum(int(row["event_count"]) for row in observed.values()),
        "reference_total_events": int(reference["total_events_sent_cell_level"]),
    }


def fetch_pageviews(article: str, *, retries: int = 3) -> dict[str, Any]:
    encoded = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = PAGEVIEW_ENDPOINT.format(
        article=encoded,
        start=PAGEVIEW_START,
        end=PAGEVIEW_END,
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Wikimedia response has no items list for {article}")
            views = [int(item["views"]) for item in items]
            return {
                "article": article,
                "status": "CONFIRMED",
                "endpoint": url,
                "months_returned": len(views),
                "pageviews_total": int(sum(views)),
                "pageviews_mean_monthly": float(np.mean(views)),
                "first_timestamp": items[0]["timestamp"] if items else None,
                "last_timestamp": items[-1]["timestamp"] if items else None,
            }
        except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
            error = exc
            if attempt < retries:
                retry_after = None
                if isinstance(exc, urllib.error.HTTPError):
                    retry_after = exc.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 10.0 * attempt
                )
                time.sleep(delay)
    return {
        "article": article,
        "status": "BLOCKED",
        "endpoint": url,
        "error": str(error),
    }


def _corr_value(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if kind == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if kind == "spearman":
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    raise ValueError(f"Unknown correlation kind: {kind}")


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(x), size=len(x))
        value = _corr_value(x[index], y[index], kind)
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < draws * 0.95:
        raise RuntimeError(
            f"Too few finite {kind} bootstrap draws: {len(estimates)}/{draws}"
        )
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return float(low), float(high), len(estimates)


def correlation_report(
    rows: list[dict[str, Any]],
    x_key: str,
    *,
    y_key: str = "raw_composite_mae",
    seed_offset: int,
) -> dict[str, Any]:
    x = np.asarray([float(row[x_key]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    pearson_low, pearson_high, pearson_draws = bootstrap_ci(
        x, y, "pearson", seed=SEED + seed_offset
    )
    spearman_low, spearman_high, spearman_draws = bootstrap_ci(
        x, y, "spearman", seed=SEED + seed_offset + 1
    )
    return {
        "x": x_key,
        "y": y_key,
        "n": len(rows),
        "pearson": {
            "r": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
            "ci_95": [pearson_low, pearson_high],
            "ci_method": "paired non-parametric percentile bootstrap",
            "bootstrap_draws_finite": pearson_draws,
        },
        "spearman": {
            "rho": float(spearman.statistic),
            "p_value": float(spearman.pvalue),
            "ci_95": [spearman_low, spearman_high],
            "ci_method": "paired non-parametric percentile bootstrap",
            "bootstrap_draws_finite": spearman_draws,
        },
    }


def _residuals(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(control), control])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _partial_value(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    return float(
        np.corrcoef(_residuals(x, control), _residuals(y, control))[0, 1]
    )


def bootstrap_partial_ci(
    x: np.ndarray,
    y: np.ndarray,
    control: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(x), size=len(x))
        value = _partial_value(x[index], y[index], control[index])
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < draws * 0.95:
        raise RuntimeError(
            f"Too few finite partial-correlation bootstrap draws: {len(estimates)}/{draws}"
        )
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return float(low), float(high), len(estimates)


def _two_sided_t_p(r: float, df: int) -> float:
    t_statistic = r * math.sqrt(df / (1.0 - r * r))
    return float(2.0 * student_t.sf(abs(t_statistic), df))


def partial_correlation_report(
    rows: list[dict[str, Any]],
    x_key: str,
    control_key: str,
    *,
    y_key: str = "raw_composite_mae",
    seed_offset: int,
) -> dict[str, Any]:
    """Linear-residualisation partial correlation of x and y given one control."""

    x = np.asarray([float(row[x_key]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    control = np.asarray([float(row[control_key]) for row in rows])
    r = _partial_value(x, y, control)
    n = len(rows)
    df = n - 3
    low, high, finite_draws = bootstrap_partial_ci(
        x, y, control, seed=SEED + seed_offset
    )
    return {
        "x": x_key,
        "y": y_key,
        "control": control_key,
        "n": n,
        "method": (
            "Pearson correlation of the residuals of x and y after separate "
            "ordinary-least-squares fits on the control."
        ),
        "r": r,
        "p_value": _two_sided_t_p(r, df),
        "df": df,
        "p_value_method": (
            "Two-sided t test on the residual correlation with df = n - 2 - 1 "
            "for the single control."
        ),
        "p_value_df_n_minus_2": _two_sided_t_p(r, n - 2),
        "p_value_df_n_minus_2_note": (
            "Reported because treating the residual correlation as an ordinary "
            "n-point correlation spends no degrees of freedom on the control and "
            "returns a slightly smaller p-value."
        ),
        "ci_95": [low, high],
        "ci_method": "paired non-parametric percentile bootstrap",
        "bootstrap_draws_finite": finite_draws,
    }


def _ci_excludes_zero(ci: list[float] | tuple[float, float]) -> bool:
    return ci[0] > 0.0 or ci[1] < 0.0


def _ci_direction(report: dict[str, Any]) -> str:
    pearson_ci = report["pearson"]["ci_95"]
    spearman_ci = report["spearman"]["ci_95"]
    if pearson_ci[0] > 0.0 and spearman_ci[0] > 0.0:
        return "positive"
    if pearson_ci[1] < 0.0 and spearman_ci[1] < 0.0:
        return "negative"
    return "indistinguishable_from_zero"


def _load_difficulty_rows() -> list[dict[str, Any]]:
    payload = _read_json(DIFFICULTY_PATH)
    if int(payload["n_countries"]) != 25:
        raise RuntimeError("Difficulty artefact is not the canonical 25-country roster")
    countries = payload["countries"]
    ukraine = next(row for row in countries if row["slug"] == "ukraine")
    if not math.isclose(float(ukraine["mean_mae_all_models"]), 7.1391, abs_tol=1e-12):
        raise RuntimeError(
            "Ukraine does not reconcile to canonical raw composite MAE 7.1391; "
            "the wrong estimand or artefact was selected"
        )
    return countries


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "country",
        "country_label",
        "raw_composite_mae",
        "difficulty_start_year",
        "difficulty_n_years_gt",
        "ground_truth_volatility",
        "event_count",
        "event_mean_per_cell",
        "event_empty_cells",
        "pageview_article",
        "pageviews_total",
        "pageviews_mean_monthly",
        "pageview_months",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _ci(bounds: list[float]) -> str:
    return f"[{_fmt(bounds[0])}, {_fmt(bounds[1])}]"


def _corr_rows(label: str, report: dict[str, Any]) -> list[str]:
    return [
        f"| {label} | Pearson r | {_fmt(report['pearson']['r'])} | "
        f"{_ci(report['pearson']['ci_95'])} | {_fmt(report['pearson']['p_value'])} |",
        f"| {label} | Spearman rho | {_fmt(report['spearman']['rho'])} | "
        f"{_ci(report['spearman']['ci_95'])} | {_fmt(report['spearman']['p_value'])} |",
    ]


def build_readme(payload: dict[str, Any]) -> str:
    correlations = payload["correlations"]
    collinearity = payload["collinearity"]
    partials = payload["partial_correlations"]
    window = payload["window_sensitivity"]
    verdict = payload["verdict"]
    event_partial = partials["event_count_vs_raw_mae_given_volatility"]
    volatility_partial = partials["volatility_vs_raw_mae_given_event_count"]
    matched = window["event_count_vs_raw_mae_matched_window"]
    reference = window["event_count_vs_raw_mae_all_25_reference"]

    header = "| Pair | Statistic | Estimate | 95% CI | Parametric p |"
    rule = "|---|---|---:|---:|---:|"

    main_rows: list[str] = []
    main_rows += _corr_rows(
        "Selected Portal-first event count, 2005-2024, versus raw composite MAE",
        correlations["event_count_vs_raw_mae"],
    )
    if "pageviews_vs_raw_mae" in correlations:
        main_rows += _corr_rows(
            "English Wikipedia pageviews, Jul 2015 to Dec 2024, versus raw composite MAE",
            correlations["pageviews_vs_raw_mae"],
        )
    main_rows += _corr_rows(
        "Annual ground-truth PRS volatility versus raw composite MAE",
        correlations["ground_truth_volatility_vs_raw_mae"],
    )

    collinearity_rows = _corr_rows(
        "Event count versus ground-truth volatility",
        collinearity["event_count_vs_volatility"],
    )
    if "pageviews_vs_volatility" in collinearity:
        collinearity_rows += _corr_rows(
            "Pageviews versus ground-truth volatility",
            collinearity["pageviews_vs_volatility"],
        )

    excluded = ", ".join(window["excluded_countries"])

    lines = [
        "# Wikipedia coverage, ground-truth volatility and raw country difficulty",
        "",
        "Does thin Wikipedia coverage explain why some countries are harder to nowcast?",
        "The answer on the canonical 25-country roster is no, and this file records both",
        "the original test and the follow-up that separates coverage from volatility.",
        "",
        f"Overall status: **{payload['status']}**. "
        "The Wikimedia Pageviews API begins in July 2015 and cannot cover the full",
        "2005-2024 panel, so the pageview measure is a part-window proxy.",
        "",
        "## Estimand guard",
        "",
        payload["estimand_guard"],
        "",
        "## Coverage, volatility and difficulty",
        "",
        f"All correlations below use {correlations['event_count_vs_raw_mae']['n']} countries. "
        "Confidence intervals are paired",
        f"non-parametric percentile bootstrap intervals from "
        f"{payload['spec']['bootstrap_draws_requested']:,} draws with deterministic",
        f"seeds anchored on {payload['spec']['bootstrap_seed']}. "
        "Full precision lives in `coverage_difficulty.json`.",
        "",
        header,
        rule,
        *main_rows,
        "",
        "Event volume runs the wrong way for the scarcity story: countries with more",
        "stored evidence have higher raw MAE, not lower. Pageviews are flat. Volatility",
        "is positively rank-correlated with difficulty.",
        "",
        "## Are coverage and volatility separable?",
        "",
        "The first version of this analysis framed coverage and volatility as competing",
        "explanations without testing whether they move together.",
        "",
        header,
        rule,
        *collinearity_rows,
        "",
        "Every interval spans zero, so coverage volume and ground-truth volatility are",
        "statistically unrelated across these countries. They are separate candidate",
        "explanations rather than two readings of one underlying quantity.",
        "",
        "## Partial correlations",
        "",
        "Each predictor is residualised on the other using an ordinary-least-squares fit,",
        "then correlated with the residualised outcome. The reported p-value uses",
        f"df = n - 2 - 1 = {event_partial['df']} for the single control.",
        "",
        "| Partial correlation | r | 95% CI | p |",
        "|---|---:|---:|---:|",
        f"| Event count versus raw MAE, controlling volatility | {_fmt(event_partial['r'])} | "
        f"{_ci(event_partial['ci_95'])} | {_fmt(event_partial['p_value'])} |",
        f"| Volatility versus raw MAE, controlling event count | {_fmt(volatility_partial['r'])} | "
        f"{_ci(volatility_partial['ci_95'])} | {_fmt(volatility_partial['p_value'])} |",
        "",
        "Because the two predictors are uncorrelated, controlling for one hardly moves the",
        "other: each partial estimate sits slightly above its raw counterpart. Event volume",
        "survives the control on both inference paths.",
        "",
        "Volatility is the weaker and less stable of the two. Its bootstrap interval excludes",
        "zero by a small margin while the t test does not reject at the 5% level, so the two",
        "inference paths disagree. The verdict block records that disagreement rather than",
        "resolving it, and the same disagreement is already visible in the raw volatility",
        "Pearson statistic above. An independent role for volatility should therefore be",
        "treated as unresolved at this sample size, not as established.",
        "",
        "## Window-mismatch sensitivity",
        "",
        window["issue"],
        "",
        f"Countries dropped for the matched-window check: {excluded}.",
        "",
        "| Sample | Pearson r | Pearson p | Spearman rho | Spearman p |",
        "|---|---:|---:|---:|---:|",
        f"| All {reference['n']} countries | {_fmt(reference['pearson_r'])} | "
        f"{_fmt(reference['pearson_p_value'])} | {_fmt(reference['spearman_rho'])} | "
        f"{_fmt(reference['spearman_p_value'])} |",
        f"| Matched window, {matched['n']} countries | {_fmt(matched['pearson']['r'])} | "
        f"{_fmt(matched['pearson']['p_value'])} | {_fmt(matched['spearman']['rho'])} | "
        f"{_fmt(matched['spearman']['p_value'])} |",
        "",
        "The positive association is not an artefact of the window mismatch. Dropping the",
        "five part-window countries leaves the estimate essentially unchanged, at",
        f"Pearson r {_fmt(matched['pearson']['r'])} with CI {_ci(matched['pearson']['ci_95'])} and Spearman rho "
        f"{_fmt(matched['spearman']['rho'])} with CI {_ci(matched['spearman']['ci_95'])}. Both intervals still",
        "exclude zero, although the rank interval only just does on the smaller sample.",
        "",
        "## Verdict",
        "",
        "| Field | Value |",
        "|---|---|",
        "| coverage_scarcity_hypothesis_supported | "
        f"{str(verdict['coverage_scarcity_hypothesis_supported']).lower()} |",
        f"| event_count_vs_raw_mae_direction | {verdict['event_count_vs_raw_mae_direction']} |",
        f"| pageviews_vs_raw_mae_direction | {verdict['pageviews_vs_raw_mae_direction']} |",
        "| event_volume_independently_predicts_difficulty | "
        f"{str(verdict['event_volume_independently_predicts_difficulty']).lower()} |",
        "| event_volume_independence_ci_and_t_test_agree | "
        f"{str(verdict['event_volume_independence_ci_and_t_test_agree']).lower()} |",
        "| volatility_independently_predicts_difficulty | "
        f"{str(verdict['volatility_independently_predicts_difficulty']).lower()} |",
        "| volatility_independence_ci_and_t_test_agree | "
        f"{str(verdict['volatility_independence_ci_and_t_test_agree']).lower()} |",
        "| event_volume_and_volatility_collinear | "
        f"{str(verdict['event_volume_and_volatility_collinear']).lower()} |",
        "| headline_survives_window_matching | "
        f"{str(verdict['headline_survives_window_matching']).lower()} |",
        "",
        "## What this means for the mechanism question",
        "",
        "Thin Wikipedia coverage does not explain per-country nowcast difficulty. The",
        "association runs in the opposite direction to the scarcity hypothesis, and it is",
        "not a disguised volatility effect, because the two predictors are uncorrelated and",
        "event volume keeps its association after volatility is partialled out. Volatility",
        "remains a plausible second driver, but its independent contribution is marginal and",
        "inference-path dependent at n = 25, so it should not be reported as an established",
        "effect on this evidence.",
        "",
        "The likeliest reading is that event volume is a marker of eventfulness rather than",
        "of evidence quality: countries that generate more recorded events are the countries",
        "whose risk ratings move, and moving ratings are harder to track. That reading is",
        "consistent with the numbers here but is not itself tested by them, so it should be",
        "carried as an interpretation and not as a result.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m analysis.coverage_difficulty.run_coverage_difficulty",
        "```",
        "",
        "The run is deterministic. Pageview totals are reused from the stored artefact when",
        "present, so a rerun needs no network access. Outputs are `coverage_difficulty.json`",
        "(full precision), `coverage_difficulty.csv` (one row per country), and this file.",
        "",
    ]
    return "\n".join(lines)


def run(input_root: Path, *, fetch_pageview_data: bool = True) -> dict[str, Any]:
    difficulty_rows = _load_difficulty_rows()
    countries = [str(row["slug"]) for row in difficulty_rows]
    event_rows, input_manifest_sha256 = read_single_source_event_coverage(
        input_root, countries
    )
    event_reconciliation = reconcile_event_counts(event_rows)
    if event_reconciliation["status"] != "CONFIRMED":
        raise RuntimeError(
            "Parked evidence-bundle counts do not reconcile to the canonical "
            f"single-source manifest: {event_reconciliation}"
        )

    cached_pageviews: dict[str, dict[str, Any]] = {}
    if OUTPUT_JSON.exists():
        previous = _read_json(OUTPUT_JSON)
        cached_pageviews = {
            str(country): row
            for country, row in previous.get("pageview_fetch", {})
            .get("records", {})
            .items()
            if row.get("status") == "CONFIRMED"
        }

    pageview_rows: dict[str, dict[str, Any]] = {}
    if fetch_pageview_data:
        for row in sorted(difficulty_rows, key=lambda item: str(item["slug"])):
            country = str(row["slug"])
            if country in cached_pageviews:
                pageview_rows[country] = cached_pageviews[country]
                continue
            article = str(row["label"])
            pageview_rows[country] = fetch_pageviews(article)
            time.sleep(2.0)
    else:
        pageview_rows = {
            country: {
                "article": next(
                    str(row["label"]) for row in difficulty_rows if row["slug"] == country
                ),
                "status": "BLOCKED",
                "error": "Pageview fetch disabled by --no-pageviews",
            }
            for country in countries
        }

    blocked_pageviews = {
        country: row
        for country, row in pageview_rows.items()
        if row["status"] != "CONFIRMED"
    }

    combined: list[dict[str, Any]] = []
    for item in sorted(difficulty_rows, key=lambda row: str(row["slug"])):
        country = str(item["slug"])
        pageviews = pageview_rows[country]
        row = {
            "country": country,
            "country_label": str(item["label"]),
            "raw_composite_mae": float(item["mean_mae_all_models"]),
            "difficulty_start_year": int(item["start_year"]),
            "difficulty_n_years_gt": int(item["n_years_gt"]),
            "ground_truth_volatility": float(item["prs_std_annual"]),
            **event_rows[country],
            "pageview_article": str(pageviews["article"]),
            "pageviews_total": (
                int(pageviews["pageviews_total"])
                if pageviews["status"] == "CONFIRMED"
                else None
            ),
            "pageviews_mean_monthly": (
                float(pageviews["pageviews_mean_monthly"])
                if pageviews["status"] == "CONFIRMED"
                else None
            ),
            "pageview_months": (
                int(pageviews["months_returned"])
                if pageviews["status"] == "CONFIRMED"
                else None
            ),
        }
        combined.append(row)

    correlations = {
        "event_count_vs_raw_mae": correlation_report(
            combined, "event_count", seed_offset=10
        ),
        "ground_truth_volatility_vs_raw_mae": correlation_report(
            combined, "ground_truth_volatility", seed_offset=20
        ),
    }
    if not blocked_pageviews:
        correlations["pageviews_vs_raw_mae"] = correlation_report(
            combined, "pageviews_total", seed_offset=30
        )

    collinearity = {
        "purpose": (
            "Coverage volume and ground-truth volatility were framed as competing "
            "explanations of raw difficulty. This block tests whether they are "
            "separable in the first place."
        ),
        "event_count_vs_volatility": correlation_report(
            combined,
            "event_count",
            y_key="ground_truth_volatility",
            seed_offset=40,
        ),
    }
    if not blocked_pageviews:
        collinearity["pageviews_vs_volatility"] = correlation_report(
            combined,
            "pageviews_total",
            y_key="ground_truth_volatility",
            seed_offset=50,
        )

    partial_correlations = {
        "purpose": (
            "Separate the two candidate explanations by residualising each on the "
            "other before correlating it with raw composite MAE."
        ),
        "event_count_vs_raw_mae_given_volatility": partial_correlation_report(
            combined, "event_count", "ground_truth_volatility", seed_offset=60
        ),
        "volatility_vs_raw_mae_given_event_count": partial_correlation_report(
            combined, "ground_truth_volatility", "event_count", seed_offset=70
        ),
    }

    matched_window_rows = [
        row for row in combined if int(row["difficulty_start_year"]) == YEAR_MIN
    ]
    headline = correlations["event_count_vs_raw_mae"]
    matched_window = correlation_report(
        matched_window_rows, "event_count", seed_offset=80
    )
    window_sensitivity = {
        "issue": (
            "Event counts cover 2005-2024 for every country, but the difficulty "
            "artefact measures MAE over 2015-2024 for five countries, so the "
            "headline all-25 correlation pairs a full-window predictor with a "
            "part-window outcome for those five."
        ),
        "matched_window_countries": [
            str(row["country"]) for row in matched_window_rows
        ],
        "excluded_countries": [
            str(row["country"])
            for row in combined
            if int(row["difficulty_start_year"]) != YEAR_MIN
        ],
        "event_count_vs_raw_mae_matched_window": matched_window,
        "event_count_vs_raw_mae_all_25_reference": {
            "n": headline["n"],
            "pearson_r": headline["pearson"]["r"],
            "pearson_p_value": headline["pearson"]["p_value"],
            "spearman_rho": headline["spearman"]["rho"],
            "spearman_p_value": headline["spearman"]["p_value"],
        },
    }

    event_ci = correlations["event_count_vs_raw_mae"]["spearman"]["ci_95"]
    volatility_ci = correlations["ground_truth_volatility_vs_raw_mae"]["spearman"][
        "ci_95"
    ]
    pageview_ci = (
        correlations["pageviews_vs_raw_mae"]["spearman"]["ci_95"]
        if "pageviews_vs_raw_mae" in correlations
        else None
    )
    coverage_negative = event_ci[1] < 0 or (
        pageview_ci is not None and pageview_ci[1] < 0
    )
    volatility_null = volatility_ci[0] <= 0 <= volatility_ci[1]

    event_partial = partial_correlations["event_count_vs_raw_mae_given_volatility"]
    volatility_partial = partial_correlations[
        "volatility_vs_raw_mae_given_event_count"
    ]
    collinear = any(
        _ci_excludes_zero(collinearity[key][statistic]["ci_95"])
        for key in collinearity
        if key != "purpose"
        for statistic in ("pearson", "spearman")
    )
    verdict = {
        "coverage_scarcity_hypothesis_supported": bool(
            coverage_negative and not blocked_pageviews
        ),
        "coverage_scarcity_rule": (
            "Supported only if a coverage measure correlates negatively with raw "
            "composite MAE with a 95% bootstrap CI lying entirely below zero."
        ),
        "event_count_vs_raw_mae_direction": _ci_direction(
            correlations["event_count_vs_raw_mae"]
        ),
        "pageviews_vs_raw_mae_direction": (
            _ci_direction(correlations["pageviews_vs_raw_mae"])
            if "pageviews_vs_raw_mae" in correlations
            else None
        ),
        "event_volume_independently_predicts_difficulty": bool(
            _ci_excludes_zero(event_partial["ci_95"])
        ),
        "event_volume_partial_r": event_partial["r"],
        "event_volume_partial_ci_95": event_partial["ci_95"],
        "event_volume_partial_p_value": event_partial["p_value"],
        "event_volume_independence_ci_and_t_test_agree": bool(
            _ci_excludes_zero(event_partial["ci_95"])
            == (event_partial["p_value"] < 0.05)
        ),
        "volatility_independently_predicts_difficulty": bool(
            _ci_excludes_zero(volatility_partial["ci_95"])
        ),
        "volatility_partial_r": volatility_partial["r"],
        "volatility_partial_ci_95": volatility_partial["ci_95"],
        "volatility_partial_p_value": volatility_partial["p_value"],
        "volatility_independence_ci_and_t_test_agree": bool(
            _ci_excludes_zero(volatility_partial["ci_95"])
            == (volatility_partial["p_value"] < 0.05)
        ),
        "independence_rule": (
            "Independent prediction requires the partial-correlation 95% bootstrap "
            "CI, computed after residualising on the rival explanation, to exclude "
            "zero. The agreement fields record whether the two-sided t test at the "
            "5% level reaches the same decision."
        ),
        "event_volume_and_volatility_collinear": bool(collinear),
        "collinearity_rule": (
            "Collinear only if a Pearson or Spearman 95% bootstrap CI between a "
            "coverage measure and ground-truth volatility excludes zero."
        ),
        "headline_survives_window_matching": bool(
            _ci_excludes_zero(matched_window["pearson"]["ci_95"])
            and _ci_excludes_zero(matched_window["spearman"]["ci_95"])
        ),
    }

    by_country = {row["country"]: row for row in combined}
    worked_pair = {
        country: by_country[country]
        for country in ("ukraine", "china")
    }

    status = "APPROXIMATE"
    payload = {
        "status": status,
        "analysis": "Wikipedia coverage density versus raw per-country composite MAE",
        "estimand_guard": (
            "Outcome is the mean across four canonical models' country-level annual "
            "composite MAEs from country_difficulty_decomposition.json. It is not "
            "nowcast skill gain relative to persistence."
        ),
        "spec": {
            "roster": "25 canonical countries",
            "difficulty_source": str(DIFFICULTY_PATH.relative_to(REPO)),
            "difficulty_window": (
                "Artefact-defined per country: 2005-2024 for 20 countries and "
                "2015-2024 for France, Israel, Japan, Kenya and Saudi Arabia."
            ),
            "event_coverage_window": f"{YEAR_MIN}-{YEAR_MAX}",
            "event_coverage_regime": (
                "EVENTS_SINGLE_SOURCE=1: Portal file when present, otherwise "
                "Year-in-Country; raw selected-bundle event count."
            ),
            "event_input_root": str(input_root),
            "event_input_manifest_sha256": input_manifest_sha256,
            "pageviews_definition": (
                "English Wikipedia country-article views, all-access/user, monthly."
            ),
            "pageviews_available_window": "2015-07-01 to 2024-12-31",
            "pageviews_window_limitation": (
                "The Wikimedia Pageviews API begins in July 2015, so pageviews do "
                "not cover the full 2005-2024 panel and are labelled APPROXIMATE "
                "for the proposed full-window mechanism."
            ),
            "bootstrap_seed": SEED,
            "bootstrap_draws_requested": BOOTSTRAP_DRAWS,
        },
        "reconciliation": {
            "ukraine_raw_composite_mae": {
                "expected": 7.1391,
                "observed": by_country["ukraine"]["raw_composite_mae"],
                "status": "CONFIRMED",
            },
            "event_bundle_counts": event_reconciliation,
        },
        "pageview_fetch": {
            "status": "BLOCKED" if blocked_pageviews else "APPROXIMATE",
            "blocked": blocked_pageviews,
            "records": pageview_rows,
        },
        "correlations": correlations,
        "mechanism_test": {
            "claim_tested": "Thinner Wikipedia coverage, not PRS volatility, drives raw MAE.",
            "supported_by_prespecified_rule": bool(
                coverage_negative and volatility_null and not blocked_pageviews
            ),
            "rule": (
                "At least one coverage measure must have a negative Spearman 95% CI "
                "excluding zero, while the volatility Spearman 95% CI includes zero."
            ),
        },
        "collinearity": collinearity,
        "partial_correlations": partial_correlations,
        "window_sensitivity": window_sensitivity,
        "verdict": verdict,
        "ukraine_china_worked_pair": worked_pair,
        "countries": combined,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(combined)
    OUTPUT_README.write_text(build_readme(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Wikipedia coverage density against raw country-level MAE."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_default_input_root(),
        help="Read-only monthly-llm-risk-signals data/input root.",
    )
    parser.add_argument(
        "--no-pageviews",
        action="store_true",
        help="Skip Wikimedia API calls and label the pageview measure BLOCKED.",
    )
    args = parser.parse_args()
    result = run(args.input_root.resolve(), fetch_pageview_data=not args.no_pageviews)
    print(f"status={result['status']}")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_README}")


if __name__ == "__main__":
    main()
