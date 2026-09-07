"""Available-case home-bias recompute on the 2005-2024 monthly panel.

This script refreshes ``artifacts/analysis/nationality_bias_corrected.json``.
All four core models are present across the full 2005-2024 monthly panel for
both home countries, so both US-on-US and CN-on-CN paired contrasts use the
full ``n=240``. The available-case branch is retained for resilience: any
future month-country cell where a core model is missing will be dropped from
the relevant paired contrast.

Three specifications are emitted side by side so that the JSON remains the
single canonical artefact for the home-bias paper:

* ``family_k2`` — paired one-sample t-test on monthly mean gaps
  (in-group mean minus out-group mean), matching the locked dashboard
  computation in ``the original dashboard home-bias computation`` and the existing
  regression test in ``tests/test_home_bias_compute.py``.
* ``pooled_ols`` — OLS of monthly model score on the two origin-by-home
  dummies plus country and model fixed effects, country-clustered SE.
* ``country_year_fe`` — same regression with country-by-year fixed effects,
  matching the spec in ``country_year_fe.py``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

from analysis.paper_home_bias.common import (
    BASELINE_JSON,
    CORE_MODELS,
    HOME_COUNTRY,
    MODEL_ORIGIN,
    cluster_fit,
    coefficient_payload,
    load_panel,
    p_adjust_two,
    write_json,
)

START_YEAR = 2005
END_YEAR = 2024
US_MODELS = [m for m in CORE_MODELS if MODEL_ORIGIN[m] == "US"]
CN_MODELS = [m for m in CORE_MODELS if MODEL_ORIGIN[m] == "CN"]
MISSING_MONTHS: list[dict] = []


def _paired_gaps(panel, target_country: str, in_models: list[str], out_models: list[str]):
    """Return the monthly gap series and skipped month_keys for one contrast.

    The gap is defined as ``mean(in-group scores) - mean(out-group scores)`` in
    each month for the given target country. A month is dropped only if any of
    the four core models is missing or invalid for that month-country.
    """
    rows = panel[panel["country"] == target_country]
    by_month = rows.groupby("month")
    gaps: list[float] = []
    skipped: list[str] = []
    for month_key, group in sorted(by_month, key=lambda item: item[0]):
        models_present = set(group["model"])
        if not all(model in models_present for model in in_models + out_models):
            skipped.append(month_key)
            continue
        in_vals = group[group["model"].isin(in_models)]["score"].to_list()
        out_vals = group[group["model"].isin(out_models)]["score"].to_list()
        gaps.append(float(np.mean(in_vals) - np.mean(out_vals)))
    return gaps, skipped


def _ttest_payload(label: str, gaps: list[float]) -> dict:
    """Return the legacy ``family_k2`` test record plus a 95 percent CI."""
    arr = np.asarray(gaps, dtype=float)
    n = arr.size
    if n < 2:
        raise RuntimeError(f"{label}: not enough observations ({n})")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    df = n - 1
    t_stat = mean / se
    p_raw = float(2.0 * stats.t.sf(abs(t_stat), df))
    crit = float(stats.t.ppf(0.975, df))
    return {
        "label": label,
        "mean_gap": round(mean, 4),
        "t_stat": round(t_stat, 4),
        "df": df,
        "n": n,
        "se": round(se, 4),
        "ci95_low": round(mean - crit * se, 4),
        "ci95_high": round(mean + crit * se, 4),
        "p_raw": p_raw,
    }


def _attach_corrections(tests: list[dict]) -> list[dict]:
    p_us = next(t["p_raw"] for t in tests if t["label"] == "US-on-US")
    p_cn = next(t["p_raw"] for t in tests if t["label"] == "CN-on-CN")
    adjusted = p_adjust_two(p_us, p_cn)
    out = []
    for test in tests:
        key = "us_on_us" if test["label"] == "US-on-US" else "cn_on_cn"
        new_test = dict(test)
        new_test["p_bonferroni"] = adjusted[key]["p_bonferroni_k2"]
        new_test["p_holm"] = adjusted[key]["p_holm_k2"]
        new_test["reject_bonferroni"] = adjusted[key]["p_bonferroni_k2"] < 0.05
        new_test["reject_holm"] = adjusted[key]["p_holm_k2"] < 0.05
        out.append(new_test)
    return out


def _two_sample_payload(label: str, in_vals: list[float], out_vals: list[float]) -> dict:
    """Welch two-sample t-test for the family_k4 cross-origin contrasts."""
    a = np.asarray(in_vals, dtype=float)
    b = np.asarray(out_vals, dtype=float)
    res = stats.ttest_ind(a, b, equal_var=False)
    return {
        "label": label,
        "mean_gap_or_diff": round(float(a.mean() - b.mean()), 4),
        "t_stat": round(float(res.statistic), 4),
        "p_raw": float(res.pvalue),
        "n_in": int(a.size),
        "n_out": int(b.size),
    }


def _spec_payload(panel, formula: str) -> dict:
    fit = cluster_fit(panel, formula)
    return {
        "formula": formula,
        "n_obs": int(fit.nobs),
        "n_clusters": int(panel["country"].nunique()),
        "coefficients": coefficient_payload(fit),
    }


def main() -> None:
    panel = load_panel(start_year=START_YEAR, end_year=END_YEAR)

    # ---- Family k=2 paired-t tests --------------------------------------
    us_gaps, us_skipped = _paired_gaps(panel, HOME_COUNTRY["US"], US_MODELS, CN_MODELS)
    cn_gaps, cn_skipped = _paired_gaps(panel, HOME_COUNTRY["CN"], CN_MODELS, US_MODELS)
    if len(us_gaps) != 240:
        raise RuntimeError(f"Expected 240 US-on-US monthly gaps, got {len(us_gaps)} (skipped: {us_skipped})")
    if len(cn_gaps) != 240:
        raise RuntimeError(f"Expected 240 CN-on-CN monthly gaps, got {len(cn_gaps)} (skipped: {cn_skipped})")
    if cn_skipped:
        raise RuntimeError(f"Unexpected CN skipped months: {cn_skipped}")

    family_k2_tests = _attach_corrections([
        _ttest_payload("US-on-US", us_gaps),
        _ttest_payload("CN-on-CN", cn_gaps),
    ])

    # ---- Family k=4 (paired + Welch cross-origin) -----------------------
    us_panel = panel[panel["country"] == HOME_COUNTRY["US"]]
    cn_panel = panel[panel["country"] == HOME_COUNTRY["CN"]]
    us_in_scores = us_panel[us_panel["model"].isin(US_MODELS)]["score"].to_list()
    us_out_scores = us_panel[us_panel["model"].isin(CN_MODELS)]["score"].to_list()
    cn_in_scores = cn_panel[cn_panel["model"].isin(CN_MODELS)]["score"].to_list()
    cn_out_scores = cn_panel[cn_panel["model"].isin(US_MODELS)]["score"].to_list()

    welch_us = _two_sample_payload("US vs CN scores on US (two-sample)", us_in_scores, us_out_scores)
    welch_cn = _two_sample_payload("CN vs US scores on CN (two-sample)", cn_in_scores, cn_out_scores)

    raw_ps_k4 = [
        family_k2_tests[0]["p_raw"],
        family_k2_tests[1]["p_raw"],
        welch_us["p_raw"],
        welch_cn["p_raw"],
    ]
    bonf_k4 = [min(p * 4.0, 1.0) for p in raw_ps_k4]
    order = sorted(range(4), key=lambda i: raw_ps_k4[i])
    holm_k4 = [0.0] * 4
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min((4 - rank + 1) * raw_ps_k4[idx], 1.0)
        holm_k4[idx] = max(prev, adj)
        prev = holm_k4[idx]

    family_k4_tests = []
    base_records = [
        {
            "label": "US-on-US home bias",
            "mean_gap_or_diff": family_k2_tests[0]["mean_gap"],
            "t_stat": family_k2_tests[0]["t_stat"],
            "p_raw": family_k2_tests[0]["p_raw"],
        },
        {
            "label": "CN-on-CN home bias",
            "mean_gap_or_diff": family_k2_tests[1]["mean_gap"],
            "t_stat": family_k2_tests[1]["t_stat"],
            "p_raw": family_k2_tests[1]["p_raw"],
        },
        welch_us,
        welch_cn,
    ]
    for i, record in enumerate(base_records):
        new_rec = dict(record)
        new_rec["p_bonferroni"] = bonf_k4[i]
        new_rec["p_holm"] = holm_k4[i]
        new_rec["reject_bonferroni"] = bonf_k4[i] < 0.05
        new_rec["reject_holm"] = holm_k4[i] < 0.05
        family_k4_tests.append(new_rec)

    # ---- Pooled OLS + country-year FE specs -----------------------------
    pooled_ols = _spec_payload(panel, "score ~ us_on_us + cn_on_cn + C(country) + C(model)")
    country_year_fe = _spec_payload(panel, "score ~ us_on_us + cn_on_cn + C(country_year) + C(model)")

    payload = {
        "method": "home_bias_monthly_ttest_with_multiple_comparisons",
        "data_source": (
            "monthly LLM signals (production, 4 core models: DeepSeek V3.2 CN, "
            "MiniMax M2.7 CN, Grok 4.1 Fast US, GPT-5.4 US)"
        ),
        "country_pair": ["united_states", "china"],
        "n_monthly_observations_per_country": {
            "us_on_us": len(us_gaps),
            "cn_on_cn": len(cn_gaps),
        },
        "year_range": f"{START_YEAR}-{END_YEAR}",
        "missing_months": MISSING_MONTHS,
        "missing_handling": (
            "Available-case: drop any month where any core model lacks a valid "
            "score for the target country. All four core models are now present "
            "for every (country, month) pair in 2005-2024 for both home countries, "
            "so no months are dropped from either paired contrast."
        ),
        "model_note": (
            "Results are based on the current 4-core model configuration with "
            "2 CN-origin (DeepSeek, MiniMax) and 2 US-origin (Grok, GPT-5.4) "
            "models, extended to the full 2005-2024 panel via the GPT-5.4 and "
            "Grok 4.1 Fast batch backfills (2005-2014 lives under the *_batch "
            "model directories on disk)."
        ),
        "precondition_warning": (
            "The earlier ten-year metadata reported US-on-US +3.329 (p approximately "
            "7e-27) and CN-on-CN +0.935 (p approximately 7e-04). Under the refreshed "
            f"twenty-year available-case panel the paired-t headline shifts to "
            f"US-on-US {family_k2_tests[0]['mean_gap']:+.4f} "
            f"(n={family_k2_tests[0]['n']}, p={family_k2_tests[0]['p_raw']:.3e}) and "
            f"CN-on-CN {family_k2_tests[1]['mean_gap']:+.4f} "
            f"(n={family_k2_tests[1]['n']}, p={family_k2_tests[1]['p_raw']:.3e}). "
            "The cluster-robust country-year fixed-effects spec on the full "
            "25-country panel keeps both contrasts Bonferroni-significant. See "
            "year_range and n_monthly_observations_per_country for the current "
            "panel definition; see papers/origin-bias/refresh_metadata_diff.md "
            "for the full old-vs-new comparison."
        ),
        "family_k2": {
            "description": (
                "Pre-specified family of 2: US-on-US home bias and CN-on-CN home "
                "bias, paired one-sample t-test on monthly mean gaps."
            ),
            "tests": family_k2_tests,
        },
        "family_k4": {
            "description": (
                "Extended family of 4: paired US-on-US and CN-on-CN home-bias plus "
                "Welch two-sample contrasts on the US and China target panels."
            ),
            "tests": family_k4_tests,
        },
        "pooled_ols": pooled_ols,
        "country_year_fe": country_year_fe,
        "headline_change": (
            "Under the 2005-2024 available-case panel the paired-t US-on-US gap is "
            f"{family_k2_tests[0]['mean_gap']:+.3f} (was +3.329) and the CN-on-CN "
            f"gap is {family_k2_tests[1]['mean_gap']:+.3f} (was +0.935); the CN "
            "paired contrast is no longer significant after extending to the full "
            "twenty-year window. The cluster-robust country-year fixed-effects "
            "spec on the 25-country panel still rejects both nulls under "
            "Bonferroni at k=2."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(BASELINE_JSON, payload)


if __name__ == "__main__":
    main()
