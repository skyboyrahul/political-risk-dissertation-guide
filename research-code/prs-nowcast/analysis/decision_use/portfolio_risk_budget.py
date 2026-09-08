"""Portfolio risk-budget simulation: does a same-year nowcast move the allocation?

STATUS: DECISION-USE ILLUSTRATION (not canonical, not a thesis headline result)

Question
--------
The canonical same-year nowcast beats a carry-forward persistence baseline on
annual PRS change by 8.95% RMSE at cutoff m=12. That is a measurement claim.
This script asks a narrower decision-shaped question: if an analyst had to turn
the 25-country score vector into a long-only risk budget, would the nowcast
scores have produced an allocation closer to the allocation implied by realised
annual PRS than the stale previous-year score would have?

This is an illustrative decision-use simulation. It is NOT a return backtest and
NOT a claim of monetary value. See "Statistical and interpretive boundary".

Design
------
1. Trace the canonical fixed BayesianRidge `delta_prs` nowcast rows for the five
   locked cutoffs m in {3, 6, 9, 11, 12} out of the frozen walk-forward artefact.
   No model is refitted here; the rows are read back exactly as
   `analysis/ml_experiments/nowcast_inference.py` reads them. Before any
   simulation the script reproduces the fold-mean RMSE and persistence RMSE from
   the rows themselves and proves each country-year-cutoff appears exactly once.
2. For every held-out country-year, form three annual PRS *levels* from the same
   previous-year anchor `prs_lag1`:
       level_persistence = prs_lag1
       level_nowcast     = prs_lag1 + predicted delta
       level_realised    = prs_lag1 + realised delta   (identically gt_composite)
3. Convert each 25-country score vector into long-only, fully invested weights
   with the pre-specified softmax rule
       w_i = exp(beta * z_i) / sum_j exp(beta * z_j)
   where z is the cross-sectional standardised score within that year and
   cutoff, so a higher PRS (lower political risk) receives more weight.
   beta = 0.5 is the primary rule; beta in {0.25, 1.0} are sensitivity checks
   only. Both betas were fixed before any result was inspected.
4. Per year and cutoff, compare the nowcast and persistence weights against the
   realised-PRS weights with half the L1 distance,
       reallocation share = 0.5 * sum_i |w_i - w_i^realised|,
   which is the share of the portfolio that would have to be reallocated to
   reach the realised-PRS allocation. Also report turnover from persistence to
   nowcast (same half-L1 metric) and portfolio concentration (HHI, effective
   number of holdings, maximum weight).
5. Inference is paired at the year level on
       d_y = reallocation_persistence(y) - reallocation_nowcast(y),
   so d_y > 0 favours the nowcast. Each cutoff gets a year-block bootstrap
   percentile confidence interval and an exact sign-flip test enumerating all
   2^14 = 16384 sign assignments. The five cutoff tests are corrected together
   with Holm-Bonferroni.

Pre-specified choices (fixed before results were seen)
-----------------------------------------------------
- Primary beta = 0.5; sensitivity betas = 0.25 and 1.0.
- Cross-sectional standardisation uses the population standard deviation
  (ddof=0), because the 25 countries are the entire investable universe of this
  simulation rather than a sample from a larger one. Using ddof=1 instead is
  exactly equivalent to rescaling beta by sqrt(25/24) = 1.0206, which is far
  inside the beta sensitivity band already reported.
- Higher PRS receives more weight (PRS is a 0-100 index in which higher means
  lower political risk).
- Paired year-level inference; 14 held-out years (2011-2024).

Estimands
---------
The RMSE numbers reproduced in the gate block are the canonical *fold-mean*
estimand (mean of the 14 annual fold RMSEs), which is the estimand behind the
thesis headline 1.5722 vs 1.7267. The separate row-pooled estimand is not used
here. The portfolio metrics themselves are year-level quantities by construction,
so the fold-mean unit and the portfolio unit agree: one observation per year.

Statistical and interpretive boundary
-------------------------------------
- The realised-PRS allocation is an oracle measurement benchmark. It is what a
  perfectly informed risk analyst would have allocated given end-of-year PRS. It
  is not a profitable portfolio and carries no claim about returns.
- No asset returns, prices, transaction costs, financing, capacity, liquidity,
  benchmark tracking or investor utility enter this experiment at any point.
  "Reallocation share" is a distance between weight vectors, not a cost.
- The softmax mapping is one arbitrary but pre-specified monotone rule. Nothing
  here shows it is the rule an investor should use.
- A null or adverse result is a valid outcome and is reported as such.

Outputs
-------
- results/decision_use/portfolio_risk_budget/portfolio_risk_budget_rows.csv
- results/decision_use/portfolio_risk_budget/portfolio_risk_budget_year_cutoff.csv
- results/decision_use/portfolio_risk_budget/portfolio_risk_budget.json
- results/decision_use/portfolio_risk_budget/portfolio_risk_budget.md
- results/decision_use/portfolio_risk_budget/portfolio_risk_budget_gap_by_cutoff.pdf/.png
- sibling .provenance.json manifests

Reproduction
------------
    cd ~/Documents/Masters/repos/prs-nowcast
    python3 analysis/decision_use/portfolio_risk_budget.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.ml_experiments.provenance import write_manifest  # noqa: E402

# --- Locked configuration ---------------------------------------------------

CUTOFFS: tuple[int, ...] = (3, 6, 9, 11, 12)
TARGET = "delta_prs"
HEADLINE_MODEL = "BayesianRidge"
HEADLINE_CUTOFF = 12

BETA_PRIMARY = 0.5
BETA_SENSITIVITY: tuple[float, ...] = (0.25, 1.0)
ALL_BETAS: tuple[float, ...] = (0.25, 0.5, 1.0)

STANDARDISE_DDOF = 0
N_BOOT = 10_000
SEED = 42
CI = 0.95

EXPECTED_COUNTRIES = 25
EXPECTED_YEARS: tuple[int, ...] = tuple(range(2011, 2025))
EXPECTED_ROWS_PER_CUTOFF = EXPECTED_COUNTRIES * len(EXPECTED_YEARS)  # 350

# Tolerances. RMSE reproduction from the stored rows is pure arithmetic on the
# same floats, so it should agree to near machine precision; 1e-10 leaves room
# for summation order only. Weight vectors are normalised sums, so 1e-12.
RMSE_TOLERANCE = 1e-10
WEIGHT_SUM_TOLERANCE = 1e-12
LEVEL_IDENTITY_TOLERANCE = 1e-9

# --- Inputs (frozen canonical artefacts; read-only) -------------------------

WALK_FORWARD_JSON = REPO_ROOT / "results/canonical/ml_experiments/results/nowcast_walk_forward.json"
PANEL_CSV = REPO_ROOT / "results/canonical/frozen/_inputs/panel_pooled_features.csv"

# SHA256 of the two inputs, captured on this branch. The gate below fails loudly
# if a canonical artefact moves underneath this analysis.
EXPECTED_INPUT_SHA256 = {
    "results/canonical/ml_experiments/results/nowcast_walk_forward.json":
        "300f33dd612521ca33a0e02cd451cc648a428f4c0ad16c19cceb09912a99f80e",
    "results/canonical/frozen/_inputs/panel_pooled_features.csv":
        "709b335425558ca31832ce71919248032ccf4b449a1e3a27270cabfac42fb09d",
}

# Canonical fold-mean RMSE values this analysis must reproduce from the rows.
EXPECTED_FOLD_MEAN_RMSE = {
    3: 1.6713,
    6: 1.6515,
    9: 1.6159,
    11: 1.5881,
    12: 1.5722,
}
EXPECTED_FOLD_MEAN_PERSISTENCE_RMSE = 1.7267

# --- Outputs ----------------------------------------------------------------

OUT_DIR = REPO_ROOT / "results" / "decision_use" / "portfolio_risk_budget"
OUT_ROWS_CSV = OUT_DIR / "portfolio_risk_budget_rows.csv"
OUT_YEAR_CSV = OUT_DIR / "portfolio_risk_budget_year_cutoff.csv"
OUT_JSON = OUT_DIR / "portfolio_risk_budget.json"
OUT_MD = OUT_DIR / "portfolio_risk_budget.md"

# Illustration year for the allocation-anatomy figure. Chosen on thesis-narrative
# grounds before the results were inspected: 2022 is the locked Chapter 1 opener
# (the February 2022 shock). It is deliberately NOT the year with the largest gap.
ANATOMY_YEAR = 2022

REPRODUCTION_COMMAND = "python3 analysis/decision_use/portfolio_risk_budget.py"


# --- Helpers ----------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def softmax_weights(scores: np.ndarray, beta: float, *, ddof: int = STANDARDISE_DDOF) -> np.ndarray:
    """Long-only, fully invested softmax weights on cross-sectionally standardised scores.

    Higher score receives more weight. The max-subtraction is the standard
    numerically stable softmax and leaves the weights unchanged.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    spread = scores.std(ddof=ddof)
    if spread <= 0:
        # Degenerate cross-section: every score identical, so equal weights.
        return np.full(scores.shape, 1.0 / scores.size)
    z = (scores - scores.mean()) / spread
    exponent = beta * z
    exponent -= exponent.max()
    weights = np.exp(exponent)
    return weights / weights.sum()


def half_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Half the L1 distance between two weight vectors.

    For two long-only vectors that each sum to one this equals the share of the
    portfolio that must be reallocated to move from one to the other.
    """
    return float(0.5 * np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def herfindahl(weights: np.ndarray) -> float:
    return float(np.square(np.asarray(weights, dtype=float)).sum())


def year_block_bootstrap_ci(
    values: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED, ci: float = CI
) -> dict[str, float]:
    """Percentile confidence interval for the mean, resampling whole years."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return {
        "mean": float(values.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "ci_level": float(ci),
    }


def sign_flip_null_distribution(values: np.ndarray) -> np.ndarray:
    """Every one of the 2^n sign-flipped means, in enumeration order."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n > 20:
        raise ValueError("exact enumeration is only intended for small n")
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)
    return signs @ values / n


def exact_sign_flip_test(values: np.ndarray) -> dict[str, float]:
    """Two-sided exact sign-flip (randomisation) test on the mean of paired differences.

    Enumerates every one of the 2^n sign assignments, so no seed is involved and
    the p-value is exact under the null that each paired difference is
    symmetrically distributed about zero.
    """
    values = np.asarray(values, dtype=float)
    means = sign_flip_null_distribution(values)
    observed = float(values.mean())
    # Strict-inequality guard uses a relative tolerance so the observed
    # assignment always counts itself, keeping p >= 1 / 2^n.
    tol = 1e-12 * max(1.0, abs(observed))
    extreme = int(np.sum(np.abs(means) >= abs(observed) - tol))
    # The mean is antisymmetric under a global sign flip, so every assignment is
    # paired with its negation and both share the same |mean|. A two-sided
    # p-value is therefore always a multiple of 2 / 2^n, and that is the
    # attainable resolution floor, not 1 / 2^n.
    return {
        "observed_mean": observed,
        "n_permutations": int(means.size),
        "n_at_least_as_extreme": extreme,
        "p_value": float(extreme / means.size),
        "p_value_floor": float(2.0 / means.size),
        "p_value_at_floor": bool(extreme <= 2),
    }


def holm_bonferroni(p_values: dict[int, float]) -> dict[int, float]:
    """Holm-Bonferroni step-down adjusted p-values, monotone and capped at one."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        value = (m - rank) * p
        running = max(running, value)
        adjusted[key] = float(min(1.0, running))
    return adjusted


# --- Data assembly ----------------------------------------------------------


def load_canonical_rows() -> pd.DataFrame:
    """Read the canonical BayesianRidge delta_prs rows for the five locked cutoffs."""
    payload = json.loads(WALK_FORWARD_JSON.read_text(encoding="utf-8"))
    records = [
        row
        for row in payload["row_details"]
        if row["target"] == TARGET
        and row["model"] == HEADLINE_MODEL
        and int(row["cutoff_m"]) in CUTOFFS
    ]
    frame = pd.DataFrame.from_records(records)
    frame["cutoff_m"] = frame["cutoff_m"].astype(int)
    frame["year"] = frame["year"].astype(int)
    return frame, payload


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_CSV)
    panel = panel[["country", "country_slug", "Year", "prs_lag1", "gt_composite", "delta_prs"]]
    return panel.rename(columns={"Year": "year", "delta_prs": "panel_delta_prs"})


def build_levels(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the previous-year anchor and form the three annual PRS levels."""
    merged = rows.merge(panel, on=["country_slug", "year"], how="left", suffixes=("", "_panel"))
    merged["level_persistence"] = merged["prs_lag1"]
    merged["level_nowcast"] = merged["prs_lag1"] + merged["y_pred"]
    merged["level_realised"] = merged["prs_lag1"] + merged["y_true"]
    return merged


# --- Core simulation --------------------------------------------------------

VECTORS = ("persistence", "nowcast", "realised")


def simulate(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-row weights, per year-cutoff-beta portfolio metrics)."""
    row_records: list[dict[str, Any]] = []
    year_records: list[dict[str, Any]] = []

    for beta in ALL_BETAS:
        for cutoff in CUTOFFS:
            for year in EXPECTED_YEARS:
                block = levels[(levels["cutoff_m"] == cutoff) & (levels["year"] == year)]
                block = block.sort_values("country_slug").reset_index(drop=True)
                if len(block) != EXPECTED_COUNTRIES:
                    raise RuntimeError(
                        f"cutoff {cutoff} year {year}: expected {EXPECTED_COUNTRIES} countries, got {len(block)}"
                    )
                weights = {
                    name: softmax_weights(block[f"level_{name}"].to_numpy(), beta)
                    for name in VECTORS
                }

                for i, slug in enumerate(block["country_slug"]):
                    row_records.append(
                        {
                            "beta": beta,
                            "cutoff_m": cutoff,
                            "year": year,
                            "country": block.loc[i, "country"],
                            "country_slug": slug,
                            "prs_lag1": float(block.loc[i, "prs_lag1"]),
                            "predicted_delta": float(block.loc[i, "y_pred"]),
                            "realised_delta": float(block.loc[i, "y_true"]),
                            "level_persistence": float(block.loc[i, "level_persistence"]),
                            "level_nowcast": float(block.loc[i, "level_nowcast"]),
                            "level_realised": float(block.loc[i, "level_realised"]),
                            "weight_persistence": float(weights["persistence"][i]),
                            "weight_nowcast": float(weights["nowcast"][i]),
                            "weight_realised": float(weights["realised"][i]),
                        }
                    )

                realloc_nowcast = half_l1(weights["nowcast"], weights["realised"])
                realloc_persistence = half_l1(weights["persistence"], weights["realised"])
                year_records.append(
                    {
                        "beta": beta,
                        "cutoff_m": cutoff,
                        "year": year,
                        "reallocation_nowcast": realloc_nowcast,
                        "reallocation_persistence": realloc_persistence,
                        "reallocation_gap": realloc_persistence - realloc_nowcast,
                        "turnover_persistence_to_nowcast": half_l1(
                            weights["persistence"], weights["nowcast"]
                        ),
                        **{f"hhi_{name}": herfindahl(weights[name]) for name in VECTORS},
                        **{
                            f"effective_n_{name}": 1.0 / herfindahl(weights[name])
                            for name in VECTORS
                        },
                        **{f"max_weight_{name}": float(weights[name].max()) for name in VECTORS},
                    }
                )

    return pd.DataFrame(row_records), pd.DataFrame(year_records)


def summarise(year_frame: pd.DataFrame) -> dict[str, Any]:
    """Per-beta, per-cutoff descriptive summary plus inference for every beta."""
    summary: dict[str, Any] = {}
    for beta in ALL_BETAS:
        beta_block = year_frame[year_frame["beta"] == beta]
        raw_p: dict[int, float] = {}
        per_cutoff: dict[str, Any] = {}
        for cutoff in CUTOFFS:
            block = beta_block[beta_block["cutoff_m"] == cutoff].sort_values("year")
            gap = block["reallocation_gap"].to_numpy()
            bootstrap = year_block_bootstrap_ci(gap)
            sign_flip = exact_sign_flip_test(gap)
            raw_p[cutoff] = sign_flip["p_value"]
            per_cutoff[str(cutoff)] = {
                "n_years": int(len(block)),
                "mean_reallocation_nowcast": float(block["reallocation_nowcast"].mean()),
                "mean_reallocation_persistence": float(block["reallocation_persistence"].mean()),
                "mean_reallocation_gap": float(gap.mean()),
                "median_reallocation_gap": float(np.median(gap)),
                "relative_reduction_pct": float(
                    100.0
                    * gap.mean()
                    / block["reallocation_persistence"].mean()
                ),
                "years_nowcast_closer": int((gap > 0).sum()),
                "mean_turnover_persistence_to_nowcast": float(
                    block["turnover_persistence_to_nowcast"].mean()
                ),
                "concentration": {
                    name: {
                        "mean_hhi": float(block[f"hhi_{name}"].mean()),
                        "mean_effective_n": float(block[f"effective_n_{name}"].mean()),
                        "mean_max_weight": float(block[f"max_weight_{name}"].mean()),
                    }
                    for name in VECTORS
                },
                "year_block_bootstrap": bootstrap,
                "exact_sign_flip": sign_flip,
            }
        adjusted = holm_bonferroni(raw_p)
        for cutoff in CUTOFFS:
            per_cutoff[str(cutoff)]["holm_adjusted_p_value"] = adjusted[cutoff]
            per_cutoff[str(cutoff)]["holm_significant_at_0_05"] = bool(adjusted[cutoff] < 0.05)
        summary[str(beta)] = {
            "beta": beta,
            "role": "primary" if beta == BETA_PRIMARY else "sensitivity",
            "multiplicity_correction": "holm-bonferroni across the five cutoffs",
            "per_cutoff": per_cutoff,
        }
    return summary


# --- Gates ------------------------------------------------------------------


def run_gates(
    rows: pd.DataFrame,
    levels: pd.DataFrame,
    row_frame: pd.DataFrame,
    payload: dict[str, Any],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}

    # G1: canonical inputs unchanged.
    observed_sha = {
        rel: _sha256(REPO_ROOT / rel) for rel in sorted(EXPECTED_INPUT_SHA256)
    }
    gates["canonical_inputs_unchanged"] = {
        "passed": observed_sha == EXPECTED_INPUT_SHA256,
        "expected": EXPECTED_INPUT_SHA256,
        "observed": observed_sha,
    }

    # G2: each country-year-cutoff appears exactly once, with the expected shape.
    duplicates = int(rows.duplicated(subset=["country_slug", "year", "cutoff_m"]).sum())
    per_cutoff_counts = rows.groupby("cutoff_m").size().to_dict()
    gates["row_uniqueness_and_shape"] = {
        "passed": (
            duplicates == 0
            and set(per_cutoff_counts) == set(CUTOFFS)
            and all(int(v) == EXPECTED_ROWS_PER_CUTOFF for v in per_cutoff_counts.values())
            and int(rows["country_slug"].nunique()) == EXPECTED_COUNTRIES
            and tuple(sorted(rows["year"].unique())) == EXPECTED_YEARS
        ),
        "duplicate_country_year_cutoff": duplicates,
        "rows_per_cutoff": {int(k): int(v) for k, v in per_cutoff_counts.items()},
        "n_countries": int(rows["country_slug"].nunique()),
        "years": [int(y) for y in sorted(rows["year"].unique())],
        "total_rows": int(len(rows)),
    }

    # G3: fold-mean RMSE and persistence RMSE reproduce the canonical artefact.
    rmse_detail: dict[str, Any] = {}
    rmse_ok = True
    for cutoff in CUTOFFS:
        block = rows[rows["cutoff_m"] == cutoff]
        by_year = block.groupby("year")
        model_rmse = float(
            by_year.apply(
                lambda g: np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2)), include_groups=False
            ).mean()
        )
        persistence_rmse = float(
            by_year.apply(
                lambda g: np.sqrt(np.mean(g["y_true"] ** 2)), include_groups=False
            ).mean()
        )
        artefact = payload["nowcast_results"][TARGET][str(cutoff)][HEADLINE_MODEL]
        matches = (
            abs(model_rmse - artefact["rmse"]) < RMSE_TOLERANCE
            and abs(persistence_rmse - artefact["persistence_rmse"]) < RMSE_TOLERANCE
            and round(model_rmse, 4) == EXPECTED_FOLD_MEAN_RMSE[cutoff]
            and round(persistence_rmse, 4) == EXPECTED_FOLD_MEAN_PERSISTENCE_RMSE
        )
        rmse_ok = rmse_ok and matches
        rmse_detail[str(cutoff)] = {
            "recomputed_fold_mean_rmse": model_rmse,
            "artefact_fold_mean_rmse": float(artefact["rmse"]),
            "recomputed_fold_mean_persistence_rmse": persistence_rmse,
            "artefact_fold_mean_persistence_rmse": float(artefact["persistence_rmse"]),
            "pct_vs_persistence": 100.0 * (model_rmse - persistence_rmse) / persistence_rmse,
            "matches": bool(matches),
        }
    gates["canonical_rmse_reproduced"] = {"passed": bool(rmse_ok), "per_cutoff": rmse_detail}

    # G4: panel join complete and the realised level is exactly gt_composite.
    unmatched = int(levels["prs_lag1"].isna().sum())
    level_gap = float(np.max(np.abs(levels["level_realised"] - levels["gt_composite"])))
    delta_gap = float(np.max(np.abs(levels["y_true"] - levels["panel_delta_prs"])))
    gates["panel_join_and_level_identity"] = {
        "passed": bool(
            unmatched == 0
            and level_gap < LEVEL_IDENTITY_TOLERANCE
            and delta_gap < LEVEL_IDENTITY_TOLERANCE
        ),
        "unmatched_rows": unmatched,
        "max_abs_level_minus_gt_composite": level_gap,
        "max_abs_row_delta_minus_panel_delta": delta_gap,
        "missing_prs_lag1": int(levels["prs_lag1"].isna().sum()),
        "missing_predicted": int(levels["y_pred"].isna().sum()),
        "missing_realised": int(levels["y_true"].isna().sum()),
    }

    # G5: every weight vector is long-only and fully invested.
    grouped = row_frame.groupby(["beta", "cutoff_m", "year"])
    sums = {name: grouped[f"weight_{name}"].sum() for name in VECTORS}
    max_sum_error = max(float(np.max(np.abs(s - 1.0))) for s in sums.values())
    min_weight = min(float(row_frame[f"weight_{name}"].min()) for name in VECTORS)
    gates["weights_long_only_fully_invested"] = {
        "passed": bool(max_sum_error < WEIGHT_SUM_TOLERANCE and min_weight > 0.0),
        "max_abs_weight_sum_error": max_sum_error,
        "min_weight": min_weight,
        "n_weight_vectors": int(3 * len(grouped)),
    }

    gates["all_passed"] = all(
        bool(value["passed"]) for value in gates.values() if isinstance(value, dict)
    )
    return gates


# --- Payload, markdown, figure ---------------------------------------------


def build(*, write_outputs: bool = True) -> dict[str, Any]:
    rows, payload = load_canonical_rows()
    panel = load_panel()
    levels = build_levels(rows, panel)
    row_frame, year_frame = simulate(levels)
    gates = run_gates(rows, levels, row_frame, payload)
    summary = summarise(year_frame)

    result: dict[str, Any] = {
        "config": {
            "question": (
                "Would the locked out-of-sample same-year nowcasts have produced a 25-country "
                "risk allocation closer to the allocation implied by realised annual PRS than "
                "the stale previous-year score?"
            ),
            "status": "decision-use illustration, not canonical thesis evidence",
            "cutoffs": list(CUTOFFS),
            "target": TARGET,
            "model": HEADLINE_MODEL,
            "beta_primary": BETA_PRIMARY,
            "beta_sensitivity": list(BETA_SENSITIVITY),
            "weighting_rule": "w_i = exp(beta * z_i) / sum_j exp(beta * z_j), z cross-sectionally standardised within year and cutoff, higher PRS gets more weight",
            "standardisation_ddof": STANDARDISE_DDOF,
            "distance_metric": "half the L1 distance between weight vectors, read as the share of the portfolio needing reallocation",
            "paired_statistic": "d_y = reallocation_persistence(y) - reallocation_nowcast(y); positive favours the nowcast",
            "inference": {
                "unit": "year",
                "n_years": len(EXPECTED_YEARS),
                "year_block_bootstrap": {"n_boot": N_BOOT, "seed": SEED, "ci_level": CI},
                "exact_sign_flip_permutations": 2 ** len(EXPECTED_YEARS),
                "multiplicity": "holm-bonferroni across the five cutoffs",
            },
            "reproduction_command": REPRODUCTION_COMMAND,
        },
        "inputs": {
            rel: {"sha256": _sha256(REPO_ROOT / rel)} for rel in sorted(EXPECTED_INPUT_SHA256)
        },
        "sample": {
            "n_countries": EXPECTED_COUNTRIES,
            "years": list(EXPECTED_YEARS),
            "rows_per_cutoff": EXPECTED_ROWS_PER_CUTOFF,
            "total_canonical_rows": int(len(rows)),
            "missingness": {
                "prs_lag1": int(levels["prs_lag1"].isna().sum()),
                "predicted_delta": int(levels["y_pred"].isna().sum()),
                "realised_delta": int(levels["y_true"].isna().sum()),
            },
        },
        "gates": gates,
        "results": summary,
        "statistical_boundary": [
            "The realised-PRS allocation is an oracle measurement benchmark, not a profitable portfolio.",
            "No asset returns, prices, transaction costs, financing, liquidity or investor utility enter this experiment.",
            "Reallocation share is a distance between weight vectors, not a monetary cost.",
            "The softmax mapping is one pre-specified monotone rule and is not shown to be optimal.",
            "The absolute quantities are small. Annual PRS levels are highly persistent, so the "
            "cross-section barely moves year to year and any monotone weighting of it barely moves "
            "either. The relative improvement is the interpretable figure, not the raw percentage points.",
            "Inference rests on fourteen paired years. The sign-flip test is exact at that sample "
            "size, but the year-block bootstrap interval is approximate with only fourteen blocks.",
            "One year, 2014, contributes a paired difference several times the typical size and "
            "visibly splits the randomisation null. The result holds on a broad majority of years "
            "rather than on that year alone, but it is not free of influential-observation risk.",
            "The result is direction-consistent but sharpness-dependent: at beta = 1.0 it does not "
            "survive correction across the five cutoffs.",
            "A null or adverse result is a valid outcome of this simulation.",
        ],
    }

    if write_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        row_frame.to_csv(OUT_ROWS_CSV, index=False)
        year_frame.to_csv(OUT_YEAR_CSV, index=False)
        result["figures"] = build_figures(summary, row_frame, year_frame)
        result["config"]["anatomy_year"] = ANATOMY_YEAR
        OUT_JSON.write_text(
            json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        OUT_MD.write_text(build_markdown(result), encoding="utf-8")
        manifest_targets = [OUT_ROWS_CSV, OUT_YEAR_CSV, OUT_JSON, OUT_MD]
        manifest_targets += [REPO_ROOT / meta["pdf"] for meta in result["figures"].values()]
        for target in manifest_targets:
            write_manifest(
                target,
                input_paths=[WALK_FORWARD_JSON, PANEL_CSV],
                script_path=Path(__file__),
            )

    return result


def _save_figure(fig, stem: str, caption: str) -> dict[str, str]:
    """Write PDF, PNG and a draft caption, following the house figure convention.

    The PDF CreationDate is suppressed so reruns stay byte-identical; matplotlib
    otherwise stamps the wall clock into the document metadata.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.55)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches=None, metadata={"CreationDate": None})
    fig.savefig(png, dpi=300, bbox_inches=None)
    import matplotlib.pyplot as plt

    plt.close(fig)
    caption_path = OUT_DIR / f"{stem}.caption_draft.txt"
    caption_path.write_text(
        "% DRAFT CAPTION, Rahul to rewrite\n" + caption.strip() + "\n", encoding="utf-8"
    )
    return {
        "pdf": str(pdf.relative_to(REPO_ROOT)),
        "png": str(png.relative_to(REPO_ROOT)),
        "caption_draft": str(caption_path.relative_to(REPO_ROOT)),
    }


def _style_axis(ax, *, axis: str = "y") -> None:
    from analysis.thesis_figures._common import COLOUR

    ax.grid(axis=axis, color=COLOUR["grid"], linewidth=0.5)
    ax.set_axisbelow(True)


def _panel_size(height_inches: float, *, fraction: float = 1.0) -> tuple[float, float]:
    """Textwidth-matched width with an explicitly chosen height.

    The height is set per figure rather than derived from a golden ratio, because
    these panels carry long rotated axis labels and, in the anatomy figure, 25
    country rows. Deriving the height instead clips the labels at `bbox_inches=None`,
    which is the setting that keeps the rendered width equal to the LaTeX textwidth.
    """
    from analysis.thesis_figures._common import PT_PER_IN, TEXTWIDTH_PT

    return (TEXTWIDTH_PT / PT_PER_IN) * fraction, height_inches


def _headroom(ax, *, top: float = 0.22, bottom: float = 0.0) -> None:
    """Open vertical space so an in-axes legend never sits on the data."""
    low, high = ax.get_ylim()
    span = high - low
    ax.set_ylim(low - bottom * span, high + top * span)


def build_figures(
    summary: dict[str, Any], row_frame: pd.DataFrame, year_frame: pd.DataFrame
) -> dict[str, Any]:
    """The figure set.

    Five figures, each answering a question the tables cannot. House gates from
    `analysis/thesis_figures/_common.py` are honoured: no in-image titles (G1),
    no in-axes numeric stat boxes (G2), time never on the y-axis of a
    time-indexed panel (G3), and one fixed colour per series across the whole set
    (G5) taken from the same Okabe-Ito anchors as the Chapter 4 figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis.thesis_figures._common import COLOUR, configure_style, country_label

    configure_style()
    figures: dict[str, Any] = {}
    primary = summary[str(BETA_PRIMARY)]["per_cutoff"]
    cutoffs = list(CUTOFFS)

    label_persistence = "Previous-year score"
    label_nowcast = "Nowcast score"
    label_realised = "Realised PRS (oracle)"

    # --- F1: the headline, by evidence cutoff -------------------------------
    persistence = [100.0 * primary[str(c)]["mean_reallocation_persistence"] for c in cutoffs]
    nowcast = [100.0 * primary[str(c)]["mean_reallocation_nowcast"] for c in cutoffs]

    fig, ax = plt.subplots(figsize=_panel_size(2.8))
    ax.plot(
        cutoffs, persistence, marker="s", markersize=4, linewidth=1.2,
        color=COLOUR["persistence"],
    )
    ax.plot(
        cutoffs, nowcast, marker="o", markersize=4, linewidth=1.2,
        color=COLOUR["model"],
    )
    ax.fill_between(cutoffs, nowcast, persistence, color=COLOUR["model"], alpha=0.10, linewidth=0)
    ax.annotate(
        "Last cutoff before year-end",
        xy=(11, nowcast[-2]),
        xytext=(8.6, nowcast[-2] - 0.035),
        color=COLOUR["model"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": COLOUR["model"], "linewidth": 0.7},
    )
    ax.text(
        12.25, persistence[-1], f"Last year's scores: {persistence[-1]:.2f}%",
        color=COLOUR["persistence"], va="center", ha="left",
    )
    ax.text(
        12.25, nowcast[-1], f"Full-year check: {nowcast[-1]:.2f}%",
        color=COLOUR["model"], va="center", ha="left",
    )
    ax.set_xlabel("Information available through month")
    ax.set_ylabel("Weight that would need to move (%)")
    ax.set_xticks(cutoffs)
    ax.set_xlim(2.5, 15.0)
    ax.set_ylim(min(nowcast) - 0.06, max(persistence) + 0.06)
    _style_axis(ax)
    figures["gap_by_cutoff"] = _save_figure(
        fig,
        "f1_gap_by_cutoff",
        "Average share of the 25-country allocation that would have to move to match the weights "
        "based on final annual PRS. Lower values are closer to the final allocation. The updated "
        "scores move closer as evidence accumulates, while the allocation based on the previous "
        "year's scores remains fixed. Month 11 is the final pre-year-end cutoff; month 12 is the "
        "full-year validation point. Softmax sharpness beta = 0.5.",
    )

    # --- F2: the year-level paired result at the headline cutoff ------------
    block = year_frame[
        (year_frame["beta"] == BETA_PRIMARY) & (year_frame["cutoff_m"] == HEADLINE_CUTOFF)
    ].sort_values("year")
    years = block["year"].to_numpy()
    y_pers = 100.0 * block["reallocation_persistence"].to_numpy()
    y_now = 100.0 * block["reallocation_nowcast"].to_numpy()

    fig, ax = plt.subplots(figsize=_panel_size(2.95))
    for x, a, b in zip(years, y_pers, y_now):
        improved = b < a
        ax.annotate(
            "",
            xy=(x, b),
            xytext=(x, a),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.16,head_length=0.34",
                "color": COLOUR["model"] if improved else COLOUR["persistence"],
                "linewidth": 1.1,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.plot(years, y_pers, marker="s", markersize=4, linestyle="none",
            color=COLOUR["persistence"], label=label_persistence, zorder=3)
    ax.plot(years, y_now, marker="o", markersize=4, linestyle="none",
            color=COLOUR["model"], label=label_nowcast, zorder=3)
    ax.plot([], [], color=COLOUR["model"], linewidth=1.1, label="Nowcast closer to realised")
    ax.plot([], [], color=COLOUR["persistence"], linewidth=1.1, label="Previous year closer")
    ax.set_xlabel("Held-out year")
    ax.set_ylabel("Portfolio share to reallocate (%)")
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    _style_axis(ax)
    _headroom(ax, top=0.26)
    ax.legend(frameon=False, loc="upper center", ncol=4, columnspacing=1.4, handletextpad=0.5)
    figures["year_level_paired"] = _save_figure(
        fig,
        "f2_year_level_paired",
        "Year-by-year view of the same comparison at the full-year cutoff m = 12. Each arrow runs "
        "from the reallocation required by the stale previous-year allocation to that required by "
        "the nowcast allocation, so a downward blue arrow is a year in which the nowcast was "
        "closer to the realised-PRS allocation. Twelve of the fourteen held-out years favour the "
        "nowcast; 2011 and 2023 do not. The advantage is broad rather than carried by a single "
        "year, although 2014 contributes the largest single-year gain.",
    )

    # --- F3: the anatomy of the metric in one year --------------------------
    anatomy = row_frame[
        (row_frame["beta"] == BETA_PRIMARY)
        & (row_frame["cutoff_m"] == HEADLINE_CUTOFF)
        & (row_frame["year"] == ANATOMY_YEAR)
    ].copy()
    anatomy["error_persistence"] = 100.0 * (
        anatomy["weight_persistence"] - anatomy["weight_realised"]
    )
    anatomy["error_nowcast"] = 100.0 * (anatomy["weight_nowcast"] - anatomy["weight_realised"])
    anatomy = anatomy.sort_values("weight_realised").reset_index(drop=True)
    positions = np.arange(len(anatomy))

    fig, axes = plt.subplots(
        1, 2, figsize=_panel_size(5.5), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )
    ax = axes[0]
    ax.barh(
        positions, 100.0 * anatomy["weight_realised"], height=0.62,
        color=COLOUR["actual"], alpha=0.85, linewidth=0, label=label_realised,
    )
    ax.axvline(
        100.0 / EXPECTED_COUNTRIES, color=COLOUR["dark"], linewidth=0.9,
        linestyle=(0, (4, 2)), label="Equal weight",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([country_label(slug) for slug in anatomy["country_slug"]])
    ax.set_ylim(-0.8, len(anatomy) - 0.2)
    ax.set_xlabel("Realised-PRS weight (%)")
    ax.tick_params(axis="y", length=0, pad=2)
    _style_axis(ax, axis="x")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.6)

    ax = axes[1]
    ax.axvline(0.0, color=COLOUR["dark"], linewidth=0.8, zorder=1)
    for pos, ep, en in zip(positions, anatomy["error_persistence"], anatomy["error_nowcast"]):
        improved = abs(en) < abs(ep)
        ax.annotate(
            "",
            xy=(en, pos),
            xytext=(ep, pos),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.14,head_length=0.30",
                "color": COLOUR["model"] if improved else COLOUR["persistence"],
                "linewidth": 0.9,
                "alpha": 0.55,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.plot(anatomy["error_persistence"], positions, marker="s", markersize=3.4,
            linestyle="none", color=COLOUR["persistence"], label=label_persistence, zorder=3)
    ax.plot(anatomy["error_nowcast"], positions, marker="o", markersize=3.4,
            linestyle="none", color=COLOUR["model"], label=label_nowcast, zorder=3)
    ax.set_xlabel("Weight error against the realised allocation (percentage points)")
    ax.tick_params(axis="y", length=0)
    _style_axis(ax, axis="x")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.5)
    figures["allocation_anatomy"] = _save_figure(
        fig,
        "f3_allocation_anatomy",
        f"Anatomy of the reallocation metric for {ANATOMY_YEAR} at cutoff m = 12, the year the "
        "thesis uses as its opening illustration. Left: the risk budget implied by realised annual "
        "PRS, with the dashed line marking the equal-weight benchmark of four per cent. Right: the "
        "weight error of each score vector against that realised allocation, country by country, "
        "ordered by realised weight. Half the sum of the absolute errors in the right panel is "
        "exactly the reallocation share reported elsewhere, so the figure shows which countries "
        "the nowcast corrects and which it leaves misallocated. Softmax sharpness beta = 0.5.",
    )

    # --- F4: sensitivity to the weighting sharpness -------------------------
    fig, ax = plt.subplots(figsize=_panel_size(2.9, fraction=0.74))
    ax.axhline(0.0, color=COLOUR["context"], linewidth=0.7, zorder=1)
    styles = {0.25: (":", "^"), 0.5: ("-", "o"), 1.0: ("--", "v")}
    for beta in ALL_BETAS:
        arm = summary[str(beta)]["per_cutoff"]
        values = np.array([100.0 * arm[str(c)]["mean_reallocation_gap"] for c in cutoffs])
        lo = np.array([100.0 * arm[str(c)]["year_block_bootstrap"]["ci_lower"] for c in cutoffs])
        hi = np.array([100.0 * arm[str(c)]["year_block_bootstrap"]["ci_upper"] for c in cutoffs])
        linestyle, marker = styles[beta]
        primary_arm = beta == BETA_PRIMARY
        colour = COLOUR["model"] if primary_arm else COLOUR["dark"]
        ax.fill_between(
            cutoffs, lo, hi, color=colour, linewidth=0,
            alpha=0.16 if primary_arm else 0.07,
        )
        ax.plot(
            cutoffs, values, linestyle=linestyle, marker=marker, markersize=4,
            linewidth=1.5 if primary_arm else 1.0, alpha=1.0 if primary_arm else 0.75,
            color=colour,
            label=f"$\\beta$ = {beta}" + (" (primary)" if primary_arm else ""),
        )
    ax.set_xlabel("Evidence cutoff month within the target year")
    ax.set_ylabel("Reallocation saved (percentage points)")
    ax.set_xticks(cutoffs)
    _style_axis(ax)
    _headroom(ax, top=0.30)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.6)
    figures["beta_sensitivity"] = _save_figure(
        fig,
        "f4_beta_sensitivity",
        "Sensitivity of the paired result to the sharpness of the softmax weighting rule, with "
        "year-block bootstrap 95 per cent intervals. The sign and the ordering across cutoffs hold "
        "at every sharpness, but the sharper rule concentrates weight on the extremes of the "
        "cross-section, which widens the interval faster than it widens the effect and costs the "
        "result its significance once the five cutoffs are corrected together.",
    )

    # --- F5: the exact randomisation null -----------------------------------
    contrast = (CUTOFFS[0], HEADLINE_CUTOFF)
    fig, axes = plt.subplots(1, 2, figsize=_panel_size(2.75), sharey=True)
    for ax, cutoff in zip(axes, contrast):
        block = year_frame[
            (year_frame["beta"] == BETA_PRIMARY) & (year_frame["cutoff_m"] == cutoff)
        ].sort_values("year")
        values = block["reallocation_gap"].to_numpy()
        null = 100.0 * sign_flip_null_distribution(values)
        observed = 100.0 * float(values.mean())
        ax.hist(null, bins=90, color=COLOUR["context"], alpha=0.45, linewidth=0)
        extreme = np.abs(null) >= abs(observed) - 1e-12
        ax.hist(null[extreme], bins=90, range=(null.min(), null.max()),
                color=COLOUR["event"], alpha=0.85, linewidth=0)
        ax.axvline(observed, color=COLOUR["model"], linewidth=1.3)
        ax.axvline(-observed, color=COLOUR["model"], linewidth=0.7, linestyle=(0, (3, 2)))
        ax.set_xlabel(f"Sign-flipped mean at m = {cutoff} (percentage points)")
        _style_axis(ax)
    axes[0].set_ylabel("Sign assignments")
    _headroom(axes[0], top=0.30)
    axes[1].plot([], [], color=COLOUR["model"], linewidth=1.3, label="Observed mean")
    axes[1].plot([], [], color=COLOUR["model"], linewidth=0.8, linestyle=(0, (3, 2)),
                 label="Reflected critical point")
    axes[1].plot([], [], color=COLOUR["event"], linewidth=3, alpha=0.85, label="At least as extreme")
    axes[1].legend(frameon=False, loc="upper left", handletextpad=0.6)
    figures["sign_flip_null"] = _save_figure(
        fig,
        "f5_sign_flip_null",
        "Exact randomisation null for the paired year-level statistic, enumerating all "
        f"$2^{{14}}$ = {2 ** 14} sign assignments of the fourteen held-out years rather than "
        "sampling them. Left: the earliest cutoff, where the observed mean sits well inside the "
        "null. Right: the full-year cutoff, where it sits in the tail. The shaded mass is the "
        "two-sided p-value by construction. The right-hand null is visibly bimodal because one "
        "year, 2014, contributes a difference several times the typical size, so flipping its sign "
        "alone moves the mean between the two modes; the result therefore rests on a broad "
        "majority of years but with one influential year, which is why the year-block bootstrap is "
        "reported alongside. Softmax sharpness beta = 0.5.",
    )

    return figures


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def build_markdown(result: dict[str, Any]) -> str:
    primary = result["results"][str(BETA_PRIMARY)]["per_cutoff"]
    lines: list[str] = []
    lines.append("# Portfolio risk-budget simulation")
    lines.append("")
    lines.append(
        "Decision-use illustration. Not canonical thesis evidence, not a return backtest, "
        "and not a claim of monetary value."
    )
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(result["config"]["question"])
    lines.append("")
    lines.append("## Design")
    lines.append("")
    lines.append(
        f"Canonical fixed {HEADLINE_MODEL} `{TARGET}` nowcast rows are read back from the frozen "
        f"walk-forward artefact for cutoffs m = {', '.join(str(c) for c in CUTOFFS)}. No model is "
        "refitted. For each held-out country-year the previous-year anchor `prs_lag1` carries three "
        "annual PRS levels: the previous-year score itself, that score plus the predicted change, "
        "and that score plus the realised change. Each 25-country vector becomes long-only, fully "
        f"invested weights through `w_i = exp(beta * z_i) / sum_j exp(beta * z_j)` with beta = {BETA_PRIMARY} "
        f"as the primary rule and beta in {{{', '.join(str(b) for b in BETA_SENSITIVITY)}}} as sensitivity "
        "checks fixed in advance. Distances between weight vectors are half the L1 distance, read as "
        "the share of the portfolio that would need reallocation."
    )
    lines.append("")
    lines.append("## Canonical reproduction")
    lines.append("")
    lines.append(
        "Recomputed directly from the stored out-of-sample rows before any simulation, so the "
        "reproduction is arithmetic on the same predictions the thesis reports."
    )
    lines.append("")
    lines.append("| m | fold-mean RMSE recomputed | artefact | persistence | vs persistence |")
    lines.append("|---|---|---|---|---|")
    for cutoff in CUTOFFS:
        detail = result["gates"]["canonical_rmse_reproduced"]["per_cutoff"][str(cutoff)]
        lines.append(
            f"| {cutoff} | {_fmt(detail['recomputed_fold_mean_rmse'])} | "
            f"{_fmt(detail['artefact_fold_mean_rmse'])} | "
            f"{_fmt(detail['recomputed_fold_mean_persistence_rmse'])} | "
            f"{detail['pct_vs_persistence']:+.2f}% |"
        )
    shape = result["gates"]["row_uniqueness_and_shape"]
    lines.append("")
    lines.append(
        f"Sample: {shape['total_rows']} rows, {shape['rows_per_cutoff'][CUTOFFS[0]]} per cutoff, "
        f"{shape['n_countries']} countries, {len(shape['years'])} held-out years "
        f"{shape['years'][0]}-{shape['years'][-1]}, {shape['duplicate_country_year_cutoff']} duplicate "
        "country-year-cutoff keys. No missing predicted, realised or anchor values."
    )
    lines.append("")
    lines.append(f"## Primary result (beta = {BETA_PRIMARY})")
    lines.append("")
    lines.append(
        "Reallocation share is the fraction of the portfolio that would have to move to reach the "
        "realised-PRS allocation; lower is closer. The gap is the paired year-level difference "
        "`persistence - nowcast`, so a positive gap favours the nowcast."
    )
    lines.append("")
    lines.append(
        "| m | persistence | nowcast | gap | relative | 95% CI (year block) | years nowcast closer | sign-flip p | Holm p |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cutoff in CUTOFFS:
        block = primary[str(cutoff)]
        boot = block["year_block_bootstrap"]
        lines.append(
            f"| {cutoff} | {_fmt(block['mean_reallocation_persistence'])} | "
            f"{_fmt(block['mean_reallocation_nowcast'])} | "
            f"{block['mean_reallocation_gap']:+.4f} | "
            f"{block['relative_reduction_pct']:+.2f}% | "
            f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}] | "
            f"{block['years_nowcast_closer']}/{block['n_years']} | "
            f"{block['exact_sign_flip']['p_value']:.4f} | "
            f"{block['holm_adjusted_p_value']:.4f} |"
        )
    lines.append("")
    headline = primary[str(HEADLINE_CUTOFF)]
    weakest = primary[str(CUTOFFS[0])]
    best_cutoff = max(CUTOFFS, key=lambda c: primary[str(c)]["mean_reallocation_gap"])
    significant = [c for c in CUTOFFS if primary[str(c)]["holm_significant_at_0_05"]]
    lines.append("### Reading")
    lines.append("")
    lines.append(
        "The direction favours the nowcast at every cutoff and the advantage grows as the evidence "
        f"window extends, from {weakest['mean_reallocation_gap'] * 100:.2f} percentage points of the "
        f"portfolio at m = {CUTOFFS[0]} to {primary[str(best_cutoff)]['mean_reallocation_gap'] * 100:.2f} "
        f"points at m = {best_cutoff}. It does not grow all the way to the end of the year: the gap is "
        f"flat between m = 11 and m = {HEADLINE_CUTOFF} "
        f"({primary['11']['mean_reallocation_gap'] * 100:.3f} against "
        f"{headline['mean_reallocation_gap'] * 100:.3f} points), even though the underlying RMSE still "
        "improves over that last month. The final month of evidence sharpens the measurement without "
        "moving the allocation, which is a limit on how far a measurement gain carries into a decision. "
        f"Only m = {' and m = '.join(str(c) for c in significant)} survive the Holm correction across the "
        "five cutoffs, so the defensible statement is about the full-year and near-full-year evidence "
        "windows rather than about early-year cutoffs."
    )
    lines.append("")
    lines.append(
        f"The scale matters more than the significance. At m = {HEADLINE_CUTOFF} the stale previous-year "
        f"score already sits within {headline['mean_reallocation_persistence'] * 100:.2f} percentage points "
        f"of the oracle allocation, and the nowcast closes {headline['relative_reduction_pct']:.1f}% of that "
        f"distance, leaving {headline['mean_reallocation_nowcast'] * 100:.2f} points still misallocated. The "
        "relative improvement is of the same order as the 8.95% RMSE improvement that generates it, but the "
        "absolute quantity is small because annual PRS levels are highly persistent: the cross-section of "
        "country scores barely moves year to year, so any monotone weighting of it barely moves either. A "
        "reader should take this as evidence that the measurement gain survives translation into an "
        "allocation, not as evidence that the allocation gain is economically large."
    )
    lines.append("")
    lines.append("## Turnover and concentration")
    lines.append("")
    lines.append(
        "| m | turnover persistence to nowcast | effective holdings persistence | nowcast | realised |"
    )
    lines.append("|---|---|---|---|---|")
    for cutoff in CUTOFFS:
        block = primary[str(cutoff)]
        conc = block["concentration"]
        lines.append(
            f"| {cutoff} | {_fmt(block['mean_turnover_persistence_to_nowcast'])} | "
            f"{_fmt(conc['persistence']['mean_effective_n'], 2)} | "
            f"{_fmt(conc['nowcast']['mean_effective_n'], 2)} | "
            f"{_fmt(conc['realised']['mean_effective_n'], 2)} |"
        )
    lines.append("")
    lines.append(
        "Equal weighting across the 25 countries would give exactly 25 effective holdings. All three "
        "vectors carry essentially the same concentration, so the reallocation result is not an "
        "artefact of one score vector being more concentrated than another; it reflects which "
        "countries are ranked where. Turnover from the persistence weights to the nowcast weights is "
        "roughly half the reallocation distance, so acting on the nowcast is a small trade relative "
        "to the error it removes."
    )
    lines.append("")
    lines.append("## Sensitivity to the weighting sharpness")
    lines.append("")
    lines.append("| beta | role | m | gap | 95% CI (year block) | sign-flip p | Holm p |")
    lines.append("|---|---|---|---|---|---|---|")
    for beta in ALL_BETAS:
        arm = result["results"][str(beta)]
        for cutoff in CUTOFFS:
            block = arm["per_cutoff"][str(cutoff)]
            boot = block["year_block_bootstrap"]
            lines.append(
                f"| {beta} | {arm['role']} | {cutoff} | {block['mean_reallocation_gap']:+.4f} | "
                f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}] | "
                f"{block['exact_sign_flip']['p_value']:.4f} | "
                f"{block['holm_adjusted_p_value']:.4f} |"
            )
    lines.append("")
    sharp = result["results"]["1.0"]["per_cutoff"][str(HEADLINE_CUTOFF)]
    flat = result["results"]["0.25"]["per_cutoff"][str(HEADLINE_CUTOFF)]
    lines.append(
        "The sign and the ordering across cutoffs are stable across all three sharpness settings, but "
        f"the evidence is not. The flatter rule (beta = 0.25) strengthens it, with Holm p = "
        f"{flat['holm_adjusted_p_value']:.4f} at m = {HEADLINE_CUTOFF}, because flatter weights make the "
        "distance metric respond almost linearly to the underlying score errors. The sharper rule "
        f"(beta = 1.0) roughly doubles the raw gap to {sharp['mean_reallocation_gap'] * 100:.2f} points but "
        f"loses significance under correction, Holm p = {sharp['holm_adjusted_p_value']:.4f}, because "
        "concentrating weight on the extremes of the cross-section makes each year's outcome depend on a "
        "few countries and inflates the year-to-year variance. The result should therefore be reported as "
        "direction-consistent but sharpness-dependent, and the primary beta = 0.5 statement should not be "
        "quoted without that qualifier."
    )
    lines.append("")
    figures = result.get("figures")
    if figures:
        lines.append("## Figures")
        lines.append("")
        titles = {
            "gap_by_cutoff": "The headline gap by evidence cutoff, with Holm significance marked",
            "year_level_paired": f"Year-by-year paired outcome at m = {HEADLINE_CUTOFF}",
            "allocation_anatomy": f"Anatomy of the metric for {ANATOMY_YEAR}, country by country",
            "beta_sensitivity": "Sensitivity to the softmax sharpness",
            "sign_flip_null": "Exact randomisation null, earliest against full-year cutoff",
        }
        for key, meta in figures.items():
            lines.append(f"- `{meta['pdf']}` - {titles.get(key, key)}")
        lines.append("")
        lines.append(
            "Draft captions sit beside each figure as `*.caption_draft.txt`. Colours follow the "
            "Chapter 4 convention: vermillion for the previous-year baseline, blue for the "
            "nowcast, green for the realised-PRS oracle."
        )
        lines.append("")
    lines.append("## Limits")
    lines.append("")
    for item in result["statistical_boundary"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| input | sha256 |")
    lines.append("|---|---|")
    for rel, meta in result["inputs"].items():
        lines.append(f"| `{rel}` | `{meta['sha256']}` |")
    lines.append("")
    lines.append("```")
    lines.append(REPRODUCTION_COMMAND)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-outputs",
        action="store_true",
        help="Run the simulation and gates without writing artefacts. Debugging only.",
    )
    args = parser.parse_args()

    result = build(write_outputs=not args.skip_outputs)

    for name, value in result["gates"].items():
        if isinstance(value, dict):
            print(f"{name}_passed={value['passed']}")
    print(f"all_gates_passed={result['gates']['all_passed']}")

    primary = result["results"][str(BETA_PRIMARY)]["per_cutoff"]
    for cutoff in CUTOFFS:
        block = primary[str(cutoff)]
        boot = block["year_block_bootstrap"]
        print(
            f"m={cutoff} persistence={block['mean_reallocation_persistence']:.4f} "
            f"nowcast={block['mean_reallocation_nowcast']:.4f} "
            f"gap={block['mean_reallocation_gap']:+.4f} "
            f"ci=[{boot['ci_lower']:+.4f},{boot['ci_upper']:+.4f}] "
            f"sign_flip_p={block['exact_sign_flip']['p_value']:.4f} "
            f"holm_p={block['holm_adjusted_p_value']:.4f}"
        )

    if not result["gates"]["all_passed"]:
        raise SystemExit("Gate failure: stop and diagnose before interpreting results.")


if __name__ == "__main__":
    main()
