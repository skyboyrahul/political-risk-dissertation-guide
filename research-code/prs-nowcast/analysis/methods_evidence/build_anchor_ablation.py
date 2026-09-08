"""Build the prior-year 12-component anchor-only nowcast ablation.

The evaluation deliberately mirrors the canonical BayesianRidge m=12
walk-forward panel: 25 countries, 2011--2024 test years, and five supervised
years before the first test fold.  Its only predictors are the twelve ICRG
component scores observed in year t-1.  No monthly LLM or disagreement value
enters the feature matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "analysis" / "ml_experiments"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from bootstrap_block import bootstrap_country_block, bootstrap_year_block  # noqa: E402
from diebold_mariano_corrections import (  # noqa: E402
    diebold_mariano_clustered,
    diebold_mariano_hln,
    diebold_mariano_two_way_clustered,
)

PANEL_PATH = REPO_ROOT / "results/canonical/frozen/_inputs/panel_pooled_features.csv"
GROUND_TRUTH_PATH = REPO_ROOT / "data/ground_truth/prs_only_data.csv"
CANONICAL_PATH = (
    REPO_ROOT
    / "results/canonical/frozen/artifacts/analysis/ml_experiments/results/nowcast_walk_forward.json"
)
OUT_DIR = REPO_ROOT / "results/methods_evidence/anchor_ablation"
MIN_TRAIN_YEARS = 5
N_BOOT = 2000
SEED = 42

COMPONENT_COLUMNS = {
    "Bureaucracy Quality": "bureaucracy_quality_lag1",
    "Corruption": "corruption_lag1",
    "Democratic Accountability": "democratic_accountability_lag1",
    "Ethnic Tensions": "ethnic_tensions_lag1",
    "External Conflict": "external_conflict_lag1",
    "Government Stability": "government_stability_lag1",
    "Internal Conflict": "internal_conflict_lag1",
    "Investment Profile": "investment_profile_lag1",
    "Law and Order": "law_and_order_lag1",
    "Military in Politics": "military_in_politics_lag1",
    "Religious Tensions": "religious_tensions_lag1",
    "Socioeconomic Conditions": "socioeconomic_conditions_lag1",
}
FEATURES = list(COMPONENT_COLUMNS.values())


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge()),
        ]
    )


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    panel["Year"] = panel["Year"].astype(int)
    countries = sorted(panel["country"].unique())

    raw = pd.read_csv(GROUND_TRUTH_PATH)
    raw["Year"] = raw["Year"].astype(int)
    raw = raw[raw["Country"].isin(countries)].copy()
    raw = raw.sort_values(["Country", "Year"])
    for source, destination in COMPONENT_COLUMNS.items():
        raw[destination] = raw.groupby("Country", sort=False)[source].shift(1)

    anchors = raw[["Country", "Year", *FEATURES]].rename(columns={"Country": "country"})
    out = panel[["country", "country_slug", "Year", "delta_prs"]].merge(
        anchors, on=["country", "Year"], how="left", validate="one_to_one"
    )
    if out[FEATURES].isna().any().any():
        missing = out.loc[out[FEATURES].isna().any(axis=1), ["country", "Year"]]
        raise RuntimeError(f"Missing prior-year components:\n{missing.to_string(index=False)}")
    return out


def load_canonical() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    summary = payload["nowcast_results"]["delta_prs"]["12"]["BayesianRidge"]
    rows = [
        row
        for row in payload["row_details"]
        if row["target"] == "delta_prs"
        and int(row["cutoff_m"]) == 12
        and row["model"] == "BayesianRidge"
    ]
    return summary, rows


def main() -> None:
    panel = load_panel()
    eligible_years = sorted(panel.loc[panel["delta_prs"].notna(), "Year"].unique())
    test_years = [year for year in eligible_years if year >= eligible_years[0] + MIN_TRAIN_YEARS]

    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for test_year in test_years:
        train = panel[(panel["Year"] < test_year) & panel["delta_prs"].notna()].copy()
        test = panel[(panel["Year"] == test_year) & panel["delta_prs"].notna()].copy()
        model = _pipeline()
        model.fit(train[FEATURES], train["delta_prs"])
        predicted = model.predict(test[FEATURES])
        actual = test["delta_prs"].to_numpy(dtype=float)
        persistence = np.zeros(len(test), dtype=float)
        folds.append(
            {
                "year": int(test_year),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "rmse": _rmse(actual, predicted),
                "persistence_rmse": _rmse(actual, persistence),
            }
        )
        for index, (_, record) in enumerate(test.iterrows()):
            rows.append(
                {
                    "country": record["country"],
                    "country_slug": record["country_slug"],
                    "year": int(test_year),
                    "actual": float(actual[index]),
                    "predicted": float(predicted[index]),
                    "persistence": 0.0,
                }
            )

    if len(rows) != 350 or len({row["country_slug"] for row in rows}) != 25:
        raise RuntimeError("Anchor ablation does not reproduce the canonical 350-row, 25-country panel")

    actual = np.asarray([row["actual"] for row in rows])
    predicted = np.asarray([row["predicted"] for row in rows])
    persistence = np.asarray([row["persistence"] for row in rows])
    country_ids = np.asarray([row["country_slug"] for row in rows], dtype=object)
    year_ids = np.asarray([row["year"] for row in rows], dtype=int)
    loss_model = (actual - predicted) ** 2
    loss_persistence = (actual - persistence) ** 2
    differential = loss_model - loss_persistence
    bootstrap_rows = [
        {
            "country": row["country_slug"],
            "year": row["year"],
            "model_loss": float(loss_model[index]),
            "persistence_loss": float(loss_persistence[index]),
        }
        for index, row in enumerate(rows)
    ]

    country_results = []
    for country, group in pd.DataFrame(rows).groupby("country", sort=True):
        country_results.append(
            {
                "country": country,
                "n": int(len(group)),
                "rmse_anchor": _rmse(group["actual"], group["predicted"]),
                "rmse_persistence": _rmse(group["actual"], group["persistence"]),
            }
        )
    for result in country_results:
        result["anchor_wins"] = result["rmse_anchor"] < result["rmse_persistence"]

    canonical_summary, canonical_rows = load_canonical()
    canonical_actual = np.asarray([float(row["actual"]) for row in canonical_rows])
    canonical_predicted = np.asarray([float(row["predicted"]) for row in canonical_rows])
    canonical_persistence = np.asarray([float(row["persistence"]) for row in canonical_rows])
    if [(row["country_slug"], row["year"]) for row in rows] != [
        (row["country_slug"], row["year"]) for row in canonical_rows
    ]:
        raise RuntimeError("Anchor and canonical row keys do not align exactly")

    summary = {
        "design": {
            "model": "BayesianRidge",
            "target": "delta_prs",
            "features": FEATURES,
            "feature_scope": "prior-year ICRG component scores only",
            "excluded": ["monthly LLM signals", "disagreement features", "prs_lag1 composite"],
            "min_train_years": MIN_TRAIN_YEARS,
            "test_years": [int(year) for year in test_years],
            "n_rows": len(rows),
            "n_countries": int(len(np.unique(country_ids))),
        },
        "anchor_ablation": {
            "fold_mean_rmse": float(np.mean([fold["rmse"] for fold in folds])),
            "fold_mean_persistence_rmse": float(np.mean([fold["persistence_rmse"] for fold in folds])),
            "pooled_row_rmse": _rmse(actual, predicted),
            "pooled_row_persistence_rmse": _rmse(actual, persistence),
            "per_country_wins": int(sum(result["anchor_wins"] for result in country_results)),
            "per_country_total": len(country_results),
        },
        "canonical_m12": {
            "fold_mean_rmse": float(canonical_summary["rmse"]),
            "fold_mean_persistence_rmse": float(canonical_summary["persistence_rmse"]),
            "pooled_row_rmse": _rmse(canonical_actual, canonical_predicted),
            "pooled_row_persistence_rmse": _rmse(canonical_actual, canonical_persistence),
        },
        "comparisons": {
            "anchor_minus_canonical_fold_mean_rmse": float(
                np.mean([fold["rmse"] for fold in folds]) - float(canonical_summary["rmse"])
            ),
            "anchor_minus_canonical_pooled_row_rmse": float(
                _rmse(actual, predicted) - _rmse(canonical_actual, canonical_predicted)
            ),
        },
        "panel_inference_anchor_vs_persistence": {
            "loss_differential": "anchor squared error minus persistence squared error",
            "hln": diebold_mariano_hln(differential, h=1),
            "country_clustered": diebold_mariano_clustered(differential, country_ids),
            "two_way_country_year_clustered": diebold_mariano_two_way_clustered(
                differential, country_ids, year_ids
            ),
            "country_block_bootstrap": bootstrap_country_block(
                bootstrap_rows, n_resamples=N_BOOT, seed=SEED
            ),
            "year_block_bootstrap": bootstrap_year_block(
                bootstrap_rows, n_resamples=N_BOOT, seed=SEED
            ),
        },
        "folds": folds,
        "country_results": country_results,
        "row_details": rows,
        "sources": {
            "panel": str(PANEL_PATH.relative_to(REPO_ROOT)),
            "ground_truth": str(GROUND_TRUTH_PATH.relative_to(REPO_ROOT)),
            "canonical_nowcast": str(CANONICAL_PATH.relative_to(REPO_ROOT)),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "anchor_ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "anchor_ablation_predictions.csv", index=False)
    pd.DataFrame(country_results).to_csv(OUT_DIR / "anchor_ablation_by_country.csv", index=False)

    anchor = summary["anchor_ablation"]
    canonical = summary["canonical_m12"]
    inference = summary["panel_inference_anchor_vs_persistence"]
    markdown = f"""# Prior-year component anchor-only ablation

This ablation uses Bayesian Ridge with only the twelve ICRG component scores from year `t-1`. It uses the same 25 countries, 14 test years (2011--2024), 350 test rows, and expanding-window `min_train=5` folds as the frozen canonical m=12 nowcast. Monthly LLM signals, disagreement features, and the prior-year composite are excluded.

| Estimand | Anchor only | Persistence | Canonical m=12 |
|---|---:|---:|---:|
| Mean of annual-fold RMSEs | {anchor['fold_mean_rmse']:.4f} | {anchor['fold_mean_persistence_rmse']:.4f} | {canonical['fold_mean_rmse']:.4f} |
| Pooled row-level RMSE | {anchor['pooled_row_rmse']:.4f} | {anchor['pooled_row_persistence_rmse']:.4f} | {canonical['pooled_row_rmse']:.4f} |

The anchor-only model beats persistence in {anchor['per_country_wins']}/{anchor['per_country_total']} country-level RMSE comparisons. Its two-way country-year clustered loss test against persistence gives p={inference['two_way_country_year_clustered']['p_value']}. The country-block bootstrap gives p={inference['country_block_bootstrap']['p_value']}, and the year-block bootstrap gives p={inference['year_block_bootstrap']['p_value']}. These tests concern the anchor-only model versus persistence, not the difference between anchor-only and the canonical nowcast.

The canonical 1.5722 value is a mean of annual-fold RMSEs, whereas its pooled row-level RMSE is {canonical['pooled_row_rmse']:.4f}. Both estimands are reported to avoid mixing aggregation conventions.
"""
    (OUT_DIR / "anchor_ablation.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
