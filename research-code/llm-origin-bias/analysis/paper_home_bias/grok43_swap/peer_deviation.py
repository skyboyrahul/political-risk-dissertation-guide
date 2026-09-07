"""Peer-deviation diagnostic for the Grok 4.3 substitution panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper_home_bias._vendor.dashboard_data import scan_signals

OUT_DIR = ROOT / "artifacts" / "analysis" / "paper_home_bias_grok43_swap"
COUNTRIES_YAML = ROOT / "config" / "countries.yaml"
MISMATCH_PATH = ROOT / "logs" / "runs" / "grok43_input_hash_mismatches.json"
GROK43_HYBRID = "xai_grok43_hybrid"
GROK43_SOURCES = ["xai_grok43_batch", "xai_grok43_realtime"]

CORE_MODELS_CANONICAL = ["deepseek_deepseekv32", "minimax_m27", "xai_grok41fast", "gpt54"]
CORE_MODELS_GROK43 = ["deepseek_deepseekv32", "minimax_m27", "xai_grok43_hybrid", "gpt54"]
MODEL_ORIGIN_GROK43 = {
    "deepseek_deepseekv32": "CN",
    "minimax_m27": "CN",
    "xai_grok43_hybrid": "US",
    "gpt54": "US",
}
MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "minimax_m27": "MiniMax M2.7",
    "xai_grok41fast": "Grok 4.1 Fast",
    "xai_grok43_hybrid": "Grok 4.3",
    "gpt54": "GPT-5.4",
}
HOME_COUNTRY = {"US": "united_states", "CN": "china"}


def _countries() -> list[str]:
    payload = yaml.safe_load(COUNTRIES_YAML.read_text(encoding="utf-8"))
    return [row["slug"] for row in payload["countries"]]


def _load_scores(models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in _countries():
        scan_models = sorted({source for model in models for source in ([model] if model != GROK43_HYBRID else GROK43_SOURCES)})
        signals = scan_signals(country, scan_models)
        for model in models:
            if model == GROK43_HYBRID:
                model_signals = {}
                for source in GROK43_SOURCES:
                    model_signals.update(signals.get(source, {}))
            else:
                model_signals = signals.get(model, {})
            for month_key, row in model_signals.items():
                year = int(month_key[:4])
                if year < 2005 or year > 2024:
                    continue
                if row.get("status") != "valid":
                    continue
                score = row.get("total_prs")
                if not isinstance(score, (int, float)):
                    continue
                rows.append(
                    {
                        "country": country,
                        "month": month_key,
                        "model": model,
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def _mismatched_country_years() -> set[tuple[str, int]]:
    """Return no exclusions after the digest differences were cleared as input-equivalent."""
    return set()


def _fit_peer_deviation(
    models: list[str],
    model_origin: dict[str, str],
    excluded_country_years: set[tuple[str, int]] | None = None,
) -> dict:
    scores = _load_scores(models)
    excluded_country_years = excluded_country_years or set()
    if excluded_country_years:
        scores["year"] = scores["month"].str[:4].astype(int)
        mask = [
            (country, year) not in excluded_country_years
            for country, year in zip(scores["country"], scores["year"], strict=True)
        ]
        scores = scores.loc[mask].drop(columns=["year"])
    wide = (
        scores.pivot_table(
            index=["country", "month"],
            columns="model",
            values="score",
            aggfunc="mean",
        )
        .dropna(subset=models)
        .sort_index()
    )
    results = {}
    for model in models:
        peers = [peer for peer in models if peer != model]
        frame = wide.copy()
        frame["peer_deviation"] = frame[model] - frame[peers].mean(axis=1)
        frame = frame.reset_index()
        home_country = HOME_COUNTRY[model_origin[model]]
        frame["home_country"] = (frame["country"] == home_country).astype(int)
        fit = smf.ols("peer_deviation ~ home_country", data=frame).fit()
        clustered = fit.get_robustcov_results(cov_type="cluster", groups=frame["country"])
        ci_low, ci_high = clustered.conf_int(alpha=0.05)[1]
        results[model] = {
            "label": MODEL_LABELS[model],
            "origin": model_origin[model],
            "home_country": home_country,
            "beta": float(clustered.params[1]),
            "se_clustered": float(clustered.bse[1]),
            "t": float(clustered.tvalues[1]),
            "p_raw": float(clustered.pvalues[1]),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "n_country_months": int(frame.shape[0]),
            "n_countries": int(frame["country"].nunique()),
            "completed_countries_in_regression": sorted(frame["country"].unique().tolist()),
        }
    return {
        "models": models,
        "n_common_country_months": int(wide.shape[0]),
        "n_common_countries": int(wide.reset_index()["country"].nunique()),
        "common_countries": sorted(wide.reset_index()["country"].unique().tolist()),
        "excluded_country_years": [
            {"country": country, "year": year}
            for country, year in sorted(excluded_country_years)
        ],
        "per_model": results,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_origin = {
        "deepseek_deepseekv32": "CN",
        "minimax_m27": "CN",
        "xai_grok41fast": "US",
        "gpt54": "US",
    }
    payload = {
        "method": "per_model_peer_deviation_home_effect",
        "definition": (
            "For each rater and country-month, peer_deviation equals that rater's score "
            "minus the mean score of the other three raters in the same country-month. "
            "For each rater, peer_deviation is regressed on that rater-origin's home-country "
            "indicator with country-clustered standard errors."
        ),
        "canonical_reference": _fit_peer_deviation(CORE_MODELS_CANONICAL, canonical_origin),
        "grok43_substitution": _fit_peer_deviation(
            CORE_MODELS_GROK43,
            MODEL_ORIGIN_GROK43,
            excluded_country_years=_mismatched_country_years(),
        ),
        "notes": [
            "The canonical CORE_MODELS constant in analysis.paper_home_bias.common is not modified.",
            "The Grok 4.3 regression uses the complete 6,000-cell panel shared by all four substituted raters.",
        ],
    }
    out = OUT_DIR / "peer_deviation_grok43_swap.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
