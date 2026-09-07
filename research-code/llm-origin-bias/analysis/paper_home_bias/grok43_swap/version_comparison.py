"""Apples-to-apples comparison of Grok 4.1 Fast vs Grok 4.3 home-bias signature.

Restricts both Grok versions to the same matched country-month panel where all
five models have valid scores. Reports per-Grok-version peer-deviation home-effect regressions, a
GPT-5.4 sanity check under both panel definitions, descriptive cell-level deltas,
and annual mean delta trajectories for US and China.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper_home_bias._vendor.dashboard_data import scan_signals

OUT_DIR = ROOT / "artifacts" / "analysis" / "paper_home_bias_grok43_swap"
COUNTRIES_YAML = ROOT / "config" / "countries.yaml"
MISMATCH_PATH = ROOT / "logs" / "runs" / "grok43_input_hash_mismatches.json"
GROK43_HYBRID = "xai_grok43_hybrid"
GROK43_SOURCES = ["xai_grok43_batch", "xai_grok43_realtime"]

ALL_MODELS = [
    "deepseek_deepseekv32",
    "minimax_m27",
    "xai_grok41fast",
    "xai_grok43_hybrid",
    "gpt54",
]

MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "minimax_m27": "MiniMax M2.7",
    "xai_grok41fast": "Grok 4.1 Fast",
    "xai_grok43_hybrid": "Grok 4.3",
    "gpt54": "GPT-5.4",
}

PEERS_FOR_GROK = ["deepseek_deepseekv32", "minimax_m27", "gpt54"]
HOME_COUNTRY_US = "united_states"
HOME_COUNTRY_CN = "china"


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
    """Return no exclusions.

    The historical register records loader-level digest differences that were
    subsequently shown not to represent differences in the evidence supplied
    to the two Grok versions. It is retained for traceability but no longer
    defines the comparison sample.
    """
    return set()


def _build_matched_panel(
    scores: pd.DataFrame,
    models: list[str],
    excluded_country_years: set[tuple[str, int]],
) -> pd.DataFrame:
    df = scores.copy()
    df["year"] = df["month"].str[:4].astype(int)
    if excluded_country_years:
        mask = [
            (country, year) not in excluded_country_years
            for country, year in zip(df["country"], df["year"], strict=True)
        ]
        df = df.loc[mask]
    wide = (
        df.pivot_table(
            index=["country", "month", "year"],
            columns="model",
            values="score",
            aggfunc="mean",
        )
        .dropna(subset=models)
        .sort_index()
    )
    return wide


def _fit_peer_deviation_on_panel(
    wide: pd.DataFrame,
    focal: str,
    peers: list[str],
    home_country: str,
) -> dict:
    frame = wide.copy()
    frame["peer_deviation"] = frame[focal] - frame[peers].mean(axis=1)
    frame = frame.reset_index()
    frame["home_country"] = (frame["country"] == home_country).astype(int)
    fit = smf.ols("peer_deviation ~ home_country", data=frame).fit()
    clustered = fit.get_robustcov_results(cov_type="cluster", groups=frame["country"])
    ci_low, ci_high = clustered.conf_int(alpha=0.05)[1]
    return {
        "focal_model": focal,
        "focal_label": MODEL_LABELS[focal],
        "peers": peers,
        "peer_labels": [MODEL_LABELS[m] for m in peers],
        "home_country": home_country,
        "beta": float(clustered.params[1]),
        "se_clustered": float(clustered.bse[1]),
        "t": float(clustered.tvalues[1]),
        "p_raw": float(clustered.pvalues[1]),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n_country_months": int(frame.shape[0]),
        "n_countries": int(frame["country"].nunique()),
    }


def _cell_level_deltas(wide: pd.DataFrame) -> dict:
    frame = wide.copy().reset_index()
    frame["delta"] = frame["xai_grok43_hybrid"] - frame["xai_grok41fast"]

    overall_mean = float(frame["delta"].mean())
    us = frame.loc[frame["country"] == HOME_COUNTRY_US, "delta"]
    cn = frame.loc[frame["country"] == HOME_COUNTRY_CN, "delta"]
    non_home = frame.loc[~frame["country"].isin([HOME_COUNTRY_US, HOME_COUNTRY_CN]), "delta"]

    us_n = int(us.shape[0])
    us_mean = float(us.mean()) if us_n else float("nan")
    if us_n > 1:
        sem_us = float(stats.sem(us.to_numpy()))
        h = sem_us * stats.t.ppf(0.975, us_n - 1)
        us_ci = [us_mean - h, us_mean + h]
    else:
        us_ci = [float("nan"), float("nan")]

    per_country = (
        frame.groupby("country")["delta"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "count": "n_country_months"})
    )
    per_country = per_country.reindex(
        per_country["mean_delta"].abs().sort_values(ascending=False).index
    )
    per_country_list = [
        {
            "country": str(row["country"]),
            "mean_delta": float(row["mean_delta"]),
            "n_country_months": int(row["n_country_months"]),
        }
        for _, row in per_country.iterrows()
    ]

    return {
        "overall_mean": overall_mean,
        "us_cell_delta_mean": us_mean,
        "us_cell_delta_n": us_n,
        "us_cell_delta_ci95": us_ci,
        "china_cell_delta_mean": float(cn.mean()) if cn.shape[0] else float("nan"),
        "china_cell_delta_n": int(cn.shape[0]),
        "non_home_cell_delta_mean": float(non_home.mean()) if non_home.shape[0] else float("nan"),
        "non_home_cell_delta_n": int(non_home.shape[0]),
        "per_country_sorted_by_abs_delta": per_country_list,
    }


def _annual_trajectories(wide: pd.DataFrame) -> dict:
    frame = wide.copy().reset_index()
    annual = (
        frame.groupby(["country", "year"])
        .agg(
            grok41_annual_mean=("xai_grok41fast", "mean"),
            grok43_annual_mean=("xai_grok43_hybrid", "mean"),
            n_months=("xai_grok41fast", "size"),
        )
        .reset_index()
    )
    annual["annual_delta"] = annual["grok43_annual_mean"] - annual["grok41_annual_mean"]

    def _country_series(slug: str) -> list[dict]:
        sub = annual.loc[annual["country"] == slug].sort_values("year")
        return [
            {
                "year": int(row["year"]),
                "grok41_annual_mean": float(row["grok41_annual_mean"]),
                "grok43_annual_mean": float(row["grok43_annual_mean"]),
                "annual_delta": float(row["annual_delta"]),
                "n_months": int(row["n_months"]),
            }
            for _, row in sub.iterrows()
        ]

    countries_in_panel = sorted(annual["country"].unique().tolist())
    return {
        "us_trajectory": _country_series(HOME_COUNTRY_US),
        "china_trajectory": _country_series(HOME_COUNTRY_CN),
        "countries_in_panel": countries_in_panel,
        "all_country_year_deltas": [
            {
                "country": str(row["country"]),
                "year": int(row["year"]),
                "annual_delta": float(row["annual_delta"]),
                "grok41_annual_mean": float(row["grok41_annual_mean"]),
                "grok43_annual_mean": float(row["grok43_annual_mean"]),
                "n_months": int(row["n_months"]),
            }
            for _, row in annual.sort_values(["country", "year"]).iterrows()
        ],
    }


def _format_md(payload: dict) -> str:
    matched = payload["matched_panel"]
    reg = payload["regression"]
    desc = payload["descriptive_deltas"]
    sanity = payload["gpt54_sanity_check"]

    def reg_row(label: str, r: dict) -> str:
        return (
            f"| {label} | {r['beta']:+.4f} | {r['se_clustered']:.4f} | "
            f"{r['t']:+.3f} | {r['p_raw']:.4g} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['n_country_months']} |"
        )

    lines = [
        "# Grok 4.1 Fast vs Grok 4.3: matched-panel home-bias comparison",
        "",
        "## Matched panel definition",
        "",
        f"- Country-months where all five models (DeepSeek V3.2, MiniMax M2.7, Grok 4.1 Fast, Grok 4.3, GPT-5.4) report a valid score.",
        "- Uses the complete panel; no country-years are excluded by the historical digest register.",
        f"- Resulting size: **{matched['n_country_months']} country-months** across "
        f"**{matched['n_countries']} countries**.",
        f"- Countries in panel: {', '.join(matched['countries'])}.",
        f"- Excluded (country, year) pairs: "
        + (
            "; ".join(f"{c}/{y}" for c, y in matched["excluded_country_years_list"])
            if matched["excluded_country_years_list"]
            else "(none)"
        )
        + ".",
        "",
        "## Headline regression (peer deviation ~ home-country indicator, country-clustered SE)",
        "",
        "Peer deviation is computed against the same three peer raters for both Grok versions: "
        "DeepSeek V3.2, MiniMax M2.7, GPT-5.4. Home country is the United States for both.",
        "",
        "| Focal model | beta | SE (clustered) | t | p | 95% CI | n |",
        "|---|---|---|---|---|---|---|",
        reg_row("Grok 4.1 Fast on US", reg["grok41_on_us"]),
        reg_row("Grok 4.3 on US", reg["grok43_on_us"]),
        "",
        "## GPT-5.4 sanity check",
        "",
        "Re-running the same regression for GPT-5.4 under both panel definitions of the fourth peer rater.",
        "",
        "| Panel (4th peer) | beta | SE (clustered) | t | p | 95% CI | n |",
        "|---|---|---|---|---|---|---|",
        reg_row("with Grok 4.1 Fast", sanity["gpt54_with_grok41"]),
        reg_row("with Grok 4.3", sanity["gpt54_with_grok43"]),
        "",
        "## Cell-level deltas (Grok 4.3 minus Grok 4.1 Fast)",
        "",
        f"- Overall mean delta: **{desc['overall_mean']:+.4f}** across {matched['n_country_months']} cells.",
        f"- US country-months (n={desc['us_cell_delta_n']}): mean = **{desc['us_cell_delta_mean']:+.4f}**, "
        f"paired-t 95% CI [{desc['us_cell_delta_ci95'][0]:+.4f}, {desc['us_cell_delta_ci95'][1]:+.4f}].",
        f"- China country-months (n={desc['china_cell_delta_n']}): mean = **{desc['china_cell_delta_mean']:+.4f}**.",
        f"- Non-home pooled (n={desc['non_home_cell_delta_n']}): mean = "
        f"**{desc['non_home_cell_delta_mean']:+.4f}**.",
        "",
        "### Per-country mean delta (sorted by absolute magnitude)",
        "",
        "| Country | mean delta | n country-months |",
        "|---|---|---|",
    ]
    for row in desc["per_country_sorted_by_abs_delta"]:
        lines.append(f"| {row['country']} | {row['mean_delta']:+.4f} | {row['n_country_months']} |")

    lines.extend(
        [
            "",
            "## Annual trajectory snapshot",
            "",
            "### United States (annual mean grok41 -> grok43, delta)",
            "",
            "| Year | grok41 | grok43 | delta |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["annual_trajectories"]["us_trajectory"]:
        lines.append(
            f"| {row['year']} | {row['grok41_annual_mean']:.3f} | "
            f"{row['grok43_annual_mean']:.3f} | {row['annual_delta']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "### China (annual mean grok41 -> grok43, delta)",
            "",
            "| Year | grok41 | grok43 | delta |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["annual_trajectories"]["china_trajectory"]:
        lines.append(
            f"| {row['year']} | {row['grok41_annual_mean']:.3f} | "
            f"{row['grok43_annual_mean']:.3f} | {row['annual_delta']:+.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = _load_scores(ALL_MODELS)
    excluded = _mismatched_country_years()
    wide = _build_matched_panel(scores, ALL_MODELS, excluded)

    matched_countries = sorted(wide.reset_index()["country"].unique().tolist())
    matched_panel = {
        "models": ALL_MODELS,
        "model_labels": [MODEL_LABELS[m] for m in ALL_MODELS],
        "n_country_months": int(wide.shape[0]),
        "n_countries": len(matched_countries),
        "countries": matched_countries,
        "excluded_country_years_list": [
            [c, y] for c, y in sorted(excluded)
        ],
    }

    # Headline regressions on matched panel.
    grok41_on_us = _fit_peer_deviation_on_panel(
        wide, "xai_grok41fast", PEERS_FOR_GROK, HOME_COUNTRY_US
    )
    grok43_on_us = _fit_peer_deviation_on_panel(
        wide, "xai_grok43_hybrid", PEERS_FOR_GROK, HOME_COUNTRY_US
    )

    # GPT-5.4 sanity check: build two separate matched panels (one with Grok 4.1
    # Fast as the fourth rater, one with Grok 4.3). Apply the same excluded
    # (country, year) pairs for symmetry. This isolates whether GPT-5.4's home
    # effect moves when the Grok version is swapped.
    panel_with_grok41 = _build_matched_panel(
        scores,
        ["deepseek_deepseekv32", "minimax_m27", "xai_grok41fast", "gpt54"],
        excluded,
    )
    panel_with_grok43 = _build_matched_panel(
        scores,
        ["deepseek_deepseekv32", "minimax_m27", "xai_grok43_hybrid", "gpt54"],
        excluded,
    )
    gpt54_with_grok41 = _fit_peer_deviation_on_panel(
        panel_with_grok41,
        "gpt54",
        ["deepseek_deepseekv32", "minimax_m27", "xai_grok41fast"],
        HOME_COUNTRY_US,
    )
    gpt54_with_grok43 = _fit_peer_deviation_on_panel(
        panel_with_grok43,
        "gpt54",
        ["deepseek_deepseekv32", "minimax_m27", "xai_grok43_hybrid"],
        HOME_COUNTRY_US,
    )

    # Descriptive deltas and annual trajectories on the matched 5-model panel.
    descriptive = _cell_level_deltas(wide)
    trajectories = _annual_trajectories(wide)

    payload = {
        "method": "matched_panel_grok_version_comparison",
        "definition": (
            "Matched panel = country-months where DeepSeek V3.2, MiniMax M2.7, Grok 4.1 Fast, "
            "Grok 4.3 and GPT-5.4 all report a valid score. For "
            "each Grok version, peer deviation = focal score minus mean(DeepSeek, MiniMax, "
            "GPT-5.4) in the same country-month. Peer deviation is regressed on a US-home "
            "indicator with country-clustered standard errors. GPT-5.4 is re-run as a sanity "
            "check on two parallel four-rater panels."
        ),
        "matched_panel": matched_panel,
        "regression": {
            "peers_for_grok": PEERS_FOR_GROK,
            "peer_labels_for_grok": [MODEL_LABELS[m] for m in PEERS_FOR_GROK],
            "home_country": HOME_COUNTRY_US,
            "grok41_on_us": grok41_on_us,
            "grok43_on_us": grok43_on_us,
        },
        "gpt54_sanity_check": {
            "gpt54_with_grok41": gpt54_with_grok41,
            "gpt54_with_grok43": gpt54_with_grok43,
            "panel_with_grok41_n_country_months": int(panel_with_grok41.shape[0]),
            "panel_with_grok43_n_country_months": int(panel_with_grok43.shape[0]),
        },
        "descriptive_deltas": descriptive,
        "annual_trajectories": trajectories,
        "notes": [
            "Peers for the Grok regressions are held constant across the two versions to make the comparison strictly within-cell.",
            "The historical digest register is retained for traceability but does not exclude cells because its differences do not represent different supplied evidence.",
            "GPT-5.4 sanity panels use different fourth-rater models, so country-month coverage and n can differ between the two GPT-5.4 rows.",
        ],
    }

    json_out = OUT_DIR / "version_comparison.json"
    md_out = OUT_DIR / "version_comparison.md"
    json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_out.write_text(_format_md(payload) + "\n", encoding="utf-8")
    print(json_out)
    print(md_out)


if __name__ == "__main__":
    main()
