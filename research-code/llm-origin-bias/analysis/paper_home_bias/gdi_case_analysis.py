#!/usr/bin/env python3
"""Analyse four-rater disagreement and render the thesis diagnostic figure.

The Geopolitical Divergence Index (GDI) is defined here as the sample standard
deviation of the four canonical model composite scores in a country-month.  The
script uses the same four-model panel as the origin analysis, summarises the
largest and zero-disagreement cells, and repeats the Wikipedia coverage-block
test on annual mean GDI over the complete 500-country-year panel.

Outputs:
    artifacts/analysis/home_bias_paper/gdi_case_analysis.json
    papers/origin-bias/figures/gdi_case_diagnostic.pdf
    papers/origin-bias/figures/gdi_case_diagnostic.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper_home_bias._vendor.thesis_figure_style import (  # noqa: E402
    CB_PALETTE,
    apply_paper_style,
)
from analysis.paper_home_bias.common import (  # noqa: E402
    ANALYSIS_DIR,
    CONTROL_COLUMNS,
    CORE_MODELS,
    WIKIPEDIA_PANEL,
    load_panel,
    write_json,
)

FIG_DIR = ROOT / "papers" / "origin-bias" / "figures"
OUT_JSON = ANALYSIS_DIR / "gdi_case_analysis.json"
OUT_PDF = FIG_DIR / "gdi_case_diagnostic.pdf"
OUT_PNG = FIG_DIR / "gdi_case_diagnostic.png"

MODEL_ORDER = [
    "deepseek_deepseekv32",
    "gpt54",
    "xai_grok41fast",
    "minimax_m27",
]
MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "gpt54": "GPT-5.4",
    "xai_grok41fast": "Grok 4.1 Fast",
    "minimax_m27": "MiniMax M2.7",
}
MODEL_MARKERS = {
    "deepseek_deepseekv32": "o",
    "gpt54": "s",
    "xai_grok41fast": "^",
    "minimax_m27": "D",
}
MODEL_COLOURS = {
    "deepseek_deepseekv32": CB_PALETTE[4],
    "gpt54": CB_PALETTE[0],
    "xai_grok41fast": CB_PALETTE[2],
    "minimax_m27": CB_PALETTE[5],
}
EXPECTED_TOP_THREE = [
    ("egypt", "2024-02"),
    ("israel", "2022-06"),
    ("philippines", "2024-09"),
]


def build_monthly_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the complete monthly GDI panel and its four score columns."""
    scores = load_panel()
    wide = scores.pivot(index=["country", "year", "month"], columns="model", values="score")

    if wide.shape != (6000, 4):
        raise AssertionError(f"Expected 6,000 country-months by four models, found {wide.shape}")
    if set(wide.columns) != set(CORE_MODELS):
        raise AssertionError(f"Unexpected model panel: {sorted(wide.columns)}")
    if wide.isna().any().any():
        raise AssertionError("The four-rater panel contains missing scores")

    wide = wide.loc[:, MODEL_ORDER]
    monthly = wide.reset_index()
    monthly["date"] = pd.to_datetime(monthly["month"] + "-01")
    monthly["gdi"] = wide.std(axis=1, ddof=1).to_numpy(dtype=float)
    monthly["consensus_score"] = wide.mean(axis=1).to_numpy(dtype=float)
    monthly = monthly.sort_values(["date", "country"]).reset_index(drop=True)
    return monthly, scores


def run_coverage_test(monthly: pd.DataFrame) -> dict:
    """Test whether four Wikipedia coverage measures jointly explain annual GDI."""
    annual = (
        monthly.groupby(["country", "year"], as_index=False)
        .agg(
            gdi_annual_mean=("gdi", "mean"),
            gdi_annual_max=("gdi", "max"),
            n_months=("gdi", "size"),
        )
    )
    coverage = pd.read_csv(WIKIPEDIA_PANEL)
    merged = annual.merge(coverage, on=["country", "year"], how="left", validate="one_to_one")
    if merged.shape[0] != 500 or merged[CONTROL_COLUMNS].isna().any().any():
        raise AssertionError("Coverage test must contain 500 complete country-years")

    control_terms: list[str] = []
    for column in CONTROL_COLUMNS:
        term = f"{column}_z"
        centred = merged[column].astype(float) - float(merged[column].mean())
        scale = float(merged[column].std(ddof=0))
        merged[term] = centred / scale
        control_terms.append(term)

    formula = (
        "gdi_annual_mean ~ "
        + " + ".join(control_terms)
        + " + C(country, Treatment(reference='argentina'))"
        + " + C(year, Treatment(reference=2005))"
    )
    fit = smf.ols(formula, data=merged).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["country"]},
    )
    hypothesis = ", ".join(f"{term} = 0" for term in control_terms)
    joint = fit.wald_test(hypothesis, scalar=True)

    return {
        "definition": "Annual mean of monthly sample standard deviation across four model composite scores.",
        "formula": formula,
        "controls": CONTROL_COLUMNS,
        "n_obs": int(fit.nobs),
        "n_countries": int(merged["country"].nunique()),
        "n_years": int(merged["year"].nunique()),
        "coverage_joint_test": {
            "statistic": float(joint.statistic),
            "p_value": float(joint.pvalue),
            "hypothesis": hypothesis,
        },
        "control_coefficients": {
            column: {
                "coef": float(fit.params[f"{column}_z"]),
                "std_err": float(fit.bse[f"{column}_z"]),
                "p_value": float(fit.pvalues[f"{column}_z"]),
            }
            for column in CONTROL_COLUMNS
        },
    }


def _month_record(row: pd.Series, scores: pd.DataFrame) -> dict:
    subset = scores.loc[
        (scores["country"] == row["country"]) & (scores["month"] == row["month"])
    ].set_index("model")
    return {
        "country": str(row["country"]),
        "month": str(row["month"]),
        "gdi": float(row["gdi"]),
        "consensus_score": float(row["consensus_score"]),
        "model_scores": {model: float(row[model]) for model in MODEL_ORDER},
        "source_paths": {model: str(subset.loc[model, "source_path"]) for model in MODEL_ORDER},
    }


def build_payload(monthly: pd.DataFrame, scores: pd.DataFrame, coverage: dict) -> dict:
    top = monthly.nlargest(10, "gdi").copy()
    observed_top_three = list(zip(top.head(3)["country"], top.head(3)["month"], strict=True))
    if observed_top_three != EXPECTED_TOP_THREE:
        raise AssertionError(f"Unexpected top-three GDI cells: {observed_top_three}")

    zero = monthly.loc[np.isclose(monthly["gdi"], 0.0)].sort_values(
        ["consensus_score", "country", "month"]
    )
    if len(zero) != 21:
        raise AssertionError(f"Expected 21 zero-GDI months, found {len(zero)}")

    score_frame = monthly.set_index(["country", "year", "month"])[MODEL_ORDER]
    row_max = score_frame.max(axis=1)
    rater_positions = {}
    for model in MODEL_ORDER:
        peers = [peer for peer in MODEL_ORDER if peer != model]
        peer_deviation = score_frame[model] - score_frame[peers].mean(axis=1)
        rater_positions[model] = {
            "label": MODEL_LABELS[model],
            "mean_score": float(score_frame[model].mean()),
            "mean_peer_deviation": float(peer_deviation.mean()),
            "share_tied_for_highest_score": float(score_frame[model].eq(row_max).mean()),
            "share_unique_highest_score": float(
                score_frame[model].gt(score_frame[peers].max(axis=1)).mean()
            ),
        }

    return {
        "definition": {
            "name": "Geopolitical Divergence Index",
            "status": "Study-specific term introduced for this analysis, not an established external index.",
            "calculation": "Sample standard deviation (ddof=1) of the four model composite scores in each country-month.",
            "units": "PRS points",
            "models": MODEL_LABELS,
        },
        "panel": {
            "n_country_months": int(len(monthly)),
            "n_countries": int(monthly["country"].nunique()),
            "n_months_per_country": int(monthly.groupby("country").size().iloc[0]),
            "year_range": [int(monthly["year"].min()), int(monthly["year"].max())],
            "mean_gdi": float(monthly["gdi"].mean()),
            "median_gdi": float(monthly["gdi"].median()),
            "p95_gdi": float(monthly["gdi"].quantile(0.95)),
            "max_gdi": float(monthly["gdi"].max()),
            "n_zero_gdi": int(len(zero)),
            "zero_gdi_consensus_score_range": [
                float(zero["consensus_score"].min()),
                float(zero["consensus_score"].max()),
            ],
        },
        "top_months": [_month_record(row, scores) for _, row in top.iterrows()],
        "zero_gdi_months": [_month_record(row, scores) for _, row in zero.iterrows()],
        "rater_positions": rater_positions,
        "coverage_test": coverage,
        "figure": str(OUT_PDF.relative_to(ROOT)),
    }


def render_figure(monthly: pd.DataFrame) -> None:
    apply_paper_style()
    fig = plt.figure(figsize=(7.25, 3.25))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.60, 1.0], wspace=0.34)
    left = outer[0].subgridspec(1, 3, width_ratios=[1.0, 0.16, 0.055], wspace=0.08)
    ax_all = fig.add_subplot(left[0])
    ax_marginal = fig.add_subplot(left[1], sharey=ax_all)
    ax_colourbar = fig.add_subplot(left[2])
    ax_scores = fig.add_subplot(outer[1])

    blues = plt.get_cmap("Blues")
    visible_blues = LinearSegmentedColormap.from_list(
        "visible_blues",
        blues(np.linspace(0.22, 0.95, 256)),
    )

    density = ax_all.hexbin(
        mdates.date2num(monthly["date"]),
        monthly["gdi"],
        gridsize=(54, 26),
        mincnt=1,
        cmap=visible_blues,
        norm=LogNorm(vmin=1, vmax=20),
        linewidths=0,
        rasterized=True,
    )
    density_bar = fig.colorbar(density, cax=ax_colourbar)
    density_bar.set_ticks([1, 3, 10, 20])
    density_bar.set_ticklabels(["1", "3", "10", "20"])
    density_bar.minorticks_off()
    density_bar.set_label("Country-months per hexagon", fontsize=6.7, labelpad=3)
    density_bar.ax.tick_params(labelsize=6.5)

    p95 = float(monthly["gdi"].quantile(0.95))
    ax_all.axhline(p95, color="#666666", linestyle=":", linewidth=0.7)
    ax_all.text(
        pd.Timestamp("2005-03-01"),
        p95 + 0.16,
        f"95th percentile = {p95:.1f}",
        fontsize=7,
        color="#555555",
        va="bottom",
    )

    top_three = monthly.nlargest(3, "gdi").copy()
    for rank, (_, row) in enumerate(top_three.iterrows(), start=1):
        ax_all.scatter(
            row["date"],
            row["gdi"],
            s=24,
            facecolor="#252525",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax_all.annotate(
            str(rank),
            xy=(row["date"], row["gdi"]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=7.0,
            color="#252525",
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=5,
        )

    histogram_bins = np.arange(0.0, 12.51, 0.5)
    ax_marginal.hist(
        monthly["gdi"],
        bins=histogram_bins,
        orientation="horizontal",
        color="#a6a6a6",
        edgecolor="white",
        linewidth=0.25,
    )
    ax_marginal.axhline(p95, color="#666666", linestyle=":", linewidth=0.7)
    ax_marginal.set_xlim(0, 950)
    ax_marginal.set_xticks([0, 800])
    ax_marginal.set_xlabel("Count", fontsize=6.7, labelpad=2)
    ax_marginal.tick_params(axis="x", labelsize=6.2)
    ax_marginal.tick_params(axis="y", left=False, labelleft=False)
    ax_marginal.grid(False)
    ax_marginal.spines["left"].set_visible(False)
    ax_marginal.spines["top"].set_visible(False)
    ax_marginal.spines["right"].set_visible(False)

    ax_all.set_title("(a) Monthly GDI across the panel", loc="left")
    ax_all.set_ylabel("GDI (standard deviation, PRS points)")
    ax_all.set_xlabel("Year")
    ax_all.set_ylim(-0.35, 12.25)
    ax_all.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_all.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_all.grid(axis="y", alpha=0.18)

    case_labels = [
        "1\nEgypt\nFeb 2024",
        "2\nIsrael\nJun 2022",
        "3\nPhilippines\nSep 2024",
    ]
    x = np.arange(len(top_three), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(MODEL_ORDER))
    for case_index, (_, row) in enumerate(top_three.iterrows()):
        values = [float(row[model]) for model in MODEL_ORDER]
        for offset, model, value in zip(offsets, MODEL_ORDER, values, strict=True):
            ax_scores.scatter(
                case_index + offset,
                value,
                marker=MODEL_MARKERS[model],
                s=31,
                color=MODEL_COLOURS[model],
                edgecolor="white",
                linewidth=0.5,
                label=MODEL_LABELS[model] if case_index == 0 else None,
                zorder=3,
            )

    ax_scores.set_title("(b) Model scores at the three peaks", loc="left")
    ax_scores.set_ylabel("Composite score (PRS points)")
    ax_scores.set_xticks(x)
    ax_scores.set_xticklabels(case_labels)
    ax_scores.tick_params(axis="x", labelsize=6.5)
    ax_scores.set_ylim(45, 91)
    ax_scores.grid(axis="y", alpha=0.18)
    ax_scores.legend(loc="upper left", ncol=2, fontsize=6.4, handletextpad=0.3, columnspacing=0.7)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.91)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    monthly, scores = build_monthly_panel()
    coverage = run_coverage_test(monthly)
    payload = build_payload(monthly, scores, coverage)
    render_figure(monthly)
    write_json(OUT_JSON, payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(
        "Top three:",
        [(row["country"], row["month"], round(row["gdi"], 4)) for row in payload["top_months"][:3]],
    )
    print(
        "Coverage joint test:",
        f"n={coverage['n_obs']}, p={coverage['coverage_joint_test']['p_value']:.6f}",
    )


if __name__ == "__main__":
    main()
