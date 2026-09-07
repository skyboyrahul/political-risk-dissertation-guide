"""Model leave-one-out and pairwise rater decomposition."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from analysis.paper_home_bias.common import (
        ANALYSIS_DIR,
        CORE_MODELS,
        MODEL_ORIGIN,
        ROOT,
        ensure_dirs,
        format_p,
        load_panel,
        model_count,
        write_json,
        write_md,
    )
    from analysis.paper_home_bias.inference_utils import fit_payload, holm_two, prepare_design
    from analysis.paper_home_bias._vendor.thesis_figure_style import (
        CN_COLOUR,
        NULL_LINE,
        US_COLOUR,
        apply_paper_style,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from analysis.paper_home_bias.common import (  # type: ignore
        ANALYSIS_DIR,
        CORE_MODELS,
        MODEL_ORIGIN,
        ROOT,
        ensure_dirs,
        format_p,
        load_panel,
        model_count,
        write_json,
        write_md,
    )
    from inference_utils import fit_payload, holm_two, prepare_design  # type: ignore
    from analysis.paper_home_bias._vendor.thesis_figure_style import (  # type: ignore
        CN_COLOUR,
        NULL_LINE,
        US_COLOUR,
        apply_paper_style,
    )

FIG_DIR = ROOT / "papers" / "origin-bias" / "figures"
MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "minimax_m27": "MiniMax M2.7",
    "xai_grok41fast": "Grok 4.1 Fast",
    "gpt54": "GPT-5.4",
}


def _fit_country_year(df):
    design = prepare_design(df, fe_groups=("country_year", "model"))
    coeffs = fit_payload(design["y"], design["x"], design["clusters"], terms=design["terms"], cov_kind="cr0")
    holm = holm_two(coeffs["us_on_us"]["p_cr0"], coeffs["cn_on_cn"]["p_cr0"])
    out = {}
    for key, values in coeffs.items():
        out[key] = {
            "beta": values["beta"],
            "se_clustered": values["se_cr0"],
            "t": values["t_cr0"],
            "p_raw": values["p_cr0"],
            "p_holm_k2": holm[key],
            "p_bonferroni_k2": min(2.0 * values["p_cr0"], 1.0),
            "ci95_low": values["ci95_low_cr0"],
            "ci95_high": values["ci95_high_cr0"],
        }
    return out, design


def _pairwise_rows(df):
    rows = []
    focal = df[df["country"].isin(["united_states", "china"])]
    for left, right in combinations(CORE_MODELS, 2):
        left_origin = MODEL_ORIGIN[left]
        right_origin = MODEL_ORIGIN[right]
        row = {
            "left_model": left,
            "left_label": MODEL_LABELS[left],
            "left_origin": left_origin,
            "right_model": right,
            "right_label": MODEL_LABELS[right],
            "right_origin": right_origin,
            "pair_type": "cross_origin" if left_origin != right_origin else "same_origin_diagnostic",
        }
        for country in ("united_states", "china"):
            sub = focal[(focal["country"] == country) & (focal["model"].isin([left, right]))]
            by_month = sub.pivot_table(index="month", columns="model", values="score", aggfunc="mean")
            by_month = by_month.dropna(subset=[left, right])
            diff = by_month[left] - by_month[right]
            row[f"{country}_left_minus_right_mean"] = float(diff.mean())
            row[f"{country}_n_months"] = int(diff.shape[0])
        rows.append(row)
    return rows


def _add_contributions(full_coeffs, loo_rows, n_clusters: int) -> None:
    crit = float(stats.t.ppf(0.975, max(n_clusters - 1, 1)))
    for row in loo_rows:
        for key, loo_coeffs in row["coefficients"].items():
            full = full_coeffs[key]
            contribution = full["beta"] - loo_coeffs["beta"]
            se_delta = float(
                np.sqrt(full["se_clustered"] ** 2 + loo_coeffs["se_clustered"] ** 2)
            )
            loo_coeffs.update(
                {
                    "contribution": float(contribution),
                    "contribution_lower": float(contribution - crit * se_delta),
                    "contribution_upper": float(contribution + crit * se_delta),
                    "contribution_se_delta": se_delta,
                    "contribution_method": "delta-method using country-clustered SEs, zero covariance",
                }
            )


def _write_plot(full_coeffs, loo_rows) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    # Single-panel leave-one-out coefficient forest. The former panel (a)
    # ("contribution" = full-panel beta minus leave-one-out beta) was an affine
    # transform of this forest and carried a zero-covariance delta-method
    # interval that is not a valid SE for nested estimators; it is dropped.
    contrasts = [
        ("us_on_us", "US-on-US", US_COLOUR),
        ("cn_on_cn", "CN-on-CN", CN_COLOUR),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), sharex=False)
    for col, (key, title, colour) in enumerate(contrasts):
        loo_records = [
            {"label": "Full panel", **full_coeffs[key]},
            *[
                {
                    "label": f"Drop {MODEL_LABELS[row['dropped_model']]}",
                    **row["coefficients"][key],
                }
                for row in loo_rows
            ],
        ]
        ax = axes[col]
        y = np.arange(len(loo_records))
        beta = np.asarray([r["beta"] for r in loo_records])
        lo = np.asarray([r["ci95_low"] for r in loo_records])
        hi = np.asarray([r["ci95_high"] for r in loo_records])
        ax.axvline(0, color=NULL_LINE, linewidth=0.4, linestyle=":", label="Null")
        ax.errorbar(
            beta,
            y,
            xerr=[beta - lo, hi - beta],
            fmt="o",
            mfc="white",
            mec=colour,
            mew=1.0,
            color=colour,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.8,
            markersize=4.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in loo_records])
        ax.invert_yaxis()
        ax.set_title(title, color=colour)
        ax.set_xlabel("Home-on-home coefficient (ICRG points)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90, wspace=0.40)
    fig.savefig(FIG_DIR / "jackknife_forest.pdf")
    fig.savefig(FIG_DIR / "jackknife_forest.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load_panel()
    full_coeffs, full_design = _fit_country_year(df)
    loo_rows = []
    for model in CORE_MODELS:
        sub = df[df["model"] != model].copy()
        coeffs, design = _fit_country_year(sub)
        loo_rows.append({
            "dropped_model": model,
            "dropped_label": MODEL_LABELS[model],
            "dropped_origin": MODEL_ORIGIN[model],
            "n_obs": design["n_obs"],
            "n_models": model_count(sub),
            "coefficients": coeffs,
        })
    _add_contributions(full_coeffs, loo_rows, full_design["n_clusters"])
    pairwise = _pairwise_rows(df)
    _write_plot(full_coeffs, loo_rows)
    result = {
        "spec": "Country-year fixed effects with model fixed effects, country-clustered SE.",
        "full_panel": {"n_obs": full_design["n_obs"], "n_models": model_count(df), "coefficients": full_coeffs},
        "leave_one_model_out": loo_rows,
        "pairwise_rater_gaps": pairwise,
        "figure": "papers/origin-bias/figures/jackknife_forest.pdf",
        "notes": [
            "There are six unordered model pairs. The four cross-origin rows are the substantive rater decomposition; the two same-origin rows are labelled same_origin_diagnostic.",
            "Positive pairwise gaps are left-model higher than right-model on the focal subject country.",
        ],
    }
    write_json(ANALYSIS_DIR / "model_jackknife.json", result)

    lines = [
        "# Model Jackknife and Pairwise Rater Decomposition",
        "",
        "Country-year fixed-effects spec with one model left out per run.",
        "",
        "| Drop model | Contrast | Beta | 95% CI | Holm p |",
        "|---|---|---:|---|---:|",
    ]
    for row in loo_rows:
        for key, label in (("us_on_us", "US-on-US"), ("cn_on_cn", "CN-on-CN")):
            c = row["coefficients"][key]
            lines.append(
                f"| {row['dropped_label']} | {label} | {c['beta']:.3f} | "
                f"[{c['ci95_low']:.3f}, {c['ci95_high']:.3f}] | {format_p(c['p_holm_k2'])} |"
            )
    lines.extend([
        "",
        "## Contribution view",
        "",
        "Contribution is `full_panel_beta - leave_one_out_beta`; positive values mean the full-panel coefficient declines when that rater is removed.",
        "",
        "| Rater | Contrast | Contribution | 95% interval |",
        "|---|---|---:|---|",
    ])
    for row in loo_rows:
        for key, label in (("us_on_us", "US-on-US"), ("cn_on_cn", "CN-on-CN")):
            c = row["coefficients"][key]
            lines.append(
                f"| {row['dropped_label']} | {label} | {c['contribution']:+.3f} | "
                f"[{c['contribution_lower']:+.3f}, {c['contribution_upper']:+.3f}] |"
            )
    lines.extend([
        "",
        "## Pairwise rater gaps",
        "",
        "| Pair | Type | USA left-right | China left-right |",
        "|---|---|---:|---:|",
    ])
    for row in pairwise:
        pair = f"{row['left_label']} minus {row['right_label']}"
        lines.append(
            f"| {pair} | {row['pair_type']} | "
            f"{row['united_states_left_minus_right_mean']:.3f} | "
            f"{row['china_left_minus_right_mean']:.3f} |"
        )
    lines.extend([
        "",
        "The forest plot is saved at `papers/origin-bias/figures/jackknife_forest.pdf`.",
    ])
    write_md(ANALYSIS_DIR / "model_jackknife.md", "\n".join(lines))


if __name__ == "__main__":
    main()
