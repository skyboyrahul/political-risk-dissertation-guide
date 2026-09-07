"""Country-year fixed-effects home-bias specification for the paper."""

from __future__ import annotations

from analysis.paper_home_bias.common import (
    ANALYSIS_DIR,
    baseline_comparison,
    cluster_fit,
    coefficient_payload,
    ensure_dirs,
    format_p,
    load_panel,
    model_count,
    write_json,
    write_md,
)


SPEC = (
    "OLS of monthly model score on target-country-by-year fixed effects, model fixed effects, "
    "and the two origin-specific home-country dummies. Standard errors are clustered by target country."
)


def main() -> None:
    ensure_dirs()
    df = load_panel()
    fit = cluster_fit(df, "score ~ us_on_us + cn_on_cn + C(country_year) + C(model)")
    coefficients = coefficient_payload(fit)
    baseline = baseline_comparison()
    compared_to_baseline = {
        key: {
            "baseline_beta": baseline[key]["beta"],
            "baseline_p_raw": baseline[key]["p_raw"],
            "country_year_fe_beta": coefficients[key]["beta"],
            "country_year_fe_p_raw": coefficients[key]["p_raw"],
            "beta_shift": coefficients[key]["beta"] - baseline[key]["beta"],
        }
        for key in ("us_on_us", "cn_on_cn")
    }
    result = {
        "coefficients": coefficients,
        "n_obs": int(fit.nobs),
        "n_clusters": int(df["country"].nunique()),
        "n_country_year_cells": int(df["country_year"].nunique()),
        "n_models": model_count(df),
        "spec": SPEC,
        "compared_to_baseline": compared_to_baseline,
        "notes": [
            "The fixed-effect cell is target country by year over the 2005-2024 panel.",
            "Input scores must be supplied separately; archived signals are excluded from this public edition.",
        ],
    }
    write_json(ANALYSIS_DIR / "country_year_fe.json", result)
    pass_note = (
        "Both home dummies retain the expected positive sign and survive Bonferroni correction at k=2."
        if all(coefficients[k]["beta"] > 0 and coefficients[k]["p_bonferroni_k2"] < 0.05 for k in coefficients)
        else "At least one home dummy does not retain sign or survive Bonferroni correction at k=2."
    )
    md = f"""# Country-Year Fixed-Effects Home-Bias Specification

{SPEC}

| Contrast | Beta | Clustered SE | Raw p | Bonferroni k=2 | Holm k=2 |
|---|---:|---:|---:|---:|---:|
| US-on-US | {coefficients['us_on_us']['beta']:.3f} | {coefficients['us_on_us']['se_clustered']:.3f} | {format_p(coefficients['us_on_us']['p_raw'])} | {format_p(coefficients['us_on_us']['p_bonferroni_k2'])} | {format_p(coefficients['us_on_us']['p_holm_k2'])} |
| CN-on-CN | {coefficients['cn_on_cn']['beta']:.3f} | {coefficients['cn_on_cn']['se_clustered']:.3f} | {format_p(coefficients['cn_on_cn']['p_raw'])} | {format_p(coefficients['cn_on_cn']['p_bonferroni_k2'])} | {format_p(coefficients['cn_on_cn']['p_holm_k2'])} |

N = {int(fit.nobs)} observations, {df['country'].nunique()} country clusters, {df['country_year'].nunique()} target-country-by-year cells, {model_count(df)} models.

Shift versus the locked baseline: US-on-US {compared_to_baseline['us_on_us']['beta_shift']:+.3f}; CN-on-CN {compared_to_baseline['cn_on_cn']['beta_shift']:+.3f}.

{pass_note}
"""
    write_md(ANALYSIS_DIR / "country_year_fe.md", md)


if __name__ == "__main__":
    main()
