"""
wealth_stratified_yearly_full.py
Wealth-stratified accuracy for the YEARLY closed-book LLM signals over the
FULL annual country set, not the 20-country subset shared with the monthly
panel. Companion to wealth_stratified_yearly.py; the record filter and the
composite-error definition are identical (prompt_frame == 'A', abstain ==
False, composite = sum of the 12 predicted components against the ICRG
composite). Income tiers come from the World Bank FY2027 classification file
in prs-classical-baselines, applied across the whole period (a current, not
historical, wealth measure). Defunct states with no current classification
are excluded and recorded.
"""

import csv
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PRS_COMPOSITE = REPO / "data/PRS/prs_composite_only.csv"
OUT_JSON = REPO / "experiments/wealth_bias_yearly_full_results.json"
WB_INCOME_ENV = "LLM_PRS_WORLD_BANK_INCOME_GROUPS"
DEFAULT_WB_INCOME = REPO / "data/external/world_bank_income_groups_fy2027.csv"

MODELS = ["gpt-5-mini", "deepseek-v3.1", "gemini-2.5-flash", "qwen3-coder"]

SIGNAL_COMPONENTS = [
    "government_stability", "socioeconomic_conditions", "investment_profile",
    "internal_conflict", "external_conflict", "corruption",
    "military_in_politics", "religious_tensions", "law_and_order",
    "ethnic_tensions", "democratic_accountability", "bureaucracy_quality",
]

# ICRG country name -> World Bank FY2027 economy name, where they differ.
WB_ALIASES = {
    "Bahamas": "Bahamas, The",
    "Brunei": "Brunei Darussalam",
    "Congo": "Congo, Rep.",
    "Congo, DR": "Congo, Dem. Rep.",
    "Cote d'Ivoire": "Côte d’Ivoire",
    "Czech Republic": "Czechia",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong": "Hong Kong SAR, China",
    "Iran": "Iran, Islamic Rep.",
    "Korea, DPR": "Korea, Dem. People's Rep.",
    "Russia": "Russian Federation",
    "Slovakia": "Slovak Republic",
    "Somalia": "Somalia, Fed. Rep.",
    "South Korea": "Korea, Rep.",
    "Syria": "Syrian Arab Republic",
    "Taiwan": "Taiwan, China",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Turkey": "Türkiye",
    "UAE": "United Arab Emirates",
    "Venezuela": "Venezuela, RB",
    "Vietnam": "Viet Nam",
    "Yemen": "Yemen, Rep.",
}

# Defunct states with no FY2027 classification; excluded by construction.
DEFUNCT = {"Czechoslovakia", "East Germany", "West Germany", "USSR"}

TIER_ORDER = ["High income", "Upper middle income", "Lower middle income", "Low income"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_income_groups_path(explicit_path=None):
    configured = explicit_path or os.environ.get(WB_INCOME_ENV)
    path = Path(configured).expanduser() if configured else DEFAULT_WB_INCOME
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "World Bank FY2027 income groups were not found. Pass "
            "--income-groups PATH, set "
            f"{WB_INCOME_ENV}, or supply data/external/world_bank_income_groups_fy2027.csv "
            f"(looked for {path})."
        )
    return path


def load_income_groups(path):
    groups = {}
    for row in read_csv(path):
        name = row["economy"]
        tier = row["income_group"].strip()
        if tier:
            groups[name] = tier
    return groups


def main(income_groups_path=None, output_path=None):
    income_groups_path = resolve_income_groups_path(income_groups_path)
    output_path = Path(output_path).expanduser().resolve() if output_path else OUT_JSON

    prs = {
        (r["country"], int(r["year"])): float(r["prs_composite"])
        for r in read_csv(PRS_COMPOSITE)
    }
    wb = load_income_groups(income_groups_path)

    errors = defaultdict(list)  # country -> list of per-record abs composite errors
    n_records = 0
    for model in MODELS:
        path = os.path.join(REPO, "results", model, "validated_assessments.csv")
        for row in read_csv(path):
            if row["prompt_frame"] != "A" or row["abstain"] == "True":
                continue
            key = (row["country"], int(row["year"]))
            if key not in prs:
                continue
            predicted = sum(float(row[c]) for c in SIGNAL_COMPONENTS)
            errors[row["country"]].append(abs(predicted - prs[key]))
            n_records += 1

    excluded = []
    tiered = {}
    for country in sorted(errors):
        if country in DEFUNCT:
            excluded.append({"country": country, "reason": "defunct state, no FY2027 classification"})
            continue
        wb_name = WB_ALIASES.get(country, country)
        tier = wb.get(wb_name)
        if tier is None:
            excluded.append({"country": country, "reason": f"no income group for '{wb_name}'"})
            continue
        tiered[country] = {
            "mae": float(np.mean(errors[country])),
            "n_records": len(errors[country]),
            "tier": tier,
        }

    by_tier = defaultdict(list)
    for country, row in tiered.items():
        by_tier[row["tier"]].append(row["mae"])

    tier_summary = {
        tier: {
            "mean_country_mae": float(np.mean(by_tier[tier])),
            "median_country_mae": float(np.median(by_tier[tier])),
            "n_countries": len(by_tier[tier]),
        }
        for tier in TIER_ORDER
        if tier in by_tier
    }

    kw = stats.kruskal(*[by_tier[t] for t in TIER_ORDER if t in by_tier])
    mw_high_lower = stats.mannwhitneyu(
        by_tier["High income"], by_tier["Lower middle income"], alternative="two-sided"
    )
    mw_high_low = stats.mannwhitneyu(
        by_tier["High income"], by_tier["Low income"], alternative="two-sided"
    )
    tier_rank = {tier: rank for rank, tier in enumerate(TIER_ORDER)}
    ranks = [tier_rank[row["tier"]] for row in tiered.values()]
    maes = [row["mae"] for row in tiered.values()]
    spearman = stats.spearmanr(ranks, maes)

    out = {
        "analysis": "wealth_stratified_yearly_full",
        "protocol": "closed-book yearly signals, prompt_frame A, abstain False, "
                    "composite = sum of 12 components vs ICRG composite",
        "models": MODELS,
        "income_tier_source": income_groups_path.name,
        "income_tier_vintage": "World Bank FY2027, classification basis year 2025, "
                               "applied across the whole evaluation period",
        "n_records": n_records,
        "n_countries": len(tiered),
        "excluded_countries": excluded,
        "tier_summary": tier_summary,
        "gradient_lower_middle_minus_high": float(
            tier_summary["Lower middle income"]["mean_country_mae"]
            - tier_summary["High income"]["mean_country_mae"]
        ),
        "gradient_low_minus_high": float(
            tier_summary["Low income"]["mean_country_mae"]
            - tier_summary["High income"]["mean_country_mae"]
        ),
        "statistical_tests": {
            "kruskal_wallis": {"H": float(kw.statistic), "p": float(kw.pvalue)},
            "mann_whitney_high_vs_lower_middle": {
                "U": float(mw_high_lower.statistic), "p": float(mw_high_lower.pvalue)
            },
            "mann_whitney_high_vs_low": {
                "U": float(mw_high_low.statistic), "p": float(mw_high_low.pvalue)
            },
            "spearman_tier_rank_vs_country_mae": {
                "rho": float(spearman.statistic), "p": float(spearman.pvalue)
            },
        },
        "country_mae": {
            country: tiered[country] for country in sorted(tiered, key=lambda c: tiered[c]["mae"])
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"records: {n_records}, countries: {len(tiered)}, excluded: {len(excluded)}")
    for tier, row in tier_summary.items():
        print(f"  {tier}: mean country MAE {row['mean_country_mae']:.3f} (n={row['n_countries']})")
    print(f"  gradient LM-High: {out['gradient_lower_middle_minus_high']:.3f}, "
          f"Low-High: {out['gradient_low_minus_high']:.3f}")
    print(f"  Kruskal-Wallis H={kw.statistic:.3f}, p={kw.pvalue:.2e}")
    print(f"  MW High vs LM p={mw_high_lower.pvalue:.2e}; High vs Low p={mw_high_low.pvalue:.2e}")
    print(f"  Spearman tier rank vs MAE rho={spearman.statistic:.3f}, p={spearman.pvalue:.2e}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Reproduce the full-country annual wealth analysis.")
    parser.add_argument(
        "--income-groups",
        type=Path,
        help=f"World Bank FY2027 income-group CSV (or set {WB_INCOME_ENV}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of the committed experiments artefact.",
    )
    args = parser.parse_args()
    try:
        main(args.income_groups, args.output)
    except FileNotFoundError as exc:
        parser.error(str(exc))
