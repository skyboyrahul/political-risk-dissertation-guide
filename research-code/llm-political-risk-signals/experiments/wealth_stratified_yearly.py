"""
wealth_stratified_yearly.py
Wealth-stratified accuracy analysis for YEARLY closed-book LLM signals.
Companion to monthly-llm-risk-signals/analysis/wealth_stratified_accuracy.py
"""

import csv
import json
import math
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
PRS_COMPOSITE = REPO / "data/PRS/prs_composite_only.csv"
PRS_COMPONENTS_DIR = REPO / "data/PRS/components_separate"
ASSUMPTIONS_PATH = REPO / "data/analysis_inputs/wealth_subset_assumptions.json"

MODELS = ["gpt-5-mini", "deepseek-v3.1", "gemini-2.5-flash", "qwen3-coder"]

COUNTRIES = [
    "Argentina", "Brazil", "China", "Egypt", "Germany",
    "India", "Indonesia", "Iran", "Mexico", "Nigeria",
    "Pakistan", "Philippines", "Poland", "Russia", "South Africa",
    "Turkey", "Ukraine", "United Kingdom", "United States", "Venezuela",
]

ASSUMPTIONS = {}
INCOME_TIERS = {}
GDP_PER_CAPITA = {}
COUNTRY_TO_TIER = {}
TIER_ORDER = []
MONTHLY_MAE = {}


def configure_assumptions(path=None):
    """Load separately supplied original-study assumptions at execution time."""
    global ASSUMPTIONS, INCOME_TIERS, GDP_PER_CAPITA, COUNTRY_TO_TIER, TIER_ORDER, MONTHLY_MAE
    source = Path(path or os.environ.get("WEALTH_ASSUMPTIONS_PATH", ASSUMPTIONS_PATH))
    if not source.is_file():
        raise FileNotFoundError("Original-study assumptions are excluded. Supply WEALTH_ASSUMPTIONS_PATH; see research-code/README.md.")
    ASSUMPTIONS = json.loads(source.read_text(encoding="utf-8"))
    INCOME_TIERS = ASSUMPTIONS["income_tiers"]
    GDP_PER_CAPITA = ASSUMPTIONS["gdp_per_capita_current_usd_approx"]
    COUNTRY_TO_TIER = {c: tier for tier, countries in INCOME_TIERS.items() for c in countries}
    TIER_ORDER = ASSUMPTIONS["tier_order"]
    MONTHLY_MAE = ASSUMPTIONS["legacy_monthly_mae_reference"]


# Column mapping: signal CSV col -> PRS component CSV col name
COMPONENT_MAP = {
    "government_stability":   "gov_stability",
    "socioeconomic_conditions": "socioeco_conditions",
    "investment_profile":     "investment_profile",
    "internal_conflict":      "internal_conflict",
    "external_conflict":      "external_conflict",
    "corruption":             "corruption",
    "military_in_politics":   "military_in_politics",
    "religious_tensions":     "religious_tensions",
    "law_and_order":          "law_and_order",
    "ethnic_tensions":        "ethnic_tensions",
    "democratic_accountability": "democratic_accountability",
    "bureaucracy_quality":    "bureaucracy_quality",
}

SIGNAL_COMPONENTS = list(COMPONENT_MAP.keys())

# Legacy monthly comparison retained for exact reproduction, not causal inference.
# Legacy reference is loaded by configure_assumptions at execution time.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    """Return list of dicts from CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_prs_composite():
    """Returns dict (country, year) -> prs_composite float."""
    rows = read_csv(PRS_COMPOSITE)
    out = {}
    for r in rows:
        out[(r["country"], int(r["year"]))] = float(r["prs_composite"])
    return out


def load_prs_components():
    """
    Returns dict component_name -> dict (country, year) -> float
    Uses PRS column names (gov_stability, etc.)
    """
    comp_data = {}
    for sig_col, prs_col in COMPONENT_MAP.items():
        # map file names
        fname_map = {
            "gov_stability": "gov_stability.csv",
            "socioeco_conditions": "socioeco_conditions.csv",
            "investment_profile": "investment_profile.csv",
            "internal_conflict": "internal_conflict.csv",
            "external_conflict": "external_conflict.csv",
            "corruption": "corruption.csv",
            "military_in_politics": "military_in_politics.csv",
            "religious_tensions": "religious_tensions.csv",
            "law_and_order": "law_and_order.csv",
            "ethnic_tensions": "ethnic_tensions.csv",
            "democratic_accountability": "democratic_accountability.csv",
            "bureaucracy_quality": "bureaucracy_quality.csv",
        }
        fpath = os.path.join(PRS_COMPONENTS_DIR, fname_map[prs_col])
        rows = read_csv(fpath)
        d = {}
        for r in rows:
            d[(r["country"], int(r["year"]))] = float(r[prs_col])
        comp_data[sig_col] = d
    return comp_data


def load_model_signals(model):
    """
    Load validated_assessments.csv for a model.
    Filters: prompt_frame == 'A', abstain == False/false/'False'
    Returns list of dicts with numeric component values.
    """
    path = os.path.join(RESULTS_DIR, model, "validated_assessments.csv")
    rows = read_csv(path)
    out = []
    for r in rows:
        if r.get("prompt_frame", "") != "A":
            continue
        abstain_val = str(r.get("abstain", "True")).strip().lower()
        if abstain_val in ("true", "1", "yes"):
            continue
        if r.get("country", "") not in COUNTRIES:
            continue
        try:
            row = {
                "country": r["country"],
                "year": int(r["year"]),
                "model": model,
            }
            for col in SIGNAL_COMPONENTS:
                row[col] = float(r[col])
            row["pred_composite"] = sum(row[col] for col in SIGNAL_COMPONENTS)
            out.append(row)
        except (ValueError, KeyError):
            continue
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(output_dir=None):
    configure_assumptions()
    experiments_dir = Path(output_dir).expanduser().resolve() if output_dir else REPO / "experiments"
    figures_dir = experiments_dir / "figures/wealth_bias_yearly"
    figures_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PRS ground truth...")
    prs_composite = load_prs_composite()
    prs_components = load_prs_components()

    print("Loading model signals...")
    all_signals = []
    for model in MODELS:
        sigs = load_model_signals(model)
        print(f"  {model}: {len(sigs)} valid rows")
        all_signals.extend(sigs)

    # Build error records: (country, model, year, composite_error, component_errors)
    records = []
    skipped = 0
    for row in all_signals:
        key = (row["country"], row["year"])
        if key not in prs_composite:
            skipped += 1
            continue
        gt_composite = prs_composite[key]
        pred_composite = row["pred_composite"]
        composite_ae = abs(pred_composite - gt_composite)

        comp_errors = {}
        for sig_col in SIGNAL_COMPONENTS:
            if key in prs_components[sig_col]:
                comp_errors[sig_col] = abs(row[sig_col] - prs_components[sig_col][key])
            else:
                comp_errors[sig_col] = None

        records.append({
            "country": row["country"],
            "year": row["year"],
            "model": row["model"],
            "tier": COUNTRY_TO_TIER[row["country"]],
            "gdp": GDP_PER_CAPITA[row["country"]],
            "composite_ae": composite_ae,
            "comp_errors": comp_errors,
        })

    print(f"Records matched: {len(records)}, skipped (no GT): {skipped}")

    # -----------------------------------------------------------------------
    # 1. Composite MAE by tier (pooled and per-model)
    # -----------------------------------------------------------------------
    tier_aes = defaultdict(list)
    model_tier_aes = defaultdict(lambda: defaultdict(list))

    for r in records:
        tier_aes[r["tier"]].append(r["composite_ae"])
        model_tier_aes[r["model"]][r["tier"]].append(r["composite_ae"])

    tier_mae = {t: np.mean(tier_aes[t]) for t in TIER_ORDER}
    model_tier_mae = {
        m: {t: np.mean(model_tier_aes[m][t]) if model_tier_aes[m][t] else float("nan")
            for t in TIER_ORDER}
        for m in MODELS
    }

    print("\n=== Composite MAE by income tier (pooled) ===")
    for t in TIER_ORDER:
        print(f"  {t}: {tier_mae[t]:.3f}  (n={len(tier_aes[t])})")

    # -----------------------------------------------------------------------
    # 2. Component MAE by tier
    # -----------------------------------------------------------------------
    tier_comp_mae = {t: {} for t in TIER_ORDER}
    for t in TIER_ORDER:
        for sig_col in SIGNAL_COMPONENTS:
            vals = [r["comp_errors"][sig_col] for r in records
                    if r["tier"] == t and r["comp_errors"][sig_col] is not None]
            tier_comp_mae[t][sig_col] = np.mean(vals) if vals else float("nan")

    # -----------------------------------------------------------------------
    # 3. Country-level composite MAE
    # -----------------------------------------------------------------------
    country_aes = defaultdict(list)
    for r in records:
        country_aes[r["country"]].append(r["composite_ae"])
    country_mae = {c: np.mean(v) for c, v in country_aes.items()}
    country_mae_sorted = sorted(country_mae.items(), key=lambda x: x[1])

    print("\n=== Country-level composite MAE (sorted) ===")
    for c, mae in country_mae_sorted:
        print(f"  {c} ({COUNTRY_TO_TIER[c]}): {mae:.3f}")

    # -----------------------------------------------------------------------
    # 4. Statistical tests
    # -----------------------------------------------------------------------
    groups = [np.array([country_mae[c] for c in INCOME_TIERS[t] if c in country_mae])
              for t in TIER_ORDER]
    groups = [g for g in groups if len(g) > 0]

    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4f}")

    mw_results = {}
    tier_pairs = [
        ("High income", "Upper-middle income"),
        ("High income", "Lower-middle income"),
        ("Upper-middle income", "Lower-middle income"),
    ]
    for t1, t2 in tier_pairs:
        g1 = np.array([country_mae[c] for c in INCOME_TIERS[t1] if c in country_mae])
        g2 = np.array([country_mae[c] for c in INCOME_TIERS[t2] if c in country_mae])
        if len(g1) > 0 and len(g2) > 0:
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            mw_results[f"{t1} vs {t2}"] = {"U": float(u), "p": float(p)}
            print(f"  Mann-Whitney {t1} vs {t2}: U={u:.1f}, p={p:.4f}")

    # OLS: country×model composite MAE ~ log(GDP) + model fixed effects
    # Build design matrix
    ols_y = []
    ols_X = []  # [log_gdp, model_dummies...]
    for r in records:
        ols_y.append(r["composite_ae"])
        row_x = [math.log(r["gdp"])]
        for m in MODELS[1:]:  # MODELS[0] is baseline
            row_x.append(1.0 if r["model"] == m else 0.0)
        ols_X.append(row_x)

    ols_y = np.array(ols_y)
    ols_X = np.column_stack([np.ones(len(ols_y))] + [np.array([x[i] for x in ols_X]) for i in range(len(ols_X[0]))])

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(ols_X, ols_y, rcond=None)
        y_pred = ols_X @ beta
        ss_res = np.sum((ols_y - y_pred) ** 2)
        ss_tot = np.sum((ols_y - np.mean(ols_y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        n, k = ols_X.shape
        # Standard errors
        if n > k:
            sigma2 = ss_res / (n - k)
            xtx_inv = np.linalg.pinv(ols_X.T @ ols_X)
            se = np.sqrt(np.diag(sigma2 * xtx_inv))
            t_stats = beta / se
            p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_stats]
        else:
            se = [float("nan")] * len(beta)
            t_stats = [float("nan")] * len(beta)
            p_vals = [float("nan")] * len(beta)

        col_names = ["intercept", "log_gdp"] + [f"model_{m}" for m in MODELS[1:]]
        print("\n=== OLS: composite MAE ~ log(GDP) + model FE ===")
        print(f"  R² = {r2:.4f}")
        for name, b, s, t, p in zip(col_names, beta, se, t_stats, p_vals):
            print(f"  {name}: β={b:.4f}, SE={s:.4f}, t={t:.3f}, p={p:.4f}")

        ols_results = {
            "r2": float(r2),
            "coefficients": {name: {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}
                             for name, b, s, t, p in zip(col_names, beta, se, t_stats, p_vals)}
        }
    except Exception as e:
        print(f"OLS failed: {e}")
        ols_results = {"error": str(e)}

    # -----------------------------------------------------------------------
    # 5. Comparison table
    # -----------------------------------------------------------------------
    print("\n=== Comparison: Yearly (closed-book) vs Monthly (Wikipedia-fed) ===")
    print(f"{'Tier':<25} {'Yearly MAE':>12} {'Monthly MAE':>12} {'Diff':>8}")
    for t in TIER_ORDER:
        y = tier_mae[t]
        m = MONTHLY_MAE[t]
        print(f"  {t:<23} {y:>12.3f} {m:>12.3f} {y-m:>+8.3f}")

    # -----------------------------------------------------------------------
    # 6. Figures
    # -----------------------------------------------------------------------

    tier_colors = {
        "High income": "#2196F3",
        "Upper-middle income": "#FF9800",
        "Lower-middle income": "#F44336",
    }

    # -- Fig 1: Grouped bar chart by model and tier
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    width = 0.25
    for i, tier in enumerate(TIER_ORDER):
        vals = [model_tier_mae[m][tier] for m in MODELS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=tier,
                      color=tier_colors[tier], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in MODELS], fontsize=10)
    ax.set_ylabel("Composite MAE", fontsize=11)
    ax.set_title("Fig 1: Composite PRS MAE by Income Tier per Model\n(Yearly Closed-Book)", fontsize=12)
    ax.legend(title="Income tier", fontsize=9)
    ax.set_ylim(0, max(model_tier_mae[m][t] for m in MODELS for t in TIER_ORDER
                       if not math.isnan(model_tier_mae[m][t])) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig1_mae_by_model_tier.png", dpi=300)
    plt.close(fig)
    print("Saved fig1")

    # -- Fig 2: Scatter country MAE vs log(GDP per capita)
    fig, ax = plt.subplots(figsize=(9, 7))
    xs = np.array([math.log(GDP_PER_CAPITA[c]) for c in country_mae])
    ys = np.array([country_mae[c] for c in country_mae])
    colors = [tier_colors[COUNTRY_TO_TIER[c]] for c in country_mae]

    ax.scatter(xs, ys, c=colors, s=80, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
    for c, x_val, y_val in zip(country_mae.keys(), xs, ys):
        ax.annotate(c, (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=7)

    # Regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(xs, ys)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, label=f"OLS (R²={r_val**2:.3f}, p={p_val:.3f})")
    ax.set_xlabel("log(GDP per capita, USD)", fontsize=11)
    ax.set_ylabel("Country composite MAE", fontsize=11)
    ax.set_title("Fig 2: Country MAE vs GDP per Capita\n(Yearly Closed-Book)", fontsize=12)
    patches = [mpatches.Patch(color=tier_colors[t], label=t) for t in TIER_ORDER]
    ax.legend(handles=patches + [ax.lines[0]], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig2_scatter_gdp_mae.png", dpi=300)
    plt.close(fig)
    print("Saved fig2")

    # -- Fig 3: Component × tier heatmap
    comp_labels = [c.replace("_", " ").title() for c in SIGNAL_COMPONENTS]
    heatmap_data = np.array([[tier_comp_mae[t][c] for c in SIGNAL_COMPONENTS] for t in TIER_ORDER])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn_r", vmin=0)
    ax.set_xticks(range(len(SIGNAL_COMPONENTS)))
    ax.set_xticklabels(comp_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(TIER_ORDER)))
    ax.set_yticklabels([t.replace(" income", "") for t in TIER_ORDER], fontsize=10)
    ax.set_title("Fig 3: Component MAE by Income Tier (Yearly Closed-Book)", fontsize=12)
    for i in range(len(TIER_ORDER)):
        for j in range(len(SIGNAL_COMPONENTS)):
            val = heatmap_data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if val < heatmap_data.max() * 0.75 else "white")
    plt.colorbar(im, ax=ax, label="MAE")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig3_component_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved fig3")

    # -- Fig 4: Comparison bar chart yearly vs monthly
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(TIER_ORDER))
    width = 0.35
    yearly_vals = [tier_mae[t] for t in TIER_ORDER]
    monthly_vals = [MONTHLY_MAE[t] for t in TIER_ORDER]

    b1 = ax.bar(x - width / 2, yearly_vals, width, label="Yearly (closed-book)",
                color="#1565C0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, monthly_vals, width, label="Legacy monthly reference",
                color="#E65100", alpha=0.85, edgecolor="white")
    for bar, v in zip(b1, yearly_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, v in zip(b2, monthly_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" income", "") for t in TIER_ORDER], fontsize=11)
    ax.set_ylabel("Composite MAE", fontsize=11)
    ax.set_title("Fig 4: Composite MAE by Income Tier\nYearly Closed-Book vs Legacy Monthly Reference", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(yearly_vals + monthly_vals) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig4_comparison_yearly_monthly.png", dpi=300)
    plt.close(fig)
    print("Saved fig4")

    # -----------------------------------------------------------------------
    # 7. Write results JSON
    # -----------------------------------------------------------------------
    results = {
        "analysis": "wealth_stratified_yearly",
        "protocol": "closed-book (no Wikipedia events)",
        "n_records": len(records),
        "models": MODELS,
        "composite_mae_by_tier": {t: float(tier_mae[t]) for t in TIER_ORDER},
        "composite_mae_by_tier_per_model": {
            m: {t: float(model_tier_mae[m][t]) for t in TIER_ORDER}
            for m in MODELS
        },
        "country_mae_sorted": [
            {"country": c, "mae": float(mae), "tier": COUNTRY_TO_TIER[c], "gdp": GDP_PER_CAPITA[c]}
            for c, mae in country_mae_sorted
        ],
        "component_mae_by_tier": {
            t: {c: float(v) for c, v in tier_comp_mae[t].items()}
            for t in TIER_ORDER
        },
        "statistical_tests": {
            "kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
            "mann_whitney": mw_results,
            "ols": ols_results,
        },
        "comparison_with_monthly": {
            t: {
                "yearly_mae": float(tier_mae[t]),
                "monthly_mae": float(MONTHLY_MAE[t]),
                "difference": float(tier_mae[t] - MONTHLY_MAE[t]),
            }
            for t in TIER_ORDER
        },
        "descriptive_inputs": "data/analysis_inputs/wealth_subset_assumptions.json",
        "monthly_source": ASSUMPTIONS["legacy_monthly_mae_provenance"],
    }

    with open(experiments_dir / "wealth_bias_yearly_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved wealth_bias_yearly_results.json")

    # -----------------------------------------------------------------------
    # 7b. Write summary markdown
    # -----------------------------------------------------------------------
    # Determine gradient direction
    yearly_gradient = tier_mae["Lower-middle income"] - tier_mae["High income"]
    monthly_gradient = MONTHLY_MAE["Lower-middle income"] - MONTHLY_MAE["High income"]
    wikipedia_amplifies = yearly_gradient < monthly_gradient

    # Regression stats for scatter
    xs_all = np.array([math.log(GDP_PER_CAPITA[c]) for c in country_mae])
    ys_all = np.array([country_mae[c] for c in country_mae])
    slope_all, intercept_all, r_all, p_all, _ = stats.linregress(xs_all, ys_all)

    md_lines = [
        "# Wealth-Stratified Accuracy Analysis: Yearly Closed-Book LLM Signals",
        "",
        "## Overview",
        "",
        "This analysis evaluates whether LLM accuracy in predicting political risk (ICRG/PRS composite) varies by country income tier, using a **closed-book protocol** (no Wikipedia events fed to the models). A legacy monthly reference is retained for exact reproduction, but its originating run was not recovered. That comparison is descriptive only and cannot identify an effect of Wikipedia coverage.",
        "",
        "**Models**: " + ", ".join(MODELS),
        f"**Total matched records**: {len(records)}",
        "**Countries**: 20 (same as monthly analysis)",
        "**Filter**: `prompt_frame == 'A'`, `abstain == False`",
        "",
        "---",
        "",
        "## Primary Result: Composite PRS MAE by Income Tier",
        "",
        "| Income Tier | Yearly MAE (closed-book) | Legacy monthly MAE | Difference |",
        "|---|---|---|---|",
    ]
    for t in TIER_ORDER:
        y = tier_mae[t]
        m = MONTHLY_MAE[t]
        diff = y - m
        sign = "+" if diff >= 0 else ""
        md_lines.append(f"| {t} | {y:.3f} | {m:.3f} | {sign}{diff:.3f} |")

    md_lines += [
        "",
        f"**Yearly wealth-accuracy gradient** (Lower-middle − High): {yearly_gradient:.3f}",
        f"**Legacy monthly-reference gradient** (Lower-middle − High): {monthly_gradient:.3f}",
        "",
        "**Provenance warning**: the exact monthly run producing these three legacy values was not recovered. See `data/analysis_inputs/wealth_subset_assumptions.json`. No causal conclusion about Wikipedia coverage is drawn from this comparison.",
        "",
    ]

    if wikipedia_amplifies:
        md_lines += [
            "### Descriptive comparison",
            "",
            f"The legacy monthly-reference gradient ({monthly_gradient:.2f}) is steeper than the yearly closed-book gradient ({yearly_gradient:.2f}). Because the monthly run provenance was not recovered, this ordering is not interpreted causally.",
        ]
    else:
        md_lines += [
            "### Descriptive comparison",
            "",
            f"The yearly closed-book gradient ({yearly_gradient:.2f}) is similar to or steeper than the legacy monthly-reference gradient ({monthly_gradient:.2f}). Because the monthly run provenance was not recovered, this ordering is not interpreted causally.",
        ]

    md_lines += [
        "",
        "---",
        "",
        "## Statistical Tests",
        "",
        f"**Kruskal-Wallis test** (country-level MAE across 3 tiers): H={kw_stat:.3f}, p={kw_p:.4f}",
        "",
        "**Mann-Whitney pairwise tests**:",
        "",
    ]
    for pair, res in mw_results.items():
        md_lines.append(f"- {pair}: U={res['U']:.1f}, p={res['p']:.4f}")

    if "r2" in ols_results:
        md_lines += [
            "",
            f"**OLS (MAE ~ log(GDP) + model FE)**: R²={ols_results['r2']:.4f}",
            "",
            "| Variable | β | SE | t | p |",
            "|---|---|---|---|---|",
        ]
        for name, coef in ols_results["coefficients"].items():
            md_lines.append(f"| {name} | {coef['beta']:.4f} | {coef['se']:.4f} | {coef['t']:.3f} | {coef['p']:.4f} |")

    md_lines += [
        "",
        f"**GDP regression (country-level)**: slope={slope_all:.4f}, R²={r_all**2:.4f}, p={p_all:.4f}",
        "",
        "---",
        "",
        "## Country-Level MAE (sorted)",
        "",
        "| Country | MAE | Tier |",
        "|---|---|---|",
    ]
    for c, mae in country_mae_sorted:
        md_lines.append(f"| {c} | {mae:.3f} | {COUNTRY_TO_TIER[c]} |")

    md_lines += [
        "",
        "---",
        "",
        "## Component MAE by Tier",
        "",
        "| Component | High | Upper-middle | Lower-middle |",
        "|---|---|---|---|",
    ]
    for sig_col in SIGNAL_COMPONENTS:
        label = sig_col.replace("_", " ").title()
        vals = [f"{tier_comp_mae[t][sig_col]:.3f}" for t in TIER_ORDER]
        md_lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    md_lines += [
        "",
        "---",
        "",
        "## Figures",
        "",
        "Saved to `experiments/figures/wealth_bias_yearly/`:",
        "",
        "- **fig1_mae_by_model_tier.png**: Composite MAE grouped bar by model and tier",
        "- **fig2_scatter_gdp_mae.png**: Country MAE vs log(GDP per capita) scatter with regression",
        "- **fig3_component_heatmap.png**: Component × tier MAE heatmap",
        "- **fig4_comparison_yearly_monthly.png**: Yearly values beside the explicitly labelled legacy monthly reference",
        "",
        "---",
        "",
        "## Key Finding",
        "",
    ]
    md_lines.append(
        f"The closed-book analysis has a {yearly_gradient:.2f}-point High-to-Lower-middle MAE gradient. "
        f"The unresolved legacy monthly reference has a {monthly_gradient:.2f}-point gradient. "
        "The retained evidence does not isolate whether Wikipedia coverage caused their difference."
    )

    with open(experiments_dir / "wealth_bias_yearly_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Saved wealth_bias_yearly_summary.md")

    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Reproduce the 20-country annual wealth analysis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write artefacts below this directory instead of the repository's experiments/ directory.",
    )
    main(parser.parse_args().output_dir)
