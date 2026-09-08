"""Exploratory annual-PRS review policy; fixed plan in docs/review_trigger_experiment_plan.md.

Run: python3 analysis/decision_use/review_trigger_experiment.py
Frozen scores only. No fitting, LLM calls, returns or live thesis modifications.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.decision_use import portfolio_risk_budget as source
from analysis.ml_experiments.provenance import write_manifest

PLAN = ROOT / "docs/review_trigger_experiment_plan.md"
OUT = ROOT / "results/decision_use/review_trigger_experiment"
THRESHOLDS = (50, 60, 70, 80)
CUTOFFS = (3, 6, 9, 11, 12)
PRIMARY_THRESHOLD = 60
PRIMARY_CUTOFF = 11
PRIMARY_MISS_COST = 10
MISS_COSTS = (1, 2, 5, 10, 20, 50, 100)
SEED = 20260905
N_BOOT = 10_000
BLOCK_LENGTH = 2
BOUNDARY_TOLERANCE = 1e-9
POLICIES = ("persistence", "nowcast", "always_review", "never_review")
COUNT_FIELDS = ("n", "tp", "fp", "fn", "tn", "reviews")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def below_threshold(scores, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    snapped = np.where(np.abs(values - threshold) <= BOUNDARY_TOLERANCE, threshold, values)
    return snapped < threshold


def confusion_counts(truth, review) -> dict:
    truth, review = np.asarray(truth), np.asarray(review)
    if (truth.ndim != 1 or review.shape != truth.shape
            or truth.dtype != bool or review.dtype != bool):
        raise ValueError("Truth and review must be matching one-dimensional Boolean arrays")
    tp = int(np.sum(truth & review))
    fp = int(np.sum(~truth & review))
    fn = int(np.sum(truth & ~review))
    tn = int(np.sum(~truth & ~review))
    n = len(truth)
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "reviews": tp + fp, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "review_rate": (tp + fp) / n if n else None}


def decision_loss(counts: dict, miss_cost: float) -> float:
    if not np.isfinite(miss_cost) or miss_cost < 0:
        raise ValueError("Miss cost must be finite and non-negative")
    return float(counts["reviews"] + miss_cost * counts["fn"])


def break_even(p_counts: dict, n_counts: dict) -> dict:
    """Nowcast minus persistence loss is intercept + r*slope, for r >= 0."""
    intercept = int(n_counts["reviews"] - p_counts["reviews"])
    slope = int(n_counts["fn"] - p_counts["fn"])
    root = -intercept / slope if slope else None
    if slope == 0:
        condition = "all r >= 0" if intercept < 0 else "none"
        kind = "equal_at_all_ratios" if intercept == 0 else "no_finite_crossing"
    else:
        kind = "positive_crossing" if root > 0 else (
            "zero_crossing" if root == 0 else "crossing_outside_nonnegative_domain")
        if slope < 0:
            condition = "all r >= 0" if root < 0 else f"r > {root:g}"
        else:
            condition = f"0 <= r < {root:g}" if root > 0 else "none"
    return {"review_difference": intercept, "miss_difference": slope,
            "root": root, "kind": kind, "nowcast_lower_for": condition}


def circular_block_indices(n_years: int, n_boot: int, block_length: int, seed: int) -> np.ndarray:
    for value in (n_years, n_boot, block_length):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("Bootstrap dimensions must be positive integers")
    if block_length > n_years:
        raise ValueError("Block cannot exceed the number of years")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_years, size=(n_boot, math.ceil(n_years / block_length)))
    return ((starts[:, :, None] + np.arange(block_length)) % n_years).reshape(n_boot, -1)[:, :n_years]


def paired_year_intervals(values, n_boot=N_BOOT, block_length=BLOCK_LENGTH, seed=SEED) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Paired values must be a nonempty finite year-by-metric matrix")
    indices = circular_block_indices(len(values), n_boot, block_length, seed)
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], axis=0)
    return {"mean": values.mean(axis=0).tolist(), "lower": lower.tolist(), "upper": upper.tolist()}


def load_levels() -> tuple[pd.DataFrame, dict]:
    observed = {rel: sha256(ROOT / rel) for rel in source.EXPECTED_INPUT_SHA256}
    if observed != source.EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen input hash mismatch; no experiment outputs permitted")
    rows, payload = source.load_canonical_rows()
    panel = source.load_panel()
    if panel.duplicated(["country_slug", "year"]).any():
        raise ValueError("Panel country-year join is not unique")
    if rows.duplicated(["country_slug", "year", "cutoff_m"]).any():
        raise ValueError("Duplicate canonical nowcast row")
    levels = source.build_levels(rows, panel)
    if len(levels) != 1750 or set(levels.cutoff_m) != set(CUTOFFS):
        raise ValueError("Unexpected experiment panel dimensions")
    expected_countries = set(levels.country_slug)
    if len(expected_countries) != 25 or set(levels.year) != set(range(2011, 2025)):
        raise ValueError("Unexpected country/year universe")
    for _, block in levels.groupby(["cutoff_m", "year"]):
        if len(block) != 25 or set(block.country_slug) != expected_countries:
            raise ValueError("Unbalanced country-year-cutoff panel")
    numeric = ["prs_lag1", "y_pred", "y_true", "gt_composite", "panel_delta_prs",
               "level_persistence", "level_nowcast", "level_realised"]
    if not np.isfinite(levels[numeric].to_numpy()).all():
        raise ValueError("Nonfinite score or incomplete panel join")
    max_level_gap = float(np.abs(levels.level_realised - levels.gt_composite).max())
    max_delta_gap = float(np.abs(levels.y_true - levels.panel_delta_prs).max())
    if max(max_level_gap, max_delta_gap) > 1e-9:
        raise ValueError("Frozen outcome and annual score disagree")
    levels["level_realised"] = levels["gt_composite"]
    checks = {}
    for cutoff, block in levels.groupby("cutoff_m"):
        by_year = block.groupby("year")
        model_rmse = float(np.mean([np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)) for _, g in by_year]))
        persistence_rmse = float(np.mean([np.sqrt(np.mean(g.y_true ** 2)) for _, g in by_year]))
        canonical = payload["nowcast_results"][source.TARGET][str(cutoff)][source.HEADLINE_MODEL]
        if (abs(model_rmse - canonical["rmse"]) > 1e-10
                or abs(persistence_rmse - canonical["persistence_rmse"]) > 1e-10):
            raise ValueError("Frozen RMSE reproduction failed")
        checks[str(cutoff)] = {"rmse": model_rmse, "persistence_rmse": persistence_rmse}
    gates = {"all_passed": True, "input_sha256": observed, "n_rows": len(levels),
             "n_countries": 25, "years": list(range(2011, 2025)), "rmse": checks,
             "max_level_identity_error": max_level_gap, "max_delta_identity_error": max_delta_gap}
    return levels.sort_values(["cutoff_m", "year", "country_slug"]), gates


def analyse(levels: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    case_parts, totals, annual, transitions, costs = [], [], [], [], []
    break_evens = []
    for threshold in THRESHOLDS:
        for cutoff in CUTOFFS:
            block = levels.loc[levels.cutoff_m == cutoff,
                               ["country_slug", "year", "level_persistence", "level_nowcast", "level_realised"]].copy()
            block.insert(0, "cutoff_m", cutoff)
            block.insert(0, "threshold", threshold)
            block["truth"] = below_threshold(block.level_realised, threshold)
            block["persistence"] = below_threshold(block.level_persistence, threshold)
            block["nowcast"] = below_threshold(block.level_nowcast, threshold)
            block["always_review"] = True
            block["never_review"] = False
            case_parts.append(block)
            summary = {}
            for policy in POLICIES:
                counts = confusion_counts(block.truth.to_numpy(), block[policy].to_numpy())
                summary[policy] = counts
                totals.append({"threshold": threshold, "cutoff_m": cutoff, "policy": policy, **counts})
                for year, yearly in block.groupby("year"):
                    annual.append({"threshold": threshold, "cutoff_m": cutoff, "year": int(year),
                                   "policy": policy,
                                   **confusion_counts(yearly.truth.to_numpy(), yearly[policy].to_numpy())})
                eligible = block.loc[~block.persistence]
                transitions.append({"threshold": threshold, "cutoff_m": cutoff, "policy": policy,
                                    **confusion_counts(eligible.truth.to_numpy(), eligible[policy].to_numpy())})
                for ratio in MISS_COSTS:
                    loss = decision_loss(counts, ratio)
                    costs.append({"threshold": threshold, "cutoff_m": cutoff, "policy": policy,
                                  "miss_cost": ratio, "loss": loss, "loss_per_100": 100 * loss / counts["n"]})
            break_evens.append({"threshold": threshold, "cutoff_m": cutoff,
                                **break_even(summary["persistence"], summary["nowcast"])})
    frames = {"cases": pd.concat(case_parts, ignore_index=True), "summary": pd.DataFrame(totals),
              "years": pd.DataFrame(annual), "transitions": pd.DataFrame(transitions),
              "costs": pd.DataFrame(costs), "break_even": pd.DataFrame(break_evens)}
    primary = frames["summary"].query("threshold == @PRIMARY_THRESHOLD and cutoff_m == @PRIMARY_CUTOFF")
    p = primary.loc[primary.policy == "persistence"].iloc[0].to_dict()
    n = primary.loc[primary.policy == "nowcast"].iloc[0].to_dict()
    yearly = frames["years"].query("threshold == @PRIMARY_THRESHOLD and cutoff_m == @PRIMARY_CUTOFF")
    yp = yearly.loc[yearly.policy == "persistence"].set_index("year").sort_index()
    yn = yearly.loc[yearly.policy == "nowcast"].set_index("year").sort_index()
    metrics = ["reviews", "fp", "fn", "loss"]
    differences = (yn[["reviews", "fp", "fn"]] - yp[["reviews", "fp", "fn"]]).copy()
    differences["loss"] = differences.reviews + PRIMARY_MISS_COST * differences.fn
    intervals = paired_year_intervals(differences.to_numpy() * 100 / 25)
    uncertainty = {metric: {"difference_per_100": intervals["mean"][i],
                            "ci_lower": intervals["lower"][i], "ci_upper": intervals["upper"][i]}
                   for i, metric in enumerate(metrics)}
    loo = [{"omitted_year": int(year),
            "loss_difference_per_100": float(differences.drop(index=year).loss.mean() * 100 / 25)}
           for year in differences.index]
    disagree = frames["cases"].query("threshold == @PRIMARY_THRESHOLD and cutoff_m == @PRIMARY_CUTOFF").copy()
    disagree = disagree.loc[disagree.persistence != disagree.nowcast]
    frames["primary_disagreements"] = disagree
    result = {"primary": {"threshold": PRIMARY_THRESHOLD, "cutoff_m": PRIMARY_CUTOFF,
                           "miss_cost": PRIMARY_MISS_COST, "policies": records(primary),
                           "loss_difference": decision_loss(n, PRIMARY_MISS_COST) - decision_loss(p, PRIMARY_MISS_COST),
                           "break_even": break_even(p, n), "pointwise_intervals_per_100": uncertainty,
                           "leave_one_year_out": loo},
              "all_settings": records(frames["summary"]),
              "all_break_even_conditions": break_evens}
    return result, frames


def markdown(result: dict, frames: dict[str, pd.DataFrame]) -> str:
    primary = result["primary"]
    policies = primary["policies"]
    p = next(row for row in policies if row["policy"] == "persistence")
    n = next(row for row in policies if row["policy"] == "nowcast")
    lines = ["# Annual PRS review-trigger experiment", "",
             "Status: completed exploratory retrospective policy illustration. No live thesis changes.", "",
             "## Primary result: month eleven, annual PRS below 60", "",
             "There are 350 country-year cases (25 countries, 2011–2024). Each cutoff is a separate assessment; "
             "review counts must not be added across cutoffs as if they were independent reviews.", "",
             "| Policy | Reviews | Correct reviews (TP) | Unnecessary reviews (FP) | Missed cases (FN) | Correct non-reviews (TN) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in policies:
        lines.append(f"| {row['policy']} | {row['reviews']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |")
    lines += ["", f"Nowcast minus persistence: {n['reviews'] - p['reviews']:+d} reviews, "
              f"{n['fp'] - p['fp']:+d} unnecessary reviews and {n['fn'] - p['fn']:+d} missed cases.", "",
              "## Hypothetical cost comparison", "",
              "Loss = all reviews + r × missed cases. One review costs one unit; r is the assumed "
              "missed-case cost measured in review-cost units. A reviewed true positive is assumed "
              "to address the stipulated review need. These are hypothetical penalties, not observed "
              "money or demonstrated harms prevented.", "",
              "| Miss cost r | Persistence loss | Nowcast loss | Difference | Always review | Never review |",
              "|---:|---:|---:|---:|---:|---:|"]
    for ratio in MISS_COSTS:
        losses = {row["policy"]: decision_loss(row, ratio) for row in policies}
        lines.append(f"| {ratio} | {losses['persistence']:.0f} | {losses['nowcast']:.0f} | "
                     f"{losses['nowcast'] - losses['persistence']:+.0f} | {losses['always_review']:.0f} | {losses['never_review']:.0f} |")
    be = primary["break_even"]
    lines += ["", f"Across non-negative r, nowcast minus persistence loss is "
              f"{be['review_difference']} + ({be['miss_difference']}) × r. "
              f"The nowcast has lower stipulated loss for: **{be['nowcast_lower_for']}**. "
              f"Break-even classification: `{be['kind']}`.", "",
              "## Uncertainty and influence", "",
              "Differences below are nowcast minus persistence per 100 country-years. Negative values "
              "indicate fewer reviews/errors or lower loss. Intervals are approximate pointwise 95% "
              "intervals from 10,000 paired circular two-year block bootstrap draws, seed 20260905. "
              "They retain complete country panels, are conditional on this country set, and do not "
              "preserve unrestricted serial dependence. Fourteen years are few. No p-values or "
              "simultaneous/confirmatory superiority claims are made.", "",
              "| Metric | Difference per 100 | Lower | Upper |", "|---|---:|---:|---:|"]
    for metric, value in primary["pointwise_intervals_per_100"].items():
        lines.append(f"| {metric} | {value['difference_per_100']:.3f} | {value['ci_lower']:.3f} | {value['ci_upper']:.3f} |")
    loo = [row["loss_difference_per_100"] for row in primary["leave_one_year_out"]]
    lines += ["", f"At r=10, the 14 leave-one-year-out loss differences per 100 cases range "
              f"from {min(loo):.3f} to {max(loo):.3f}. All individual omissions are in the JSON.", "",
              "## Previously at or above the review threshold", "",
              "This diagnostic includes previous annual PRS at or above 60. It concerns transitions "
              "in annual score categories, not event onset or within-year lead time. Persistence "
              "cannot flag these new crossings by construction.", "",
              "| Policy | Eligible cases | Final below threshold | Correct flags | False flags | Misses |",
              "|---|---:|---:|---:|---:|---:|"]
    transitions = frames["transitions"].query("threshold == @PRIMARY_THRESHOLD and cutoff_m == @PRIMARY_CUTOFF")
    for row in transitions.to_dict(orient="records"):
        lines.append(f"| {row['policy']} | {row['n']} | {row['tp'] + row['fn']} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ["", "## Complete fixed sensitivity grid", "",
              "Month eleven / threshold 60 remains primary. Other combinations are descriptive; "
              "month twelve offers no advance timing under the thesis's assumed annual release clock.", "",
              "| Threshold | Cutoff | Persistence reviews / FP / FN | Nowcast reviews / FP / FN | Loss difference, r=10 |",
              "|---:|---:|---|---|---:|"]
    for (threshold, cutoff), setting in frames["summary"].groupby(["threshold", "cutoff_m"]):
        counts = {r["policy"]: r for r in setting.to_dict(orient="records")}
        a, b = counts["persistence"], counts["nowcast"]
        lines.append(f"| {threshold} | {cutoff} | {a['reviews']} / {a['fp']} / {a['fn']} | "
                     f"{b['reviews']} / {b['fp']} / {b['fn']} | {decision_loss(b, 10) - decision_loss(a, 10):+.0f} |")
    lines += ["", "## Interpretation boundaries and reproduction", "",
              "The threshold is grounded in the PRS Group's political-risk bands; triggering an analyst "
              "review at that threshold is our illustrative policy. Annual PRS is a rating benchmark. "
              "The experiment does not observe actual review effectiveness, portfolio returns, losses, "
              "trading costs or monthly risk. Retrospective LLM knowledge and unverified historical "
              "availability remain limitations. Thresholding a conditional-mean score is not an "
              "optimised probability-based decision rule; r does not retune its threshold.", "",
              "The design was committed before computing these outcomes, but the dataset and "
              "nowcast accuracy were already examined. This is exploratory, not a new independent "
              "validation sample. An unfavourable primary result must not be replaced by a better "
              "sensitivity setting.", "",
              "Source: [PRS Group ICRG methodology, PDF p. 7](https://www.prsgroup.com/wp-content/uploads/2018/01/icrgmethodology.pdf).", "",
              "Plan: `docs/review_trigger_experiment_plan.md` (pre-outcome commit `e7187839`).", "",
              "```bash", "python3 analysis/decision_use/review_trigger_experiment.py",
              "python3 -m pytest -q tests/test_review_trigger_experiment.py", "```", "",
              "CSV files contain complete cases, year-level counts, all policy summaries, all cost "
              "settings, annual transitions and primary disagreements. JSON records the primary "
              "intervals, every leave-one-year-out check and analytic break-even conditions. "
              "Each output has a provenance sidecar with input, script, plan and runtime records.", ""]
    return "\n".join(lines)


def figure(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    policies = {row["policy"]: row for row in result["primary"]["policies"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colours = {"persistence": "#777777", "nowcast": "#167284", "always_review": "#ad7844", "never_review": "#a45166"}
    x = np.arange(3)
    for offset, policy in ((-0.18, "persistence"), (0.18, "nowcast")):
        vals = [policies[policy][key] for key in ("reviews", "fp", "fn")]
        bars = axes[0].bar(x + offset, vals, 0.36, label=policy.capitalize(), color=colours[policy])
        axes[0].bar_label(bars, padding=3)
    axes[0].set_xticks(x, ["All reviews", "Unnecessary\nreviews", "Missed\ncases"])
    axes[0].set_ylabel("Count across 350 country-years")
    axes[0].set_title("Review workload and classification errors")
    axes[0].legend(frameon=False)
    axes[0].margins(y=0.18)
    ratios = np.geomspace(1, 100, 200)
    for policy in POLICIES:
        losses = [decision_loss(policies[policy], r) * 100 / 350 for r in ratios]
        axes[1].plot(ratios, losses, label=policy.replace("_", " ").capitalize(), color=colours[policy], lw=2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Missed-case cost / review cost (assumed)")
    axes[1].set_ylabel("Hypothetical loss per 100 country-years (log scale)")
    axes[1].set_title("Cost sensitivity of fixed policies")
    axes[1].axvline(PRIMARY_MISS_COST, color="#999999", ls=":", lw=1)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Annual PRS below 60 · month eleven · retrospective illustration", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build(write_outputs: bool = True, out_dir: Path = OUT) -> dict:
    levels, gates = load_levels()
    result, frames = analyse(levels)
    result["config"] = {"status": "exploratory retrospective policy illustration",
                         "cutoffs": list(CUTOFFS), "thresholds": list(THRESHOLDS),
                         "miss_costs": list(MISS_COSTS), "loss": "reviews + r * fn",
                         "contrast": "nowcast minus persistence", "boundary_tolerance": BOUNDARY_TOLERANCE,
                         "bootstrap": {"n_boot": N_BOOT, "block_length": BLOCK_LENGTH, "seed": SEED},
                         "plan_sha256": sha256(PLAN)}
    result["gates"] = gates
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for name, frame in frames.items():
            path = out_dir / f"review_trigger_{name}.csv"
            frame.to_csv(path, index=False)
            outputs.append(path)
        path = out_dir / "review_trigger_results.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        outputs.append(path)
        path = out_dir / "review_trigger_report.md"
        path.write_text(markdown(result, frames))
        outputs.append(path)
        path = out_dir / "review_trigger_primary.png"
        figure(result, path)
        outputs.append(path)
        for path in outputs:
            write_manifest(path, input_paths=[ROOT / rel for rel in source.EXPECTED_INPUT_SHA256] + [PLAN, Path(source.__file__)],
                           script_path=Path(__file__), extra={"status": result["config"]["status"]})
    return result


if __name__ == "__main__":
    outcome = build()
    print(json.dumps(outcome["primary"], indent=2, allow_nan=False))
