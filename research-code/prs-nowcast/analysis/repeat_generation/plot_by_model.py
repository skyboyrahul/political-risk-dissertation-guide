"""Render the per-model self-consistency figure for the MSc dissertation.

Reads the original and regenerated ICRG signal pairs directly from disk, computes
the absolute difference between the two composite scores for every month pair,
and draws one horizontal distribution per model, ordered from most to least
reproducible.

Path confound: the 36 Brazil / Grok 4.1 Fast pairs compare xAI batch-endpoint
originals against OpenRouter regenerations, so they mix serving-route variation
with repeat-generation variation. The thesis retains and discloses these pairs;
all four plotted model rows therefore contain 180 pairs.

Outputs (``results/methods_evidence/repeat_generation/``):
  self_consistency_by_model.pdf   vector, 300 DPI
  self_consistency_by_model.png   raster preview
  self_consistency_by_model.provenance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
REGENS_DIR = (
    REPO_ROOT
    / "results"
    / "external_evidence"
    / "monthly-llm-risk-signals"
    / "experiments"
    / "self_consistency"
    / "regens"
)
FIGURES_DIR = REPO_ROOT / "results" / "methods_evidence" / "repeat_generation"
STEM = "self_consistency_by_model"

# House style module lives in this repository's analysis directory.
STYLE_DIR = REPO_ROOT / "analysis"
if not (STYLE_DIR / "thesis_figure_style.py").exists():
    raise SystemExit(f"thesis_figure_style.py not found under {STYLE_DIR}")
sys.path.insert(0, str(STYLE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from thesis_figure_style import apply_paper_style  # noqa: E402

# The 12 ICRG components; the composite is their sum on a 0-100 scale.
COMPONENTS = [
    "government_stability",
    "socioeconomic_conditions",
    "investment_profile",
    "internal_conflict",
    "external_conflict",
    "corruption",
    "military_in_politics",
    "religious_tensions",
    "law_and_order",
    "ethnic_tensions",
    "democratic_accountability",
    "bureaucracy_quality",
]
COUNTRIES = ["south_africa", "china", "united_states", "russia", "brazil"]
MODELS = ["deepseek_deepseekv32", "minimax_m27", "xai_grok41fast", "gpt54"]
YEARS = [2015, 2018, 2021]
MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "minimax_m27": "MiniMax M2.7",
    "xai_grok41fast": "Grok 4.1 Fast",
    "gpt54": "GPT-5.4",
}

# Verified all-pairs composite figures (FINDINGS.md section 2): mean abs diff, exact %.
VERIFIED_ALL_PAIRS = {
    "deepseek_deepseekv32": (2.2278, 21.67),
    "minimax_m27": (4.0806, 10.56),
    "xai_grok41fast": (1.6056, 17.22),
    "gpt54": (0.7167, 45.00),
}

# Thesis text width: A4 with 1in margins, matching the prs-nowcast figure set.
TEXTWIDTH_IN = 452.9679 / 72.27

POINT_COLOUR = "#0072B2"
BOX_COLOUR = "#5A5A5A"
DARK = "#1A1A1A"
GREY_TEXT = "#5A5A5A"
GRID_COLOUR = "#E4E4E4"

COLUMN_MAX_HEIGHT = 0.55  # data units allotted to the tallest share column
BASE_OFFSET = 0.045       # gap between the row baseline and the first dot
BOX_Y_OFFSET = -0.175     # summary glyph sits below the row baseline
BOX_HALF_HEIGHT = 0.045
MEDIAN_HALF_HEIGHT = 0.088


def composite(payload: dict) -> float:
    return float(sum(float(payload[component]) for component in COMPONENTS))


def original_path(country: str, model: str, year: int, month: int) -> tuple[Path, bool]:
    """Return the original signal path and whether the `_batch` fallback was used."""
    filename = f"{year}_{month:02d}.json"
    base = REPO_ROOT / "signals" / country / model / str(year) / filename
    if base.exists():
        return base, False
    return REPO_ROOT / "signals" / country / f"{model}_batch" / str(year) / filename, True


def load_pairs() -> list[dict]:
    rows: list[dict] = []
    for country in COUNTRIES:
        for model in MODELS:
            for year in YEARS:
                for month in range(1, 13):
                    filename = f"{year}_{month:02d}.json"
                    orig, _ = original_path(country, model, year, month)
                    regen = REGENS_DIR / country / model / str(year) / filename
                    if not orig.exists() or not regen.exists():
                        raise SystemExit(f"Missing pair: {country}/{model}/{year}_{month:02d}")
                    diff = composite(json.loads(orig.read_text())) - composite(json.loads(regen.read_text()))
                    rows.append(
                        {
                            "country": country,
                            "model": model,
                            "year": year,
                            "month": month,
                            "abs_diff": abs(diff),
                            "path_confounded": country == "brazil" and model == "xai_grok41fast",
                        }
                    )
    return rows


def summarise(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean_abs_diff": round(float(values.mean()), 4),
        "exact_match_pct": round(float(100.0 * np.mean(values == 0)), 2),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "max_abs_diff": float(values.max()),
    }


def check_verified(rows: list[dict]) -> None:
    """Fail loudly if the all-pairs recomputation drifts from the verified figures."""
    for model, (expected_mean, expected_exact) in VERIFIED_ALL_PAIRS.items():
        values = np.array([row["abs_diff"] for row in rows if row["model"] == model])
        stats = summarise(values)
        if stats["n"] != 180:
            raise SystemExit(f"{model}: expected 180 all-pairs, got {stats['n']}")
        if abs(stats["mean_abs_diff"] - expected_mean) > 5e-4:
            raise SystemExit(
                f"{model}: mean abs diff {stats['mean_abs_diff']} != verified {expected_mean}"
            )
        if abs(stats["exact_match_pct"] - expected_exact) > 5e-3:
            raise SystemExit(
                f"{model}: exact match {stats['exact_match_pct']}% != verified {expected_exact}%"
            )


def draw(plotted: dict[str, np.ndarray], stats: dict[str, dict]) -> plt.Figure:
    order = sorted(plotted, key=lambda model: stats[model]["mean_abs_diff"])
    baselines = {model: float(len(order) - 1 - index) for index, model in enumerate(order)}

    x_max = max(float(values.max()) for values in plotted.values())
    counts = {
        model: {int(value): int(count) for value, count in zip(*np.unique(values, return_counts=True))}
        for model, values in plotted.items()
    }
    max_share = max(
        count / stats[model]["n"] for model, table in counts.items() for count in table.values()
    )
    share_to_height = COLUMN_MAX_HEIGHT / max_share

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.35))

    ax.set_xlim(-1.35, x_max + 0.8)
    ax.set_ylim(-0.42, len(order) - 1 + 1.18)
    ax.set_xticks(range(0, int(x_max) + 1, 2))
    ax.set_yticks([])
    ax.set_xlabel("Absolute composite difference between the two generations (PRS points)")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    left_text = blended_transform_factory(ax.transAxes, ax.transData)

    for model in order:
        base = baselines[model]
        model_stats = stats[model]
        n = model_stats["n"]

        ax.plot(
            [-1.35, x_max + 0.8], [base, base],
            color="#D9D9D9", linewidth=0.5, zorder=1, solid_capstyle="butt",
        )

        # Dot column per integer difference; column height encodes the share of pairs,
        # so a single pair stays a full-size dot rather than an invisible sliver.
        for value, count in sorted(counts[model].items()):
            height = share_to_height * (count / n)
            offsets = np.linspace(0.0, height, count) if count > 1 else np.array([0.0])
            ax.plot(
                np.full(count, float(value)), base + BASE_OFFSET + offsets,
                marker="o", markersize=1.7, markeredgewidth=0.0,
                linestyle="none", color=POINT_COLOUR, zorder=3,
            )

        # Exact-reproduction share, sitting above the zero column and clear of
        # every other column in the row.
        tallest = share_to_height * (max(counts[model].values()) / n)
        ax.text(
            0.0, base + BASE_OFFSET + tallest + 0.06, f"{model_stats['exact_match_pct']:.1f}% exact",
            ha="center", va="bottom", fontsize=7.2, color=DARK, zorder=4,
        )

        # Summary glyph below the baseline: IQR bar, median tick, mean diamond.
        box_y = base + BOX_Y_OFFSET
        ax.add_patch(
            Rectangle(
                (model_stats["q1"], box_y - BOX_HALF_HEIGHT),
                model_stats["q3"] - model_stats["q1"], 2 * BOX_HALF_HEIGHT,
                facecolor=BOX_COLOUR, edgecolor="none", zorder=3,
            )
        )
        ax.plot(
            [model_stats["median"]] * 2,
            [box_y - MEDIAN_HALF_HEIGHT, box_y + MEDIAN_HALF_HEIGHT],
            color=DARK, linewidth=1.0, zorder=4, solid_capstyle="butt",
        )
        ax.plot(
            [model_stats["mean_abs_diff"]], [box_y],
            marker="D", markersize=3.4, markerfacecolor="white",
            markeredgecolor=DARK, markeredgewidth=0.7, linestyle="none", zorder=5,
        )

        ax.text(
            -0.012, base + 0.055, MODEL_LABELS[model],
            transform=left_text, ha="right", va="bottom", fontsize=8.6, color=DARK,
        )
        ax.text(
            -0.012, base - 0.02, f"n = {n}",
            transform=left_text, ha="right", va="top", fontsize=7.0, color=GREY_TEXT,
        )

    handles = [
        Line2D([], [], marker="o", markersize=2.6, markeredgewidth=0.0,
               linestyle="none", color=POINT_COLOUR, label="month pair"),
        Line2D([], [], color=BOX_COLOUR, linewidth=4.0, label="interquartile range"),
        Line2D([], [], color=DARK, linewidth=1.0, label="median"),
        Line2D([], [], marker="D", markersize=3.4, markerfacecolor="white",
               markeredgecolor=DARK, markeredgewidth=0.7, linestyle="none", label="mean"),
    ]
    ax.legend(
        handles=handles, loc="upper right", ncol=4, frameon=False,
        fontsize=7.0, handlelength=1.3, handletextpad=0.45,
        columnspacing=1.1, borderaxespad=0.0,
    )
    return fig


def main() -> int:
    rows = load_pairs()
    check_verified(rows)

    all_stats = {
        model: summarise(np.array([row["abs_diff"] for row in rows if row["model"] == model]))
        for model in MODELS
    }
    plotted = {
        model: np.array([row["abs_diff"] for row in rows if row["model"] == model])
        for model in MODELS
    }
    plotted_stats = {model: summarise(values) for model, values in plotted.items()}
    confounded = np.array([row["abs_diff"] for row in rows if row["path_confounded"]])

    apply_paper_style(usetex=False)
    fig = draw(plotted, plotted_stats)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{STEM}.pdf"
    png_path = FIGURES_DIR / f"{STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    provenance = {
        "figure": STEM,
        "source_signals": "signals/<country>/<model>/<year>/<year>_<month>.json",
        "source_regens": (
            "results/external_evidence/monthly-llm-risk-signals/experiments/"
            "self_consistency/regens/<country>/<model>/<year>/<year>_<month>.json"
        ),
        "grid": {"countries": COUNTRIES, "models": MODELS, "years": YEARS, "months": 12},
        "measure": "absolute difference of the 12-component composite (0-100 scale)",
        "plotted_excludes": None,
        "serving_path_note": {
            "reason": "36 Brazil / Grok 4.1 Fast originals came from the xAI batch endpoint "
                      "while the regenerations came from OpenRouter (FINDINGS.md section 6). "
                      "Same model either way, so all 180 pairs are plotted.",
            "n_affected": int(confounded.size),
            "affected_mean_abs_diff": round(float(confounded.mean()), 4),
            "affected_exact_match_pct": round(float(100.0 * np.mean(confounded == 0)), 2),
        },
        "plotted": {MODEL_LABELS[model]: plotted_stats[model] for model in MODELS},
        "all_pairs": {MODEL_LABELS[model]: all_stats[model] for model in MODELS},
        "outputs": [pdf_path.name, png_path.name],
    }
    (FIGURES_DIR / f"{STEM}.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{'model':16s} {'n':>4s} {'mean':>7s} {'exact%':>7s} {'median':>7s} {'IQR':>9s} {'max':>5s}")
    for model in sorted(MODELS, key=lambda m: plotted_stats[m]["mean_abs_diff"]):
        s = plotted_stats[model]
        print(
            f"{MODEL_LABELS[model]:16s} {s['n']:4d} {s['mean_abs_diff']:7.4f} "
            f"{s['exact_match_pct']:7.2f} {s['median']:7.1f} "
            f"{s['q1']:.0f}-{s['q3']:<7.0f} {s['max_abs_diff']:5.0f}"
        )
    print(
        f"\n36 Brazil / Grok 4.1 Fast pairs were served batch-vs-realtime and are included: "
        f"mean={confounded.mean():.4f} exact={100.0 * np.mean(confounded == 0):.2f}% "
        f"against {plotted_stats['xai_grok41fast']['mean_abs_diff']:.4f} for all 180."
    )
    print(f"\nWrote {pdf_path}\n      {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
