"""Shared helpers for Chapter 4 thesis figure scripts (Goal 3 / ch4-figures-v3).

Ports the per-figure logic from `misc/wch4_figures/build_refined_figures_v2.py`
and the v3 scratch renderers into this repo's production home. Most figures
read the historical snapshot under `results/canonical/`; figures tied to a
documented live correction name that source explicitly. Figure generation
never regenerates a result or an LLM signal.

Global gates applied here for every figure (goal Section 2.1):
  G1: no in-image titles/subtitles (no `set_title`, no `suptitle` calls).
  G2: no in-axes stat boxes (no `bbox=` text boxes carrying numbers).
  G3: months/time never on the y-axis for time-indexed figures.
  G5: persistence and nowcast colours identical across every figure, pulled
      from this module (which in turn matches
      `analysis/thesis_figure_style.py`'s Okabe-Ito-derived palette).

Colour source of truth: this repo's `analysis/thesis_figure_style.py` (serif,
300 DPI, IEEE style, vector PDF). Its CB_PALETTE is Wong/Okabe-Ito; the
specific model/persistence anchors below match the exact hex values already
used across the refined_v2 set and the v3 scratch F5/N1/N4 renders, so the
whole figure set (old and new) shares one consistent grammar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "canonical" / "ml_experiments" / "results"
VIGNETTES = REPO / "analysis" / "vignette_figures"
CLASSICAL_PROVENANCE = (
    REPO.parent / "prs-classical-baselines" / "results" / "figures" / "classical_model_skill_score.provenance.json"
)

FIGURES_OUT = REPO / "figures" / "thesis"

# Okabe-Ito derived palette, identical across every figure in the set (G5).
# Matches build_refined_figures_v2.py::COLOUR and the v3 scratch F5/N1/N4 scripts.
COLOUR = {
    "model": "#0072B2",         # nowcast, Okabe-Ito blue
    "persistence": "#D55E00",   # persistence baseline, Okabe-Ito vermillion
    "actual": "#009E73",        # Okabe-Ito green
    "midas": "#009E73",
    "context": "#8A8A8A",
    "dark": "#222222",
    "grid": "#E6E6E6",
    "light": "#F2F2F2",
    "loss": "#6B6B6B",
    "event": "#CC79A7",         # Okabe-Ito reddish purple
    "null": "#56B4E9",          # Okabe-Ito sky blue
    "scrambled_mean": "#222222",
}

MODEL_LABELS = {
    "deepseek_deepseekv32": "DeepSeek V3.2",
    "gpt54": "GPT-5.4",
    "minimax_m27": "MiniMax M2.7",
    "xai_grok41fast": "Grok 4.1 Fast",
}

TEXTWIDTH_PT = 452.9679  # A4 report, geometry margin=1in: matches build_refined_figures_v2.py
PT_PER_IN = 72.27
GOLDEN = (5 ** 0.5 - 1) / 2


def configure_style() -> None:
    """Vector PDF, embedded serif fonts, 300 DPI raster preview (D4)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.045,
        }
    )


def figure_size(
    fraction: float = 1.0,
    *,
    subplots: tuple[float, float] = (1.0, 1.0),
    min_height: float | None = None,
    max_height: float | None = None,
) -> tuple[float, float]:
    width = (TEXTWIDTH_PT / PT_PER_IN) * fraction
    height = width * GOLDEN * (subplots[0] / subplots[1])
    if min_height is not None:
        height = max(height, min_height)
    if max_height is not None:
        height = min(height, max_height)
    return width, height


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def country_label(value: str) -> str:
    special = {
        "usa": "United States",
        "united_states": "United States",
        "united_kingdom": "United Kingdom",
        "south_africa": "South Africa",
        "saudi_arabia": "Saudi Arabia",
    }
    return special.get(value, value.replace("_", " ").replace("-", " ").title())


def save(fig: plt.Figure, stem: str) -> dict[str, str]:
    """Write PDF + PNG into figures/thesis/. No title/suptitle should ever be
    set on `fig` before calling this (G1); callers are responsible for that.
    """
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_OUT / f"{stem}.pdf"
    png = FIGURES_OUT / f"{stem}.png"
    fig.tight_layout(pad=0.6)
    fig.savefig(pdf, bbox_inches=None)
    fig.savefig(png, dpi=300, bbox_inches=None)
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO))}


def write_provenance(stem: str, provenance: dict[str, Any]) -> Path:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"{stem}.provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def write_caption(stem: str, caption: str) -> Path:
    """Draft caption text file, marked clearly as a draft (G4). Phase 4 (tex
    wiring) consumes this alongside the provenance JSON.
    """
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"{stem}.caption_draft.txt"
    header = "% DRAFT CAPTION, Rahul to rewrite\n"
    path.write_text(header + caption.strip() + "\n", encoding="utf-8")
    return path


def load_vignette(key: str):
    import pandas as pd

    return pd.read_csv(VIGNETTES / key / "plotted_values.csv")


def error_bracket(ax, x: float, y0: float, y1: float, text: str, *, side: str = "right") -> None:
    ax.annotate(
        "",
        xy=(x, y0),
        xytext=(x, y1),
        arrowprops=dict(arrowstyle="<->", lw=0.75, color=COLOUR["dark"], shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    dx = 0.25 if side == "right" else -0.25
    ha = "left" if side == "right" else "right"
    ax.text(x + dx, (y0 + y1) / 2, text, ha=ha, va="center", fontsize=7.0, color=COLOUR["dark"])


def nowcast_m12_rows() -> list[dict[str, Any]]:
    nowcast = load_json(RESULTS / "nowcast_walk_forward.json")
    return [
        row
        for row in nowcast["row_details"]
        if row["target"] == "delta_prs" and row["model"] == "BayesianRidge" and int(row["cutoff_m"]) == 12
    ]


def cutoff_table(cutoffs: list[int] | None = None) -> tuple[list[int], list[dict[str, Any]]]:
    canonical = load_json(RESULTS / "canonical_nowcast_inference_combined.json")
    cutoffs = cutoffs or [3, 6, 9, 11, 12]
    rows = [canonical["per_cutoff"][str(cut)]["delta_prs"] for cut in cutoffs]
    return cutoffs, rows
