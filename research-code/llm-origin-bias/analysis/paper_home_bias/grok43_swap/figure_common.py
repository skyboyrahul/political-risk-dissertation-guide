"""Shared helpers for Grok swap paper figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.paper_home_bias._vendor.thesis_figure_style import CB_PALETTE, apply_thesis_style  # noqa: E402

ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "paper_home_bias_grok43_swap"
PLOTS_DIR = ROOT / "artifacts" / "plots" / "origin_bias_grok_swap"

TIER_COLOURS = {
    "High income": CB_PALETTE[4],
    "Upper-middle income": CB_PALETTE[1],
    "Lower-middle income": CB_PALETTE[2],
}


def load_json(name: str) -> dict:
    return json.loads((ANALYSIS_DIR / name).read_text(encoding="utf-8"))


def save(fig, stem: str) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in ("pdf", "png"):
        path = PLOTS_DIR / f"{stem}.{suffix}"
        fig.savefig(path)
        out.append(path)
    plt.close(fig)
    return out


def style() -> None:
    apply_thesis_style()
