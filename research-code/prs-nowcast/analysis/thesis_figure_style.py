"""Shared figure style for thesis and paper outputs."""

from __future__ import annotations

import shutil
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

# IEEE single-column: 3.5", full text width: 7.25".
IEEE_COLUMN_WIDTH_IN = 3.5
IEEE_TEXT_WIDTH_IN = 7.25

# Backwards-compatible names used by existing scripts.
SINGLE_COL_WIDTH = IEEE_COLUMN_WIDTH_IN
DOUBLE_COL_WIDTH = IEEE_TEXT_WIDTH_IN

# Wong (2011) colorblind-safe palette
CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
              "#0072B2", "#D55E00", "#CC79A7", "#999999"]
BREWER_SET1_BLUE = "#377eb8"
BREWER_SET1_RED = "#e41a1c"
GREY_ANCHOR = "#4d4d4d"
NULL_LINE = "#777777"

# Wong-palette anchors for US and CN origin markers.  Imported by figure
# scripts that need per-subject colour routing.
US_COLOUR = "#0072B2"
CN_COLOUR = "#D55E00"

# SUBJECT_MARKERS: per-subject visual spec using distinct Wong-palette colours
# plus shape redundancy so colour-blind readers and B&W printouts still parse.
SUBJECT_MARKERS = {
    "US": {"marker": "s", "facecolor": US_COLOUR, "edgecolor": US_COLOUR, "linewidths": 0.8},
    "CN": {"marker": "o", "facecolor": CN_COLOUR, "edgecolor": CN_COLOUR, "linewidths": 0.8},
    "ICRG": {"marker": "^", "color": GREY_ANCHOR},
}

ORIGIN_MARKERS = {
    "US": {"marker": "s", "facecolors": "none", "edgecolors": "black", "linewidths": 0.8},
    "CN": {"marker": "o", "facecolors": "none", "edgecolors": "black", "linewidths": 0.8},
    "ICRG": {"marker": "^", "color": GREY_ANCHOR},
}
ORIGIN_COLOURS = {"US": BREWER_SET1_BLUE, "CN": BREWER_SET1_RED, "ICRG": GREY_ANCHOR}
ORIGIN_GREYSCALE = {"US": "black", "CN": "black", "ICRG": GREY_ANCHOR}

# Colorblind-safe diverging colourmap (replaces red-green)
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "YlGnBu"

PAPER_RC_BASE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.35,
}


def _kpsewhich(package: str) -> bool:
    if shutil.which("kpsewhich") is None:
        return False
    try:
        subprocess.run(
            ["kpsewhich", package],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _latex_available() -> bool:
    return shutil.which("latex") is not None and _kpsewhich("times.sty") and _kpsewhich("stix.sty")


def paper_rc(*, greyscale_only: bool = False, usetex: bool | None = None) -> dict:
    """Return rcParams for publication figures."""
    rc = dict(PAPER_RC_BASE)
    rc["text.usetex"] = _latex_available() if usetex is None else bool(usetex)
    if not rc["text.usetex"]:
        rc.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "stix",
            }
        )
    if greyscale_only:
        rc["axes.prop_cycle"] = plt.cycler(
            color=["black", "#4d4d4d", "#8c8c8c"],
            linestyle=["-", "--", ":"],
        )
    else:
        rc["axes.prop_cycle"] = plt.cycler(color=CB_PALETTE)
    return rc


def apply_paper_style(*, greyscale_only: bool = False, usetex: bool | None = None) -> None:
    """Apply publication styling globally for paper figure scripts."""
    sns.set_theme(style="white", font_scale=1.0, rc=paper_rc(greyscale_only=greyscale_only, usetex=usetex))
    plt.rcParams.update(paper_rc(greyscale_only=greyscale_only, usetex=usetex))


def apply_thesis_style():
    """Backward-compatible wrapper for older figure scripts."""
    apply_paper_style()


def make_figure(width: str = "column", aspect: float = 0.62):
    """Return a `(fig, ax)` pair sized for IEEE-style paper output."""
    if width == "column":
        width_in = IEEE_COLUMN_WIDTH_IN
    elif width == "text":
        width_in = IEEE_TEXT_WIDTH_IN
    else:
        raise ValueError("width must be 'column' or 'text'")
    return plt.subplots(figsize=(width_in, width_in * aspect))


THESIS_RC = PAPER_RC_BASE


def thesis_context():
    """Context manager for thesis style (does not pollute global state)."""
    return plt.rc_context(paper_rc())
