"""Figure 1: per-rater home-country peer-deviation effects."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.paper_home_bias.grok43_swap.figure_common import load_json, save, style
from analysis.paper_home_bias._vendor.thesis_figure_style import CN_COLOUR, US_COLOUR


def render() -> list:
    style()
    peer = load_json("peer_deviation_grok43_swap.json")
    version = load_json("version_comparison.json")
    by_label = {}
    for model in ("deepseek_deepseekv32", "minimax_m27", "gpt54"):
        block = peer["canonical_reference"]["per_model"][model]
        by_label[block["label"]] = {
            "label": block["label"],
            "origin": block["origin"],
            "beta": block["beta"],
            "low": block["ci95_low"],
            "high": block["ci95_high"],
        }
    for key, label in (("grok41_on_us", "Grok 4.1 Fast"), ("grok43_on_us", "Grok 4.3")):
        block = version["regression"][key]
        by_label[label] = {
            "label": label,
            "origin": "US",
            "beta": block["beta"],
            "low": block["ci95_low"],
            "high": block["ci95_high"],
            "successor": key == "grok43_on_us",
        }

    order = ["Grok 4.1 Fast", "Grok 4.3", "GPT-5.4", "MiniMax M2.7", "DeepSeek V3.2"]
    rows = [by_label[label] for label in order]

    fig, ax = plt.subplots(figsize=(7.25, 3.3))
    positions = list(range(len(rows), 0, -1))
    for position, row in zip(positions, rows):
        colour = US_COLOUR if row["origin"] == "US" else CN_COLOUR
        successor = row.get("successor", False)
        ax.errorbar(
            row["beta"],
            position,
            xerr=[[row["beta"] - row["low"]], [row["high"] - row["beta"]]],
            fmt="D" if successor else "o",
            color=colour,
            markerfacecolor="white" if successor else colour,
            markeredgecolor=colour,
            markeredgewidth=1.0 if successor else 0.5,
            markersize=5.5,
            capsize=3,
            elinewidth=1.1,
            zorder=3,
        )

    ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    ax.axhline(2.5, color="#d0d0d0", linewidth=0.6, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlim(-2.25, 4.75)
    ax.set_xlabel("Home-country peer deviation (PRS points)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=US_COLOUR, label="United States origin"),
            Line2D([0], [0], marker="o", linestyle="", color=CN_COLOUR, label="China origin"),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                color=US_COLOUR,
                markerfacecolor="white",
                label="Successor version",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.28),
        ncol=3,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.23, bottom=0.19, top=0.78)
    return save(fig, "fig01_per_model_home_effects")


def main() -> None:
    for path in render():
        print(path)


if __name__ == "__main__":
    main()
