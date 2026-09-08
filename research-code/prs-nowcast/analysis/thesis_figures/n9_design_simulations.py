#!/usr/bin/env python3
"""N9: what the anticipation screen could have detected.

This is a simulation of the *evaluation design*, not a further analysis of the
political-risk panel. It reads the screen's own dispersion and effect estimates
from the stored artefact and asks a question the screen cannot answer about
itself: how large an anticipation effect would have had to be before this
comparison would have found it.

The screen (Section 4.7 and the anticipation-screen appendix) z-scores each
model's monthly composite over the 12 months before a shock against the 36
months before that window, then compares seven shock country-months against 18
comparison country-months with a Welch two-sample test. It returns p between
0.67 and 0.99 for all four models.

A power curve assuming the same shift on all seven shocks would be the
optimistic case. Training exposure need not be uniform: a model may hold a great
deal about one globally salient event and very little about the others. The
simulation therefore varies how many of the seven shocks actually carry the
effect, holding the per-event shift fixed. k=7 is the homogeneous alternative,
k=1 the sparse one.

Design constants are read from the screen artefact rather than hardcoded, so the
simulated dispersion is the dispersion the Welch test actually faces. An earlier
version of this script used the standard deviation of the cross-model
differential (0.27 to 0.51) in place of the standard deviation of the
anticipation scores (0.77 to 1.24), and so reported a minimum detectable effect
roughly three times too small.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import stats

from _common import COLOUR, REPO, configure_style, figure_size, save, write_caption, write_provenance

STEM = "n9_design_simulations"

# Use both colour and shape to distinguish the observed value in colour and
# greyscale copies. This is the Okabe-Ito orange, not the persistence vermillion.
OBSERVED_COLOUR = "#E69F00"

SEED = 20260803
N_SIM = 20000

N_SHOCK = 7
N_PLACEBO = 18
ALPHA = 0.05
K_GRID = (1, 3, 7)
EFFECTS = np.linspace(0.0, 2.6, 53)

SCREEN_ARTEFACT = (
    REPO / "results" / "methods_evidence" / "temporal_anticipation" / "anticipation_screen.json"
)
ARTEFACT_DIR = REPO / "results" / "methods_evidence" / "design_simulations"


def read_screen() -> dict[str, Any]:
    """Pull the dispersion and observed effects the screen actually produced.

    The Welch test operates on the anticipation z-scores, so the relevant
    dispersion is `events_std_z` / `placebo_std_z`. `std_differential_events`
    describes a different statistic, the cross-model differential, and must not
    be substituted for it.
    """
    screen = json.loads(SCREEN_ARTEFACT.read_text(encoding="utf-8"))

    rows = []
    for name, v in screen["per_model"].items():
        rows.append(
            {
                "model": name,
                "events_mean_z": v["events_mean_z"],
                "placebo_mean_z": v["placebo_mean_z"],
                "observed_effect": v["events_mean_z"] - v["placebo_mean_z"],
                "events_std_z": v["events_std_z"],
                "placebo_std_z": v["placebo_std_z"],
                "welch_p": v["welch_events_vs_placebos"]["p_value"],
            }
        )

    return {
        "source": "results/methods_evidence/temporal_anticipation/anticipation_screen.json",
        "per_model": rows,
        "sd_events_mean": float(np.mean([r["events_std_z"] for r in rows])),
        "sd_placebo_mean": float(np.mean([r["placebo_std_z"] for r in rows])),
        "sd_events_min": float(np.min([r["events_std_z"] for r in rows])),
        "sd_events_max": float(np.max([r["events_std_z"] for r in rows])),
        "sd_placebo_min": float(np.min([r["placebo_std_z"] for r in rows])),
        "sd_placebo_max": float(np.max([r["placebo_std_z"] for r in rows])),
        # Deterioration is negative on this scale, so the strongest apparent
        # anticipation is the most negative shock-minus-comparison difference.
        "largest_observed_effect": float(min(r["observed_effect"] for r in rows)),
    }


def power_curve(
    rng: np.random.Generator, k: int, sd_events: float, sd_placebo: float
) -> list[float]:
    """P(reject) for the 7-vs-18 Welch screen when k of the 7 shocks carry the effect.

    Deterioration appears as a negative z on this scale, so the shift is applied
    negatively. The test is two-sided, so only the magnitude affects power.
    """
    power = []
    for delta in EFFECTS:
        shift = np.zeros(N_SHOCK)
        shift[:k] = -delta
        shock = rng.normal(0.0, sd_events, size=(N_SIM, N_SHOCK)) + shift
        placebo = rng.normal(0.0, sd_placebo, size=(N_SIM, N_PLACEBO))
        _, p = stats.ttest_ind(shock, placebo, axis=1, equal_var=False)
        power.append(float(np.mean(p < ALPHA)))
    return power


def mde(power: list[float]) -> float:
    """Smallest simulated per-event effect reaching 80% power."""
    for e, pw in zip(EFFECTS, power):
        if pw >= 0.80:
            return float(e)
    return float("nan")


def simulate(rng: np.random.Generator, screen: dict[str, Any]) -> dict[str, Any]:
    central = {
        f"k{k}": power_curve(rng, k, screen["sd_events_mean"], screen["sd_placebo_mean"])
        for k in K_GRID
    }

    # Scenario envelope, not a confidence band: the same design run at the
    # least and the most dispersed model in the panel.
    favourable = power_curve(rng, 7, screen["sd_events_min"], screen["sd_placebo_min"])
    unfavourable = power_curve(rng, 7, screen["sd_events_max"], screen["sd_placebo_max"])

    return {
        "effects": EFFECTS.tolist(),
        "power_central": central,
        "mde_80_central": {f"k{k}": mde(central[f"k{k}"]) for k in K_GRID},
        "power_k7_lowest_dispersion": favourable,
        "power_k7_highest_dispersion": unfavourable,
        "mde_80_k7_lowest_dispersion": mde(favourable),
        "mde_80_k7_highest_dispersion": mde(unfavourable),
        "size_at_zero_effect": {f"k{k}": central[f"k{k}"][0] for k in K_GRID},
        "n_shock": N_SHOCK,
        "n_placebo": N_PLACEBO,
        "alpha": ALPHA,
        "two_sided": True,
        "test": "Welch two-sample t-test, unequal variances, scipy.stats.ttest_ind",
        "n_sim": N_SIM,
        "mc_se_max": float(0.5 / np.sqrt(N_SIM)),
    }


def build_figure(sim: dict[str, Any], screen: dict[str, Any]):
    import matplotlib.pyplot as plt

    configure_style()
    width, height = figure_size(0.78, subplots=(1.0, 1.45))
    fig, ax = plt.subplots(figsize=(width, height))

    observed = abs(screen["largest_observed_effect"])
    threshold = sim["mde_80_central"]["k7"]
    threshold_low = sim["mde_80_k7_lowest_dispersion"]
    threshold_high = sim["mde_80_k7_highest_dispersion"]

    labels = [
        "Largest difference recorded",
        "All 7 shock cases affected",
        "3 of 7 affected",
        "1 of 7 affected",
    ]
    y = np.arange(len(labels))[::-1]

    ax.scatter(
        observed,
        y[0],
        marker="D",
        facecolor=OBSERVED_COLOUR,
        edgecolor=COLOUR["dark"],
        linewidths=0.6,
        s=38,
        zorder=3,
    )
    ax.text(observed + 0.08, y[0], f"{observed:.2f}", color=COLOUR["dark"], va="center")

    ax.hlines(y[1], threshold_low, threshold_high, color=COLOUR["model"], linewidth=2.2)
    ax.vlines(
        [threshold_low, threshold_high],
        y[1] - 0.08,
        y[1] + 0.08,
        color=COLOUR["model"],
        linewidth=0.8,
    )
    ax.scatter(threshold, y[1], color=COLOUR["model"], s=30, zorder=3)
    ax.text(
        threshold_high + 0.08,
        y[1],
        f"{threshold:.1f}  (range {threshold_low:.1f} to {threshold_high:.1f})",
        color=COLOUR["model"],
        va="center",
    )

    rates_at_maximum = {
        y[2]: sim["power_central"]["k3"][-1],
        y[3]: sim["power_central"]["k1"][-1],
    }
    for row, rate in rates_at_maximum.items():
        ax.scatter(2.6, row, color=COLOUR["context"], s=20, zorder=3)
        ax.text(
            2.5,
            row,
            f"At 2.6: detected in {rate:.0%}",
            color=COLOUR["context"],
            ha="right",
            va="center",
        )

    ax.text(
        0.0,
        1.08,
        "Each scenario: 20 000 simulations; threshold: detected in at least 16 000 (80%)",
        transform=ax.transAxes,
        color=COLOUR["context"],
        fontsize=7.2,
    )
    ax.set_xlabel("Difference in pre-shock scores (standard deviations)")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlim(0, 3.0)
    ax.grid(axis="x", color=COLOUR["grid"], linewidth=0.45)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    return fig


def main() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    screen = read_screen()
    sim = simulate(rng, screen)

    fig = build_figure(sim, screen)
    outputs = save(fig, STEM)

    artefact = {
        "script": "analysis/thesis_figures/n9_design_simulations.py",
        "seed": SEED,
        "scope": (
            "Simulation of the anticipation screen's operating characteristics. "
            "No political-risk signal, score or PRS value is read. The design "
            "constants are the screen's own sample sizes and dispersion, taken "
            "from its stored artefact."
        ),
        "screen_inputs": screen,
        "simulation": sim,
        "outputs": outputs,
    }
    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTEFACT_DIR / "design_simulations.json").write_text(
        json.dumps(artefact, indent=2) + "\n", encoding="utf-8"
    )
    write_provenance(STEM, artefact)
    write_caption(
        STEM,
        "Recorded pre-shock difference and simulated detection results.\n\n"
        "Note: Grey points show detection rates at 2.6 standard deviations. "
        "Tests are two-sided Welch tests at the 5% level.",
    )
    return artefact


if __name__ == "__main__":
    result = main()
    s, sim = result["screen_inputs"], result["simulation"]
    print("dispersion faced by the Welch test (shock):", round(s["sd_events_mean"], 4))
    print("dispersion faced by the Welch test (comparison):", round(s["sd_placebo_mean"], 4))
    print("observed effects:", {r["model"]: round(r["observed_effect"], 4) for r in s["per_model"]})
    print("largest observed effect:", round(s["largest_observed_effect"], 4))
    print("size at zero effect:", sim["size_at_zero_effect"])
    print("MDE at 80% power, central dispersion:", sim["mde_80_central"])
    print(
        "MDE at 80% power, k=7, dispersion range:",
        sim["mde_80_k7_lowest_dispersion"],
        "to",
        sim["mde_80_k7_highest_dispersion"],
    )
    print("max Monte Carlo SE:", round(sim["mc_se_max"], 5))
    print("pdf:", result["outputs"]["pdf"])
