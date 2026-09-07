"""Finite-sample inference checks for the origin-bias contrasts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from analysis.paper_home_bias.common import load_panel
from analysis.paper_home_bias.inference_utils import CONTRASTS, fit_payload, holm_two, prepare_design, wild_cluster_bootstrap_p


OUT_DIR = Path("artifacts/analysis/home_bias_paper")

SPECS = {
    "country_year_fe": ("country_year", "model"),
    "country_month_fe": ("country_time", "model"),
}


def _merge_payloads(design: dict, bootstrap_b: int) -> dict:
    cr0 = fit_payload(design["y"], design["x"], design["clusters"], terms=design["terms"], cov_kind="cr0")
    cr2 = fit_payload(design["y"], design["x"], design["clusters"], terms=design["terms"], cov_kind="cr2")
    cr3 = fit_payload(design["y"], design["x"], design["clusters"], terms=design["terms"], cov_kind="cr3")
    holm_cr0 = holm_two(cr0["us_on_us"]["p_cr0"], cr0["cn_on_cn"]["p_cr0"])
    holm_cr2 = holm_two(cr2["us_on_us"]["p_cr2"], cr2["cn_on_cn"]["p_cr2"])
    holm_cr3 = holm_two(cr3["us_on_us"]["p_cr3"], cr3["cn_on_cn"]["p_cr3"])

    out: dict[str, dict] = {}
    for idx, term in enumerate(design["terms"]):
        boot = wild_cluster_bootstrap_p(
            design["y"],
            design["x"],
            design["clusters"],
            term_index=idx,
            b=bootstrap_b,
            seed=20260511 + 101 * idx,
        )
        out[term] = {
            "beta": cr0[term]["beta"],
            "n_obs": design["n_obs"],
            "n_clusters": design["n_clusters"],
            "cr0": {
                "se": cr0[term]["se_cr0"],
                "t": cr0[term]["t_cr0"],
                "p": cr0[term]["p_cr0"],
                "p_holm_k2": holm_cr0[term],
                "ci95_low": cr0[term]["ci95_low_cr0"],
                "ci95_high": cr0[term]["ci95_high_cr0"],
            },
            "cr2": {
                "se": cr2[term]["se_cr2"],
                "t": cr2[term]["t_cr2"],
                "p": cr2[term]["p_cr2"],
                "p_holm_k2": holm_cr2[term],
                "ci95_low": cr2[term]["ci95_low_cr2"],
                "ci95_high": cr2[term]["ci95_high_cr2"],
            },
            "cr3": {
                "se": cr3[term]["se_cr3"],
                "t": cr3[term]["t_cr3"],
                "p": cr3[term]["p_cr3"],
                "p_holm_k2": holm_cr3[term],
                "ci95_low": cr3[term]["ci95_low_cr3"],
                "ci95_high": cr3[term]["ci95_high_cr3"],
            },
            "wild_cluster_bootstrap_t": boot,
        }
    return out


def _p_fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _write_md(payload: dict, path: Path) -> None:
    lines = [
        "# Finite-sample inference",
        "",
        "Restricted-null wild-cluster bootstrap-t uses Rademacher country weights with B=99999 unless overridden by `HOME_BIAS_BOOTSTRAP_B` for local testing.",
        "CR2 and CR3 are internal cluster-robust sensitivity columns applied to the residualised fixed-effects design. [ASSUMED: the in-house CR2/CR3 implementation is used because no project-local `clubSandwich` or `boottest` binding is documented.]",
        "Interpretive rule: the restricted-null wild-cluster bootstrap-t is the only finite-sample p-value interpreted as inferential in the draft.",
        "",
        "| spec | contrast | beta | CR0 p | internal CR2 p | internal CR3 p | wild p | internal CR2 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for spec, spec_payload in payload["specifications"].items():
        for term in CONTRASTS:
            row = spec_payload["contrasts"][term]
            lines.append(
                "| {spec} | {term} | {beta:.3f} | {cr0} | {cr2} | {cr3} | {wild} | [{lo:.3f}, {hi:.3f}] |".format(
                    spec=spec,
                    term=term,
                    beta=row["beta"],
                    cr0=_p_fmt(row["cr0"]["p_holm_k2"]),
                    cr2=_p_fmt(row["cr2"]["p_holm_k2"]),
                    cr3=_p_fmt(row["cr3"]["p_holm_k2"]),
                    wild=_p_fmt(row["wild_cluster_bootstrap_t"]["p_wild_cluster_bootstrap_t"]),
                    lo=row["cr2"]["ci95_low"],
                    hi=row["cr2"]["ci95_high"],
                )
            )
    lines.extend(
        [
            "",
            "## Verify pass",
            "",
            "- Both pre-specified contrasts are present for the country-year and country-month FE specifications.",
            "- Each contrast reports CR0, internal CR2/CR3 sensitivity columns, and wild-cluster bootstrap-t p-values.",
            f"- Bootstrap replications recorded: B={payload['bootstrap_B']}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    bootstrap_b = int(os.environ.get("HOME_BIAS_BOOTSTRAP_B", "99999"))
    df = load_panel()
    payload = {
        "bootstrap_B": bootstrap_b,
        "bootstrap_weight": "Rademacher at country-cluster level",
        "cluster": "country",
        "family": list(CONTRASTS),
        "specifications": {},
    }
    for spec_name, fe_groups in SPECS.items():
        design = prepare_design(df, fe_groups=fe_groups)
        payload["specifications"][spec_name] = {
            "fixed_effects": list(fe_groups),
            "contrasts": _merge_payloads(design, bootstrap_b),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "finite_sample_inference.json"
    md_path = OUT_DIR / "finite_sample_inference.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_md(payload, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
