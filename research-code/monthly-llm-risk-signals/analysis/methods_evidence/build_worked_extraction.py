"""Freeze a no-network worked extraction example for Ukraine, March 2022."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.generate import (  # noqa: E402
        MONTH_NAMES,
        _build_anchor_block,
        _build_components_list,
        _build_context_block,
        _build_output_schema,
        _build_step_1_block,
        _format_events_for_prompt,
        _load_country_display_name,
        _load_month_events,
        _load_prs_anchor,
        load_prompt_template,
        load_prs_component_descriptions,
        render_prompt,
    )
except ModuleNotFoundError as exc:
    raise SystemExit("This historical producer is source for review. Its original generation helpers and licensed inputs are excluded; see research-code/README.md.") from exc

COUNTRY = "ukraine"
YEAR = 2022
MONTH = 3
MODEL = "xai_grok43_realtime"
SIGNAL_PATH = REPO_ROOT / f"signals/{COUNTRY}/{MODEL}/{YEAR}/{YEAR}_{MONTH:02d}.json"
PROMPT_PATH = REPO_ROOT / "config/prompts/master_prompt.md"
OUT_DIR = REPO_ROOT / "artifacts/methods_evidence/worked_example"
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _composite(payload: dict) -> float:
    return float(sum(float(payload[component]) for component in COMPONENTS))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signal = json.loads(SIGNAL_PATH.read_text(encoding="utf-8"))
    metadata = signal["generation_metadata"]

    os.environ["EVENTS_SINGLE_SOURCE"] = "1"
    events, event_paths = _load_month_events(REPO_ROOT / "data/input", COUNTRY, YEAR, MONTH)
    country_display = _load_country_display_name(COUNTRY, REPO_ROOT)
    anchor = _load_prs_anchor(COUNTRY, YEAR, REPO_ROOT)
    template = load_prompt_template(PROMPT_PATH)
    context = {
        "country_slug": COUNTRY,
        "country_display": country_display,
        "year": YEAR,
        "month": MONTH,
        "month_name": MONTH_NAMES[MONTH],
        "event_source": "wikipedia" if events else "parametric",
        "events": json.dumps(events, ensure_ascii=False, indent=2),
        "events_text": _format_events_for_prompt(events),
        "context_block": _build_context_block(country_display, MONTH_NAMES[MONTH], YEAR, events),
        "step_1_block": _build_step_1_block(country_display, MONTH_NAMES[MONTH], YEAR, events),
        "prs_anchor": json.dumps(anchor, ensure_ascii=False, indent=2),
        "anchor_block": _build_anchor_block(country_display, anchor),
        "prs_scoring_guide": load_prs_component_descriptions(REPO_ROOT),
        "components_list": _build_components_list(),
        "json_schema": _build_output_schema(country_display, YEAR, MONTH),
    }
    prompt = render_prompt(template, context)
    prompt_hash = _sha256_text(prompt)
    expected_hash = metadata["prompt_sha256"]
    if prompt_hash != expected_hash:
        raise RuntimeError(f"Rendered prompt hash {prompt_hash} does not match stored {expected_hash}")
    if len(events) != int(metadata["event_count"]) or [str(path.relative_to(REPO_ROOT)) for path in event_paths] != metadata["event_paths"]:
        raise RuntimeError("Reconstructed evidence inputs do not match generation metadata")

    (OUT_DIR / "rendered_prompt.txt").write_text(prompt, encoding="utf-8")
    excerpt = _format_events_for_prompt(events[:8])
    if excerpt not in prompt:
        raise RuntimeError("Evidence excerpt is not a literal substring of the rendered prompt")
    (OUT_DIR / "evidence_excerpt.txt").write_text(excerpt + "\n", encoding="utf-8")
    shutil.copyfile(SIGNAL_PATH, OUT_DIR / "raw_json_response.json")

    parsed = {
        "country": signal["country"],
        "year": signal["year"],
        "month": signal["month"],
        "model": MODEL,
        "component_scores": {component: signal[component] for component in COMPONENTS},
        "parsed_composite": _composite(signal),
        "rationale": signal["rationale"],
    }
    (OUT_DIR / "parsed_response.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    annual_rows = []
    for month in range(1, 13):
        path = REPO_ROOT / f"signals/{COUNTRY}/{MODEL}/{YEAR}/{YEAR}_{month:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        annual_rows.append({"month": month, "composite": _composite(payload)})
    annual = pd.DataFrame(annual_rows)
    annual_feature = pd.DataFrame(
        [
            {
                "country": country_display,
                "year": YEAR,
                "model": MODEL,
                "months_observed": len(annual),
                "annual_mean": annual["composite"].mean(),
                "annual_std": annual["composite"].std(ddof=1),
                "annual_trend": np.polyfit(annual["month"], annual["composite"], 1)[0],
                "annual_min": annual["composite"].min(),
                "annual_max": annual["composite"].max(),
            }
        ]
    )
    annual_feature.to_csv(OUT_DIR / "annual_feature_row.csv", index=False)

    prediction_input = os.environ.get("WORKED_EXTRACTION_PREDICTIONS")
    if not prediction_input:
        raise ValueError("Supply the authorised frozen prediction input through WORKED_EXTRACTION_PREDICTIONS.")
    canonical_path = Path(prediction_input)
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    prediction = next(
        row
        for row in canonical["row_details"]
        if row["country_slug"] == COUNTRY
        and int(row["year"]) == YEAR
        and int(row["cutoff_m"]) == 12
        and row["target"] == "delta_prs"
        and row["model"] == "BayesianRidge"
    )
    pd.DataFrame(
        [
            {
                "country": country_display,
                "year": YEAR,
                "canonical_model": "BayesianRidge m=12 four-core panel",
                "predicted_delta_prs": prediction["predicted"],
                "actual_delta_prs": prediction["actual"],
                "persistence_delta_prs": prediction["persistence"],
                "prediction_error": prediction["predicted"] - prediction["actual"],
            }
        ]
    ).to_csv(OUT_DIR / "prediction_vs_actual.csv", index=False)

    manifest = {
        "status": "VERIFIED_RECONSTRUCTION",
        "example": "Ukraine, March 2022",
        "model": MODEL,
        "generated_at_utc": metadata["generated_at_utc"],
        "source_repo_commit": metadata["repo_commit"],
        "single_source_mode": True,
        "event_count": len(events),
        "event_paths": metadata["event_paths"],
        "prompt_sha256_stored": expected_hash,
        "prompt_sha256_reconstructed": prompt_hash,
        "prompt_hash_match": True,
        "evidence_excerpt_literal_prompt_substring": True,
        "parsed_composite": parsed["parsed_composite"],
        "raw_response_note": "The retained signal JSON is the validated provider JSON content plus generation metadata. The provider transport envelope and pre-parse byte string were not retained.",
        "prediction_scope_note": "The downstream prediction is the frozen canonical four-core BayesianRidge m=12 row. Grok 4.3 is shown for extraction traceability but is not a predictor in that canonical model.",
        "sources": {
            "signal": str(SIGNAL_PATH.relative_to(REPO_ROOT)),
            "prompt_template": str(PROMPT_PATH.relative_to(REPO_ROOT)),
            "canonical_prediction": canonical_path.name,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown = f"""# Worked extraction: Ukraine, March 2022

This example is a verified offline reconstruction of the prompt recorded for `{MODEL}`. `EVENTS_SINGLE_SOURCE=1` selects the Portal file first; all {len(events)} filtered Portal events were included. The reconstructed prompt SHA256 exactly matches the stored generation metadata (`{prompt_hash}`). The eight-event excerpt is a literal substring of that prompt.

The persisted validated JSON response produces a composite of {parsed['parsed_composite']:.1f}/100. The full provider transport envelope and pre-parse byte string were not retained, so `raw_json_response.json` is the persisted validated JSON content rather than an API envelope. The annual feature row aggregates this model's twelve 2022 monthly composites. The final prediction row is separately taken from the frozen four-core BayesianRidge m=12 model; Grok 4.3 is not one of that model's predictors.
"""
    (OUT_DIR / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
