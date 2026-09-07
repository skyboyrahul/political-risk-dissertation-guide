"""Audit the frozen local event-input snapshot without network access."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "artifacts/methods_evidence/event_input_snapshot.json"


def _keys(pattern: str, suffix: str) -> set[tuple[str, int, int]]:
    keys = set()
    for path in (REPO_ROOT / "data/input").glob(pattern):
        stem = path.stem.removesuffix(suffix)
        parts = stem.rsplit("_", 2)
        keys.add((path.parent.name, int(parts[-2]), int(parts[-1])))
    return keys


def main() -> None:
    countries = sorted(
        path.name
        for path in (REPO_ROOT / "signals").iterdir()
        if path.is_dir() and (path / "gpt54").exists()
    )
    expected = {
        (country, year, month)
        for country in countries
        for year in range(2005, 2025)
        for month in range(1, 13)
    }
    portal = _keys("*/*_portal_filtered.json", "_portal_filtered") & expected
    yic = _keys("*/*_yic_filtered.json", "_yic_filtered") & expected
    payload = {
        "snapshot_scope": "current checked-out filtered inputs for the 25-country, 2005-2024 panel",
        "countries": countries,
        "expected_country_months": len(expected),
        "portal_present": len(portal),
        "year_in_country_present": len(yic),
        "both_present": len(portal & yic),
        "portal_only": len(portal - yic),
        "year_in_country_only": len(yic - portal),
        "neither_present": len(expected - (portal | yic)),
        "single_source_selection_if_run_on_this_snapshot": {
            "portal_first": len(portal),
            "year_in_country_fallback": len(yic - portal),
            "parametric_no_file": len(expected - (portal | yic)),
        },
        "historical_caveat": "These are present-day repository snapshot counts, not a reconstruction of which file existed when each legacy canonical signal was generated.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
