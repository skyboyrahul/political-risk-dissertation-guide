"""Context for interpreting the five-country repeat-generation coefficients."""

from __future__ import annotations

import json
import statistics

from .reanalyse import COUNTRIES, MODELS, MONTHS, REPO_ROOT, YEARS, composite


OUT = REPO_ROOT / "results" / "methods_evidence" / "repeat_generation" / "monthly_movement_context.json"


def original_composite(country: str, model: str, year: int, month: int) -> float:
    path = REPO_ROOT / "signals" / country / model / str(year) / f"{year}_{month:02d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return composite(payload)


def build_context() -> dict:
    differences: list[float] = []
    values: dict[str, list[float]] = {country: [] for country in COUNTRIES}
    for country in COUNTRIES:
        for model in MODELS:
            for year in YEARS:
                series = [original_composite(country, model, year, month) for month in MONTHS]
                values[country].extend(series)
                differences.extend(abs(right - left) for left, right in zip(series, series[1:]))

    means = {country: statistics.mean(country_values) for country, country_values in values.items()}
    return {
        "scope": "original signals only; 5 countries x 4 models x 3 years x 12 months",
        "month_to_month_within_country_model_year": {
            "n_transitions": len(differences),
            "mean_abs_change": round(statistics.mean(differences), 4),
            "median_abs_change": statistics.median(differences),
            "zero_change_share_pct": round(100 * differences.count(0) / len(differences), 2),
        },
        "country_mean_composites": {
            country: round(value, 4) for country, value in means.items()
        },
        "spread_of_country_means": round(max(means.values()) - min(means.values()), 4),
    }


def main() -> int:
    output = build_context()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(output["month_to_month_within_country_model_year"]),
        output["spread_of_country_means"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
