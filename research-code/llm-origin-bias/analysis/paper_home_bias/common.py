"""Shared helpers for the home-bias paper experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "home_bias_paper"
PLOTS_DIR = ROOT / "artifacts" / "plots" / "home_bias_paper"
BASELINE_JSON = ROOT / "artifacts" / "analysis" / "nationality_bias_corrected.json"
WIKIPEDIA_PANEL = ROOT / "artifacts" / "external" / "wikipedia_coverage_panel.csv"
GDI_PANEL = ROOT / "artifacts" / "analysis" / "disagreement" / "gdi_features_annual.csv"
MODELS_YAML = ROOT / "config" / "models.yaml"
COUNTRIES_YAML = ROOT / "config" / "countries.yaml"
MANIFEST_CSV = ROOT / "logs" / "manifest.csv"

CORE_MODELS = ["deepseek_deepseekv32", "minimax_m27", "xai_grok41fast", "gpt54"]
MODEL_ORIGIN = {
    "deepseek_deepseekv32": "CN",
    "minimax_m27": "CN",
    "xai_grok41fast": "US",
    "gpt54": "US",
}
HOME_COUNTRY = {"US": "united_states", "CN": "china"}
CONTROL_COLUMNS = ["article_bytes", "edit_count", "dispute_revisions", "ref_count"]


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_countries() -> list[str]:
    payload = yaml.safe_load(COUNTRIES_YAML.read_text(encoding="utf-8"))
    return [row["slug"] for row in payload["countries"]]


def load_panel(start_year: int = 2005, end_year: int = 2024) -> pd.DataFrame:
    from analysis.paper_home_bias._vendor.dashboard_data import scan_signals

    rows: list[dict[str, Any]] = []
    for country in load_countries():
        signals = scan_signals(country, CORE_MODELS)
        for model in CORE_MODELS:
            for month_key, model_row in sorted(signals.get(model, {}).items()):
                year = int(month_key[:4])
                if year < start_year or year > end_year:
                    continue
                if model_row.get("status") != "valid":
                    continue
                score = model_row.get("total_prs")
                if not isinstance(score, (int, float)):
                    continue
                origin = MODEL_ORIGIN[model]
                month_number = int(month_key[5:7])
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "month": month_key,
                        "month_number": month_number,
                        "country_time": f"{country}_{month_key}",
                        "country_year": f"{country}_{year}",
                        "model": model,
                        "origin": origin,
                        "score": float(score),
                        "us_on_us": int(origin == "US" and country == "united_states"),
                        "cn_on_cn": int(origin == "CN" and country == "china"),
                        "rationale": str(model_row.get("rationale") or ""),
                        "status": str(model_row.get("status") or ""),
                        "source_path": str(model_row.get("path") or ""),
                        **{
                            component: value
                            for component, value in model_row.get("components", {}).items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
    if not rows:
        raise RuntimeError("No score rows found in the raw signal tree")
    return pd.DataFrame(rows)


def add_annual_controls(df: pd.DataFrame, wikipedia: bool = False, gdi: bool = False) -> pd.DataFrame:
    out = df.copy()
    if wikipedia:
        wiki = pd.read_csv(WIKIPEDIA_PANEL)
        for col in CONTROL_COLUMNS:
            mean = wiki[col].mean()
            std = wiki[col].std(ddof=0)
            wiki[f"{col}_z"] = (wiki[col] - mean) / std
        keep = ["country", "year"] + [f"{col}_z" for col in CONTROL_COLUMNS]
        out = out.merge(wiki[keep], on=["country", "year"], how="left", validate="many_to_one")
    if gdi:
        gdi_df = pd.read_csv(GDI_PANEL)
        keep = [
            "country",
            "year",
            "gdi_annual_mean",
            "gdi_annual_max",
            "gdi_annual_std",
            "gdi_annual_velocity_mean",
        ]
        for col in keep[2:]:
            mean = gdi_df[col].mean()
            std = gdi_df[col].std(ddof=0)
            gdi_df[f"{col}_z"] = (gdi_df[col] - mean) / std
        out = out.merge(
            gdi_df[["country", "year"] + [f"{col}_z" for col in keep[2:]]],
            on=["country", "year"],
            how="left",
            validate="many_to_one",
        )
    return out


def cluster_fit(df: pd.DataFrame, formula: str):
    model = smf.ols(formula, data=df).fit()
    return model.get_robustcov_results(cov_type="cluster", groups=df["country"])


def p_adjust_two(p_us: float, p_cn: float) -> dict[str, dict[str, float]]:
    pvals = {"us_on_us": float(p_us), "cn_on_cn": float(p_cn)}
    ordered = sorted(pvals.items(), key=lambda item: item[1])
    holm: dict[str, float] = {}
    prev = 0.0
    for rank, (key, p) in enumerate(ordered, start=1):
        adjusted = min((2 - rank + 1) * p, 1.0)
        holm[key] = max(prev, adjusted)
        prev = holm[key]
    return {
        key: {
            "p_bonferroni_k2": min(value * 2.0, 1.0),
            "p_holm_k2": holm[key],
        }
        for key, value in pvals.items()
    }


def coefficient_payload(fit, names: tuple[str, str] = ("us_on_us", "cn_on_cn")) -> dict[str, dict[str, float]]:
    param_names = list(fit.model.exog_names)
    raw = {}
    for name in names:
        idx = param_names.index(name)
        beta = float(fit.params[idx])
        se = float(fit.bse[idx])
        t_value = float(fit.tvalues[idx])
        p_raw = float(fit.pvalues[idx])
        ci_low, ci_high = fit.conf_int(alpha=0.05)[idx]
        raw[name] = {
            "beta": beta,
            "se_clustered": se,
            "t": t_value,
            "p_raw": p_raw,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
        }
    adjusted = p_adjust_two(raw["us_on_us"]["p_raw"], raw["cn_on_cn"]["p_raw"])
    for key, values in adjusted.items():
        raw[key].update(values)
    return raw


def baseline_comparison() -> dict[str, dict[str, float]]:
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for test in payload["family_k2"]["tests"]:
        key = "us_on_us" if test["label"] == "US-on-US" else "cn_on_cn"
        out[key] = {
            "beta": float(test["mean_gap"]),
            "p_raw": float(test["p_raw"]),
            "p_bonferroni_k2": float(test["p_bonferroni"]),
            "p_holm_k2": float(test["p_holm"]),
        }
    return out


def model_count(df: pd.DataFrame) -> int:
    return int(df["model"].nunique())


def infer_origin(tag: str, provider: str, slug: str, notes: str = "") -> str:
    text = f"{tag} {provider} {slug} {notes}".lower()
    if any(token in text for token in ["deepseek", "minimax", "qwen", "z-ai", "glm"]):
        return "CN"
    if any(token in text for token in ["openai", "x-ai", "xai", "gemini"]):
        return "US"
    return "other"


def format_p(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def zscore_within_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("model")["score"]
    out["score_z_model"] = grouped.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    return out


def t_p_from_beta(beta: float, se: float, df_resid: float) -> tuple[float, float]:
    t_value = beta / se
    return float(t_value), float(2.0 * stats.t.sf(abs(t_value), df_resid))
