"""Small linear-inference helpers for the home-bias robustness scripts."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


CONTRASTS = ("us_on_us", "cn_on_cn")


def residualise_multiway(df: pd.DataFrame, cols: list[str], groups: tuple[str, ...]) -> pd.DataFrame:
    """Demean columns by several fixed-effect groups using alternating projections."""
    arr = df[cols].to_numpy(dtype=float)
    out = arr - arr.mean(axis=0)
    codes = [df[group].astype("category").cat.codes.to_numpy() for group in groups]
    for _ in range(500):
        old = out.copy()
        for code in codes:
            tmp = pd.DataFrame(out)
            tmp["_g"] = code
            out -= tmp.groupby("_g").transform("mean").to_numpy(dtype=float)
        if float(np.max(np.abs(out - old))) < 1e-11:
            break
    return pd.DataFrame(out, columns=cols, index=df.index)


def prepare_design(
    df: pd.DataFrame,
    *,
    outcome: str = "score",
    fe_groups: tuple[str, ...] = ("country_year", "model"),
    contrasts: tuple[str, ...] = CONTRASTS,
) -> dict:
    cols = [outcome, *contrasts]
    resid = residualise_multiway(df, cols, fe_groups)
    return {
        "y": resid[outcome].to_numpy(dtype=float),
        "x": resid[list(contrasts)].to_numpy(dtype=float),
        "clusters": df["country"].to_numpy(),
        "terms": list(contrasts),
        "n_obs": int(len(df)),
        "n_clusters": int(df["country"].nunique()),
        "fe_groups": list(fe_groups),
        "outcome": outcome,
    }


def _cluster_indices(clusters: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(clusters == cluster) for cluster in np.unique(clusters)]


def ols_fit(y: np.ndarray, x: np.ndarray) -> dict:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    return {"beta": beta, "resid": resid, "xtx_inv": xtx_inv}


def cluster_covariance(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    *,
    kind: str = "cr0",
) -> np.ndarray:
    fit = ols_fit(y, x)
    xtx_inv = fit["xtx_inv"]
    resid = fit["resid"]
    n, p = x.shape
    cluster_ids = _cluster_indices(clusters)
    meat = np.zeros((p, p), dtype=float)
    for idx in cluster_ids:
        xg = x[idx]
        ug = resid[idx]
        if kind == "cr0":
            adj_u = ug
        else:
            hg = xg @ xtx_inv @ xg.T
            mat = np.eye(len(idx)) - hg
            if kind == "cr2":
                eigval, eigvec = np.linalg.eigh((mat + mat.T) / 2.0)
                eigval = np.clip(eigval, 1e-10, None)
                adj = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
            elif kind == "cr3":
                adj = np.linalg.pinv(mat)
            else:
                raise ValueError(kind)
            adj_u = adj @ ug
        score = xg.T @ adj_u
        meat += np.outer(score, score)
    g = len(cluster_ids)
    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - p, 1.0))
    return correction * xtx_inv @ meat @ xtx_inv


def fit_payload(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    *,
    terms: Iterable[str] = CONTRASTS,
    cov_kind: str = "cr0",
) -> dict[str, dict[str, float]]:
    beta = ols_fit(y, x)["beta"]
    cov = cluster_covariance(y, x, clusters, kind=cov_kind)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    df = max(len(np.unique(clusters)) - 1, 1)
    out: dict[str, dict[str, float]] = {}
    crit = float(stats.t.ppf(0.975, df))
    for idx, term in enumerate(terms):
        t_value = float(beta[idx] / se[idx]) if se[idx] > 0 else float("nan")
        p_value = float(2.0 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else float("nan")
        out[term] = {
            "beta": float(beta[idx]),
            f"se_{cov_kind}": float(se[idx]),
            f"t_{cov_kind}": t_value,
            f"p_{cov_kind}": p_value,
            f"ci95_low_{cov_kind}": float(beta[idx] - crit * se[idx]),
            f"ci95_high_{cov_kind}": float(beta[idx] + crit * se[idx]),
        }
    return out


def holm_two(p_us: float, p_cn: float) -> dict[str, float]:
    if p_us <= p_cn:
        return {"us_on_us": min(2.0 * p_us, 1.0), "cn_on_cn": max(p_us, p_cn)}
    return {"cn_on_cn": min(2.0 * p_cn, 1.0), "us_on_us": max(p_cn, p_us)}


def wild_cluster_bootstrap_p(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    *,
    term_index: int,
    b: int = 9999,
    seed: int = 20260511,
) -> dict[str, float]:
    """Restricted-null Rademacher wild-cluster bootstrap-t for one coefficient."""
    rng = np.random.default_rng(seed + term_index)
    full = ols_fit(y, x)
    cov_full = cluster_covariance(y, x, clusters, kind="cr0")
    se_full = math.sqrt(max(float(cov_full[term_index, term_index]), 0.0))
    observed_t = float(full["beta"][term_index] / se_full) if se_full > 0 else float("nan")

    keep = [idx for idx in range(x.shape[1]) if idx != term_index]
    if keep:
        restricted = ols_fit(y, x[:, keep])
        yhat_r = x[:, keep] @ restricted["beta"]
    else:
        yhat_r = np.zeros_like(y)
    resid_r = y - yhat_r
    cluster_values, cluster_codes = np.unique(clusters, return_inverse=True)
    cluster_ids = _cluster_indices(clusters)
    xtx_inv = full["xtx_inv"]
    x_t = x.T
    n, p = x.shape
    g = len(cluster_values)
    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - p, 1.0))
    lever_j = [x[idx] @ xtx_inv[term_index, :] for idx in cluster_ids]
    extreme = 0
    batch = 500
    done = 0
    while done < b:
        size = min(batch, b - done)
        weights = rng.choice([-1.0, 1.0], size=(g, size))
        y_star = yhat_r[:, None] + resid_r[:, None] * weights[cluster_codes]
        beta_star = xtx_inv @ (x_t @ y_star)
        resid_star = y_star - x @ beta_star
        variance = np.zeros(size, dtype=float)
        for idx, lever in zip(cluster_ids, lever_j):
            score_j = lever @ resid_star[idx]
            variance += score_j * score_j
        se_boot = np.sqrt(np.clip(correction * variance, 0.0, None))
        valid = se_boot > 0
        t_boot = np.full(size, np.nan, dtype=float)
        t_boot[valid] = beta_star[term_index, valid] / se_boot[valid]
        extreme += int(np.sum(np.abs(t_boot[valid]) >= abs(observed_t)))
        done += size
    return {
        "t_observed": observed_t,
        "p_wild_cluster_bootstrap_t": float((extreme + 1.0) / (b + 1.0)),
        "B": int(b),
        "seed": int(seed + term_index),
    }


def paired_hac_test(gaps: np.ndarray, *, max_lag: int = 12) -> dict[str, float]:
    arr = np.asarray(gaps, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    resid = arr - mean
    lag = min(max_lag, n - 1)
    long_run = float(np.mean(resid * resid))
    for k in range(1, lag + 1):
        cov = float(np.mean(resid[k:] * resid[:-k]))
        weight = 1.0 - k / (lag + 1.0)
        long_run += 2.0 * weight * cov
    se = math.sqrt(max(long_run, 0.0) / n)
    t_value = mean / se if se > 0 else float("nan")
    p_value = float(2.0 * stats.t.sf(abs(t_value), n - 1)) if math.isfinite(t_value) else float("nan")
    crit = float(stats.t.ppf(0.975, n - 1))
    return {
        "mean_gap": mean,
        "n": int(n),
        "se_hac": se,
        "t_hac": float(t_value),
        "p_hac": p_value,
        "ci95_low": float(mean - crit * se),
        "ci95_high": float(mean + crit * se),
        "max_lag": int(lag),
    }


def block_bootstrap_mean_p(
    gaps: np.ndarray,
    blocks: np.ndarray,
    *,
    b: int = 9999,
    seed: int = 20260511,
) -> dict[str, float]:
    arr = np.asarray(gaps, dtype=float)
    block_values = np.asarray(sorted(set(blocks)))
    by_block = [arr[blocks == block] for block in block_values]
    observed = abs(float(arr.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(b):
        sampled = rng.integers(0, len(by_block), size=len(by_block))
        boot_arr = np.concatenate([by_block[idx] for idx in sampled])
        centred = boot_arr - arr.mean()
        if abs(float(centred.mean())) >= observed:
            extreme += 1
    return {"p_block_bootstrap": float((extreme + 1.0) / (b + 1.0)), "B": int(b), "seed": int(seed)}
