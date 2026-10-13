"""Decision engine: Monte Carlo metrics -> hard constraints -> Pareto filter -> weighted score."""
import zlib
from collections import Counter
import numpy as np
import pandas as pd
from .config import (CRITERIA, CRIT_COL, RISK_W_CONG, RISK_W_GEO, economics,
                     default_weights, WEIGHT_CONC, COST_PREMIUM_CAP, TIME_CAP)
from .simulate import simulate_route, summarize

PARETO_COLS = [CRIT_COL[k] for k in CRITERIA]


def normalise_weights(w):
    s = sum(max(0.0, float(w[k])) for k in CRITERIA)
    if s <= 0:
        raise ValueError("weights must have a positive sum")
    return {k: max(0.0, float(w[k])) / s for k in CRITERIA}


def route_metrics(routes, shp, n_sims=2000, seed=0):
    """Add composite risk and Monte Carlo reliability metrics to a shipment's routes."""
    df = routes.reset_index(drop=True).copy()
    df["risk"] = RISK_W_CONG * df["congestion"] + RISK_W_GEO * df["geo_risk"]
    rows = []
    for _, r in df.iterrows():
        rng = np.random.default_rng([seed, zlib.crc32(str(r["route_id"]).encode())])
        arrival, cost = simulate_route(r, shp, n_sims, rng, jitter=True)
        rows.append(summarize(arrival, cost, float(shp["deadline_days"])))
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def nondominated(vals):
    """Boolean mask of Pareto non-dominated rows (all columns minimised)."""
    vals = np.asarray(vals, dtype=float)
    keep = np.ones(len(vals), bool)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i != j and np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                keep[i] = False
                break
    return keep


def _minmax(s):
    lo, hi = s.min(), s.max()
    return pd.Series(0.0, index=s.index) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def criterion_scores(m, shp, norm="absolute"):
    """Map each objective to a 0-1 penalty (0 = best).

    absolute (default): scales are tied to decision-relevant references, so one extreme route
        (e.g. air freight) cannot distort the others:
        cost = premium over the cheapest route (+100% -> 1), time = P90 / deadline,
        late = P(miss deadline), risk = composite risk.
    minmax: classic min-max over the shipment's routes (kept as an ablation).
    """
    if norm == "minmax":
        return {k: _minmax(m[CRIT_COL[k]]) for k in CRITERIA}
    prem = m["cost_usd"] / m["cost_usd"].min() - 1.0
    return {"cost": (prem / COST_PREMIUM_CAP).clip(0, 1),
            "time": (m["p90_days"] / float(shp["deadline_days"])).clip(0, TIME_CAP) / TIME_CAP,
            "late": m["p_late"].clip(0, 1),
            "risk": m["risk"].clip(0, 1)}


def decide(m, shp, weights, method="weighted", constrained=True, norm="absolute"):
    """Pure function over a metrics table. Returns dict(table, chosen, fallback).

    method: 'weighted'       rank by priority-weighted normalised score (default)
            'expected_cost'  rank by Monte Carlo expected total cost
    constrained=False skips hard constraints + Pareto filtering (used for baselines/ablations).
    """
    m = m.copy()
    e = economics(shp)
    m["fail_ontime"] = m["p_on_time"] < e["min_on_time"]
    m["fail_risk"] = m["risk"] > float(shp["max_risk"])
    m["feasible"] = ~(m["fail_ontime"] | m["fail_risk"])
    m["violation"] = (np.maximum(0, e["min_on_time"] - m["p_on_time"])
                      + np.maximum(0, m["risk"] - float(shp["max_risk"])))
    fallback = False
    if constrained:
        cand = m["feasible"].to_numpy().copy()
        if not cand.any():
            # Nothing meets every requirement. Keep risk-compliant routes (the risk cap is a safety
            # rule) and rank them by expected total cost, which prices *how late* each one will be.
            fallback = True
            ok = (~m["fail_risk"]).to_numpy()
            cand = ok if ok.any() else np.ones(len(m), bool)
    else:
        cand = np.ones(len(m), bool)
    m["candidate"] = cand

    m["dominated"] = False
    if constrained and cand.sum() > 1:
        idx = np.flatnonzero(cand)
        nd = nondominated(m.loc[idx, PARETO_COLS].to_numpy())
        m.loc[idx[~nd], "dominated"] = True
    pool = m["candidate"] & ~m["dominated"]
    m["in_pool"] = pool

    w = normalise_weights(weights)
    for k, v in criterion_scores(m, shp, norm).items():
        m[f"n_{k}"] = v
    m["score"] = sum(w[k] * m[f"n_{k}"] for k in CRITERIA)
    rank_method = "expected_cost" if fallback else method
    key = "score" if rank_method == "weighted" else "exp_total_cost"
    ranked = m[pool].sort_values([key, "cost_usd"])
    chosen = ranked["route_id"].iloc[0]
    return dict(table=m, chosen=chosen, fallback=fallback, weights=w, method=method,
                rank_method=rank_method)


def stability(m, shp, weights, method="weighted", n=300, seed=0):
    """Re-run the decision under perturbed weights and perturbed metric estimates."""
    rng = np.random.default_rng(seed)
    base = decide(m, shp, weights, method)["chosen"]
    w0 = normalise_weights(weights)
    alpha = np.array([w0[k] for k in CRITERIA]) * WEIGHT_CONC
    counts = Counter()
    for _ in range(n):
        wp = dict(zip(CRITERIA, rng.dirichlet(alpha)))
        mm = m.copy()
        k = len(mm)
        mm["cost_usd"] = mm["cost_usd"] * rng.lognormal(0, .03, k)
        mm["p90_days"] = mm["p90_days"] * rng.lognormal(0, .05, k)
        mm["p_late"] = np.clip(mm["p_late"] + rng.normal(0, .05, k), 0, 1)
        mm["p_on_time"] = 1 - mm["p_late"]
        mm["risk"] = np.clip(mm["risk"] + rng.normal(0, .05, k), 0, 1)
        counts[decide(mm, shp, wp, method)["chosen"]] += 1
    share = counts[base] / n
    others = [(r, c / n) for r, c in counts.most_common() if r != base]
    return dict(chosen=base, share=share, alternatives=others)


def recommend(routes, shp, weights=None, method="weighted", n_sims=2000, seed=0,
              n_stability=300, explain_text=True):
    """Full pipeline for one shipment. `routes` = that shipment's routes (observed estimates)."""
    from .explain import explain
    weights = weights or default_weights(shp["priority"])
    m = route_metrics(routes, shp, n_sims=n_sims, seed=seed)
    out = decide(m, shp, weights, method)
    out["stability"] = stability(m, shp, weights, method, n=n_stability, seed=seed) if n_stability else None
    if explain_text:
        out["explanation"] = explain(out, shp)
    out["metrics"] = m
    return out


def load_data(data_dir="data"):
    from pathlib import Path
    d = Path(data_dir)
    return (pd.read_csv(d / "shipments.csv"), pd.read_csv(d / "routes.csv"),
            pd.read_csv(d / "routes_truth.csv"))
