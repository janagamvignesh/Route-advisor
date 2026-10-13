"""Evaluation harness.

Ground truth = Monte Carlo simulation using the HIDDEN true route parameters (routes_truth.csv),
i.e. what would actually happen if we shipped. Each policy only sees the noisy observed data.
Oracle = the route with the lowest true expected total cost (an unattainable benchmark).
"""
import json
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RISK_W_CONG, RISK_W_GEO, economics, default_weights, SEED
from .engine import route_metrics, decide, stability, nondominated, PARETO_COLS, load_data
from .simulate import simulate_route, summarize
from .sensitivity import parameter_sweep

POLICIES = {
    "cheapest":              "Always the lowest freight cost",
    "fastest":               "Always the lowest P90 transit time",
    "lowest_risk":           "Always the lowest composite risk",
    "expected_cost":         "Min Monte Carlo expected total cost (no constraints)",
    "weighted_unconstrained": "Weighted score, no constraints/Pareto (ablation)",
    "weighted_minmax":       "Constraints + weighted score with min-max normalisation (ablation)",
    "weighted (ours)":       "Constraints + Pareto + weighted score",
    "constrained_expected_cost": "Constraints + Pareto + min expected cost",
}


def _pick(policy, m, shp):
    w = default_weights(shp["priority"])
    if policy == "cheapest": return m.loc[m.cost_usd.idxmin(), "route_id"]
    if policy == "fastest": return m.loc[m.p90_days.idxmin(), "route_id"]
    if policy == "lowest_risk": return m.loc[m.risk.idxmin(), "route_id"]
    if policy == "expected_cost": return m.loc[m.exp_total_cost.idxmin(), "route_id"]
    if policy == "weighted_unconstrained": return decide(m, shp, w, "weighted", constrained=False)["chosen"]
    if policy == "weighted_minmax": return decide(m, shp, w, "weighted", norm="minmax")["chosen"]
    if policy == "weighted (ours)": return decide(m, shp, w, "weighted")["chosen"]
    if policy == "constrained_expected_cost": return decide(m, shp, w, "expected_cost")["chosen"]
    raise KeyError(policy)


def true_outcomes(shp, truth_routes, n=4000, seed=SEED + 1):
    rows = {}
    for _, r in truth_routes.iterrows():
        rng = np.random.default_rng([seed, zlib.crc32(str(r["route_id"]).encode())])
        arr, cost = simulate_route(r, shp, n, rng, jitter=False)
        s = summarize(arr, cost, float(shp["deadline_days"]))
        s["true_risk"] = RISK_W_CONG * r["congestion"] + RISK_W_GEO * r["geo_risk"]
        rows[r["route_id"]] = s
    return rows


def run(data_dir="data", out_dir="results", n_sims=2000, quick=False):
    ships, obs, truth = load_data(data_dir)
    rows, per_ship = [], []
    stab = []
    for _, shp in ships.iterrows():
        sid = shp["shipment_id"]
        m = route_metrics(obs[obs.shipment_id == sid], shp, n_sims=n_sims)
        tr = true_outcomes(shp, truth[truth.shipment_id == sid])
        e = economics(shp)
        oracle = min(tr, key=lambda k: tr[k]["exp_total_cost"])
        oracle_cost = tr[oracle]["exp_total_cost"]
        mt = m.set_index("route_id")
        nd = nondominated(m[PARETO_COLS].to_numpy())
        pareto_ids = set(m.loc[nd, "route_id"])
        for pol in POLICIES:
            rid = _pick(pol, m, shp)
            t = tr[rid]
            rows.append(dict(
                shipment_id=sid, priority=shp.priority, policy=pol, route_id=rid,
                true_cost=t["exp_total_cost"], oracle_cost=oracle_cost,
                regret=t["exp_total_cost"] - oracle_cost,
                regret_pct=(t["exp_total_cost"] - oracle_cost) / oracle_cost,
                match_oracle=rid == oracle, true_p_on_time=t["p_on_time"],
                violates=(t["p_on_time"] < e["min_on_time"]) or (t["true_risk"] > float(shp["max_risk"])),
                quoted_cost=mt.loc[rid, "cost_usd"], days=t["mean_days"],
                pareto_optimal=rid in pareto_ids))
        st = stability(m, shp, default_weights(shp.priority), n=200)
        stab.append(dict(shipment_id=sid, priority=shp.priority, share=st["share"]))
        per_ship.append(dict(shipment_id=sid, oracle=oracle))
    df = pd.DataFrame(rows)
    summ = df.groupby("policy").agg(
        mean_regret_pct=("regret_pct", "mean"), total_regret=("regret", "sum"),
        oracle_match=("match_oracle", "mean"), mean_true_on_time=("true_p_on_time", "mean"),
        constraint_violation_rate=("violates", "mean"), mean_quoted_cost=("quoted_cost", "mean"),
        mean_days=("days", "mean"), pareto_rate=("pareto_optimal", "mean")).reindex(list(POLICIES))
    stab_df = pd.DataFrame(stab)
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    df.to_csv(out / "per_shipment_results.csv", index=False)
    summ.to_csv(out / "policy_comparison.csv")
    stab_df.to_csv(out / "stability.csv", index=False)
    sweep = parameter_sweep(ships, obs, n_sims=800) if not quick else pd.DataFrame()
    if len(sweep):
        sweep.to_csv(out / "parameter_sensitivity.csv", index=False)
    meta = dict(n_shipments=len(ships), n_routes=len(obs),
                stability_mean=float(stab_df.share.mean()),
                stability_ge85=float((stab_df.share >= .85).mean()),
                stability_lt60=float((stab_df.share < .6).mean()))
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    return summ, stab_df, sweep, df


if __name__ == "__main__":
    summ, stab, sweep, df = run()
    pd.set_option("display.width", 200)
    print(summ.round(3))
    print(stab.share.describe().round(3))
    print(sweep)
