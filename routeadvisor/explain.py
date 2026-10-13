"""Plain-language explanations for logistics operators."""
from .config import economics

DRIVER_TEXT = {"cost": "lower freight cost", "time": "shorter transit time",
               "late": "a lower chance of missing the deadline", "risk": "lower route risk"}


def _compare(c, o, label):
    dc = c.cost_usd - o.cost_usd
    parts = []
    if abs(dc) >= 1:
        parts.append(f"costs ${abs(dc):,.0f} ({abs(dc) / o.cost_usd:.0%}) {'more' if dc > 0 else 'less'}")
    dl = c.p_late - o.p_late
    if abs(dl) >= 0.02:
        parts.append(f"{'cuts' if dl < 0 else 'raises'} the chance of missing the deadline "
                     f"from {o.p_late:.0%} to {c.p_late:.0%}")
    dd = c.p90_days - o.p90_days
    if abs(dd) >= 0.5:
        parts.append(f"is {abs(dd):.1f} days {'faster' if dd < 0 else 'slower'} (P90)")
    dr = c.risk - o.risk
    if abs(dr) >= 0.05:
        parts.append(f"has {'lower' if dr < 0 else 'higher'} route risk ({o.risk:.2f} -> {c.risk:.2f})")
    if not parts:
        return None
    return f"vs {label} {o.route_id}: this route " + "; ".join(parts) + "."


def _reject_reason(r, e, shp):
    why = []
    if r.fail_ontime:
        why.append(f"only {r.p_on_time:.0%} chance of meeting the {shp['deadline_days']:.0f}-day deadline "
                   f"(needs at least {e['min_on_time']:.0%})")
    if r.fail_risk:
        why.append(f"route risk {r.risk:.2f} exceeds the cap of {float(shp['max_risk']):.2f}")
    return " and ".join(why)


def explain(result, shp):
    t = result["table"]
    e = economics(shp)
    cid = result["chosen"]
    c = t[t.route_id == cid].iloc[0]
    L = []
    L.append(f"RECOMMENDATION: {cid} - {c.service} ({c.path})")
    L.append(f"  ${c.cost_usd:,.0f} freight | ~{c.mean_days:.1f} days (P90 {c.p90_days:.1f}) | "
             f"{c.p_on_time:.0%} chance on time | risk {c.risk:.2f} | expected total cost ${c.exp_total_cost:,.0f}")
    L.append(f"  Shipment is '{shp['priority']}': must arrive within {shp['deadline_days']:.0f} days with "
             f">= {e['min_on_time']:.0%} confidence and route risk <= {float(shp['max_risk']):.2f}.")
    if result["fallback"]:
        best_p = t.p_on_time.max()
        L.append(f"  WARNING: no route meets every requirement (best on-time chance is {best_p:.0%}). Ranked by expected "
                 "total cost instead, which accounts for how late each route is likely to be. "
                 "Consider renegotiating the deadline or approving a faster mode.")

    pool = t[t.in_pool]
    cheapest = pool.sort_values("cost_usd").iloc[0]
    fastest = pool.sort_values("p90_days").iloc[0]
    L.append("WHY THIS ROUTE")
    seen = set()
    for label, o in [("the cheapest eligible route", cheapest), ("the fastest eligible route", fastest)]:
        if o.route_id != cid and o.route_id not in seen:
            s = _compare(c, o, label)
            if s:
                L.append("  - " + s)
            seen.add(o.route_id)
    if len(pool) == 1:
        L.append("  - It is the only route that satisfies all requirements and is not dominated.")
    else:
        rest = pool[pool.route_id != cid].sort_values("score")
        if len(rest):
            ru = rest.iloc[0]
            w = result["weights"]
            gains = {k: w[k] * (ru[f"n_{k}"] - c[f"n_{k}"]) for k in w}
            top = max(gains, key=gains.get)
            if result["rank_method"] == "weighted" and gains[top] > 1e-6:
                L.append(f"  - Closest alternative is {ru.route_id}; the deciding factor is {DRIVER_TEXT[top]}.")
            elif result["rank_method"] == "expected_cost":
                L.append(f"  - Lowest expected total cost (delay, holding and disruption costs included); "
                         f"closest alternative {ru.route_id} would cost about "
                         f"${ru.exp_total_cost - c.exp_total_cost:,.0f} more in expectation.")

    rej = t[~t.in_pool & (t.route_id != cid)]
    if len(rej):
        L.append("ROUTES NOT RECOMMENDED")
        for _, r in rej.iterrows():
            if not r.candidate:
                reason = _reject_reason(r, e, shp)
            else:  # candidate but Pareto-dominated
                reason = "dominated: another eligible route is at least as good on cost, time, reliability and risk"
            L.append(f"  - {r.route_id} {r.service}: {reason}")

    st = result.get("stability")
    if st:
        msg = f"CONFIDENCE: {st['share']:.0%} of 300 stress tests (weights and estimates perturbed) pick this route"
        if st["alternatives"]:
            msg += f"; main alternative {st['alternatives'][0][0]} ({st['alternatives'][0][1]:.0%})"
        label = "high" if st["share"] >= .85 else "moderate" if st["share"] >= .6 else "LOW - a judgement call"
        L.append(msg + f". Decision stability: {label}.")
    return "\n".join(L)
