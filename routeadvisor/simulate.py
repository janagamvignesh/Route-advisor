"""Monte Carlo simulation of one route's outcome (arrival time and realised total cost)."""
import numpy as np
from .config import (economics, DISRUPT_RATE, DISRUPT_DAYS, DISRUPT_LOSS,
                     JIT_P, JIT_D, JIT_G)


def simulate_route(route, shp, n, rng, jitter=False):
    """Return (arrival_days, total_cost) arrays of length n.

    jitter=True adds *parameter uncertainty* (our estimates are noisy) on top of the natural
    randomness of delays. The evaluation harness uses jitter=False on hidden true parameters.
    """
    p = float(route["delay_prob"]); days = float(route["transit_days"])
    geo = float(route["geo_risk"]); md = float(route["avg_delay_days"])
    if jitter:
        p = np.clip(p + rng.normal(0, JIT_P, n), 0.01, 0.95)
        days = days * rng.normal(1, JIT_D, n)
        geo = np.clip(geo + rng.normal(0, JIT_G, n), 0, 1)
    base = days * rng.lognormal(0, 0.04, n)
    delayed = rng.random(n) < p
    delay = np.where(delayed, rng.gamma(2.0, md / 2.0, n), 0.0)
    disrupted = rng.random(n) < DISRUPT_RATE * geo
    extra = np.where(disrupted, DISRUPT_DAYS, 0.0)
    arrival = base + delay + extra

    e = economics(shp)
    late_days = np.maximum(0.0, arrival - float(shp["deadline_days"]))
    is_late = late_days > 0
    cost = (float(route["cost_usd"])
            + e["holding_per_day"] * arrival
            + e["delay_per_day"] * (delay + extra)
            + DISRUPT_LOSS * float(shp["cargo_value"]) * disrupted
            + is_late * (e["sla_penalty"] + e["delay_per_day"] * late_days))
    return arrival, cost


def summarize(arrival, cost, deadline):
    return dict(
        p_on_time=float(np.mean(arrival <= deadline)),
        p_late=float(np.mean(arrival > deadline)),
        p90_days=float(np.quantile(arrival, 0.9)),
        mean_days=float(arrival.mean()),
        exp_total_cost=float(cost.mean()),
    )
