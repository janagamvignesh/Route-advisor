"""All tunable assumptions live here so they are easy to audit and change."""

SEED = 42

# Composite route risk = weighted blend of congestion and geopolitical risk (both in [0, 1]).
RISK_W_CONG = 0.4
RISK_W_GEO = 0.6

CRITERIA = ["cost", "time", "late", "risk"]          # the four objectives (all minimised)
CRIT_COL = {"cost": "cost_usd", "time": "p90_days", "late": "p_late", "risk": "risk"}

# Priority profiles: weights express *what the shipper cares about*, economics express *what
# lateness costs them*. min_on_time is the hard reliability requirement for the deadline.
PROFILES = {
    "urgent": dict(
        weights=dict(cost=0.15, time=0.30, late=0.35, risk=0.20),
        min_on_time=0.85, delay_per_day_frac=0.004, delay_per_day_base=400, sla_frac=0.15),
    "standard": dict(
        weights=dict(cost=0.35, time=0.20, late=0.25, risk=0.20),
        min_on_time=0.70, delay_per_day_frac=0.0015, delay_per_day_base=150, sla_frac=0.05),
    "cost_sensitive": dict(
        weights=dict(cost=0.55, time=0.10, late=0.15, risk=0.20),
        min_on_time=0.60, delay_per_day_frac=0.0005, delay_per_day_base=50, sla_frac=0.02),
}

HOLDING_FRAC_PER_DAY = 0.0003     # inventory/capital cost per day, as a fraction of cargo value

# Disruption model (geopolitical events): P(disruption) = DISRUPT_RATE * geo_risk
DISRUPT_RATE = 0.10
DISRUPT_DAYS = 7.0                # extra days if disrupted
DISRUPT_LOSS = 0.15               # fraction of cargo value lost/damaged if disrupted

# Uncertainty we assume in *our own estimates* (used by Monte Carlo + stability analysis)
JIT_P = 0.05                      # sd of delay-probability estimate error
JIT_D = 0.05                      # relative sd of transit-time estimate error
JIT_G = 0.08                      # sd of geopolitical-risk estimate error
COST_PREMIUM_CAP = 1.0             # a route costing +100% over the cheapest option scores the maximum cost penalty
TIME_CAP = 1.5                    # transit time is scored as P90/deadline, saturating at 1.5x the deadline
WEIGHT_CONC = 30                  # Dirichlet concentration for weight perturbation (higher = tighter)


def economics(shp):
    """Per-shipment cost parameters derived from priority + cargo value (overridable)."""
    prof = PROFILES[shp["priority"]]
    cv = float(shp["cargo_value"])
    return dict(
        delay_per_day=prof["delay_per_day_base"] + prof["delay_per_day_frac"] * cv,
        sla_penalty=prof["sla_frac"] * cv,
        holding_per_day=HOLDING_FRAC_PER_DAY * cv,
        min_on_time=float(shp.get("min_on_time", prof["min_on_time"])),
    )


def default_weights(priority):
    return dict(PROFILES[priority]["weights"])
