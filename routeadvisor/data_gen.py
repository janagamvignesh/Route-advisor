"""Synthetic dataset generator.

Each shipment gets 3-5 routes built from *archetypes* that encode real-world trade-offs:
  ocean_direct    cheap, slow, goes through congested hubs  -> high delay probability
  ocean_reliable  ~15% dearer, a bit slower, safe hubs      -> low delay probability
  ocean_shortcut  cheapest + fastest ocean, via a chokepoint -> high geopolitical risk
  sea_air         2.3x cost, much faster
  air_express     4.8x cost, fastest and most reliable

We first draw TRUE parameters, then derive noisy OBSERVED estimates (what a planner would see).
Only the observed file is used by the recommender; the true file is used only in evaluation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from .config import SEED

# (origin, destination, approx. sea distance km)
LANES = [("Shanghai", "Rotterdam", 19500), ("Shenzhen", "Hamburg", 19800),
         ("Singapore", "Rotterdam", 15500), ("Mumbai", "Felixstowe", 11500),
         ("Shanghai", "Los Angeles", 10500), ("Busan", "Long Beach", 9800),
         ("Chennai", "Jebel Ali", 3400), ("Shanghai", "Chennai", 7200),
         ("Rotterdam", "New York", 6200), ("Mumbai", "Mombasa", 4300)]

# hub -> (congestion, geopolitical risk)
HUBS = {
    "safe": {"Singapore": (.45, .08), "Colombo": (.35, .12), "Tanjung Pelepas": (.30, .08),
             "Busan Hub": (.35, .05), "Piraeus": (.40, .15), "Salalah": (.25, .20),
             "Cape Town": (.20, .10)},
    "congested": {"Port Klang": (.75, .10), "Los Angeles Hub": (.80, .05),
                  "Jebel Ali Hub": (.55, .30), "Colombo Hub": (.70, .12)},
    "risky": {"Red Sea / Suez": (.50, .85), "Strait of Hormuz": (.35, .70),
              "Black Sea": (.40, .80)},
    "air": {"Doha": (.20, .20), "Hong Kong": (.30, .10), "Frankfurt": (.30, .05)},
}

# mode, label, hub pool, #hubs, cost mult, speed km/day, dwell days/hub, p_delay range, delay-days range
ARCH = {
    "ocean_direct":   ("ocean", "Ocean - direct", "congested", (1, 1), 1.00, 650, 2.0, (.30, .45), (4, 7)),
    "ocean_reliable": ("ocean", "Ocean - reliable hubs", "safe", (1, 2), 1.15, 620, 1.5, (.10, .20), (2, 4)),
    "ocean_shortcut": ("ocean", "Ocean - chokepoint shortcut", "risky", (1, 1), 0.92, 720, 1.5, (.18, .30), (3, 6)),
    "sea_air":        ("sea_air", "Sea-air combo", "safe", (1, 1), 2.30, 1800, 2.0, (.10, .18), (1.5, 3)),
    "air_express":    ("air", "Air express", "air", (0, 1), 4.80, 9000, 1.0, (.05, .10), (1, 2)),
}
FAST = ("sea_air", "air_express")
DEADLINE_FACTOR = {"urgent": (.55, .75), "standard": (1.25, 1.50), "cost_sensitive": (1.60, 2.00)}


def _build_route(rng, arch, dist, weight_t):
    mode, label, pool, nh, cmult, speed, dwell, prange, drange = ARCH[arch]
    k = int(rng.integers(nh[0], nh[1] + 1))
    names = [str(h) for h in rng.choice(list(HUBS[pool]), size=k, replace=False)] if k else []
    attrs = [HUBS[pool][h] for h in names]
    cong = float(np.mean([a[0] for a in attrs])) if attrs else 0.15
    geo = float(max(a[1] for a in attrs)) if attrs else 0.05
    cong = float(np.clip(cong + rng.normal(0, .04), 0.02, 0.98))
    geo = float(np.clip(geo + rng.normal(0, .04), 0.0, 1.0))
    base_ocean = 600 + weight_t * dist * 0.012
    cost = base_ocean * cmult * float(rng.lognormal(0, .06))
    days = dist / speed + dwell * k + (1.0 if mode != "ocean" else 0.0)
    p = float(np.clip(rng.uniform(*prange) + 0.15 * (cong - 0.4), 0.02, 0.8))
    md = float(rng.uniform(*drange))
    return dict(mode=mode, service=label, names=names, cost_usd=cost, transit_days=days,
                delay_prob=p, avg_delay_days=md, congestion=cong, geo_risk=geo, arch=arch)


def generate(n_shipments=40, seed=SEED, out_dir="data"):
    rng = np.random.default_rng(seed)
    ship_rows, true_rows, obs_rows = [], [], []
    rid = 1
    for i in range(1, n_shipments + 1):
        origin, dest, dist = LANES[int(rng.integers(len(LANES)))]
        weight_t = float(rng.uniform(2, 18))
        cargo_value = float(np.clip(rng.lognormal(11.6, .8), 20_000, 600_000))
        priority = str(rng.choice(["urgent", "standard", "cost_sensitive"], p=[.25, .45, .30]))
        archs = ["ocean_direct"]
        for a, pr in [("ocean_reliable", .85), ("ocean_shortcut", .5), ("sea_air", .5), ("air_express", .5)]:
            if rng.random() < pr:
                archs.append(a)
        if priority == "urgent" and not any(a in FAST for a in archs) and rng.random() > .12:
            archs.append("sea_air")
        while len(archs) < 3:
            archs.append(str(rng.choice([a for a in ARCH if a not in archs])))
        archs = archs[:5]
        built = [_build_route(rng, a, dist, weight_t) for a in archs]
        direct_days = next(b["transit_days"] for b in built if b["arch"] == "ocean_direct")
        deadline = direct_days * float(rng.uniform(*DEADLINE_FACTOR[priority]))
        max_risk = 0.50 if cargo_value > 250_000 else 0.70
        sid = f"S{i:03d}"
        ship_rows.append(dict(shipment_id=sid, origin=origin, destination=dest,
                              weight_t=round(weight_t, 1), cargo_value=round(cargo_value, -2),
                              priority=priority, deadline_days=round(deadline, 1), max_risk=max_risk))
        for b in built:
            route_id = f"R{rid:03d}"; rid += 1
            path = " > ".join([origin] + b["names"] + [dest])
            common = dict(route_id=route_id, shipment_id=sid, mode=b["mode"], service=b["service"],
                          path=path, n_legs=len(b["names"]) + 1)
            cols = ["cost_usd", "transit_days", "delay_prob", "avg_delay_days", "congestion", "geo_risk"]
            true_rows.append({**common, **{k: round(b[k], 4) for k in cols}})
            # Observed estimates = truth + estimation noise (the freight quote is exact)
            obs = dict(cost_usd=b["cost_usd"],
                       transit_days=b["transit_days"] * rng.normal(1, .06),
                       delay_prob=np.clip(b["delay_prob"] + rng.normal(0, .06), .01, .9),
                       avg_delay_days=b["avg_delay_days"] * rng.normal(1, .15),
                       congestion=np.clip(b["congestion"] + rng.normal(0, .08), 0, 1),
                       geo_risk=np.clip(b["geo_risk"] + rng.normal(0, .10), 0, 1))
            obs_rows.append({**common, **{k: round(float(v), 4) for k, v in obs.items()}})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ships, obs_df, true_df = pd.DataFrame(ship_rows), pd.DataFrame(obs_rows), pd.DataFrame(true_rows)
    ships.to_csv(out / "shipments.csv", index=False)
    obs_df.to_csv(out / "routes.csv", index=False)
    true_df.to_csv(out / "routes_truth.csv", index=False)
    return ships, obs_df, true_df


if __name__ == "__main__":
    s, o, t = generate()
    print(f"{len(s)} shipments, {len(o)} routes written to data/")
