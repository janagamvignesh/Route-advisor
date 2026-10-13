"""Sensitivity analyses: weight sweeps and input-parameter perturbations."""
import numpy as np
import pandas as pd
from .config import CRITERIA, default_weights
from .engine import route_metrics, decide


def weight_sweep(m, shp, base_w, crit="cost", steps=21):
    """Vary one criterion's weight 0..1 (others rescaled proportionally); return chosen route per step."""
    others = [k for k in CRITERIA if k != crit]
    s = sum(base_w[k] for k in others)
    rows = []
    for x in np.linspace(0, 1, steps):
        w = {crit: x, **{k: (1 - x) * base_w[k] / s for k in others}}
        rows.append((float(x), decide(m, shp, w)["chosen"]))
    return rows


def parameter_sweep(shipments, routes, factors=(0.75, 1.0, 1.25), n_sims=1000, seed=0):
    """Scale delay-probability and geopolitical-risk estimates; report share of changed decisions."""
    def choose(scale_col, f):
        res = {}
        for _, shp in shipments.iterrows():
            r = routes[routes.shipment_id == shp.shipment_id].copy()
            if scale_col:
                r[scale_col] = np.clip(r[scale_col] * f, 0, 1)
            m = route_metrics(r, shp, n_sims=n_sims, seed=seed)
            res[shp.shipment_id] = decide(m, shp, default_weights(shp.priority))["chosen"]
        return res

    base = choose(None, 1.0)
    out = []
    for target in ["delay_prob", "geo_risk"]:
        for f in factors:
            if f == 1.0:
                continue
            alt = choose(target, f)
            changed = sum(base[k] != alt[k] for k in base)
            out.append(dict(parameter=target, scale=f, pct_changed=changed / len(base)))
    return pd.DataFrame(out)
