"""Figures for the report (reads results/ produced by evaluate)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from .config import default_weights
from .engine import load_data, route_metrics, decide, stability
from .sensitivity import weight_sweep

OFFSETS = {"Ocean - direct": (8, -14), "Ocean - chokepoint shortcut": (8, -26),
           "Ocean - reliable hubs": (8, 10), "Air express": (-48, 10)}
COL = {"recommended": "#1b7f5c", "eligible": "#4a6fa5", "rejected": "#c0392b"}


def pick_example(ships, obs):
    """First shipment whose recommendation is neither the cheapest nor the fastest route and where
    at least one route is rejected on constraints: the clearest illustration of a trade-off."""
    fallback = None
    for _, shp in ships.iterrows():
        r = obs[obs.shipment_id == shp.shipment_id]
        if len(r) < 4:
            continue
        m = route_metrics(r, shp, n_sims=1000)
        d = decide(m, shp, default_weights(shp.priority))
        t = d["table"]
        mid = (d["chosen"] != t.loc[t.cost_usd.idxmin(), "route_id"]
               and d["chosen"] != t.loc[t.p90_days.idxmin(), "route_id"])
        if not d["fallback"] and mid and (~t.candidate).any():
            return shp, m, d
        fallback = fallback or (shp, m, d)
    return fallback


def make_all(out="results/figures"):
    Path(out).mkdir(parents=True, exist_ok=True)
    ships, obs, _ = load_data()
    summ = pd.read_csv("results/policy_comparison.csv", index_col=0)
    stab = pd.read_csv("results/stability.csv")
    shp, m, d = pick_example(ships, obs)
    t = d["table"]
    Path("results/example_shipment.txt").write_text(shp["shipment_id"])
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    # 1. trade-off scatter for the example shipment
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for _, r in t.iterrows():
        kind = "recommended" if r.route_id == d["chosen"] else ("eligible" if r.candidate else "rejected")
        ax.scatter(r.cost_usd, r.p90_days, s=60 + 400 * r.risk, c=COL[kind], alpha=.85,
                   edgecolor="k", linewidth=.6 if kind != "recommended" else 1.6, zorder=3)
        ax.annotate(f"{r.service.replace('Ocean - ', 'Ocean ')}\nP(late) {r.p_late:.0%}", (r.cost_usd, r.p90_days),
                    textcoords="offset points", xytext=OFFSETS.get(r.service, (7, 6)), fontsize=6.5)
    ax.axhline(shp.deadline_days, color="gray", ls="--", lw=.8)
    ax.text(ax.get_xlim()[1], shp.deadline_days, " deadline", va="bottom", ha="right", fontsize=6.5, color="gray")
    ax.set_xlabel("Freight cost (USD)"); ax.set_ylabel("P90 transit days")
    ax.set_title(f"{shp.shipment_id} ({shp.priority}): green = recommended, red = fails constraints\n"
                 "bubble size = route risk", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{out}/fig_tradeoff.png", dpi=200); plt.close(fig)

    # 2. policy comparison
    pol = ["cheapest", "fastest", "lowest_risk", "weighted_minmax", "weighted (ours)", "expected_cost"]
    names = ["Cheapest", "Fastest", "Lowest\nrisk", "Weighted\n(min-max)", "Weighted\n(ours)", "Min exp.\ncost*"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    cols = ["#9aa5b1"] * 3 + ["#c9a227", "#1b7f5c", "#4a6fa5"]
    axs[0].bar(names, summ.loc[pol, "mean_regret_pct"] * 100, color=cols)
    axs[0].set_ylabel("Mean regret vs oracle (%)"); axs[0].set_title("Realised cost regret (lower = better)", fontsize=8)
    axs[1].bar(names, summ.loc[pol, "constraint_violation_rate"] * 100, color=cols)
    axs[1].set_ylabel("Shipments violating constraints (%)"); axs[1].set_title("Requirement violations on true outcomes", fontsize=8)
    for a in axs: a.tick_params(axis="x", labelsize=6.3)
    fig.tight_layout(); fig.savefig(f"{out}/fig_policies.png", dpi=200); plt.close(fig)

    # 3. sensitivity: weight sweep + stability histogram
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    sweep = weight_sweep(m, shp, default_weights(shp.priority), "cost", steps=41)
    ids = list(dict.fromkeys(r for _, r in sweep)); lab = {r: t.set_index("route_id").loc[r, "service"] for r in ids}
    xs = [x for x, _ in sweep]; ys = [ids.index(r) for _, r in sweep]
    axs[0].step(xs, ys, where="post", color="#1b7f5c", lw=2)
    axs[0].set_yticks(range(len(ids))); axs[0].set_yticklabels([lab[r].replace("Ocean - ", "Ocean ") for r in ids], fontsize=6.5)
    base_cost_w = default_weights(shp.priority)["cost"]
    axs[0].axvline(base_cost_w, color="gray", ls="--", lw=.8)
    axs[0].set_xlabel("Weight on cost (others rescaled)"); axs[0].set_title(f"{shp.shipment_id}: chosen route vs cost weight", fontsize=8)
    axs[1].hist(stab.share * 100, bins=np.arange(0, 105, 10), color="#4a6fa5", edgecolor="w")
    axs[1].set_xlabel("% of stress tests choosing the same route"); axs[1].set_ylabel("Shipments")
    axs[1].set_title("Decision stability across 40 shipments", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig_sensitivity.png", dpi=200); plt.close(fig)
    print("figures written; example shipment:", shp.shipment_id)


if __name__ == "__main__":
    make_all()
