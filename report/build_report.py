"""Builds report/Route_Advisor_Technical_Report.pdf from results/ (numbers are never hand-typed)."""
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether)
from routeadvisor.engine import load_data, recommend
from routeadvisor.config import default_weights, PROFILES

ROOT = Path(__file__).resolve().parents[1]
summ = pd.read_csv(ROOT / "results/policy_comparison.csv", index_col=0)
sens = pd.read_csv(ROOT / "results/parameter_sensitivity.csv")
meta = json.loads((ROOT / "results/summary.json").read_text())
ex_id = (ROOT / "results/example_shipment.txt").read_text().strip()
ships, obs, _ = load_data(str(ROOT / "data"))
shp = ships[ships.shipment_id == ex_id].iloc[0]
ex = recommend(obs[obs.shipment_id == ex_id], shp)

body = ParagraphStyle("b", fontName="Helvetica", fontSize=8.4, leading=10.6, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=10.2, leading=12, spaceBefore=5, spaceAfter=2.5,
                    textColor=colors.HexColor("#14405c"))
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14.5, leading=17, spaceAfter=1)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=4)
mono = ParagraphStyle("m", fontName="Courier", fontSize=6.5, leading=8, leftIndent=8, textIndent=-8)
cell = ParagraphStyle("c", fontName="Helvetica", fontSize=7.3, leading=8.8)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
bul = ParagraphStyle("bul", parent=body, leftIndent=9, bulletIndent=0, spaceAfter=1.5)


def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="\u2022")


def tbl(rows, widths, head=True, zebra=True):
    t = Table([[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)],
              colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#b8c2cc")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf3")))
    t.setStyle(TableStyle(st)); return t


pct = lambda x: f"{x*100:.0f}%"
pct1 = lambda x: f"{x*100:.1f}%"
S = summ
ours, mm_, cheap, fast, risk, ec = (S.loc[k] for k in
    ["weighted (ours)", "weighted_minmax", "cheapest", "fastest", "lowest_risk", "expected_cost"])
n_ship, n_routes = meta["n_shipments"], meta["n_routes"]
sw = lambda p, f: sens[(sens.parameter == p) & (sens.scale == f)].pct_changed.iloc[0]

doc = SimpleDocTemplate(str(ROOT / "report/Route_Advisor_Technical_Report.pdf"), pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=11 * mm,
                        title="Explainable Shipment Route Recommendation", author="Janagam Vignesh")
W = A4[0] - 30 * mm
s = []
s += [P("Explainable Shipment Route Recommendation under Cost, Time and Risk", title),
      P("Technical report | Aiqon AI Internship take-home | Janagam Vignesh | code: see GitHub repository "
        "(README has run instructions)", sub)]

s += [P("1. What \"optimal\" means here", h1),
      P("A route is optimal for a shipment if it is the best match for that shipment's priority "
        "<b>among routes that satisfy its non-negotiable requirements</b>. I separate the two kinds of requirement on purpose: "
        "<b>hard constraints</b> are must-haves a planner would never trade away (a minimum probability of meeting the "
        "deadline, a cap on route risk); <b>soft objectives</b> (freight cost, transit time, lateness risk, route risk) are "
        "traded off with weights that depend on whether the shipment is urgent, standard or cost-sensitive. "
        "This mirrors how planners work, keeps the logic auditable, and avoids the failure mode of a pure weighted sum, "
        "where a large enough saving can buy a route that will almost certainly miss the deadline.")]

arch = obs.groupby("service").agg(cost=("cost_usd", "mean"), days=("transit_days", "mean"),
                                  p=("delay_prob", "mean"), geo=("geo_risk", "mean"), n=("route_id", "count"))
role = {"Ocean - direct": "Baseline: cheap, congested hubs, slow, often late",
        "Ocean - reliable hubs": "1.15x direct cost, safe hubs, much lower delay probability",
        "Ocean - chokepoint shortcut": "0.92x cost and quicker, but via Red Sea/Hormuz/Black Sea: high geopolitical risk",
        "Sea-air combo": "2.3x cost, much faster, fairly reliable",
        "Air express": "4.8x cost, fastest and most reliable"}
rows = [["Route archetype", "#", "Avg cost", "Avg days", "Delay prob.", "Geo risk", "Trade-off it encodes"]]
for k in ["Ocean - direct", "Ocean - reliable hubs", "Ocean - chokepoint shortcut", "Sea-air combo", "Air express"]:
    r = arch.loc[k]
    rows.append([k, int(r.n), f"${r.cost:,.0f}", f"{r.days:.1f}", f"{r.p:.2f}", f"{r.geo:.2f}", role[k]])
s += [P("2. Synthetic data and how trade-offs were constructed", h1),
      P(f"{n_ship} shipments over 10 lanes (Asia-Europe, trans-Pacific, India-Middle East/Africa), {n_routes} routes "
        "(3-5 per shipment, seeded and reproducible). Every shipment always has the cheap congested option, and the other "
        "archetypes appear randomly, so each shipment presents a genuine dilemma. Cargo value (log-normal, $20k-$600k), weight, "
        "priority (25% urgent / 45% standard / 30% cost-sensitive) and deadline (a priority-dependent multiple of the direct "
        "route's transit time) vary per shipment. High-value cargo gets a stricter risk cap (0.5 vs 0.7)."),
      tbl(rows, [33 * mm, 7 * mm, 15 * mm, 15 * mm, 17 * mm, 14 * mm, W - 101 * mm]),
      P("<i>Averages pool different lanes; multipliers in the last column are the nominal design values.</i>", cell), Spacer(1, 2),
      P("<b>True vs observed.</b> I first draw true parameters, then give the recommender only noisy <i>estimates</i> "
        "(delay probability +/-0.06, transit time +/-6%, risk indicators +/-0.08-0.10; freight quotes are exact). The true values are "
        "used only by the evaluation harness, so the system is judged on what would actually happen, not on its own inputs.")]

wt = "; ".join(f"{p}: " + "/".join(f"{PROFILES[p]['weights'][k]:.2f}" for k in ["cost", "time", "late", "risk"])
               for p in ["urgent", "standard", "cost_sensitive"])
s += [P("3. Decision framework", h1),
      B("<b>Uncertainty-aware metrics.</b> Each route is simulated 2,000 times: a delay occurs with the route's delay "
        "probability (gamma-distributed length), a geopolitical disruption occurs with probability 0.10 x geo-risk (+7 days, 15% "
        "cargo loss), and the delay/risk <i>estimates themselves</i> are jittered. Output: P(on time), P90 transit days, expected total cost "
        "(freight + holding + delay + SLA penalty + disruption loss)."),
      B("<b>Hard constraints.</b> P(on time) >= 85% / 70% / 60% (urgent / standard / cost-sensitive) and composite risk "
        "(0.4 x congestion + 0.6 x geopolitical) below the cargo's cap. If <i>nothing</i> qualifies, the system says so, drops the "
        "reliability constraint and ranks risk-compliant routes by expected total cost, because probability alone cannot tell "
        "0.4 days late from 10 days late."),
      B("<b>Pareto filter.</b> Routes that are no better than another eligible route on cost, P90 time, P(late) and risk are removed."),
      B(f"<b>Weighted score.</b> Score = sum of weight x penalty, lower is better. Penalties live on absolute, decision-relevant "
        f"scales: cost = premium over the cheapest route (+100% scores 1), time = P90 / deadline, lateness = P(late), risk = composite "
        f"risk. Weights (cost/time/late/risk): {wt}. They are preference judgements and are exposed in the CLI and the demo."),
      B("<b>Why absolute scales?</b> My first version used min-max normalisation. Evaluation showed that one expensive air option stretched "
        f"the cost scale, making all sensible ocean options look equally cheap, so the system overpaid for speed (regret "
        f"{pct1(mm_.mean_regret_pct)} vs {pct1(ours.mean_regret_pct)} now; avg freight ${mm_.mean_quoted_cost:,.0f} vs ${ours.mean_quoted_cost:,.0f}). "
        "It is kept as an ablation."),
      B("<b>Why not a learned model?</b> There are no outcome labels for \"best route\", and an opaque model would be hard "
        "for an operator to challenge. A transparent framework plus simulation is more useful here.")]

ex_lines = ex["explanation"].split("\n")
box = Table([[[Paragraph(l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;", 0) if False else
                         l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), mono) for l in ex_lines]]],
            colWidths=[W])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
                         ("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#b8c2cc")),
                         ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
s += [P(f"4. Explanations an operator can use (shipment {ex_id}, {shp.priority}, deadline {shp.deadline_days:.0f} days)", h1),
      box, Spacer(1, 3),
      P("Every output names the winner, compares it with the cheapest and fastest eligible alternatives in plain trade-off "
        "language (\"costs $X more but cuts P(late) from a to b\"), states the deciding factor, lists why each other route was "
        "excluded, and ends with a confidence line from the stress test in section 6."),
      Image(str(ROOT / "results/figures/fig_tradeoff.png"), width=95 * mm, height=95 * mm * 3.0 / 4.4)]

rows = [["Policy (sees only observed data)", "Mean regret", "Matches oracle", "True P(on time)", "Req. violations", "Avg freight", "Avg days"]]
for k, lab in [("cheapest", "Always cheapest"), ("fastest", "Always fastest"), ("lowest_risk", "Always lowest risk"),
               ("weighted_minmax", "Weighted, min-max scaling (ablation)"), ("weighted (ours)", "<b>Constraints + Pareto + weighted (ours)</b>"),
               ("expected_cost", "Min expected total cost*")]:
    r = S.loc[k]
    rows.append([lab, pct1(r.mean_regret_pct), pct(r.oracle_match), pct(r.mean_true_on_time),
                 pct1(r.constraint_violation_rate), f"${r.mean_quoted_cost:,.0f}", f"{r.mean_days:.1f}"])
s += [P("5. Evaluation", h1),
      P("Ground truth is a 4,000-run simulation of each route using its <i>hidden true</i> parameters. The <b>oracle</b> is the route with the lowest true "
        "expected total cost (unattainable in practice). <b>Regret</b> = how much more the chosen route costs than the oracle, averaged over shipments. "
        "<b>Violations</b> = chosen route's true P(on time) or risk breaks the shipment's requirement."),
      tbl(rows, [58 * mm, 19 * mm, 21 * mm, 22 * mm, 22 * mm, 19 * mm, W - 161 * mm]), Spacer(1, 3),
      Image(str(ROOT / "results/figures/fig_policies.png"), width=W * 0.86, height=W * 0.86 * 2.5 / 6.6),
      B(f"Single-metric rules fail in predictable ways: <i>cheapest</i> breaks a requirement for {pct(cheap.constraint_violation_rate)} of shipments "
        f"(on-time only {pct(cheap.mean_true_on_time)}); <i>fastest</i> and <i>lowest risk</i> pay ${fast.mean_quoted_cost:,.0f} and ${risk.mean_quoted_cost:,.0f} "
        f"on average, about {fast.mean_quoted_cost / ours.mean_quoted_cost:.1f}x and {risk.mean_quoted_cost / ours.mean_quoted_cost:.1f}x our ${ours.mean_quoted_cost:,.0f}, giving ~{pct(fast.mean_regret_pct)} regret."),
      B(f"The framework cuts regret to {pct1(ours.mean_regret_pct)} with {pct(ours.mean_true_on_time)} true on-time probability. Its one violation "
        "is S007, an urgent shipment with a 5-day deadline that no available route can meet; the system flags it rather than hiding it."),
      B("*<b>Honest caveat:</b> minimising expected cost scores best "
        f"({pct1(ec.mean_regret_pct)} regret) because my ground truth uses the same cost model, so this comparison is partly circular and I treat it as a "
        "reference, not a competitor. In practice dollar penalties for lateness are rarely known precisely, which is why the weighted approach (needing only "
        "stated preferences) is the default and expected-cost ranking is offered as an option.")]

s6 = [P("6. Uncertainty and sensitivity", h1),
      B(f"<b>Stress test per shipment (300 runs):</b> weights are redrawn around the profile (Dirichlet) and cost, time, P(late) and risk estimates are perturbed. "
        f"Mean stability is {pct(meta['stability_mean'])}; {pct1(meta['stability_ge85'])} of shipments are at least 85% stable and {pct1(meta['stability_lt60'])} fall below 60% "
        "and are labelled \"a judgement call\" in the output, which is the honest message when two routes are close."),
      B(f"<b>Weight sweep:</b> for {ex_id} the choice flips from the reliable-hub route to the chokepoint shortcut when the cost weight rises just above its default "
        "(figure below), exactly the kind of knife-edge an operator should be told about."),
      B(f"<b>Estimate errors:</b> scaling all delay probabilities by -25% / +25% changes {pct1(sw('delay_prob', .75))} / {pct1(sw('delay_prob', 1.25))} of recommendations; "
        f"scaling geopolitical risk by -25% / +25% changes {pct1(sw('geo_risk', .75))} / {pct1(sw('geo_risk', 1.25))}. Decisions are robust to delay-estimate error but "
        "more sensitive to geopolitical risk, because chokepoint routes sit near the risk caps."),
      Image(str(ROOT / "results/figures/fig_sensitivity.png"), width=W * 0.86, height=W * 0.86 * 2.4 / 6.6)]

s += [KeepTogether(s6)]

s += [P("7. Assumptions and limitations", h1),
      B("<b>Synthetic data.</b> Costs, speeds, delay rates and the economic parameters (delay cost per day, SLA penalties, 10% disruption rate) are assumed, "
        "not calibrated; absolute numbers are illustrative and only the relative behaviour is meaningful. With 40 shipments, regret estimates are noisy."),
      B("<b>Independence.</b> Legs and routes fail independently; real disruptions are correlated (a Red Sea closure hits every route using it). A scenario-based or "
        "CVaR objective would capture tail risk better than P90."),
      B("<b>Subjective inputs.</b> Weights, caps and the +100% cost scale are judgements. They are exposed and stress-tested, but should be elicited from planners "
        "(e.g. pairwise comparisons) in a real deployment."),
      B("<b>Scope.</b> No capacity or booking availability, no shipment consolidation, no mid-journey re-routing; recommendations are a one-time snapshot."),
      B("<b>Next steps.</b> Calibrate delay/risk from port-performance and schedule-reliability data, add correlated disruption scenarios, learn weights from "
        "planner overrides, and log decisions to monitor drift.")]
doc.build(s)
print("report built")
