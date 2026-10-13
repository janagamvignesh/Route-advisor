# Route Advisor: explainable shipment route recommendation

Take-home for the Aiqon AI Internship. Given a shipment and 3-5 candidate routes, the system
**evaluates** each route under uncertainty, **recommends** one, and **explains** the choice in
plain language, including why the other routes were not picked and how confident the decision is.

Full write-up: [`report/Route_Advisor_Technical_Report.pdf`](report/Route_Advisor_Technical_Report.pdf) (3 pages).

## Quick start

```bash
pip install -r requirements.txt

python -m routeadvisor.cli recommend S003              # recommendation + explanation
python -m routeadvisor.cli recommend S007              # an infeasible shipment (shows the warning path)
python -m routeadvisor.cli recommend S003 --weights cost=0.7,time=0.1,late=0.1,risk=0.1
python -m routeadvisor.cli recommend S003 --method expected_cost --deadline 20 --max-risk 0.4
python -m routeadvisor.cli batch                       # all shipments -> results/recommendations.csv
streamlit run app.py                                   # interactive demo
pytest -q                                              # 13 tests
```

Reproduce everything from scratch (about 2 minutes):

```bash
python -m routeadvisor.cli generate      # synthetic data (seeded) -> data/
python -m routeadvisor.cli evaluate      # evaluation + sensitivity -> results/
python -m routeadvisor.cli figures       # report figures
python report/build_report.py            # PDF report (numbers are read from results/)
```

## How it works

1. **Monte Carlo metrics** (`simulate.py`, `engine.route_metrics`): 2,000 simulated trips per route
   (random delays, geopolitical disruptions, and jitter on our own estimates) give P(on time),
   P90 transit days, and expected total cost.
2. **Hard constraints**: minimum on-time probability (by priority) and a route-risk cap. If nothing
   qualifies, the system warns and ranks risk-compliant routes by expected total cost.
3. **Pareto filter**: drop routes that are no better than another eligible one on every objective.
4. **Weighted score** over cost, time, lateness and risk, with weights per priority profile
   (urgent / standard / cost-sensitive) on absolute, decision-relevant scales.
5. **Explanation** (`explain.py`) and a **stability test** (300 perturbations of weights and estimates).

All assumptions and parameters live in `routeadvisor/config.py`.

## Data

Synthetic and reproducible (`data_gen.py`): 40 shipments, 142 routes, five route archetypes that
encode real trade-offs (cheap but congested, reliable but dearer, risky shortcut, sea-air, air).
`routes.csv` holds noisy *observed* estimates; `routes_truth.csv` holds hidden *true* parameters,
used only by the evaluation harness.

## Evaluation (summary)

Each policy sees only observed data and is scored by simulating outcomes with the hidden true
parameters. Regret is measured against an oracle that knows the true expected cost.

| Policy | Mean regret | True P(on time) | Requirement violations | Avg freight |
|---|---|---|---|---|
| Always cheapest | 108.5% | 74% | 32.5% | $1,901 |
| Always fastest | 89.7% | 97% | 5.0% | $7,199 |
| Always lowest risk | 89.4% | 92% | 7.5% | $5,494 |
| Weighted, min-max scaling (ablation) | 20.5% | 96% | 2.5% | $3,972 |
| **Constraints + Pareto + weighted (ours)** | **10.0%** | **94%** | **2.5%** | **$2,758** |
| Min expected total cost* | 1.7% | 94% | 2.5% | $2,589 |

\*Scores best because the ground-truth simulator uses the same cost model, so the comparison is
partly circular; see the report. The single remaining violation is S007, a shipment no route can serve in time.

## Layout

```
routeadvisor/   config, data_gen, simulate, engine, explain, sensitivity, evaluate, figures, cli
app.py          Streamlit demo
tests/          13 unit tests (constraints, Pareto, fallback, sensitivity, determinism, ...)
data/ results/  generated datasets, evaluation tables and figures
report/         PDF report and its builder
docs/           demo script
```

## Limitations

Synthetic, uncalibrated inputs; independent route failures (no correlated disruptions); weights are
stated preferences, not learned; no capacity or re-routing. See section 7 of the report.
