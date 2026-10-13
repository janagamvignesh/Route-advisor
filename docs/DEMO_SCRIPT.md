# 5-minute demo script (screen recording)

**0:00-0:30 Problem.** "Choosing a route is a trade-off between cost, time and risk. This tool recommends one and explains why."

**0:30-1:30 Data and framework.** Show `data/routes.csv` briefly. State the idea: hard requirements first
(on-time probability, risk cap), then a priority-weighted score. Mention Monte Carlo for uncertainty.

**1:30-3:00 Streamlit, shipment S003.** `streamlit run app.py`. Point at the recommendation card, the trade-off map
(red = excluded, green = chosen), and the explanation text. Read one sentence: it costs a little more than the cheapest
route but cuts the chance of missing the deadline and the route risk.

**3:00-3:45 What-if.** Tick "Override priority weights", raise Cost; watch the recommendation flip to the shortcut.
Say: "this is why the stability number is shown; for S003 it is a judgement call."

**3:45-4:20 Infeasible case.** Select S007 (5-day deadline). Show the warning and that it ranks by expected total cost.

**4:20-5:00 Evaluation and limits.** Show the policy table from the report: single-metric rules have about 90-108% regret,
this framework about 10%. Be upfront about the circularity caveat and the synthetic data.
