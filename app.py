"""Streamlit demo:  streamlit run app.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from routeadvisor.config import CRITERIA, default_weights, PROFILES
from routeadvisor.engine import load_data, recommend

st.set_page_config(page_title="Route Advisor", layout="wide")
st.title("Shipment Route Advisor")
st.caption("Balances cost, time, reliability and risk. Explains every recommendation.")


@st.cache_data
def _data():
    return load_data("data")


@st.cache_data(show_spinner="Simulating route outcomes...")
def _recommend(sid, weights, method, deadline, min_on_time, max_risk):
    ships, obs, _ = _data()
    shp = ships[ships.shipment_id == sid].iloc[0].to_dict()
    shp.update(deadline_days=deadline, min_on_time=min_on_time, max_risk=max_risk)
    return shp, recommend(obs[obs.shipment_id == sid], shp, weights=dict(weights), method=method)


ships, obs, _ = _data()
sid = st.sidebar.selectbox("Shipment", ships.shipment_id.tolist())
row = ships[ships.shipment_id == sid].iloc[0]
st.sidebar.markdown(f"**{row.origin} -> {row.destination}**  \n{row.weight_t} t, cargo ${row.cargo_value:,.0f}  \n"
                    f"Priority: **{row.priority}**")

st.sidebar.subheader("Requirements")
deadline = st.sidebar.slider("Deadline (days)", 1.0, 80.0, float(row.deadline_days), 0.5)
min_on_time = st.sidebar.slider("Min. chance of on-time arrival", 0.3, 0.99, float(PROFILES[row.priority]["min_on_time"]), 0.01)
max_risk = st.sidebar.slider("Max. route risk", 0.2, 1.0, float(row.max_risk), 0.05)

st.sidebar.subheader("What matters (auto-normalised)")
base = default_weights(row.priority)
use_custom = st.sidebar.checkbox("Override priority weights", False)
weights = {}
for k, label in [("cost", "Cost"), ("time", "Transit time"), ("late", "Lateness risk"), ("risk", "Route risk")]:
    weights[k] = st.sidebar.slider(label, 0.0, 1.0, float(base[k]), 0.05, disabled=not use_custom)
if not use_custom:
    weights = base
method = st.sidebar.radio("Ranking method", ["weighted", "expected_cost"],
                          format_func=lambda x: {"weighted": "Weighted score (preferences)",
                                                 "expected_cost": "Expected total cost ($)"}[x])
if sum(weights.values()) <= 0:
    st.error("Set at least one weight above zero."); st.stop()

shp, res = _recommend(sid, tuple(sorted(weights.items())), method, deadline, min_on_time, max_risk)
t = res["table"]
c = t[t.route_id == res["chosen"]].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommended", f"{c.route_id}", c.service)
m2.metric("Freight cost", f"${c.cost_usd:,.0f}")
m3.metric("On-time chance", f"{c.p_on_time:.0%}", f"deadline {deadline:.0f} d", delta_color="off")
m4.metric("Decision stability", f"{res['stability']['share']:.0%}")
if res["fallback"]:
    st.warning("No route meets every requirement. Showing the best compromise.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Explanation")
    st.code(res["explanation"], language=None)
with right:
    st.subheader("Trade-off map")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for _, r in t.iterrows():
        col = "#1b7f5c" if r.route_id == res["chosen"] else ("#4a6fa5" if r.candidate else "#c0392b")
        ax.scatter(r.cost_usd, r.p90_days, s=60 + 400 * r.risk, c=col, edgecolor="k", alpha=.85, zorder=3)
        ax.annotate(r.route_id, (r.cost_usd, r.p90_days), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.axhline(deadline, color="gray", ls="--", lw=.8)
    ax.set_xlabel("Freight cost (USD)"); ax.set_ylabel("P90 transit days")
    ax.set_title("green = recommended | red = fails requirements | size = risk", fontsize=8)
    st.pyplot(fig)

st.subheader("All route options")
show = t[["route_id", "service", "path", "cost_usd", "mean_days", "p90_days", "p_on_time", "risk",
          "exp_total_cost", "score", "in_pool"]].copy()
show["status"] = ["RECOMMENDED" if r == res["chosen"] else ("eligible" if p else "excluded")
                  for r, p in zip(show.route_id, show.in_pool)]
st.dataframe(show.drop(columns="in_pool").round(2), hide_index=True, width="stretch")
st.caption("Lower score is better. P(on time) and P90 come from 2,000 Monte Carlo simulations that include "
           "uncertainty in the delay and risk estimates.")
