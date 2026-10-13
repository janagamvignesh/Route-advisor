import numpy as np
import pandas as pd
import pytest
from routeadvisor.config import CRITERIA, default_weights, economics
from routeadvisor.data_gen import generate
from routeadvisor.engine import (nondominated, normalise_weights, route_metrics, decide,
                                 recommend, stability, load_data)
from routeadvisor.simulate import simulate_route, summarize
from routeadvisor.sensitivity import weight_sweep


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(n_shipments=12, seed=7, out_dir=str(d))
    return load_data(str(d))


def mk_shipment(**kw):
    s = dict(shipment_id="T1", origin="A", destination="B", weight_t=10, cargo_value=100_000,
             priority="standard", deadline_days=30.0, max_risk=0.6)
    s.update(kw)
    return s


def mk_routes(rows):
    base = dict(shipment_id="T1", mode="ocean", service="x", path="A > B", n_legs=1,
                avg_delay_days=3.0, congestion=0.3, geo_risk=0.1)
    return pd.DataFrame([{**base, "route_id": f"T{i}", **r} for i, r in enumerate(rows)])


def test_dataset_shape(data):
    ships, obs, truth = data
    counts = obs.groupby("shipment_id").size()
    assert counts.between(3, 5).all()
    assert len(truth) == len(obs) and set(ships.shipment_id) == set(obs.shipment_id)
    assert (obs.delay_prob.between(0, 1)).all() and (obs.geo_risk.between(0, 1)).all()


def test_generation_is_deterministic(tmp_path):
    a = generate(5, seed=1, out_dir=str(tmp_path / "a"))[1]
    b = generate(5, seed=1, out_dir=str(tmp_path / "b"))[1]
    pd.testing.assert_frame_equal(a, b)


def test_pareto_filter():
    vals = [[1, 5], [2, 2], [3, 3], [5, 1]]   # [3,3] is dominated by [2,2]
    assert nondominated(vals).tolist() == [True, True, False, True]
    assert nondominated([[1, 1], [1, 1]]).tolist() == [True, True]  # identical rows do not dominate


def test_weights_normalise():
    w = normalise_weights(dict(cost=2, time=1, late=1, risk=0))
    assert pytest.approx(sum(w.values())) == 1 and w["cost"] == 0.5
    with pytest.raises(ValueError):
        normalise_weights(dict(cost=0, time=0, late=0, risk=0))


def test_simulation_is_reproducible_and_sensible():
    shp = mk_shipment()
    r = mk_routes([dict(cost_usd=2000, transit_days=20, delay_prob=0.3)]).iloc[0]
    a1, c1 = simulate_route(r, shp, 5000, np.random.default_rng(1))
    a2, c2 = simulate_route(r, shp, 5000, np.random.default_rng(1))
    assert np.allclose(a1, a2) and np.allclose(c1, c2)
    assert a1.min() > 15 and c1.min() >= 2000          # never cheaper than the quote
    s = summarize(a1, c1, 30.0)
    assert s["p_on_time"] > 0.9 and s["p_late"] == pytest.approx(1 - s["p_on_time"])


def test_hard_constraint_beats_cheapness():
    shp = mk_shipment(deadline_days=25.0, priority="standard")
    routes = mk_routes([
        dict(cost_usd=1000, transit_days=40, delay_prob=.4),            # cheap but misses the deadline
        dict(cost_usd=1500, transit_days=18, delay_prob=.1)])
    res = recommend(routes, shp, n_stability=20)
    assert res["chosen"] == "T1" and not res["fallback"]
    assert bool(res["table"].set_index("route_id").loc["T0", "fail_ontime"])
    assert "T0" in res["explanation"] and "deadline" in res["explanation"]


def test_risk_cap_excludes_risky_route():
    shp = mk_shipment(max_risk=0.5)
    routes = mk_routes([
        dict(cost_usd=900, transit_days=15, delay_prob=.1, geo_risk=.95, congestion=.5),   # risk ~0.77
        dict(cost_usd=1400, transit_days=18, delay_prob=.1, geo_risk=.1, congestion=.3)])
    assert recommend(routes, shp, n_stability=10)["chosen"] == "T1"


def test_weights_change_decision():
    shp = mk_shipment(deadline_days=60.0)
    routes = mk_routes([
        dict(cost_usd=1000, transit_days=30, delay_prob=.3),
        dict(cost_usd=1500, transit_days=14, delay_prob=.1)])
    m = route_metrics(routes, shp, n_sims=1500)
    cheap = decide(m, shp, dict(cost=1, time=0, late=0, risk=0))["chosen"]
    fast = decide(m, shp, dict(cost=0, time=1, late=0, risk=0))["chosen"]
    assert cheap == "T0" and fast == "T1"
    sweep = weight_sweep(m, shp, default_weights("standard"), "cost", steps=11)
    assert sweep[0][1] == "T1" and sweep[-1][1] == "T0"      # decision flips as cost weight rises


def test_dominated_route_is_removed():
    shp = mk_shipment(deadline_days=60.0)
    routes = mk_routes([
        dict(cost_usd=1000, transit_days=20, delay_prob=.1),
        dict(cost_usd=1200, transit_days=25, delay_prob=.2, geo_risk=.2)])  # worse on everything
    res = decide(route_metrics(routes, shp, n_sims=1000), shp, default_weights("standard"))
    t = res["table"].set_index("route_id")
    assert bool(t.loc["T1", "dominated"]) and res["chosen"] == "T0"


def test_fallback_when_nothing_feasible():
    shp = mk_shipment(deadline_days=3.0, priority="urgent")
    routes = mk_routes([dict(cost_usd=1000, transit_days=30, delay_prob=.3),
                        dict(cost_usd=4000, transit_days=8, delay_prob=.1)])
    res = recommend(routes, shp, n_stability=10)
    assert res["fallback"] and res["chosen"] == "T1" and "WARNING" in res["explanation"]


def test_stability_in_unit_interval_and_clear_winner_is_stable():
    shp = mk_shipment(deadline_days=25.0)
    routes = mk_routes([dict(cost_usd=1000, transit_days=45, delay_prob=.5),
                        dict(cost_usd=1100, transit_days=15, delay_prob=.05)])
    m = route_metrics(routes, shp, n_sims=1000)
    st = stability(m, shp, default_weights("standard"), n=100)
    assert 0 <= st["share"] <= 1 and st["share"] > 0.95


def test_economics_override():
    shp = mk_shipment(min_on_time=0.99)
    assert economics(shp)["min_on_time"] == 0.99
    assert economics(mk_shipment())["min_on_time"] == 0.70


def test_every_dataset_shipment_gets_a_recommendation(data):
    ships, obs, _ = data
    for _, s in ships.iterrows():
        res = recommend(obs[obs.shipment_id == s.shipment_id], s, n_sims=500, n_stability=10)
        assert res["chosen"] in set(obs[obs.shipment_id == s.shipment_id].route_id)
        assert res["chosen"] in res["explanation"]
