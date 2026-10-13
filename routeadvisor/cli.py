"""Command line interface:  python -m routeadvisor.cli <command>"""
import argparse
import json
import pandas as pd
from .config import default_weights, CRITERIA
from .engine import load_data, recommend


def _parse_weights(s, priority):
    if not s:
        return default_weights(priority)
    w = {k: 0.0 for k in CRITERIA}
    for part in s.split(","):
        k, v = part.split("=")
        w[k.strip()] = float(v)
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(prog="routeadvisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="create the synthetic dataset in data/")
    g.add_argument("--n", type=int, default=40)
    r = sub.add_parser("recommend", help="recommend a route for one shipment")
    r.add_argument("shipment_id")
    r.add_argument("--weights", help="e.g. cost=0.5,time=0.1,late=0.2,risk=0.2 (auto-normalised)")
    r.add_argument("--method", choices=["weighted", "expected_cost"], default="weighted")
    r.add_argument("--deadline", type=float, help="override deadline in days")
    r.add_argument("--min-on-time", type=float, help="override required on-time probability (0-1)")
    r.add_argument("--max-risk", type=float, help="override route-risk cap (0-1)")
    r.add_argument("--json", action="store_true", help="machine-readable output")
    b = sub.add_parser("batch", help="recommend for every shipment, write results/recommendations.csv")
    e = sub.add_parser("evaluate", help="run the full evaluation + sensitivity study")
    e.add_argument("--quick", action="store_true")
    sub.add_parser("figures", help="regenerate report figures")
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        from .data_gen import generate
        s, o, _ = generate(a.n)
        print(f"Wrote {len(s)} shipments and {len(o)} routes to data/")
    elif a.cmd == "recommend":
        ships, obs, _ = load_data()
        shp = ships[ships.shipment_id == a.shipment_id]
        if shp.empty:
            raise SystemExit(f"unknown shipment {a.shipment_id}")
        shp = shp.iloc[0].to_dict()
        if a.deadline: shp["deadline_days"] = a.deadline
        if a.min_on_time: shp["min_on_time"] = a.min_on_time
        if a.max_risk: shp["max_risk"] = a.max_risk
        res = recommend(obs[obs.shipment_id == a.shipment_id], shp,
                        weights=_parse_weights(a.weights, shp["priority"]), method=a.method)
        if a.json:
            print(json.dumps(dict(shipment=shp, recommended=res["chosen"], fallback=res["fallback"],
                                  stability=res["stability"]["share"], explanation=res["explanation"]), default=str, indent=2))
        else:
            print(res["explanation"])
    elif a.cmd == "batch":
        ships, obs, _ = load_data()
        rows = []
        for _, shp in ships.iterrows():
            res = recommend(obs[obs.shipment_id == shp.shipment_id], shp, n_stability=100, explain_text=False)
            c = res["table"].set_index("route_id").loc[res["chosen"]]
            rows.append(dict(shipment_id=shp.shipment_id, priority=shp.priority, recommended=res["chosen"],
                             service=c.service, cost_usd=round(c.cost_usd), p_on_time=round(c.p_on_time, 3),
                             risk=round(c.risk, 3), fallback=res["fallback"], stability=res["stability"]["share"]))
        out = pd.DataFrame(rows)
        out.to_csv("results/recommendations.csv", index=False)
        print(out.to_string(index=False))
    elif a.cmd == "evaluate":
        from .evaluate import run
        summ, stab, sweep, _ = run(quick=a.quick)
        pd.set_option("display.width", 200)
        print(summ.round(3).to_string())
        print(sweep.to_string() if len(sweep) else "")
    elif a.cmd == "figures":
        from .figures import make_all
        make_all()


if __name__ == "__main__":
    main()
