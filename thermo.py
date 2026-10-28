"""Thermodynamics of the agent firm, from the saved runs (no GPU needed).
1. Decision entropy vs sampling temperature: marginal entropy and seed-to-seed disagreement per (ticker, day).
2. Debate relaxation: bull/bear and risk-team stances round by round (3-round arm).
3. Energy: kJ per decision vs portfolio Sharpe along the ablation ladder.
Usage: python thermo.py [--out results/]  (reads TA_RUNS)"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import data
from run import RUNS

SUB = ["AAPL", "GOOGL", "AMZN"]
ACTIONS = ("BUY", "HOLD", "SELL")


def entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def load(arm, cols=("window", "ticker", "date", "action", "state")):
    rows = [json.loads(l) for l in (RUNS / f"{arm}.jsonl").read_text().splitlines() if l.strip()]
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows])


def temperature_table():
    out = []
    for T in ("0.0", "0.3", "0.7", "1.0", "1.3"):
        arms = [f"qwen__full__real__t{T}__s{s}__r1" for s in range(3)]
        dfs = [load(a) for a in arms]
        # T=0.7 arms cover all tickers and both windows: cut them to the sweep's slice
        dfs = [d[(d.window == "A") & d.ticker.isin(SUB)].set_index(["ticker", "date"]).sort_index() for d in dfs]
        acts = pd.concat([d.action.rename(i) for i, d in enumerate(dfs)], axis=1).dropna()
        pooled = collections.Counter(acts.values.ravel())
        per_item = acts.apply(lambda r: entropy(collections.Counter(r).values()), axis=1)
        gaps = [abs(s[0] - s[1]) for d in dfs for s in d.state.map(lambda x: x["investment_stances"])
                if None not in s[:2]]
        rets = backtest.all_returns([str(RUNS / f"{a}.jsonl") for a in arms], ["A"])
        srs = [backtest.metrics(rets[("A", a)][SUB].mean(axis=1), data.risk_free(data.trading_days("AAPL", "A")[0]))["SR"]
               for a in arms]
        out.append({"T": float(T), "items": len(acts), "buy_share": pooled["BUY"] / sum(pooled.values()),
                    "H_marginal_bits": entropy(pooled.values()), "H_seed_bits": per_item.mean(),
                    "seed_disagree": (acts.nunique(axis=1) > 1).mean(), "bull_bear_gap": np.mean(gaps),
                    "SR_mean": np.mean(srs), "SR_sd": np.std(srs, ddof=1), "SR_seeds": [round(s, 2) for s in srs]})
    return pd.DataFrame(out)


def relaxation_table(arm="qwen__full__real__t0.7__s0__r3", rounds=3):
    st = load(arm).state
    rows = []
    for k in range(rounds):
        inv = [s["investment_stances"][2 * k:2 * k + 2] for s in st]
        inv = [x for x in inv if len(x) == 2 and None not in x]
        risk = [s["risk_stances"][3 * k:3 * k + 3] for s in st]
        risk = [x for x in risk if len(x) == 3 and None not in x]
        rows.append({"round": k + 1, "n": len(inv), "bull": np.mean([b for b, _ in inv]),
                     "bear": np.mean([b for _, b in inv]), "bull_bear_gap": np.mean([abs(b - r) for b, r in inv]),
                     "risk_mean": np.mean([np.mean(x) for x in risk]), "risk_spread": np.mean([np.ptp(x) for x in risk])})
    return pd.DataFrame(rows)


def energy_table():
    out = []
    for model, variants in (("qwen", ("single", "analysts", "debate", "full")), ("llama", ("single", "full"))):
        for v in variants:
            arm = f"{model}__{v}__real__t0.7__s0__r1"
            p = json.loads((RUNS / f"{arm}.progress.json").read_text())
            rets = backtest.all_returns([str(RUNS / f"{arm}.jsonl")], ["A", "B"])
            row = {"model": model, "variant": v, "kJ_per_decision": p["kj_per_decision"],
                   "kWh_total": p["gpu_kj"] / 3600, "tokens_per_decision": p["tokens"] / p["finished"]}
            for w in ("A", "B"):
                r = rets[(w, arm)].mean(axis=1)
                bh = rets[(w, "BuyHold")].mean(axis=1)
                rf = data.risk_free(r.index[0])
                row[f"SR_{w}"] = backtest.metrics(r, rf)["SR"]
                row[f"dSR_vs_BH_{w}"] = row[f"SR_{w}"] - backtest.metrics(bh, rf)["SR"]
            out.append(row)
    return pd.DataFrame(out)


def selfcheck():
    assert entropy([1, 1, 1]) == math.log2(3) and entropy([5, 0, 0]) == 0
    assert abs(entropy([1, 1]) - 1) < 1e-12
    print("thermo selfcheck ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    selfcheck()
    if not a.selfcheck:
        Path(a.out).mkdir(exist_ok=True)
        for name, fn in (("thermo_temperature", temperature_table), ("thermo_relaxation", relaxation_table),
                         ("thermo_energy", energy_table)):
            df = fn()
            print(f"\n== {name}\n{df.round(3).to_string(index=False)}")
            df.to_csv(Path(a.out) / f"{name}.csv", index=False)
