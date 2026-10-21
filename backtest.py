"""Turn daily BUY/SELL/HOLD signals into P&L, compare with the paper's rule-based baselines.

Execution: a signal for day D is decided before the open (data <= D-1) and filled at D's open.
Daily return = old position over the overnight gap, compounded with the new position open->close,
minus costs on turnover.
BUY -> +1, SELL -> -1 (0 if long-only), HOLD -> keep. Returns use split+dividend-adjusted prices.
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data

COST_BPS = 10  # commission + slippage per unit of turnover


def positions(signals, allow_short=True):
    pos, cur = [], 0.0
    for s in signals:
        cur = 1.0 if s == "BUY" else (-1.0 if allow_short else 0.0) if s == "SELL" else cur
        pos.append(cur)
    return np.array(pos)


def daily_returns(t, days, signals, allow_short=True, cost_bps=COST_BPS):
    px = data._load_prices(t)
    idx = px.index.get_indexer(pd.DatetimeIndex(days))
    assert (idx > 0).all(), "need the previous close for every day"
    o, c, prev_c = px["open"].values[idx], px["close"].values[idx], px["close"].values[idx - 1]
    pos = positions(signals, allow_short)
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    # compound the overnight leg (old position) with the intraday leg (new position); daily-rebalanced shorts
    r = (1 + prev_pos * (o / prev_c - 1)) * (1 + pos * (c / o - 1)) - 1 - np.abs(pos - prev_pos) * cost_bps / 1e4
    return pd.Series(r, index=pd.DatetimeIndex(days))


def metrics(r, rf_annual=0.0):
    eq = (1 + r).cumprod()
    cr = eq.iloc[-1] - 1
    ar = (1 + cr) ** (252 / len(r)) - 1          # the paper's S2 formula done right (N in years)
    ex = r - rf_annual / 252
    sr = ex.mean() / r.std(ddof=1) * np.sqrt(252) if r.std() > 0 else 0.0
    mdd = (1 - eq / np.maximum.accumulate(np.concatenate([[1.0], eq.values]))[1:]).max()
    return {"CR%": 100 * cr, "AR%": 100 * ar, "SR": sr, "MDD%": 100 * mdd}


@functools.lru_cache(maxsize=None)  # same day count -> same resamples: cheap, and comparable across strategies
def _boot_idx(n, n_boot, block, seed):
    """Stationary block bootstrap index sets (keeps short-range autocorrelation of daily returns)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        i = rng.integers(n)
        for j in range(n):
            out[b, j] = i
            i = rng.integers(n) if rng.random() < 1 / block else (i + 1) % n
    return out


def _sr(x):
    sd = x.std(axis=-1, ddof=1)
    return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)


def sharpe_ci(r, rf_annual=0.0, n_boot=2000, block=5, seed=0):
    """95% CI for the annualized Sharpe ratio."""
    x = r.values - rf_annual / 252
    return np.percentile(_sr(x[_boot_idx(len(x), n_boot, block, seed)]), [2.5, 97.5])


def paired_sharpe_diff(r1, r2, rf_annual=0.0, n_boot=2000, block=5, seed=0):
    """SR(r1) - SR(r2) with a 95% CI from resampling the SAME days for both: the 'does X beat Y' test."""
    x1, x2 = r1.values - rf_annual / 252, r2.values - rf_annual / 252
    idx = _boot_idx(len(x1), n_boot, block, seed)
    d = _sr(x1[idx]) - _sr(x2[idx])
    return float(_sr(x1) - _sr(x2)), *np.percentile(d, [2.5, 97.5])


# ---------------------------------------------------------------- baselines (Appendix S1.1)

def baseline_signals(t, days, name):
    """Each rule sees indicators computed through D-1 only."""
    first, last = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
    ind = data.indicators(data.history(t, last + pd.Timedelta(days=1)))
    ind = ind.shift(1).reindex(pd.DatetimeIndex(days))    # row D holds values as of D-1's close
    prev = ind.shift(1)                                    # as of D-2: for crossovers
    sig = pd.Series("HOLD", index=ind.index)
    if name == "BuyHold":
        sig[:] = "BUY"
    elif name == "MACD":
        sig[(ind.macd > ind.macd_signal) & (prev.macd <= prev.macd_signal)] = "BUY"
        sig[(ind.macd < ind.macd_signal) & (prev.macd >= prev.macd_signal)] = "SELL"
    elif name == "KDJ_RSI":
        sig[(ind.kdj_j < 20) & (ind.rsi14 < 40)] = "BUY"
        sig[(ind.kdj_j > 80) & (ind.rsi14 > 60)] = "SELL"
    elif name == "ZMR":
        sig[ind.zscore20 < -1] = "BUY"
        sig[ind.zscore20 > 1] = "SELL"
    elif name == "SMA":
        sig[(ind.sma10 > ind.sma50) & (prev.sma10 <= prev.sma50)] = "BUY"
        sig[(ind.sma10 < ind.sma50) & (prev.sma10 >= prev.sma50)] = "SELL"
        # crossover strategies start in the regime they find themselves in
        sig.iloc[0] = "BUY" if ind.sma10.iloc[0] > ind.sma50.iloc[0] else "SELL"
    if name == "MACD":
        sig.iloc[0] = "BUY" if ind.macd.iloc[0] > ind.macd_signal.iloc[0] else "SELL"
    return sig.tolist()


BASELINES = ["BuyHold", "MACD", "KDJ_RSI", "ZMR", "SMA"]


def load_arm(path):
    df = pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])
    return df[["window", "ticker", "date", "action", "confidence", "parse_error"]]


def all_returns(arm_files, windows, allow_short=True):
    """{(window, strategy): DataFrame[days x tickers] of daily returns}; arms must cover every day."""
    out = {}
    for w in windows:
        for t in data.TICKERS:
            days = data.trading_days(t, w)
            want = [f"{d:%Y-%m-%d}" for d in days]
            strategies = {b: baseline_signals(t, days, b) for b in BASELINES}
            for f in arm_files:
                df = load_arm(f)
                df = df[(df.window == w) & (df.ticker == t)].drop_duplicates("date").set_index("date")
                if set(want) <= set(df.index):
                    strategies[Path(f).stem] = df.loc[want, "action"].tolist()
            for name, sig in strategies.items():
                out.setdefault((w, name), {})[t] = daily_returns(t, days, sig, allow_short)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def table(arm_files, windows, allow_short=True, n_boot=1000):
    rows = []
    rets = all_returns(arm_files, windows, allow_short)
    for (w, name), R in rets.items():
        rf = data.risk_free(R.index[0])
        # per ticker, plus an equal-weight portfolio over the tickers this strategy covers
        cols = {t: R[t] for t in R.columns}
        if len(R.columns) > 1:
            cols["PORTFOLIO"] = R.mean(axis=1)
        bh = rets[(w, "BuyHold")]
        for t, r in cols.items():
            lo, hi = sharpe_ci(r, rf, n_boot)
            ref = bh[R.columns].mean(axis=1) if t == "PORTFOLIO" else bh[t]
            d, dlo, dhi = paired_sharpe_diff(r, ref, rf, n_boot)
            rows.append({"window": w, "ticker": t, "strategy": name, **metrics(r, rf), "SR_lo": lo, "SR_hi": hi,
                         "dSR_vs_BH": d, "dSR_lo": dlo, "dSR_hi": dhi, "n_tickers": len(R.columns)})
    return pd.DataFrame(rows)


def compare(arm_a, arm_b, windows, allow_short=True):
    """Paired portfolio-level Sharpe difference between two arms (e.g. full vs single)."""
    rets = all_returns([arm_a, arm_b], windows, allow_short)
    for w in windows:
        A, B = rets.get((w, Path(arm_a).stem)), rets.get((w, Path(arm_b).stem))
        if A is None or B is None:
            print(w, "incomplete"); continue
        common = A.columns.intersection(B.columns)
        d, lo, hi = paired_sharpe_diff(A[common].mean(axis=1), B[common].mean(axis=1), data.risk_free(A.index[0]))
        print(f"window {w}: SR({Path(arm_a).stem}) - SR({Path(arm_b).stem}) = {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  over {len(common)} tickers")


def selfcheck():
    days = data.trading_days("AAPL", "A")
    px = data._load_prices("AAPL")
    i0 = px.index.get_loc(days[0])
    # buy & hold on day 1: earn the open->close of day 1, then close-to-close afterwards
    r = daily_returns("AAPL", days, ["BUY"] * len(days), cost_bps=0)
    expect = px["close"].loc[days[-1]] / px["open"].loc[days[0]] - 1
    assert abs((1 + r).prod() - 1 - expect) < 1e-9
    # costs: one entry costs exactly cost_bps
    r_c = daily_returns("AAPL", days, ["BUY"] * len(days), cost_bps=10)
    assert abs(r.iloc[0] - r_c.iloc[0] - 0.001) < 1e-12
    # HOLD keeps position; SELL flips to short
    assert positions(["HOLD", "BUY", "HOLD", "SELL", "HOLD"]).tolist() == [0, 1, 1, -1, -1]
    assert positions(["BUY", "SELL"], allow_short=False).tolist() == [1, 0]
    # metrics on a known path: +10% then -50% -> CR -45%, MDD 50%
    m = metrics(pd.Series([0.10, -0.50]))
    assert abs(m["CR%"] + 45) < 1e-9 and abs(m["MDD%"] - 50) < 1e-9
    # no look-ahead in baselines: changing prices from D onward cannot change the signal for D
    sig = baseline_signals("AAPL", days, "MACD")
    saved = px.copy()
    data._cache["AAPL"] = px.assign(sa_close=px.sa_close.where(px.index < days[30], px.sa_close * 3))
    assert baseline_signals("AAPL", days, "MACD")[:31] == sig[:31]
    data._cache["AAPL"] = saved
    d, lo, hi = paired_sharpe_diff(r, r)
    assert d == 0 and lo == 0 and hi == 0
    d, lo, hi = paired_sharpe_diff(r, -r)
    assert lo < d < hi and abs(d - 2 * metrics(r)["SR"]) < 1e-9
    print("backtest selfcheck ok; AAPL B&H window A:", {k: round(v, 2) for k, v in metrics(r).items()}, "rows", i0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="*")
    p.add_argument("--windows", default="A,B")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--compare", action="store_true", help="paired Sharpe difference of the two given arms")
    p.add_argument("--out", default="results.csv")
    a = p.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.compare:
        compare(*a.arms, a.windows.split(","), not a.long_only)
    else:
        df = table(a.arms, a.windows.split(","), not a.long_only)
        print(df.round(2).to_string())
        df.to_csv(a.out, index=False)
