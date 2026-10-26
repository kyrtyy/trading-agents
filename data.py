"""Point-in-time market data: fetch once to disk, then serve views as of a decision date.

Rule everywhere: a decision for trading day D sees only data dated <= D-1
(prices through D-1 close, headlines dated D-1 or earlier, filings filed before D).
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("TA_DATA", Path(__file__).parent / "data"))

# ticker -> (news query, aliases scrubbed in anonymized mode)
TICKERS = {
    "AAPL": ("AAPL Apple", ["Apple", "iPhone", "iPad", "MacBook", "Mac", "Vision Pro", "Tim Cook", "Cupertino",
                         "Steve Jobs", "iOS", "App Store", "AirPods", "Masimo"]),
    "GOOGL": ("GOOGL Alphabet Google", ["Alphabet", "Google", "YouTube", "Gemini", "Sundar Pichai", "Waymo"]),
    "AMZN": ("AMZN Amazon", ["Amazon", "AWS", "Andy Jassy", "Jeff Bezos", "Prime"]),
    "NVDA": ("NVDA Nvidia", ["Nvidia", "NVIDIA", "Jensen Huang", "GeForce", "Blackwell", "Hopper", "CUDA"]),
    "MSFT": ("MSFT Microsoft", ["Microsoft", "Azure", "Satya Nadella", "Windows", "Copilot", "Xbox", "OpenAI"]),
    "META": ("META Meta Platforms", ["Meta Platforms", "Meta", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"]),
    "JPM": ("JPM JPMorgan", ["JPMorgan Chase", "JPMorgan", "JP Morgan", "Jamie Dimon", "Chase"]),
    "XOM": ("XOM Exxon", ["Exxon Mobil", "ExxonMobil", "Exxon", "Darren Woods", "Pioneer"]),
    "JNJ": ("JNJ Johnson & Johnson", ["Johnson & Johnson", "J&J", "Kenvue", "Janssen"]),
    "KO": ("KO Coca-Cola", ["Coca-Cola", "Coca Cola", "Coke", "James Quincey"]),
}
GLOBAL_QUERY = "stock market economy Federal Reserve"
WINDOWS = {"A": ("2024-01-02", "2024-03-28"), "B": ("2026-04-01", "2026-09-30")}
WARMUP_DAYS = 400   # history before a window, for SMA200/MACD warm-up
# Anonymized runs move dates FORWARD past every model's cutoff (364 weeks: keeps weekdays), so the model
# can neither recall the real period nor import the mood of some other period it remembers.
ANON_SHIFT = dt.timedelta(days=-2548)


def _get(url, headers=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:  # network blips, 429s
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))
            print("retry", url[:80], e, file=sys.stderr)


def _span(window):
    s, e = WINDOWS[window]
    return pd.Timestamp(s), pd.Timestamp(e)


# ---------------------------------------------------------------- fetch (run once)

def fetch_prices(tickers):
    import yfinance as yf
    start = min(_span(w)[0] for w in WINDOWS) - pd.Timedelta(days=WARMUP_DAYS)
    end = max(_span(w)[1] for w in WINDOWS) + pd.Timedelta(days=7)
    (DATA / "prices").mkdir(parents=True, exist_ok=True)
    for t in list(tickers) + ["^IRX"]:
        out = DATA / "prices" / f"{t}.csv"
        if out.exists():
            continue
        tk = yf.Ticker(t)
        adj = tk.history(start=start, end=end, auto_adjust=True)        # split+dividend adjusted: for returns
        raw = tk.history(start=start, end=end, auto_adjust=False)       # split-adjusted only
        df = adj[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        # yfinance back-adjusts for every split up to today. split_factor[r] = product of splits after r;
        # history() rescales by the factor as of D so agents see prices as quoted then (NVDA 2024 ~$500).
        factor = pd.Series(1.0, index=raw.index)
        for when, ratio in tk.splits.items():
            factor[raw.index < when] *= ratio
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df["sa_" + c.lower()] = raw[c]
        df["split_factor"] = factor
        df.index = df.index.tz_localize(None).normalize()
        df.to_csv(out)
        print("prices", t, len(df))


def fetch_news(tickers, pause=2.0):
    """One Google News RSS query per (query, calendar day); resumable, one file per day."""
    days = set()
    for w in WINDOWS:
        s, e = _span(w)
        days |= set(pd.date_range(s - pd.Timedelta(days=10), e))
    queries = {t: TICKERS[t][0] for t in tickers} | {"_GLOBAL": GLOBAL_QUERY}
    for key, q in queries.items():
        d_out = DATA / "news" / key
        d_out.mkdir(parents=True, exist_ok=True)
        for d in sorted(days):
            out = d_out / f"{d:%Y-%m-%d}.json"
            if out.exists():
                continue
            nxt = d + pd.Timedelta(days=1)
            url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
                {"q": f"{q} after:{d:%Y-%m-%d} before:{nxt:%Y-%m-%d}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
            items = _rss_items(url)
            if not items and d.weekday() < 5:  # empty weekday: likely a transient miss, try once more
                time.sleep(pause * 3)
                items = _rss_items(url)
            out.write_text(json.dumps(items))
            time.sleep(pause)
        print("news", key, "done")


def _rss_items(url):
    items = []
    for it in ET.fromstring(_get(url)).findall(".//item"):
        pub = parsedate_to_datetime(it.findtext("pubDate"))
        items.append({"date": pub.strftime("%Y-%m-%d"), "source": it.findtext("source"),
                      "title": it.findtext("title")})
    return items


def fetch_finnhub(tickers):
    key = os.environ["FINNHUB_API_KEY"]
    base = "https://finnhub.io/api/v1/"
    for t in tickers:
        out = DATA / "finnhub" / f"{t}.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        q = lambda path: json.loads(_get(base + path + f"&token={key}"))
        blob = {"insider_tx": [], "insider_sent": []}
        for w in WINDOWS:  # one request per window: a single multi-year request can get truncated
            s, e = _span(w)
            lo, hi = f"{s - pd.Timedelta(days=200):%Y-%m-%d}", f"{e:%Y-%m-%d}"
            blob["insider_tx"] += q(f"stock/insider-transactions?symbol={t}&from={lo}&to={hi}").get("data", [])
            blob["insider_sent"] += q(f"stock/insider-sentiment?symbol={t}&from={lo}&to={hi}").get("data", [])
        blob["insider_sent"].sort(key=lambda x: (x["year"], x["month"]))
        blob["financials"] = q(f"stock/financials-reported?symbol={t}&freq=quarterly").get("data", [])
        out.write_text(json.dumps(blob))
        print("finnhub", t, {k: len(v) for k, v in blob.items()})
        time.sleep(2)


# ---------------------------------------------------------------- point-in-time views

_cache = {}


def _load_prices(t):
    if t not in _cache:
        _cache[t] = pd.read_csv(DATA / "prices" / f"{t}.csv", index_col=0, parse_dates=True)
    return _cache[t]


def trading_days(t, window):
    s, e = _span(window)
    idx = _load_prices(t).index
    return [d for d in idx if s <= d <= e]


def history(t, D):
    """Point-in-time OHLCV strictly before D (as traders saw it)."""
    df = _load_prices(t)
    h = df[df.index < pd.Timestamp(D)]
    k = h["split_factor"].iloc[-1]  # splits after D-1 hadn't happened yet
    out = h[["sa_open", "sa_high", "sa_low", "sa_close"]].set_axis(["open", "high", "low", "close"], axis=1) * k
    out["volume"] = h["sa_volume"] / k
    return out


def indicators(h):
    """Standard technical indicators on a price frame; last row = latest known values."""
    c, hi, lo = h["close"], h["high"], h["low"]
    out = pd.DataFrame(index=h.index)
    out["close"] = c
    for n in (10, 20, 50, 200):
        out[f"sma{n}"] = c.rolling(n).mean()
    out["ema10"] = c.ewm(span=10, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd"], out["macd_signal"] = macd, macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi14"] = 100 - 100 / (1 + gain / loss)
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    out["boll_up"], out["boll_mid"], out["boll_low"] = m + 2 * s, m, m - 2 * s
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rsv = 100 * (c - lo.rolling(9).min()) / (hi.rolling(9).max() - lo.rolling(9).min())
    out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]
    out["zscore20"] = (c - m) / s
    return out


NEWS_PER_DAY = 10  # Google returns ~10/day for 2024 but up to ~50 for 2026; equal caps keep windows comparable


def news(key, D, days=7, limit=40):
    """Headlines dated D-days .. D-1 (never D itself: Google dates are day-granular)."""
    D = pd.Timestamp(D)
    rows, seen = [], set()
    for k in range(1, days + 1):
        day = f"{(D - pd.Timedelta(days=k)):%Y-%m-%d}"
        f = DATA / "news" / key / f"{day}.json"
        if not f.exists():
            continue
        # the query for `day` also returns some items dated the next day: keep only its own date,
        # in Google's relevance order, capped
        for it in [x for x in json.loads(f.read_text()) if x["date"] == day][:NEWS_PER_DAY]:
            if it["title"] not in seen:
                seen.add(it["title"])
                rows.append(it)
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:limit]


def _finnhub(t):
    if ("fh", t) not in _cache:
        blob = json.loads((DATA / "finnhub" / f"{t}.json").read_text())
        # per-window requests can overlap / be ignored by the API: dedupe
        blob["insider_tx"] = list({json.dumps(x, sort_keys=True): x for x in blob["insider_tx"]}.values())
        blob["insider_tx"].sort(key=lambda x: x.get("filingDate") or "", reverse=True)
        blob["insider_sent"] = sorted({(x["year"], x["month"]): x for x in blob["insider_sent"]}.values(),
                                      key=lambda x: (x["year"], x["month"]))
        _cache[("fh", t)] = blob
    return _cache[("fh", t)]


def insider(t, D, days=90):
    blob = _finnhub(t)
    lo = (pd.Timestamp(D) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(D) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tx = [x for x in blob["insider_tx"] if lo <= (x.get("filingDate") or "") <= hi]
    D = pd.Timestamp(D)
    # monthly MSPR is only known once the month is over
    sent = [x for x in blob["insider_sent"] if (x["year"], x["month"]) < (D.year, D.month)][-6:]
    return tx, sent


FIN_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
                "EarningsPerShareDiluted", "GrossProfit", "OperatingIncomeLoss", "Assets", "Liabilities",
                "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
                "NetCashProvidedByUsedInOperatingActivities", "LongTermDebtNoncurrent"]


def _lines(r, sections):
    lines = {}
    for section in sections:
        for it in r.get("report", {}).get(section, []):
            name = it.get("concept", "").split("_")[-1]
            if name in FIN_CONCEPTS and name not in lines:
                lines[name] = it.get("value")
    return lines


def fundamentals(t, D, n=4):
    """Last n quarterly/annual reports filed before D, reduced to key XBRL lines.
    10-Q income/cash-flow lines are fiscal-year-to-date and 10-K lines are full-year, so each is turned into its
    own quarter by subtracting the previous filing of the same fiscal year (always filed earlier: no look-ahead).
    Balance-sheet lines are point-in-time and kept as is. "months" says what period the flow lines cover."""
    blob = _finnhub(t)
    reps = [r for r in blob["financials"] if (r.get("filedDate") or "9")[:10] < f"{pd.Timestamp(D):%Y-%m-%d}"]
    reps.sort(key=lambda r: r["filedDate"], reverse=True)
    out = []
    for r in reps[:n]:
        start, end = r.get("startDate", "")[:10], r.get("endDate", "")[:10]
        flows = _lines(r, ("ic", "cf"))
        prev = [p for p in reps if p.get("startDate", "")[:10] == start and p.get("endDate", "")[:10] < end]
        if prev:  # YTD minus the previous YTD of the same fiscal year = this quarter alone
            p = max(prev, key=lambda p: p["endDate"])
            pf = _lines(p, ("ic", "cf"))
            flows = {k: round(v - pf[k], 2) if isinstance(v, (int, float)) and isinstance(pf.get(k), (int, float)) else None
                     for k, v in flows.items()}
            start = p["endDate"][:10]
        flows = {k: v for k, v in flows.items() if v is not None}
        months = round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4) if start and end else None
        out.append({"form": r.get("form"), "period_end": end, "months": months,
                    "filed": r["filedDate"][:10], "values": {**flows, **_lines(r, ("bs",))}})
    return out


def risk_free(D):
    """Annualized 13-week T-bill yield known before D, as a fraction."""
    irx = _load_prices("^IRX")
    return float(irx[irx.index < pd.Timestamp(D)]["close"].iloc[-1]) / 100


def coverage():
    """Per ticker x window: how much each agent actually gets to see. A vs B should be comparable."""
    rows = []
    for w in WINDOWS:
        for t in list(TICKERS) + ["_GLOBAL"]:
            days = trading_days("AAPL" if t == "_GLOBAL" else t, w)
            n_news = [len(news(t, D, days=1)) for D in days]
            n_news7 = [len(news(t, D)) for D in days]
            row = {"window": w, "ticker": t, "days": len(days),
                   "pct_days_with_news_yday": round(100 * np.mean([n > 0 for n in n_news]), 1),
                   "headlines_7d_mean": round(float(np.mean(n_news7)), 1),
                   "headlines_7d_min": int(np.min(n_news7))}
            if t != "_GLOBAL":
                ins = [insider(t, D) for D in days]
                row["insider_tx_90d_mean"] = round(float(np.mean([len(x[0]) for x in ins])), 1)
                row["insider_sent_months_min"] = min(len(x[1]) for x in ins)
                row["filings_min"] = min(len(fundamentals(t, D)) for D in days)
                row["fin_lines_min"] = min(len(fundamentals(t, D, 1)[0]["values"]) if fundamentals(t, D, 1) else 0
                                           for D in days)
            rows.append(row)
    return pd.DataFrame(rows)


def selfcheck():
    for w, n_expected in (("A", 61), ("B", None)):
        for t in TICKERS:
            days = trading_days(t, w)
            assert len(days) == (n_expected or len(trading_days("AAPL", w))), (t, w, len(days))
            for D in days[::10]:
                ds = f"{D:%Y-%m-%d}"
                h = history(t, D)
                assert h.index.max() < D and len(h) > 250, (t, ds, len(h))
                assert all(r["date"] < ds for r in news(t, D) + news("_GLOBAL", D))
                assert all(f["filed"] < ds for f in fundamentals(t, D))
                tx, sent = insider(t, D)
                assert all(x["filingDate"] < ds for x in tx)
                # every input an agent sees must be non-empty
                assert news(t, D), f"no headlines for {t} before {ds}"
                assert news("_GLOBAL", D), f"no macro headlines before {ds}"
                assert fundamentals(t, D), f"no filings for {t} before {ds}"
    nv = history("NVDA", "2024-02-01")["close"].iloc[-1]
    assert nv > 300, f"NVDA should show pre-split price, got {nv}"
    print("data selfcheck ok")
    print(coverage().to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    tks = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TICKERS)
    if cmd == "prices":
        fetch_prices(tks)
    elif cmd == "news":
        fetch_news(tks)
    elif cmd == "finnhub":
        fetch_finnhub(tks)
    elif cmd == "selfcheck":
        selfcheck()
    elif cmd == "coverage":
        print(coverage().to_string())
