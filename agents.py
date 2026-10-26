"""TradingAgents graph (Xiao et al., arXiv 2412.20138), plus the ablation variants.

Agents talk through a structured state dict (the paper's "communication protocol"): each role
reads only the fields it needs and writes one report; debates are the only free-form dialogue.
Deviation from the paper: analysts get their tool outputs pre-fetched into the prompt instead of
calling tools via ReAct, so every model/variant sees byte-identical inputs.
"""
import asyncio
import json
import re
import time
import zlib

import pandas as pd

import data

VARIANTS = ["single", "analysts", "debate", "full"]
ACTIONS = ["BUY", "SELL", "HOLD"]
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["action", "confidence", "rationale"],
}
ACTION_RULES = ("Actions: BUY = hold a long position from today's open; SELL = hold a short position "
                "(exit any long); HOLD = keep whatever position is currently held.")


# ---------------------------------------------------------------- anonymization

class View:
    """Renders point-in-time data as text; in anon mode hides identity (name, real dates, price level)."""

    def __init__(self, ticker, D, anon):
        self.t, self.D, self.anon = ticker, pd.Timestamp(D), anon
        h = data.history(ticker, D)
        # anon: rescale prices so the last known close is 100 (level would reveal the stock)
        self.k = 100 / h["close"].iloc[-1] if anon else 1.0
        self.h = h

    def name(self):
        return "Company XYZ (ticker XYZ)" if self.anon else self.t

    def date(self, d):
        d = pd.Timestamp(d)
        return f"{(d - data.ANON_SHIFT if self.anon else d):%Y-%m-%d}"

    def scrub(self, text):
        if not self.anon:
            return text
        for a in sorted(data.TICKERS[self.t][1], key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(a)}(s|'s)?\b", "the Company", text, flags=re.I)
        text = re.sub(rf"\b{self.t}\b", "XYZ", text)
        return re.sub(r"\b(19|20)\d\d\b", "[year]", text)

    def money(self, v):
        return v * self.k if self.anon and isinstance(v, (int, float)) else v

    # -- tool outputs (what the paper's analysts would retrieve)
    def market(self):
        ind = data.indicators(self.h).tail(15)
        cols = ["close", "sma20", "sma50", "sma200", "macd", "macd_signal", "rsi14", "boll_up", "boll_low",
                "atr14", "kdj_k", "kdj_d", "zscore20"]
        ind = ind[cols].copy()
        price_cols = ["close", "sma20", "sma50", "sma200", "macd", "macd_signal", "boll_up", "boll_low", "atr14"]
        ind[price_cols] = ind[price_cols] * self.k
        ind.index = [self.date(d) for d in ind.index]
        vol = self.h["volume"].tail(15)
        ind["volume_vs_20d_avg"] = (vol / self.h["volume"].rolling(20).mean().tail(15)).round(2).values
        r = self.h["close"].pct_change()
        stats = (f"Returns: 5d {self.h['close'].iloc[-1] / self.h['close'].iloc[-6] - 1:+.2%}, "
                 f"20d {self.h['close'].iloc[-1] / self.h['close'].iloc[-21] - 1:+.2%}, "
                 f"60d {self.h['close'].iloc[-1] / self.h['close'].iloc[-61] - 1:+.2%}; "
                 f"20d annualized vol {r.tail(20).std() * 252 ** 0.5:.1%}; "
                 f"52w high {self.h['high'].tail(252).max() * self.k:.2f}, low {self.h['low'].tail(252).min() * self.k:.2f}")
        return stats + "\n\nLast 15 sessions (technical indicators):\n" + ind.round(2).to_string()

    def headlines(self, key, days, limit):
        rows = data.news(key, self.D, days, limit)
        # anon drops the outlet: names like "MacRumors" or "9to5Mac" identify the company
        return "\n".join(f"[{self.date(r['date'])}] " + ("" if self.anon else f"({r['source']}) ") +
                         self.scrub(re.sub(r"\s+-\s+[^-]+$", "", r["title"]) if self.anon else r["title"])
                         for r in rows) or "(none)"

    def company_news(self):
        # 20 = what window A (2024) always reaches; window B has more, capping keeps the windows equally informed
        return self.headlines(self.t, 7, 20)

    def global_news(self):
        return self.headlines("_GLOBAL", 3, 20)

    def insider(self):
        tx, sent = data.insider(self.t, self.D)
        lines = [f"[{self.date(x['filingDate'])}] {'insider' if self.anon else x['name']}: change {x['change']:+,} sh"
                 f" at {self.money(x.get('transactionPrice') or 0):.2f} (code {x.get('transactionCode')})" for x in tx[:25]]
        mspr = [self.date(pd.Timestamp(x["year"], x["month"], 15))[:7] +
                f": MSPR {x['mspr']:+.1f}, net change {x['change']:+,}" for x in sent]
        return ("Insider transactions, last 90 days (newest first):\n" + ("\n".join(lines) or "(none)") +
                "\n\nMonthly insider sentiment (MSPR, -100..100):\n" + ("\n".join(mspr) or "(none)"))

    def fundamentals(self):
        out = []
        for r in data.fundamentals(self.t, self.D):
            vals = {k: self.money(v) for k, v in r["values"].items()}
            out.append(f"{r['form']}, income/cash-flow lines cover the {r['months']} months ending "
                       f"{self.date(r['period_end'])}, balance sheet as of that date (filed {self.date(r['filed'])}): "
                       + ", ".join(f"{k}={v:,.0f}" if isinstance(v, (int, float)) and abs(v) > 100 else f"{k}={v}"
                                   for k, v in vals.items()))
        return "\n".join(out) or "(no filings available)"


# ---------------------------------------------------------------- LLM plumbing

class Agent:
    """One decision's LLM session: records every call for the transcript and token/latency accounting."""

    def __init__(self, client, model, temperature, seed):
        self.client, self.model, self.temp, self.seed = client, model, temperature, seed
        self.calls = []

    async def ask(self, role, system, user, max_tokens=1000, schema=None):
        kw = {}
        if schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}}
        t0 = time.time()
        r = await self.client.chat.completions.create(
            # per-role seed: reproducible regardless of the order concurrent calls finish in
            model=self.model, temperature=self.temp, seed=self.seed * 1000 + zlib.crc32(role.encode()) % 1000,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        text = r.choices[0].message.content or ""
        self.calls.append({"role": role, "text": text, "prompt_tokens": r.usage.prompt_tokens,
                           "completion_tokens": r.usage.completion_tokens, "seconds": round(time.time() - t0, 2),
                           "finish": r.choices[0].finish_reason})
        return text

    async def decide(self, role, system, user):
        text = await self.ask(role, system, user, max_tokens=400, schema=DECISION_SCHEMA)
        try:
            d = json.loads(text)
        except json.JSONDecodeError:  # truncated JSON: salvage the action, flag it
            m = re.search(r'"action"\s*:\s*"(BUY|SELL|HOLD)"', text)
            d = {"action": m.group(1) if m else "HOLD", "confidence": None, "rationale": text, "parse_error": True}
        return d


STANCE_RULE = ("End your reply with one final line exactly of the form 'STANCE: N' where N is an integer from "
               "-2 to +2 ({}).")
STANCE_RE = re.compile(r"STANCE:\s*([+-]?\d)")


def stance(text):
    m = STANCE_RE.findall(text)
    return max(-2, min(2, int(m[-1]))) if m else None


def _sys(role, goal, name, today):
    return (f"You are the {role} at a trading firm. Goal: {goal}\nThe stock under analysis is {name}. "
            f"Today is {today}, before the market opens; you only know information up to yesterday's close. "
            "Be specific, cite the numbers you rely on, and keep it under 350 words.")


# ---------------------------------------------------------------- the graph

async def run_decision(client, model, ticker, D, variant, anon=False, temperature=0.7, seed=0, rounds=1):
    v = View(ticker, D, anon)
    ag = Agent(client, model, temperature, seed)
    name, today = v.name(), v.date(D)
    S = {}  # the structured shared state

    if variant == "single":
        blob = (f"MARKET DATA\n{v.market()}\n\nCOMPANY NEWS\n{v.company_news()}\n\nMACRO NEWS\n{v.global_news()}\n\n"
                f"INSIDER ACTIVITY\n{v.insider()}\n\nFUNDAMENTALS\n{v.fundamentals()}")
        S["decision"] = await ag.decide(
            "single", _sys("sole trader", "decide today's trade from all available data.", name, today),
            blob + f"\n\n{ACTION_RULES}\nReturn your decision as JSON.")
        return S, ag.calls

    # I. Analyst team (independent, so the four calls run concurrently)
    specs = {
        "market_report": ("Market (technical) Analyst", "analyze price trends and technical indicators "
                          "(MACD, RSI, Bollinger Bands, moving averages, KDJ, volume) to judge momentum and timing.",
                          v.market()),
        "sentiment_report": ("Sentiment Analyst", "gauge investor and insider sentiment from insider trading "
                             "activity and the tone of recent coverage.", v.insider() + "\n\nRECENT HEADLINES\n" + v.company_news()),
        "news_report": ("News Analyst", "assess company-specific events and the macroeconomic backdrop that "
                        "could move the stock.", "COMPANY NEWS\n" + v.company_news() + "\n\nMACRO NEWS\n" + v.global_news()),
        "fundamentals_report": ("Fundamentals Analyst", "evaluate financial health, growth, profitability and "
                                "valuation from the latest filings.", v.fundamentals()),
    }
    reports = await asyncio.gather(*[ag.ask(k, _sys(r, g, name, today), f"Data:\n{d}\n\nWrite your analysis report.")
                                     for k, (r, g, d) in specs.items()])
    S.update(dict(zip(specs, reports)))
    analyst_block = "\n\n".join(f"## {k.replace('_', ' ').title()}\n{S[k]}" for k in specs)

    if variant == "analysts":
        S["decision"] = await ag.decide(
            "trader", _sys("Trader", "turn the analyst reports into today's trade.", name, today),
            analyst_block + f"\n\n{ACTION_RULES}\nReturn your decision as JSON.")
        return S, ag.calls

    # II. Researcher team: bull vs bear, n rounds, then the facilitator picks a side
    hist = []
    for i in range(rounds):
        for side, goal in (("Bull Researcher", "build the strongest evidence-based case FOR investing."),
                           ("Bear Researcher", "build the strongest evidence-based case AGAINST investing.")):
            debate = "\n\n".join(hist) or "(you open the debate)"
            arg = await ag.ask(f"{side.split()[0].lower()}_r{i}", _sys(side, goal, name, today),
                               f"{analyst_block}\n\nDEBATE SO FAR:\n{debate}\n\nMake your argument and rebut the other "
                               "side directly. " + STANCE_RULE.format("-2 = the stock should clearly be avoided/shorted, "
                                                                      "+2 = clearly bought; be honest about how the "
                                                                      "evidence actually leans"))
            hist.append(f"{side}: {arg}")
    S["investment_debate"] = hist
    S["investment_stances"] = [stance(h) for h in hist]
    S["investment_plan"] = await ag.ask(
        "research_manager", _sys("Research Manager (debate facilitator)", "judge the bull/bear debate, pick the "
                                 "prevailing view and write an investment plan.", name, today),
        f"{analyst_block}\n\nDEBATE:\n" + "\n\n".join(hist) + "\n\nState which side prevailed and why, then the plan.")

    if variant == "debate":
        S["decision"] = await ag.decide(
            "trader", _sys("Trader", "turn the investment plan into today's trade.", name, today),
            f"{analyst_block}\n\nINVESTMENT PLAN:\n{S['investment_plan']}\n\n{ACTION_RULES}\nReturn your decision as JSON.")
        return S, ag.calls

    # III. Trader proposes
    S["trader_plan"] = await ag.ask(
        "trader", _sys("Trader", "propose today's trade with sizing and risk notes.", name, today),
        f"{analyst_block}\n\nINVESTMENT PLAN:\n{S['investment_plan']}\n\n{ACTION_RULES}\n"
        "End with a line 'FINAL TRANSACTION PROPOSAL: BUY|SELL|HOLD'.")

    # IV. Risk team: aggressive / conservative / neutral, n rounds
    hist = []
    for i in range(rounds):
        for side, goal in (("Aggressive Risk Analyst", "champion high-reward opportunities in the trader's plan."),
                           ("Conservative Risk Analyst", "protect capital; stress the downside and volatility."),
                           ("Neutral Risk Analyst", "weigh both views for a balanced risk-adjusted stance.")):
            debate = "\n\n".join(hist) or "(you speak first)"
            arg = await ag.ask(f"{side.split()[0].lower()}_r{i}", _sys(side, goal, name, today),
                               f"TRADER PLAN:\n{S['trader_plan']}\n\nMARKET REPORT:\n{S['market_report']}\n\n"
                               f"RISK DISCUSSION SO FAR:\n{debate}\n\nGive your assessment and respond to the others. "
                               + STANCE_RULE.format("-2 = cut exposure / go defensive, 0 = keep the trader's plan, "
                                                    "+2 = take more risk than proposed"))
            hist.append(f"{side}: {arg}")
    S["risk_debate"] = hist
    S["risk_stances"] = [stance(h) for h in hist]

    # V. Fund manager approves / adjusts
    S["decision"] = await ag.decide(
        "fund_manager", _sys("Fund Manager", "make the final call after the risk discussion.", name, today),
        f"TRADER PLAN:\n{S['trader_plan']}\n\nRISK DISCUSSION:\n" + "\n\n".join(hist) +
        f"\n\n{ACTION_RULES}\nReturn your final decision as JSON.")
    return S, ag.calls
