"""Memorization probe: ask the bare model for closing prices with no context.

If error on window A is far below a no-knowledge baseline (the last close before the window), the
model has memorized prices from the backtest period. Window B (after every model's cutoff) is the control.
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

import data


PRICE_SCHEMA = {"type": "object", "properties": {"price": {"type": "number"}}, "required": ["price"]}


async def ask(client, model, t, d):
    # JSON-constrained so refusals/hedges ("I don't have data for 2024...") can't be misparsed as a price
    r = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=30,
        response_format={"type": "json_schema", "json_schema": {"name": "price", "schema": PRICE_SCHEMA}},
        messages=[{"role": "user", "content": f"What was the closing price of {t} stock on {d}, in USD? "
                                               "Give your best guess even if unsure."}])
    try:
        return float(json.loads(r.choices[0].message.content)["price"])
    except (ValueError, KeyError, TypeError):
        return None


async def identify(client, model, t, D):
    """Show the anonymized analyst inputs and ask the model to de-anonymize them."""
    import agents
    v = agents.View(t, D, anon=True)
    blob = "\n\n".join([v.market(), v.company_news(), v.global_news(), v.insider(), v.fundamentals()])
    r = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=60,
        messages=[{"role": "user", "content": blob + "\n\nThis data was anonymized. Which real US-listed company "
                   "is it, and in which real calendar year and month was it observed? Answer exactly as "
                   "'TICKER: <ticker>; YEAR: <yyyy>; MONTH: <mm>' with your best guess."}])
    text = r.choices[0].message.content or ""
    tk = re.search(r"TICKER:\s*\$?([A-Z.]+)", text)
    yr = re.search(r"YEAR:\s*(\d{4})", text)
    mo = re.search(r"MONTH:\s*(\d{1,2})", text)
    return {"ticker": t, "date": f"{pd.Timestamp(D):%Y-%m-%d}", "answer": text,
            "ticker_hit": bool(tk and tk.group(1) in (t, "GOOG" if t == "GOOGL" else t)),
            "year_hit": bool(yr and int(yr.group(1)) == pd.Timestamp(D).year),
            "month_hit": bool(yr and mo and int(yr.group(1)) == pd.Timestamp(D).year
                              and int(mo.group(1)) == pd.Timestamp(D).month)}


async def main(a):
    client = AsyncOpenAI(base_url=a.base_url, api_key="local")
    if a.identify:
        rng = np.random.default_rng(0)
        picks = [(t, d) for w in ("A", "B") for t in data.TICKERS for d in data.trading_days(t, w)]
        picks = [picks[i] for i in rng.choice(len(picks), a.identify, replace=False)]
        res = await asyncio.gather(*[identify(client, a.model, t, d) for t, d in picks])
        Path(a.out, f"identify_{a.model}.json").write_text(json.dumps(res, indent=1))
        for k in ("ticker_hit", "year_hit", "month_hit"):
            print(f"{a.model} anonymization {k}: {np.mean([r[k] for r in res]):.0%} of {len(res)} "
                  f"(chance: ticker 10%)")
        return
    rows = []
    for w in ("A", "B"):
        for t in data.TICKERS:
            days = data.trading_days(t, w)[:: a.every]
            anchor = data.history(t, days[0])["close"].iloc[-1]   # naive guess: last close before the window
            guesses = await asyncio.gather(*[ask(client, a.model, t, f"{d:%Y-%m-%d}") for d in days])
            for d, g in zip(days, guesses):
                true = data.history(t, d + np.timedelta64(1, "D"))["close"].iloc[-1]  # close of d, as quoted then
                rows.append({"window": w, "ticker": t, "date": f"{d:%Y-%m-%d}", "true": true, "guess": g,
                             "naive": anchor})
    Path(a.out, f"probe_{a.model}.json").write_text(json.dumps(rows))
    for w in ("A", "B"):
        r = [x for x in rows if x["window"] == w and x["guess"]]
        err = np.median([abs(x["guess"] / x["true"] - 1) for x in r])
        naive = np.median([abs(x["naive"] / x["true"] - 1) for x in r])
        print(f"{a.model} window {w}: median |error| {err:.1%} vs naive {naive:.1%} ({len(r)} answers)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--every", type=int, default=5, help="probe every Nth trading day")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--identify", type=int, default=0, help="run the de-anonymization test on N samples instead")
    p.add_argument("--out", default=".", help="directory for probe results")
    asyncio.run(main(p.parse_args()))
