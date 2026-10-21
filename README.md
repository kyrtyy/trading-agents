# TradingAgents, replicated and ablated

This project re-runs **TradingAgents** (Xiao et al., [arXiv 2412.20138](https://arxiv.org/abs/2412.20138)), a multi-agent LLM trading framework, and tests whether its results hold up under stricter conditions:

- **Honest backtest:** trading costs, no look-ahead, correctly annualized returns, and confidence intervals.
- **Ablation:** does each added stage of agents improve the result?
- **Leakage audit:** is the model trading on knowledge it memorized during training?
- **Physical cost:** how many joules each decision uses, how random the decisions are at different sampling temperatures, and how the debate settles toward agreement.

> **Status:** the code and the data are verified. The full experiment (about 21k decisions on 2× RTX 4090) is queued and has not run yet. Numbers under *Expected results* are hypotheses, not findings. Numbers under *Done so far* are measured.

## The agent graph

Every trading decision for (ticker, day D) uses only data available before day D. The full variant makes **12 LLM calls**:

```
4 analysts (technical / sentiment + insider / news + macro / fundamentals)   run concurrently
        → Bull vs Bear debate (each ends with STANCE −2..+2) → research manager
        → trader ("FINAL TRANSACTION PROPOSAL")
        → aggressive / conservative / neutral risk debate → fund manager → BUY | SELL | HOLD (JSON)
```

The ablation ladder uses four variants: `single` (1 call) → `analysts` → `debate` → `full`.

**Deviation from the paper:** tool outputs are fetched in advance and pasted into the prompts, instead of being called through ReAct. This makes runs reproducible and keeps every agent's input identical across arms.

## Setup

| | |
|---|---|
| Tickers | AAPL, GOOGL, AMZN, NVDA, MSFT, META, JPM, XOM, JNJ, KO (10, vs. the paper's 3; WMT dropped because of a 3:1 split inside window A) |
| Window A | 2024-01-02 → 2024-03-28 (61 trading days, the paper's period) |
| Window B | 2026-04-01 → 2026-09-30 (~126 trading days, after both models' training cutoffs) |
| Models | Qwen3-30B-A3B-Instruct-2507 (FP8, split 26/22 layers across 2 GPUs with vLLM pipeline parallelism); Llama-3.1-8B-Instruct as the control (stated cutoff Dec 2023) |
| Data | Daily Google News per ticker and for the global market, Finnhub insider trades, insider sentiment and filed financials, yfinance prices (split- and dividend-adjusted) |
| Execution | Signal for day D is filled at D's open; 10 bps cost per unit of turnover; BUY = long, SELL = short, HOLD = keep the position |
| Baselines | Buy & Hold, MACD, KDJ+RSI, Z-score mean reversion, SMA crossover (the paper's set) |
| Metrics | Cumulative return, correctly annualized return, Sharpe ratio in excess of the T-bill rate (^IRX), max drawdown. Sharpe CIs and paired "does X beat Y" tests use a stationary block bootstrap. |

### Run plan (`run_all.sh`, about 21k decisions)

| Phase | Arms | Decisions |
|---|---|---|
| Qwen: replication + ablation | full, single, analysts, debate (10 tickers × both windows) | 7,480 |
| Qwen: leakage + noise | full on anonymized inputs; full with seeds 1 and 2 | 5,610 |
| Qwen: thermodynamics | T ∈ {0, 0.3, 1.0, 1.3} × 3 seeds, plus 3 debate rounds (AAPL/GOOGL/AMZN, window A) | 2,379 |
| Llama control | full real, full anonymized, single | 5,610 |

Two probes run first:
- **Price recall:** asks the bare model for closing prices, compared with a naive "last close" guess.
- **De-anonymization:** 100 anonymized inputs; the model is asked to name the company, year and month.

## Done so far (measured)

**Infrastructure**
- Point-in-time data pipeline with a self-check (`python data.py selfcheck`). It asserts, for every ticker and window, that no input is dated on or after D and that no input is empty. **Passes.**
- Backtester self-check (`python backtest.py --selfcheck`). It covers costs, compounding, short positions, look-ahead in the baselines (prices from D onward are changed, and the signal for D must not move), and the bootstrap statistics. **Passes.**
- Runs survive being killed: every decision is appended and fsynced, finished work is skipped on restart, and progress is snapshotted hourly. Tested by killing a run mid-arm: 35 unique records, no duplicates.
- Energy comes from NVML hardware energy counters on our own GPUs only (the cluster is shared). Idle power is subtracted. New work pauses when a GPU passes 83 °C.

**Anonymization:**
- The company name and its aliases are replaced with "Company XYZ".
- Dates are shifted forward by about 7 years, and years are masked.
- Prices are rescaled so that the last close is 100, and money values are scaled to match.
- News sources are dropped.

**Results so far**

| Check | Result |
|---|---|
| AAPL Buy & Hold, window A | CR −8.26%, AR −29.95%, Sharpe −1.69, MDD 13.3% (the paper reports CR −5.23%; we don't reproduce it exactly) |
| **Paper's annualization** | Its "annualized return" is not annualized: a 26.62% return over 3 months is reported as 30.5% AR. The correct annualized figure is about 165%. |
| **Noise floor** | A *random* BUY/SELL/HOLD strategy scored Sharpe 3.58 (+23.2%) on AAPL in window A. With 3 tickers and 61 days, skill can't be told apart from luck. |
| Llama-3.1-8B price recall | Window A median error 12.4%, vs. 4.2% for the naive guess; window B 32.7% vs. 11.2%. No memorization, so it's a clean control. |
| Llama throughput (full variant) | About 428 decisions/h at concurrency 16; about 25k tokens and **about 3.1 kJ of GPU energy per decision**; idle 28.8 W |
| Pipeline smoke tests | All 4 variants, real and anonymized: 0 JSON parse errors |

**Data coverage** (`python data.py coverage`):
- All 11 news streams are downloaded: 290 days each, no errors.
- Every ticker has ≥ 4 filings and 6 months of insider sentiment in view.
- **Known confound:** window A has thinner news than window B. Average headlines per week:

  | | Window A | Window B |
  |---|---|---|
  | Big tech | 26–32 | about 40 |
  | KO | 6 | 33 |

  The prompt is capped at 20 headlines per week for both windows. Comparisons within a window (ablation, real vs. anonymized, Qwen vs. Llama) are unaffected. Comparisons of A against B need to account for the gap.

## Expected results (hypotheses, to be replaced)

| Question | Expectation | Why |
|---|---|---|
| Does the full system beat Buy & Hold? | Point estimate maybe; **95% CI on ΔSharpe includes 0** in each window | 61 days of returns give a Sharpe CI about ±4 wide; the random-strategy result above |
| Full vs. single agent (ablation) | ΔSharpe CI includes 0, at about **6–12× the tokens and energy** | Most of the information is already in the analyst reports; the debate mainly adds narrative |
| Leakage: real vs. anonymized, Qwen | Real beats anonymized in **window A** (2024, inside Qwen's training data); no gap in **window B** | If the gain comes from memory, hiding the company and date should remove it only where memory exists |
| Leakage: Llama | No real-vs-anonymized gap in either window | Its cutoff (Dec 2023) predates both windows; the price probe already shows no recall |
| De-anonymization | Company identified **well above the 10% chance rate** (sector and fundamentals leak through); year rarely | Market cap, product mix and insider names are hard to hide fully |
| Energy per decision | Qwen full: about 3–6 kJ; single: about 0.3–0.6 kJ; whole run about 20–30 kWh | Llama measured at 3.1 kJ; Qwen activates 3B parameters per token but runs across 2 GPUs |
| Decision entropy vs. temperature | Rises with T; 0 across seeds at T = 0 | Greedy decoding is deterministic; sampling spreads the 12-call chain |
| Debate "relaxation" | Bull and bear stances converge toward each other, and with 3 rounds they settle by round 2–3 | Models tend to accommodate their counterpart in multi-turn debate |

## Layout

| File | What it does |
|---|---|
| `data.py` | Point-in-time fetchers and views (prices, news, Finnhub), `selfcheck`, `coverage` |
| `agents.py` | Rendering (real or anonymized), the agent graph, `run_decision` |
| `run.py` | Runs one experiment arm: resumable, energy/temperature monitor, hourly progress |
| `backtest.py` | Signals → P&L, baselines, metrics, bootstrap CIs, `--compare` |
| `probe.py` | Price-recall probe and de-anonymization test (`--identify N`) |
| `serve.sh` | vLLM server (`qwen` on both GPUs, `llama` on GPU1) |
| `run_all.sh` / `stop_all.sh` | Full queue, resumable; clean stop |

## Running

```bash
# env.sh (not in the repo) sets FINNHUB_API_KEY, HF_HOME, TA_DATA, TA_RUNS
python data.py prices && python data.py news && python data.py finnhub
python data.py selfcheck && python backtest.py --selfcheck
setsid nohup bash run_all.sh > logs/run_all.log 2>&1 < /dev/null &   # rerun to resume
python backtest.py runs/qwen__full__real__t0.7__s0__r1.jsonl --out results.csv
python backtest.py --compare runs/qwen__full__real__t0.7__s0__r1.jsonl runs/qwen__single__real__t0.7__s0__r1.jsonl
```

The paths in `serve.sh`, `run_all.sh` and `stop_all.sh` point to the original cluster; edit `W=` for your own machine.

## Next

- Run the queue.
- Fill in the results tables.
- Thermodynamics analysis: entropy vs. T, stance trajectories, joules per unit of Sharpe.
- Dashboard.
