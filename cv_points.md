# CV points: TradingAgents, replicated and ablated

Legend: plain numbers are already measured or fixed by the design. [brackets] are estimates, to be replaced after the run.

## Title line
**Multi-Agent LLM Trading: Replication, Ablation & Leakage Audit of TradingAgents (arXiv 2412.20138)**
Python, vLLM, PyTorch, NVML, pandas | self-hosted Qwen3-30B-A3B (FP8) + Llama-3.1-8B on 2× RTX 4090

## Main bullets (pick 4–5)
- Reimplemented the TradingAgents framework: a 12-call LLM graph per decision with 4 analysts, a bull/bear debate, a trader, and a 3-way risk debate. Ran [~21k] trading decisions ([~0.4B] tokens) over 10 large-cap US stocks and 187 trading days on self-hosted open-weight models, served with vLLM pipeline parallelism across 2 GPUs.
- Built a point-in-time data pipeline with no look-ahead: daily news, insider trades, filings and split-adjusted prices, all asserted to be visible only before day D. Built an event-driven backtester with open-price fills, 10 bps costs, a risk-free-adjusted Sharpe ratio, and stationary block-bootstrap confidence intervals.
- Found two evaluation flaws in the original paper:
  - Its "annualized return" is not annualized: a 26.6% 3-month return is reported as 30.5% AR, while the correct figure is ~165%.
  - Its 3-stock, 3-month test cannot separate skill from noise: a random strategy scored Sharpe 3.58 on AAPL in the same window.
- Ablation ladder (single agent → analysts → +debate → full system), using paired bootstrap tests: the multi-agent system changed the portfolio Sharpe ratio by [+0.X, 95% CI −a to +b] vs. a single agent at [~6×] the token cost. [Fill in: significant or not.]
- Designed a training-data leakage audit:
  - Ran the same model on real vs. anonymized inputs (company renamed, dates shifted 7 years, prices rescaled), plus a de-anonymization attack and a price-recall memorization probe.
  - Used Llama-3.1-8B (cutoff Dec 2023) as the control. It does not memorize prices: 12.4% error, worse than the 4.2% of a naive last-close guess.
  - Result: [Qwen recovered the ticker in X% of anonymized samples vs. 10% chance; real − anon Sharpe gap = Y].
- Measured agentic compute thermodynamically from GPU hardware energy counters:
  - [~3–5] kJ and ~25k tokens per decision ([~25] kWh total).
  - Decision entropy vs. sampling temperature (T = 0–1.3).
  - Debate stance "relaxation": [stances converge within N rounds].

## Short variants
**Quant:** Built a look-ahead-free backtester with bootstrap Sharpe CIs. Showed that the published LLM-agent trading gains [do / do not] survive costs, proper annualization and a 10-stock, 2-window out-of-sample test (a random strategy hits Sharpe 3.58 in the original setup).

**ML engineering:** Served a 30B MoE (FP8) across 2× 24 GB GPUs with vLLM pipeline parallelism (26/22 layer split) at [~X] decisions/h with concurrency 16. Runs resume after a crash, with per-record fsync and hourly checkpoints, and pause under thermal limits on a shared 7-user cluster.

**AI agents:** Ablated a 12-call multi-agent LLM debate system. Quantified what each stage adds in return per token and per joule, plus how much of the apparent skill comes from training-data leakage.

## To fill after the run
| Placeholder | Source |
|---|---|
| total decisions / tokens / kWh | `runs/*.progress.json` |
| Qwen decisions/h, kJ/decision | Qwen arm progress.json |
| full vs single dSR + CI | `backtest.py --compare` |
| full vs Buy&Hold, per window | `backtest.py` table, PORTFOLIO rows |
| de-anon hit rates | `probes/identify_{qwen,llama}.json` |
| Qwen price-recall error vs naive | `probes/probe_qwen.json` |
| real − anon gap | `--compare` full real vs full anon |
| entropy vs T, stance convergence | thermodynamics analysis (to write) |
