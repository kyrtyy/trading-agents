"""Run one experiment arm over (window, ticker, day) tasks. Kill any time; rerun the same command to resume.

Every finished decision is appended (and fsynced) to runs/<arm>.jsonl; on start, done keys are skipped.
A sampler thread logs GPU power/temperature to runs/<arm>.energy.csv, pauses new work while any GPU
is hotter than --max-temp, and writes runs/<arm>.progress.json every hour.
"""
import argparse
import warnings
import asyncio
import json
import os
import signal
import threading
import time
from pathlib import Path

from openai import AsyncOpenAI

warnings.filterwarnings("ignore", message=".*pynvml.*")

import agents
import data

RUNS = Path(os.environ.get("TA_RUNS", Path(__file__).parent / "runs"))


class GPUMonitor(threading.Thread):
    def __init__(self, csv_path, max_temp, gpus, period=1.0):
        super().__init__(daemon=True)
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        # only the GPUs serving our model (NVML numbering = nvidia-smi): the box is shared, and another
        # user's job must neither count toward our energy nor trip our thermal pause
        self.gpus = gpus
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpus]
        self.csv, self.max_temp, self.period = open(csv_path, "a"), max_temp, period
        if self.csv.tell() == 0:
            self.csv.write("ts,gpu,power_w,temp_c,util,mem_mib,energy_mj\n")
        self.hot, self.temps = False, []
        self.e0 = self.energy_mj()

    def energy_mj(self):
        """Hardware energy counters (mJ since driver load), summed over GPUs: exact, no sampling error."""
        return sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles)

    @property
    def joules(self):
        return (self.energy_mj() - self.e0) / 1000

    def idle_watts(self, seconds=60):
        e, t = self.energy_mj(), time.time()
        time.sleep(seconds)
        return (self.energy_mj() - e) / 1000 / (time.time() - t)

    def run(self):
        nv = self.nv
        while True:
            temps = []
            for i, h in zip(self.gpus, self.handles):
                p = nv.nvmlDeviceGetPowerUsage(h) / 1000
                t = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                u = nv.nvmlDeviceGetUtilizationRates(h).gpu
                m = nv.nvmlDeviceGetMemoryInfo(h).used // 2**20
                e = nv.nvmlDeviceGetTotalEnergyConsumption(h)
                self.csv.write(f"{time.time():.1f},{i},{p:.1f},{t},{u},{m},{e}\n")
                temps.append(t)
            self.csv.flush()
            self.temps = temps
            # hysteresis: pause above max_temp, resume 5C below
            self.hot = max(temps) > self.max_temp or (self.hot and max(temps) > self.max_temp - 5)
            time.sleep(self.period)


def arm_name(a):
    return f"{a.model}__{a.variant}__{'anon' if a.anon else 'real'}__t{a.temperature}__s{a.seed}__r{a.rounds}"


def all_tasks(a):
    return [(w, t, f"{d:%Y-%m-%d}") for w in a.windows.split(",") for t in a.tickers.split(",")
            for d in data.trading_days(t, w)]


async def main(a):
    RUNS.mkdir(parents=True, exist_ok=True)
    arm = arm_name(a)
    out_path = RUNS / f"{arm}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["window"], r["ticker"], r["date"]))
            except json.JSONDecodeError:
                pass  # half-written last line from a kill: that task simply reruns
    todo = [x for x in all_tasks(a) if x not in done][: a.limit or None]
    print(f"{arm}: {len(done)} done, {len(todo)} to run", flush=True)

    mon = GPUMonitor(RUNS / f"{arm}.energy.csv", a.max_temp, [int(g) for g in a.gpus.split(",")]) if not a.no_gpu else None
    idle_w = None
    if mon:
        if todo and a.idle_seconds:
            idle_w = round(mon.idle_watts(a.idle_seconds), 1)  # model loaded, no requests: the idle floor
            print(f"idle power {idle_w} W", flush=True)
        mon.e0 = mon.energy_mj()
        mon.start()
    client = AsyncOpenAI(base_url=a.base_url, api_key="local", timeout=900)
    out = open(out_path, "a")
    sem = asyncio.Semaphore(a.concurrency)
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    stats = {"arm": arm, "started": time.time(), "done_at_start": len(done), "finished": 0, "errors": 0,
             "error_streak": 0, "tokens": 0, "idle_w": idle_w, "actions": {x: 0 for x in agents.ACTIONS},
             "parse_errors": 0}

    def snapshot():
        el = time.time() - stats["started"]
        rate = stats["finished"] / el if el else 0
        left = len(todo) - stats["finished"] - stats["errors"]
        snap = {**stats, "elapsed_h": round(el / 3600, 2), "decisions_per_h": round(rate * 3600, 1),
                "remaining": left, "eta_h": round(left / rate / 3600, 1) if rate else None,
                "gpu_kj": round(mon.joules / 1000, 1) if mon else None,
                "kj_per_decision": round(mon.joules / 1000 / stats["finished"], 2) if mon and stats["finished"] else None,
                "gpu_temps": mon.temps if mon else None, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        (RUNS / f"{arm}.progress.json").write_text(json.dumps(snap, indent=1))
        return snap

    async def hourly():
        while not stop.is_set():
            await asyncio.sleep(3600)
            print("snapshot", snapshot(), flush=True)

    async def one(task):
        w, t, d = task
        async with sem:
            while mon and mon.hot and not stop.is_set():
                await asyncio.sleep(10)
            if stop.is_set():
                return
            t0 = time.time()
            try:
                S, calls = await agents.run_decision(client, a.model, t, d, a.variant, a.anon, a.temperature,
                                                     a.seed, a.rounds)
            except Exception as e:
                stats["errors"] += 1
                stats["error_streak"] += 1
                print("ERROR", task, repr(e)[:300], flush=True)
                if stats["error_streak"] >= a.max_error_streak:  # server down: don't burn through the arm
                    print(f"ABORT: {stats['error_streak']} consecutive errors", flush=True)
                    stop.set()
                return
            stats["error_streak"] = 0
            rec = {"window": w, "ticker": t, "date": d, "arm": arm, "model": a.model, "variant": a.variant,
                   "anon": a.anon, "temperature": a.temperature, "seed": a.seed, "rounds": a.rounds,
                   "action": S["decision"]["action"], "confidence": S["decision"].get("confidence"),
                   "parse_error": S["decision"].get("parse_error", False), "t_start": t0, "t_end": time.time(),
                   "n_calls": len(calls), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                   "completion_tokens": sum(c["completion_tokens"] for c in calls), "state": S, "calls": calls}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            os.fsync(out.fileno())
            stats["finished"] += 1
            stats["tokens"] += rec["prompt_tokens"] + rec["completion_tokens"]
            stats["actions"][rec["action"]] += 1
            stats["parse_errors"] += rec["parse_error"]
            if stats["finished"] % 25 == 0:
                print(f"{stats['finished']}/{len(todo)} done", flush=True)

    hb = asyncio.create_task(hourly())
    await asyncio.gather(*[one(x) for x in todo])
    hb.cancel()
    print("final", snapshot(), flush=True)
    complete = stats["finished"] == len(todo)
    return 0 if complete else (3 if stats["error_streak"] >= a.max_error_streak else 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen")
    p.add_argument("--variant", default="full", choices=agents.VARIANTS)
    p.add_argument("--anon", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--windows", default="A,B")
    p.add_argument("--tickers", default=",".join(data.TICKERS))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only run the first N pending tasks (smoke tests)")
    p.add_argument("--max-temp", type=float, default=83)
    p.add_argument("--gpus", default="1,0", help="nvidia-smi indices serving the model (llama: 1)")
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--no-gpu", action="store_true", help="skip NVML (e.g. when not on the GPU box)")
    p.add_argument("--idle-seconds", type=int, default=60, help="measure idle GPU power before starting")
    p.add_argument("--max-error-streak", type=int, default=20)
    raise SystemExit(asyncio.run(main(p.parse_args())))
