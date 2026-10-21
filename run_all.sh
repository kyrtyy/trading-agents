#!/usr/bin/env bash
# The whole experiment, most important arms first. Resumable: rerun to continue (finished arms skip in seconds).
# Start:  setsid nohup bash run_all.sh > $W/logs/run_all.log 2>&1 < /dev/null &
# Stop:   bash stop_all.sh
set -uo pipefail
W=/data3/yashdewangan/tradingagents
. $W/env.sh; . $W/venv/bin/activate; cd $W/app
LOG=$W/logs; mkdir -p $LOG $W/probes
echo $$ > $W/run_all.pid
SUB=AAPL,GOOGL,AMZN   # the paper's three tickers: subset for the temperature / long-debate studies

serve() {  # $1 = qwen|llama, $2 = nvidia-smi GPU indices it needs
  for g in ${2//,/ }; do  # never start on a card another user is on; wait for it instead
    while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do
      echo "$(date '+%F %T') GPU$g busy, waiting"; sleep 300; done
  done
  setsid bash serve.sh $1 > $LOG/serve_$1.log 2>&1 < /dev/null & echo $! > $W/serve.pid
  for i in $(seq 90); do curl -sf localhost:8011/health > /dev/null && { echo "$(date '+%F %T') $1 up"; return 0; }; sleep 10; done
  echo "server $1 failed to start, see $LOG/serve_$1.log"; exit 1
}
unserve() { [ -f $W/serve.pid ] && kill $(cat $W/serve.pid) 2>/dev/null; sleep 20; }
arm() {  # one arm; stop the queue if the server is gone or the arm aborts
  curl -sf localhost:8011/health > /dev/null || { echo "server down before: $*"; exit 1; }
  echo "$(date '+%F %T') ARM $*"
  python run.py "$@" || { echo "$(date '+%F %T') arm stopped (exit $?): $*"; exit 1; }
}

# ---- Phase 1: Qwen (both GPUs, GPU1 carries 26/48 layers)
serve qwen 0,1
[ -f $W/probes/probe_qwen.json ] || python probe.py --model qwen --out $W/probes
[ -f $W/probes/identify_qwen.json ] || python probe.py --model qwen --identify 100 --out $W/probes
Q="--model qwen --gpus 1,0"
arm $Q --variant full                     # headline replication
arm $Q --variant single                   # ablation ladder
arm $Q --variant analysts
arm $Q --variant debate
arm $Q --variant full --anon              # leakage: same model, identity hidden
arm $Q --variant full --seed 1            # sampling noise
arm $Q --variant full --seed 2
# ---- Phase 3 (on Qwen while it is loaded): thermodynamics studies, window A, paper's tickers
for T in 0.0 0.3 1.0 1.3; do for s in 0 1 2; do
  arm $Q --variant full --windows A --tickers $SUB --temperature $T --seed $s
done; done
arm $Q --variant full --windows A --tickers $SUB --rounds 3   # debate relaxation over more rounds
unserve

# ---- Phase 2: Llama control (GPU1 only), training cutoff Dec 2023
serve llama 1
[ -f $W/probes/identify_llama.json ] || python probe.py --model llama --identify 100 --out $W/probes
L="--model llama --gpus 1"
arm $L --variant full
arm $L --variant full --anon
arm $L --variant single
unserve
echo "$(date '+%F %T') ALL DONE"
