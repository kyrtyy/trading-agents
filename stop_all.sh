#!/usr/bin/env bash
# Stop the queue, the running arm (it exits cleanly, nothing finished is lost) and the model server.
W=/data3/yashdewangan/tradingagents
[ -f $W/run_all.pid ] && kill $(cat $W/run_all.pid) 2>/dev/null
for p in $(pgrep -u "$USER" -f "python run.py"); do kill -TERM $p; done
sleep 5
[ -f $W/serve.pid ] && kill $(cat $W/serve.pid) 2>/dev/null
echo stopped; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
