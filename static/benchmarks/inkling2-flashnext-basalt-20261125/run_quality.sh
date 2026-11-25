#!/usr/bin/env bash
# Quality suite for one model on the local Basalt server.
# Same five steps, in the same order, as run_comparison.py of 2026-11-05 (the Flash-Next quant comparison).
#   run_quality.sh LABEL MODEL [thinking effort for the focused reruns; empty = the model's default]
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export BENCH_API=${BENCH_API:-http://127.0.0.1:8080}
L=$1; M=$2; EFFORT=${3:-}
step() { local name=$1; shift; echo "$(date +%T) START $name"; "$@" >"$L-$name.log" 2>&1; local rc=$?; echo "$(date +%T) END   $name rc=$rc"; }
step direct-speed python3 benchmark.py --label "$L" --api "$BENCH_API/v1" --model "$M" --phase all
step thinking     python3 benchmark.py --label "$L" --api "$BENCH_API/v1" --model "$M" --phase quality --thinking
step humaneval    python3 humaneval_subset.py --label "$L" --model "$M"
export BENCH_SETTINGS_JSON="$(python3 - "$EFFORT" <<'PY'
import json, sys
s = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
     "reasoning_effort": None, "preserve_thinking": True, "seed": 729}
if sys.argv[1]: s["reasoning_effort"] = sys.argv[1]
print(json.dumps(s))
PY
)"
step recommended python3 recommended_settings.py --label "$L" --model "$M"
unset BENCH_SETTINGS_JSON
step context      python3 check_context.py --label "$L-256k" --model "$M" --target-tokens 250000
echo "$(date +%T) ALL DONE"
