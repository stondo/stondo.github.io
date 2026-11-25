#!/usr/bin/env bash
# The llama.cpp reference pass for Inkling 2 Mini UD-Q8_K_XL on the BENCH_API server:
# the HumanEval subset, the six focused reruns, and async_ordered_map with thinking off and on.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
L=$1; M=$2
export BENCH_API=${BENCH_API:-http://127.0.0.1:8080}
export BENCH_MAX_TOKENS=16384 BENCH_HUMANEVAL_MAX_TOKENS=8192
export BENCH_REQUEST_KWARGS='{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0}'
# llama.cpp ignores enable_thinking for this template: the steps that stand for "thinking off" ask for effort none.
OFF='{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"reasoning_effort":"none"}'
export BENCH_SETTINGS_JSON='{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"reasoning_effort":null,"preserve_thinking":true,"seed":729}'
step() { local name=$1; shift; echo "$(date +%T) START $name"; "$@" >"$L-$name.log" 2>&1; local rc=$?; echo "$(date +%T) END   $name rc=$rc"; }
step humaneval env BENCH_REQUEST_KWARGS="$OFF"    python3 humaneval_subset.py --label "$L" --model "$M"
step speed env BENCH_REQUEST_KWARGS="$OFF"        python3 benchmark.py --label "$L" --api "$BENCH_API/v1" --model "$M" --phase speed
step focused      python3 recommended_settings.py --label "$L" --model "$M"
step async-off env BENCH_REQUEST_KWARGS="$OFF" python3 run_cases.py --label "$L" --model "$M" async_ordered_map
step async-on python3 run_cases.py --label "$L" --model "$M" --thinking async_ordered_map
echo "$(date +%T) ALL DONE"
