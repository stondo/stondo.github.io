#!/usr/bin/env bash
# One more sampled run of the core cases and the HumanEval subset with another seed, on the server at BENCH_API.
#   run_seed.sh SEED          label: inkling2-<seed>; Inkling's publisher settings: temperature 1.0
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export BENCH_API=${BENCH_API:-http://127.0.0.1:8080}
export BENCH_MAX_TOKENS=16384 BENCH_HUMANEVAL_MAX_TOKENS=8192
S=$1; L=inkling2-rec-s$S; M=${2:-inkling-2-mini}
export BENCH_REQUEST_KWARGS="{\"temperature\":1.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"seed\":$S}"
step() { local name=$1; shift; echo "$(date +%T) START $name"; "$@" >"$L-$name.log" 2>&1; local rc=$?; echo "$(date +%T) END   $name rc=$rc"; }
step thinking-off python3 benchmark.py --label "$L" --api "$BENCH_API/v1" --model "$M" --phase quality
step thinking-on  python3 benchmark.py --label "$L" --api "$BENCH_API/v1" --model "$M" --phase quality --thinking
step humaneval    python3 humaneval_subset.py --label "$L" --model "$M"
echo "$(date +%T) ALL DONE"
