# Inkling 2 Mini against Flash-Next on one RTX 5090: the local comparison

Measured on 2026-11-25. RTX 5090 32 GB, 96 GB host RAM, 450 W power limit.
Inkling 2 Mini ran on Basalt 0.5.7, with one reference pass on Unsloth's
llama.cpp build. The Flash-Next speed numbers were taken on Basalt 0.5.7 the
same day. The Flash-Next quality scores and its 256K numbers come from the
archived run of 2026-11-05 on Strata 0.1.57, with the same weights:
/benchmarks/qwen38-strata-5090-20261105/.

Article: https://stondo.github.io/posts/inkling-2-mini-vs-flash-next-basalt-rtx5090/

Download all files: [inkling2-flashnext-tests.zip](inkling2-flashnext-tests.zip).

## What is here

- `measurements.json`: curated measurements, exact fractions, failures and
  timings for both models.
- `provenance.json`: engines, packs with SHA-256 and repository revisions,
  server settings, sampling, output budgets and limits.
- `benchmark.py` / `tasks.py`: the sixteen-case direct/thinking suite and the
  speed probes. These are the files of the earlier bundle plus the environment
  overrides listed below.
- `run_quality.sh LABEL MODEL [EFFORT]`: the five steps of the earlier
  comparison in the same order, with a model's publisher sampling:
  speed and direct cases, the same cases with thinking, the HumanEval subset,
  six focused reruns, and the recall check. The focused reruns use temperature
  1.0, top_p 1.0, top_k 0, min_p 0, seed 729 and the effort given as the third
  argument (empty: the model's default).
- `run_seed.sh SEED [MODEL]`: another sampled run of the core cases in both
  modes and of the HumanEval subset, with the given seed.
- `python run_cases.py --label LABEL --model MODEL [--thinking] CASE...`: chosen
  core cases, same result files.
- `run_llamacpp_pass.sh LABEL MODEL`: the reference pass used on the llama.cpp
  server: the HumanEval subset, speed, the focused reruns, and
  `async_ordered_map` with thinking off and on.
- `humaneval_subset.py`, `prepare_humaneval.py`, `humaneval-provenance.json`:
  reconstruct and test the same 32 public HumanEval problems; source bytes are
  SHA256 pinned.
- `check_context.py`: three-marker recall at about 250,000 input tokens.
- `gsm8k-inkling-2-mini-first-200.json`: Inkling 2 Mini's per-problem results for the
  first 200 GSM8K problems, greedy with thinking on. The Flash-Next file of the
  same run no longer exists, so GSM8K is compared on totals only.
- `loopcd-2026-11-26.json`: the follow-up of 2026-11-26. LoopCD on and off for
  Inkling 2 Mini with capy-flash 0.16.5 on one RTX 5090, in greedy mode and at
  temperature 1.0, built by script from the result files of the same harness.
- `drafter-kv-2026-11-26.json`: the follow-up of 2026-11-26 on Basalt: Inkling 2
  Mini at temperature 0 with the drafter on and off, and with an fp16 and an
  int8 cache, built by script from the result files of the same harness.
- `flashnext-basalt-greedy-2026-11-26.json`: Flash-Next at temperature 0 on Basalt, two
  passes of the 16 core cases against the running service, with the run of
  2026-11-05 on Strata as the reference.
- `both-models-basalt-publisher-2026-11-26.json`: both models on Basalt at the
  settings their publishers recommend, three runs each, case by case.
- `flashnext-expert-cache-2026-11-26.json`: the share of Flash-Next's expert lookups
  that missed the card's cache during those runs, from the service's own log.

The public bundle contains the harness and curated measurements, without local
provider credentials, private configuration files, result files or
machine-specific launch profiles. The underlying HumanEval dataset comes from
https://github.com/openai/human-eval; see that repository for its licensing
and methodology.

## Environment overrides

All are optional. Unset, the scripts behave as the earlier bundle's.

- `BENCH_API`: base URL of the server. Default `http://127.0.0.1:8080`.
- `BENCH_MAX_TOKENS`: output budget of the core cases. Default 4096.
- `BENCH_HUMANEVAL_MAX_TOKENS`: output budget of the HumanEval subset.
  Default 4096.
- `BENCH_CONTEXT_MAX_TOKENS`: reply budget of the recall check. Default 256.
- `BENCH_REQUEST_KWARGS`: JSON with request fields for each request of a
  step, for example `{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0}`.
- `BENCH_SETTINGS_JSON`: JSON with the settings of the focused reruns.

The speed probes always use temperature 0.6. The recall check always uses
temperature 0.

## Reproduction

Use a Linux environment with `bubblewrap`, `/usr/bin/python` and `nvidia-smi`.
Generated Python is executed inside bubblewrap with network and home access
removed and with resource limits. Run one model at a time and choose a fresh
label for each run; existing result files are resumed, so a reused label is
not a fresh trial.

```bash
python prepare_humaneval.py
MODEL=$(curl -fsS http://127.0.0.1:8080/v1/models |
  python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')

# Inkling 2 Mini: the settings its publisher recommends, and a reply budget
# for a model that always reasons
export BENCH_API=http://127.0.0.1:8080
export BENCH_MAX_TOKENS=16384 BENCH_HUMANEVAL_MAX_TOKENS=8192 BENCH_CONTEXT_MAX_TOKENS=2048
export BENCH_REQUEST_KWARGS='{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0}'
./run_quality.sh inkling-run-1 "$MODEL"
./run_seed.sh 730 "$MODEL"
```

For Flash-Next, the instructions of the earlier bundle apply without these
overrides: temperature 0, an output budget of 4,096, and `top_p` 0.95 with
`top_k` 20 and `xhigh` on the focused reruns.

The dataset preparation also accepts `--data-file /path/to/HumanEval.jsonl.gz`
for an offline run, with the same SHA256 check.

## Limits

Sixteen local cases, repeated in different modes and, for Inkling 2 Mini, in
three sampled runs, plus 32 of 164 public HumanEval problems and the first 200
GSM8K problems. Public-test familiarity can affect the result. Inkling 2 Mini
answers at temperature 1.0 are samples; each case has three. Flash-Next was
run once, at temperature 0. Its quality numbers come from another engine with
the same weights. GSM8K is compared on totals only. Minimum RAM is sampled
system-wide MemAvailable, not model RSS. The context test checks three markers
near 250,000 tokens; it is not long-context coding validation. No sustained
full-context decode throughput or multi-day stability claim is made.
