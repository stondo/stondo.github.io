# Flash-Next on one RTX 5090: the local comparison

Measured on 2026-11-05. RTX5090 32GB, 96GB host RAM, Omarchy/Arch,
Python3.14.8, Strata CLI0.2.11 and engine/control0.1.57, 450W power limit.

Article: https://stondo.github.io/posts/qwen38-flash-next-strata-rtx5090-quant-comparison/

Download all files: [flash-next-tests.zip](flash-next-tests.zip).

## What is here

- `measurements.json`: curated measurements, exact fractions, failures and timings.
- `provenance.json`: engine, weight revisions, context and sampling settings.
- `benchmark.py` / `tasks.py`: the sixteen-case direct/thinking suite and speed probes.
- `recommended_settings.py`: six selected cases rerun with xhigh, preserved thinking,
  temperature1, top_p0.95, top_k20, min_p0, presence_penalty0,
  repetition_penalty1, seed729, output budget8192. These are not six independent
  new tasks or a broad agentic benchmark.
- `humaneval_subset.py`, `prepare_humaneval.py`, `humaneval-provenance.json`:
  reconstruct and test the same32 public HumanEval tasks; source bytes are SHA256 pinned.
- `check_context.py`: three-marker recall at approximately256.6K input tokens.

The public bundle contains our harness and curated measurements, without local
provider credentials, private configuration files or machine-specific launch profiles.
The underlying HumanEval dataset comes from https://github.com/openai/human-eval;
see that repository for its licensing and methodology.

## Reproduction

Use a Linux environment with `bubblewrap`, `/usr/bin/python`, and `nvidia-smi`.
Generated Python is executed inside bubblewrap with network/home access removed
and resource limits. The harness asserts a450W GPU limit because that was the
measured configuration. It uses the local unauthenticated endpoint at
`http://127.0.0.1:8080/v1`.

The article contains the launch settings. Run one model at a time and choose a
fresh result label for each model/configuration; existing files are resumed,
so reusing a label is not a fresh trial.

```bash
python prepare_humaneval.py
MODEL=$(curl -fsS http://127.0.0.1:8080/v1/models |
  python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
LABEL=iq4-v31-reproduction
python benchmark.py --label "$LABEL" --api http://127.0.0.1:8080/v1 --model "$MODEL" --phase all
python benchmark.py --label "$LABEL" --api http://127.0.0.1:8080/v1 --model "$MODEL" --phase quality --thinking
python humaneval_subset.py --label "$LABEL" --model "$MODEL"
python recommended_settings.py --label "$LABEL" --model "$MODEL"
python check_context.py --label "$LABEL-256k" --model "$MODEL" --target-tokens 250000
```

The dataset preparation also accepts `--data-file /path/to/HumanEval.jsonl.gz`
for an offline run, with the same SHA256 check.

Then stop the model, start the other candidate, set a new label and repeat.
The NVFP4-GPTQ deployment used `STRATA_PREFILL_NVFP4=w4a8`; the Unsloth v3.1
deployment used `STRATA_HC_Q8=1`. Both used int8KV, one slot, vision off, MTP4,
spec-min-p0.5, reserve3072MiB, route-tail-skip0, native262144 context and no tuning.
The pack-specific internal precision/storage choices remain part of the comparison.

Weight sources:
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/d96201194f103c906994f283f504209dc8d2a10c
- https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata/tree/640a1db97446ddd5e89d2b70ecd0297987192c93

## Limits

Sixteen local cases, repeated in different modes, plus32 of164 public HumanEval
problems. Public-test familiarity can affect the result. The six recommended-settings
reruns each use one seeded sample. Core quality uses temperature0 and output4096;
short-prompt speed uses temperature0.6 and up to512 output tokens, a warm-up excluded,
and a three-run median. Minimum RAM is sampled system-wide MemAvailable, not modelRSS.
The context test reports256553 API prompt tokens for both candidates and checks
three markers; it is not long-context coding validation. No sustained full-context
decode throughput, multi-day stability or selected-v3.1 512K quality claim is made.
