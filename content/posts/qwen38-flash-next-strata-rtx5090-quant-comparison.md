---
title: "Qwen3.8-Flash-Next on One RTX 5090: Why I Kept IQ4_XS v3.1 at 256K"
date: 2026-11-06T00:24:32+01:00
draft: false
description: "A practical comparison of Unsloth IQ4_XS v3.1 and NVFP4-GPTQ on one RTX 5090 with 96 GB RAM: matched Strata engine, coding and tool checks, near-full 256K recall, 176 versus 145 tokens/s, and the checkpoint revision that changed the decision."
summary: "Rebuilding my workstation from Fedora 44 to Omarchy turned into a quantization experiment. NVFP4 was my first pick; then a fresh Unsloth release changed the candidate. With both remeasured on Strata 0.1.57, IQ4_XS v3.1 tied the core checks, passed one focused coding rerun NVFP4 failed, decoded about 21% faster on short prompts, and left nearly 8 GiB more system RAM available. The account includes the exact settings, the limits of the tests, and why I kept native 256K instead of declaring victory at 512K."
tags:
  - llm-inference
  - qwen
  - quantization
  - nvfp4
  - rtx-5090
  - blackwell
  - strata
  - opencode
  - long-context
  - self-hosting
images:
  - "/qwen38-strata-5090-benchmarks.png"
categories:
  - engineering
  - infrastructure
keywords:
  - Qwen3.8-Flash-Next
  - Unsloth Dynamic v3.1
  - UD-IQ4_XS-v31
  - NVFP4 GPTQ
  - Strata
  - RTX 5090
  - 96 GB RAM
  - 256K context
  - coding agents
---

I wiped Fedora 44, installed Omarchy, and set about recovering the workstation
I had spent months teaching to be useful. Projects, SSH access, agent configs,
the memory in [cairnkeep](/posts/announcing-cairnkeep-durable-memory-coding-agents/).
The important things came back from the NAS. Then I reached the large,
replaceable things: the model weights.

That seemed like a good time to question the default. My RTX 5090 had been
running [a 27B with NVFP4 KV
cache](/posts/qwen38-27b-nvfp4kv-262k-single-rtx5090/). Could
Qwen3.8-Flash-Next become the local coding model on this box instead?

The requirements were concrete: **one 32 GB RTX 5090, 96 GB of system RAM,
native 256K context, and enough headroom to keep using the desktop**. I wanted
working code and tool calls in OpenCode. A fast answer to a tiny prompt was
useful evidence, but it would not settle the decision.

NVFP4 looked like a sensible candidate for a Blackwell GPU. A higher-precision
GGUF looked attractive on published quality charts. I had a coding agent run
the experiments rather than choosing from the filenames.

The first recommendation was NVFP4. The model I eventually kept was
**Unsloth UD-IQ4_XS-v31**. The reason starts with a date.

## The quant I tested was older than the guide I was reading

The initial comparison used Strata's existing Unsloth downloads. Then I read
the [current Unsloth guide](https://unsloth.ai/docs/models/qwen3.8-next),
which announced Dynamic v3.1 on November 4.

My installed files came from the September 30 revision. Same model family,
familiar quant names, a different candidate from the one in the new charts.
That is an easy way to finish a careful benchmark and answer yesterday's
question.

There was a second trap: a changed download did not necessarily mean changed
weights. The [Strata 0.2.11 release
notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.11) distinguish
the cases explicitly. The November Q4_K_XL upload contains the same weights
with different shard boundaries. IQ4_XS-v31 has different tensor formats and
four shards instead of three. It gets a separate model name.

So I updated Strata and downloaded v3.1 separately. The fresh comparison used
**CLI 0.2.11, control and engine 0.1.57**, for both candidates. NVFP4 got a
fresh run too. Carrying its old score across the engine update would have
added another variable to the comparison.

The pinned revisions were:

| Candidate | Repository revision |
|---|---|
| Unsloth IQ4_XS-v31 | `d96201194f103c906994f283f504209dc8d2a10c` |
| Maximilian228 NVFP4-GPTQ Strata pack | `640a1db97446ddd5e89d2b70ecd0297987192c93` |

## What I actually held constant

This is a comparison of two working deployments on my workstation. Each
pack still brings its own precision and storage choices. It is not a clean
laboratory isolation of weight format alone.

Strata makes this a GPU-and-host-RAM deployment. The weights extend well
beyond the card's 32 GB, so host residency and the prepared pack are part of
the memory budget. That is why I measured available system RAM alongside
the GPU configuration.

The shared settings were:

- RTX 5090 with a **450 W power limit**, the same cap for every run.
- Native **262,144-token context**, int8 KV, one slot, vision disabled.
- MTP drafting, four draft tokens, minimum acceptance probability 0.5.
- A 3,072 MiB VRAM reserve and route-tail skipping disabled.
- One model loaded at a time, weights on the local NVMe, tuning skipped.

The model-specific paths matter too: v3.1 used `STRATA_HC_Q8=1`; the NVFP4
deployment used `STRATA_PREFILL_NVFP4=w4a8`. Those settings belong with the
results. A format name does not describe every operation the engine performs.

The 450 W cap is an operating choice from this machine's history with GPU
instability. These measurements do not describe an unrestricted 600 W card.
They also do not certify months of agent-session stability.

## A small suite that could tell me when the answer was wrong

I wanted executable checks. The core suite had **16 cases**, covering Python
code, structured output, tool arguments, context records, and small repair
tasks with read/write/test tools. SSE chunk handling, exact decimal accounting,
and a bounded asynchronous mapper were among the coding tasks.

Each candidate ran those cases directly and again with thinking enabled.
That is the same 16 cases under two modes, not 32 independent problems. The
output budget was 4,096 tokens per request.

Then came a frozen **32-problem subset of
[HumanEval](https://github.com/openai/human-eval)**, selected with seed
`20261105`. All selected reference solutions had passed the local grader.
Generated code ran in bubblewrap with no network or home-directory access,
plus CPU, memory, file-size and wall-time limits.

Finally, six selected cases got another run with a larger, 8,192-token
budget and the reasoning/sampling settings described by Unsloth: temperature
1.0, `top_p=0.95`, `top_k=20`, `min_p=0`, no presence or repetition penalty,
`xhigh` reasoning, and preserved thinking. The harness retained assistant
reasoning on tool continuations; I also checked that the template rendered it.
Each of those six cases used one seeded sample.

Those focused reruns are useful diagnostics. They are a very long way from
Unsloth's broad, repeated agentic evaluation. Their guide uses native 256K
for agentic benchmarks too; it does not establish a 512K agent-quality result.

## The fresh results

| Measurement | IQ4_XS-v31 | NVFP4-GPTQ |
|---|---:|---:|
| Core direct checks | 16/16 | 16/16 |
| Same checks with thinking | 16/16 | 16/16 |
| Frozen HumanEval subset | 31/32 | 31/32 |
| Six focused recommended-settings reruns | **6/6** | 5/6 |
| Short-prompt decode, median of three runs | **175.7 tok/s** | 145.3 tok/s |
| Minimum sampled system `MemAvailable` | **14.6 GiB** | 6.7 GiB |
| Near-full 256K three-marker recall | Pass | Pass |
| Near-full-context time to first token | 40.84 s | 41.37 s |

![Measured short-prompt decode and system RAM headroom for the two deployments](/qwen38-strata-5090-benchmarks.png)

The decode measurement used short prompts and up to 512 output tokens at
temperature 0.6, with a warm-up excluded. **176 tok/s is not a measured
generation rate at a full 256K context.** The RAM figure is system-wide
available memory sampled during the tests, including the desktop and OS;
it is not isolated model RSS.

The quality difference needs similar restraint. Both candidates failed the
same HumanEval problem, number 127, on interval-intersection length. In the
focused reruns, NVFP4 generated an accounting implementation with an
uninitialized local variable. The test raised `UnboundLocalError`. The
response had finished within its token budget; this was an implementation
bug, rather than an answer cut off by the harness.

That one failure does not establish a general quality ranking. What it gave
me was a practical result: the new quant showed no deficit in the core
checks, passed the focused rerun, generated about **21% faster** in the
short-prompt test, and left **nearly 8 GiB more RAM available**.

That last column matters when this is also the machine on which I browse,
edit, and run the coding agent. I wanted a usable workstation around the model.

## Filling the context instead of admiring the allocation

A server accepting `--context 256k` proves that a configuration loaded.
I also wanted to know whether it could answer with the window nearly full.

Both candidates received a synthetic prompt with markers near the beginning,
middle and end. The API reported **256,553 prompt tokens** for each. Both
returned all three markers, with the first answer token arriving in roughly
41 seconds.

That is a recall smoke test. It establishes something useful about this
configuration, but three recovered strings cannot certify long-context
coding quality. I have not measured a sustained agent session at that depth,
and a short-prompt throughput number cannot fill that gap.

I kept **native 256K** as the daily setting. YaRN at 512K remains a separate
experiment for the selected v3.1 deployment. More addressable context would
need its own quality, latency and headroom checks before becoming my default.

## Where Q4_K_XL fits into the story

Q4_K_XL was part of the earlier experiment. On this 96 GB host, a
full-residency trial crossed the minimum-RAM guard and was stopped. A
budgeted configuration ran, using a 64 GiB expert-residency budget.

Those runs used engine 0.1.56. Their scores belong with those settings;
the fresh table above is the two-candidate comparison on 0.1.57. The
November Q4 resharding also does not turn the earlier weights into a new
quantization algorithm.

Published quality charts helped me choose what to try. My final choice had
to include the host-memory budget, engine path, context requirement and
desktop workload. A nominally higher-precision download can be a perfectly
reasonable choice on a different machine. Here, v3.1 was the deployment
I wanted to keep.

## The configuration I kept

These are the launch choices used for the selected candidate; the first
run downloads and prepares the model. This recipe is scoped to the versions
above:

```bash
strata launch \
  --family unsloth --model UD-IQ4_XS-v31 \
  --context 256k --kv int8 --parallel 1 --vision off \
  --drafting mtp --tune skip --access local \
  --data-dir "$HOME/data/strata" \
  --set vram-reserve-mib=3072 \
  --set spec=4 --set spec-min-p=0.5 \
  --set STRATA_ROUTE_TAIL_SKIP=0 \
  --set STRATA_HC_Q8=1 \
  --save-preset "Flash Next IQ4 v31 256K" \
  --yes --no-tui
```

If another model is already serving, stop it first with `strata launch --stop`.
For later starts, use the saved preset:

```bash
strata launch --preset "Flash Next IQ4 v31 256K" --no-tui
```

I also kept a small identity check close at hand:

```bash
curl -fsS http://127.0.0.1:8080/v1/models | python -m json.tool
```

OpenCode now has one local Strata entry for the selected model, pointing at
`http://127.0.0.1:8080/v1`, with tool calling and reasoning enabled. Its
configured context is 262,144 tokens. The endpoint check tells me what is
actually loaded when I come back to the machine.

## One model on NVMe, the experiments on the NAS

The remaining cleanup was pleasantly concrete: keep the selected model and
its required pack local, move the old IQ4, Q4_K_XL, NVFP4 and retired 27B
weights and their unused model-specific packs to the NAS. The move freed
about **353 GB** on the NVMe; the old paths now point to the NAS copies.
The shared MTP files stay local.

There was one final lesson about automation budgets. My agent initially
gave every model a full source SHA256 pass, a copy, and a complete read-back
hash over the NAS link. That is a lot of network time for public weights
whose recovery procedure is to download them again. After the first model,
I changed the remaining moves to rsync's transfer checks plus file-list and
size validation, retaining the source-unchanged check before local removal.
The encrypted backups of configurations and agent memory deserve a different
standard from replaceable downloads.

The conclusion I can support is deliberately local: **IQ4_XS v3.1 is the
Flash-Next deployment I kept for this RTX 5090, 96 GB of RAM and a native
256K coding context**. The measurements leave me a comfortable reason to
use it, and a versioned baseline for the next candidate.

The [measurement data and runnable test
bundle](/benchmarks/qwen38-strata-5090-20261105/README.md) include the task
selection, settings and limitations. If a newer engine changes the answer,
there is something concrete to rerun.
