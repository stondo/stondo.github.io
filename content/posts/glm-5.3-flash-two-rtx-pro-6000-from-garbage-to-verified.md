---
title: "From locklocklock to 262K Context: GLM-5.3-Flash on Two RTX PRO 6000"
date: 2026-08-30T09:00:00+00:00
draft: false
description: "GLM-5.3-Flash booted on my two RTX PRO 6000 Blackwell and answered every prompt with deterministic garbage. The investigation ran from a fake kernel bug I wrote myself, through a full exoneration of the linear-attention stack, to a live per-layer bisect that cornered the real culprit in the sparse-MLA prefill path. The fix came from a completely different quantization and kernel stack, and the model now passes 261,900-token needle retrieval on my desk."
summary: "A 320B sparse-hybrid MoE serving word salad at temperature 0, a repro whose bugs outlived the bug it hunted, a per-layer bisect that cleared the KDA layers and convicted the sparse full-attention path, and an EXL3/TR3 plus B12X deployment that ends the story with verified 262K-context retrieval at 39 tok/s."
tags:
  - llm-inference
  - vllm
  - glm-5.3-flash
  - rtx-pro-6000
  - blackwell
  - self-hosting
  - debugging
categories:
  - engineering
  - infrastructure
keywords:
  - GLM-5.3-Flash
  - RTX PRO 6000
  - SM120
  - sparse MLA
  - KDA
  - EXL3
  - B12X
  - vLLM
  - phrase looping
  - garbage output
---

The prompt was `The capital of France is`. The answer, at temperature 0, in perfect determinism, was:

```
 the capital France capital France capital France capital France...
```

That is what a broken model sounds like when nothing is random. Every request, same flavor of word salad. Thinking prompts burned their entire token budget on noise. Weights verified byte-perfect, engine boots clean, and output that a toddler would reject. This is the story of getting GLM-5.3-Flash, a 320B sparse-hybrid MoE, from that state to serving verified 262K-context retrieval on two RTX PRO 6000 Blackwell cards under my desk.

## The cast

GLM-5.3-Flash mixes two attention families in one 45-layer stack: 34 layers of KDA (Kimi Delta Attention, the gated linear-attention recurrence) and 11 layers of sparse full attention (MLA-style with a top-k token indexer). The model I wanted to serve, the hardware I have, and two software stacks that could not be more different in how much pain they delivered.

Attempt one: the NVFP4 checkpoint (181 GB, routed experts in 4-bit, attention in BF16) on a vLLM build carrying the GLM support PR. The port itself went fine, and the engine served requests. Garbage requests.

## Act one: the kernel bug that wasn't

The KDA chunk kernel looked guilty. Its returned final state disagreed wildly between a full-length run and a split run with a carried state, outputs stayed bit-exact while states diverged, and smaller chunk sizes produced NaN. I wrote a minimal repro and filed a detailed issue with a ruling-out table.

The repro was wrong twice, and both bugs are worth writing down because they will eat somebody else's week.

First, the gate kernel reads `g_bias` as a flat H-by-D tensor: 1024 floats at these shapes. My repro passed eight. Nothing checks the shape; the kernel just reads past the end of an eight-element tensor, and whatever lives in the next kilobyte of GPU memory becomes your gate bias. Fresh pages read as zeros and the output looks almost right. Dirty pages produce gates with norms in the eleven digits. Every "memory corruption" observation in my first investigation, the exploding tensors, the values that changed when I reordered unrelated calls, all of it was one out-of-bounds read.

Second, the chunk entry point destroys its `v` input in place. Somewhere in the pipeline the layer's output is written directly over the caller's input tensor, an undocumented memory optimization. My repro reused `v` across calls, so every call after the first was reading the previous call's *output* as its input. States drifting to zero on identical inputs, results that depended on call order, the works.

Both are real API hazards (I reported them upstream: shape assertions and documentation are cheap), but neither is a numerical bug. With correct tensors and fresh buffers, the whole KDA library checked out beautifully: chunk and recurrent paths agree to 0.3% at every length I tested, everything is bit-deterministic, and a pure torch reference of the chunk-state kernel matches it to six significant digits. A single KDA layer loaded with real checkpoint weights runs prefill and decode at 0.5% consistency. The linear-attention stack was innocent.

I also killed my own favorite theory properly: I compiled the exact failing kernel for sm_120a and sm_121a and got byte-identical PTX and identical SASS. Not an architecture codegen problem.

## Act two: the per-layer bisect

If the library was innocent, the engine was not. Time to stop probing kernels and interrogate the running thing.

The recipe: boot the real engine with per-layer activation hooks recording input and output norms for every layer and sublayer, swap the MoE backend to the pure-torch dequantization path to rule out the expert kernels, send the five-token capital-of-France prompt, and read the table.

The table was unambiguous. Every KDA sublayer: healthy, output norms 0.4 to 9.9, even while digesting an increasingly poisoned residual stream. MoE and shared experts: healthy. And every sparse full-attention layer exploded: layer 3 outputs at 42 times its input norm, layers 7 through 43 at 44 to 96x. The residual stream was being set on fire at the first MLA layer and never recovered.

Every component of that path I could test in isolation passed: the decode attention wrapper matches a torch reference to 0.2%, the top-k selection ops are exact, the block-table index conversion is exact. Two structural facts completed the pin-down: no dense-MLA prefill backend supports this model's NoPE dimensions (256/0/256) on *any* architecture, so prefill always runs through the top-k MQA path, and the masked-MHA fallback path is hard-rejected on SM120 silicon. The defect lives in the prefill wiring of that path. Layer-level isolation, not a line-level fix; I ran out of bootable patience (TP2 warmup on this box is its own lottery) before the final buffer dump.

## Act three: the recipe that works

While digging I found something better than another lead: [a validated deployment](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell) targeting my exact hardware, down to the mixed Max-Q/workstation GPU pair. Different stack, same silicon:

| | Broken stack | Working stack |
|---|---|---|
| Quant | NVFP4, 181 GB | EXL3/TR3 4-bpw, 176 GB |
| Runtime | vLLM PR nightly | pinned vLLM build with B12X kernels |
| Sparse attention | stock sparse-MLA path | `B12X_MLA_SPARSE` backend |
| Parallelism | TP2 | TP2 + DCP2 over PCIe |
| Validation | word salad | 128K and 261.9K needle retrieval passes |

The B12X kernel suite replaces precisely the component my bisect convicted: the sparse full-attention path including its own indexers. The checkpoint is [brandonmusic's EXL3/TR3 build](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (source-available ShapleyMCG license, review before use), the container is T.J. Purtell's pinned image, and samuelcardillo's repo wraps it in scripts with preflight checks and a vision-template repair.

It boots in about five minutes, and then it is boring in the best way:

| Check | Result |
|---|---|
| Text / tools / vision verifier | all pass (vision reads a number out of a generated PNG) |
| `The capital of France is` | ` Paris. It is known as the "City of Light"...` |
| Single-stream decode | ~39 tok/s, TTFT 40-90 ms |
| Context ceiling | 262,144 tokens, with KV headroom for 2.8x concurrency |

One podman quirk for reproducers: `--shm-size` is rejected alongside `--ipc=host` (the host's 62 GB `/dev/shm` makes the limit unnecessary anyway).

## What I'd tell past me

1. **A deterministic wrong answer is a gift.** Nothing is flaky, racey, or thermal. Something computes the wrong function on every call, which means you can bisect it.
2. **Suspect your repro before the kernel.** A mis-shaped tensor and a reused input produced three days of phantom memory corruption. Assert your shapes; the library didn't.
3. **In-place APIs must be documented or they will be rediscovered, expensively, by everyone.**
4. **When the parts test clean, instrument the whole.** Per-layer norm hooks in one instrumented boot did more than a week of standalone kernel probes.
5. **Somebody with your exact problem has already solved it with a different stack.** Finding their repo is a debugging step.

The full evidence trail, including the exonerations and the footguns, is on the vLLM PR thread and in the issue where this started. My thanks to the recipe authors, and to the person who ran my first broken repro on their GB10 fleet before anyone knew it was broken; their coherence data was the clue that the difference was in my stack, not my silicon.

GLM-5.3-Flash now serves from a systemd unit on the GPU box, behind the same OpenAI-compatible endpoint as everything else I run. Total timeline from first boot to verified serving: four days, of which three were my own bugs and one was reading someone else's README properly.

## Epilogue, same day: the crash that looks like a clean stop

The recipe above survived four days of debugging and a 261.9K needle test. It did not survive its first afternoon as the production brain.

Under a multi-agent workload — dozens of concurrent long-prompt prefills sharing one system prefix, adaptive MTP spec decode active — the EngineCore died:

```
ValueError: ReplaySSM prefill source/state row count mismatch
  vllm/third_party/flash_linear_attention/ops/kda_replayssm_spec_decode.py:482
  (materialize_kda_replayssm_state, called from the KDA layer forward, kda.py:651)
```

The irony is not lost on me: the crash lives in the KDA stack I spent three days exonerating — but in the ReplaySSM state materialization for speculative decode, not in the attention numerics. Solo calls and light batches are fine; this needed sustained batching to trigger, which is why four days of interactive testing never produced it and one afternoon of agent fan-out did. There is a softer failure mode on the same path, too: under batching, requests stochastically run away into tens of thousands of thinking tokens for a kilobyte answer, while the identical request solo answers in seconds.

The worse bug was operational. When the EngineCore raises, the vLLM API server shuts down *cleanly* — exit status 0. My quadlet had `Restart=on-failure`, so systemd read a dead engine as a deliberate stop and left the default model offline, mid-workload, without a word. The fix is one word: `Restart=always`. An explicit `systemctl stop` never triggers a restart, so deliberate downtime still works; only self-exits — crash or "clean" — bring the service back. I applied it to every serving unit on the box, not just this one. The v4flash unit already had it; that lesson had been learned once and not propagated.

If you run this stack under real concurrency: check your `Restart=` line before you trust the dashboard, and watch for the ReplaySSM signature under batched prefill. Reported upstream: [glm-5.3-flash-ext3-4-bit-2x-rtx#1](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx/issues/1).
