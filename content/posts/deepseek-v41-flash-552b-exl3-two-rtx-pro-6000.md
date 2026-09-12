---
title: "552B Parameters and the 203 GB You Keep on Disk: DeepSeek-V4.1-Flash on Two RTX PRO 6000"
date: 2026-09-12T10:30:00+00:00
draft: false
description: "DeepSeek dropped V4.1-Flash on a Thursday: 552B backbone parameters plus a 196B Engram memory, activating 8-16B per token. Two days later it runs on my two RTX PRO 6000s at EXL3 2bpw — because a quant existed for exactly this hardware. This is the fit analysis, the deployment, and the honest gap between my 123 GB of host RAM and the 113 tok/s headline."
summary: "The quant landscape sweep that ended at a 2.0bpw EXL3 pack built for exactly two RTX PRO 6000s, why 196B of Engram tables live on NVMe in every configuration that exists, what eager-mode serving actually measures (94-104 tok/s decode, 2.7K tok/s prefill), and the AM5 ceiling that keeps the last 15% of the recipe's performance behind a RAM upgrade the platform cannot physically provide."
tags:
  - llm-inference
  - deepseek
  - deepseek-v4.1-flash
  - exl3
  - vllm
  - rtx-pro-6000
  - blackwell
  - sm120
  - self-hosting
categories:
  - engineering
  - infrastructure
keywords:
  - DeepSeek-V4.1-Flash
  - EXL3
  - RTX PRO 6000
  - SM120
  - Engram
  - vLLM
  - EXL3 2bpw
  - tensor parallelism
  - local LLM
---

The weights landed on a Thursday. DeepSeek-V4.1-Flash: 552B backbone parameters, a new `DeepseekV41ForCausalLM` encoder-decoder architecture, vision in the base checkpoint, and — the part that changes everything about running it locally — **196B parameters of "Engram" memory**, sparse n-gram lookup tables that the model reads selectively instead of computing. Eight billion parameters active per token in prefill, sixteen in decode.

My GPU box has two RTX PRO 6000 Blackwell cards, 96 GB each, and it was already earning its keep serving the [uncensored V4-Flash-Vision lane](/posts/deepseek-v4-flash-vision-uncensored-two-rtx-pro-6000/). The question wasn't whether V4.1 was better — the benchmarks said yes, loudly. The question was whether 192 GB of VRAM and 123 GB of host RAM could hold a model that is bigger than its predecessor *and* drags a 203 GB sidecar of memory tables behind it.

Short answer: yes, but only because somebody already built the exact quant this hardware needs. This is the story of finding it, deploying it, and measuring exactly where the performance ceiling sits — because on this machine, the ceiling is made of DDR5 slots.

## First, the landscape sweep

Rule one of running frontier-scale models on owned hardware: before you download 180 GB of anything, sweep the quant landscape. The candidates for V4.1-Flash, two days after release:

| Candidate | Why not |
|---|---|
| Official FP8 / dealignai uncensored FP8 | ~550 GB. Not a VRAM problem, a *physics* problem |
| NVFP4 builds (msuiche, LibertAIDAI, AtomicChat) | ~290 GB of experts at 4-bit. Still doesn't fit 192 GB |
| EXL3 3.5bpw (bot-lab-21 "Pollard") | ~232 GB. Two cards short |
| ds4 (antirez) Q2 GGUF | 341 GB file, 152 GB resident — fits! Except: **Metal only**. More below |
| **diffbot EXL3 2.0bpw for 2× RTX PRO 6000** | fits, with 8 GiB of KV to spare |

The diffbot pack is almost suspiciously specific: routed experts re-encoded at EXL3 2 bits per weight with an MCG codebook, attention and dense weights left at DeepSeek's FP8, the DSpark draft layers untouched, the vision aligner untouched, and the 203 GB of Engram tables shipped as-is in shards 47 and 48. It adapts sfxnz's 2× DGX Spark recipe to x86 Blackwell, and its benchmark table was measured on two RTX PRO 6000 Max-Q cards at a 300 W cap — my silicon, my power envelope.

Two days after the weights dropped. Somebody moved fast. I benefit.

## The antirez detour

I wanted ds4 to be the answer. [DwarfStar](https://github.com/antirez/ds4) is antirez's single-C-file inference engine for exactly this family of models, and its whole design philosophy matches my constraint: **Engram tables stay on disk in every mode**. Rows are read from the GGUF as needed, never resident. The Q2 build needs only 152 GiB of live weights — my two cards could hold that with room for a real context.

But the commit that matters is dated the morning I checked: *"DeepSeek v4.1 Flash support for Metal."* Metal. The docs are refreshingly blunt: "DSpark, pipeline execution and non-Metal backends are not implemented for V4.1." No CUDA PR in flight. V4.1 is an encoder-decoder with a new architecture string; the port is not trivial, and antirez turned around the Metal path in roughly 48 hours, which gives hope for a CUDA follow-up — but hope is not a deployment plan.

So: vLLM with the diffbot pack. Filed under "watch this repo."

## What 123 GB of RAM actually means

The recipe's headline configuration — 113 tok/s single-stream decode, 4,000+ tok/s prefill, GSM8K 98.5% — has a footnote written in memory dims: **Engram tables pinned in host RAM, ~190 GiB of them, with serving RSS around 280 GB.** Pinned Engram plus CUDA graphs is the fast path. Engram on NVMe is the slow path, and it *requires* eager mode, because per-step disk reads can't be captured into a CUDA graph.

My host has 123 GB of RAM. The pinned path needs more than double that, before you count anything else the box does.

The serve script I adapted says it out loud, because I want future me to read the comment before getting clever:

```bash
# the ~190 GiB Engram tables CANNOT be pinned, so ENGRAM_DISK=1 (NVMe reads
# via async threads) is the default, which REQUIRES EAGER=1 (no CUDA graphs).
# Expect bench-row-1 class decode, not the 113 t/s headline.
```

The upgrade math is where it gets cruel. The board is an AM5 ProArt X870E with two 64 GB DIMMs installed. Two slots free. AM5 tops out at **256 GB** — still below the ~280 GB the pinned configuration consumes. The platform that can hold my GPUs cannot hold the RAM the fastest configuration wants. I could buy the missing performance for the price of two DIMMs, except I can't.

## What eager-plus-NVMe actually delivers

Cold start to first token is about two minutes (Engram load off NVMe, FlashInfer JIT). After that, the numbers, measured on the live lane, not copied from anyone's README:

| Measurement | Result |
|---|---|
| Decode, short context | ~94 tok/s single-stream |
| Decode, 54K-token context | ~104 tok/s (after prefill; the tables are warm in page cache) |
| Prefill, 54K tokens | ~2,700 tok/s (20.2 s wall) |
| Prefill, 152K tokens | ~2,400 tok/s (62.8 s wall) |
| Context ceiling | 524,288 tokens, fp8 KV, 8 GiB pool ≈ 3.8M tokens of headroom |

The recipe's full-stack ladder tops out at 4,044 tok/s prefill and 113-117 tok/s decode. So this configuration sits at roughly **85-90% of the headline**, and the missing slice is almost entirely prefill — decode past warmup is nearly there, because the xmoe MoE kernel rewrite, the Marlin dense GEMMs, and DSpark speculative decoding all work in eager mode. The kernel authors did their part; my DDR5 slots did not do theirs.

For an agent-coding workload this is a good trade. Prefix caching absorbs the re-prefill of system prompts (the recipe's own agentic benchmark: 87% of prompt tokens served from cache), and the turns that matter are decode-bound.

## The daily driver shape

The lane runs as a plain systemd user unit over rootless podman, on port 8004, sharing the box with a GLM-5.3-Flash EXL3 lane via `Conflicts=` (they both want all 192 GB; they take turns). Rootless cost two recipe adjustments worth remembering: `--shm-size` is rejected next to `--ipc=host` (host `/dev/shm` is 62 GB, so nobody misses it), and the pinned-RAM mode's `CAP_IPC_LOCK` gymnastics are moot when you can't pin anyway.

One deliberate text-only flag: `--language-model-only`. The vision tower ships in the pack, but the serving recipe is text-first, day-zero code — and this lane's job is being the biggest brain in the fleet, not reading screenshots.

The follow-up post is about the week's actual mystery: the lane that benchmarked clean and then, in real sessions, occasionally started answering in fluent nonsense — [the word salad only opencode could see](/posts/dsv41-flash-word-salad-top-p-opencode/).

## What I'd tell past me

1. **Sweep the landscape before downloading anything.** Two hours of Hugging Face archaeology found a pack built for my exact GPU pair. That never happens; check anyway.
2. **The quant that fits is the best quant.** A 3.5bpw build I can't load loses to a 2.0bpw build that serves at 104 tok/s.
3. **Engram changes the memory math.** 552B of weights is a VRAM problem; 196B of lookup tables is a *storage bandwidth* problem. Budget them separately.
4. **Read the recipe's footnotes about host RAM before envying its headline.** Then check your motherboard's DIMM ceiling. Sometimes the bottleneck is the socket.
5. **Watch ds4.** An engine whose design keeps the 203 GB sidecar on disk by default is the right shape for this hardware — the day CUDA support lands for V4.1, it's worth a fresh comparison against eager vLLM.

DeepSeek-V4.1-Flash now serves from the GPU box alongside its predecessors, one `systemctl` flip away from being the default brain — while a head-to-head against the GLM lane decides who earns the boot slot. The comparison is ongoing. The tables, all 203 GB of them, are on disk either way.
