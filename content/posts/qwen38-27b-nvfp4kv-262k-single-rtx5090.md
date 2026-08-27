---
title: "262,144 Tokens on One RTX 5090: Qwen3.8-27B with an NVFP4 KV Cache That Actually Works"
date: 2026-08-27T13:00:00+00:00
draft: false
description: "How I got the full native 256K context of Qwen3.8-27B serving on a single 32 GB RTX 5090 — NVFP4 everything checkpoint, a patched vLLM nightly that unlocks NVFP4 KV cache on consumer Blackwell, 380K tokens of resident KV pool, and every landmine I stepped on: silent download stalls, a tokenizer that cost me an hour, an MTP setting that benched 18% faster and then exploded in production."
summary: "My 27B coding brain was hard-capped at ~107K context by fp8 KV and fat weights. A stranger's gist promised a 451K-token KV pool on the same GPU via NVFP4 KV cache and a patched vLLM. I ported it, pinned it, validated it with planted-password recall tests at 235K tokens — and then watched it crash spectacularly the first time a real agent used it. The final stack: full 262,144-token context, ~125 tok/s decode, vision, MTP speculative decoding, all on one card."
tags:
  - llm-inference
  - vllm
  - quantization
  - nvfp4
  - rtx-5090
  - blackwell
  - qwen
  - long-context
  - speculative-decoding
  - self-hosting
categories:
  - engineering
  - infrastructure
keywords:
  - NVFP4 KV cache
  - vLLM
  - RTX 5090
  - Qwen3.8-27B
  - 256K context
  - SM120
  - FlashInfer
  - MTP
  - speculative decoding
  - long context serving
---

For about a year my rule of thumb has been: a 27B-class dense model on a single 32 GB card means a compromise menu. You can have weights that fit, or context that fits, or decode speed — pick two. My daily coding brain, an unsloth NVFP4 Qwen3.8-27B, sat at ~107K context because fp8 KV cache plus 21 GiB of weights is simply all a 5090 holds. Anything deeper went to a slower single-stream lane.

Then a friend-of-the-internet's gist crossed my desk claiming a **451,541-token KV pool with vision and speculative decoding enabled, on the same GPU I own**. The trick isn't smarter weights — it's quantizing the **KV cache itself to NVFP4**, which nobody does on consumer Blackwell because vLLM's gate refuses it: the NVFP4 KV path is hardwired to SM100 datacenter parts. The gist ships a patch that reroutes it through FlashInfer's FA2 paged reader on SM120, plus a tiny standalone CUDA kernel that writes V-scales in the layout that reader expects.

Skepticism first, benchmarks later. This post is the full account: what held up, what didn't, and the one setting that benched 18% faster and then detonated the first time a real agent touched it.

## Why NVFP4 KV changes the math

Qwen3.8-27B has a hybrid architecture: 16 full-attention layers out of 64 (the rest are linear attention with fixed-size state). Full-attention KV at 16 layers × 4 KV heads × 256 head dim is already lean — about 64 KiB/token in bf16. The problem is that a 262K-token window still wants ~16 GiB of KV, and after 21 GiB of unsloth weights there's no such thing left on a 32 GB card.

Two multipliers stack in this stack:

1. **A smaller checkpoint.** `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` is ModelOpt NVFP4 *everywhere* — attention projections, even the 248K-vocabulary lm_head, which matters more than people think at decode time (a full-vocab GEMM runs every token; in BF16 that's 2.5 GB read per token, in FP4 it's 0.7 GB). Weights: **17.9 GiB** vs my old 21.3.
2. **NVFP4 KV cache** ≈ 3.5× the token capacity of fp8 in the same bytes.

Multiply those and the same silicon holds ~380K tokens of resident KV instead of ~190K. Enough for the full native **262,144-token window** with room for a second concurrent request.

## Pinning the unpinnable

The patch targets a vLLM nightly. Nightlies are a moving target, so step one was refusing to build against "latest". Docker Hub turned out to keep commit-pinned nightly tags, and the gist's boot logs leaked the exact build it was validated on: `0.26.1rc1.dev1102+ge9d1398d9`. That string is a git describe — and sure enough, `vllm/vllm-openai:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013` exists, dated the same day. Everything downstream builds from that digest. When a patch applies to a pinned base, "it stopped working" stops being a category of problem.

Two porting notes for anyone reproducing on rootless podman: the compile step bind-mounts a cache directory that rootful podman auto-creates and rootless refuses (`mkdir` it yourself), and the `hf download` of the 18.8 GB checkpoint silently stalled at 2.2 GB with the process alive and zero bytes moving — kill and restart with `HF_HUB_DISABLE_XET=1`. That failure mode ate an afternoon once before; this time it only ate ten minutes.

## Validation before vibes

A long-context stack that only *looks* alive is worse than no stack, so before pointing anything real at it I ran the checklist. Boot log first: the patched path announces itself (`Using LBHNC KV cache layout`, PIECEWISE cudagraphs, and the number that matters — `GPU KV cache size: 385,934 tokens`). If that number comes back small, you've silently fallen back to fp8 and everything downstream is a lie.

Then real prompts, not allocated-but-empty context — the single most common way long-context benchmarks lie:

- **Greedy determinism**: same prompt six times, temperature 0 → six byte-identical outputs. This also doubles as the cudagraph-corruption detector.
- **Needle recall on genuine haystacks**: 138K tokens → both planted codes; 179K → 2/2; and later, at the full-window config, **235,655 real tokens with both codes recalled in 100 seconds cold**. That is, comfortably, the deepest verified context this GPU has ever held for me.
- **Tools and vision**: parser round-trips, a green test image correctly named. The checkpoint is multimodal; that comes along for free.
- **Throughput**, single-stream, against my incumbent brain on the same card:

| | fp8-KV tiers brain | NVFP4-KV experiment |
|---|---|---|
| Context ceiling | 106,656 | **262,144** |
| KV pool | ~191K tokens | **~380K tokens** |
| Prefill (8K, cold) | 7,936 t/s | **11,730 ±300 t/s** |
| Decode (512) | 90.5 t/s | **108–125 t/s** |
| Verified recall depth | ~47K | **235K** |

Speculative decoding acceptance sat at 84–85% with a mean accepted length of 3.5 — the model's built-in MTP head is very good at drafting itself.

## The setting that lied to me

Qwen3.8's MTP head is multi-step trained, and two independent sources measured the speculative sweet spot at `num_speculative_tokens=3`. My A/B said **4**: +18% decode (124 vs 105 t/s), tight variance, clean determinism. I shipped ns=4 into the config.

Twenty-five minutes into the first real agentic session on it — a qwen CLI chat about 40K tokens deep — the engine died with a CUDA illegal memory access. The crash dump told the story: a five-token speculative decode step, the fault surfacing one step after launch, somewhere in the patched FA2/draft path. No external validation of ns=4 on this stack exists; the gist's own day-long agent-session validation ran at ns=3.

Two lessons, one config change:

1. **Single-turn benchmarks don't certify agentic stability.** Mixed prefill/decode with a growing session exercises kernel interleavings a benchmark loop never produces. ns=4's +18% stays in my notes; ns=3 stays in the config.
2. The crash exposed a second bug for free: after an engine death, vLLM's API server **exits 0**, so `Restart=on-failure` never fires and the unit just... sits there, dead, port open. `Restart=always` on the unit. If your brain can die with a smiley exit code, make resurrection unconditional.

## The prefix-cache law of small pools

One operational truth worth knowing before you promise anyone "256K interactive": prefix-cache retention needs **2× the request** to fit in the pool alongside its own cached blocks. With a ~380K pool that means repeated turns cache-hit below ~185K — snappy. Above that, every turn re-prefills the whole conversation at ~2.5–3K t/s. A 235K-token turn costs ~90 seconds of re-reading before the first new token. One-shot deep ingestion (huge document, one question): perfect. Deep *multi-turn* chat: consider capping `max-model-len` at ~196K, where re-asks hit cache in ~10 s.

## The stack, for the impatient

Pinned base `nightly-e9d1398d9...`, the gist's flashinfer.py diff + V-scale writer kernel baked into a local image, and a serve line that boils down to:

**Turnkey bundle (builder + serve script, tested): [stondo/33016facc72c6439d836be64e87ffb8e](https://gist.github.com/stondo/33016facc72c6439d836be64e87ffb8e)**

```
VLLM_KV_CACHE_LAYOUT=HND   # leave this out and output is garbage
--kv-cache-dtype nvfp4 --max-model-len 262144
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--compilation-config '{"cudagraph_mode":"piecewise"}'   # FULL = silent corruption
--tool-call-parser qwen3_coder
--gpu-memory-utilization 0.92    # 0.98 headless → ~451K pool
```

The three invariants that will bite anyone who skips them: the **HND layout env** must reach the container, cudagraphs must stay **PIECEWISE** (FULL capture *succeeds* and then replays corrupted reasoning — the nastiest failure mode in this stack because nothing crashes), and **ns=3** until someone validates 4 under real load.

## One recipe, three checkpoints

The recipe isn't married to the gittensor checkpoint — it's the KV cache doing the work, so any NVFP4 Qwen3.8-27B that loads in this vLLM build can ride it. I A/B'd the two other checkpoints I had on disk (same image, same flags, same validation harness):

| | gittensor (everything-FP4) | unsloth NVFP4 (FP8 attn) | heretic-ara NVFP4 (W4A16) |
|---|---|---|---|
| Weights in VRAM | 17.9 GiB | 21.3 GiB | 19.6 GiB |
| Max context | **262,144** | 196,608* | **262,144** |
| KV pool | ~388K tokens | ~217-326K* | ~352K tokens |
| Prefill (8K, cold) | **11,730 t/s** | 7,554 t/s | 3,569 t/s |
| Decode (512) | **108-125 t/s** | 88.5 t/s | 91.5 t/s |
| 140K-token needle | PASS | PASS | PASS |
| Uncensored | — | — | **yes** |

\* the unsloth weights leave less room; pool varies with how much VRAM the desktop happens to hold.

Three takeaways. The everything-FP4 checkpoint keeps both crowns — its lighter weights feed the KV pool *and* the decode path (the full-vocab lm_head in FP4 is not a rounding error). The unsloth build on this recipe gains context (106K → 196K on my old fp8-KV setup) but no speed: FP8-attention dequant dominates. And the pleasant surprise: the **abliterated heretic build holds the full 262,144-token window too** — the uncensored lane used to cap at 127K, so that's a straight 2x for a model family that usually eats a quality-or-context tax.

One hygiene note while I was in there: `--trust-remote-code` was cargo cult in every serve line I'd copied (the gist's, mine). None of these checkpoints ship `auto_map` or a single `.py` file — the architecture is native in vLLM — so I proved a flag-less boot and dropped it everywhere. If a checkpoint ever genuinely needs remote code, that's a decision to make consciously, not a flag to inherit.

## Credits and expiry date

The patch stack is [co-l's gist](https://gist.github.com/co-l/c2aeaf40b53fcacfe9dd3293be75f23a) — genuinely one of the best day-0 engineering write-ups I've read, and this post is mostly me standing on it with a validation harness. The upstream endgame already exists as open PRs (vLLM #46329 and friends wire SM120 NVFP4 KV in natively); when they merge, stock vLLM takes these flags, the patched image retires, and this becomes just a config file. That's the good kind of obsolescence.

Meanwhile: 262,144 tokens, one card, ~125 t/s. The compromise menu finally lost an item.
