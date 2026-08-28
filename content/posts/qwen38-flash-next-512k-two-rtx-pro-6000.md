---
title: "Qwen3.8-Flash-Next at 512K Context on Two RTX PRO 6000: 146 t/s, a 51B-Entry Table in Host RAM, and the FP8 KV Door That Stays Shut"
date: 2026-08-27T22:00:00+00:00
draft: false
description: "The official vLLM recipe for Qwen3.8-Flash-Next trusts GB300 at TP4. I wanted it on two workstation-class RTX PRO 6000 cards: TP2, the PLE n-gram table offloaded to pinned host RAM, YaRN stretched to 512K context, MTP speculative decoding. That stack delivers a 1.42M-token KV pool, 146.5 tok/s decode and working vision. It also taught me that fp8 KV is architecturally dead on QSA (two engines, two failure modes), that systemd strips quotes out of JSON args, and that 1M context dies at compile, not at KV budgeting."
summary: "Flash-Next is Qwen's sparse-attention flagship and its official FP8 recipe assumes datacenter GB300. Adapting it to 2x96 GB workstation Blackwell took one legit trick (offloading the 51B-entry PLE n-gram table to pinned host RAM, freeing ~25 GiB per GPU for KV), one config archaeology exercise (YaRN via --hf-overrides because the build has no --rope-scaling flag), and discipline about what not to retry: fp8 KV hard-requires BF16 on QSA and fails differently on vLLM and SGLang, and 786K+ contexts OOM during compile no matter the KV math. Final stack: 512K context, 1.42M-token pool, 146.5 tok/s decode with MTP ns=3, 9-11.2K tok/s prefill, vision and tools validated. The SGLang NVFP4 day-0 lane runs too (after three SM120 gates), but serves one request at 262K slower than the vLLM lane serves two at 512K, so vLLM keeps the port."
tags:
  - llm-inference
  - vllm
  - sglang
  - qwen
  - sparse-attention
  - long-context
  - speculative-decoding
  - rtx-pro-6000
  - blackwell
  - self-hosting
  - podman
  - systemd
categories:
  - engineering
  - infrastructure
keywords:
  - Qwen3.8-Flash-Next
  - vLLM
  - SGLang
  - sparse attention
  - QSA
  - 512K context
  - YaRN
  - PLE offload
  - MTP
  - speculative decoding
  - RTX PRO 6000
  - SM120
---

My rule for new frontier-ish releases: the day the official recipe lands, try to break it on the hardware you actually own. Qwen3.8-Flash-Next is the new entry in Qwen's sparse-attention Flash line: hybrid GDN linear-attention layers plus QSA (query sparse attention) full-attention layers, an always-on thinking mode, native image and video input, and a multi-token prediction head for self-drafting. The official FP8 checkpoint is **172.78 GiB**, and the official vLLM recipe (vLLM 0.28.0+, served from the `vllm/vllm-openai:qwen38-flash-next` image; PyPI installs are explicitly not supported for this model) was validated on GB300 at TP4.

My `deep` box is two RTX PRO 6000 cards: 96 GB each, workstation Blackwell (SM120), 123 GB of host RAM. Half the GPUs, same architecture family, and a recipe that says TP2 is the validated *minimum* for FP8. So the question was never "does it fit" but "what has to move out of video memory to make room for context".

The answer turned out to be a lookup table. And the failure modes along the way were better teachers than the success.

## What actually ships in the checkpoint

The 172.78 GiB on disk is not all transformer weights. Flash-Next carries a **PLE component: a 51-billion-entry n-gram table**, sharded across tensor-parallel ranks at load time (the boot log literally streams `Loading safetensors checkpoint shards(PLE-offload)` as its own phase). On the GB300 recipe this sits in GPU memory and nobody notices, because a GB300 has memory to burn.

On two 96 GB cards it is the whole ballgame. Weights-plus-table on-GPU lands around 90 GiB per card, which after runtime overheads leaves a KV pool too small to be interesting. The recipe ships an escape hatch for exactly this: `VLLM_PLE_CPU_OFFLOAD=1` parks each rank's table shard in **pinned host RAM** (~51 GiB across ranks plus runtime headroom, comfortably inside deep's 123 GB), and the GPU footprint drops to roughly 61-65 GiB per card. That is ~25 GiB per GPU handed back to KV cache, which at Flash-Next's hybrid-architecture rate of ~32 KiB/token is most of a million tokens of context capacity, bought with RAM that was idle anyway.

One requirement that will bite rootless-podman users: pinned host pages need `LimitMEMLOCK=infinity` in the unit's `[Service]` block (plus `--ulimit memlock=-1:-1` on the container args), and the user manager must actually be allowed to raise it. Without that, the offload fails in a way that looks like a mysterious weights-loading crash.

## Stretching to 512K: YaRN through the side door

Flash-Next trains to 262,144 tokens. The recipe's context extension is YaRN, and here is the first bit of archaeology: **this vLLM build has no `--rope-scaling` flag for it.** The YaRN parameters go through `--hf-overrides`, as JSON, into `text_config.rope_parameters`:

```json
{"text_config": {"rope_parameters": {
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 2.0,
  "original_max_position_embeddings": 262144
}}}
```

Factor 2.0 doubles the window to **524,288**. Because that exceeds the trained length, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` has to ride along or vLLM refuses the config on principle.

The KV budget per 95.01 GiB card then works out to: ~64.6 GiB weights, ~2.1 non-torch, ~2.0 CUDA graphs, ~2.5 activation/compile slack, and **20 GiB of KV per GPU** (`--kv-cache-memory 21474836480`). Measured pool after boot: **1,424,140 tokens**, which is 2.7 concurrent requests at the full 512K, or 5.4 at the native 262K.

Where it stops is worth knowing in advance: 786K and 1M contexts **OOM during compilation**, in activation and CUDA-graph capture territory, before KV budgeting even gets a vote. More KV memory would not have saved them. And static YaRN past factor 2 degrades short-text quality, so 1M was not a prize worth chasing on this stack anyway.

## Numbers, after the validation pass

| Metric (single stream unless noted) | Value |
|---|---|
| Decode with MTP ns=3 | **146.5 tok/s** |
| Prefill, 16K-token chunks | 9,000-11,200 tok/s |
| llama-benchy aggregate, 4 concurrent | 430-540 |
| KV pool | 1,424,140 tokens (~32 KiB/token) |
| Context | **524,288** (YaRN factor 2.0) |
| Concurrency at full 512K / at 262K | 2.7 / 5.4 |
| Vision (image and video input) | validated |
| Tools (`qwen3_coder` + `qwen3` reasoning parser) | validated |

For orientation: my single-RTX-5090 27B brain decodes at ~125 tok/s with its own MTP. A sparse-attention flagship outrunning a 27B dense model per stream, on two cards, while holding double the context window, is the whole Flash pitch made real.

Two recipe details that look cosmetic and are not. `--max-num-seqs` **must stay 256**: other values trip a mamba-cache capacity error at startup (the hybrid GDN layers carry fixed-size state, and the cache is sized against that exact sequence count). And the sampling recipe differs by mode: thinking mode wants temp 1.0 / top_p 0.95 / top_k 20; instruct mode temp 0.7 / top_p 0.8 with **presence_penalty 1.5**, which doubles as the anti-repetition-loop knob on an always-thinking model.

## Two engines, two walls: fp8 KV stays dead

With a BF16 main KV, the obvious next lever is fp8 KV cache: halve the bytes, roughly double the pool. On this architecture it is a hard no, and the two engines fail in usefully different ways.

**vLLM fails honestly.** Load time, immediately: `NotImplementedError`. The QSA path requires a BF16 main KV cache; the quantized-KV plumbing simply is not wired for it. No crash loop, no mystery, just a closed door with a sign on it.

**SGLang fails entertainingly.** It boots. A 2.69M-token pool, correct short answers, everything looks great, right up until the scheduler dies inside the QSA indexer's Triton kernel with `AssertionError: Unsupported rhs dtype fp8e4nv in tl.dot(q_values, keys)`. A boot that works is not a deployment that works.

I burned an evening on the second one before trusting the first one. The takeaway generalizes: when an engine tells you an architecture combination is unsupported, believe it before you go ask the other engine to surprise you. On QSA + SM120, BF16 KV is the only KV.

## The other lane: SGLang's day-0 recipe, three gates to pass

SGLang shipped same-day support for Flash-Next (their blog post, plus the `lmsysorg/sglang:qwen38flashnext` image carrying model support that is still an unmerged PR, sglang#36497). Unlike vLLM, they also shipped **their own quant of the model**: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, ~93 GiB of W4A4. On paper, W4A4 plus SGLang's IndexShare MTP should outrun an FP8 checkpoint. The cookbook's validated hardware matrix is B200/B300/GB300, and SM120 is not on it. Running this lane was the test of exactly that.

Three things break on SM120, each with a fix:

1. **The MoE backend.** The default `FLASHINFER_TRTLLM` path is SM100-only and dies during CUDA-graph capture with a `NotImplementedError` whose message actually prescribes the cure: `--fp4-gemm-backend flashinfer_cutlass`.
2. **The GDN half of the hybrid.** The SM100 "all-FlashInfer" flag set does not transfer: FlashInfer prefill wants FP32 checkpoints on SM120. The working trio is `--linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer --mamba-ssm-dtype bfloat16`.
3. **The QSA decode path.** The stock image falls back to the FlashAttention-4 CuTe DSL on SM120 and dies at warmup with a "weakly congruent" MLIRError. The fix is a single hardware gate in `qwen_sparse_attn_backend.py` that opens the FlashInfer trtllm-gen XQA decode kernel to SM120, bind-mounted over the image's copy:

```diff
+    from sglang.srt.utils.common import is_sm120_supported
...
-    if not is_sm100_supported():
+    if not (is_sm100_supported() or is_sm120_supported()):
```

That is the entire patch. One gate, three lines, and the sparse-attention decode path that GB300 gets natively starts working on a workstation card. It is a patch against a moving target (unmerged PR image), so re-verify on every image pull.

Credit where due: while wiring this up I found Infatoshi's write-up (`qwen38-flash-next-2x-rtxpro6000s`) running the *identical* hardware and the same image digest, which pinned the validated contract before I touched anything: single concurrent request at 262,144 context, 127.8-128.3 tok/s decode, 8.8-11.6K tok/s prefill, ~79 GiB per GPU, `--mem-fraction-static 0.80`, and a startling **115 GB host RAM peak during load** (their 91 GB machine survived on 128 GB of swap; deep's 123 GB held without it). My own measured numbers on this lane: 114-117 tok/s single-stream, in family with theirs.

Two SGLang-specific behaviors worth knowing: it applies the checkpoint's own `generation_config.json`, so leave sampling alone unless you have measured cause; and like the vLLM side, thinking cannot be turned off, with `--reasoning-parser auto` sorting the output streams.

## Why vLLM won anyway

| | vLLM FP8 lane (production) | SGLang NVFP4 lane (fallback) |
|---|---|---|
| Decode, single stream | **146.5 tok/s** (MTP ns=3) | 114-117 tok/s (MTP unvalidated here) |
| Context | **524,288** (YaRN 2.0) | 262,144 |
| KV pool | 1.42M tokens | sized for c=1 |
| Concurrency | max-num-seqs 256; c2.7 at full 512K | `--max-running-requests 1` |
| Vision | validated | not exercised |
| Recipe status | official vLLM recipe; TP2 = validated minimum | day-0 image, unmerged PR #36497, SM120 off-matrix |
| SM120 modifications | none, stock image | MoE backend flag + GDN backend trio + one-gate patch |
| PLE table | offloaded to pinned host RAM (~51 GB) | stays on GPU (host offload is BF16-checkpoint-only) |
| Host RAM profile | steady ~51 GB pinned | 115 GB transient peak at load |

The honest read: the SGLang lane is not a failure, it is an earlier-stage thing. W4A4 plus IndexShare MTP probably does beat FP8 once MTP is validated on SM120, and the flag shape is already known from SGLang's blog numbers (EAGLE, 2 draft steps, top-k 1, 3 draft tokens, validated at TP4 on H200-class parts). But today, on this hardware, it serves one request at 262K slower than the vLLM lane serves two at 512K, off a moving-target image, with a bind-mounted patch keeping its decode path alive. The quadlet stays installed as the fallback brain and as the fast path the day that MTP A/B lands.

## The crash loop that was never a bug

The vLLM unit was restarting 65 times in a row at one point, and the error was `cannot be converted to <function loads>`: invalid JSON in `--speculative-config` and `--hf-overrides`. The JSON was valid. The problem is that **systemd's `Exec=` parsing strips plain double quotes** before the container ever sees the arguments.

The fix is ugly and mandatory: backslash-escape every quote inside JSON passed through quadlet `Exec=`. Sixty-five restarts for four backslashes. Related operational notes from the field: the image's entrypoint is `vllm serve` itself, so `Exec=` is arguments-only (a leading command prefix crashes the container); `TimeoutStartSec=1800` is not paranoia when cold start means 173 GiB of shards plus compile; and when you stop either lane, **port 8001 drains for up to 180 seconds**. Starting the next brain during the drain fails with "Address already in use" no matter how dead the previous one looks. Wait out the drain, `systemctl --user reset-failed`, then start.

Last ops fact, by design not accident: this lane owns port 8001, which is also the home of my usual deep brain. The quadlet's `Conflicts=` handles the handover atomically, and the router loses its default model until the lane stops again. Test windows, planned as such.

## Footnote: GLM-5.3-Flash

Also on this box: GLM-5.3-Flash now fits the same two cards (switching the MoE backend to `flashinfer_cutlass` skips the 3.4 GiB Marlin repack that used to OOM the load), and it serves, and every reply degenerates into `locklock` loops. The cause is an SM120-specific Triton miscompile in the linear-attention KDA chunk kernel: bit-exact token outputs, but a poisoned recurrent state handed to decode ([minimal repro filed upstream](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark/issues/4)). That debugging story deserves its own post once the model actually works; it stays parked until then.

## The actual recipes

Both lanes are podman quadlets in `~/.config/containers/systemd/`, deliberately manual-only (no `[Install]` section; starting either atomically stops whatever else owns :8001 via `Conflicts=`). Full unit files, verbatim:

- **vLLM FP8 lane** (the production candidate, PLE offload + MTP ns=3 + YaRN 512K): [stondo/d6c467b59d0880dd5511671e63f44a2e](https://gist.github.com/stondo/d6c467b59d0880dd5511671e63f44a2e)
- **SGLang NVFP4 lane** (fallback, all three SM120 gates + the validated c=1 contract): [stondo/90bf3e3f76f8d6a2e6d754f7a21833a8](https://gist.github.com/stondo/90bf3e3f76f8d6a2e6d754f7a21833a8)

The vLLM unit distilled to its load-bearing lines:

```ini
[Container]
Image=docker.io/vllm/vllm-openai:qwen38-flash-next
Volume=/var/aios/models/qwen38-flashnext-fp8:/models:ro,Z
PodmanArgs=--ipc=host --ulimit memlock=-1:-1
Environment=VLLM_PLE_CPU_OFFLOAD=1
Environment=VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
Exec=--model /models \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3}" \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 \
  --kv-cache-memory 21474836480 \
  --max-num-seqs 256 \
  --max-model-len 524288 \
  --hf-overrides "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":2.0,\"original_max_position_embeddings\":262144}}}" \
  --enable-prefix-caching --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3

[Service]
LimitMEMLOCK=infinity
Restart=on-failure
TimeoutStartSec=1800
```

And the operational loop, because a manual lane is only as good as its start/stop hygiene:

```bash
systemctl --user start qwen38flashnext-vllm   # ~10-15 min cold start; journal shows
                                               # the PLE-offload shard phase, then compile
systemctl --user stop  qwen38flashnext-vllm   # then wait out the 180s port drain before
                                               # starting any other brain on :8001
```

The lane graduated from test lane to production candidate on my fleet: wired as the `deep-flashnext` provider (model `flashnext`) in my agents, one start away, one Conflicts-driven stop away from handing the port back.

## What I'd tell past me

1. **Offload the table before you shrink the model.** The PLE table in pinned host RAM is worth ~25 GiB per GPU of real KV. That is the difference between a demo and a 512K-context server on workstation cards.
2. **When the build lacks a flag, the config object behind it is still there.** `--hf-overrides` carrying the full YaRN parameter block is just as recipe-faithful as a dedicated flag, once you stop looking for the flag.
3. **Believe the honest engine first.** vLLM's load-time `NotImplementedError` on fp8-KV-plus-QSA described reality; SGLang's successful boot described a scheduler crash waiting for a quiet moment.
4. **A faster-on-paper quant outside the validated matrix is a science project, not a deployment.** The SGLang NVFP4 lane is worth every hour I put in, and it still lost the port on concurrency, context, and maturity, not on anyone's mistake.
5. **Test the ceiling where it actually breaks.** Context limits here die in compile, not in KV budgeting; no amount of `--kv-cache-memory` arithmetic predicts them.
6. **Escape your quotes.** systemd strips them, vLLM gets garbage JSON, and the failure mode is a restart loop, not a config error. Four backslashes, sixty-five restarts.
