---
title: "Every Flag Justified: The Complete DeepSeek-V4.1-Flash Lane Recipe for Two RTX PRO 6000"
date: 2026-09-12T14:15:00+00:00
draft: false
description: "The companion runbook to the V4.1-Flash deployment and debugging posts: the exact serve script, container invocation, systemd wiring, and generation defaults the lane runs in production right now — with the reason every flag is what it is, and the numbers to expect from each concession."
summary: "A copy-paste serving recipe for DeepSeek-V4.1-Flash at EXL3 2bpw on 2× RTX PRO 6000 Blackwell (sm_120): rootless podman over the diffbot pack, eager mode forced by the Engram-on-NVMe path, the generation_config.json that ships official sampling to every client, the reasoning-effort knob that wants strings and not integers, and the measured performance envelope of the result."
tags:
  - llm-inference
  - deepseek
  - deepseek-v4.1-flash
  - exl3
  - vllm
  - podman
  - systemd
  - rtx-pro-6000
  - self-hosting
  - runbook
categories:
  - engineering
  - infrastructure
keywords:
  - DeepSeek-V4.1-Flash
  - serving recipe
  - vLLM
  - EXL3 2bpw
  - RTX PRO 6000
  - sm_120
  - podman rootless
  - systemd user service
  - generation_config.json
---

The [deployment post](/posts/deepseek-v41-flash-552b-exl3-two-rtx-pro-6000/) explains why this lane exists and where its ceiling sits. The [debugging post](/posts/dsv41-flash-word-salad-top-p-opencode/) explains the two-line sampling fix. This one is the thing you came for: every file the lane runs in production right now, in order, with nothing paraphrased.

Hardware assumption throughout: two RTX PRO 6000 Blackwell (96 GB each, sm_120), a consumer AM5 board with 123 GB of host RAM, and a fast local NVMe. If your host RAM is north of ~280 GB, stop reading this and go pin the Engram tables instead — the recipe's own ladder will thank you.

## 1. The pack

Everything hangs off [diffbot's `DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000): routed experts re-encoded at EXL3 2 bits-per-weight (MCG codebook), attention and dense weights kept at DeepSeek's FP8, DSpark draft layers and vision aligner untouched, and the 203 GB of Engram tables shipped unchanged in the last shards. Its `recipe/` directory contains the Dockerfiles, sm_120 kernel patches and build scripts — build the image locally, don't improvise the base: the EXL3 MoE kernel, the vLLM plugin and the FlashInfer prefill fix are load-bearing.

Two files you must add yourself, because the pack ships without them and the lane misbehaves without the second:

```bash
# generation_config.json — placed in the model directory root.
# vLLM applies these as defaults whenever a request omits the params.
# This is DeepSeek's own evaluation recipe (model card: "temperature=1.0, top_p=0.95").
{
  "temperature": 1.0,
  "top_p": 0.95
}
```

Without this file, any client that sends no sampling parameters — and several popular coding CLIs send none at all — runs an uncapped `top_p 1.0` tail on a 2-bit quant. That is how you grow the word salad from the debugging post.

## 2. The serve script

The lane wraps one command: `vllm serve` inside the locally built image, under rootless podman. The interesting part is that every deviation from the upstream recipe is a forced move, documented at the top of the script:

```bash
#!/usr/bin/env bash
# dsv41-serve — systemd entrypoint.
# Differences vs upstream serve.sh, forced by deep's 128 GB host RAM:
#   the ~190 GiB Engram tables CANNOT be pinned, so ENGRAM_DISK=1 (NVMe reads
#   via async threads) is the default, which REQUIRES EAGER=1 (no CUDA graphs).
#   Expect bench-row-1 class prefill, not the 113 t/s headline.
# Rootless podman: --device nvidia.com/gpu=all (CDI) instead of --gpus all,
#   no --shm-size alongside --ipc host, no --cap-add IPC_LOCK (pinned mode unused).
```

The invocation, annotated with the reason each flag survives scrutiny:

```bash
podman run --rm --name dsv41-flash-exl3 \
  --device nvidia.com/gpu=all --network host --ipc host \
  --security-opt label=disable \
  -v /var/aios/models:/var/aios/models:ro \
  -v $RECIPE/patches/vllm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \
  -v $RECIPE/patches/flashinfer/sparse_mla_sm120_prefill.cu:...ro \
  -e XMOE_EXT=xmoe -e XMOE_DIR=/opt/xmoe/build \
  -v $RECIPE/kernels/build:/opt/xmoe/build:ro \
  -e VLLM_PLUGINS=vllm_exl3 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 \
  -e NCCL_CUMEM_ENABLE=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm dsv41-flash-exl3-sm120 serve /var/aios/models/dsv41-flash-exl3-2bpw \
  --tensor-parallel-size 2 --host 0.0.0.0 --port 8004 \
  --max-model-len 524288 \                 # 512K; the model supports 1M, the KV pool doesn't care
  --kv-cache-dtype fp8 --kv-cache-memory 8589934592 \  # 8 GiB ≈ 3.8M tokens — V4.1 KV is tiny
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --quantization exl3 --language-model-only \           # text lane; vision tower ships but serving is day-zero
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --enforce-eager \                        # REQUIRED with DSV41_ENGRAM_DISK=1 — graphs can't capture NVMe reads
  --disable-custom-all-reduce \            # custom AR fails graph capture upstream; kept for parity in eager too
  --attention-config '{"indexer_kv_dtype":"mxfp4"}' \   # native FP4 indexer cache; sm_120 needs it for 32-token pages
  --linear-backend marlin \                # Marlin W8A16 beats FlashInfer W8A8 on MXFP8 at decode sizes
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --served-model-name deepseek-v4.1-flash
```

The kernel env vars are the recipe's real payload: `XMOE_EXT=xmoe` mounts the rewritten sm_120 MoE kernel (multi-row prefill plus the flat decode scheduler — the single biggest decode win that survives eager mode), and the FlashInfer bind-mount replaces the sparse-MLA sm_120 prefill source with the patched variant. On image rebuilds, re-derive both from the recipe's `docker/` and `kernels/` directories.

Cold start to first token: about two minutes, dominated by Engram load off NVMe and FlashInfer JIT. Set `TimeoutStartSec=3600` and don't panic at the silence.

## 3. systemd wiring

A plain user unit (not a quadlet — the entrypoint is a wrapper that loads the API key from an env file):

```ini
[Unit]
Description=DeepSeek-V4.1-Flash EXL3 2bpw via vLLM TP2 (engram-disk)
Conflicts=glm53-tr3.service        # both lanes want all 192 GB; they take turns
After=network-online.target

[Service]
Type=simple
Environment=HOME=%h
ExecStart=%h/.local/bin/dsv41-serve
ExecStopPost=-/usr/bin/podman rm -f dsv41-flash-exl3
Restart=on-failure
RestartSec=15
TimeoutStartSec=3600               # engram-disk load + FlashInfer JIT
```

Operational notes that have each cost me an hour:

- `Restart=always` if this is the default brain — vLLM's API server can exit 0 on engine death, and `on-failure` will read a dead lane as a deliberate stop (a lesson from the GLM post that carries over).
- The lane shares the box with a GLM alternative via `Conflicts=`; flipping is two `systemctl --user` commands and the cold-start wait.
- `/metrics` on the API port is **unauthenticated** even with `--api-key` on completions — Prometheus scrapes it directly, no bearer dance.

## 4. Talking to it correctly

The client-side contract, learned the hard way:

| To get | Send | Do not send |
|---|---|---|
| Official sampling | nothing — `generation_config.json` has it | your own top_p unless you mean it |
| Reasoning on | top-level `"reasoning_effort": "low"|"medium"|"high"|"max"` | `chat_template_kwargs` with an **integer** effort — ints silently no-op in this template |
| Max effort | `reasoning_effort: "max"` **and** max_tokens ≥ 4096 | small budgets — max will spend all of them thinking and return empty content |
| Non-thinking | top-level `"reasoning_effort": "none"` | — |

The lane's default is thinking-on at `high` (the agent-coding sweet spot; official benchmarks use max, interactive sessions rarely want to fund it). Note that at effort-high the reasoning often lands inline in `content` rather than `reasoning_content` — a vLLM parser gap, cosmetic, known.

## 5. What to expect

Measured on the production lane, 2× Max-Q at stock power:

| Workload | Result |
|---|---|
| Decode, short context, single stream | ~94 tok/s |
| Decode, 54K-token context | ~104 tok/s |
| Prefill, 54K tokens | ~2,700 tok/s |
| Prefill, 152K tokens | ~2,400 tok/s |
| Cold start | ~120 s |
| Agentic sessions with prefix caching | the recipe's own bench: 87% of prompt tokens from cache, 240-254 tok/s across 3-4 streams |

The 113 tok/s / 4,000 tok/s prefill headline exists — pinned Engram plus CUDA graphs, roughly 280 GB of host RAM. This recipe is the honest version for a 128 GB host: 85-90% of it, decode nearly whole, prefill paying the NVMe tax.

## 6. Maintenance map

- **Image digest changes**: re-extract and re-apply the two bind-mounted patches (pattern is in the recipe's `patches/`), rebuild xmoe via `kernels/build.sh`.
- **New inference port on the GPU box**: add it to the Prometheus `inference-deep` job or your dashboards go dark for exactly the lane you're watching.
- **The recipe's upstream** ([sfxnz's 2× DGX Spark original](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark)) is worth watching; the diffbot adaptation tracks it.
- **ds4**: if antirez ships CUDA support for V4.1, benchmark it against this lane — an engine designed to keep the 203 GB of Engram on disk by default is structurally the right answer for this hardware class.

That's the whole lane: one pack, one image, one script, one unit, one JSON file, and the discipline to send string effort levels. May your tables stay warm in page cache.
