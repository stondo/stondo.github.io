---
title: "Three Days, Four Wrong Hypotheses, and One Uncensored 27B That Finally Serves on a Single RTX 5090"
date: 2026-08-25T12:00:00+00:00
draft: false
description: "The full story of quantizing OBLITERATUS/Qwen3.8-27B-OBLITERATED to NVFP4 for a single RTX 5090 with vLLM and LMCache: every ModelOpt trap I fell into, the assertion that survived three fixes, and the one debugging trick that cracked it in a single run."
summary: "No NVFP4 build of the uncensored Qwen3.8-27B existed, so I made one. It took a fake-quantization scare, a rename odyssey across 800 tensors, an MTP file that quietly loaded the vision tower twice, and one instrumented container run that proved everything I believed was wrong. 40 tok/s single stream, 317 tok/s with 8 concurrent subagents."
tags:
  - llm-inference
  - vllm
  - quantization
  - nvfp4
  - rtx-5090
  - blackwell
  - modelopt
  - qwen
  - self-hosting
categories:
  - engineering
  - infrastructure
keywords:
  - NVFP4
  - vLLM
  - RTX 5090
  - Qwen3.8-27B
  - OBLITERATUS
  - abliterated
  - ModelOpt
  - LMCache
  - speculative decoding
  - MTP
---

## What I wanted, and why the internet could not give it to me

I run a small fleet of machines that serve local models for my daily work. The workhorse is a single RTX 5090 with 32 GB of VRAM, and on it I had the stock Qwen3.8-27B in Unsloth's NVFP4 build, served through vLLM with tiered KV caching. That lane answers as `qwen3.8-27b-coder` and everything from my editor to my job dispatcher is configured to talk to it.

The problem: I wanted an uncensored variant for security research work. Not because I have exotic needs, but because abliterated models stop wasting your time. You ask for a stack smashing proof of concept for a vulnerability class and you get the proof of concept, not a paragraph about responsible disclosure you did not ask for.

The obvious candidate was [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED). The V3 release is good work: iterative refinement on top of direction ablation, zero hard refusals, zero soft deflection lectures, and only a 2.1 point MMLU drop from stock. It ships GGUF files, MLX, and bf16 safetensors.

What it does not ship is NVFP4. And on a Blackwell card, NVFP4 is the format you want. The 5090 has a native FP4 tensor core path and roughly 1.8 TB/s of memory bandwidth, so a 27B in FP4 is about 20 GB of weights, which leaves real room for KV cache at 126K context. BF16 does not fit. Full stop.

There were community NVFP4 builds of other uncensored variants, and I had even run one of them before and not loved the model. So I did the reasonable thing: I quantized OBLITERATUS myself. How hard could it be? NVIDIA's ModelOpt does this in a few lines.

Reader, it took three days.

## Day one: the API fights back

The quantization script itself is short. Load the model in bf16 on a big GPU, call `mtq.quantize()` with the NVFP4 config, save. I ran it on a machine with two 96 GB RTX PRO 6000 cards so I did not have to think about memory. It finished quickly. Too quickly, I would later understand.

The first crash was honest at least: `quantize() got an unexpected keyword argument 'filter_fn'`. Fine. ModelOpt 0.45 does filtering differently. You express exclusions as ordered rules in the config, appended after the enabling rules, and the last matching rule wins. Want the vision tower in bf16? Append `{"quantizer_name": "*visual*", "enable": False}`. The base `W4A16_NVFP4_CFG` already excludes `lm_head`, the vision tower, and the small internal projections of the hybrid attention layers, which saved me some work.

The second crash was subtler and it is the one I want on a mug: after a "successful" run, my output directory was **51 GB**. For comparison, the reference NVFP4 build I was cloning ideas from is 20 GB. What had I produced? A beautiful fake.

Here is the thing nobody told me: `mtq.quantize()` installs *fake quantization* modules. The weights stay in bf16 and the quantization is simulated at forward time. That is useful for accuracy evaluation and completely useless for deployment. To get real packed FP4 weights you must call `mtq.compress(model)`. And when you call it, you must call it bare, with no arguments, because the config you passed to `quantize()` contains a field literally named `"algorithm"` whose value is the string `"max"` (it is a calibration setting), and if you pass that config to `compress()` a pydantic validator somewhere dies of embarrassment.

After that, the save produced 19.7 GB of properly packed uint8 weights. I could see the tensors shrink. Progress.

Then transformers 5.15 ambushed me twice more. It saved everything as a single unindexed `model.safetensors` with no shard index file. And it silently dropped the MTP head, because the source repo ships the multi token prediction weights in a separate file following a vLLM convention that `AutoModelForImageTextToText` simply does not know about. So I wrote a post processing script: build the index myself, copy the MTP file in, merge its fifteen tensors into the index. All CPU work, no GPU needed.

The GPU part, by the way, took about **eight minutes**. Everything else was archaeology.

## Day two: the assertion

I wired the model into a systemd managed vLLM lane, cloned from my existing stock lane, same pinned container image with the LMCache tiering patches. Started it. And vLLM died, forty seconds in, with:

```
AssertionError
  at parameter.py, load_merged_column_weight
  assert param_data.shape == loaded_weight.shape
```

No tensor names. No shapes. Just the assertion and my self confidence leaving the room.

What followed was a comedy of reasonable hypotheses, each wrong:

**Hypothesis one: the config format.** My `config.json` embedded the quantization config in the nested form ModelOpt writes, and I found that a vLLM pre flight checker only accepts a flat, merged form for this quant method. The reference build ships exactly that flat form, with compressed-tensors style `config_groups` plus a top level `quant_algo`. I rebuilt my config to match, byte for byte in structure. The assertion stayed.

**Hypothesis two: tensor naming.** Comparing safetensors headers tensor by tensor, I found my export used the transformers style names, `weight_quantizer._scale` and friends, while the loader expects deployment names: `weight_scale`, `weight_scale_2`. I renamed 608 tensors and dropped 304 calibration leftovers in one pass. The assertion stayed.

**Hypothesis three: the MTP file.** This one was a genuine find. The file I copied as `model_mtp.safetensors` contained the fifteen MTP tensors *and the entire 333 tensor vision tower*. The same vision tower already lived in the main checkpoint. vLLM was loading it twice, and the second load path hit a fused projection without the context it needed. I stripped the file down to MTP only. The assertion stayed.

At this point I did the thing I should feel worse about: I ran manual tests, each flag isolated, and they all passed, so I told myself the service must have been reading a stale mmap or something. That was wrong in an instructive way. My manual containers had a five minute timeout, and this build spends the first several minutes in torch.compile before it ever touches a weight file. I was killing every test before the failing code ran. My "passes" were silent no ops.

Meanwhile the unit kept flapping. I learned something valuable about systemd here that is worth sharing: while a service is in its restart loop, `is-active` reports `activating`, which sounds like progress. My monitoring treated that as "still starting". The machine cycled through **27 crashes** while my dashboards showed a calm, slow boot. If you want to detect flapping, watch the `NRestarts` counter, never the state string. I now have a shell loop doing exactly that and it is embarrassing how much grief it would have saved.

## Day three: stop guessing, instrument

The breakthrough was abandoning hypothesis testing entirely. I patched the vLLM source *inside a throwaway container*: the assertion in `parameter.py` now wrote the actual parameter shape, the loaded shape, the module prefix, and the dtype to a file before dying. Same for the fused layer loader in `linear.py`, and for the exclusion checker in the ModelOpt integration. One run. One log file. Every question answered.

The log said:

```
FUSED cls=MergedColumnParallelLinear
     prefix=language_model.model.layers.0.linear_attn.in_proj_qkvz
     param_data=(16384, 2560) torch.uint8
     wshape=(10240, 5120) torch.bfloat16
```

And below it, page after page of:

```
EXCL prefix=...linear_attn.in_proj_qkvz verdict=False
EXCL prefix=...linear_attn.in_proj_ba    verdict=True
```

Now the story was readable. Qwen3.8 uses a hybrid architecture: 48 of 64 layers use linear attention (a Mamba style recurrent path) and vLLM implements the input projection as one fused module with four shards, `in_proj_qkvz`. The quantization config excluded the small projections fine, but for the two big ones my exclude patterns had been mangled by vLLM's own name remapper, which rewrites `something*` into `something` plus `something.*`. A pattern written against the checkpoint name can never match the remapped module name. So the fused module stayed quantized, its parameter was a packed uint8 blob, and my checkpoint (where I had spliced bf16 originals in during an earlier fix attempt) handed it a bf16 matrix. Shape mismatch. Assertion.

Two more facts fell out of the same log. The module parameter in bf16 would be `(16384, 5120)`, and my checkpoint's tensor was `(10240, 5120)`, which is only the qkv part. So even with correct exclusion, my splice could never have loaded. And the exclusion path itself was a dead end for a different reason: leaving those projections in bf16 costs roughly 6 GB across 48 layers, and on a 32 GB card that is the difference between 191K tokens of KV cache and no KV cache at all.

The reference build had the answer all along, visible in its safetensors headers if I had known how to read them: it ships those projections **quantized**. The packed FP4 path loads fused tensors whole, no shard splitting, no assertion. My very first quantization run had actually produced exactly that, and I had spent a day surgically breaking it.

## The fix

Re run the quantization with the original, correct exclusion set (only the tiny projections and the usual suspects stay bf16). Re apply the finisher: drop the calibration buffers, rename the scale tensors to the deployment convention, build the index, ship the MTP file with exactly fifteen tensors, write the flat config. Sync to the 5090 machine. Swap the lane.

One hundred and forty seconds later:

```
Loading safetensors checkpoint shards: 100%
GPU KV cache size: 191,118 tokens
```

## What it delivers

Same alias as the stock lane, so zero client configuration changed anywhere. Same tool call surface (the template dialect differs under the hood, `qwen3_coder` instead of `qwen3_xml`, but that is server side).

| Metric | Result |
|---|---|
| Single stream decode | 40 tok/s |
| 4 concurrent streams | 161 tok/s aggregate |
| 8 concurrent streams | 317 tok/s aggregate, ~40 per stream |
| GPU KV cache | 191K tokens at 126K context |
| Startup | 140 seconds warm |

The concurrency scaling is the part I care about most, because the whole point was a swarm of security research subagents sharing one GPU. Eight agents each get full single stream speed. Coherence checks pass, multi turn conversations hold together, and the uncensored behavior is exactly what the OBLITERATUS card promised: a direct technical answer about frame pointer overwrites with no preamble and no lecture.

## The checklist I would hand my past self

1. `mtq.quantize()` fakes it. `mtq.compress()` makes it real. Call it bare.
2. Exclusion rules are an ordered list and the base NVFP4 config already covers `lm_head`, vision, and the small linear attention internals. Add less than you think.
3. transformers 5.15 saves one big file with no index and drops vLLM convention MTP extras. Write your own finisher.
4. Scale tensors must be renamed to `weight_scale` / `weight_scale_2` for the deployment loader.
5. The quantization config in `config.json` must be the flat form: `config_groups` plus a top level `quant_algo` plus `quant_method: modelopt`.
6. Never exclude a fused module from quantization on a memory constrained card. If the reference build ships it packed, ship it packed.
7. When a load fails with a bare assertion, do not read tea leaves. Patch the code in a container, write the shapes to a file, run once. One instrumented run beat a day of hypotheses.
8. Watch `NRestarts`, not `is-active`. `activating` is how a crash loop looks you in the eye and lies.

The weights are on my Hugging Face if you want to skip the fun part: [Joestar79/Qwen3.8-27B-OBLITERATED-NVFP4](https://huggingface.co/Joestar79/Qwen3.8-27B-OBLITERATED-NVFP4). But honestly? The fun part taught me more than the model ever will.

If this saved you an evening, or you are fighting an assertion right now at 2 AM, my sympathies. Instrument first. The assertions are always telling the truth. It is the rest of the pipeline that lies.
