---
title: "One Server, Every Mode: MiniMax H3 Goes From Static Frames to Full Video Editing on Two RTX PRO 6000"
date: 2026-08-27T17:00:00+00:00
draft: false
description: "My MiniMax H3 deployment could only start videos from still images. Getting the Ref2VA partition online — video-in generation and text-instructed video editing — turned into a lesson in how vLLM-Omni actually resolves model layouts: a 196 GB download of the wrong layout, a crash loop caused by a mount point basename, and an API field whose plural I needed."
summary: "H3 ships as two checkpoints: FL2VA (text + first/last frame) and Ref2VA (references including video). Serving both DiTs from one process took three failed hypotheses, a basename-sensitive model resolver, a 66 GB minimal partition download, and one multipart field spelled plural. Now one systemd unit serves text-to-video, keyframe animation, and video editing — 56 GiB per GPU, validated end to end."
tags:
  - llm-inference
  - vllm
  - vllm-omni
  - minimax-h3
  - video-generation
  - diffusion
  - rtx-pro-6000
  - blackwell
  - self-hosting
  - podman
categories:
  - engineering
  - infrastructure
keywords:
  - MiniMax H3
  - Ref2VA
  - FL2VA
  - vLLM-Omni
  - video editing
  - video generation
  - RTX PRO 6000
  - FP8 quantization
  - tensor parallelism
  - systemd quadlet
---

I have been running [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) on `deep`, my two-RTX-PRO-6000 box, since it shipped — but only half of it. The deployment served the **FL2VA** checkpoint: text-to-video, or a video grown out of one or two still keyframes. That is the mode for storyboarding from images. What I actually wanted was the other half of the model: **Ref2VA**, the omni-reference mode that starts from real inputs — up to nine images, three video clips, three audio clips — and does instructed *editing*: feed it a clip and say "make this a rainy night", and it regenerates the clip with the car, the motion, and the camera work intact.

This is the story of getting from "static pictures only" to "all modes, one server, one systemd unit" — which turned out to be less about GPUs and more about how vLLM-Omni decides what a model directory *is*.

## H3's mode map, for orientation

| Mode | Task | Inputs | What it's for |
|---|---|---|---|
| T2VA | text-to-audio-video | text only | pure generation |
| FL2VA | first/last-frame | 0–2 images | keyframe-driven generation |
| Ref2VA | omni-reference | ≤9 images, ≤3 videos, ≤3 audio (≤12 files) | video-in generation, editing, style/identity transfer |

(There are also two pipeline modules — H3-Context-IR for prompt refinement and H3-Regenerate-2K for a 2K upscale pass — but they're workflow stages, not separately servable checkpoints.)

The important discovery: since [vLLM-Omni PR #5720](https://github.com/vllm-project/vllm-omni/pull/5720), you don't pick a mode at server start anymore. Serve the combined model and `extra_params.task` routes each request — no restart between "animate this still" and "edit this clip". The vLLM recipe's guidance is to use one DiT per server only when memory forces you to it. My cards have 96 GB each; the math said both DiTs at FP8 would land around 60–70 GB per GPU with the shared Qwen3-VL encoder and both VAEs. Worth a shot.

## The 196 GB detour

H3's HF repository hosts *two layouts side by side*: the original task checkpoints in `FL2VA/` and `Ref2VA/` subdirectories, and a newer diffusers-style modular root (`transformer/`, `transformer_ref/`, shared component directories, `model_index.json` with `_class_name: MiniMaxH3ModularPipeline`). I read "serve the repo root", looked at the root layout, and downloaded exactly that: **196 GB of root-level components**, using my hardened downloader (HF token, `HF_HUB_DISABLE_XET=1`, hf_transfer, retry loop, stall watcher — Xet CAS corruption and silent hf_transfer stalls are the two failure modes that eat afternoons here).

That part worked. Then the fun started.

### Trap one: `--include` only eats one pattern

If you write:

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir ... \
  --include "model_index.json" "text_encoder/*" "transformer/*"
```

the CLI's `--include` takes **one** value. Everything after the first pattern becomes *positional filename arguments*, and the downloader helpfully warns — in a warning you will not see unless you read the log — that it is "ignoring `--include` since filenames have been explicitly set". Directory globs as positionals happen to work, so the download *looks* completely healthy. What silently doesn't happen: the first pattern — `model_index.json`, the 3 KB file that makes the root a model — never downloads. Later, a literal non-matching glob positional fails outright with a URL-escaped `File not found ... %2A`.

Lesson: `--include` once per pattern. And verify the *small* files after any big download, not just the byte count.

### Trap two: the resolver cares about your mount point's *basename*

With the root layout on disk, the server crashed on boot:

```
WARNING: No registered PipelineConfig resolved for model '/models'.
Legacy `stage_args` YAMLs are no longer supported ...
RuntimeError: Orchestrator initialization failed
```

I assumed the combined-serving feature was too new for my (Aug 19) nightly image. Before pinning or rebuilding anything, I grepped the installed package. The resolution chain in `config_factory.py` is: transformers config → root `config.json` → `model_index.json` `_class_name` matched against pipelines that declare a `diffusers_class_name` (only three models do — H3 isn't one) → and then a **basename fallback**: the last path component of the model path, lowercased, dashes and underscores stripped, substring-matched against registered pipeline keys.

My quadlet mounted the repo at `/models`. Basename: `models`. No match. The fix was one character of path:

```
Volume=/var/aios/models/minimax-h3:/models/MiniMax-H3:ro,Z
Exec=vllm serve /models/MiniMax-H3 --omni ...
```

`MiniMax-H3` → `minimaxh3` → matches the registered `minimax_h3` pipeline. Every example in the vLLM recipe serves a path ending in `MiniMax-H3` — now I know that's not a convention, it's a *requirement*. (Upstream fix would be registering `diffusers_class_name` for H3; the basename fallback is doing load-bearing work until then.)

### Trap three: this build wants the *old* layout

Boot proceeded past resolution and immediately failed with a refreshingly explicit error:

```
ValueError: Ref2VA partition not found at /models/MiniMax-H3/Ref2VA
```

Reading the pipeline source settled it: my omni build's combined mode loads `FL2VA/` as the base partition, then requires `Ref2VA/` — reads its `model_index.json` metadata and its `transformer/` weights, and takes **everything else** (encoder, both VAEs, tokenizer, processor) from FL2VA. The modular root directories I'd downloaded at such length are for a newer diffusers-path consumer. This build simply does not read them.

So the 196 GB root layout was the wrong purchase. The right one was a *minimal* Ref2VA partition: `Ref2VA/model_index.json` (3 KB) plus `Ref2VA/transformer/` (61.7 GB). Not the full 134 GB self-contained subtree — its duplicated encoder and VAEs are dead weight when FL2VA sits next to it.

A note for anyone reproducing: I kept the root layout on disk because a future omni nightly will presumably consume it, and disk is cheaper than re-downloading 196 GB. But if you're starting fresh on the current image: you need `FL2VA/` (full) + `Ref2VA/model_index.json` + `Ref2VA/transformer/*`. That's it.

### Trap four: the plural field

Server up, both partitions loaded, T2VA generating. Then the edit test — upload a clip, ask for the rainy night — returned:

```
RuntimeError: ref2va accepts at most 3 video references
```

...for one uploaded video. The H3 reference path lives behind `input_references` — **plural**, a repeated multipart file list. The singular `input_reference` routes through the generic single-media decoder, whose output the H3 reference counter doesn't understand, and the error message points anywhere but at the field name. The form parser accepts the plural field as repeated files: MP4/MOV for video, JPG/PNG for images, WAV/MP3 for audio, mixed in one request, subject to H3's 9/3/3/12 reference contract.

## The working stack

```ini
# ~/.config/containers/systemd/minimax-h3.container (podman quadlet)
[Container]
Image=docker.io/vllm/vllm-omni:nightly
PublishPort=8091:8091
Volume=/var/aios/models/minimax-h3:/models/MiniMax-H3:ro,Z
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
Environment=VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
Exec=vllm serve /models/MiniMax-H3 --omni --trust-remote-code \
  --host 0.0.0.0 --port 8091 \
  --num-gpus 2 --tensor-parallel-size 2 \
  --usp 1 --ring 1 --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

No `--task-type`: that's the whole point — requests pick their DiT. FP8 online quantization touches the two DiTs; the encoder and VAEs stay BF16. Result: **56.3 GiB per GPU** out of 96, both DiTs resident, ~2 minutes from `systemctl --user start` to health-200. The unit keeps its `Conflicts=` on the LLM brain services, so starting H3 hands the GPUs over atomically and stopping it lets the brains come back — video generation and token generation time-share the silicon by design.

And the three smoke tests, all first-try after the fixes (4-second clips, 24 steps — half the reference step count, smoke-quality):

| Task | Request | Time | Output |
|---|---|---|---|
| T2VA | "red vintage car, coastal road, sunset" | 2:26 | H.264 1344×768 + stereo AAC |
| FL2VA | first frame of that clip + "continue the drive" | 2:37 | same shape, motion continues |
| Ref2VA | the T2VA clip back in + "rainy night, wet asphalt reflections, keep the car and camera work" | 8:08 | regenerated, structure retained |

For the Ref2VA check I compared frames numerically (this session's coding model doesn't do images, so no eyeballs): per-pixel mean absolute difference of 20–28 across frames — a genuine regeneration, not a re-encode (which would sit under 5) — with a consistent cooling color shift, which is what "sunset → rainy night" should look like to a histogram. Reference-video soundtracks ride along as audio conditions, so an edit inherits its source's sound bed.

The curl that matters, for posterity:

```bash
curl -X POST http://deep:8091/v1/videos/sync \
  -F prompt="Edit this clip: rainy night, wet asphalt, keep motion unchanged" \
  -F "input_references=@source.mp4;type=video/mp4" \
  -F fps=24 -F num_inference_steps=50 -F flow_shift=12 -F seed=1101 \
  -F 'extra_params={"task":"ref2va","duration":4.0,"audio_flow_shift":3.0}' \
  -o edited.mp4
```

## What I'd tell past me

1. **Read the consumer before you download the producer.** Twenty minutes in the installed pipeline source would have shown that combined mode wants `FL2VA/` + minimal `Ref2VA/` — a 66 GB download instead of 196 GB of beautiful, unused diffusers root.
2. **Verify small files, not byte totals.** A 3 KB missing `model_index.json` hides perfectly inside a 330 GB success.
3. **Model-path basenames are load-bearing in vLLM-Omni.** Mount H3 at a path that *names* H3, or the resolver falls off its last-resort ladder with an error message that never mentions naming.
4. **When an error message is absurd** ("at most 3 video references", one video provided), suspect the request *schema*, not the content — the plural field was the whole bug.
5. One server, all modes, 56 GiB/GPU: if you have the VRAM, don't split H3 per-task. The mode-switching friction was entirely self-inflicted.
