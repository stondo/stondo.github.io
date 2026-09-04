---
title: "An Uncensored 305B With Eyes: DeepSeek-V4-Flash-Vision on 2× RTX PRO 6000, and the Two-Engine Week That Tamed It"
date: 2026-09-04T17:00:00+00:00
draft: false
description: "Replacing DeepSeek-V4-Flash-0731 with the abliterated Vision-Exp build on two RTX PRO 6000s: the SM120 kernel that rejected vision prefill, the DSML tool-call bugs that only fired in real sessions, and why I swapped sglang for a community-qualified vLLM image mid-flight — plus a router fix this adventure forced into existence."
summary: "The uncensored build of DeepSeek's Vision-Exp is a byte-for-byte drop-in — which meant every failure afterwards was mine to fix. A sparse-MLA kernel that rejects vision prefill on SM120, a streaming tool-call parser that corrupted exactly the arguments coding agents live on, and a health check that proved the port but not the model. 142 tok/s single-stream, native vision, zero refusals, one new router feature."
tags:
  - llm-inference
  - deepseek
  - vllm
  - sglang
  - blackwell
  - rtx-pro-6000
  - sm120
  - abliterated
  - tool-calling
  - self-hosting
categories:
  - engineering
  - infrastructure
keywords:
  - DeepSeek-V4-Flash-Vision-Exp
  - RTX PRO 6000
  - SM120
  - sglang
  - vLLM
  - DSML
  - tool calling
  - flashinfer
  - sparse MLA
  - token-miser
---

## The model that arrived already uncensored

My deep machine serves the fleet's heavy lane: two RTX PRO 6000 Blackwell cards, 96 GB each, one port (`:8001`), one model at a time. For a while that model was DeepSeek-V4-Flash-0731, and it was fine. Then DeepSeek shipped Vision-Exp — 305B total, ~18B active, a native vision tower, DSpark speculative decoding from the same checkpoint, and multimodal agent numbers that embarrassed its sibling (ApexBench 36.5 vs 26.2, with text-agent parity or better everywhere else).

I wanted the uncensored variant, for the same reason I always do: abliterated models don't waste your time with paragraphs you didn't ask for. And this time someone had done the hard part properly. OrcaRouter's `DeepSeek-V4-Flash-Vision-Uncensored` bakes the refusal-direction edit **directly into the official mixed-precision shards** — same tensor names, same dtypes, same shard layout, same `index.json`. It is a drop-in for the official checkpoint in any stack that already serves it. The vision tower is byte-identical; thirty-three tensors moved, most of them in the aligner. Their published evals show refusal rates collapsing from 0.85–0.98 down to 0.02–0.16 with MMLU/GSM8K inside ±1 point, and — the part I found most convincing — image-conditioned refusal at 0.583 dropping to 0.025 with *over-refusal dropping too*, all 67 discordant pairs flipping one direction.

One hundred sixty-eight gigabytes later (the download pipeline earns its name `robust-dl.sh`), I had the weights verified shard-for-shard against the index and a maintenance window to boot the lane.

Which is when the model started rejecting my GPUs. Politely. In CUDA.

## sglang bring-up: the kernel that hadn't met a vision token

The good news first: text serving worked almost immediately. The 0731 recipe carried over — TP2, DSpark, `flashinfer_mxfp4`, Think-Max default — and the very first candidate image (`lmsysorg/sglang:dev-dsv4-flash-vision`, cut two days after the weights dropped) booted in about four minutes with a 643,840-token KV pool at `mem-fraction 0.93`. Asked what 17×23 was, it said 391. Asked what color a red square was, it killed both TP schedulers.

```
tvm.error.InternalError: Check failed: (ok) is false:
Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=64 topk=256 page_block_size=64 topk_extra=512
```

That's flashinfer's SM120 sparse-MLA kernel, refusing the *vision* prefill shape specifically — the extra-cache path that image tokens widen from 128 to 512 candidates. Text prefill never touched it. This is the same kernel family I'd already fought on the GLM lane, so I knew the neighborhood; what I needed was a way around the config table.

The fix ladder, in order of how wrong each rung was:

1. **`--page-size 256`** — the official vLLM recipe uses it, so maybe the kernel just dislikes 64-token pages. The config string still said `page_block_size=64`. The extra path has its own block geometry; the flag never reaches it.
2. **`SGLANG_SM120_FLASHMLA_BACKEND=triton`** — the image ships a Triton drop-in for exactly this entry point, same kwargs including the `extra_*` vision args. Vision worked. Decode fell from ~113 tok/s to **12 tok/s**, because the env var swaps the kernel everywhere, decode included. Unusable.
3. **The nightly image** — dead on arrival with `No accelerator available` against my 610.57 driver.
4. **The DGX Spark image** — arm64. My bad, GB10 is an ARM chip.

The working fix was surgical: the dispatch inside `flash_mla_sm120.py` picks flashinfer's split-K path for decode-sized batches and its paged-attention path for extend-sized ones. Only the *paged* branch rejects the vision config. So I bind-mounted a ten-line patch that routes `B > _DECODE_MAX_TOKENS` batches to the Triton kernel and leaves decode on flashinfer. Vision answered "Red." Decode stayed at 113 tok/s. Cold start ~280s.

I want to be honest about what that patch is: a maintenance liability, pinned inside a quadlet with a comment telling future-me to re-derive it whenever the image digest changes. It bought the lane. It did not buy peace.

## The tool-calling ghosts

First real work session, and my coding agents started failing. A lot. `write` tool calls — the ones carrying an entire file as one string argument — came back broken in two distinct ways across two afternoons:

**Day one:** `JSON parsing failed ... Expected '}'`. The streamed arguments arrived as `{"arguments": {"content": "...", "filePath": "..."}.` — the DSML envelope leaking into the args, the outer brace missing, a stray period where structure should be. DeepSeek-V4 doesn't emit OpenAI-style JSON tool calls; it emits **DSML** — XML tags like `<｜DSML｜invoke name="...">` and `<｜DSML｜parameter name="content" string="true">` — and the serving stack translates. The translator was the problem.

**Day two, after I fixed day one:** `SchemaError(Missing key at ["content"])`. Arguments now parsed as valid JSON — with the entire content parameter gone. The long string value was being swallowed somewhere between the reasoning parser's `<think>` boundary handling and the DSML detector's streaming state machine.

Here's the part that made this miserable: **my synthetic tests never reproduced either bug.** Ten streamed large-content writes in a row: ten clean, well-formed calls. Multi-round loops, parallel calls, nested schemas, router in the path — all green. The failures were chunk-boundary dependent. They needed a real session's timing, a real harness's prompt stack, probably the DSpark speculative stream interleaving. I only had the failure signatures because my agent sessions kept screenshotting them — and there's a detail I enjoyed: I couldn't read those screenshots myself (the model running my editor that day has no vision), so I OCR'd them through the very lane that was failing. The diagnostic instrument was the patient.

The archaeology, though, was conclusive. sglang's `DeepSeekV4Detector` is a 67-line subclass; everything lives in `DeepSeekV32Detector`. And upstream had the receipts:

- **PR #36339** — "Fix DeepSeekV32Detector streaming corrupting tool-call arguments" — the envelope leak, still open.
- **PR #34600** — "Harden DeepSeek-V4 tool-call streaming" — *four* more bugs, including the `potentially_dsml` trap that silently discards text at stream end (my missing `content`) and a reasoning-parser `tool_start_token` of `<｜DSML｜` that matches *any* DSML sub-tag and terminates reasoning early.
- **PR #36748**, **#35563** — siblings.

vLLM had this exact bug family too — issue #40801, "intermittently leaks DSML fragments in streaming, worst with MTP" — and *fixed it*, in released versions, months ago.

I backported anyway, because the lane was hot and I wanted the model that day. #36339's head file mounted cleanly (self-contained, one import the image already ships). #34600's base was too old to mount wholesale — the image's tree had drifted past it — so I ported the two relevant bugs by hand: the `_DSML_TOOL_TAGS` narrowing and `finish()` flush into the mounted detector, the token narrowing into the image's reasoning parser. Two lessons from that afternoon, one of which cost me a silent double-apply:

- `patch(1)` run twice over one file (a `||` fallback that also fired) applies hunks twice, produces duplicate imports, and **still compiles**. Diff PR-head files against image copies instead of patching blind.
- Verify that a reported "change applied" actually changed the bytes. My patcher once reported success while its replace targeted a group that excluded the match. grep the config afterwards, every time.

Both fixes validated clean in testing. Real sessions still failed. Not as often — but "less often" is not a property you can ship to agents whose whole day is tool calls.

## The engine swap

So I did what I should have done at the first eviction: I asked what people with my exact hardware run. The answer was not ambiguous.

The [RTX 6000 Pro wiki](https://github.com/local-inference-lab/rtx6kpro) — nine hundred stars of field notes on SM120 PCIe serving — lists vLLM as its *primary* runtime for DeepSeek-V4, with sglang filed under "historical and alternate." Their DS4 runbook ("Jovian Judgement r5", updated the day I was debugging) ships a source-locked image with the vision-relevant SM120 fixes **already integrated** — the same topk-512 dual-cache prefill my own kernel patch was working around — qualified on exactly two 96 GB cards, with a validation receipt. Meanwhile a Hugging Face discussion titled "Working recipe: DeepSeek-V4-Flash-Vision-Exp on 2× RTX PRO 6000 (SM120) with vLLM" documented the three walls *they'd* hit, and the yhfgyyf SM89/SM120 fork maintains wheels and containers for the same model family. Nobody credible was running this model on sglang on this silicon by choice.

The r5 launcher turned out to be a masterclass in env-only configuration — model path, variant, DSpark depth, backend, all environment; the tool parsers (`deepseek_v4` tool + reasoning), `thinking=true`/`reasoning_effort=high` defaults, KV fp8, block 256, all baked in. Vision profile: TP2, fixed probabilistic DSpark K3 (the vision checkpoint has three draft layers, so K3 is its ceiling), b12x allreduce over PCIe, 0.975 utilization, a million-token advertised context. Two quadlet gotchas later — rootless podman can't raise `nofile` past the user's hard cap, and quadlet's `Entrypoint=` doesn't split into argv (the image's default entrypoint was already correct; my override broke it) — the lane came up in 230 seconds.

The same battery that could never catch the old bugs now runs against a stack where the bug family is *fixed upstream*: streaming large-content writes, multi-round loops, parallel calls, vision, the uncensored probes. All green, across several sessions since. Single-stream decode measured at **142 tok/s** (the sglang lane managed 113 on a good day; the qualified r5 numbers on server-class EPYC are higher still). And with DSpark K3 instead of the text model's K5, agent loops feel snappier than the raw numbers suggest.

The orca weights dropped in unchanged — that's the quiet payoff of a byte-for-byte abliteration. Whatever broke belonged to the serving stack, and the serving stack is now someone else's well-maintained problem.

## The bug the model swap exposed

One more thing, because it's the part I'd want you to steal.

My router (token-miser, the little Go tiering proxy this fleet runs on every machine) picks upstreams from health-checked chains. Deep-lane deployments check `GET /health`. Here is the thing about `/health` on a shared GPU lane: **it proves the port answers. It does not prove which model holds it.** My lanes flip via systemd `Conflicts=` — GLM and DeepSeek and flashnext all take turns owning `:8001` — and every router deployment and every harness entry just follows whoever's up.

With the sglang dev build, a request for `glm-5.3-flash` routed at the vision lane got *silently served by the vision model*. With vLLM, which validates model names strictly, it got a 404. The lenient server was lying to me politely; the strict one at least complained. Both were wrong, and the router was the right place to fix it — so I fixed it there:

```
[deployments."glm53@deep"]
base_url = "http://127.0.0.1:8001/v1"
model = "glm-5.3-flash"
verify_model = true   # new: healthy only when THIS model holds the port
```

`verify_model` adds a second stage to the health probe: after `/health` passes, the task fetches `{base_url}/models` and requires the deployment's model to be listed before recording a success. Failures go through the normal hysteresis, and the eviction reason names reality:

```
glm53@deep -> evicted
  model "glm-5.3-flash" not served (lane serves: [DeepSeek-V4-Flash-Vision])
```

That's now upstream in token-miser (commit `1de921c`, with unit tests including a live one-shot HTTP server that asserts the probe actually hits `/v1/models`). The chains went back to quality-first ordering — GLM primary, DeepSeek fallback — because the router can finally tell the difference between a dead deployment and a live one serving someone else.

## What I'd tell you to do

- **Run the orca build.** A byte-for-byte drop-in removes the weights from your blast radius. When something breaks you get to suspect your stack, which is a *gift*, because your stack is the thing you can fix.
- **On 2× RTX PRO 6000 (SM120), serve Vision-Exp with the rtx6kpro-qualified vLLM image**, TP2, DSpark K3, b12x. Cold start ~4 minutes, 142 tok/s single-stream measured, 1M context advertised. Don't hand-roll the sglang kernel patch unless you enjoy that kind of thing — and if you do, mount it, comment it, and date it.
- **DSML tool calls are the model family's weak point.** If your agent harness fails intermittently on big `write` arguments while every isolated test passes, it's not your imagination and it's probably not the model. Check your serving stack's parser issue tracker for "DSML", "streaming", "arguments". vLLM ≥ 0.20.1 has the fixes; on sglang, verify #36339/#34600 have merged before you trust a green smoke test.
- **A health check that doesn't verify the model name is a rumor, not a check.** If one port can serve several models, your router needs to know who's actually home. `verify_model` in token-miser is forty lines; steal the idea for whatever you run.
- And the meta-lesson, same as the last post, worth repeating until I stop needing it: **synthetic green means nothing for timing-dependent bugs.** The failing input is a real session. Reproduce with reality or don't claim reproduction.
