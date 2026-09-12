---
title: "The Word Salad Only opencode Could See: Hunting a Phantom Garbage Bug on DeepSeek-V4.1-Flash"
date: 2026-09-12T13:00:00+00:00
draft: false
description: "The new V4.1-Flash lane benchmarked clean for hours, then a user report: it starts producing garbage after a while, and only in some CLIs. The hunt ran through four test batteries that found nothing, a tcpdump that captured nothing, a reverse proxy that caught the actual payloads — and ended at a two-line JSON file fixing a sampling default nobody had set."
summary: "CLI-dependent degradation on a quantized 552B MoE: concurrency hammers, a logging proxy that exposed what opencode really sends (nothing — and that was the bug), one reproduced episode of fluent Chinese nonsense in twelve cycles, and the missing generation_config.json that separated an uncapped top_p 1.0 tail from DeepSeek's official recipe. Plus the reasoning-effort knob that exists as 0-100 in one engine and string levels in another."
tags:
  - llm-inference
  - deepseek
  - deepseek-v4.1-flash
  - vllm
  - opencode
  - debugging
  - sampling
  - self-hosting
categories:
  - engineering
  - infrastructure
keywords:
  - DeepSeek-V4.1-Flash
  - vLLM
  - sampling defaults
  - generation_config.json
  - top_p
  - opencode
  - speculative decoding
  - DSpark
  - garbage output
---

The lane was two days old and had passed everything. [DeepSeek-V4.1-Flash at EXL3 2bpw](/posts/deepseek-v41-flash-552b-exl3-two-rtx-pro-6000/) on the two RTX PRO 6000s, benchmarked at every context length I could construct, tool calls round-tripping as valid JSON, streaming clean, 152K-token prompts answered correctly. Then the report came in from the person actually using it:

> It starts producing garbage after a while. It depends on the CLI. opencode is bad. qwen is better.

The two sentences every debugging session should fear. *After a while* — so not the first request. *Depends on the CLI* — so not obviously the engine. And the CLIs disagree, which means somewhere in the stack, two clients that speak the same API are getting different models.

## Batteries one and two: everything I knew how to test, clean

First move: reproduce against the raw API, no CLIs involved. I built a scoring harness that flags the failure signatures — n-gram repetition loops, entropy collapse, mojibake, template-token leakage, and the one that would matter later, wrong-script drift in what should be an English answer.

- Single shots at every temperature I suspected: greedy, 0.7, 1.0. Clean.
- A 24-turn conversation accumulating history. Clean.
- Tool-call histories shaped like an agent's — assistant messages with `tool_calls`, tool results, a 3 KB file body in the arguments (the exact shape that broke sglang's DSML parser on the [previous lane](/posts/deepseek-v4-flash-vision-uncensored-two-rtx-pro-6000/)). Clean, valid JSON both directions.
- Thinking on and off via template kwargs. Streaming, chunk by chunk. A 152K-token prompt. All clean.
- Eight turns × 700 tokens at the server-default sampling — because my leading theory was "nobody set the temperature and temp 1.0 on a 2-bit quant eventually derails." Clean. Every tail coherent. Theory dead, and I missed the eulogy that mattered: *the temperature was never the parameter to watch.*

Two scorer flags fired along the way and both were my own harness misreading `finish_reason: tool_calls` as "empty content." A garbage detector that can't tell a tool call from a seizure generates its own false positives. Noted, fixed, moved on.

The raw API wouldn't break. Which meant the difference lived in what the CLIs actually send — and I was going to have to look.

## The capture that wouldn't

I knew the shape of the answer I needed: the literal JSON bodies each CLI puts on the wire. Plan: tcpdump on the GPU box, port 8004, run each CLI, read the POSTs.

Tcpdump captured nothing. Twice. First because `sudo -n` inside an SSH session failed silently behind a `2>/dev/null` — the classic "I hid the error that would have explained me" — and second because my cleanup `pkill -f "tcpdump.*8004"` matched the SSH shell that contained the pattern in its own command line, and killed the session instead. I have this exact pitfall written down from a previous adventure. It got me anyway. That's why it's a pitfall.

New plan, no root, no packet parsing: a twenty-line Python reverse proxy on my workstation, listening on 127.0.0.1:18004, logging every request body to a file, forwarding to the lane. Registered as a transient systemd unit so no process-group cleanup could eat it — a lesson from the same pitfall family. A temporary `deep-dsv41-cap` provider in opencode pointed at it, one agentic task with file reads, six request bodies captured verbatim.

## What opencode actually sends

Six requests told the whole story:

1. **No `temperature`. No `top_p`. No `max_tokens`. No `reasoning_effort`.** Nothing. Every sampling decision delegated downward.
2. `store: false`, `prompt_cache_key: <session>` — ignored by vLLM, harmless.
3. A **concurrent pair of requests every turn**: a tiny title-generation call alongside the main one. Two streams, same engine, every single turn.
4. Assistant history echoed back including a `reasoning_content` field — all of them empty. Red herring, but a beautiful one for about ten minutes.

And the lane, remember, ships **no `generation_config.json`** — so the server-side defaults for anything unsent were vLLM's built-ins: `temperature 1.0, top_p 1.0`. Uncapped nucleus. On a 2-bit quant, with DSpark speculative decoding doing Gumbel-style probabilistic acceptance underneath.

The official DeepSeek-V4.1-Flash evaluation recipe, from the model card, in a paragraph I'd skimmed past during deployment: *"Evaluations use `temperature=1.0, top_p=0.95`."*

Temperature 1.0 — by design, my battery had proven that. **top_p 0.95 — the missing half.** The difference between opencode and qwen wasn't the model; it was that one CLI's requests ran the uncapped tail and the other's didn't (and qwen, single-stream per turn, had half the exposure to whatever lives in the tail).

## Reproducing it: once in twelve

The concurrency hammer: fire the title request and the main request simultaneously, exactly like opencode, twelve cycles, server-default sampling. Sanity probes between cycles.

Cycle two, the main stream came back as this — an English-language request about Python exception propagation, answered in fluent, grammatical, completely deranged Chinese:

```
。对于异常的层级抛送结构，这样的向上实搜体现了识别局限性的层级根本约束。
一旦层级上某匹配无法凭借连续栈树开展筛查，异常突破点则收敛到算法位移模式的栈式障碍区。
丢弃的异常往往通过框架化的次序基于最底层栈入口先将关联性打破消化…
```

Fluent nonsense is the signature of a *sampling* failure, not a numerics failure — the grammar model survived; the meaning coordinator didn't. Every sanity probe before and after: perfect. Fourteen more rounds of a single-stream agent simulation with tools: perfect. So: per-request degeneration under concurrent batching, at uncapped top_p, on a 2bpw quant. Rare, real, and client-dependent exactly as reported.

I tried to reproduce it statistically and got a lesson in intermittency instead: three arms × sixteen cycles — control, official sampling, conservative sampling — zero anomalies anywhere. When the failure rate is one-in-thirty, a clean run proves nothing except that the fix is cheap enough to apply without further conviction.

## The fix is a file

Two lines in the model directory:

```json
{
  "temperature": 1.0,
  "top_p": 0.95
}
```

vLLM reads `generation_config.json` as the default for any sampling parameter a request omits. One lane restart (~120 s cold start) and the log confirmed it: *default sampling parameters overridden by generation_config.json*. Every CLI that sends nothing now gets the recipe DeepSeek actually evaluates with — no client changes, no per-CLI settings, nothing for the next CLI I adopt to forget.

The vLLM logs had corroborated the mechanism without my noticing at first: the top-p sampling kernels (`_topp_sb_*`, FlashInfer) only JIT-compile the first time a request *sets* top_p. Every request before the fix had been skipping the nucleus entirely.

Post-fix: the same concurrency pattern that produced the word salad, zero anomalies. And the lane kept its official-sampling shape for the CLIs that do send parameters — nothing overrides a client that speaks up.

## Second act: the effort knob that isn't

While verifying the fix I chased one more discrepancy. antirez's ds4 engine exposes V4.1's reasoning effort as `--think-level`, an integer 0-100. The official API: integer 1-100. My lane:

- `chat_template_kwargs: {"thinking": true, "reasoning_effort": 95}` — **silently produces no thinking at all.** The integer scale doesn't exist in this vLLM template build; it doesn't error, it just quietly drops the feature.
- Top-level request parameter `reasoning_effort` with **string** levels — `none | minimal | low | medium | high | xhigh | max` — works. ("max" is DeepSeek-V4-specific, per the protocol source.) Same request, `high`: reasoning triples, answer unchanged.
- `max` works too, in the sense that it consumed an entire 3,000-token budget on reasoning and returned `finish_reason: length` with empty content — the Think-Max pitfall from the old lane, unchanged by the new model. Budget ≥4K or don't ask for max.

Silent no-ops are worse than errors; an integer effort that quietly disables thinking is how a "quality lane" serves fast-mode answers for a week before anybody notices. The lane now defaults to thinking-on at `high`, and per-request overrides use strings.

## What I'd tell past me

1. **The sampling parameters a client doesn't send are also a configuration** — chosen by a default nobody read. When different clients behave differently against the same engine, diff their bytes, not their reputations.
2. **A repro you can't repeat is still a repro.** One degraded stream in twelve cycles, with clean sanity probes bracketing it, localizes the fault better than an hour of passing tests.
3. **Fluent garbage means sampling; broken garbage means numerics.** The grammar/meaning split is the cheapest triage question in the book.
4. **Ship the model card's eval config as the server default.** `generation_config.json` is two lines and it fixes every future client at once.
5. **Watch for silent parameter no-ops on day-zero engines.** An int where the template wants a string doesn't raise — it just quietly turns your reasoning model into a fast one.
6. **Read your own pitfall notes before the pitfall reads you.** The `pkill` self-match got me twice now. There will not be a third; the capture path is a systemd transient unit and a dumb proxy, and it works on the first try.

The lane is serving as I write this, thinking on, effort high, top_p 0.95, one `generation_config.json` heavier. The GLM-5.3 lane next to it watches politely from the other side of a `Conflicts=` line while the head-to-head decides who boots. And somewhere in my harness, the garbage scorer keeps running on every response — because a bug you caught once deserves a permanent witness.
