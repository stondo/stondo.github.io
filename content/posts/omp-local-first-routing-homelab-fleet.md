---
title: "One Coding Agent, Three GPUs, Two Cloud Keys: Local-First Routing and the Judge That Was Quietly Burning My 5090"
date: 2026-10-02T09:00:00+00:00
draft: false
description: "I audited the model routing of my coding agent over the three-box fleet and found one dead fallback, one silent GPU tax, and one model name that does not exist. The fix routing: local models carry the token volume, cloud keys carry the decisions, and a 0.9-second local decision model now does every internal judgment."
summary: "An audit of my coding agent's model routing turned into a routing philosophy plus three real bugs. The principle: route by token volume, not vibes — the default loop, subagent fan-out, and background work run on local GPUs for free, while planning, review, and deep reasoning buy cloud quality. Found along the way: a context-promotion target pointing at a provider I have no credential for (a silent no-op), omp's internal judge role degrading to prompted 27B judging because no TypeSafe key exists (fixed with Ollaya's winnow:e4b decision model on the senses node, calibrated answers in 0.93 s on CPU), and the DeepSeek V4.1 naming trap where the literal model id does not exist — deepseek-flash IS V4.1-Flash, and deepseek-v4-pro server-routes to it anyway. Plus the GLM-5.3 flash/base vision trap, verified with live image requests."
tags:
  - ai-agents
  - llm
  - vllm
  - self-hosted
  - omp
  - deepseek
  - glm
  - ollaya
  - homelab
cover:
  image: "/omp-routing-card.png"
  alt: "Terminal-style card: local-first AI model routing across three boxes and two cloud providers"
---

My coding agent spends most of its life talking to hardware I own. That was the
design intent, anyway. So when I sat down to answer one question — *is my agent
config actually using my local AI in a meaningful way, or is it decoration?* —
I expected a config review. What I got was a routing philosophy check plus
three genuine bugs, one of which had been quietly taxing my GPU on every
single turn.

The agent is [omp](https://omp.sh) (oh-my-pi, [can1357's fork of
Pi](https://github.com/can1357/oh-my-pi)), and the hardware is the fleet I
described in [the three-node agentic stack
post](/posts/aios-three-node-agentic-coding-stack/): a two-GPU brain, a
workhorse, and a senses node. Nothing here is exotic. What follows is the
routing that came out of the audit, and the traps I hit while verifying it.

## The fleet, in one paragraph

The **brain** is a pair of RTX PRO 6000s serving `qwen3.8-flash-next` at 512K
context through vLLM — the box I built for [half-a-million-context
prefills](/posts/qwen38-flash-next-512k-two-rtx-pro-6000/). The **workhorse**
is an RTX 5090 running my NVFP4-KV quantization of Qwen3.8-27B — the one from
[the 262K single-card build](/posts/qwen38-27b-nvfp4kv-262k-single-rtx5090/)
and [the OBLITERATUS
quantization saga](/posts/obliteratus-qwen38-27b-nvfp4-rtx5090/) — which
handles eight concurrent subagents without breaking a sweat. The **senses
node** is an RTX 4080 running SearXNG, embeddings, a reranker, speech in and
out, and the monitoring stack. On top of that, exactly two cloud API keys:
z.ai for GLM 5.3, and DeepSeek.

Local compute for volume. Cloud for decisions. That's the whole thesis — the
interesting part is making an agent harness actually respect it.

## Route by token volume, not vibes

The mistake I see in most "hybrid" setups is routing by *task type* ("coding
is local, research is cloud") instead of by *token economics*. The right
question for each role is: where does the volume go, and where does the
stakes-per-token go?

omp has nine model roles plus a few model-kind roles, which makes this
concrete. Here's the split after the audit:

| Role | What it does | Token volume | Runs on |
|---|---|---|---|
| `default` | every turn, every file read, all tool results | ~70-80% of session tokens | brain, local |
| `task` / `smol` | subagent fan-out, summaries | bursty, huge | 5090, local |
| `tiny` / `memory` / `commit` | titles, memory extraction, commit messages | high-frequency, small | 5090, local |
| `judge` | internal agent judgments, **every turn** | tiny but constant | senses node, local |
| `web` | all searches | frequent | senses node, local |
| `plan` / `advisor` | planning, review, oversight | rare, critical | GLM 5.3, cloud |
| `slow` | deep reasoning | rare, heavy | DeepSeek, cloud |
| `vision` | screenshots, diagrams | rare | glm-5.3-flash, cloud |

The single most expensive thing an agent does — a 400K-token prefill on the
default loop — happens on hardware I already paid for. Cloud spend scales with
*decisions*, not with *context*. That asymmetry is the entire game.

Two timeouts make the local-first part practical, and both were learned the
hard way:

```yaml
providers:
  streamFirstEventTimeoutSeconds: 600   # local prefill of 400k+ prompts
  streamIdleTimeoutSeconds: 120         # long reasoning streams between tokens
```

If you've ever watched a local vLLM chew through a giant prefill (I wrote
about a related prefill footgun in [the zoysh
port](/posts/zoysh-yosh-zsh-port/)), you know the default first-event
timeouts in agent harnesses assume cloud latencies. They don't survive
contact with a 512K-context local box.

## Bug 1: the dead promotion target

omp has a lovely feature called context promotion: when a turn overflows the
context window, instead of compacting, it can switch to a designated
larger-context model and retry. My deep model had:

```yaml
contextPromotionTarget: "anthropic/claude-opus-5.5"
```

Sounds great. One problem: **there is no Anthropic credential on this
machine**. The promotion target is availability-checked at overflow time, and
a target that can't resolve a credential is silently skipped — every overflow
went straight to compaction instead. A one-line config value, dead on
arrival, failing quietly forever.

The fix was to point it at something I actually have a key for, with a bigger
window:

```yaml
contextPromotionTarget: "deepseek/deepseek-v4-pro"   # 1M ctx, key present
```

Lesson: any "fall back to X on failure" config is a liability until you've
watched it fail at least once. Dead references don't error — they no-op.

## Bug 2: the judge that was quietly burning my 5090

This is my favorite find of the audit. omp makes small typed decisions about
its own state — classifying a prompt to pick a thinking-effort level,
detecting when a stream stopped for unexpected reasons, AI-staging files in
the git UI. These run through a `judge` role with a native chain that starts
at TypeSafe's System One models and degrades through progressively cheaper
options.

I have no TypeSafe key. So every one of those judgments — on every turn — was
falling through to the `tiny` role: **a prompted 27B LLM generating tokens to
answer yes/no questions**, uncalibrated, on the same GPU that runs my
subagent fan-out. The config looked innocent. The GPU bill was real.

The fix is a project I'd been waiting for an excuse to deploy:
[Ollaya](https://ollaya.dev/), a local server for *decision models* —
single-forward-pass models that return calibrated probabilities instead of
generated prose. And the reason it drops straight into omp is a small
miracle of protocol archaeology: omp's `models.yml` supports `api: typesafe`
as a custom provider API — the exact wire format (`/v1/systemone`) Ollaya
speaks.

```yaml
# models.yml
ollaya:
  baseUrl: http://senses-node:11435
  api: typesafe          # System One judgment wire -> the judge role
  auth: none
  models:
    - id: winnow:e4b
      name: Winnow E4B (local decision model)
      contextWindow: 8192
      maxTokens: 1024
```

```yaml
# config.yml
modelRoles:
  judge: ollaya/winnow:e4b
```

Live test, from the same request shape omp sends:

```json
{"root_cause": {"choice": "db_outage", "confidence": 0.957,
                "probabilities": {"db_outage": 0.97, "app_bug": 0.027, "config": 0.002}},
 "is_urgent":  {"choice": "yes", "confidence": 0.973}}
```

0.93 seconds warm — on **CPU**, because the 4080's VRAM belongs to the
embeddings and reranker services (on GPU it's ~90 ms). Even on CPU it beats
prompted-27B judging on latency, returns actual calibrated probabilities, and
costs the 5090 exactly nothing. The judge role now runs on a box whose job is
already "the senses."

## The V4.1 naming trap

"Does the DeepSeek API offer V4.1?" Yes. No — yes, but hear me out.

Every catalog names it differently. omp's bundled catalog showed
`deepseek-v4-pro` and `deepseek-flash` with no v4.1 anywhere. OpenRouter had
a literal `deepseek-v4.1-flash`. HuggingFace had `DeepSeek-V4.1-Flash`. So I
stopped trusting catalogs and asked the endpoint:

```
$ curl https://api.deepseek.com/v1/models -H "Authorization: Bearer $KEY"
{"data":[
  {"id":"deepseek-flash","name":"DeepSeek-V4.1-Flash", ...},
  {"id":"deepseek-v4-pro","name":"DeepSeek-V4-Pro", ...}]}
```

Ground truth: **there is no literal v4.1 model id on the DeepSeek API.**
`deepseek-flash` *is* V4.1-Flash — 1M context, vision-capable, rolling
alias. And the kicker from their release notes: since September 14, *all
`deepseek-v4-pro` requests are server-side routed to V4.1-Flash* at Flash
prices, "until V4.1-Pro launches."

Which means my `slow` role — set to `deepseek-v4-pro:high` — is already
serving V4.1-Flash, and when V4.1-Pro ships it will auto-upgrade to the real
flagship without me touching anything. Sometimes the lazy selector is the
correct one.

The transferable lesson is older than LLMs: **`GET /v1/models` on the live
endpoint beats every catalog.** Catalogs drift; endpoints don't lie about
what they serve.

## Vision: the flash/base trap

One more verified-fact detour, because it's a trap wearing a friendly name.
GLM-5.3 — the big flagship — is **text-only**. GLM-5.3-*Flash* is the
multimodal one: native vision encoder, 1M context, image/video/file input.
If you skim model names the way I do, "use the bigger model for vision" is
exactly the mistake you'd make.

And the local option? Doesn't exist. My 27B workhorse is not a VLM, and I
now have the receipt — a live image request returning:

```
HTTP 400: "At most 0 image(s) may be provided in one prompt."
```

So `vision` lives on `zai/glm-5.3-flash:high`, which is genuinely good at it
(screenshot-to-app understanding is apparently a design goal) and comes with
3× quota on the coding plan. Test your assumptions with actual requests; the
config comment that says "not a VLM" might be stale, and the one that says
nothing might be wrong too.

## The lattice

Fallback chains are where local-first gets to be clever, because the chain
can cross the local/cloud boundary in both directions:

```yaml
retry:
  fallbackChains:
    deep/*:      [5090-27b, zai/glm-5.3]          # local -> local -> cloud
    fast/*:      [deep/flash-next, zai/glm-5.3]   # local -> local -> cloud
    zai/*:       [deepseek/deepseek-v4-pro, deep] # cloud -> other cloud -> local
    deepseek/*:  [deep/flash-next, zai/glm-5.3]   # cloud -> local -> other cloud
```

A cloud outage degrades to my hardware. A hardware reboot degrades to the
other cloud. No single failure takes the agent down.

One honest footnote: the `zai/*` chain technically serves `vision` too, and
its fallbacks are text-only models. If z.ai 429s mid-screenshot-analysis, the
fallback lands somewhere that can't see the image. It only matters in that
narrow intersection and I've left it — but a vision-scoped chain ending in
`deepseek-flash` (which has vision) is the correct fix if it ever bites.

## What's next

The audit left a queue: point omp's memory backend at mnemopi with the local
embeddings server doing the vectors; offload the `tiny` roles to omp's
embedded tiny models and raise subagent concurrency; wire the senses node's
jury and dispatch MCPs into the agent; and try the [Context Language
Models](https://github.com/facebookresearch/context-language-models)
extension — context-as-file self-management, which their benchmarks claim
cuts long-agent compute substantially (mind the CC BY-NC license).

## Verdict

Is the config using local AI in a meaningful way? After the audit, honestly:
yes. Local carries the daily driving, all fan-out, all background machinery,
judging, and search; cloud is reserved for exactly the four places quality
justifies it. The bugs weren't in the philosophy — they were in the
verification. Dead promotion targets, silent judge degradation, and phantom
model names all share one cure:

Don't read the config. **Query the endpoints.**
