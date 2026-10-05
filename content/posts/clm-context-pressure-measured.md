---
title: "Attention Fails Before the Guard Does: Measuring Context Management at 700K Tokens"
date: 2026-10-05T15:30:00+02:00
draft: false
description: "Everyone has opinions about long-context agents; nobody had measurements of what the machinery actually does under sustained pressure on a local fleet. I built a harness that drives a real coding-agent session past a 524K-token model window, unattended, and measured five things: where promotion fires, what compaction destroys, when verbatim recall collapses, why the truncation guard I was testing never fired, and how the whole story reorders my assumptions about agent memory hygiene."
summary: "fleet-brain's CLM harness drove real omp sessions past the local 524K window: promotion fires at 84% of the window and rides the prompt cache; with promotion disabled, a snapcompact compaction cut 445K tokens to 45K while preserving gist and destroying every verbatim canary; the headline surprise — canary recall collapsed between 389K and 443K with no context-management event at all, meaning attention had already failed exactly where the managed boundary sits. Plus: the overflow-withhold mechanism I set out to observe is unreachable on the tool surface, because every tool result is capped upstream."
tags:
  - ai-agents
  - context-management
  - coding-agents
  - local-llm
  - measurement
  - omp
categories:
  - engineering
keywords:
  - context window
  - context promotion
  - compaction
  - lost in the middle
  - long context
---

# Attention fails before the guard does

*Five measurements from driving a real coding-agent session past a
half-million-token window, and what each one changes about how I run agents.*

Every claim I've made about context management — promotion, compaction,
"persist before pruning" — has been theory: it composes on paper, nobody has
watched it happen under genuine load. My fleet serves a local model with a
524,288-token window, my agent harness (omp) has context machinery I've never
seen fire, and my memory policy asks agents to write durable notes *before*
material leaves context — but when does material actually leave?

So fleet-brain, the cross-repo index project I wrote about before, grew a
second purpose: a harness that drives a real `omp --mode rpc` session past
the boundary with no manual rescue, logging every turn's context size, every
promotional jump, every compaction, and every durable-memory write. Each turn
makes the agent read a large real doc in full and analyze it — honest load,
not lorem ipsum. Every workload turn also injects a unique canary token
(`CANARY-014-ZEPHYR82`), and every few turns a probe asks the agent, tools
locked: *which canaries can you still literally see? Which file did turn one
cover?* Gist versus verbatim, measured.

Five results. Each one changed something I do.

## 1. Promotion fires early and rides the cache

Session one, defaults on. Context grew 11K → 442K over 24 turns; at
**445,432 tokens — 85% of the window** — the harness switched the session to
the cloud promotion target and kept going: **546,631 tokens**, 140% of the
local window, no error, no rescue, no compaction. The crossing turns show
457K–539K of the prompt arriving as `cacheRead`: the cloud tier didn't
re-prefill my history, it picked up a cached context and kept growing it.
Turn latency after the jump: 24–40 seconds.

The design point: promotion fires **before** the wall, not at it. The switch
cost is paid while there is still slack, so an over-budget request can never
fail. I had assumed the opposite — that machinery triggers on overflow. It
triggers on *approach*.

## 2. Compaction destroys verbatim, preserves gist

Second session, promotion disabled via config overlay, so something *had* to
give. Nothing crashed and — surprise #1 — no token was ever withheld.
Instead, at **445,681 tokens (again 85%)**, the harness ran a `snapcompact`:
76,459 characters of history archived onto five summary frames. Presented
context: **445,681 → 44,979. A 90% cut.** Session continued, epoch bumped,
context grew back, and at the next 85% it happened again.

The canaries say what that 90% *lost*:

| generation | oldest-4 canaries | newest canaries | turn-1 file path |
|---|---|---|---|
| before any compaction | present | present | recalled |
| after one compaction | **LOST, every probe** | present | recalled |

Verbatim strings do not survive summarization. High-entropy tokens — IDs,
secrets, exact numbers, the literal content of a diff — die first, every
time. The *structure* survives: which files were read, what they cover, which
decision followed which. Compaction is a gist-preservation machine.

Which is exactly why "persist before pruning" is the right policy: durable
memory entries have to be written while the verbatim detail is still in
context, because after compaction the agent still knows *what happened* but
can no longer *quote it*. My memory-hygiene rule now has a mechanism behind
it.

## 3. The headline: recall collapses before any machinery fires

Third session, promotion back **on**, pushed further — toward the cloud
model's own 1M window. One promotion, at 84% of the local window; the target
re-tokenized the intact history (445K → 426K tokens, different tokenizer, no
summarization) and kept growing to **732,088** — with **zero compactions and
zero further model changes**. No thrash: the promotion target is terminal.
A follow-up run closed the last open question: driven deeper, the cloud model
grew to **848,598 and ran its own snapcompact there** — so the relief
threshold is proportional to whatever window you are *currently* in, and the
85% promotion point proved deterministic across runs to within ~10 tokens.
The canary curve below reproduced on the second run almost exactly (7/8 at
364K → 3/8 at 453K → flat to 805K, gist perfect). The cliff is not noise.

And here is the result I did not predict. Look at the canary curve —
*nothing has happened* the entire time. No compaction, no promotion after
turn 22, every token still formally in context:

| probe | context size | oldest canaries survived |
|---|---|---|
| 8 | 237K | yes (7/7) |
| 16 | 389K | yes (8/8) |
| 24 | 443K | **no (4/8 — oldest four LOST)** |
| 32–48 | 517K → 664K | never recovered, even on the cloud model |

Between 389K and 443K — **74% to 85% of the window** — the agent stopped
being able to see the oldest verbatim content, while gist recall stayed
perfect. Classic lost-in-the-middle, measured on a real local fleet under
load, before any context-management mechanism touched anything. And both
managed mechanisms fire at 84–85%.

Let that land: **the managed boundary is scheduled around already-failed
attention.** Compaction at 85% is not destroying information the model could
still use; it is formally filing away what the model had already stopped
reading. The working context that matters — the region where the model can
actually attend verbatim — is meaningfully smaller than the context window
that gets advertised, and the gap is exactly where all the clever machinery
claims to operate.

Practical rule now pinned in my steering docs: the deadline for durable
writes is **~75% of window**, set by attention — not the first pruning event.
Anything older than the last quarter of the context is already lost for
verbatim purposes; it just hasn't been deleted yet.

## 4. The guard I came to measure is unreachable

The README theory I set out to verify: "the overflow guard withholds the
oldest observations." After four separate attempts, **withhold events: zero**.
Why:

- A 6.4MB (~1.6M-token) file demanded in one read call: the read tool
  truncates at ~300 lines — ~23K tokens into context. The agent noticed,
  fetched the tail, quoted my needle verbatim.
- A raw bash job printing ~1M tokens, run bare: the tool result capped at
  ~765 bytes, the rest spilled to a disk artifact. The agent's own summary:
  *"the full output never entered my context; retrieval can only ever be
  sampled fragments, never the blob inline."*

Every tool surface is capped *upstream* of the window. A single result cannot
overflow a context that no tool is allowed to fill in one shot. The
overflow-withhold path exists in the code and is unreachable by any workload
ordinary tools can produce. On this stack, context pressure is cumulative or
it is nothing — and cumulative pressure is promotion and snapcompact, both
measured above.

## 5. Defense in depth, ordered by when it acts

The window, top-down, with measured trigger points:

1. **Tool-result caps** — per call, orders of magnitude below the window
   (lines/bytes, spill to artifacts).
2. **Attention's own cliff** — unmanaged, ~75–85% of window, verbatim recall
   of oldest content collapses while gist survives.
3. **snapcompact** — 85% of current window, gist kept, verbatim discarded,
   epoch bumped, run continues.
4. **promotion** — 84% of window when a larger model exists; one hop,
   terminal, cache-carried.

None of these save your facts. Only the durable write does — and by the time
any *visible* machinery fires, the verbatim content it will lose was already
unavailable to the model. Measure your own stack if you run one of these; the
harness that produced all of this is ~400 lines of Python that just drives
`omp --mode rpc` and reads the session JSONL. The surprising results were all
things I was sure I already knew.

---

*Method notes: sessions driven by fleet-brain's `clm run` (persistent
`omp --mode rpc` process, one session, 28–55 workload prompts, cairnkeep docs
as load); canary/probe scoring via `clm canaries`; promotion disabled with a
session-scoped config overlay (`contextPromotion.enabled: false`, empty retry
chains so no silent model hop could mask the mechanism). Hostnames and IPs
sanitized; model identifiers kept.*
