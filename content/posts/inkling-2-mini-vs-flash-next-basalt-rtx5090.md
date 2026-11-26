---
title: "Inkling 2 Mini vs Qwen3.8-Flash-Next on One RTX 5090: Close on the Tests, Faster, and 54 GiB of RAM Free"
date: 2026-11-25T19:20:57+0100
draft: false
description: "Inkling 2 Mini against my Flash-Next IQ4_XS v3.1 deployment on one RTX 5090 with 96 GB of RAM: the same 16 core cases run three times, a 32-problem HumanEval subset, GSM8K, a 256K recall check, speed, token cost, and the temperature trap that changes the scores."
summary: "Thinking Machines Lab's 26.4B MoE looked strong against the dense Qwen3.8-27B, but my RTX 5090 runs Flash-Next. I ran Inkling 2 Mini through the same tests as my last post, three times at its publisher's recommended temperature. On average it scored within about two cases of Flash-Next on the core cases and within one on HumanEval, 194 against 195 on GSM8K. It decoded 36% faster, read prompts more than twice as fast, and gave back 54 GiB of RAM. At temperature 0 it stalls on the hard cases, and it cannot answer without reasoning. An update of November 26 puts both models on one engine. With thinking on, the two score the same on the core cases. Inkling 2 Mini takes a quarter fewer tokens in 40 % of the time, so it is my default now."
tags:
  - llm-inference
  - inkling-2-mini
  - thinking-machines
  - qwen
  - qwen3.8-flash-next
  - basalt
  - llama-cpp
  - rtx-5090
  - blackwell
  - long-context
  - self-hosting
  - benchmark
images:
  - "/inkling2-flashnext-benchmarks.png"
categories:
  - engineering
  - infrastructure
keywords:
  - Inkling 2 Mini
  - Qwen3.8-Flash-Next
  - Basalt
  - RTX 5090
  - NVFP4
  - UD-Q8_K_XL
  - HumanEval
  - GSM8K
  - 256K context
---

Basalt added Inkling 2 Mini on November 21. The model comes from Thinking
Machines Lab, with 26.4 billion parameters and 5.4 billion of them active per
token. It ships under Apache 2.0, reads images, and takes a one-million-token
context.

The charts I found put it next to Qwen3.8-27B, a dense model. It leads that
one on DeepSWE and on the hard split of SWE-bench Pro, as Unsloth's
[guide](https://unsloth.ai/docs/models/inkling-2-mini) shows. My RTX 5090 runs
a different Qwen, [Flash-Next IQ4_XS
v3.1](/posts/qwen38-flash-next-strata-rtx5090-quant-comparison/). I found no
comparison of the two, so I ran one.

My question was narrow. Is Inkling 2 Mini as good as Flash-Next on the tests I
already trust? And what does it cost in speed, tokens and memory?

## What I compared, and what stayed different

Both models ran on one RTX 5090 with a **450 W power limit**, a 32 GB card in
a machine with 96 GB of RAM. Both quality runs used a 262,144-token context. I
loaded one model at a time.

The quality tests are the ones from my last post. The core suite has 16 cases:
Python code, structured output, tool arguments and small repairs with tools.
Each case ran with thinking off and again with thinking on. Then came a fixed
32-problem subset of HumanEval and six focused reruns. I added the first 200
problems of GSM8K and a three-marker recall check near 256K tokens.

Five things stayed different. The numbers belong with them:

- Weights. Flash-Next is Unsloth's IQ4_XS v3.1 in a 91 GiB pack that keeps most
  of its experts in system RAM. Inkling is Basalt's default 19.6 GB pack,
  which holds the release's expert weights as they are.
- Drafting. Flash-Next uses MTP drafts and Inkling uses its DSpark drafter. Both
  were on.
- Sampling. Each model ran with the settings its publisher recommends.
  Flash-Next got temperature 0 on the core cases and HumanEval, and
  temperature 1.0 with `xhigh` reasoning on the focused reruns. Inkling got
  temperature 1.0 on everything but GSM8K.
- Engine and day behind the older numbers. Flash-Next's quality scores and its
  RAM figure come from its run of November 5 on Strata, with the same weights
  it runs today. I measured its speed and its GSM8K score on Basalt.
- Server for the speed figures. Both ran on Basalt 0.5.7, one request at a time.
  Inkling ran on a fresh server with one slot and a 262,144-token context.
  Flash-Next ran on its usual configuration: four slots that share a
  one-million-token context, and an int8 cache.

## Thinking off is not off, and temperature 0 stalls it

Inkling has no switch for thinking. Basalt maps `enable_thinking=false` to the
effort level `none`. At that level the model still writes a short reasoning
block before it answers. In this post, "thinking off" means that level.

Temperature 0 was the bigger surprise. My test suite uses it for the core
cases and for HumanEval, because Flash-Next answers well that way. Inkling
does not. With a 16,384-token budget it reasons for 52,000 to 75,000
characters on the hard cases and never gets to the code. Of the eleven
core-case attempts that ran out of budget, across both budgets, seven repeat
one sentence roughly 60 to 340 times.

| Inkling 2 Mini at temperature 0 | Score | What failed |
|---|---:|---|
| Thinking off, budget 4,096 tokens | 12/16 | 4 cases ran out of budget before any code |
| Thinking off, budget 16,384 tokens | 13/16 | 3 cases ran out of budget |
| Thinking on, budget 16,384 tokens | 11/16 | 4 cases ran out of budget, 1 tool chain was wrong |
| HumanEval subset, budget 8,192 tokens | 29/32 | 3 problems ran out of budget |

All but one of those misses are answers that never arrived. The exception is a
tool chain in which Inkling called the same lookup twice. The stall shows on
hard code cases. It does not show on GSM8K, where Inkling scored 194 of 200
with greedy decoding.

Unsloth recommends temperature 1.0, `top_p` 1.0, no top-k and `min_p` 0 for
this model. Inkling's suite scores from here on use those settings. A
benchmark that runs this model greedy can underrate it.

An update of November 26 qualifies the stall above: with an fp16 cache it
halves.

## One sample is not a score

At temperature 1.0 every answer is a draw. I ran the suite three times with
the recommended settings: once with an int8 KV cache and twice with an fp16
cache, on two seeds.

| Inkling 2 Mini, three runs | Run 1 | Run 2 | Run 3 | Passes over all runs |
|---|---:|---:|---:|---:|
| Core cases, thinking off | 16/16 | 14/16 | 13/16 | 43 of 48 |
| Core cases, thinking on | 15/16 | 16/16 | 15/16 | 46 of 48 |
| HumanEval subset | 31/32 | 31/32 | 30/32 | 92 of 96 |

Two runs differ by up to three cases on the same tests. One run cannot rank
this model, so I count passes over all runs and give ranges. Flash-Next
answers at temperature 0, so one run is the whole story for it. An update of
November 26 corrects this for Flash-Next on Basalt.

| Measurement | Flash-Next IQ4_XS v3.1 | Inkling 2 Mini |
|---|---:|---:|
| Core cases, thinking off | 16/16 | 43 of 48 case-runs |
| Core cases, thinking on | 16/16 | 46 of 48 case-runs |
| HumanEval subset | 31/32 | 92 of 96 case-runs |
| Six focused reruns, three runs | 6/6 | 5/6, 5/6, 6/6 |
| GSM8K, first 200 problems | 195/200 | 194/200 |
| Recall near 256K tokens | Pass | Pass |

The focused reruns ran three times: once before these runs, and then in runs
1 and 2. The first two used the same seed and the same int8 cache.

Five problems account for all 13 of Inkling's misses in these runs:

- `merge_patch` failed in three of six attempts, with the same mistake each
  time. The code looks right. It copies a nested object from the patch when
  the target has no such key, where RFC 7396 says to apply it.
- `async_ordered_map` passed five of nine attempts. Twice it reasoned for all
  16,384 tokens without answering. Twice it answered and used
  `asyncio.Semaphore` without importing `asyncio`. Its five passes took 5,700
  to 12,800 tokens.
- `literal_tool_payload` failed twice with thinking off. Both times it wrote the
  tool call as plain text, and no call reached the test.
- HumanEval problem 116 never passed. Twice it produced no code within 8,192
  tokens. Once it produced code that failed its test.
- HumanEval problem 32 failed once, with no code within 8,192 tokens.

GSM8K ran the same way for both models: greedy, with thinking on, one request
at a time. It has no paired comparison. The per-problem file from my earlier
Flash-Next run was in a temporary directory. It no longer exists. On totals
the two are one problem apart out of 200.

## Speed and memory

Both models ran the same speed probes on Basalt 0.5.7, one request at a time.
The last row comes from the 256K check:

| Measurement | Flash-Next | Inkling 2 Mini |
|---|---:|---:|
| Generation, prompt under 100 tokens | 239.1 tok/s | **324.8 tok/s** |
| Time to first token, same request | 0.248 s | 0.055 s |
| Prompt read, same text (4,115 and 2,716 tokens) | 4,055 tok/s | 16,855 tok/s |
| Prompt read, same text (16,285 and 10,505 tokens) | 7,560 tok/s | 16,929 tok/s |
| `basalt-bench`: conversation, structured, prose | 258, 495, 261 tok/s | 360, 642, 288 tok/s |
| `basalt-bench`: prompt read at 16K, 64K and 131K tokens | 8,181, 8,638, 8,368 tok/s | 16,286, 12,918, 9,389 tok/s |
| RAM free during the 256K check, minimum | 15.9 GiB | **70.2 GiB** |

![Measured generation speed, prompt read speed and free RAM for the two models](/inkling2-flashnext-benchmarks.png)

The last row is not from Basalt. Flash-Next's 256K check ran on Strata on
November 5. Inkling's figure is from the run with the fp16 cache. Its two
recorded runs of the same check with an int8 cache had 70.8 and 71.1 GiB free.

I ran the speed probes four times on Inkling. Each round takes the median of
three 512-token runs, and the four rounds gave 316 to 329 tok/s. The table
shows the median of the four rounds. Flash-Next's figure is the median of
three runs in one round. The probes ran at temperature 0.6 for both models.

The two tokenizers count the same text differently, so the prompt-read rows
give both token counts. Inkling read the long text in 0.62 s against 2.15 s.
It read the short one in 0.16 s against 1.01 s.

The `basalt-bench` rows for Inkling are the mean of two repetitions from
today. Flash-Next's are from my trial of November 24. These are short-prompt
numbers, as in my last post. They say nothing about generation at a full
context.

The RAM figure is why I care. With Flash-Next loaded, the server holds about
74 GB of RAM, most of it experts that do not fit on the card. With Inkling
loaded it holds about 1 GB. On a machine that also runs a browser, a compiler
and a coding agent, that is the difference between 16 GiB free and 70.

## The cost in tokens

Flash-Next ran the core cases with a 4,096-token budget and Inkling with
16,384.

| Over the 16 core cases | Flash-Next | Inkling 2 Mini |
|---|---:|---:|
| Completion tokens, thinking off | 5,851 | 36,000 to 41,000 |
| Completion tokens, thinking on | 26,502 | 52,000 to 63,000 |
| Seconds, thinking off | 53 | 102 to 117 |
| Seconds, thinking on | 185 | 149 to 171 |
| Seconds, six focused reruns | 162 | 59 to 65 |

Inkling cannot answer without reasoning. Where Flash-Next replies at once,
Inkling spends six to seven times the tokens and takes about twice the time.

With thinking on it spends about twice the tokens for each solved case, a
median of 3,106 against 1,656. It also finishes those cases sooner: 8.2 s
against 11.6 s. That count covers only the cases it solved. The failed
attempts add their whole budget, so the totals in the table are closer.

## A recall check that needed a bigger reply budget

The recall check asks for three markers from a synthetic prompt of about
250,000 tokens. The test script lets the model answer in 256 tokens.
Flash-Next
used 46 of them and passed.

Inkling failed at 256 tokens, twice. In the one I read, it had found all
three markers and listed them in its reasoning, and it spent the 256 tokens
there. No answer followed. With a 2,048-token reply budget it passed twice,
at 249,972 prompt tokens. With the fp16 cache it answered in 352 tokens, and
the first token came after 34.2 s. Flash-Next's first token came after 40.8 s
at 256,553 tokens.

That is the same smoke test as in my last post: three recovered strings, and
no claim about coding quality at that depth.

## How close the two packs are to full precision

No BF16 checkpoint of Inkling 2 Mini exists. Thinking Machines Lab released
the model with its experts in NVFP4 and MXFP8. Basalt's default pack keeps
those expert weights bit for bit. It stores the attention, gate, convolution
and output projections as 8-bit copies of the BF16 values. Basalt's notes say
that BF16 attention there scored closer to the reference, at 1.2 GB more reads
per token. Basalt did not take that option.

So two things in my runs were below the best a file can hold. The KV cache was
int8 in the early runs and fp16 in the later ones. The projections were 8-bit
throughout. Unsloth calls its UD-Q8_K_XL file full precision: it keeps the
release's NVFP4 expert weights and stores the rest in BF16. I tested the
effect of both.

The cache format changed which tokens were drawn, and nothing else that I can
see. With thinking off, the two fp16 runs scored 14 and 13 against 16 with
int8. With thinking on they scored 16 and 15 against 15. Their misses fall on
cases that fail in other runs too. I read the difference as sampling noise.
Generation on a short prompt was 328.6 tok/s with fp16 and 325.7 tok/s with
int8, in two runs that were otherwise the same.

For the weights, I ran UD-Q8_K_XL in Unsloth's llama.cpp build, on the same
tests and at the same temperature:

| Test | Basalt, default pack | llama.cpp, UD-Q8_K_XL, one run |
|---|---:|---:|
| HumanEval subset | 30 to 31 of 32, three runs | 32 of 32 |
| Six focused reruns | 5 to 6 of 6, three runs | 4 of 6 |
| `async_ordered_map`, thinking off and on, budget 16,384 | 4 of 6 passes, three runs | 2 of 2 passes |
| Generation, prompt under 100 tokens | 325 tok/s | 114 tok/s |

The Q8_K_XL file did not remove the failure mode. In llama.cpp, two of the six
focused cases also ran out of their 8,192-token budget without an answer. One
of them was `async_ordered_map` again. The scores overlap with Basalt's. One
run at temperature 1.0 cannot rank two files, and I would not trade two thirds
of the speed for that table.

llama.cpp ran without the DSpark drafter. Unsloth reports 10 to 20 percent
more with it on a B200. I did not test that.

Basalt's default pack is good enough for this model. The fp16 cache costs
nothing I can measure on a short prompt, so that is the setting I keep.

## Update of November 26: the same tests on Basalt, LoopCD, and Flash-Next

A reader pointed me to LoopCD, a decoding method for looped transformers
([arXiv 2610.02185](https://arxiv.org/abs/2610.02185)). The model runs its
core three times for each token. At each step, LoopCD moves the logits away
from what the model predicted after the first of those passes. It needs no
training. Basalt does not have it, so I tested it with capy-flash 0.16.5, the
engine that implements it, on the same card with the same tests.

That test led to more runs on Basalt. They change three statements in this
post, and they decided which model I run.

### The stall at temperature 0 follows the cache format

The capy-flash run stalls less at temperature 0 than my Basalt runs above,
with LoopCD off: 57 of 64 attempts pass against 53, and 5 run out of budget
against 10. So I looked for the cause on Basalt. I ran the same greedy tests
with the drafter on and off, and with an fp16 and an int8 cache.

The cache format makes the difference. With the int8 cache, 53 of 64 attempts
pass and 10 run out of budget. With fp16, 57 pass and 5 run out. The drafter
changes nothing: every attempt that finished in two runs has the same output.
The runs above used the int8 cache.

| Greedy, temperature 0, 64 attempts | Passes | Ran out of budget | Decode |
|---|---:|---:|---:|
| Basalt 0.5.7, int8 cache | 53 | 10 | 321 tok/s |
| Basalt 0.5.7, fp16 cache | 57 | 5 | 330 tok/s |
| capy-flash 0.16.5, LoopCD off | 57 | 5 | 151 tok/s |
| capy-flash 0.16.5, LoopCD at 0.5 | 60 | 1 | 132 tok/s |
| capy-flash 0.16.5, LoopCD at 0.3 | 61 | 1 | 132 tok/s |

The 64 attempts are the 16 core cases in both modes and the 32 HumanEval
problems. Decode is with the drafter on. Without it Basalt runs at 253 and 260
tok/s. With fp16, 6 attempts pass that fail with int8, and 2 pass with int8
that run out with fp16. A greedy path changes with the numerics, so one miss
means little.

### LoopCD on the same tests

On capy-flash, LoopCD made the same three core cases stop looping at both
strengths, and HumanEval problem 116 passed. One case got worse at strength
0.5 and none at 0.3. Decode fell by 13 %. The capy-flash docs give 11.5 to
16.5 %.

At the publisher's settings LoopCD made no difference to the score. Over three
runs at temperature 1.0, 88 of 96 case-runs passed without it and 89 with it.
It used 14.5 % fewer completion tokens with thinking off and 23.2 % fewer with
thinking on.

This is a small suite, and greedy decoding gives one path per case. Read the
gain as a direction. In Basalt's issue 685 a contributor tested a LoopCD
variant of their own on the same pack. On the full HumanEval and on MMLU it
gave no gain.

capy-flash also decoded at 151 tok/s on this card, against Basalt's 330. It is
text only and serves one request at a time. So Basalt stays my engine for
Inkling 2 Mini, with an fp16 cache.

### Flash-Next on Basalt

Flash-Next's scores and token counts above come from Strata. I ran the same 16
core cases on Basalt, on the service I use, which keeps most of the model's
experts in RAM.

In greedy mode with thinking off, two passes gave 15 of 16.
`batch_transactions` failed both times: the code checks a duplicate row before
it ignores it, and the task says to ignore it first.

With thinking on the passes gave 15 of 16 and 16 of 16. The same case ran out
of budget in the first pass. Five other code cases took 11,900 to 12,700
tokens each, against 3,300 to 3,450 on Strata. The reasoning lists possible
hidden tests one by one. One pass took 84,000 tokens and 372 s, against 26,502
and 185 s above.

Two passes on the same server also gave different output. With thinking off 9
of 16 answers were the same, with thinking on 2 of 16. Inkling 2 Mini gave the
same output in 54 of 54 attempts on a freshly started server. I did not
isolate the cause. The likeliest candidates are the service's warm caches. In
these requests a median of 18 % of its expert lookups missed the cache on the
card.

Then I ran each model on Basalt at the settings its publisher recommends,
three times. Flash-Next got `xhigh` reasoning with thinking on. The HumanEval
subset ran without thinking for both.

| Both on Basalt, publisher's settings, three runs | Inkling 2 Mini | Flash-Next |
|---|---:|---:|
| Core cases, thinking off | 43 of 48 case-runs | 46 of 48 case-runs |
| Core cases, thinking on | 46 of 48 case-runs | 46 of 48 case-runs |
| HumanEval subset, no thinking | 92 of 96 case-runs | 92 of 96 case-runs |
| Six focused reruns | 16 of 18 | 4 of 6, one run |
| Completion tokens, thinking off | 116,342 | 16,118 |
| Completion tokens, thinking on | 171,693 | 232,865 |
| Seconds, thinking off | 334 | 110 |
| Seconds, thinking on | 478 | 1,208 |

On one engine the two are close in score. With thinking off, Flash-Next passed
3 more case-runs and used a seventh of the tokens. With thinking on, both
passed 46 of 48. Inkling 2 Mini used a quarter fewer tokens and 40 % of the
time. Flash-Next missed `batch_transactions` twice with thinking off and
`sse_parser` twice with thinking on. In the focused reruns it failed both,
where Strata passed all six.

### What changes in this post

- Inkling 2 Mini's stall at temperature 0 follows the int8 cache as much as
  the model. With fp16 it halves.
- On Basalt, Flash-Next at temperature 0 did not answer every core case, and a
  second pass did not give the same answer. Its token cost with thinking on is
  higher than the table above says. It is about 83,000 per run at temperature
  0 and 78,000 at its publisher's settings, against 26,502 on Strata.
- On one engine, at the publisher's settings, the two models are close in
  score. Inkling 2 Mini is the cheaper one with thinking on. Flash-Next is far
  cheaper with thinking off.
- Inkling 2 Mini is my default now. My agents run with thinking on, and with
  thinking on the two score the same on the core cases.

The numbers are in the bundle: the
[LoopCD](/benchmarks/inkling2-flashnext-basalt-20261125/loopcd-2026-11-26.json),
[cache
format](/benchmarks/inkling2-flashnext-basalt-20261125/drafter-kv-2026-11-26.json),
[Flash-Next
greedy](/benchmarks/inkling2-flashnext-basalt-20261125/flashnext-basalt-greedy-2026-11-26.json),
[both
models](/benchmarks/inkling2-flashnext-basalt-20261125/both-models-basalt-publisher-2026-11-26.json)
and [expert
cache](/benchmarks/inkling2-flashnext-basalt-20261125/flashnext-expert-cache-2026-11-26.json)
measurement files.

## What runs on the machine now

I now start Inkling 2 Mini by default, with an fp16 cache. It matches Flash-Next
with thinking on at a quarter fewer tokens, and it leaves 54 GiB of RAM free.
During the 256K check, 70.2 GiB of RAM were free with Inkling 2 Mini. With
Flash-Next, the figure was 15.9 GiB. My agents run with thinking on, so that is
the comparison that counts.

This is the command, with the publisher's sampling as the server's defaults:

```bash
basalt serve Inkling-2-Mini-NVFP4-MX8.basalt \
  --kv fp16 --max-context 262144 --parallel 1 --vision dynamic \
  --temperature 1.0 --top-p 1.0 --top-k 0 --min-p 0
```

A request that leaves temperature, `top_p`, `top_k` and `min_p` out gets those
values. It can still send its own. One that sends temperature 0 meets the stall
described above. The bare command listens on Basalt's default port and reports
the model as `Inkling 2 Mini`.

The fp16 cache halves the stalls at temperature 0. The slots share the context.
With one slot the cache holds all 262,144 tokens, and other requests wait. A
second slot would halve the context of each.

Basalt takes the whole cache up front. The card holds about 30 GB from the
start, desktop included, on a 32 GB card. The model loads 17 GiB, and 1.2 GiB
of that stay in RAM. Stop the server before anything else needs most of the
card.

Flash-Next stays installed, for a session by hand with this server stopped.

OpenCode has one local entry for it, `basalt/inkling-2-mini`. The model's
`options` carry the publisher's sampling: temperature 1.0, `top_p` 1.0,
`top_k` 0 and `min_p` 0.

## Limits

Sixteen local cases, 32 HumanEval problems, six focused reruns and 200 GSM8K
problems are a small suite. Public-test familiarity can affect the HumanEval
and GSM8K results.

Inkling's answers at temperature 1.0 are samples. I have three per case, over
two seeds. I ran Flash-Next once, at temperature 0, so its scores carry no
sampling noise and no repetition. Its quality numbers come from Strata with
the same weights, and the update of November 26 adds three runs of each model
on Basalt. The llama.cpp pass is one run, and its prompt-read numbers are not
comparable because of prefix caching. I compare GSM8K on totals only.

The 256K check recovers three markers and says nothing about coding at that
depth. I did not measure a sustained agent session on Inkling 2 Mini. The
short-prompt speeds do not describe generation at a full context.

The [measurement data and runnable test
bundle](/benchmarks/inkling2-flashnext-basalt-20261125/README.md) include the
task selection, the settings, every result in curated form and the limits of
the comparison. If a newer engine changes the answer, there is something
concrete to rerun.
