---
title: "AIOS: a Three-Node Self-Hosted Agentic Coding Stack with DeepSeek V4 Flash, vLLM, and a Custom Model Router"
date: 2026-08-21T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
description: "How I wired three GPU machines (2x RTX PRO 6000, RTX 5090, RTX 4080) into a single agentic coding stack: SGLang serving DeepSeek V4 Flash as the brain, a Qwen3.8 27B NVFP4 on the 5090 as the workhorse for subagents and vision, a custom chain-based router for failover and code exploration, and persistent agent memory across opencode, pi, and qwen CLI."
summary: "Instead of picking one model per task, I made three machines cooperate: a big thinking model for primary reasoning, a fast local 27B for subagent loops and images, a router with automatic failover in front of everything, and shared agent memory. Total cost: electricity. Here is the full architecture, the benchmarks, and the lessons learned the hard way."
tags:
  - llm
  - self-hosted
  - agentic-coding
  - vllm
  - sglang
  - deepseek
  - qwen
  - rtx-5090
  - rtx-pro-6000
  - lmcache
  - nvfp4
  - opencode
  - homelab
  - linux
  - systemd
categories:
  - engineering
  - infrastructure
keywords:
  - self-hosted LLM
  - agentic coding stack
  - DeepSeek V4 Flash
  - Qwen3.8 27B NVFP4
  - vLLM LMCache
  - SGLang DSpark
  - model router failover
  - opencode subagents
  - pi agent extensions
  - qwen cli configuration
  - RTX 5090 power limit
  - multi-node inference
cover:
  image: ""
  alt: "AIOS three-node agentic coding stack"
  relative: false
---

Most self-hosted LLM setups I see are one machine, one model, one OpenAI-compatible endpoint. That works, but it wastes the interesting part: if you have more than one GPU box, the models can *cooperate* instead of acting as each other's fallback. This post describes the stack I ended up with after several iterations: three machines with distinct roles, a custom router in front of everything, and agent CLIs configured so the right model is used for the right job automatically.

I call it AIOS. Everything below is running in production on my desk right now, autostarts at boot, and costs me electricity only.

## The hardware

| Node | Role | GPU | OS |
|:-----|:-----|:----|:---|
| `deep` | The brain | 2x RTX PRO 6000 (192 GB VRAM) | Fedora Silverblue |
| `fast` | The workhorse (also my daily desktop) | RTX 5090 32 GB | Fedora |
| `perception` | The senses: embeddings, audio, monitoring, routing | RTX 4080 | Debian |

The machines talk over my LAN, with a self-hosted Tailscale control plane (headscale, on my own VPS) as the overlay so everything also works when I am away from home. Zero SaaS in the coordination path: if the control server dies, existing tunnels keep working, and it is my server to fix.

## The layers

### deep: primary reasoning

The biggest model I can run lives here: **DeepSeek V4 Flash 0731**, served by **SGLang** in MXFP4 with DSpark, on port 8001. Thinking mode and max reasoning effort are injected by default via `--default-chat-template-kwargs`, so every request gets the full reasoning budget without the client having to ask for it.

This is the model that does the actual hard thinking: architecture decisions, tricky refactors, debugging sessions.

Measured on this exact box with llama-benchy:

| test | t/s | ttft (ms) |
|:-----|:----|:----------|
| pp2048 | 7735 | 341 |
| tg32 | 133 | |
| pp2048 @ 32k ctx | 6265 | 5634 |
| tg32 @ 32k ctx | 111 | |

Prompt processing north of 6k tok/s even at 32k context, generation steady around 110 to 133 tok/s. For a model of this class running on a desk, I have no complaints.

### fast: subagents, vision, and speed

My desktop runs **Qwen3.8 27B in NVFP4** under **vLLM**, with three things that make it punch above its weight:

- **LMCache** with an L2 tier on a dedicated NVMe: KV cache survives across sessions, so repeated work in the same repo gets near-instant prefill on cache hits.
- **MTP speculative decoding** for extra generation speed.
- **Multimodal enabled**: it accepts images, which matters more than I expected (see the vision section below).

An update since I first wrote this: the 27B on this node is no longer the stock Unsloth build. I swapped it for a **self-quantized NVFP4 build of the OBLITERATUS abliterated variant**, because a growing share of my subagent work is security research and the uncensored model gives straight technical answers instead of refusal fluff. The fun part is that no client needed to change: it serves under the same `qwen3.8-27b-coder` alias, so the router, the CLIs, and every subagent pin kept working. Same speed too, about 40 tok/s single stream and 317 tok/s aggregate with 8 concurrent subagents. The full quantization saga, including every ModelOpt trap I fell into, [is a post of its own](/posts/obliteratus-qwen38-27b-nvfp4-rtx5090/), and the weights are [on my Hugging Face](https://huggingface.co/Joestar79/Qwen3.8-27B-OBLITERATED-NVFP4).

The 27B's job in the stack is everything that should *not* disturb the big model: subagent loops (explore, review, research workers), quick questions, and image understanding. It is fast enough that a subagent fan-out feels instant, and SGLang's RadixCache on deep handles in-session prefix reuse for the brain, so each model's cache does what it is best at.

One hard-won lesson, with an honest correction. For months I believed **a 5090 at stock power crashes with dense models**, because mine did, and a systemd unit capping it to 450 W at boot made the crashes stop. Case closed, or so I thought. The real culprit turned out to be the GPU-boxed 4x8-pin to 12V-2x6 adapter cable: sharp load transients were tripping its protection, not the card. After switching to the PSU's native 12V-2x6 cable, stock 600 W has been rock solid, the cap is gone, and the box now runs a clock-lock plus undervolt profile instead (cooler and slightly faster than stock). Two lessons for the price of one: when a fix works, that does not mean the theory behind it was right, and always suspect the cheapest component in the chain first.

Also note for desktop users: set `--gpu-memory-utilization` with your desktop's appetite in mind. Your browser and compositor eat a couple of GiB that vLLM cannot plan around. I run 0.94 now, but only because I treat the GPU as the model's: when I need it for something heavy like video encoding, I stop the lane first instead of sharing.

### perception: embeddings, audio, monitoring, and the router

The 4080 box runs the small but essential services, and it has picked up new senses since the first version of this post:

- An **embedding endpoint** (Qwen3-Embedding-4B) and a **reranker** (bge-reranker-v2-m3) feeding the memory and retrieval stack.
- **Ears and a voice.** Speech to text runs on faster-whisper large-v3-turbo (through speaches), text to speech on Qwen3-TTS 1.7B with preset voices. Both share the 4080 with everything else and I never notice them.
- **A private Telegram bot as my remote control and pager.** It bridges chat to the STT and TTS endpoints and the router: I send a voice note, it transcribes, the router picks whichever model is on duty, and I get the answer back as audio. The same bot is how the stack notifies me when something deserves attention, which beats discovering a dead lane mid session.
- **Prometheus and Grafana watching the whole fleet.** Every node exports host metrics, the GPUs export NVML/DCGM counters, and the vLLM and SGLang engines expose their own request and cache metrics. One Prometheus scrapes it all, Grafana draws it, and there is a live tok/s speedometer dashboard I check more often than I should admit.

The piece that ties the fleet together also lives here: **token-miser**, a custom model router with two tricks:

1. **Chain-based failover.** Clients point at a single logical model (`auto`), and the router walks a configured chain until something answers. My fast lane is `v4flash@deep` then `qwen38@fast`: if the big model ever goes down, requests silently land on the 27B instead of erroring out mid-session.
2. **An explore agent loop.** Given a natural-language question about a codebase, it drives an agent that navigates the repo and answers with `file:line` citations. This loop runs on the 27B on fast, so exploration never competes with the brain for VRAM.

## The agent layer: where it all comes together

Here is the part that took the most iteration. I use three agent CLIs depending on mood and task: **opencode**, **pi**, and **qwen CLI**. All three are now configured identically at the strategy level:

- **Default model is the router** (`auto`), never a raw endpoint. Failover and routing come for free.
- **Subagents are pinned to the local 27B.** In opencode, the explore/general/reviewer subagents all run on `qwen3.8-27b-coder`. In pi, the subagents extension has `defaultModel` set to the 27B with the router as fallback. The big model stays focused on the main thread; the swarm runs on the workhorse.
- **Vision routes automatically.** Each CLI's model registry declares which models accept images (`modalities` in opencode, `input` in pi, `generationConfig.modalities` in qwen). Paste a screenshot and it goes to the multimodal 27B; text goes to the brain. No manual model switching.
- **A native `explore` tool.** I wrote a small pi extension that shells out to token-miser's explore loop, so the agent can ask "where is X implemented?" and get cited answers instead of grepping blind.
- **Persistent memory via MCP.** All three CLIs share a memory server (cairn-memory) that stores decisions, pitfalls, and conventions. This survives context compaction and, more importantly, *carries across the different CLIs*. A lesson learned in an opencode session is available to pi the next day.
- **Standing delegation rules in AGENTS.md**: subagent code review before claiming done, explore before grep, escalate risky architectural calls to the big model. The CLIs follow these without being asked each time.

The net effect: I type into whichever CLI I like, and the fleet decides where each piece of work runs. Highest automation, zero model babysitting.

## Boot resilience

Everything starts at boot, no manual steps:

- **deep**: a systemd quadlet for the SGLang container, with `[Install] WantedBy=default.target` *inside the .container file*. Gotcha: `systemctl enable` on a generated quadlet unit fails with "transient or generated", which is expected. The Install section in the quadlet itself is what wires autostart.
- **fast**: user services for the vLLM launch script, plus the nvidia power-limit unit.
- **perception**: services for the embedder, reranker, STT and TTS, the Telegram voice bot, Prometheus and Grafana, and the router.

## Things I tried and removed

Honest engineering notes, because not everything survived contact with reality:

- **A dedicated 4B "fast context" model for repo exploration.** I benchmarked it against the 27B on a real repo: 0/4 correct citations versus 4/4, and it was not even faster in wall-clock time. Quantization was never the bottleneck; capability was. I deleted it entirely and gave its VRAM back. Lesson: a slightly bigger model you already have beats a tiny specialist model, and *measure before you keep*.
- **Fronting the big model with LMCache via the router chain.** Looked good on paper, but SGLang's RadixCache already covers in-session prefix reuse, and the chain flip would have shadowed the smart model behind caching logic. The current split (LMCache on the 27B, RadixCache on the brain) is the cleaner design.

## Was it worth it?

Yes, and the reason is not raw throughput. It is that the stack behaves like one system: the CLI I am in does not matter, the failure of any single node does not kill my session, subagents are cheap enough that I delegate aggressively, and memory persists across tools and days. The models cooperate instead of queuing for my attention.

If you have the hardware sitting idle, the highest-leverage pieces in order were:

1. A router in front of everything, so clients never hardcode endpoints.
2. Subagents pinned to a fast local model, so delegation is free.
3. Shared memory via MCP, so lessons accumulate instead of evaporating at compaction.
4. Boot resilience, because a stack you have to babysit is a stack you will stop using.

The code and configs live in my repos; if there is interest, I will write a follow-up on the router internals and the pi extension. Happy to answer questions.
