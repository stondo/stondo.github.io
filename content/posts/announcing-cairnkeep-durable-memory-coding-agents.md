---
title: "Announcing Cairnkeep: Durable, Local-First Memory for Coding Agents, Without the Autonomy"
date: 2026-08-27T23:00:00+00:00
draft: false
description: "Every coding agent I run forgets everything when the session ends, every harness silos its own memory, and most agent-memory projects gradually turn into autonomous agents that write and approve things on their own. Cairnkeep is my answer: a durable, local-first memory and context layer for coding agents that works across Claude Code, OpenCode, Codex, Kimi, Qwen and Pi, ships a 26-lesson release-verified curriculum, and is built around one stubborn rule: memory is durable context, never an authority."
summary: "I have been building Cairnkeep since early July: project-scoped memory that survives sessions and crosses harnesses, a retrieval-first protocol agents must follow instead of new autonomy, least-authority MCP surfaces (read-only tool profiles, capability gates, immutable context packs), and optional, explicit opt-ins for everything networked. This post is the what and the why, the production deployment I run it in (a VPS-hosted memory server for a four-machine fleet, with the work laptop deliberately isolated), the 26-lesson learning path with its course labs repo, and the companion video series on my BitsEntangled YouTube channel."
tags:
  - cairnkeep
  - ai-agents
  - mcp
  - memory
  - claude-code
  - opencode
  - local-first
  - open-source
  - self-hosting
categories:
  - engineering
  - open-source
keywords:
  - Cairnkeep
  - coding agent memory
  - MCP
  - Model Context Protocol
  - durable memory
  - local-first
  - Claude Code
  - OpenCode
  - Codex CLI
  - agent context
---

Here is a pattern every heavy agent user knows. Mid-session, your coding agent finally understands the project: which quadlet owns port 8001, why the tokenizer guard exists, which PR is waiting on whom. Then the session ends. The next agent, sometimes in a different harness on a different machine, starts from zero and pays the rediscovery tax again, in tokens, in GPU time, and in your patience.

The obvious fix is a memory system. The problem is what most memory systems want to become. Give an agent durable storage and, within a release cycle or two, it wants to write autonomously, act on its memories, approve its own promotions. The memory stops being a notebook and starts being a second agent you have to supervise.

So I built the thing I actually wanted. [Cairnkeep](https://github.com/cairnkeep/cairnkeep) is durable, local-first memory and context for coding agents: it stores decisions, constraints, patterns and lessons in scoped memory, and it gives agents a deliberately bounded protocol for finding and verifying that context. It does not execute work, does not grant approval, does not phone home. It has been running my own fleet since the 1.0.1 release in July (it replaced a pile of bespoke harness glue I had jokingly named work-agent-infra), and it is at [v2.16.0 on npm](https://www.npmjs.com/package/@cairnkeep/cli) as `@cairnkeep/cli`, Apache-2.0.

## The three problems it attacks

1. **Continuity.** Recall project decisions and recurring failures instead of rediscovering them from scratch. The memory-wakeup hook in my setup surfaces project memory automatically at session start; agents start a session already knowing last week's conclusions.
2. **Harness independence.** The same memory through Claude Code, OpenCode, Codex CLI, Kimi Code, Qwen Code, Pi, or any MCP client. A decision recorded from OpenCode on the Fedora workstation is recalled by Claude Code on the MacBook an hour later. No silos, no export scripts.
3. **Trust.** Local by default: the stdio server stores data on your computer, and remote access, embeddings, capture, and external context are explicit opt-ins you can enumerate.

## Memory is context, not truth, and definitely not authority

This is the design hill I picked, so it deserves its own section. Everything else follows from it.

The managed instruction block that `cairn setup` installs teaches compatible agents a four-step protocol, and the whole thing is retrieval-first:

1. For a nontrivial task that may depend on prior project knowledge, make **one task-derived, project-scoped `memory_search`** the first tool call.
2. Treat results as **locators, not truth**: verify against the repository sources they reference. Memory that contradicts the code loses, every time.
3. If memory is unavailable or empty, continue with normal repository inspection. Graceful fallback is part of the contract, not an error path.
4. **Never write, supersede, or approve durable memory** unless the user or a reviewed workflow explicitly asks for it.

Playbooks extend this pattern: they can recommend or require existing review, verification, security or documentation steps, but they never run those steps, activate skills, or authorize destructive work. And the MCP surface itself follows least authority: complete tool annotations, read-only and custom tool profiles, capability gates, and immutable context packs keep observation strictly separate from action.

There is a trust-boundary page in the docs that says it more bluntly than marketing ever would: Cairnkeep is not a hosted agent, not a telemetry service, not an automatic remote knowledge collector. Ordinary uninstall retains your durable data unless you explicitly ask for a purge. I wrote that page to be re-read in six months, when the feature list is longer and the temptation bigger.

## The core loop, and the five-minute version

Node 22 or newer, Linux, macOS, native Windows x64, or WSL (yes, it runs on real Bash 3.2; that compatibility was deliberate and cost an afternoon). Guided setup presents your Git and memory choices plus a checkbox for every supported harness:

```bash
npm install --global @cairnkeep/cli
cairn setup /path/to/project
cd /path/to/project
cairn doctor
./.ai/start-codex.sh        # or start-claude.sh, start-kimi.sh, start-qwen.sh ...
```

The daily loop is intentionally small. In Claude Code and OpenCode, `/remember` and `/recall` are wired up for you; in any other MCP client it is `memory_write` and `memory_search` directly. Store a concise, verified project fact. Close the session. Retrieve it when a later task needs it. That is the entire religion.

## Running it in production: one memory server, four machines, one hard boundary

My deployment is the configuration Cairnkeep's optional remote mode exists for. A single `cairn-memory` HTTP MCP server runs on a small VPS, behind a bearer token, serving every personal machine: the Fedora workstation, the Debian box, the MacBook, and the VPS agents themselves. Every harness on every machine points at the same scoped memory. Work-related projects use a local-only memory on the work laptop by explicit policy; that machine never touches the personal server, and personal endpoints and credentials live in a separate overlay repo rather than anything public.

Two production details I appreciate having escaped into seams instead of the core:

- **RAG is an opt-in seam.** Long-form documentation goes to an AnythingLLM workspace via a sync script wired through an environment variable. Cairnkeep core knows nothing about it; the embedding and retrieval backends run on my GPU nodes and can be swapped without touching the memory server.
- **Scope discipline is a feature.** When I built a vision roleplay app that needed relationship memory, I did not reuse Cairnkeep for it. Coding-agent memory is the wrong shape for that job, and knowing what not to bolt a tool onto is part of the tool. That app got a simple per-user JSON store and is happier for it.

## Learning it: 26 lessons, four tracks, real labs

The part of the project I am proudest of is not the code, it is the [learning path](https://github.com/cairnkeep/cairnkeep/tree/main/docs/learning). Four tracks: Quickstart (prove the core value, including the one-command Codex project path, in about 110 minutes), Practitioner (daily project work), Evidence and Evaluation (session evidence, capability governance, measuring changes without overstating results), and Operator (storage, services, containers, managed distributions).

Twenty-six lessons, L00 through L25. Eleven are release-verified today, meaning the exercise has been run against the actual release; the rest are published as outcomes with acceptance criteria and will only be marked ready once their labs are executable. Every complete lesson follows the same teaching contract: a practical problem and observable outcome, prerequisites and a clean start, a hands-on exercise with non-sensitive sample data, verification and recovery steps, an explicit privacy or security boundary, and release compatibility metadata. A learner never needs optional infrastructure to get through the fundamentals, and the [course labs repository](https://github.com/cairnkeep/cairnkeep-course-labs) provides one synthetic project with tagged checkpoints shared by lessons, articles, workshops, and videos.

The curriculum map even records where every public feature is introduced, practiced, and operated, and marks design-only work that must not be taught as a released capability. That last clause exists because I have watched too many project docs teach the roadmap as if it were the product.

## The videos: BitsEntangled

Reading docs is not how everyone learns, so there is a companion video series on my YouTube channel, [BitsEntangled](https://www.youtube.com/@BitsEntangled). It is a growing set of short, numbered episodes (fifteen and counting as I write this, most in the one-to-four-minute range) covering the same ground as the lesson track: getting Cairnkeep across machines without losing state, declaring capabilities and MCP tools explicitly, evaluating changes without overstating results, and the rest of the daily workflow. If you want the feel of the project in five minutes instead of a blog post's worth of scrolling, [the playlist is here](https://www.youtube.com/@BitsEntangled/videos).

## Where to start

- L00, "Why Cairnkeep?", is a ten-minute read and the honest elevator pitch: [github.com/cairnkeep/cairnkeep](https://github.com/cairnkeep/cairnkeep)
- The [quickstart](https://github.com/cairnkeep/cairnkeep/blob/main/docs/quickstart.md) walks Claude Code setup, the first remember/recall cycle, and recovery.
- The [coding-agent guide](https://github.com/cairnkeep/cairnkeep/blob/main/docs/agents.md) is the normative reference for the retrieval-first protocol, the authority model, and fallback behavior.

The project is two months old, at 2.16.0, and the roadmap's direction is more of the same: more verified lessons, more harness surfaces, stricter authority boundaries. Memory that outlives sessions is easy. Memory that stays in its lane is the actual product.
