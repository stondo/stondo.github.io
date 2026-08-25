---
title: "Zoysh: Yosh's yo Comes to zsh"
date: 2026-08-25T13:00:00+00:00
draft: false
description: "Zoysh is a zsh port of Yosh, the LLM-enabled shell by Fil Pizlo. It generates shell commands from natural language, prefills them for review, streams answers, runs multi-step plans, and never executes anything you did not press Enter on. Works with local models by default."
summary: "A zsh plugin that ports Yosh's yo interaction model: natural language in, reviewed command at your prompt, streaming answers, multi-step plans, scrollback context, and an experimental native module. Local-first, multi-provider, and safe by construction."
tags:
  - zsh
  - llm
  - cli
  - opensource
  - shell
  - terminal
  - developer-tools
  - self-hosting
categories:
  - engineering
  - tools
keywords:
  - zsh plugin
  - LLM shell
  - Yosh
  - Zoysh
  - natural language shell
  - zle
  - Fil Pizlo
  - Fil-C
---

## What it is

[Zoysh](https://github.com/stondo/zoysh) is an LLM-powered shell assistant for zsh. It is a port and adaptation of [Yosh](https://yoshell.ai/), the LLM-enabled Bash created by [Fil Pizlo](https://github.com/pizlonator), and it brings Yosh's `yo` interaction model to zsh as a plain plugin. No custom shell, no compiled extensions required, nothing to switch away from. If you live in zsh, you keep living in zsh.

```
$ yo find all python files modified today
find . -type f -name "*.py" -newermt "$(date +%Y-%m-%d)"
# ↑ prefilled at your prompt, press Enter to run or edit first

$ yo -c what does the -exec flag in find do?
The -exec flag runs a command on each matched file...
```

Type `yo` followed by what you want. The model generates a command and it appears at your prompt, as if you had typed it yourself. Edit it, cancel it, or press Enter. Ask with `yo -c` and the answer prints inline, right there in the terminal.

The core safety rule is the same one Yosh established: **zoysh never executes generated output**. Commands reach your prompt as editor buffer state through a `zle-line-init` hook, and nothing runs until you accept a line you can actually see. Not a suggestion box, not a confirmation dialog, and not the shell input stack either: the hook-based placement exists precisely because pushing text onto the input stack lets already-queued keystrokes accept a line before you reviewed it.

## What it does

**Command generation.** Natural language in, zsh command out, prefilled for review. The context includes your OS, zsh version, working directory, and git branch, so the commands fit where you are.

**Streaming answers.** Chat responses stream over SSE as they are generated. Reasoning models stream with their `<think>` blocks hidden, including tags that split across chunk boundaries, and the settled text is rendered as terminal Markdown, byte-identical to the non-streaming path. That equality is enforced by tests. A `streaming 0` directive restores the old blocking behavior, and non-SSE responses fall back automatically.

**Cancellation.** Ctrl-C kills the in-flight request and its whole process group, keeps whatever partial answer already printed, and returns you to the prompt with a short notice. The interrupt trap is verified not to leak into normal shell behavior afterwards.

**A ZLE widget.** `M-y` on what you are typing turns the buffer into a generated command without leaving the line editor. On an empty buffer it opens an inline mini-prompt. The result is assigned to `BUFFER` and `CURSOR` directly, so you stay in ZLE from question to command.

**Multi-step plans.** With `continuation 1`, the model may answer with a fenced `zoysh:plan` block: one command per line, executed as a queue you drive. Each step is prefilled for review, and the queue advances only when the prefilled command itself ran. Type anything else and the queue drops. `yo --skip` and `yo --abort` manage it. Follow-up queries carry the plan and completed steps as context, so the model can adjust what remains.

**Scrollback capture.** With `scrollback_enabled 1`, plan steps prefill as `zoysh-run <command>`, a wrapper that tees the command and its output into a bounded ring and preserves the exit status. Later queries carry that ring, so "what did the last command print" just works. You can see exactly what will be captured before you press Enter. Ambient capture of the whole terminal stays a Yosh feature; the design notes in `doc/pty-design.md` explain why a script-honest port stops here and what a native implementation would take.

**Session memory.** Follow-ups work. "Now exclude the tests directory" does what you hope. Memory is bounded by exchange count and an approximate token budget.

**An experimental native module.** Phase 2 of the port has begun behind an opt-in switch: vendored cJSON, a `zoysh-status` builtin with a C port of the config parser, and a `zoysh-call` streaming client that speaks the exact same record protocol as the script engine. The two engines are verified byte-identical by gated tests (`make check-module`), the script engine remains the default, and `make check` never requires the module.

## Providers

Zoysh is local-first. With no configuration at all it talks to an OpenAI-compatible endpoint at `http://127.0.0.1:8001/v1/`, auto-detects the served model through `/models`, and explains what to start if the server is down. In my setup that endpoint is a router in front of a small GPU fleet, so `yo` lands on whichever model is on duty that day.

Hosted providers are supported when you want them: Anthropic, OpenAI, OpenRouter, Kimi, DeepSeek, Qwen, and z.ai, each with sane defaults and key-file conventions. Keys resolve from the config, the environment, or provider key files, and they are passed to curl through a file-descriptor-backed header source, so they never appear in `ps` output.

Configuration lives in `~/.yoconf` and follows Yosh's conventions: every portable directive Yosh defines is honored, including the display styling ones (`chat_prefix`, `enable_bold`, `code_delimiter`, and friends). The file is re-read before every command, so edits take effect immediately. Zoysh's additions, like `streaming`, `continuation`, and `scrollback_enabled`, are additive.

## Installing

Any zsh framework works, or none:

```zsh
zinit light stondo/zoysh
```

antidote, zplug, and oh-my-zsh are equally supported, and `source zoysh.plugin.zsh` is fine if you enjoy minimalism. Dependencies are zsh 5.8+, curl, and python3 3.8+ for the JSON handling. The test suite runs on GitHub Actions against a bundled OpenAI-compatible stub server, so nothing in CI needs a model or a GPU.

## Privacy and provenance

The query, current directory, OS, zsh version, git branch, bounded session history, and (if you opt in) the scrollback ring are sent to the configured API endpoint. No telemetry, no persisted conversation history, no keys in process arguments.

Zoysh exists because of Yosh. The design, the `yo` interaction model, the config conventions, and the original `yo.c` implementation are Fil Pizlo's work; the man built a memory-safe C runtime, [Fil-C](https://fil-c.org/), and recompiled the entire GNU stack with it to make a shell. Zoysh is GPL-3.0-only because it is a port of that GPL work, the NOTICE file records provenance down to the copyright holders, and it is an independent project, not affiliated with or endorsed by Fil Pizlo or Epic Games. If Zoysh makes you curious about Yosh and Fil-C, it has done its job twice.

The repo has the full picture: [github.com/stondo/zoysh](https://github.com/stondo/zoysh). Try it if you live in zsh. The commands are yours to review, and the shell stays exactly where you left it.
