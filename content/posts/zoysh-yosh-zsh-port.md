---
title: "Porting Yosh's yo to zsh in One Long Day: How Zoysh Was Born"
date: 2026-08-25T13:00:00+00:00
draft: false
description: "How I fell in love with Yosh, the LLM-enabled shell built by Fil Pizlo, and ported its yo command to zsh as a plain script plugin. The print -z trick, Yosh-compatible config, local-first defaults, and everything I deliberately did not port (yet)."
summary: "Yosh is Bash with an integrated LLM, built by Fil Pizlo on Fil-C. I wanted its yo command in my zsh without building a custom shell, so I ported the whole interaction model to a 900-line script plugin in one long day. Here is what made the port possible, what stays honest about being a port, and why the safety model survives translation."
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
  - print -z
  - Fil Pizlo
  - Fil-C
---

## The shell that got away

Some weeks ago I stumbled on [Yosh](https://yoshell.ai/), and I want to be precise about what happened next, because "I found a cool tool" undersells it. Yosh is Bash with an integrated LLM. You type `yo` followed by what you want, and the model places a command at your prompt *as if you had typed it yourself*. Not a suggestion box. Not a chat window with copy-paste. The command is sitting there in your prompt buffer, cursor blinking, and you either edit it, cancel it with Ctrl-C, or press Enter. Your hands never leave the keyboard and your eyes never leave the terminal.

I recommend watching what [Fil Pizlo](https://github.com/pizlonator), the author, did with the implementation, because it is gloriously unhinged in the best way: the entire stack, bash, readline, curl, openssl, zlib, libc, is compiled with [Fil-C](https://fil-c.org/), his memory-safe C runtime. A custom-built, memory-safe Bash with an LLM socket inside. It is a research-grade answer to the question "what if the shell trusted itself?"

There was one problem. I do not live in Bash. I live in zsh, the way some people live in a particular chair: years of muscle memory, a prompt I am fond of, plugins, widgets, the works. Yosh is a shell you switch to. I wanted `yo` in the shell I already had.

So I did the thing you do at two in the afternoon when you should be doing something else: I read Yosh's interaction model carefully and asked how much of its soul could survive a transplant into a plain zsh plugin. No custom shell. No Fil-C. Just a script you can load with zinit or antidote.

Twelve hours later, including a detour through terminal Markdown rendering and an argument with myself about config file semantics, [Zoysh](https://github.com/stondo/zoysh) v0.3.0 existed. The git log is almost embarrassing in its honesty: initial scaffold at 14:22, v0.2.0 at 14:27, and the last doc polish at 02:35 the following night.

## The one zsh feature that makes it all possible

Everything in Yosh's `yo` interaction hangs on a single behavior: putting text into the prompt buffer for the human to review. Bash needs a patched readline to do that from inside the shell. Zsh has it built in, and it is almost a crime how little it is used:

```zsh
print -z 'find . -type f -name "*.py" -newermt "$(date +%Y-%m-%d)"'
```

That is it. That is the whole magic. The string lands on your prompt, editable, cancellable, runnable. The LLM never executes anything. You do, after looking at it. Zoysh's entire safety model is Yosh's safety model, and `print -z` is what carries it across.

```
$ yo find all python files modified today
find . -type f -name "*.py" -newermt "$(date +%Y-%m-%d)"
# ↑ prefilled at your prompt — press Enter to run, or edit first

$ yo -c what does the -exec flag in find do?
The -exec flag runs a command on each matched file...
```

The `-c` mode is the other half: inline questions answered in place, with session memory so follow-ups work. "Now exclude the tests directory" does what you hope.

## What I ported, and what porting means when you respect the original

A port is a translation, and translations have rules. Mine were:

**The interaction model translates exactly.** `yo <intent>` prefills a command. `yo -c <question>` answers inline. Session memory bounds itself the way Yosh does, by exchange count and an approximate token budget. The keys never leave your terminal unless you configure a hosted provider.

**The config file is Yosh's config file.** Zoysh reads `~/.yoconf` and understands every portable directive Yosh defines, including the display styling ones: `chat_prefix`, `enable_bold`, `code_delimiter`, all of it. If Yosh documents a directive and it makes sense without a custom shell, Zoysh honors it. My additions are additive: OpenRouter as a provider, and local OpenAI-compatible servers as the *default* rather than an afterthought.

**The defaults are mine, and they are opinionated.** With no config file at all, Zoysh talks to `http://127.0.0.1:8001/v1/`. That number is not random: it is my fleet's model router. [I wrote about that stack earlier](/posts/aios-three-node-agentic-coding-stack/); the short version is that `yo` in my terminal lands on whatever model the router decides is on duty, usually a 27B running on the GPU next to my desk. A shell assistant that defaults to local hardware is a small statement, and I wanted to make it.

**The thinking models get de-thinked.** Local reasoning models emit `<think>` blocks before the answer. Zoysh strips them, because a terminal is not a place for watching a model deliberate, it is a place for the command.

**The API keys stay out of `ps`.** This one I am unreasonably proud of. Keys are passed to curl through a file-descriptor-backed header source, so they never appear in curl's process arguments. Small thing, habitual thing, the kind of thing you do because you looked at `ps aux` once at the wrong moment and never recovered.

## What I did not port (yet), and why that is the honest choice

Yosh has two flagship features that a script plugin cannot honestly deliver, and rather than fake them, Zoysh tells you.

**Scrollback awareness.** Yosh runs a transparent PTY proxy and shows the LLM what your terminal sees. That requires a native module, so Zoysh recognizes the `scrollback_enabled` directives for forward compatibility, refuses to pretend, and warns if you enable them. The PTY proxy is Phase 2, alongside a ZLE widget, and the repository carries the module scaffolding to prove the direction is real.

**Multi-step continuation.** Yosh chains commands automatically for complex tasks. In a pure script, without controlling the terminal, auto-chaining is how you get a half-executed pipeline and a sad evening. It waits for Phase 2 too.

There is a philosophical line here worth spelling out: a port that quietly drops features is lying to its users. A port that names the gaps is a roadmap.

## Under the hood, or: zsh is not a JSON runtime

The plugin is about 900 lines of zsh, which is 850 more zsh than any sane person should write before lunch. Zsh is a glorious, cursed language where quoting is an extreme sport and associative arrays are considered a modern convenience. For anything that touches JSON, Zoysh shells out to a small Python helper, because parsing model responses in pure zsh would be a betrayal of both the model and the reader.

The rest is the boring, durable stuff: provider tables for Anthropic, OpenAI, OpenRouter, Kimi, DeepSeek, Qwen, and z.ai; model auto-detection against a local server's `/models` endpoint; a Markdown renderer that knows what to do with bold, italics, headings, lists, and fenced code in a terminal; and a test suite wired into CI, because "it is just a shell script" stops being an excuse the moment other people install it.

It installs the way zsh people expect:

```zsh
zinit light stondo/zoysh
```

or antidote, or zplug, or oh-my-zsh, or literally `source` it if you enjoy minimalism.

## Standing on a giant's shell

Let me be clear about the credit hierarchy, because Zoysh has a family tree and I am the smallest branch on it. Yosh's design, its `yo` interaction model, its config conventions, and its actual `yo.c` implementation are Fil Pizlo's work. The man built a memory-safe C runtime and then recompiled the whole GNU stack with it *to make a shell*, which is exactly the kind of excessive, principled engineering the rest of us get to enjoy from downhill. Zoysh is GPL-3.0-only precisely because it is a port of his GPL work, the NOTICE file records the provenance down to who holds which copyright, and if Zoysh sends anyone curious toward Yosh and Fil-C, it has done its job twice.

Try it if you live in zsh: [github.com/stondo/zoysh](https://github.com/stondo/zoysh). And whether or not you do, go look at [Yosh](https://yoshell.ai/). Some tools are worth knowing about even when you cannot switch to them.

As for me, the next time I catch myself writing `| xargs grep -l` wrong twice in a row, I just type what I meant. The prompt fills in the rest, and my chair, my widgets, and my shell all stay exactly where I left them.
