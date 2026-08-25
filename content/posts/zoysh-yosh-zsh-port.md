---
title: "Porting Yosh's yo to zsh in One Long Day: How Zoysh Was Born"
date: 2026-08-25T13:00:00+00:00
draft: false
description: "How I fell in love with Yosh, the LLM-enabled shell built by Fil Pizlo, and ported its yo command to zsh as a plain script plugin: streaming answers, Ctrl-C cancellation, multi-step plans, a ZLE widget, scrollback capture, and an experimental native module. Plus the print -z footgun I shipped first and fixed honestly."
summary: "Yosh is Bash with an integrated LLM, built by Fil Pizlo on Fil-C. I wanted its yo command in my zsh without building a custom shell, so I ported the interaction model in one long day, then spent a second round closing the gaps: streaming, cancellation, plans, a widget, capture, and a native module. Including the prefill footgun I found in my own first release."
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

Then I got greedy, in the disciplined way. The first release was a translation of Yosh's surface: the `yo` interaction, the config conventions, the providers. But Yosh has deeper machinery, and I wanted it too. So I wrote myself a six-feature plan, executed it branch by branch with a test for everything, and v0.4.0 is the result. More on what actually shipped below, including one place where porting the feature honestly meant changing my own first design.

## The one zsh feature that makes it all possible

Everything in Yosh's `yo` interaction hangs on a single behavior: putting text into the prompt buffer for the human to review. Bash needs a patched readline to do that from inside the shell. Zsh has it built in, and it is almost a crime how little it is used:

```zsh
print -z 'find . -type f -name "*.py" -newermt "$(date +%Y-%m-%d)"'
```

That is it. That is what I thought was the whole magic. The string lands on your prompt, editable, cancellable, runnable. The LLM never executes anything. You do, after looking at it. Zoysh's entire safety model is Yosh's safety model.

Then, while porting multi-step continuation, I found a footnote in my own
footgun: `print -z` does not put text in the editor buffer, it pushes onto
the shell's *input stack*, and anything already queued in the terminal,
like an Enter impatiently pressed while the model was still streaming, gets
applied to that pushed line and can run it before you ever saw it. zpty
made the failure visible; the fix is a small `zle-line-init` hook that
places the command into `BUFFER` as pure editor state. Nothing executes
until you accept the line you can actually see. The safety promise now has
a mechanism that keeps it, not just a convention.

```
$ yo find all python files modified today
find . -type f -name "*.py" -newermt "$(date +%Y-%m-%d)"
# ↑ prefilled at your prompt, press Enter to run or edit first

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

When the first draft of this post was written, two Yosh flagship features
were missing and the post said so plainly. Both have since landed, in
shapes that respect what a script can honestly do, and one gap remains.

**Scrollback awareness.** Yosh runs a transparent PTY proxy and shows the
LLM what your terminal sees. I investigated porting that wholesale, using
zsh's own `zpty` and a fork/exec PTY pair, and wrote the verdict down in
`doc/pty-design.md` before writing code: both reduce to building a
terminal multiplexer, with resize and signal forwarding across a pty
boundary, which is a bug farm out of all proportion to the value. So v1
capture is narrower and truthful: with `scrollback_enabled 1`, plan steps
prefill as `zoysh-run <command>`, a wrapper that tees the command and its
output into a bounded ring, and later questions carry that ring as
context. You can see exactly what will be captured before you press
Enter. Ambient whole-terminal capture stays Yosh-only until the native
module grows it, and the README says exactly that.

**Multi-step continuation.** Now opt-in via `continuation 1`: the model
may answer with a fenced `zoysh:plan` block, one command per line, and
zoysh prefills each step in order, advancing only when the prefilled
command itself ran. Type anything else and the queue drops. `yo --skip`
and `yo --abort` manage it, and no step ever runs without your Enter.

Along the way the port grew the rest of the Yosh feel, one branch at a
time: answers stream over SSE with `<think>` reasoning hidden live, then
settle into rendered Markdown byte-identical to the non-streaming path;
Ctrl-C kills the request's whole process group and keeps your partial
answer; `M-y` turns the buffer you are typing into a generated command
without ever leaving the line editor; and an experimental native module
(`zoysh-status`, a C config parser, a curl streaming client speaking the
exact same record protocol) proves the Phase 2 pipeline, with the script
engine still the default and the two engines verified byte-identical by
tests.

There is a philosophical line here worth spelling out: a port that quietly
drops features is lying to its users. A port that names the gaps is a
roadmap, and then, slowly and honestly, closes them.

## Under the hood, or: zsh is not a JSON runtime

The plugin is now about 1,800 lines of zsh, which is roughly 1,750 more zsh than any sane person should write before lunch. Zsh is a glorious, cursed language where quoting is an extreme sport and associative arrays are considered a modern convenience. For anything that touches JSON, Zoysh shells out to a small Python helper, because parsing model responses in pure zsh would be a betrayal of both the model and the reader.

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
