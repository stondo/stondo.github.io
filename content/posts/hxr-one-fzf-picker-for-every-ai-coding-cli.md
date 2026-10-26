---
title: "hxr: One fzf Picker for Every AI Coding CLI"
date: 2026-10-26T02:00:00+00:00
draft: false
description: "Seven coding agents run on my machines, and every one of them hides its session history in a different place: two generations of sqlite schema, a decoy sessions directory, AI-generated titles buried at the tail of JSONL transcripts, and a database column that sometimes contains an entire pasted conversation. hxr reads all of them read-only in about 50 milliseconds and gives me one fzf picker to find a past session and resume it in the right directory with the right command. This is the story of mapping seven storage formats, the fzf tab problem that forced a state file, and the drift guards that keep the whole thing from silently rotting."
summary: "A single Rust binary over seven undocumented session stores: bounded reads and parallel discovery at 900 sessions in 50ms, per-CLI resume semantics, fzf tabs cycled through a reload state file, and fixture tests with synthetic stores so upstream format drift fails loudly instead of hiding a tab."
tags:
  - rust
  - cli
  - fzf
  - developer-tools
  - ai-agents
  - open-source
categories:
  - engineering
  - tools
keywords:
  - fzf
  - rust cli
  - claude code
  - codex
  - opencode
  - session management
  - terminal
---

I use seven AI coding CLIs. opencode for most work, Claude Code and Codex when a specific model earns its slot, pi and omp and Qwen Code and reasonix for the corners where each of them is the right tool. Every one of them is good at remembering conversations. Every one of them remembers in a different place, in a different format, with a different resume flag.

The Monday morning problem looks like this: I know I was debugging the NFS mount guard in one of these tools two days ago. Which one? The muscle memory is `claude -c` except that continues the wrong project, `codex resume` opens a picker I cannot pre-filter, opencode buries sessions behind a subcommand, and the other four I would have to look up. Again.

![hxr picker](/hxr-card.png)

My colleague Patrick at Siemens had the same problem for exactly one tool and solved it in about 60 lines of Python: query the opencode sqlite database, format the rows, pipe them into fzf, `exec` the resume in the session's own directory. I ran that script for months. This post is about what happened when it became obvious that the script was the right idea pointed at one seventh of the problem.

## Seven stores, one Session struct

The first real work was archaeology, not code. Nobody documents where these CLIs keep sessions, so I read them all live, and each one had an opinion:

| Harness | Sessions live in | The catch |
|---|---|---|
| opencode | sqlite, two schema generations | union both tables, dedupe ids |
| Claude Code | one JSONL transcript per session | titles are `ai-title` entries appended at the file tail |
| Codex | `state_*.sqlite`, a threads table | version-numbered filename, must not `SELECT *` (rows embed full conversations) |
| pi | JSONL with a session-meta first line | no titles anywhere, derive from first user message |
| omp | a history.db with AI titles | session directory names are broken munges that drop the home prefix |
| Qwen Code | `projects/*/chats/*.jsonl` | the obvious `sessions/` directory is a decoy full of pid locks |
| reasonix | named handles plus a `.meta.json` | archived sessions get suffixes you must filter |

Three of those deserve their own warning label. Qwen Code's `~/.qwen/sessions/` looks exactly like the place history would live, and it is not; the real transcripts sit under `projects/<munged-cwd>/chats/`. omp's session directories munge paths so aggressively that `/home/me/.config` and `/home/me/config` collide, and the munge drops the home prefix entirely, so I never trust those names and read cwd from the database instead. And my favorite: Codex's `threads.title`, which is usually a neat model-generated summary, will occasionally contain an entire pasted conversation. The first time my picker rendered a three-thousand-character title I thought I had a string bug. I had a data bug that was upstream of me, so now every title gets truncated at 80 characters at ingestion.

All of this is read-only. hxr never writes to a harness store, opens sqlite databases in read-only mode, and reads transcripts with hard caps: 256 KiB from the head for cwd and the first user message, 64 KiB from the tail for the newest title entry. A listing pass never scans a full transcript, even the 13 MB ones.

## Tabs were harder than they look

The obvious UI is one merged fzf list, newest first, with a two-character tag per harness. That shipped in a day. The interesting part was per-harness tabs.

fzf key bindings are static. A binding says "run this command", it cannot say "figure out which tab is active and advance it". Alt+number bindings work around that by making each target explicit, but Alt+digit collides with desktop and terminal shortcuts on Fedora and plenty of other setups, so that was out.

The fix is that the list is already generated by a subprocess. Tab and Shift+Tab re-run hxr itself with a cycle direction and a state file: read the current tab, compute the next one, write it back, print the new list. The tab bar rides inside the feed as fzf `--header-lines`, so it reloads with the list and no dynamic binding machinery is needed at all. The whole tab system is one small file in `/tmp` and two bindings:

```
--bind=tab:reload("hxr" --cycle next --state "$STATE")
--bind=btab:reload("hxr" --cycle prev --state "$STATE")
```

A tab switch costs one scoped subprocess, and because the lister is Rust that is about 30 milliseconds end to end. That was the performance argument for Rust over porting the Python script: not raw scan speed, but the fact that a Python interpreter restart on every tab press lands in the hundreds of milliseconds, which reads as lag.

## Speed is a schema decision

Discovery runs all adapters in parallel and lands around 50 milliseconds for roughly 900 sessions across the seven stores. The numbers that got it there are boring and worth writing down:

- Never shell out for data you can read. The first version asked the opencode CLI for its database path on every run, which is a 30 millisecond subprocess. The standard XDG path is checked first now, and the CLI is only the fallback.
- Bounded reads everywhere: head and tail caps per transcript mean listing cost does not grow with transcript size.
- Parallel adapters: seven independent scans make the whole discovery cost roughly the slowest adapter, not the sum.

## These CLIs will drift. Guard against it.

Every harness in that table ships weekly and rewrites its storage without asking. Codex alone has moved through at least five state database versions. hxr defends two ways.

First, adapters fail loud and isolated: a broken store prints one warning and loses its tab, it never takes the other six down. Second, the repository carries fixture tests: a synthetic home directory containing one minimal store per harness, sqlite schemas included, with assertions on ids, titles, fallback chains, and resume argv. When an upstream format changes, a named fixture assertion fails in CI before anyone notices a missing tab in daily use. A smoke script does the same against live stores, including dry-run resume resolution, without ever launching a harness TUI.

The fixture home also makes CI hermetic: the test suite needs none of the seven CLIs installed, which is why the GitHub Actions matrix can be two plain runner images.

## Release

hxr is open source (MIT or Apache-2.0, your pick), Linux and macOS, one binary plus fzf:

```
cargo install hxr-picker   # installs the `hxr` command
```

The package name is `hxr-picker` because someone took `hxr` on crates.io in 2024 for an HTTP client that never shipped a second release. The command is `hxr`. Sources and the full format reference for all seven stores:

- Repository: [github.com/stondo/hxr](https://github.com/stondo/hxr)
- Crate: [crates.io/crates/hxr-picker](https://crates.io/crates/hxr-picker)

If you use a coding CLI that is not on the list, the adapter trait is two functions, `discover()` and `resume()`, and the fixture file shows exactly what a new adapter has to prove. Pull requests welcome, and bring your archaeology notes.
