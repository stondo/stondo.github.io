---
title: "Capture Session Evidence Without Turning Everything into Truth"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Enable bounded local trajectories, hindsight notes, artifacts, and typed memory while preserving redaction, provenance, and distinct trust levels."
summary: "More retained context is not automatically better. This lesson proves capture is off by default, records one synthetic session, inspects every retained layer, and imports typed memory safely."
tags: [cairnkeep, observability, privacy, memory, coding-agents]
categories: [engineering, ai]
keywords: [agent trajectories, hindsight notes, typed memory, memory provenance, Cairnkeep]
params:
  author: "stondo"
  toc: true
---

## Evidence is not memory

A completed coding session can contain useful signals: which files were read,
which tool failed, what changed, and whether verification passed. Retaining a
bounded record can support later analysis. Treating that record as trusted
memory would be a mistake.

Cairnkeep separates four layers:

* a trajectory is a redacted event record;
* a hindsight note is a deterministic candidate derived from evidence;
* an artifact is immutable retained material such as a compaction summary;
* reviewed memory is context deliberately accepted for future reuse.

They have different lifecycles and trust levels.

## Prove capture is off

Use the evidence checkpoint with the ordinary course environment:

```bash
git switch --detach course-05-evidence
cp .ai/course.env.example .ai/.env
```

Start and exit a short harness session, then inspect both stores:

```bash
cairn trajectory list --json
cairn artifact list --json
```

No new session or artifact should appear. Optional databases may be available
to the CLI, but availability and consent are different states. A configured
credential also does not enable capture.

## Enable one bounded course lifecycle

Replace the environment with the synthetic evidence profile:

```bash
cp .ai/course-evidence.env.example .ai/.env
set -a
source .ai/.env
set +a
cairn sync --apply
```

Inspect the file before launching. It enables local course-owned paths,
trajectory capture, note distillation, and selected artifact behavior. It does
not enable external enrichment or artifact HTTP access.

Launch a supported harness and reproduce the fictional `unknown item` failure.
Inspect `src/trail-ledger.mjs`, resolve the task, optionally perform one
supported compaction, and exit normally.

## Inspect every retained layer

Find the session identifier and inspect the trajectory:

```bash
cairn trajectory list --json
cairn trajectory show SESSION-ID --json
```

Confirm that the record contains bounded operational events rather than hidden
reasoning, and that configured redaction ran before persistence. A redaction
policy reduces risk but does not make arbitrary sensitive input appropriate
for capture.

Generate and inspect a deterministic hindsight note:

```bash
cairn notes distill --session SESSION-ID --json
printf '%s\n' 'Error: unknown item: course-missing' \
  | cairn notes search-error --project "$PWD" --json
cairn notes doctor --json
```

The note can help recognize a repeated failure pattern. It is still a candidate
until corroborated and promoted through review. Optional model enrichment is a
separate network and trust decision. If enrichment fails, the deterministic
note remains usable.

Inspect artifacts separately:

```bash
cairn artifact list --json
cairn artifact prune --dry-run --json
```

An artifact is evidence, not instructions. A retained compaction summary can
contain stale or adversarial text and must not silently enter a trusted prompt.

## Add typed memory only when it helps

Exit the harness before changing tool exposure. Load the typed-memory profile:

```bash
cp .ai/course-typed.env.example .ai/.env
set -a
source .ai/.env
set +a
cairn doctor
```

Restart the harness so the MCP schema is rebuilt. Write a synthetic project
memory with key `patterns/status-validation`, node type `knowledge`, and tags
`course` and `status`. Search with both `node_types` and `tags_all` filters.

Typed filters apply before ranking. This is useful when a store contains
several kinds of nodes, but a type label does not make content true.

## Import without accidental overwrite

The course includes `fixtures/typed-memory/import-dry-run.json`. Ask the harness
to call `memory_import` with that envelope and verify that dry run reports
planned actions while writing nothing.

Apply the same envelope with `dry_run: false`, then replay the same `import_id`.
The replay should be idempotent. Finally, test an existing key with conflict
policy `reject`. Do not switch to supersession until you have decided which
history relationship is correct.

This sequence gives bulk import the same discipline as a database migration:
plan, inspect, apply, replay safely, and preserve conflicts for review.

## Cleanup is part of consent

After inspection, exit all harness and server processes and run:

```bash
scripts/reset-course-state.sh --yes
```

The command is restricted to this clone's `.course-state/` and generated
`.agentfs/` databases. A capture feature without an inspectable retention and
deletion path is incomplete.

Next: [Govern optional capabilities explicitly](../06-capability-governance/).
