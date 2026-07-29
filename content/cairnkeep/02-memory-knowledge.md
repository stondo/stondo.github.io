---
title: "Build a Reviewed Memory Lifecycle"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Write, recall, review, and supersede durable coding-agent memory while keeping canonical sources and derived project knowledge distinct."
summary: "Useful agent memory is not a transcript. This lesson stores one synthetic convention, proves cross-session recall, preserves a correction, and compares memory with a tracked project wiki."
tags: [cairnkeep, coding-agents, memory, knowledge-management, mcp]
categories: [engineering, ai]
keywords: [agent memory lifecycle, Cairnkeep remember recall, memory supersession, project wiki]
params:
  author: "stondo"
  toc: true
---

## Memory should be smaller than the conversation

Recording every prompt and response creates a large archive, not necessarily a
useful memory. Durable context earns its place when it can change a future
decision: a convention that is easy to miss, a pitfall discovered through
debugging, or a design choice with a reason.

It also needs a correction model. If a decision changes, overwriting the old
text destroys provenance. Keeping both as unrelated facts leaves future agents
to guess which one is current. Cairnkeep uses explicit supersession and history
so the current record remains useful without rewriting the past.

## Start from the memory checkpoint

Use a disposable clone at the second checkpoint:

```bash
git switch --detach course-02-memory
cp .ai/course.env.example .ai/.env
cairn sync --apply
cairn doctor
cairn memory path
```

Confirm the store is inside `.course-state/`, then launch your supported
harness through the generated `.ai` script.

Ask the agent to inspect these two canonical sources:

```text
docs/requirements.md
docs/decisions/0001-bounded-status-set.md
```

The project defines a bounded status set and uses UTC timestamps in examples.
The source files remain authoritative. We will store only a concise convention
that helps later work.

## Remember one reviewed convention

Inside the harness, run:

```text
/remember Course convention: Trail Ledger uses UTC timestamps in examples.
```

Then recall it by meaning:

```text
/recall Trail Ledger timestamps
```

Inspect the proposed key, scope, value, and provenance. A good memory key is
short, stable, and categorized, such as `conventions/utc-examples`. The value
should contain the decision, not a transcript of how the agent found it.

Now exit the harness completely, relaunch it through the same project script,
and repeat the recall. The successful result demonstrates persistence across
sessions. It does not yet prove that another project or machine shares the same
scope.

Run `/memory-review` and accept only the synthetic convention. Review is not a
ceremonial step. It is the boundary between a candidate observation and context
you are prepared to reuse.

## Correct without erasing history

Suppose someone proposes adding `lost` to the valid status set. Record the
proposal as a new decision only after checking the canonical documents. If it
replaces an earlier memory, use supersession rather than rewriting the old
value.

Inspect the resulting history with the lifecycle tools. You should be able to
answer:

* Which record is current?
* What did it supersede?
* Why was the correction made?
* Which source or review supports the current claim?

The lesson is not that memory can override the repository. It is that a changed
belief can remain traceable until the canonical source is updated and reviewed.

## Add derived project knowledge

Run the wiki workflow:

```text
/wiki-ingest docs
/wiki-query bounded status
/wiki-lint
```

Now compare the outputs.

The memory record is scoped durable context served from the memory store. The
wiki is a set of tracked files under `.planning/wiki/`. It should cite the
source documents, survive ordinary Git review, and be rebuildable if the source
changes.

These two mechanisms are complementary:

| Question | Durable memory | Derived wiki |
|---|---|---|
| Optimized for later agent recall? | Yes | Indirectly |
| Tracked and reviewed in Git? | No | Yes |
| Preserves memory supersession history? | Yes | No |
| Rebuilt from canonical documents? | No | Yes |
| Replaces canonical sources? | No | No |

The same principle applies to alignment reports and knowledge graphs. They can
accelerate navigation, but they remain derived views.

## Verify the default boundary

Core memory does not silently enable session evidence. Outside the harness,
run:

```bash
cairn trajectory list --json
cairn artifact list --json
```

No new trajectory or artifact should exist. Credentials or installed binaries
alone do not enable capture.

## A practical memory test

Before accepting any memory, ask four questions:

1. Will this information change a future action?
2. Is the scope narrow enough?
3. Can I cite or explain its provenance?
4. Would supersession be safer than editing an existing record?

If the answer to the first question is no, the observation probably belongs in
the session, not durable memory. If it is canonical project policy, it probably
belongs in a reviewed repository document first.

Next: [Find real defects with repository review and security audit](../03-review-security/).
