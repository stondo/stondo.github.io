---
title: "Why Coding Agents Need Durable Project Memory"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Start with a small coding task and see why chat history, repository documentation, and durable reviewed memory solve different problems."
summary: "A coding agent can understand a repository today and still repeat the same investigation tomorrow. This first Cairnkeep lesson defines the missing layer and establishes a safe, reproducible demo project."
tags: [cairnkeep, coding-agents, memory, mcp, developer-tools]
categories: [engineering, ai]
keywords: [Cairnkeep, coding agent memory, durable context, MCP, agent memory]
params:
  author: "stondo"
  toc: true
---

## The repeated investigation problem

A coding agent starts a new session with an empty working memory. It can read
the repository, inspect tests, and reconstruct the architecture, but that work
has a cost. Worse, a subtle decision may not live in code at all. It may have
been discovered during debugging, agreed during review, or learned from a
failed approach.

Chat history does not fully solve this problem. It belongs to a harness and is
usually organized around conversations rather than durable project knowledge.
Putting every observation into `README.md` does not solve it either. A
repository document should contain reviewed information that humans expect to
maintain, not every transient clue produced during a session.

The useful missing layer is smaller than a new agent platform. It needs to:

* survive sessions and harness changes;
* separate projects and scopes;
* preserve corrections instead of silently replacing history;
* remain inspectable and exportable;
* make optional capture and network behavior explicit.

That is the problem Cairnkeep is designed to address.

## A project before memory

The public course uses
[Trail Ledger](https://github.com/cairnkeep/cairnkeep-course-labs), a deliberately
small Node.js application that records fictional shared equipment. Start from
the first checkpoint in a disposable clone:

```bash
git clone https://github.com/cairnkeep/cairnkeep-course-labs.git
cd cairnkeep-course-labs
git switch --detach course-00-app
npm test
```

The application has a bounded set of item statuses. Ask a coding agent a few
questions without giving it prepared context:

1. Which statuses are valid?
2. Why is that set bounded?
3. Where should a new status be validated?
4. What convention should examples use for timestamps?

The agent can answer by reading the code and documents. That is expected. Now
end the session, begin another, and ask again. The answer may still be correct,
but the investigation starts over. Any useful debugging discovery that was not
committed also disappeared with the first session.

This is not a model intelligence problem. The information was available, but
the lifecycle was wrong.

## Three kinds of context

It helps to distinguish three things that are often grouped under "memory."

### Canonical project sources

Code, tests, architecture decisions, and maintained documentation are the
authority. An agent should cite and verify them. Cairnkeep does not replace
them.

### Derived project knowledge

A wiki, alignment report, or graph can make canonical sources easier to
navigate. These artifacts are useful because they are tracked, reviewable, and
rebuildable. They are derived views, not a second source of truth.

### Scoped durable memory

A reviewed convention, pitfall, or decision can be recalled in a later session
without replaying the entire conversation. It should have a scope, a stable
key, provenance, and history. It should also be correctable without pretending
the earlier belief never existed.

Keeping these categories separate is more important than storing more data.

## Why MCP is part of the design

Cairnkeep exposes memory operations through the Model Context Protocol. A CLI
still handles installation, diagnosis, backup, and maintenance. MCP handles the
small set of structured operations that a compatible coding harness can call
during a session.

This split is deliberate. The CLI is better for operators. MCP gives different
harnesses the same typed memory contract without teaching each one how to open
or mutate the underlying database. In the ordinary local setup, the MCP server
runs over stdio as a child process. That does not imply a remote service or a
network hop.

## What this series will prove

The remaining articles add one layer at a time:

* local installation and predictable storage;
* reviewed memory, history, and derived knowledge;
* repository review and security auditing against a vulnerable fixture;
* backup, restoration, multiple-machine topology, and optional integrations;
* explicitly enabled session evidence and typed memory;
* capability state with restart boundaries;
* deterministic evaluation with honest claim limits.

Every optional stage can be skipped. A useful local memory server does not
require document RAG, session capture, a remote host, containers, or an
evaluation framework.

## Before the next article

Keep this first checkout disposable. Do not configure it with personal data,
private repositories, or a real remote endpoint. The next article starts from
`course-01-bootstrap` and answers the operational question that matters before
the first write: where will the memory actually be stored?

Next: [Install Cairnkeep and know where every byte goes](../01-install-storage/).
