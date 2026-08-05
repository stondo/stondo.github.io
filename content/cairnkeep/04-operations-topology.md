---
title: "Operate Cairnkeep Safely Across Stores, Machines, and Integrations"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Predict storage placement, take a consistent backup, reason about multiple machines, and add optional integrations without changing Cairnkeep's local-first default."
summary: "Once memory matters, storage location and recovery matter too. This lesson covers WAL-safe export, restore rehearsal, explicit remote topology, optional RAG and exploration, containers, and managed overlays."
tags: [cairnkeep, operations, backup, containers, developer-tools]
categories: [engineering, infrastructure]
keywords: [Cairnkeep backup, remote agent memory, multiple machines, AnythingLLM, Podman]
params:
  author: "stondo"
  toc: true
---

## Durable means operationally accountable

A memory feature becomes infrastructure as soon as you care about what it has
stored. At that point, "the agent remembers" is not enough. You need to predict
which process owns the database, take a consistent snapshot, restore it safely,
and explain every network boundary.

This article groups several optional surfaces because they share one question:
where do code and context go?

## Locate before backup

Start from the operations checkpoint in a disposable clone:

```bash
git switch --detach course-04-operation
cp .ai/course.env.example .ai/.env
mkdir -p .course-state/backups
cairn doctor
cairn memory path
```

For the course, the named and global store must be under
`.course-state/memory`. Project memory remains separate under
`.agentfs/project.db` because it is tied to the server working directory.

Do not infer the path from a default you remember. Ask the installed CLI for
the effective path after the launcher environment is loaded.

## Take a consistent snapshot

Export the disposable named and global databases:

```bash
cairn memory export .course-state/backups/global-memory.tgz
```

The export uses SQLite-aware snapshot behavior and includes write-ahead log
state. It requires the `sqlite3` CLI. Copying only the main database file while
a server is writing can omit committed data still represented in WAL files.

Project memory needs its own treatment. Either stop the server before a
filesystem copy or use SQLite's online backup operation against
`.agentfs/project.db`.

An archive is not a proven backup until restoration is rehearsed. In the
course's disposable store:

```bash
cairn memory import .course-state/backups/global-memory.tgz
cairn doctor
```

Import backs up databases it replaces. It is replacement, not record-level
merge. Restore a production archive into a disposable location first, validate
database integrity and recall behavior, and only then plan a production change.

## Reason about multiple machines

Draw two computers and place the `cairn memory-server` process explicitly.

In stdio mode, each harness starts a server child on its own machine. Each
server writes to its own effective storage path. Installing Cairnkeep on two
machines does not create shared memory.

For shared storage, an operator must run an authenticated HTTP memory server
and configure every client with its URL and token. The data then lives on the
server host. Project-routing headers can choose a project scope, but routing is
not authorization. The server still needs a deliberate trust and access model.

The course simulates two machines with two clones and two isolated
`.course-state/memory` directories. It deliberately never connects to a real
service.

## Optional document RAG

Cairnkeep memory works without a document retrieval service. Prove that first:

```bash
env -u ANYTHINGLLM_BASE_URL -u ANYTHINGLLM_API_KEY cairn doctor
```

If you already operate a disposable local RAG service, keep its credentials in
`.ai/.env`, configure a course-only workspace, and sync only the public README
and documentation. Search through the `domain_knowledge_*` tools and verify
citations against the files.

Then remove the variables and restart. Core memory should remain healthy. An
optional integration that cannot be removed cleanly has become an accidental
dependency.

## Optional repository exploration

`/context-explore` delegates repository exploration to an operator-owned
binary. With `CAIRN_EXPLORE_BINARY` unset, it should return a clear
configuration error without affecting memory.

When configured, verify every returned file and line citation. Automatic
prompt-time invocation is a separate opt-in, not a consequence of installing
the binary.

## Container boundaries

The public container launchers support two different goals:

```bash
cairn-container --help
cairn-container stdio --volume cairnkeep-course-memory
cairn-container workspace --repo "$PWD" --mode sandbox
```

A named stdio volume persists memory when a container is replaced. Workspace
sandbox mode copies the repository into a named volume. Shared mode mounts the
host checkout read/write and therefore does not isolate it from changes made in
the container.

List mounts and volumes before use. Clean up only resources with the explicit
course name.

## What an overlay should own

A managed overlay is a separate distribution policy, not a hidden fork. It can
pin a tested core version, provide a normal `cairn` wrapper, apply policy, write
a profile lock, and offer fleet upgrade and rollback gates.

Private endpoints, registry credentials, and organization policy belong in the
overlay or machine configuration. They do not belong in public Cairnkeep core
or a bootstrapped project template. The wrapper should make effective policy
inspectable and retain a neutral way to invoke upstream core behavior.

## The operational checklist

Before enabling any topology or integration, record:

1. the process that reads source or context;
2. the host where that process runs;
3. the data it stores and for how long;
4. the credentials and authorization boundary;
5. the verification command;
6. the reversal and cleanup procedure.

If one item is unknown, the feature is not ready for sensitive work.

Next: [Capture session evidence without turning everything into truth](../05-evidence-typed-memory/).
