---
title: "Install Cairnkeep and Know Where Every Byte Goes"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Install Cairnkeep locally, bootstrap a project, register the memory server, and verify storage before writing durable context."
summary: "A sound installation is not just a successful command. This lesson verifies the binary, generated project layer, harness registration, operating assets, and exact database location."
tags: [cairnkeep, installation, mcp, storage, developer-tools]
categories: [engineering, ai]
keywords: [install Cairnkeep, MCP server, agent memory storage, Cairnkeep bootstrap]
params:
  author: "stondo"
  toc: true
---

## Installation is a chain, not one command

An installer can finish successfully while the software remains unusable. A
coding-agent memory layer has several boundaries to verify:

1. the CLI is the version you intended to install;
2. the project scaffold was created in the intended repository;
3. the harness starts the intended memory server;
4. operating commands and hooks match the installed version;
5. the effective database path is predictable.

The last point is critical. Installing Cairnkeep never discovers an account,
remote host, or storage server. The default server is local. A remote topology
exists only when an operator explicitly configures an authenticated HTTP URL
and token.

## Prepare an isolated course checkout

Use the bootstrap checkpoint rather than a real project:

```bash
git clone https://github.com/cairnkeep/cairnkeep-course-labs.git
cd cairnkeep-course-labs
git switch --detach course-01-bootstrap
npm test
```

Install the current public CLI using the method documented in the
[Cairnkeep README](https://github.com/cairnkeep/cairnkeep#installation), then
verify what your shell resolves:

```bash
command -v cairn
cairn version
```

The course is written against Cairnkeep 2.4.0. If you are using a newer
version, read its release notes before assuming screenshots or tool lists are
identical.

## Bootstrap the project layer

From the course repository:

```bash
cairn bootstrap "$PWD"
```

Bootstrap creates project launchers, an environment template, planning policy,
and ignored state locations. It does not silently select private policy or a
remote memory server. Inspect the result before continuing:

```bash
git status --short
find .ai .planning .agentfs -maxdepth 2 -type f -print
```

For this course, use the provided isolated environment rather than your normal
machine profile:

```bash
cp .ai/course.env.example .ai/.env
```

Read `.ai/.env`. The named and global store must resolve beneath this clone's
`.course-state/` directory. Secrets never belong in the committed example.

## Register the MCP server once

Harness registration is a user-level operation. The exact command depends on
the supported harness, but the server command should ultimately launch:

```bash
cairn memory-server
```

For example, the public course supports generated launchers for Claude Code and
OpenCode. Follow the current
[installation lesson](https://github.com/cairnkeep/cairnkeep/blob/main/docs/learning/lessons/L02-installation.md)
for the selected harness rather than copying an old registration command.

After registration, install the matching operating layer:

```bash
cairn sync --apply
cairn sync --check
```

`sync --check` should report current state. This matters after upgrades because
a new CLI can coexist with old generated commands or hooks until the project is
synchronized.

## Diagnose before writing

Run the three checks that establish the effective installation:

```bash
cairn doctor
cairn sync --check
cairn memory path
```

For the course, stop if the named or global path points outside
`.course-state/`. Do not test by writing and then searching your filesystem for
the result. Fix the environment, exit any running harness, and relaunch so the
server inherits the corrected variables.

Project-scoped memory has a different location. It belongs to the server's
working directory under `.agentfs/project.db`. Named and global scopes use the
configured base directory. This distinction lets a project carry an isolated
working memory boundary while operators decide where broader scopes live.

## The three-layer mental model

A working installation has three layers:

* **Memory server:** owns scoped durable storage and exposes MCP tools.
* **Project scaffold:** owns launchers, private environment, and tracked
  planning policy.
* **Operating layer:** owns harness commands, agents, and hooks synchronized
  from the installed package.

Diagnosing "Cairnkeep is installed" without naming the layer is ambiguous. A
healthy CLI does not prove the MCP server is registered. A registered server
does not prove a project launcher loads the intended environment. A working
launcher does not prove its generated operating assets are current.

## Local and remote are explicit choices

In local stdio mode, each machine launches its own server and writes to its own
configured storage. Installing the package on a second computer does not point
that computer at the first.

Shared memory requires a separately operated HTTP server plus explicit client
configuration. The database then belongs to the server host, not the client.
This is an operational decision covered later in the series, not an installation
default.

## Recovery and cleanup

If diagnosis fails, preserve the store and inspect configuration before trying
repair commands. Do not delete a database to fix registration or environment
problems.

When the course is complete, exit the harness and server child processes, then
use the repository's bounded cleanup:

```bash
scripts/reset-course-state.sh --yes
```

The script refuses to operate outside fixed course-owned locations.

Next: [Build a reviewed memory lifecycle](../02-memory-knowledge/).
