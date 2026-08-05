---
title: "Govern Optional Capabilities Explicitly"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Inspect, disable, reset, and verify Cairnkeep capabilities while understanding precedence, process restart boundaries, and minimal callback logging."
summary: "Environment flags become difficult to reason about as optional features grow. This lesson uses an inspectable capability contract to make effective state and ownership explicit."
tags: [cairnkeep, governance, configuration, mcp, developer-tools]
categories: [engineering, ai]
keywords: [capability governance, MCP tools, feature flags, Cairnkeep capabilities]
params:
  author: "stondo"
  toc: true
---

## Optional features need an effective-state model

A few environment flags are easy to understand. A larger system may combine
package compatibility defaults, distribution policy, project overrides, and
strict process constraints. At that point, asking "is this feature enabled?"
requires more than searching one file.

Cairnkeep's capability contract makes the effective answer inspectable. It also
distinguishes two kinds of change:

* an operating workflow reads state on its next invocation;
* an MCP tool is part of the server schema and requires a server restart when
  exposure changes.

## Inspect the initial state

Start from the governance checkpoint:

```bash
git switch --detach course-06-governance
cp .ai/course-governance.env.example .ai/.env
set -a
source .ai/.env
set +a
cairn capabilities list
cairn capabilities status --json
```

The status output identifies each capability, its owner, its effective state,
and the source of that state. Save the configuration digest for comparison.

The digest identifies a configuration. It is not an approval, a signature, a
security score, or evidence that the capability works correctly.

## Disable and reset an operating capability

Use context exploration as an operating example:

```bash
cairn capabilities disable context.explore --json
cairn capabilities status --json
cairn capabilities reset context.explore --json
```

The disable operation creates a project override. The workflow sees it on its
next invocation. Reset removes that override and reveals the inherited state.

Reset does not mean enable. If the inherited policy disables the capability,
reset returns to disabled. This distinction is essential when a managed
distribution owns the higher-level policy.

## Disable an MCP tool

Now disable semantic memory search:

```bash
cairn capabilities disable memory.search --json
```

An already running memory server still exposes the schema it built at startup.
Exit and restart the harness. Confirm `memory_search` is absent while ordinary
memory read remains healthy.

Then remove the override:

```bash
cairn capabilities reset memory.search --json
```

Restart again and confirm the tool returns. This proves both the state change
and its process boundary. Testing only the configuration file would miss half
the behavior.

## Understand precedence

The effective capability follows a deliberate order:

1. compatibility default provides safe behavior for installations without a
   contract;
2. distribution or user policy can establish an inherited baseline;
3. project overrides can narrow or select behavior;
4. a strict process override can impose the final constraint.

The status command should explain the winning value rather than merely print
it. That makes configuration drift diagnosable across projects and machines.

## Callback logging does not grant capture consent

Exercise the logging state:

```bash
cairn capabilities logging enable --json
cairn capabilities status --json
cairn capabilities logging reset --json
```

Logging state alone cannot enable trajectory capture. The independent capture
flag must also grant local consent. When both apply, a callback record is
limited to capability identity, owner, timing, outcome, and state identity. It
must not contain prompts, tool arguments, results, or memory values.

This is a useful general rule: governance telemetry should prove which control
was applied without becoming a second copy of the controlled data.

## What to verify in a real distribution

For each managed capability, verify:

* who owns the default;
* whether a project may override it;
* whether the state changes tool exposure or invocation behavior;
* which process must restart;
* whether ordinary core memory survives disabling it;
* how reset behaves under inherited policy.

Capability governance is not required for a simple local Cairnkeep setup. It
becomes useful when optional features and managed policy make effective state
otherwise difficult to explain.

Next: [Evaluate the memory layer without overstating the result](../07-evaluation-boundaries/).
