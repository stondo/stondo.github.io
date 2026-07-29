---
title: "Evaluate the Memory Layer Without Overstating the Result"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Validate and run a deterministic two-pass Cairnkeep experiment, inspect missingness, perform one-capability ablation, and state the result's limits."
summary: "An evaluation framework can create precise reports and still prove very little. The final lesson exercises Cairnkeep's coordinator with an offline fixture and labels every claim by evidence scope."
tags: [cairnkeep, evaluation, experimentation, coding-agents, measurement]
categories: [engineering, ai]
keywords: [agent evaluation, memory ablation, deterministic experiment, Cairnkeep eval]
params:
  author: "stondo"
  toc: true
---

## Measurement starts with a bounded claim

The question "does memory improve coding agents?" is too broad for one
experiment. Results depend on tasks, harness, model, prompts, repository state,
verification, and what "improve" means.

Cairnkeep's evaluation coordinator helps define and preserve those boundaries.
It does not choose a harness or model, and it does not turn fixture output into
evidence of product quality.

This final course exercise uses a deterministic offline adapter. Its purpose is
to verify experiment mechanics: validation, two-pass execution, independent
verification, missingness, reports, retention, and one-capability ablation.

## Validate without executing

Start at the evaluation checkpoint:

```bash
git switch --detach course-07-evaluation
export CAIRN_EVAL=1
output="$PWD/.agentfs/eval/experiments"
core=$(node scripts/locate-cairnkeep-core.mjs)
```

Use the installed package's digest-bound fixtures:

```bash
"$core/bin/cairn" eval validate \
  --task-set "$core/examples/eval/task-set.json" \
  --adapter "$core/examples/eval/adapter.json" \
  --output "$output" --json
```

Validation must not invoke the adapter or create an experiment. It checks the
task set, adapter contract, package version and digest binding, output boundary,
and estimated work before execution.

This separation matters for live adapters. A validation command should not
spend tokens, modify a repository, or contact a model merely to tell you that
the experiment definition is invalid.

## Run the two-pass experiment

After inspection, run with explicit confirmation:

```bash
"$core/bin/cairn" eval run \
  --task-set "$core/examples/eval/task-set.json" \
  --adapter "$core/examples/eval/adapter.json" \
  --output "$output" --seed course-1 --yes --json
```

Use the returned experiment identifier:

```bash
"$core/bin/cairn" eval report \
  --experiment EXPERIMENT-ID --json
```

Inspect the full, executed, eligible, paired, and missing populations. The
fresh second run is not redundant. It exposes order effects and distinguishes a
repeat from a continuation of the first state.

The independent verifier assigns task pass or fail. The adapter reports what it
did, but it does not grade itself.

## Preserve missingness

Reports become misleading when unavailable measurements are replaced by zero
or estimated values. If an adapter cannot report a token count, the field must
remain missing. If a pair is ineligible, it must not silently enter an average.

The offline fixture includes designed fields so the framework can be exercised
without a model. Label its output `offline-framework`. It demonstrates that the
coordinator works against the fixture. It does not demonstrate better quality,
lower cost, faster completion, or statistical significance.

## Ablate one capability

Estimate a treatment that disables memory search:

```bash
"$core/bin/cairn" eval ablate --disable memory.search \
  --task-set "$core/examples/eval/task-set.json" \
  --adapter "$core/examples/eval/adapter.json" \
  --output "$output" --seed course-1 --json
```

Confirm that exactly one capability changes and record both configuration
digests. Then repeat with `--yes` to execute.

The fake adapter does not use memory, so a null or designed difference is the
expected result. That is valuable: the exercise catches a pipeline that invents
an effect where none can exist.

A real claim would require representative tasks, a real adapter, controlled
state, repeated runs, appropriate metrics, and a verifier independent of the
system being evaluated.

## Retention and deletion

Inspect retention operations before applying them:

```bash
"$core/bin/cairn" eval prune --older-than-days 0 --dry-run --json
"$core/bin/cairn" eval delete \
  --experiment EXPERIMENT-ID --dry-run --json
```

The course's bounded cleanup can remove the disposable evaluation store after
inspection.

## The product boundary

Cairnkeep owns durable memory and context infrastructure plus the evaluation
coordinator. It does not own the coding harness, model inference loop, task
policy, human approvals, or repository governance.

A future bounded meta-agent configuration loop exists as a design contract, not
a released command. It should not be demonstrated or advertised as current
functionality.

That boundary is the final lesson of the series. Durable context is valuable
when it is inspectable, optional features are explicit, and evidence is allowed
to say only what the experiment actually measured.

Return to the [Cairnkeep series index](../).
