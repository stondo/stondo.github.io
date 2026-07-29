---
title: "Find Real Defects with Repository Review and Security Audit"
date: 2026-07-29T00:00:00+02:00
lastmod: 2026-07-29T00:00:00+02:00
draft: true
description: "Use Cairnkeep review workflows against a deliberately vulnerable fixture and turn findings into reproducible tests instead of trusting plausible reports."
summary: "Happy-path tests can pass while traversal, authorization, and output-encoding bugs remain. This lesson uses a synthetic target to distinguish findings, proof, repairs, and durable lessons."
tags: [cairnkeep, code-review, security, testing, coding-agents]
categories: [engineering, security]
keywords: [AI code review, repository review, security audit, regression tests, Cairnkeep]
params:
  author: "stondo"
  toc: true
---

## A green test suite is not a security argument

Code review with a coding agent can uncover serious defects, but a confident
finding is still a hypothesis. The useful workflow connects each claim to a
file and line, demonstrates behavior with a concrete input, and adds a
regression test that fails before the repair.

Cairnkeep provides `/repo-review` and `/security-audit` as structured operating
workflows. They do not certify a repository. They help an agent gather evidence,
separate overlapping findings, and preserve only generalizable lessons after a
human reviews the result.

## Use the intentionally vulnerable fixture

Switch a disposable course clone to the quality checkpoint:

```bash
git switch --detach course-03-quality
node --test fixtures/review-target/report-viewer.test.mjs
```

The tests pass. That is intentional.

Read `fixtures/review-target/README.md`, but do not inspect the solution branch.
The fixture is not a deployable application. It is a compact adversarial input
boundary containing three classes of defect:

* filesystem containment;
* authorization;
* output encoding.

Launch a supported harness through the project `.ai` script.

## Run repository review first

Ask for a bounded target:

```text
/repo-review fixtures/review-target
```

Require every finding to include:

1. a specific file and line;
2. the input or condition that reaches the behavior;
3. the user or system impact;
4. the regression test that is currently missing.

Reject vague statements such as "path handling may be unsafe." A useful
finding explains how an input escapes the expected root, which operation then
uses the escaped path, and how a negative test would prove containment.

Repository review should also identify ordinary correctness and maintainability
risks. Keep these separate from vulnerabilities so severity does not become a
synonym for inconvenience.

## Run the security audit against the same boundary

Next run:

```text
/security-audit fixtures/review-target
```

The security workflow should treat file names, report contents, and caller
identity as untrusted inputs. Compare its output with the repository review.
Merge duplicate findings rather than counting the same root cause twice.

For this fixture, the acceptance bar is concrete:

* demonstrate a path that escapes filesystem containment;
* demonstrate an operation that lacks the required authorization decision;
* demonstrate unescaped content reaching generated output.

The original passing tests are not evidence against any of these behaviors.
They cover only expected inputs.

## Repair with failing-before tests

Create a branch from the checkpoint. Add one negative test per root cause before
changing implementation. Run it and confirm the failure represents the
reported behavior, not a test setup error.

Then make the smallest repair:

* resolve and validate paths against an allowed root;
* enforce authorization before the protected operation;
* encode untrusted content for the output context.

Run the complete fixture test suite after each repair. Finally, repeat both
review workflows and classify findings as fixed, residual, or still untested.

The course includes `solutions/review-target` for comparison after the exercise:

```bash
git show solutions/review-target
```

It is a reference repair, not the only correct implementation.

## Store the lesson, not the vulnerable source

After reviewing the result, `/remember` can preserve a general pattern such as:

```text
Validate resolved filesystem paths against the allowed root before opening the file.
```

Do not store a source dump, report transcript, token, exploit payload from a
real system, or a claim that the complete application was audited. Memory
should make the next review better without becoming a shadow vulnerability
database.

## What this workflow can claim

It can claim that a defined version of a defined target was examined, that
specific behaviors were reproduced, and that regression tests cover the
repairs.

It cannot claim complete security, absence of undiscovered defects, or coverage
of code outside the stated target. That boundary makes the result more useful,
not less.

Next: [Operate Cairnkeep safely across stores, machines, and integrations](../04-operations-topology/).
