---
title: "Cairnkeep: Durable Context for Coding Agents"
description: "An eight-part, hands-on series about durable memory, project knowledge, review workflows, operations, evidence, governance, and evaluation for coding agents."
draft: true
params:
  author: "stondo"
---

Coding agents are effective inside a session, but projects outlive sessions,
models, and tools. This series builds a durable context layer one capability at
a time, starting with an ordinary Node.js project and ending with a bounded,
inspectable evaluation.

Every article uses the public
[Trail Ledger course repository](https://github.com/cairnkeep/cairnkeep-course-labs).
Its Git tags provide stable starting points, so readers can reproduce the same
exercise instead of copying fragments from screenshots.

## The series

1. [Why coding agents need durable project memory](00-why-durable-context/)
2. [Install Cairnkeep and know where every byte goes](01-install-storage/)
3. [Build a reviewed memory lifecycle](02-memory-knowledge/)
4. [Find real defects with repository review and security audit](03-review-security/)
5. [Operate Cairnkeep safely across stores, machines, and integrations](04-operations-topology/)
6. [Capture session evidence without turning everything into truth](05-evidence-typed-memory/)
7. [Govern optional capabilities explicitly](06-capability-governance/)
8. [Evaluate the memory layer without overstating the result](07-evaluation-boundaries/)

## Safety boundary

The exercises use only fictional data. The course directs all disposable state
into the clone, never points at a real remote service, and includes a bounded
cleanup script. Optional capture and network integrations remain disabled until
an exercise explicitly enables them.
