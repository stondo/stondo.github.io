# Episode 00: Why coding agents need durable project memory

**Target:** 8 minutes

**Checkpoint:** `course-00-app`

**Article:** `content/cairnkeep/00-why-durable-context.md`

## Recording setup

- Open a clean Trail Ledger clone at `course-00-app`.
- Prepare one slide with three boxes: canonical sources, derived knowledge,
  durable memory.
- Do not configure Cairnkeep yet.

## 00:00 - Hook

**Say:** "A coding agent can solve a difficult project question today and
repeat the complete investigation tomorrow. The model did not forget a fact it
was trained on. The project failed to preserve a reviewed result in a reusable
form."

**Show:** Two empty session windows labeled Session 1 and Session 2.

## 00:35 - Outcome

**Say:** "We will use a tiny application to identify what should survive a
session, what belongs in Git, and what should not be retained at all."

## 01:00 - Establish the baseline

**Do:** Show `git describe --tags --exact-match`, `git status --short`, and
`npm test`.

**Say:** "Trail Ledger is synthetic and deliberately small. The code is not the
product. It is a stable subject for every lesson in the series."

## 02:00 - Repeated investigation

**Do:** Ask the agent which statuses are valid, where validation occurs, and
which timestamp convention examples use.

**Point out:** The agent reads the same source files needed to answer each
question.

**Say:** "A second session can reconstruct this answer, but reconstruction is
not the same as durable project learning."

## 03:30 - Mental model

**Show:** Reveal the three boxes.

**Say:** "Code, tests, decisions, and maintained documentation are canonical.
A wiki or graph is a derived, reviewable view. Durable memory stores a concise
reviewed convention, decision, or pitfall for later recall. None of these
should silently replace another."

## 05:10 - Why CLI plus MCP

**Say:** "The CLI handles installation, diagnosis, backup, and maintenance. MCP
gives supported coding harnesses a shared typed memory contract. The ordinary
stdio server is local and does not imply a remote network service."

## 06:15 - Boundary

**Show:** A list labeled Off unless enabled: remote storage, RAG, trajectories,
artifacts, typed memory, evaluation.

**Say:** "Cairnkeep does not need to record every conversation. Optional
capture and network paths stay off until an operator chooses them."

## 07:10 - Recap

**Say:** "The design goal is selective, scoped, correctable memory around
canonical project sources. In the next episode we install the complete local
workflow and verify its exact storage path before writing anything."

## Description links

- Series article 00
- Cairnkeep repository
- Course repository and `course-00-app` tag
