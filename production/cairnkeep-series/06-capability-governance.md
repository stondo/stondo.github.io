# Episode 06: Govern optional capabilities explicitly

**Target:** 11 minutes

**Checkpoint:** `course-06-governance`

**Article:** `content/cairnkeep/06-capability-governance.md`

## Recording setup

- Load only `.ai/course-governance.env.example`.
- Prepare a slide showing default, inherited policy, project override, and
  strict process constraint.
- Ensure the harness can be exited and restarted twice without editing.

## 00:00 - Hook

**Say:** "Once optional features have defaults, distribution policy, project
overrides, and process constraints, an environment search no longer answers a
simple question: what is actually enabled?"

## 00:35 - Outcome

**Say:** "We will inspect effective state, disable and reset an operating
workflow, remove an MCP tool after restart, and verify minimal callback
logging."

## 01:00 - Status and digest

**Do:** Run `cairn capabilities list` and `cairn capabilities status --json`.

**Point out:** Owner, effective state, source, and configuration digest.

**Say:** "The digest identifies state. It is not an approval or quality score."

## 02:40 - Operating capability

**Do:** Disable `context.explore`, inspect status, and reset it.

**Say:** "Operating workflows read the new state on their next invocation.
Reset removes the project override and returns to inherited state. Reset does
not mean enable."

## 04:30 - MCP tool boundary

**Do:** Disable `memory.search`. Before restart, explain that the running schema
is unchanged. Exit and restart, then confirm search is absent while memory read
remains healthy.

**Do:** Reset `memory.search`, restart again, and confirm it returns.

## 07:30 - Precedence model

**Show:** Reveal the prepared precedence slide.

**Say:** "The status output should explain which layer wins. This is what makes
fleet drift diagnosable rather than mysterious."

## 08:30 - Logging boundary

**Do:** Enable capability logging, inspect state, and reset it.

**Say:** "Logging state does not grant trajectory consent. When independent
capture consent exists, callback records contain identity, timing, outcome, and
state, but not prompts, arguments, results, or memory values."

## 10:00 - Recap

**Say:** "We made ownership, precedence, and restart semantics observable. The
last episode uses those configuration identities in a deterministic evaluation
without claiming the fixture proves agent improvement."

## Description links

- Series article 06
- Capability contract documentation
- Course `course-06-governance` tag
