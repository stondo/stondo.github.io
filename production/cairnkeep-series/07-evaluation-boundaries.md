# Episode 07: Evaluate without overstating the result

**Target:** 16 minutes

**Checkpoint:** `course-07-evaluation`

**Article:** `content/cairnkeep/07-evaluation-boundaries.md`

## Recording setup

- Start with an empty `.agentfs/eval` directory.
- Use only the package-owned offline task set and adapter.
- Prepare a slide labeled `offline-framework`, not product benchmark.
- Keep one experiment ID ready only as a fallback for retakes. Prefer the ID
  created on screen.

## 00:00 - Hook

**Say:** "A precise report can still support an invalid conclusion. Today we
will verify an evaluation framework while carefully refusing to claim it proves
that memory improves coding agents."

## 00:40 - Outcome

**Say:** "We will validate without execution, run two fresh passes, inspect
missingness, disable one capability, and state exactly what the result cannot
show."

## 01:10 - Locate fixtures

**Do:** Show the checkpoint, export `CAIRN_EVAL=1`, set the contained output,
and locate the installed Cairnkeep core.

**Say:** "The fixtures are package-owned and digest-bound, so the experiment
records the exact definition it ran."

## 02:30 - Validate

**Do:** Run the complete `cairn eval validate` command from the article.

**Point out:** No experiment was created and no adapter was invoked.

**Say:** "Validation must not spend model tokens or modify a repository."

## 04:30 - Two-pass run

**Do:** Run with seed `course-1` and explicit `--yes`, then open the report.

**Point out:** Run 1, fresh Run 2, independent verifier, and configuration
identity.

## 07:00 - Populations and missingness

**Show:** Full, executed, eligible, paired, and missing populations.

**Say:** "Unavailable values remain missing. They do not become zero. Ineligible
pairs do not silently enter an average."

## 09:00 - One-capability ablation

**Do:** Estimate `eval ablate --disable memory.search` without `--yes`. Compare
both configuration digests and confirm exactly one capability changes.

**Do:** Execute only after the estimate is understood.

**Say:** "This adapter never uses memory, so no meaningful quality effect can
exist. A null or designed difference is the expected result."

## 12:00 - Claim boundary

**Show:** The `offline-framework` slide.

**Say:** "This proves that the coordinator executes its deterministic fixture,
preserves state identity, and reports missingness. It does not prove quality,
cost, latency, causality, or statistical significance for a real agent."

## 13:30 - Retention

**Do:** Run prune and delete with `--dry-run`.

**Say:** "Experiment evidence also needs an inspectable retention and deletion
path."

## 14:30 - Product boundary and close

**Say:** "Cairnkeep owns memory and context infrastructure plus this evaluation
coordinator. It does not own the harness, model, task policy, or human approval.
The proposed meta-agent loop is a design contract, not a released feature."

**Say:** "Across this series, we started with one repeated investigation and
ended with scoped memory, explicit operations, bounded evidence, inspectable
governance, and honest measurement."

## Description links

- Series article 07
- Evaluation documentation
- Course `course-07-evaluation` tag
- Complete Cairnkeep feature guide
