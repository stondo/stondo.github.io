# Episode 03: Find real defects with repository review and security audit

**Target:** 14 minutes

**Checkpoint:** `course-03-quality`

**Article:** `content/cairnkeep/03-review-security.md`

## Recording setup

- Use the original vulnerable fixture and do not open the solution branch.
- Prepare a second working tree for the repair so the before state remains
  visible.
- Never substitute a real repository for the public fixture.

## 00:00 - Hook

**Do:** Run the fixture tests and show they pass.

**Say:** "This green suite misses filesystem traversal, authorization, and
output-encoding defects. Passing expected-input tests is not security evidence."

## 00:40 - Outcome

**Say:** "We will turn agent findings into reproducible behavior and
failing-before tests, then review the repair again."

## 01:10 - Repository review

**Do:** Run `/repo-review fixtures/review-target`.

**Say before execution:** "Every accepted finding must identify a file and
line, reachable condition, impact, and missing regression test."

**Point out:** One concrete containment finding rather than scrolling through
the entire report.

## 04:00 - Security audit

**Do:** Run `/security-audit fixtures/review-target`.

**Say:** "The target is explicit. I merge duplicate root causes across the two
reports instead of treating tool output volume as confidence."

## 06:00 - Reproduce

**Do:** Demonstrate one adversarial input for each root cause. Keep payloads
limited to the synthetic fixture.

**Say:** "A reproducible behavior upgrades a plausible finding into evidence
about this version of this target."

## 08:00 - Regression-first repair

**Do:** Add and run one negative test. Show it failing, apply the smallest
repair, and show it passing with the complete suite.

**Point out:** Authorization occurs before access, paths remain inside the
allowed root, and output is encoded for its context.

## 11:00 - Review again

**Do:** Repeat the bounded review and classify fixed, residual, and untested
risk. Compare with `solutions/review-target` only now.

## 12:20 - Memory boundary

**Say:** "I can remember a reviewed general pattern. I do not store the source
dump, report transcript, real exploit data, or a claim that the whole
application is secure."

## 13:10 - Recap

**Say:** "Review workflows create hypotheses. Reproduction and regression
tests create durable evidence. Next we operate the resulting memory safely
across backup, machines, and optional integrations."

## Description links

- Series article 03
- Vulnerable fixture warning
- Course `course-03-quality` tag
- Solution branch, labeled as post-exercise material
