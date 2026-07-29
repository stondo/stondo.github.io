# Episode 05: Capture evidence without turning everything into truth

**Target:** 15 minutes

**Checkpoint:** `course-05-evidence`

**Article:** `content/cairnkeep/05-evidence-typed-memory.md`

## Recording setup

- Begin with no course trajectory or artifact records.
- Prepare the three public environment profiles in separate editor tabs.
- Use only the synthetic `unknown item` task.
- Keep the trajectory output large enough to inspect redacted fields on screen.

## 00:00 - Hook

**Say:** "A session log can help explain what happened. It can also retain too
much and be mistaken for truth. Cairnkeep separates evidence, notes, artifacts,
and reviewed memory so each has a different trust level."

## 00:40 - Prove default-off

**Do:** Start and exit a short session with `.ai/course.env.example`, then run
`cairn trajectory list --json` and `cairn artifact list --json`.

**Point out:** No session evidence appeared.

## 02:00 - Enable bounded capture

**Do:** Load `.ai/course-evidence.env.example`, source it, inspect the redaction
configuration, and run `cairn sync --apply`.

**Say:** "This profile writes only beneath the disposable clone. External note
enrichment and artifact HTTP remain disabled."

## 03:30 - Synthetic session

**Do:** Launch the harness, reproduce `unknown item: course-missing`, inspect
the implementation, resolve the task, optionally compact once, and exit.

## 06:00 - Inspect trajectory and note

**Do:** Run trajectory list and show for the session ID. Highlight bounded
events and omitted reasoning.

**Do:** Distill the session, search for the synthetic error, and run notes
doctor.

**Say:** "The deterministic note is a candidate, not reviewed memory. Model
enrichment would be a separate network decision."

## 09:00 - Inspect artifacts

**Do:** List artifacts and run prune with `--dry-run`.

**Say:** "An immutable artifact may contain stale or adversarial text. It is
evidence, not trusted instructions."

## 10:30 - Typed memory and import

**Do:** Exit, load `.ai/course-typed.env.example`, and restart. Write the
synthetic typed knowledge node and search with hard type and tag filters.

**Do:** Import the provided envelope as dry run, apply it, replay its import ID,
and demonstrate conflict policy `reject`.

**Point out:** Dry run writes nothing, replay is idempotent, and reject does not
overwrite history.

## 13:40 - Cleanup and recap

**Do:** Show `scripts/reset-course-state.sh --yes` but apply only after all
evidence has been inspected.

**Say:** "Consent includes storage, inspection, retention, and deletion. Next
we govern optional capabilities and verify exactly when state changes take
effect."

## Description links

- Series article 05
- Course safety boundary
- Course `course-05-evidence` tag
