# Episode 02: Build a reviewed memory lifecycle

**Target:** 14 minutes

**Checkpoint:** `course-02-memory`

**Article:** `content/cairnkeep/02-memory-knowledge.md`

## Recording setup

- Prepare a clean clone and isolated `.ai/.env`.
- Ensure the store contains no previous course memory.
- Keep the two canonical Trail Ledger documents open.

## 00:00 - Hook

**Say:** "A transcript is not a memory system. Useful memory is smaller,
scoped, reviewed, and correctable without erasing history."

## 00:30 - Outcome

**Say:** "We will remember one convention, prove it survives a new session,
supersede a changed belief, and compare memory with a tracked project wiki."

## 01:00 - Verify the boundary

**Do:** Show the checkpoint, run `cairn doctor` and `cairn memory path`, then
launch through the project script.

## 02:10 - Read canonical sources

**Do:** Ask the harness to inspect the requirements and decision document.

**Say:** "The repository remains authoritative. Memory will hold only the
concise convention that helps future work."

## 03:20 - Remember and recall

**Do:** Run `/remember Course convention: Trail Ledger uses UTC timestamps in
examples.` Then run `/recall Trail Ledger timestamps`.

**Point out:** Key, scope, value, and provenance.

## 05:15 - Prove persistence

**Do:** Exit the harness on screen, relaunch it, and repeat the recall.

**Say:** "Do not cut this restart from the recording. It is the persistence
proof."

## 06:30 - Review and supersede

**Do:** Run `/memory-review`, accept only the synthetic convention, create the
course correction, and inspect history.

**Say:** "Supersession names the current record without pretending the earlier
belief never existed."

## 09:00 - Derived wiki

**Do:** Run `/wiki-ingest docs`, `/wiki-query bounded status`, and `/wiki-lint`.
Show the tracked `.planning/wiki` files.

**Say:** "Memory optimizes later recall. The wiki is a reviewable derived view
with citations. Neither replaces the canonical source."

## 11:20 - Default-off proof

**Do:** Outside the harness, run `cairn trajectory list --json` and
`cairn artifact list --json`.

**Point out:** No evidence was silently captured.

## 12:30 - Recovery and recap

**Say:** "If the store path is unexpected, exit without deleting anything,
repair `.ai/.env`, and relaunch. We now have selective, reviewed memory and a
tracked knowledge view. Next we test review workflows against code whose happy
path tests pass."

## Description links

- Series article 02
- Course memory lab
- Course `course-02-memory` tag
