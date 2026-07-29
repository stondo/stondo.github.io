# Cairnkeep article and video production pack

These scripts pair with the unpublished articles under `content/cairnkeep/`.
Each recording starts from a clean disposable clone of
`cairnkeep-course-labs` at the named tag.

## Production order

| Episode | Article | Checkpoint | Target |
|---|---|---|---:|
| 00 | Why durable context | `course-00-app` | 8 min |
| 01 | Install and storage | `course-01-bootstrap` | 12 min |
| 02 | Memory and knowledge | `course-02-memory` | 14 min |
| 03 | Review and security | `course-03-quality` | 14 min |
| 04 | Operations and topology | `course-04-operation` | 15 min |
| 05 | Evidence and typed memory | `course-05-evidence` | 15 min |
| 06 | Capability governance | `course-06-governance` | 11 min |
| 07 | Evaluation and boundaries | `course-07-evaluation` | 16 min |

## Recording rules

1. Rehearse every command from a fresh clone before recording.
2. Show `git status --short` and the checkpoint at the start.
3. Use only synthetic course data and isolated course environment profiles.
4. Keep credentials, unrelated repositories, shell history, notifications, and
   other MCP server names off screen.
5. Say the intent before running a command and pause on the relevant output.
6. Include one failure or recovery path instead of editing it out.
7. Publish the article and video together and link the same checkpoint.
