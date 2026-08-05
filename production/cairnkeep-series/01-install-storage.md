# Episode 01: Install Cairnkeep and know where every byte goes

**Target:** 12 minutes

**Checkpoint:** `course-01-bootstrap`

**Article:** `content/cairnkeep/01-install-storage.md`

## Recording setup

- Use a disposable user profile or VM with Node.js 22 or newer.
- Remove prior Cairnkeep harness registration from that profile.
- Keep private environment variables and unrelated MCP servers off screen.

## 00:00 - Hook

**Say:** "A successful package install does not prove your coding harness can
use Cairnkeep, and a running server does not prove you know where it writes. We
will verify every boundary before the first memory."

## 00:35 - Outcome

**Show:** CLI, MCP server, project scaffold, operating layer, storage path.

**Say:** "At the end, all five have an observable check."

## 01:05 - Runtime and CLI

**Do:** Run `node --version`, `npm --version`, the documented pinned install,
`command -v cairn`, and `cairn version`.

**Point out:** The resolved binary and version.

## 02:40 - Course checkpoint

**Do:** Show the exact Git tag, clean status, and passing app tests.

**Say:** "I am not using a personal project because installation testing should
not risk existing memory or source."

## 03:30 - Bootstrap

**Do:** Run `cairn bootstrap "$PWD"`, then show the generated `.ai`,
`.planning`, and `.agentfs` boundaries.

**Say:** "Bootstrap creates the project layer. It does not discover a remote
server or private policy."

## 05:00 - Isolated environment

**Do:** Copy `.ai/course.env.example` to `.ai/.env` and show only the safe path
variables.

**Point out:** Named and global state remain inside `.course-state/`.

## 06:00 - Register and synchronize

**Do:** Follow the current harness registration command, verify the stable name
`cairn-memory`, then run `cairn sync --apply` and `cairn sync --check`.

**Say:** "Registration supplies MCP tools. Sync supplies the matching commands,
agents, and hooks."

## 08:20 - Prove storage

**Do:** Run `cairn doctor`, `cairn sync --check`, and `cairn memory path`.

**Pause:** Keep the path result visible.

**Say:** "Project scope is tied to `.agentfs/project.db`. Named and global
scopes follow the configured base directory."

## 09:45 - Recovery demonstration

**Do:** Temporarily show an intentionally incorrect course path without
starting the harness, then restore the example.

**Say:** "If the path is wrong, stop. Fix the environment and restart the
server. Do not write a test memory into an unknown store and do not delete a
database to solve routing."

## 11:00 - Recap

**Say:** "The installation is complete only when the CLI, registration,
project environment, operating layer, and storage path agree. Next we will
write one reviewed convention and recall it from a new session."

## Description links

- Series article 01
- Current Cairnkeep installation documentation
- Course `course-01-bootstrap` tag
