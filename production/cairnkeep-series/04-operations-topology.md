# Episode 04: Operate Cairnkeep across stores, machines, and integrations

**Target:** 15 minutes

**Checkpoint:** `course-04-operation`

**Article:** `content/cairnkeep/04-operations-topology.md`

## Recording setup

- Install `sqlite3` and use only the course-owned store.
- Prepare a simple two-machine topology diagram.
- Do not configure a real RAG service, remote memory server, or credential.
- If Podman is unavailable, record the container explanation from inspected
  help output and clearly label it as not executed.

## 00:00 - Hook

**Say:** "Once memory is useful, it is data you can lose. Before adding more
features, we need to locate it, snapshot it consistently, restore it, and draw
every network boundary."

## 00:35 - Outcome

**Say:** "We will restore a disposable backup and then distinguish local
stdio, remote memory, document RAG, context exploration, containers, and
managed overlays."

## 01:00 - Locate and export

**Do:** Show the checkpoint, load `.ai/course.env.example`, run
`cairn memory path`, and create `.course-state/backups`.

**Do:** Run `cairn memory export
.course-state/backups/global-memory.tgz`.

**Say:** "The export is SQLite-aware and includes WAL state. A live copy of
only the main database file is not equivalent."

## 03:20 - Restore rehearsal

**Do:** Import only into the disposable course store and run `cairn doctor`.

**Say:** "Import replaces databases and preserves backups of what it replaces.
It is not a merge. Production restore should be rehearsed elsewhere first."

## 05:00 - Two-machine diagram

**Show:** In stdio mode, draw one memory-server child and local store on each
client.

**Say:** "Installation never discovers a shared server. Remote storage requires
an authenticated HTTP server and explicit URL and token on every client. The
database belongs to the server host."

## 07:00 - Optional RAG and exploration

**Do:** Run the doctor command with document-RAG variables unset. Show the
configuration error for `/context-explore` with its binary unset.

**Say:** "Both integrations can add value, but removing either must leave core
memory healthy. Automatic prompt-time exploration is another explicit opt-in."

## 09:20 - Containers

**Do:** Show `cairn-container --help`, inspect stdio volume mode and workspace
sandbox mode, and display mounts before running anything.

**Say:** "A named volume persists. Sandbox mode copies the repository into a
volume. Shared mode mounts the host checkout read/write and is not host
isolation."

## 11:40 - Managed overlays

**Show:** Core, overlay wrapper, machine-private configuration, project profile
lock.

**Say:** "An overlay pins tested core and applies inspectable policy. It must
not hide a fork or silently change upstream storage defaults. Secrets stay in
the overlay or machine configuration, never public core."

## 13:20 - Boundary and recap

**Say:** "For every integration, name the process, host, data, credential,
verification, and reversal path. Next we enable bounded session evidence and
inspect every retained layer."

## Description links

- Series article 04
- Cairnkeep storage and privacy guides
- Course `course-04-operation` tag
