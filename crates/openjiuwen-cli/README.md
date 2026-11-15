# openjiuwen-cli

The `openjiuwen` executable — the process entry point of the Rust openJiuwen
agent SDK.

The binary is a thin process boundary. It owns the operating-system concerns
(`.env` loading, `tracing` installation, clap's exit codes, the stderr
diagnostic for a failed run), the outer command grammar, the `auto-harness`
command group, which the harness has no words for, and the harness-surface
commands `lsp`, `cron`, `team`, `watch` and `sessions`, which drive the harness
tools, protocols and persistent-agent stores from a terminal. Everything else —
option resolution, the one-shot runner, the interactive REPL, permission
prompts and session storage — lives in `openjiuwen_harness::cli`.

```console
$ openjiuwen run "summarise src/lib.rs"
$ openjiuwen chat
$ openjiuwen auto-harness run --task "fix the parser" --dry-run
$ openjiuwen lsp diagnostics --file src/lib.rs
$ openjiuwen cron list
$ openjiuwen team --socket /tmp/team.sock members
$ openjiuwen watch --list
$ openjiuwen sessions list
$ openjiuwen run "and now?" --continue
```

## Build and run

The workspace builds offline; every dependency is already in the local cargo
cache.

```console
# from the repository root
cargo build --offline                     # debug binary: target/debug/openjiuwen
cargo build --release -p openjiuwen-cli   # release binary: target/release/openjiuwen
cargo run -p openjiuwen-cli -- --help
cargo run -p openjiuwen-cli -- run "hello"
cargo run -p openjiuwen-cli -- auto-harness run --task "tidy docs" --dry-run
cargo run -p openjiuwen-cli -- cron list
cargo run -p openjiuwen-cli -- watch --list
```

Definition of done for changes in this crate:

```console
cargo fmt -p openjiuwen-cli
cargo test -p openjiuwen-cli --offline
cargo clippy -p openjiuwen-cli --all-targets --offline -- -D warnings
cargo metadata --offline --no-deps
```

## Command reference

```
openjiuwen [GLOBAL OPTIONS] [COMMAND]
```

Without a command the binary chooses one from the terminal: an interactive
terminal starts `chat`, anything else behaves like `run` with the prompt read
from stdin. That dispatch belongs to the harness.

### `run [PROMPT] [-f|--output-format <FORMAT>] [--worktree <NAME>] [--resume <ID>|--continue]`

Runs a single prompt and exits.

| Argument / flag | Meaning |
| --- | --- |
| `PROMPT` | Prompt text. Omitted or `-` reads stdin to end of input and joins the lines. |
| `-f`, `--output-format` | `text` (default), `json` or `stream-json`. |
| `--worktree <NAME>` | Create (or reuse) a git worktree of the workspace's repository and run the prompt inside it. |
| `--resume <ID>` | Continue the session `ID` (a CLI session id, a persistent session id or an execution id) instead of starting a new one. |
| `--continue` | Continue the most recent resumable session; conflicts with `--resume`. |

Output shapes:

- `text` — chunks are rendered as they stream, as plain text: a one-shot run
  uses a non-interactive I/O handle, so it never emits ANSI styling and
  `--no-color` / `NO_COLOR` change nothing here (they matter for `chat`).
- `json` — nothing is printed while streaming; one pretty-printed object at the
  end with the keys `answer`, `result_type`, `usage` and `session_id`.
- `stream-json` — one JSON object per streamed chunk
  (`{"type", "index", "payload"}`), then a terminal object which is the same
  `answer`/`result_type`/`usage`/`session_id` object with `"type": "result"`
  added.

An agent turn may call tools, so a run is not limited to one model request; the
answer is whatever the agent finished with.

`--worktree <NAME>` creates (or reuses) a git worktree of the repository that
contains the resolved workspace — `-w`, then `OPENJIUWEN_WORKSPACE`, then
`settings.json`, then `<home>/workspace`, with a relative value resolved against
the working directory — at `<repo-root>/.worktrees/<slug>`, on branch
`worktree-<slug>`. The run's workspace becomes the worktree path; everything
else (option resolution, permissions, sessions) is unchanged. The creation is
announced on stderr as one line, `worktree: <slug> -> <path> (branch <branch>)`,
so stdout stays the result stream. A workspace outside a git repository, or a
host without git, is a typed `config` error before any model is built.

### `chat [--watch] [--worktree <NAME>] [--resume <ID>|--continue]`

Starts the interactive REPL on a terminal. Beyond the global options it takes
flags of its own; without them the REPL behaves exactly as before.

| Flag | Meaning |
| --- | --- |
| `--watch` | Watch the workspace and report skill, spec, instruction and MCP reloads on stderr while the REPL runs. |
| `--worktree <NAME>` | Create (or reuse) a worktree as `run --worktree` does and chat inside it. |
| `--resume <ID>` | Seed the REPL with the transcript of session `ID` (the same ids `run --resume` accepts). |
| `--continue` | Seed the REPL with the most recent resumable session; conflicts with `--resume`. |

The REPL is wired by the harness and provides the in-session slash commands
`/exit` (`/quit`), `/new`, `/clear`, `/mode`, `/help`, `/status`, `/cost`,
`/compact`, `/sessions` and `/<skill>`, plus `!<command>` shell passthrough,
persistent input history, and the first-run onboarding wizard when the home
directory has no settings file yet.

`chat --watch` starts the same watcher `openjiuwen watch` uses, with stderr as
its sink: one line per reported update (a debounced change, a reload failure, a
dropped-notification notice or a watcher error), preceded by a `watching N
target(s)` readiness line. The notices are diagnostics only — the running REPL
keeps the instructions, skills and MCP configuration it started with; a restart
applies the change. The conversation on stdout is untouched.

### Resuming a session (`--resume <ID>` / `--continue`)

Both flags resolve their id through the same catalogue `openjiuwen sessions`
lists: the CLI session records under `<home>/sessions`, the persistent-agent
execution journals under the workspace's `persistent_agent/logs`, and the
persistent-agent session records under `persistent_agent/sessions`. A
persistent session record wins over a CLI record, which wins over an execution
journal. `--continue` picks the newest usable execution, else the newest
persistent session record, else the newest CLI record.

The restored history is seeded into the turn and one note goes to **stderr**, so
stdout keeps the run's output format:

```text
resuming session cli-1a2b3c4d from the CLI session record (4 messages)
resuming session sess-1 from the persistent session record via execution exec-7 (9 messages restored)
resuming the most recent session exec-7 from execution exec-7 (9 messages restored)
```

`run --resume` continues the resolved session id: the kernel session and the CLI
session record keep that id and the new turns are appended to it. `chat
--resume` cannot adopt it — the harness REPL always mints a fresh `cli-*` id — so
the transcript is seeded into a new REPL session and the note says so:
`… ; the REPL continues as cli-….`

An id no store knows is `error: not_found: unknown session: <id>`; an empty
catalogue is `error: not_found: no resumable session found`; giving both flags at
once is a usage error (exit `2`).

### `auto-harness <COMMAND>`

Auto Harness Agent: optimise the harness framework itself. Ported from Python's
`openjiuwen auto-harness` group.

Global to the group:

| Flag | Meaning |
| --- | --- |
| `--json` | Print machine-readable JSON instead of text (Rust addition; every command keeps its text form as the default). |

#### `auto-harness run`

| Flag | Meaning |
| --- | --- |
| `--task <TEXT>` | Run one manually specified task. Wins over `--task-file`. |
| `--task-file <PATH>` | Read the task list from a JSON file: an array of entries or a single object. Each entry needs `topic`; `description` (string) and `files` (array of strings) are optional and unknown keys are ignored. |
| `--dry-run` | Resolve and print the task list only; implement nothing. |
| `--stage <STAGE>` | Restrict the run to one stage: `assess`, `plan`, `implement` or `verify`. |
| `--no-push` | Accepted for command-line parity only; the Rust port has no push step, so it changes nothing. |
| `--budget <SECS>` | Override the session budget in seconds (also lowers the per-task timeout to 95 % of it). |
| `--goal <TEXT>` | Natural-language optimisation goal for the session. |
| `--pipeline <PIPELINE>` | `meta` (the default when the flag is absent), `extended` or `auto`. |

`--dry-run` writes the resolved list to
`<data-dir>/runs/tasks.json`, prints the same array on stdout, and prints
`[dry-run] task list -> <path>` on stderr. Without `--dry-run` the command
resolves its inputs and then fails with a typed `unsupported` error: session
execution needs a harness evaluation runner the Rust port does not ship (see
*Rust-port differences* below).

#### `auto-harness experience search <QUERY>`

Keyword search over the experience store.

- The query is lowercased and split on whitespace; words shorter than two
  characters are dropped, and a query with no usable word matches nothing.
- A keyword is counted once per field it appears in, over `topic`, `summary`
  and `details`; `outcome` is never searched.
- Hits are scored by hit count plus a recency bonus (1.0 for a fresh record,
  decaying linearly to 0.0 after thirty days) and the best ten are printed,
  newest-and-best first.
- Text output is one line per hit, `[<type>] <topic>: <summary or outcome>`;
  `no matching experiences` when nothing matched (exit code `0`).

#### `auto-harness experience list [--type <TYPE>] [--limit <N>]`

- `--type <TYPE>` filters by the exact, case-sensitive record type (`optimization`,
  `failure`, `insight`, …).
- `--limit <N>` caps the number of rows (default `10`).
- Records are sorted newest-first, **truncated to the limit, and only then
  filtered by type** — a one-row window holding only a failure comes back empty
  when `--type insight` is asked for (Python parity).
- `<workspace>/auto_harness/experience/` is created on first read; a missing
  `experiences.jsonl` is an empty store. Blank and malformed lines are skipped.
- `no experience records` when the selection is empty (exit code `0`).

#### `auto-harness history [--limit <N>]`

Shows the optimisation history: the same store, the same line format and the
same JSON form as `experience list`, without a type filter (default `20` rows).

#### `auto-harness verify-ext [--ext-path <DIR>]`

Validates an extension package (structure + lint) in Python. **Deferred:** it
needs the runtime extension loader (synthetic `sys.modules` hot-loading and an
external linter), so it always exits `1` with
`error: unsupported: runtime extension loader is not implemented in the Rust port: …`.
The path is not validated — the loader is the first thing the command touches.

#### `auto-harness gap-analyze`

Analyses capability gaps against a competing implementation. A placeholder in
both ports (no model is wired in): it prints
`phase 1 placeholder: gap analysis is not wired to a model in this port` and
exits `0`. With `--json` it prints
`{"gaps": [], "placeholder": true, "note": "<the placeholder text>"}`.

### `lsp <ACTION> [OPTIONS]`

Drives the harness `lsp` tool from the terminal: the client, the server table
and the action set belong to `openjiuwen_harness::lsp`, and the command only
translates a command line into one tool call.

| Argument / flag | Meaning |
| --- | --- |
| `ACTION` | `hover`, `definition`, `references`, `completion`, `symbols`, `workspace-symbols`, `diagnostics`, `implementation`, `prepare-call-hierarchy`, `incoming-calls` or `outgoing-calls`. |
| `--file <PATH>` | File to act on, relative to the workspace or absolute. Every action except `workspace-symbols` needs one, and `diagnostics` without one reports the buffered diagnostics of all files; a missing path is the tool's `path is required for '<action>'` error. |
| `--line <N>` | One-based line of the position. |
| `--column <N>` | One-based column of the position (alias `--col`). |
| `--query <TEXT>` | Query text for `workspace-symbols`. |
| `--context-lines <N>` | Context lines to include around a position result. |
| `--server <NAME>` | Built-in server by name: `pyright`, `rust`, `typescript`, `gopls`, `jdtls`. An unknown name is a `validation` error naming that list. |
| `--server-command <PROGRAM>` | Explicit language-server program; wins over `--server`. |
| `--server-arg <ARG>` | Argument for `--server-command`; repeatable. |
| `--language-id <ID>` | LSP language id for `--server-command` (for example `rust`, `python`); without `--server-command` it is a tool-level hint. |
| `--json` | Print the whole tool payload document instead of text. |

A `--file` that does not exist is rejected as `error: not_found:` before a
language server is selected or launched.

Output is the tool's own answer: one payload per action. Text mode prints the
payload's rendered `data.result`; when an action has no textual rendering the
`data` JSON is pretty-printed instead. `--json` prints the entire payload
document (`success`, `data`, `error`, plus the optional extras such as
`extracted_content`) as one object. A payload reporting
`success: false` is a typed `error: tool:` (exit `1`, empty stdout), so a caller
can pipe either form without the two being mixed.

### `cron <COMMAND>`

A host-side view of the same scheduler the agent's `cron` tool uses. Jobs live
at `<workspace>/cron/jobs.json` and the bounded run history at
`<workspace>/cron/runs.json`; the `-w` workspace is created before the store is
opened.

| Flag | Meaning |
| --- | --- |
| `--json` | Group-global: print one pretty JSON document instead of text. |

| Command | Meaning |
| --- | --- |
| `cron list` | List the stored jobs. |
| `cron status` | Report job counts, the due count and the earliest pending run. |
| `cron get <JOB_ID>` | Show one job. |
| `cron create <CRON> --prompt <TEXT> [--description <TEXT>]` | Store a new job from a cron expression: classic five fields, six/seven Quartz fields, or a supported `@` shorthand (`@daily`, …). |
| `cron toggle <JOB_ID>` | Flip a job between enabled and disabled. |
| `cron preview <JOB_ID> [--count <N>]` | Show the next occurrences without mutating the job; `--count` defaults to `5` and is clamped to `1..=20`. |
| `cron run <JOB_ID>` | Record a manual run and advance the job's next occurrence. |
| `cron runs [--job-id <ID>] [--limit <N>]` | Show the run history, newest first; `--limit` defaults to `10`, and `0` means all. |
| `cron cancel <JOB_ID>` | Delete a job and its run records. |

Text output, one item per line:

- `list` / `get` — `<id>\t<cron>\tnext <ts>\t<enabled|disabled>`, or
  `no cron jobs` for an empty store.
- `status` — `N cron job(s): N enabled, N disabled, N due; N run record(s)`,
  plus `next run at <ts>` when an enabled job is pending.
- `create` — `created <job line>`; `toggle` — the new state word followed by the
  job line (`enabled …` / `disabled …`).
- `preview` — one RFC 3339 occurrence per line, or `no upcoming occurrences`.
- `run` — `ran <id>; next run at <ts>`.
- `runs` — `<recorded>\t<job-id>\t<manual|scheduled>\toccurrence <ts>`, or
  `no cron runs`.
- `cancel` — `cancelled <id>`.

`--json` prints exactly one document per command: the job array of `list`, the
`CronStatus` object of `status`, one `CronJob` for `get`/`create`/`toggle`/
`cancel`, one `CronRun` for `run`, the occurrence strings of `preview`, and the
run array of `runs`. Timestamps are RFC 3339 in UTC (`…Z`, fractional seconds
only when present).

An unknown id is `error: not_found: cron job not found: <id>` (exit `1`, empty
stdout); an unparsable expression is an `error: validation:` naming the cron
expression, and stores nothing.

### `team <COMMAND>`

Operates an agent team hosted by **another** process over the
`openjiuwen-agent-teams` attach protocol: one JSON request/response per line on
a Unix domain socket. One process opens one connection and performs exactly one
operation — there is no reconnect and no retry, so a failure is reported as the
failure it is, with its wire kind (a remote `not_found` reaches the exit path as
`error: not_found: …`).

| Flag | Meaning |
| --- | --- |
| `--socket <PATH>` | Attach socket of the hosting team. Falls back to `OPENJIUWEN_TEAM_SOCKET`; with neither set the command is a `validation` error naming both sources. |
| `--json` | Group-global: print one pretty JSON document instead of text. |

Both flags are global to the group, so they may follow the subcommand.

| Command | Meaning |
| --- | --- |
| `team members` | List the roster. |
| `team member <NAME>` | Show one member. |
| `team inbox` | Read and drain the leader's unread inbox. |
| `team send <TO> <CONTENT>` | Send a direct message. |
| `team broadcast <CONTENT>` | Send a message to every member. |
| `team create-task <TITLE> <CONTENT> [--task-id <ID>] [--dep <ID>]…` | Create a board task; `--dep` is repeatable and the team assigns an id when `--task-id` is omitted. |
| `team task list [--status <STATUS>]` | List tasks, optionally filtered: `pending`, `blocked`, `planning`, `in_progress`, `in_review`, `completed` or `cancelled`. |
| `team task claimable` | List the tasks this caller may claim. |
| `team task get <TASK_ID>` | Show one task in full. |
| `team claim <TASK_ID>` | Claim a task for this caller. |
| `team complete <TASK_ID>` | Mark a claimed task completed. |
| `team update <TASK_ID> [--title <TEXT>] [--content <TEXT>]` | Patch a task's title and/or content. |
| `team snapshot` | Print the whole team snapshot (members, tasks, state). |
| `team stop` | Ask the hosting team to stop. |

Text output, one item per line:

- `members` / `member` — `<member_name>\t<role>\t<status>\t<execution>`, plus a
  fifth `<display_name>` column when the roster carries one that differs from
  the member name; an empty roster prints `no team members`.
- `inbox` — the text exactly as the host returned it, or `inbox is empty`.
- `send` / `broadcast` — `<message_id> from <from> to <to|*>` (`*` for a
  broadcast); a payload without those fields is printed as compact JSON.
- `create-task` / `update` — `<task_id>\t<status>\t<title>`; any other payload
  is printed as compact JSON.
- `task list` / `task claimable` — `<task_id>\t<status>\t<title>`, plus
  `\t<assignee>` when the task has an owner; `no tasks` / `no claimable tasks`
  when the board selection is empty.
- `task get` — `task_id`, `title`, `status` and `assignee` rows (`-` when
  unowned), then the task content; an id the host does not know is
  `error: not_found: task not found: <id>`.
- `claim` / `complete` — `claimed <id>` / `completed <id>` when the payload
  carries `ok: true`; a payload carrying `ok: false` — the board's policy
  answer, such as claiming a task somebody else owns — is an
  `error: tool: <host reason>` (exit `1`, empty stdout) in both output modes; a
  payload without `ok` is printed as compact JSON.
- `snapshot` — `team <name> (<state>): N member(s), M task(s)` followed by one
  member line and one task line each; states are `idle`, `running`, `pausing`,
  `paused` and `terminated`.
- `stop` — `stop requested`.

`--json` prints exactly one document per command: the member array of `members`,
one `MemberSnapshot` for `member`, the inbox text, the operation payload of
`send`/`broadcast`/`create-task`/`claim`/`complete`/`update`/`stop`, the task
array of `task list`/`task claimable`, one `TaskDetail` for `task get`, and the
whole `TeamSnapshot` for `snapshot`.

`invoke_tool` exists on the wire and in the client but is not exposed as a
command: the in-repo attach host always answers it with `unsupported`.

### `watch [--json] [--list] [--debounce-ms <MS>] [--max-events <N>]`

The terminal end of the harness workspace watcher: a debounced, coalesced change
stream over the files the harness reloads from — skill libraries,
agent-template/plugin roots, instruction documents (`OPENJIUWEN.md`,
`IDENTITY.md` in the workspace and in the home directory) and the MCP
configurations — plus the reload handlers that re-read them.

| Flag | Meaning |
| --- | --- |
| `--list` | Print the resolved watch target plan and exit without starting a watcher. |
| `--json` | Print one JSON object per update (with `--list`: one pretty document). |
| `--debounce-ms <MS>` | Debounce window in milliseconds (default `250`). |
| `--max-events <N>` | Exit after this many updates; without it the watcher runs until Ctrl-C. |

`--list` is the deterministic, offline plan view: nothing has to exist yet, and
a candidate whose watch root is missing is reported rather than dropped. A
target line is `<kind>\t<path>`, with ` (recursive)` appended for a directory
target; kinds are `skills`, `spec`, `instruction_file`, `mcp_config` and
`other`, and each missing root is printed as `skipped\t<path>`. With `--json`
the document is `{"targets": […], "skipped": […]}`, where each target carries
`kind`, `path`, `label` and `recursive`, and every skipped entry is an absolute
path string.

Watching announces `watching N target(s)` on **stderr** (flushed, so a
supervisor can use it as a readiness signal) and then reports one line per
update on stdout:

- a change — `event <kind> <change> <path>`, with ` <- <previous path>` appended
  after a rename; JSON `{"type": "event", "sequence", "kind", "path", "change",
  "previous_path"?, "label"?}`;
- a reload handler failure — `reload failed [<handler>] <path>: <message>`; JSON
  type `reload_failed`;
- dropped notifications — `dropped <n> <stage>` (`intake` or `dispatch`, the two
  bounded queues); JSON type `dropped`;
- an OS watcher failure — `watcher error: <message>`; JSON type
  `watcher_failed`.

`--max-events N` shuts the watcher down after the N-th rendered update and exits
`0`, which is what makes the command scriptable. A plan with no target at all is
a typed `error: config:` (`… requires at least one watch target`) instead of a
silently idle process.

### `sessions <COMMAND>`

The read-only catalogue behind `--resume`/`--continue`: it merges the CLI
session records, the persistent-agent execution journals and the persistent-agent
session records into one list, newest first. Listing creates nothing, so it works
on a home and a workspace that were never materialised.

| Flag | Meaning |
| --- | --- |
| `--data-dir <DIR>` | Group-global: base directory holding the persistent-agent state (`logs/`, `sessions/`, `links/`). Defaults to the workspace's `persistent_agent` node. |
| `--json` | Group-global: print one pretty JSON document instead of text. |

| Command | Meaning |
| --- | --- |
| `sessions list [--kind <KIND>] [--limit <N>]` | List the persisted rows, newest first. `--kind` is `cli`, `execution` or `state`; `--limit` defaults to `20` and `0` means all (the kind filter applies before the limit). |
| `sessions show <ID>` | Show one row and exactly what resuming it would seed. |
| `sessions prune [--older-than <DAYS>] [--all] [--dry-run]` | Delete rows older than `DAYS` (fractional days allowed) or every row with `--all`; one of the two is required. `--dry-run` prints the selection without deleting. |

Row lines are `<id>\t<created_at or ->\t<kind>\t<detail>`, where the detail names
the store: `<n> messages, model <model>` (cli), `<n> entries, session <id>, usable
<bool>` (execution), or `agent <id>, waiting <bool>, ending <bool>` (state, with
`created_at` taken from the `state.json` modification time). An empty selection is
`no sessions`.

`sessions show` prints `session <id>`, `kind <kind>`, `created <ts or ->`, the
detail line, `messages <n>` (the seeded window), one `<role>: <content>` line per
seeded message, and — when a persistent session record was traced through its
journal — `plan execution <ref or -> resume-seq <n>`. With `--json` it is one
object with the keys `id`, `kind`, `created_at`, `detail`,
`messages: [{"role", "content"}]`, `plan` and `record`.

`prune` prints `would prune <kind>\t<id>` for a dry run, `pruned <kind>\t<id>`
otherwise, and `nothing to prune` when the selector matches nothing; `--json`
prints `{"dry_run", "pruned", "entries"}`. Deleting a `cli` row removes
`<home>/sessions/<id>.json`, an `execution` row removes its journal directory, and
a `state` row removes its `<agent>/<session>/` directory.

## Global options

| Flag | Meaning |
| --- | --- |
| `-m`, `--model <MODEL>` | Model name (default `gpt-4o`). |
| `--provider <PROVIDER>` | Model provider, for example `openai` or `anthropic` (default `openai`). |
| `--api-key <KEY>` | API key for the provider. Parsed as plain text, turned into a secret by option resolution, never logged. |
| `--api-base <URL>` | Base URL of the provider API. |
| `-w`, `--workspace <DIR>` | Workspace directory the agent may read and write. |
| `-v`, `--verbose` | Stream reasoning chunks and extra diagnostics. |
| `--no-color` | Force ANSI colour off. Colour is already off when stdout is not a terminal or when `NO_COLOR` is set. |

Every option is global and optional, so it may appear before or after a
subcommand. The effective value of each field is resolved per level:
**flag → `OPENJIUWEN_*` → generic SDK variable → `~/.openjiuwen/settings.json` →
built-in default**, blank values skipped at every level.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success, including `--help` and `--version`, and a `watch` that stopped at `--max-events`. |
| `1` | A runtime failure from `run`, `chat`, `auto-harness`, `lsp`, `cron`, `team` or `watch`. Reported as `error: <kind>: <cause>` on stderr (for example `error: unsupported: …`, `error: not_found: cron job not found: …`, `error: validation: team socket is required: …`). |
| `2` | A usage error, decided by clap before any command runs: unknown subcommand or flag, missing value, invalid enum value (`--pipeline bogus`, `--stage bogus`, `--output-format bogus`, `lsp definitely-not-an-action`). |

## Streams: logs on stderr, results on stdout

- **stdout is the result stream.** `run -f json` prints exactly one JSON object;
  `run -f stream-json` prints one JSON object per line; `auto-harness --json`,
  `lsp --json`, `cron --json`, `team --json` and `watch --list --json` print one
  JSON document; `watch --json` prints one JSON object per reported update. A
  `watch --list` text plan and every text-mode command print their lines here
  too. `auto-harness run --dry-run` prints only the task array.
- **stderr carries everything else**: the `tracing` log stream (always installed
  with a stderr writer), permission prompts, the `error: …` diagnostic,
  `warning: …` startup notes, the `[dry-run] task list -> …` note, the
  `worktree: …` announcement, the `watching N target(s)` readiness line, and the
  reload notices of `chat --watch`. Redirecting stderr therefore leaves a clean,
  machine-readable stream on stdout.

## Environment variables

The CLI honours the variables below (unset, empty and whitespace-only values are
treated as absent).

| Variable | Effect |
| --- | --- |
| `OPENJIUWEN_HOME` | Runtime-state root (default `~/.openjiuwen`): settings, permission allow rules, session records, `mcp.json`, input history, and the `auto_harness/` tree when no `-w` is given. |
| `OPENJIUWEN_MODEL`, `MODEL_NAME` | Model name. |
| `OPENJIUWEN_PROVIDER`, `MODEL_PROVIDER` | Provider name. |
| `OPENJIUWEN_API_KEY`, `API_KEY`, `OPENAI_API_KEY` | API key (first non-blank wins). |
| `OPENJIUWEN_API_BASE`, `API_BASE`, `OPENAI_BASE_URL` | Provider base URL. |
| `OPENJIUWEN_WORKSPACE` | Workspace directory when `-w` is absent. |
| `OPENJIUWEN_MAX_ITERATIONS` | Agent iteration budget (must be an integer ≥ 1). |
| `OPENJIUWEN_MAX_TOKENS` | Per-response token budget (must be an integer ≥ 256). |
| `RUST_LOG` | `tracing` filter, wins over the level/format below. |
| `OPENJIUWEN_LOG_LEVEL`, `OPENJIUWEN_LOG_FORMAT` | Log level and format when `RUST_LOG` is unset. |
| `NO_COLOR` | Disables colour when set to any non-blank value. |
| `OPENJIUWEN_MCP_CONFIG` | Explicit `mcp.json` path. |
| `OPENJIUWEN_TEAM_SOCKET` | Attach socket of the hosting team, dialled by `openjiuwen team` when `--socket` is absent. |
| `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY` | REPL history file and replay switch. |
| `OPENJIUWEN_NO_ONBOARDING` | Skips the first-run onboarding wizard. |
| `SSRF_PROTECT_ENABLED` | Guard for outbound URL fetching tools. |

A `.env` file is loaded from the working directory **and its ancestors** at
startup; its absence is normal.

`watch` and `chat --watch` need the harness file watcher, so this binary builds
`openjiuwen-harness` with its `notify` feature enabled. There is no runtime
switch: the watcher is compiled in, and no environment variable turns it on or
off.

## Files the CLI reads and writes

```
~/.openjiuwen/                     (or $OPENJIUWEN_HOME)
  settings.json                    provider/model defaults written by onboarding
  permissions.yaml                 permanent "always allow" tool answers
  sessions/                        append-only session records (read by sessions list/prune)
  mcp.json                         MCP server declarations (optional)
  .openjiuwen_history              REPL input history
  auto_harness/                    auto-harness state when no -w is given
    config.yaml                    optional AutoHarnessConfig document
    runs/tasks.json                last --dry-run task list
    experience/experiences.jsonl   experience records (read-only in Rust)

<workspace>/                       -w, OPENJIUWEN_WORKSPACE or <home>/workspace
  persistent_agent/                execution journals, session records and links (sessions list/show/prune)
  cron/jobs.json                   the cron schedule written by cron create/toggle/cancel/run
  cron/runs.json                   the bounded run history written by cron run, read by cron runs
  .openjiuwen/mcp.json             workspace MCP configuration (a watch target)
  .worktree/index.json             worktree registry shared with the harness worktree tools

<repo>/.worktrees/<slug>/          worktree created by --worktree <NAME> (branch worktree-<slug>)
```

`auto-harness run` and the store commands resolve the auto-harness data
directory differently, matching Python: `run` uses
`<-w workspace>/auto_harness` when `-w` was given and `<home>/auto_harness`
otherwise, while `experience list`/`search` and `history` read
`<resolved workspace>/auto_harness/experience/experiences.jsonl`, where the
resolved workspace is `-w`, then `OPENJIUWEN_WORKSPACE`, then
`<home>/workspace`; a relative value resolves against the process directory.

## Rust-port differences and deferred surfaces

The port keeps the Python command surface and output wording where it can, and
is explicit where it cannot:

- **No auto-harness session execution.** The assess → plan → implement → verify
  pipeline needs a harness evaluation runner and an LLM-driven pipeline; those
  are not part of `openjiuwen-rsi`, so `auto-harness run` without `--dry-run`
  exits `1` with a typed `unsupported` error naming the missing runner. Task
  resolution, `--task` / `--task-file` parsing, the budget/goal/pipeline
  overrides and the dry run are implemented.
- **`verify-ext` is deferred.** The runtime extension loader (synthetic
  `sys.modules` hot-loading plus an external linter) has no Rust equivalent, so
  the command reports `unsupported: runtime extension loader is not implemented
  in the Rust port: …` instead of printing a fake `PASSED`.
- **`gap-analyze` is the same placeholder as in Python** — no model is wired in
  in either port.
- **`experience` is read-only.** Nothing in the Rust port produces experience
  records yet; the commands read the documented JSONL, and the directory is
  created on first read so scripts can seed it.
- **`team` needs Unix domain sockets.** The grammar and the wire types compile
  everywhere, but the attach transport is Unix-only: on a non-Unix target the
  command keeps its help and returns `unsupported: cross-process attach requires
  Unix domain sockets`. The socket comes from `--socket` or
  `OPENJIUWEN_TEAM_SOCKET`; the protocol carries no authentication, so trust is
  the socket directory's file mode.
- **`lsp` never probes `PATH`.** A server is either a built-in table entry
  (`pyright`, `rust`, `typescript`, `gopls`, `jdtls`, each with its fixed
  program, arguments, language id, extensions and root markers) or an explicit
  `--server-command`. Nothing is discovered by scanning the filesystem or the
  environment, so a program that is not installed surfaces as a typed launch
  failure (`error: tool: …`) on first use instead of a silent fallback.
- **`watch` reports reloads; it does not hot-swap a running REPL.** The
  `chat --watch` notices tell the user that an instruction file, skill library or
  MCP configuration changed and that the harness reloaded it, but the running
  REPL keeps the instructions, skills and MCP servers it started with — a
  restart applies them. Only `openjiuwen watch` is a pure observer, and
  `--max-events` exists so scripts can stop it deterministically.
- **`chat --resume` starts a new REPL session.** The harness `Repl` always mints
  a fresh `cli-*` id (its `session_id` field is private and the constructors take
  no id), so a resumed chat seeds the restored transcript into a new session and
  the stderr note names both ids. `run --resume` does continue the resolved id:
  the kernel session and the CLI session record keep it and the new turns are
  appended. The resume summary uses the CLI's fresh request id for the
  persistent-agent `resume` call, so a state-record resume settles only its own
  waiting scene.
- **No remote mode.** Python's client/server (`OPENJIUWEN_SERVER_URL`) path is
  not exposed by this binary: `run` and `chat` always drive an in-process agent.
- **Rust additions.** `--json` on the `auto-harness`, `cron`, `team`, `lsp` and
  `sessions` groups and on `watch`, `--max-events`/`--debounce-ms` on `watch`,
  `--worktree`/`--watch` on `run`/`chat`, `--resume`/`--continue` on `run`/`chat`
  and the whole `sessions` group, the stderr dry-run note and resume note
  (Python prints the former on stdout), English user-facing messages, and the
  `error: <kind>: <cause>` diagnostic on stderr with exit `1`.
- **Deferred harness features** are documented in `../openjiuwen-harness/` and
  `../../docs/PARITY.md`: the browser/CUA/mobile presets, MCP resource tools that
  need a host backend, and OTel export.

## Tests

`tests/` holds the process-level suites, all offline and hermetic (temporary
working directory, temporary `OPENJIUWEN_HOME`, ambient `OPENJIUWEN_*` /
`OPENAI_*` / `API_KEY` / `API_BASE` / `MODEL_*` removed, bounded child waits).
The eight suites hold 78 tests; no case needs the network, credentials or a
terminal.

| Suite | Covers |
| --- | --- |
| `tests/cli_smoke.rs` | `--version`, the `--help` rows of `run`/`chat`/`lsp`/`cron`/`team`/`watch`/`auto-harness`, every subcommand's help and flag list (`lsp` actions, `cron` and `team` subcommands, `watch` flags, `run --worktree`, `chat --watch`/`--worktree`), `--no-color` placement, the unknown-subcommand usage error. |
| `tests/cli_run.rs` | A real `run` against the loopback mock model server in all three output formats, `--no-color` through a run, and `chat --help`. |
| `tests/auto_harness.rs` | Task resolution, the dry run and its files, the deferred `run`/`verify-ext` errors, the experience store (ordering, limit, type filter, search, `--json`, empty store), `history` and `gap-analyze`. |
| `tests/cli_features.rs` | `run --worktree` end to end (worktree created, stderr announcement, run inside it), `--worktree` outside a repository as a typed `config` error, and `chat --watch` starting the watcher on stderr and exiting on stdin EOF. |
| `tests/cron.rs` | The empty store, `create`/`list`/`get` persistence, `toggle` both ways, `preview --count`, `run` plus the `runs` receipt, `status` counts, `cancel`, and the `not_found`/`validation` failures. |
| `tests/lsp.rs` | The action and flag help, an unknown action as a usage error, a missing `--file` as `not_found` (also with `--json`), and a failing or unspawnable `--server-command` as a typed `tool` error. |
| `tests/team.rs` | A stub attach host behind a Unix socket: the grammar, one request per command with the documented arguments, the member/task/message/outcome renderings and empty markers, the JSON documents, a remote `not_found`, an unknown `--status`, and the missing-socket validation error. |
| `tests/watch.rs` | `watch --list` text and JSON plans (targets plus skipped roots), an empty plan as a typed `config` error, and a live change reported as one JSON event with `--max-events 1` shutting the watcher down. |

The session catalogue is covered in-crate by the `sessions` module's inline
tests: the pure helpers (row ordering with mixed timestamps, role mapping,
resume-note prefixes, kind tags) plus async tests that drive real stores in a
temporary home and workspace — a persistent session record listing, resolving
and pruning, a CLI record resolving with its transcript, and the typed
`not_found`/`validation` failures.

In-process CLI behaviour (streaming render, REPL transcripts, permissions) is
covered by the harness suites; nothing here duplicates it.
