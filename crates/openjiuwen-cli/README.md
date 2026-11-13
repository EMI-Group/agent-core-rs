# openjiuwen-cli

The `openjiuwen` executable — the process entry point of the Rust openJiuwen
agent SDK.

The binary is a thin process boundary. It owns the operating-system concerns
(`.env` loading, `tracing` installation, clap's exit codes, the stderr
diagnostic for a failed run), the outer command grammar, and the
`auto-harness` command group, which the harness has no words for. Everything
else — option resolution, the one-shot runner, the interactive REPL, permission
prompts and session storage — lives in `openjiuwen_harness::cli`.

```console
$ openjiuwen run "summarise src/lib.rs"
$ openjiuwen chat
$ openjiuwen auto-harness run --task "fix the parser" --dry-run
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

### `run [PROMPT] [-f|--output-format <FORMAT>]`

Runs a single prompt and exits.

| Argument / flag | Meaning |
| --- | --- |
| `PROMPT` | Prompt text. Omitted or `-` reads stdin to end of input and joins the lines. |
| `-f`, `--output-format` | `text` (default), `json` or `stream-json`. |

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

### `chat`

Starts the interactive REPL on a terminal. It takes no options of its own; the
global options still apply. The REPL is wired by the harness and provides the
in-session slash commands `/exit` (`/quit`), `/new`, `/clear`, `/mode`, `/help`,
`/status`, `/cost`, `/compact`, `/sessions` and `/<skill>`, plus `!<command>`
shell passthrough, persistent input history, and the first-run onboarding wizard
when the home directory has no settings file yet.

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
| `0` | Success, including `--help` and `--version`. |
| `1` | A runtime failure from the run, the REPL or an `auto-harness` command. Reported as `error: <kind>: <cause>` on stderr (for example `error: unsupported: …`). |
| `2` | A usage error, decided by clap before any command runs: unknown subcommand or flag, missing value, invalid enum value (`--pipeline bogus`, `--stage bogus`, `--output-format bogus`). |

## Streams: logs on stderr, results on stdout

- **stdout is the result stream.** `run -f json` prints exactly one JSON object;
  `run -f stream-json` prints one JSON object per line; `auto-harness --json`
  prints one JSON document. `auto-harness run --dry-run` prints only the task
  array on stdout, while its `[dry-run] task list -> …` note goes to stderr.
- **stderr carries everything else**: the `tracing` log stream (always installed
  with a stderr writer), permission prompts, the `error: …` diagnostic and
  `warning: …` startup notes. Redirecting stderr therefore leaves a clean,
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
| `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY` | REPL history file and replay switch. |
| `OPENJIUWEN_NO_ONBOARDING` | Skips the first-run onboarding wizard. |
| `SSRF_PROTECT_ENABLED` | Guard for outbound URL fetching tools. |

A `.env` file is loaded from the working directory **and its ancestors** at
startup; its absence is normal.

## Files the CLI reads and writes

```
~/.openjiuwen/                     (or $OPENJIUWEN_HOME)
  settings.json                    provider/model defaults written by onboarding
  permissions.yaml                 permanent "always allow" tool answers
  sessions/                        append-only session records
  mcp.json                         MCP server declarations (optional)
  .openjiuwen_history              REPL input history
  auto_harness/                    auto-harness state when no -w is given
    config.yaml                    optional AutoHarnessConfig document
    runs/tasks.json                last --dry-run task list
    experience/experiences.jsonl   experience records (read-only in Rust)
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
- **No remote mode.** Python's client/server (`OPENJIUWEN_SERVER_URL`) path is
  not exposed by this binary: `run` and `chat` always drive an in-process agent.
- **Rust additions.** `--json` on the `auto-harness` group, the stderr dry-run
  note (Python prints it on stdout), English user-facing messages, and the
  `error: <kind>: <cause>` diagnostic on stderr with exit `1`.
- **Deferred harness features** are documented in `../openjiuwen-harness/` and
  `../../docs/PARITY.md`: the browser/CUA/mobile presets, the LSP client, MCP
  resource tools that need a host backend, and OTel export.

## Tests

`tests/` holds the process-level suites, all offline and hermetic (temporary
working directory, temporary `OPENJIUWEN_HOME`, ambient `OPENJIUWEN_*` /
`OPENAI_*` / `API_KEY` / `API_BASE` / `MODEL_*` removed, bounded child waits):

| Suite | Covers |
| --- | --- |
| `tests/cli_smoke.rs` | `--version`, `--help` rows for `run`/`chat`/`auto-harness`, every subcommand's help, `--no-color` placement, the unknown-subcommand usage error. |
| `tests/cli_run.rs` | A real `run` against the loopback mock model server in all three output formats, `--no-color` through a run, and `chat --help`. |
| `tests/auto_harness.rs` | Task resolution, the dry run and its files, the deferred `run`/`verify-ext` errors, the experience store (ordering, limit, type filter, search, `--json`, empty store), `history` and `gap-analyze`. |

In-process CLI behaviour (streaming render, REPL transcripts, permissions) is
covered by the harness suites; nothing here duplicates it.
