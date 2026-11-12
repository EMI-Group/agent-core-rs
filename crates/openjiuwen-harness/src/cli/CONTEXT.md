# cli

## Intent

Command-line logic for the `openjiuwen` binary: a clap-based `run` command and an interactive REPL that build a `DeepAgent`, stream output, and answer permission/interaction prompts.
Ports the agent-facing parts of `openjiuwen/harness/cli/` (`cli.py`, `agent/factory.py`, `ui/repl.py`); the actual binary lives in `openjiuwen-cli`.

## API Surface

- `Cli` (clap `Parser`): global options `--model/-m`, `--provider`, `--api-key`, `--api-base`, `--workspace/-w` (default `~/.openjiuwen/workspace`), `--verbose/-v`; subcommands `run [PROMPT]` (`--output-format/-f` = `text|json|stream-json`) and `chat`; no subcommand dispatches to `chat` on a TTY, otherwise `run` with stdin.
- `run(cli: Cli) -> Result<()>` — async entry point used by `openjiuwen-cli`.
- `Repl::new(agent, options)`, `Repl::run() -> Result<()>`: `You> ` prompt, `/exit`, `/new`, `/help` commands, `!cmd` shell passthrough (deferred), Ctrl-C abort handling, streaming renderer for `llm_output`/`answer`/`tool_call`/`tool_result`/`__interaction__`.
- `permissions.rs`: terminal confirmation renderer (`⚠ Approve {tool}?`, `(y/yes, a/always, n/no, feedback)`, empty Enter = y) and `CliPermissionHost` implementing `permission::PermissionHost`.
- `config.rs`: resolve model/provider/key/base/workspace from CLI flags then `OPENJIUWEN_*`, `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_*`, then `~/.openjiuwen/settings.json`; defaults `OpenAI`/`gpt-4o`.
- Session id `cli-<uuid8>`, persisted under `~/.openjiuwen/sessions/<session_id>.json` (append-only history).

## Constraints

- The CLI owns stdout/stderr; libraries log through `tracing` only.
- No `dotenvy` in the library; the binary entry point enables `.env`.
- Prompt text and user-visible messages match Python where ported; unanswered permission prompts reject (fail closed) except empty Enter which approves, matching Python.
- Secrets are never echoed; `--api-key` is read into `SecretString`.
- `run` supports non-TTY stdin as the prompt (`-` or piped), empty prompt → usage error.
- Deferred: `auto-harness` subcommands, remote mode, file-based session history restore, slash completion UI, rich renderer.

## Routing Table

- `./mod.rs` → `Cli`, subcommands, `run`, dispatch logic
- `./config.rs` → option/env/settings resolution
- `./run.rs` → one-shot `run` execution and output formats
- `./repl.rs` → interactive REPL loop and rendering
- `./permissions.rs` → terminal permission prompts and `CliPermissionHost`

## Test Strategy

- Integration tests (`tests/cli.rs`): clap parsing (flags, defaults, output-format), no-subcommand dispatch rule, `run` with a scripted model prints the answer, permission prompt rendering/answers, empty prompt usage error.
- Parity references: `tests/unit_tests/cli/test_agent_factory.py`, `test_repl_interrupt_kinds.py`, `tests/cli/unit/test_renderer.py`.
