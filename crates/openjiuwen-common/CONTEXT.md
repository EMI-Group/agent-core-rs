# openjiuwen-common

## Intent

Shared kernel types for the Rust rewrite: typed errors, Cards, logging, config/env, secrets and redaction, an async HTTP client, a structured task manager, and general utilities.
Maps to the Python package `openjiuwen.core.common` (schema, exception, logging, clients, task_manager, security, utils, constants, run_context).
Every other crate depends on this crate, so its surface is small, provider-agnostic, and free of workspace dependencies (it may only depend on external crates).
Python mechanisms are replaced deliberately: the exception class hierarchy becomes one `Error` enum plus `ErrorKind`; `contextvars` + `LogManager` become `tracing` plus `LogScope`; the pydantic `BaseCard`/`Param` become serde structs with `schemars` schemas.

## API Surface

### `error` — `crate::error::{Error, ErrorKind, Result, ResultExt, OptionExt}`
`pub enum Error { Config, Validation, Llm, Tool, Workflow, Graph, Session, Memory, Retrieval, SysOp, Security, Mcp, Transport, Timeout, NotFound, Unsupported, Cancelled, External }` exactly as `docs/ARCHITECTURE.md` §2.1, where all string variants hold `String` and `External(Box<dyn std::error::Error + Send + Sync>)`.
`pub type Result<T, E = Error> = std::result::Result<T, E>`.
`ErrorKind` is the non-payload discriminant (`Copy + Eq + Hash`, snake_case `Display`/serde) with `Error::kind()`.
Constructors: `Error::config/validation/llm/tool/workflow/graph/session/memory/retrieval/sys_op/security/mcp/transport/timeout/not_found/unsupported`, `Error::cancelled()`, `Error::external(e)`, all taking `impl Into<String>` / `impl std::error::Error + Send + Sync + 'static`; every constructor accepts an optional detail suffix via `format!` at the call site.
`Error::message() -> Cow<'_, str>`, `Error::is_cancelled()`, `Error::is_retryable()` (true for `Transport`, `Timeout`), `Error::with_context(ctx) -> Error` (prefixes string variants, wraps `External` in the crate-private `ContextError` so `source()` keeps working, leaves `Cancelled` untouched).
`impl std::error::Error for Error` exposes `source()` for `External`; `Display` is `"<kind>: <message>"` (Python's `[code] message` form is not reproduced — single enum is the documented deviation).
`ResultExt<T>::context(ctx)` for every `std::result::Result<T, E> where E: Into<Error>`, and `OptionExt<T>::ok_or_not_found/ok_or_validation` helpers.
`From` impls: `io::Error`, `serde_json::Error`, `url::ParseError`, `reqwest::Error` (timeouts map to `Timeout`), `tokio::task::JoinError`, `Box<dyn Error + Send + Sync>`.
No panics on user input: all fallible paths return `Result`.

### `card` — `crate::card::{Card, CardBase, CardId}`
`pub trait Card: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static { fn id(&self) -> &str; fn name(&self) -> &str; fn description(&self) -> Option<&str>; }`.
`pub struct CardBase { id: String, name: String, description: Option<String>, version: Option<String>, metadata: BTreeMap<String, Value> }` with `#[serde(rename_all = "snake_case", default)]`, unknown fields tolerated, `None` fields skipped on serialize; constructors `CardBase::new(name)`, builders `with_id/with_name/with_description/with_version/with_metadata/with_metadata_entry`, accessors, and `to_str()` = `"id=<id>,name=<name>"` (Python `BaseCard.to_str`).
`CardBase` implements `Card`, `Default`, `JsonSchema`, `PartialEq`.
`pub struct CardId(String)` is a transparent serde/schemars newtype with `Display`, `From<&str>`, `From<String>`, `as_str()`, `CardId::generate()` (Python `uuid4().hex`, 32 lowercase hex chars), and `Default` = generated.
Card JSON is one-to-one with Python `BaseCard.model_dump()` for the shared fields; round-trips through `serde_json` and rejects no unknown keys.
`AgentCard` and the other concrete cards are NOT defined here: `AgentCard` lives in `openjiuwen-agent` (`src/card.rs`) and embeds its fields directly instead of flattening `CardBase`.
`card_schema::<C: Card>() -> Result<Value>` and `card_base_schema() -> Result<Value>` emit deterministic JSON Schemas (generated-id defaults stripped).

### `logging` — `crate::logging`
`levels`: `normalize_level(&str) -> tracing::level_filters::LevelFilter` with Python aliases (`CRITICAL`/`FATAL` → `ERROR`, `WARN` → `WARN`, case-insensitive), unknown → `WARNING`; `parse_level(&str) -> Option<tracing::Level>`; `DEFAULT_CONFIG_LEVEL = INFO`, `DEFAULT_FALLBACK_LEVEL = WARN`.
`subscriber`: `LoggingConfig { level, format, writer, ansi, filter }`, `LogFormat::{Text, Json}`, `LogWriter::{Stdout, Stderr, File(PathBuf)}`, `init() -> Result<()>` (reads `OPENJIUWEN_LOG_LEVEL`/`OPENJIUWEN_LOG_FORMAT`, defaults `INFO`/`Text`), `init_with(config: &LoggingConfig) -> Result<()>`, both idempotent (second call returns `Ok(())`), `is_initialized() -> bool`; never panics and never installs a second global subscriber; file writer is a `MakeWriter` (rotation is deferred — a binary owns log rotation).
`scope`: `LogScope { trace_id, member_id, session_id, invocation_id, agent_id, conversation_id }` (all optional except trace_id), `LogScope::new()` (generates trace id), builders, `span() -> tracing::Span` (records scope fields), `with_scope(Arc<LogScope>, fut)` (tokio task-local), `Arc<LogScope>::instrument(fut)` (span + task-local in one future, the supported way to cross `tokio::spawn`), `current() -> Option<Arc<LogScope>>`, `current_trace_id() -> String` (default `"default_trace_id"`, Python parity), `current_member_id() -> String` (default `""`; `DEFAULT_TRACE_ID` names the default).
`events`: `ModuleType`, `EventStatus`, `EventType` (string newtype + well-known consts), `LogEvent` with Python `BaseLogEvent` fields (`event_id`, `event_type`, `log_level`, `timestamp`, `module_type`, `module_id`, `module_name`, `session_id`, `conversation_id`, `trace_id`, `correlation_id`, `parent_event_id`, `status`, `error_code`, `error_message`, `message`, `stacktrace`, `exception`, `metadata`) plus `to_dict()`/`to_value()` dropping nulls, serializing enums as their Python values, timestamps as ISO-8601, and setting `error_code = "-1"` for attached exceptions (Python parity); `redacted(fields)` applies `security::redact`.
`targets`: `&'static str` target names for the Python loggers (`COMMON`, `AGENT`, `WORKFLOW`, `LLM`, `TOOL`, `SESSION`, ...).
`emit(&LogEvent)` forwards to `tracing` at the event's level with structured fields; library code never uses `println!`.

### `config` — `crate::config::{env, home, load_dotenv}`
`env`: `var`, `flag`, `truthy` (`1/true/yes/on`), `parse<T: FromStr>`, plus typed resolvers `api_key() -> Option<SecretString>` (`OPENJIUWEN_API_KEY` → `API_KEY` → `OPENAI_API_KEY`), `api_base() -> Option<String>` (`API_BASE` → `OPENAI_BASE_URL`), `model_provider() -> String` (default `"openai"`), `model_name() -> Option<String>`, `openjiuwen_home() -> Option<PathBuf>`, `server_url()`, `bash_strict() -> bool` (default false), `mcp_reconnect_attempts() -> u32` (default 3), `otel_global_instrument_enabled() -> bool` (default false), `ssrf_protect_enabled() -> bool` (default true), `http_proxy()`/`no_proxy()`.
`home`: `OpenJiuwenHome { root: PathBuf }` with `resolve()` (`OPENJIUWEN_HOME` → `$HOME/.openjiuwen` → `./.openjiuwen`), `from_root`, `root()`, `join(rel)`, `logs_dir()`, `sessions_dir()`, `ensure() -> Result<()>`.
`load_dotenv()` is the only `.env` entry point and is documented as binary-only; library constructors never read `.env`.

### `security` — `crate::security::{SecretString, constant_time_eq, redact}`
`SecretString` redacts in `Debug`/`Display` (`<REDACTED>`), serializes its value (config round-trip, Python parity), never logs, exposes `expose_secret()`, `is_empty()`.
`redact`: `REDACTED = "<REDACTED>"`, `DEFAULT_SENSITIVE_FIELDS` (Python list: `messages`, `response_content`, `input_content`, `query`, `arguments`, `result`, `message_content`, `tool_calls`, `input_data`, `output_data`, `retrieved_memories`), `redact_json(&Value, Option<&[&str]>) -> Value` (recursive key match), `redact_str`, `redact_url_password(&str) -> String` (masks only the password, Python parity), `sanitize_headers(&BTreeMap<String,String>) -> BTreeMap<String,String>` (drops blank keys/values and protected `host`, `content-length`, `transfer-encoding`, `connection`, `authorization` case-insensitively), `constant_time_eq(&[u8], &[u8]) -> bool`.

### `http` — `crate::http`
`HttpClientConfig` (serde, `Default`, builders): timeouts, `max_retries`, retry backoff bounds, retryable statuses, default headers, proxy, `verify_ssl`, user agent, `max_response_bytes`; no provider-specific fields.
`RetryPolicy` with `backoff(attempt) -> Duration` (exponential, capped, jittered) and `is_retryable_status`.
`HttpClient::{new, with_defaults, request, get, get_json, post_json, execute, execute_cancellable, stream}`; `HttpRequest` builder (method, url, `query`, `header`/`with_headers`, `SecretString` `bearer`, JSON/text/bytes body, per-request `timeout`, `cancel` token — `headers()` is the accessor, the bulk builder is `with_headers`); `HttpResponse { status, headers, url, body }` with `text`, `json<T>`, `is_success`, `error_for_status`; `HttpStream = BoxStream<'static, Result<Bytes>>`; dropping the stream cancels the transfer.
Retries cover transport errors, timeouts, and retryable statuses (default 429/500/502/503/504) with replayable bodies; non-2xx responses are returned, not raised (Python `raise_for_status=False` default); cancellation maps to `Error::Cancelled`, timeouts to `Error::Timeout`, transport failures to `Error::Transport`.

### `task` — `crate::task::{TaskManager, TaskHandle, TaskStatus, TaskInfo, TaskOptions, TaskId}`
`TaskManager::{new, global, spawn, spawn_in_group, spawn_with, list, get, len, is_empty, cancel, cascade_cancel, cancel_group, cancel_all, wait_all, remove_finished, shutdown, subscribe}` (Clone over one `Arc`; `new() -> Self`, `global() -> Self`); `TaskHandle::{id, name, group, status, info, cancellation, cancel, abort, join, detach}`; `TaskStatus::{Pending, Running, Completed, Failed, Cancelled, Timeout}` with `is_terminal`; `TaskOptions` (`name`, `group`, `timeout`, `metadata`); `TaskInfo` is a serializable snapshot (`id`, `name`, `group`, `status`, `parent_id`, timestamps, `error`, `cancel_reason`, `cancelled_by`, `metadata`) with `display_name()` and `duration_ms()`.
Every task is tracked in the manager registry with its `AbortHandle`; there are no untracked detached tasks, and `Drop` for `TaskManager` aborts the remainder.
Panics are isolated with `FutureExt::catch_unwind` and recorded as `Failed`; `join` returns `Error::cancelled()` / `Error::timeout(...)` / an external panic error.
`TaskManager::global()` is the documented process-global instance (Python singleton parity); tests use `TaskManager::new()`.

### `utils` — `crate::utils::{json, id, string, hash}`
`json`: `merge(&mut Value, Value)` (deep object merge, otherwise replace), `get_path(&Value, "a.b[0].c")`, `set_path(&mut Value, path, value)`, `content_to_text(&Value) -> Option<String>` (tolerant string / text-part array / `{"text": ...}`), `flatten(&Value, "...")`, `is_empty_object`.
`id`: `uuid_v4()`, `uuid_hex()` (Python card/event id form), `now_utc()`, `now_local()`, `format_iso8601`, `now_rfc3339`, `unix_millis()`.
`string`: `sanitize_message` (escapes control chars as `\r \n \t \b \v \f \0` / `\xNN`, Python `_sanitize_message`), `truncate_chars`, `format_log_filename` (Python `{name} {ext} {pid} {timestamp=...} {date} {time} {datetime}` placeholders, UTC), `format_schema_error`.
`hash`: `generate_key(api_key, api_base, model_provider)` = sha256 hex of the sorted concatenation (Python `utils.hash_util` parity).

### `run_context` and `constants`
`RunContext { reason: Option<HeartbeatReason>, session_id: Option<String>, context_mode: Option<String>, extra: BTreeMap<String, Value> }`, `HeartbeatReason::{Interval, Manual}` (`"interval"`/`"manual"`), `normalize_run_context(Option<RunContextInput>) -> Option<RunContext>` where unknown mapping keys fold into `extra` (Python parity).
`constants` holds Python `constant.py`/`enums.py` values (`INTERACTION`, `INTERACTIVE_INPUT`, `INPUTS_KEY`, `CONFIG_KEY`, `END_FRAME`, `END_NODE_STREAM`, `LOOP_ID`, `INDEX`, `FINISH_INDEX`, `USER_FIELDS`, `SYSTEM_FIELDS`, `QUERY`, size limits) and `ControllerType`/`TaskType` enums.

## Constraints

- No workspace dependencies; external crates only (`reqwest`, `tokio`, `serde`, `thiserror`, `schemars`, `tracing`, `tracing-subscriber`, `futures`).
- Async-first tokio; no blocking work on async tasks; shared state uses `parking_lot`/`tokio` synchronization and locks are never held across `.await`.
- `unsafe` is forbidden (workspace lint); the workspace clippy lints (`unwrap_used`, `expect_used`, `pedantic`) are `-D warnings`, so library code returns `Result` and tests may `#[allow(clippy::unwrap_used, clippy::expect_used)]`.
- Public items carry doc comments (`missing_docs` is a warning, which CI promotes to an error).
- Files stay around ≤1000 lines; each module owns one concern.
- Public data types are `serde`-serializable with snake_case field names, tolerant of unknown fields.
- Feature flags: none required; everything compiles offline with no external services.
- Documented deviations from Python: numeric `StatusCode` codes and the exception class tree are not reproduced (`ErrorKind` identifies the category); `clients/` connector pools, session managers, and `ClientRegistry` are not ported (each crate owns its own clients, `http` is the shared transport); `security` crypto (AES-GCM), SSRF URL checks, SSL helpers, and sensitive-path checks are deferred to `openjiuwen-security`/`openjiuwen-sysop`; log rotation and invocation JSONL sinks are not ported.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`) and are deterministic and offline: 182 tests.
- Integration tests exercise the crate as an external consumer (public paths only) and live in `tests/`: `error_contract.rs` (15), `card_json.rs` (12), `config_env.rs` (17), `logging_scope.rs` (16), `http_retry.rs` (11, `wiremock`), `task_manager.rs` (4).
- Doc tests cover every public helper's documented example (26).
- Env-var tests serialize with a shared mutex and restore the previous value, because environment variables are process-global.
- Behavior ports cite the Python test file in a comment, e.g. `// port of tests/unit_tests/core/common/test_errors.py`.
- `cargo test -p openjiuwen-common`, `cargo clippy -p openjiuwen-common --all-targets -- -D warnings`, and `cargo fmt -p openjiuwen-common` must pass.

## Design Decisions

- `Error` is a single enum; Python's 400-value `StatusCode` enum and exception class tree are replaced by `ErrorKind` (categories) and per-crate message construction, so no numeric codes are exposed.
- `CardBase` derives serde defaults, so a JSON schema generator would bake a fresh random id into the schema; `card_schema`/`card_base_schema` strip generated defaults so schemas are deterministic (Python's `default_factory` emits none either).
- `LogEvent` keeps the Python field set and `to_dict()` quirks (nulls dropped, exception ⇒ `error_code == "-1"` as a string) but only a documented subset of the ~90 `LogEventType` names; `EventType` is an open string newtype so unknown event names round-trip.
- Retry/backoff lives in the shared HTTP client (Python's `HttpClient` has none) because every model and retrieval client needs it; defaults are zero retries so behavior is opt-in per call site.
- The task manager is structured concurrency, not an asyncio task-group wrapper: spawn registers before polling, every task keeps its `AbortHandle`, panics are caught with `catch_unwind`, and `Drop` aborts leftovers.
- `OpenJiuwenHome::ensure()` creates only the root; callers that need `logs/` or `sessions/` create them explicitly.

## Notes for Agents

- Import errors from `openjiuwen_common::error::{Error, Result}`; `use openjiuwen_common::error::ResultExt;` enables `.context("...")` on any `E: Into<Error>` result.
- `src` never logs with `println!`; use `openjiuwen_common::logging::{info, warn, error, debug, trace}` re-exported from `tracing`, and pick a target from `logging::targets`.
- Tokio task-locals are not inherited by `tokio::spawn`; propagate a `LogScope` with `Arc<LogScope>::instrument(fut)` (or `with_scope`) around the spawned future.
- `LogEvent::redacted(Some(fields))` replaces `DEFAULT_SENSITIVE_FIELDS`; pass `None` for the Python default list.
- `Error::external` requires a real `std::error::Error`; after `with_context` the cause is reachable by walking `source()`, not by downcasting the outer `Error`.
- Id formats differ by purpose: `utils::id::uuid_hex()` (32 lowercase hex, Python card/event ids) vs `uuid_v4()` (hyphenated, task/trace ids).
- `HttpClientConfig.headers` are sanitized: `authorization`, `host`, `content-length`, `transfer-encoding`, and `connection` are dropped; pass credentials through `HttpRequest::bearer` or per-request headers.
- Timeouts and cancellation surface as `ErrorKind::Timeout` / `ErrorKind::Cancelled`; non-2xx responses are returned normally — call `error_for_status()` when a failure must become an error.

## Routing Table

- `./src/error.rs` → shared `Error`/`Result`, `ErrorKind`, context chaining, conversions
- `./src/card/` → `Card` trait, `CardBase`, `CardId`, JSON schema
- `./src/logging/` → tracing initializer, levels, `LogScope`, structured `LogEvent`s, logger targets
- `./src/config/` → environment variable resolvers, `OpenJiuwenHome`, opt-in `.env` loading
- `./src/security/` → `SecretString`, redaction, header sanitization, constant-time compare
- `./src/http/` → provider-agnostic async HTTP client, retry policy, streaming bodies
- `./src/task/` → task manager, handles, statuses, structured cancellation and shutdown
- `./src/utils/` → JSON path/merge helpers, ids and timestamps, string sanitization, hashing
- `./src/run_context.rs` → heartbeat/run context normalization (Python `run_context.py`)
- `./src/constants.rs` → shared protocol constants and controller/task type enums
