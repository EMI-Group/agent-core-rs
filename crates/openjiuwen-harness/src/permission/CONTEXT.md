# permission

## Intent

Tool-permission policy for the DeepAgent: a tiered rule pipeline (builtin rules, user rules, tool baselines, approval overrides), a file-path guard, a network guard and a shell-structure guard folded into one allow/ask/deny outcome, plus the HITL interrupt rail that turns an ASK into a confirmation prompt.
Ports `openjiuwen/harness/security/permission_engine/` and `openjiuwen/harness/rails/security/tool_security_rail.py` (Python v0.1.20).
Fidelity target: decision outcomes, `matched_rule` label strings, `[PERMISSION_DENIED] …` tool results, the confirm answer schema, session/auto-confirm semantics — internals are idiomatic Rust.

## API Surface

Every public item is reachable as `crate::permission::…`.

- `model.rs` — `PermissionLevel {Allow, Ask, Deny}` (wire `allow`/`ask`/`deny`, `parse` is trim+lower), `PermissionResult {permission, matched_rule, reason, external_paths}` with `is_allowed`/`is_denied`/`needs_approval`, `PermissionConfirmResponse {approved, feedback, auto_confirm, persist_allow}` with `wants_permanent_persist`/`wants_session_persist`/`parse`, `FileAction {Read, Write, Exec}`, `PathAccess {path, action}`, `RuleMatchType {Command, Path}`, `PathMatch {Prefix, Glob}`, and the config data: `PermissionsSection`, `PermissionRule`, `ApprovalOverrideEntry`, `FileGuardSection`, `FileGuardDefaults`, `FileGuardPathEntry`, `NetGuardSection`, `ShellGuardSection`.
- `patterns.rs` — `match_wildcard(value, pattern)` (shell-safe glob, `*` never crosses shell metacharacters), `match_glob(pattern, path)` (`**` path glob), `PolicyMatcher` (exact / wildcard / `re:` regex) and `match_net_pattern(url, pattern)`.
- `builtin.rs` + `builtin_rules.yaml` — packaged `resources/builtin_rules.yaml` equivalent: `package_command_rules()`, `package_sensitive_paths()`, `package_net_urls()` (cached, never IO at check time).
- `rules.rs` — `normalize_tool_name` / `TOOL_NAME_ALIASES`, tool categories (`tool_category`, `is_shell_tool`, `shell_tools_from_config`, `DEFAULT_SHELL_TOOLS`), rule applicability + label formatting, `severity_to_action` / `severity_to_decision` / `normalize_permission_mode`.
- `shell.rs` — `canonicalize_shell_command`, `analyze_shell` → `ShellCommand {kind, canonical, subcommands, flags}` (conservative scanner replacing Python's tree-sitter), `shell_auto_confirm_key(tool_name, command)` (`"bash:ls"`, empty for risky/compound/empty).
- `paths.rs` — `extract_accesses(tool, args, workspace_root) -> Vec<PathAccess>` (file-tool spec table + shell path tokens), `resolve_path`, `tool_action`.
- `tiered.rs` — `evaluate_tiered_policy(section, tool_name, args) -> (PermissionLevel, String)`; never returns "no opinion".
- `file_guard.rs` — `FileGuardChecker::{build, evaluate, collect_ask_accesses}` (prefix/glob match, write/exec⇒read implication, longest-prefix-wins, `None` when it has no opinion).
- `net_guard.rs` — `NetGuardChecker::{build, evaluate}`, `extract_fetch_url`; allow/deny only (ask entries dropped), `None` or DENY.
- `engine.rs` — `PermissionEngineConfig {permissions, workspace_root, trusted_dirs, enabled}`, `PermissionEngine::{new, check_permission, evaluate_global_policy_directly, update_permissions, config, workspace_root, enabled}`, `strictest` / `strictest_all`, `prepare_permissions_for_engine(section) -> PermissionsSection` (inlines builtin rules, sensitive paths, net urls; fills legacy `severity` actions; idempotent via `layer: "builtin"`).
- `session.rs` — `PermissionHost` (async: `request_confirmation`, `persist_allow_rule`, `persist_session_allow_rule`; sync: `permissions_snapshot`, `resolve_workspace_dir`, `session_id`, `permission_yaml_path`, `scene_hook`), `NoopPermissionHost`, `PermissionConfirmationRequest`, `ConfirmationOutcome`, `SceneDecision`, `SessionPermissionStore`, `merge_allow_rule_into_permissions`, `merge_file_guard_access_allows`, `should_store_auto_confirm`.
- `presentation.rs` — `AskPresentation {category, title, summary}`, `build_ask_presentation`, `render_ask_message` (Python ask copy, Chinese titles).
- `interrupt.rs` — `PermissionInterruptRail` (priority 90, `AgentRail`, kind `permission`, owner `openjiuwen.harness.rails.security.tool_security_rail.PermissionInterruptRail`), `PERMISSION_DENIED_PREFIX`, `PERMISSION_REJECTED_MESSAGE`.

## Constraints

- Fail closed: parse errors, unknown structures, malformed levels, missing config and absent matching rules widen to ASK (or keep DENY); nothing silently ALLOWs but an explicit allow rule / `defaults["*"]`.
- `Option<PermissionLevel>` = "no opinion", never allow; every fold goes through `strictest` (deny < ask < allow).
- Behavioural labels are API: `tiered_policy:fallback(no_config)`, `tiered_policy:builtin:deny:<ids>`, `tiered_policy:approval_overrides`, `tools.<tool>`, `file_guard…`, `net_guard…`, joined with `|` in pipeline order (A → file_guard → net_guard).
- `check_permission` is sync and never blocks: no filesystem/network IO, no locks held across calls; compiled config lives behind `RwLock` in the engine (`Arc<PermissionEngine>` shared).
- Config types are serde snake_case, tolerant of unknown fields, `JsonSchema` derived — `PermissionsSection` is embedded in `crate::schema::DeepAgentConfig`.
- No dependency on `crate::schema` or any other harness module; no `unsafe`; no panics/`unwrap` outside tests; `tracing` only (no stdout).
- Errors: `Error::validation` for bad config, `Error::security` for denials raised by callers; checkers return decisions, not errors.
- Files ≤1000 lines, split by responsibility; shared helpers live in the lowest owning module (`patterns`, `model`, `rules`).

## Design decisions

- Python's engine is `async` because the rail awaits host hooks; in Rust `check_permission` is synchronous (no IO) and only the rail/host path is async.
- Python's tree-sitter shell AST is replaced by a conservative lexer in `shell.rs` that reports `Simple` / `TooComplex`; unknown or risky structure widens to ASK exactly like Python's `too_complex` / `parse_unavailable` floors.
- Python rule dicts are typed Rust data (`PermissionRule`, ...). `action: Option<PermissionLevel>` so ingest can fill it from `severity`; unknown action words fail at deserialization (Python raises `ValueError`).
- Python's `tools.<name>` accepts `{"*": level}`; the Rust field deserializes both forms into `BTreeMap<String, PermissionLevel>`.
- Session decisions are layered like Python: `SessionPermissionStore` (auto-confirm keys + in-memory allow overlay) and host hooks (snapshot/persist). Persisting a permanent allow calls the host only; the rail never writes YAML itself (no `permission_yaml_path` writer).
- The permission rail's interrupt request carries an empty `auto_confirm_key` (Python parity), so the kernel does not auto-store it; the rail stores auto-confirm itself.
- Persist-once across resume rounds is tracked by tool-call id in `ctx.extra` (`_permission_persisted_approvals`) because the kernel has no "carried answer" flag.

## Deferred Python areas

- `external_directory` legacy section and `trusted_dirs` projection: only `PermissionEngineConfig.trusted_dirs` (prefix allow/allow/ask) is supported; the deprecated `external_directory` mapping is dropped.
- Literal-shell allow shortcut (`literal_shell.py`, `parse_literal_shell`): not ported; approvals come from `approval_overrides`.
- `approve/ask_presentation.py` suggestion/chip rendering: only category/title/summary + always-allow hint are ported.
- `skill_authorization/` remains a separate security axis (not this node).
- YAML write-back of a `permissions:` section (`write_permissions_section_to_agent_config_yaml`) belongs to the CLI/config layer, not this module.
- `permission_checks_active` predicate seam (`set_permission_checks_active`).

## Routing Table

- `./model.rs` → permission levels/results, confirmation payload, config sections and rule data
- `./patterns.rs` → wildcard/glob/regex/URL matchers
- `./builtin.rs` + `./builtin_rules.yaml` → packaged command rules, sensitive paths, metadata URLs
- `./rules.rs` → tool-name normalization, categories, rule applicability, labels, severity mapping
- `./shell.rs` → command canonicalization, structure analysis, auto-confirm keys
- `./paths.rs` → path-access extraction and resolution
- `./tiered.rs` → pipeline A tier ladder
- `./file_guard.rs` → path read/write/exec guard (pipeline B)
- `./net_guard.rs` → URL guard (pipeline C)
- `./engine.rs` → `PermissionEngine`, fold/reasons, `prepare_permissions_for_engine`
- `./session.rs` → host trait, session store, allow-rule merges, auto-confirm policy
- `./presentation.rs` → ask category/title/summary and message rendering
- `./interrupt.rs` → `PermissionInterruptRail`
- `./tests.rs` → unit-test suite for the whole module (module tests may also live inline)

## Test Strategy

- `tests.rs` covers: tier precedence and labels (builtin deny > user deny > approval override > builtin > user > tool baseline > defaults > ASK), approval override never relaxing DENY (compound subcommand case), severity mapping tables, wildcard/glob/regex matching, file-guard prefix/glob + implication + longest-prefix-wins, net-guard ask-dropping/defaults/authority matching, shell structure widening + interpreter sink + auto-confirm keys, engine reason templates, session merge shapes and auto-confirm truth table, rail allow-once/allow-always(permanent+session)/deny/reject/invalid-payload/carried-persist-once.
- Parity references: `tests/unit_tests/harness/security/*` and `tests/unit_tests/harness/rails/test_typed_interrupts.py` in the Python reference.
- Crate-level integration tests live in `../../tests/permissions.rs` (sibling node — read-only from here).
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`.
