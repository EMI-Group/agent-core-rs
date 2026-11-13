# permission

## Intent

Tool-permission policy for the `DeepAgent`: a tiered rule pipeline (builtin rules, user rules, tool baselines, approval overrides), a file-path guard, a network guard and a shell-structure guard folded into one allow/ask/deny outcome, plus the HITL interrupt rail that turns an ASK into a confirmation prompt.
Ports `openjiuwen/harness/security/permission_engine/` and `openjiuwen/harness/rails/security/tool_security_rail.py` (Python v0.1.20).
Fidelity target: decision outcomes, `matched_rule` label strings, `[PERMISSION_DENIED] …` tool results, the confirm answer schema and session/auto-confirm semantics; internals are idiomatic Rust.

## API Surface

Every public item is also re-exported flat from this module (`crate::permission::PermissionEngine`, …).

- `model` — `PermissionLevel {Allow, Ask, Deny}` (wire `allow`/`ask`/`deny`, `parse` = trim+lower, `Default = Ask`), `PermissionResult {permission, matched_rule, reason, external_paths}` (`is_allowed`/`is_denied`/`needs_approval`), `PermissionConfirmResponse {approved, feedback, auto_confirm, persist_allow}` (`wants_permanent_persist`/`wants_session_persist`/`parse`/`payload_schema`), `FileAction`, `PathAccess`, `RuleMatchType {Command, Path}`, `PathMatch {Prefix, Glob}`, `PermissionRule`, `ApprovalOverrideEntry`, `FileGuardDefaults/Section/PathEntry`, `NetGuardSection`, `ShellGuardSection`, `PermissionsSection`.
- `patterns` — `match_wildcard(value, pattern)`, `match_glob(pattern, path)`, `PolicyMatcher` (exact / wildcard / `re:` regex), `match_net_pattern(url, pattern)`, `extract_host`.
- `builtin` + `builtin_rules.yaml` — `include_str!` packaged floors: `package_command_rules()` (17 rules), `package_sensitive_paths()` (18 globs), `package_net_urls()` (4 deny hosts), `package_command_rule_count()`.
- `rules` — `normalize_tool_name`/`TOOL_NAME_ALIASES`, `DEFAULT_SHELL_TOOLS`/`PATH_TOOLS`/`NETWORK_TOOLS`/`OS_CONTROL_TOOLS`, `ToolCategory`, `shell_tools_from_config`/`is_shell_tool`/`tool_category`, `command_text`/`os_control_arg_text`, `rule_applies_to`/`rule_tools_consistent`/`rule_matches`/`rule_ref`, `approval_override_applies`, `normalize_permission_mode`, `severity_to_action`/`severity_to_decision`.
- `shell` — `INTERPRETER_SINKS`, `ShellKind {Simple, TooComplex, ParseUnavailable}`, `ShellFlags`, `ShellCommand`, `canonicalize_shell_command`, `strip_fd_alias_tokens`, `analyze_shell`, `has_interpreter_sink`, `shell_auto_confirm_key`.
- `paths` — `tool_action`, `tool_path_args`, `is_write_command`, `resolve_path`, `is_path_like`, `normalize_lexically`, `extract_file_accesses`, `extract_shell_accesses`.
- `tiered` — `evaluate_tiered_policy(section, tool, args) -> (PermissionLevel, String)` (never "no opinion"), `FALLBACK_NO_CONFIG = "tiered_policy:fallback(no_config)"`.
- `file_guard` — `FileGuardChecker::{build(section, workspace_root, trusted_dirs), evaluate, collect_ask_accesses, workspace_root}`; `None` checker when disabled, `None` decision when it has no opinion or would allow.
- `net_guard` — `NetGuardChecker::{build, evaluate}`, `extract_fetch_url`, `FETCH_TOOLS`, `URL_ARG_KEYS`; deny-only.
- `engine` — `PermissionEngineConfig {permissions, workspace_root, trusted_dirs, enabled}` (`new` + `with_*`), `PermissionEngine::{new, config, enabled, workspace_root, trusted_dirs, update_permissions, check_permission, evaluate_global_policy_directly}`, `strictest`, `strictest_all`, `prepare_permissions_for_engine`.
- `session` — `PermissionHost` (async `scene_hook`, `request_confirmation`, `persist_allow_rule`, `persist_session_allow_rule`; sync `permissions_snapshot`, `resolve_workspace_dir`, `session_id`, `permission_yaml_path`), `NoopPermissionHost`, `SceneHookInput`, `SceneDecision {Approve, Reject}`, `PermissionConfirmationRequest`, `ConfirmationOutcome {Interrupt, Response, Failed, Invalid}`, `SessionPermissionStore`, `MergeOutcome`, `merge_allow_rule_into_permissions`, `merge_file_guard_access_allows`, `should_store_auto_confirm`.
- `presentation` — `AskCategory {Shell, Path, Network, Tool, Finding}`, `AskPresentation {category, title, summary}`, `build_ask_presentation`, `render_ask_message`, `always_allow_hint`, `category_title`.
- `interrupt` — `PermissionInterruptRail` (`AgentRail`, name `permission_interrupt`, priority 90, kind `permission`), `RAIL_OWNER`, `RAIL_PRIORITY`, `RESUME_ANSWERS_KEY`, `AUTO_CONFIRM_KEY`, `PERSISTED_APPROVALS_KEY`, `DENIED_PREFIX`, `REJECTED_MESSAGE`, `auto_confirm_key`, `parse_confirm_payload`.

## Constraints

- Fail closed: parse errors, unknown structures, malformed levels, missing config and absent matching rules widen to ASK (or keep DENY); only an explicit allow rule, `approval_overrides`, `tools.<tool>` or `defaults["*"]` can allow.
- `Option<PermissionLevel>` = "no opinion", never allow; every fold goes through `strictest`/`strictest_all` (deny < ask < allow).
- Labels are API: `tiered_policy:fallback(no_config)` byte-exact, `tiered_policy:builtin:deny:<refs>`, `tiered_policy:builtin:builtin[id]`, `tiered_policy:rules:…`, `tiered_policy:approval_overrides:approval_overrides[ids]`, `tools.<tool>`, `tiered_policy:defaults.*`, `file_guard:prefix:<path>`/`file_guard:glob`/`file_guard:defaults`, `net_guard:url:<pattern>`/`net_guard:defaults`; pipelines A → file_guard → net_guard joined with `|`.
- `check_permission` is synchronous and non-blocking: no filesystem/network IO, compiled policy behind `parking_lot::RwLock`, shared as `Arc<PermissionEngine>`; locks are never held across `.await`.
- Config types are serde snake_case, tolerate unknown fields, and derive `JsonSchema` (no `serde_json::Value` fields) because `PermissionsSection` is embedded by `crate::schema::DeepAgentConfig`; `tools` accepts both `"bash": "ask"` and legacy `"bash": {"*": "ask"}` and serializes as the plain level string.
- No dependency on `crate::schema` or any other harness module — only kernel crates (`openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-agent`).
- No on-disk permission YAML load or save exists in this module or the crate: `builtin_rules.yaml` is `include_str!`-embedded, sections arrive from `PermissionsSection::default()` (CLI) or provider JSON, and `PermissionHost::permission_yaml_path` has no consumer.
- No `unsafe`, no `panic`/`unwrap`/`expect` outside `#[cfg(test)]`, `tracing` only (no stdout).
- Files ≤1000 lines of implementation; `model.rs` and `paths.rs` exceed that only because of their inline test modules (called out in a top-of-file size note).

## Design decisions

- Python's `await engine.check_permission(...)` becomes a sync `check_permission`: decisions need no IO. Only the rail and the host hooks are async.
- Python's tree-sitter shell AST is replaced by a conservative lexer (`shell::analyze_shell`): `TooComplex` for substitution/heredoc/subshell, `ParseUnavailable` for unbalanced quoting, `Simple` otherwise with top-level subcommand splitting. Risky or unknown structure widens to ASK exactly like Python's floor; a parser limitation can never allow.
- The tiered ladder is evaluated on both the raw argument text and the canonical command (strictest wins), so `bash -lc "rm -v x"` cannot hide the inner command, and compound `Simple` commands are aggregated per subcommand (a subcommand DENY beats a whole-command approval override).
- Python's `_allow_overrides_unless_denied` runs after the shell floors: a remembered exact command relaxes an ASK (including `interpreter_sink`/`too_complex` floors) but never a DENY.
- Builtin patterns are `regex`-crate syntax: Python lookaheads were rewritten as alternations of the accepted shapes (equivalent-or-stricter). A unit test asserts every packaged `re:` pattern compiles.
- File guard deviates from Python where Python touches the disk: paths are normalized LEXICALLY (no `Path::resolve`), glob patterns also get `~` expansion, and there is no implicit workspace-root allow rule (the workspace axes/defaults must say so). `trusted_dirs` project to prefix `allow/allow/ask` entries and enable the layer, replacing the deprecated `external_directory` section.
- `net_guard` keeps its rules in a `BTreeMap`, so deny patterns are scanned in sorted rather than insertion order — the decision is identical, only the reported pattern can differ.
- Remembered decisions are layered: `SessionPermissionStore` holds auto-confirm keys, the rail always applies the merged allow rule to the in-memory engine config, and durability is the host's call (`persist_allow_rule` = permanent, `persist_session_allow_rule` = session). The permission rail's interrupt carries an EMPTY `auto_confirm_key` (Python parity), so the kernel never stores it — the rail stores it itself.
- Persist-once across resume rounds is tracked per tool-call id in `ctx.extra[PERSISTED_APPROVALS_KEY]`; the kernel has no "carried answer" flag.

## Deferred Python areas

- `external_directory` legacy section (superseded by `file_guard` + `trusted_dirs`).
- Literal-shell allow shortcut (`literal_shell.py`, `parse_literal_shell`).
- `ask_presentation.py` suggestion chips/categories beyond category/title/summary and the always-allow hint.
- YAML write-back of the `permissions:` block (`write_permissions_section_to_agent_config_yaml`) — belongs to the CLI/config layer.
- `permission_checks_active` predicate seam (`set_permission_checks_active`).
- `skill_authorization/` (a separate security axis, not this module).

## Routing Table

- `./model.rs` → permission levels/results, confirmation payload, config sections and rule data
- `./patterns.rs` → wildcard/glob/regex/URL matchers
- `./builtin.rs` + `./builtin_rules.yaml` → packaged command rules, sensitive paths, metadata URLs
- `./rules.rs` → tool-name normalization, categories, rule applicability, labels, severity mapping
- `./shell.rs` → command canonicalization, structure analysis, auto-confirm keys
- `./paths.rs` → path-access extraction and lexical resolution
- `./tiered.rs` → pipeline A tier ladder, approval-override relaxation, shell floors
- `./file_guard.rs` → path read/write/exec guard (pipeline B)
- `./net_guard.rs` → URL guard (pipeline C)
- `./engine.rs` → `PermissionEngine`, fold/reasons, `prepare_permissions_for_engine`
- `./session.rs` → host trait, session store, allow-rule merges, auto-confirm policy
- `./presentation.rs` → ask category/title/summary and message rendering
- `./interrupt.rs` → `PermissionInterruptRail` (must be registered by the agent layer at priority 90)
- `./tests.rs` → cross-module integration suite; per-module unit tests live inline in each file

## Test Strategy

- Per-module inline unit tests: defaults/serde tolerance (`model`), matcher semantics (`patterns`), packaged-rule compile + counts (`builtin`), category/severity tables (`rules`), canonicalization/structure/auto-confirm keys (`shell`), extraction + lexical resolution (`paths`), ladder precedence and labels (`tiered`), prefix/glob/implication folds (`file_guard`), ask-dropping and authority matching (`net_guard`), reason templates and packaged injection (`engine`), merge shapes and auto-confirm truth table (`session`), titles/summaries (`presentation`), rail allow/deny/ask/remember/deny-after-approval/persist-once (`interrupt`).
- `tests.rs` drives the seams end to end: pipeline folding, packaged floors, shell floors + remembered relaxations, net/sensitive-path floors, the session remember loop, hosted confirmation, Python-shaped config JSON parity, `JsonSchema` generation, interrupt wire stability, rail priority ordering, trusted dirs.
- A clean policy needs `PermissionsSection { package_builtin_rules: false, .. }`; `PermissionsSection::default()` inlines the 17 packaged command rules and sets `defaults["*"] = ask` (file/net guards stay disabled).
- There is no per-rule mutation API: rules are registered by replacing the whole section through `PermissionEngine::update_permissions` (construction takes `PermissionEngineConfig`); `prepare_permissions_for_engine`/`builtin::package_command_rules` are the standalone ingest helpers.
- `PermissionRule` with `match_type: Path` never matches: pipeline A matches only shell command text and `os_control` arguments; path policy comes from `file_guard.paths`, which does not read `section.rules`.
- `PermissionRule.tools` entries are exact names (plus the `"shell"` category token); `"*"` is a literal tool name, not a wildcard.
- No `Ord`/`PartialOrd` on `PermissionLevel` (fold with `strictest`/`strictest_all`) and no `Clone`/`Default` on `PermissionEngine` (share as `Arc`).
- Verification: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness` (this module: 249 unit + 3 doc tests; crate-wide 1088 unit + 80 integration + 8 doc tests, also with `--no-default-features`), `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` (clean crate-wide), `cargo doc --no-deps` warning-free.
- Parity references: `tests/unit_tests/harness/security/*` and `tests/unit_tests/harness/rails/test_typed_interrupts.py` in the Python reference.
