# permission

## Intent

Tool-permission policy for the DeepAgent: a tiered engine (builtin rules, user rules, tool defaults), file-path guard, network guard and shell guard folded into one allow/ask/deny outcome, plus the HITL interrupt rail that turns an ASK into a confirmation prompt.
Ports `openjiuwen/harness/security/permission_engine/` and `rails/security/tool_security_rail.py`.

## API Surface

- `model.rs`: `PermissionLevel {Allow, Ask, Deny}` (wire `allow`/`ask`/`deny`), `PermissionResult {permission, matched_rule, reason, external_paths}` with `is_allowed`/`is_denied`/`needs_approval`; `PermissionConfirmResponse {approved, feedback, auto_confirm, persist_allow}` with `wants_permanent_persist`/`wants_session_persist`; `RuleMatchType {Command, Path}`, `ApprovalOverrideEntry`, `FileGuardPathEntry`, `NetGuardSection`, `ShellGuardSection`, `PermissionsSection {enabled, package_builtin_rules, defaults, tools, categories, rules, approval_overrides, file_guard, net_guard, shell_guard}` with Python defaults (`enabled=true`, `defaults["*"]="ask"`, `file_guard` default `read=allow,write=allow,exec=ask`).
- `engine.rs`: `PermissionEngine::new(PermissionEngineConfig)` (config, workspace_root, trusted_dirs, enabled), `check_permission(tool_name, args) -> PermissionResult` (rail-facing: unknown → ASK with `matched_rule="default"`), `evaluate_global_policy_directly(...) -> (Option<PermissionLevel>, Option<String>)`, `strictest(a,b)` fold (deny < ask < allow), `prepare_permissions_for_engine` (inline builtin rules, legacy `severity` mapping).
- `rules.rs`: exact / wildcard (`*?[`) / regex (`re:` prefix) matching, tool categories (`shell`, `path`, `network`, `os_control`), builtin rule set packaged from Python `resources/builtin_rules.yaml`, rule labels `tiered_policy:<ns>:<level>:<ids>`.
- `file_guard.rs`: read/write/exec actions, `prefix`/`glob` match (`**/` semantics), write/exec allow implies read allow, overall-allow returns `None` (never lifts a deny), longest-prefix-wins with strictest fold.
- `net_guard.rs`: fetch-tool URL rules; only allow/deny (ask entries dropped); returns DENY or `None`.
- `shell.rs`: unknown-structure / interpreter-sink widening to ASK (fail closed), literal command canonicalization.
- `session.rs`: `PermissionHost` trait (`request_confirmation`, `persist_allow_rule`, `persist_session_allow_rule`, `resolve_workspace_dir`) and `SessionPermissionStore` (in-memory session approvals).
- `interrupt.rs`: `PermissionInterruptRail` (priority 90) resolving every tool call: host scene hook → engine check → session auto-confirm → host confirmation → interrupt; DENY rejects with `[PERMISSION_DENIED] reason`.

## Constraints

- Fail closed: parse errors, unknown structures, missing config and absent rules always widen to ASK or keep DENY; never silently ALLOW.
- `None` means "no opinion" and is never treated as allow; all folds use `strictest`.
- No blocking I/O inside checkers; YAML is parsed once at construction.
- Decisions are plain serde data; the engine is an owned runtime object shared as `Arc<PermissionEngine>`.
- Integrates with `openjiuwen_agent::{BaseInterruptRail, ConfirmInterruptRail, InterruptRequest}`; no dependency on `crate::schema`.
- Errors: `Error::security` for denials, `Error::validation` for bad config.

## Routing Table

- `./model.rs` → permission levels/results, confirmation response, section and rule data types
- `./engine.rs` → `PermissionEngine`, strictest fold, tiered pipeline ordering
- `./rules.rs` → rule matching and the packaged builtin rule set
- `./file_guard.rs` → path read/write/exec guard
- `./net_guard.rs` → URL guard
- `./shell.rs` → shell-structure guard
- `./session.rs` → host trait, session-scoped decisions, persistence helpers
- `./interrupt.rs` → `PermissionInterruptRail`

## Test Strategy

- Unit tests: allow/deny/ask precedence (builtin deny > user deny > approval override > tool rule > default > ASK), regex/wildcard/exact matching, file-guard prefix/glob and implication rules, net-guard ask-dropping, session-scoped allow re-check, `strictest` ordering.
- Rail tests: allow once, allow always (persist), deny → skip tool; denial after resume wins.
- Parity references: `tests/unit_tests/harness/security/*` (net_guard, file_guard_native, approval_override_vs_deny, shell_guard, severity_mapping), `rails/test_typed_interrupts.py`.
