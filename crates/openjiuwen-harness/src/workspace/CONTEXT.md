# workspace

## Intent

The agent's on-disk workspace: named directory nodes, the default bilingual schema (`AGENT.md`, `memory/`, `todo/`, `skills/`, `context/`, …), safe directory materialization through `openjiuwen_sysop`, and team/worktree link management.
Ports `openjiuwen/harness/workspace/{workspace.py,directory_builder.py}` plus the packaged `harness/prompts/workspace_content` assets (v0.1.20).
This module is the security boundary for every model-supplied path in the harness: tools resolve paths through it before touching the filesystem.

## API Surface

Re-exported flat from `openjiuwen_harness::workspace`:

- `WorkspaceNode` (string enum) — `AgentMd` `"AGENT.md"`, `SoulMd` `"SOUL.md"`, `HeartbeatMd` `"HEARTBEAT.md"`, `IdentityMd` `"IDENTITY.md"`, `UserMd` `"USER.md"`, `Memory` `"memory"`, `CodingMemory` `"coding_memory"`, `Todo` `"todo"`, `Messages` `"messages"`, `Skills` `"skills"`, `Agents` `"agents"`, `MemoryMd` `"MEMORY.md"`, `DailyMemory` `"daily_memory"`, `TeamLinks` `".team"`, `WorktreeLinks` `".worktree"`; `as_str()`, `Display`, `FromStr` (unknown → `Error::validation`), `From<WorkspaceNode> for &'static str`, `ALL`.
- `DirectoryNode { name: String, path: Option<String>, description: Option<String>, children: Vec<DirectoryNode>, is_file: bool, default_content: String }` — serde snake_case, absent optional keys omitted, unknown keys ignored, `Default`, `PartialEq`, `JsonSchema`; `file(name, path, content)`, `dir(name, path)`, `with_description(..)`, `with_children(..)`, `resolved_path()`, `validate()`, `is_file()`, `content()`.
- `Workspace { root_path: PathBuf, directories: Vec<DirectoryNode>, language: String }` — `new(root)`, `for_language(root, language)`, `with_language(lang)`, `with_directories(Vec<DirectoryNode>) -> Result<Self>`, `validate() -> Result<()>`, `ensure_default_schema() -> usize`, `get_directory(&str) -> Option<String>`, `get_directory_for(WorkspaceNode)`, `get_node_path(&str) -> Option<PathBuf>`, `get_node_path_for(WorkspaceNode)`, `set_directory(DirectoryNode) -> Result<()>`, `set_directories(iter) -> Result<()>`, `resolve(&str) -> Result<PathBuf>`, `resolve_existing(&str) -> Result<PathBuf>`, `is_english()`, `language()`, and the associated function `get_default_directory(language) -> Vec<DirectoryNode>`.
- Links — `TEAM_LINKS_DIR` (`".team"`), `WORKTREE_LINKS_DIR` (`".worktree"`), `LINK_MARKER` (`".workspace-link"`), `WorkspaceLink { name, target: PathBuf, kind: LinkKind }`, `LinkKind::{Symlink, Directory}`; `Workspace::{link_team, unlink_team, list_team_links, link_worktree, unlink_worktree, list_worktree_links}`.
- `DirectoryBuilder { sys_operation: SysOperation, root_path: PathBuf }` — `new`, `build(&[DirectoryNode]) -> Result<()>`; `WORKSPACE_MARKER` (`".workspace"`).
- `is_safe_relative_path(&str) -> bool`.
- `get_workspace_schema(language) -> Vec<DirectoryNode>` — independent tree per call; only the exact string `"en"` selects English, everything else is `cn` (Python parity).
- `default_content(language, relative_path) -> &'static str` — packaged resource text, `""` when unknown.

## Constraints

- Path sandboxing is the security boundary: every model-supplied path goes through `Workspace::resolve` / `resolve_existing` or `is_safe_relative_path`. `resolve` rejects empty/absolute/drive-letter/UNC/`..`-escaping paths lexically, then canonicalizes the longest existing prefix (reusing `openjiuwen_sysop::workspace::{clean_path, canonicalize_containing, is_within}`) so symlink escapes are rejected (`Error::security`).
- Path resolution failures: `Error::validation` for a syntactically unsafe path, `Error::security` for an escape, `Error::not_found` from `resolve_existing`.
- Default schema is packaged: `include_str!` from `./resources/{cn,en}/**`; an unknown language or resource key yields `""`, never an error, and never a cross-language fallback.
- `cn` has 12 top-level nodes including `coding_memory`; `en` has 11 (no `coding_memory`) — a consumer relies on `get_node_path("coding_memory")` being `None` in English.
- `ensure_default_schema` supplements only top-level nodes, matched by `name` (caller node wins, default order preserved, defaults cannot be opted out). `get_node_path` is top-level only (nested lookups return `None`) and falls back to the node name when `path` is absent.
- `DirectoryBuilder` creates files only when absent (existence probed through the sysop backend, not the process filesystem) and writes an empty `.workspace` marker into every directory node; unsafe node paths abort before any I/O for that node.
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics, no `unwrap` outside tests. `tracing` only, no stdout.
- Link methods are synchronous filesystem setup helpers (Python parity): symlink on Unix, directory symlink on Windows, recursive copy when symlinking is denied; listing is sorted by name and canonicalizes targets.
- Files ≤1000 lines; `Workspace` is `Clone + Debug + Eq + PartialEq + Serialize + Deserialize + Default + JsonSchema`; `schemars` uses `#[schemars(with = "String")]` for `PathBuf` fields.
- Items are reachable only as `openjiuwen_harness::workspace::*` (plus `Workspace` methods); the crate root declares `pub mod workspace` and no flat re-exports, so integration tests must use the qualified path.
- The submodules (`node`, `workspace`, `resolve`, `links`, `directory_builder`, `schema`) and the free fns `find_node`/`path_to_string`/`file_exists` are private; only the names in `./mod.rs` are re-exported.
- `DirectoryBuilder`'s `sys_operation` and `root_path` fields are public and `build` needs a filesystem backend (`SysOperation::local(LocalWorkConfig::unrestricted(root))?`); `Workspace::new`/`for_language` accept `impl Into<PathBuf>` and perform no I/O, so a `TempDir` must be passed as `temp.workspace()` / `temp.root().to_path_buf()`.

## Design decisions

- `DirectoryNode` is a typed struct, not Python's `Dict[str, Any]`; validation replaces `_validate_directory_node` and reports the Python messages (`Directory `name` must be a non-empty string.`, …) through `Error::validation`.
- `Workspace` deserialization normalizes like Python's `__post_init__`: an empty/absent `directories` list becomes the language default, then missing defaults are supplemented and the tree validated (invalid input becomes a deserialization error).
- `with_language` reconciles the schema: untouched defaults are swapped for the new language's schema, custom directories are preserved and supplemented.
- `resolve`/`resolve_existing` are async (canonicalization uses tokio); `get_node_path`, `get_directory` and the link helpers are synchronous.
- A directory node is identified on disk by its `.workspace` marker; markers and seeded files are written with `prepend_newline = false` so an empty default content produces a zero-byte file (Python's local backend writes a lone `"\n"`; documented deviation).
- Fresh `.workspace` markers are written only when absent (Python rewrites them on every build), so `build` is idempotent and cheap.
- `list_team_links`/`list_worktree_links` report both symlinks and copy-fallback directories with a `LinkKind` (Python silently hides copied links, which would make a degraded link invisible to the agent).
- Both languages seed `USER.md` with a short preferences stub; Python ships no `USER.md` resource (it creates an empty file), so the seeded note is a documented deviation.
- `get_directory` searches nested nodes first and only then falls back to the language defaults; the fallback builds a fresh default tree so no reference into the workspace is returned.
- `mod workspace` keeps the Python file name, hence `#[allow(clippy::module_inception)]` on the declaration.
- `tests.rs` is legitimately long (~1050 lines, 49 tests): it is the single cross-file suite for this module and splitting it would scatter closely related fixtures.

## Deferred Python areas

- Windows junction creation (`cmd /c mklink /J`, `winerror 1314`) and reparse-point detection: the Rust port relies on `symlink_dir` and falls back to a recursive copy.
- `Workspace.set_directory` deep-copy quirks and import-time module-level schema mutation: Rust returns fresh trees per call.
- Per-language prompt header constants (`workspace_header.py`) belong to `prompts`, not this module.

## Routing Table

- `./node.rs` → `WorkspaceNode`, `DirectoryNode`, node validation
- `./workspace.rs` → `Workspace` construction, defaults, lookup, set, validation, serde
- `./links.rs` → `.team`/`.worktree` link creation, removal, listing, `WorkspaceLink`
- `./resolve.rs` → `Workspace::resolve`/`resolve_existing` sandbox checks (impl block)
- `./directory_builder.rs` → `DirectoryBuilder`, `is_safe_relative_path`, marker writing
- `./schema.rs` → default cn/en schema topology and `default_content` resource lookup
- `./resources/` → packaged `AGENT.md`, `SOUL.md`, `HEARTBEAT.md`, `IDENTITY.md`, `USER.md`, `memory/MEMORY.md`, `context/session_memory.md` per language
- `./tests.rs` → `#[cfg(test)]` cross-file unit tests (schema, builder, links, resolution, serde)

## Test Strategy

- Unit tests live beside each module plus `./tests.rs` (`#[cfg(test)]`), driven by `openjiuwen_test_support::TempWorkspace` and `tempfile`; every test is offline and deterministic.
- Schema: cn/en top-level names and order, en lacking `coding_memory`, bilingual descriptions, required literal content (`智能体`, `This folder is home`), independent trees, unknown language → cn, missing resource → `""`.
- Workspace: default construction, auto-supplement idempotency, nested `get_node_path` → `None`, `get_directory` fallback for a custom parent, `set_directory` upsert/validation, JSON round-trip and normalization on deserialize.
- Builder: files only when absent, zero-byte empty content, `.workspace` markers for nested directories, unsafe paths rejected before I/O, idempotent rebuild, full cn schema materialized.
- Links: symlink target/content, idempotent relink, sorted listing (`beta, alpha, gamma` → `alpha, beta, gamma`), unlink true/false, dangling symlinks listed from the `read_link` fallback without panicking, copy-fallback links discovered through `LINK_MARKER`, foreign files neither listed nor deleted.
- Resolution: inside paths allowed (including missing leaves), `..`/absolute/drive/UNC/empty rejected, symlink escape rejected (Unix), `resolve_existing` missing → `Error::NotFound`.
- Parity references: `tests/unit_tests/harness/test_deep_agent_workspace.py`, `test_workspace_links.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness -- --check`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` all clean (49 unit tests + 1 doc test).

## Known Issues

- Crate-level `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` still fails on three pre-existing scaffold lints in `../lib.rs` (outside this node): `doc_markdown` on `DeepAgent` (lines 1 and 4) and `must_use_candidate` on `crate_name` (line 28). This module itself is clippy-clean; the fix belongs to the crate root.

