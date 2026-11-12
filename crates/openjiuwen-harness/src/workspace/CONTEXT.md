# workspace

## Intent

The agent's on-disk workspace: named directory nodes, the default bilingual schema (`AGENT.md`, `memory/`, `todo/`, `skills/`, …), team/worktree link management, and safe directory materialization through `openjiuwen_sysop` or the local filesystem.
Ports `openjiuwen/harness/workspace/{workspace.py,directory_builder.py}`.

## API Surface

- `WorkspaceNode` (string enum): `AGENT_MD`, `SOUL_MD`, `HEARTBEAT_MD`, `IDENTITY_MD`, `USER_MD`, `MEMORY`, `CODING_MEMORY`, `TODO`, `MESSAGES`, `SKILLS`, `AGENTS`, `MEMORY_MD`, `DAILY_MEMORY`, `TEAM_LINKS` (".team"), `WORKTREE_LINKS` (".worktree").
- `DirectoryNode { name, path (relative), description, children, is_file, default_content }`.
- `Workspace { root_path: PathBuf (default "./"), directories: Vec<DirectoryNode>, language: String ("cn") }`:
  `new(root_path)`, `with_language`, `validate`, `ensure_default_schema` (append missing default nodes; defaults cannot be opted out), `get_directory(name) -> Option<&str>`, `get_node_path(name) -> Option<PathBuf>` (top-level only), `set_directory(nodes)`, `resolve(path) -> Result<PathBuf>` (sandbox check), `link_team`/`unlink_team`/`link_worktree`/`unlink_worktree`/`list_team_links`/`list_worktree_links` (`.team`/`.worktree` subdirs), `get_workspace_schema(language) -> Vec<DirectoryNode>`.
- `DirectoryBuilder { sys_operation: SysOperation, root_path: PathBuf }`: `build(&[DirectoryNode])` creates directories/files through sysop; file nodes write only when absent; directory nodes get a `.workspace` marker; unsafe paths are rejected.
- `is_safe_relative_path(path) -> bool`: rejects empty/absolute paths, leading `/`/`\`, drive letters, UNC prefixes, and any `..` segment.

## Constraints

- Path sandboxing is the security boundary: every model-supplied path is resolved through `Workspace::resolve` or `is_safe_relative_path`; symlink escapes are rejected after canonicalizing the existing prefix.
- Uses `openjiuwen_sysop` for I/O when available (so sandbox backends work); local `std`/`tokio::fs` only for the fallback constructor.
- Default schema is loaded from packaged resources under `src/workspace/resources/{cn,en}/**` (include_str!), mirroring Python `prompts/workspace_content`; missing resource → empty content, never an error.
- `get_node_path` is top-level only (nested lookups return `None`), matching Python.
- No panics; errors are `Error::config`/`Error::validation`/`Error::sys_op`.

## Routing Table

- `./node.rs` → `WorkspaceNode`, `DirectoryNode`
- `./workspace.rs` → `Workspace` (lookup, default-schema merge, links, resolution)
- `./directory_builder.rs` → `DirectoryBuilder`, `is_safe_relative_path`
- `./schema.rs` → default cn/en workspace schema and packaged markdown resources
- `./resources/` → packaged `AGENT.md`, `SOUL.md`, `MEMORY.md`, `session_memory.md` content per language

## Test Strategy

- Unit tests: default schema content differs by language, auto-supplement adds missing nodes idempotently, nested `get_node_path` returns `None`, `is_safe_relative_path` rejects `..`/absolute/drive/UNC, `DirectoryBuilder` writes only missing files and creates the `.workspace` marker.
- Parity references: `tests/unit_tests/harness/test_deep_agent_workspace.py`, `test_workspace_links.py`.
