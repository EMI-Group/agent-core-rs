# memory

## Intent

The eight model-facing memory tools, split into the backend seam, the file-backed implementation, the tool-kind enum and the tool implementation: ports `openjiuwen/harness/tools/memory.py` and `coding_memory.py`.
Storage and retrieval sit behind `MemoryToolBackend`; `WorkspaceMemoryBackend` is the production implementation over a harness `Workspace`, and the tool layer owns only the model-facing contract — cards, Python validation messages, payload shapes and per-kind `Tool::render_for_llm` strings.

## API Surface

- `mod.rs` — the `//!` port docs, module declarations and the re-exports consumed by `crate::tools::mod`: `MemoryScope`, `MemorySearchRequest`, `MemoryReadData`, `MemoryWriteData`, `MemoryEditData`, `MemoryToolBackend`, `MemoryToolKind`, `MemoryTool`, `WorkspaceMemoryBackend`.
- `backend.rs` — the `MemoryToolBackend` async trait (`search`, `read`, `write`, `edit`) and the four data structs plus `MemoryScope`.
- `workspace_backend.rs` — `WorkspaceMemoryBackend` (`new(Workspace)`, `workspace()`): Python path mapping, line windows, coding-memory front-matter rules and the offline keyword search.
- `workspace_backend_tests.rs` — `#[cfg(test)] mod tests` of the backend (path mapping, line windows, tokenization, front-matter messages), pulled in through `#[path]` so the implementation file stays within budget.
- `kinds.rs` — `MemoryToolKind` (`Search`, `Get`, `Write`, `Read`, `Edit`, `CodingRead`, `CodingWrite`, `CodingEdit`) with `ALL`, `name`, `from_name`.
- `tool.rs` — `MemoryTool` (card building, per-kind arg structs, invoke dispatch, payload builders, schema defaults, render), the default constants and `NO_BACKEND_MESSAGE`.
- `tests.rs` — the per-kind card/payload/render parity table with the `FakeBackend` fixture, moved verbatim.

## Workspace backend

`WorkspaceMemoryBackend` ports `openjiuwen/core/memory/lite/memory_tool_ops.py` (storage half) and `coding_memory_tool_ops.py` (front-matter half).
`new` performs no I/O: every root is read from the workspace schema on demand (`get_directory`), so a backend can be built before the directories exist.

| model path | resolved workspace path (`MemoryScope::Memory`) |
| --- | --- |
| `USER.md` | `<workspace>/USER.md` |
| `MEMORY.md` | `<workspace>/<memory>/MEMORY.md` |
| `YYYY-MM-DD.md` | `<workspace>/<memory>/<daily_memory>/YYYY-MM-DD.md` |
| anything else | `<workspace>/<memory>/<basename>` |

`MemoryScope::CodingMemory` maps `*.md` to `<workspace>/<coding_memory>/<basename>` and rejects any other extension with `Path must end with .md`.
Only the basename of a model-supplied path is used (`os.path.basename`); an absolute path or any `..` segment is rejected up front with `Invalid path: directory traversal not allowed`, and every resolved path then goes through `Workspace::resolve`, so a symlink escaping the workspace is an `Error::security`.
Reported `path`/`full_path` are the workspace-relative path (`memory/MEMORY.md`) and the absolute path.

- `read` windows the whole file by 1-based `offset` (`0` = first line) and `limit`, reporting the real `total_lines`; a missing file is `Error::not_found`.
- `write` creates parent directories, reports `appended`/`file_existed` (non-empty before the write) and, when `append` is set, prepends a newline unless the existing file already ends with one.
- `edit` replaces exactly one occurrence: `old_text not found in file. Use read_memory tool to check exact content.`, `old_text appears N times in file. Be more specific.`; coding memory keeps Python's own wording (`old_text not found in file`, `old_text appears N times, please be more specific`, `old_text cannot be empty`).
- Coding-memory `write` requires `---` front matter with `name`/`description`/`type`, `type` one of `user`/`feedback`/`project`/`reference`, and a non-empty body; it renders canonical front matter (via `openjiuwen_memory::lite::markdown::FrontMatter`) with `created_at` on creation and a fresh `updated_at`.
- Coding-memory `append` validates the new chunk, appends only its body to the existing file and refreshes `updated_at` (Python's `_append_to_existing_file`); Python's LLM conflict/redundancy checker and the `MEMORY.md` index upsert are not ported.
- `search` is a deterministic offline keyword scan: it walks `<memory>/**/*.md`, `<memory>/sessions/**/*.jsonl`, `USER.md` and (coding scope) `<coding_memory>/**/*.md`, skips files over 4 MiB, searches at most 512 files (sorted by relative path) and groups contiguous matching lines into `start_line`/`end_line` windows scored as the share of query tokens present (0..=1).
- `search` returns the Python object: `results[]` (`id`, `path`, `source`, `start_line`, `end_line`, `score`, `snippet` capped at 700 chars, `citation` = `path#L<n>` or `path#L<a>-L<b>`), `provider: "local"`, `model: null`, `search_mode: "keyword_only"`, `embedding_available: false`, `embedding_error: null`, `disabled: false`; a missing tree yields empty results, never an error.
- `session_key`, when given, keeps only session-log hits under `sessions/<session_key>/` (Python's lite manager ignores the option).
- Query tokens are lowercased alphanumeric runs with duplicates removed; runs shorter than 3 characters are dropped unless every run is that short.

## Constraints

- Tool names, JSON schemas, payload shapes and message text are unchanged from the pre-split module.
- No backend configured → `success=false` with `memory tool backend is not configured` (`tool.rs::NO_BACKEND_MESSAGE`; checked after the required-argument validation, so a missing argument is reported first). `ToolRegistry::new` leaves the backend unset; `ToolRegistry::with_workspace_memory` installs `WorkspaceMemoryBackend`, `with_memory` substitutes a host backend, and `DeepAgentBuilder::build` mounts the workspace backend by default.
- The crate's other memory seam, `rails::memory::LongTermMemoryAdapter`, implements `MemoryRecall`/`MemoryWriteBack` (not this trait) over `openjiuwen_memory::LongTermMemory`.
- No embeddings, no SQLite index and no LLM in this backend: search is keyword-only and the result object says so (`search_mode: "keyword_only"`, `embedding_available: false`).
- `MemoryToolKind` is the only selector of card/scope/render shape; per-kind arg structs stay private to `tool.rs`.
- Files ≤1000 lines; no duplicated card/payload scaffolding between kinds; no blocking I/O on the async path (`tokio::fs` only).

## Routing Table

- `./mod.rs` → docs, declarations, re-exports
- `./backend.rs` → `MemoryToolBackend` trait, scope and data structs
- `./workspace_backend.rs` → `WorkspaceMemoryBackend` (path mapping, line windows, coding front matter, keyword search)
- `./workspace_backend_tests.rs` → `#[cfg(test)] mod tests` of the backend
- `./kinds.rs` → `MemoryToolKind`
- `./tool.rs` → `MemoryTool`, cards, invoke/render, payload and schema helpers
- `./tests.rs` → `#[cfg(test)] mod tests`
