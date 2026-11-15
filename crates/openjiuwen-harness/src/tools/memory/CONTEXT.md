# memory

## Intent

The eight model-facing memory tools, split into the backend seam, the tool-kind enum and the tool implementation: ports `openjiuwen/harness/tools/memory.py` and `coding_memory.py`.
Storage and retrieval sit behind `MemoryToolBackend` (the SQLite/embedding hybrid lives in `openjiuwen_memory`); this module owns only the model-facing contract — cards, Python validation messages, payload shapes and per-kind `Tool::render_for_llm` strings.

## API Surface

- `mod.rs` — the `//!` port docs, module declarations and the re-exports consumed by `crate::tools::mod`: `MemoryScope`, `MemorySearchRequest`, `MemoryReadData`, `MemoryWriteData`, `MemoryEditData`, `MemoryToolBackend`, `MemoryToolKind`, `MemoryTool`.
- `backend.rs` — the `MemoryToolBackend` async trait (`search`, `read`, `write`, `edit`) and the four data structs plus `MemoryScope`.
- `kinds.rs` — `MemoryToolKind` (`Search`, `Get`, `Write`, `Read`, `Edit`, `CodingRead`, `CodingWrite`, `CodingEdit`) with `ALL`, `name`, `from_name`.
- `tool.rs` — `MemoryTool` (card building, per-kind arg structs, invoke dispatch, payload builders, schema defaults, render), the default constants and `NO_BACKEND_MESSAGE`.
- `tests.rs` — the per-kind card/payload/render parity table with the `FakeBackend` fixture, moved verbatim.

## Constraints

- Tool names, JSON schemas, payload shapes and message text are unchanged from the pre-split module.
- No backend configured → `success=false` with `memory tool backend is not configured` (`tool.rs::NO_BACKEND_MESSAGE`; checked after the required-argument validation, so a missing argument is reported first).
- No production `MemoryToolBackend` implementation exists in the crate or anywhere in the workspace — only the unit-test `FakeBackend` in `tests.rs`; a real adapter over `openjiuwen_memory` is the host's job, wired through `ToolRegistry::with_memory`. The crate's other memory seam, `rails::memory::LongTermMemoryAdapter`, implements `MemoryRecall`/`MemoryWriteBack` (not this trait) over `openjiuwen_memory::LongTermMemory`.
- `MemoryToolKind` is the only selector of card/scope/render shape; per-kind arg structs stay private to `tool.rs`.
- Files ≤1000 lines; no duplicated card/payload scaffolding between kinds.

## Routing Table

- `./mod.rs` → docs, declarations, re-exports
- `./backend.rs` → `MemoryToolBackend` trait, scope and data structs
- `./kinds.rs` → `MemoryToolKind`
- `./tool.rs` → `MemoryTool`, cards, invoke/render, payload and schema helpers
- `./tests.rs` → `#[cfg(test)] mod tests`
