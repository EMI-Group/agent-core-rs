# src/memory — team memory wiring

## Intent

The runtime half of team memory: turn the `openjiuwen_memory::team` engine into something a
`TeamAgent` drives — a per-team wiring handle, the metadata config that selects the backend, the
model-facing tools each member mounts, the `TEAM_MEMORY.md` export/import, and the observer bridge.
Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
`MemberMemoryToolkit`, `extract_team_memories` and the shared `TEAM_MEMORY.md` summary.
The engine (records, store, facade, extractors, hooks, toolkit, observer seam) stays in
`openjiuwen_memory::team`; this node never re-implements it.

## API Surface

```rust
pub const MEMORY_METADATA_KEY: &str = "memory";
pub use config::{MemoryConfig, MemoryStoreConfig, MemoryExtractorKind, memory_config};
pub use wiring::{TeamMemoryWiring, TEAM_MEMORY_FILE};
pub use tools::TeamMemoryTool;
```

## Constraints

- Only `crates/openjiuwen-agent-teams/src/memory/**` is edited here.
- Errors are `openjiuwen_common::error::{Error, Result}`; user-supplied config never panics.
- Hook failures are logged and never fail a member lifecycle; every multi-record write goes
  through one batch call (`remember_many`), so a failed pass leaves no partial set.
- `tracing` only; no `println!`/`eprintln!`.

## Routing Table

- `./config.rs` → `MemoryConfig`, store/extractor selection, the tolerant `metadata["memory"]` parser
- `./wiring.rs` → `TeamMemoryWiring`: facade + hooks + per-member toolkits + `TEAM_MEMORY.md` export/import
- `./tools.rs` → `TeamMemoryTool`: the four `MemberMemoryToolkit` tools as `openjiuwen_foundation::tool::Tool` handles
