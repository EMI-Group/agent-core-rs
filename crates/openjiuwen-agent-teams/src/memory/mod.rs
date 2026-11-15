//! Team-memory wiring: the `openjiuwen-memory` team facade inside a team runtime.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
//! `MemberMemoryToolkit` and the shared `TEAM_MEMORY.md` summary. The memory engine itself lives
//! in [`openjiuwen_memory::team`]; this node owns the *runtime* half: which members get a
//! toolkit, when the lifecycle hooks run, where the team summary is exported, and which observer
//! sees the operations.
//!
//! The node has four parts, each documented in its own module:
//!
//! - [`config`] reads the optional `memory` block out of
//!   [`TeamAgentSpec`](crate::schema::TeamAgentSpec) metadata ([`memory_config`]);
//! - [`wiring`] builds the facade, the lifecycle hooks, the per-member toolkits and the
//!   `TEAM_MEMORY.md` export/import ([`TeamMemoryWiring`], [`TEAM_MEMORY_FILE`]);
//! - [`tools`] projects a member toolkit onto the crate-wide tool seam ([`TeamMemoryTool`],
//!   [`team_memory_tools`]);
//! - [`trace`] attaches the crate's team-memory observer where the facade is built, so every
//!   operation is visible through [`observability`](crate::observability).
//!
//! ```
//! use openjiuwen_agent_teams::memory::{MemoryConfig, MemoryStoreConfig};
//!
//! let config = MemoryConfig::default();
//! assert!(config.enabled);
//! assert!(config.trace);
//! assert!(config.otel_sink.is_none());
//! assert!(config.observer.is_none());
//! assert!(!config.store.is_file());
//! assert_eq!(config.store, MemoryStoreConfig::Memory);
//! assert!(config.validate().is_ok());
//! ```
//!
//! # Observability
//!
//! A traced configuration ([`MemoryConfig::trace`], on by default) makes the facade report every
//! observable operation: one `memory.{operation}` span and one structured `tracing` event at
//! target `openjiuwen_agent_teams::memory` per operation, plus — on the `otel` build — one
//! `SpanRecord` in the sink [`MemoryConfig::otel_sink`] carries.
//!
//! A host that owns its own sink supplies [`MemoryConfig::observer`] instead (the OTLP bridge on
//! the `otlp` build, for instance), or opens through
//! [`TeamMemoryWiring::open_with_observer`](crate::memory::TeamMemoryWiring::open_with_observer):
//! the supplied observer replaces the crate's bridge, and `trace: false` still attaches nothing.
//!
//! Two honest limitations:
//!
//! - The bridge roots its own `team.{team_name}` trace, so these spans are a *sibling* trace of
//!   the runtime's team trace rather than children of it; joining them needs a
//!   `MemorySpanObserver::from_root(TeamSpan)` constructor in the observability node.
//! - Real OTLP export is deferred: the exporter lives in `openjiuwen-extensions`, which is in this
//!   crate's dependency layer and cannot be depended on here, so the `otel` feature only fills the
//!   in-memory sink.
pub mod config;
pub mod tools;
pub mod trace;
pub mod wiring;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
pub(crate) mod test_support;

pub use config::{
    memory_config, MemoryConfig, MemoryExtractorKind, MemoryStoreConfig, MEMORY_METADATA_KEY,
};
pub use tools::{team_memory_tools, TeamMemoryTool};
pub use trace::OtelSpanSink;
pub use wiring::{TeamMemoryWiring, TEAM_MEMORY_FILE};
