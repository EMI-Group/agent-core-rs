//! Team-memory wiring: the `openjiuwen-memory` team facade inside a team runtime.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
//! `MemberMemoryToolkit` and the shared `TEAM_MEMORY.md` summary. The memory engine itself lives
//! in [`openjiuwen_memory::team`]; this node owns the *runtime* half: which members get a
//! toolkit, when the lifecycle hooks run, and where the team summary is exported.
//!
//! The node has three parts, each documented in its own module:
//!
//! - [`config`] reads the optional `memory` block out of
//!   [`TeamAgentSpec`](crate::schema::TeamAgentSpec) metadata ([`memory_config`]);
//! - [`wiring`] builds the facade, the lifecycle hooks, the per-member toolkits and the
//!   `TEAM_MEMORY.md` export/import ([`TeamMemoryWiring`], [`TEAM_MEMORY_FILE`]);
//! - [`tools`] projects a member toolkit onto the crate-wide tool seam ([`TeamMemoryTool`],
//!   [`team_memory_tools`]).
//!
//! ```
//! use openjiuwen_agent_teams::memory::{MemoryConfig, MemoryStoreConfig};
//!
//! let config = MemoryConfig::default();
//! assert!(config.enabled);
//! assert!(!config.store.is_file());
//! assert_eq!(config.store, MemoryStoreConfig::Memory);
//! assert!(config.validate().is_ok());
//! ```
pub mod config;
pub mod tools;
pub mod wiring;

pub use config::{
    memory_config, MemoryConfig, MemoryExtractorKind, MemoryStoreConfig, MEMORY_METADATA_KEY,
};
pub use tools::{team_memory_tools, TeamMemoryTool};
pub use wiring::{TeamMemoryWiring, TEAM_MEMORY_FILE};
