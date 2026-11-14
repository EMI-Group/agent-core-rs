//! Team-memory wiring: the `openjiuwen-memory` team facade inside a team runtime.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/` (v0.1.20) — `TeamMemoryManager`,
//! `MemberMemoryToolkit` and the `TEAM_MEMORY.md` shared summary. The memory engine itself lives in
//! the [`openjiuwen_memory::team`] node; this node owns the *runtime* half: which members get a
//! toolkit, when the lifecycle hooks run, and where the team summary is exported.
//!
//! The sub-modules are documented in [`CONTEXT.md`](https://example.invalid) (this directory's
//! `CONTEXT.md`) and re-exported flat at [`crate::memory`].
