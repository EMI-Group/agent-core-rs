//! Model pool and per-request request router (Python `openjiuwen.agent_teams.models`).
//!
//! Python parity source: `openjiuwen/agent_teams/models/{pool,allocator}.py`. This node ships the
//! two *data/routing* pieces of that package:
//!
//! * [`ModelPool`] — the team's ordered, validated list of LLM endpoints. Declaration order is the
//!   entry's identity (a persisted member reference is `(model_name, index)`), so a pool is never
//!   re-sorted and duplicate or blank model names are rejected at construction.
//! * [`ModelRouter`] — the per-request choice among those endpoints, with four strategies
//!   ([`RouterStrategy::ByName`], [`RouterStrategy::RoundRobin`], [`RouterStrategy::LeastBusy`],
//!   [`RouterStrategy::RuleBased`]), an ordered [`RoutingRule`] list, an in-flight reservation
//!   [`RouteGuard`] and a [`RouterDecision::fallback_chain`] for retry.
//!
//! # Composition with `runtime::ModelAllocator`
//!
//! Allocation and routing answer different questions and both are needed:
//!
//! * [`crate::runtime::allocator::ModelAllocator`] chooses **one endpoint per member at build time**
//!   and persists it as `MemberRecord::model_ref`. A member's model is therefore stable across a
//!   conversation and survives a restart via the persisted `(model_name, index)` reference.
//! * [`ModelRouter`] chooses **one endpoint per request** among the same pool. Its only state — a
//!   round-robin cursor and one in-flight counter per entry — is process-local, never persisted, and
//!   worthless after a restart by design.
//!
//! Neither replaces the other: the allocator prevents every member from piling onto one endpoint,
//! while the router keeps a single busy member's requests spread (or pinned to the endpoint a rule
//! names) and reports which entry to retry when one fails. A team with an empty pool uses neither.
//!
//! ```
//! use openjiuwen_agent_teams::models::{ModelPool, ModelRouter, RequestHint, RouterStrategy};
//! use openjiuwen_agent_teams::schema::ModelPoolEntry;
//!
//! let entries = vec![ModelPoolEntry::new(
//!     "gpt-4o",
//!     "openai",
//!     "https://api.openai.com/v1",
//!     "sk-a",
//! )];
//! let pool = ModelPool::new(entries).expect("one distinct name");
//! assert_eq!(pool.model_group_label(), "gpt-4o");
//!
//! let router = ModelRouter::new(pool, RouterStrategy::RoundRobin);
//! let decision = router.decide(&RequestHint::new()).expect("pooled");
//! assert_eq!(decision.index, 0);
//! assert!(decision.fallback_chain.is_empty());
//! ```
//!
//! Deferred (see the module `CONTEXT.md`): the `IntelliRouter` client-side reliable router and its
//! `intelli_router` provider, vendor CLI model catalogs, and weighted/LRU/affinity strategies — all
//! of them need a vendor SDK, a network call, or a persisted weight model that this port does not
//! carry.

mod pool;
mod router;

pub use pool::ModelPool;
pub use router::{
    Complexity, ModelRouter, RequestHint, RouteGuard, RouteReason, RouterDecision, RouterStrategy,
    RoutingRule, RuleMatcher,
};
