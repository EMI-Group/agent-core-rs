//! The parameter-update pipeline: update contracts, apply helpers, updaters, RL.
//!
//! Ports Python `openjiuwen.agent_evolving.updater`, `types.py` and
//! `update_execution.py`, plus the RL parameter-update half the crate previously
//! documented as deferred (batching, advantages, gradient application,
//! checkpoint emission).
//!
//! # Layers
//!
//! - **Contracts** ([`UpdateValue`], [`Updates`], [`ApplyResult`]) — what an
//!   update is and how it reports its outcome. [`normalize_update_value`] wraps a
//!   legacy payload into the structured contract and gives `experiences` updates
//!   their append/pending-change shape.
//! - **Apply** ([`UpdateTarget`], [`execute_updates`]) — the two-pass execution
//!   Python performs: non-`None` values in map order, `None`-valued keys last, and
//!   one [`ApplyResult`] per key, failures included.
//! - **Routing** ([`Updater`], [`SingleDimUpdater`], [`MultiDimUpdater`]) — turning
//!   trajectories plus signals (or evaluated cases) into an update map by driving a
//!   [`TextualOptimizer`].
//! - **RL** ([`compute_advantages`], [`GradientOptimizer`], [`RlUpdatePipeline`]) —
//!   the ported half of the PPO loop: group-relative advantages, an ordered
//!   gradient application through a target, and an [`crate::rl::RlCheckpoint`] per
//!   step. [`GpuGradientOptimizer`] is the typed `Unsupported` for the
//!   `torch`/`verl`/`ray` optimiser.
//!
//! Nothing here persists, approves or commits an update: [`execute_updates`] is the
//! last write, and a caller owns whatever store lives behind it.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_agent_evolving::updater::{
//!     execute_updates, summarize_apply_results, RecordingTarget, UpdateTarget, UpdateValue,
//!     Updates,
//! };
//! use serde_json::json;
//!
//! let target = Arc::new(RecordingTarget::new());
//! let mut operators = std::collections::BTreeMap::new();
//! let erased: Arc<dyn UpdateTarget> = target.clone();
//! operators.insert("agent/llm_1".to_owned(), erased);
//!
//! let mut updates = Updates::new();
//! updates.insert_update("agent/llm_1", "description", json!("new prompt"));
//! updates.insert(("agent/llm_1".to_owned(), "body".to_owned()), None);
//!
//! let results = execute_updates(&operators, &updates);
//! let summary = summarize_apply_results(&results);
//! assert_eq!((summary.total, summary.applied, summary.failed), (2, 1, 1));
//! // The `None` value is reported last, never dropped.
//! assert_eq!(results[1].errors, ["update value is None"]);
//! assert_eq!(target.value("agent/llm_1", "description"), Some(json!("new prompt")));
//! assert_eq!(UpdateValue::default().mode.as_str(), "replace");
//! ```

mod apply;
mod rl_update;
mod types;
// The Python package names this module `updater`; the nested path keeps parity.
#[allow(clippy::module_inception)]
mod updater;

pub use apply::{
    apply_updates, execute_updates, summarize_apply_results, OperatorMap, RecordingTarget,
    UpdateTarget,
};
pub use rl_update::{
    compute_advantages, policy_update_key, AdvantageGroupStats, AdvantageReport,
    GpuGradientOptimizer, GradientOptimizer, OfflinePolicyOptimizer, ParameterUpdate,
    RlUpdatePipeline, POLICY_TARGET,
};
pub use types::{
    normalize_update_value, normalize_updates, ApplyResult, ApplySummary, LifecycleStage, Update,
    UpdateDomain, UpdateEffect, UpdateEntry, UpdateKey, UpdateMode, UpdateValue, Updates,
    EXPERIENCES_TARGET, SKILL_EXPERIENCE_ENTRY,
};
pub use updater::{
    domain_for_signal, signals_from_cases, MultiDimUpdater, RecordingOptimizer, SingleDimUpdater,
    TextualOptimizer, UpdateConfig, Updater,
};
