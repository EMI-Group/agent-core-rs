//! Runtime wiring of the sub-agent lifecycle hooks.
//!
//! The runtime owns one [`LifecycleHooks`] registry ([`Shared::lifecycle`]) and awaits it at the
//! four transition points of [`LifecycleStage`]: `Spawn` before the first turn, `Ready` once the
//! worker parks the instance idle (and after a resume), `BeforeEvict` ahead of an explicit
//! eviction, and `Close` after the tombstone is written. Every invocation happens *outside* the
//! state lock: the runtime builds the owned [`SubagentLifecycleEvent`] snapshot under the lock,
//! drops it, then awaits [`run_stage`]. A hook may therefore call back into the runtime (list
//! instances, read transcripts or activity, register another hook) without deadlocking.
//!
//! [`LifecycleHooks::run`] contains hook failures (it logs and reports them), so a failing hook
//! never fails a spawn, a turn, a close or an eviction.

use super::lifecycle::{LifecycleHooks, LifecycleStage, SubagentLifecycleEvent};
use super::{Shared, SubAgentRuntime};

impl SubAgentRuntime {
    /// Copies every hook of `hooks` into this runtime's registry, in registration order.
    ///
    /// The call is additive: hooks already registered keep their position, and a hook with a name
    /// that is already present replaces it in place (see [`LifecycleHooks::register`]). Repeating
    /// the call with the same registry is therefore idempotent, while two calls with different
    /// registries stack their hooks. Call it before sharing the runtime.
    #[must_use]
    pub fn with_lifecycle_hooks(self, hooks: LifecycleHooks) -> Self {
        for hook in hooks.hooks() {
            self.shared.lifecycle.register(hook);
        }
        self
    }

    /// Returns the lifecycle registry this runtime invokes.
    ///
    /// The registry is shared with every clone of the runtime handle, so registering a hook here
    /// takes effect on the next transition; cloning the runtime always yields the same registry.
    #[must_use]
    pub fn lifecycle_hooks(&self) -> &LifecycleHooks {
        &self.shared.lifecycle
    }
}

/// Awaits `stage` on the runtime's hooks for every event, in order.
///
/// The runtime calls this after releasing its state lock; failures are contained by
/// [`LifecycleHooks::run`].
pub(super) async fn run_stage(
    shared: &Shared,
    stage: LifecycleStage,
    events: impl IntoIterator<Item = SubagentLifecycleEvent>,
) {
    for event in events {
        let report = shared.lifecycle.run(stage, &event).await;
        if !report.is_ok() {
            tracing::debug!(
                stage = stage.as_str(),
                subagent_id = %event.subagent_id,
                failures = report.failures.len(),
                "subagent lifecycle hooks reported failures"
            );
        }
    }
}
