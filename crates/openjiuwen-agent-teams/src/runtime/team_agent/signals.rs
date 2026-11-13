//! Observability surfaces of the handle: lifecycle signals, the monitor stream and KV affinity.
use std::sync::Arc;

use super::super::observability_bind::TeamSignals;
use super::TeamAgent;
use crate::kv_cache::TeamKvCacheAffinity;
use crate::monitor::{BackpressurePolicy, MonitorReceiver};

impl TeamAgent {
    /// Returns the team lifecycle signals (observability hooks + monitor sink).
    #[must_use]
    pub fn signals(&self) -> TeamSignals {
        self.inner.signals.clone()
    }

    /// Attaches a monitor stream to this team and returns its receiving end.
    ///
    /// Events are emitted best-effort from the lifecycle calls (start/stop, member spawn/stop,
    /// rounds, task transitions); installing a second stream replaces the first emitter.
    #[must_use]
    pub fn monitor_stream(
        &self,
        capacity: usize,
        policy: BackpressurePolicy,
    ) -> Option<MonitorReceiver> {
        self.inner.backend.monitor_stream(capacity, policy)
    }

    /// Returns the KV-cache affinity registry of the run.
    #[must_use]
    pub fn kv_cache(&self) -> Arc<TeamKvCacheAffinity> {
        Arc::clone(&self.inner.kv_cache)
    }

    /// Returns the KV-cache identity env a member session should inherit, if registered.
    #[must_use]
    pub fn kv_env(&self, member_name: &str) -> Option<std::collections::BTreeMap<String, String>> {
        self.inner.kv_cache.env(member_name)
    }
}
