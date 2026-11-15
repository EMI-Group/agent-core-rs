//! The typed outcome of a runner shutdown: [`StopReport`] and its parts.
//!
//! [`Runner::stop`](crate::Runner::stop) answers one [`Result`] for the whole
//! process-level shutdown, but a shutdown has three independent components — the
//! message queue, the team runtime pool, and the installed dreaming service — and
//! each can fail on its own. This module is that decomposition as a value:
//! [`StopReport`] carries one [`ComponentStop`] per component (always all three,
//! even when a component was skipped), whether the runner had been started, and
//! whether the registries were released.
//!
//! **Nothing is swallowed.** Where the pre-report shutdown logged a pool sweep
//! failure and returned the queue's result, [`Runner::stop`](crate::Runner::stop)
//! now aggregates *every* failed component into one error, and
//! [`Runner::stop_with_report`](crate::Runner::stop_with_report) hands the whole
//! per-component report back to the caller. A [`StopStatus::Skipped`] component is
//! not a silent success either: it says the component had nothing to do (the queue
//! of a runner that never started, the dreaming slot of a runner with no installed
//! [`DreamService`](crate::DreamService)).
//!
//! **The wire shape is the contract.** [`StopReport`] is `snake_case` JSON, every
//! field is optional on input (a missing component defaults to `skipped`), and
//! unknown fields are ignored — a host may persist a shutdown report next to its
//! own telemetry and read it back with a later build.
//!
//! ```no_run
//! use openjiuwen_runner::Runner;
//!
//! # async fn example() {
//! let runner = Runner::new();
//! let report = runner.stop_with_report().await;
//! if !report.is_clean() {
//!     for failure in report.failures() {
//!         eprintln!("{}: {:?}", failure.component, failure.error);
//!     }
//! }
//! # }
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

/// One component of a runner shutdown.
///
/// The three components stop independently and in a fixed order — dreaming, the
/// team pool, then the queue — and each reports its own outcome in a
/// [`StopReport`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopComponent {
    /// The message queue installed at the moment of the stop
    /// ([`MessageQueue`](crate::MessageQueue)).
    #[default]
    Queue,
    /// The team runtime pool ([`TeamPool`](crate::TeamPool)), swept and replaced
    /// with a fresh open pool.
    Pool,
    /// The installed [`DreamService`](crate::DreamService), when one is installed.
    Dreaming,
}

impl StopComponent {
    /// The `snake_case` name used in [`Display`](fmt::Display) and in the JSON
    /// wire shape — the same word a [`StopReport`] and an aggregated shutdown
    /// error use.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queue => "queue",
            Self::Pool => "pool",
            Self::Dreaming => "dreaming",
        }
    }
}

impl fmt::Display for StopComponent {
    /// Writes [`StopComponent::as_str`], so log lines and error messages name a
    /// component the same way the JSON does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How one shutdown component ended.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopStatus {
    /// The component had nothing to do: the queue of a runner that never
    /// started, or the dreaming slot of a runner with no installed service.
    ///
    /// This is the default, so a hand-written or older report that omits a
    /// component claims no work for it rather than a success it cannot prove.
    #[default]
    Skipped,
    /// The component stopped cleanly (or had nothing left to stop but answered
    /// so).
    Stopped,
    /// The component did not stop cleanly; the message is in
    /// [`ComponentStop::error`].
    Failed,
}

impl StopStatus {
    /// The `snake_case` name used in [`Display`](fmt::Display) and in the JSON
    /// wire shape.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Skipped => "skipped",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }

    /// Whether this status is a failure ([`StopStatus::Failed`]).
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Failed)
    }
}

impl fmt::Display for StopStatus {
    /// Writes [`StopStatus::as_str`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One component's outcome: which component, how it ended, and why it failed.
///
/// `error` is `Some` exactly when [`StopStatus::Failed`] is the status; the two
/// fields are kept apart because a host usually branches on the status and only
/// renders the message on failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ComponentStop {
    /// The component this outcome belongs to.
    pub component: StopComponent,
    /// How the component ended.
    pub status: StopStatus,
    /// The failure message, when the component `Failed`; `None` otherwise.
    ///
    /// Serialized as `null` rather than omitted, so every component's JSON object
    /// has the same three fields.
    pub error: Option<String>,
}

impl Default for ComponentStop {
    /// The report default: the queue component, skipped, with no error.
    fn default() -> Self {
        Self::skipped(StopComponent::Queue)
    }
}

impl ComponentStop {
    /// A component with nothing to do ([`StopStatus::Skipped`]).
    #[must_use]
    pub const fn skipped(component: StopComponent) -> Self {
        Self {
            component,
            status: StopStatus::Skipped,
            error: None,
        }
    }

    /// A component that stopped cleanly ([`StopStatus::Stopped`]).
    #[must_use]
    pub const fn stopped(component: StopComponent) -> Self {
        Self {
            component,
            status: StopStatus::Stopped,
            error: None,
        }
    }

    /// A component that did not stop cleanly ([`StopStatus::Failed`]).
    ///
    /// `message` is the reason a caller can act on: the backend's own error
    /// string, or the wording of the bound that expired.
    #[must_use]
    pub fn failed(component: StopComponent, message: impl Into<String>) -> Self {
        Self {
            component,
            status: StopStatus::Failed,
            error: Some(message.into()),
        }
    }

    /// Whether the component failed.
    #[must_use]
    pub const fn is_failure(&self) -> bool {
        self.status.is_failure()
    }

    /// Whether the component did not fail — a [`StopStatus::Stopped`] or
    /// [`StopStatus::Skipped`] outcome.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        !self.is_failure()
    }
}

/// The per-component outcome of one [`Runner::stop_with_report`](crate::Runner::stop_with_report).
///
/// `started` is the started flag the runner held when the stop began — `false`
/// means the queue had already been stopped (or never started), so its component
/// is [`StopStatus::Skipped`] and the registries were left alone.
/// `registries_released` records that the checkpointer was dropped and every
/// resource-registry entry released during this stop.
///
/// JSON is `snake_case`, tolerant of unknown fields, and every field is optional
/// on input.
///
/// ```no_run
/// use openjiuwen_runner::Runner;
///
/// # async fn example() {
/// let runner = Runner::new();
/// runner.start().await.expect("start");
/// let report = runner.stop_with_report().await;
/// assert!(report.started);
/// assert!(report.is_clean());
/// # }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StopReport {
    /// Whether the runner was started when the stop began; `false` for a runner
    /// that was already stopped.
    pub started: bool,
    /// The message-queue outcome.
    pub queue: ComponentStop,
    /// The team-runtime-pool outcome.
    pub pool: ComponentStop,
    /// The dreaming-service outcome; [`StopStatus::Skipped`] when no service is
    /// installed.
    pub dreaming: ComponentStop,
    /// Whether this stop dropped the checkpointer and released the registries.
    pub registries_released: bool,
}

impl StopReport {
    /// A report for a stop that began with the runner `started` and has not
    /// stopped anything yet: every component is `skipped` and no registry was
    /// released.
    ///
    /// The facade fills the components in as it stops them; a caller that builds
    /// an expectation starts here.
    #[must_use]
    pub fn new(started: bool) -> Self {
        Self {
            started,
            queue: ComponentStop::skipped(StopComponent::Queue),
            pool: ComponentStop::skipped(StopComponent::Pool),
            dreaming: ComponentStop::skipped(StopComponent::Dreaming),
            registries_released: false,
        }
    }

    /// Replaces the queue outcome.
    #[must_use]
    pub fn with_queue(mut self, queue: ComponentStop) -> Self {
        self.queue = queue;
        self
    }

    /// Replaces the pool outcome.
    #[must_use]
    pub fn with_pool(mut self, pool: ComponentStop) -> Self {
        self.pool = pool;
        self
    }

    /// Replaces the dreaming outcome.
    #[must_use]
    pub fn with_dreaming(mut self, dreaming: ComponentStop) -> Self {
        self.dreaming = dreaming;
        self
    }

    /// Records whether the registries were released.
    #[must_use]
    pub fn with_registries_released(mut self, released: bool) -> Self {
        self.registries_released = released;
        self
    }

    /// Whether every component stopped without a failure.
    ///
    /// A [`StopStatus::Skipped`] component is clean: it had nothing to do.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.components().iter().all(|stop| stop.is_clean())
    }

    /// Every component's outcome in shutdown order: queue, pool, dreaming.
    #[must_use]
    pub fn components(&self) -> [&ComponentStop; 3] {
        [&self.queue, &self.pool, &self.dreaming]
    }

    /// The failed components, in shutdown order; empty for a clean stop.
    #[must_use]
    pub fn failures(&self) -> Vec<&ComponentStop> {
        self.components()
            .into_iter()
            .filter(|stop| stop.is_failure())
            .collect()
    }

    /// One line naming every failed component and its message, or `None` when
    /// the stop was clean.
    ///
    /// This is what [`Runner::stop`](crate::Runner::stop) puts into its
    /// aggregated error, so the wording a caller sees and the report it can ask
    /// for agree.
    #[must_use]
    pub fn failure_summary(&self) -> Option<String> {
        let failures = self.failures();
        if failures.is_empty() {
            return None;
        }
        let details: Vec<String> = failures
            .iter()
            .map(|stop| {
                format!(
                    "{}: {}",
                    stop.component,
                    stop.error.as_deref().unwrap_or("failed without a reason")
                )
            })
            .collect();
        Some(details.join("; "))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! Unit suite for the report value: defaults, the failure projection, the
    //! aggregated summary, and the JSON wire shape.
    use super::*;

    #[test]
    fn a_fresh_report_skips_every_component_and_is_clean() {
        let report = StopReport::new(true);
        assert!(report.started);
        assert!(!report.registries_released);
        assert!(!report.queue.is_failure());
        assert_eq!(report.queue.component, StopComponent::Queue);
        assert_eq!(report.pool.component, StopComponent::Pool);
        assert_eq!(report.dreaming.component, StopComponent::Dreaming);
        for stop in report.components() {
            assert_eq!(stop.status, StopStatus::Skipped);
            assert!(stop.error.is_none());
        }
        assert!(report.is_clean());
        assert_eq!(report.failures().len(), 0);
        assert!(report.failure_summary().is_none());
    }

    #[test]
    fn constructors_pin_the_status_and_the_message() {
        let skipped = ComponentStop::skipped(StopComponent::Dreaming);
        assert_eq!(skipped.status, StopStatus::Skipped);
        assert!(skipped.error.is_none());

        let stopped = ComponentStop::stopped(StopComponent::Pool);
        assert_eq!(stopped.status, StopStatus::Stopped);
        assert!(stopped.error.is_none());
        assert!(stopped.is_clean());

        let failed = ComponentStop::failed(StopComponent::Queue, "boom");
        assert_eq!(failed.status, StopStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("boom"));
        assert!(failed.is_failure());
        assert!(!failed.is_clean());
    }

    #[test]
    fn failures_are_projected_in_shutdown_order_and_summarised() {
        let report = StopReport::new(true)
            .with_queue(ComponentStop::stopped(StopComponent::Queue))
            .with_pool(ComponentStop::failed(StopComponent::Pool, "pool boom"))
            .with_dreaming(ComponentStop::failed(StopComponent::Dreaming, "dream boom"));
        assert!(!report.is_clean());
        let failures = report.failures();
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].component, StopComponent::Pool);
        assert_eq!(failures[1].component, StopComponent::Dreaming);
        assert_eq!(
            report.failure_summary().as_deref(),
            Some("pool: pool boom; dreaming: dream boom")
        );
    }

    #[test]
    fn a_failure_without_a_message_still_summarises() {
        let mut report = StopReport::new(false);
        report.queue = ComponentStop {
            component: StopComponent::Queue,
            status: StopStatus::Failed,
            error: None,
        };
        assert_eq!(
            report.failure_summary().as_deref(),
            Some("queue: failed without a reason")
        );
    }

    #[test]
    fn the_wire_shape_is_snake_case_and_tolerates_unknown_fields() {
        let report = StopReport::new(true)
            .with_queue(ComponentStop::stopped(StopComponent::Queue))
            .with_registries_released(true);
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["started"], serde_json::json!(true));
        assert_eq!(json["registries_released"], serde_json::json!(true));
        assert_eq!(
            json["queue"],
            serde_json::json!({"component": "queue", "status": "stopped", "error": null})
        );
        assert_eq!(json["dreaming"]["status"], serde_json::json!("skipped"));

        let back: StopReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, report);

        // An unknown field is tolerated, and a missing component falls back to
        // `skipped` rather than to a success the input never claimed.
        let partial: StopReport = serde_json::from_value(serde_json::json!({
            "started": true,
            "queue": {"component": "queue", "status": "failed", "error": "boom"},
            "future_component": {"status": "failed"},
        }))
        .expect("an unknown field is tolerated");
        assert!(partial.started);
        assert!(partial.queue.is_failure());
        assert_eq!(partial.pool.status, StopStatus::Skipped);
        assert_eq!(partial.dreaming.status, StopStatus::Skipped);
        assert!(!partial.registries_released);
        assert!(!partial.is_clean());
    }

    #[test]
    fn component_and_status_names_are_the_wire_words() {
        for (component, name) in [
            (StopComponent::Queue, "queue"),
            (StopComponent::Pool, "pool"),
            (StopComponent::Dreaming, "dreaming"),
        ] {
            assert_eq!(component.as_str(), name);
            assert_eq!(component.to_string(), name);
            assert_eq!(
                serde_json::to_value(component).expect("serialize"),
                serde_json::json!(name)
            );
        }
        for (status, name) in [
            (StopStatus::Skipped, "skipped"),
            (StopStatus::Stopped, "stopped"),
            (StopStatus::Failed, "failed"),
        ] {
            assert_eq!(status.as_str(), name);
            assert_eq!(status.to_string(), name);
            assert_eq!(
                serde_json::to_value(status).expect("serialize"),
                serde_json::json!(name)
            );
        }
        assert_eq!(StopComponent::default(), StopComponent::Queue);
        assert_eq!(StopStatus::default(), StopStatus::Skipped);
        assert_eq!(
            ComponentStop::default(),
            ComponentStop::skipped(StopComponent::Queue)
        );
    }
}
