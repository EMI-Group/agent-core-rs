//! The replaceable seams of the [`Runner`] facade: the run monitor, the
//! human-in-the-loop hub, and the message queue.
//!
//! Each of the three collaborators is built by [`Runner::with_config`] from the
//! default configuration and already usable ([`Runner::monitor`],
//! [`Runner::hitl`], [`Runner::message_queue`]). A host that owns its own
//! implementation — a monitor rendering into a dashboard, a hub shared with a
//! UI process, a broker-backed queue — installs it with the setters here and
//! gets the previously installed instance back.
//!
//! **Installing is forward-looking.** The setters replace the seam for every run
//! started afterwards; they never rewrite state that already exists. An in-flight
//! run keeps the monitor and hub it was admitted with (the run path snapshots
//! them once, see [`crate::run::RunServices`]), the old monitor keeps its
//! finished-run history and its subscribers, and the old hub keeps its live
//! registrations.
//!
//! This module mirrors `runner_spawn.rs`: an `impl Runner` block in a private
//! module keeps the facade file under the crate's size budget.

use std::sync::Arc;

use crate::hitl::InteractionHub;
use crate::monitor::RunMonitor;
use crate::mq::MessageQueue;
use crate::runner::Runner;

/// `tracing` target every seam-installation log uses.
const LOG_TARGET: &str = "openjiuwen_runner::runner";

impl Runner {
    /// Installs `monitor` for every run started afterwards and returns the
    /// previously installed monitor.
    ///
    /// The installed hub is re-pointed at `monitor` as well
    /// ([`InteractionHub::attach_monitor`]), so the hub's own transitions keep
    /// mirroring into the monitor the run paths report into — the pairing
    /// [`Runner::with_config`] establishes.
    ///
    /// **Not retroactive.** Runs already in flight keep the monitor they were
    /// admitted with: each run path snapshots the seams once, when it builds its
    /// [`crate::run::RunServices`]. The retired monitor keeps its live statuses,
    /// its finished-run retention, and its subscribers, and nothing is migrated
    /// into the new monitor. Every later run — direct or pooled — reports into
    /// the new one.
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use openjiuwen_runner::{RunMonitor, Runner};
    ///
    /// let runner = Runner::new();
    /// let dashboard = Arc::new(RunMonitor::new());
    /// let retired = runner.set_monitor(Arc::clone(&dashboard));
    /// assert!(Arc::ptr_eq(&runner.monitor(), &dashboard));
    /// assert!(!Arc::ptr_eq(&retired, &dashboard));
    /// ```
    pub fn set_monitor(&self, monitor: Arc<RunMonitor>) -> Arc<RunMonitor> {
        let hub = self.hitl();
        let previous = std::mem::replace(&mut *self.monitor.write(), Arc::clone(&monitor));
        let hub_previous = hub.attach_monitor(Some(monitor));
        if hub_previous.is_none() {
            tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                "the installed hub had no monitor; it now mirrors the runner's",
            );
        }
        tracing::debug!(
            target: LOG_TARGET,
            runner_id = %self.runner_id(),
            "run monitor replaced; in-flight runs keep the previous one",
        );
        previous
    }

    /// Installs `hub` as the human-in-the-loop hub every run started afterwards
    /// registers on, and returns the previously installed hub.
    ///
    /// The hub is used **as given**: the setter does not attach the runner's
    /// monitor to it implicitly, so a host that wants the hub's transitions
    /// mirrored passes a hub built with
    /// [`InteractionHub::with_monitor`] or re-points it with
    /// [`InteractionHub::attach_monitor`].
    ///
    /// **Not retroactive.** Live registrations stay on the retired hub, whose
    /// runs keep being addressable through it; they are not migrated. Runs
    /// already in flight likewise keep the hub they were admitted with. The
    /// facade's HITL shortcuts ([`Runner::interact_agent_team`],
    /// [`Runner::answer_interaction`], [`Runner::pause_agent_team`], …) address
    /// the installed hub, so they see the new registrations only.
    pub fn set_hitl(&self, hub: Arc<InteractionHub>) -> Arc<InteractionHub> {
        let previous = std::mem::replace(&mut *self.hitl.write(), hub);
        tracing::debug!(
            target: LOG_TARGET,
            runner_id = %self.runner_id(),
            "HITL hub replaced; live registrations stay on the previous hub",
        );
        previous
    }

    /// Installs a host-owned message queue and returns the previously installed
    /// one.
    ///
    /// The intended use is a broker-backed backend (the seam
    /// [`MessageQueue`] defines) instead of the in-process queue
    /// [`Runner::with_config`] builds from
    /// [`RunnerConfig::message_queue`](crate::config::RunnerConfig).
    ///
    /// **The lifecycle still owns the slot.** [`Runner::start`] rebuilds the
    /// queue from the live configuration, so an installed queue is in force
    /// until the next `start` (or the next `start` after a
    /// [`Runner::set_config`]), which replaces it with the configured backend;
    /// [`Runner::stop`] stops whatever queue is installed at that moment.
    /// The queue is used in whatever started state the host left it in — the
    /// runner does not call [`MessageQueue::start`] on it implicitly, so a host
    /// that wants it serving subscribes and starts it itself.
    pub fn set_message_queue(&self, queue: Arc<dyn MessageQueue>) -> Arc<dyn MessageQueue> {
        let previous = std::mem::replace(&mut *self.queue.write(), queue);
        tracing::debug!(
            target: LOG_TARGET,
            runner_id = %self.runner_id(),
            "message queue replaced; `start` rebuilds it from the configuration",
        );
        previous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mq::InMemoryMessageQueue;

    #[test]
    fn set_monitor_returns_the_previous_monitor_and_retargets_the_hub() {
        let runner = Runner::new();
        let previous = runner.monitor();
        let dashboard = Arc::new(RunMonitor::new());

        let retired = runner.set_monitor(Arc::clone(&dashboard));
        assert!(
            Arc::ptr_eq(&retired, &previous),
            "the setter returns the old one"
        );
        assert!(Arc::ptr_eq(&runner.monitor(), &dashboard));

        let hub_monitor = runner.hitl().monitor().expect("the hub mirrors a monitor");
        assert!(
            Arc::ptr_eq(&hub_monitor, &dashboard),
            "the installed hub follows the runner's monitor"
        );
    }

    #[test]
    fn set_monitor_keeps_the_hub_that_is_installed() {
        let runner = Runner::new();
        let hub = runner.hitl();
        runner.set_monitor(Arc::new(RunMonitor::new()));
        assert!(
            Arc::ptr_eq(&runner.hitl(), &hub),
            "installing a monitor never swaps the hub"
        );
        assert!(hub.monitor().is_some(), "the same hub is re-pointed");
    }

    #[test]
    fn set_hitl_returns_the_previous_hub_without_attaching_a_monitor() {
        let runner = Runner::new();
        let previous = runner.hitl();
        let host_hub = Arc::new(InteractionHub::new());

        let retired = runner.set_hitl(Arc::clone(&host_hub));
        assert!(Arc::ptr_eq(&retired, &previous));
        assert!(Arc::ptr_eq(&runner.hitl(), &host_hub));
        assert!(
            host_hub.monitor().is_none(),
            "the setter does not attach the runner's monitor implicitly"
        );
        assert!(
            previous.monitor().is_some(),
            "the retired hub keeps the monitor it was built with"
        );
    }

    #[test]
    fn set_message_queue_returns_the_previous_queue() {
        let runner = Runner::new();
        let previous = runner.message_queue();
        let broker: Arc<dyn MessageQueue> = InMemoryMessageQueue::with_capacity(8);

        let retired = runner.set_message_queue(Arc::clone(&broker));
        assert!(Arc::ptr_eq(&retired, &previous));
        assert!(Arc::ptr_eq(&runner.message_queue(), &broker));
    }
}
