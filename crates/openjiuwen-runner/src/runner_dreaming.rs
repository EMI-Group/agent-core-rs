//! The dreaming half of the [`Runner`] facade: install, start, stop, and observe
//! a [`DreamService`].
//!
//! Dreaming is **installed, not constructed**: a host builds the
//! [`DreamService`] over its own
//! [`DreamOrchestrator`](openjiuwen_memory::dreaming::DreamOrchestrator) — the
//! repository, the consolidator, the schedule, and any sweep lock belong to the
//! application — and hands it to [`Runner::set_dreaming_service`]. Nothing in
//! [`crate::config::RunnerConfig`] describes it, because a repository provider and
//! a consolidator are runtime capabilities rather than serializable settings.
//!
//! **A lifecycle of its own.** [`Runner::start`] does not start dreaming and
//! [`Runner::start_dreaming`] does not start the runner: the two are independent,
//! so a host can run consolidation without a message queue and vice versa.
//! [`Runner::stop`] *does* stop a running dreaming service (bounded by the
//! configured shutdown grace period) and reports the verdict as one
//! [`ComponentStop`] of the shutdown report — an unjoinable loop is a reported
//! failure, not a log line — and it leaves the service installed so a later
//! [`Runner::start_dreaming`] can run it again.
//!
//! **Typed absence.** Without an installed service the three lifecycle/status
//! entry points return [`Error::Config`](openjiuwen_common::error::Error) naming
//! [`Runner::set_dreaming_service`]; none of them silently does nothing and none of
//! them claims the capability is unsupported — the runner simply has no
//! orchestrator to drive until the host installs one.
//!
//! This module mirrors `runner_seams.rs` and `runner_spawn.rs`: an `impl Runner`
//! block in a private module keeps the facade file under the crate's size budget.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::Result;

use crate::dreaming::{DreamService, DreamServiceStatus, DEFAULT_DREAMING_STOP_TIMEOUT};
use crate::error;
use crate::runner::Runner;
use crate::stop::{ComponentStop, StopComponent};

/// `tracing` target every dreaming facade log uses.
const LOG_TARGET: &str = "openjiuwen_runner::runner";

impl Runner {
    /// Installs `service` (or clears the slot with `None`) and returns the
    /// previously installed service.
    ///
    /// Installing replaces the slot; it is **not** retroactive for a running
    /// service. A replaced service keeps its own loop task until somebody calls
    /// its [`DreamService::stop`], and [`Runner::stop`] only stops the service
    /// installed at that moment — so a host that replaces a running service stops
    /// the retired one itself (a `Runner` never owns two drivers).
    pub fn set_dreaming_service(
        &self,
        service: Option<Arc<DreamService>>,
    ) -> Option<Arc<DreamService>> {
        if let Some(retired) = self.dreaming.read().as_ref() {
            if retired.is_running() {
                tracing::warn!(
                    target: LOG_TARGET,
                    runner_id = %self.runner_id(),
                    "replacing a running dreaming service; its loop keeps running until its own stop"
                );
            }
        }
        let installed = service.is_some();
        let previous = std::mem::replace(&mut *self.dreaming.write(), service);
        tracing::debug!(
            target: LOG_TARGET,
            runner_id = %self.runner_id(),
            installed,
            "dreaming service slot updated",
        );
        previous
    }

    /// The installed dreaming service, when one is installed.
    #[must_use]
    pub fn dreaming_service(&self) -> Option<Arc<DreamService>> {
        self.dreaming.read().clone()
    }

    /// Starts the installed dreaming service's background loop.
    ///
    /// Idempotent in the same way [`DreamService::start`] is: a second call while
    /// the loop is live returns the current status and spawns nothing. Starting
    /// does not require [`Runner::start`].
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when no dreaming service is
    /// installed, naming [`Runner::set_dreaming_service`].
    pub async fn start_dreaming(&self) -> Result<DreamServiceStatus> {
        self.require_dreaming_service()?.start().await
    }

    /// Stops the installed dreaming service's background loop.
    ///
    /// Idempotent: stopping a service that is not running returns its status.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when no dreaming service is
    /// installed, naming [`Runner::set_dreaming_service`].
    pub async fn stop_dreaming(&self) -> Result<DreamServiceStatus> {
        self.require_dreaming_service()?.stop().await
    }

    /// A snapshot of the installed dreaming service and its orchestrator.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when no dreaming service is
    /// installed, naming [`Runner::set_dreaming_service`].
    pub fn dreaming_status(&self) -> Result<DreamServiceStatus> {
        Ok(self.require_dreaming_service()?.status())
    }

    /// Stops the installed dreaming service as part of [`Runner::stop`] and
    /// reports the outcome as one [`ComponentStop`].
    ///
    /// A runner without a service has nothing to stop, so the component is
    /// [`StopStatus::Skipped`] — the typed absence, not a silent success. With a
    /// service, the loop is joined with the runner's shutdown grace period
    /// ([`crate::RunnerConfig::shutdown_timeout`], capped at
    /// [`DEFAULT_DREAMING_STOP_TIMEOUT`]) and the verdict is reported:
    /// [`StopStatus::Failed`] when the status carries a
    /// [`DreamServiceStatus::last_stop_error`] (a loop that ignored its
    /// cancellation and was aborted, or a task that failed), otherwise
    /// [`StopStatus::Stopped`]. Nothing here is log-only: the runner's own stop
    /// result aggregates this component.
    pub(crate) async fn stop_dreaming_component(&self) -> ComponentStop {
        let Some(service) = self.dreaming_service() else {
            tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                "no dreaming service is installed; the dreaming component is skipped",
            );
            return ComponentStop::skipped(StopComponent::Dreaming);
        };
        let bound = self.dreaming_stop_timeout();
        let status = match service.stop_with_timeout(bound).await {
            Ok(status) => status,
            Err(error) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    runner_id = %self.runner_id(),
                    %error,
                    "stopping the dreaming service failed",
                );
                return ComponentStop::failed(StopComponent::Dreaming, error.to_string());
            }
        };
        if let Some(message) = status.last_stop_error {
            tracing::warn!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                reason = %message,
                "the dreaming service did not shut down cleanly",
            );
            ComponentStop::failed(StopComponent::Dreaming, message)
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                ticks = status.orchestrator.ticks,
                "the dreaming service stopped with the runner",
            );
            ComponentStop::stopped(StopComponent::Dreaming)
        }
    }

    /// The join bound [`Runner::stop`] gives the dreaming loop: the configured
    /// shutdown grace period, capped at [`DEFAULT_DREAMING_STOP_TIMEOUT`].
    ///
    /// One configured number bounds the whole shutdown, so a host that sets a
    /// short grace period for the queue gets the same bound for dreaming. A zero
    /// grace period is read as "no configured bound", which for a dreaming loop
    /// means [`DEFAULT_DREAMING_STOP_TIMEOUT`] rather than an instant abort.
    fn dreaming_stop_timeout(&self) -> Duration {
        let configured = self.config().shutdown_timeout();
        if configured.is_zero() {
            DEFAULT_DREAMING_STOP_TIMEOUT
        } else {
            configured.min(DEFAULT_DREAMING_STOP_TIMEOUT)
        }
    }

    /// The installed service, or the typed `Error::Config` naming the fix.
    fn require_dreaming_service(&self) -> Result<Arc<DreamService>> {
        self.dreaming_service()
            .ok_or_else(error::no_dreaming_service)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! The facade's typed-absence contract, proven without a running loop.
    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::config::RunnerConfig;

    fn runner() -> Arc<Runner> {
        Runner::with_config(RunnerConfig::default())
    }

    #[test]
    fn a_runner_without_a_service_answers_config_errors() {
        let runner = runner();
        assert!(runner.dreaming_service().is_none());
        let error = runner.dreaming_status().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error.to_string().contains("set_dreaming_service"),
            "`{error}` should name the fix"
        );
    }

    #[tokio::test]
    async fn the_lifecycle_calls_need_an_installed_service() {
        let runner = runner();
        for error in [
            runner.start_dreaming().await.unwrap_err(),
            runner.stop_dreaming().await.unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::Config);
            assert!(
                error.to_string().contains("set_dreaming_service"),
                "`{error}` should name the fix"
            );
        }
    }

    #[test]
    fn setting_and_clearing_the_slot_returns_the_previous_service() {
        use openjiuwen_memory::dreaming::{
            DreamConfig, DreamConsolidator, DreamOrchestratorConfig, NoopConsolidator,
        };

        let runner = runner();
        assert!(runner.set_dreaming_service(None).is_none());
        assert!(runner.dreaming_service().is_none());

        let consolidator: Arc<dyn DreamConsolidator> = Arc::new(NoopConsolidator);
        let service = DreamService::with_in_memory(
            consolidator,
            DreamConfig::default(),
            DreamOrchestratorConfig::default(),
        )
        .unwrap();
        let replaced = runner.set_dreaming_service(Some(Arc::clone(&service)));
        assert!(replaced.is_none(), "the slot was empty");
        assert!(runner
            .dreaming_service()
            .is_some_and(|installed| Arc::ptr_eq(&installed, &service)));

        let previous = runner.set_dreaming_service(None).expect("installed");
        assert!(Arc::ptr_eq(&previous, &service));
        assert!(runner.dreaming_service().is_none());
    }
}
