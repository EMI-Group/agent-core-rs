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
//! [`Runner::stop`] *does* stop a running dreaming service (bounded and log-only on
//! failure, so the runner's own shutdown never fails because of dreaming), and it
//! leaves the service installed so a later [`Runner::start_dreaming`] can run it
//! again.
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

use openjiuwen_common::error::Result;

use crate::dreaming::{DreamService, DreamServiceStatus};
use crate::error;
use crate::runner::Runner;

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

    /// Stops the installed dreaming service as part of [`Runner::stop`].
    ///
    /// Best effort by design: the service reports a failed join through its own
    /// status, and anything else is logged here, so the runner's shutdown result
    /// stays the queue's. A runner without a service does nothing.
    pub(crate) async fn stop_dreaming_quietly(&self) {
        let Some(service) = self.dreaming_service() else {
            return;
        };
        match service.stop().await {
            Ok(status) => tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                ticks = status.orchestrator.ticks,
                "the dreaming service stopped with the runner",
            ),
            Err(error) => tracing::warn!(
                target: LOG_TARGET,
                runner_id = %self.runner_id(),
                %error,
                "stopping the dreaming service failed; the runner keeps stopping",
            ),
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
