//! The runtime-only request a host hands to the pool.
//!
//! A [`PoolRequest`] is what a host wants to run (a [`TeamRef`], its inputs, a
//! session, and [`RunOptions`]); the pool derives the entry's key from it and
//! writes the key's session id back into the options the run actually sees, so
//! the pooled entry and the run can never disagree about the session.

use std::fmt;
use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::{RunnerConfig, TeamPoolConfig};
use crate::run::{resolve_session_id, RunOptions, TeamRef};

use super::key::PoolKey;

/// What a host asks the pool to run.
///
/// Not serde-serializable on purpose: it carries a live [`TeamRef`] (which may
/// hold an `Arc<dyn Team>`) and [`RunOptions`] (which carries sessions and a
/// cancellation token).
///
/// **Session precedence.** [`PoolRequest::key`] derives the entry's session id
/// through `crate::run::resolve_session_id`:
/// `inputs["conversation_id"]` first, then this request's
/// [`PoolRequest::session_id`], then [`RunOptions::session_id`], then
/// `RunnerConfig::default_session_id`. The derived id always wins, and it is
/// written back into the [`RunOptions`] handed to the run, so the run and the
/// pool entry can never disagree.
#[derive(Clone)]
pub struct PoolRequest {
    /// The team to run: a registry id or a live instance.
    pub team: TeamRef,
    /// The inputs the team's strategy receives.
    pub inputs: Value,
    /// The session id this request asks for, when it asks for one.
    pub session_id: Option<String>,
    /// The label reported on the monitor and the snapshot; defaults to the team
    /// name.
    pub label: Option<String>,
    /// Per-run options (cancellation, deadline, session overrides, metadata).
    pub options: RunOptions,
}

impl PoolRequest {
    /// Builds a request for `team` with `inputs` and default options.
    #[must_use]
    pub fn new(team: impl Into<TeamRef>, inputs: Value) -> Self {
        Self {
            team: team.into(),
            inputs,
            session_id: None,
            label: None,
            options: RunOptions::default(),
        }
    }

    /// Requests `session_id` for the run.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the label reported on the monitor and the snapshot.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Replaces the per-run options.
    #[must_use]
    pub fn with_options(mut self, options: RunOptions) -> Self {
        self.options = options;
        self
    }

    /// Cancels the run when `cancellation` is triggered.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.options.cancellation = cancellation;
        self
    }

    /// Bounds the run with `timeout` (overriding the pool's `run_timeout`).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = Some(timeout);
        self
    }

    /// Derives the key this request is pooled under.
    ///
    /// The team name comes from [`PoolKey::for_team`] (an id reference verbatim,
    /// a live team's card id); the session id comes from
    /// `crate::run::resolve_session_id`.
    #[must_use]
    pub fn key(&self, config: &RunnerConfig) -> PoolKey {
        let session_id = resolve_session_id(&self.inputs, &self.options_for_key(), config);
        PoolKey::for_team(&self.team, session_id)
    }

    /// The label this request reports under `team_name`.
    #[must_use]
    pub fn label_or(&self, team_name: &str) -> String {
        self.label.clone().unwrap_or_else(|| team_name.to_owned())
    }

    /// The deadline this request runs under, given the pool configuration.
    #[must_use]
    pub fn deadline(&self, config: &TeamPoolConfig) -> Option<Duration> {
        self.options.timeout.or_else(|| config.run_timeout())
    }

    /// The effective options handed to the run, with the key's session id.
    #[must_use]
    pub fn effective_options(&self, session_id: &str) -> RunOptions {
        let mut options = self.options.clone();
        options.session_id = Some(session_id.to_owned());
        options
    }

    /// The options the key derivation sees: this request's session id, when it
    /// names one, takes precedence over the options' own.
    fn options_for_key(&self) -> RunOptions {
        let mut options = self.options.clone();
        if self.session_id.is_some() {
            options.session_id.clone_from(&self.session_id);
        }
        options
    }
}

impl fmt::Debug for PoolRequest {
    /// Reports the reference, inputs, labels, and options.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolRequest")
            .field("team", &self.team)
            .field("inputs", &self.inputs)
            .field("session_id", &self.session_id)
            .field("label", &self.label)
            .field("options", &self.options)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(session: Option<&str>) -> PoolRequest {
        let mut request = PoolRequest::new("crew", json!({"query": "hi"}));
        if let Some(session) = session {
            request = request.with_session_id(session);
        }
        request
    }

    #[test]
    fn a_request_defaults_to_empty_options() {
        let request = PoolRequest::new("crew", json!({"q": 1}));
        assert!(matches!(request.team, TeamRef::Id(ref id) if id == "crew"));
        assert_eq!(request.inputs, json!({"q": 1}));
        assert!(request.session_id.is_none());
        assert!(request.label.is_none());
        assert!(request.options.session_id.is_none());
        let debug = format!("{request:?}");
        assert!(debug.contains("PoolRequest"), "{debug}");
        assert!(debug.contains("crew"), "{debug}");
    }

    #[test]
    fn builders_set_the_label_session_cancellation_and_timeout() {
        let token = CancellationToken::new();
        let request = PoolRequest::new("crew", json!({}))
            .with_session_id("s-9")
            .with_label("nightly")
            .with_options(RunOptions::default().with_session_id("s-other"))
            .with_cancellation(token.clone())
            .with_timeout(Duration::from_millis(250));
        assert_eq!(request.session_id.as_deref(), Some("s-9"));
        assert_eq!(request.label.as_deref(), Some("nightly"));
        assert_eq!(request.options.timeout, Some(Duration::from_millis(250)));
        token.cancel();
        assert!(request.options.cancellation.is_cancelled());
        // The explicit request session id wins over the options'.
        let config = RunnerConfig::default();
        assert_eq!(request.key(&config).session_id, "s-9");
    }

    #[test]
    fn the_key_prefers_the_conversation_then_the_request_then_the_options() {
        let config = RunnerConfig::default().with_default_session_id("runner-default");
        assert_eq!(
            request(None).key(&config),
            PoolKey::new("crew", "runner-default")
        );
        assert_eq!(
            request(Some("s-9")).key(&config),
            PoolKey::new("crew", "s-9")
        );

        let options_request =
            request(None).with_options(RunOptions::default().with_session_id("s-opt"));
        assert_eq!(options_request.key(&config), PoolKey::new("crew", "s-opt"));

        let mut with_conversation = request(Some("s-9"));
        with_conversation.inputs = json!({"conversation_id": "conv-1"});
        assert_eq!(
            with_conversation.key(&config),
            PoolKey::new("crew", "conv-1")
        );
    }

    #[test]
    fn a_label_falls_back_to_the_team_name_and_a_deadline_to_the_pool() {
        assert_eq!(request(None).label_or("crew"), "crew");
        assert_eq!(request(None).with_label("n").label_or("crew"), "n");

        let pool = TeamPoolConfig::default();
        assert!(request(None).deadline(&pool).is_none());
        let pool = pool.with_run_timeout_secs(Some(2.0));
        assert_eq!(request(None).deadline(&pool), Some(Duration::from_secs(2)));
        assert_eq!(
            request(None)
                .with_timeout(Duration::from_millis(5))
                .deadline(&pool),
            Some(Duration::from_millis(5))
        );
    }

    #[test]
    fn the_effective_options_carry_the_key_session() {
        let options = request(Some("s-1")).effective_options("s-2");
        assert_eq!(options.session_id.as_deref(), Some("s-2"));
    }
}
