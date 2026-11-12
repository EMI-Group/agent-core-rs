//! Case runners: the seam that turns a [`Case`] into a prediction.
//!
//! Python runs the agent through its own pipeline and feeds the resulting dict to
//! the evaluator. This crate splits that in two: a [`CaseRunner`] produces the
//! prediction, then an [`openjiuwen_agent_evolving::evaluator::Evaluator`] scores
//! it. [`AgentCaseRunner`] adapts a runnable agent; [`MapCaseRunner`] replays
//! fixed predictions for offline tests and baselines.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::session::Session;
use openjiuwen_core::single_agent::Agent;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::CaseRunner;
use crate::dataset::Case;

/// Runs a [`Case`] through an [`Agent`], optionally inside a session.
pub struct AgentCaseRunner {
    agent: Arc<dyn Agent>,
    session: Option<Arc<dyn Session>>,
}

impl AgentCaseRunner {
    /// Wraps `agent` with no session.
    #[must_use]
    pub const fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            session: None,
        }
    }

    /// Runs the agent inside `session`.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// The wrapped agent.
    #[must_use]
    pub const fn agent(&self) -> &Arc<dyn Agent> {
        &self.agent
    }

    /// The configured session, when any.
    #[must_use]
    pub fn session(&self) -> Option<&Arc<dyn Session>> {
        self.session.as_ref()
    }
}

#[async_trait]
impl CaseRunner for AgentCaseRunner {
    fn name(&self) -> &'static str {
        "agent"
    }

    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        let inputs = serde_json::to_value(&case.inputs).map_err(|error| {
            Error::validation(format!("case inputs are not serializable: {error}"))
        })?;
        let output = self.agent.invoke(inputs, self.session.clone()).await?;
        serde_json::to_value(&output).map_err(|error| {
            Error::validation(format!("agent output is not serializable: {error}"))
        })
    }
}

/// Replays fixed predictions keyed by [`Case::case_id`].
///
/// Useful for offline tests, deterministic baselines, and feeding pre-computed
/// agent answers back into evaluation.
#[derive(Debug, Clone, Default)]
pub struct MapCaseRunner {
    entries: BTreeMap<String, Value>,
}

impl MapCaseRunner {
    /// Creates a runner from `(case_id, prediction)` pairs.
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = (String, Value)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// Registers or replaces the prediction for `case_id`.
    pub fn insert(&mut self, case_id: impl Into<String>, prediction: Value) {
        self.entries.insert(case_id.into(), prediction);
    }

    /// The registered predictions.
    #[must_use]
    pub const fn entries(&self) -> &BTreeMap<String, Value> {
        &self.entries
    }
}

#[async_trait]
impl CaseRunner for MapCaseRunner {
    fn name(&self) -> &'static str {
        "map"
    }

    async fn run_case(&self, case: &Case, cancel: &CancellationToken) -> Result<Value> {
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        self.entries.get(&case.case_id).cloned().ok_or_else(|| {
            Error::not_found(format!(
                "no prediction registered for case_id {}",
                case.case_id
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_core::single_agent::{AgentCard, AgentChunk, AgentOutput};

    use super::*;

    fn sample_case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String("hi".to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String("hello".to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    #[tokio::test]
    async fn map_runner_returns_registered_predictions() {
        let mut runner = MapCaseRunner::default();
        runner.insert("case-1", serde_json::json!({"answer": "hello"}));
        assert_eq!(runner.name(), "map");
        assert_eq!(runner.entries().len(), 1);

        let value = runner
            .run_case(&sample_case("case-1"), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(value, serde_json::json!({"answer": "hello"}));
    }

    #[tokio::test]
    async fn map_runner_reports_unknown_cases() {
        let runner =
            MapCaseRunner::new([("case-1".to_owned(), serde_json::json!({"answer": "hello"}))]);
        let error = runner
            .run_case(&sample_case("missing"), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::NotFound(message) if message.contains("missing")),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn map_runner_rejects_cancelled_runs() {
        let runner =
            MapCaseRunner::new([("case-1".to_owned(), serde_json::json!({"answer": "hello"}))]);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = runner
            .run_case(&sample_case("case-1"), &cancel)
            .await
            .unwrap_err();
        assert!(error.is_cancelled(), "unexpected error: {error}");
    }

    struct FakeAgent {
        card: AgentCard,
        output: AgentOutput,
        seen_inputs: std::sync::Mutex<Option<Value>>,
    }

    impl FakeAgent {
        fn new(output: AgentOutput) -> Self {
            Self {
                card: AgentCard::new("fake").with_id("fake-agent"),
                output,
                seen_inputs: std::sync::Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            *self.seen_inputs.lock().unwrap() = Some(inputs);
            Ok(self.output.clone())
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            futures::stream::empty().boxed()
        }
    }

    #[tokio::test]
    async fn agent_runner_forwards_inputs_and_serializes_the_output() {
        let agent = Arc::new(FakeAgent::new(AgentOutput::answer("hello")));
        let runner = AgentCaseRunner::new(agent.clone());
        assert_eq!(runner.name(), "agent");
        assert!(runner.session().is_none());
        assert_eq!(runner.agent().card().id, "fake-agent");

        let value = runner
            .run_case(&sample_case("case-1"), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(value["output"], Value::String("hello".to_owned()));
        assert_eq!(value["result_type"], Value::String("answer".to_owned()));

        let seen = agent.seen_inputs.lock().unwrap().clone().unwrap();
        assert_eq!(seen, serde_json::json!({"question": "hi"}));
    }

    #[tokio::test]
    async fn agent_runner_rejects_cancelled_runs() {
        let runner = AgentCaseRunner::new(Arc::new(FakeAgent::new(AgentOutput::answer("hello"))));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = runner
            .run_case(&sample_case("case-1"), &cancel)
            .await
            .unwrap_err();
        assert!(error.is_cancelled(), "unexpected error: {error}");
    }
}
