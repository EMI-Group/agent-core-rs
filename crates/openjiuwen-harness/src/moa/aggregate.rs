//! Labelled or model-synthesised aggregation of the reference proposals.
//!
//! Ports the integration half of `openjiuwen/harness/moa/consultation.py`: once the
//! propositions have been collected, one answer (or next step) has to be produced from
//! them. Two strategies are offered:
//!
//! - [`AggregationStrategy::Labelled`] (the default) is deterministic and fully offline. It
//!   concatenates the proposals under stable headings — the
//!   `## Integrated answer (mixture of agents)` header followed by one
//!   `### Proposal {index}` section per source, in configured order, separated by blank
//!   lines — so a caller, a transcript or a test can read the exact text that was
//!   produced. An empty set yields the header plus a clear marker instead of an empty
//!   string, so a caller never has to distinguish "no advice" from "failed to render".
//! - [`AggregationStrategy::Synthesis`] asks the injected aggregator model to reconcile
//!   the proposals into one final answer, using [`crate::moa::prompts::aggregation_messages`].
//!
//! The module never opens a network connection: the aggregator model is a caller-injected
//! `Arc<dyn Model>`, and a synthesis without one is a configuration error rather than a
//! silent fallback to the labelled output.
// The public names (`AggregationStrategy`, `Aggregator`, ...) mirror the Python contract
// and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::ModelRequestConfig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::config::MoaConfig;
use super::prompts::aggregation_messages;
use super::state::ProposalSource;

/// Heading that opens the deterministic aggregation output.
const LABELLED_HEADER: &str = "## Integrated answer (mixture of agents)";

/// Body of the deterministic aggregation when no proposal was available.
const EMPTY_MARKER: &str = "_No proposals were available._";

/// How the retained proposals are turned into one answer.
///
/// Serializes as `snake_case`, so a persisted configuration or a JSON diagnostic carries the
/// same spellings the Python SDK uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AggregationStrategy {
    /// Deterministic offline concatenation of the proposals under stable headings.
    #[default]
    Labelled,
    /// One final answer produced by the injected aggregator model.
    Synthesis,
}

impl AggregationStrategy {
    /// The wire spelling of the strategy (`"labelled"`, `"synthesis"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Labelled => "labelled",
            Self::Synthesis => "synthesis",
        }
    }

    /// Parses an exact `snake_case` strategy name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "labelled" => Some(Self::Labelled),
            "synthesis" => Some(Self::Synthesis),
            _ => None,
        }
    }
}

/// Turns retained proposals into one answer under a fixed strategy.
///
/// The type holds no model handle: [`AggregationStrategy::Synthesis`] takes the model from
/// the [`MoaConfig`] it is called with, exactly like the rest of the module.
#[derive(Debug, Clone, Copy)]
pub struct Aggregator {
    /// Strategy applied by [`Aggregator::aggregate`].
    strategy: AggregationStrategy,
}

impl Aggregator {
    /// Builds an aggregator for `strategy`.
    #[must_use]
    pub const fn new(strategy: AggregationStrategy) -> Self {
        Self { strategy }
    }

    /// The strategy this aggregator applies.
    #[must_use]
    pub const fn strategy(&self) -> AggregationStrategy {
        self.strategy
    }

    /// Renders the proposals under stable headings, without calling any model.
    ///
    /// The output is the `## Integrated answer (mixture of agents)` header, a blank line,
    /// then one `### Proposal {index}` block per source in slice order, blocks separated
    /// by blank lines. Proposal bodies are trimmed so a trailing newline in a model
    /// response cannot collapse the spacing. An empty slice yields the header plus
    /// `_No proposals were available._`; the function never panics.
    #[must_use]
    pub fn labelled(sources: &[ProposalSource]) -> String {
        let mut rendered = String::from(LABELLED_HEADER);
        rendered.push_str("\n\n");
        if sources.is_empty() {
            rendered.push_str(EMPTY_MARKER);
            return rendered;
        }
        let blocks: Vec<String> = sources
            .iter()
            .map(|source| {
                format!(
                    "### Proposal {index}\n\n{body}",
                    index = source.index,
                    body = source.text.trim()
                )
            })
            .collect();
        rendered.push_str(&blocks.join("\n\n"));
        rendered
    }

    /// Produces the integrated answer for `problem` from `sources`.
    ///
    /// [`AggregationStrategy::Labelled`] returns [`Aggregator::labelled`] directly and
    /// never observes `cancel`. [`AggregationStrategy::Synthesis`] sends
    /// [`aggregation_messages`] to `config.aggregator_model` with the configured
    /// temperature, bounded by `config.reference_timeout`, and returns the trimmed
    /// assistant text. There is no retry: a provider failure is reported once.
    ///
    /// # Errors
    ///
    /// - [`Error::Config`] when the synthesis strategy has no `aggregator_model`.
    /// - [`Error::Cancelled`] when `cancel` is already or becomes cancelled.
    /// - [`Error::Timeout`] when the aggregator does not answer within
    ///   `config.reference_timeout`.
    /// - [`Error::Llm`] when the provider fails or the answer is empty.
    pub async fn aggregate(
        &self,
        config: &MoaConfig,
        problem: &str,
        sources: &[ProposalSource],
        cancel: &CancellationToken,
    ) -> Result<String> {
        if self.strategy == AggregationStrategy::Labelled {
            return Ok(Self::labelled(sources));
        }

        let model = config
            .aggregator_model
            .as_ref()
            .ok_or_else(|| Error::config("MoA synthesis requires an aggregator model"))?;
        let messages = aggregation_messages(problem, sources);
        let request = ModelRequestConfig::new(model.model_name())
            .with_temperature(config.aggregator_temperature.unwrap_or(0.0));

        let response = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(Error::cancelled()),
            result = tokio::time::timeout(
                config.reference_timeout,
                model.invoke(&messages, &request),
            ) => match result {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => return Err(aggregation_failure(&error)),
                Err(_) => return Err(Error::timeout("MoA aggregation timed out")),
            },
        };

        let answer = response
            .message()
            .map(|message| message.text().trim().to_owned())
            .unwrap_or_default();
        if answer.is_empty() {
            return Err(Error::llm("MoA aggregator returned an empty answer"));
        }
        Ok(answer)
    }
}

/// Maps a provider failure onto the aggregation contract.
///
/// Cancellation keeps its kind so a caller can distinguish it from a provider error; every
/// other failure becomes [`Error::Llm`] carrying the provider message.
fn aggregation_failure(error: &Error) -> Error {
    if error.is_cancelled() {
        Error::cancelled()
    } else {
        Error::llm(format!("MoA aggregation failed: {}", error.message()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::Model;
    use openjiuwen_test_support::ScriptedModel;

    use super::*;

    fn source(index: usize, text: &str) -> ProposalSource {
        ProposalSource::new(index, text)
    }

    fn sources() -> Vec<ProposalSource> {
        vec![source(1, "first proposal"), source(2, "second proposal")]
    }

    fn synthesis_config(answer: &str) -> MoaConfig {
        MoaConfig::new(vec![Arc::new(ScriptedModel::text("p")) as Arc<dyn Model>])
            .with_aggregator_model(Arc::new(ScriptedModel::text(answer)))
    }

    #[test]
    fn labelled_lists_proposals_in_order() {
        let rendered = Aggregator::labelled(&sources());
        assert!(rendered.starts_with("## Integrated answer (mixture of agents)"));
        let first = rendered.find("### Proposal 1").unwrap();
        let second = rendered.find("### Proposal 2").unwrap();
        assert!(first < second, "proposals keep their configured order");
        assert!(rendered.contains("first proposal"));
        assert!(rendered.contains("second proposal"));
        assert!(rendered.contains("\n\n### Proposal 2\n\nsecond proposal"));
    }

    #[test]
    fn labelled_without_sources_reports_the_marker() {
        let rendered = Aggregator::labelled(&[]);
        assert_eq!(
            rendered,
            "## Integrated answer (mixture of agents)\n\n_No proposals were available._"
        );
        assert!(!rendered.contains("### Proposal"));
    }

    #[tokio::test]
    async fn synthesis_returns_the_aggregator_text() {
        let config = synthesis_config("final answer");
        let aggregator = Aggregator::new(AggregationStrategy::Synthesis);
        let answer = aggregator
            .aggregate(
                &config,
                "fix the build",
                &sources(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(answer, "final answer");
    }

    #[tokio::test]
    async fn synthesis_without_model_is_a_config_error() {
        let config = MoaConfig::new(vec![Arc::new(ScriptedModel::text("p")) as Arc<dyn Model>]);
        let aggregator = Aggregator::new(AggregationStrategy::Synthesis);
        let error = aggregator
            .aggregate(
                &config,
                "fix the build",
                &sources(),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "MoA synthesis requires an aggregator model"
        );
    }

    #[tokio::test]
    async fn pre_cancelled_token_aborts_synthesis() {
        let config = synthesis_config("unused");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let aggregator = Aggregator::new(AggregationStrategy::Synthesis);
        let error = aggregator
            .aggregate(&config, "fix the build", &sources(), &cancel)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
    }

    #[test]
    fn strategy_names_round_trip() {
        for strategy in [
            AggregationStrategy::Labelled,
            AggregationStrategy::Synthesis,
        ] {
            assert_eq!(
                AggregationStrategy::parse(strategy.as_str()),
                Some(strategy)
            );
        }
        assert_eq!(
            AggregationStrategy::parse("labelled").unwrap().as_str(),
            "labelled"
        );
        assert_eq!(AggregationStrategy::parse("mixed"), None);
        assert_eq!(
            AggregationStrategy::default(),
            AggregationStrategy::Labelled
        );
        assert_eq!(
            serde_json::to_value(AggregationStrategy::Synthesis).unwrap(),
            serde_json::json!("synthesis")
        );
        assert_eq!(
            Aggregator::new(AggregationStrategy::Synthesis).strategy(),
            AggregationStrategy::Synthesis
        );
    }
}
