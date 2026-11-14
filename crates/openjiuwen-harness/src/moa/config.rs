//! Mixture-of-agents configuration: reference models, routing mode, verifier cadence,
//! budgets and the consultation skill directory.
//!
//! Ports `openjiuwen/harness/moa/config.py` and `openjiuwen/harness/moa/_validation.py`.
//! Python keeps the live reference `Model` handles and the routing policy in one frozen
//! dataclass whose `__post_init__` raises on an invalid combination; Rust keeps the model
//! handles in [`MoaConfig`] (so it is not `serde`) and moves every check into
//! [`MoaConfig::validate`], because construction itself cannot fail without hiding the
//! handle type behind a wrapper.
//!
//! Validation messages are parity-critical: each one reproduces the Python text exactly,
//! so a caller migrating from the Python SDK sees the same diagnostic.
// The public names (`MoaConfig`, `MoaMode`, `VerifierConfig`, ...) mirror the Python
// contract and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::Model;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Character budget applied to one reference response before it is stored.
pub const DEFAULT_MAX_REFERENCE_CHARS: usize = 12_000;
/// Total character budget of the advice block injected into the acting agent's prompt.
pub const DEFAULT_MAX_ADVICE_CHARS: usize = 48_000;
/// Seconds one reference model has to answer a consultation request.
pub const DEFAULT_REFERENCE_TIMEOUT_SECS: u64 = 60;
/// Proposer fan-out width used when a caller configures none.
pub const DEFAULT_MAX_CONCURRENCY: usize = 8;
/// Number of consultation records kept by the bounded history.
pub const DEFAULT_HISTORY_LIMIT: usize = 16;

/// The directory name a consultation skill root must have.
const CONSULT_SKILL_DIR_NAME: &str = "consult-reference-models";
/// The skill body file every consultation skill root must contain.
const CONSULT_SKILL_FILE: &str = "SKILL.md";

/// Routing policy of a mixture-of-agents configuration.
///
/// Serializes as the Python `Literal` strings, so persisted configuration written by the
/// Python SDK loads unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MoaMode {
    /// References are consulted when the acting agent explicitly asks for proposals.
    #[default]
    AggregatorOnDemand,
    /// References are consulted automatically when progress verification stalls.
    VerifierRouted,
    /// An external System One router decides when references are consulted.
    JevRouted,
}

impl MoaMode {
    /// The Python `Literal` spelling of the mode (`"aggregator_on_demand"`, …).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AggregatorOnDemand => "aggregator_on_demand",
            Self::VerifierRouted => "verifier_routed",
            Self::JevRouted => "jev_routed",
        }
    }

    /// Parses an exact snake-case mode name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "aggregator_on_demand" => Some(Self::AggregatorOnDemand),
            "verifier_routed" => Some(Self::VerifierRouted),
            "jev_routed" => Some(Self::JevRouted),
            _ => None,
        }
    }
}

impl fmt::Display for MoaMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How progress verification is performed in [`MoaMode::VerifierRouted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VerifierSelection {
    /// Offline structural progress check; never calls a model.
    #[default]
    Structural,
    /// Model-driven progress check using [`MoaConfig::verifier_model`].
    Llm,
}

impl VerifierSelection {
    /// The snake-case spelling of the selection (`"structural"`, `"llm"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Llm => "llm",
        }
    }

    /// Parses an exact snake-case selection name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "structural" => Some(Self::Structural),
            "llm" => Some(Self::Llm),
            _ => None,
        }
    }
}

/// Scoring cadence and attempt budget of the progress verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct VerifierConfig {
    /// Completed turns between two verifier checkpoints.
    pub interval: usize,
    /// Number of repeated evaluations averaged into one score.
    pub n_evaluations: usize,
    /// Maximum verifier calls per invocation; `None` means unlimited.
    pub max_verifier_calls: Option<usize>,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            interval: 3,
            n_evaluations: 3,
            max_verifier_calls: None,
        }
    }
}

impl VerifierConfig {
    /// Validates the verifier cadence and budget.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when [`Self::interval`] or [`Self::n_evaluations`] is
    /// zero, or when [`Self::max_verifier_calls`] is the unusable `usize::MAX` sentinel.
    pub fn validate(&self) -> Result<()> {
        require_positive("verifier interval", self.interval)?;
        require_positive("verifier n_evaluations", self.n_evaluations)?;
        validate_optional_budget("max_verifier_calls", self.max_verifier_calls)?;
        Ok(())
    }

    /// Returns this configuration with [`Self::interval`] replaced.
    #[must_use]
    pub fn with_interval(mut self, interval: usize) -> Self {
        self.interval = interval;
        self
    }

    /// Returns this configuration with [`Self::n_evaluations`] replaced.
    #[must_use]
    pub fn with_n_evaluations(mut self, n_evaluations: usize) -> Self {
        self.n_evaluations = n_evaluations;
        self
    }

    /// Returns this configuration with [`Self::max_verifier_calls`] replaced.
    #[must_use]
    pub fn with_max_verifier_calls(mut self, max_verifier_calls: Option<usize>) -> Self {
        self.max_verifier_calls = max_verifier_calls;
        self
    }
}

/// Routing cadence of [`MoaMode::JevRouted`].
///
/// Python also holds the caller-owned `JevSystemOneClient` here; this port keeps the
/// serde-able knobs only — the client is an injected seam the caller owns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct JevConfig {
    /// Completed turns between two router checkpoints.
    pub interval: usize,
    /// Recent conversation messages handed to the router.
    pub recent_messages: usize,
    /// Seconds the complete checkpoint (todo reads and client retries included) may take.
    pub timeout_secs: f64,
    /// Maximum router calls per invocation; `None` means unlimited.
    pub max_jev_calls: Option<usize>,
    /// Maximum consultation batches per invocation; `None` means unlimited.
    pub max_reference_calls: Option<usize>,
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            interval: 3,
            recent_messages: 5,
            timeout_secs: 30.0,
            max_jev_calls: None,
            max_reference_calls: None,
        }
    }
}

impl JevConfig {
    /// Validates the routing cadence, timeout and budgets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a cadence counter is zero, when
    /// [`Self::timeout_secs`] is not finite and positive, or when a budget is the
    /// unusable `usize::MAX` sentinel.
    pub fn validate(&self) -> Result<()> {
        require_positive("JEV interval", self.interval)?;
        require_positive("JEV recent_messages", self.recent_messages)?;
        if !self.timeout_secs.is_finite() || self.timeout_secs <= 0.0 {
            return Err(Error::config("JEV timeout must be finite and positive"));
        }
        validate_optional_budget("JEV max_jev_calls", self.max_jev_calls)?;
        validate_optional_budget("JEV max_reference_calls", self.max_reference_calls)?;
        Ok(())
    }
}

/// Reference models and routing policy of one mixture-of-agents consultation.
///
/// Holds live `Arc<dyn Model>` handles, so it is deliberately not `serde` and its
/// [`Debug`] implementation prints presence, never model internals. Validate with
/// [`Self::validate`] before building a rail or a consultation.
///
/// # Examples
///
/// ```
/// use openjiuwen_harness::moa::{MoaConfig, MoaMode};
///
/// let config = MoaConfig::new(Vec::new()).with_mode(MoaMode::VerifierRouted);
/// assert_eq!(config.mode, MoaMode::VerifierRouted);
/// // Construction keeps the handles; validation reports the missing references.
/// assert!(config.validate().is_err());
/// ```
pub struct MoaConfig {
    /// Reference models consulted for independent proposals, in configured order.
    pub reference_models: Vec<Arc<dyn Model>>,
    /// Routing policy; see [`MoaMode`].
    pub mode: MoaMode,
    /// Verifier cadence and budget.
    pub verifier: VerifierConfig,
    /// How progress is verified in [`MoaMode::VerifierRouted`].
    pub verifier_selection: VerifierSelection,
    /// Model used by [`VerifierSelection::Llm`].
    pub verifier_model: Option<Arc<dyn Model>>,
    /// Model that synthesises the final answer; `None` keeps the labelled concatenation.
    pub aggregator_model: Option<Arc<dyn Model>>,
    /// Number of reference models to use; `None` uses all of them.
    pub max_proposals: Option<usize>,
    /// Maximum concurrent proposer requests.
    pub max_concurrency: usize,
    /// Sampling temperature of proposer requests; `None` keeps the model default.
    pub proposer_temperature: Option<f32>,
    /// Sampling temperature of the aggregator request; `None` keeps the model default.
    pub aggregator_temperature: Option<f32>,
    /// Character budget of one stored reference response.
    pub max_reference_chars: usize,
    /// Character budget of the advice block injected into the prompt.
    pub max_advice_chars: usize,
    /// Deadline of one reference response.
    pub reference_timeout: Duration,
    /// Maximum consultation batches per invocation; `None` means unlimited.
    pub max_reference_calls: Option<usize>,
    /// Skill root materialised for callers that expose consultation as a skill.
    pub consultation_skill_dir: Option<PathBuf>,
    /// Router configuration; present exactly in [`MoaMode::JevRouted`].
    pub jev: Option<JevConfig>,
    /// Whether the consultation rail is active.
    pub enabled: bool,
    /// Number of records kept by the bounded consultation history.
    pub history_limit: usize,
    /// Dispatch priority of the consultation rail.
    pub rail_priority: i32,
}

impl MoaConfig {
    /// Builds a configuration for `reference_models` with every knob at its Python default.
    ///
    /// The constructor never fails: call [`Self::validate`] to apply the Python
    /// `__post_init__` rules (an empty reference list among them).
    #[must_use]
    pub fn new(reference_models: Vec<Arc<dyn Model>>) -> Self {
        Self {
            reference_models,
            mode: MoaMode::default(),
            verifier: VerifierConfig::default(),
            verifier_selection: VerifierSelection::default(),
            verifier_model: None,
            aggregator_model: None,
            max_proposals: None,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            proposer_temperature: None,
            aggregator_temperature: None,
            max_reference_chars: DEFAULT_MAX_REFERENCE_CHARS,
            max_advice_chars: DEFAULT_MAX_ADVICE_CHARS,
            reference_timeout: Duration::from_secs(DEFAULT_REFERENCE_TIMEOUT_SECS),
            max_reference_calls: None,
            consultation_skill_dir: None,
            jev: None,
            enabled: true,
            history_limit: DEFAULT_HISTORY_LIMIT,
            rail_priority: crate::rails::rail_priority::MOA,
        }
    }

    /// Applies the Python `MoaConfig.__post_init__` checks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] with the Python message when the reference list is
    /// empty, the mode and the routing pieces disagree, a counter or budget is out of
    /// range, the reference timeout is zero, or the consultation skill directory is not
    /// a `<…>/consult-reference-models/SKILL.md` path in `aggregator_on_demand` mode.
    ///
    /// Skill-directory validation reads the filesystem (synchronously, since
    /// configuration validation is not async).
    pub fn validate(&self) -> Result<()> {
        if self.reference_models.is_empty() {
            return Err(Error::config("MoA requires at least one reference model"));
        }
        // The enum cannot hold an unsupported variant; the check keeps the Python
        // message reachable for callers that map a foreign string through `parse`.
        if MoaMode::parse(self.mode.as_str()).is_none() {
            return Err(Error::config(format!(
                "unsupported MoA mode: {}",
                self.mode.as_str()
            )));
        }
        if self.mode == MoaMode::JevRouted {
            if self.jev.is_none() {
                return Err(Error::config("jev_routed requires a JevConfig"));
            }
        } else if self.jev.is_some() {
            return Err(Error::config(
                "JEV and verifier routing are mutually exclusive; jev is only supported in jev_routed",
            ));
        }
        if self.mode == MoaMode::VerifierRouted
            && self.verifier_selection == VerifierSelection::Llm
            && self.verifier_model.is_none()
        {
            return Err(Error::config("verifier_routed requires a verifier model"));
        }
        self.verifier.validate()?;
        if let Some(jev) = &self.jev {
            jev.validate()?;
        }
        require_positive("max_reference_chars", self.max_reference_chars)?;
        require_positive("max_advice_chars", self.max_advice_chars)?;
        require_positive("max_concurrency", self.max_concurrency)?;
        validate_optional_budget("max_reference_calls", self.max_reference_calls)?;
        if self.reference_timeout.is_zero() {
            return Err(Error::config(
                "reference_timeout must be finite and positive",
            ));
        }
        if let Some(directory) = &self.consultation_skill_dir {
            self.validate_consultation_skill_dir(directory)?;
        }
        Ok(())
    }

    /// Number of proposals one consultation batch is expected to produce.
    ///
    /// [`Self::max_proposals`] caps the fan-out, the model count caps the result, and a
    /// non-empty model list always yields at least one slot.
    #[must_use]
    pub fn proposal_count(&self) -> usize {
        let total = self.reference_models.len();
        if total == 0 {
            return 0;
        }
        self.max_proposals.unwrap_or(total).min(total).max(1)
    }

    /// Returns this configuration with [`Self::mode`] replaced.
    #[must_use]
    pub fn with_mode(mut self, mode: MoaMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns this configuration with [`Self::verifier`] replaced.
    #[must_use]
    pub fn with_verifier(mut self, verifier: VerifierConfig) -> Self {
        self.verifier = verifier;
        self
    }

    /// Returns this configuration with [`Self::verifier_selection`] replaced.
    #[must_use]
    pub fn with_verifier_selection(mut self, verifier_selection: VerifierSelection) -> Self {
        self.verifier_selection = verifier_selection;
        self
    }

    /// Returns this configuration with [`Self::verifier_model`] set.
    #[must_use]
    pub fn with_verifier_model(mut self, verifier_model: Arc<dyn Model>) -> Self {
        self.verifier_model = Some(verifier_model);
        self
    }

    /// Returns this configuration with [`Self::aggregator_model`] set.
    #[must_use]
    pub fn with_aggregator_model(mut self, aggregator_model: Arc<dyn Model>) -> Self {
        self.aggregator_model = Some(aggregator_model);
        self
    }

    /// Returns this configuration with [`Self::max_proposals`] replaced.
    #[must_use]
    pub fn with_max_proposals(mut self, max_proposals: Option<usize>) -> Self {
        self.max_proposals = max_proposals;
        self
    }

    /// Returns this configuration with [`Self::max_concurrency`] replaced.
    #[must_use]
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Returns this configuration with [`Self::proposer_temperature`] set.
    #[must_use]
    pub fn with_proposer_temperature(mut self, proposer_temperature: f32) -> Self {
        self.proposer_temperature = Some(proposer_temperature);
        self
    }

    /// Returns this configuration with [`Self::aggregator_temperature`] set.
    #[must_use]
    pub fn with_aggregator_temperature(mut self, aggregator_temperature: f32) -> Self {
        self.aggregator_temperature = Some(aggregator_temperature);
        self
    }

    /// Returns this configuration with [`Self::max_reference_chars`] replaced.
    #[must_use]
    pub fn with_max_reference_chars(mut self, max_reference_chars: usize) -> Self {
        self.max_reference_chars = max_reference_chars;
        self
    }

    /// Returns this configuration with [`Self::max_advice_chars`] replaced.
    #[must_use]
    pub fn with_max_advice_chars(mut self, max_advice_chars: usize) -> Self {
        self.max_advice_chars = max_advice_chars;
        self
    }

    /// Returns this configuration with [`Self::reference_timeout`] replaced.
    #[must_use]
    pub fn with_reference_timeout(mut self, reference_timeout: Duration) -> Self {
        self.reference_timeout = reference_timeout;
        self
    }

    /// Returns this configuration with [`Self::max_reference_calls`] replaced.
    #[must_use]
    pub fn with_max_reference_calls(mut self, max_reference_calls: Option<usize>) -> Self {
        self.max_reference_calls = max_reference_calls;
        self
    }

    /// Returns this configuration with [`Self::consultation_skill_dir`] set.
    #[must_use]
    pub fn with_consultation_skill_dir(
        mut self,
        consultation_skill_dir: impl Into<PathBuf>,
    ) -> Self {
        self.consultation_skill_dir = Some(consultation_skill_dir.into());
        self
    }

    /// Returns this configuration with [`Self::jev`] set.
    #[must_use]
    pub fn with_jev(mut self, jev: JevConfig) -> Self {
        self.jev = Some(jev);
        self
    }

    /// Returns this configuration with [`Self::enabled`] replaced.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Returns this configuration with [`Self::history_limit`] replaced.
    #[must_use]
    pub fn with_history_limit(mut self, history_limit: usize) -> Self {
        self.history_limit = history_limit;
        self
    }

    /// Returns this configuration with [`Self::rail_priority`] replaced.
    #[must_use]
    pub fn with_rail_priority(mut self, rail_priority: i32) -> Self {
        self.rail_priority = rail_priority;
        self
    }

    /// Applies the Python `consultation_skill_dir` rules to `directory`.
    fn validate_consultation_skill_dir(&self, directory: &Path) -> Result<()> {
        if self.mode != MoaMode::AggregatorOnDemand {
            return Err(Error::config(
                "consultation_skill_dir is only supported in aggregator_on_demand mode",
            ));
        }
        if directory.to_string_lossy().trim().is_empty() {
            return Err(Error::config(
                "consultation_skill_dir must be a nonempty skill directory path",
            ));
        }
        let named_correctly = directory
            .file_name()
            .is_some_and(|name| name == CONSULT_SKILL_DIR_NAME);
        if !named_correctly || !directory.join(CONSULT_SKILL_FILE).is_file() {
            return Err(Error::config(
                "consultation_skill_dir must be a consult-reference-models directory with SKILL.md",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for MoaConfig {
    /// Renders presence and knobs only; model handles and their configuration never leak.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoaConfig")
            .field("reference_models", &self.reference_models.len())
            .field("mode", &self.mode)
            .field("verifier", &self.verifier)
            .field("verifier_selection", &self.verifier_selection)
            .field("verifier_model", &Presence(self.verifier_model.is_some()))
            .field(
                "aggregator_model",
                &Presence(self.aggregator_model.is_some()),
            )
            .field("max_proposals", &self.max_proposals)
            .field("max_concurrency", &self.max_concurrency)
            .field("proposer_temperature", &self.proposer_temperature)
            .field("aggregator_temperature", &self.aggregator_temperature)
            .field("max_reference_chars", &self.max_reference_chars)
            .field("max_advice_chars", &self.max_advice_chars)
            .field("reference_timeout", &self.reference_timeout)
            .field("max_reference_calls", &self.max_reference_calls)
            .field("consultation_skill_dir", &self.consultation_skill_dir)
            .field("jev", &self.jev)
            .field("enabled", &self.enabled)
            .field("history_limit", &self.history_limit)
            .field("rail_priority", &self.rail_priority)
            .finish()
    }
}

/// Prints `Some` or `None` for an optional live handle without touching the handle.
struct Presence(bool);

impl fmt::Debug for Presence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(if self.0 { "Some" } else { "None" })
    }
}

/// Rejects a zero count with the Python `validate_positive_int` message.
fn require_positive(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(Error::config(format!("{name} must be a positive integer")));
    }
    Ok(())
}

/// Rejects an attempt budget that cannot behave as a finite budget.
///
/// Python rejects negative and non-integral budgets; a `usize` can hold neither, so the
/// only representable violation is `usize::MAX`, which every `calls >= limit` comparison
/// silently treats as unlimited while looking finite. That sentinel is rejected with the
/// Python message so an accidental "huge budget" fails loudly.
fn validate_optional_budget(name: &str, value: Option<usize>) -> Result<()> {
    match value {
        Some(usize::MAX) => Err(Error::config(format!(
            "{name} must be a nonnegative integer or None"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use openjiuwen_test_support::ScriptedModel;

    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::text("reference"))
    }

    fn config() -> MoaConfig {
        MoaConfig::new(vec![model(), model(), model()])
    }

    fn assert_config_error(result: Result<()>, expected: &str) {
        let error = result.expect_err("expected a configuration error");
        assert!(
            matches!(error, Error::Config(_)),
            "expected a config error, got {error:?}"
        );
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }

    #[test]
    fn defaults_match_python() {
        let verifier = VerifierConfig::default();
        assert_eq!(verifier.interval, 3);
        assert_eq!(verifier.n_evaluations, 3);
        assert_eq!(verifier.max_verifier_calls, None);

        let jev = JevConfig::default();
        assert_eq!(jev.interval, 3);
        assert_eq!(jev.recent_messages, 5);
        assert!((jev.timeout_secs - 30.0).abs() < f64::EPSILON);
        assert_eq!(jev.max_jev_calls, None);
        assert_eq!(jev.max_reference_calls, None);

        let config = config();
        assert_eq!(config.mode, MoaMode::AggregatorOnDemand);
        assert_eq!(config.verifier_selection, VerifierSelection::Structural);
        assert_eq!(config.max_proposals, None);
        assert_eq!(config.max_concurrency, DEFAULT_MAX_CONCURRENCY);
        assert_eq!(config.max_reference_chars, DEFAULT_MAX_REFERENCE_CHARS);
        assert_eq!(config.max_advice_chars, DEFAULT_MAX_ADVICE_CHARS);
        assert_eq!(
            config.reference_timeout,
            Duration::from_secs(DEFAULT_REFERENCE_TIMEOUT_SECS)
        );
        assert_eq!(config.history_limit, DEFAULT_HISTORY_LIMIT);
        assert!(config.enabled);
        assert_eq!(config.rail_priority, crate::rails::rail_priority::MOA);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn modes_round_trip_through_serde_and_parse() {
        assert_eq!(
            serde_json::to_string(&MoaMode::VerifierRouted).unwrap(),
            "\"verifier_routed\""
        );
        assert_eq!(
            serde_json::from_str::<MoaMode>("\"jev_routed\"").unwrap(),
            MoaMode::JevRouted
        );
        for mode in [
            MoaMode::AggregatorOnDemand,
            MoaMode::VerifierRouted,
            MoaMode::JevRouted,
        ] {
            assert_eq!(MoaMode::parse(mode.as_str()), Some(mode));
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!(MoaMode::parse("on_demand"), None);
        assert!(serde_json::from_str::<MoaMode>("\"unknown\"").is_err());
        assert_eq!(MoaMode::default(), MoaMode::AggregatorOnDemand);
        assert_eq!(
            VerifierSelection::parse("llm"),
            Some(VerifierSelection::Llm)
        );
        assert_eq!(VerifierSelection::parse("judge"), None);
        assert_eq!(VerifierSelection::default().as_str(), "structural");
    }

    #[test]
    fn verifier_and_jev_configs_round_trip() {
        let verifier = VerifierConfig {
            interval: 5,
            n_evaluations: 2,
            max_verifier_calls: Some(4),
        };
        let json = serde_json::to_string(&verifier).unwrap();
        assert_eq!(
            serde_json::from_str::<VerifierConfig>(&json).unwrap(),
            verifier
        );
        let jev: JevConfig =
            serde_json::from_str("{\"interval\": 2, \"timeout_secs\": 12.5}").unwrap();
        assert_eq!(jev.interval, 2);
        assert_eq!(jev.recent_messages, 5);
        assert!((jev.timeout_secs - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn proposal_count_respects_caps() {
        assert_eq!(config().proposal_count(), 3);
        assert_eq!(config().with_max_proposals(Some(2)).proposal_count(), 2);
        assert_eq!(config().with_max_proposals(Some(9)).proposal_count(), 3);
        assert_eq!(config().with_max_proposals(Some(0)).proposal_count(), 1);
        assert_eq!(MoaConfig::new(Vec::new()).proposal_count(), 0);
    }

    #[test]
    fn validate_requires_a_reference_model() {
        assert_config_error(
            MoaConfig::new(Vec::new()).validate(),
            "MoA requires at least one reference model",
        );
    }

    #[test]
    fn validate_verifier_routing_requires_a_model() {
        let missing = config()
            .with_mode(MoaMode::VerifierRouted)
            .with_verifier_selection(VerifierSelection::Llm);
        assert_config_error(
            missing.validate(),
            "verifier_routed requires a verifier model",
        );

        let complete = missing.with_verifier_model(model());
        assert!(complete.validate().is_ok());
    }

    #[test]
    fn validate_jev_routing_and_mutual_exclusion() {
        assert_config_error(
            config().with_mode(MoaMode::JevRouted).validate(),
            "jev_routed requires a JevConfig",
        );
        assert!(config()
            .with_mode(MoaMode::JevRouted)
            .with_jev(JevConfig::default())
            .validate()
            .is_ok());
        assert_config_error(
            config().with_jev(JevConfig::default()).validate(),
            "JEV and verifier routing are mutually exclusive; jev is only supported in jev_routed",
        );
    }

    #[test]
    fn validate_positive_counters() {
        assert_config_error(
            config().with_max_reference_chars(0).validate(),
            "max_reference_chars must be a positive integer",
        );
        assert_config_error(
            config().with_max_advice_chars(0).validate(),
            "max_advice_chars must be a positive integer",
        );
        assert_config_error(
            config().with_max_concurrency(0).validate(),
            "max_concurrency must be a positive integer",
        );
        assert_config_error(
            config()
                .with_verifier(VerifierConfig::default().with_interval(0))
                .validate(),
            "verifier interval must be a positive integer",
        );
        assert_config_error(
            config()
                .with_verifier(VerifierConfig::default().with_n_evaluations(0))
                .validate(),
            "verifier n_evaluations must be a positive integer",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    interval: 0,
                    ..JevConfig::default()
                })
                .validate(),
            "JEV interval must be a positive integer",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    recent_messages: 0,
                    ..JevConfig::default()
                })
                .validate(),
            "JEV recent_messages must be a positive integer",
        );
    }

    #[test]
    fn validate_timeouts() {
        assert_config_error(
            config().with_reference_timeout(Duration::ZERO).validate(),
            "reference_timeout must be finite and positive",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    timeout_secs: 0.0,
                    ..JevConfig::default()
                })
                .validate(),
            "JEV timeout must be finite and positive",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    timeout_secs: f64::NAN,
                    ..JevConfig::default()
                })
                .validate(),
            "JEV timeout must be finite and positive",
        );
    }

    #[test]
    fn validate_optional_budgets() {
        assert_config_error(
            config()
                .with_max_reference_calls(Some(usize::MAX))
                .validate(),
            "max_reference_calls must be a nonnegative integer or None",
        );
        assert_config_error(
            config()
                .with_verifier(VerifierConfig::default().with_max_verifier_calls(Some(usize::MAX)))
                .validate(),
            "max_verifier_calls must be a nonnegative integer or None",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    max_jev_calls: Some(usize::MAX),
                    ..JevConfig::default()
                })
                .validate(),
            "JEV max_jev_calls must be a nonnegative integer or None",
        );
        assert_config_error(
            config()
                .with_mode(MoaMode::JevRouted)
                .with_jev(JevConfig {
                    max_reference_calls: Some(usize::MAX),
                    ..JevConfig::default()
                })
                .validate(),
            "JEV max_reference_calls must be a nonnegative integer or None",
        );
        assert!(config()
            .with_max_reference_calls(Some(0))
            .validate()
            .is_ok());
    }

    #[test]
    fn validate_consultation_skill_dir() {
        assert_config_error(
            config()
                .with_mode(MoaMode::VerifierRouted)
                .with_consultation_skill_dir("/tmp/consult-reference-models")
                .validate(),
            "consultation_skill_dir is only supported in aggregator_on_demand mode",
        );
        assert_config_error(
            config().with_consultation_skill_dir("   ").validate(),
            "consultation_skill_dir must be a nonempty skill directory path",
        );
        assert_config_error(
            config()
                .with_consultation_skill_dir("/tmp/skills")
                .validate(),
            "consultation_skill_dir must be a consult-reference-models directory with SKILL.md",
        );

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("consult-reference-models");
        std::fs::create_dir_all(&root).unwrap();
        assert_config_error(
            config().with_consultation_skill_dir(&root).validate(),
            "consultation_skill_dir must be a consult-reference-models directory with SKILL.md",
        );

        std::fs::write(root.join("SKILL.md"), "---\nname: x\n---\n").unwrap();
        let valid = config().with_consultation_skill_dir(&root);
        assert!(valid.validate().is_ok());
        assert_eq!(valid.consultation_skill_dir.as_ref(), Some(&root));
    }

    #[test]
    fn builders_replace_every_knob() {
        let reference = model();
        let config = MoaConfig::new(vec![reference])
            .with_mode(MoaMode::VerifierRouted)
            .with_verifier(VerifierConfig::default().with_interval(2))
            .with_verifier_selection(VerifierSelection::Llm)
            .with_verifier_model(model())
            .with_aggregator_model(model())
            .with_max_proposals(Some(1))
            .with_max_concurrency(2)
            .with_proposer_temperature(0.2)
            .with_aggregator_temperature(0.7)
            .with_max_reference_chars(100)
            .with_max_advice_chars(200)
            .with_reference_timeout(Duration::from_secs(5))
            .with_max_reference_calls(Some(1))
            .with_jev(JevConfig::default())
            .with_enabled(false)
            .with_history_limit(4)
            .with_rail_priority(7);
        assert_eq!(config.mode, MoaMode::VerifierRouted);
        assert_eq!(config.verifier.interval, 2);
        assert_eq!(config.verifier_selection, VerifierSelection::Llm);
        assert!(config.verifier_model.is_some());
        assert!(config.aggregator_model.is_some());
        assert_eq!(config.max_proposals, Some(1));
        assert_eq!(config.max_concurrency, 2);
        assert_eq!(config.proposer_temperature, Some(0.2));
        assert_eq!(config.aggregator_temperature, Some(0.7));
        assert_eq!(config.max_reference_chars, 100);
        assert_eq!(config.max_advice_chars, 200);
        assert_eq!(config.reference_timeout, Duration::from_secs(5));
        assert_eq!(config.max_reference_calls, Some(1));
        assert!(config.jev.is_some());
        assert!(!config.enabled);
        assert_eq!(config.history_limit, 4);
        assert_eq!(config.rail_priority, 7);
        // `jev` is only legal in `jev_routed`; the mode-specific rule still reports it.
        assert!(config.validate().is_err());
    }

    #[test]
    fn debug_prints_presence_only() {
        let config = config().with_aggregator_model(model());
        let rendered = format!("{config:?}");
        assert!(rendered.contains("MoaConfig"));
        assert!(rendered.contains("reference_models: 3"));
        assert!(rendered.contains("aggregator_model: Some"));
        assert!(rendered.contains("verifier_model: None"));
        assert!(rendered.contains("enabled: true"));
        assert!(!rendered.contains("aggregator_model: Some("));
    }
}
