//! Tip induction from trajectories and evolution signals.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.ttse.induction`:
//! failure evidence and explicit signals become candidate tips. The offline
//! path is deterministic; an injected [`Model`](openjiuwen_core::foundation::llm::Model)
//! enables a prompt-driven path that falls back to the deterministic one.
//!
//! Two invariants make the offline path reproducible: tips derive from the
//! recorded steps only (no clock, no I/O, no random source) and candidates keep
//! input order, so identical inputs yield identical records. Successful
//! trajectories never produce a tip — they are evidence that a run went fine,
//! not a lesson — so only failures and error-carrying `Unknown` runs are mined.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;

use crate::signal::EvolutionSignal;
use crate::trajectory::{render_digest, Trajectory, TrajectoryOutcome};
use crate::ttse::bank::{jaccard, tokenize};
use crate::ttse::classify::TipClassifier;
use crate::ttse::config::TtseConfig;
use crate::ttse::models::{truncate_chars, TipCategory, TipRecord};
use crate::ttse::parse::TipParser;

/// The fixed instruction the model-driven path sends as its system message.
const INDUCE_INSTRUCTION: &str = "You update a bank of short, actionable engineering tips. Reply with JSON: {\"tips\": [{\"rule\": \"...\", \"category\": \"tool|workflow|pitfall|environment|communication|other\"}]}";

/// The provenance label of tips that came from a model answer.
const MODEL_SOURCE: &str = "model";
/// The provenance label of tips mined from a trajectory.
const TRAJECTORY_SOURCE: &str = "trajectory";
/// The provenance label of tips mined from an evolution signal.
const SIGNAL_SOURCE: &str = "signal";

/// How many trajectory digests one induction prompt carries.
const MAX_PROMPT_TRAJECTORIES: usize = 12;
/// How many signal lines one induction prompt carries.
const MAX_PROMPT_SIGNALS: usize = 12;
/// How many characters of each trajectory digest the prompt keeps.
const DIGEST_CHARS: usize = 600;
/// How many characters of an error step become part of a rule.
const ERROR_EXCERPT_CHARS: usize = 80;
/// How many distinct error signatures one trajectory may contribute.
const MAX_TIPS_PER_TRAJECTORY: usize = 8;
/// How many keywords a tip keeps.
const MAX_KEYWORDS: usize = 32;
/// The lowest signal confidence that still produces a tip.
const MIN_SIGNAL_CONFIDENCE: f64 = 0.5;

/// Produces candidate tips from trajectories and signals.
///
/// The deterministic path needs no model and is always available; injecting one
/// through [`TipInducer::with_model`] enables the prompt-driven path, which
/// falls back to the deterministic result whenever the model fails or answers
/// with something that is not a tip list. Model errors are never propagated.
#[derive(Clone)]
pub struct TipInducer {
    /// The shared TTSE tunables (`max_tips_per_observation`, `dedup_threshold`).
    config: TtseConfig,
    /// Classifies a rule into a [`TipCategory`].
    classifier: TipClassifier,
    /// Parses a model answer into tip records.
    parser: TipParser,
    /// The optional model driving the prompt path; `None` keeps induction offline.
    model: Option<Arc<dyn Model>>,
    /// The request sent with every model call.
    request: ModelRequestConfig,
}

impl fmt::Debug for TipInducer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("TipInducer");
        debug
            .field("config", &self.config)
            .field("classifier", &self.classifier)
            .field("parser", &self.parser)
            .field("request", &self.request);
        if let Some(model) = &self.model {
            debug.field("model", &model.model_name());
        }
        debug.finish()
    }
}

impl TipInducer {
    /// Builds an offline inducer for `config`.
    ///
    /// The classifier and parser follow the config
    /// ([`TipParser::from_config`]); no model is installed, so induction is
    /// deterministic until [`Self::with_model`] is called.
    #[must_use]
    pub fn new(config: TtseConfig) -> Self {
        let parser = TipParser::from_config(&config);
        Self {
            config,
            classifier: TipClassifier::default(),
            parser,
            model: None,
            request: ModelRequestConfig::default(),
        }
    }

    /// Installs the model used by the prompt-driven path.
    ///
    /// The model is optional at runtime: a failing or unusable answer falls back
    /// to the deterministic path, so injecting one never makes induction failable.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        self.model = Some(model);
        self.request = request;
        self
    }

    /// The configuration this inducer applies.
    #[must_use]
    pub fn config(&self) -> &TtseConfig {
        &self.config
    }

    /// Whether the prompt-driven path is available.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Induces candidate tips from `trajectories` and `signals`.
    ///
    /// With a model installed the prompt path runs first; when it fails, or when
    /// its answer yields no tip, the deterministic path below runs instead. The
    /// result is capped at [`TtseConfig::max_tips_per_observation`] and holds no
    /// duplicates: a candidate whose token Jaccard similarity against an
    /// already-kept candidate reaches [`TtseConfig::dedup_threshold`] is skipped.
    ///
    /// The deterministic path mines, in this order:
    ///
    /// 1. one tip per distinct error signature of a failed trajectory (or of an
    ///    `Unknown` trajectory that recorded an error), plus one
    ///    `Run failed: …` tip for a failed trajectory without error steps;
    /// 2. one tip per signal with a confidence of at least `0.5`, weighted
    ///    `1.0 + signal.weight`, so a signal tip weighs `1.0..=2.0`.
    ///
    /// Successful trajectories contribute nothing. Empty input yields no tips.
    pub async fn induce(
        &self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
    ) -> Result<Vec<TipRecord>> {
        if trajectories.is_empty() && signals.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(tips) = self.induce_with_model(trajectories, signals).await {
            return Ok(tips);
        }
        Ok(self.induce_offline(trajectories, signals))
    }

    /// Runs the prompt-driven path, returning `None` when it yields nothing.
    ///
    /// A model error, an unusable answer, or an answer without tips all log at
    /// [`tracing::debug`] and return `None`; no model error is propagated.
    async fn induce_with_model(
        &self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
    ) -> Option<Vec<TipRecord>> {
        let model = self.model.as_ref()?;
        let messages = build_prompt(trajectories, signals);
        let response = match model.invoke(&messages, &self.request).await {
            Ok(response) => response,
            Err(error) => {
                tracing::debug!(
                    model = model.model_name(),
                    error = %error,
                    "ttse induction fell back to the deterministic path"
                );
                return None;
            }
        };
        let content = response
            .message()
            .map_or_else(String::new, |message| message.text().into_owned());
        if !looks_like_tip_list(&content) {
            tracing::debug!(
                model = model.model_name(),
                "ttse induction answer is not a tip list, using the deterministic path"
            );
            return None;
        }
        let mut tips: Vec<TipRecord> = self
            .parser
            .parse(&content)
            .into_iter()
            .map(|tip| tip.with_source(MODEL_SOURCE))
            .collect();
        tips.truncate(self.config.max_tips_per_observation);
        if tips.is_empty() {
            tracing::debug!(
                model = model.model_name(),
                "ttse induction answer held no tip, using the deterministic path"
            );
            return None;
        }
        Some(tips)
    }

    /// The deterministic offline path: failure evidence first, then signals.
    fn induce_offline(
        &self,
        trajectories: &[Trajectory],
        signals: &[EvolutionSignal],
    ) -> Vec<TipRecord> {
        let mut collector = Collector::new(&self.config, &self.classifier);
        for trajectory in trajectories {
            if !is_failed(trajectory) {
                continue;
            }
            collector.add_trajectory(trajectory);
        }
        for signal in signals {
            collector.add_signal(signal);
        }
        collector.tips
    }
}

/// Collects candidates in input order, applying dedupe and the tip cap.
struct Collector<'a> {
    /// The shared tunables: the cap and the dedupe threshold.
    config: &'a TtseConfig,
    /// Classifies a rule into a [`TipCategory`].
    classifier: &'a TipClassifier,
    /// The tokens of every kept candidate, used for the similarity check.
    kept_tokens: Vec<Vec<String>>,
    /// The accepted candidates, in input order.
    tips: Vec<TipRecord>,
}

impl<'a> Collector<'a> {
    /// Builds an empty collector.
    fn new(config: &'a TtseConfig, classifier: &'a TipClassifier) -> Self {
        Self {
            config,
            classifier,
            kept_tokens: Vec::new(),
            tips: Vec::new(),
        }
    }

    /// Whether the tip cap is reached.
    fn is_full(&self) -> bool {
        self.tips.len() >= self.config.max_tips_per_observation
    }

    /// Adds one candidate unless the cap is reached or it duplicates a kept tip.
    fn push(&mut self, rule: &str, category: TipCategory, source: &str, weight: f64) {
        if self.is_full() {
            return;
        }
        let Ok(tip) = TipRecord::new(rule, category) else {
            return;
        };
        let tokens = tokenize(rule);
        let duplicate = self
            .kept_tokens
            .iter()
            .any(|kept| jaccard(kept, &tokens) >= self.config.dedup_threshold);
        if duplicate {
            return;
        }
        let mut keywords = tokens.clone();
        keywords.truncate(MAX_KEYWORDS);
        let tip = tip.with_keywords(keywords).with_source(source);
        let Ok(tip) = tip.with_weight(weight) else {
            return;
        };
        self.tips.push(tip);
        self.kept_tokens.push(tokens);
    }

    /// Adds one tip per distinct error signature of a failed trajectory.
    fn add_trajectory(&mut self, trajectory: &Trajectory) {
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        let mut produced = 0usize;
        for step in trajectory.steps() {
            if self.is_full() {
                return;
            }
            if !step.is_error() {
                continue;
            }
            let excerpt = normalize_excerpt(&step.content, ERROR_EXCERPT_CHARS);
            if excerpt.is_empty() {
                continue;
            }
            let tool = step
                .tool_name
                .as_deref()
                .map(str::trim)
                .filter(|name| !name.is_empty());
            let signature = (tool.unwrap_or("-").to_owned(), excerpt.clone());
            if !seen.insert(signature) {
                continue;
            }
            let rule = match tool {
                Some(name) => format!("Check {name} before retrying: {excerpt}"),
                None => format!("Avoid repeating: {excerpt}"),
            };
            let category = self.classifier.classify(&rule);
            self.push(&rule, category, TRAJECTORY_SOURCE, 1.0);
            produced += 1;
            if produced >= MAX_TIPS_PER_TRAJECTORY {
                return;
            }
        }
        if produced > 0 {
            return;
        }
        // A failed run without error steps still teaches something: keep a
        // bounded excerpt of where it stopped.
        let Some(last) = trajectory
            .steps()
            .iter()
            .rev()
            .find(|step| !step.content.trim().is_empty())
        else {
            return;
        };
        let excerpt = normalize_excerpt(&last.content, ERROR_EXCERPT_CHARS);
        if excerpt.is_empty() {
            return;
        }
        let rule = format!("Run failed: {excerpt}");
        let category = self.classifier.classify(&rule);
        self.push(&rule, category, TRAJECTORY_SOURCE, 1.0);
    }

    /// Adds one tip for a signal that is confident enough.
    ///
    /// The weight is `1.0 + signal.weight` over a weight clamped into
    /// `0.0..=1.0`, so a signal tip always weighs `1.0..=2.0`; a non-finite
    /// weight falls back to `1.0` instead of rejecting the tip.
    fn add_signal(&mut self, signal: &EvolutionSignal) {
        if self.is_full() {
            return;
        }
        if !is_confident(signal.confidence) {
            return;
        }
        let rule = signal.excerpt.trim();
        if rule.is_empty() {
            return;
        }
        let category = self.classifier.classify(rule);
        self.push(rule, category, SIGNAL_SOURCE, signal_weight(signal.weight));
    }
}

/// Whether a signal confidence clears [`MIN_SIGNAL_CONFIDENCE`].
///
/// Written as a positive predicate so a `NaN` confidence is rejected instead of
/// slipping through a negated comparison.
fn is_confident(confidence: f64) -> bool {
    confidence >= MIN_SIGNAL_CONFIDENCE
}

/// The tip weight one signal contributes, clamped into `1.0..=2.0`.
fn signal_weight(weight: f64) -> f64 {
    if weight.is_finite() {
        1.0 + weight.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Whether a trajectory counts as failed evidence.
///
/// A recorded [`TrajectoryOutcome::Failure`] always does; an
/// [`TrajectoryOutcome::Unknown`] run only when it recorded an error step, since
/// an error-free run is simply unscored.
fn is_failed(trajectory: &Trajectory) -> bool {
    match trajectory.outcome {
        TrajectoryOutcome::Failure => true,
        TrajectoryOutcome::Unknown => trajectory.error_count() > 0,
        TrajectoryOutcome::Success => false,
    }
}

/// Collapses whitespace and caps `text` at `max_chars` characters.
fn normalize_excerpt(text: &str, max_chars: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&collapsed, max_chars)
}

/// Whether a model answer looks like a tip list rather than prose.
///
/// The induction prompt asks for JSON, so a structured answer is a JSON payload
/// or a markdown list. Unstructured prose (for example a model replying "I could
/// not find anything") is treated as a failed parse instead of becoming one
/// meaningless tip through the parser's line fallback.
fn looks_like_tip_list(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return true;
    }
    text.lines().any(|line| {
        let line = line.trim();
        let first = line.chars().next();
        matches!(first, Some('{' | '[' | '-' | '*' | '+')) || {
            let lowered = line.to_ascii_lowercase();
            lowered.starts_with("tip:") || lowered.starts_with("fact:")
        }
    })
}

/// Builds the deterministic induction prompt for `trajectories` and `signals`.
///
/// At most [`MAX_PROMPT_TRAJECTORIES`] digests (each bounded to
/// [`DIGEST_CHARS`] characters, prefixed by `trajectory <id> (<outcome>)`) and
/// [`MAX_PROMPT_SIGNALS`] signal lines (`- [<signal_type>] <excerpt>`) render in
/// input order, so identical inputs produce an identical prompt.
fn build_prompt(trajectories: &[Trajectory], signals: &[EvolutionSignal]) -> Vec<BaseMessage> {
    let mut lines: Vec<String> = vec!["Recent trajectories:".to_owned()];
    if trajectories.is_empty() {
        lines.push("(none)".to_owned());
    }
    for trajectory in trajectories.iter().take(MAX_PROMPT_TRAJECTORIES) {
        lines.push(format!(
            "trajectory {} ({})",
            trajectory.trajectory_id, trajectory.outcome
        ));
        lines.push(render_digest(trajectory, DIGEST_CHARS));
    }
    lines.push("Evolution signals:".to_owned());
    if signals.is_empty() {
        lines.push("(none)".to_owned());
    }
    for signal in signals.iter().take(MAX_PROMPT_SIGNALS) {
        lines.push(format!(
            "- [{}] {}",
            signal.signal_type,
            signal.excerpt.trim()
        ));
    }
    vec![
        BaseMessage::system(INDUCE_INSTRUCTION),
        BaseMessage::user(lines.join("\n")),
    ]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_core::foundation::llm::Model;
    use openjiuwen_test_support::ScriptedModel;

    use super::*;
    use crate::signal::EvolutionTarget;
    use crate::trajectory::{StepKind, StepStatus, TrajectoryStep};

    fn step(kind: StepKind, content: &str) -> TrajectoryStep {
        TrajectoryStep::new(kind, content)
    }

    fn failed_step(content: &str) -> TrajectoryStep {
        step(StepKind::ToolResult, content).with_status(StepStatus::Error)
    }

    fn trajectory(id: &str, outcome: TrajectoryOutcome, steps: Vec<TrajectoryStep>) -> Trajectory {
        let mut trajectory = Trajectory::new(id).expect("valid id").with_outcome(outcome);
        trajectory.extend(steps);
        trajectory
    }

    fn signal(excerpt: &str, confidence: f64, weight: f64) -> EvolutionSignal {
        EvolutionSignal::new("execution_failure", EvolutionTarget::Memory, excerpt)
            .expect("valid signal")
            .with_confidence(confidence)
            .expect("valid confidence")
            .with_weight(weight)
            .expect("valid weight")
    }

    #[tokio::test]
    async fn failed_trajectory_yields_tool_and_plain_rules() {
        let inducer = TipInducer::new(TtseConfig::default());
        let trajectory = trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            vec![
                step(StepKind::ToolCall, "cargo test"),
                failed_step("error[E0308]: mismatched types in lib.rs").with_tool_name("shell"),
                failed_step("command timed out after 30s"),
            ],
        );

        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("induction succeeds");

        assert_eq!(tips.len(), 2, "one tip per distinct error signature");
        assert_eq!(
            tips[0].rule,
            "Check shell before retrying: error[E0308]: mismatched types in lib.rs"
        );
        assert_eq!(tips[0].source, "trajectory");
        assert_eq!(tips[0].category, TipCategory::Pitfall);
        assert!((tips[0].weight - 1.0).abs() < f64::EPSILON);
        assert!(tips[0].keywords.contains(&"shell".to_owned()));
        assert_eq!(tips[1].rule, "Avoid repeating: command timed out after 30s");
        assert_eq!(tips[1].source, "trajectory");
        assert!(tips.iter().all(|tip| tip.created_tick == 0));
    }

    #[tokio::test]
    async fn successful_trajectory_yields_nothing() {
        let inducer = TipInducer::new(TtseConfig::default());
        let trajectory = trajectory(
            "t-1",
            TrajectoryOutcome::Success,
            vec![
                failed_step("a recovered error"),
                step(StepKind::Message, "all good"),
            ],
        );

        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("induction succeeds");

        assert!(tips.is_empty(), "success is not a lesson: {tips:?}");
    }

    #[tokio::test]
    async fn unknown_with_errors_counts_as_failed() {
        let inducer = TipInducer::new(TtseConfig::default());
        let with_error = trajectory(
            "t-1",
            TrajectoryOutcome::Unknown,
            vec![failed_step("permission denied on /root").with_tool_name("write_file")],
        );
        let without_error = trajectory(
            "t-2",
            TrajectoryOutcome::Unknown,
            vec![step(StepKind::Message, "nothing recorded")],
        );

        let tips = inducer
            .induce(&[with_error, without_error], &[])
            .await
            .expect("induction succeeds");

        assert_eq!(tips.len(), 1);
        assert_eq!(
            tips[0].rule,
            "Check write_file before retrying: permission denied on /root"
        );
    }

    #[tokio::test]
    async fn failed_without_error_steps_uses_the_last_content() {
        let inducer = TipInducer::new(TtseConfig::default());
        let trajectory = trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            vec![
                step(StepKind::Message, "starting"),
                step(StepKind::Model, "   "),
                step(StepKind::Message, "gave up on the build"),
            ],
        );

        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("induction succeeds");

        assert_eq!(tips.len(), 1);
        assert_eq!(tips[0].rule, "Run failed: gave up on the build");
        assert_eq!(tips[0].source, "trajectory");
    }

    #[tokio::test]
    async fn low_confidence_signals_are_skipped_and_weights_add() {
        let inducer = TipInducer::new(TtseConfig::default());
        let signals = vec![
            signal("Prefer the offline registry cache", 0.9, 0.25),
            signal("Ignore this weak hint", 0.49, 1.0),
            signal("Run cargo fmt before the tests", 0.5, 1.0),
        ];

        let tips = inducer.induce(&[], &signals).await.expect("induction");

        assert_eq!(tips.len(), 2, "the 0.49-confidence signal is skipped");
        assert_eq!(tips[0].rule, "Prefer the offline registry cache");
        assert_eq!(tips[0].source, "signal");
        assert!((tips[0].weight - 1.25).abs() < f64::EPSILON);
        assert_eq!(tips[1].rule, "Run cargo fmt before the tests");
        assert!((tips[1].weight - 2.0).abs() < f64::EPSILON);
        assert!((signal_weight(f64::NAN) - 1.0).abs() < f64::EPSILON);
        assert!((signal_weight(5.0) - 2.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn dedupe_and_cap_are_deterministic() {
        let config = TtseConfig {
            max_tips_per_observation: 2,
            ..TtseConfig::default()
        };
        let inducer = TipInducer::new(config);
        let first = trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            vec![failed_step("network unreachable").with_tool_name("http")],
        );
        let duplicate = trajectory(
            "t-2",
            TrajectoryOutcome::Failure,
            vec![failed_step("network unreachable").with_tool_name("http")],
        );
        let other = trajectory(
            "t-3",
            TrajectoryOutcome::Failure,
            vec![failed_step("disk full")],
        );
        let capped = trajectory(
            "t-4",
            TrajectoryOutcome::Failure,
            vec![failed_step("out of memory")],
        );

        let tips = inducer
            .induce(&[first, duplicate, other, capped], &[])
            .await
            .expect("induction");

        assert_eq!(
            tips.iter().map(|tip| tip.rule.clone()).collect::<Vec<_>>(),
            vec![
                "Check http before retrying: network unreachable".to_owned(),
                "Avoid repeating: disk full".to_owned(),
            ],
            "the duplicate is dropped and the cap keeps input order"
        );

        let empty = inducer.induce(&[], &[]).await.expect("empty induction");
        assert_eq!(empty, Vec::<TipRecord>::new());
    }

    #[tokio::test]
    async fn model_path_returns_parsed_tips() {
        let answer = concat!(
            "{\"tips\": [",
            "{\"rule\": \"Check the offline registry before building\", \"category\": \"environment\"},",
            "{\"rule\": \"Prefer rg over grep\", \"category\": \"tool\"}",
            "]}"
        );
        let model = Arc::new(ScriptedModel::text(answer)) as Arc<dyn Model>;
        let inducer =
            TipInducer::new(TtseConfig::default()).with_model(model, ModelRequestConfig::default());
        assert!(inducer.has_model());

        let tips = inducer.induce(&[], &[]).await.expect("induction");

        assert!(
            tips.is_empty(),
            "empty input short-circuits before the model"
        );

        let trajectory = trajectory("t-1", TrajectoryOutcome::Failure, vec![failed_step("boom")]);
        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("induction");

        assert_eq!(tips.len(), 2);
        assert!(tips.iter().all(|tip| tip.source == "model"));
        assert_eq!(tips[0].rule, "Check the offline registry before building");
        assert_eq!(tips[0].category, TipCategory::Environment);
        assert_eq!(tips[1].category, TipCategory::Tool);
    }

    #[tokio::test]
    async fn model_junk_falls_back_to_the_deterministic_path() {
        let model = Arc::new(ScriptedModel::text("Default mock response")) as Arc<dyn Model>;
        let inducer =
            TipInducer::new(TtseConfig::default()).with_model(model, ModelRequestConfig::default());
        let trajectory = trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            vec![failed_step("the build is broken")],
        );

        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("a junk answer never errors");

        assert_eq!(
            tips.iter().map(|tip| tip.rule.clone()).collect::<Vec<_>>(),
            vec!["Avoid repeating: the build is broken".to_owned()],
            "prose is not a tip list, so the offline path decides"
        );
        assert!(tips.iter().all(|tip| tip.source == "trajectory"));
    }

    #[tokio::test]
    async fn model_errors_fall_back_to_the_deterministic_path() {
        use openjiuwen_test_support::ScriptedTurn;

        let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")])) as Arc<dyn Model>;
        let inducer =
            TipInducer::new(TtseConfig::default()).with_model(model, ModelRequestConfig::default());
        let trajectory = trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            vec![failed_step("the build is broken")],
        );

        let tips = inducer
            .induce(std::slice::from_ref(&trajectory), &[])
            .await
            .expect("a model error never propagates");

        assert_eq!(tips.len(), 1);
        assert_eq!(tips[0].rule, "Avoid repeating: the build is broken");
    }

    #[test]
    fn prompt_is_deterministic_and_bounded() {
        let trajectories: Vec<Trajectory> = (0..20)
            .map(|index| {
                trajectory(
                    &format!("t-{index}"),
                    TrajectoryOutcome::Failure,
                    vec![failed_step(&format!("failure {index}"))],
                )
            })
            .collect();
        let signals: Vec<EvolutionSignal> = (0..20)
            .map(|index| signal(&format!("signal {index}"), 1.0, 1.0))
            .collect();

        let first = build_prompt(&trajectories, &signals);
        let second = build_prompt(&trajectories, &signals);
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);

        let user = first[1].text().into_owned();
        assert!(user.contains("trajectory t-0 (failure)"));
        assert!(user.contains("trajectory t-11 (failure)"));
        assert!(!user.contains("trajectory t-12 (failure)"), "12 digests");
        assert!(user.contains("- [execution_failure] signal 11"));
        assert!(!user.contains("signal 12"), "12 signal lines");
        assert!(first[0].text().contains("Reply with JSON"));
    }

    #[test]
    fn success_detector_gating_matches_the_outcome_helpers() {
        assert!(is_failed(&trajectory(
            "t-1",
            TrajectoryOutcome::Failure,
            Vec::new()
        )));
        assert!(!is_failed(&trajectory(
            "t-1",
            TrajectoryOutcome::Success,
            vec![failed_step("ignored")]
        )));
        assert!(is_failed(&trajectory(
            "t-1",
            TrajectoryOutcome::Unknown,
            vec![failed_step("counted")]
        )));
        assert!(!is_failed(&trajectory(
            "t-1",
            TrajectoryOutcome::Unknown,
            Vec::new()
        )));
        assert_eq!(normalize_excerpt("  a   b\n c ", 80), "a b c");
        assert_eq!(normalize_excerpt("abc", 2), "a…");
        assert_eq!(normalize_excerpt("   ", 80), "");
        assert!(!looks_like_tip_list("  "));
        assert!(looks_like_tip_list("[{\"rule\": \"x\"}]"));
        assert!(looks_like_tip_list("- a tip"));
        assert!(looks_like_tip_list("TIP: keep the cache warm"));
        assert!(!looks_like_tip_list("Default mock response"));
    }
}
