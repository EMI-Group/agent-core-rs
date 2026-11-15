//! [`ChatAgent`]: the tuned agent and the adapters that bind tuning to it.
//!
//! Ports the role of Python's `openjiuwen/dev_tools/tune/chat_agent/chat_agent.py`
//! (`ChatAgent` wraps one `LLMCall` and answers `{"output": content,
//! "tool_calls": …}`) and of `create_chat_agent`, without Python's runtime: the
//! execution step is a caller-supplied [`ChatRunner`], and the agent itself is
//! the thing a tuning run is bound to.
//!
//! The conversation is assembled here:
//!
//! - **system** — the candidate prompt under optimisation. When
//!   [`freeze_system_prompt`](ChatConfig::freeze_system_prompt) is set and the
//!   configured [`system_prompt`](ChatConfig::system_prompt) is non-empty, the
//!   frozen prompt comes first and the candidate is appended as its own
//!   [`CANDIDATE_BLOCK_HEADING`] block, so a caller can pin the agent's persona
//!   while still tuning an instruction.
//! - **user** — [`user_prompt`](ChatConfig::user_prompt) with its `{{name}}`
//!   placeholders substituted from the case inputs, or the rendered case inputs
//!   ([`render_case_inputs`]) when that prompt is empty. A frozen user prompt is
//!   sent verbatim, with no substitution, so it must be non-empty
//!   ([`ChatConfig::validate`] enforces that).
//!
//! The reply text becomes the answer map `{"output": text}` — the same single
//! entry Python's `dict(output=…)` produces — which [`LabelScorer`] already
//! scores against a case label when the answer has no matching key.
//!
//! The adapters are thin on purpose: [`ChatAgent::evaluator`] composes the agent
//! into a [`RunnerEvaluator`], and [`ChatAgent::optimize`] /
//! [`ChatAgent::train`] attach that evaluator to the corresponding optimizer, so
//! a caller who wants a model-driven rewrite or a different scorer composes
//! [`InstructionOptimizer::with_model`] / [`Trainer::with_config`] around
//! `agent.evaluator()` instead.
//!
//! [`LabelScorer`]: crate::tune::LabelScorer

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde_json::Value;

use crate::tune::evaluator::{render_case_inputs, PromptRunner, RunnerEvaluator};
use crate::tune::{Case, InstructionOptimizer, OptimizationResult, Trainer};

use super::config::ChatConfig;
use super::runner::{ChatRunner, ModelChatRunner};

/// The heading that separates a frozen system prompt from the candidate prompt.
pub const CANDIDATE_BLOCK_HEADING: &str = "## Candidate instructions";

/// The single answer-map key a reply is reported under.
const OUTPUT_KEY: &str = "output";

/// A chat agent: a validated [`ChatConfig`] plus the runner that answers it.
///
/// Python: `openjiuwen.dev_tools.tune.chat_agent.chat_agent.ChatAgent`. The
/// agent is cheap to clone — the configuration is cloned and the runner shared —
/// so an evaluator can own one while the caller keeps another.
#[derive(Clone)]
pub struct ChatAgent {
    /// The validated settings every conversation is assembled from.
    config: ChatConfig,
    /// The execution seam that answers a rendered conversation.
    runner: Arc<dyn ChatRunner>,
}

impl fmt::Debug for ChatAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatAgent")
            .field("agent_id", &self.config.agent_id)
            .field("model_name", &self.config.model_name)
            .field("frozen_system_prompt", &self.config.freeze_system_prompt)
            .field("frozen_user_prompt", &self.config.freeze_user_prompt)
            .finish_non_exhaustive()
    }
}

impl ChatAgent {
    /// Creates an agent that answers through `model`.
    ///
    /// The model-backed runner sends the configuration's model name and sampling
    /// settings, so `config.model_name` — not the model's own name — is what a
    /// request carries.
    ///
    /// # Errors
    ///
    /// Whatever [`ChatConfig::validate`] reports.
    pub fn new(config: ChatConfig, model: Arc<dyn Model>) -> Result<Self> {
        let runner = ModelChatRunner::from_config(&config, model);
        Self::with_runner(config, Arc::new(runner))
    }

    /// Creates an agent that answers through a caller-supplied [`ChatRunner`].
    ///
    /// This is the seam a test (or a host runtime) binds: the agent still owns
    /// the conversation assembly, and only the execution step is replaced.
    ///
    /// # Errors
    ///
    /// Whatever [`ChatConfig::validate`] reports.
    pub fn with_runner(config: ChatConfig, runner: Arc<dyn ChatRunner>) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, runner })
    }

    /// The validated settings.
    #[must_use]
    pub fn config(&self) -> &ChatConfig {
        &self.config
    }

    /// The execution seam.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn ChatRunner> {
        &self.runner
    }

    /// Runs one candidate prompt over one case and returns the raw reply text.
    ///
    /// The system and user messages are assembled exactly as
    /// [`PromptRunner::run`] assembles them, which makes this the entry point for
    /// a caller that wants the text rather than the answer map.
    ///
    /// # Errors
    ///
    /// The runner's own error.
    pub async fn reply(&self, prompt: &str, case: &Case) -> Result<String> {
        let system = render_system(&self.config, prompt);
        let user = render_user(&self.config, case);
        self.runner.reply(&system, &user).await
    }

    /// An evaluator over a clone of this agent, scored by the default
    /// [`LabelScorer`](crate::tune::LabelScorer).
    ///
    /// The clone shares the runner (an `Arc`), so the evaluator and the caller
    /// observe one conversation history — a [`ScriptedChatRunner`]'s recorded
    /// calls, for example, stay visible to the caller.
    ///
    /// [`ScriptedChatRunner`]: super::ScriptedChatRunner
    #[must_use]
    pub fn evaluator(&self) -> RunnerEvaluator {
        RunnerEvaluator::new(Arc::new(self.clone()))
    }

    /// Tunes `prompt` against `cases` with an [`InstructionOptimizer`] bound to
    /// this agent's evaluator.
    ///
    /// The optimizer has no model attached, so candidate prompts come from the
    /// deterministic mutation bank; a caller who wants model rewrites composes
    /// `InstructionOptimizer::with_model(…)` around
    /// [`evaluator`](Self::evaluator) instead.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] when the prompt is blank
    /// or the case list is empty; otherwise the runner's own error.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        InstructionOptimizer::new()
            .with_evaluator(Arc::new(self.evaluator()))
            .optimize(prompt, cases)
            .await
    }

    /// Trains `prompt` on `train` and selects on `val`, through a [`Trainer`]
    /// bound to this agent's evaluator.
    ///
    /// As with [`optimize`](Self::optimize) no model is attached, so each epoch's
    /// candidate comes from the deterministic mutation bank and the winner is
    /// always measured on `val`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] when the prompt is blank
    /// or the train split is empty; otherwise the runner's own error.
    pub async fn train(
        &self,
        prompt: &str,
        train: &[Case],
        val: &[Case],
    ) -> Result<OptimizationResult> {
        Trainer::new()
            .with_evaluator(Arc::new(self.evaluator()))
            .train(prompt, train, val)
            .await
    }
}

#[async_trait]
impl PromptRunner for ChatAgent {
    async fn run(&self, prompt: &str, case: &Case) -> Result<BTreeMap<String, Value>> {
        let text = self.reply(prompt, case).await?;
        let mut answer = BTreeMap::new();
        answer.insert(OUTPUT_KEY.to_owned(), Value::String(text));
        Ok(answer)
    }
}

/// Renders the system message for one candidate prompt.
///
/// The candidate prompt is the system message — it is what tuning rewrites —
/// unless a non-empty frozen system prompt precedes it as its own block.
fn render_system(config: &ChatConfig, candidate: &str) -> String {
    let frozen = config.system_prompt.trim();
    if config.freeze_system_prompt && !frozen.is_empty() {
        return format!("{frozen}\n\n{CANDIDATE_BLOCK_HEADING}\n\n{candidate}");
    }
    candidate.to_owned()
}

/// Renders the user message for one case.
fn render_user(config: &ChatConfig, case: &Case) -> String {
    if config.freeze_user_prompt {
        return config.user_prompt.clone();
    }
    if config.user_prompt.trim().is_empty() {
        return render_case_inputs(case);
    }
    substitute(&config.user_prompt, case)
}

/// Substitutes every `{{name}}` of `template` with the case input `name`.
///
/// A single left-to-right pass, so a substituted value is never itself
/// substituted. A placeholder no input supplies — and a stray `{{` or `}}` —
/// stays verbatim, which keeps the template visible to the model instead of
/// silently dropping it.
fn substitute(template: &str, case: &Case) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find("{{") {
        rendered.push_str(&rest[..open]);
        let after_open = &rest[open + 2..];
        let Some(close) = after_open.find("}}") else {
            rendered.push_str(&rest[open..]);
            return rendered;
        };
        let name = &after_open[..close];
        if let Some(value) = case.inputs.get(name) {
            rendered.push_str(&render_input_value(value));
        } else {
            rendered.push_str("{{");
            rendered.push_str(name);
            rendered.push_str("}}");
        }
        rest = &after_open[close + 2..];
    }
    rendered.push_str(rest);
    rendered
}

/// Renders a case input value the way [`render_case_inputs`] does.
fn render_input_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::json;

    use super::*;
    use crate::tune::evaluator::CaseEvaluator;
    use crate::tune::ScriptedChatRunner;

    /// A case with one string input `question` and one label `answer`.
    fn case(id: &str, question: &str, answer: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(question.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(answer.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// An agent over a runner that echoes nothing back but records the turn.
    fn recording_agent(config: ChatConfig) -> (ChatAgent, Arc<ScriptedChatRunner>) {
        let runner = Arc::new(ScriptedChatRunner::new().with_fallback("reply"));
        let agent =
            ChatAgent::with_runner(config, Arc::clone(&runner) as Arc<dyn ChatRunner>).unwrap();
        (agent, runner)
    }

    fn config() -> ChatConfig {
        ChatConfig::new("agent-1", "gpt-4o-mini")
    }

    #[test]
    fn construction_validates_the_config() {
        let model = Arc::new(ScriptedModel::text("hi"));
        let error = ChatAgent::new(ChatConfig::new("", "gpt-4o-mini"), model).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let runner = Arc::new(ScriptedChatRunner::new());
        let error =
            ChatAgent::with_runner(ChatConfig::new("agent-1", " "), runner.clone()).unwrap_err();
        assert!(error.to_string().contains("model_name"), "{error}");

        let error = ChatAgent::with_runner(config().with_frozen_user_prompt(true), runner.clone())
            .unwrap_err();
        assert!(error.to_string().contains("freeze_user_prompt"), "{error}");

        let error = ChatAgent::with_runner(config().with_temperature(3.0), runner).unwrap_err();
        assert!(error.to_string().contains("temperature"), "{error}");
    }

    #[tokio::test]
    async fn readers_expose_the_config_and_runner() {
        let (agent, runner) = recording_agent(config().with_version("0.2.0"));
        assert_eq!(agent.config().agent_id, "agent-1");
        assert_eq!(agent.config().version, "0.2.0");
        agent.reply("prompt", &case("c1", "q", "a")).await.unwrap();
        assert_eq!(
            runner.call_count(),
            1,
            "the reader reports the shared runner"
        );
    }

    #[tokio::test]
    async fn the_candidate_prompt_becomes_the_system_message() {
        let (agent, runner) = recording_agent(config().with_system_prompt("unused baseline"));
        let reply = agent
            .reply("Answer briefly.", &case("c1", "2 + 2?", "4"))
            .await
            .unwrap();
        assert_eq!(reply, "reply");

        let call = runner.last_call().unwrap();
        assert_eq!(call.system, "Answer briefly.");
        assert_eq!(call.user, "[question]: 2 + 2?");
    }

    #[tokio::test]
    async fn a_frozen_system_prompt_precedes_the_candidate() {
        let (agent, runner) = recording_agent(
            config()
                .with_system_prompt("You are a careful assistant.")
                .with_frozen_system_prompt(true),
        );
        agent
            .reply("Answer in one word.", &case("c1", "q", "a"))
            .await
            .unwrap();

        let call = runner.last_call().unwrap();
        assert_eq!(
            call.system,
            "You are a careful assistant.\n\n## Candidate instructions\n\nAnswer in one word."
        );
        assert_eq!(call.user, "[question]: q");
    }

    #[test]
    fn the_frozen_system_prompt_is_a_labelled_block() {
        let frozen = config()
            .with_system_prompt("  You are a careful assistant.  ")
            .with_frozen_system_prompt(true);
        let system = render_system(&frozen, "Answer in one word.");
        assert_eq!(
            system,
            "You are a careful assistant.\n\n## Candidate instructions\n\nAnswer in one word."
        );
        assert!(system.contains(CANDIDATE_BLOCK_HEADING));
    }

    #[test]
    fn a_blank_frozen_system_prompt_falls_back_to_the_candidate() {
        let blank_frozen = config()
            .with_system_prompt("   ")
            .with_frozen_system_prompt(true);
        assert_eq!(render_system(&blank_frozen, "candidate"), "candidate");

        let unfrozen = config().with_system_prompt("baseline");
        assert_eq!(render_system(&unfrozen, "candidate"), "candidate");
    }

    #[test]
    fn the_user_prompt_template_substitutes_case_inputs() {
        let templated =
            config().with_user_prompt("Question: {{question}} ({{count}} of {{missing}})");
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
        inputs.insert("count".to_owned(), json!(4));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String("4".to_owned()));
        let example = Case::new(inputs, label).unwrap();

        assert_eq!(
            render_user(&templated, &example),
            "Question: 2 + 2? (4 of {{missing}})"
        );
    }

    #[test]
    fn substitution_does_not_cascade_into_inserted_values() {
        let templated = config().with_user_prompt("{{a}}|{{b}}");
        let mut inputs = BTreeMap::new();
        inputs.insert("a".to_owned(), Value::String("{{b}}".to_owned()));
        inputs.insert("b".to_owned(), Value::String("second".to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String("x".to_owned()));
        let example = Case::new(inputs, label).unwrap();

        assert_eq!(render_user(&templated, &example), "{{b}}|second");
    }

    #[test]
    fn malformed_braces_stay_verbatim() {
        let open = config().with_user_prompt("open {{question");
        let example = case("c1", "unused", "unused");
        assert_eq!(render_user(&open, &example), "open {{question");

        let close = config().with_user_prompt("close }} only");
        assert_eq!(render_user(&close, &example), "close }} only");
    }

    #[test]
    fn an_empty_user_prompt_renders_the_case_inputs() {
        let case = case("c1", "2 + 2?", "4");
        assert_eq!(render_user(&config(), &case), render_case_inputs(&case));
        assert_eq!(
            render_user(&config().with_user_prompt("  "), &case),
            "[question]: 2 + 2?"
        );
    }

    #[test]
    fn a_frozen_user_prompt_is_sent_verbatim() {
        let config = config()
            .with_user_prompt("Literal {{question}}")
            .with_frozen_user_prompt(true);
        assert_eq!(
            render_user(&config, &case("c1", "2 + 2?", "4")),
            "Literal {{question}}"
        );
    }

    #[tokio::test]
    async fn the_reply_becomes_the_output_answer_entry() {
        let runner = Arc::new(ScriptedChatRunner::new().with_fallback("Paris"));
        let agent = ChatAgent::with_runner(config(), runner).unwrap();
        let answer = agent
            .run(
                "Answer the question.",
                &case("c1", "capital of France?", "Paris"),
            )
            .await
            .unwrap();
        assert_eq!(answer.len(), 1);
        assert_eq!(
            answer.get("output"),
            Some(&Value::String("Paris".to_owned()))
        );
    }

    #[tokio::test]
    async fn the_evaluator_scores_through_the_agents_runner() {
        let runner = Arc::new(
            ScriptedChatRunner::new()
                .with_reply("capital of France", "Paris")
                .with_reply("capital of Italy", "Rome"),
        );
        let agent = ChatAgent::with_runner(config(), runner.clone()).unwrap();
        let evaluator = agent.evaluator();

        let cases = [
            case("c1", "capital of France?", "Paris"),
            case("c2", "capital of Italy?", "Rome"),
        ];
        let (mean, evaluated) = evaluator
            .score_cases("Answer the question.", &cases)
            .await
            .unwrap();
        assert_eq!(evaluated.len(), 2);
        assert!((mean - 1.0).abs() < 1e-9, "both answers match: {mean}");
        assert_eq!(runner.call_count(), 2);
        assert!(
            evaluated
                .iter()
                .all(|case| case.reason.contains("matched exactly")),
            "{:?}",
            evaluated
                .iter()
                .map(|case| case.reason.as_str())
                .collect::<Vec<_>>()
        );

        let missed = evaluator
            .score_cases(
                "Answer the question.",
                &[case("c3", "capital of Spain?", "Madrid")],
            )
            .await;
        assert_eq!(
            missed.unwrap_err().kind(),
            openjiuwen_common::error::ErrorKind::NotFound
        );
    }

    #[tokio::test]
    async fn a_model_backed_agent_optimizes_end_to_end() {
        let model = ScriptedModel::builder()
            .model_name("ignored-by-the-request")
            .fallback(ScriptedTurn::text("4"))
            .respond_text("4")
            .build();
        let config = ChatConfig::new("agent-1", "gpt-4o-mini").with_temperature(0.1);
        let agent = ChatAgent::new(config.clone(), Arc::new(model.clone())).unwrap();
        let cases = [case("c1", "2 + 2?", "4"), case("c2", "3 + 1?", "4")];

        let result = agent
            .optimize("Answer the question.", &cases)
            .await
            .unwrap();

        assert!(
            (result.score - 1.0).abs() < 1e-9,
            "both labels match: {}",
            result.score
        );
        assert_eq!(result.iterations, 0, "an already-perfect seed stops early");
        assert_eq!(result.best_prompt, "Answer the question.");

        let calls = model.calls();
        assert!(
            !calls.is_empty(),
            "the optimizer evaluated through the agent"
        );
        for call in &calls {
            assert_eq!(call.messages.len(), 2);
            assert!(call.messages[0].role.is_system());
            assert_eq!(call.messages[0].text(), "Answer the question.");
            assert!(call.messages[1].role.is_user());
            assert_eq!(call.request.model_name, "gpt-4o-mini");
            assert_eq!(call.request.temperature, Some(0.1));
        }
    }

    #[tokio::test]
    async fn a_model_backed_agent_runs_search_rounds_when_the_seed_is_weak() {
        let model = ScriptedModel::builder()
            .fallback(ScriptedTurn::text("wrong"))
            .respond_text("wrong")
            .build();
        let agent = ChatAgent::new(config(), Arc::new(model.clone())).unwrap();
        let cases = [case("c1", "2 + 2?", "4")];

        let result = agent
            .optimize("Answer the question.", &cases)
            .await
            .unwrap();

        assert!(
            result.score.abs() < 1e-9,
            "no candidate matches: {}",
            result.score
        );
        assert!(result.iterations >= 1, "the search ran at least one round");
        assert!(model.call_count() > 1, "candidates were evaluated");
        let systems: Vec<String> = model
            .calls()
            .iter()
            .map(|call| call.messages[0].text().into_owned())
            .collect();
        assert!(
            systems
                .iter()
                .any(|system| system.contains("Think step by step")),
            "the mutation bank's candidates ran through the agent: {systems:?}"
        );
        assert!(
            systems
                .iter()
                .all(|system| system.starts_with("Answer the question.")),
            "every candidate keeps the seed as its base: {systems:?}"
        );
    }

    #[tokio::test]
    async fn a_model_backed_agent_trains_end_to_end() {
        let model = ScriptedModel::builder().respond_text("Paris").build();
        let agent = ChatAgent::new(config(), Arc::new(model.clone())).unwrap();
        let train = [case("t1", "capital of France?", "Paris")];
        let val = [case("v1", "capital of France?", "Paris")];

        let result = agent
            .train("Answer the question.", &train, &val)
            .await
            .unwrap();

        assert!(
            (result.score - 1.0).abs() < 1e-9,
            "the validation score: {}",
            result.score
        );
        assert_eq!(
            result.iterations, 0,
            "the baseline already passes early stop"
        );
        assert_eq!(result.best_prompt, "Answer the question.");
        assert_eq!(
            model.call_count(),
            1,
            "only the validation baseline was scored"
        );
    }

    #[tokio::test]
    async fn training_validates_its_inputs_through_the_trainer() {
        let agent = ChatAgent::new(config(), Arc::new(ScriptedModel::text("x"))).unwrap();
        let error = agent
            .train("   ", &[case("t1", "q", "a")], &[])
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = agent.train("Answer.", &[], &[]).await.unwrap_err();
        assert!(error.to_string().contains("train split"), "{error}");
    }

    #[tokio::test]
    async fn a_scripted_agent_optimizes_through_its_recorded_runner() {
        let runner = Arc::new(ScriptedChatRunner::new().with_fallback("4"));
        let agent =
            ChatAgent::with_runner(config().with_user_prompt("{{question}}"), runner.clone())
                .unwrap();
        let cases = [case("c1", "2 + 2?", "4")];

        let result = agent
            .optimize("Answer the question.", &cases)
            .await
            .unwrap();

        assert!((result.score - 1.0).abs() < 1e-9);
        let call = runner.last_call().unwrap();
        assert_eq!(call.user, "2 + 2?");
        assert_eq!(call.system, "Answer the question.");
    }

    #[tokio::test]
    async fn declared_tools_reach_the_optimizer_rewrite_prompt() {
        use openjiuwen_foundation::tool::ToolInfo;

        use crate::tune::InstructionOptimizer;

        let cases = [case("c1", "2 + 2?", "4").with_tools(vec![ToolInfo::new(
            "lookup",
            "Look something up",
            json!({"type": "object"}),
        )])];
        assert!(cases[0].has_tools());

        let chat = Arc::new(ScriptedChatRunner::new().with_fallback("wrong"));
        let agent = ChatAgent::with_runner(config(), chat).unwrap();
        let rewrite_model = ScriptedModel::builder()
            .respond_text("no block here")
            .build();

        let result = InstructionOptimizer::new()
            .with_evaluator(Arc::new(agent.evaluator()))
            .with_model(Arc::new(rewrite_model.clone()))
            .optimize("Answer the question.", &cases)
            .await
            .unwrap();
        assert!(result.score.abs() < 1e-9, "no answer matches the label");

        let rewrite_calls = rewrite_model.calls();
        assert!(
            !rewrite_calls.is_empty(),
            "the optimizer asked for a rewrite"
        );
        let prompt = rewrite_calls[0].last_user_text().unwrap();
        assert!(prompt.contains("<tools_description>"), "{prompt}");
        assert!(prompt.contains("lookup"), "{prompt}");
    }

    #[test]
    fn the_agent_is_debug() {
        let (agent, _) = recording_agent(config().with_frozen_system_prompt(true));
        let rendered = format!("{agent:?}");
        assert!(rendered.contains("ChatAgent"), "{rendered}");
        assert!(rendered.contains("agent-1"), "{rendered}");
        assert!(rendered.contains("gpt-4o-mini"), "{rendered}");
    }
}
