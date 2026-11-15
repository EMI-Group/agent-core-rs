//! Integration suite for the `tune` adapter layers.
//!
//! The optimizers produce prompt strings; these layers are what binds that output
//! back to a real agent. The suite exercises them through the public
//! `openjiuwen_dev_tools::tune` surface only:
//!
//! - **Binding** — [`TextualParameter`] rendering and validation, the
//!   [`ParameterBinding`]/[`ParameterUpdate`] placeholder contract, the
//!   multi-parameter [`BoundAgent`], and the [`ToolBinding`] join between a case's
//!   declared tools and the runnable [`ToolStubs`].
//! - **Adoption** — the pure [`AdoptPolicy`]/[`AdoptConfig`] gates, the async
//!   [`AdoptService`] that measures, decides, and archives, [`AdoptMemory`], and
//!   the deterministic gradient artifacts ([`render_gradient`],
//!   [`select_bad_cases`]).
//! - **Chat agent** — [`ChatConfig`] validation and the offline [`ChatAgent`] over
//!   a [`ScriptedChatRunner`]: the conversation it assembles, the recorded calls,
//!   and the `optimize` adapter.
//! - **Telemetry** — [`RunTelemetry`] carried by an [`Execution`] driving the
//!   latency/token/cost reward components, the [`CompositeReward`] combination,
//!   and the evaluator seam ([`PromptRunner::run_traced`]) that surfaces a
//!   runner's telemetry into those same components.
//!
//! Everything runs offline and deterministically: no clock (timestamps are
//! literals), no environment reads, no network, no sleeps, and no randomness —
//! chat replies are scripted and telemetry is caller-supplied.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_dev_tools::tune::{
    create_chat_config, describe_gradient_context, render_gradient, select_bad_cases,
    select_good_cases, AdoptConfig, AdoptDecision, AdoptMemory, AdoptPolicy, AdoptReason,
    AdoptService, BoundAgent, Case, CaseEvaluator, CaseResult, ChatAgent, ChatConfig, ChatRunner,
    CompositeReward, CorrectnessReward, CostReward, DeterministicGradientPolicy, EvaluatedCase,
    Evaluation, Execution, GradientPolicy, LatencyReward, LexicalDriftJudge, ParameterBinding,
    ParameterUpdate, PromptCandidate, PromptRecord, PromptRunner, PromptTaskSpec, RewardBreakdown,
    RewardComponent, RunOutcome, RunTelemetry, RunnerEvaluator, ScriptedChatRunner,
    TextualParameter, TokenUsageReward, ToolBinding, ToolBindingReport, ToolCallRecord, ToolStub,
    ToolStubs, ToolTrace, CANDIDATE_BLOCK_HEADING, DEFAULT_BAD_CASES_SAMPLE_NUM,
};
use openjiuwen_foundation::tool::ToolInfo;
use serde_json::json;

/// The timestamp every adoption decision stamps its records with (no clock read).
const NOW: f64 = 1_700_000_000.0;

/// The objective the adoption scenarios reason about.
const OBJECTIVE: &str = "Answer the question accurately.";

/// The variable-price objective the composition scenario scores against.
const ARITHMETIC: &str = "answer arithmetic questions accurately";

/// A case whose inputs and label carry `question`/`answer`.
fn case(id: &str, question: &str, answer: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!(question))]),
        BTreeMap::from([("answer".to_owned(), json!(answer))]),
    )
    .expect("a case with non-empty inputs and label")
    .with_case_id(id)
}

/// An evaluated case with `id`, `score`, and an evaluator reason.
fn evaluated(id: &str, score: f64, reason: &str) -> EvaluatedCase {
    EvaluatedCase::new(
        case(id, "2 + 2?", "4"),
        BTreeMap::from([("answer".to_owned(), json!("5"))]),
        score,
        reason,
    )
    .expect("a score inside 0.0..=1.0")
}

/// Asserts two floats agree to the crate's comparison tolerance.
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

/// A name → text candidate map, the shape the binding layer takes.
fn prompts(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(name, prompt)| ((*name).to_owned(), (*prompt).to_owned()))
        .collect()
}

/// The telemetry the telemetry scenarios measure.
fn measured() -> RunTelemetry {
    RunTelemetry::default()
        .with_latency_ms(1000.0)
        .with_prompt_tokens(600)
        .with_completion_tokens(400)
        .with_cost(0.5)
}

// 1. Textual parameters, bindings, and the multi-parameter agent adapter.

#[allow(clippy::too_many_lines)]
#[test]
fn textual_parameters_render_validate_and_apply_across_a_bound_agent() {
    // A parameter is its template plus the provenance an optimizer staged.
    let parameter =
        TextualParameter::new("system", "Question: {{question}}\nAnswer in {{language}}.")
            .with_description("the system prompt")
            .with_gradient("instruction", "be terse")
            .with_candidates(
                "instruction",
                vec!["be terse".to_owned(), "be kind".to_owned()],
            );
    assert_eq!(parameter.name, "system");
    assert_eq!(parameter.description, "the system prompt");
    assert_eq!(parameter.gradient("instruction"), Some("be terse"));
    assert_eq!(parameter.gradient("missing"), None);
    assert_eq!(
        parameter.candidates("instruction"),
        vec!["be terse".to_owned(), "be kind".to_owned()]
    );
    assert_eq!(
        parameter.placeholder_names(),
        vec!["language".to_owned(), "question".to_owned()],
        "placeholders are sorted and deduplicated"
    );
    assert!(parameter.validate().is_ok());

    // Rendering substitutes the values it has and leaves an unknown placeholder
    // verbatim rather than dropping the text the template declares.
    let filled = BTreeMap::from([
        ("question".to_owned(), json!("2 + 2?")),
        ("language".to_owned(), json!("French")),
    ]);
    assert_eq!(
        parameter
            .render(&filled)
            .expect("every placeholder is filled"),
        "Question: 2 + 2?\nAnswer in French."
    );

    // A blank name and a blank template are validation errors.
    let error = TextualParameter::new("  ", "Answer.")
        .validate()
        .expect_err("a blank name is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("name"), "{error}");
    let error = TextualParameter::new("system", " \n ")
        .validate()
        .expect_err("a blank template is invalid");
    assert!(error.to_string().contains("template"), "{error}");

    // A binding owns the placeholder contract: a rewrite that drops or invents a
    // placeholder is a validation error, because nothing downstream could fill it.
    let binding = ParameterBinding::new(parameter.clone());
    assert_eq!(binding.parameter().template, parameter.template);
    assert!(binding.validate(&parameter.template).is_ok());
    let error = binding
        .validate("Question: {{question}}")
        .expect_err("a dropped placeholder is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("dropped declared placeholders"),
        "{message}"
    );
    assert!(message.contains("{{language}}"), "{message}");
    let message = binding
        .validate("Question: {{question}} in {{language}} via {{tool}}")
        .expect_err("an invented placeholder is rejected")
        .to_string();
    assert!(
        message.contains("introduced undeclared placeholders"),
        "{message}"
    );
    assert!(message.contains("{{tool}}"), "{message}");

    // An accepted rewrite is reported as a change; nothing is mutated.
    let rewrite = "Question: {{question}}\nAnswer briefly in {{language}}.";
    let update = binding
        .apply(rewrite)
        .expect("the rewrite keeps every placeholder");
    assert_eq!(update.parameter_name, "system");
    assert_eq!(update.previous_prompt, parameter.template);
    assert_eq!(update.new_prompt, rewrite);
    assert!(update.changed());
    assert_eq!(
        update.preserved_placeholders,
        vec!["language".to_owned(), "question".to_owned()]
    );
    assert_eq!(update.gain, None, "no optimizer result was applied");
    assert_eq!(binding.parameter().template, parameter.template);

    // The agent-side adapter validates a whole candidate map in name order, and
    // ignores candidates for parameters it does not bind.
    let agent = BoundAgent::new(vec![
        ParameterBinding::from_template("user", "Question: {{question}}"),
        binding,
    ])
    .expect("two distinctly named parameters");
    assert_eq!(agent.parameters().len(), 2);
    assert_eq!(agent.parameters()[0].parameter().name, "system");
    assert!(agent.parameter("user").is_some());
    assert!(agent.parameter("System").is_none(), "names match exactly");

    let candidates = prompts(&[
        ("system", "Answer in {{language}}: {{question}}"),
        ("user", "Ask: {{question}}"),
        ("subagent", "anything {{unknown}}"),
    ]);
    let updates = agent.apply(&candidates).expect("both rewrites are valid");
    assert_eq!(
        updates
            .iter()
            .map(|update| update.parameter_name.as_str())
            .collect::<Vec<&str>>(),
        vec!["system", "user"],
        "updates follow the agent's name order"
    );
    assert!(updates.iter().all(ParameterUpdate::changed));
    assert_eq!(
        updates,
        agent.validate_all(&candidates).expect("same check")
    );

    // A declared parameter with no candidate is reported by name.
    let error = agent
        .validate_all(&prompts(&[("user", "Ask: {{question}}")]))
        .expect_err("every declared parameter needs a candidate");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("no candidate prompt for bound parameter 'system'"),
        "{error}"
    );

    let encoded = serde_json::to_value(&updates[1]).expect("an update serializes");
    assert_eq!(encoded["parameter_name"], json!("user"));
    assert_eq!(encoded["preserved_placeholders"], json!(["question"]));
}

// 2. Tool binding: a case's declarations joined to runnable stubs.

#[test]
fn tool_binding_joins_case_declarations_to_the_stub_table() {
    // The stub table is the runnable half: named fixtures, looked up by name.
    let mut stubs = ToolStubs::new();
    assert!(stubs.is_empty());
    stubs.insert(ToolStub::constant("search", json!({"hits": 2})).unwrap());
    stubs.insert(ToolStub::constant("calculator", json!({"value": 4})).unwrap());
    assert_eq!(stubs.len(), 2);
    assert!(stubs.contains("search"));
    assert!(stubs.get("weather").is_none());

    // A case declares descriptors only; `declare_only` is that list.
    let declaring = case("c1", "capital of France", "Paris").with_tools(vec![
        ToolInfo::new("search", "Search the web", json!({"type": "object"})),
        ToolInfo::new("weather", "Look up the weather", json!({"type": "object"})),
        ToolInfo::new(
            "calculator",
            "Compute an expression",
            json!({"type": "object"}),
        ),
        ToolInfo::new("search", "Search again", json!({"type": "object"})),
    ]);
    assert_eq!(
        ToolBinding::declare_only(&declaring),
        vec![
            "search".to_owned(),
            "weather".to_owned(),
            "calculator".to_owned()
        ],
        "declarations are deduplicated in declaration order"
    );

    // `bind_case` partitions the declarations and reports the gap.
    let binding = ToolBinding::new(stubs);
    let report: ToolBindingReport = binding
        .bind_case(&declaring)
        .expect("a partial binding is reported, not rejected");
    assert_eq!(
        report.declared,
        vec![
            "search".to_owned(),
            "weather".to_owned(),
            "calculator".to_owned()
        ]
    );
    assert_eq!(
        report.bound,
        vec!["search".to_owned(), "calculator".to_owned()]
    );
    assert_eq!(report.missing, vec!["weather".to_owned()]);
    assert!(!report.is_complete());
    assert_eq!(report.missing_count(), 1);

    // A fully bound case is complete; a tool-free case is trivially complete.
    let bound = case("c2", "2 + 2?", "4").with_tools(vec![ToolInfo::new(
        "calculator",
        "Compute",
        json!({}),
    )]);
    let complete = binding.bind_case(&bound).expect("the tool is bound");
    assert!(complete.is_complete());
    assert_eq!(complete.missing_count(), 0);
    let vacuous = ToolBinding::new(ToolStubs::new())
        .bind_case(&case("c3", "2 + 2?", "4"))
        .expect("a tool-free case binds vacuously");
    assert_eq!(vacuous, ToolBindingReport::default());

    // `validate_case` is the strict entry point and names the first gap.
    binding.validate_case(&bound).expect("every tool is bound");
    let error = binding
        .validate_case(&declaring)
        .expect_err("an unbound declaration fails validation");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("case 'c1'"), "{message}");
    assert!(
        message.contains("unbound tool 'weather'"),
        "the first gap: {message}"
    );
    assert!(message.contains("1 of 3"), "{message}");

    // A bound stub is callable through the binding; the trace counts the calls.
    assert_eq!(
        binding
            .tool_stubs()
            .call("search", &json!({"query": "openjiuwen"}))
            .unwrap(),
        json!({"hits": 2})
    );
    let trace = ToolTrace::new([
        ToolCallRecord::success("search", json!({}), json!({"hits": 2})),
        ToolCallRecord::success("search", json!({}), json!({"hits": 1})),
        ToolCallRecord::failure("calculator", json!({}), "division by zero"),
    ]);
    assert_eq!(
        binding.usage(&trace),
        BTreeMap::from([
            ("calculator".to_owned(), 1usize),
            ("search".to_owned(), 2usize)
        ])
    );
    assert!(binding.usage(&ToolTrace::default()).is_empty());
}

// 3. The adoption service: measure, gate, decide, and archive.

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn adopt_service_gates_candidates_and_archives_only_adoptions() {
    // The service exposes the policy, its thresholds, and its memory.
    let service = AdoptService::new(AdoptPolicy::new());
    let config = *service.config();
    assert_eq!(config, AdoptConfig::default());
    assert_close(config.min_gain, 0.01);
    assert_close(config.max_drift, 0.25);
    assert_close(config.min_correctness, 0.5);
    assert_eq!(config.min_history, 1);
    assert!(config.require_improvement && config.validate().is_ok());
    assert!(service.characteristics().is_none());
    assert_eq!(service.memory().adopted(), Vec::<PromptRecord>::new());

    let task = PromptTaskSpec::new(OBJECTIVE).with_base_prompt("Answer the question.");
    let candidate = PromptCandidate::new("Answer the question accurately and briefly.", "seed")
        .with_id("cand-1");

    // Nothing comparable yet: the first candidate becomes the baseline.
    let first = service
        .decide(&task, &candidate, 0.6, &[1.0, 0.5], &[], NOW)
        .await
        .expect("the deterministic path cannot fail");
    assert!(first.adopted());
    assert_eq!(first.decision.reason(), &AdoptReason::FirstCandidate);
    assert_close(first.correctness, 0.75);
    assert_close(first.drift, 0.0);
    assert!(first.convergence.is_none());
    let archived = first
        .record
        .clone()
        .expect("an adopted candidate is archived");
    assert_eq!(archived.record_id, "adopt-cand-1");
    assert_eq!(archived.prompt, candidate.prompt);
    assert_eq!(archived.objective, OBJECTIVE);
    assert_eq!(archived.task_characteristics, OBJECTIVE);
    assert_close(archived.reward, 0.6);
    assert!(archived.applied, "the archived prompt is marked applied");
    assert_eq!(archived.applied_at, Some(NOW), "`now` is the caller's");
    assert!(archived.baseline_reward.is_none(), "no earlier baseline");
    assert_close(archived.metrics["correctness"].as_f64().unwrap(), 0.75);
    assert_eq!(service.memory().adopted().len(), 1);
    assert_eq!(
        service.memory().previous_best(OBJECTIVE),
        Some(0.6),
        "the run's own adoption is the next baseline"
    );

    // A reward that does not beat the best history by `min_gain` is rejected and
    // archived nowhere.
    let worse = PromptCandidate::new("Answer briefly.", "round two").with_id("cand-2");
    let rejected = service
        .decide(&task, &worse, 0.5, &[1.0, 0.0], &[0.6], NOW)
        .await
        .expect("the deterministic path cannot fail");
    assert!(!rejected.adopted());
    assert!(matches!(
        rejected.decision.reason(),
        AdoptReason::NoImprovement { .. }
    ));
    assert_close(rejected.decision.gain(), -0.1);
    assert!(
        rejected.record.is_none(),
        "a regression is archived nowhere"
    );
    assert_eq!(service.memory().adopted().len(), 1, "the run is unchanged");

    // Correctness below the floor is rejected before any comparison.
    let below = service
        .decide(&task, &candidate, 1.0, &[0.2, 0.2], &[0.6], NOW)
        .await
        .expect("the deterministic path cannot fail");
    match below.decision.reason() {
        AdoptReason::InsufficientCorrectness { score } => assert_close(*score, 0.2),
        other => panic!("expected the correctness gate, got {other:?}"),
    }
    assert!(below.record.is_none());

    // A history shallower than `min_history` is not trusted.
    let picky =
        AdoptService::new(AdoptPolicy::new().with_config(AdoptConfig::new().with_min_history(2)));
    let shallow = picky
        .decide(&task, &candidate, 0.9, &[1.0, 1.0], &[0.5], NOW)
        .await
        .expect("the deterministic path cannot fail");
    assert_eq!(
        shallow.decision.reason(),
        &AdoptReason::InsufficientHistory { rounds: 1 },
        "a one-deep history is not comparable"
    );

    // Drift above the allowance wins on nothing, however high the reward.
    let driftless =
        AdoptService::new(AdoptPolicy::new()).with_drift_judge(Arc::new(LexicalDriftJudge::new()));
    let repurposed =
        PromptCandidate::new("Write a haiku about oceans.", "unrelated").with_id("cand-drift");
    let drifted = driftless
        .decide(&task, &repurposed, 1.0, &[1.0, 1.0], &[0.6], NOW)
        .await
        .expect("the deterministic path cannot fail");
    assert_role(&drifted, AdoptReason::Drift { deviation: 1.0 });
    assert_close(drifted.drift, 1.0);
    assert!(drifted.record.is_none());

    // The characteristics override replaces the task's own signature.
    let labelled = AdoptService::new(AdoptPolicy::new()).with_characteristics("tuned by hand");
    assert_eq!(labelled.characteristics(), Some("tuned by hand"));

    // `AdoptMemory` keeps both lists: adopted records are applied, rejected ones
    // stay in the run and never reach a persistent store.
    let memory = AdoptMemory::new();
    let applied = memory.record_adopted(PromptRecord::new("r1", "prompt", 0.8, OBJECTIVE), NOW);
    assert!(applied.applied);
    assert_eq!(applied.applied_at, Some(NOW));
    let refused = memory.record_rejected(PromptRecord::new("r2", "prompt", 0.1, OBJECTIVE), NOW);
    assert!(!refused.applied && refused.applied_at.is_none());
    assert_eq!(refused.created_at, NOW);
    assert_eq!(memory.adopted().len(), 1);
    assert_eq!(memory.rejected().len(), 1);
    assert_eq!(memory.previous_best(OBJECTIVE), Some(0.8));

    // The decision and its reason keep the Python wire shape.
    let encoded = serde_json::to_value(AdoptDecision::new(
        true,
        AdoptReason::Improved { gain: 0.3 },
    ))
    .expect("a decision serializes");
    assert_eq!(encoded["adopted"], json!(true));
    assert_eq!(encoded["reason"], json!({"kind": "improved", "gain": 0.3}));
}

/// Asserts `outcome` was rejected for exactly `reason` (gain compared loosely).
fn assert_role(outcome: &openjiuwen_dev_tools::tune::AdoptOutcome, reason: AdoptReason) {
    assert!(!outcome.adopted(), "expected a rejection for {reason:?}");
    assert_eq!(outcome.decision.reason(), &reason);
}

// 4. The deterministic gradient artifact the adopt layer renders.

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn adopt_gradient_selects_cases_and_renders_the_artifact() {
    let cases = [
        evaluated("d", 0.0, "wrong result"),
        evaluated("a", 1.0, "correct"),
        evaluated("c", 0.5, "partly right"),
        evaluated("b", 0.0, ""),
    ];

    // Only exact scores are evidence: 0.0 is a failure, 1.0 a success.
    let bad = select_bad_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM);
    assert_eq!(
        bad.iter()
            .map(|case| case.case_id().to_owned())
            .collect::<Vec<String>>(),
        vec!["b".to_owned(), "d".to_owned()],
        "failed cases in case-id order"
    );
    assert_eq!(
        select_good_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM)
            .iter()
            .map(|case| case.case_id().to_owned())
            .collect::<Vec<String>>(),
        vec!["a".to_owned()]
    );
    assert_eq!(
        select_bad_cases(&cases, 0).len(),
        0,
        "a zero limit selects none"
    );
    assert_eq!(select_bad_cases(&cases, 1).len(), 1, "the limit truncates");

    // The rendered artifact names each failing case and its evidence.
    let rendered = describe_gradient_context(&bad);
    let lines: Vec<&str> = rendered.lines().collect();
    assert_eq!(lines[0], "case 0: b");
    assert_eq!(lines[1], "  score: 0.000");
    assert_eq!(lines[2], r#"  input: {"question":"2 + 2?"}"#);
    assert_eq!(lines[5], "  feedback: (no feedback recorded)");
    assert_eq!(lines[6], "case 1: d");
    assert_eq!(lines[11], "  feedback: wrong result");
    assert_eq!(describe_gradient_context(&[]), "no failing cases");

    // The default policy renders the context under its documented header.
    let task = PromptTaskSpec::new(OBJECTIVE);
    let candidate = PromptCandidate::new("Answer {{question}}.", "seed").with_id("cand-1");
    let gradient = DeterministicGradientPolicy::new();
    assert_eq!(DeterministicGradientPolicy, gradient, "a unit value");
    let text = gradient
        .gradient(&task, &candidate, &bad)
        .await
        .expect("the deterministic policy never fails");
    assert_eq!(text, render_gradient(&task, &candidate, &bad));
    assert!(text.starts_with(&format!(
        "objective: {OBJECTIVE}\ncandidate: cand-1\ngradient:\n"
    )));
    assert_eq!(
        render_gradient(&task, &PromptCandidate::default(), &[]),
        format!(
            "objective: {OBJECTIVE}\ncandidate: (unnamed candidate)\ngradient:\nno failing cases"
        )
    );
}

// 5. The offline chat agent.

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn chat_agent_renders_the_conversation_and_tunes_a_scripted_runner() {
    // The factory validates, and every rejection names the offending field.
    let config = create_chat_config("tuner", "1.0.0", "tunes a chat agent", "scripted-model")
        .expect("a valid chat configuration");
    assert_eq!(config.agent_id, "tuner");
    assert_eq!(config.version, "1.0.0");
    assert_eq!(config.model_name, "scripted-model");
    config.validate().expect("the factory already validated it");
    let error = create_chat_config("", "1.0.0", "", "scripted-model")
        .expect_err("a blank agent id is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("agent_id"), "{error}");
    assert!(create_chat_config("tuner", "1.0.0", "", " ")
        .expect_err("a blank model name is rejected")
        .to_string()
        .contains("model_name"));
    ChatConfig::new("tuner", "m")
        .with_frozen_system_prompt(true)
        .validate()
        .expect("a frozen but blank system prompt is accepted");

    // `reply` assembles system and user exactly as the runner sees them: the
    // frozen persona first, the candidate under its own heading, and the case
    // inputs substituted into the user prompt template.
    let config = ChatConfig::new("tuner", "scripted-model")
        .with_system_prompt("You are a careful assistant.")
        .with_frozen_system_prompt(true)
        .with_user_prompt("Q: {{question}}");
    let runner = Arc::new(ScriptedChatRunner::new().with_ordered_reply("queued answer"));
    let handle: Arc<dyn ChatRunner> = runner.clone();
    let agent = ChatAgent::with_runner(config, handle).expect("a valid agent");
    assert_eq!(
        agent
            .reply("Answer in one word.", &case("c1", "2 + 2?", "4"))
            .await
            .expect("the queued reply answers the first call"),
        "queued answer"
    );
    assert_eq!(runner.remaining(), 0, "the ordered reply was consumed");
    let call = runner.last_call().expect("the call was recorded");
    assert_eq!(
        call.system,
        format!("You are a careful assistant.\n\n{CANDIDATE_BLOCK_HEADING}\n\nAnswer in one word.")
    );
    assert_eq!(call.user, "Q: 2 + 2?");

    // An unresolved call is an error, and is still recorded.
    let error = agent
        .reply("Answer in one word.", &case("c1", "2 + 2?", "4"))
        .await
        .expect_err("the runner has no reply left");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(runner.call_count(), 2, "every call is recorded");

    // Marker-keyed replies are prompt-independent, so the evaluator and the
    // optimizer score every candidate the same way.
    let scoring = Arc::new(
        ScriptedChatRunner::new()
            .with_reply("capital of France", "Paris")
            .with_reply("capital of Italy", "Milan"),
    );
    let scoring_handle: Arc<dyn ChatRunner> = scoring.clone();
    let agent = ChatAgent::with_runner(
        ChatConfig::new("tuner", "scripted-model").with_user_prompt("{{question}}"),
        scoring_handle,
    )
    .expect("a valid agent");
    let cases = [
        case("c1", "capital of France", "Paris"),
        case("c2", "capital of Italy", "Rome"),
    ];

    let (mean, evaluated) = agent
        .evaluator()
        .score_cases("Answer the question.", &cases)
        .await
        .expect("a scripted evaluation");
    assert_close(mean, 0.5);
    assert_close(evaluated[0].score, 1.0);
    assert_close(evaluated[1].score, 0.0);
    let calls = scoring.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0].system, "Answer the question.",
        "an unfrozen system prompt is the candidate prompt itself"
    );
    assert_eq!(calls[0].user, "capital of France", "the template is filled");
    assert_eq!(calls[1].user, "capital of Italy");

    // `optimize` binds the instruction search to the agent's own runner.
    let result = agent
        .optimize("Answer the question.", &cases)
        .await
        .expect("a valid optimization run");
    assert_close(result.score, 0.5);
    assert!(result.iterations >= 1, "the seed leaves room to search");
    assert_ne!(result.best_prompt.trim(), "");
    let after = scoring.calls();
    assert!(
        after.len() > calls.len(),
        "the optimizer used the agent's runner"
    );
    assert_eq!(
        after[calls.len()].system,
        "Answer the question.",
        "the seed is evaluated first, unmodified"
    );

    // The adapter validates its inputs through the optimizer.
    let error = agent
        .optimize("   ", &cases)
        .await
        .expect_err("a blank prompt");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("prompt"), "{error}");
    assert!(agent
        .optimize("Answer the question.", &[])
        .await
        .expect_err("an empty case list is rejected")
        .to_string()
        .contains("cases"));
}

// 6. Telemetry-carrying rewards, including the evaluator seam that surfaces them.

/// A runner that reports a fixed telemetry through the traced seam.
struct TelemetryRunner {
    /// What every run reports it cost.
    telemetry: RunTelemetry,
    /// The answer map every case gets.
    answer: BTreeMap<String, serde_json::Value>,
}

#[async_trait]
impl PromptRunner for TelemetryRunner {
    async fn run(
        &self,
        _prompt: &str,
        _case: &Case,
    ) -> Result<BTreeMap<String, serde_json::Value>> {
        Ok(self.answer.clone())
    }

    async fn run_traced(&self, _prompt: &str, _case: &Case) -> Result<RunOutcome> {
        Ok(RunOutcome::new(self.answer.clone()).with_telemetry(self.telemetry))
    }
}

/// An execution of `candidate` carrying `telemetry` and two graded results.
fn graded_run(candidate: PromptCandidate, telemetry: RunTelemetry) -> Execution {
    let mut execution = Execution::new(candidate);
    execution.telemetry = telemetry;
    execution.push_result(CaseResult::new("c1", "4").with_score(1.0));
    execution.push_result(CaseResult::new("c2", "5").with_score(0.8));
    execution
}

#[tokio::test]
async fn telemetry_carrying_executions_drive_the_reward_components() {
    let task = PromptTaskSpec::new(ARITHMETIC);
    let telemetry = measured();
    let candidate = PromptCandidate::new(ARITHMETIC, "seed");
    let execution = graded_run(candidate.clone(), telemetry);

    assert!(!telemetry.is_empty());
    assert_eq!(telemetry.total_tokens(), Some(1000));
    assert_close(execution.measured_latency_s(), 1.0);
    assert_eq!(execution.measured_total_tokens(), 1000);
    assert_eq!(execution.measured_cost(), Some(0.5));

    // Each component is exactly half credit at its documented half-life, and a
    // longer half-life is more forgiving.
    let latency = LatencyReward::new(1.0);
    let tokens = TokenUsageReward::new(1000.0);
    let cost = CostReward::new(0.0, 0.5);
    assert_close(latency.half_life_s(), 1.0);
    assert_close(tokens.half_life_tokens(), 1000.0);
    assert_close(cost.half_life_cost(), 0.5);

    // Measured exactly at the half-life, each component awards half credit, and a
    // longer half-life is strictly more forgiving.
    assert_close(latency.score(&execution, &task), 0.5);
    assert_close(tokens.score(&execution, &task), 0.5);
    assert_close(cost.score(&execution, &task), 0.5);
    assert!(LatencyReward::new(4.0).score(&execution, &task) > 0.5);
    assert!(TokenUsageReward::new(4000.0).score(&execution, &task) > 0.5);
    assert!(CostReward::new(0.0, 2.0).score(&execution, &task) > 0.5);

    // Cost is derived from the token count when the caller reported none, and
    // nothing reported leaves every component at its documented constant.
    let unpriced = graded_run(
        candidate.clone(),
        RunTelemetry::default()
            .with_prompt_tokens(600)
            .with_completion_tokens(400),
    );
    assert_eq!(unpriced.measured_cost(), None);
    assert_close(
        CostReward::new(1000.0, 0.5).score(&unpriced, &task),
        1.0 / 3.0,
    );
    let bare = Execution::new(candidate.clone());
    assert!(bare.telemetry.is_empty());
    assert_close(latency.score(&bare, &task), 1.0);
    assert_close(tokens.score(&bare, &task), 1.0);
    assert_close(CostReward::new(3.0, 0.5).score(&bare, &task), 1.0);

    // The composite combines the measured values by weight.
    let composite = CompositeReward::new(
        vec![
            Box::new(CorrectnessReward::new()),
            Box::new(latency),
            Box::new(tokens),
            Box::new(cost),
        ],
        BTreeMap::from([
            ("correctness".to_owned(), 1.0),
            ("latency".to_owned(), 0.1),
            ("token_usage".to_owned(), 0.1),
            ("cost".to_owned(), 0.05),
        ]),
    )
    .with_min_correctness(0.5);
    assert_eq!(
        composite.component_names(),
        vec!["correctness", "latency", "token_usage", "cost"]
    );
    assert_close(composite.min_correctness(), 0.5);
    let breakdowns = composite.evaluate(std::slice::from_ref(&execution), &task, &BTreeMap::new());
    let breakdown: &RewardBreakdown = &breakdowns[0];
    assert_close(breakdown.correctness, 0.9);
    for name in ["latency", "token_usage", "cost"] {
        assert_close(breakdown.components[name], 0.5);
    }
    assert_close(
        breakdown.score,
        (0.9 + 0.1 * 0.5 + 0.1 * 0.5 + 0.05 * 0.5) / 1.25,
    );
    assert!(!breakdown.is_gated());
    assert_eq!(breakdown.notes, Vec::<String>::new());

    // The breakdown keeps the Python wire shape.
    let encoded = serde_json::to_value(breakdown).expect("a breakdown serializes");
    assert_close(encoded["score"].as_f64().unwrap(), breakdown.score);
    assert_close(encoded["components"]["latency"].as_f64().unwrap(), 0.5);
    assert_eq!(encoded["gated"], json!(false));
    assert_eq!(encoded["notes"], json!([]));

    // The evaluator seam carries the same telemetry: a runner that measures
    // reports it through `score_runs`, and the reward components then score on
    // what the run actually reported.
    let measuring: Arc<dyn PromptRunner> = Arc::new(TelemetryRunner {
        telemetry,
        answer: BTreeMap::from([("answer".to_owned(), json!("4"))]),
    });
    let (mean, runs) = RunnerEvaluator::new(measuring)
        .score_runs("prompt", &[case("c1", "2 + 2?", "4")])
        .await
        .expect("a measured evaluation");
    assert_close(mean, 1.0);
    assert_eq!(runs[0].telemetry, telemetry, "the telemetry is surfaced");
    assert!(!runs[0].is_empty_telemetry());
    let surfaced = Execution {
        telemetry: runs[0].telemetry,
        ..Execution::new(candidate.clone())
    };
    for component_score in [
        LatencyReward::new(1.0).score(&surfaced, &task),
        TokenUsageReward::new(1000.0).score(&surfaced, &task),
        CostReward::new(0.0, 0.5).score(&surfaced, &task),
    ] {
        assert_close(component_score, 0.5);
    }
}

// 7. Composition: a telemetry-backed reward drives an adoption decision.

#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn a_telemetry_backed_composite_reward_drives_the_adoption_decision() {
    let task = PromptTaskSpec::new(ARITHMETIC);
    let fast = PromptCandidate::new("answer arithmetic questions accurately and briefly", "run")
        .with_id("cand-fast");
    let slow = PromptCandidate::new("answer arithmetic questions accurately, slowly", "run")
        .with_id("cand-slow");
    let ideal = PromptCandidate::new("answer arithmetic questions accurately and tersely", "run")
        .with_id("cand-ideal");

    // Three runs of the same graded answers, differing only in what they cost —
    // the only signal telemetry adds.
    let fast_run = graded_run(
        fast.clone(),
        RunTelemetry::default()
            .with_latency_ms(500.0)
            .with_prompt_tokens(300)
            .with_completion_tokens(200)
            .with_cost(0.01),
    );
    let slow_run = graded_run(
        slow.clone(),
        RunTelemetry::default()
            .with_latency_ms(4000.0)
            .with_prompt_tokens(2000)
            .with_completion_tokens(1000)
            .with_cost(0.2),
    );
    let ideal_run = graded_run(
        ideal.clone(),
        RunTelemetry::default()
            .with_latency_ms(100.0)
            .with_prompt_tokens(60)
            .with_completion_tokens(40)
            .with_cost(0.0),
    );

    let reward = CompositeReward::new(
        vec![
            Box::new(CorrectnessReward::new()),
            Box::new(LatencyReward::new(1.0)),
            Box::new(TokenUsageReward::new(1000.0)),
            Box::new(CostReward::new(0.0, 0.1)),
        ],
        BTreeMap::from([
            ("correctness".to_owned(), 1.0),
            ("latency".to_owned(), 0.1),
            ("token_usage".to_owned(), 0.1),
            ("cost".to_owned(), 0.05),
        ]),
    )
    .with_min_correctness(0.5)
    .with_drift_penalty(0.5);

    // The same correctness, a cheaper run: the reward comes out strictly higher.
    let batch = [fast_run.clone(), slow_run.clone(), ideal_run.clone()];
    let breakdowns = reward.evaluate(&batch, &task, &BTreeMap::new());
    let fast_reward = breakdowns[0].score;
    let ideal_reward = breakdowns[2].score;
    assert_close(breakdowns[0].correctness, 0.9);
    assert_close(breakdowns[1].correctness, 0.9);
    assert_close(breakdowns[1].components["latency"], 0.2);
    assert_close(breakdowns[2].components["latency"], 1.0 / 1.1);
    assert!(
        fast_reward > breakdowns[1].score,
        "the fast run outscores the slow one"
    );
    assert!(ideal_reward > fast_reward, "the cheapest run wins");

    // The adoption service reasons over exactly those composite scores.
    let evaluations = [
        Evaluation {
            execution: fast_run,
            reward: breakdowns[0].clone(),
        },
        Evaluation {
            execution: slow_run,
            reward: breakdowns[1].clone(),
        },
        Evaluation {
            execution: ideal_run,
            reward: breakdowns[2].clone(),
        },
    ];
    let graded = [1.0, 0.8];
    let service = AdoptService::new(AdoptPolicy::new());

    // The first composite score becomes the baseline.
    let first = service
        .decide(
            &task,
            evaluations[0].candidate(),
            evaluations[0].score(),
            &graded,
            &[],
            NOW,
        )
        .await
        .expect("a deterministic decision");
    assert_eq!(first.decision.reason(), &AdoptReason::FirstCandidate);
    assert_close(first.record.as_ref().expect("archived").reward, fast_reward);

    // The same answers at four times the latency are a regression: rejected.
    let worse = service
        .decide(
            &task,
            evaluations[1].candidate(),
            evaluations[1].score(),
            &graded,
            std::slice::from_ref(&fast_reward),
            NOW,
        )
        .await
        .expect("a deterministic decision");
    assert!(matches!(
        worse.decision.reason(),
        AdoptReason::NoImprovement { .. }
    ));
    assert!(worse.record.is_none());

    // The ideal run beats the baseline by more than `min_gain`, so it is adopted
    // and archived against the earlier composite score.
    let improved = service
        .decide(
            &task,
            evaluations[2].candidate(),
            evaluations[2].score(),
            &graded,
            std::slice::from_ref(&fast_reward),
            NOW,
        )
        .await
        .expect("a deterministic decision");
    assert!(improved.adopted());
    match improved.decision.reason() {
        AdoptReason::Improved { gain } => assert_close(*gain, ideal_reward - fast_reward),
        other => panic!("expected an improvement, got {other:?}"),
    }
    let record = improved.record.as_ref().expect("an adopted prompt");
    assert_eq!(record.prompt, ideal.prompt);
    assert_close(record.reward, ideal_reward);
    assert_close(record.baseline_reward.expect("a baseline"), fast_reward);
    assert_close(record.gain(), ideal_reward - fast_reward);
    assert_eq!(
        service.memory().adopted().len(),
        2,
        "two adoptions recorded"
    );
}
