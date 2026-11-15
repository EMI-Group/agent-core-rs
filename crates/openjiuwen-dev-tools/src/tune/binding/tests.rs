//! End-to-end binding tests: optimizer output, rendered parameters, runnable tools.
//!
//! Every suite is offline and deterministic: the instruction optimizer runs its
//! deterministic evaluator (no model), and tool execution goes through in-process
//! stubs, so no network, clock, environment, or filesystem is involved.
use super::{BoundAgent, ParameterBinding, ToolBinding, ToolBindingReport};
use crate::tune::optimizer::placeholders;
use crate::tune::tool::{ToolCallRecord, ToolStub, ToolStubs, ToolTrace};
use crate::tune::{Case, InstructionOptimizer};
use openjiuwen_foundation::tool::ToolInfo;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

/// A case whose inputs carry the `question` a template renders.
fn case(id: &str, question: &str, answer: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!(question))]),
        BTreeMap::from([("output".to_owned(), json!(answer))]),
    )
    .unwrap()
    .with_case_id(id)
}

/// A three-case set with a shared `question`-shaped input.
fn cases() -> Vec<Case> {
    vec![
        case("c1", "capital of France", "Paris"),
        case("c2", "capital of Italy", "Rome"),
        case("c3", "capital of Spain", "Madrid"),
    ]
}

#[tokio::test]
async fn instruction_optimizer_winner_binds_back_into_the_template() {
    let template = "You are a careful assistant.\nQuestion: {{question}}\nAnswer in {{language}}.";
    let binding = ParameterBinding::from_template("system", template);

    let result = InstructionOptimizer::new()
        .optimize(template, &cases())
        .await
        .unwrap();
    let update = binding.apply_result(&result).unwrap();

    assert_eq!(update.parameter_name, "system");
    assert_eq!(update.previous_prompt, template);
    assert_eq!(
        update.new_prompt, result.best_prompt,
        "the winning prompt is reported verbatim"
    );
    assert!(
        update.changed(),
        "the deterministic bank should rewrite the seed: {:?}",
        result.best_prompt
    );
    assert_eq!(
        update.preserved_placeholders,
        vec!["language".to_owned(), "question".to_owned()],
        "every declared placeholder survives the optimizer's rewrite"
    );
    let declared: BTreeSet<String> = binding
        .parameter()
        .placeholder_names()
        .into_iter()
        .collect();
    assert_eq!(
        placeholders(&update.new_prompt),
        declared,
        "the winner introduces no placeholder the template did not declare"
    );
    assert_eq!(
        update.gain,
        Some(result.score),
        "the winning score is reported as the gain"
    );
    assert!(
        result.score > 0.0 && result.score.is_finite(),
        "the deterministic evaluator reports a bounded positive score: {}",
        result.score
    );
}

#[tokio::test]
async fn bound_agent_applies_optimizer_winners_per_parameter() {
    let system = ParameterBinding::from_template(
        "system",
        "You answer questions.\nQuestion: {{question}}\nBe accurate.",
    );
    let user = ParameterBinding::from_template("user", "Question: {{question}}");
    let agent = BoundAgent::new(vec![user, system]).unwrap();

    let mut prompts = BTreeMap::new();
    for binding in agent.parameters() {
        let template = binding.parameter().template.as_str();
        let result = InstructionOptimizer::new()
            .optimize(template, &cases())
            .await
            .unwrap();
        prompts.insert(binding.parameter().name.clone(), result.best_prompt);
    }

    let updates = agent.apply(&prompts).unwrap();
    assert_eq!(
        updates
            .iter()
            .map(|update| update.parameter_name.as_str())
            .collect::<Vec<&str>>(),
        vec!["system", "user"],
        "updates follow the agent's name order"
    );
    assert!(updates
        .iter()
        .all(|update| update.preserved_placeholders == vec!["question".to_owned()]));
    assert!(updates
        .iter()
        .all(|update| update.new_prompt == prompts[&update.parameter_name]));
    assert_eq!(updates, agent.validate_all(&prompts).unwrap());
}

#[test]
fn rendered_parameter_matches_the_bound_prompt() {
    let binding = ParameterBinding::from_template("system", "Answer {{question}} in {{language}}.");
    let update = binding
        .apply("Answer {{question}} in {{language}}, briefly.")
        .unwrap();

    let mut values = binding.prompt_values();
    values.insert("question".to_owned(), json!("2 + 2?"));
    assert_eq!(
        binding.parameter().render(&values).unwrap(),
        "Answer 2 + 2? in {{language}}.",
        "a partially filled skeleton leaves the other placeholders verbatim"
    );

    values.insert("language".to_owned(), json!("French"));
    assert_eq!(
        binding.parameter().render(&values).unwrap(),
        "Answer 2 + 2? in French."
    );
    assert_eq!(
        update.new_prompt,
        "Answer {{question}} in {{language}}, briefly."
    );
    assert_eq!(
        binding.parameter().template,
        "Answer {{question}} in {{language}}."
    );
}

#[test]
fn declared_tools_bind_and_run_through_the_tool_layer() {
    let stubs = ToolStubs::new()
        .with_stub(
            ToolStub::new("lookup", |arguments: &serde_json::Value| {
                let question = arguments["question"].clone();
                Ok(json!({"question": question, "answer": "Paris"}))
            })
            .unwrap(),
        )
        .with_stub(ToolStub::constant("clock", json!({"now": "12:00"})).unwrap());
    let binding = ToolBinding::new(stubs);

    let case = case("c1", "capital of France", "Paris").with_tools(vec![ToolInfo::new(
        "lookup",
        "Look a fact up",
        json!({"type": "object", "properties": {"question": {"type": "string"}}}),
    )]);

    let report = binding.bind_case(&case).unwrap();
    assert_eq!(
        report,
        ToolBindingReport {
            declared: vec!["lookup".to_owned()],
            bound: vec!["lookup".to_owned()],
            missing: Vec::new(),
        }
    );
    assert!(report.is_complete());
    binding.validate_case(&case).unwrap();

    let arguments = json!({"question": case.inputs["question"]});
    let result = binding.tool_stubs().call("lookup", &arguments).unwrap();
    assert_eq!(
        result,
        json!({"question": "capital of France", "answer": "Paris"})
    );

    let trace = ToolTrace::new([
        ToolCallRecord::success("lookup", arguments, result.clone()),
        ToolCallRecord::success("lookup", json!({"question": "capital of Italy"}), json!({})),
        ToolCallRecord::failure("search", json!({}), "not bound"),
    ]);
    assert_eq!(
        binding.usage(&trace),
        BTreeMap::from([("lookup".to_owned(), 2), ("search".to_owned(), 1),])
    );
    assert!(
        !binding.tool_stubs().contains("search"),
        "the trace may mention an unbound tool, but the table does not offer it"
    );
}

#[test]
fn tool_binding_reports_a_partial_declaration_without_failing() {
    let binding = ToolBinding::new(
        ToolStubs::new()
            .with_stub(ToolStub::constant("lookup", json!({"answer": "Paris"})).unwrap()),
    );
    let case = case("c1", "capital of France", "Paris").with_tools(vec![
        ToolInfo::new("lookup", "Look a fact up", json!({})),
        ToolInfo::new("weather", "Read the forecast", json!({})),
    ]);

    let report = binding.bind_case(&case).unwrap();
    assert_eq!(
        report.declared,
        vec!["lookup".to_owned(), "weather".to_owned()]
    );
    assert_eq!(report.bound, vec!["lookup".to_owned()]);
    assert_eq!(report.missing, vec!["weather".to_owned()]);

    let error = binding.validate_case(&case).unwrap_err();
    assert!(
        error.to_string().contains("unbound tool 'weather'"),
        "{error}"
    );
}
