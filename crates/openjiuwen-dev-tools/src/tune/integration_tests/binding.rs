//! Scenario 8: the optimizer → binding seam.
//!
//! An optimizer winner bound back into a parameter, a two-parameter
//! `BoundAgent`, and a `ToolBinding` joining a case's declarations to runnable
//! stubs.

use std::collections::BTreeMap;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::tool::ToolInfo;
use serde_json::json;

use super::support::*;
use crate::tune::*;

#[tokio::test]
async fn optimizer_winner_binds_back_into_a_parameter() {
    let template = "You are a careful assistant.\nQuestion: {{question}}\nAnswer in {{language}}.";
    let binding = ParameterBinding::from_template("system", template);
    let cases = vec![
        case("c1", "capital of France", "Paris"),
        case("c2", "capital of Italy", "Rome"),
        case("c3", "capital of Spain", "Madrid"),
    ];

    let result = InstructionOptimizer::new()
        .optimize(template, &cases)
        .await
        .expect("a valid optimization run");
    let update = binding
        .apply_result(&result)
        .expect("the winner keeps every declared placeholder");

    assert_eq!(update.parameter_name, "system");
    assert_eq!(update.previous_prompt, template);
    assert_eq!(update.new_prompt, result.best_prompt);
    assert!(update.changed(), "the winner differs from the template");
    assert_eq!(
        update.preserved_placeholders,
        vec!["language".to_owned(), "question".to_owned()]
    );
    assert_eq!(
        binding.parameter().placeholder_names(),
        vec!["language".to_owned(), "question".to_owned()],
        "the binding still declares exactly the template's placeholders"
    );
    assert_eq!(update.gain, Some(result.score));
}

#[test]
fn bound_agent_validates_two_parameters() {
    let system =
        ParameterBinding::from_template("system", "You answer questions.\nQuestion: {{question}}");
    let user = ParameterBinding::from_template("user", "Question: {{question}}");
    let agent = BoundAgent::new(vec![user, system]).expect("two distinctly named parameters");

    let candidates = BTreeMap::from([
        (
            "system".to_owned(),
            "You answer questions.\nQuestion: {{question}}\nBe precise.".to_owned(),
        ),
        (
            "user".to_owned(),
            "Question: {{question}}\nReply briefly.".to_owned(),
        ),
    ]);
    let updates = agent
        .validate_all(&candidates)
        .expect("both candidates keep the declared placeholder");
    assert_eq!(
        updates
            .iter()
            .map(|update| update.parameter_name.as_str())
            .collect::<Vec<&str>>(),
        vec!["system", "user"],
        "updates follow the agent's name order"
    );
    assert!(updates.iter().all(ParameterUpdate::changed));
    assert!(updates
        .iter()
        .all(|update| update.preserved_placeholders == vec!["question".to_owned()]));
    assert_eq!(updates, agent.apply(&candidates).expect("the same check"));

    // A candidate that drops a declared placeholder is a validation error.
    let dropped = BTreeMap::from([
        ("system".to_owned(), "You answer questions.".to_owned()),
        (
            "user".to_owned(),
            "Question: {{question}}\nReply briefly.".to_owned(),
        ),
    ]);
    let error = agent
        .validate_all(&dropped)
        .expect_err("a dropped placeholder is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    // A missing candidate for a declared parameter is rejected as well.
    let partial = BTreeMap::from([(
        "system".to_owned(),
        "You answer questions.\nQuestion: {{question}}".to_owned(),
    )]);
    assert!(agent.validate_all(&partial).is_err());
}

#[test]
fn tool_binding_partitions_declarations_against_stubs() {
    let declarations = vec![
        ToolInfo::new("search", "Search the web", json!({"type": "object"})),
        ToolInfo::new(
            "calculator",
            "Compute an expression",
            json!({"type": "object"}),
        ),
    ];
    let declaring = case("c1", "what is two plus two", "four").with_tools(declarations);
    assert_eq!(declaring.tool_names(), vec!["search", "calculator"]);

    let stubs = ToolStubs::new()
        .with_stub(ToolStub::constant("search", json!({"hits": 2})).expect("a named stub"));
    let binding = ToolBinding::new(stubs);

    let report = binding
        .bind_case(&declaring)
        .expect("an unbound declaration is reported, not rejected");
    assert_eq!(report.declared, vec!["search", "calculator"]);
    assert_eq!(report.bound, vec!["search"]);
    assert_eq!(report.missing, vec!["calculator"]);
    assert_eq!(report.missing_count(), 1);
    assert!(!report.is_complete());
    assert_eq!(ToolBinding::declare_only(&declaring), report.declared);

    // The strict entry point names the unbound declaration.
    let error = binding
        .validate_case(&declaring)
        .expect_err("an unbound declaration fails validation");
    assert_eq!(error.kind(), ErrorKind::Validation);

    // A bound stub is callable through the binding and its use is counted.
    let value = binding
        .tool_stubs()
        .call("search", &json!({"query": "openjiuwen"}))
        .expect("the bound stub answers");
    assert_eq!(value, json!({"hits": 2}));
    let trace = ToolTrace::new([
        ToolCallRecord::success("search", json!({}), json!({"hits": 2})),
        ToolCallRecord::success("search", json!({}), json!({"hits": 2})),
    ]);
    assert_eq!(binding.usage(&trace).get("search"), Some(&2));
    assert_eq!(
        binding
            .tool_stubs()
            .call("calculator", &json!({}))
            .expect_err("an unregistered tool is unknown")
            .kind(),
        ErrorKind::NotFound
    );
}
