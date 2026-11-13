//! Unit tests for the `intent_detection` component and its router.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unnecessary_literal_bound
)]
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::{Session, WorkflowSession};
use serde_json::json;

use super::*;

fn config() -> IntentDetectionCompConfig {
    IntentDetectionCompConfig::default()
        .with_category_names(vec!["weather".to_owned(), "travel".to_owned()])
}

fn component(config: IntentDetectionCompConfig) -> IntentDetectionComponent {
    IntentDetectionComponent::new(config).expect("the config is valid")
}

fn context() -> ComponentContext {
    let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("intent-tests"));
    ComponentContext::new(session, "intent", "wf")
}

#[test]
fn categories_are_prefixed_per_language() {
    let zh = component(config());
    assert_eq!(zh.default_class(), "分类0");
    assert_eq!(
        zh.category_list().to_vec(),
        vec!["分类0".to_owned(), "分类1".to_owned(), "分类2".to_owned()]
    );
    assert_eq!(
        zh.categories().to_vec(),
        vec![
            "默认意图".to_owned(),
            "weather".to_owned(),
            "travel".to_owned()
        ]
    );

    let en = component(config().with_accept_language("en"));
    assert_eq!(en.default_class(), "Category0");
    assert_eq!(
        en.category_list().to_vec(),
        vec![
            "Category0".to_owned(),
            "Category1".to_owned(),
            "Category2".to_owned()
        ]
    );
    assert_eq!(en.categories()[0], "Default intent");
}

#[test]
fn no_categories_leaves_only_the_default_class() {
    let component = component(IntentDetectionCompConfig::default());
    assert_eq!(component.category_list(), ["分类0"]);
    assert_eq!(component.categories(), ["默认意图"]);
}

#[test]
fn an_unsupported_language_is_code_101055() {
    let error = IntentDetectionComponent::new(config().with_accept_language("fr")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().contains("[101055]"), "{error}");
    assert!(error.message().contains("'fr'"), "{error}");
}

#[test]
fn a_blank_category_name_is_code_101053() {
    let error = IntentDetectionComponent::new(
        config().with_category_names(vec!["weather".to_owned(), "  ".to_owned()]),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().contains("[101053]"), "{error}");
}

#[test]
fn a_duplicate_category_name_is_code_101054() {
    let error = IntentDetectionComponent::new(
        config().with_category_names(vec!["weather".to_owned(), "weather".to_owned()]),
    )
    .unwrap_err();
    assert!(error.message().contains("[101054]"), "{error}");
    assert!(error.message().contains("weather"), "{error}");
}

#[test]
fn a_category_colliding_with_the_default_name_is_code_101060() {
    let error =
        IntentDetectionComponent::new(config().with_category_names(vec!["默认意图".to_owned()]))
            .unwrap_err();
    assert!(error.message().contains("[101060]"), "{error}");
}

#[test]
fn an_unknown_keyword_category_is_code_101056() {
    let keywords = BTreeMap::from([("ghost".to_owned(), vec!["x".to_owned()])]);
    let error = IntentDetectionComponent::new(config().with_keywords(keywords)).unwrap_err();
    assert!(error.message().contains("[101056]"), "{error}");
    assert!(error.message().contains("ghost"), "{error}");
}

#[test]
fn a_blank_keyword_is_code_101057() {
    let keywords = BTreeMap::from([("weather".to_owned(), vec![" ".to_owned()])]);
    let error = IntentDetectionComponent::new(config().with_keywords(keywords)).unwrap_err();
    assert!(error.message().contains("[101057]"), "{error}");
}

#[test]
fn keyword_keys_may_use_the_default_display_name() {
    let keywords = BTreeMap::from([("默认意图".to_owned(), vec!["hello".to_owned()])]);
    let component = component(config().with_keywords(keywords));
    assert_eq!(
        component.classify_by_keywords("hello there"),
        (0, "keyword matched: hello".to_owned())
    );
}

#[test]
fn a_blank_example_is_code_101058() {
    let error = IntentDetectionComponent::new(
        config().with_example_content(vec!["ok".to_owned(), String::new()]),
    )
    .unwrap_err();
    assert!(error.message().contains("[101058]"), "{error}");
}

#[test]
fn a_whitespace_only_prompt_is_code_101059() {
    let error = IntentDetectionComponent::new(config().with_user_prompt("   ")).unwrap_err();
    assert!(error.message().contains("[101059]"), "{error}");
}

#[test]
fn history_without_room_for_a_message_is_code_101061() {
    let error = IntentDetectionComponent::new(
        config()
            .with_enable_history(true)
            .with_chat_history_max_turn(0),
    )
    .unwrap_err();
    assert!(error.message().contains("[101061]"), "{error}");
}

#[test]
fn an_english_reply_selects_the_configured_class() {
    let component = component(config().with_accept_language("en"));
    let (label, reason) = component.post_process(r#"{"class": "Category2", "reason": "travel"}"#);
    assert_eq!(label, "Category2");
    assert_eq!(reason, "travel");
    assert_eq!(component.class_id(&label), 2);
}

#[test]
fn a_chinese_reply_selects_the_configured_class() {
    let component = component(config());
    let (label, reason) = component.post_process(r#"{"class": "分类1"}"#);
    assert_eq!(label, "分类1");
    assert_eq!(reason, "");
    assert_eq!(component.class_id(&label), 1);
}

#[test]
fn class_matching_is_case_insensitive() {
    let component = component(config().with_accept_language("en"));
    assert_eq!(
        component
            .post_process(r#"{"class": "category1", "reason": "w"}"#)
            .0,
        "Category1"
    );
}

#[test]
fn prose_around_the_json_is_ignored() {
    let component = component(config().with_accept_language("en"));
    let reply = r#"Sure! Here it is: {"class": "Category2"} — thanks."#;
    assert_eq!(component.post_process(reply).0, "Category2");
}

#[test]
fn quoted_and_multiline_class_values_are_sanitised() {
    let component = component(config().with_accept_language("en"));
    let multiline = r#"{"class": "Category\n2"}"#;
    assert_eq!(component.post_process(multiline).0, "Category2");
    let quoted = r#"{"class": "\"category2\""}"#;
    assert_eq!(component.post_process(quoted).0, "Category2");
}

#[test]
fn an_unknown_class_returns_the_default_with_the_validation_reason() {
    let component = component(config().with_accept_language("en"));
    let (label, reason) = component.post_process(r#"{"class": "Category9"}"#);
    assert_eq!(label, "Category0");
    assert_eq!(component.class_id(&label), 0);
    assert!(reason.contains("'Category9'"), "{reason}");
    assert!(
        reason.contains("['Category0', 'Category1', 'Category2']"),
        "{reason}"
    );
}

#[test]
fn an_unparsable_reply_returns_the_default_with_the_parse_reason() {
    let english = component(config().with_accept_language("en"));
    let (label, reason) = english.post_process("no json here");
    assert_eq!(label, "Category0");
    assert!(reason.contains("no json here"), "{reason}");
    assert!(
        reason.contains("does not conform to valid JSON"),
        "{reason}"
    );

    let chinese = component(config());
    let (label, reason) = chinese.post_process("不是 JSON");
    assert_eq!(label, "分类0");
    assert!(reason.contains("不符合有效的JSON规范"), "{reason}");
}

#[test]
fn a_missing_class_returns_the_default_with_the_missing_reason() {
    let component = component(config().with_accept_language("en"));
    let (label, reason) = component.post_process(r#"{"reason": "no class"}"#);
    assert_eq!(label, "Category0");
    assert!(reason.contains("missing the required 'class'"), "{reason}");
    assert!(reason.contains(r#"{"reason":"no class"}"#), "{reason}");
}

#[test]
fn keyword_hit_selects_the_best_category() {
    let keywords = BTreeMap::from([
        (
            "weather".to_owned(),
            vec!["weather".to_owned(), "rain".to_owned()],
        ),
        ("travel".to_owned(), vec!["flight".to_owned()]),
    ]);
    let component = component(config().with_keywords(keywords));
    assert_eq!(
        component.classify_by_keywords("Rain or weather today?"),
        (1, "keyword matched: weather, rain".to_owned())
    );
    assert_eq!(
        component.classify_by_keywords("Book a FLIGHT"),
        (2, "keyword matched: flight".to_owned())
    );
}

#[test]
fn a_keyword_tie_goes_to_the_lowest_class_id() {
    let keywords = BTreeMap::from([
        ("weather".to_owned(), vec!["rain".to_owned()]),
        ("travel".to_owned(), vec!["flight".to_owned()]),
    ]);
    let component = component(config().with_keywords(keywords));
    assert_eq!(component.classify_by_keywords("rain and flight").0, 1);
}

#[test]
fn duplicate_keywords_count_once() {
    let keywords = BTreeMap::from([(
        "weather".to_owned(),
        vec!["rain".to_owned(), "rain".to_owned()],
    )]);
    let component = component(config().with_keywords(keywords));
    assert_eq!(
        component.classify_by_keywords("rain"),
        (1, "keyword matched: rain".to_owned())
    );
}

#[test]
fn no_keyword_match_returns_the_default() {
    let keywords = BTreeMap::from([("travel".to_owned(), vec!["flight".to_owned()])]);
    let component = component(config().with_keywords(keywords));
    assert_eq!(
        component.classify_by_keywords("hello"),
        (0, NO_KEYWORD_MATCH_REASON.to_owned())
    );
}

#[test]
fn display_names_are_the_fallback_keywords() {
    let component = component(config());
    assert_eq!(
        component.classify_by_keywords("any weather today?"),
        (1, "keyword matched: weather".to_owned())
    );
}

#[test]
fn the_output_omits_an_empty_reason() {
    let component = component(config());
    assert_eq!(
        component.output(1, ""),
        json!({"classification_id": 1, "category_name": "weather"})
    );
    assert_eq!(
        component.output(0, "no keyword matched"),
        json!({
            "classification_id": 0,
            "category_name": "默认意图",
            "reason": "no keyword matched"
        })
    );
}

#[test]
fn route_is_deliberately_not_implemented() {
    let component = component(config());
    assert!(component.route(&json!({"query": "hi"})).is_none());
}

#[test]
fn identity_matches_the_python_component() {
    let component = component(config());
    assert_eq!(component.component_type(), "intent_detection");
    assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
    let card = component.card();
    assert_eq!(card.id, "intent_detection");
    assert_eq!(card.component_type, "intent_detection");
    assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
}

#[tokio::test]
async fn the_rule_path_answers_without_a_model() {
    let keywords = BTreeMap::from([("travel".to_owned(), vec!["flight".to_owned()])]);
    let component = component(config().with_keywords(keywords));
    let output = component
        .invoke(json!({"query": "book a flight"}), &context())
        .await
        .unwrap();
    assert_eq!(
        output,
        json!({
            "classification_id": 2,
            "category_name": "travel",
            "reason": "keyword matched: flight"
        })
    );
}

#[tokio::test]
async fn a_non_string_query_is_code_101050() {
    let component = component(config());
    let error = component
        .invoke(json!({"query": 7}), &context())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().contains("[101050]"), "{error}");
    assert!(error.message().contains("number"), "{error}");
}

#[tokio::test]
async fn the_rule_path_falls_back_to_the_whole_input() {
    let component = component(config());
    let output = component
        .invoke(json!({"query": null, "text": "weather please"}), &context())
        .await
        .unwrap();
    assert_eq!(output["classification_id"], json!(1));
}

#[test]
fn the_router_rejects_an_ambiguous_configuration() {
    let duplicate = IntentRouter::new(
        "intent",
        vec![(1, vec!["a".to_owned()]), (1, vec!["b".to_owned()])],
        Vec::new(),
    );
    let error = duplicate.validate().unwrap_err();
    assert!(error.message().contains("[101062]"), "{error}");

    let blank_id = IntentRouter::new(" ", Vec::new(), Vec::new());
    assert!(blank_id.validate().is_err());

    let blank_target = IntentRouter::new("intent", Vec::new(), vec![" ".to_owned()]);
    assert!(blank_target.validate().is_err());
}

#[test]
fn the_router_exposes_its_configuration() {
    let router = IntentRouter::new(
        "intent",
        vec![(2, vec!["end_b".to_owned()])],
        vec!["end_a".to_owned()],
    );
    assert!(router.validate().is_ok());
    assert_eq!(router.component_id(), "intent");
    assert_eq!(router.branches(), [(2, vec!["end_b".to_owned()])]);
    assert_eq!(router.fallback(), ["end_a"]);
    let handle = router.router();
    assert_eq!(handle.name(), "intent_detection");
}

#[test]
fn the_config_builders_are_chainable() {
    let config = IntentDetectionCompConfig::default()
        .with_model_id("m")
        .with_category_names(vec!["weather".to_owned()])
        .with_user_prompt("detect")
        .with_example_content(vec!["example".to_owned()])
        .with_enable_history(true)
        .with_chat_history_max_turn(2)
        .with_accept_language("en")
        .with_keywords(BTreeMap::from([(
            "weather".to_owned(),
            vec!["rain".to_owned()],
        )]));
    assert_eq!(config.model_id.as_deref(), Some("m"));
    assert_eq!(config.chat_history_max_turn, 2);
    assert_eq!(config.accept_language, "en");
    assert!(config.enable_history);
    // The runtime-only fields are not serialized.
    let wire = serde_json::to_value(&config).unwrap();
    assert!(wire.get("keywords").is_none(), "{wire}");
    assert!(wire.get("model").is_none(), "{wire}");
    let restored: IntentDetectionCompConfig = serde_json::from_value(wire).unwrap();
    assert!(restored.keywords.is_empty());
    assert!(restored.model.is_none());
}
