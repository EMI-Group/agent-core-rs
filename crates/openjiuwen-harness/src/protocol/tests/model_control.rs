//! Model discovery and selection of the native harness.

use super::*;

#[tokio::test]
async fn list_models_reports_the_configured_model() {
    let fixture = fixture();
    let models = fixture
        .harness
        .list_models()
        .await
        .expect("model discovery works");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "scripted-model");
    assert_eq!(models[0].display_name, "scripted-model");
    assert!(models[0].is_default);
    assert_eq!(
        models[0].efforts,
        vec!["low".to_owned(), "medium".to_owned(), "high".to_owned()]
    );

    let bare = fixture_with(ScriptedModel::text("hi"), DeepAgentConfig::default());
    let models = bare.harness.list_models().await.unwrap();
    assert_eq!(
        models.len(),
        0,
        "an agent without a model name reports no catalog entry"
    );
}

#[tokio::test]
async fn set_model_switches_the_model_and_effort() {
    let fixture = fixture();

    let error = fixture
        .harness
        .set_model(ModelSelection {
            model: Some("other-model".to_owned()),
            effort: None,
        })
        .await
        .expect_err("an unstarted harness rejects model selection");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "native harness has not been started");

    fixture.start().await;
    fixture
        .harness
        .set_model(ModelSelection {
            model: Some("other-model".to_owned()),
            effort: Some("high".to_owned()),
        })
        .await
        .expect("model selection is supported");

    let config = fixture.agent.config();
    assert_eq!(config.model_name.as_deref(), Some("other-model"));
    let request = config.model_config.expect("a request config is written");
    assert_eq!(request.model_name, "other-model");
    assert_eq!(request.temperature, Some(1.0));
    assert_eq!(
        fixture.harness.list_models().await.unwrap()[0].model_id,
        "other-model"
    );

    let error = fixture
        .harness
        .set_model(ModelSelection {
            model: None,
            effort: Some("extreme".to_owned()),
        })
        .await
        .expect_err("unknown efforts are rejected");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "unsupported reasoning effort 'extreme'");
}

#[tokio::test]
async fn set_model_keeps_the_current_model_when_only_the_effort_changes() {
    let fixture = fixture();
    fixture.start().await;

    fixture
        .harness
        .set_model(ModelSelection {
            model: None,
            effort: Some("low".to_owned()),
        })
        .await
        .expect("an effort-only selection is supported");

    let config = fixture.agent.config();
    assert_eq!(config.model_name.as_deref(), Some("scripted-model"));
    let request = config.model_config.expect("a request config is written");
    assert_eq!(request.model_name, "scripted-model");
    assert_eq!(request.temperature, Some(0.2));
}
