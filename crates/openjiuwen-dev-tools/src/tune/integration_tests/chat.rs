//! Scenario 9: the offline chat-agent adapter.
//!
//! A scripted chat runner behind `ChatAgent` evaluates a case list and tunes a
//! prompt through the same runner, and the recorded conversation shows the
//! assembled system and user messages.

use std::sync::Arc;

use super::support::*;
use crate::tune::*;

#[tokio::test]
async fn chat_agent_evaluates_and_optimizes_offline() {
    let config = create_chat_config("tuner", "1.0.0", "tunes a chat agent", "scripted-model")
        .expect("a valid chat configuration");
    config.validate().expect("the builder already validated it");
    assert_eq!(config.agent_id, "tuner");
    assert!(create_chat_config("", "1.0.0", "tunes a chat agent", "scripted-model").is_err());
    assert!(ChatConfig::new("tuner", "scripted-model")
        .with_temperature(3.0)
        .validate()
        .is_err());

    // The scripted runner answers `Paris` to everything, so one case matches its
    // label exactly and the other does not.
    let runner = Arc::new(ScriptedChatRunner::new().with_fallback("Paris"));
    let handle: Arc<dyn ChatRunner> = runner.clone();
    let agent = ChatAgent::with_runner(config, handle).expect("a valid agent");
    let cases = vec![
        case("c1", "capital of France", "Paris"),
        case("c2", "capital of Italy", "Rome"),
    ];

    let (mean, evaluated) = agent
        .evaluator()
        .score_cases("Answer the question.", &cases)
        .await
        .expect("a scripted evaluation");
    assert_eq!(evaluated.len(), 2);
    assert_close(mean, 0.5);
    assert_close(evaluated[0].score, 1.0);
    assert_close(evaluated[1].score, 0.0);

    // The conversation the adapter assembled: the candidate prompt is the system
    // message and the rendered case inputs are the user message.
    let calls = runner.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].system, "Answer the question.");
    assert_eq!(calls[0].user, render_case_inputs(&cases[0]));
    assert_eq!(calls[0].user, "[question]: capital of France");
    assert_eq!(calls[1].user, "[question]: capital of Italy");

    // Tuning through the same runner keeps the reply fixed, so the score cannot
    // improve past the seed's.
    let result = agent
        .optimize("Answer the question.", &cases)
        .await
        .expect("a valid optimization run");
    assert_close(result.score, 0.5);
    assert!(result.iterations >= 1, "the seed leaves room to improve");
    assert_ne!(result.best_prompt, "", "the winner is a real prompt");
    assert!(
        runner.call_count() > calls.len(),
        "the optimizer scored candidates through the agent's runner"
    );
    assert!(runner
        .calls()
        .iter()
        .skip(calls.len())
        .all(|call| !call.system.is_empty()));
}
