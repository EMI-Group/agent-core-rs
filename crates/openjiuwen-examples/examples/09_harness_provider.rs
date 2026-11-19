//! `09_harness_provider` — the harness provider registry: `create_harness("native", …)`,
//! then start, send, observe events, and stop.
//!
//! It teaches the host-facing seam of the harness tier: a provider is looked up by
//! name (`provider_descriptors` lists what this build can serve), a JSON create
//! mapping is validated by the provider, and the resulting `HarnessProtocol` is
//! driven through one turn — `start` with a `HarnessContext`, `send_auto` with a
//! `HarnessInput`, and a single-consumer cursor that ends at the turn's terminal
//! `TurnLifecycle` event.
//!
//! There is **no Python counterpart**: `create_harness` has zero hits under the
//! Python reference `examples/`. The native provider resolves its model lazily
//! from the create mapping, which is how the example stays offline: `api_base`
//! points at a loopback `MockLlmServer`, so the real OpenAI-compatible HTTP path
//! runs against `127.0.0.1`. The `api_key` is a dummy value that is never printed.
//! `enable_task_loop = false` keeps the turn to a single model call, so one
//! scripted text answer is enough to reach the terminal event.

#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod common;

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use openjiuwen::harness_protocol::{
    HarnessCard, HarnessContext, HarnessEvent, HarnessEventCursor, HarnessEventPayload,
    HarnessInput, HarnessProtocol, JsonObject, JsonValue, TurnResult, TurnStatus,
};
use openjiuwen::harness_providers::{create_harness, provider_descriptors};

/// Registry name the example creates its harness through.
const PROVIDER: &str = "native";
/// Model name written into the create mapping.
const MODEL_NAME: &str = "gpt-4o-mini";
/// Provider written into the create mapping (the model client family).
const MODEL_PROVIDER: &str = "openai";
/// Dummy credential: accepted by the mapping, never printed.
const API_KEY: &str = "test-key";
/// System prompt the create mapping carries into the `DeepAgent`.
const SYSTEM_PROMPT: &str = "You are a concise example agent.";
/// The text the host sends as one turn.
const PROMPT: &str = "Say hello to the host.";
/// The answer the loopback server is scripted with.
const SCRIPTED_ANSWER: &str = "hello from the loopback provider";
/// Bound on one lifecycle call (`start`, `send`, `stop`).
const CALL_TIMEOUT: Duration = Duration::from_secs(20);
/// Bound on reading one event from the cursor.
const EVENT_TIMEOUT: Duration = Duration::from_secs(15);
/// Upper bound on the events one turn may produce.
///
/// A drain loop needs both a per-event timeout and a cap: a script mismatch can
/// otherwise deliver events forever without ever reaching a terminal one.
const MAX_EVENTS: usize = 64;

/// What one observed turn produced.
struct Observation {
    /// Number of events read from the cursor.
    events: usize,
    /// Text accumulated from the `answer` output channel.
    answer: String,
    /// Terminal turn result, when the turn reported one.
    result: Option<TurnResult>,
}

/// Prints the providers this build can name, with their capabilities.
fn print_registry() {
    common::banner::step("provider registry");
    for descriptor in provider_descriptors() {
        common::banner::note(&format!(
            "{} (card {} {}) implemented={} configures_models={} builtin={} requires={}",
            descriptor.name,
            descriptor.card_name,
            descriptor.implementation_version,
            descriptor.implemented,
            descriptor.configures_models,
            descriptor.builtin,
            descriptor.requires.as_deref().unwrap_or("nothing"),
        ));
    }
}

/// Creates the unstarted native harness over the loopback model endpoint.
fn create_native_harness(api_base: &str, workspace: &Path) -> Result<Box<dyn HarnessProtocol>> {
    // Only keys the provider declares: unknown keys are rejected outright.
    let mut config: JsonObject = JsonObject::new();
    config.insert("model".to_owned(), JsonValue::String(MODEL_NAME.to_owned()));
    config.insert(
        "provider".to_owned(),
        JsonValue::String(MODEL_PROVIDER.to_owned()),
    );
    config.insert("api_key".to_owned(), JsonValue::String(API_KEY.to_owned()));
    config.insert(
        "api_base".to_owned(),
        JsonValue::String(api_base.to_owned()),
    );
    config.insert(
        "workspace".to_owned(),
        JsonValue::String(workspace.display().to_string()),
    );
    config.insert("enable_task_loop".to_owned(), JsonValue::Bool(false));
    config.insert("max_iterations".to_owned(), JsonValue::from(4_u64));
    config.insert(
        "system_prompt".to_owned(),
        JsonValue::String(SYSTEM_PROMPT.to_owned()),
    );

    // The narrative shows the mapping with the credential redacted.
    let entries: Vec<String> = config
        .iter()
        .map(|(key, value)| {
            if key == "api_key" {
                format!("{key}=<redacted>")
            } else {
                format!("{key}={value}")
            }
        })
        .collect();
    common::banner::step(&format!("create mapping for {PROVIDER}"));
    common::banner::note(&entries.join(", "));

    create_harness(PROVIDER, config)
        .with_context(|| format!("the {PROVIDER} provider refused the create mapping"))
}

/// Prints the card a created harness reports.
fn print_card(card: &HarnessCard) {
    common::banner::step(&format!(
        "created harness {} {} (protocol {})",
        card.name, card.implementation_version, card.protocol_version
    ));
    let capabilities: Vec<&str> = card.capabilities.iter().map(|c| c.as_str()).collect();
    common::banner::note(&format!("capabilities: {}", capabilities.join(", ")));
}

/// Returns a short label for the payload an event carries.
fn payload_label(event: &HarnessEvent) -> String {
    match &event.event {
        HarnessEventPayload::TurnLifecycle(payload) => {
            format!("turn_lifecycle/{}", payload.kind.as_str())
        }
        HarnessEventPayload::Output(payload) => format!("output/{}", payload.channel.as_str()),
        other => other.wire_event_type().to_owned(),
    }
}

/// Returns the terminal result of `turn_id`, when `event` is its final lifecycle event.
fn terminal_result<'a>(event: &'a HarnessEvent, turn_id: &str) -> Option<&'a TurnResult> {
    let HarnessEventPayload::TurnLifecycle(payload) = &event.event else {
        return None;
    };
    if event.turn_id.as_deref() != Some(turn_id) || !payload.kind.is_terminal() {
        return None;
    }
    payload.result.as_ref()
}

/// Returns the host-facing name of a turn status.
fn status_label(status: TurnStatus) -> &'static str {
    match status {
        TurnStatus::Completed => "completed",
        TurnStatus::Interrupted => "interrupted",
        TurnStatus::Failed => "failed",
    }
}

/// Drains the cursor until the turn ends, printing every event it sees.
///
/// Both bounds are deliberate: a per-event timeout and an event cap, so a script
/// mismatch prints a note instead of hanging the demo.
async fn observe_turn(cursor: &mut Box<dyn HarnessEventCursor>, turn_id: &str) -> Observation {
    let mut observation = Observation {
        events: 0,
        answer: String::new(),
        result: None,
    };
    loop {
        if observation.events >= MAX_EVENTS {
            common::banner::note(&format!(
                "stopped after {MAX_EVENTS} events without a terminal turn event"
            ));
            return observation;
        }

        let event = match tokio::time::timeout(EVENT_TIMEOUT, cursor.next_event()).await {
            Ok(Ok(Some(event))) => event,
            Ok(Ok(None)) => {
                common::banner::note("the observation cursor closed before the turn ended");
                return observation;
            }
            Ok(Err(error)) => {
                common::banner::note(&format!("the observation cursor reported {error}"));
                return observation;
            }
            Err(_) => {
                common::banner::note("no further event arrived before the timeout");
                return observation;
            }
        };
        observation.events += 1;

        common::banner::note(&format!(
            "{:>3} {:<28} turn={} item={}",
            event.sequence,
            payload_label(&event),
            event.turn_id.as_deref().unwrap_or("-"),
            event.item_id.as_deref().unwrap_or("-"),
        ));

        if let HarnessEventPayload::Output(output) = &event.event {
            if let Some(text) = output.content.as_str() {
                common::banner::note(&format!("output[{}]: {text}", output.channel.as_str()));
                observation.answer.push_str(text);
            }
        }

        if let Some(terminal) = terminal_result(&event, turn_id) {
            observation.result = Some(terminal.clone());
            return observation;
        }
    }
}

/// Returns the answer one observed turn produced, or an error naming why it has none.
///
/// A turn that ends without a single `answer` output is an event-stream
/// regression, not a reason to print the scripted text: the example would exit
/// `0` with a convincing transcript that never happened. The error carries the
/// event count and the terminal status, so the failure says what was observed.
fn require_answer(observation: &Observation) -> Result<String> {
    if !observation.answer.is_empty() {
        return Ok(observation.answer.clone());
    }
    let ending = match &observation.result {
        Some(result) => format!(
            "turn {} (stop reason: {})",
            status_label(result.status),
            result.stop_reason.as_deref().unwrap_or("<none>")
        ),
        None => "no terminal turn result".to_owned(),
    };
    Err(anyhow::anyhow!(
        "the turn produced no `answer` output after {} event(s), {ending}: the harness \
         event stream is not delivering the model's output (scripted answer was {:?})",
        observation.events,
        SCRIPTED_ANSWER,
    ))
}
#[tokio::main]
async fn main() -> Result<()> {
    let mode = common::setup()?;
    common::banner::title("09 harness provider");

    if mode.is_live() {
        common::banner::note(
            "live mode does not apply here: the create mapping points the provider at the \
             loopback server on purpose",
        );
    }
    common::banner::note(
        "the native provider builds its model from the create mapping, so api_base points at a \
         loopback mock server and no real provider is contacted",
    );

    print_registry();

    // The workspace and the server outlive the harness: the `TempDir` keeps the
    // per-agent files inside it, and the server must still be listening while the
    // turn runs.
    let workspace = common::tempdir()?;
    let server = common::mock_server(vec![common::text_response(SCRIPTED_ANSWER)]).await?;
    common::banner::note(&format!(
        "loopback model endpoint: {} (workspace: {})",
        server.base_url(),
        workspace.path().display()
    ));

    let harness = create_native_harness(server.base_url(), workspace.path())?;
    print_card(harness.card());
    common::banner::note(&format!("state before start: {:?}", harness.state()));

    common::banner::step("starting the harness");
    tokio::time::timeout(
        CALL_TIMEOUT,
        harness.start(HarnessContext::new(
            "deep_agent",
            "agent-1",
            "host-1",
            SYSTEM_PROMPT,
        )),
    )
    .await
    .context("the harness start timed out")?
    .context("the harness refused to start")?;
    common::banner::note(&format!(
        "provider session: {}",
        harness.provider_session_id().unwrap_or("<none>")
    ));

    // One consumer for the whole cycle: `events()` and `turn_events()` share a
    // single logical channel, so a second cursor here would be an error.
    let mut cursor: Box<dyn HarnessEventCursor> = harness.events();

    common::banner::step(&format!("sending: {PROMPT}"));
    let receipt = tokio::time::timeout(CALL_TIMEOUT, harness.send_auto(HarnessInput::text(PROMPT)))
        .await
        .context("the send timed out")?
        .context("the harness refused the input")?;
    common::banner::note(&format!(
        "accepted as {} for turn {} ({:?})",
        receipt.message_id, receipt.turn_id, receipt.accepted_mode
    ));

    common::banner::step("observing the turn");
    let observation = observe_turn(&mut cursor, &receipt.turn_id).await;
    match &observation.result {
        Some(result) => common::banner::note(&format!(
            "turn {} {} (stop reason: {})",
            receipt.turn_id,
            status_label(result.status),
            result.stop_reason.as_deref().unwrap_or("<none>"),
        )),
        None => common::banner::note(&format!(
            "turn {} did not report a terminal result in {} events",
            receipt.turn_id, observation.events
        )),
    }
    common::banner::note(&format!(
        "{} events observed, server saw {} request(s)",
        observation.events,
        server.request_count()
    ));

    cursor
        .close()
        .await
        .context("the observation cursor failed to close")?;

    common::banner::step("stopping the harness");
    tokio::time::timeout(CALL_TIMEOUT, harness.stop())
        .await
        .context("the harness stop timed out")?
        .context("the harness failed to stop")?;
    common::banner::note(&format!("state after stop: {:?}", harness.state()));

    server.shutdown().await;

    // No fallback text: an empty turn is a regression this example must report,
    // not paper over with the answer it scripted.
    common::banner::answer(&require_answer(&observation)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    /// Builds the observation of a turn that delivered no `answer` output.
    fn silent_turn(events: usize, result: Option<TurnResult>) -> Observation {
        Observation {
            events,
            answer: String::new(),
            result,
        }
    }

    #[test]
    fn an_empty_turn_fails_with_the_observed_counts() {
        let error = require_answer(&silent_turn(3, None)).expect_err("silence is a regression");
        let text = error.to_string();
        assert!(text.contains("no `answer` output"), "{text}");
        assert!(text.contains("after 3 event(s)"), "{text}");
        assert!(text.contains("no terminal turn result"), "{text}");
    }

    #[test]
    fn a_terminal_turn_without_output_names_its_status() {
        let result: TurnResult = serde_json::from_value(serde_json::json!({
            "status": "failed",
            "error": {"message": "the model failed"},
        }))
        .expect("a terminal failed result");
        let error = require_answer(&silent_turn(7, Some(result))).expect_err("still no answer");
        let text = error.to_string();
        assert!(text.contains("after 7 event(s)"), "{text}");
        assert!(text.contains("turn failed"), "{text}");
    }

    #[test]
    fn a_produced_answer_is_returned_verbatim() {
        let observation = Observation {
            events: 2,
            answer: SCRIPTED_ANSWER.to_owned(),
            result: None,
        };
        assert_eq!(
            require_answer(&observation).expect("the turn answered"),
            SCRIPTED_ANSWER
        );
    }
}
