//! Turns, observation cursors, delivery modes, and pause/resume.

use super::*;

#[tokio::test]
async fn send_returns_a_receipt_and_the_turn_completes_with_an_answer() {
    let fixture = fixture();
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    let receipt = fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    assert_eq!(receipt.message_id, "message-1");
    assert_eq!(receipt.turn_id, "turn-1");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);

    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;

    let started = events
        .iter()
        .find(|event| is_turn_event(event, "turn-1", TurnEventKind::Started))
        .expect("the turn starts");
    assert_eq!(started.correlation_id.as_deref(), Some("message-1"));
    assert_eq!(started.provider_session_id.as_deref(), Some("host-1"));
    assert!(
        events.iter().any(|event| matches!(
            &event.event,
            HarnessEventPayload::Output(output)
                if output.channel == OutputChannel::Answer
        )),
        "the answer delta was observed"
    );

    let finished = events.last().expect("the terminal event is buffered");
    assert_eq!(finished.turn_id.as_deref(), Some("turn-1"));
    let result = lifecycle(finished)
        .and_then(|payload| payload.result.as_ref())
        .expect("the terminal event carries a result");
    assert_eq!(result.status, TurnStatus::Completed);
    assert!(result.stop_reason.is_some(), "the stop reason is reported");
    assert_eq!(
        result.provider_data.get("result_type"),
        Some(&JsonValue::String("answer".to_owned()))
    );
}

#[tokio::test]
async fn steering_needs_a_running_turn() {
    let fixture = fixture();
    fixture.start().await;

    let error = fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Steer)
        .await
        .expect_err("there is nothing to steer");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "there is no active turn to steer");
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
}

#[tokio::test]
async fn steering_a_running_turn_returns_its_turn_id() {
    let (model, entered, release) = GateModel::new(ScriptedModel::text("gated answer"));
    let fixture = fixture_with(model, scripted_config());
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    let first = fixture
        .harness
        .send(HarnessInput::text("one"), DeliveryMode::Auto)
        .await
        .expect("the first input is accepted");
    let permit = entered
        .acquire()
        .await
        .expect("the turn enters the model call");
    drop(permit);

    let steer = fixture
        .harness
        .send(HarnessInput::text("change course"), DeliveryMode::Steer)
        .await
        .expect("a running turn accepts steering");
    assert_eq!(steer.turn_id, first.turn_id);
    assert_eq!(steer.accepted_mode, DeliveryMode::Steer);
    assert_eq!(steer.message_id, "message-2");

    release.add_permits(1);
    drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
}

#[tokio::test]
async fn abort_without_an_active_turn_is_a_no_op() {
    let fixture = fixture();
    fixture.start().await;

    fixture
        .harness
        .abort(AbortMode::Graceful)
        .await
        .expect("aborting an idle harness is a no-op");
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
}

#[tokio::test]
async fn abort_interrupts_the_active_turn() {
    let (model, entered, release) = GateModel::new(ScriptedModel::text("gated answer"));
    let fixture = fixture_with(model, scripted_config());
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    fixture
        .harness
        .send(HarnessInput::text("one"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    let permit = entered
        .acquire()
        .await
        .expect("the turn enters the model call");
    drop(permit);

    fixture
        .harness
        .abort(AbortMode::Graceful)
        .await
        .expect("a running turn accepts an abort");
    // Let the parked stream deliver one frame so the turn observes the abort at
    // its next frame boundary.
    release.add_permits(1);
    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Aborted)
    })
    .await;

    let aborted = events.last().expect("the terminal event is buffered");
    let result = lifecycle(aborted)
        .and_then(|payload| payload.result.as_ref())
        .expect("the terminal event carries a result");
    assert_eq!(result.status, TurnStatus::Interrupted);
    let termination = result
        .termination
        .as_ref()
        .expect("a termination is carried");
    assert_eq!(
        termination.kind,
        openjiuwen_harness_protocol::TurnTerminationKind::UserAbort
    );
}

#[tokio::test]
async fn turn_events_end_at_the_terminal_event() {
    let fixture = fixture_with_sink(false);
    fixture.start().await;
    let sink = fixture.sink_state();

    fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;

    let mut cursor = fixture.harness.turn_events(Some("turn-1"));
    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
    assert!(is_turn_event(&events[0], "turn-1", TurnEventKind::Started));
    assert!(cursor.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn turn_events_reject_a_foreign_turn_id() {
    let fixture = fixture_with_sink(false);
    fixture.start().await;
    let sink = fixture.sink_state();

    fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;

    let mut cursor = fixture.harness.turn_events(Some("turn-99"));
    let error = next_error(&mut cursor).await;
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(
        error.message(),
        "requested turn is not the next unconsumed turn: expected turn-99, found turn-1"
    );
    assert!(cursor.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn a_second_cursor_is_rejected_while_one_is_active() {
    let fixture = fixture();
    fixture.start().await;

    let mut first = fixture.harness.events();
    let _event = next_event(&mut first)
        .await
        .expect("the transition is buffered");

    let mut second = fixture.harness.events();
    let error = next_error(&mut second).await;
    assert_eq!(
        error.message(),
        "the harness observation stream already has an active consumer"
    );

    let mut third = fixture.harness.turn_events(None);
    let error = next_error(&mut third).await;
    assert_eq!(
        error.message(),
        "the harness observation stream already has an active consumer"
    );
}

#[tokio::test]
async fn a_queued_follow_up_runs_after_the_active_turn() {
    let (model, entered, release) = GateModel::new(ScriptedModel::text("gated answer"));
    let fixture = fixture_with(model, scripted_config());
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    let first = fixture
        .harness
        .send(HarnessInput::text("one"), DeliveryMode::Auto)
        .await
        .expect("the first input is accepted");
    assert_eq!(first.turn_id, "turn-1");
    assert_eq!(first.accepted_mode, DeliveryMode::Auto);

    // Park the turn inside the model call so the second input must be queued.
    let permit = entered
        .acquire()
        .await
        .expect("the turn enters the model call");
    drop(permit);
    let second = fixture
        .harness
        .send(HarnessInput::text("two"), DeliveryMode::Auto)
        .await
        .expect("the second input is accepted");
    assert_eq!(second.accepted_mode, DeliveryMode::FollowUp);
    assert_eq!(second.turn_id, "turn-2");
    assert_eq!(second.message_id, "message-2");

    release.add_permits(2);
    let mut starts = 0;
    let mut finishes = 0;
    while finishes < 2 {
        let Some(event) = next_event(&mut cursor).await else {
            panic!("the cycle cursor ended before both turns finished");
        };
        if let Some(payload) = lifecycle(&event) {
            match payload.kind {
                TurnEventKind::Started => starts += 1,
                TurnEventKind::Finished => {
                    let result = payload.result.as_ref().expect("the result is carried");
                    assert_eq!(result.status, TurnStatus::Completed);
                    finishes += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(starts, 2, "both queued turns started on one cursor");
}

#[tokio::test]
async fn pause_defers_input_until_resume() {
    let fixture = fixture_with_sink(false);
    fixture.start().await;
    let mut cursor = fixture.harness.events();
    let sink = fixture.sink_state();

    fixture.harness.pause().await.expect("pause succeeds");
    assert_eq!(fixture.harness.state(), HarnessState::Paused);

    let receipt = fixture
        .harness
        .send(HarnessInput::text("queued"), DeliveryMode::Auto)
        .await
        .expect("a paused harness accepts a follow-up");
    assert_eq!(receipt.accepted_mode, DeliveryMode::FollowUp);
    assert_eq!(receipt.turn_id, "turn-1");
    assert_eq!(
        fixture.harness.state(),
        HarnessState::Paused,
        "a paused harness must not start the queued turn"
    );

    fixture.harness.resume(None).await.expect("resume succeeds");
    let error = fixture
        .harness
        .resume(None)
        .await
        .expect_err("a resumed harness is not paused");
    assert_eq!(error.message(), "native harness is not paused");

    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
    assert!(events
        .iter()
        .any(|event| is_turn_event(event, "turn-1", TurnEventKind::Started)));
    let paused = events
        .iter()
        .find(|event| state_change(event).is_some_and(|change| change.new == HarnessState::Paused))
        .expect("the pause transition was observed");
    assert_eq!(
        state_change(paused).map(|change| change.old),
        Some(HarnessState::Idle)
    );
    // The resumed turn settles after its terminal event, so the Idle assertion
    // waits for the checkpoint published by the settlement.
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;
    assert_eq!(fixture.harness.state(), HarnessState::Idle);
}

#[tokio::test]
async fn resume_with_input_runs_the_supplied_turn() {
    let fixture = fixture();
    fixture.start().await;
    let mut cursor = fixture.harness.events();

    fixture.harness.pause().await.expect("pause succeeds");
    fixture
        .harness
        .resume(Some(HarnessInput::text("after pause")))
        .await
        .expect("resume with an input succeeds");

    let events = drain_until(&mut cursor, |event| {
        is_turn_event(event, "turn-1", TurnEventKind::Finished)
    })
    .await;
    let finished = events.last().expect("the terminal event is buffered");
    let result = lifecycle(finished)
        .and_then(|payload| payload.result.as_ref())
        .expect("the terminal event carries a result");
    assert_eq!(result.status, TurnStatus::Completed);
}
