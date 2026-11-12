//! Checkpoint export, host-sink publication, and resume.

use super::*;

#[tokio::test]
async fn export_checkpoint_describes_the_fresh_cycle() {
    let fixture = fixture();
    fixture.start().await;

    let checkpoint = fixture
        .harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("start publishes a checkpoint");
    assert_eq!(checkpoint.provider, HARNESS_NAME);
    assert_eq!(checkpoint.schema_version, "1");
    assert_eq!(checkpoint.agent_id, "agent-1");
    assert_eq!(checkpoint.host_session_id, "host-1");
    assert_eq!(checkpoint.provider_session_id.as_deref(), Some("host-1"));
    assert_eq!(checkpoint.sequence, 1);

    let state = decoded(&checkpoint);
    assert_eq!(state.schema_version, "1");
    assert_eq!(state.agent_id, "agent-1");
    assert_eq!(state.host_session_id, "host-1");
    assert_eq!(state.session_id, "host-1");
    assert_eq!(state.turn_counter, 0);
    assert_eq!(state.plan_mode, "normal");
    assert_eq!(state.pending_follow_ups.len(), 0);
}

#[tokio::test]
async fn checkpoint_sink_receives_activation_and_turn_completion() {
    let fixture = fixture_with_sink(false);
    fixture.start().await;
    let sink = fixture.sink_state();

    let activated = wait_for_checkpoint(&sink, CheckpointReason::SessionActivated).await;
    assert_eq!(activated.sequence, 1);

    fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;

    let sequences = sink.sequences();
    assert_eq!(
        sink.reasons(),
        vec![
            CheckpointReason::SessionActivated,
            CheckpointReason::TurnCompleted
        ]
    );
    assert!(
        sequences.windows(2).all(|pair| pair[0] < pair[1]),
        "checkpoint sequences must increase: {sequences:?}"
    );
}

#[tokio::test]
async fn a_conflicting_checkpoint_sink_never_fails_the_turn() {
    let fixture = fixture_with_sink(true);
    fixture.start().await;
    let sink = fixture.sink_state();

    fixture
        .harness
        .send(HarnessInput::text("hi"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;

    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    // The checkpoint is recorded before the (conflicting) sink write, so it is
    // exportable as soon as the sink has seen it.
    let checkpoint = fixture
        .harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("the checkpoint is exportable while the sink write is in flight");
    assert_eq!(checkpoint.provider, HARNESS_NAME);
    let state = decoded(&checkpoint);
    assert_eq!(state.turn_counter, 1, "the turn still advanced the counter");
    assert!(
        sink.reasons().contains(&CheckpointReason::TurnCompleted),
        "the conflicting sink still received the write"
    );
}

#[tokio::test]
async fn require_resume_without_a_checkpoint_fails() {
    let fixture = fixture();

    let error = fixture
        .harness
        .start(context().with_resume_policy(ResumePolicy::RequireResume))
        .await
        .expect_err("require_resume demands a checkpoint");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "native harness requires a checkpoint to resume"
    );
    assert_eq!(fixture.harness.state(), HarnessState::Terminated);
}

#[tokio::test]
async fn require_resume_rejects_a_foreign_scope() {
    let foreign_agent = fixture();
    let error = foreign_agent
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint_for("agent-9", "host-1")),
        )
        .await
        .expect_err("a checkpoint of another agent cannot be resumed");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "native checkpoint scope does not match the context"
    );

    let foreign_session = fixture();
    let error = foreign_session
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint_for("agent-1", "host-9")),
        )
        .await
        .expect_err("a checkpoint of another host session cannot be resumed");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "native checkpoint scope does not match the context"
    );
}

#[tokio::test]
async fn require_resume_restores_the_turn_counter_and_plan_mode() {
    let fixture = fixture();

    fixture
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint_for("agent-1", "host-1")),
        )
        .await
        .expect("the checkpoint is restored");

    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    assert_eq!(fixture.harness.provider_session_id(), Some("host-1"));

    // The activation checkpoint is published after the restore, so it already
    // reports the restored counters and plan mode.
    let checkpoint = fixture
        .harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("start publishes a checkpoint");
    let state = decoded(&checkpoint);
    assert_eq!(state.turn_counter, 4);
    assert_eq!(state.plan_mode, "plan");
    assert_eq!(fixture.agent.config().default_mode.as_str(), "plan");

    // The restored counter of 4 makes the next input turn-5.
    let receipt = fixture
        .harness
        .send(HarnessInput::text("resume"), DeliveryMode::Auto)
        .await
        .expect("the resumed harness accepts input");
    assert_eq!(receipt.turn_id, "turn-5");
    assert_eq!(receipt.message_id, "message-1");
}

#[tokio::test]
async fn require_resume_applies_the_restored_state_before_the_activation_checkpoint() {
    let fixture = fixture();
    let mut checkpoint = checkpoint_for("agent-1", "host-1");
    let mut restored = decoded(&checkpoint);
    restored.turn_counter = 7;
    restored.iteration = 3;
    restored.plan_mode = "plan".to_owned();
    restored.pending_follow_ups = vec!["restored".to_owned()];
    checkpoint.data = restored.to_json();

    fixture
        .start_with(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint),
        )
        .await;

    let activation = fixture
        .harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("start publishes a checkpoint");
    let state = decoded(&activation);
    assert_eq!(state.iteration, 3, "the restored iteration is applied");
    assert_eq!(state.plan_mode, "plan");
    assert_eq!(state.pending_follow_ups, vec!["restored".to_owned()]);
    assert!(
        state.turn_counter >= 7,
        "the activation checkpoint reports the restored turn counter, got {}",
        state.turn_counter
    );
}

#[tokio::test]
async fn the_checkpoint_sequence_continues_across_a_restart() {
    let fixture = fixture();
    let checkpoint = HarnessCheckpoint {
        sequence: 5,
        ..checkpoint_for("agent-1", "host-1")
    };

    fixture
        .start_with(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint),
        )
        .await;

    let activation = fixture
        .harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("start publishes a checkpoint");
    assert!(
        activation.sequence >= 6,
        "the checkpoint sequence continues after the resumed checkpoint, got {}",
        activation.sequence
    );
}

#[tokio::test]
async fn require_resume_accepts_an_exported_checkpoint() {
    let source = fixture_with_sink(false);
    source.start().await;
    let sink = source.sink_state();
    source
        .harness
        .send(HarnessInput::text("first"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    wait_for_checkpoint(&sink, CheckpointReason::TurnCompleted).await;
    // `export_settled` still lives in the parent test module; the checkpoint is
    // already recorded when the sink write completes, so it returns on the
    // first pass.
    let exported = export_settled(&source.harness).await;
    assert_eq!(decoded(&exported).turn_counter, 1);

    let resumed = fixture();
    resumed
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(exported),
        )
        .await
        .expect("the exported checkpoint is restored");
    assert_eq!(resumed.harness.state(), HarnessState::Idle);
    assert_eq!(resumed.harness.provider_session_id(), Some("host-1"));

    let receipt = resumed
        .harness
        .send(HarnessInput::text("second"), DeliveryMode::Auto)
        .await
        .expect("the resumed harness accepts input");
    assert_eq!(
        receipt.turn_id, "turn-2",
        "the exported turn counter is restored"
    );
}

#[tokio::test]
async fn require_resume_rejects_an_undecodable_checkpoint() {
    let fixture = fixture();
    let mut data = JsonObject::new();
    data.insert("unexpected".to_owned(), JsonValue::Bool(true));
    let checkpoint = HarnessCheckpoint {
        data,
        ..checkpoint_for("agent-1", "host-1")
    };

    let error = fixture
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint),
        )
        .await
        .expect_err("undecodable data cannot be resumed");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(error.message(), "invalid native checkpoint");
}

#[tokio::test]
async fn require_resume_rejects_a_foreign_schema_version() {
    let fixture = fixture();
    let checkpoint = HarnessCheckpoint {
        schema_version: "2".to_owned(),
        ..checkpoint_for("agent-1", "host-1")
    };

    let error = fixture
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::RequireResume)
                .with_checkpoint(checkpoint),
        )
        .await
        .expect_err("a foreign checkpoint schema cannot be resumed");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(error.message(), "unsupported native checkpoint version");
}

#[tokio::test]
async fn resume_if_available_starts_cold_with_a_garbage_checkpoint() {
    let fixture = fixture();
    let mut data = JsonObject::new();
    data.insert("unexpected".to_owned(), JsonValue::Bool(true));
    let checkpoint = HarnessCheckpoint {
        data,
        ..checkpoint_for("agent-9", "host-9")
    };

    fixture
        .harness
        .start(
            context()
                .with_resume_policy(ResumePolicy::ResumeIfAvailable)
                .with_checkpoint(checkpoint),
        )
        .await
        .expect("resume_if_available ignores the unusable checkpoint");

    assert_eq!(fixture.harness.state(), HarnessState::Idle);
    assert_eq!(fixture.harness.provider_session_id(), Some("host-1"));
    let receipt = fixture
        .harness
        .send(HarnessInput::text("cold"), DeliveryMode::Auto)
        .await
        .expect("the cold-start harness accepts input");
    assert_eq!(receipt.turn_id, "turn-1", "the counter was not restored");
}
