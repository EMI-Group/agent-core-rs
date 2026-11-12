//! Mail and bus seams: `TeamAgent` mail → backend rows → in-process transport → leader steering.
//!
//! The backend unit tests pin routing on a bare facade; these cases drive it through a
//! manager-built team so the *same* transport the leader listens on is the one an observer
//! subscribes to, and the leader's coordination loop is what turns unread mail and board events
//! into steering deliveries.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use serde_json::json;

use crate::schema::{EventPayload, TaskStatus, TaskStatusChangedEvent, TeamTopic};

use super::support::{
    blueprint, drain, event_types, recorded, subscribe_observer, wait_until, Fixture,
};

#[tokio::test]
async fn a_direct_message_reaches_one_mailbox_with_one_event() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Message]);

    let record = team.send_message(Some("dev-1"), "hello").await.unwrap();

    assert_eq!(record.from_member_name, "boss");
    assert_eq!(record.to_member_name.as_deref(), Some("dev-1"));
    assert!(!record.is_broadcast(), "a targeted message is direct");
    let unread = team.backend().unread("dev-1").await.unwrap();
    assert_eq!(unread.len(), 1, "one row in the target's mailbox");
    assert_eq!(unread[0].content, "hello");
    assert!(
        team.backend().unread("boss").await.unwrap().is_empty(),
        "the leader does not receive its own mail"
    );

    assert_eq!(event_types(&recorder), vec!["message".to_owned()]);
    let events = recorded(&recorder);
    assert_eq!(events[0].sender_id, "boss");
    assert_eq!(events[0].payload["to_member_name"], json!("dev-1"));
}

#[tokio::test]
async fn a_broadcast_persists_one_row_and_one_broadcast_event() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Message]);

    let record = team.send_message(None, "all hands").await.unwrap();

    assert!(record.is_broadcast());
    assert_eq!(record.to_member_name, None);
    let unread = team.backend().unread("dev-1").await.unwrap();
    assert_eq!(
        unread.len(),
        1,
        "the broadcast is readable from the mailbox"
    );
    assert!(unread[0].is_broadcast());

    assert_eq!(event_types(&recorder), vec!["broadcast".to_owned()]);
    assert_eq!(recorded(&recorder)[0].payload["from_member_name"], "boss");
}

#[tokio::test]
async fn a_multicast_target_list_is_de_duplicated() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Message]);

    let record = team
        .send_message(Some("dev-1,dev-1"), "twice?")
        .await
        .unwrap();

    assert_eq!(record.to_member_name.as_deref(), Some("dev-1"));
    let unread = team.backend().unread("dev-1").await.unwrap();
    assert_eq!(unread.len(), 1, "one row, not one per repeated target");
    assert_eq!(event_types(&recorder), vec!["message".to_owned()]);
}

#[tokio::test]
async fn the_leader_loop_turns_mail_and_board_events_into_steering() {
    let fixture = Fixture::new();
    fixture.factory.hanging_for("boss");
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(
        team.backend().messager(),
        &[TeamTopic::Message, TeamTopic::Task],
    );

    let stream = team.stream(json!("plan"), None);
    wait_until("the initial leader query", || {
        fixture.sends("boss").iter().any(|text| text == "plan")
    })
    .await;

    let published = team
        .backend()
        .post_event(
            TeamTopic::Task,
            EventPayload::TaskStatusChanged(TaskStatusChangedEvent {
                team_name: "alpha".to_owned(),
                task_id: "task_1".to_owned(),
                status: TaskStatus::InProgress,
                assignee: Some("dev-1".to_owned()),
            }),
        )
        .await
        .unwrap();
    assert_eq!(published, "task_status_changed");

    team.backend()
        .send_message("dev-1", Some("boss"), "status?")
        .await
        .unwrap();

    wait_until("the leader steering batch", || {
        let sends = fixture.sends("boss");
        sends
            .iter()
            .any(|text| text.contains("dev-1 -> boss: status?"))
            && sends
                .iter()
                .any(|text| text.contains("task task_1: in_progress"))
    })
    .await;

    let events = recorded(&recorder);
    let task_event = events
        .iter()
        .find(|event| event.event_type == "task_status_changed")
        .expect("the board event reached the observer");
    assert_eq!(
        task_event.sender_id, "alpha",
        "organisation events are stamped with the team, not a member"
    );
    assert_eq!(task_event.payload["task_id"], json!("task_1"));

    team.stop().await.unwrap();
    let _frames = drain(stream).await;
}
