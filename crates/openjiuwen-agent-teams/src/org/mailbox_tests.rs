//! Behavioural tests of [`crate::org::mailbox`] over the in-memory [`crate::storage::TeamStore`].
//!
//! Every test is deterministic and offline: one store per test, no sleeps, and only monotonicity or
//! round-trip assertions about stamps (never a wall-clock value). Rows inserted straight through the
//! store use caller-supplied stamps.
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};

use super::*;
use crate::constants::MAX_MESSAGE_CONTENT_CHARS;
use crate::org::{is_org_namespace, ORG_MESSAGE_PROTOCOL, ORG_SYSTEM_SENDER};
use crate::schema::TaskRecord;
use crate::storage::{InMemoryTeamStore, MessageRecord, TeamStore};

fn memory_store() -> Arc<InMemoryTeamStore> {
    Arc::new(InMemoryTeamStore::new())
}

fn mailbox(store: &Arc<InMemoryTeamStore>, organization_id: &str) -> OrgMailbox {
    OrgMailbox::new(store.clone(), organization_id).unwrap()
}

fn org_payload(kind: &str, subject: &str, body: &str) -> String {
    json!({ "kind": kind, "subject": subject, "body": body }).to_string()
}

async fn insert_row(
    store: &dyn TeamStore,
    namespace: &str,
    message_id: &str,
    to: Option<&str>,
    content: &str,
    timestamp: u64,
    protocol: &str,
) {
    let mut row = match to {
        Some(team) => {
            MessageRecord::direct(namespace, message_id, "alpha", team, content, timestamp)
        }
        None => MessageRecord::broadcast(namespace, message_id, "alpha", content, timestamp),
    };
    row.protocol = String::from(protocol);
    store.insert_message(row).await.unwrap();
}

#[tokio::test]
async fn send_persists_one_message_for_the_receiver_only() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let sent = mailbox
        .send("alpha", "beta", "handoff", "please review")
        .await
        .unwrap();
    assert_eq!(sent.kind, OrgMessageKind::Leader);
    assert_eq!(sent.organization_id, "acme");
    assert_eq!(sent.from_team, "alpha");
    assert_eq!(sent.to_team.as_deref(), Some("beta"));
    assert_eq!(sent.subject, "handoff");
    assert_eq!(sent.body, "please review");
    assert!(sent.timestamp > 0);
    assert!(!sent.read);
    assert!(!sent.is_broadcast());
    assert!(!sent.is_system());

    let hex = sent.message_id.strip_prefix(ORG_MESSAGE_ID_PREFIX).unwrap();
    assert_eq!(hex.chars().count(), 32);
    assert!(hex
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

    let inbox = mailbox.inbox("beta").await.unwrap();
    assert_eq!(inbox, vec![sent]);
    assert_eq!(mailbox.inbox("alpha").await.unwrap(), Vec::new());
    assert_eq!(mailbox.unread_count("beta").await.unwrap(), 1);
}

#[tokio::test]
async fn send_refuses_self_and_invalid_team_names() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let error = mailbox.send("alpha", "alpha", "s", "b").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("team 'alpha' cannot message itself"),
        "{error}"
    );

    for bad in ["", "org::alpha", "alpha::beta"] {
        let error = mailbox.send(bad, "beta", "s", "b").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{bad}");
        let error = mailbox.send("alpha", bad, "s", "b").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{bad}");
    }

    let long = "t".repeat(MAX_MAILBOX_TEAM_NAME_CHARS + 1);
    let error = mailbox.send(&long, "beta", "s", "b").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .to_string()
        .contains(&MAX_MAILBOX_TEAM_NAME_CHARS.to_string()));

    // The limit itself is accepted; one character more is refused and names the limit.
    let at_limit = "x".repeat(MAX_MESSAGE_CONTENT_CHARS);
    assert!(mailbox.send("alpha", "beta", "s", &at_limit).await.is_ok());
    let over = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
    let error = mailbox.send("alpha", "beta", "s", &over).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .to_string()
        .contains(&MAX_MESSAGE_CONTENT_CHARS.to_string()));
    assert!(error.to_string().contains("body"), "{error}");
    let error = mailbox.send("alpha", "beta", &over, "b").await.unwrap_err();
    assert!(error
        .to_string()
        .contains(&MAX_MESSAGE_CONTENT_CHARS.to_string()));
    assert!(error.to_string().contains("subject"), "{error}");
}

#[tokio::test]
async fn broadcast_is_unread_for_every_other_team() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let sent = mailbox
        .broadcast("alpha", "standup", "in five")
        .await
        .unwrap();
    assert_eq!(sent.kind, OrgMessageKind::Broadcast);
    assert_eq!(sent.to_team, None);
    assert!(sent.is_broadcast());
    assert!(!sent.read);

    assert_eq!(mailbox.inbox("alpha").await.unwrap(), Vec::new());
    for team in ["beta", "gamma"] {
        assert_eq!(mailbox.inbox(team).await.unwrap(), vec![sent.clone()]);
    }

    for bad in ["", "org::alpha", "a::b"] {
        let error = mailbox.broadcast(bad, "s", "b").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{bad}");
    }
    let over = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
    let error = mailbox.broadcast("alpha", "s", &over).await.unwrap_err();
    assert!(error
        .to_string()
        .contains(&MAX_MESSAGE_CONTENT_CHARS.to_string()));
    let error = mailbox.broadcast("alpha", &over, "b").await.unwrap_err();
    assert!(error
        .to_string()
        .contains(&MAX_MESSAGE_CONTENT_CHARS.to_string()));
}

#[tokio::test]
async fn acknowledging_a_broadcast_moves_the_watermark_forward() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let first = mailbox.broadcast("alpha", "one", "first").await.unwrap();
    let second = mailbox.broadcast("alpha", "two", "second").await.unwrap();
    assert!(second.timestamp > first.timestamp);
    assert_eq!(mailbox.inbox("beta").await.unwrap().len(), 2);

    let cleared = mailbox
        .acknowledge("beta", std::slice::from_ref(&first.message_id))
        .await
        .unwrap();
    assert_eq!(cleared, 1);
    assert_eq!(
        mailbox.watermark("beta").await.unwrap(),
        Some(first.timestamp)
    );
    assert_eq!(mailbox.inbox("beta").await.unwrap(), vec![second.clone()]);

    // A later broadcast is unread again; the acknowledged one stays read.
    let third = mailbox.broadcast("alpha", "three", "third").await.unwrap();
    assert_eq!(
        mailbox.inbox("beta").await.unwrap(),
        vec![second, third.clone()]
    );

    // Re-acknowledging an older message changes nothing and never regresses the watermark.
    let cleared = mailbox
        .acknowledge("beta", std::slice::from_ref(&first.message_id))
        .await
        .unwrap();
    assert_eq!(cleared, 0);
    assert_eq!(
        mailbox.watermark("beta").await.unwrap(),
        Some(first.timestamp)
    );

    // Acknowledging the newest broadcast clears everything below it.
    let cleared = mailbox
        .acknowledge("beta", std::slice::from_ref(&third.message_id))
        .await
        .unwrap();
    assert_eq!(cleared, 1);
    assert_eq!(
        mailbox.watermark("beta").await.unwrap(),
        Some(third.timestamp)
    );
    assert_eq!(mailbox.inbox("beta").await.unwrap(), Vec::new());
}

#[tokio::test]
async fn acknowledge_direct_is_idempotent_and_ignores_foreign_ids() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let sent = mailbox.send("alpha", "beta", "s", "b").await.unwrap();
    let other = mailbox.send("alpha", "gamma", "s", "b").await.unwrap();

    let cleared = mailbox
        .acknowledge("beta", &["missing".to_owned(), other.message_id.clone()])
        .await
        .unwrap();
    assert_eq!(cleared, 0);
    assert_eq!(mailbox.inbox("beta").await.unwrap(), vec![sent.clone()]);

    let cleared = mailbox
        .acknowledge("beta", std::slice::from_ref(&sent.message_id))
        .await
        .unwrap();
    assert_eq!(cleared, 1);
    assert_eq!(mailbox.inbox("beta").await.unwrap(), Vec::new());

    let cleared = mailbox
        .acknowledge("beta", &[sent.message_id])
        .await
        .unwrap();
    assert_eq!(cleared, 0);
    assert_eq!(mailbox.acknowledge("beta", &[]).await.unwrap(), 0);
    assert_eq!(mailbox.unread_count("beta").await.unwrap(), 0);

    // The other team's copy is untouched.
    assert_eq!(mailbox.inbox("gamma").await.unwrap(), vec![other]);
}

#[tokio::test]
async fn acknowledge_all_clears_direct_and_broadcast_state() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let direct = mailbox.send("alpha", "beta", "s", "b").await.unwrap();
    let announcement = mailbox.broadcast("alpha", "s", "b").await.unwrap();
    let notice = mailbox.notify("beta", "maintenance").await.unwrap();

    let unread = mailbox.unread_count("beta").await.unwrap();
    assert_eq!(unread, 3);
    assert_eq!(mailbox.inbox("beta").await.unwrap().len(), unread);

    assert_eq!(mailbox.acknowledge_all("beta").await.unwrap(), 3);
    assert_eq!(mailbox.inbox("beta").await.unwrap(), Vec::new());
    assert_eq!(mailbox.unread_count("beta").await.unwrap(), 0);
    let newest = [direct.timestamp, announcement.timestamp, notice.timestamp]
        .into_iter()
        .max()
        .unwrap();
    assert_eq!(mailbox.watermark("beta").await.unwrap(), Some(newest));
    assert_eq!(mailbox.acknowledge_all("beta").await.unwrap(), 0);

    // Only the announcement was visible to the other teams.
    assert_eq!(mailbox.unread_count("gamma").await.unwrap(), 1);
    assert_eq!(mailbox.unread_count("alpha").await.unwrap(), 0);
    assert_eq!(mailbox.acknowledge_all("alpha").await.unwrap(), 0);
}

#[tokio::test]
async fn system_notifications_are_normal_inbox_rows() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let notice = mailbox.notify("beta", "maintenance window").await.unwrap();
    assert_eq!(notice.from_team, ORG_SYSTEM_SENDER);
    assert_eq!(notice.kind, OrgMessageKind::System);
    assert_eq!(notice.subject, ORG_SYSTEM_DEFAULT_SUBJECT);
    assert_eq!(notice.to_team.as_deref(), Some("beta"));
    assert!(notice.is_system());
    assert!(!notice.is_broadcast());
    assert_eq!(mailbox.inbox("beta").await.unwrap(), vec![notice]);

    let announcement = mailbox.announce("all hands").await.unwrap();
    assert_eq!(announcement.from_team, ORG_SYSTEM_SENDER);
    assert_eq!(announcement.kind, OrgMessageKind::System);
    assert!(announcement.to_team.is_none());
    assert!(announcement.is_system());

    // The organisation never receives its own announcement.
    assert_eq!(mailbox.inbox(ORG_SYSTEM_SENDER).await.unwrap(), Vec::new());
    assert_eq!(mailbox.inbox("beta").await.unwrap().len(), 2);
    assert_eq!(mailbox.inbox("gamma").await.unwrap(), vec![announcement]);

    let error = mailbox.notify("org::beta", "b").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let over = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
    assert!(mailbox.notify("beta", &over).await.is_err());
    assert!(mailbox.announce(&over).await.is_err());
}

#[tokio::test]
async fn inbox_is_oldest_first_and_equal_stamps_stay_stable() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");

    let first = mailbox.send("alpha", "beta", "one", "1").await.unwrap();
    let second = mailbox.send("alpha", "beta", "two", "2").await.unwrap();
    let third = mailbox.send("alpha", "beta", "three", "3").await.unwrap();
    assert!(first.timestamp < second.timestamp);
    assert!(second.timestamp < third.timestamp);

    let inbox = mailbox.inbox("beta").await.unwrap();
    let ids: Vec<&str> = inbox.iter().map(|m| m.message_id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            first.message_id.as_str(),
            second.message_id.as_str(),
            third.message_id.as_str()
        ]
    );

    // Two rows sharing a stamp keep the store's `(timestamp, message_id)` order.
    let tie_stamp = third.timestamp + 10;
    let a_id = format!("{ORG_MESSAGE_ID_PREFIX}{}", "a".repeat(32));
    let b_id = format!("{ORG_MESSAGE_ID_PREFIX}{}", "b".repeat(32));
    insert_row(
        store.as_ref(),
        mailbox.namespace(),
        &b_id,
        Some("beta"),
        &org_payload("leader", "tie", "b"),
        tie_stamp,
        ORG_MESSAGE_PROTOCOL,
    )
    .await;
    insert_row(
        store.as_ref(),
        mailbox.namespace(),
        &a_id,
        Some("beta"),
        &org_payload("leader", "tie", "a"),
        tie_stamp,
        ORG_MESSAGE_PROTOCOL,
    )
    .await;

    let inbox = mailbox.inbox("beta").await.unwrap();
    assert_eq!(inbox.len(), 5);
    assert_eq!(inbox[0].message_id, first.message_id);
    assert_eq!(inbox[4].message_id, b_id);
    assert_eq!(inbox[3].message_id, a_id);
}

#[tokio::test]
async fn namespaces_keep_organisations_and_teams_apart() {
    let store = memory_store();
    let acme = mailbox(&store, "acme");
    let globex = mailbox(&store, "globex");

    assert_eq!(acme.namespace(), "org::acme");
    assert!(is_org_namespace(acme.namespace()));
    assert_eq!(globex.namespace(), "org::globex");
    assert!(is_org_namespace(globex.namespace()));
    assert_ne!(acme.namespace(), globex.namespace());

    let acme_message = acme.send("alpha", "beta", "a", "acme mail").await.unwrap();
    let globex_message = globex
        .send("alpha", "beta", "g", "globex mail")
        .await
        .unwrap();
    assert_ne!(acme_message.message_id, globex_message.message_id);

    // A real team's mailbox, and its task board, share the store but not the namespace rows.
    store
        .insert_message(MessageRecord::direct(
            "alpha",
            "plain-1",
            "team_leader",
            "beta",
            "hello",
            1,
        ))
        .await
        .unwrap();
    store
        .insert_task(TaskRecord::new(acme.namespace(), "task-1", "t", "c"))
        .await
        .unwrap();
    assert_eq!(
        store.unread_messages("alpha", "beta").await.unwrap().len(),
        1
    );
    assert_eq!(acme.inbox("alpha").await.unwrap(), Vec::new());

    // A plain-protocol row inside the namespace itself is invisible to the mailbox.
    insert_row(
        store.as_ref(),
        acme.namespace(),
        "plain-2",
        Some("beta"),
        "plain body",
        2,
        "plain",
    )
    .await;

    assert_eq!(acme.inbox("beta").await.unwrap(), vec![acme_message]);
    assert_eq!(acme.unread_count("beta").await.unwrap(), 1);
    assert_eq!(globex.inbox("beta").await.unwrap(), vec![globex_message]);
    assert_eq!(globex.unread_count("beta").await.unwrap(), 1);
}

#[tokio::test]
async fn foreign_protocol_and_corrupt_rows_are_skipped() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");
    let good = mailbox.send("alpha", "beta", "s", "b").await.unwrap();

    let corrupt = [
        ("msg_corrupt1", "not json"),
        ("msg_corrupt2", r#"{"kind":"leader""#),
        ("msg_corrupt3", "[]"),
        (
            "msg_corrupt4",
            r#"{"kind":"nope","subject":"s","body":"b"}"#,
        ),
    ];
    for (index, (id, content)) in corrupt.into_iter().enumerate() {
        insert_row(
            store.as_ref(),
            mailbox.namespace(),
            id,
            Some("beta"),
            content,
            good.timestamp + 1 + u64::try_from(index).unwrap(),
            ORG_MESSAGE_PROTOCOL,
        )
        .await;
    }

    let inbox = mailbox.inbox("beta").await.unwrap();
    assert_eq!(inbox.len(), 1);
    assert_eq!(inbox[0].message_id, good.message_id);
    assert_eq!(mailbox.unread_count("beta").await.unwrap(), 1);

    // Corruption never blocks the acknowledgment path.
    assert_eq!(mailbox.acknowledge_all("beta").await.unwrap(), 1);
    assert_eq!(mailbox.inbox("beta").await.unwrap(), Vec::new());
}

#[tokio::test]
async fn message_and_kind_round_trip_through_serde() {
    for (kind, wire) in [
        (OrgMessageKind::Leader, "leader"),
        (OrgMessageKind::System, "system"),
        (OrgMessageKind::Broadcast, "broadcast"),
    ] {
        assert_eq!(serde_json::to_value(kind).unwrap(), json!(wire));
        let decoded: OrgMessageKind = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(decoded, kind);
    }
    assert_eq!(OrgMessageKind::default(), OrgMessageKind::Leader);
    assert_eq!(OrgMessageKind::Leader.as_str(), "leader");

    let broadcast = OrgMessage {
        message_id: "msg_1".to_owned(),
        organization_id: "acme".to_owned(),
        from_team: "alpha".to_owned(),
        to_team: None,
        kind: OrgMessageKind::Broadcast,
        subject: "s".to_owned(),
        body: "b".to_owned(),
        timestamp: 7,
        read: false,
    };
    let encoded = serde_json::to_value(&broadcast).unwrap();
    assert_eq!(encoded["to_team"], Value::Null);
    assert_eq!(encoded["kind"], "broadcast");
    assert_eq!(encoded["read"], false);
    let decoded: OrgMessage = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, broadcast);

    // Unknown fields are tolerated and every field defaults.
    let decoded: OrgMessage = serde_json::from_value(json!({
        "message_id": "m",
        "from_team": "alpha",
        "kind": "leader",
        "junk": true,
    }))
    .unwrap();
    assert_eq!(decoded.message_id, "m");
    assert_eq!(decoded.kind, OrgMessageKind::Leader);
    assert_eq!(decoded.organization_id, "");
    assert_eq!(decoded.subject, "");
    assert_eq!(decoded.timestamp, 0);
    assert!(decoded.to_team.is_none());
    assert!(!decoded.read);
    assert_eq!(
        serde_json::from_value::<OrgMessage>(json!({})).unwrap(),
        OrgMessage::default()
    );
}

#[test]
fn preview_truncates_on_a_char_boundary() {
    let message = OrgMessage {
        subject: "ignored".to_owned(),
        body: "日本語テキスト".to_owned(),
        ..OrgMessage::default()
    };

    let preview = message.preview(3);
    assert_eq!(preview, "日本…");
    assert_eq!(preview.chars().count(), 3);
    assert_eq!(message.preview(0), "");
    assert_eq!(message.preview(7), "日本語テキスト");
    assert_eq!(message.preview(100), "日本語テキスト");

    // An empty body falls back to the subject.
    let subject_only = OrgMessage {
        subject: "subject only".to_owned(),
        ..OrgMessage::default()
    };
    assert_eq!(subject_only.preview(5), "subj…");
    assert_eq!(subject_only.preview(5).chars().count(), 5);
    assert_eq!(OrgMessage::default().preview(10), "");
}

#[test]
fn constructor_validates_the_organization_id() {
    let store = memory_store();
    let oversize = "a".repeat(97);
    for bad in ["", "acme/core", "acme core", oversize.as_str()] {
        let error = OrgMailbox::new(store.clone(), bad).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{bad}");
    }

    let mailbox = OrgMailbox::new(store, "acme.core-1").unwrap();
    assert_eq!(mailbox.organization_id(), "acme.core-1");
    assert_eq!(mailbox.namespace(), "org::acme.core-1");
    assert_eq!(mailbox.store().name(), "memory");
}

#[tokio::test]
async fn clones_share_the_store_and_the_stamp_guard() {
    let store = memory_store();
    let first = mailbox(&store, "acme");
    let second = first.clone();

    let to_beta = first.send("alpha", "beta", "a", "1").await.unwrap();
    let to_gamma = second.send("beta", "gamma", "b", "2").await.unwrap();
    assert!(to_gamma.timestamp > to_beta.timestamp);

    assert_eq!(second.inbox("beta").await.unwrap(), vec![to_beta]);
    assert_eq!(first.inbox("gamma").await.unwrap(), vec![to_gamma]);
    assert!(format!("{first:?}").contains("acme"));
}

#[tokio::test]
async fn read_paths_validate_the_team_name() {
    let store = memory_store();
    let mailbox = mailbox(&store, "acme");
    for bad in ["", "org::alpha", "a::b"] {
        assert!(mailbox.inbox(bad).await.is_err(), "{bad}");
        assert!(mailbox.watermark(bad).await.is_err(), "{bad}");
        assert!(mailbox.unread_count(bad).await.is_err(), "{bad}");
        assert!(mailbox.acknowledge(bad, &[]).await.is_err(), "{bad}");
        assert!(mailbox.acknowledge_all(bad).await.is_err(), "{bad}");
    }
}
