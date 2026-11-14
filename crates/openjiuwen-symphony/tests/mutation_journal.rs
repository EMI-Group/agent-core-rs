//! Contract tests for the graph-mutation request half of the port: batch normalization, the
//! order-independent request digest, the change-set check, and the durable idempotency
//! journal.
//!
//! Everything here is offline and deterministic: temp directories for the journal, pinned
//! digests for the hashing rules, and no assertion that depends on the wall clock (an
//! `inserted_at` is only checked for being present).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::orchestration::mutation::{
    request_digest, validate_mutation_against_inventory, JournalRecord, MutationJournal,
    MutationOperation, MutationRequest, JOURNAL_SCHEMA_VERSION, MAX_CHANGED_IDS,
    MAX_REQUEST_ID_CHARS, MAX_SOURCE_REVISION_CHARS, REQUEST_SCHEMA_VERSION,
};
use serde_json::{json, Value};

/// The repository's `add` fixture: two qualified capability ids.
fn ids() -> Vec<String> {
    vec!["tool:fetch".to_string(), "tool:sum".to_string()]
}

/// A prepared `add` batch declaring `source_revision = v2`.
fn request() -> MutationRequest {
    MutationRequest::prepare(MutationOperation::Add, &ids(), "req-digest", Some("v2"))
        .expect("prepare")
}

/// Every `*.json` file name under `root`, sorted.
fn json_files(root: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .expect("read_dir")
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

#[test]
fn prepare_normalizes_and_documents_the_json_shape() {
    let request = MutationRequest::prepare(
        MutationOperation::Update,
        &[" tool:sum ".to_string(), "tool:fetch".to_string()],
        "  req-1  ",
        Some("  v7  "),
    )
    .expect("prepare");
    assert_eq!(request.request_id, "req-1");
    assert_eq!(request.operation, MutationOperation::Update);
    assert_eq!(request.source_revision.as_deref(), Some("v7"));
    assert_eq!(request.changed_skill_ids, vec!["tool:fetch", "tool:sum"]);

    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(
        json,
        json!({
            "request_id": "req-1",
            "operation": "update",
            "source_revision": "v7",
            "changed_skill_ids": ["tool:fetch", "tool:sum"],
        })
    );
    let restored: MutationRequest = serde_json::from_value(json).expect("deserialize");
    assert_eq!(restored, request);
}

#[test]
fn prepare_rejects_every_invalid_batch() {
    let blank_id = MutationRequest::prepare(MutationOperation::Add, &ids(), "   ", None)
        .expect_err("a blank request id must be rejected");
    assert_eq!(blank_id.kind(), ErrorKind::Validation);
    assert!(blank_id.message().contains("request_id"));

    let long_id = "x".repeat(MAX_REQUEST_ID_CHARS + 1);
    let oversized_id = MutationRequest::prepare(MutationOperation::Add, &ids(), &long_id, None)
        .expect_err("an oversized request id must be rejected");
    assert!(oversized_id.message().contains("request_id"));

    let empty = MutationRequest::prepare(MutationOperation::Delete, &[], "req", None)
        .expect_err("a delete without ids must be rejected");
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.message().contains("changed_skill_ids"));

    let oversized: Vec<String> = (0..=MAX_CHANGED_IDS)
        .map(|index| format!("tool:cap-{index}"))
        .collect();
    let oversized_batch = MutationRequest::prepare(MutationOperation::Add, &oversized, "req", None)
        .expect_err("an oversized batch must be rejected");
    assert!(oversized_batch.message().contains("changed_skill_ids"));

    let duplicate = MutationRequest::prepare(
        MutationOperation::Add,
        &["tool:fetch".to_string(), "  tool:fetch".to_string()],
        "req",
        None,
    )
    .expect_err("a duplicate id must be rejected");
    assert!(duplicate.message().contains("tool:fetch"));

    let blank_member =
        MutationRequest::prepare(MutationOperation::Add, &["  ".to_string()], "req", None)
            .expect_err("a blank id must be rejected");
    assert!(blank_member.message().contains("changed_skill_ids"));

    let blank_revision =
        MutationRequest::prepare(MutationOperation::Add, &ids(), "req", Some("  "))
            .expect_err("a blank source revision must be rejected");
    assert!(blank_revision.message().contains("source_revision"));

    let long_revision = "v".repeat(MAX_SOURCE_REVISION_CHARS + 1);
    let oversized_revision =
        MutationRequest::prepare(MutationOperation::Add, &ids(), "req", Some(&long_revision))
            .expect_err("an oversized source revision must be rejected");
    assert!(oversized_revision.message().contains("source_revision"));
}

#[test]
fn digest_is_pinned_and_covers_every_field() {
    let request = request();
    assert_eq!(
        request.digest(),
        "74cdb358db562085bb398581e919aad98b598c9685e9d708deee9369bb8f022c",
        "the digest document must stay exactly Python's field set"
    );
    assert_eq!(request_digest(&request), request.digest());

    let mut reordered = request.clone();
    reordered.changed_skill_ids.reverse();
    assert_eq!(
        reordered.digest(),
        request.digest(),
        "id order must not change the replay identity"
    );

    let mut without_revision = request.clone();
    without_revision.source_revision = None;
    assert_eq!(
        without_revision.digest(),
        "d5a6bfd20379312a5f6a7423248b8ace6d10da06117e10e15bc233696cbb15e0",
        "an undeclared source revision hashes as JSON null"
    );

    let mut variants = Vec::new();
    let mut other = request.clone();
    other.request_id = "req-other".to_string();
    variants.push(other);
    let mut other = request.clone();
    other.operation = MutationOperation::Delete;
    variants.push(other);
    let mut other = request.clone();
    other.source_revision = Some("v3".to_string());
    variants.push(other);
    let mut other = request.clone();
    other.changed_skill_ids = vec!["tool:fetch".to_string()];
    variants.push(other);
    for variant in variants {
        assert_ne!(variant.digest(), request.digest(), "{variant:?}");
    }

    assert_eq!(
        REQUEST_SCHEMA_VERSION,
        "Symphony-skill-graph-mutation-request-v2"
    );
}

#[test]
fn change_set_validation_matches_the_declared_batch() {
    let request = request();
    let inventory = ids();
    validate_mutation_against_inventory(&request, &ids(), &inventory)
        .expect("an exact change set must validate");
    {
        let mut reordered = ids();
        reordered.reverse();
        validate_mutation_against_inventory(&request, &reordered, &inventory)
            .expect("the change set is compared as a set");
    }

    let missing = validate_mutation_against_inventory(&request, &ids()[..1], &inventory)
        .expect_err("an id that did not change must be rejected");
    assert_eq!(missing.kind(), ErrorKind::Validation);
    assert!(missing.message().contains("tool:sum"));

    let extra = validate_mutation_against_inventory(
        &request,
        &[
            "tool:fetch".to_string(),
            "tool:sum".to_string(),
            "tool:ghost".to_string(),
        ],
        &inventory,
    )
    .expect_err("an undeclared change must be rejected");
    assert!(extra.message().contains("tool:ghost"));

    let absent = validate_mutation_against_inventory(&request, &ids(), &ids()[..1])
        .expect_err("an added capability must be present afterwards");
    assert!(absent.message().contains("tool:sum"));
}

#[tokio::test]
async fn journal_round_trips_records_in_request_id_order() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("journal.json");
    let mut journal = MutationJournal::new(&path);
    assert!(journal.is_empty());
    assert_eq!(journal.len(), 0);
    assert_eq!(journal.path(), path.as_path());

    let second = JournalRecord::new(MutationOperation::Update, "v4", "digest-b")
        .with_inserted_at("2026-01-02T00:00:00Z")
        .with_result(json!({"request_id": "req-b"}));
    journal
        .record("req-b", second.clone())
        .await
        .expect("record b");
    journal
        .record(
            "req-a",
            JournalRecord::new(MutationOperation::Add, "v3", "digest-a")
                .with_inserted_at("2026-01-01T00:00:00Z"),
        )
        .await
        .expect("record a");
    assert_eq!(journal.len(), 2);
    assert!(journal.contains("req-b"));
    assert_eq!(
        journal.request_ids().collect::<Vec<_>>(),
        vec!["req-a", "req-b"]
    );

    let reopened = MutationJournal::open(&path).await.expect("reopen");
    assert_eq!(reopened.len(), 2);
    let recorded = reopened
        .lookup("req-b")
        .await
        .expect("request b is recorded");
    assert_eq!(recorded, second);
    assert!(reopened.contains("req-a"));
    assert!(
        reopened.lookup("req-missing").await.is_none(),
        "an unknown request id must not be invented"
    );

    // The document is stable: records sorted by request id, one envelope field each, and no
    // temporary sibling left behind by the atomic write.
    let text = std::fs::read_to_string(&path).expect("read journal");
    let payload: Value = serde_json::from_str(&text).expect("parse journal");
    assert_eq!(payload["schema_version"], json!(JOURNAL_SCHEMA_VERSION));
    let records = payload["records"].as_object().expect("records object");
    let keys: Vec<&String> = records.keys().collect();
    assert_eq!(keys, vec!["req-a", "req-b"]);
    assert!(
        text.find("\"req-a\"").expect("req-a") < text.find("\"req-b\"").expect("req-b"),
        "records must be written in request-id order"
    );
    assert_eq!(
        json_files(directory.path()),
        vec!["journal.json".to_string()],
        "the atomic write must not leave a temporary file behind"
    );

    // A second write of the same records is byte-identical.
    let mut again = MutationJournal::open(&path).await.expect("reopen");
    again
        .record(
            "req-a",
            JournalRecord::new(MutationOperation::Add, "v3", "digest-a")
                .with_inserted_at("2026-01-01T00:00:00Z"),
        )
        .await
        .expect("rewrite a");
    assert_eq!(std::fs::read_to_string(&path).expect("reread"), text);
}

#[tokio::test]
async fn journal_clear_empties_the_document() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("journal.json");
    let mut journal = MutationJournal::open(&path).await.expect("open");
    journal
        .record(
            "req-a",
            JournalRecord::new(MutationOperation::Add, "v1", "digest"),
        )
        .await
        .expect("record");
    journal.clear().await.expect("clear");
    assert!(journal.is_empty());

    let reopened = MutationJournal::open(&path).await.expect("reopen");
    assert!(reopened.is_empty());
    let payload: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
    assert_eq!(payload["records"], json!({}));
}

#[tokio::test]
async fn journal_rejects_malformed_documents_and_skips_malformed_records() {
    let directory = tempfile::tempdir().expect("tempdir");

    let absent = MutationJournal::open(directory.path().join("missing.json"))
        .await
        .expect("an absent journal is empty");
    assert!(absent.is_empty());

    let broken = directory.path().join("broken.json");
    tokio::fs::write(&broken, b"{not json")
        .await
        .expect("write");
    let error = MutationJournal::open(&broken)
        .await
        .expect_err("malformed JSON must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("broken.json"),
        "the error must name the file: {}",
        error.message()
    );

    let wrong_schema = directory.path().join("wrong.json");
    tokio::fs::write(
        &wrong_schema,
        serde_json::to_vec(&json!({"schema_version": "other-v9", "records": {}})).expect("json"),
    )
    .await
    .expect("write");
    let error = MutationJournal::open(&wrong_schema)
        .await
        .expect_err("an unknown schema version must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("schema_version"));

    let partial = directory.path().join("partial.json");
    tokio::fs::write(
        &partial,
        serde_json::to_vec(&json!({
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "records": {
                "req-good": {"digest": "d", "operation": "add", "revision": "v1"},
                "req-bad": 7,
            },
        }))
        .expect("json"),
    )
    .await
    .expect("write");
    let journal = MutationJournal::open(&partial)
        .await
        .expect("a malformed record is skipped, not fatal");
    assert_eq!(journal.len(), 1);
    assert!(journal.contains("req-good"));

    let mut journal = journal;
    let error = journal
        .record("  ", JournalRecord::new(MutationOperation::Add, "v1", "d"))
        .await
        .expect_err("a blank request id must not be journaled");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = MutationJournal::open(directory.path())
        .await
        .expect_err("a directory must not open as a journal");
    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Validation),
        "a directory is not a journal: {error}"
    );
}
