//! Offline tests for the `sharing` subsystem: keyword extraction, bundle ids,
//! the stager gates and both backends.
//!
//! Everything is deterministic — no clocks, no models, no network — so every
//! assertion is hand-computable from the fixtures.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty,
    clippy::match_wildcard_for_single_variants
)]
use std::collections::BTreeSet;

use openjiuwen_agent_evolving::experience::{ExperienceRecord, ExperienceSection};
use openjiuwen_agent_evolving::sharing::{
    KeywordExtractor, LocalFileSharingBackend, MemorySharingBackend, QueryKeywords, ShareStager,
    SharedExperience, SharedSkillBundle, SharingBackend, SharingMeta, SkillPackageMeta,
    SkillSearchResult, StagedBundle,
};
use openjiuwen_common::error::Error;

/// The skill every fixture record belongs to.
const SKILL: &str = "pdf-report";
/// A record about the canonical demo skill.
const CONTENT_A: &str = "raise the read timeout before parsing large pdf files";
/// A near-duplicate of [`CONTENT_A`] (Jaccard similarity 8/9).
const CONTENT_A_VARIANT: &str = "raise the read timeout before parsing large pdf files again";
/// An unrelated record about images.
const CONTENT_B: &str = "resize images with the thumbnail helper";
/// An unrelated record about archives.
const CONTENT_C: &str = "compress archives into zip volumes";
/// Another record that mentions `timeout`, for search ranking.
const CONTENT_D: &str = "network timeout while fetching the remote mirror";

/// A record with a fixed skill, keywords and confidence.
fn record(id: &str, content: &str, confidence: f64) -> ExperienceRecord {
    ExperienceRecord::new(id, SKILL, ExperienceSection::Tool, content)
        .unwrap()
        .with_keywords(["pdf", "timeout"])
        .with_confidence(confidence)
        .unwrap()
}

/// A record whose success counter is set explicitly.
fn record_with_successes(
    id: &str,
    content: &str,
    confidence: f64,
    successes: u64,
) -> ExperienceRecord {
    let mut record = record(id, content, confidence);
    record.successes = successes;
    record
}

/// Metadata for the demo skill.
fn meta() -> SharingMeta {
    SharingMeta::new(SKILL, "1.2.0")
}

/// A stager that publishes without approval.
fn stager() -> ShareStager {
    ShareStager::default().with_require_approval(false)
}

/// Builds a staged bundle through the stager.
fn stage(records: &[ExperienceRecord], stager: &ShareStager) -> StagedBundle {
    stager.build_bundle(records, &meta()).unwrap()
}

/// Builds a bundle for `meta`, with aggregates filled by the stager.
fn bundle_for(meta: &SharingMeta, records: &[ExperienceRecord]) -> SharedSkillBundle {
    stager().build_bundle(records, meta).unwrap().bundle
}

/// Wraps `record` for direct bundle construction.
fn wrapped(record: ExperienceRecord, meta: &SharingMeta) -> SharedExperience {
    SharedExperience::new(record, vec!["pdf".to_owned()], "summary", meta.clone())
}

#[test]
fn extracts_english_keywords_by_frequency_then_alphabetically() {
    let extractor = KeywordExtractor::new();
    assert_eq!(
        extractor.extract("timeout parse timeout file parse timeout"),
        vec!["timeout", "parse", "file"]
    );
}

#[test]
fn keeps_underscored_identifiers_and_drops_stopwords() {
    let extractor = KeywordExtractor::new();
    assert_eq!(
        extractor.extract("read_file the and read_file"),
        vec!["read_file"]
    );
    assert!(extractor.extract("the and of to").is_empty());
}

#[test]
fn extracts_cjk_characters_and_filters_chinese_stopwords() {
    let extractor = KeywordExtractor::new();
    assert_eq!(
        extractor.extract("文件读取的失败 文件"),
        vec!["件", "文", "取", "失", "读", "败"]
    );
}

#[test]
fn honours_max_keywords_and_min_len() {
    let extractor = KeywordExtractor::new().with_max_keywords(2);
    assert_eq!(
        extractor.extract("timeout parse file retry"),
        vec!["file", "parse"]
    );

    let unlimited = KeywordExtractor::new().with_max_keywords(0);
    assert_eq!(unlimited.extract("timeout parse file").len(), 3);

    assert!(KeywordExtractor::new().extract("go go").is_empty());
    let shorter = KeywordExtractor::new().with_min_len(2);
    assert_eq!(shorter.extract("go go"), vec!["go"]);
}

#[test]
fn accepts_a_custom_stopword_set() {
    let stopwords = BTreeSet::from(["timeout".to_owned()]);
    let extractor = KeywordExtractor::new().with_stopwords(stopwords);
    assert_eq!(extractor.extract("timeout parse"), vec!["parse"]);
}

#[test]
fn summarise_returns_short_text_trimmed() {
    let extractor = KeywordExtractor::new();
    assert_eq!(extractor.summarise("  Short note  ", 32), "Short note");
}

#[test]
fn summarise_stops_at_a_sentence_boundary() {
    let extractor = KeywordExtractor::new();
    assert_eq!(
        extractor.summarise("First sentence. Second sentence is long.", 20),
        "First sentence."
    );
    assert_eq!(extractor.summarise("One. Two. Three.", 11), "One. Two.");
    assert_eq!(extractor.summarise("你好。世界。", 5), "你好。");
    assert_eq!(extractor.summarise("Alpha\nBeta", 8), "Alpha");
}

#[test]
fn summarise_hard_truncates_to_exactly_max_chars() {
    let extractor = KeywordExtractor::new();
    let ascii = extractor.summarise("abcdefghij", 5);
    assert_eq!(ascii, "abcd…");
    assert_eq!(ascii.chars().count(), 5);

    let cjk = extractor.summarise("文件读取失败的问题", 4);
    assert_eq!(cjk, "文件读…");
    assert_eq!(cjk.chars().count(), 4);

    let sentence = extractor.summarise("A very long sentence without terminator", 10);
    assert_eq!(sentence, "A very lo…");
    assert_eq!(sentence.chars().count(), 10);
}

#[test]
fn summarise_handles_zero_and_one_char_budgets() {
    let extractor = KeywordExtractor::new();
    assert_eq!(extractor.summarise("anything at all", 0), "");
    assert_eq!(extractor.summarise("anything at all", 1), "…");
}

#[test]
fn aggregates_keywords_once_per_text() {
    let extractor = KeywordExtractor::new();
    let texts = vec!["timeout parse".to_owned(), "timeout retry".to_owned()];
    assert_eq!(
        extractor.aggregate(&texts),
        vec!["timeout", "parse", "retry"]
    );

    let repeated = vec!["timeout timeout".to_owned(), "parse".to_owned()];
    assert_eq!(extractor.aggregate(&repeated), vec!["parse", "timeout"]);
}

#[test]
fn bundle_ids_are_deterministic_and_order_independent() {
    let meta = meta();
    let first = record("exp-1", CONTENT_A, 0.9);
    let second = record("exp-2", CONTENT_B, 0.9);
    let forward = SharedSkillBundle::new(
        SKILL,
        SKILL,
        "1.0",
        vec![
            wrapped(first.clone(), &meta),
            wrapped(second.clone(), &meta),
        ],
    )
    .unwrap();
    let reversed = SharedSkillBundle::new(
        SKILL,
        SKILL,
        "1.0",
        vec![wrapped(second, &meta), wrapped(first, &meta)],
    )
    .unwrap();
    assert_eq!(forward.bundle_id, reversed.bundle_id);
    assert!(forward.bundle_id.starts_with("bundle-"));
    assert_eq!(forward.bundle_id.len(), 23);
    assert_eq!(forward.record_ids(), vec!["exp-1", "exp-2"]);
    assert_eq!(forward.len(), 2);
    assert!(!forward.is_empty());
}

#[test]
fn different_record_sets_get_different_bundle_ids() {
    let meta = meta();
    let one = SharedSkillBundle::new(
        SKILL,
        SKILL,
        "1.0",
        vec![wrapped(record("exp-1", CONTENT_A, 0.9), &meta)],
    )
    .unwrap();
    let two = SharedSkillBundle::new(
        SKILL,
        SKILL,
        "1.0",
        vec![
            wrapped(record("exp-1", CONTENT_A, 0.9), &meta),
            wrapped(record("exp-2", CONTENT_B, 0.9), &meta),
        ],
    )
    .unwrap();
    assert_ne!(one.bundle_id, two.bundle_id);
}

#[test]
fn bundle_rejects_a_blank_skill_id() {
    let error = SharedSkillBundle::new("  ", SKILL, "1.0", Vec::new()).unwrap_err();
    assert!(matches!(error, Error::Validation(_)));

    let empty = SharedSkillBundle::new(SKILL, SKILL, "", Vec::new()).unwrap();
    assert!(empty.is_empty());
    assert!(empty.record_ids().is_empty());
}

#[test]
fn stager_filters_records_below_min_confidence() {
    let records = [
        record("keep", CONTENT_A, 0.9),
        record("drop", CONTENT_B, 0.4),
    ];
    let staged = stage(&records, &stager());
    assert_eq!(staged.bundle.len(), 1);
    assert_eq!(staged.bundle.experiences[0].record.record_id, "keep");
    assert_eq!(staged.skipped, vec!["drop"]);
    assert!(!staged.requires_approval);
}

#[test]
fn stager_dedupes_near_duplicates_in_favour_of_confidence() {
    let records = [
        record("low", CONTENT_A, 0.8),
        record("high", CONTENT_A_VARIANT, 0.95),
    ];
    let staged = stage(&records, &stager());
    assert_eq!(staged.bundle.len(), 1);
    assert_eq!(staged.bundle.experiences[0].record.record_id, "high");
    assert_eq!(staged.skipped, vec!["low"]);
}

#[test]
fn dedupe_prefers_more_successes_then_the_smaller_id() {
    let records = [
        record_with_successes("few", CONTENT_A, 0.9, 1),
        record_with_successes("many", CONTENT_A, 0.9, 3),
    ];
    let staged = stage(&records, &stager());
    assert_eq!(staged.bundle.experiences[0].record.record_id, "many");
    assert_eq!(staged.skipped, vec!["few"]);

    let records = [
        record("beta", CONTENT_A, 0.9),
        record("alpha", CONTENT_A, 0.9),
    ];
    let staged = stage(&records, &stager());
    assert_eq!(staged.bundle.experiences[0].record.record_id, "alpha");
    assert_eq!(staged.skipped, vec!["beta"]);
}

#[test]
fn stager_caps_experiences_and_reports_the_dropped_ids() {
    let records = [
        record("a", CONTENT_A, 0.9),
        record("b", CONTENT_B, 0.8),
        record("c", CONTENT_C, 0.7),
    ];
    let capped = stager().with_max_experiences(2);
    let staged = stage(&records, &capped);
    assert_eq!(staged.bundle.len(), 2);
    assert_eq!(staged.bundle.record_ids(), vec!["a", "b"]);
    assert_eq!(staged.skipped, vec!["c"]);

    let unlimited = stager().with_max_experiences(0);
    let staged = stage(&records, &unlimited);
    assert_eq!(staged.bundle.len(), 3);
    assert!(staged.skipped.is_empty());
}

#[test]
fn stager_reports_invalid_and_low_confidence_skips_sorted() {
    let mut invalid = record("alpha", CONTENT_A, 0.9);
    invalid.content = "   ".to_owned();
    let records = [
        record("zeta", CONTENT_B, 0.1),
        invalid,
        record("middle", CONTENT_C, 0.2),
        record("keep", CONTENT_D, 0.9),
    ];
    let staged = stage(&records, &stager());
    assert_eq!(staged.bundle.record_ids(), vec!["keep"]);
    assert_eq!(staged.skipped, vec!["alpha", "middle", "zeta"]);
}

#[test]
fn stager_rejects_a_bundle_with_no_surviving_experience() {
    let records = [record("low", CONTENT_A, 0.1)];
    let error = stager().build_bundle(&records, &meta()).unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
    assert!(error.to_string().contains("no shareable experiences"));
}

#[test]
fn stager_rejects_a_blank_skill_name_in_meta() {
    let records = [record("exp-1", CONTENT_A, 0.9)];
    let error = stager()
        .build_bundle(&records, &SharingMeta::new("  ", "1.0"))
        .unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
}

#[test]
fn stager_fills_bundle_metadata_and_experience_summaries() {
    let staged = stage(&[record("exp-1", CONTENT_A, 0.9)], &stager());
    assert_eq!(staged.bundle.skill_id, SKILL);
    assert_eq!(staged.bundle.skill_name, SKILL);
    assert_eq!(staged.bundle.skill_version, "1.2.0");
    assert!(!staged.bundle.keywords_aggregate.is_empty());
    assert!(!staged.bundle.summary_aggregate.is_empty());
    let experience = &staged.bundle.experiences[0];
    assert_eq!(experience.record.record_id, "exp-1");
    assert_eq!(experience.sharing_meta, meta());
    assert!(!experience.keywords.is_empty());
    assert!(experience.summary.chars().count() <= 160);
}

#[test]
fn sharing_meta_builders_and_validation() {
    let meta = SharingMeta::new(SKILL, "1.2")
        .with_confidence(0.9)
        .unwrap()
        .with_source_user("user-1")
        .with_feedback("great tip")
        .with_keywords(["timeout", "pdf", "timeout"]);
    assert_eq!(meta.source_user_id.as_deref(), Some("user-1"));
    assert_eq!(meta.feedback_excerpt.as_deref(), Some("great tip"));
    assert_eq!(meta.keywords, vec!["pdf", "timeout"]);
    assert_eq!(meta.upload_trigger, "user_approval");
    meta.validate().unwrap();

    assert!(SharingMeta::new("", "").validate().is_err());
    let mut blank_trigger = SharingMeta::new(SKILL, "");
    blank_trigger.upload_trigger = " ".to_owned();
    assert!(blank_trigger.validate().is_err());
    assert!(SharingMeta::new(SKILL, "").with_confidence(1.5).is_err());
}

#[tokio::test]
async fn publish_and_share_honour_the_approval_gate() {
    let stager = ShareStager::default();
    let records = [record("exp-1", CONTENT_A, 0.9)];
    let prepared = stager.build_bundle(&records, &meta()).unwrap();
    assert!(prepared.requires_approval);

    let backend = MemorySharingBackend::new();
    let error = stager
        .publish(&backend, &prepared, false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("requires approval"));
    assert!(backend.list().await.unwrap().is_empty());

    let id = stager.publish(&backend, &prepared, true).await.unwrap();
    assert_eq!(id, prepared.bundle.bundle_id);
    assert!(backend.get(&id).await.unwrap().is_some());
}

#[tokio::test]
async fn share_returns_none_until_approved() {
    let stager = ShareStager::default();
    let records = [record("exp-1", CONTENT_A, 0.9)];
    let backend = MemorySharingBackend::new();

    assert_eq!(
        stager
            .share(&backend, &records, &meta(), false)
            .await
            .unwrap(),
        None
    );
    let id = stager
        .share(&backend, &records, &meta(), true)
        .await
        .unwrap()
        .unwrap();
    assert!(backend.get(&id).await.unwrap().is_some());
    assert_eq!(backend.list().await.unwrap().len(), 1);
}

#[tokio::test]
async fn share_publishes_without_approval_when_gate_is_disabled() {
    let records = [record("exp-1", CONTENT_A, 0.9)];
    let backend = MemorySharingBackend::new();
    let id = stager()
        .share(&backend, &records, &meta(), false)
        .await
        .unwrap()
        .unwrap();
    assert!(backend.get(&id).await.unwrap().is_some());
}

#[tokio::test]
async fn local_file_backend_publish_list_get_remove() {
    let dir = tempfile::tempdir().unwrap();
    let backend = LocalFileSharingBackend::new(dir.path().join("bundles"));
    assert_eq!(backend.dir().to_path_buf(), dir.path().join("bundles"));

    let first = stage(
        &[
            record("exp-1", CONTENT_A, 0.9),
            record("exp-2", CONTENT_B, 0.9),
        ],
        &stager(),
    )
    .bundle;
    let second = stage(&[record("exp-3", CONTENT_C, 0.9)], &stager()).bundle;

    let first_id = backend.publish(&first).await.unwrap();
    let second_id = backend.publish(&second).await.unwrap();
    assert_eq!(first_id, first.bundle_id);
    assert_eq!(backend.get(&first_id).await.unwrap(), Some(first));

    let ids: Vec<String> = backend
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|bundle| bundle.bundle_id)
        .collect();
    let mut expected = vec![first_id.clone(), second_id];
    expected.sort();
    assert_eq!(ids, expected);

    assert!(backend.remove(&first_id).await.unwrap());
    assert_eq!(backend.get(&first_id).await.unwrap(), None);
    assert!(!backend.remove(&first_id).await.unwrap());
}

#[tokio::test]
async fn local_file_backend_treats_a_missing_directory_as_empty() {
    let dir = tempfile::tempdir().unwrap();
    let backend = LocalFileSharingBackend::new(dir.path().join("absent"));
    assert!(backend.list().await.unwrap().is_empty());
    assert!(backend
        .get("bundle-0000000000000000")
        .await
        .unwrap()
        .is_none());
    assert!(!backend.remove("bundle-0000000000000000").await.unwrap());
}

#[tokio::test]
async fn local_file_backend_errors_on_malformed_json() {
    let dir = tempfile::tempdir().unwrap();
    let backend = LocalFileSharingBackend::new(dir.path());
    tokio::fs::write(dir.path().join("bundle-bad.json"), "{not json")
        .await
        .unwrap();
    assert!(backend.list().await.is_err());
    assert!(backend.get("bundle-bad").await.is_err());
}

#[tokio::test]
async fn local_file_search_ranks_filters_and_limits() {
    let dir = tempfile::tempdir().unwrap();
    let backend = LocalFileSharingBackend::new(dir.path().join("bundles"));
    let pdf = bundle_for(
        &SharingMeta::new("pdf-report", "1.0"),
        &[record("exp-pdf", CONTENT_A, 0.9)],
    );
    let network = bundle_for(
        &SharingMeta::new("network-tool", "1.0"),
        &[record("exp-net", CONTENT_D, 0.9)],
    );
    let image = bundle_for(
        &SharingMeta::new("image-tool", "1.0"),
        &[record("exp-img", CONTENT_B, 0.9)],
    );
    backend.publish(&pdf).await.unwrap();
    backend.publish(&network).await.unwrap();
    backend.publish(&image).await.unwrap();

    let query = QueryKeywords::new(vec!["timeout".to_owned()], "", "");
    let all = backend.search(&query, 0).await.unwrap();
    let mut expected = vec![pdf.bundle_id.clone(), network.bundle_id.clone()];
    expected.sort();
    let ids: Vec<String> = all.iter().map(|result| result.bundle_id.clone()).collect();
    assert_eq!(ids, expected);

    let limited = backend.search(&query, 1).await.unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].bundle_id, expected[0]);

    let thumbnail = QueryKeywords::new(vec!["thumbnail".to_owned()], "", "");
    let results = backend.search(&thumbnail, 0).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].bundle_id, image.bundle_id);
    assert_eq!(results[0].description, image.summary_aggregate);
    assert_eq!(results[0].experience_count, 1);
    assert_eq!(results[0].keywords, image.keywords_aggregate);
}

#[tokio::test]
async fn memory_backend_matches_file_semantics() {
    let backend = MemorySharingBackend::new();
    let empty = MemorySharingBackend::default();
    assert!(empty.list().await.unwrap().is_empty());
    assert!(backend
        .search(&QueryKeywords::default(), 0)
        .await
        .unwrap()
        .is_empty());

    let first = bundle_for(
        &SharingMeta::new("pdf-report", "1.0"),
        &[record("exp-pdf", CONTENT_A, 0.9)],
    );
    let second = bundle_for(
        &SharingMeta::new("image-tool", "1.0"),
        &[record("exp-img", CONTENT_B, 0.9)],
    );
    assert_eq!(backend.publish(&first).await.unwrap(), first.bundle_id);
    backend.publish(&second).await.unwrap();
    assert_eq!(
        backend.get(&first.bundle_id).await.unwrap(),
        Some(first.clone())
    );

    let ids: Vec<String> = backend
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|bundle| bundle.bundle_id)
        .collect();
    let mut expected = vec![first.bundle_id.clone(), second.bundle_id.clone()];
    expected.sort();
    assert_eq!(ids, expected);

    let query = QueryKeywords::new(vec!["timeout".to_owned()], "", "");
    let results = backend.search(&query, 0).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].bundle_id, first.bundle_id);
    assert_eq!(backend.search(&query, 1).await.unwrap().len(), 1);

    assert!(backend.remove(&first.bundle_id).await.unwrap());
    assert!(backend.get(&first.bundle_id).await.unwrap().is_none());
    assert!(!backend.remove(&first.bundle_id).await.unwrap());
}

#[test]
fn sharing_meta_serde_round_trip_and_unknown_fields() {
    let meta = SharingMeta::new(SKILL, "1.2")
        .with_confidence(0.9)
        .unwrap()
        .with_source_user("user-1")
        .with_feedback("great tip")
        .with_keywords(["pdf", "timeout"]);
    let json = serde_json::to_string(&meta).unwrap();
    let decoded: SharingMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, meta);

    let tolerant: SharingMeta =
        serde_json::from_str(r#"{"skill_name":"s","unknown_field":42}"#).unwrap();
    assert_eq!(tolerant.skill_name, "s");
    assert_eq!(tolerant.upload_trigger, "user_approval");
    assert!((tolerant.confidence - 0.7).abs() < f64::EPSILON);
}

#[test]
fn shared_experience_serde_round_trip() {
    let experience = SharedExperience::new(
        record("exp-1", CONTENT_A, 0.9),
        vec!["pdf".to_owned(), "timeout".to_owned()],
        "raise the timeout",
        meta(),
    );
    let json = serde_json::to_string(&experience).unwrap();
    let decoded: SharedExperience = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, experience);
}

#[test]
fn shared_skill_bundle_serde_round_trip() {
    let bundle = stage(&[record("exp-1", CONTENT_A, 0.9)], &stager()).bundle;
    let json = serde_json::to_string(&bundle).unwrap();
    let decoded: SharedSkillBundle = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(decoded.record_ids(), vec!["exp-1"]);
}

#[test]
fn query_keywords_serde_and_emptiness() {
    let query = QueryKeywords::new(vec!["timeout".to_owned()], "parse pdf", "raw excerpt");
    assert!(!query.is_empty());
    let json = serde_json::to_string(&query).unwrap();
    let decoded: QueryKeywords = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, query);

    assert!(QueryKeywords::default().is_empty());
    assert!(QueryKeywords::new(vec!["  ".to_owned()], "  ", "raw").is_empty());
}

#[test]
fn skill_search_result_and_package_meta_serde() {
    let result = SkillSearchResult {
        bundle_id: "bundle-0000000000000001".to_owned(),
        skill_id: SKILL.to_owned(),
        skill_name: SKILL.to_owned(),
        description: "summary".to_owned(),
        experience_count: 2,
        keywords: vec!["pdf".to_owned()],
        score: 0.7,
    };
    let json = serde_json::to_string(&result).unwrap();
    let decoded: SkillSearchResult = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, result);

    let package = SkillPackageMeta::new(SKILL);
    assert_eq!(package.skill_name, "");
    let json = serde_json::to_string(&package).unwrap();
    let decoded: SkillPackageMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, package);

    let tolerant: SkillPackageMeta =
        serde_json::from_str(r#"{"skill_id":"pdf","extra":true}"#).unwrap();
    assert_eq!(tolerant.skill_id, "pdf");
}
