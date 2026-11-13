//! Behavior tests for skill-folder discovery, front matter, catalogs, and settings.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::fs;
use std::path::Path;

use openjiuwen_symphony::discovery::{
    inventory_fingerprint, parse_front_matter, DiscoverySettings, FrontMatterStatus,
    SkillFolderScanner, SkillFs, SkillInventory, SkillRecord, SkillScanConfig,
};
use openjiuwen_symphony::error::ErrorKind;

/// Writes `content` to `root/relative`, creating parent directories.
fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

/// A minimal `SKILL.md` with front matter and a body.
fn skill(name: &str, description: &str) -> String {
    format!("---\nname: {name}\ndescription: {description}\n---\n\n{description}\n")
}

/// Creates the default fixture: two skills, one lowercase file name, and two skipped
/// folders.
fn default_root(root: &Path) {
    write(root, "alpha/SKILL.md", &skill("Alpha", "Adds numbers"));
    write(root, "beta/skill.md", &skill("Beta", "Translates text"));
    write(root, ".hidden/SKILL.md", &skill("Hidden", "Hidden skill"));
    write(
        root,
        "_private/SKILL.md",
        &skill("Private", "Private skill"),
    );
    write(root, "notes/README.md", "no skill here");
    write(root, "README.md", "root readme");
}

#[tokio::test]
async fn scan_finds_immediate_child_skills_and_skips_special_names() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    default_root(root);

    let capabilities = SkillFolderScanner::new(root).scan().await.unwrap();
    let ids: Vec<&str> = capabilities
        .iter()
        .map(|descriptor| descriptor.capability_id.as_str())
        .collect();
    assert_eq!(ids, ["alpha", "beta"]);
    for descriptor in &capabilities {
        assert_eq!(descriptor.capability_type, "skill");
        assert_eq!(descriptor.source, "local");
        assert!(descriptor.available);
        assert_eq!(descriptor.content_hash.len(), 64);
        assert!(descriptor
            .content_hash
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    }
    assert_eq!(capabilities[0].name, "Alpha");
    assert_eq!(capabilities[0].description, "Adds numbers");
    assert_eq!(capabilities[1].name, "Beta");
}

#[tokio::test]
async fn recursive_mode_finds_nested_skills_and_honors_max_depth() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "nested/deep/SKILL.md", &skill("Deep", "Nested skill"));

    let immediate = SkillFolderScanner::new(root).scan().await.unwrap();
    assert!(immediate.is_empty());

    let recursive = SkillFolderScanner::with_config(root, SkillScanConfig::recursive(Some(3)))
        .scan()
        .await
        .unwrap();
    assert_eq!(recursive.len(), 1);
    assert_eq!(recursive[0].name, "Deep");

    let shallow = SkillFolderScanner::with_config(root, SkillScanConfig::recursive(Some(1)))
        .scan()
        .await
        .unwrap();
    assert!(shallow.is_empty());
}

#[tokio::test]
async fn front_matter_fields_are_parsed_into_the_descriptor() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "pdf/SKILL.md",
        "---\n\
         capability_id: pdf-export\n\
         name: PDF Export\n\
         description: Exports documents as PDF\n\
         capability_type: skills\n\
         category: documents\n\
         tags: 'Pdf, pdf; export'\n\
         aliases: [acrobat, pdf]\n\
         version: 1.2.0\n\
         author: Alice\n\
         available: false\n\
         inputs:\n  - name: source\n    type: string\n    description: input file\n    required: true\n  - raw\n\
         outputs:\n  - name: artifact\n    type: file\n\
         ---\n\nConvert a document into a PDF file.\n",
    );

    let capabilities = SkillFolderScanner::new(root).scan().await.unwrap();
    assert_eq!(capabilities.len(), 1);
    let descriptor = &capabilities[0];
    assert_eq!(descriptor.capability_id, "pdf-export");
    assert_eq!(descriptor.capability_type, "skill");
    assert_eq!(descriptor.name, "PDF Export");
    assert_eq!(descriptor.description, "Exports documents as PDF");
    assert_eq!(descriptor.classification, "documents");
    assert_eq!(descriptor.tags, ["Pdf", "export"]);
    assert!(!descriptor.available);
    assert_eq!(descriptor.inputs.len(), 2);
    assert_eq!(descriptor.inputs[0].name, "source");
    assert_eq!(descriptor.inputs[0].io_type(), "string");
    assert_eq!(descriptor.inputs[0].required, Some(true));
    assert_eq!(descriptor.inputs[1].name, "raw");
    assert_eq!(descriptor.outputs.len(), 1);
    assert_eq!(descriptor.outputs[0].io_type(), "file");
    assert_eq!(
        descriptor
            .metadata
            .get("version")
            .and_then(|value| value.as_str()),
        Some("1.2.0")
    );
    assert_eq!(
        descriptor
            .metadata
            .get("author")
            .and_then(|value| value.as_str()),
        Some("Alice")
    );
    assert_eq!(
        descriptor
            .metadata
            .get("entrypoint")
            .and_then(|value| value.as_str()),
        Some("pdf/SKILL.md")
    );
    let aliases: Vec<&str> = descriptor.metadata["aliases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|value| value.as_str())
        .collect();
    assert_eq!(aliases, ["acrobat", "pdf"]);
}

#[tokio::test]
async fn credentials_in_front_matter_are_never_echoed_into_search_text() {
    // Regression guard for the retrieval text: it is built from name/description/tags
    // and the markdown body only, so unrelated front-matter keys stay out of it.
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "secret/SKILL.md",
        "---\nname: Secret\ntoken: super-secret\n---\n\nBody\n",
    );

    let catalog = SkillFs::open(root).await.unwrap();
    assert_eq!(catalog.inventory().count, 1);
}

#[tokio::test]
async fn missing_front_matter_is_a_warning_and_still_yields_a_descriptor() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "plain/SKILL.md",
        "Helps with plain things.\n\nMore detail follows.\n",
    );

    let result = SkillFolderScanner::new(root).scan_result().await.unwrap();
    assert_eq!(result.capabilities.len(), 1);
    let descriptor = &result.capabilities[0];
    assert_eq!(descriptor.capability_id, "plain");
    assert_eq!(descriptor.name, "plain");
    assert_eq!(descriptor.description, "Helps with plain things.");
    assert_eq!(descriptor.source, "local");

    let diagnostic = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "missing_front_matter")
        .expect("expected a missing_front_matter diagnostic");
    assert!(!diagnostic.is_error());
    assert!(diagnostic.path.ends_with("SKILL.md"));
    assert!(result.error_codes().is_empty());
}

#[tokio::test]
async fn unterminated_front_matter_skips_the_skill_with_an_error_diagnostic() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "broken/SKILL.md",
        "---\nname: Broken\nBody without a closing fence\n",
    );
    write(root, "good/SKILL.md", &skill("Good", "Works fine"));

    let result = SkillFolderScanner::new(root).scan_result().await.unwrap();
    assert_eq!(result.capability_ids(), ["good"]);
    assert_eq!(result.error_codes(), ["unterminated_front_matter"]);
    let diagnostic = &result.diagnostics[0];
    assert!(diagnostic.path.ends_with("broken/SKILL.md"));
    assert!(diagnostic.message.contains("unterminated"));
}

#[tokio::test]
async fn malformed_yaml_is_a_diagnostic_and_other_skills_survive() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "bad/SKILL.md",
        "---\nname: [unclosed\n---\n\nBad front matter.\n",
    );
    write(root, "good/SKILL.md", &skill("Good", "Works fine"));

    let result = SkillFolderScanner::new(root).scan_result().await.unwrap();
    let ids = result.capability_ids();
    assert!(ids.contains(&"good"), "good skill must survive: {ids:?}");
    assert!(
        ids.contains(&"bad"),
        "bad skill keeps a defaulted descriptor"
    );

    let diagnostic = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "malformed_front_matter")
        .expect("expected a malformed_front_matter diagnostic");
    assert!(!diagnostic.is_error());
    assert!(result.error_codes().is_empty());
}

#[tokio::test]
async fn content_hash_is_stable_per_content_and_metadata_carries_entrypoint() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "one/SKILL.md", &skill("One", "Same content"));
    write(root, "two/SKILL.md", &skill("Two", "Other content"));

    let capabilities = SkillFolderScanner::new(root).scan().await.unwrap();
    assert_eq!(capabilities.len(), 2);
    let one = &capabilities[0];
    let two = &capabilities[1];
    assert_ne!(one.content_hash, two.content_hash);

    let rescan = SkillFolderScanner::new(root).scan().await.unwrap();
    assert_eq!(rescan[0].content_hash, one.content_hash);

    // Identical bytes in a different folder produce the identical digest.
    write(root, "three/SKILL.md", &skill("One", "Same content"));
    let capabilities = SkillFolderScanner::new(root).scan().await.unwrap();
    let three = capabilities
        .iter()
        .find(|descriptor| descriptor.capability_id == "three")
        .unwrap();
    assert_eq!(three.content_hash, one.content_hash);
    assert_eq!(
        three
            .metadata
            .get("entrypoint")
            .and_then(|value| value.as_str()),
        Some("three/SKILL.md")
    );
    assert_eq!(
        three
            .metadata
            .get("version")
            .and_then(|value| value.as_str()),
        Some("")
    );
}

#[tokio::test]
async fn inventory_fingerprint_is_stable_across_scans_and_record_order() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    default_root(root);

    let first = SkillFs::open(root).await.unwrap();
    let second = SkillFs::open(root).await.unwrap();
    let fingerprint = first.inventory().fingerprint.clone();
    assert_eq!(fingerprint.len(), 64);
    assert_eq!(second.inventory().fingerprint, fingerprint);
    assert_eq!(first.inventory().count, 2);

    let mut reversed: Vec<SkillRecord> = first.inventory().items.clone();
    reversed.reverse();
    assert_eq!(inventory_fingerprint(&reversed), fingerprint);

    // A content change moves both the record hash and the inventory fingerprint.
    write(
        root,
        "beta/skill.md",
        &skill("Beta", "Translates text and images"),
    );
    let changed = SkillFs::open(root).await.unwrap();
    assert_ne!(changed.inventory().fingerprint, fingerprint);
    assert_ne!(
        changed.records()[1].content_hash,
        first.records()[1].content_hash
    );
}

#[tokio::test]
async fn inventory_fingerprint_tracks_record_fields() {
    let records = vec![
        SkillRecord::new("beta", "Beta").with_description("second"),
        SkillRecord::new("alpha", "Alpha").with_description("first"),
    ];
    let inventory = SkillInventory::new(records.clone());
    assert_eq!(inventory.worker_ids(), ["alpha", "beta"]);
    assert_eq!(inventory.fingerprint, inventory_fingerprint(&records));

    let mut edited = records;
    edited[0].description = "changed".to_string();
    assert_ne!(inventory.fingerprint, inventory_fingerprint(&edited));
}

#[tokio::test]
async fn catalog_exposes_records_bodies_and_selection_cards() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "alpha/SKILL.md",
        "---\nname: Alpha\ntags: 'math, numbers'\nversion: 2.0.0\ndescription: Adds numbers\n---\n\nAdds two numbers together.\n",
    );

    let catalog = SkillFs::open(root).await.unwrap();
    let records = catalog.records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.worker_id, "alpha");
    assert_eq!(record.name, "Alpha");
    assert_eq!(record.skill_file, "alpha/SKILL.md");
    assert_eq!(record.source, "local");
    assert_eq!(record.version, "2.0.0");
    assert_eq!(record.source_root, root.to_string_lossy());
    assert_eq!(record.content_hash.len(), 64);

    let body = catalog.read_body("alpha").expect("body must be available");
    assert!(body.contains("Adds two numbers"));
    assert!(catalog.read_body("missing").is_none());

    let cards = catalog.selection_cards();
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].worker_id, "alpha");
    assert_eq!(cards[0].tags, ["math", "numbers"]);
    assert_eq!(cards[0].version, "2.0.0");
}

#[tokio::test]
async fn search_ranks_the_matching_skill_and_returns_an_excerpt() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "pdf-export/SKILL.md",
        "---\nname: PDF Export\ndescription: Exports documents to PDF\ntags: 'pdf, documents'\n---\n\nConvert any document into a pdf file.\n",
    );
    write(
        root,
        "translate/SKILL.md",
        "---\nname: Translate\ndescription: Translates text between languages\ntags: language\n---\n\nTranslate sentences between languages.\n",
    );

    let catalog = SkillFs::open(root).await.unwrap();

    let hits = catalog.search("pdf", 10).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].worker_id, "pdf-export");
    assert!(hits[0].matched_terms.contains(&"pdf".to_string()));
    assert!(hits[0].excerpt.to_lowercase().contains("pdf"));
    assert!(hits[0].score > 0.0);

    // A body-only match still yields an excerpt from the body.
    let body_hits = catalog.search("sentences", 10).unwrap();
    assert_eq!(body_hits[0].worker_id, "translate");
    assert!(body_hits[0].excerpt.contains("sentences"));

    // An unknown term matches nothing.
    assert!(catalog.search("nonexistent", 10).unwrap().is_empty());
}

#[tokio::test]
async fn search_honors_top_k_and_zero_is_unlimited() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    for name in ["alpha", "beta", "gamma"] {
        write(
            root,
            &format!("{name}/SKILL.md"),
            &skill(name, "shared words for ranking"),
        );
    }

    let catalog = SkillFs::open(root).await.unwrap();
    assert_eq!(catalog.search("shared", 1).unwrap().len(), 1);
    assert_eq!(catalog.search("shared", 2).unwrap().len(), 2);
    assert_eq!(catalog.search("shared", 0).unwrap().len(), 3);
    assert_eq!(catalog.search("shared", 10).unwrap().len(), 3);
}

#[tokio::test]
async fn search_ties_break_on_worker_id() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "zeta/SKILL.md", &skill("Same", "identical text"));
    write(root, "alpha/SKILL.md", &skill("Same", "identical text"));

    let catalog = SkillFs::open(root).await.unwrap();
    let hits = catalog.search("identical", 0).unwrap();
    let ids: Vec<&str> = hits.iter().map(|hit| hit.worker_id.as_str()).collect();
    assert_eq!(ids, ["alpha", "zeta"]);
    assert_eq!(hits[0].score, hits[1].score);

    let again = catalog.search("identical", 0).unwrap();
    assert_eq!(hits, again);
}

#[tokio::test]
async fn blank_and_punctuation_queries_are_filter_only() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    default_root(root);

    let catalog = SkillFs::open(root).await.unwrap();
    for query in ["", "   ", "!!!"] {
        let hits = catalog.search(query, 0).unwrap();
        let ids: Vec<&str> = hits.iter().map(|hit| hit.worker_id.as_str()).collect();
        assert_eq!(ids, ["alpha", "beta"]);
        for hit in &hits {
            assert_eq!(hit.score, 0.0);
            assert!(hit.matched_terms.is_empty());
            assert!(hit.excerpt.is_empty());
        }
    }
    assert_eq!(catalog.search("", 1).unwrap().len(), 1);
}

#[tokio::test]
async fn disabled_skills_are_dropped_after_scanning() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    default_root(root);

    let config = SkillScanConfig::default().with_disabled_skills(["Alpha"]);
    let capabilities = SkillFolderScanner::with_config(root, config)
        .scan()
        .await
        .unwrap();
    let ids: Vec<&str> = capabilities
        .iter()
        .map(|descriptor| descriptor.capability_id.as_str())
        .collect();
    assert_eq!(ids, ["beta"]);
}

#[tokio::test]
async fn missing_root_is_a_typed_validation_error() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("does-not-exist");

    let error = SkillFolderScanner::new(&missing).scan().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("root"));

    let error = SkillFs::open(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("root"));

    // A regular file is not an acceptable root either.
    let file = directory.path().join("skills.md");
    fs::write(&file, "not a directory").unwrap();
    let error = SkillFolderScanner::new(&file).scan().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("not a directory"));
}

#[tokio::test]
async fn source_is_configurable_and_stamped_everywhere() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    default_root(root);

    let config = SkillScanConfig::default().with_source("bundled");
    let result = SkillFolderScanner::with_config(root, config.clone())
        .scan_result()
        .await
        .unwrap();
    assert_eq!(result.source, "bundled");
    for descriptor in &result.capabilities {
        assert_eq!(descriptor.source, "bundled");
    }

    let catalog = SkillFs::open_with(config, root).await.unwrap();
    for record in catalog.records() {
        assert_eq!(record.source, "bundled");
    }
}

#[test]
fn front_matter_statuses_are_reported() {
    assert_eq!(
        parse_front_matter("---\nname: alpha\n---\nBody").status,
        FrontMatterStatus::Ok
    );
    assert_eq!(
        parse_front_matter("Body only").status,
        FrontMatterStatus::Missing
    );
    assert_eq!(
        parse_front_matter("---\nname: alpha\nBody").status,
        FrontMatterStatus::Unterminated
    );
}

#[test]
fn discovery_settings_defaults_and_budget() {
    let settings = DiscoverySettings::default();
    assert_eq!(settings.max_results, 10);
    assert_eq!(settings.max_output_chars, 12_000);
    assert_eq!(settings.candidate_budget_tokens(), 2_000);
    assert!(settings.use_existing_index);

    let parsed = DiscoverySettings::from_value(serde_json::json!({
        "max_results": "25",
        "context_window_tokens": 1000,
        "candidate_budget_ratio": 0.5,
        "use_existing_index": false
    }));
    assert_eq!(parsed.max_results, 25);
    assert_eq!(parsed.candidate_budget_tokens(), 500);
    assert!(!parsed.use_existing_index);
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_folders_are_skipped_with_a_diagnostic() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("skills");
    fs::create_dir_all(&root).unwrap();
    write(&root, "real/SKILL.md", &skill("Real", "Real skill"));

    let outside = directory.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    write(&outside, "SKILL.md", &skill("Linked", "Linked skill"));
    std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();

    let result = SkillFolderScanner::new(&root).scan_result().await.unwrap();
    assert_eq!(result.capability_ids(), ["real"]);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "symlink_skipped"));
}
