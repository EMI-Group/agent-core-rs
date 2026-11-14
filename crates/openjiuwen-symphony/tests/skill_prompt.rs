//! Behavior tests for the model-facing prompt snapshot and incremental skill notices.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]
use std::fs;
use std::path::Path;

use openjiuwen_symphony::discovery::{
    inventory_fingerprint, PromptSnapshotOptions, SkillFs, SkillInventory, SkillNoticeState,
    SkillRecord, SkillScanConfig, DEFAULT_NOTICE_MAX_CHARS, PROMPT_EMPTY_TEXT,
    SKILL_INDEX_TOOL_NAME,
};

/// Writes one `SKILL.md` under `root/<folder>`.
fn write_skill(root: &Path, folder: &str, front_matter: &str, body: &str) {
    let directory = root.join(folder);
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("SKILL.md"),
        format!("---\n{front_matter}\n---\n\n{body}\n"),
    )
    .unwrap();
}

/// A minimal skill with a name and description.
fn skill(root: &Path, folder: &str, name: &str, description: &str) {
    write_skill(
        root,
        folder,
        &format!("name: {name}\ndescription: {description}"),
        description,
    );
}

#[tokio::test]
async fn visible_skill_names_use_pythons_casefold_order_and_drop_disabled_skills() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write_skill(
        root,
        "first",
        "capability_id: Beta\nname: Beta\ndescription: Second",
        "Second",
    );
    write_skill(
        root,
        "second",
        "capability_id: alpha\nname: Alpha\ndescription: First",
        "First",
    );
    write_skill(
        root,
        "third",
        "capability_id: gamma\nname: Gamma\ndescription: Third",
        "Third",
    );

    let catalog = SkillFs::open(root).await.unwrap();
    assert_eq!(catalog.visible_skill_names(), ["alpha", "Beta", "gamma"]);

    let disabled = SkillFs::open_with(
        SkillScanConfig::default().with_disabled_skills(["beta"]),
        root,
    )
    .await
    .unwrap();
    assert_eq!(disabled.visible_skill_names(), ["alpha", "gamma"]);
}

#[tokio::test]
async fn prompt_snapshot_renders_categories_and_bookkeeping() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write_skill(
        root,
        "alpha",
        "name: Alpha\ndescription: Adds numbers\ncategory: documents",
        "Adds numbers",
    );
    write_skill(
        root,
        "beta",
        "name: Beta\ndescription: Translates text",
        "Translates text",
    );
    let catalog = SkillFs::open(root).await.unwrap();

    let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::default());
    assert_eq!(
        snapshot.rendered,
        "# Skills\n\n- alpha: Adds numbers\n- beta: Translates text"
    );
    assert_eq!(snapshot.included, ["alpha", "beta"]);
    assert!(snapshot.omitted.is_empty());
    assert!(!snapshot.truncated);
    assert_eq!(snapshot.fingerprint, catalog.inventory().fingerprint);

    let groups = PromptSnapshotOptions::default().with_categories(true);
    let snapshot = catalog.prompt_snapshot(&groups);
    assert_eq!(
        snapshot.rendered,
        "# Skills\n\n\
         ## documents\n\n\
         - alpha: Adds numbers\n\n\
         ## Uncategorized\n\n\
         - beta: Translates text"
    );
    assert_eq!(snapshot.included, ["alpha", "beta"]);
}

#[tokio::test]
async fn prompt_snapshot_truncates_on_a_multibyte_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    skill(root, "alpha", "Alpha", "转换文档为 PDF 并保持排版布局");
    skill(root, "beta", "Beta", "Translates text");
    let catalog = SkillFs::open(root).await.unwrap();

    let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::new(36));
    assert!(snapshot.truncated);
    assert_eq!(snapshot.included, ["alpha"]);
    assert_eq!(snapshot.omitted, ["beta"]);
    assert!(snapshot.rendered.starts_with("# Skills\n\n- alpha: "));
    assert!(snapshot.rendered.contains("..."));
    assert!(snapshot.rendered.ends_with("[truncated]"));
    assert!(snapshot.rendered.chars().count() <= 36);
    assert!(!snapshot.rendered.contains('\u{fffd}'));
    assert_eq!(snapshot.fingerprint, {
        let record = catalog.inventory().get("alpha").unwrap().clone();
        inventory_fingerprint(&[record])
    });
}

#[tokio::test]
async fn prompt_snapshot_handles_tiny_budgets_without_a_blank_result() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    skill(root, "alpha", "Alpha", "Adds numbers");
    let catalog = SkillFs::open(root).await.unwrap();

    let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::new(0));
    assert!(snapshot.truncated);
    assert!(snapshot.included.is_empty());
    assert_eq!(snapshot.omitted, ["alpha"]);
    assert_eq!(snapshot.rendered, "# Skills\n\n[truncated]");

    let snapshot = catalog.prompt_snapshot(&PromptSnapshotOptions::new(4));
    assert!(snapshot.truncated);
    assert_eq!(snapshot.rendered, "# Skills\n\n[truncated]");
    assert_eq!(snapshot.fingerprint, inventory_fingerprint(&[]));
}

#[tokio::test]
async fn prompt_snapshot_filters_to_one_worker_id() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    skill(root, "alpha", "Alpha", "Adds numbers");
    skill(root, "beta", "Beta", "Translates text");
    let catalog = SkillFs::open(root).await.unwrap();

    let options = PromptSnapshotOptions::default().with_worker_id("beta");
    let snapshot = catalog.prompt_snapshot(&options);
    assert_eq!(snapshot.rendered, "# Skills\n\n- beta: Translates text");
    assert_eq!(snapshot.included, ["beta"]);
    assert!(snapshot.omitted.is_empty());
    assert!(!snapshot.truncated);

    let missing = catalog.prompt_snapshot(&PromptSnapshotOptions::default().with_worker_id("nope"));
    assert_eq!(missing.rendered, PROMPT_EMPTY_TEXT);
    assert!(missing.included.is_empty());
    assert!(missing.omitted.is_empty());
    assert!(!missing.truncated);
}

#[tokio::test]
async fn prompt_snapshot_fingerprint_is_stable_and_content_sensitive() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    skill(root, "alpha", "Alpha", "Adds numbers");
    skill(root, "beta", "Beta", "Translates text");

    let first = SkillFs::open(root).await.unwrap();
    let rebuilt = SkillFs::open(root).await.unwrap();
    assert_eq!(
        first
            .prompt_snapshot(&PromptSnapshotOptions::default())
            .fingerprint,
        rebuilt
            .prompt_snapshot(&PromptSnapshotOptions::default())
            .fingerprint
    );

    skill(root, "alpha", "Alpha", "Adds numbers and more");
    let changed = SkillFs::open(root).await.unwrap();
    assert_ne!(
        first
            .prompt_snapshot(&PromptSnapshotOptions::default())
            .fingerprint,
        changed
            .prompt_snapshot(&PromptSnapshotOptions::default())
            .fingerprint
    );
}

fn record(worker_id: &str, description: &str) -> SkillRecord {
    SkillRecord::new(worker_id, worker_id.to_uppercase())
        .with_description(description)
        .with_content_hash(format!("hash:{worker_id}:{description}"))
}

#[tokio::test]
async fn notices_report_added_removed_and_changed_with_a_python_reminder() {
    let baseline = SkillInventory::new(vec![
        record("alpha", "Adds numbers"),
        record("beta", "Translates text"),
    ]);
    let mut state = SkillNoticeState::new(&baseline);
    assert_eq!(state.max_chars, DEFAULT_NOTICE_MAX_CHARS);
    assert_eq!(state.discovery_tool_name, SKILL_INDEX_TOOL_NAME);

    let unchanged = state.consume(&baseline).await;
    assert!(unchanged.added.is_empty());
    assert!(unchanged.removed.is_empty());
    assert!(unchanged.changed.is_empty());
    assert!(unchanged.reminder.is_none());

    let updated = SkillInventory::new(vec![
        record("alpha", "Adds numbers and more"),
        record("charlie", "Summarizes documents"),
    ]);
    let notice = state.consume(&updated).await;
    assert_eq!(notice.added, ["charlie"]);
    assert_eq!(notice.removed, ["beta"]);
    assert_eq!(notice.changed, ["alpha"]);
    assert_eq!(
        notice.reminder.as_deref(),
        Some(
            "<system-reminder>\n\
             The following skills are newly available for use with the Skill tool:\n\n\
             - charlie: Summarizes documents\n\
             </system-reminder>"
        )
    );

    let again = state.consume(&updated).await;
    assert!(again.added.is_empty());
    assert!(again.removed.is_empty());
    assert!(again.changed.is_empty());
    assert!(again.reminder.is_none());
}

#[tokio::test]
async fn notices_sort_names_and_fall_back_when_the_reminder_is_too_long() {
    let empty = SkillInventory::new(Vec::new());
    let mut state = SkillNoticeState::new(&empty);
    let updated = SkillInventory::new(vec![
        record("Beta", "Second"),
        record("alpha", "First"),
        record("gamma", "Third"),
    ]);
    let notice = state.consume(&updated).await;
    assert_eq!(notice.added, ["alpha", "Beta", "gamma"]);
    assert_eq!(notice.removed, Vec::<String>::new());
    assert_eq!(notice.changed, Vec::<String>::new());
    let reminder = notice.reminder.unwrap();
    assert!(reminder.contains("- alpha: First"));
    assert!(reminder.contains("- Beta: Second"));
    assert!(reminder.contains("- gamma: Third"));

    let mut small = SkillNoticeState::new(&empty)
        .with_max_chars(40)
        .with_discovery_tool_name("custom_index");
    let notice = small.consume(&updated).await;
    assert_eq!(
        notice.reminder.as_deref(),
        Some(
            "<system-reminder>\n\
             3 new skills are available. Use `custom_index` to search the current catalog.\n\
             </system-reminder>"
        )
    );
}

#[tokio::test]
async fn notice_state_is_plain_persistable_data() {
    let baseline = SkillInventory::new(vec![record("alpha", "First")]);
    let mut state = SkillNoticeState::new(&baseline);
    let first = SkillInventory::new(vec![record("beta", "Second")]);
    let notice = state.consume(&first).await;
    assert_eq!(notice.added, ["beta"]);

    let encoded = serde_json::to_string(&state).unwrap();
    let mut restored: SkillNoticeState = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored, state);

    let second = SkillInventory::new(vec![record("gamma", "Third")]);
    let notice = restored.consume(&second).await;
    assert_eq!(notice.added, ["gamma"]);
    assert_eq!(notice.removed, ["beta"]);
    assert_eq!(notice.changed, Vec::<String>::new());
}

#[tokio::test]
async fn skill_fs_notice_state_tracks_newly_installed_skills() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    skill(root, "alpha", "Alpha", "Adds numbers");
    let catalog = SkillFs::open(root).await.unwrap();
    let mut state = catalog.notice_state();

    skill(root, "beta", "Beta", "Translates text");
    let refreshed = SkillFs::open(root).await.unwrap();
    let notice = state.consume(refreshed.inventory()).await;
    assert_eq!(notice.added, ["beta"]);
    assert!(notice.removed.is_empty());
    assert!(notice.changed.is_empty());
    assert!(notice.reminder.is_some());
}
