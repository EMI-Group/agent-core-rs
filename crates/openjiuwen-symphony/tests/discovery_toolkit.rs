//! Behavior tests for the model-facing `skill_index` foundation layer: the virtual
//! category view, the pinned row renderers and output budgeting, and request validation.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::too_many_lines, clippy::assert_is_empty)]

use std::fs;
use std::path::Path;

use openjiuwen_symphony::discovery::directory_toolkit::{
    append_note, apply_simple_defaults, bounded_summary_line, category_segments, compact,
    empty_message, fit_rows, inline_text, list_row, metadata_card, row_count, safe_match_snippet,
    sanitize_model_text, search_directory_row, search_row, summary_line, validate,
    validate_category_paths, validate_identifiers, validate_paths, validate_pipeline,
    validate_queries, DirectoryEntry, EntryKind, FittedRows, ListView, Operation, OutputMode,
    PipelineStage, ReadMode, RequestIssue, ResponseSummary, Row, SearchMatch, SearchResult,
    SkillDirectoryView, SkillIndexRequest, ValidatedRequest, SHORTENED_MARKER,
};
use openjiuwen_symphony::discovery::SkillFs;

/// Writes `content` to `root/relative`, creating parent directories.
fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

/// Builds a `SKILL.md` with front matter plus an optional extra field block.
fn skill(name: &str, description: &str, extra: &str) -> String {
    format!("---\nname: {name}\ndescription: {description}\n{extra}---\n\n{description}\n")
}

/// Writes the shared 8-skill fixture across five categories.
fn fixture_root(root: &Path) {
    write(
        root,
        "alpha/SKILL.md",
        &skill(
            "Alpha",
            "Adds numbers",
            "category: Data\nversion: 1.2.3\nauthor: Ada\n",
        ),
    );
    write(
        root,
        "omega/SKILL.md",
        &skill("Omega", "Sorts rows", "category: Data\n"),
    );
    write(
        root,
        "beta/SKILL.md",
        &skill("Beta", "Exports tables", "category: Data > Export\n"),
    );
    write(
        root,
        "delta/SKILL.md",
        &skill("Delta", "Takes notes", "category: Notes\n"),
    );
    write(
        root,
        "tau/SKILL.md",
        &skill("Tau", "Exports notes", "category: Notes > 导出\n"),
    );
    write(
        root,
        "sigma/SKILL.md",
        &skill("Sigma", "导出数据", "category: 数据 > 导出\n"),
    );
    write(
        root,
        "gamma/SKILL.md",
        &skill("Gamma", "Standalone helper", ""),
    );
    write(
        root,
        "notes/SKILL.md",
        &skill("Notes", "A skill called Notes", ""),
    );
}

/// Opens the shared fixture as a catalog inside a fresh temporary directory.
async fn fixture() -> (tempfile::TempDir, SkillFs) {
    let directory = tempfile::tempdir().unwrap();
    fixture_root(directory.path());
    let catalog = SkillFs::open(directory.path()).await.unwrap();
    let view = SkillDirectoryView::from_catalog(&catalog);
    assert_eq!(view.record_ids().len(), 8);
    (directory, catalog)
}

/// Builds the view over the shared fixture.
async fn fixture_view() -> (tempfile::TempDir, SkillFs, SkillDirectoryView) {
    let (directory, catalog) = fixture().await;
    let view = SkillDirectoryView::from_catalog(&catalog);
    (directory, catalog, view)
}

/// The entry at `path` among the view's tree rows.
fn entry_at(view: &SkillDirectoryView, path: &str) -> DirectoryEntry {
    view.entries(&[view.root_path().to_string()], true, None, false, false)
        .unwrap()
        .into_iter()
        .find(|entry| entry.path == path)
        .unwrap_or_else(|| panic!("no entry at {path}"))
}

/// A summary in the shape the renderers expect.
fn summary(operation: &str, skills: usize, categories: usize, shown: bool) -> ResponseSummary {
    ResponseSummary {
        operation: operation.to_string(),
        returned_skill_count: skills,
        returned_category_count: categories,
        previously_shown: shown,
    }
}

/// The index of the first issue naming `field`.
fn issue_index(issues: &[RequestIssue], field: &str) -> Option<usize> {
    issues.iter().position(|issue| issue.field == field)
}

// ---------------------------------------------------------------------------
// view
// ---------------------------------------------------------------------------

#[test]
fn normalize_path_collapses_and_rejects_traversal() {
    assert_eq!(SkillDirectoryView::normalize_path("/").unwrap(), "/");
    assert_eq!(SkillDirectoryView::normalize_path("").unwrap(), "/");
    assert_eq!(SkillDirectoryView::normalize_path("/a/b").unwrap(), "/a/b");
    assert_eq!(SkillDirectoryView::normalize_path("a/b/").unwrap(), "/a/b");
    assert_eq!(
        SkillDirectoryView::normalize_path("//a//b//").unwrap(),
        "/a/b"
    );
    assert_eq!(
        SkillDirectoryView::normalize_path(" /Data/Export ").unwrap(),
        "/Data/Export"
    );
    let error = SkillDirectoryView::normalize_path("/Data/../etc").unwrap_err();
    assert!(error.to_string().contains("must not contain '..'"));
    // `ROOT` is a category alias, not a path alias.
    assert_eq!(SkillDirectoryView::normalize_path("ROOT").unwrap(), "/ROOT");
}

#[tokio::test]
async fn the_view_builds_a_deterministic_category_tree() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(view.root_path(), "/");
    assert_eq!(view.root_label(), "ROOT");
    assert_eq!(
        view.node_paths(),
        [
            "/",
            "/Data",
            "/Data/Export",
            "/Notes",
            "/Notes/导出",
            "/数据",
            "/数据/导出"
        ]
    );
    assert_eq!(
        view.children("/")
            .unwrap()
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        ["/Data", "/gamma", "/Notes", "/notes~2", "/数据"]
    );
    assert_eq!(
        view.children("/Data")
            .unwrap()
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        ["/Data/alpha", "/Data/Export", "/Data/omega"]
    );
    // The `Notes` directory wins the label; the skill is disambiguated.
    let notes_skill = entry_at(&view, "/notes~2");
    assert_eq!(notes_skill.kind, EntryKind::Skill);
    assert_eq!(notes_skill.worker_id, "notes");
    assert_eq!(notes_skill.label, "Notes");
    let notes_dir = view.node("/Notes").unwrap();
    assert_eq!(notes_dir.label, "Notes");
    assert_eq!(view.node("/Data/Export").unwrap().label, "Export");
    assert_eq!(view.node("/Notes/导出").unwrap().label, "导出");
}

#[tokio::test]
async fn skill_counts_include_every_descendant() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(view.skill_count("/").unwrap(), 8);
    assert_eq!(view.skill_count("/Data").unwrap(), 3);
    assert_eq!(view.skill_count("/Data/Export").unwrap(), 1);
    assert_eq!(view.skill_count("/Notes").unwrap(), 2);
    assert_eq!(view.skill_count("/Notes/导出").unwrap(), 1);
    assert_eq!(view.skill_count("/数据").unwrap(), 1);
    assert_eq!(view.skill_count("Data").unwrap(), 3);
    assert_eq!(view.skill_count_or_zero("/nope"), 0);
    let error = view.skill_count("/nope").unwrap_err();
    assert_eq!(error.message(), "No such Skill directory: /nope");
    assert_eq!(
        view.children("/nope").unwrap_err().message(),
        "No such Skill directory: /nope"
    );
}

#[tokio::test]
async fn entries_reproduce_the_python_scope_rules() {
    let (_directory, _catalog, view) = fixture_view().await;
    let root = vec!["/".to_string()];

    // Direct children only.
    let direct = view.entries(&root, false, None, false, false).unwrap();
    assert_eq!(direct.len(), 5);
    assert!(direct.iter().all(|entry| entry.depth == 1));

    // Recursive, nothing filtered: parent before children, siblings in tree order.
    let recursive = view.entries(&root, true, None, false, false).unwrap();
    let shape: Vec<(&str, usize)> = recursive
        .iter()
        .map(|entry| (entry.path.as_str(), entry.depth))
        .collect();
    assert_eq!(
        shape,
        [
            ("/Data", 1),
            ("/Data/alpha", 2),
            ("/Data/Export", 2),
            ("/Data/Export/beta", 3),
            ("/Data/omega", 2),
            ("/gamma", 1),
            ("/Notes", 1),
            ("/Notes/delta", 2),
            ("/Notes/导出", 2),
            ("/Notes/导出/tau", 3),
            ("/notes~2", 1),
            ("/数据", 1),
            ("/数据/导出", 2),
            ("/数据/导出/sigma", 3),
        ]
    );

    // `max_depth` is counted relative to the scope root.
    let shallow = view.entries(&root, true, Some(1), false, false).unwrap();
    assert_eq!(shallow.len(), 5);
    let two = view.entries(&root, true, Some(2), false, false).unwrap();
    assert_eq!(two.len(), 11);

    // `directory_entry` returns the scope row itself: Python passes `recursive=None`,
    // which skips `_walk` entirely.
    let with_root = view.entries(&root, true, None, false, true).unwrap();
    assert_eq!(with_root.len(), 1);
    assert_eq!(with_root[0].path, "/");
    assert_eq!(with_root[0].kind, EntryKind::Dir);
    assert_eq!(with_root[0].label, "ROOT");
    assert_eq!(with_root[0].depth, 0);
    assert_eq!(with_root[0].description, "Installed Skills");
    let nested = view
        .entries(&["/Data".to_string()], true, None, false, true)
        .unwrap();
    assert_eq!(nested.len(), 1);
    assert_eq!(nested[0].path, "/Data");
    assert_eq!(nested[0].label, "Data");

    // `directories_only` drops Skill rows.
    let directories = view.entries(&root, true, None, true, false).unwrap();
    assert_eq!(
        directories
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        [
            "/Data",
            "/Data/Export",
            "/Notes",
            "/Notes/导出",
            "/数据",
            "/数据/导出"
        ]
    );

    // Two scopes de-duplicate by `(kind, path)` and keep scope order.
    let two_scopes = view
        .entries(
            &["/Data".to_string(), "/Data/Export".to_string()],
            true,
            None,
            false,
            false,
        )
        .unwrap();
    let mut paths: Vec<&str> = two_scopes.iter().map(|entry| entry.path.as_str()).collect();
    paths.dedup();
    assert_eq!(
        paths,
        [
            "/Data/alpha",
            "/Data/Export",
            "/Data/Export/beta",
            "/Data/omega"
        ]
    );
    assert!(view
        .entries(&["/nope".to_string()], true, None, false, false)
        .is_err());
}

#[tokio::test]
async fn tree_and_searchable_entries_agree_on_order() {
    let (_directory, _catalog, view) = fixture_view().await;
    let root = vec!["/".to_string()];
    let tree = view.tree_entries(&root, None, false).unwrap();
    assert_eq!(tree.len(), 15);
    assert_eq!(tree[0].path, "/");
    let recursive = view.entries(&root, true, None, false, false).unwrap();
    assert_eq!(
        tree[1..]
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        recursive
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>()
    );
    let shallow = view.tree_entries(&root, Some(1), false).unwrap();
    assert_eq!(shallow.len(), 6);
    let searchable = view.searchable_entries(&root, None).unwrap();
    assert_eq!(
        searchable
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        tree.iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>()
    );
    // A `META.md` scope resolves to its Skill row.
    let resolved = view
        .searchable_entries(&["/Data/alpha/META.md".to_string()], None)
        .unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].kind, EntryKind::Skill);
    assert_eq!(resolved[0].path, "/Data/alpha");
    assert_eq!(resolved[0].worker_id, "alpha");
}

#[tokio::test]
async fn scoped_records_and_metadata_paths_round_trip() {
    let (_directory, _catalog, view) = fixture_view().await;
    let root = vec!["/".to_string()];
    let all = view.scoped_records(&root, None).unwrap();
    let works: Vec<&str> = all.iter().map(|record| record.worker_id.as_str()).collect();
    assert_eq!(
        works,
        ["alpha", "beta", "omega", "gamma", "delta", "tau", "notes", "sigma"]
    );
    let data = view
        .scoped_records(&["/Data".to_string()], Some(2))
        .unwrap();
    let scoped: Vec<&str> = data
        .iter()
        .map(|record| record.worker_id.as_str())
        .collect();
    assert_eq!(scoped, ["alpha", "beta", "omega"]);

    assert_eq!(view.metadata_path("alpha").unwrap(), "/Data/alpha/META.md");
    let record = view.resolve_metadata_path("/Data/alpha/META.md").unwrap();
    assert_eq!(record.worker_id, "alpha");
    assert_eq!(record.name, "Alpha");
    assert_eq!(
        view.resolve_metadata_path("/Data/alpha/SKILL.md")
            .unwrap_err()
            .message(),
        "No such Skill metadata path: /Data/alpha/SKILL.md"
    );
    assert_eq!(
        view.metadata_path("nope").unwrap_err().message(),
        "Unknown Skill ID: nope"
    );
    assert_eq!(view.record_path("beta"), Some("/Data/Export/beta"));
    assert_eq!(view.record_path("nope"), None);
    assert_eq!(
        view.record("gamma").unwrap().description,
        "Standalone helper"
    );
}

#[tokio::test]
async fn category_names_render_from_paths_and_skills() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(view.category_from_path("/"), "ROOT");
    assert_eq!(view.category_from_path("/Data"), "Data");
    assert_eq!(view.category_from_path("/Data/Export"), "Data > Export");
    assert_eq!(view.category_from_path("/Notes/导出"), "Notes > 导出");
    assert_eq!(view.skill_category("beta").unwrap(), "Data > Export");
    assert_eq!(view.skill_category("gamma").unwrap(), "ROOT");
    assert_eq!(view.skill_category_or_empty("nope"), "");
    assert_eq!(view.category_text("beta"), "Data\nExport");
    assert_eq!(view.category_text("gamma"), "");
    assert_eq!(
        view.skill_category("nope").unwrap_err().message(),
        "Unknown Skill ID: nope"
    );
    assert!(view.entry_labels().contains(&"ROOT".to_string()));
    assert!(view.entry_labels().contains(&"导出".to_string()));
    assert!(view
        .entry_labels()
        .windows(2)
        .all(|pair| pair[0].to_lowercase() <= pair[1].to_lowercase()));
}

#[tokio::test]
async fn category_paths_accept_aliases_labels_and_chains() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(view.category_path("/").unwrap(), "/");
    assert_eq!(view.category_path("ROOT").unwrap(), "/");
    assert_eq!(view.category_path("root").unwrap(), "/");
    assert_eq!(view.category_path("Data").unwrap(), "/Data");
    assert_eq!(view.category_path("data").unwrap(), "/Data");
    assert_eq!(view.category_path("Data > Export").unwrap(), "/Data/Export");
    assert_eq!(view.category_path(" Export ").unwrap(), "/Data/Export");
    assert_eq!(view.category_path("数据 > 导出").unwrap(), "/数据/导出");
    assert_eq!(
        view.category_path("Missing").unwrap_err().message(),
        "Unknown Skill category: Missing"
    );
    assert_eq!(
        view.category_path("导出").unwrap_err().message(),
        "Ambiguous Skill category; use its full category chain: 导出"
    );
    assert!(view.category_path("  ").is_err());
}

#[tokio::test]
async fn category_chains_split_on_angle_brackets_then_slashes() {
    assert_eq!(category_segments("Data > Export"), ["Data", "Export"]);
    assert_eq!(category_segments("Data/Export"), ["Data", "Export"]);
    assert_eq!(category_segments(" > 数据 >  > 导出 > "), ["数据", "导出"]);
    assert!(category_segments("").is_empty());
    assert!(category_segments("   ").is_empty());
}

#[tokio::test]
async fn building_the_view_twice_is_deterministic() {
    let (_directory, catalog) = fixture().await;
    let first = SkillDirectoryView::from_catalog(&catalog);
    let second = SkillDirectoryView::from_catalog(&catalog);
    assert_eq!(first.node_paths(), second.node_paths());
    let root = vec!["/".to_string()];
    let left = first.entries(&root, true, None, false, false).unwrap();
    let right = second.entries(&root, true, None, false, false).unwrap();
    assert_eq!(left, right);
    let rows_left: Vec<Row> = left
        .iter()
        .map(|entry| list_row(entry, ListView::Tree, &first))
        .collect();
    let rows_right: Vec<Row> = right
        .iter()
        .map(|entry| list_row(entry, ListView::Tree, &second))
        .collect();
    assert_eq!(rows_left, rows_right);
    assert_eq!(
        rows_left[1].text,
        "    - [skill] alpha  category: Data  path: alpha/SKILL.md  desc: Adds numbers"
    );
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_rows_pin_every_view() {
    let (_directory, _catalog, view) = fixture_view().await;
    let data = entry_at(&view, "/Data");
    assert_eq!(
        list_row(&data, ListView::Names, &view).text,
        "- [category] Data"
    );
    assert_eq!(
        list_row(&data, ListView::Details, &view).text,
        "- [category] Data  desc: Contains 3 skills."
    );
    let export = entry_at(&view, "/Data/Export");
    assert_eq!(
        list_row(&export, ListView::Tree, &view).text,
        "    - [category] Data > Export  desc: Contains 1 skill."
    );
    assert_eq!(list_row(&data, ListView::Details, &view).worker_id, "");
    assert!(list_row(&data, ListView::Names, &view).is_category_row());

    let alpha = entry_at(&view, "/Data/alpha");
    assert_eq!(
        list_row(&alpha, ListView::Names, &view).text,
        "- [skill] alpha  category: Data  path: alpha/SKILL.md"
    );
    assert_eq!(
        list_row(&alpha, ListView::Details, &view).text,
        "- [skill] alpha  category: Data  path: alpha/SKILL.md  desc: Adds numbers"
    );
    assert_eq!(
        list_row(&alpha, ListView::Tree, &view).text,
        "    - [skill] alpha  category: Data  path: alpha/SKILL.md  desc: Adds numbers"
    );
    assert_eq!(list_row(&alpha, ListView::Tree, &view).worker_id, "alpha");

    let gamma = entry_at(&view, "/gamma");
    assert_eq!(
        list_row(&gamma, ListView::Details, &view).text,
        "- [skill] gamma  category: ROOT  path: gamma/SKILL.md  desc: Standalone helper"
    );
}

#[tokio::test]
async fn search_rows_pin_their_fields() {
    let (_directory, _catalog, view) = fixture_view().await;
    let alpha = view.record("alpha").unwrap();
    assert_eq!(
        search_row(alpha, &view, &[], None).text,
        "- [skill] alpha  category: Data  desc: Adds numbers  path: alpha/SKILL.md"
    );
    assert_eq!(
        search_row(alpha, &view, &[1, 3], None).text,
        "- [skill] alpha  category: Data  matches: 1, 3  desc: Adds numbers  path: alpha/SKILL.md"
    );
    assert_eq!(
        search_row(alpha, &view, &[2], Some("Adds")).text,
        "- [skill] alpha  category: Data  matches: 2  match: Adds  desc: Adds numbers  path: alpha/SKILL.md"
    );
    assert_eq!(
        search_row(alpha, &view, &[], Some("")).text,
        search_row(alpha, &view, &[], None).text
    );
    let gamma = view.record("gamma").unwrap();
    assert_eq!(
        search_row(gamma, &view, &[], None).text,
        "- [skill] gamma  category: ROOT  desc: Standalone helper  path: gamma/SKILL.md"
    );
    assert_eq!(search_row(alpha, &view, &[], None).worker_id, "alpha");

    let data = entry_at(&view, "/Data");
    assert_eq!(
        search_directory_row(&data, &view, &[]).text,
        "- [category] Data  desc: Contains 3 skills."
    );
    assert_eq!(
        search_directory_row(&data, &view, &[2, 4]).text,
        "- [category] Data  matches: 2, 4  desc: Contains 3 skills."
    );
    assert_eq!(search_directory_row(&data, &view, &[]).worker_id, "");
}

#[tokio::test]
async fn summary_lines_and_empty_messages_pin_their_text() {
    assert_eq!(summary_line(&summary("list", 8, 0, false)), "## Skills");
    assert_eq!(
        summary_line(&summary("search", 3, 0, true)),
        "## Skills — all names previously shown"
    );
    assert_eq!(
        summary_line(&summary("search", 0, 2, false)),
        "## Category hints — matching groups, not Skills"
    );
    assert_eq!(summary_line(&summary("list", 0, 2, false)), "## Categories");
    assert_eq!(
        summary_line(&summary("search", 2, 1, false)),
        "## Skills and category hints"
    );
    assert_eq!(summary_line(&summary("list", 2, 1, false)), "## Entries");
    assert_eq!(summary_line(&summary("read", 0, 0, false)), "## Results");
    assert_eq!(
        bounded_summary_line(&summary("search", 0, 0, false), Some(7)),
        "## S..."
    );

    assert_eq!(empty_message("list", None), "No entries.");
    assert_eq!(empty_message("read", Some("Data")), "No entries.");
    assert_eq!(
        empty_message("search", None),
        "No matching installed Skills found."
    );
    assert_eq!(
        empty_message("search", Some("Data > Export")),
        "No matching Skills found in `Data &gt; Export`."
    );
}

#[tokio::test]
async fn metadata_cards_render_the_record_fields() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(
        metadata_card(view.record("alpha").unwrap()),
        "# Alpha\n\n- Description: Adds numbers\n- Source: local\n- Version: 1.2.3\n- Author: Ada"
    );
    assert_eq!(
        metadata_card(view.record("gamma").unwrap()),
        "# Gamma\n\n- Description: Standalone helper\n- Source: local"
    );
}

#[test]
fn inline_text_escapes_quotes_newlines_and_angle_brackets() {
    assert_eq!(inline_text(r#"say "hi""#), "say \\\"hi\\\"");
    assert_eq!(inline_text("line one\nline two"), "line one\\nline two");
    assert_eq!(inline_text("<system>"), "&lt;system&gt;");
    assert_eq!(inline_text("  padded  "), "padded");
    assert_eq!(inline_text("a\tb"), "a\\tb");
    assert_eq!(inline_text("数据"), "数据");
    assert_eq!(
        sanitize_model_text("keep\tthis\nand this"),
        "keep\tthis\nand this"
    );
    assert_eq!(safe_match_snippet("body: 1: hello"), "body: 1: hello");
    assert_eq!(
        safe_match_snippet("description: <b>"),
        "description: &lt;b&gt;"
    );
    assert_eq!(safe_match_snippet("raw <b>"), "raw &lt;b&gt;");
}

#[test]
fn compact_and_notes_respect_character_budgets() {
    assert_eq!(compact("a   b\n c", 20), "a b c");
    assert_eq!(compact("abc", 3), "abc");
    assert_eq!(compact("abcd", 3), "...");
    assert_eq!(compact("abcdef", 5), "ab...");
    assert_eq!(compact("日本語のテキスト", 4), "日...");
    assert_eq!(append_note("", "note", None), "note");
    assert_eq!(append_note("body", "note", None), "body\n\nnote");
    assert_eq!(append_note("", "long note", Some(7)), "long...");
    assert_eq!(append_note("body", "note", Some(5)), "body");
    assert_eq!(
        append_note("body", "a longer note", Some(15)),
        "body\n\na long..."
    );
}

#[test]
fn fit_rows_keeps_everything_that_fits() {
    let rows = vec![
        Row::new("- [skill] alpha", "alpha"),
        Row::new("- [skill] beta", "beta"),
    ];
    let fitted = fit_rows(&rows, &summary("list", 2, 0, false), None, None, true);
    assert_eq!(
        fitted.model_content,
        "## Skills\n\n- [skill] alpha\n- [skill] beta"
    );
    assert_eq!(fitted.body, "- [skill] alpha\n- [skill] beta");
    assert_eq!(fitted.observed, ["alpha", "beta"]);
    assert!(!fitted.shortened);
    assert_eq!(fitted.shown_count, 2);
    assert_eq!(row_count(&rows), 2);
}

#[test]
fn fit_rows_drops_rows_and_suppresses_the_marker() {
    let rows = vec![
        Row::new("- [skill] alpha", "alpha"),
        Row::new("- [skill] beta", "beta"),
    ];
    let fitted = fit_rows(&rows, &summary("list", 2, 0, false), Some(30), None, true);
    assert_eq!(fitted.model_content, "## Skills\n\n- [skill] alpha");
    assert_eq!(fitted.observed, ["alpha"]);
    assert!(fitted.shortened);
    assert_eq!(fitted.shown_count, 1);
    assert!(!fitted.body.contains(SHORTENED_MARKER));

    let suppressed = fit_rows(&rows, &summary("list", 2, 0, false), Some(100), None, false);
    assert_eq!(
        suppressed.model_content,
        "## Skills\n\n- [skill] alpha\n- [skill] beta"
    );
    assert!(!suppressed.shortened);
    assert!(!suppressed.body.contains(SHORTENED_MARKER));
}

#[test]
fn fit_rows_appends_the_marker_when_it_fits() {
    let first = format!("- [skill] alpha {}", "x".repeat(44));
    let second = format!("- [skill] beta {}", "y".repeat(45));
    let rows = vec![
        Row::new(first.clone(), "alpha"),
        Row::new(second.clone(), "beta"),
    ];
    let header = "## Skills".chars().count();
    let marker_budget = header + 2 + first.chars().count() + 1 + SHORTENED_MARKER.chars().count();
    assert!(marker_budget < header + 2 + first.chars().count() + 1 + second.chars().count());
    let fitted = fit_rows(
        &rows,
        &summary("list", 2, 0, false),
        Some(marker_budget),
        None,
        true,
    );
    assert!(fitted.shortened);
    assert_eq!(fitted.body, format!("{first}\n{SHORTENED_MARKER}"));
    assert_eq!(
        fitted.model_content,
        format!("## Skills\n\n{first}\n{SHORTENED_MARKER}")
    );
    assert_eq!(fitted.observed, ["alpha"]);
    let without = fit_rows(
        &rows,
        &summary("list", 2, 0, false),
        Some(marker_budget),
        None,
        false,
    );
    assert_eq!(without.body, first);
    assert!(without.shortened);
}

#[test]
fn fit_rows_falls_back_to_one_compacted_row() {
    let rows = vec![
        Row::new("- [skill] alpha", "alpha"),
        Row::new("- [skill] beta", "beta"),
    ];
    let fitted = fit_rows(&rows, &summary("list", 2, 0, false), Some(20), None, true);
    assert_eq!(fitted.body, "- [ski...");
    assert_eq!(fitted.model_content, "## Skills\n\n- [ski...");
    assert!(fitted.shortened);
    assert_eq!(fitted.observed, ["alpha"]);
    assert_eq!(fitted.shown_count, 1);

    let tiny = fit_rows(&rows, &summary("list", 2, 0, false), Some(9), None, true);
    assert_eq!(tiny.model_content, "## Skills");
    assert!(tiny.body.is_empty());
    assert!(tiny.observed.is_empty());
    assert_eq!(tiny.shown_count, 0);

    let empty = row_count(&[]);
    assert_eq!(empty, 0);
    let fitted = fit_rows(&[], &summary("list", 0, 0, false), Some(20), None, true);
    assert!(!fitted.shortened);
    assert_eq!(fitted.model_content, "## Results");
}

#[test]
fn fit_rows_uses_the_pipeline_count() {
    let rows = vec![
        Row::new("- [skill] alpha", "alpha"),
        Row::new("- [skill] beta", "beta"),
    ];
    let fitted = fit_rows(&rows, &summary("list", 2, 0, false), None, Some(42), true);
    assert_eq!(fitted.shown_count, 42);
    assert_eq!(fitted.observed.len(), 2);
    let uncounted = fit_rows(&rows, &summary("list", 2, 0, false), None, None, true);
    assert_eq!(uncounted.shown_count, 2);
}

#[test]
fn fit_rows_counts_characters_not_bytes() {
    let rows = vec![
        Row::new("- 日本語のスキル", "jp"),
        Row::new("- [skill] alpha", "alpha"),
    ];
    // The first row is 10 characters but 26 bytes: a byte budget of 21 would drop it.
    let fitted = fit_rows(&rows, &summary("list", 2, 0, false), Some(21), None, false);
    assert_eq!(fitted.body, "- 日本語のスキル");
    assert!(fitted.shortened);
    assert_eq!(fitted.observed, ["jp"]);
    let both = fit_rows(&rows, &summary("list", 2, 0, false), Some(38), None, false);
    assert_eq!(both.body, "- 日本語のスキル\n- [skill] alpha");
    assert!(!both.shortened);
}

#[test]
fn row_shapes_round_trip_through_json() {
    let fitted = FittedRows {
        body: "body".to_string(),
        model_content: "## Skills\n\nbody".to_string(),
        observed: vec!["alpha".to_string()],
        shortened: true,
        shown_count: 1,
    };
    let encoded = serde_json::to_value(&fitted).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({
            "body": "body",
            "model_content": "## Skills\n\nbody",
            "observed": ["alpha"],
            "shortened": true,
            "shown_count": 1
        })
    );
    let decoded: FittedRows = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, fitted);
    let row: Row = serde_json::from_str(r#"{"text":"t","worker_id":"","extra":1}"#).unwrap();
    assert_eq!(row, Row::text_only("t"));
    assert_eq!(ListView::parse("tree"), Some(ListView::Tree));
    assert_eq!(ListView::parse("bonsai"), None);
    assert_eq!(ListView::ALL.len(), 3);
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

#[test]
fn a_valid_list_request_passes() {
    let request = SkillIndexRequest::new(Operation::List);
    let validated = validate(&request).unwrap();
    assert_eq!(validated.operation, Operation::List);
    assert_eq!(validated.view, ListView::Names);
    assert_eq!(validated.paths, ["/"]);
    assert!(validated.queries.is_empty());
    assert!(validated.skills.is_empty());
    assert_eq!(validated.page_size, None);

    let tree = SkillIndexRequest {
        operation: "list".to_string(),
        view: "tree".to_string(),
        recursive: false,
        max_depth: Some(3),
        directories_only: true,
        ..SkillIndexRequest::default()
    };
    let validated = validate(&tree).unwrap();
    assert_eq!(validated.view, ListView::Tree);
    assert_eq!(validated.max_depth, Some(3));
    assert!(validated.directories_only);
}

#[test]
fn a_valid_search_request_passes() {
    let request = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["pdf".to_string(), " pdf ".to_string()]),
        search_match: "name".to_string(),
        case_insensitive: true,
        ..SkillIndexRequest::default()
    };
    let validated = validate(&request).unwrap();
    assert_eq!(validated.operation, Operation::Search);
    assert_eq!(validated.queries, ["pdf"]);
    assert_eq!(validated.search_match, SearchMatch::Name);
    assert_eq!(validated.search_result, SearchResult::Files);

    let content = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["needle".to_string()]),
        result: "matches".to_string(),
        fixed_strings: true,
        per_query_limit: Some(4),
        ..SkillIndexRequest::default()
    };
    let validated = validate(&content).unwrap();
    assert_eq!(validated.search_result, SearchResult::Matches);
    assert!(validated.fixed_strings);
    assert_eq!(validated.per_query_limit, Some(4));
}

#[test]
fn a_valid_read_request_passes() {
    let by_skill = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string(), " alpha ".to_string()]),
        ..SkillIndexRequest::default()
    };
    let validated = validate(&by_skill).unwrap();
    assert_eq!(validated.skills, ["alpha"]);
    assert_eq!(validated.paths, ["/"]);
    assert_eq!(validated.read_mode, ReadMode::Full);

    let by_path = SkillIndexRequest {
        operation: "read".to_string(),
        paths: Some(vec!["/Data/alpha/META.md".to_string()]),
        read_mode: "range".to_string(),
        start_line: Some(2),
        end_line: Some(4),
        ..SkillIndexRequest::default()
    };
    let validated = validate(&by_path).unwrap();
    assert_eq!(validated.paths, ["/Data/alpha/META.md"]);
    assert_eq!(validated.read_mode, ReadMode::Range);
    assert_eq!(validated.start_line, Some(2));
    assert_eq!(validated.end_line, Some(4));

    let head = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "head".to_string(),
        line_count: Some(3),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&head).unwrap().line_count, Some(3));
}

#[test]
fn unknown_enums_are_field_issues() {
    let request = SkillIndexRequest {
        operation: "browse".to_string(),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&request).unwrap_err();
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].field, "operation");
    assert_eq!(issues[0].message, "must be one of: list, search, read");
    assert_eq!(
        issues[0].to_string(),
        "operation: must be one of: list, search, read"
    );

    let view = SkillIndexRequest {
        view: "wide".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(
        validate(&view).unwrap_err()[0].message,
        "must be one of: names, details, tree"
    );
    let search_match = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        search_match: "body".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&search_match).unwrap_err()[0].field, "match");
    let result = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        result: "snippets".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&result).unwrap_err()[0].field, "result");
    let read_mode = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "all".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&read_mode).unwrap_err()[0].field, "read_mode");
    let output = SkillIndexRequest {
        operation: "list".to_string(),
        output_mode: "rows".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&output).unwrap_err()[0].field, "output_mode");
    let stage = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage {
            operation: "shuffle".to_string(),
            ..PipelineStage::default()
        }],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&stage).unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0].operation");
}

#[test]
fn non_positive_numbers_are_field_issues() {
    let depth = SkillIndexRequest {
        operation: "list".to_string(),
        view: "tree".to_string(),
        max_depth: Some(0),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&depth).unwrap_err();
    let index = issue_index(&issues, "max_depth").unwrap();
    assert_eq!(issues[index].message, "must be a positive integer");

    let negative = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "head".to_string(),
        line_count: Some(-2),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&negative).unwrap_err();
    assert!(issue_index(&issues, "line_count").is_some());
    assert!(issue_index(&issues, "read_mode").is_none());

    let zero_chars = SkillIndexRequest {
        max_output_chars: Some(0),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&zero_chars).unwrap_err();
    let index = issue_index(&issues, "max_output_chars").unwrap();
    assert_eq!(issues[index].message, "must be a positive integer");

    let out_of_range = SkillIndexRequest {
        max_output_chars: Some(100),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&out_of_range).unwrap_err();
    let index = issue_index(&issues, "max_output_chars").unwrap();
    assert_eq!(issues[index].message, "must be between 512 and 48000");
    let too_big = SkillIndexRequest {
        max_output_chars: Some(1_000_000),
        ..SkillIndexRequest::default()
    };
    assert!(issue_index(&validate(&too_big).unwrap_err(), "max_output_chars").is_some());
    let in_range = SkillIndexRequest {
        max_output_chars: Some(4_000),
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&in_range).unwrap().max_output_chars, Some(4_000));
    let disabled = SkillIndexRequest {
        max_output_chars: Some(1),
        disable_output_truncation: true,
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&disabled).unwrap().max_output_chars, None);
}

#[test]
fn query_arrays_are_bounded() {
    let too_many = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some((0..9).map(|index| format!("q{index}")).collect()),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&too_many).unwrap_err();
    let index = issue_index(&issues, "query").unwrap();
    assert_eq!(issues[index].message, "must contain 1-8 strings");

    let empty = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(Vec::new()),
        ..SkillIndexRequest::default()
    };
    assert!(issue_index(&validate(&empty).unwrap_err(), "query").is_some());

    let missing = SkillIndexRequest {
        operation: "search".to_string(),
        ..SkillIndexRequest::default()
    };
    assert!(issue_index(&validate(&missing).unwrap_err(), "query").is_some());

    let blank = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["   ".to_string()]),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&blank).unwrap_err();
    assert_eq!(issues[0].message, "must be a non-empty string");

    let nul = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["a\0b".to_string()]),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&nul).unwrap_err();
    assert_eq!(issues[0].message, "must not contain a null byte");

    let long = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".repeat(513)]),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&long).unwrap_err();
    assert_eq!(issues[0].message, "must not exceed 512 characters");

    assert_eq!(
        validate_queries(&Some(vec![" pdf ".to_string()])).unwrap(),
        ["pdf"]
    );
    assert!(validate_queries(&None).is_err());
}

#[test]
fn paths_and_identifiers_are_bounded() {
    let many = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        paths: Some((0..33).map(|index| format!("/p{index}")).collect()),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&many).unwrap_err();
    let index = issue_index(&issues, "paths").unwrap();
    assert_eq!(issues[index].message, "must contain at most 32 items");

    let nul = SkillIndexRequest {
        operation: "read".to_string(),
        paths: Some(vec!["/a\0b".to_string()]),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&nul).unwrap_err();
    assert_eq!(issues[0].message, "must not contain null bytes");

    let long = SkillIndexRequest {
        operation: "read".to_string(),
        paths: Some(vec!["x".repeat(4_097)]),
        ..SkillIndexRequest::default()
    };
    assert!(issue_index(&validate(&long).unwrap_err(), "paths").is_some());

    assert_eq!(validate_paths(&None, false).unwrap(), ["/"]);
    assert!(validate_paths(&None, true).is_err());
    assert_eq!(
        validate_paths(&Some(vec![" /a ".to_string(), "/a".to_string()]), false).unwrap(),
        ["/a"]
    );
    assert_eq!(
        validate_identifiers(&Some(vec![" alpha ".to_string()]), "skills", true).unwrap(),
        ["alpha"]
    );
    assert!(validate_identifiers(&None, "skills", true).is_err());
    assert!(validate_identifiers(&Some(Vec::new()), "skills", false).is_err());
}

#[test]
fn fields_are_rejected_outside_their_operation() {
    let list_with_query = SkillIndexRequest {
        operation: "list".to_string(),
        query: Some(vec!["x".to_string()]),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&list_with_query).unwrap_err();
    assert_eq!(issues[0].field, "query");
    assert_eq!(issues[0].message, "is only valid for operation=search");

    let list_with_view = SkillIndexRequest {
        operation: "list".to_string(),
        view: "details".to_string(),
        recursive: true,
        ..SkillIndexRequest::default()
    };
    assert!(validate(&list_with_view).is_ok());

    let search_with_view = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        view: "details".to_string(),
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&search_with_view)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["view"]);

    let read_with_tree = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        directories_only: true,
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&read_with_tree)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["directories_only"]);

    let tree_with_recursive = SkillIndexRequest {
        operation: "list".to_string(),
        view: "tree".to_string(),
        recursive: true,
        directory_entry: true,
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&tree_with_recursive)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["recursive", "directory_entry"]);

    let flat_with_depth = SkillIndexRequest {
        operation: "list".to_string(),
        view: "details".to_string(),
        max_depth: Some(2),
        directories_only: true,
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&flat_with_depth)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["max_depth", "directories_only"]);

    let name_with_result = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        search_match: "name".to_string(),
        result: "matches".to_string(),
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&name_with_result)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["result"]);
}

#[test]
fn read_rules_require_a_scope_and_a_consistent_mode() {
    let no_scope = SkillIndexRequest {
        operation: "read".to_string(),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&no_scope).unwrap_err();
    assert!(issue_index(&issues, "skills").is_some());

    let both = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        paths: Some(vec!["/Data".to_string()]),
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&both)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert!(fields.contains(&"paths".to_string()));

    let full_with_lines = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        line_count: Some(2),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&full_with_lines).unwrap_err();
    assert!(issue_index(&issues, "line_count").is_some());

    let head_with_range = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "head".to_string(),
        line_count: Some(2),
        start_line: Some(1),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&head_with_range).unwrap_err();
    assert!(issue_index(&issues, "start_line").is_some());

    let range_without_bounds = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "range".to_string(),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&range_without_bounds).unwrap_err();
    assert!(issue_index(&issues, "start_line").is_some());

    let reversed = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        read_mode: "range".to_string(),
        start_line: Some(5),
        end_line: Some(2),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&reversed).unwrap_err();
    let index = issue_index(&issues, "end_line").unwrap();
    assert_eq!(
        issues[index].message,
        "must be greater than or equal to start_line"
    );

    let category = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        category: Some("Data".to_string()),
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&category)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["category"]);

    let scoped = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        category: Some("Data".to_string()),
        paths: Some(vec!["/Data".to_string()]),
        ..SkillIndexRequest::default()
    };
    let fields: Vec<String> = validate(&scoped)
        .unwrap_err()
        .into_iter()
        .map(|issue| issue.field)
        .collect();
    assert_eq!(fields, ["paths"]);
}

#[test]
fn output_modes_reject_reads_and_pipelines() {
    let read_count = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        output_mode: "count".to_string(),
        ..SkillIndexRequest::default()
    };
    let issues = validate(&read_count).unwrap_err();
    assert_eq!(issues[0].field, "output_mode");

    let list_count = SkillIndexRequest {
        operation: "list".to_string(),
        output_mode: "count".to_string(),
        ..SkillIndexRequest::default()
    };
    assert_eq!(
        validate(&list_count).unwrap().output_mode,
        OutputMode::Count
    );

    let count_with_pipeline = SkillIndexRequest {
        operation: "list".to_string(),
        output_mode: "count".to_string(),
        pipeline: vec![PipelineStage::limit(2)],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&count_with_pipeline).unwrap_err();
    assert_eq!(issues[0].field, "pipeline");
    assert_eq!(issues[0].message, "is not valid with output_mode=count");
}

#[test]
fn pipeline_stages_are_validated_per_operation() {
    assert_eq!(
        validate_pipeline(&[PipelineStage::limit(3)]).unwrap(),
        [PipelineStage::limit(3)]
    );
    assert_eq!(
        validate_pipeline(&[PipelineStage::slice(2, 4), PipelineStage::count()])
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        validate_pipeline(&[PipelineStage::filter("pdf")]).unwrap()[0]
            .query
            .as_deref(),
        Some("pdf")
    );

    let missing_lines = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage {
            operation: "limit".to_string(),
            ..PipelineStage::default()
        }],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&missing_lines).unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0].lines");
    assert_eq!(issues[0].message, "is required for limit");

    let mut extra = PipelineStage::limit(2);
    extra.query = Some("pdf".to_string());
    let issues = validate(&SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![extra],
        ..SkillIndexRequest::default()
    })
    .unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0]");
    assert_eq!(issues[0].message, "has fields not valid for limit");

    let reversed = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage::slice(5, 4)],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&reversed).unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0].end_line");
    assert_eq!(issues[0].message, "must be >= start_line");

    let too_many_lines = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage::limit(5_001)],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&too_many_lines).unwrap_err();
    assert_eq!(issues[0].message, "must not exceed 5000");

    let count_with_fields = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage {
            operation: "count".to_string(),
            invert: true,
            ..PipelineStage::default()
        }],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&count_with_fields).unwrap_err();
    assert_eq!(issues[0].message, "count takes no other fields");

    let filter_missing_query = SkillIndexRequest {
        operation: "list".to_string(),
        pipeline: vec![PipelineStage {
            operation: "filter".to_string(),
            ..PipelineStage::default()
        }],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&filter_missing_query).unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0].query");
    assert_eq!(issues[0].message, "is required for filter");
}

#[test]
fn batch_queries_accept_one_limit_stage() {
    let batched = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string(), "beta".to_string()]),
        pipeline: vec![PipelineStage::limit(4)],
        ..SkillIndexRequest::default()
    };
    let validated = validate(&batched).unwrap();
    assert_eq!(validated.per_query_limit, Some(4));

    let too_many = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string(), "beta".to_string()]),
        pipeline: vec![PipelineStage::limit(11)],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&too_many).unwrap_err();
    assert_eq!(issues[0].field, "pipeline[0].lines");
    assert_eq!(issues[0].message, "must not exceed 10 for queries");

    let two_stages = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string(), "beta".to_string()]),
        pipeline: vec![PipelineStage::limit(2), PipelineStage::count()],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&two_stages).unwrap_err();
    assert_eq!(issues[0].field, "pipeline");
    assert_eq!(
        issues[0].message,
        "queries support only one limit pipeline stage"
    );

    let conflicting = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string()]),
        per_query_limit: Some(3),
        pipeline: vec![PipelineStage::limit(2)],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&conflicting).unwrap_err();
    assert_eq!(issues[0].field, "per_query_limit");
    assert_eq!(issues[0].message, "conflicts with pipeline limit");

    let agreeing = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string()]),
        per_query_limit: Some(2),
        pipeline: vec![PipelineStage::limit(2)],
        ..SkillIndexRequest::default()
    };
    assert_eq!(validate(&agreeing).unwrap().per_query_limit, Some(2));
}

#[test]
fn issues_accumulate_in_one_pass() {
    let request = SkillIndexRequest {
        operation: "read".to_string(),
        view: "tree".to_string(),
        query: Some(Vec::new()),
        max_output_chars: Some(1),
        line_count: Some(2),
        pipeline: vec![PipelineStage {
            operation: "limit".to_string(),
            ..PipelineStage::default()
        }],
        ..SkillIndexRequest::default()
    };
    let issues = validate(&request).unwrap_err();
    let fields: Vec<&str> = issues.iter().map(|issue| issue.field.as_str()).collect();
    for expected in [
        "skills",
        "view",
        "query",
        "max_output_chars",
        "line_count",
        "pipeline[0].lines",
    ] {
        assert!(
            fields.contains(&expected),
            "missing {expected} in {fields:?}"
        );
    }
    assert!(issues.len() >= 6);
    for issue in &issues {
        assert!(!issue.message.is_empty());
        assert!(issue.to_string().starts_with(&issue.field));
    }
}

#[test]
fn simple_defaults_shape_the_model_facing_call() {
    let list = SkillIndexRequest {
        operation: "list".to_string(),
        view: "names".to_string(),
        ..SkillIndexRequest::default()
    };
    let validated = apply_simple_defaults(&list).unwrap();
    assert_eq!(validated.view, ListView::Details);
    assert_eq!(validated.page_size, Some(0));

    let limited = SkillIndexRequest {
        operation: "list".to_string(),
        limit: Some(5),
        ..SkillIndexRequest::default()
    };
    let validated = apply_simple_defaults(&limited).unwrap();
    assert_eq!(validated.view, ListView::Details);
    assert_eq!(validated.page_size, Some(5));
    assert_eq!(validated.limit, Some(5));

    let search = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string(), "beta".to_string()]),
        ..SkillIndexRequest::default()
    };
    let validated = apply_simple_defaults(&search).unwrap();
    assert_eq!(validated.page_size, Some(20));

    let one = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["alpha".to_string()]),
        ..SkillIndexRequest::default()
    };
    assert_eq!(apply_simple_defaults(&one).unwrap().page_size, Some(10));

    let read = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        ..SkillIndexRequest::default()
    };
    assert_eq!(apply_simple_defaults(&read).unwrap().page_size, None);

    let too_big = SkillIndexRequest {
        operation: "list".to_string(),
        limit: Some(11),
        ..SkillIndexRequest::default()
    };
    let issues = apply_simple_defaults(&too_big).unwrap_err();
    assert_eq!(issues[0].field, "limit");
    assert_eq!(issues[0].message, "must not exceed 10 for list");

    let list_with_query = SkillIndexRequest {
        operation: "list".to_string(),
        query: Some(vec!["x".to_string()]),
        ..SkillIndexRequest::default()
    };
    assert_eq!(
        apply_simple_defaults(&list_with_query).unwrap_err()[0].field,
        "query"
    );

    let search_with_skills = SkillIndexRequest {
        operation: "search".to_string(),
        query: Some(vec!["x".to_string()]),
        skills: Some(vec!["alpha".to_string()]),
        ..SkillIndexRequest::default()
    };
    assert_eq!(
        apply_simple_defaults(&search_with_skills).unwrap_err()[0].field,
        "skills"
    );

    let read_with_query = SkillIndexRequest {
        operation: "read".to_string(),
        skills: Some(vec!["alpha".to_string()]),
        query: Some(vec!["x".to_string()]),
        ..SkillIndexRequest::default()
    };
    assert!(apply_simple_defaults(&read_with_query).is_err());
}

#[tokio::test]
async fn category_paths_resolve_against_the_view() {
    let (_directory, _catalog, view) = fixture_view().await;
    assert_eq!(
        validate_category_paths(&view, Some("Data > Export"), None).unwrap(),
        ["/Data/Export"]
    );
    assert_eq!(
        validate_category_paths(&view, None, Some(&["/Data".to_string()])).unwrap(),
        ["/Data"]
    );
    assert_eq!(validate_category_paths(&view, None, None).unwrap(), ["/"]);
    let conflict =
        validate_category_paths(&view, Some("Data"), Some(&["/Data".to_string()])).unwrap_err();
    assert_eq!(
        conflict.message(),
        "paths: category and paths are mutually exclusive"
    );
    assert!(validate_category_paths(&view, Some("Missing"), None).is_err());
}

#[test]
fn requests_round_trip_through_json() {
    let decoded: SkillIndexRequest = serde_json::from_str(
        r#"{"operation":"search","match":"name","query":["pdf"],"pipeline":[{"operation":"limit","lines":2}],"unknown":true}"#,
    )
    .unwrap();
    assert_eq!(decoded.operation, "search");
    assert_eq!(decoded.search_match, "name");
    assert_eq!(decoded.query, Some(vec!["pdf".to_string()]));
    assert_eq!(decoded.pipeline[0].lines, Some(2));
    assert!(decoded.case_insensitive);

    let encoded = serde_json::to_value(&decoded).unwrap();
    assert_eq!(encoded["match"], "name");
    assert!(encoded.get("search_match").is_none());

    let validated = validate(&decoded).unwrap();
    assert_eq!(validated.search_match, SearchMatch::Name);
    let round_trip: ValidatedRequest =
        serde_json::from_value(serde_json::to_value(&validated).unwrap()).unwrap();
    assert_eq!(round_trip, validated);

    let defaults = SkillIndexRequest::default();
    assert_eq!(defaults.operation, "list");
    assert_eq!(defaults.read_mode, "full");
    assert_eq!(defaults.output_mode, "entries");
    assert_eq!(defaults.result, "files");
    assert!(defaults.paths.is_none());
    assert_eq!(defaults.view, "names");
}

#[tokio::test]
async fn rendered_rows_stay_stable_across_rebuilds() {
    let (directory, catalog) = fixture().await;
    let reopened = SkillFs::open(directory.path()).await.unwrap();
    assert_eq!(catalog.records().len(), reopened.records().len());
    let first_view = SkillDirectoryView::from_catalog(&catalog);
    let second_view = SkillDirectoryView::from_catalog(&reopened);
    assert_eq!(first_view.node_paths(), second_view.node_paths());
    let root = vec!["/".to_string()];
    let render = |view: &SkillDirectoryView| -> Vec<String> {
        view.entries(&root, true, None, false, false)
            .unwrap()
            .iter()
            .map(|entry| list_row(entry, ListView::Tree, view).text)
            .collect()
    };
    let first = render(&first_view);
    assert_eq!(first, render(&second_view));
    assert_eq!(first.len(), 14);
    assert_eq!(first[0], "  - [category] Data  desc: Contains 3 skills.");
    assert_eq!(
        first[3],
        "      - [skill] beta  category: Data > Export  path: beta/SKILL.md  desc: Exports tables"
    );
    assert_eq!(
        first[13],
        "      - [skill] sigma  category: 数据 > 导出  path: sigma/SKILL.md  desc: 导出数据"
    );
}
