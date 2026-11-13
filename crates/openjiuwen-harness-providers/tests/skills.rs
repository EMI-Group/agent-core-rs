#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Offline integration suite for the portable skill installer.
//!
//! Behavior ports name their Python counterpart in
//! `tests/unit_tests/harness_providers/test_skills.py` where one exists.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use openjiuwen_common::error::Error;
use openjiuwen_harness_providers::{
    install_for_provider, install_skills, normalize_skills, skill_scan_dir, SkillConflict,
    SkillInstaller, SkillLayout, SkillMode, SkillSource,
};
use serde_json::json;
use tempfile::tempdir;

/// Writes a complete bundle (front matter, body, a script and a binary resource).
///
/// Port of `test_skills.py::bundle`.
fn write_bundle(root: &Path, name: &str, text: &str) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: Test skill\n---\n{text}\n"),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("scripts")).unwrap();
    std::fs::write(root.join("scripts").join("run.sh"), "echo resource\n").unwrap();
    std::fs::write(root.join(".hidden"), b"\x00\x01resource").unwrap();
    root.to_path_buf()
}

/// Creates (if needed) and canonicalizes a project directory.
fn project_dir(path: &Path) -> PathBuf {
    std::fs::create_dir_all(path).unwrap();
    path.canonicalize().unwrap()
}

/// Returns `true` while a staging directory is left inside `project`.
fn has_stage_dir(project: &Path) -> bool {
    std::fs::read_dir(project).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".openjiuwen-skill-")
    })
}

/// Asserts that a declaration fails with the given Python message.
fn assert_message(value: &serde_json::Value, expected: &str) {
    let error = SkillSource::from_json(value).expect_err("invalid declaration");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(error.message().as_ref(), expected);
}

#[test]
fn from_json_accepts_a_string_directory() {
    let source = SkillSource::from_json(&json!("/tmp/skills")).unwrap();
    assert_eq!(source, SkillSource::new("/tmp/skills"));
    assert_eq!(source.dir, PathBuf::from("/tmp/skills"));
    assert_eq!(source.mode, SkillMode::All);
    assert_eq!(source.enabled_skills, None);
}

#[test]
fn from_json_accepts_object_forms() {
    let by_dir = SkillSource::from_json(&json!({"dir": "/tmp/one"})).unwrap();
    assert_eq!(by_dir, SkillSource::new("/tmp/one"));

    let by_source = SkillSource::from_json(&json!({"source": "/tmp/two"})).unwrap();
    assert_eq!(by_source, SkillSource::new("/tmp/two"));

    let all = SkillSource::from_json(&json!({"dir": "/tmp/three", "mode": "all"})).unwrap();
    assert_eq!(all.mode, SkillMode::All);

    let listed = SkillSource::from_json(&json!({
        "source": "/tmp/four",
        "mode": "auto_list",
        "enabled_skills": ["one", "two"],
    }))
    .unwrap();
    assert_eq!(listed.dir, PathBuf::from("/tmp/four"));
    assert_eq!(listed.mode, SkillMode::AutoList);
    assert_eq!(
        listed.enabled_skills,
        Some(vec!["one".to_owned(), "two".to_owned()])
    );

    let unbounded =
        SkillSource::from_json(&json!({"dir": "/tmp/five", "enabled_skills": []})).unwrap();
    assert_eq!(unbounded.enabled_skills, Some(Vec::new()));
}

#[test]
fn builders_set_mode_and_enabled_skills() {
    let source = SkillSource::new("/tmp/skills")
        .with_mode(SkillMode::AutoList)
        .with_enabled_skills(["alpha", "beta"]);
    assert_eq!(source.dir, PathBuf::from("/tmp/skills"));
    assert_eq!(source.mode, SkillMode::AutoList);
    assert_eq!(
        source.enabled_skills,
        Some(vec!["alpha".to_owned(), "beta".to_owned()])
    );
}

#[test]
fn from_json_rejects_invalid_declarations() {
    assert_message(&json!(""), "skill dir must be a non-empty path");
    assert_message(&json!({}), "skill dir must be a non-empty path");
    assert_message(&json!({"dir": ""}), "skill dir must be a non-empty path");
    assert_message(
        &json!({"mode": "all"}),
        "skill dir must be a non-empty path",
    );
    assert_message(
        &json!({"dir": "/tmp/x", "mode": "merge"}),
        "skill mode must be all or auto_list",
    );
    assert_message(
        &json!({"dir": "/tmp/x", "mode": 1}),
        "skill mode must be all or auto_list",
    );
    assert_message(
        &json!({"dir": "/tmp/x", "enabled_skills": "one"}),
        "enabled_skills must be an array of names",
    );
    assert_message(
        &json!({"dir": "/tmp/x", "enabled_skills": ["one", 2]}),
        "enabled_skills must be an array of names",
    );
    assert_message(
        &json!({"dir": "/tmp/x", "frobnicate": true, "alpha": 1}),
        "unknown skill source fields: alpha, frobnicate",
    );
    for value in [json!(["a"]), json!(7), json!(null), json!(true)] {
        assert_message(
            &value,
            "skill source must be a directory or manifest skill declaration",
        );
    }
}

#[test]
fn skill_source_round_trips_through_serde() {
    let source = SkillSource::new("/tmp/skills")
        .with_mode(SkillMode::AutoList)
        .with_enabled_skills(["alpha"]);

    let value = serde_json::to_value(&source).unwrap();
    assert_eq!(
        value,
        json!({"dir": "/tmp/skills", "mode": "auto_list", "enabled_skills": ["alpha"]})
    );
    assert_eq!(
        serde_json::from_value::<SkillSource>(value).unwrap(),
        source
    );

    let plain: SkillSource = serde_json::from_value(json!("/tmp/plain")).unwrap();
    assert_eq!(plain, SkillSource::new("/tmp/plain"));

    let invalid: Result<SkillSource, serde_json::Error> =
        serde_json::from_value(json!({"dir": ""}));
    assert!(invalid.is_err());
}

#[test]
fn skill_mode_and_conflict_serialize_snake_case() {
    assert_eq!(serde_json::to_value(SkillMode::All).unwrap(), json!("all"));
    assert_eq!(
        serde_json::to_value(SkillMode::AutoList).unwrap(),
        json!("auto_list")
    );
    assert_eq!(
        serde_json::from_value::<SkillMode>(json!("auto_list")).unwrap(),
        SkillMode::AutoList
    );

    assert_eq!(
        serde_json::to_value(SkillConflict::Skip).unwrap(),
        json!("skip")
    );
    assert_eq!(
        serde_json::to_value(SkillConflict::Replace).unwrap(),
        json!("replace")
    );
    assert_eq!(
        serde_json::to_value(SkillConflict::Append).unwrap(),
        json!("append")
    );
    assert_eq!(
        serde_json::from_value::<SkillConflict>(json!("append")).unwrap(),
        SkillConflict::Append
    );
}

#[test]
fn skill_conflict_parses_wire_values() {
    assert_eq!(
        SkillConflict::from_str("skip").unwrap(),
        SkillConflict::Skip
    );
    assert_eq!(
        SkillConflict::from_str("replace").unwrap(),
        SkillConflict::Replace
    );
    assert_eq!(
        SkillConflict::from_str("append").unwrap(),
        SkillConflict::Append
    );

    let error = SkillConflict::from_str("merge").unwrap_err();
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(
        error.message().as_ref(),
        "skill_conflict must be skip, replace or append"
    );
    assert_eq!(
        error.to_string(),
        "validation: skill_conflict must be skip, replace or append"
    );
}

#[test]
fn normalize_skills_mixes_strings_and_objects() {
    let values = [
        json!("/tmp/one"),
        json!({"source": "/tmp/two", "mode": "auto_list"}),
    ];
    let sources = normalize_skills(&values, SkillConflict::Skip).unwrap();
    assert_eq!(
        sources,
        vec![
            SkillSource::new("/tmp/one"),
            SkillSource::new("/tmp/two").with_mode(SkillMode::AutoList),
        ]
    );

    assert_eq!(
        normalize_skills(&[], SkillConflict::Append).unwrap(),
        Vec::<SkillSource>::new()
    );

    let error = normalize_skills(&[json!("/tmp/one"), json!({})], SkillConflict::Skip).unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "skill dir must be a non-empty path"
    );
}

#[test]
fn skill_scan_dir_maps_builtin_providers() {
    assert_eq!(skill_scan_dir("claudecode"), Some(".claude/skills"));
    assert_eq!(skill_scan_dir("codex"), Some(".agents/skills"));
    assert_eq!(skill_scan_dir("dsh"), Some(".dsh/skills"));
    assert_eq!(skill_scan_dir("native"), None);
    assert_eq!(skill_scan_dir(""), None);
}

#[test]
fn install_copies_bundle_and_applies_conflict_policies() {
    // Port of `test_skills.py::test_complete_bundle_and_conflict_policies` (claudecode branch).
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "instructions");
    let project = project_dir(&temp.path().join("project"));
    let sources = [SkillSource::new(&source)];

    let target = project.join(".claude/skills/example");
    assert_eq!(
        install_skills(&sources, "claudecode", &project, SkillConflict::Skip).unwrap(),
        vec![target.clone()]
    );
    assert_eq!(
        std::fs::read(target.join(".hidden")).unwrap(),
        b"\x00\x01resource"
    );
    assert!(target.join("scripts").join("run.sh").is_file());

    // Skip keeps the installed copy even when the source changed and the target is stale.
    std::fs::write(
        source.join("SKILL.md"),
        "---\nname: example\ndescription: Test skill\n---\nupdated\n",
    )
    .unwrap();
    std::fs::write(target.join("stale"), "old").unwrap();
    assert_eq!(
        install_skills(&sources, "claudecode", &project, SkillConflict::Skip).unwrap(),
        Vec::<PathBuf>::new()
    );
    let installed_text = std::fs::read_to_string(target.join("SKILL.md")).unwrap();
    assert!(!installed_text.contains("updated"));
    assert!(target.join("stale").exists());

    // Replace swaps the bundle wholesale.
    assert_eq!(
        install_skills(&sources, "claudecode", &project, SkillConflict::Replace).unwrap(),
        vec![target.clone()]
    );
    assert!(std::fs::read_to_string(target.join("SKILL.md"))
        .unwrap()
        .contains("updated"));
    assert!(!target.join("stale").exists());

    // Append installs name-2, then name-3, rewriting the front-matter name.
    let second = project.join(".claude/skills/example-2");
    let third = project.join(".claude/skills/example-3");
    assert_eq!(
        install_skills(&sources, "claudecode", &project, SkillConflict::Append).unwrap(),
        vec![second.clone()]
    );
    assert_eq!(
        install_skills(&sources, "claudecode", &project, SkillConflict::Append).unwrap(),
        vec![third.clone()]
    );
    let second_text = std::fs::read_to_string(second.join("SKILL.md")).unwrap();
    assert!(second_text.contains("name: example-2"));
    assert!(second_text.contains("description: Test skill"));
    assert!(second_text.ends_with("---\nupdated\n"));
    let third_text = std::fs::read_to_string(third.join("SKILL.md")).unwrap();
    assert!(third_text.contains("name: example-3"));
    assert_eq!(
        std::fs::read_to_string(target.join("SKILL.md")).unwrap(),
        std::fs::read_to_string(source.join("SKILL.md")).unwrap()
    );
    assert_eq!(
        std::fs::read(second.join(".hidden")).unwrap(),
        b"\x00\x01resource"
    );
    assert!(!has_stage_dir(&project));
}

#[test]
fn install_discovers_nested_libraries_and_filters_enabled_skills() {
    // Port of `test_skills.py::test_library_selection_and_frontmatter_name_collision`.
    let temp = tempdir().unwrap();
    let library = temp.path().join("library");
    write_bundle(&library.join("different-folder"), "selected", "one");
    write_bundle(&library.join("group/other"), "other", "two");
    let project = project_dir(&temp.path().join("project"));
    let existing = write_bundle(
        &project.join(".agents/skills/legacy-folder"),
        "selected",
        "keep",
    );

    let sources = normalize_skills(
        &[json!({
            "dir": library.to_string_lossy(),
            "enabled_skills": ["selected"],
            "mode": "auto_list",
        })],
        SkillConflict::Skip,
    )
    .unwrap();

    // The existing bundle declares `selected` under another directory name, so it is a match.
    assert_eq!(
        install_skills(&sources, "codex", &project, SkillConflict::Skip).unwrap(),
        Vec::<PathBuf>::new()
    );
    assert_eq!(
        install_skills(&sources, "codex", &project, SkillConflict::Replace).unwrap(),
        vec![existing.clone()]
    );
    let replaced = std::fs::read_to_string(existing.join("SKILL.md")).unwrap();
    assert!(!replaced.contains("keep"));
    assert!(replaced.contains("one"));
    assert!(!project.join(".agents/skills/other").exists());
    assert!(!project.join(".agents/skills/selected").exists());

    // A filter that selects no bundle installs nothing.
    let none = normalize_skills(
        &[json!({"dir": library.to_string_lossy(), "enabled_skills": ["absent"]})],
        SkillConflict::Skip,
    )
    .unwrap();
    assert_eq!(
        install_skills(&none, "codex", &project, SkillConflict::Skip).unwrap(),
        Vec::<PathBuf>::new()
    );
}

#[test]
fn installed_destinations_are_sorted_by_path() {
    let temp = tempdir().unwrap();
    let zeta = write_bundle(&temp.path().join("zeta"), "zeta", "z");
    let alpha = write_bundle(&temp.path().join("alpha"), "alpha", "a");
    let project = project_dir(&temp.path().join("project"));

    let installed = install_skills(
        &[SkillSource::new(&zeta), SkillSource::new(&alpha)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap();
    assert_eq!(
        installed,
        vec![
            project.join(".agents/skills/alpha"),
            project.join(".agents/skills/zeta"),
        ]
    );
}

#[test]
fn install_is_idempotent_when_the_source_is_the_destination() {
    // Port of the self-copy part of `test_skills.py::test_symlink_escape_is_rejected_and_self_copy_is_noop`.
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let project = project_dir(&temp.path().join("project"));
    let target = project.join(".agents/skills/example");

    assert_eq!(
        install_skills(
            &[SkillSource::new(&source)],
            "codex",
            &project,
            SkillConflict::Skip
        )
        .unwrap(),
        vec![target.clone()]
    );
    assert_eq!(
        install_skills(
            &[SkillSource::new(&target)],
            "codex",
            &project,
            SkillConflict::Replace
        )
        .unwrap(),
        Vec::<PathBuf>::new()
    );
    assert!(target.join("SKILL.md").is_file());
    assert!(!has_stage_dir(&project));
}

#[test]
fn install_with_no_sources_returns_early() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));
    assert_eq!(
        install_skills(&[], "native", &project, SkillConflict::Skip).unwrap(),
        Vec::<PathBuf>::new()
    );
    assert!(!project.join(".agents").exists());
}

#[test]
fn install_rejects_providers_without_a_scan_dir() {
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let project = project_dir(&temp.path().join("project"));

    let error = install_skills(
        &[SkillSource::new(&source)],
        "native",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "provider 'native' declares no skill discovery directory"
    );
}

#[test]
fn install_rejects_sources_that_are_not_directories() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));

    let file = temp.path().join("not-a-directory");
    std::fs::write(&file, "x").unwrap();
    let error = install_skills(
        &[SkillSource::new(&file)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    let expected = format!(
        "skill source is not a directory: {}",
        file.canonicalize().unwrap().display()
    );
    assert_eq!(error.message().as_ref(), expected.as_str());

    let missing = temp.path().join("missing");
    let error = install_skills(
        &[SkillSource::new(&missing)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert!(matches!(error, Error::External(_)), "{error:?}");
}

#[test]
fn install_rejects_libraries_without_bundles() {
    let temp = tempdir().unwrap();
    let library = temp.path().join("library");
    std::fs::create_dir_all(library.join("empty")).unwrap();
    let project = project_dir(&temp.path().join("project"));

    let error = install_skills(
        &[SkillSource::new(&library)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    let expected = format!(
        "no SKILL.md bundles found in {}",
        library.canonicalize().unwrap().display()
    );
    assert_eq!(error.message().as_ref(), expected.as_str());
}

#[test]
fn install_rejects_containment_between_source_and_destination() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));
    let library = project.join(".dsh/skills/library");
    write_bundle(&library.join("nested"), "library", "inner");

    // The bundle resolves to `.dsh/skills/library`, which contains the source itself.
    let error = install_skills(
        &[SkillSource::new(&library)],
        "dsh",
        &project,
        SkillConflict::Replace,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "skill source and destination must not contain each other"
    );
}

#[test]
fn install_rejects_invalid_portable_names() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));

    for (declared, message) in [
        ("../escape", "invalid portable skill name: '../escape'"),
        ("bad name", "invalid portable skill name: 'bad name'"),
        (
            "'quoted name'",
            "invalid portable skill name: 'quoted name'",
        ),
    ] {
        let source = temp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(
            source.join("SKILL.md"),
            format!("---\nname: {declared}\n---\nbody\n"),
        )
        .unwrap();
        let error = install_skills(
            &[SkillSource::new(&source)],
            "claudecode",
            &project,
            SkillConflict::Skip,
        )
        .unwrap_err();
        assert_eq!(error.message().as_ref(), message);
    }
}

#[test]
fn install_rejects_unterminated_front_matter() {
    let temp = tempdir().unwrap();
    let source = temp.path().join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("SKILL.md"), "---\nname: example\n").unwrap();
    let project = project_dir(&temp.path().join("project"));

    let error = install_skills(
        &[SkillSource::new(&source)],
        "claudecode",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        format!(
            "unterminated skill front matter: {}",
            source.canonicalize().unwrap().display()
        )
    );
}

#[test]
fn install_of_a_bundle_without_front_matter_uses_the_directory_name() {
    let temp = tempdir().unwrap();
    let source = temp.path().join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("SKILL.md"), "just instructions\n").unwrap();
    let project = project_dir(&temp.path().join("project"));

    assert_eq!(
        install_skills(
            &[SkillSource::new(&source)],
            "codex",
            &project,
            SkillConflict::Skip
        )
        .unwrap(),
        vec![project.join(".agents/skills/source")]
    );
    assert!(!has_stage_dir(&project));
}

#[cfg(unix)]
#[test]
fn install_rejects_scan_dir_escaping_the_project() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, project.join(".claude")).unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");

    let error = install_skills(
        &[SkillSource::new(&source)],
        "claudecode",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "skill discovery directory escapes the project"
    );
}

#[cfg(unix)]
#[test]
fn install_rejects_symlinks_escaping_the_bundle() {
    // Port of `test_skills.py::test_symlink_escape_is_rejected_and_self_copy_is_noop` (first half).
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let outside = temp.path().join("outside.txt");
    std::fs::write(&outside, "outside").unwrap();
    std::os::unix::fs::symlink(&outside, source.join("escape")).unwrap();
    let project = project_dir(&temp.path().join("project"));

    let error = install_skills(
        &[SkillSource::new(&source)],
        "dsh",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert!(
        error.message().contains("skill symlink escapes its bundle"),
        "{error:?}"
    );
}

#[cfg(unix)]
#[test]
fn install_rejects_dangling_symlinks() {
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    std::os::unix::fs::symlink(source.join("missing"), source.join("dangling")).unwrap();
    let project = project_dir(&temp.path().join("project"));

    let error = install_skills(
        &[SkillSource::new(&source)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert!(matches!(error, Error::External(_)), "{error:?}");
}

#[cfg(unix)]
#[test]
fn install_materializes_internal_symlinks() {
    // Port of `test_skills.py::test_internal_symlinks_are_materialized`.
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    std::os::unix::fs::symlink(source.join(".hidden"), source.join("reference")).unwrap();
    let project = project_dir(&temp.path().join("project"));

    install_skills(
        &[SkillSource::new(&source)],
        "codex",
        &project,
        SkillConflict::Skip,
    )
    .unwrap();
    let target = project.join(".agents/skills/example/reference");
    assert!(!std::fs::symlink_metadata(&target)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::read(&target).unwrap(),
        std::fs::read(source.join(".hidden")).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn install_replaces_a_symlinked_destination_without_mutating_the_source() {
    // Port of `test_skills.py::test_replace_destination_symlink_copies_files_without_mutating_source`.
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let project = project_dir(&temp.path().join("project"));
    let scan = project.join(".agents/skills");
    std::fs::create_dir_all(&scan).unwrap();
    std::os::unix::fs::symlink(&source, scan.join("example")).unwrap();

    install_skills(
        &[SkillSource::new(&source)],
        "codex",
        &project,
        SkillConflict::Replace,
    )
    .unwrap();
    let destination = scan.join("example");
    assert!(destination.is_dir());
    assert!(!std::fs::symlink_metadata(&destination)
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(destination.join("scripts").join("run.sh").is_file());
    assert!(source.join("SKILL.md").is_file());
}
/// Writes a flat `*.md` skill document and returns its path.
fn write_flat_document(root: &Path, file_name: &str, name: Option<&str>) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    let document = root.join(file_name);
    let text = match name {
        Some(name) => format!("---\nname: {name}\ndescription: Flat skill\n---\nflat body\n"),
        None => "flat body\n".to_owned(),
    };
    std::fs::write(&document, text).unwrap();
    document
}
#[test]
fn installer_install_targets_each_vendor_discovery_directory() {
    let temp = tempdir().unwrap();
    let library = temp.path().join("library");
    write_bundle(&library.join("first"), "first", "one");
    write_bundle(&library.join("second"), "second", "two");
    let sources = [SkillSource::new(&library)];
    for (provider, scan, layout) in [
        ("claudecode", ".claude/skills", SkillLayout::Directory),
        ("codex", ".agents/skills", SkillLayout::Directory),
        ("dsh", ".dsh/skills", SkillLayout::DirectoryOrFlat),
    ] {
        let project = project_dir(&temp.path().join(provider));
        let installer = SkillInstaller::for_provider(provider, SkillConflict::Skip).unwrap();
        assert_eq!(installer.provider(), provider);
        assert_eq!(installer.scan_dir(), scan);
        assert_eq!(installer.layout(), layout);
        assert_eq!(
            installer.install(&sources, Some(&project)).unwrap(),
            vec![
                project.join(scan).join("first"),
                project.join(scan).join("second"),
            ]
        );
        for name in ["first", "second"] {
            assert!(project.join(scan).join(name).join("SKILL.md").is_file());
            assert!(project
                .join(scan)
                .join(name)
                .join("scripts")
                .join("run.sh")
                .is_file());
        }
        assert!(!has_stage_dir(&project));
    }
}
#[test]
fn installer_keeps_the_conflict_policy() {
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let project = project_dir(&temp.path().join("project"));
    let sources = [SkillSource::new(&source)];
    let target = project.join(".agents/skills/example");
    let installer = SkillInstaller::for_provider("codex", SkillConflict::Skip).unwrap();
    assert_eq!(
        installer.install(&sources, Some(&project)).unwrap(),
        vec![target.clone()]
    );
    // Skip keeps the installed copy even when the source changed.
    std::fs::write(
        source.join("SKILL.md"),
        "---\nname: example\ndescription: Test skill\n---\nupdated\n",
    )
    .unwrap();
    assert_eq!(
        installer.install(&sources, Some(&project)).unwrap(),
        Vec::<PathBuf>::new()
    );
    assert!(!std::fs::read_to_string(target.join("SKILL.md"))
        .unwrap()
        .contains("updated"));
    // Replace swaps the bundle wholesale.
    let replace = SkillInstaller::for_provider("codex", SkillConflict::Replace).unwrap();
    assert_eq!(
        replace.install(&sources, Some(&project)).unwrap(),
        vec![target.clone()]
    );
    assert!(std::fs::read_to_string(target.join("SKILL.md"))
        .unwrap()
        .contains("updated"));
    // Append installs the next free name-N and rewrites the front-matter name.
    let append = SkillInstaller::for_provider("codex", SkillConflict::Append).unwrap();
    assert_eq!(
        append.install(&sources, Some(&project)).unwrap(),
        vec![project.join(".agents/skills/example-2")]
    );
    assert!(
        std::fs::read_to_string(project.join(".agents/skills/example-2/SKILL.md"))
            .unwrap()
            .contains("name: example-2")
    );
    assert!(!has_stage_dir(&project));
}
#[test]
fn installer_normalizes_a_dsh_flat_bundle_into_a_directory_bundle() {
    let temp = tempdir().unwrap();
    let document = write_flat_document(
        &temp.path().join("source"),
        "flat-skill.md",
        Some("flat-skill"),
    );
    let project = project_dir(&temp.path().join("project"));
    let installed = install_for_provider(
        &[SkillSource::new(&document)],
        "dsh",
        Some(&project),
        SkillConflict::Skip,
    )
    .unwrap();
    let bundle = project.join(".dsh/skills/flat-skill");
    assert_eq!(installed, vec![bundle.clone()]);
    assert!(bundle.join("SKILL.md").is_file());
    assert!(!project.join(".dsh/skills/flat-skill.md").exists());
    assert_eq!(
        std::fs::read_to_string(bundle.join("SKILL.md")).unwrap(),
        std::fs::read_to_string(&document).unwrap()
    );
}
#[test]
fn installer_rejects_flat_documents_for_the_directory_layouts() {
    let temp = tempdir().unwrap();
    let document = write_flat_document(
        &temp.path().join("source"),
        "flat-skill.md",
        Some("flat-skill"),
    );
    let project = project_dir(&temp.path().join("project"));
    for provider in ["claudecode", "codex"] {
        let error = install_for_provider(
            &[SkillSource::new(&document)],
            provider,
            Some(&project),
            SkillConflict::Skip,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(
            error.message().as_ref(),
            format!("the {provider} skill layout requires a SKILL.md directory bundle")
        );
    }
    assert_eq!(std::fs::read_dir(&project).unwrap().count(), 0);
}
#[test]
fn installer_rejects_a_non_kebab_dsh_name_without_creating_the_directory() {
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));
    let document = write_flat_document(&temp.path().join("flat"), "My_Skill.md", None);
    let error = install_for_provider(
        &[SkillSource::new(&document)],
        "dsh",
        Some(&project),
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "DSH skill names must be kebab-case: 'My_Skill'"
    );
    let bundle = write_bundle(&temp.path().join("bundle"), "My_Skill", "text");
    let error = install_for_provider(
        &[SkillSource::new(&bundle)],
        "dsh",
        Some(&project),
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "DSH skill names must be kebab-case: 'My_Skill'"
    );
    assert!(!project.join(".dsh").exists());
    assert_eq!(std::fs::read_dir(&project).unwrap().count(), 0);
}
#[test]
fn installer_with_an_unknown_provider_does_no_filesystem_work() {
    let temp = tempdir().unwrap();
    let source = write_bundle(&temp.path().join("source"), "example", "text");
    let project = project_dir(&temp.path().join("project"));
    let error = install_for_provider(
        &[SkillSource::new(&source)],
        "native",
        Some(&project),
        SkillConflict::Skip,
    )
    .unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "unknown skill provider 'native'; expected one of claudecode, codex, dsh"
    );
    assert_eq!(std::fs::read_dir(&project).unwrap().count(), 0);
    let error = SkillInstaller::for_provider("claude-code", SkillConflict::Skip).unwrap_err();
    assert_eq!(
        error.message().as_ref(),
        "unknown skill provider 'claude-code'; expected one of claudecode, codex, dsh"
    );
}
#[test]
fn installer_resolves_the_names_the_vendor_harnesses_install_with() {
    // Wiring check: `providers::{claudecode,codex,dsh}/harness.rs` call
    // `install_for_provider(&sources, "<registry name>", Some(&project), conflict)`, so every
    // literal provider string used there must resolve to a vendor installer.
    let temp = tempdir().unwrap();
    let project = project_dir(&temp.path().join("project"));
    for provider in ["claudecode", "codex", "dsh"] {
        let installer = SkillInstaller::for_provider(provider, SkillConflict::Skip).unwrap();
        assert_eq!(skill_scan_dir(provider), Some(installer.scan_dir()));
        assert_eq!(
            install_for_provider(&[], provider, Some(&project), SkillConflict::Skip).unwrap(),
            Vec::<PathBuf>::new()
        );
    }
    assert_eq!(std::fs::read_dir(&project).unwrap().count(), 0);
}
