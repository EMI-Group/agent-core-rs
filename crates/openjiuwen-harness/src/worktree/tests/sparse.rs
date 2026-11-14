//! Sparse-checkout scenarios: path validation, normalisation, serde shape, and
//! the exact `git` command construction of every sparse helper.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use openjiuwen_common::error::ErrorKind;

use super::ScriptedGitRunner;
use crate::worktree::{
    checkout_head, is_sparse_failure, sparse_checkout_disable, sparse_checkout_reapply,
    sparse_checkout_set, sparse_failure_error, validate_sparse_path, worktree_add_no_checkout,
    worktree_add_no_checkout_existing, SparseConfig, SparseMode, MAX_SPARSE_PATHS,
    SPARSE_FAILURE_MARKER,
};

/// The worktree path every command test uses.
fn worktree() -> PathBuf {
    PathBuf::from("/repo/.worktrees/feature")
}

#[test]
fn hostile_paths_are_refused_and_never_normalised() {
    for path in [
        "/etc/passwd",
        "\\windows",
        "~/.ssh",
        "C:/repo",
        "D:\\repo",
        "../outside",
        "src/../etc",
        "src/..",
    ] {
        let error = validate_sparse_path(path).expect_err(path);
        assert_eq!(error.kind(), ErrorKind::Security, "{path}");
    }

    let error = validate_sparse_path("src/\0name").expect_err("NUL");
    assert_eq!(error.kind(), ErrorKind::Security);
    assert!(error.to_string().contains("NUL"));
}

#[test]
fn empty_and_control_paths_are_validation_errors() {
    for path in ["", "   ", "\t", "src/\u{7}name"] {
        let error = validate_sparse_path(path).expect_err(path);
        assert_eq!(error.kind(), ErrorKind::Validation, "{path}");
    }
}

#[test]
fn config_validation_counts_paths_and_dedupes_on_normalisation() {
    let duplicated = SparseConfig::cone([" src/ ", "src", "lib\\mod", "src//", "docs"])
        .expect("config accepts duplicates");
    assert_eq!(duplicated.paths.len(), 5);
    assert_eq!(duplicated.normalized().paths, ["src", "lib/mod", "docs"]);

    let too_many: Vec<String> = (0..=MAX_SPARSE_PATHS).map(|i| format!("p{i}")).collect();
    let error = SparseConfig::cone(too_many).expect_err("too many paths");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn config_round_trips_with_its_mode_and_defaults_to_cone() {
    let config = SparseConfig::patterns(["*.rs", "docs/"]).expect("config");
    let json = serde_json::to_string(&config).expect("serialize");
    assert!(json.contains(r#""mode":"patterns""#));
    let decoded: SparseConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, config);
    assert_eq!(decoded.mode, SparseMode::Patterns);

    let defaulted: SparseConfig = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(defaulted.paths.len(), 0);
    assert_eq!(defaulted.mode, SparseMode::Cone);
}

#[tokio::test]
async fn sparse_checkout_set_passes_the_cone_flag_and_paths() {
    let runner = ScriptedGitRunner::new();
    runner.push_success();
    let config = SparseConfig::cone(["src", "docs"]).expect("config");
    sparse_checkout_set(&runner, &worktree(), &config)
        .await
        .expect("configure");

    assert_eq!(
        runner.commands(),
        ["sparse-checkout set --cone -- src docs"]
    );
    assert_eq!(runner.calls()[0].cwd, worktree());
}

#[tokio::test]
async fn sparse_checkout_set_normalises_and_uses_no_cone_for_patterns() {
    let runner = ScriptedGitRunner::new();
    runner.push_success();
    let config = SparseConfig::patterns(["*.rs", "src/", "src"]).expect("config");
    sparse_checkout_set(&runner, &worktree(), &config)
        .await
        .expect("configure");

    assert_eq!(
        runner.commands(),
        ["sparse-checkout set --no-cone -- *.rs src"]
    );
}

#[tokio::test]
async fn an_empty_selection_is_refused_without_running_git() {
    let runner = ScriptedGitRunner::new();
    let error = sparse_checkout_set(&runner, &worktree(), &SparseConfig::default())
        .await
        .expect_err("empty selection");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(runner.commands(), Vec::<String>::new());
}

#[tokio::test]
async fn a_hostile_selection_is_refused_without_running_git() {
    let runner = ScriptedGitRunner::new();
    let unvalidated = SparseConfig {
        paths: vec!["../etc".to_owned()],
        mode: SparseMode::Cone,
    };
    let error = sparse_checkout_set(&runner, &worktree(), &unvalidated)
        .await
        .expect_err("hostile selection");
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(runner.commands(), Vec::<String>::new());
}

#[tokio::test]
async fn worktree_add_no_checkout_builds_the_sparse_creation_commands() {
    let runner = ScriptedGitRunner::new();
    runner.push_success();
    runner.push_success();
    worktree_add_no_checkout(
        &runner,
        Path::new("/repo"),
        &worktree(),
        "worktree-feature",
        "HEAD",
    )
    .await
    .expect("create");
    worktree_add_no_checkout_existing(&runner, Path::new("/repo"), &worktree(), "worktree-feature")
        .await
        .expect("recover");

    assert_eq!(
        runner.commands(),
        [
            "worktree add --no-checkout -b worktree-feature /repo/.worktrees/feature HEAD",
            "worktree add --no-checkout /repo/.worktrees/feature worktree-feature",
        ]
    );
    assert_eq!(runner.calls()[0].cwd, PathBuf::from("/repo"));
}

#[tokio::test]
async fn checkout_reapply_and_disable_run_in_the_worktree() {
    let runner = ScriptedGitRunner::new();
    runner.push_success();
    runner.push_success();
    runner.push_success();
    checkout_head(&runner, &worktree()).await.expect("checkout");
    sparse_checkout_reapply(&runner, &worktree())
        .await
        .expect("reapply");
    sparse_checkout_disable(&runner, &worktree())
        .await
        .expect("disable");

    assert_eq!(
        runner.commands(),
        [
            "checkout HEAD",
            "sparse-checkout reapply",
            "sparse-checkout disable"
        ]
    );
    for call in runner.calls() {
        assert_eq!(call.cwd, worktree());
    }
}

#[test]
fn sparse_failures_are_typed_and_marked() {
    let error = sparse_failure_error(Path::new("/repo/.worktrees/feature"), " fatal: boom\n");
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(is_sparse_failure(&error));
    assert!(error.to_string().contains(SPARSE_FAILURE_MARKER));
    assert!(error.to_string().contains("fatal: boom"));
    assert!(error.to_string().contains("/repo/.worktrees/feature"));

    let other = openjiuwen_common::error::Error::tool("git worktree add failed");
    assert!(!is_sparse_failure(&other));
}
