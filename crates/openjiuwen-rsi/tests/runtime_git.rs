//! Offline integration suite for the git command layer.
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/auto_harness/infra/git_operations.py`
//! (v0.1.20, `GitOperations`) and `worktree_manager.py`'s `_slugify`, plus
//! `git_auth.py`'s `build_git_auth_env` and `git_operations`'
//! `_sanitize_git_output`. Every case scripts a [`FakeCommandRunner`] with the
//! exact `git` output a real repository would produce, so no test touches a
//! repository, the network, or a subprocess.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use openjiuwen_rsi::runtime::command::{
    CommandOutcome, CommandRunner, FakeCommandRunner, FakeReply,
};
use openjiuwen_rsi::runtime::git::{
    build_auth_env, scrub_credentials, slugify_topic, GitCherryPick, GitOperations, GitStatus,
    GitStepResult,
};

/// Builds operations over a fake runner rooted at `/repo`.
fn operations(runner: &Arc<FakeCommandRunner>) -> GitOperations {
    let concrete: Arc<FakeCommandRunner> = Arc::clone(runner);
    let runner: Arc<dyn CommandRunner> = concrete;
    GitOperations::new(runner, "/repo")
}

/// The porcelain fixture: modified, added, deleted, renamed, untracked, and a
/// duplicate modified entry.
const STATUS_FIXTURE: &str = " M src/a.rs\nA  src/b.rs\n D src/c.rs\nR  old/name.rs -> new/name.rs\n?? notes.txt\n M src/a.rs\n";

#[tokio::test]
async fn collect_status_parses_and_normalises_every_bucket() {
    let runner = Arc::new(FakeCommandRunner::new().reply(
        FakeReply::outcome("git", CommandOutcome::success_with(STATUS_FIXTURE)).with_args([
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]),
    ));
    let git = operations(&runner);
    let status = git.collect_status().await.unwrap();
    assert_eq!(
        status.tracked_modified_files,
        vec!["new/name.rs", "src/a.rs", "src/b.rs", "src/c.rs"]
    );
    assert_eq!(status.untracked_files, vec!["notes.txt"]);
    assert_eq!(status.renamed_files, vec!["new/name.rs"]);
    assert_eq!(
        status.dirty_files,
        vec![
            "new/name.rs",
            "notes.txt",
            "src/a.rs",
            "src/b.rs",
            "src/c.rs"
        ]
    );

    let calls = runner.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].args,
        vec!["status", "--porcelain", "--untracked-files=all"]
    );
    assert_eq!(calls[0].cwd.as_deref(), Some(Path::new("/repo")));
    assert!(calls[0].merge_stderr);
    assert_eq!(
        calls[0].env.get("GIT_TERMINAL_PROMPT").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        calls[0].env.get("GIT_ASKPASS").map(String::as_str),
        Some("")
    );
    assert_eq!(
        calls[0].env.get("GCM_INTERACTIVE").map(String::as_str),
        Some("never")
    );
}

#[tokio::test]
async fn collect_status_handles_backslashes_and_git_failure() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with(" M dir\\file.rs\n"))
                .with_args(["status"]),
        ),
    );
    let git = operations(&runner);
    let status = git.collect_status().await.unwrap();
    assert_eq!(status.tracked_modified_files, vec!["dir/file.rs"]);
    assert_eq!(status.dirty_files, vec!["dir/file.rs"]);

    let failing = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::failure_with(128, "not a repository"))
                .with_args(["status"]),
        ),
    );
    let failed = operations(&failing).collect_status().await.unwrap();
    assert_eq!(failed, GitStatus::default());
}

#[tokio::test]
async fn list_dirty_files_uses_the_status_union() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with(STATUS_FIXTURE))
                .with_args(["status"]),
        ),
    );
    let dirty = operations(&runner).list_dirty_files().await.unwrap();
    assert_eq!(
        dirty,
        vec![
            "new/name.rs",
            "notes.txt",
            "src/a.rs",
            "src/b.rs",
            "src/c.rs"
        ]
    );
}

#[tokio::test]
async fn add_paths_rejects_an_empty_list_without_running_git() {
    let runner = Arc::new(FakeCommandRunner::new());
    let result = operations(&runner).add_paths(&[]).await.unwrap();
    assert!(!result.success);
    assert_eq!(result.output, "no paths provided");
    assert_eq!(runner.call_count(), 0);
}

#[tokio::test]
async fn add_paths_stages_after_a_double_dash() {
    let runner = Arc::new(FakeCommandRunner::new().reply(
        FakeReply::outcome("git", CommandOutcome::success_with("")).with_args(["add", "--"]),
    ));
    let paths = vec!["src/a.rs".to_owned(), "docs/b.md".to_owned()];
    let result = operations(&runner).add_paths(&paths).await.unwrap();
    assert!(result.success);
    assert_eq!(
        runner.calls()[0].args,
        vec!["add", "--", "src/a.rs", "docs/b.md"]
    );
}

#[tokio::test]
async fn create_branch_reports_failure_as_data() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome(
                "git",
                CommandOutcome::failure_with(128, "fatal: a branch named 'feature' already exists"),
            )
            .with_args(["checkout", "-b", "feature"]),
        ),
    );
    let result = operations(&runner).create_branch("feature").await.unwrap();
    assert!(!result.success);
    assert_eq!(result.branch.as_deref(), Some("feature"));
    assert!(result.output.contains("already exists"));
}

#[tokio::test]
async fn current_branch_and_head_map_to_options() {
    let branch_runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome(
                "git",
                CommandOutcome::success_with("auto-harness/fix-issue-9\n"),
            )
            .with_args(["rev-parse", "--abbrev-ref"]),
        ),
    );
    let branch = operations(&branch_runner).current_branch().await.unwrap();
    assert_eq!(branch.as_deref(), Some("auto-harness/fix-issue-9"));

    let head_runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with("abc123\n"))
                .with_args(["rev-parse", "HEAD"]),
        ),
    );
    let head = operations(&head_runner).current_head().await.unwrap();
    assert_eq!(head.as_deref(), Some("abc123"));

    let failing = Arc::new(
        FakeCommandRunner::new().fallback(CommandOutcome::failure_with(
            128,
            "fatal: not a git repository",
        )),
    );
    let git = operations(&failing);
    assert_eq!(git.current_branch().await.unwrap(), None);
    assert_eq!(git.current_head().await.unwrap(), None);
}

#[tokio::test]
async fn has_commits_against_base_parses_the_count() {
    let runner = Arc::new(FakeCommandRunner::new().reply(
        FakeReply::outcome("git", CommandOutcome::success_with("3\n")).with_args([
            "rev-list",
            "--count",
            "origin/develop..HEAD",
        ]),
    ));
    assert!(operations(&runner)
        .has_commits_against_base()
        .await
        .unwrap());

    let zero = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with("0"))
                .with_args(["rev-list", "--count"]),
        ),
    );
    assert!(!operations(&zero).has_commits_against_base().await.unwrap());

    let unparseable = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with("not-a-number\n"))
                .with_args(["rev-list", "--count"]),
        ),
    );
    assert!(!operations(&unparseable)
        .has_commits_against_base()
        .await
        .unwrap());

    let failing = Arc::new(
        FakeCommandRunner::new().fallback(CommandOutcome::failure_with(128, "fatal: bad revision")),
    );
    assert!(!operations(&failing)
        .has_commits_against_base()
        .await
        .unwrap());
}

#[tokio::test]
async fn diff_helpers_normalise_deduplicate_and_target_the_base() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome(
                "git",
                CommandOutcome::success_with("src\\a.rs\nsrc/a.rs\n\nb.rs\n"),
            )
            .with_args(["diff", "--name-only"]),
        ),
    );
    let git = operations(&runner).with_base_branch("main");
    let files = git.diff_name_only("HEAD").await.unwrap();
    assert_eq!(files, vec!["src/a.rs", "b.rs"]);

    let against = git.diff_name_only_against_base().await.unwrap();
    assert_eq!(against, vec!["src/a.rs", "b.rs"]);
    let calls = runner.calls();
    assert_eq!(calls[0].args, vec!["diff", "--name-only", "HEAD"]);
    assert_eq!(
        calls[1].args,
        vec!["diff", "--name-only", "origin/main...HEAD"]
    );
}

#[tokio::test]
async fn diff_stat_helpers_trim_and_use_the_base_revision() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome("git", CommandOutcome::success_with(" 1 file changed\n"))
                .with_args(["diff", "--stat"]),
        ),
    );
    let git = operations(&runner).with_base_branch("main");
    assert_eq!(git.diff_stat().await.unwrap(), "1 file changed");
    assert_eq!(
        git.diff_stat_against_base().await.unwrap(),
        "1 file changed"
    );
    assert_eq!(
        runner.calls()[1].args,
        vec!["diff", "--stat", "origin/main...HEAD"]
    );
}

#[tokio::test]
async fn diff_against_base_keeps_raw_output() {
    let runner = Arc::new(
        FakeCommandRunner::new().reply(
            FakeReply::outcome(
                "git",
                CommandOutcome::success_with("diff --git a/x b/x\n+line"),
            )
            .with_args(["diff", "origin/develop...HEAD"]),
        ),
    );
    let diff = operations(&runner).diff_against_base().await.unwrap();
    assert_eq!(diff, "diff --git a/x b/x\n+line");
}

#[tokio::test]
async fn small_helpers_forward_git_output() {
    let runner =
        Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("fake output")));
    let git = operations(&runner);
    let commit = git.commit("message").await.unwrap();
    assert_eq!(commit.output, "fake output");
    assert!(commit.success);
    assert_eq!(git.status_porcelain().await.unwrap(), "fake output");
    assert_eq!(git.show_last_commit_stat().await.unwrap(), "fake output");
    assert!(git.discard_worktree_changes().await.unwrap());
    assert!(git.worktree_prune().await.unwrap().success);

    let calls = runner.calls();
    assert_eq!(calls[0].args, vec!["commit", "-m", "message"]);
    assert_eq!(
        calls[1].args,
        vec!["status", "--porcelain", "--untracked-files=all"]
    );
    assert_eq!(
        calls[2].args,
        vec!["show", "--stat", "--format=fuller", "-1"]
    );
    assert_eq!(calls[3].args, vec!["checkout", "."]);
    assert_eq!(calls[4].args, vec!["worktree", "prune"]);
}

#[tokio::test]
async fn worktree_add_and_remove_use_the_expected_arguments() {
    let runner = Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("ok")));
    let git = operations(&runner);
    let added = git
        .worktree_add(
            Path::new("/tmp/wt"),
            "auto-harness/fix-issue-9",
            "origin/develop",
        )
        .await
        .unwrap();
    assert!(added.success);
    let removed = git.worktree_remove(Path::new("/tmp/wt")).await.unwrap();
    assert!(removed.success);

    let calls = runner.calls();
    assert_eq!(
        calls[0].args,
        vec![
            "worktree",
            "add",
            "-b",
            "auto-harness/fix-issue-9",
            "/tmp/wt",
            "origin/develop",
        ]
    );
    assert_eq!(
        calls[1].args,
        vec!["worktree", "remove", "--force", "/tmp/wt"]
    );
}

#[tokio::test]
async fn push_escalates_and_stops_at_the_first_success() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome("git", CommandOutcome::failure_with(1, "lease rejected"))
                    .with_args(["--force-with-lease"]),
            )
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with("pushed"))
                    .with_args(["HEAD:refs/heads/fix-issue-9"]),
            )
            .fallback(CommandOutcome::failure_with(2, "plain rejected")),
    );
    let git = operations(&runner).with_remote("origin");
    let result = git.push("fix-issue-9").await.unwrap();
    assert!(result.success);
    assert!(result
        .output
        .contains("[1] cmd=git push --force-with-lease origin fix-issue-9 code=1"));
    assert!(result.output.contains("lease rejected"));
    assert!(result
        .output
        .contains("[2] cmd=git push origin fix-issue-9 code=2"));
    assert!(result.output.contains("plain rejected"));
    assert!(
        result
            .output
            .contains("[3] cmd=git push origin HEAD:refs/heads/fix-issue-9 code=0"),
        "{}",
        result.output
    );
    assert_eq!(runner.call_count(), 3);
    assert_eq!(
        runner.command_lines(),
        vec![
            "git push --force-with-lease origin fix-issue-9",
            "git push origin fix-issue-9",
            "git push origin HEAD:refs/heads/fix-issue-9",
        ]
    );
}

#[tokio::test]
async fn push_reports_every_failed_attempt() {
    let runner =
        Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::failure_with(3, "rejected")));
    let result = operations(&runner).push("topic").await.unwrap();
    assert!(!result.success);
    assert_eq!(runner.call_count(), 3);
    for attempt in 1..=3 {
        assert!(
            result.output.contains(&format!("[{attempt}] cmd=git push")),
            "{}",
            result.output
        );
    }
    assert_eq!(result.output.matches("code=3").count(), 3);
}

#[tokio::test]
async fn cherry_pick_failure_aborts_and_keeps_the_commit_list() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with("abc123\ndef456\n"))
                    .with_args(["rev-list", "--reverse", "origin/develop..feature"]),
            )
            .reply(
                FakeReply::outcome("git", CommandOutcome::failure_with(1, "CONFLICT (content)"))
                    .with_args(["cherry-pick", "origin/develop..feature"]),
            )
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with("aborted"))
                    .with_args(["cherry-pick", "--abort"]),
            ),
    );
    let result = operations(&runner)
        .cherry_pick_ref_commits("feature")
        .await
        .unwrap();
    assert!(!result.success);
    assert_eq!(result.commits, vec!["abc123", "def456"]);
    assert!(result.output.contains("CONFLICT"));
    assert!(result.output.contains("aborted"));

    let calls = runner.calls();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls[0].args,
        vec!["rev-list", "--reverse", "origin/develop..feature"]
    );
    assert_eq!(
        calls[1].args,
        vec!["cherry-pick", "origin/develop..feature"]
    );
    assert_eq!(calls[2].args, vec!["cherry-pick", "--abort"]);
}

#[tokio::test]
async fn cherry_pick_success_does_not_abort() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with("abc123"))
                    .with_args(["rev-list", "--reverse"]),
            )
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with("applied"))
                    .with_args(["cherry-pick"]),
            )
            .fallback(CommandOutcome::failure_with(1, "should not be called")),
    );
    let result = operations(&runner)
        .cherry_pick_ref_commits("feature")
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.commits, vec!["abc123"]);
    assert_eq!(result.output, "applied");
    assert_eq!(runner.call_count(), 2);
    assert_eq!(
        runner.calls()[1].args,
        vec!["cherry-pick", "origin/develop..feature"]
    );
}

#[tokio::test]
async fn raw_applies_workspace_environment_and_timeout() {
    let runner = Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("ok")));
    let git = operations(&runner)
        .with_base_branch("main")
        .with_remote("fork")
        .with_env("FOO", "bar")
        .with_timeout(Duration::from_secs(7));
    assert_eq!(git.base_branch(), "main");
    assert_eq!(git.remote(), "fork");
    assert_eq!(git.program(), "git");
    assert_eq!(git.workspace(), Path::new("/repo"));
    assert_eq!(git.timeout(), Some(Duration::from_secs(7)));
    assert_eq!(
        git.env().get("GIT_TERMINAL_PROMPT").map(String::as_str),
        Some("0")
    );
    assert_eq!(git.runner().name(), "fake");

    assert!(git.raw(&["fetch", "origin"]).await.unwrap().success());
    let spec = &runner.calls()[0];
    assert_eq!(spec.program, "git");
    assert_eq!(spec.args, vec!["fetch", "origin"]);
    assert_eq!(spec.cwd.as_deref(), Some(Path::new("/repo")));
    assert!(spec.merge_stderr);
    assert_eq!(spec.env.get("FOO").map(String::as_str), Some("bar"));
    assert_eq!(spec.timeout, Some(Duration::from_secs(7)));
}

#[test]
fn slugify_topic_matches_the_python_rules() {
    assert_eq!(slugify_topic("fix-issue-42"), "fix-issue-42");
    assert_eq!(slugify_topic("#7"), "fix-issue-7");
    assert_eq!(slugify_topic("Issue 9"), "fix-issue-9");
    assert_eq!(slugify_topic("issue-12: crash"), "fix-issue-12");
    assert_eq!(slugify_topic("Fix a broken thing!"), "fix-a-broken-thing");
    assert_eq!(slugify_topic(""), "task");
    assert_eq!(slugify_topic("修复问题"), "task");
    let long = slugify_topic(&"a".repeat(60));
    assert_eq!(long, "a".repeat(40));
    assert_eq!(long.chars().count(), 40);
}

#[test]
fn build_auth_env_injects_the_basic_extraheader() {
    let env = build_auth_env("alice", "secret-token");
    assert_eq!(
        env.get("GIT_TERMINAL_PROMPT").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        env.get("GCM_INTERACTIVE").map(String::as_str),
        Some("never")
    );
    assert_eq!(env.get("GIT_CONFIG_COUNT").map(String::as_str), Some("6"));
    assert_eq!(
        env.get("GIT_CONFIG_KEY_2").map(String::as_str),
        Some("http.https://gitcode.com/.extraheader")
    );
    let expected_header = format!(
        "AUTHORIZATION: basic {}",
        STANDARD.encode("alice:secret-token")
    );
    assert_eq!(
        env.get("GIT_CONFIG_VALUE_2").map(String::as_str),
        Some(expected_header.as_str())
    );
    assert_eq!(env.get("GIT_CONFIG_VALUE_0").map(String::as_str), Some(""));

    let anonymous = build_auth_env("", "");
    assert!(!anonymous.contains_key("GIT_CONFIG_COUNT"));
    assert_eq!(
        anonymous.get("GIT_TERMINAL_PROMPT").map(String::as_str),
        Some("0")
    );
}

#[test]
fn scrub_credentials_redacts_tokens_urls_and_headers() {
    let basic = STANDARD.encode("alice:secret-token");
    let text = format!(
        "url https://alice:secret-token@gitcode.com/repo basic {basic} Authorization: Bearer abc private-token=xyz&next=1"
    );
    let scrubbed = scrub_credentials(&text, "alice", "secret-token");
    assert!(!scrubbed.contains("secret-token"));
    assert!(!scrubbed.contains(&basic));
    assert!(scrubbed.contains("https://***:***@gitcode.com/repo"));
    assert!(scrubbed.contains("Authorization: ***"));
    assert!(scrubbed.contains("private-token=***&next=1"));

    // Without a user only the token and header values are redacted.
    let token_only = scrub_credentials("token=secret-token access_token=v", "", "secret-token");
    assert!(!token_only.contains("secret-token"));
    assert!(token_only.contains("access_token=***"));
}

#[test]
fn git_results_round_trip_and_tolerate_unknown_fields() {
    let step: GitStepResult =
        serde_json::from_str(r#"{"success":true,"output":"ok","extra":1}"#).unwrap();
    assert!(step.success);
    assert_eq!(step.output, "ok");

    let pick: GitCherryPick = serde_json::from_str(r#"{"success":false,"commits":["a"]}"#).unwrap();
    assert!(!pick.success);
    assert_eq!(pick.commits, vec!["a"]);
    assert_eq!(pick.output, "");

    let status: GitStatus =
        serde_json::from_str(r#"{"dirty_files":["a"],"unknown":{"x":1}}"#).unwrap();
    assert_eq!(status.dirty_files, vec!["a"]);
    assert_eq!(status.untracked_files, Vec::<String>::new());
}
