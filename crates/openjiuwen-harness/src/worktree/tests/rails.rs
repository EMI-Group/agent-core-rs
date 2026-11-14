//! `WorktreeRail` scenarios: inert without a worktree, the published section's
//! shape and content, the explicit-name override, a failing summary, the
//! character bound, the notice text, and the three publishing hooks.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_agent::RailEvent;
use serde_json::Value;

use super::{listing, temp_root, ScriptedGitRunner};
use crate::prompts::PromptCategory;
use crate::rails::{prompt_sections, HarnessRail, HarnessRailContext};
use crate::worktree::{
    build_worktree_notice, MemoryWorktreeSessionStore, WorktreeChangeSummary, WorktreeContext,
    WorktreeManager, WorktreeRail, WorktreeSessionState, WorktreeSessionStore,
    WORKTREE_RAIL_PRIORITY, WORKTREE_SECTION_MAX_CHARS, WORKTREE_SECTION_NAME,
    WORKTREE_SECTION_PRIORITY, WORKTREE_SUMMARY_EXTRA_KEY,
};

/// A manager over a scripted runner plus the session store shared with the tools.
struct Fixture {
    manager: Arc<WorktreeManager>,
    runner: Arc<ScriptedGitRunner>,
    store: Arc<MemoryWorktreeSessionStore>,
}

impl Fixture {
    fn new(repo: &Path) -> Self {
        let runner = Arc::new(ScriptedGitRunner::new());
        let manager = Arc::new(WorktreeManager::with_runner(repo, runner.clone()));
        Self {
            manager,
            runner,
            store: Arc::new(MemoryWorktreeSessionStore::new()),
        }
    }

    fn target(&self, name: &str) -> PathBuf {
        self.manager.target_path(name).unwrap()
    }

    fn rail(&self, language: &str) -> WorktreeRail {
        WorktreeRail::new(self.manager.clone(), self.store.clone(), language)
    }
}

/// The stdout of `rev-parse --show-toplevel` for `repo`.
fn root(repo: &Path) -> String {
    repo.to_string_lossy().into_owned()
}

/// A session for the worktree `name` of `repo`.
fn session(fixture: &Fixture, repo: &Path, name: &str) -> WorktreeSessionState {
    let mut session =
        WorktreeSessionState::new(name, fixture.target(name), Some(format!("worktree-{name}")));
    session.original_cwd = repo.to_path_buf();
    session
}

/// The porcelain status of a single modified file.
const STATUS: &str = " M src/lib.rs\n";

/// The structured summary published for the fixture worktree.
fn expected_summary() -> Value {
    serde_json::json!({
        "changed_files": 1,
        "files": ["src/lib.rs"],
        "commits": 0,
        "unpushed": true,
        "diffstat": "",
        "truncated": false,
    })
}

#[tokio::test]
async fn no_worktree_publishes_nothing_and_runs_no_git() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let rail = fixture.rail("en");

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    rail.before_model_call(&mut ctx).await.unwrap();

    assert_eq!(prompt_sections(&ctx).len(), 0);
    assert!(!ctx.extra().contains_key(WORKTREE_SUMMARY_EXTRA_KEY));
    assert_eq!(fixture.runner.calls().len(), 0);
    assert_eq!(fixture.store.session(), None);
}

#[tokio::test]
async fn a_session_publishes_the_section_and_the_structured_summary() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.target("feature");
    fixture
        .store
        .set_session(Some(session(&fixture, &repo, "feature")));
    let rail = fixture.rail("en");

    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    fixture.runner.push_ok(STATUS);
    fixture.runner.push_ok("deadbeef\n");

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    rail.before_model_call(&mut ctx).await.unwrap();

    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    let section = &sections[0];
    assert_eq!(section.name, WORKTREE_SECTION_NAME);
    assert_eq!(section.name, "worktree");
    assert_eq!(section.priority, WORKTREE_SECTION_PRIORITY);
    assert_eq!(section.priority, 85);
    assert_eq!(section.category, Some(PromptCategory::SystemPrompt));
    assert_eq!(section.carrier, "system_message");

    let path = target.display().to_string();
    for language in ["cn", "en"] {
        let content = section.content.get(language).unwrap();
        assert!(content.contains(&path), "{language} content names the path");
        assert!(
            content.contains("worktree-feature"),
            "{language} content names the branch"
        );
        assert!(
            content.contains("feature"),
            "{language} content names the tree"
        );
    }
    assert_ne!(section.content["cn"], section.content["en"]);

    let summary = ctx.extra().get(WORKTREE_SUMMARY_EXTRA_KEY).unwrap();
    assert_eq!(summary, &expected_summary());

    assert_eq!(
        fixture.runner.commands(),
        [
            "rev-parse --show-toplevel",
            "worktree list --porcelain",
            "status --porcelain",
            "rev-list --max-count=1 HEAD --not --remotes",
        ]
    );
    assert_eq!(fixture.runner.remaining(), 0);
}

#[tokio::test]
async fn an_explicit_worktree_overrides_the_absent_session() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.target("feature");
    assert_eq!(fixture.store.session(), None);
    let rail = fixture.rail("cn").with_worktree("feature");
    assert_eq!(rail.worktree_name(), Some("feature"));

    // Resolution lists the worktrees, the summary lists them again.
    for _ in 0..2 {
        fixture.runner.push_ok(&root(&repo));
        fixture
            .runner
            .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    }
    fixture.runner.push_ok(STATUS);
    fixture.runner.push_ok("deadbeef\n");

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
    rail.before_invoke(&mut ctx).await.unwrap();

    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    let path = target.display().to_string();
    for language in ["cn", "en"] {
        let content = &sections[0].content[language];
        assert!(content.contains(&path));
        assert!(content.contains("worktree-feature"));
    }
    assert_eq!(ctx.extra()[WORKTREE_SUMMARY_EXTRA_KEY], expected_summary());
    assert_eq!(fixture.runner.commands().len(), 6);
}

#[tokio::test]
async fn a_failing_summary_still_publishes_the_section() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.target("feature");
    fixture
        .store
        .set_session(Some(session(&fixture, &repo, "feature")));
    let rail = fixture.rail("en");

    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    fixture.runner.push_fail("fatal: unable to read status");

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    rail.before_model_call(&mut ctx).await.unwrap();

    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    let path = target.display().to_string();
    assert!(sections[0].content["en"].contains(&path));
    assert!(sections[0].content["en"].contains("worktree-feature"));
    assert!(sections[0].content["en"].contains("Change summary: unavailable"));
    assert!(sections[0].content["cn"].contains("变更摘要: 不可用"));
    assert!(!ctx.extra().contains_key(WORKTREE_SUMMARY_EXTRA_KEY));
    assert_eq!(
        fixture.runner.commands(),
        [
            "rev-parse --show-toplevel",
            "worktree list --porcelain",
            "status --porcelain",
        ]
    );
}

#[tokio::test]
async fn an_unlisted_worktree_publishes_a_minimal_section() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let rail = fixture.rail("en").with_worktree("gone");

    // Resolution lists only the main worktree; the summary probe cannot find it.
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));
    fixture.runner.push_ok(&root(&repo));
    fixture.runner.push_ok(&listing(&repo, None));

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
    rail.before_invoke(&mut ctx).await.unwrap();

    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    let path = fixture.target("gone").display().to_string();
    assert!(sections[0].content["en"].contains(&path));
    assert!(sections[0].content["en"].contains("Change summary: unavailable"));
    assert!(!ctx.extra().contains_key(WORKTREE_SUMMARY_EXTRA_KEY));
}

#[tokio::test]
async fn a_long_summary_is_bounded_in_both_languages() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.target("feature");
    fixture
        .store
        .set_session(Some(session(&fixture, &repo, "feature")));
    let rail = fixture.rail("en");

    let mut long = String::new();
    for index in 0..30 {
        long.push_str(" M src/");
        long.push_str(&"a".repeat(80));
        long.push_str(&index.to_string());
        long.push('\n');
    }

    fixture.runner.push_ok(&root(&repo));
    fixture
        .runner
        .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
    fixture.runner.push_ok(&long);
    fixture.runner.push_ok("");

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
    rail.before_model_call(&mut ctx).await.unwrap();

    let sections = prompt_sections(&ctx);
    assert_eq!(sections.len(), 1);
    for language in ["cn", "en"] {
        let content = &sections[0].content[language];
        assert!(content.chars().count() <= WORKTREE_SECTION_MAX_CHARS);
        assert!(content.ends_with("..."), "{language} marks truncation");
    }
    let summary = ctx.extra().get(WORKTREE_SUMMARY_EXTRA_KEY).unwrap();
    assert_eq!(summary["truncated"], Value::Bool(true));
    assert_eq!(summary["changed_files"], Value::from(30));
}

#[test]
fn a_direct_context_render_is_bounded_for_a_long_diffstat() {
    let context = WorktreeContext {
        name: "feature".to_owned(),
        path: PathBuf::from("/repo/.worktrees/feature"),
        branch: Some("worktree-feature".to_owned()),
        summary: Some(WorktreeChangeSummary {
            changed_files: 40,
            files: (0..25)
                .map(|index| format!("src/file-{index}.rs"))
                .collect(),
            commits: 12,
            unpushed: Some(true),
            diffstat: "z".repeat(5_000),
            truncated: true,
        }),
    };
    for language in ["cn", "en"] {
        let rendered = context.render(language);
        assert!(rendered.chars().count() <= WORKTREE_SECTION_MAX_CHARS);
        assert!(rendered.ends_with("..."));
    }
}

#[test]
fn the_notice_is_deterministic_and_names_both_roots() {
    let parent = Path::new("/parent");
    let worktree = Path::new("/parent/.worktrees/feature");
    let en = build_worktree_notice(parent, worktree, "en");
    assert_eq!(en, build_worktree_notice(parent, worktree, "en"));
    assert!(en.starts_with("You are operating in an isolated git worktree at"));
    assert!(en.contains("/parent/.worktrees/feature"));
    assert!(en.contains("refer to /parent"));
    assert!(en.contains("Your changes stay in this worktree"));

    let cn = build_worktree_notice(parent, worktree, "cn");
    assert_eq!(cn, build_worktree_notice(parent, worktree, "zh"));
    assert_ne!(cn, en);
    assert!(cn.contains("/parent/.worktrees/feature"));
    assert!(cn.contains("/parent"));
}

#[tokio::test]
async fn every_hook_publishes_and_the_rail_reports_its_identity() {
    let (_dir, repo) = temp_root();
    let fixture = Fixture::new(&repo);
    let target = fixture.target("feature");
    fixture
        .store
        .set_session(Some(session(&fixture, &repo, "feature")));

    let rail = fixture.rail("en");
    assert_eq!(rail.name(), "worktree");
    assert_eq!(rail.priority(), WORKTREE_RAIL_PRIORITY);
    assert_eq!(rail.priority(), 100);
    assert_eq!(rail.language(), "en");
    assert_eq!(rail.worktree_name(), None);
    assert_eq!(rail.manager().repo_root(), fixture.manager.repo_root());

    for _ in 0..3 {
        fixture.runner.push_ok(&root(&repo));
        fixture
            .runner
            .push_ok(&listing(&repo, Some((&target, "worktree-feature"))));
        fixture.runner.push_fail("fatal: unable to read status");
    }

    for event in [
        RailEvent::BeforeInvoke,
        RailEvent::BeforeModelCall,
        RailEvent::AfterInvoke,
    ] {
        let mut ctx = HarnessRailContext::for_event(event);
        let outcome = match event {
            RailEvent::BeforeInvoke => rail.before_invoke(&mut ctx).await,
            RailEvent::AfterInvoke => rail.after_invoke(&mut ctx).await,
            _ => rail.before_model_call(&mut ctx).await,
        };
        outcome.unwrap();
        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1, "{event:?} publishes one section");
        assert_eq!(sections[0].name, WORKTREE_SECTION_NAME);
        assert_eq!(sections[0].priority, WORKTREE_SECTION_PRIORITY);
        assert_eq!(sections[0].category, Some(PromptCategory::SystemPrompt));
        assert_eq!(
            ctx.event(),
            event,
            "the rail must not change the firing event"
        );
    }
    assert_eq!(fixture.runner.commands().len(), 9);

    let lowered = rail.with_priority(7);
    assert_eq!(lowered.priority(), 7);
    assert_eq!(lowered.name(), "worktree");
}
