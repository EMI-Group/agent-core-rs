//! Worktree rail: a bounded worktree context section for the harness prompt.
//!
//! Port of the prompt-facing half of
//! `openjiuwen/harness/tools/worktree/{rails,notice}.py`. Python's
//! `WorktreeRail` is an injection rail: it owns a `WorktreeManager`, mounts
//! `enter_worktree`/`exit_worktree` on the agent, restores the session cwd in
//! `before_invoke` and persists the session in `after_invoke`; Python's
//! `DiffSummaryRail` is a lifecycle hook that logs `git diff --stat <base>..HEAD`
//! before an exit. Python has no prompt section for a worktree, so the bounded
//! section published here is the requested Rust surface.
//!
//! # Hooks
//!
//! [`WorktreeRail`] implements the harness [`HarnessRail`] seam at Python's
//! priority [`WORKTREE_RAIL_PRIORITY`] (`100`, the `SysOperationRail` tier) and
//! publishes its section from `before_invoke`, `before_model_call` and
//! `after_invoke` — the outer loop may assemble the system prompt at any of the
//! three points, and publishing into the prompt-section channel is idempotent
//! (Python's upsert-by-name behaviour).
//!
//! A firing with no active worktree publishes nothing and runs no git command.
//! Every expected failure — an unresolvable name, a failing `git worktree list`
//! or change summary — is logged at `debug` and swallowed: a rail must not
//! abort a run because a summary could not be produced. Only the two
//! impossible-in-practice serialization failures (the section summary value) are
//! swallowed the same way. Hooks therefore always return `Ok(())`.
//!
//! # Section
//!
//! One [`PromptSection`] named [`WORKTREE_SECTION_NAME`], priority
//! [`WORKTREE_SECTION_PRIORITY`], [`PromptCategory::SystemPrompt`], bilingual
//! content (`cn` plus `en`) rendered by [`WorktreeContext::render`]. The text
//! names the worktree, its path, its branch and — when a summary was produced —
//! the bounded [`render_change_summary`] output. The whole rendering is capped at
//! [`WORKTREE_SECTION_MAX_CHARS`] characters and ends with a `...` marker when
//! the cap cut anything.
//!
//! The structured summary is additionally written into the rail context's
//! `extra` map under [`WORKTREE_SUMMARY_EXTRA_KEY`], so a host can read the
//! summary without re-parsing the prompt; a firing without a summary removes the
//! key rather than leaving a stale one behind.
//!
//! # Deviations from Python
//!
//! - Python's rail injects the two worktree tools and mirrors the session
//!   through the agent `Session` (including a cwd switch via `set_cwd`); the
//!   tools and the session store already live in [`crate::worktree::tools`], so this rail
//!   only *reads* a [`SharedWorktreeSessions`] store, never mutates it, and
//!   never touches the process cwd.
//! - `DiffSummaryRail` is folded into the section: the summary is computed while
//!   the agent works inside the worktree instead of only on exit, so the model
//!   sees what changed without leaving.
//! - [`build_worktree_notice`] keeps Python's `notice.py` text but is a plain
//!   helper: the host decides where to inject it (the section carries no parent
//!   cwd).
//! - Python's `WorktreeLifecycleRail` hook chain (`AutoSetupRail` setup
//!   commands, commit lint, sync filtering) and the lifecycle/cleanup timers are
//!   deferred; [`WorktreeManager`] has no hook chain.
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::prompts::{PromptCategory, PromptSection};
use crate::rails::{publish_prompt_section, HarnessRail, HarnessRailContext};

use super::changes::{render_change_summary, WorktreeChangeSummary};
use super::manager::WorktreeManager;
use super::slug::slugify;
use super::tools::SharedWorktreeSessions;

/// Name of the prompt section the rail publishes.
pub const WORKTREE_SECTION_NAME: &str = "worktree";

/// Render priority of the worktree prompt section inside the system prompt.
pub const WORKTREE_SECTION_PRIORITY: i32 = 85;

/// Character bound of one rendered worktree section.
pub const WORKTREE_SECTION_MAX_CHARS: usize = 1_200;

/// Dispatch priority of [`WorktreeRail`] (Python's `WorktreeRail.priority`).
pub const WORKTREE_RAIL_PRIORITY: i32 = 100;

/// Key in the rail context `extra` map holding the structured change summary.
pub const WORKTREE_SUMMARY_EXTRA_KEY: &str = "harness.worktree_summary";

/// Marker appended when [`WorktreeContext::render`] has to cut content.
const TRUNCATION_MARKER: &str = "...";

/// Python's `notice.py` text, English.
const NOTICE_EN: &str = "\
You are operating in an isolated git worktree at {worktree_cwd}. \
The parent context lives in {parent_cwd} — same repository, \
same relative file structure, separate working copy.\n\n\
Important:\n\
- Paths from the parent context refer to {parent_cwd}\n\
- Translate them to your worktree root before use\n\
- Re-read files before editing if the parent may have modified them\n\
- Your changes stay in this worktree and will not affect the parent";

/// The Chinese counterpart of [`NOTICE_EN`].
const NOTICE_CN: &str = "\
你正在位于 {worktree_cwd} 的隔离 git worktree 中工作。\
父上下文位于 {parent_cwd} —— 同一个仓库、相同的相对文件结构，但是独立的工作副本。\n\n\
重要提示：\n\
- 父上下文中的路径指向 {parent_cwd}\n\
- 使用前先转换到你的 worktree 根目录\n\
- 如果父上下文可能修改过文件，编辑前请重新读取\n\
- 你的改动会保留在该 worktree 中，不会影响父上下文";

/// Builds the worktree context notice injected into a system prompt.
///
/// Port of `openjiuwen/harness/tools/worktree/notice.py`: it explains that the
/// agent works in an isolated worktree, that parent-context paths refer to
/// `parent_cwd`, that files should be re-read, and that changes stay in the
/// worktree. `language == "en"` selects Python's English text; every other value
/// selects the Chinese counterpart. Pure and deterministic: no clock, no
/// environment, no filesystem.
#[must_use]
pub fn build_worktree_notice(parent_cwd: &Path, worktree_cwd: &Path, language: &str) -> String {
    let template = if language == "en" {
        NOTICE_EN
    } else {
        NOTICE_CN
    };
    template
        .replace("{parent_cwd}", &parent_cwd.to_string_lossy())
        .replace("{worktree_cwd}", &worktree_cwd.to_string_lossy())
}

/// The worktree facts rendered into the prompt section.
///
/// `summary` is absent when the worktree is not currently listed by git or the
/// change summary could not be produced; the rendering says so instead of
/// guessing a clean state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorktreeContext {
    /// Name the worktree was created under.
    pub name: String,
    /// Path of the worktree directory.
    pub path: PathBuf,
    /// Branch checked out in the worktree, when one is known.
    pub branch: Option<String>,
    /// Change summary of the worktree, when one could be produced.
    pub summary: Option<WorktreeChangeSummary>,
}

impl WorktreeContext {
    /// Renders the section text in `language`.
    ///
    /// `language == "en"` selects the English text; every other value selects
    /// the Chinese one (the harness prompt-language contract). The output names
    /// the worktree, its path and its branch, then either the bounded
    /// [`render_change_summary`] text or a note that no summary is available.
    /// The whole rendering is capped at [`WORKTREE_SECTION_MAX_CHARS`] and ends
    /// with a `...` marker when anything was cut. Deterministic: no clock and no
    /// environment is read.
    #[must_use]
    pub fn render(&self, language: &str) -> String {
        let english = language == "en";
        let path = self.path.display().to_string();
        let name = self.name.as_str();
        let mut lines = Vec::new();
        lines.push(if english {
            "## Worktree\n".to_owned()
        } else {
            "## Worktree（隔离工作树）\n".to_owned()
        });
        lines.push(match (&self.branch, english) {
            (Some(branch), true) => format!(
                "You are working in the git worktree `{name}` at `{path}` on branch `{branch}`."
            ),
            (None, true) => format!("You are working in the git worktree `{name}` at `{path}`."),
            (Some(branch), false) => {
                format!("你正在 git worktree `{name}`（路径 `{path}`，分支 `{branch}`）中工作。")
            }
            (None, false) => format!("你正在 git worktree `{name}`（路径 `{path}`）中工作。"),
        });
        match &self.summary {
            Some(summary) => {
                lines.push(if english {
                    "### Changes\n".to_owned()
                } else {
                    "### 变更\n".to_owned()
                });
                lines.push(render_change_summary(
                    summary,
                    if english { "en" } else { "cn" },
                ));
            }
            None => lines.push(if english {
                "Change summary: unavailable — the worktree is not currently listed by git, \
                 or `git status` failed."
                    .to_owned()
            } else {
                "变更摘要: 不可用 —— git 当前未列出该 worktree，或 `git status` 执行失败。"
                    .to_owned()
            }),
        }
        bound_chars(&lines.join("\n"), WORKTREE_SECTION_MAX_CHARS)
    }
}

/// Caps `text` at `max_chars` characters, appending `...` when it clips.
fn bound_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let keep = max_chars.saturating_sub(TRUNCATION_MARKER.len());
    let mut bounded: String = text.chars().take(keep).collect();
    bounded.push_str(TRUNCATION_MARKER);
    bounded
}

/// The resolved location of the active worktree.
struct WorktreeTarget {
    /// Directory the agent works in.
    path: PathBuf,
    /// Branch checked out there, when known.
    branch: Option<String>,
}

/// Publishes a bounded worktree context section into the system prompt.
///
/// The active worktree name comes from [`Self::with_worktree`] when set,
/// otherwise from the session recorded by the worktree tools
/// ([`crate::worktree::tools::EnterWorktreeTool`]); without a name the rail is inert.
/// The path and branch come from a matching session, then from the git worktree
/// listing, then from the layout path ([`WorktreeManager::target_path`]).
/// Change detection runs through [`WorktreeManager::change_summary`], which stays
/// outside the repository lock.
pub struct WorktreeRail {
    /// The manager the summary and the layout path come from.
    manager: Arc<WorktreeManager>,
    /// The store shared with the worktree tools; read-only here.
    sessions: SharedWorktreeSessions,
    /// Prompt language of the owning agent; informational for this rail.
    language: String,
    /// Explicit worktree name overriding the session.
    worktree: Option<String>,
    /// Dispatch priority.
    priority: i32,
}

impl WorktreeRail {
    /// Builds a rail over `manager` reading `sessions`.
    #[must_use]
    pub fn new(
        manager: Arc<WorktreeManager>,
        sessions: SharedWorktreeSessions,
        language: impl Into<String>,
    ) -> Self {
        Self {
            manager,
            sessions,
            language: language.into(),
            worktree: None,
            priority: WORKTREE_RAIL_PRIORITY,
        }
    }

    /// Sets an explicit worktree name, overriding the session.
    #[must_use]
    pub fn with_worktree(mut self, name: impl Into<String>) -> Self {
        self.worktree = Some(name.into());
        self
    }

    /// Sets the dispatch priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// The manager the rail reads worktrees through.
    #[must_use]
    pub fn manager(&self) -> &Arc<WorktreeManager> {
        &self.manager
    }

    /// The prompt language of the owning agent.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The explicit worktree name, when one was set.
    #[must_use]
    pub fn worktree_name(&self) -> Option<&str> {
        self.worktree.as_deref()
    }

    /// The active worktree name: the explicit name, else the recorded session's.
    fn resolve_name(&self) -> Option<String> {
        if let Some(name) = self.worktree.as_deref() {
            let name = name.trim();
            if !name.is_empty() {
                return Some(name.to_owned());
            }
        }
        self.sessions
            .session()
            .map(|session| session.worktree_name.trim().to_owned())
            .filter(|name| !name.is_empty())
    }

    /// Resolves the path and branch of the worktree `name`.
    ///
    /// A recorded session wins when it names the same worktree and carries a
    /// path; then the git listing (matched by the directory name, so a raw name
    /// matches its slug); finally the layout path, without a branch.
    async fn resolve_target(&self, name: &str) -> WorktreeTarget {
        if let Some(session) = self.sessions.session() {
            if session.worktree_name.trim() == name && !session.worktree_path.as_os_str().is_empty()
            {
                return WorktreeTarget {
                    path: session.worktree_path,
                    branch: session.worktree_branch,
                };
            }
        }
        match self.manager.list().await {
            Ok(infos) => {
                let slug = slugify(name);
                if let Some(info) = infos.into_iter().find(|info| {
                    let listed = info.name();
                    listed == name || listed == slug
                }) {
                    return WorktreeTarget {
                        path: info.path,
                        branch: info.branch,
                    };
                }
            }
            Err(error) => {
                tracing::debug!(
                    worktree = %name,
                    error = %error,
                    "worktree listing unavailable; falling back to the layout path"
                );
            }
        }
        match self.manager.target_path(name) {
            Ok(path) => WorktreeTarget { path, branch: None },
            Err(error) => {
                tracing::debug!(
                    worktree = %name,
                    error = %error,
                    "worktree name has no usable path; falling back to the layout directory"
                );
                WorktreeTarget {
                    path: self.manager.worktrees_dir().join(slugify(name)),
                    branch: None,
                }
            }
        }
    }

    /// Publishes the section (and the structured summary) for the active worktree.
    ///
    /// Never fails: every resolution, git or serialization error is logged at
    /// `debug` and the section is still published with whatever was resolved, or
    /// skipped entirely when there is no active worktree.
    async fn publish(&self, ctx: &mut HarnessRailContext) {
        let Some(name) = self.resolve_name() else {
            return;
        };
        let target = self.resolve_target(&name).await;
        let summary = match self.manager.change_summary(&name).await {
            Ok(summary) => Some(summary),
            Err(error) => {
                tracing::debug!(
                    worktree = %name,
                    language = %self.language,
                    error = %error,
                    "worktree change summary unavailable; publishing the section without it"
                );
                None
            }
        };
        let context = WorktreeContext {
            name: name.clone(),
            path: target.path,
            branch: target.branch,
            summary: summary.clone(),
        };
        let section = PromptSection::bilingual(
            WORKTREE_SECTION_NAME,
            context.render("cn"),
            context.render("en"),
            WORKTREE_SECTION_PRIORITY,
        )
        .with_category(PromptCategory::SystemPrompt);
        publish_prompt_section(ctx, &section);
        Self::publish_summary(ctx, summary);
    }

    /// Writes (or clears) the structured summary in the context `extra` map.
    fn publish_summary(ctx: &mut HarnessRailContext, summary: Option<WorktreeChangeSummary>) {
        let Some(summary) = summary else {
            ctx.extra_mut().remove(WORKTREE_SUMMARY_EXTRA_KEY);
            return;
        };
        match serde_json::to_value(&summary) {
            Ok(value) => {
                ctx.extra_mut()
                    .insert(WORKTREE_SUMMARY_EXTRA_KEY.to_owned(), value);
            }
            Err(error) => {
                tracing::debug!(
                    error = %error,
                    "worktree summary could not be serialized; dropping the extra entry"
                );
                ctx.extra_mut().remove(WORKTREE_SUMMARY_EXTRA_KEY);
            }
        }
    }
}

impl fmt::Debug for WorktreeRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorktreeRail")
            .field("repo_root", &self.manager.repo_root())
            .field("language", &self.language)
            .field("worktree", &self.worktree)
            .field("priority", &self.priority)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl HarnessRail for WorktreeRail {
    /// The rail name; it is also the name of the prompt section it publishes.
    fn name(&self) -> &'static str {
        WORKTREE_SECTION_NAME
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn before_invoke(
        &self,
        ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        self.publish(ctx).await;
        Ok(())
    }

    async fn before_model_call(
        &self,
        ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        self.publish(ctx).await;
        Ok(())
    }

    async fn after_invoke(
        &self,
        ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        self.publish(ctx).await;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn notice_keeps_the_python_text_and_mentions_both_paths() {
        let parent = Path::new("/repo");
        let worktree = Path::new("/repo/.worktrees/feature");
        let en = build_worktree_notice(parent, worktree, "en");
        assert_eq!(
            en,
            "You are operating in an isolated git worktree at /repo/.worktrees/feature. \
             The parent context lives in /repo — same repository, \
             same relative file structure, separate working copy.\n\n\
             Important:\n\
             - Paths from the parent context refer to /repo\n\
             - Translate them to your worktree root before use\n\
             - Re-read files before editing if the parent may have modified them\n\
             - Your changes stay in this worktree and will not affect the parent"
        );
        assert!(en.contains("/repo/.worktrees/feature"));
        assert!(en.contains("/repo —"));

        let cn = build_worktree_notice(parent, worktree, "cn");
        assert_ne!(cn, en);
        assert_eq!(cn, build_worktree_notice(parent, worktree, "zh"));
        assert!(cn.contains("/repo/.worktrees/feature"));
        assert!(cn.contains("/repo ——"));
    }

    #[test]
    fn context_renders_worktree_branch_and_summary() {
        let context = WorktreeContext {
            name: "feature".to_owned(),
            path: PathBuf::from("/repo/.worktrees/feature"),
            branch: Some("worktree-feature".to_owned()),
            summary: Some(WorktreeChangeSummary {
                changed_files: 1,
                files: vec!["src/lib.rs".to_owned()],
                commits: 2,
                unpushed: Some(true),
                diffstat: "src/lib.rs | 2 +-".to_owned(),
                truncated: false,
            }),
        };
        let en = context.render("en");
        assert!(en.contains("`feature`"));
        assert!(en.contains("/repo/.worktrees/feature"));
        assert!(en.contains("worktree-feature"));
        assert!(en.contains("### Changes"));
        assert!(en.contains("Changed files: 1"));

        let cn = context.render("cn");
        assert!(cn.contains("/repo/.worktrees/feature"));
        assert!(cn.contains("worktree-feature"));
        assert!(cn.contains("### 变更"));
        assert!(cn.contains("变更文件: 1"));
        assert_ne!(cn, en);
    }

    #[test]
    fn context_without_a_summary_says_it_is_unavailable() {
        let context = WorktreeContext {
            name: "feature".to_owned(),
            path: PathBuf::from("/repo/.worktrees/feature"),
            branch: None,
            summary: None,
        };
        assert!(context.render("en").contains("Change summary: unavailable"));
        assert!(context.render("en").contains("/repo/.worktrees/feature"));
        assert!(context.render("zh").contains("变更摘要: 不可用"));
    }

    #[test]
    fn rendering_is_bounded_and_marks_truncation() {
        let context = WorktreeContext {
            name: "feature".to_owned(),
            path: PathBuf::from("/repo/.worktrees/feature"),
            branch: Some("worktree-feature".to_owned()),
            summary: Some(WorktreeChangeSummary {
                changed_files: 200,
                files: (0..30)
                    .map(|index| format!("src/file-{index}.rs"))
                    .collect(),
                commits: 40,
                unpushed: Some(true),
                diffstat: "z".repeat(5_000),
                truncated: true,
            }),
        };
        for language in ["en", "cn"] {
            let rendered = context.render(language);
            assert!(rendered.chars().count() <= WORKTREE_SECTION_MAX_CHARS);
            assert!(rendered.ends_with(TRUNCATION_MARKER));
        }
    }

    #[test]
    fn short_renderings_are_not_marked() {
        let context = WorktreeContext {
            name: "f".to_owned(),
            path: PathBuf::from("/repo/.worktrees/f"),
            branch: None,
            summary: None,
        };
        assert!(!context.render("en").ends_with(TRUNCATION_MARKER));
    }
}
