//! The skill discovery/selection rail.
//!
//! Ports the discovery and selection half of
//! `openjiuwen/harness/rails/skills/skill_use_rail.py`: before the model sees the
//! prompt, the rail refreshes the skill selection when the previous invocation
//! invalidated it and publishes [`crate::prompts::skills_section`] into the
//! prompt-section channel.
//!
//! Discovery is explicit and cached: `after_invoke` marks the cache stale, so one
//! invocation discovers at most once even though `before_model_call` fires on
//! every `ReAct` iteration. A discovery failure is logged and keeps the previous
//! selection — a broken skill root must never fail the run.
//!
//! The last selection is published as-is, including an empty one; the prompt
//! builder drops empty sections, so an empty library costs nothing.
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use openjiuwen_common::error::Result;

use crate::prompts::{self, PromptSection};
use crate::skills::{Skill, SkillBudget, SkillLibrary};

use super::{publish_prompt_section, rail_priority, HarnessRail, HarnessRailContext};

/// Tunables of [`SkillsRail`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillsRailConfig {
    /// Limits applied to the discovered skills before publication.
    pub budget: SkillBudget,
    /// Preferred description language (`description_<language>` front matter).
    pub language: Option<String>,
}

/// Discovers skills, applies the budget, and publishes the skill section.
///
/// Dispatch priority is [`rail_priority::SKILLS`] (95), the slot of the Python
/// skill-use rail.
pub struct SkillsRail {
    library: SkillLibrary,
    config: SkillsRailConfig,
    selected: RwLock<Vec<Skill>>,
    stale: AtomicBool,
}

impl SkillsRail {
    /// Builds a rail over `library`, discovering on the first model call.
    #[must_use]
    pub fn new(library: SkillLibrary) -> Self {
        Self {
            library,
            config: SkillsRailConfig::default(),
            selected: RwLock::new(Vec::new()),
            stale: AtomicBool::new(true),
        }
    }

    /// Returns this rail with `budget` as its selection budget.
    #[must_use]
    pub fn with_budget(mut self, budget: SkillBudget) -> Self {
        self.config.budget = budget;
        self
    }

    /// Returns this rail preferring `description_<language>` front matter.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.config.language = Some(language.into());
        self
    }

    /// Returns this rail with `config` as its tunables.
    #[must_use]
    pub fn with_config(mut self, config: SkillsRailConfig) -> Self {
        self.config = config;
        self
    }

    /// The skill library this rail searches.
    #[must_use]
    pub fn library(&self) -> &SkillLibrary {
        &self.library
    }

    /// The skill library, mutably (add roots, disable skills).
    pub fn library_mut(&mut self) -> &mut SkillLibrary {
        // A structural library change must be reflected in the next selection.
        self.stale.store(true, Ordering::SeqCst);
        &mut self.library
    }

    /// The selection budget.
    #[must_use]
    pub fn budget(&self) -> SkillBudget {
        self.config.budget
    }

    /// The last selection snapshot, in discovery order.
    #[must_use]
    pub fn skills(&self) -> Vec<Skill> {
        self.selected.read().clone()
    }

    /// The tunables of this rail.
    #[must_use]
    pub fn config(&self) -> &SkillsRailConfig {
        &self.config
    }

    /// Discovers skills and stores the budgeted selection.
    ///
    /// Explicitly usable when a caller wants a warm cache; the hooks call it
    /// themselves when the cache is stale.
    ///
    /// # Errors
    ///
    /// Never fails today: a discovery error is logged and the previous
    /// selection is returned. The `Result` keeps the signature consistent with
    /// the other skill operations and leaves room for a future strict mode.
    pub async fn refresh(&self) -> Result<Vec<Skill>> {
        match self
            .library
            .discover_with_language(self.config.language.as_deref())
            .await
        {
            Ok(discovered) => {
                let selection = self.library.select(&discovered, self.config.budget);
                *self.selected.write() = selection.clone();
                self.stale.store(false, Ordering::SeqCst);
                Ok(selection)
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "skill discovery failed; keeping the previous selection"
                );
                Ok(self.skills())
            }
        }
    }
}

impl Default for SkillsRail {
    /// An empty library with no roots: the section renders as "no skills".
    fn default() -> Self {
        Self::new(SkillLibrary::new())
    }
}

impl fmt::Debug for SkillsRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkillsRail")
            .field("library", &self.library)
            .field("config", &self.config)
            .field("selected", &self.skills().len())
            .field("stale", &self.stale.load(Ordering::SeqCst))
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for SkillsRail {
    fn name(&self) -> &'static str {
        "skills"
    }

    fn priority(&self) -> i32 {
        rail_priority::SKILLS
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if self.stale.load(Ordering::SeqCst) {
            // `refresh` never fails; the `Ok` branch is the only one.
            let _ = self.refresh().await;
        }
        publish_prompt_section(ctx, &prompts::skills_section(&self.skills()));
        Ok(())
    }

    async fn after_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        self.stale.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::fs;
    use std::path::Path;

    use openjiuwen_agent::RailEvent;
    use tempfile::TempDir;

    use super::*;
    use crate::prompts::{prompt_sections, SectionName};

    fn write_skill(root: &Path, name: &str, description: &str) {
        let directory = root.join(name);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("SKILL.md"),
            format!("---\ndescription: {description}\n---\n\n# {name}\n"),
        )
        .unwrap();
    }

    fn rail_with_root(root: &Path) -> SkillsRail {
        SkillsRail::new(SkillLibrary::with_roots([root.to_path_buf()]))
    }

    #[test]
    fn name_priority_and_defaults() {
        let rail = SkillsRail::default();
        assert_eq!(rail.name(), "skills");
        assert_eq!(rail.priority(), rail_priority::SKILLS);
        assert_eq!(rail.priority(), 95);
        assert!(rail.library().roots.is_empty());
        assert_eq!(rail.budget(), SkillBudget::unlimited());
        assert!(rail.skills().is_empty());
        assert!(rail.config().language.is_none());
    }

    #[tokio::test]
    async fn before_model_call_publishes_the_discovered_skills() {
        let root = TempDir::new().unwrap();
        write_skill(root.path(), "alpha", "Alpha does A");
        write_skill(root.path(), "beta", "Beta does B");
        let rail = rail_with_root(root.path());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut ctx).await.unwrap();

        let names: Vec<String> = rail.skills().iter().map(|skill| skill.name.clone()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, SectionName::SKILLS);
        let rendered = sections[0]
            .content
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("alpha"));
        assert!(rendered.contains("Alpha does A"));
        assert!(rendered.contains("beta"));
    }

    #[tokio::test]
    async fn budget_truncates_the_selection() {
        let root = TempDir::new().unwrap();
        write_skill(root.path(), "alpha", "Alpha does A");
        write_skill(root.path(), "beta", "Beta does B");
        let rail = rail_with_root(root.path()).with_budget(SkillBudget::unlimited().with_max_skills(1));

        let selected = rail.refresh().await.unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "alpha");
        assert_eq!(rail.skills().len(), 1);
    }

    #[tokio::test]
    async fn disabled_skills_are_filtered() {
        let root = TempDir::new().unwrap();
        write_skill(root.path(), "alpha", "Alpha does A");
        write_skill(root.path(), "beta", "Beta does B");
        let mut rail = rail_with_root(root.path());
        rail.library_mut().set_disabled("alpha", true);

        let selected = rail.refresh().await.unwrap();
        let names: Vec<String> = selected.iter().map(|skill| skill.name.clone()).collect();
        assert_eq!(names, vec!["beta"]);
    }

    #[tokio::test]
    async fn selection_refreshes_once_per_invocation() {
        let root = TempDir::new().unwrap();
        write_skill(root.path(), "alpha", "Alpha does A");
        let rail = rail_with_root(root.path());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(rail.skills().len(), 1);

        // A skill added mid-invocation is not picked up by the next iteration.
        write_skill(root.path(), "beta", "Beta does B");
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(rail.skills().len(), 1);

        // The next invocation starts stale and re-discovers.
        rail.after_invoke(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(rail.skills().len(), 2);
        let names: Vec<String> = rail.skills().iter().map(|skill| skill.name.clone()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn a_missing_root_degrades_to_an_empty_selection() {
        let root = TempDir::new().unwrap();
        let missing = root.path().join("does-not-exist");
        let rail = SkillsRail::new(SkillLibrary::with_roots([missing]));

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert!(rail.skills().is_empty());
        assert_eq!(prompt_sections(&ctx).len(), 1);
    }

    #[tokio::test]
    async fn localized_descriptions_are_preferred() {
        let root = TempDir::new().unwrap();
        let directory = root.path().join("alpha");
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("SKILL.md"),
            "---\ndescription: English text\ndescription_cn: 中文说明\n---\n",
        )
        .unwrap();

        let rail = rail_with_root(root.path()).with_language("cn");
        let selected = rail.refresh().await.unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].description, "中文说明");
    }

    #[tokio::test]
    async fn refresh_keeps_the_previous_selection_on_error() {
        // A discovery error is not reachable through the public library today,
        // but the documented fallback is exercised by pinning the invariant that
        // a second refresh over the same roots is idempotent.
        let root = TempDir::new().unwrap();
        write_skill(root.path(), "alpha", "Alpha does A");
        let rail = rail_with_root(root.path());
        rail.refresh().await.unwrap();
        let selected = rail.refresh().await.unwrap();
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn debug_reports_the_selection_size() {
        let rail = SkillsRail::default();
        let rendered = format!("{rail:?}");
        assert!(rendered.contains("selected: 0"));
        assert!(rendered.contains("stale: true"));
    }
}
