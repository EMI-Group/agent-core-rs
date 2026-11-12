//! Skill discovery, library state, mounting, and `SKILL.md` validation.
//!
//! Ports `openjiuwen/harness/skills/{library_state.py,mount.py}` and the
//! discovery/loading half of `rails/skills/skill_use_rail.py`. A skill is a directory
//! holding a `SKILL.md` with YAML front matter; [`SkillLibrary`] discovers them under its
//! roots, filters the ones disabled in `skills_state.json`, and applies a selection
//! budget. Workspace paths arrive as [`PathBuf`]; the caller decides which roots are
//! visible to the agent.

mod discovery;
mod library_state;
mod mount;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use library_state::{
    collect_disabled_skills, parse_disabled_skills, read_disabled_skills, save_disabled_skills,
    SKILLS_STATE_FILENAME,
};
pub use mount::{
    parse_skill_front_matter, split_skill_front_matter, SkillMetadata, SkillResourceValidator,
    SkillRootMount, FRONT_MATTER_PROBE_LINES, SKILL_FILENAME,
};

pub(crate) use discovery::discover_skills;

use mount::normalize_path;

/// One discovered skill: a directory holding a `SKILL.md`.
///
/// `name` is the bare directory name, never a front-matter `name:` field, and the first
/// skill with a given name wins across the library roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Skill {
    /// Bare directory name of the skill.
    pub name: String,
    /// Description from the front matter, or a generated fallback.
    pub description: String,
    /// Absolute directory holding the skill's `SKILL.md`.
    pub directory: PathBuf,
}

impl Skill {
    /// Creates a skill from its name, description and directory.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        directory: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            directory: directory.into(),
        }
    }

    /// Returns the path of the skill's `SKILL.md` file.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.directory.join(SKILL_FILENAME)
    }
}

/// Limits applied when a library selects the skills shown to the model.
///
/// Both limits are optional and independent; [`SkillBudget::unlimited`] disables both.
/// `max_total_chars` counts `description.chars().count()` of whole skills, so a selection
/// is never cut mid-description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct SkillBudget {
    /// Maximum number of skills kept, applied before the character budget.
    pub max_skills: Option<usize>,
    /// Soft cap on the summed description length of the kept skills.
    pub max_total_chars: Option<usize>,
}

impl SkillBudget {
    /// Creates a budget with no limits.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_skills: None,
            max_total_chars: None,
        }
    }

    /// Returns `true` when neither limit is set.
    #[must_use]
    pub const fn is_unlimited(&self) -> bool {
        self.max_skills.is_none() && self.max_total_chars.is_none()
    }

    /// Returns this budget with a maximum skill count.
    #[must_use]
    pub const fn with_max_skills(self, max_skills: usize) -> Self {
        Self {
            max_skills: Some(max_skills),
            max_total_chars: self.max_total_chars,
        }
    }

    /// Returns this budget with a maximum summed description length.
    #[must_use]
    pub const fn with_max_total_chars(self, max_total_chars: usize) -> Self {
        Self {
            max_skills: self.max_skills,
            max_total_chars: Some(max_total_chars),
        }
    }
}

/// A skill library: discovery roots plus the library-wide disabled names.
///
/// The fields are public so callers can build a library with a struct literal. There is
/// no discovery cache: [`discover`](Self::discover) is explicit and
/// [`select`](Self::select) is a pure filter over already-discovered skills, so callers
/// control when the library is refreshed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillLibrary {
    /// Skill roots, in discovery order.
    pub roots: Vec<PathBuf>,
    /// Names switched off in the library state file or by the caller.
    pub disabled: BTreeSet<String>,
}

impl SkillLibrary {
    /// Creates an empty library with no roots and no disabled skills.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a library from `roots`, appending each one that is not already present.
    #[must_use]
    pub fn with_roots<R: Into<PathBuf>, I: IntoIterator<Item = R>>(roots: I) -> Self {
        let mut library = Self::new();
        for root in roots {
            library.add_root(root);
        }
        library
    }

    /// Appends `root` when it is not present and returns `true` only when it was added.
    ///
    /// Paths are compared after absolute/lexical normalization, so `./a` and `a` are the
    /// same root.
    pub fn add_root(&mut self, root: impl Into<PathBuf>) -> bool {
        SkillRootMount::mount(&mut self.roots, root).owned()
    }

    /// Removes `root` when present and returns whether one was removed.
    ///
    /// The comparison is normalized like [`add_root`](Self::add_root).
    pub fn remove_root(&mut self, root: &Path) -> bool {
        let target = normalize_path(root.to_path_buf());
        let Some(index) = self
            .roots
            .iter()
            .position(|candidate| normalize_path(candidate.clone()) == target)
        else {
            return false;
        };
        self.roots.remove(index);
        true
    }

    /// Returns `true` when `name` is disabled for this library.
    #[must_use]
    pub fn is_disabled(&self, name: &str) -> bool {
        self.disabled.contains(name)
    }

    /// Enables or disables the skill named `name`.
    pub fn set_disabled(&mut self, name: impl Into<String>, disabled: bool) {
        let name = name.into();
        if disabled {
            self.disabled.insert(name);
        } else {
            self.disabled.remove(&name);
        }
    }

    /// Returns the disabled names, in sorted order.
    #[must_use]
    pub fn disabled_names(&self) -> &BTreeSet<String> {
        &self.disabled
    }

    /// Discovers the skills under every root without language-aware descriptions.
    ///
    /// Disabled skills are not filtered here; use [`select`](Self::select) for that.
    ///
    /// # Errors
    ///
    /// Reserved for API consistency: discovery logs and skips IO problems instead of
    /// failing, so this currently never returns `Err`.
    pub async fn discover(&self) -> Result<Vec<Skill>> {
        self.discover_with_language(None).await
    }

    /// Discovers the skills under every root, preferring descriptions in `language`.
    ///
    /// `language` is trimmed and lowercased before it is used as the
    /// `description_<language>` front-matter key. Disabled skills are not filtered here.
    ///
    /// # Errors
    ///
    /// Reserved for API consistency: discovery logs and skips IO problems instead of
    /// failing, so this currently never returns `Err`.
    pub async fn discover_with_language(&self, language: Option<&str>) -> Result<Vec<Skill>> {
        discover_skills(&self.roots, language).await
    }

    /// Applies the library's disabled set and `budget` to already-discovered skills.
    ///
    /// Disabled names are dropped first, preserving the input order. `max_skills` then
    /// truncates to the first `n` skills. `max_total_chars` drops whole skills from the
    /// end of the selection while the summed description length exceeds the cap, but
    /// always keeps at least one skill; a lone survivor over the cap is kept with a
    /// warning. An unlimited budget returns the enabled skills unchanged.
    #[must_use]
    pub fn select(&self, skills: &[Skill], budget: SkillBudget) -> Vec<Skill> {
        let mut selected: Vec<Skill> = skills
            .iter()
            .filter(|skill| !self.disabled.contains(&skill.name))
            .cloned()
            .collect();

        if let Some(max_skills) = budget.max_skills {
            if selected.len() > max_skills {
                let dropped: Vec<String> = selected[max_skills..]
                    .iter()
                    .map(|skill| skill.name.clone())
                    .collect();
                tracing::debug!(
                    ?dropped,
                    max_skills,
                    "skill budget dropped skills beyond the maximum count"
                );
                selected.truncate(max_skills);
            }
        }

        let Some(max_total_chars) = budget.max_total_chars else {
            return selected;
        };
        let mut total: usize = selected
            .iter()
            .map(|skill| skill.description.chars().count())
            .sum();
        while selected.len() > 1 && total > max_total_chars {
            let Some(dropped) = selected.pop() else {
                break;
            };
            total -= dropped.description.chars().count();
        }
        if total > max_total_chars {
            if let Some(survivor) = selected.first() {
                tracing::warn!(
                    skill = %survivor.name,
                    total,
                    max_total_chars,
                    "a single skill exceeds the description character budget; keeping it"
                );
            }
        }
        selected
    }

    /// Merges one root's library state file into the disabled set, failing open.
    pub async fn load_state(&mut self, root: impl AsRef<Path>) {
        self.disabled
            .extend(read_disabled_skills(root.as_ref()).await);
    }

    /// Merges every root's library state file into the disabled set, failing open.
    pub async fn load_all_state(&mut self) {
        self.disabled
            .extend(collect_disabled_skills(&self.roots).await);
    }

    /// Writes the disabled set to `root`'s library state file.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::sys_op`] when the root cannot be
    /// created or the state file cannot be written.
    pub async fn save_state(&self, root: impl AsRef<Path>) -> Result<()> {
        save_disabled_skills(root.as_ref(), &self.disabled).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str) -> Skill {
        Skill::new(name, description, Path::new("/skills").join(name))
    }

    fn names(skills: &[Skill]) -> Vec<&str> {
        skills.iter().map(|skill| skill.name.as_str()).collect()
    }

    #[test]
    fn new_builds_a_skill_pointing_at_its_skill_file() {
        let skill = Skill::new("alpha", "A skill", "/skills/alpha");
        assert_eq!(skill.name, "alpha");
        assert_eq!(skill.description, "A skill");
        assert_eq!(
            skill.path(),
            Path::new("/skills/alpha").join(SKILL_FILENAME)
        );
        assert!(skill.path().ends_with(SKILL_FILENAME));
    }

    #[test]
    fn unlimited_budget_reports_itself_and_keeps_enabled_skills() {
        let library = SkillLibrary::new();
        let skills = [skill("alpha", "one"), skill("beta", "two")];
        let budget = SkillBudget::unlimited();
        assert!(budget.is_unlimited());
        assert_eq!(library.select(&skills, budget), skills);
    }

    #[test]
    fn select_drops_disabled_names_and_preserves_order() {
        let mut library = SkillLibrary::new();
        library.set_disabled("beta", true);
        let skills = [
            skill("alpha", "one"),
            skill("beta", "two"),
            skill("gamma", "three"),
        ];
        assert_eq!(
            names(&library.select(&skills, SkillBudget::unlimited())),
            ["alpha", "gamma"]
        );
        assert!(library.is_disabled("beta"));
        library.set_disabled("beta", false);
        assert!(!library.is_disabled("beta"));
        assert_eq!(library.select(&skills, SkillBudget::unlimited()).len(), 3);
    }

    #[test]
    fn max_skills_truncates_and_zero_selects_nothing() {
        let library = SkillLibrary::new();
        let skills = [skill("a", "1"), skill("b", "2"), skill("c", "3")];
        let budget = SkillBudget::default().with_max_skills(2);
        assert_eq!(names(&library.select(&skills, budget)), ["a", "b"]);
        let budget = SkillBudget::default().with_max_skills(0);
        assert_eq!(library.select(&skills, budget), Vec::<Skill>::new());
    }

    #[test]
    fn max_total_chars_drops_lowest_ranked_whole_skills() {
        let library = SkillLibrary::new();
        let skills = [skill("a", "aaaa"), skill("b", "bbbb"), skill("c", "cc")];
        let budget = SkillBudget::default().with_max_total_chars(6);
        let selected = library.select(&skills, budget);
        assert_eq!(names(&selected), ["a"]);
    }

    #[test]
    fn max_total_chars_keeps_at_least_one_skill() {
        let library = SkillLibrary::new();
        let skills = [skill("a", "aaaaaaaaaa"), skill("b", "b")];
        let budget = SkillBudget::default().with_max_total_chars(3);
        let selected = library.select(&skills, budget);
        assert_eq!(names(&selected), ["a"]);
    }

    #[test]
    fn add_root_dedupes_and_remove_root_removes() {
        let mut library = SkillLibrary::new();
        assert!(library.add_root("one"));
        assert!(!library.add_root("./one"));
        assert!(library.add_root("two"));
        assert_eq!(library.roots.len(), 2);
        assert!(library.remove_root(Path::new("./one")));
        assert!(!library.remove_root(Path::new("one")));
        assert_eq!(library.roots.len(), 1);
        assert!(library.roots[0].ends_with("two"));
    }

    #[test]
    fn with_roots_appends_every_distinct_root() {
        let library = SkillLibrary::with_roots(["a", "b", "./a"]);
        assert_eq!(library.roots.len(), 2);
    }

    #[tokio::test]
    async fn load_state_merges_a_roots_disabled_names() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::write(
            root.join(SKILLS_STATE_FILENAME),
            r#"{"skill_configs": {"gamma": {"enabled": false}}}"#,
        )
        .unwrap();

        let mut library = SkillLibrary::new();
        library.load_state(root).await;
        assert_eq!(
            library.disabled_names().iter().cloned().collect::<Vec<_>>(),
            vec!["gamma"]
        );
    }

    #[tokio::test]
    async fn save_state_round_trips_through_collect() {
        let temp = tempfile::tempdir().unwrap();
        let mut library = SkillLibrary::new();
        library.set_disabled("alpha", true);
        library.set_disabled("gamma", true);
        library.set_disabled("beta", false);

        library.save_state(temp.path()).await.unwrap();

        let collected = collect_disabled_skills(&[temp.path().to_path_buf()]).await;
        assert_eq!(collected, library.disabled);
        let text = std::fs::read_to_string(temp.path().join(SKILLS_STATE_FILENAME)).unwrap();
        assert!(text.contains("\"alpha\""));
        assert!(!text.contains("\"beta\""));
    }
}
