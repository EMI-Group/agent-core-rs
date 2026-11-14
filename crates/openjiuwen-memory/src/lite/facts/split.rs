//! The `MEMORY.md` / `USER.md` rendered-index split.
//!
//! The `facts/` directory is the source of truth; the two top-level documents are
//! derived views over it. `user`-level facts (durable preferences and identity facts
//! about the person) belong to `USER.md`, and every other fact type (`feedback`,
//! `project`, `reference`) belongs to `MEMORY.md`.
//!
//! A rendered index uses the existing `crate::lite::markdown::MarkdownMemory`
//! document shape — schema front matter, a title, and one `## <Type>` section per fact
//! type with a `### <name>` entry per fact — so a rendered index is also readable by
//! the local store's parser and by hand. Rendering is byte-deterministic: the facts of
//! one level are sorted by fact type (user, feedback, project, reference) and then by
//! name ascending, which makes the output independent of storage order.
use crate::lite::fact::{CodingFact, CodingFactType};
use crate::lite::markdown::MarkdownMemory;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
/// Which of the two rendered indexes a fact belongs to.
///
/// [`IndexLevel::User`] holds the `USER.md` view of the `user` fact type;
/// [`IndexLevel::Memory`] holds the `MEMORY.md` view of every other type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexLevel {
    /// The project-level index, rendered into `MEMORY.md`.
    Memory,
    /// The user-level index, rendered into `USER.md`.
    User,
}
impl IndexLevel {
    /// Both levels, in file order (`MEMORY.md`, then `USER.md`).
    pub const ALL: [Self; 2] = [Self::Memory, Self::User];
    /// The file name this level renders into.
    #[must_use]
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Memory => super::MEMORY_FILE,
            Self::User => super::USER_FILE,
        }
    }
    /// The document title this level renders with (the text after `# `).
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Memory => "Memory",
            Self::User => "User",
        }
    }
    /// Whether a fact of `fact_type` is rendered into this level.
    #[must_use]
    pub fn contains(self, fact_type: CodingFactType) -> bool {
        match self {
            Self::User => fact_type == CodingFactType::User,
            Self::Memory => fact_type != CodingFactType::User,
        }
    }
}
/// The level a fact type renders into.
///
/// `user` facts go to [`IndexLevel::User`] (`USER.md`); every other fact type goes to
/// [`IndexLevel::Memory`] (`MEMORY.md`).
#[must_use]
pub fn level_of(fact_type: CodingFactType) -> IndexLevel {
    match fact_type {
        CodingFactType::User => IndexLevel::User,
        CodingFactType::Feedback | CodingFactType::Project | CodingFactType::Reference => {
            IndexLevel::Memory
        }
    }
}
/// Renders the `level` view of `facts` as a Markdown index document.
///
/// Facts of the other level are dropped, the rest are sorted by fact type (user,
/// feedback, project, reference) and then by name ascending, and each is written as a
/// `### <name>` entry with its description, tags, path, and timestamp. An empty slice
/// still renders a valid document (schema front matter plus the level title), and the
/// result always ends with exactly one newline.
#[must_use]
pub fn render_level(facts: &[CodingFact], level: IndexLevel) -> String {
    let mut selected: Vec<&CodingFact> = facts
        .iter()
        .filter(|fact| level.contains(fact.fact_type))
        .collect();
    selected.sort_by(|left, right| {
        type_rank(left.fact_type)
            .cmp(&type_rank(right.fact_type))
            .then_with(|| left.name.cmp(&right.name))
    });
    let mut document = MarkdownMemory::new().with_title(level.title());
    for fact in selected {
        let _replaced = document.upsert_fact(fact);
    }
    document.render()
}
/// Parses a rendered index document (or a hand-edited one) back into its facts.
///
/// The parse is tolerant in exactly the way [`MarkdownMemory::parse`] is: unknown
/// front-matter keys, unknown sections, extra prose, and freely edited entries are kept
/// or ignored rather than rejected. The returned facts carry the default scope, a
/// fact's `updated_at` is second-precision, and the entries come back in document order.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Validation`] only for an input that opens a
/// front-matter block and never closes it.
pub fn parse_level(text: &str) -> Result<Vec<CodingFact>> {
    Ok(MarkdownMemory::parse(text)?.facts())
}
/// The canonical sort rank of a fact type (its position in [`CodingFactType::ALL`]).
fn type_rank(fact_type: CodingFactType) -> usize {
    CodingFactType::ALL
        .iter()
        .position(|candidate| *candidate == fact_type)
        .unwrap_or(CodingFactType::ALL.len())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::MemoryScope;
    use chrono::DateTime;
    /// A fact with a second-precision timestamp (the precision a rendered index keeps).
    fn fact(name: &str, fact_type: CodingFactType, index: i64) -> CodingFact {
        let mut fact = CodingFact::new(&MemoryScope::default(), name, format!("body of {name}"))
            .with_type(fact_type)
            .with_description(format!("about {name}"));
        fact.updated_at = DateTime::from_timestamp(1_700_000_000 + index, 0).unwrap_or_default();
        fact
    }
    /// Facts sorted the way [`render_level`] emits them.
    fn sorted(mut facts: Vec<CodingFact>) -> Vec<CodingFact> {
        facts.sort_by(|left, right| {
            type_rank(left.fact_type)
                .cmp(&type_rank(right.fact_type))
                .then_with(|| left.name.cmp(&right.name))
        });
        facts
    }
    #[test]
    fn levels_map_fact_types_and_name_their_files() {
        assert_eq!(level_of(CodingFactType::User), IndexLevel::User);
        assert_eq!(level_of(CodingFactType::Feedback), IndexLevel::Memory);
        assert_eq!(level_of(CodingFactType::Project), IndexLevel::Memory);
        assert_eq!(level_of(CodingFactType::Reference), IndexLevel::Memory);
        assert_eq!(IndexLevel::Memory.file_name(), "MEMORY.md");
        assert_eq!(IndexLevel::User.file_name(), "USER.md");
        assert_eq!(IndexLevel::Memory.title(), "Memory");
        assert_eq!(IndexLevel::User.title(), "User");
        assert_eq!(IndexLevel::ALL, [IndexLevel::Memory, IndexLevel::User]);
    }
    #[test]
    fn every_fact_type_belongs_to_exactly_one_level() {
        for fact_type in CodingFactType::ALL {
            let levels: Vec<IndexLevel> = IndexLevel::ALL
                .into_iter()
                .filter(|level| level.contains(fact_type))
                .collect();
            assert_eq!(levels.len(), 1, "{fact_type:?} landed in {levels:?}");
        }
    }
    #[test]
    fn an_empty_level_still_renders_a_valid_document() {
        let memory = render_level(&[], IndexLevel::Memory);
        assert_eq!(
            memory,
            concat!(
                "---\n",
                "schema: openjiuwen-memory-lite\n",
                "version: 1\n",
                "---\n\n",
                "# Memory\n",
            )
        );
        let user = render_level(&[], IndexLevel::User);
        assert!(user.contains("# User\n"));
        assert_eq!(
            parse_level(&memory).expect("the empty document parses"),
            Vec::<CodingFact>::new()
        );
        assert_eq!(
            parse_level(&user).expect("the empty document parses"),
            Vec::<CodingFact>::new()
        );
    }
    #[test]
    fn rendering_splits_the_levels_and_sorts_by_type_then_name() {
        let facts = vec![
            fact("zeta", CodingFactType::Reference, 5),
            fact("beta", CodingFactType::Project, 4),
            fact("alpha", CodingFactType::User, 3),
            fact("delta", CodingFactType::Feedback, 2),
            fact("gamma", CodingFactType::Project, 1),
        ];
        let memory = render_level(&facts, IndexLevel::Memory);
        assert!(
            !memory.contains("### alpha"),
            "user facts stay out of MEMORY.md"
        );
        let order: Vec<&str> = memory
            .lines()
            .filter_map(|line| line.strip_prefix("### "))
            .collect();
        assert_eq!(order, vec!["delta", "beta", "gamma", "zeta"]);
        let user = render_level(&facts, IndexLevel::User);
        let order: Vec<&str> = user
            .lines()
            .filter_map(|line| line.strip_prefix("### "))
            .collect();
        assert_eq!(order, vec!["alpha"]);
    }
    #[test]
    fn the_round_trip_preserves_every_field_of_the_level() {
        let mut projected = fact("build", CodingFactType::Project, 9)
            .with_tags(vec!["build".to_owned(), "ci".to_owned()])
            .with_path("docs/BUILD.md");
        projected.content = "# Build\n\ncargo build\ncargo test".to_owned();
        let mut user = fact("theme", CodingFactType::User, 8);
        user.description = "prefers dark mode".to_owned();
        let facts = vec![projected, user];
        for level in IndexLevel::ALL {
            let expected: Vec<CodingFact> = sorted(
                facts
                    .iter()
                    .filter(|fact| level.contains(fact.fact_type))
                    .cloned()
                    .collect(),
            );
            let rendered = render_level(&facts, level);
            let parsed = parse_level(&rendered).expect("the rendered index parses");
            assert_eq!(
                parsed,
                expected,
                "round trip failed for {}",
                level.file_name()
            );
            assert_eq!(render_level(&parsed, level), rendered);
        }
    }
    #[test]
    fn rendering_is_independent_of_input_order() {
        let facts = vec![
            fact("b", CodingFactType::Project, 1),
            fact("a", CodingFactType::Project, 2),
            fact("c", CodingFactType::Feedback, 3),
        ];
        let mut reversed = facts.clone();
        reversed.reverse();
        assert_eq!(
            render_level(&facts, IndexLevel::Memory),
            render_level(&reversed, IndexLevel::Memory)
        );
    }
    #[test]
    fn parsing_tolerates_hand_edits_prose_and_unknown_keys() {
        let text = concat!(
            "---\n",
            "schema: openjiuwen-memory-lite\n",
            "version: 1\n",
            "owner: alice\n",
            "---\n\n",
            "# Memory\n\n",
            "some loose prose\n\n",
            "## Project\n\n",
            "### build\n\n",
            "- description: How to build\n",
            "- tags: build, ci\n",
            "- updated_at: 2023-11-14T22:13:20Z\n\n",
            "cargo build\n\n",
            "## Unknown Section\n\n",
            "whatever text\n",
        );
        let facts = parse_level(text).expect("a hand-edited index parses");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "build");
        assert_eq!(facts[0].description, "How to build");
        assert_eq!(facts[0].tags, vec!["build".to_owned(), "ci".to_owned()]);
        assert_eq!(facts[0].content, "cargo build");
    }
    #[test]
    fn an_unterminated_front_matter_is_a_validation_error() {
        let error = parse_level("---\nname: build\n").expect_err("an unclosed block is invalid");
        assert!(error.to_string().contains("never closed"), "got {error}");
    }
    #[test]
    fn a_fence_line_without_a_block_is_tolerated_as_text() {
        let facts = parse_level("---\n# Memory\n").expect("plain markdown does not fail");
        assert_eq!(facts, Vec::<CodingFact>::new());
    }
}
