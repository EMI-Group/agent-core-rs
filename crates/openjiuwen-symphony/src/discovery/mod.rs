//! Skill-folder discovery: `SKILL.md` scanning and lexical skill search.
//!
//! Ports the offline half of Python `openjiuwen.symphony.discovery` (v0.1.20) plus the
//! front-matter scanner from `openjiuwen.symphony.shared.fingerprint`:
//!
//! - [`frontmatter`](crate::discovery::frontmatter) — split a markdown document into
//!   YAML front matter and body, reporting okay/missing/unterminated fences and never
//!   panicking on bad YAML;
//! - [`scanner`](crate::discovery::scanner) — walk a skill root (immediate children by
//!   default, bounded recursion on request) and derive one
//!   [`CapabilityDescriptor`](crate::model::CapabilityDescriptor) per `SKILL.md`,
//!   collecting [`SkillDiagnostic`](crate::discovery::SkillDiagnostic)s instead of
//!   failing the whole scan;
//! - [`models`](crate::discovery::models) — the host-visible
//!   [`SkillRecord`](crate::discovery::SkillRecord) and the fingerprinted
//!   [`SkillInventory`](crate::discovery::SkillInventory);
//! - [`catalog`](crate::discovery::catalog) — [`SkillFs`](crate::discovery::SkillFs),
//!   the installed-skill catalog with BM25 search and excerpted
//!   [`SkillHit`](crate::discovery::SkillHit)s;
//! - [`settings`](crate::discovery::settings) —
//!   [`DiscoverySettings`](crate::discovery::DiscoverySettings), the result/output/token
//!   budget for one discovery round.
//!
//! Everything is offline and deterministic: directory entries are sorted, skills are
//! ordered by capability id, search ties break on worker id, and no clock or hash-map
//! iteration order leaks into an output.
//!
//! ```
//! # use openjiuwen_symphony::discovery::{SkillFolderScanner, SkillScanConfig};
//! # async fn example() -> openjiuwen_symphony::Result<()> {
//! let scanner = SkillFolderScanner::with_config(
//!     "skills",
//!     SkillScanConfig::recursive(Some(3)).with_source("bundled"),
//! );
//! let result = scanner.scan_result().await?;
//! # let _ = result;
//! # Ok(())
//! # }
//! ```

pub mod catalog;
pub mod frontmatter;
pub mod models;
pub mod scanner;
pub mod settings;

pub use catalog::{SelectionCard, SkillFs, SkillHit, EXCERPT_LEAD, EXCERPT_WIDTH};
pub use frontmatter::{
    boolean_value, io_values, normalize_capability_type, parse_front_matter, tag_values,
    FrontMatterOutcome, FrontMatterStatus, ALTERNATE_DELIMITER, DELIMITER,
};
pub use models::{
    inventory_fingerprint, sort_records, SkillInventory, SkillRecord, DEFAULT_SOURCE,
};
pub use scanner::{
    DiagnosticSeverity, ScannedSkill, SkillDiagnostic, SkillDiscovery, SkillFolderScanner,
    SkillScanConfig, SkillScanResult, DEFAULT_MAX_DEPTH, MAX_DESCRIPTION_CHARS,
    RECURSIVE_SKILL_FILE, SKILL_CAPABILITY_TYPE, SKILL_FILE_NAMES,
};
pub use settings::{DiscoverySettings, DEFAULT_MAX_OUTPUT_CHARS, DEFAULT_MAX_RESULTS};

pub use scanner::{
    DIAGNOSTIC_DUPLICATE_SKILL, DIAGNOSTIC_INVALID_SKILL, DIAGNOSTIC_MALFORMED_FRONT_MATTER,
    DIAGNOSTIC_MISSING_FRONT_MATTER, DIAGNOSTIC_SYMLINK_SKIPPED, DIAGNOSTIC_UNREADABLE_DIRECTORY,
    DIAGNOSTIC_UNREADABLE_FILE, DIAGNOSTIC_UNTERMINATED_FRONT_MATTER,
};
pub use settings::{DEFAULT_CANDIDATE_BUDGET_RATIO, DEFAULT_CONTEXT_WINDOW_TOKENS};
