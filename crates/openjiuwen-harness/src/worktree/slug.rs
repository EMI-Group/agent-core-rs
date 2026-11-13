//! Worktree slug validation, normalisation, and layout derivation.
//!
//! Port of `openjiuwen/harness/tools/worktree/slug.py`: a model- or caller-
//! supplied name becomes a safe slug (`[A-Za-z0-9._-]` only), and the slug
//! becomes both the directory under [`WORKTREES_DIR`] and the git branch name
//! `worktree-<slug>`.
//!
//! Two entry points, on purpose:
//!
//! - [`validate_raw_name`] is the security gate. It rejects path traversal,
//!   absolute paths and NUL bytes *before* any normalisation, so a hostile name
//!   can never be silently rewritten into something acceptable.
//! - [`slugify`] is the normaliser used for display, directory naming and
//!   branch naming. It maps every unsafe character to `-`, collapses runs, trims
//!   separators and caps the length, so the derived path always stays below the
//!   repository's `.worktrees/` directory.
use std::path::Path;

use openjiuwen_common::error::{Error, Result};

/// Longest accepted slug, matching the Python `MAX_SLUG_LENGTH`.
pub const MAX_SLUG_LENGTH: usize = 64;

/// Slug used when the supplied name normalises to nothing.
pub const DEFAULT_SLUG: &str = "worktree";

/// Parent directory of every worktree, inside the repository root.
///
/// The prompt guidance in [`crate::tools::descriptions`] tells the model to
/// create worktrees under `.worktrees/<name>`, so the helper uses the same
/// layout.
pub const WORKTREES_DIR: &str = ".worktrees";

/// Prefix of the derived branch name (`worktree-<slug>`).
pub const BRANCH_PREFIX: &str = "worktree-";

/// Rejects a raw, untrusted worktree name.
///
/// The check runs before [`slugify`], so a name such as `../escape`, `/etc` or
/// `a/../../b` is refused instead of being normalised into a harmless-looking
/// slug.
///
/// # Errors
///
/// Returns [`Error::security`] for an absolute path, a `..`/`.` segment, a NUL
/// byte, or a `~` prefix; [`Error::validation`] for an empty name.
pub fn validate_raw_name(raw: &str) -> Result<()> {
    if raw.trim().is_empty() {
        return Err(Error::validation("worktree name is required"));
    }
    if raw.contains('\0') {
        return Err(Error::security(format!(
            "invalid worktree name `{raw}`: name contains a NUL byte"
        )));
    }
    if Path::new(raw).is_absolute()
        || raw.starts_with('/')
        || raw.starts_with('\\')
        || raw.starts_with('~')
        || has_windows_drive_prefix(raw)
    {
        return Err(Error::security(format!(
            "invalid worktree name `{raw}`: the name must be relative and stay inside the repository"
        )));
    }
    for segment in raw.split(['/', '\\']) {
        if segment == "." || segment == ".." {
            return Err(Error::security(format!(
                "invalid worktree name `{raw}`: `.` and `..` path segments are not allowed"
            )));
        }
    }
    Ok(())
}

/// True for a `C:` / `C:\` style drive prefix.
fn has_windows_drive_prefix(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Normalises `raw` into a safe slug.
///
/// Unsafe characters become `-`, runs of `-` collapse, leading/trailing `-` and
/// `.` are trimmed, and the result is capped at [`MAX_SLUG_LENGTH`]. An empty
/// result becomes [`DEFAULT_SLUG`].
///
/// ```
/// use openjiuwen_harness::worktree::slugify;
/// assert_eq!(slugify("Feature Login!"), "Feature-Login");
/// assert_eq!(slugify("../../etc"), "etc");
/// assert_eq!(slugify(""), "worktree");
/// ```
#[must_use]
pub fn slugify(raw: &str) -> String {
    let mut slug = String::with_capacity(raw.len().min(MAX_SLUG_LENGTH));
    let mut previous_was_separator = false;
    for character in raw.chars() {
        let accepted = character.is_ascii_alphanumeric() || character == '_' || character == '.';
        if accepted {
            if character == '.' && previous_was_separator {
                continue;
            }
            slug.push(character);
            previous_was_separator = false;
        } else if !previous_was_separator {
            slug.push('-');
            previous_was_separator = true;
        }
        if slug.chars().count() >= MAX_SLUG_LENGTH {
            break;
        }
    }
    let trimmed = slug.trim_matches(|character: char| character == '-' || character == '.');
    if trimmed.is_empty() {
        return DEFAULT_SLUG.to_owned();
    }
    trimmed.chars().take(MAX_SLUG_LENGTH).collect()
}

/// Returns `slug`, or `slug-2`, `slug-3`, … for the first value not in `taken`.
///
/// The suffix is applied before the length cap, so the result always stays a
/// valid slug. `taken` is compared case-sensitively, like a filesystem on Linux.
///
/// ```
/// use openjiuwen_harness::worktree::unique_slug;
/// let taken = ["feature".to_owned(), "feature-2".to_owned()];
/// assert_eq!(unique_slug("feature", &taken), "feature-3");
/// ```
#[must_use]
pub fn unique_slug(slug: &str, taken: &[String]) -> String {
    if !taken.iter().any(|candidate| candidate == slug) {
        return slug.to_owned();
    }
    for suffix in 2..=u32::MAX {
        let tail = format!("-{suffix}");
        let keep = MAX_SLUG_LENGTH.saturating_sub(tail.len());
        let base: String = slug.chars().take(keep).collect();
        let candidate = format!("{base}{tail}");
        if !taken.iter().any(|existing| existing == &candidate) {
            return candidate;
        }
    }
    slug.to_owned()
}

/// Derives the git branch name of `raw` (`worktree-<slug>`).
///
/// ```
/// use openjiuwen_harness::worktree::branch_name;
/// assert_eq!(branch_name("feature-auth"), "worktree-feature-auth");
/// ```
#[must_use]
pub fn branch_name(raw: &str) -> String {
    format!("{BRANCH_PREFIX}{}", slugify(raw))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn slugify_keeps_the_accepted_alphabet() {
        assert_eq!(slugify("feature-auth_2.1"), "feature-auth_2.1");
        assert_eq!(slugify("A1_b-c"), "A1_b-c");
    }

    #[test]
    fn slugify_replaces_and_collapses_unsafe_characters() {
        assert_eq!(slugify("feature login!"), "feature-login");
        assert_eq!(slugify("a  b///c"), "a-b-c");
        assert_eq!(slugify("  spaced  "), "spaced");
        assert_eq!(slugify("emoji-🚀-name"), "emoji-name");
    }

    #[test]
    fn slugify_strips_traversal_into_a_flat_name() {
        assert_eq!(slugify("../../etc"), "etc");
        assert_eq!(slugify(".."), DEFAULT_SLUG);
        assert_eq!(slugify("/"), DEFAULT_SLUG);
    }

    #[test]
    fn slugify_caps_the_length_and_retrims() {
        let long = "a".repeat(MAX_SLUG_LENGTH * 2);
        assert_eq!(slugify(&long).len(), MAX_SLUG_LENGTH);

        let long_with_tail = format!("{}{}", "b".repeat(MAX_SLUG_LENGTH - 1), "--tail");
        let slug = slugify(&long_with_tail);
        assert!(slug.chars().count() <= MAX_SLUG_LENGTH);
        assert!(!slug.ends_with('-'));
    }

    #[test]
    fn empty_input_falls_back_to_the_default_slug() {
        assert_eq!(slugify(""), DEFAULT_SLUG);
        assert_eq!(slugify("///"), DEFAULT_SLUG);
        assert_eq!(slugify("..."), DEFAULT_SLUG);
    }

    #[test]
    fn unique_slug_suffixes_collisions() {
        let taken = vec!["feature".to_owned()];
        assert_eq!(unique_slug("other", &taken), "other");
        assert_eq!(unique_slug("feature", &taken), "feature-2");

        let taken = vec!["feature".to_owned(), "feature-2".to_owned()];
        assert_eq!(unique_slug("feature", &taken), "feature-3");

        let long = "a".repeat(MAX_SLUG_LENGTH);
        let taken = vec![long.clone()];
        let unique = unique_slug(&long, &taken);
        assert_eq!(unique.chars().count(), MAX_SLUG_LENGTH);
        assert!(unique.ends_with("-2"));
    }

    #[test]
    fn branch_names_prefix_the_slug() {
        assert_eq!(branch_name("feature-auth"), "worktree-feature-auth");
        assert_eq!(branch_name("a/b"), "worktree-a-b");
        assert_eq!(branch_name(""), "worktree-worktree");
    }

    #[test]
    fn raw_names_refuse_traversal_and_absolute_paths() {
        for raw in [
            "",
            "   ",
            "../escape",
            "a/../../b",
            ".",
            "..",
            "/etc/passwd",
            "\\windows",
            "~/home",
            "C:\\repo",
            "a\0b",
        ] {
            let error = validate_raw_name(raw).unwrap_err();
            assert!(
                matches!(error.kind(), ErrorKind::Security | ErrorKind::Validation),
                "unexpected kind for {raw:?}: {error}"
            );
        }
    }

    #[test]
    fn raw_names_accept_plain_and_nested_names() {
        for raw in ["feature", "feature-auth_2", "user/feature-login", "a.b"] {
            validate_raw_name(raw).unwrap();
        }
    }
}
