//! Fact names as safe, stable file names under `facts/`.
//!
//! A fact name is free-form (it may contain spaces, punctuation, or non-ASCII text) but a
//! file name must be portable, so a fact file is named
//! `<slug>-<hash>.md`:
//!
//! - the slug keeps ASCII alphanumerics (lowercased) and folds every other run of
//!   characters into a single `-`, is trimmed, and is capped at
//!   [`MAX_SLUG_CHARS`](self) characters;
//! - the hash is the first eight lowercase hex digits of a 64-bit FNV-1a digest of the
//!   exact, unmodified name.
//!
//! The slug alone would collide (`Build Command` and `build-command` share a slug), and
//! the hash alone would be unreadable, so [`file_name_for`] pairs the two: the readable
//! prefix keeps a fact file findable by hand, and the hash keeps distinct names distinct
//! without depending on any hashing crate.
use std::fmt::Write as _;
/// Maximum number of characters kept from the readable part of a file name.
pub(super) const MAX_SLUG_CHARS: usize = 48;
/// Slug used when a name holds no usable ASCII alphanumeric character.
const FALLBACK_SLUG: &str = "fact";
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// The readable part of the file name of `name`.
///
/// ASCII alphanumerics are kept and lowercased, every other run of characters becomes a
/// single `-`, the result is trimmed of edge separators and capped at 48 characters, and
/// a name with no usable character yields `"fact"`.
///
/// # Examples
///
/// ```
/// use openjiuwen_memory::lite::facts::slug;
///
/// assert_eq!(slug("Build Command"), "build-command");
/// assert_eq!(slug("  ...  "), "fact");
/// assert_eq!(slug("Ünïcode ✔ 名前"), "n-code");
/// ```
#[must_use]
pub fn slug(name: &str) -> String {
    let mut folded = String::with_capacity(name.len());
    let mut separator_pending = false;
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            if separator_pending && !folded.is_empty() {
                folded.push('-');
            }
            separator_pending = false;
            folded.push(character.to_ascii_lowercase());
        } else {
            separator_pending = true;
        }
    }
    let capped: String = folded.chars().take(MAX_SLUG_CHARS).collect();
    let trimmed = capped.trim_matches('-');
    if trimmed.is_empty() {
        return FALLBACK_SLUG.to_owned();
    }
    trimmed.to_owned()
}
/// The file name of the fact called `name`: `<slug>-<hash>.md`.
///
/// The hash is the first eight lowercase hex digits of the FNV-1a 64-bit digest of the
/// exact name, so two distinct names whose slugs coincide still get distinct files, and
/// the same name always maps to the same file.
///
/// # Examples
///
/// ```
/// use openjiuwen_memory::lite::facts::file_name_for;
///
/// let name = file_name_for("Build Command");
/// assert!(name.starts_with("build-command-"));
/// assert!(name.ends_with(".md"));
/// assert_ne!(name, file_name_for("build-command"));
/// ```
#[must_use]
pub fn file_name_for(name: &str) -> String {
    // The leading eight hex digits of the 16-digit digest are its high 32 bits.
    let digest = fnv1a_64(name.as_bytes()) >> 32;
    let mut file_name = slug(name);
    file_name.push('-');
    let _ = write!(file_name, "{digest:08x}");
    file_name.push_str(".md");
    file_name
}
/// FNV-1a 64-bit hash of `bytes` (deterministic across platforms and releases).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slug_keeps_ascii_alphanumerics_and_folds_the_rest() {
        assert_eq!(slug("Build Command"), "build-command");
        assert_eq!(slug("build_command"), "build-command");
        assert_eq!(slug("  spaced   out  "), "spaced-out");
        assert_eq!(slug("v1.2.3"), "v1-2-3");
        assert_eq!(slug("already-slugged"), "already-slugged");
        assert_eq!(slug("MiXeD"), "mixed");
    }
    #[test]
    fn slug_drops_non_ascii_and_falls_back_when_nothing_remains() {
        assert_eq!(slug("Ünïcode ✔ 名前"), "n-code");
        assert_eq!(slug(""), "fact");
        assert_eq!(slug("---"), "fact");
        assert_eq!(slug("日本語"), "fact");
    }
    #[test]
    fn slug_is_capped_and_trimmed() {
        let long = "a".repeat(200);
        let capped = slug(&long);
        assert_eq!(capped.chars().count(), MAX_SLUG_CHARS);
        assert!(capped.chars().all(|character| character == 'a'));
        let separators = "a-".repeat(60);
        let trimmed = slug(&separators);
        assert!(trimmed.chars().count() <= MAX_SLUG_CHARS);
        assert!(!trimmed.ends_with('-'));
    }
    #[test]
    fn file_names_are_deterministic_and_end_with_md() {
        let first = file_name_for("Build Command");
        let second = file_name_for("Build Command");
        assert_eq!(first, second);
        assert!(first.starts_with("build-command-"), "got {first}");
        assert_eq!(
            std::path::Path::new(&first)
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("md"),
            "got {first}"
        );
        assert_eq!(
            first
                .trim_end_matches(".md")
                .rsplit('-')
                .next()
                .map(str::len),
            Some(8)
        );
    }
    #[test]
    fn distinct_names_with_the_same_slug_get_distinct_files() {
        let names = [
            "Build Command",
            "build-command",
            "build_command",
            "BUILD  COMMAND",
        ];
        let mut files: Vec<String> = names.iter().map(|name| file_name_for(name)).collect();
        assert_eq!(files.len(), 4);
        files.sort();
        files.dedup();
        assert_eq!(files.len(), 4, "names collided: {files:?}");
    }
    #[test]
    fn even_the_fallback_slug_stays_distinct() {
        assert_ne!(file_name_for("日本語"), file_name_for("한국어"));
        assert!(file_name_for("日本語").starts_with("fact-"));
    }
    #[test]
    fn file_names_are_safe_for_any_directory() {
        for name in ["a/b", "a\\b", "..", "../../etc/passwd", "a\0b"] {
            let file_name = file_name_for(name);
            assert!(!file_name.contains('/'), "got {file_name}");
            assert!(!file_name.contains('\\'), "got {file_name}");
            assert!(!file_name.contains('\0'), "got {file_name}");
            assert!(!file_name.starts_with('.'), "got {file_name}");
        }
    }
    #[test]
    fn hashing_is_fnv1a_64() {
        // The well-known FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
