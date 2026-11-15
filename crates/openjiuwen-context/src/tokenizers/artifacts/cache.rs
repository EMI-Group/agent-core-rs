//! Cache-directory precedence, artefact naming, deterministic search and digests.
//!
//! The manager in [`super`] owns the resolution policy; this module owns the
//! filesystem primitives it needs: where the cache lives, which file name an
//! engine expects, how a directory is searched deterministically, how an id
//! becomes a relative cache path, and how a SHA-256 digest is computed without
//! adding a dependency.
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::tokenizers::TokenizerEngine;

/// The cache directory used when nothing overrides it.
///
/// It is resolved against `HOME`, so the default is
/// `~/.cache/openjiuwen/tokenizers`. When `HOME` is unset as well the relative
/// path is used (a documented fallback, never a temp directory: a tokenizer
/// cache is meant to survive the process).
pub const DEFAULT_CACHE_SUBDIR: &str = ".cache/openjiuwen/tokenizers";

/// Environment variable that overrides the cache directory.
pub const CACHE_DIR_ENV: &str = "OPENJIUWEN_TOKENIZER_CACHE_DIR";

/// Environment variable consulted for the `HuggingFace` endpoint.
pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";

/// How many directory levels below an artefact root are searched.
pub const MAX_ARTIFACT_SEARCH_DEPTH: usize = 4;

/// The upper bound on artefacts collected by one directory search.
const MAX_ARTIFACT_CANDIDATES: usize = 64;

/// Resolves the cache directory: explicit argument, environment, then `HOME`.
///
/// An empty explicit value is ignored, which keeps a blank configuration field
/// from meaning "the current directory".
#[must_use]
pub fn cache_dir_from(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit.filter(|path| !path.as_os_str().is_empty()) {
        return expand_home(path);
    }
    if let Some(value) = env_non_empty(CACHE_DIR_ENV) {
        return expand_home(value);
    }
    match home_dir() {
        Some(home) => home.join(DEFAULT_CACHE_SUBDIR),
        None => PathBuf::from(DEFAULT_CACHE_SUBDIR),
    }
}

/// The default cache directory: `~/.cache/openjiuwen/tokenizers`.
///
/// This is [`cache_dir_from`] with no explicit override, so it honours
/// `OPENJIUWEN_TOKENIZER_CACHE_DIR` first and then `HOME`.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    cache_dir_from(None)
}

/// Expands a leading `~` component against `HOME`.
///
/// Only a leading `~` is expanded, and only when it is a whole component; a
/// path that merely starts with the character (for example `~cache`) is left
/// untouched. When `HOME` is unset the path is returned unchanged.
#[must_use]
pub fn expand_home(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    let mut components = path.components();
    let Some(Component::Normal(first)) = components.next() else {
        return path.to_path_buf();
    };
    if first != "~" {
        return path.to_path_buf();
    }
    let Some(home) = home_dir() else {
        return path.to_path_buf();
    };
    let rest: PathBuf = components.collect();
    home.join(rest)
}

/// The artefact file name an engine expects.
///
/// Python derives it from the spec's engine string: `tiktoken.model` for the
/// `tiktoken` engine and `tokenizer.json` for everything else (including
/// [`TokenizerEngine::Auto`] and [`TokenizerEngine::Native`]).
#[must_use]
pub fn artifact_name(engine: TokenizerEngine) -> &'static str {
    match engine {
        TokenizerEngine::Tiktoken => "tiktoken.model",
        TokenizerEngine::Auto | TokenizerEngine::Native => "tokenizer.json",
    }
}

/// Finds an artefact below `root`, or returns `root` itself when it is a file.
///
/// The search is deterministic, unlike Python's `Path.rglob`:
///
/// 1. `root` itself when it is a file (any name: an explicit `artifact_path`
///    may carry a project-specific file name);
/// 2. `root/<artifact name>` when that is a file;
/// 3. otherwise a depth-first walk over directories sorted by path, up to
///    [`MAX_ARTIFACT_SEARCH_DEPTH`] levels and `MAX_ARTIFACT_CANDIDATES`
///    hits, taking the lexicographically smallest absolute match.
#[must_use]
pub fn find_artifact(root: &Path, engine: TokenizerEngine) -> Option<PathBuf> {
    let name = artifact_name(engine);
    if root.is_file() {
        return Some(root.to_path_buf());
    }
    if !root.is_dir() {
        return None;
    }
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let mut found = Vec::new();
    collect_artifacts(root, name, MAX_ARTIFACT_SEARCH_DEPTH, &mut found);
    found.sort();
    found.into_iter().next()
}

/// Collects artefacts named `name` below `dir`, depth first in sorted order.
///
/// A directory the process cannot read, and an entry it cannot stat, are logged and skipped instead of ending the search silently: the walk is best-effort by design, so one unreadable subdirectory must not hide the artefacts beside it.
fn collect_artifacts(dir: &Path, name: &str, depth: usize, found: &mut Vec<PathBuf>) {
    if depth == 0 || found.len() >= MAX_ARTIFACT_CANDIDATES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::debug!(
                path = %dir.display(),
                %error,
                "tokenizer artefact search could not read a directory"
            );
            return;
        }
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| match entry {
            Ok(entry) => Some(entry.path()),
            Err(error) => {
                tracing::debug!(
                    path = %dir.display(),
                    %error,
                    "tokenizer artefact search skipped an unreadable entry"
                );
                None
            }
        })
        .collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_artifacts(&path, name, depth - 1, found);
        } else if path.file_name().is_some_and(|file| file == name) {
            found.push(path);
        }
    }
}

/// Turns a tokenizer id into a relative cache path.
///
/// The id keeps its `/` separators (`org/model` is two directories) but must
/// stay a relative path: absolute paths, `.`/`..` components, URL schemes,
/// query/fragment characters, backslashes and control characters are refused,
/// so a remote id can never escape the cache directory or reshape a request.
#[must_use]
pub fn relative_artifact_id(id: &str) -> Option<PathBuf> {
    let id = id.trim();
    if id.is_empty() || id.contains("://") {
        return None;
    }
    let mut out = PathBuf::new();
    for component in Path::new(id).components() {
        let Component::Normal(part) = component else {
            return None;
        };
        let part = part.to_str()?;
        let refused = part.contains(['?', '#', '\\']) || part.chars().any(char::is_control);
        if refused {
            return None;
        }
        out.push(part);
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// The lowercase hex SHA-256 of one file, read in bounded chunks.
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] when the file cannot be read.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        if let Some(chunk) = buffer.get(..read) {
            hasher.update(chunk);
        }
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

/// The lowercase hex SHA-256 of a byte slice.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize().as_slice())
}

/// Renders a digest as lowercase hex.
fn hex_digest(digest: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The value of `name` when it is set and not blank.
fn env_non_empty(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

/// The user's home directory, from `HOME`.
fn home_dir() -> Option<PathBuf> {
    env_non_empty("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_names_follow_the_engine() {
        assert_eq!(artifact_name(TokenizerEngine::Tiktoken), "tiktoken.model");
        assert_eq!(artifact_name(TokenizerEngine::Native), "tokenizer.json");
        assert_eq!(artifact_name(TokenizerEngine::Auto), "tokenizer.json");
    }

    #[test]
    fn ids_stay_relative_and_dot_free() {
        assert_eq!(
            relative_artifact_id("org/model"),
            Some(PathBuf::from("org").join("model"))
        );
        for id in ["../escape", "/abs", "a/../b", "https://x/y", "id?query", ""] {
            assert_eq!(relative_artifact_id(id), None, "{id:?} must be refused");
        }
    }

    #[test]
    fn digests_are_lowercase_hex() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_missing_root_has_no_artifact() {
        let missing = std::env::temp_dir().join("openjiuwen-missing-artifact-root");
        assert_eq!(find_artifact(&missing, TokenizerEngine::Auto), None);
    }

    #[test]
    fn home_expansion_only_touches_a_leading_component() {
        let expanded = expand_home("~cache/x");
        assert_eq!(expanded, PathBuf::from("~cache/x"));
    }
}
