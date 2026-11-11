//! Recursive file search for the local filesystem backend.
//!
//! Ports `search_files` and `_search_files_internal_sync` from Python
//! `openjiuwen/core/sys_operation/local/fs_operation.py`. The Python implementation calls
//! `Path.rglob(pattern)` under the resolved base directory, keeps regular files, and
//! removes the union of `Path.rglob(pat)` for every entry of `exclude_patterns`.
//!
//! # `rglob` mapping
//!
//! - A pattern without `/` is matched against the file *name* (`*.txt` therefore matches
//!   at any depth).
//! - A pattern containing `/` is matched against the path relative to the resolved root,
//!   with `*` / `?` / `[...]` confined to one component (`sub/*.txt` does not match
//!   `sub/deep/a.txt`).
//! - `**` is a whole-component wildcard that matches zero or more components, exactly
//!   like Python's recursive glob (`**/*.txt` matches a root-level `a.txt`, and
//!   `src/**/mod.rs` matches both `src/mod.rs` and `src/lib/mod.rs`).
//!
//! Exclusions use the same matching rule. A pattern that does not compile is reported as a
//! `199003` failure instead of panicking.
//!
//! Recursion never follows directory symlinks (`follow_links(false)`); symlinked files are
//! still matched because the per-entry `metadata` call follows the link. The whole walk
//! runs inside [`tokio::task::spawn_blocking`], so no blocking filesystem call happens on
//! the async task.

use std::path::Path;

use crate::fs::list::file_system_item;
use crate::fs::types::{FileSystemItem, SearchFilesData, SearchFilesParams, SearchPatternMode};
use crate::result::{fs_failure, OpResult};
use crate::workspace::PathPolicy;

/// Matching options applied to path-like glob patterns.
const MATCH_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// The compiled pattern of one search run.
enum Matcher {
    /// Glob matching (Python `Path.rglob` semantics).
    Glob(glob::Pattern),
    /// Regular-expression matching against the file name (Rust addition).
    Regex(regex::Regex),
}

impl Matcher {
    /// Returns `true` when `relative` (a path below the search root) matches.
    fn matches(&self, relative: &Path) -> bool {
        match self {
            Self::Glob(pattern) => glob_matches(pattern, relative),
            Self::Regex(regex) => relative
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| regex.is_match(name)),
        }
    }
}

/// Searches `params.path` recursively for files matching `params.pattern`.
///
/// `mode` selects Python-compatible glob matching ([`SearchPatternMode::Glob`]) or the
/// additive regular-expression variant ([`SearchPatternMode::Regex`], matched against the
/// file name only). Only regular files are returned, in filesystem order, and no match is
/// a successful empty result. `exclude_patterns` are globs using the same matching rule.
///
/// Failures (a blank or disallowed path, a missing path, one that is not a directory, or
/// an uncompilable pattern) are returned as Python-compatible `OpResult` values with code
/// `199003`.
pub async fn search_files(
    policy: &PathPolicy,
    params: &SearchFilesParams,
    mode: SearchPatternMode,
) -> OpResult<SearchFilesData> {
    let root = match policy.resolve_existing(&params.path).await {
        Ok(root) => root,
        Err(error) => return fs_failure("search_files", &error.message()),
    };
    let is_directory = tokio::fs::metadata(&root)
        .await
        .is_ok_and(|metadata| metadata.is_dir());
    if !is_directory {
        let root_path = crate::util::format_path(&root);
        return fs_failure(
            "search_files",
            &format!("Path is not a directory: {root_path}"),
        );
    }

    let matcher = match compile_pattern(&params.pattern, mode) {
        Ok(matcher) => matcher,
        Err(message) => return fs_failure("search_files", &message),
    };
    let excludes = match compile_excludes(params.exclude_patterns.as_deref()) {
        Ok(excludes) => excludes,
        Err(message) => return fs_failure("search_files", &message),
    };

    let search_root = root.clone();
    let matching_files = match tokio::task::spawn_blocking(move || {
        collect_matches(&search_root, &matcher, &excludes)
    })
    .await
    {
        Ok(files) => files,
        Err(error) => return fs_failure("search_files", &error.to_string()),
    };

    OpResult::success(SearchFilesData {
        total_matches: matching_files.len(),
        matching_files,
        search_path: crate::util::format_path(&root),
        search_pattern: params.pattern.clone(),
        exclude_patterns: params.exclude_patterns.clone(),
    })
}

/// Compiles the search pattern for `mode`.
fn compile_pattern(pattern: &str, mode: SearchPatternMode) -> Result<Matcher, String> {
    match mode {
        SearchPatternMode::Glob => glob::Pattern::new(pattern)
            .map(Matcher::Glob)
            .map_err(|error| format!("invalid pattern '{pattern}': {error}")),
        SearchPatternMode::Regex => regex::Regex::new(pattern)
            .map(Matcher::Regex)
            .map_err(|error| format!("invalid regex '{pattern}': {error}")),
    }
}

/// Compiles the exclusion globs, keeping the first failure's message.
fn compile_excludes(patterns: Option<&[String]>) -> Result<Vec<glob::Pattern>, String> {
    patterns
        .unwrap_or_default()
        .iter()
        .map(|pattern| {
            glob::Pattern::new(pattern)
                .map_err(|error| format!("invalid pattern '{pattern}': {error}"))
        })
        .collect()
}

/// Walks `root` and returns the matching regular files in filesystem order.
fn collect_matches(
    root: &Path,
    matcher: &Matcher,
    excludes: &[glob::Pattern],
) -> Vec<FileSystemItem> {
    let mut items = Vec::new();
    for entry in walkdir::WalkDir::new(root).min_depth(1).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::debug!(error = %error, "skipping unreadable directory entry");
                continue;
            }
        };
        let path = entry.path();
        let Ok(metadata) = std::fs::metadata(path) else {
            tracing::debug!(path = %path.display(), "skipping entry whose metadata could not be read");
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        if !matcher.matches(relative) {
            continue;
        }
        if excludes
            .iter()
            .any(|pattern| glob_matches(pattern, relative))
        {
            continue;
        }
        items.push(file_system_item(path, &metadata));
    }
    items
}

/// Returns `true` when `relative` matches the glob `pattern` with `rglob` semantics.
fn glob_matches(pattern: &glob::Pattern, relative: &Path) -> bool {
    if !pattern.as_str().contains('/') {
        return relative
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| pattern.matches(name));
    }
    if pattern.matches_path_with(relative, MATCH_OPTIONS) {
        return true;
    }
    // `glob::Pattern` requires at least one component for every `**`. Python's `**`
    // matches zero or more, so retry with each `**` component removed in turn.
    drop_recursive_components(pattern.as_str(), relative)
}

/// Retries matching after removing the first `**` component from `pattern`.
fn drop_recursive_components(pattern: &str, relative: &Path) -> bool {
    let mut components: Vec<&str> = pattern.split('/').collect();
    let Some(index) = components.iter().position(|component| *component == "**") else {
        return false;
    };
    components.remove(index);
    let variant = components.join("/");
    if variant.is_empty() {
        return false;
    }
    match glob::Pattern::new(&variant) {
        Ok(pattern) => glob_matches(&pattern, relative),
        Err(_) => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::config::LocalWorkConfig;
    use crate::fs::types::SearchFilesParams;

    fn policy(dir: &tempfile::TempDir) -> PathPolicy {
        PathPolicy::new(
            &LocalWorkConfig::default().with_workspace(crate::util::format_path(dir.path())),
        )
    }

    fn build_tree(root: &Path) {
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        std::fs::write(root.join("a.txt"), b"a").unwrap();
        std::fs::write(root.join("b.md"), b"b").unwrap();
        std::fs::write(root.join("sub/c.txt"), b"c").unwrap();
        std::fs::write(root.join("sub/deep/d.txt"), b"d").unwrap();
        std::fs::write(root.join("sub/deep/e.md"), b"e").unwrap();
    }

    fn names(result: &SearchFilesData) -> Vec<String> {
        let mut paths: Vec<String> = result
            .matching_files
            .iter()
            .map(|item| item.name.clone())
            .collect();
        paths.sort();
        paths
    }

    #[tokio::test]
    async fn glob_name_matches_at_any_depth() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "*.txt"),
            SearchPatternMode::Glob,
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(result.total_matches, 3);
        assert_eq!(names(&result), ["a.txt", "c.txt", "d.txt"]);
    }

    #[tokio::test]
    async fn recursive_glob_matches_zero_or_more_directories() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "**/*.txt"),
            SearchPatternMode::Glob,
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(names(&result), ["a.txt", "c.txt", "d.txt"]);

        std::fs::create_dir_all(dir.path().join("src/lib")).unwrap();
        std::fs::write(dir.path().join("src/mod.rs"), b"mod").unwrap();
        std::fs::write(dir.path().join("src/lib/mod.rs"), b"mod").unwrap();
        let nested = search_files(
            &policy,
            &SearchFilesParams::new(".", "src/**/mod.rs"),
            SearchPatternMode::Glob,
        )
        .await
        .into_result()
        .unwrap();
        let mut paths: Vec<String> = nested
            .matching_files
            .iter()
            .map(|item| item.path.clone())
            .collect();
        paths.sort();
        assert_eq!(paths.len(), 2, "matches both zero and one nested level");
    }

    #[tokio::test]
    async fn path_glob_does_not_cross_separators() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "sub/*.txt"),
            SearchPatternMode::Glob,
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(names(&result), ["c.txt"], "sub/deep/d.txt must not match");
    }

    #[tokio::test]
    async fn exclude_patterns_remove_matches() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);

        let params = SearchFilesParams::new(".", "*").with_excludes(["*.md"]);
        let result = search_files(&policy, &params, SearchPatternMode::Glob)
            .await
            .into_result()
            .unwrap();
        assert_eq!(names(&result), ["a.txt", "c.txt", "d.txt"]);

        let params = SearchFilesParams::new(".", "*.txt").with_excludes(["sub/*.txt"]);
        let result = search_files(&policy, &params, SearchPatternMode::Glob)
            .await
            .into_result()
            .unwrap();
        assert_eq!(
            names(&result),
            ["a.txt", "d.txt"],
            "sub/c.txt is excluded, sub/deep/d.txt is not"
        );
        assert_eq!(
            result.exclude_patterns.as_deref(),
            Some(&["sub/*.txt".to_string()][..])
        );
    }

    #[tokio::test]
    async fn no_match_is_a_successful_empty_result() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "*.xyz"),
            SearchPatternMode::Glob,
        )
        .await
        .into_result()
        .unwrap();
        assert_eq!(result.total_matches, 0);
        assert_eq!(result.matching_files, Vec::new());
    }

    #[tokio::test]
    async fn missing_directory_is_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new("missing", "*.txt"),
            SearchPatternMode::Glob,
        )
        .await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("Path is not a directory"),
            "{}",
            result.message
        );
        assert!(result.message.contains("search_files"));
    }

    #[tokio::test]
    async fn file_path_is_not_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), b"file").unwrap();
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new("file.txt", "*.txt"),
            SearchPatternMode::Glob,
        )
        .await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("Path is not a directory"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn regex_mode_matches_file_names_only() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);

        let params = SearchFilesParams::new(".", r"^[ac]\.txt$");
        let result = search_files(&policy, &params, SearchPatternMode::Regex)
            .await
            .into_result()
            .unwrap();
        assert_eq!(names(&result), ["a.txt", "c.txt"]);

        // Anchored at a directory separator: only a path-based matcher could match it.
        let params = SearchFilesParams::new(".", "^sub/");
        let result = search_files(&policy, &params, SearchPatternMode::Regex)
            .await
            .into_result()
            .unwrap();
        assert_eq!(result.total_matches, 0, "regex sees only the file name");
    }

    #[tokio::test]
    async fn result_echoes_the_request_parameters() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);
        let params = SearchFilesParams::new(".", "*.txt").with_excludes(["*.csv"]);
        let result = search_files(&policy, &params, SearchPatternMode::Glob)
            .await
            .into_result()
            .unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        assert_eq!(result.search_path, crate::util::format_path(&root));
        assert_eq!(result.search_pattern, "*.txt");
        assert_eq!(
            result.exclude_patterns.as_deref(),
            Some(&["*.csv".to_string()][..])
        );
        assert!(result
            .matching_files
            .iter()
            .all(|item| item.file_type.as_deref() == Some(".txt")));
    }

    #[tokio::test]
    async fn invalid_pattern_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        build_tree(dir.path());
        let policy = policy(&dir);

        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "["),
            SearchPatternMode::Glob,
        )
        .await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("invalid pattern '['"),
            "{}",
            result.message
        );

        let result = search_files(
            &policy,
            &SearchFilesParams::new(".", "("),
            SearchPatternMode::Regex,
        )
        .await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("invalid regex '('"),
            "{}",
            result.message
        );

        let params = SearchFilesParams::new(".", "*.txt").with_excludes(["[bad"]);
        let result = search_files(&policy, &params, SearchPatternMode::Glob).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("invalid pattern '[bad'"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn blank_path_is_rejected_by_the_policy() {
        let dir = tempfile::tempdir().unwrap();
        let policy = policy(&dir);
        let result = search_files(
            &policy,
            &SearchFilesParams::new("", "*.txt"),
            SearchPatternMode::Glob,
        )
        .await;
        assert_eq!(result.code, 199_003);
        assert!(
            result.message.contains("path must not be empty"),
            "{}",
            result.message
        );
    }
}
