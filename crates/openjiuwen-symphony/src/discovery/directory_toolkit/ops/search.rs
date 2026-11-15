//! The `search` operation: content and entry matching, snippet assembly, and the
//! category-hint fallback.

use std::collections::{BTreeMap, BTreeSet};

use crate::discovery::directory_toolkit::matcher::{
    case_fold, glob_match, has_wildcard, literal_match_rank,
};
use crate::discovery::directory_toolkit::render::{
    compact, directory_description, safe_match_snippet, sanitize_model_text, search_directory_row,
    search_row, Row,
};
use crate::discovery::directory_toolkit::validate::{
    SearchMatch, SearchResult, ValidatedRequest, MAX_QUERY_CHARS,
};
use crate::discovery::directory_toolkit::view::{
    category_segments, DirectoryEntry, SkillDirectoryView,
};
use crate::discovery::models::SkillRecord;
use crate::discovery::SkillFs;
use crate::error::{validation, Result};
use crate::model::CapabilityDescriptor;
use crate::search::LexicalIndex;

use super::{
    Execution, CATEGORY_CAPABILITY_TYPE, CATEGORY_HINT_CHARS, DEFAULT_MULTI_QUERY_LIMIT,
    SNIPPET_CHARS,
};

/// Runs the `search` operation.
///
/// `match = content` ranks whole records by content; `name`/`path` rank directory entries
/// by their name or path. `result = matches` renders one row per matching field line
/// instead of one row per Skill.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for an unknown scope path or a query the matcher cannot
/// use, and propagates catalog search failures.
#[allow(clippy::too_many_lines)]
pub fn execute_search(
    catalog: &SkillFs,
    view: &SkillDirectoryView,
    paths: &[String],
    request: &ValidatedRequest,
) -> Result<Execution> {
    if request.queries.is_empty() {
        return Err(validation("query", "must contain 1-8 strings"));
    }
    let simple_page = request.page_size.is_some();
    let content = request.search_match == SearchMatch::Content;
    let records: Vec<SkillRecord> = if content {
        view.scoped_records(paths, request.max_depth)?
    } else {
        Vec::new()
    };
    let entries: Vec<DirectoryEntry> = if content {
        Vec::new()
    } else {
        view.searchable_entries(paths, request.max_depth)?
    };
    let limit = if request.queries.len() > 1 && !simple_page {
        Some(request.per_query_limit.unwrap_or(DEFAULT_MULTI_QUERY_LIMIT))
    } else {
        request.per_query_limit
    };

    let mut selected: BTreeMap<String, SearchItem> = BTreeMap::new();
    let mut indexes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut query_counts: Vec<usize> = Vec::new();
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut all_matches: BTreeSet<String> = BTreeSet::new();

    for (position, query) in request.queries.iter().enumerate() {
        let query_index = position + 1;
        let matches: Vec<SearchItem> = if content {
            let hits = content_matches(
                catalog,
                Some(view),
                &records,
                query,
                request.case_insensitive,
                request.fixed_strings,
            )?;
            if simple_page && hits.is_empty() {
                search_categories(view, paths, query, request.max_depth)?
                    .into_iter()
                    .map(SearchItem::Entry)
                    .collect()
            } else {
                hits.into_iter().map(SearchItem::Record).collect()
            }
        } else {
            entry_hits(
                &entries,
                view,
                query,
                request.search_match,
                request.case_insensitive,
            )
            .into_iter()
            .map(SearchItem::Entry)
            .collect()
        };
        all_matches.extend(matches.iter().map(SearchItem::identity));
        let chosen: Vec<SearchItem> = match limit {
            None => matches,
            Some(limit) => matches.into_iter().take(limit).collect(),
        };
        query_counts.push(chosen.len());
        let mut group: Vec<String> = Vec::with_capacity(chosen.len());
        for item in chosen {
            let identity = item.identity();
            group.push(identity.clone());
            if !selected.contains_key(&identity) {
                selected.insert(identity.clone(), item);
            }
            if request.queries.len() > 1 {
                let recorded = indexes.entry(identity).or_default();
                if !recorded.contains(&query_index) {
                    recorded.push(query_index);
                }
            }
        }
        groups.push(group);
    }

    let mut order: Vec<String> = Vec::new();
    if simple_page && groups.len() > 1 {
        let deepest = groups.iter().map(Vec::len).max().unwrap_or(0);
        for rank in 0..deepest {
            for group in &groups {
                if let Some(identity) = group.get(rank) {
                    order.push(identity.clone());
                }
            }
        }
    } else {
        for group in &groups {
            order.extend(group.iter().cloned());
        }
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    order.retain(|identity| seen.insert(identity.clone()));

    let mut rows: Vec<Row> = Vec::new();
    for identity in &order {
        let Some(item) = selected.get(identity) else {
            continue;
        };
        let matched = if simple_page {
            Vec::new()
        } else {
            indexes.get(identity).cloned().unwrap_or_default()
        };
        match item {
            SearchItem::Entry(entry) if entry.is_directory() => {
                rows.push(search_directory_row(entry, view, &matched));
            }
            SearchItem::Entry(entry) => {
                let Some(record) = view.record(&entry.worker_id) else {
                    continue;
                };
                rows.push(search_row(record, view, &matched, None));
            }
            SearchItem::Record(hit) => {
                let hit_queries = matched_queries(&request.queries, &matched);
                if request.search_result == SearchResult::Files {
                    let snippet = if simple_page && !hit.excerpt.is_empty() {
                        Some(safe_match_snippet(&hit.excerpt))
                    } else {
                        None
                    };
                    rows.push(search_row(&hit.record, view, &matched, snippet.as_deref()));
                    continue;
                }
                for snippet in matched_snippets_with(
                    catalog,
                    Some(view),
                    &hit.record,
                    &hit_queries,
                    request.case_insensitive,
                    request.fixed_strings,
                ) {
                    rows.push(search_row(&hit.record, view, &matched, Some(&snippet)));
                }
            }
        }
    }

    let complete = match limit {
        None => true,
        Some(limit) => query_counts.iter().all(|count| *count < limit),
    };
    let total_count = if request.search_result == SearchResult::Matches {
        rows.len()
    } else {
        all_matches.len()
    };
    Ok(Execution {
        rows,
        total_count,
        complete,
    })
}

/// Ranks the categories under `paths` against `query`.
///
/// The category rows are indexed by label plus their routing description, exactly like
/// Python's `_search_categories`, and ranked by the crate's BM25 index so the hints are
/// deterministic. The root row is never a hint.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for an unknown scope path.
pub fn search_categories(
    view: &SkillDirectoryView,
    paths: &[String],
    query: &str,
    max_depth: Option<usize>,
) -> Result<Vec<DirectoryEntry>> {
    let mut by_path: BTreeMap<String, DirectoryEntry> = BTreeMap::new();
    for entry in view.tree_entries(paths, max_depth, true)? {
        if entry.path == view.root_path() {
            continue;
        }
        by_path.insert(entry.path.clone(), entry);
    }
    if by_path.is_empty() {
        return Ok(Vec::new());
    }
    let descriptors: Vec<CapabilityDescriptor> = by_path
        .values()
        .map(|entry| {
            CapabilityDescriptor::new(
                CATEGORY_CAPABILITY_TYPE,
                entry.path.clone(),
                entry.label.clone(),
            )
            .with_description(directory_description(
                &entry.description,
                CATEGORY_HINT_CHARS,
            ))
        })
        .collect();
    let index = LexicalIndex::build(descriptors);
    let mut hints: Vec<DirectoryEntry> = Vec::new();
    for hit in index.search(query, 0) {
        let Some((_, path)) = hit.qualified_id.split_once(':') else {
            continue;
        };
        if let Some(entry) = by_path.get(path) {
            hints.push(entry.clone());
        }
    }
    Ok(hints)
}

/// Returns the scoped records whose text matches `query`.
///
/// The catalog's BM25 ranking comes first (so the strongest matches lead), followed by a
/// literal pass in scope order over the records whose name, aliases, description, body,
/// or category contains the query verbatim. Results are de-duplicated.
///
/// This entry point has no view, so the category text a match reads is the record's own
/// front-matter chain ([`category_segments`]); the toolkit always passes a view, which
/// renders the same chain.
///
/// # Errors
///
/// Returns [`Error::Validation`](crate::error::Error::Validation) for an unusable query, and propagates a catalog search
/// failure.
pub fn content_hits(
    catalog: &SkillFs,
    records: &[SkillRecord],
    query: &str,
    case_insensitive: bool,
    fixed_strings: bool,
) -> Result<Vec<SkillRecord>> {
    Ok(content_matches(
        catalog,
        None,
        records,
        query,
        case_insensitive,
        fixed_strings,
    )?
    .into_iter()
    .map(|hit| hit.record)
    .collect())
}

/// Ranks the directory entries of `entries` that match `query` by name or path.
///
/// Mirrors Python's `_search_entry_one`: a query without a wildcard matches as
/// `*<query>*`, a query with one is a glob, and the rank is `3` for a glob or the best
/// literal rank (`0` exact, `1` prefix, `2` substring) otherwise. Ties break on
/// `(path.casefold(), path, kind)`.
#[must_use]
pub fn entry_hits(
    entries: &[DirectoryEntry],
    view: &SkillDirectoryView,
    query: &str,
    search_match: SearchMatch,
    case_insensitive: bool,
) -> Vec<DirectoryEntry> {
    let value = if case_insensitive {
        case_fold(query)
    } else {
        query.to_string()
    };
    let wildcard = has_wildcard(&value);
    let pattern = if wildcard {
        value.clone()
    } else {
        format!("*{value}*")
    };
    let mut matched: Vec<(usize, DirectoryEntry)> = Vec::new();
    for entry in entries {
        let candidates: Vec<String> = if entry.is_directory() {
            if search_match == SearchMatch::Name {
                let segment = last_segment(&entry.path);
                vec![
                    if segment.is_empty() { "." } else { segment }.to_string(),
                    entry.label.clone(),
                ]
            } else {
                vec![entry.path.clone()]
            }
        } else {
            let Some(record) = view.record(&entry.worker_id) else {
                continue;
            };
            if search_match == SearchMatch::Name {
                vec![
                    last_segment(&entry.path).to_string(),
                    record.worker_id.clone(),
                    record.name.clone(),
                ]
            } else {
                vec![record.skill_file.clone()]
            }
        };
        let compared: Vec<String> = candidates
            .iter()
            .map(|candidate| {
                if case_insensitive {
                    case_fold(candidate)
                } else {
                    candidate.clone()
                }
            })
            .collect();
        if !compared
            .iter()
            .any(|candidate| glob_match(candidate, &pattern))
        {
            continue;
        }
        let rank = if wildcard {
            3
        } else {
            compared
                .iter()
                .map(|candidate| literal_match_rank(candidate, &value))
                .min()
                .unwrap_or(3)
        };
        matched.push((rank, entry.clone()));
    }
    matched.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| case_fold(&left.1.path).cmp(&case_fold(&right.1.path)))
            .then_with(|| left.1.path.cmp(&right.1.path))
            .then_with(|| left.1.kind.cmp(&right.1.kind))
    });
    matched.into_iter().map(|(_, entry)| entry).collect()
}

/// Collects the `matches` field lines of one record that any query hits.
///
/// The fields are `id`, `name`, `alias`, `description`, `body`, `category`, and `path`;
/// a body line is labelled `body:<line>: <text>`, every other field `"<field>: <text>"`.
/// Snippets are sanitized, compacted to [`SNIPPET_CHARS`], and de-duplicated. When no
/// single line matches but a whole field does (a match spanning a line break), the field
/// is returned once, bounded.
#[must_use]
pub fn matched_snippets(
    catalog: &SkillFs,
    view: &SkillDirectoryView,
    record: &SkillRecord,
    queries: &[String],
    case_insensitive: bool,
) -> Vec<String> {
    matched_snippets_with(
        catalog,
        Some(view),
        record,
        queries,
        case_insensitive,
        false,
    )
}

/// One search hit: a content-ranked record or a matched directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SearchItem {
    /// A whole-record content match.
    Record(ContentMatch),
    /// A directory entry match.
    Entry(DirectoryEntry),
}

impl SearchItem {
    /// The de-duplication identity Python's `_search_identity` builds.
    fn identity(&self) -> String {
        match self {
            Self::Record(hit) => format!("skill\0{}", hit.record.worker_id),
            Self::Entry(entry) => format!("{}\0{}", entry.kind.as_str(), entry.path),
        }
    }
}

/// One content match: the scoped record and the catalog's bounded evidence, if any.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ContentMatch {
    /// The matching record.
    record: SkillRecord,
    /// The catalog excerpt around a matched term; empty for a literal-only match.
    excerpt: String,
}

/// Ranks the scoped records against `query`, keeping the catalog excerpt per hit.
fn content_matches(
    catalog: &SkillFs,
    view: Option<&SkillDirectoryView>,
    records: &[SkillRecord],
    query: &str,
    case_insensitive: bool,
    fixed_strings: bool,
) -> Result<Vec<ContentMatch>> {
    let matcher = TextMatcher::new(query, case_insensitive, fixed_strings)?;
    let mut by_id: BTreeMap<&str, &SkillRecord> = BTreeMap::new();
    for record in records {
        by_id.insert(record.worker_id.as_str(), record);
    }
    let mut found: Vec<ContentMatch> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for hit in catalog.search(query, 0)? {
        let Some(record) = by_id.get(hit.worker_id.as_str()) else {
            continue;
        };
        if !seen.insert(record.worker_id.clone()) {
            continue;
        }
        found.push(ContentMatch {
            record: (*record).clone(),
            excerpt: hit.excerpt,
        });
    }
    for record in records {
        if seen.contains(&record.worker_id) {
            continue;
        }
        if !matcher.matches(&searchable_text(view, catalog, record)) {
            continue;
        }
        seen.insert(record.worker_id.clone());
        found.push(ContentMatch {
            record: record.clone(),
            excerpt: String::new(),
        });
    }
    Ok(found)
}

/// Collects [`matched_snippets`] with an explicit `fixed_strings`.
fn matched_snippets_with(
    catalog: &SkillFs,
    view: Option<&SkillDirectoryView>,
    record: &SkillRecord,
    queries: &[String],
    case_insensitive: bool,
    fixed_strings: bool,
) -> Vec<String> {
    let matchers: Vec<TextMatcher> = queries
        .iter()
        .filter_map(|query| TextMatcher::new(query, case_insensitive, fixed_strings).ok())
        .collect();
    if matchers.is_empty() {
        return Vec::new();
    }
    let any_match = |value: &str| matchers.iter().any(|matcher| matcher.matches(value));
    let body = read_body(catalog, &record.worker_id);
    let aliases = record.aliases.join("\n");
    let category = category_text(catalog, view, record);
    let fields: [(&str, &str); 7] = [
        ("id", record.worker_id.as_str()),
        ("name", record.name.as_str()),
        ("alias", aliases.as_str()),
        ("description", record.description.as_str()),
        ("body", body),
        ("category", category.as_str()),
        ("path", record.skill_file.as_str()),
    ];
    let mut snippets: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (label, value) in fields {
        let lines: Vec<&str> = if value.is_empty() {
            vec![""]
        } else {
            value.split('\n').collect()
        };
        for (index, line) in lines.iter().enumerate() {
            if !any_match(line) {
                continue;
            }
            let safe = compact(&sanitize_model_text(line), SNIPPET_CHARS);
            if safe.is_empty() {
                continue;
            }
            let location = if label == "body" {
                format!("body:{}", index + 1)
            } else {
                label.to_string()
            };
            let snippet = format!("{location}: {safe}");
            if seen.insert(snippet.clone()) {
                snippets.push(snippet);
            }
        }
    }
    if !snippets.is_empty() {
        return snippets;
    }
    for (label, value) in fields {
        if !any_match(value) {
            continue;
        }
        let safe = compact(&sanitize_model_text(value), SNIPPET_CHARS);
        if !safe.is_empty() {
            return vec![format!("{label}: {safe}")];
        }
    }
    Vec::new()
}

/// Reads one Skill's body, warning instead of failing when the file cannot be read.
///
/// A body that cannot be read still leaves the catalog's identity, name, description and
/// category searchable, so the failure degrades one field instead of aborting a search.
fn read_body<'a>(catalog: &'a SkillFs, worker_id: &str) -> &'a str {
    if let Some(body) = catalog.read_body(worker_id) {
        body
    } else {
        tracing::debug!(
            skill = worker_id,
            "skill body unavailable for skill_index search"
        );
        ""
    }
}

/// The text a content search reads for one record.
///
/// Python's `_search_text`: the identity, name, aliases, description, body, and category
/// of the document, newline separated.
fn searchable_text(
    view: Option<&SkillDirectoryView>,
    catalog: &SkillFs,
    record: &SkillRecord,
) -> String {
    let body = read_body(catalog, &record.worker_id);
    let aliases = record.aliases.join("\n");
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        record.worker_id,
        record.name,
        aliases,
        record.description,
        body,
        category_text(catalog, view, record)
    )
}

/// The category text of one record: the view's chain, or the record's own front-matter
/// chain when no view is at hand.
fn category_text(
    catalog: &SkillFs,
    view: Option<&SkillDirectoryView>,
    record: &SkillRecord,
) -> String {
    if let Some(view) = view {
        return view.category_text(&record.worker_id);
    }
    let classification = catalog
        .descriptor(&record.worker_id)
        .map(|descriptor| descriptor.classification.as_str())
        .unwrap_or_default();
    category_segments(classification).join("\n")
}

/// A text matcher: a case-(in)sensitive substring, or a glob when the pattern carries a
/// wildcard and `fixed_strings` is unset.
///
/// The stand-in for Python's `compile_matcher`, which compiles a bounded regular
/// expression; this crate ships no regex engine. `fixed_strings` therefore only disables
/// the glob reading.
#[derive(Debug, Clone)]
pub(super) struct TextMatcher {
    /// The (optionally folded) pattern.
    pattern: String,
    /// Whether the pattern carries a `*`/`?`/`[` wildcard.
    wildcard: bool,
    /// Whether both sides are folded before comparing.
    case_insensitive: bool,
}

impl TextMatcher {
    /// Builds a matcher for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for a blank query or one longer than
    /// [`MAX_QUERY_CHARS`].
    pub(super) fn new(query: &str, case_insensitive: bool, fixed_strings: bool) -> Result<Self> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(validation("query", "must be a non-empty string"));
        }
        if trimmed.chars().count() > MAX_QUERY_CHARS {
            return Err(validation(
                "query",
                format!("must not exceed {MAX_QUERY_CHARS} characters"),
            ));
        }
        let pattern = if case_insensitive {
            case_fold(trimmed)
        } else {
            trimmed.to_string()
        };
        Ok(Self {
            wildcard: !fixed_strings && has_wildcard(&pattern),
            pattern,
            case_insensitive,
        })
    }

    /// Whether `value` matches this pattern.
    pub(super) fn matches(&self, value: &str) -> bool {
        let candidate = if self.case_insensitive {
            case_fold(value)
        } else {
            value.to_string()
        };
        if !self.wildcard {
            return candidate.contains(&self.pattern);
        }
        // A wildcard pattern matches a whole value or a run inside it, so `pdf-*` still
        // finds "see pdf-export" in a longer line.
        glob_match(&candidate, &self.pattern)
            || glob_match(&candidate, &format!("*{}*", self.pattern))
    }
}

/// The queries behind the matched 1-based indexes, or every query when none is recorded.
fn matched_queries(queries: &[String], indexes: &[usize]) -> Vec<String> {
    if indexes.is_empty() {
        return queries.to_vec();
    }
    indexes
        .iter()
        .filter_map(|index| queries.get(index - 1).cloned())
        .collect()
}

/// The last non-empty segment of a virtual path.
fn last_segment(path: &str) -> &str {
    path.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or_default()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use crate::discovery::directory_toolkit::validate::{OutputMode, PipelineStage};

    use super::super::apply_pipeline;

    #[test]
    fn text_matcher_reads_substrings_globs_and_fixed_strings() {
        let plain = TextMatcher::new("PDF", true, false).unwrap();
        assert!(plain.matches("a pdf export"));
        assert!(!plain.matches("a ocr export"));
        assert!(plain.matches("PDF"));
        let glob = TextMatcher::new("pdf-*", true, false).unwrap();
        assert!(glob.matches("pdf-export"));
        assert!(glob.matches("a pdf-export skill"));
        assert!(!glob.matches("export"));
        let fixed = TextMatcher::new("pdf-*", true, true).unwrap();
        assert!(!fixed.matches("pdf-export"));
        assert!(fixed.matches("a pdf-* literal"));
        let sensitive = TextMatcher::new("PDF", false, false).unwrap();
        assert!(sensitive.matches("PDF export"));
        assert!(!sensitive.matches("pdf export"));
        assert!(TextMatcher::new(" ", true, false).is_err());
        assert!(TextMatcher::new(&"a".repeat(MAX_QUERY_CHARS + 1), true, false).is_err());
    }

    #[test]
    fn entry_hits_rank_name_and_path_matches() {
        let view = SkillDirectoryView::from_records(&[], &BTreeMap::new());
        let entries = vec![
            DirectoryEntry::directory("/Data/Export", "Export"),
            DirectoryEntry::directory("/Data", "Data"),
            DirectoryEntry::directory("/Notes", "Notes"),
        ];
        let name = entry_hits(&entries, &view, "export", SearchMatch::Name, true);
        assert_eq!(name.len(), 1);
        assert_eq!(name[0].path, "/Data/Export");
        let path = entry_hits(&entries, &view, "/Data", SearchMatch::Path, true);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].path, "/Data");
        let glob = entry_hits(&entries, &view, "*port", SearchMatch::Name, true);
        assert_eq!(glob.len(), 1);
        let none = entry_hits(&entries, &view, "missing", SearchMatch::Name, true);
        assert!(none.is_empty(), "{none:?}");
        let sensitive = entry_hits(&entries, &view, "Export", SearchMatch::Name, false);
        assert_eq!(sensitive.len(), 1);
        let strict = entry_hits(&entries, &view, "export", SearchMatch::Name, false);
        assert!(strict.is_empty(), "{strict:?}");
    }

    #[test]
    fn last_segment_ignores_trailing_separators() {
        assert_eq!(last_segment("/Data/Export"), "Export");
        assert_eq!(last_segment("/Data/Export/"), "Export");
        assert_eq!(last_segment("/"), "");
        assert_eq!(last_segment("a"), "a");
    }

    #[test]
    fn pipelines_flatten_and_count() {
        let rows = vec![Row::new("a\nb", "alpha"), Row::text_only("c")];
        let outcome = apply_pipeline(&rows, &[], OutputMode::Entries).unwrap();
        assert_eq!(outcome.rows.len(), 2);
        assert!(outcome.complete);
        assert_eq!(outcome.count_value, None);
        let limit = apply_pipeline(&rows, &[PipelineStage::limit(1)], OutputMode::Entries).unwrap();
        assert_eq!(limit.rows.len(), 1);
        assert!(!limit.complete);
        let counted =
            apply_pipeline(&rows, &[PipelineStage::count()], OutputMode::Entries).unwrap();
        assert_eq!(counted.count_value, Some(3));
        assert_eq!(counted.rows.len(), 1);
        assert_eq!(counted.rows[0].text, "3");
        let counted_then_limited = apply_pipeline(
            &rows,
            &[PipelineStage::count(), PipelineStage::limit(1)],
            OutputMode::Entries,
        )
        .unwrap();
        assert_eq!(counted_then_limited.count_value, None);
        assert_eq!(counted_then_limited.transformed_count, Some(1));
        let mode = apply_pipeline(&rows, &[], OutputMode::Count).unwrap();
        assert_eq!(mode.count_value, Some(3));
    }

    #[test]
    fn pipelines_filter_slice_and_reject_bad_stages() {
        let rows = vec![
            Row::new("alpha one", "alpha"),
            Row::new("beta two", "beta"),
            Row::new("ALPHA three", "gamma"),
        ];
        let filtered = apply_pipeline(
            &rows,
            &[PipelineStage::filter("alpha")],
            OutputMode::Entries,
        )
        .unwrap();
        assert_eq!(filtered.rows.len(), 2);
        let mut inverted = PipelineStage::filter("alpha");
        inverted.invert = true;
        let inverted = apply_pipeline(&rows, &[inverted], OutputMode::Entries).unwrap();
        assert_eq!(inverted.rows.len(), 1);
        let sliced =
            apply_pipeline(&rows, &[PipelineStage::slice(2, 3)], OutputMode::Entries).unwrap();
        assert_eq!(sliced.rows.len(), 2);
        assert!(!sliced.complete);
        let all =
            apply_pipeline(&rows, &[PipelineStage::slice(1, 3)], OutputMode::Entries).unwrap();
        assert!(all.complete);
        let mut blank = PipelineStage::filter("x");
        blank.query = Some("   ".to_string());
        assert!(apply_pipeline(&rows, &[blank], OutputMode::Entries).is_err());
        let mut missing = PipelineStage::limit(1);
        missing.lines = None;
        assert!(apply_pipeline(&rows, &[missing], OutputMode::Entries).is_err());
        let mut unknown = PipelineStage::count();
        unknown.operation = "shuffle".to_string();
        assert!(apply_pipeline(&rows, &[unknown], OutputMode::Entries).is_err());
    }
}
