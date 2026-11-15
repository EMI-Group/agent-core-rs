//! Case folding, glob matching, and literal match ranking.
//!
//! Ports the small matching helpers Python's `directory_toolkit` uses for `name`/`path`
//! search and for the `filter` pipeline stage: `str.casefold()`,
//! `fnmatch.fnmatchcase`, and `_literal_match_rank`.
//!
//! The glob matcher is hand rolled (Python's `fnmatch.translate` builds a regular
//! expression, and this crate has no `regex` dependency). It supports the syntax the
//! Python path relies on — `*`, `?`, `[set]`, `[!set]`, `[^set]`, `a-z` ranges, and a
//! backslash escape inside a set — and treats an unterminated `[` as a literal `[`
//! exactly like `fnmatch.translate` does.
//!
//! Deviation: Python's `str.casefold()` performs full case folding (`ß` → `ss`,
//! `İ` → `i̇`); the port uses [`str::to_lowercase`], which maps `ß` to `ß` and `İ` to
//! `i̇`-without-the-combining-dot. Every case-insensitive comparison in this module (and
//! in the view, render, and validate layers) uses the same folding, so the deviation is
//! internally consistent.

/// Folds `value` for case-insensitive comparison.
///
/// This is [`str::to_lowercase`], the port's stand-in for Python's `str.casefold()`.
///
/// ```
/// use openjiuwen_symphony::discovery::directory_toolkit::matcher::case_fold;
///
/// assert_eq!(case_fold("PDF Export"), "pdf export");
/// ```
#[must_use]
pub fn case_fold(value: &str) -> String {
    value.to_lowercase()
}

/// Reports whether `value` contains a glob metacharacter (`*`, `?`, or `[`).
///
/// Python's search decides between "match this pattern" and "match `*query*`" with the
/// same test.
#[must_use]
pub fn has_wildcard(value: &str) -> bool {
    value
        .chars()
        .any(|character| matches!(character, '*' | '?' | '['))
}

/// Reports whether `candidate` matches `pattern` under Python's `fnmatch.fnmatchcase`.
///
/// `*` matches any run of characters (including `/`), `?` matches exactly one
/// character, `[set]` matches one character in `set`, `[!set]`/`[^set]` matches one
/// character outside it, and `a-z` inside a set is an inclusive range. Matching is
/// case-sensitive; callers fold both sides first for a case-insensitive match.
///
/// ```
/// use openjiuwen_symphony::discovery::directory_toolkit::matcher::glob_match;
///
/// assert!(glob_match("pdf-export", "pdf-*"));
/// assert!(glob_match("Export", "E?port"));
/// assert!(!glob_match("pdf", "[!p]*"));
/// ```
#[must_use]
pub fn glob_match(candidate: &str, pattern: &str) -> bool {
    let text: Vec<char> = candidate.chars().collect();
    let glob: Vec<char> = pattern.chars().collect();
    let width = text.len() + 1;
    // `table[pattern_index][text_index]` is true when the pattern suffix matches the
    // text suffix; the table is filled bottom-up so no recursion (and no stack depth
    // proportional to the input) is needed.
    let mut table = vec![vec![false; width]; glob.len() + 1];
    table[glob.len()][text.len()] = true;
    for pattern_index in (0..glob.len()).rev() {
        for text_index in (0..=text.len()).rev() {
            let rest = text_index < text.len() && table[pattern_index + 1][text_index + 1];
            table[pattern_index][text_index] = match glob[pattern_index] {
                '*' => {
                    table[pattern_index + 1][text_index]
                        || (text_index < text.len() && table[pattern_index][text_index + 1])
                }
                '?' => rest,
                '[' => match class_end(&glob, pattern_index) {
                    Some(end) => {
                        text_index < text.len()
                            && class_matches(&glob, pattern_index, end, text[text_index])
                            && table[end + 1][text_index + 1]
                    }
                    None => {
                        text_index < text.len()
                            && text[text_index] == '['
                            && table[pattern_index + 1][text_index + 1]
                    }
                },
                literal => rest && text[text_index] == literal,
            };
        }
    }
    table[0][0]
}

/// Returns the index of the `]` closing the set opened at `start`, or `None`.
///
/// Mirrors `fnmatch.translate`: a leading `!`/`^` negation marker and a `]` that appears
/// before the closing bracket are both part of the set body.
fn class_end(glob: &[char], start: usize) -> Option<usize> {
    let mut index = start + 1;
    if index < glob.len() && matches!(glob[index], '!' | '^') {
        index += 1;
    }
    if index < glob.len() && glob[index] == ']' {
        index += 1;
    }
    while index < glob.len() {
        if glob[index] == ']' {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Reports whether `character` is a member of the set delimited by `start`..`end`.
fn class_matches(glob: &[char], start: usize, end: usize, character: char) -> bool {
    let mut index = start + 1;
    let mut negated = false;
    if index < end && matches!(glob[index], '!' | '^') {
        negated = true;
        index += 1;
    }
    let mut matched = false;
    while index < end {
        let mut low = glob[index];
        if low == '\\' && index + 1 < end {
            index += 1;
            low = glob[index];
        }
        if index + 2 < end && glob[index + 1] == '-' {
            let mut high = glob[index + 2];
            let mut width = 3;
            if high == '\\' && index + 3 < end {
                high = glob[index + 3];
                width = 4;
            }
            if low <= character && character <= high {
                matched = true;
            }
            index += width;
            continue;
        }
        if low == character {
            matched = true;
        }
        index += 1;
    }
    matched != negated
}

/// Ranks how directly `candidate` matches the literal `query`.
///
/// `0` is an exact match, `1` is a prefix match, and `2` is any other (substring)
/// match — the ordering Python's `_literal_match_rank` returns.
///
/// ```
/// use openjiuwen_symphony::discovery::directory_toolkit::matcher::literal_match_rank;
///
/// assert_eq!(literal_match_rank("pdf", "pdf"), 0);
/// assert_eq!(literal_match_rank("pdf-export", "pdf"), 1);
/// assert_eq!(literal_match_rank("export-pdf", "pdf"), 2);
/// ```
#[must_use]
pub fn literal_match_rank(candidate: &str, query: &str) -> usize {
    if candidate == query {
        0
    } else if candidate.starts_with(query) {
        1
    } else {
        2
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn case_folding_lowercases_and_keeps_non_ascii() {
        assert_eq!(case_fold("PDF Export"), "pdf export");
        assert_eq!(case_fold("日本語"), "日本語");
        assert_eq!(case_fold(""), "");
    }

    #[test]
    fn wildcards_are_detected() {
        assert!(has_wildcard("*.md"));
        assert!(has_wildcard("a?"));
        assert!(has_wildcard("[abc]"));
        assert!(!has_wildcard("plain"));
    }

    #[test]
    fn glob_matches_literals_stars_and_questions() {
        assert!(glob_match("pdf-export", "pdf-export"));
        assert!(!glob_match("pdf-export", "pdf"));
        assert!(glob_match("pdf-export", "pdf*"));
        assert!(glob_match("pdf-export", "*export"));
        assert!(glob_match("pdf-export", "*"));
        assert!(glob_match("", "*"));
        assert!(!glob_match("", "?"));
        assert!(glob_match("a", "?"));
        assert!(glob_match("a/b", "a*"));
        assert!(glob_match("Export", "E?port"));
    }

    #[test]
    fn glob_matches_sets_ranges_and_negation() {
        assert!(glob_match("a", "[abc]"));
        assert!(!glob_match("d", "[abc]"));
        assert!(glob_match("d", "[!abc]"));
        assert!(glob_match("d", "[^abc]"));
        assert!(glob_match("m", "[a-z]"));
        assert!(!glob_match("M", "[a-z]"));
        assert!(glob_match("]", "[]]"));
        assert!(glob_match("-", "[a-]"));
        assert!(glob_match("x", "[x]"));
    }

    #[test]
    fn an_unterminated_set_is_a_literal_bracket() {
        assert!(glob_match("[abc", "[abc"));
        assert!(!glob_match("a", "[abc"));
    }

    #[test]
    fn literal_rank_orders_exact_prefix_and_substring() {
        assert_eq!(literal_match_rank("pdf", "pdf"), 0);
        assert_eq!(literal_match_rank("pdf-export", "pdf"), 1);
        assert_eq!(literal_match_rank("export-pdf", "pdf"), 2);
        assert_eq!(literal_match_rank("", ""), 0);
    }
}
