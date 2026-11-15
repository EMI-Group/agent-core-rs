//! The tolerant selection parser for a model reply.
//!
//! Ports `_parse_selected_codes` and its helpers from Python
//! `openjiuwen.symphony.retrieval.search.runtime.render.disclosure` (v0.1.20). The parser
//! accepts every reply shape the reference tolerates and never fails: an empty output, an
//! unknown output and garbage all yield an empty vector.
//!
//! # Cascade
//!
//! Every line is normalized first: surrounding whitespace and backticks are stripped, then
//! one list marker (`1.`, `1)`, `1:`, `-`, `*`, each followed by whitespace). The remaining
//! text is tried against the mode-independent *code-handle* shapes, in this order:
//!
//! 1. JSON: an object (or array of objects) whose `id`, `c` or `code` field names a code;
//! 2. the whole line as an exact code;
//! 3. a `[XX]` bracket prefix;
//! 4. one or more `[id: XX]` handles anywhere in the line;
//! 5. a whitespace/comma/semicolon separated list in which *every* token is a known code;
//! 6. a `Candidate XX`, `id: XX`, `c: XX` or `code: XX` prefix;
//! 7. a prefix match against the known codes, longest code first, requiring a delimiter
//!    (whitespace, `|`, `:`, `-`, `,`, `;`, `]`, `}`, `)`, `"`, `'`) or the end of the line
//!    after the code.
//!
//! Names mode (compact codes disabled) additionally accepts, in order: a case-insensitive
//! code match, a unique label match, a unique `camelCase`-token overlap against
//! `display_name`/`label`, a plain digit string (first as a matching numeric code, then as a
//! 1-based position among the resolutions), and a multi-token list of codes. Under compact
//! codes those label and position fallbacks are deliberately not applied.
//!
//! # Joined-code fallback
//!
//! Only when nothing at all parsed, compact codes are enabled, the whole output has no
//! newline and no space, its length is a multiple of a non-zero code width and every chunk
//! of that width is a known code, the output is read as one concatenated code list. This
//! recovers replies such as `ABAC` for codes `AB`, `AC`.
//!
//! # Ordering
//!
//! The model's line order is preserved. A resolution is emitted once: duplicates are
//! dropped by `canonical_id`, keeping the first occurrence (Python's `seen` set).

use std::collections::BTreeSet;

use serde_json::Value;

use crate::retrieval::search::flow::{ExposedFragment, SelectableResolution};

/// The delimiters accepted after a code in the prefix-match strategy.
const CODE_DELIMITERS: [char; 11] = [' ', '\t', '|', ':', '-', ',', ';', ']', '}', ')', '"'];

/// The characters that separate the tokens of a code list.
const CODE_LIST_SEPARATORS: [char; 4] = [',', ';', ' ', '\t'];

/// The JSON field names a compact reply may carry a code in.
const CODE_FIELDS: [&str; 3] = ["id", "c", "code"];

/// The prefixes a reply may put in front of a code.
const CODE_PREFIXES: [&str; 4] = ["candidate", "id", "c", "code"];

/// Parses `output` into the resolutions of `fragment` it selects.
///
/// The result preserves the model's line order and holds each `canonical_id` at most once
/// (the first occurrence wins). Empty or unparseable output yields an empty vector; the
/// function never fails and never panics.
#[must_use]
pub fn parse_selected_codes(fragment: &ExposedFragment, output: &str) -> Vec<SelectableResolution> {
    let mut resolved: Vec<SelectableResolution> = Vec::new();
    if output.trim().is_empty() || fragment.code_to_resolution.is_empty() {
        return resolved;
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for raw_line in output.lines() {
        let line = normalize_line(raw_line);
        if line.is_empty() {
            continue;
        }
        push_entries(
            fragment,
            parse_line(fragment, line),
            &mut resolved,
            &mut seen,
        );
    }
    if resolved.is_empty() && fragment.compact_codes_enabled {
        if let Some(entries) = parse_joined_codes(fragment, output.trim()) {
            push_entries(fragment, entries, &mut resolved, &mut seen);
        }
    }
    resolved
}

/// Parses one normalized line into `(code, strategy)` entries.
fn parse_line(fragment: &ExposedFragment, line: &str) -> Vec<(String, &'static str)> {
    let mut entries: Vec<(String, &'static str)> = Vec::new();
    entries.extend(parse_code_handles(fragment, line));
    if !fragment.compact_codes_enabled {
        entries.extend(parse_name_fallbacks(fragment, line));
    }
    entries
}

/// The mode-independent code-handle shapes.
fn parse_code_handles(fragment: &ExposedFragment, line: &str) -> Vec<(String, &'static str)> {
    if let Some(entries) = parse_json_codes(fragment, line) {
        return entries;
    }
    if fragment.code_to_resolution.contains_key(line) {
        return vec![(line.to_owned(), "exact")];
    }
    if let Some(entries) = parse_bracket_code(fragment, line) {
        return entries;
    }
    if let Some(entries) = parse_id_handles(fragment, line) {
        return entries;
    }
    if let Some(entries) = parse_code_list(fragment, line) {
        return entries;
    }
    if let Some(entries) = parse_prefixed_code(fragment, line) {
        return entries;
    }
    parse_code_prefix(fragment, line).unwrap_or_default()
}

/// Strategy 1: JSON objects carrying `id`, `c` or `code`.
fn parse_json_codes(fragment: &ExposedFragment, line: &str) -> Option<Vec<(String, &'static str)>> {
    let trimmed = line.trim();
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    let mut codes: Vec<String> = Vec::new();
    collect_json_codes(&value, &mut codes);
    if codes.is_empty() {
        return None;
    }
    Some(
        codes
            .into_iter()
            .filter(|code| fragment.code_to_resolution.contains_key(code))
            .map(|code| (code, "json"))
            .collect::<Vec<(String, &'static str)>>(),
    )
}

/// Collects the codes a JSON value names, recursing into arrays.
fn collect_json_codes(value: &Value, codes: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_json_codes(item, codes);
            }
        }
        Value::Object(object) => {
            for field in CODE_FIELDS {
                let named = match object.get(field) {
                    Some(Value::String(text)) => Some(text.trim().to_owned()),
                    Some(Value::Number(number)) => Some(number.to_string()),
                    _ => None,
                };
                if let Some(code) = named {
                    if !code.is_empty() {
                        codes.push(code);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Strategy 3: a `[XX]` bracket prefix.
fn parse_bracket_code(
    fragment: &ExposedFragment,
    line: &str,
) -> Option<Vec<(String, &'static str)>> {
    let rest = line.strip_prefix('[')?;
    let close = rest.find(']')?;
    let inner = rest[..close].trim();
    if !inner.is_empty() && fragment.code_to_resolution.contains_key(inner) {
        return Some(vec![(inner.to_owned(), "bracket_code")]);
    }
    None
}

/// Strategy 4: one or more `[id: XX]` handles.
fn parse_id_handles(fragment: &ExposedFragment, line: &str) -> Option<Vec<(String, &'static str)>> {
    let mut codes: Vec<(String, &'static str)> = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("[id:") {
        let tail = &rest[start + "[id:".len()..];
        let Some(close) = tail.find(']') else {
            break;
        };
        let inner = tail[..close].trim();
        if !inner.is_empty() && fragment.code_to_resolution.contains_key(inner) {
            codes.push((inner.to_owned(), "id_handle"));
        }
        rest = &tail[close..];
    }
    if codes.is_empty() {
        None
    } else {
        Some(codes)
    }
}

/// Strategy 5: a list whose every token is a known code.
fn parse_code_list(fragment: &ExposedFragment, line: &str) -> Option<Vec<(String, &'static str)>> {
    let tokens: Vec<&str> = line
        .split(CODE_LIST_SEPARATORS)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        return None;
    }
    if !tokens
        .iter()
        .all(|token| fragment.code_to_resolution.contains_key(*token))
    {
        return None;
    }
    Some(
        tokens
            .into_iter()
            .map(|token| (token.to_owned(), "code_list"))
            .collect(),
    )
}

/// Strategy 6: a `Candidate XX` / `id: XX` / `c: XX` / `code: XX` prefix.
fn parse_prefixed_code(
    fragment: &ExposedFragment,
    line: &str,
) -> Option<Vec<(String, &'static str)>> {
    for prefix in CODE_PREFIXES {
        let Some(head) = line.get(..prefix.len()) else {
            continue;
        };
        if !head.eq_ignore_ascii_case(prefix) {
            continue;
        }
        let rest = line[prefix.len()..].trim_start();
        let rest = rest.strip_prefix([':', '=']).map_or(rest, str::trim_start);
        let code: String = rest
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        if !code.is_empty() && fragment.code_to_resolution.contains_key(&code) {
            return Some(vec![(code, "prefixed")]);
        }
    }
    None
}

/// Strategy 7: a longest-code-first prefix match followed by a delimiter.
fn parse_code_prefix(
    fragment: &ExposedFragment,
    line: &str,
) -> Option<Vec<(String, &'static str)>> {
    let mut codes: Vec<&String> = fragment.code_to_resolution.keys().collect();
    codes.sort_by(|left, right| {
        right
            .chars()
            .count()
            .cmp(&left.chars().count())
            .then_with(|| left.cmp(right))
    });
    for code in codes {
        let Some(rest) = line.strip_prefix(code.as_str()) else {
            continue;
        };
        let accepted = rest.is_empty()
            || rest
                .chars()
                .next()
                .is_some_and(|character| CODE_DELIMITERS.contains(&character) || character == '\'');
        if accepted {
            return Some(vec![(code.clone(), "code_prefix")]);
        }
    }
    None
}

/// The names-mode fallbacks.
fn parse_name_fallbacks(fragment: &ExposedFragment, line: &str) -> Vec<(String, &'static str)> {
    if let Some(code) = unique_case_insensitive_code(fragment, line) {
        return vec![(code, "case_insensitive")];
    }
    if let Some(code) = unique_label_code(fragment, line) {
        return vec![(code, "label")];
    }
    if let Some(code) = unique_token_code(fragment, line) {
        return vec![(code, "camel_tokens")];
    }
    if let Some(entries) = parse_digit_code(fragment, line) {
        return entries;
    }
    Vec::new()
}

/// A code that only differs from the line by case.
fn unique_case_insensitive_code(fragment: &ExposedFragment, line: &str) -> Option<String> {
    let lowered = line.to_lowercase();
    let matches: Vec<&String> = fragment
        .code_to_resolution
        .keys()
        .filter(|code| code.to_lowercase() == lowered)
        .collect();
    if matches.len() == 1 {
        Some(matches[0].clone())
    } else {
        None
    }
}

/// A code whose label (or display name) equals the line.
fn unique_label_code(fragment: &ExposedFragment, line: &str) -> Option<String> {
    let lowered = line.to_lowercase();
    let mut matches: Vec<&String> = Vec::new();
    for (code, resolution) in &fragment.code_to_resolution {
        let label_matches = resolution.label.trim().to_lowercase() == lowered;
        let name_matches = resolution.display_name.trim().to_lowercase() == lowered;
        if label_matches || name_matches {
            matches.push(code);
        }
    }
    if matches.len() == 1 {
        Some(matches[0].clone())
    } else {
        None
    }
}

/// A code whose display name and label share every significant `camelCase` token of the
/// line.
fn unique_token_code(fragment: &ExposedFragment, line: &str) -> Option<String> {
    let wanted: BTreeSet<String> = camel_tokens(line)
        .into_iter()
        .filter(|token| token.chars().count() > 2)
        .collect();
    if wanted.is_empty() {
        return None;
    }
    let mut matches: Vec<&String> = Vec::new();
    for (code, resolution) in &fragment.code_to_resolution {
        let mut available: BTreeSet<String> =
            camel_tokens(&resolution.display_name).into_iter().collect();
        available.extend(camel_tokens(&resolution.label));
        if wanted.is_subset(&available) {
            matches.push(code);
        }
    }
    if matches.len() == 1 {
        Some(matches[0].clone())
    } else {
        None
    }
}

/// A plain digit string: a numeric code first, then a 1-based position.
fn parse_digit_code(fragment: &ExposedFragment, line: &str) -> Option<Vec<(String, &'static str)>> {
    if line.is_empty() || !line.chars().all(|character| character.is_ascii_digit()) {
        return None;
    }
    if fragment.code_to_resolution.contains_key(line) {
        return Some(vec![(line.to_owned(), "numeric_code")]);
    }
    let position: usize = line.parse().ok()?;
    if position == 0 {
        return None;
    }
    let mut codes: Vec<&String> = fragment.code_to_resolution.keys().collect();
    codes.sort();
    codes
        .get(position - 1)
        .map(|code| vec![((*code).clone(), "position")])
}

/// The joined-code fallback (`ABAC` for codes `AB`, `AC`).
fn parse_joined_codes(
    fragment: &ExposedFragment,
    output: &str,
) -> Option<Vec<(String, &'static str)>> {
    let width = fragment.code_width;
    if width == 0 || output.contains('\n') || output.contains(' ') {
        return None;
    }
    let characters: Vec<char> = output.chars().collect();
    if characters.is_empty() || characters.len() % width != 0 {
        return None;
    }
    let mut codes: Vec<(String, &'static str)> = Vec::new();
    for chunk in characters.chunks(width) {
        let code: String = chunk.iter().collect();
        if !fragment.code_to_resolution.contains_key(&code) {
            return None;
        }
        codes.push((code, "joined_codes"));
    }
    if codes.is_empty() {
        None
    } else {
        Some(codes)
    }
}

/// Resolves `entries` into `resolved`, dropping unknown and already-seen codes.
fn push_entries(
    fragment: &ExposedFragment,
    entries: Vec<(String, &'static str)>,
    resolved: &mut Vec<SelectableResolution>,
    seen: &mut BTreeSet<String>,
) {
    for (code, strategy) in entries {
        if let Some(resolution) = fragment.code_to_resolution.get(&code) {
            if seen.insert(resolution.canonical_id.clone()) {
                resolved.push(resolution.clone());
            }
        } else {
            tracing::debug!(
                %code,
                strategy,
                "the reply named a code that this exposure does not carry",
            );
        }
    }
}

/// Strips whitespace, backticks and one list marker from a reply line.
fn normalize_line(raw: &str) -> &str {
    let line = raw.trim().trim_matches('`');
    strip_list_marker(line)
}

/// Strips one `1.` / `1)` / `1:` / `-` / `*` marker, each followed by whitespace.
fn strip_list_marker(line: &str) -> &str {
    let trimmed = line.trim_start();
    let digits = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if digits > 0 {
        let rest = &trimmed[digits..];
        if let Some(after) = rest.strip_prefix(['.', ')', ':']) {
            if let Some(stripped) = strip_leading_whitespace(after) {
                return stripped;
            }
        }
    }
    if let Some(after) = trimmed.strip_prefix(['-', '*']) {
        if let Some(stripped) = strip_leading_whitespace(after) {
            return stripped;
        }
    }
    trimmed
}

/// Returns `text` without its leading whitespace, or `None` when it has none.
fn strip_leading_whitespace(text: &str) -> Option<&str> {
    let stripped = text.trim_start();
    if stripped.len() == text.len() {
        None
    } else {
        Some(stripped)
    }
}

/// Splits a name into `camelCase` tokens, lowercased.
///
/// Mirrors the reference's `[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+` tokenizer: a lowercase run
/// with at most one leading uppercase, a run of uppercase letters that is followed by an
/// uppercase letter or the end of the name, or a digit run.
fn camel_tokens(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character.is_ascii_uppercase() {
            if characters
                .get(index + 1)
                .is_some_and(char::is_ascii_lowercase)
            {
                let mut end = index + 1;
                while characters.get(end).is_some_and(char::is_ascii_lowercase) {
                    end += 1;
                }
                tokens.push(characters[index..end].iter().collect());
                index = end;
                continue;
            }
            let mut end = index;
            while characters.get(end).is_some_and(char::is_ascii_uppercase) {
                end += 1;
            }
            // A trailing uppercase that starts a new word belongs to the next token.
            if end < characters.len() && characters[end].is_ascii_lowercase() && end - index > 1 {
                end -= 1;
            }
            tokens.push(characters[index..end].iter().collect());
            index = end;
            continue;
        }
        if character.is_ascii_lowercase() || character.is_ascii_digit() {
            let mut end = index;
            while characters
                .get(end)
                .is_some_and(|next| next.is_ascii_lowercase() == character.is_ascii_lowercase())
            {
                end += 1;
            }
            tokens.push(characters[index..end].iter().collect());
            index = end;
            continue;
        }
        index += 1;
    }
    tokens
        .into_iter()
        .map(|token| token.to_lowercase())
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::retrieval::search::flow::{ExposedFragment, SelectableResolution};
    use std::collections::BTreeMap;

    /// `(code, display_name, label)` triples, with `canonical_id = "custom.<code>"`.
    fn build(compact: bool, code_width: usize, entries: &[(&str, &str, &str)]) -> ExposedFragment {
        let mut code_to_resolution: BTreeMap<String, SelectableResolution> = BTreeMap::new();
        let mut candidate_codes: Vec<String> = Vec::new();
        for (code, display_name, label) in entries {
            candidate_codes.push((*code).to_owned());
            code_to_resolution.insert(
                (*code).to_owned(),
                SelectableResolution {
                    code: (*code).to_owned(),
                    canonical_id: format!("custom.{code}"),
                    display_name: (*display_name).to_owned(),
                    label: (*label).to_owned(),
                    description: String::new(),
                    is_terminal: true,
                    branch_path: vec!["ROOT".to_owned()],
                    node_id: None,
                    item: None,
                },
            );
        }
        ExposedFragment {
            rendered_tree: String::new(),
            code_to_resolution,
            candidate_codes,
            fragment_fingerprint: "0123456789abcdef".to_owned(),
            code_width,
            compact_codes_enabled: compact,
            flat_list_mode: false,
            ..ExposedFragment::default()
        }
    }

    fn compact_fragment() -> ExposedFragment {
        build(
            true,
            2,
            &[
                ("AA", "Summary", "Summary"),
                ("AB", "Translate", "Translate"),
                ("AC", "Search", "Search"),
            ],
        )
    }

    fn names_fragment() -> ExposedFragment {
        build(
            false,
            0,
            &[
                ("AlphaGo", "AlphaGo", "Alpha Go"),
                ("BetaTest", "BetaTest", "Beta Test"),
                ("Gamma", "Gamma", "Gamma"),
            ],
        )
    }

    fn codes(fragment: &ExposedFragment, output: &str) -> Vec<String> {
        parse_selected_codes(fragment, output)
            .into_iter()
            .map(|resolution| resolution.canonical_id)
            .collect()
    }

    #[test]
    fn empty_and_garbage_output_parse_to_nothing() {
        let fragment = compact_fragment();
        assert!(parse_selected_codes(&fragment, "").is_empty());
        assert!(parse_selected_codes(&fragment, "   \n\t\n").is_empty());
        assert!(parse_selected_codes(&fragment, "hello there").is_empty());
        assert!(parse_selected_codes(&fragment, "ZZ\nYY").is_empty());
        assert!(parse_selected_codes(&fragment, "{\"id\": \"ZZ\"}").is_empty());
        assert!(parse_selected_codes(&ExposedFragment::default(), "AA").is_empty());
    }

    #[test]
    fn an_exact_code_and_its_list_markers_parse() {
        let fragment = compact_fragment();
        assert_eq!(codes(&fragment, "AB"), vec!["custom.AB"]);
        assert_eq!(codes(&fragment, "1. AB"), vec!["custom.AB"]);
        assert_eq!(codes(&fragment, "AB\nAC"), vec!["custom.AB", "custom.AC"]);
        assert_eq!(codes(&fragment, "- AC"), vec!["custom.AC"]);
        assert_eq!(codes(&fragment, "* AA"), vec!["custom.AA"]);
        assert_eq!(codes(&fragment, "2) AC"), vec!["custom.AC"]);
        assert_eq!(codes(&fragment, "3: AA"), vec!["custom.AA"]);
        assert_eq!(codes(&fragment, "`AB`"), vec!["custom.AB"]);
    }

    #[test]
    fn a_copied_candidate_line_parses() {
        let fragment = compact_fragment();
        assert_eq!(
            codes(&fragment, "- Summary [id: AA]: Summarize long documents"),
            vec!["custom.AA"]
        );
        assert_eq!(
            codes(&fragment, "Summary [id: AB]: Translate"),
            vec!["custom.AB"]
        );
        assert_eq!(
            codes(&fragment, "[id: AC] Search\n[id: AA] Summary"),
            vec!["custom.AC", "custom.AA"]
        );
    }

    #[test]
    fn json_codes_parse() {
        let fragment = compact_fragment();
        assert_eq!(codes(&fragment, r#"{"id": "AB"}"#), vec!["custom.AB"]);
        assert_eq!(
            codes(&fragment, r#"{"c": "AC", "name": "x"}"#),
            vec!["custom.AC"]
        );
        assert_eq!(
            codes(&fragment, r#"{"code": "AA", "score": 1}"#),
            vec!["custom.AA"]
        );
        assert_eq!(
            codes(&fragment, r#"[{"id": "aa"}, {"id": "AC"}]"#),
            vec!["custom.AC"]
        );
        assert_eq!(
            codes(&fragment, r#"{"id": 7, "name": "no code"}"#),
            Vec::<String>::new()
        );
    }

    #[test]
    fn bracket_and_prefixed_codes_parse() {
        let fragment = compact_fragment();
        assert_eq!(codes(&fragment, "[AB] Translate"), vec!["custom.AB"]);
        assert_eq!(codes(&fragment, "Candidate AC"), vec!["custom.AC"]);
        assert_eq!(codes(&fragment, "id: AA"), vec!["custom.AA"]);
        assert_eq!(codes(&fragment, "c = AB"), vec!["custom.AB"]);
        assert_eq!(codes(&fragment, "code:AC"), vec!["custom.AC"]);
    }

    #[test]
    fn code_lists_and_prefix_matches_parse() {
        let fragment = compact_fragment();
        assert_eq!(codes(&fragment, "AA, AB"), vec!["custom.AA", "custom.AB"]);
        assert_eq!(codes(&fragment, "AA AC"), vec!["custom.AA", "custom.AC"]);
        assert_eq!(
            codes(&fragment, "AA; AB; AC"),
            vec!["custom.AA", "custom.AB", "custom.AC"]
        );
        // A list with one unknown token is not a code list, but the prefix match still
        // recovers the leading code.
        assert_eq!(codes(&fragment, "AA nope"), vec!["custom.AA"]);
        assert_eq!(codes(&fragment, "AB: Translate"), vec!["custom.AB"]);
        assert_eq!(codes(&fragment, "AB|other"), vec!["custom.AB"]);
        assert!(!codes(&fragment, "ABC").contains(&"custom.AB".to_owned()));
    }

    #[test]
    fn joined_codes_recover_when_nothing_else_parsed() {
        let fragment = compact_fragment();
        assert_eq!(codes(&fragment, "ABAC"), vec!["custom.AB", "custom.AC"]);
        assert_eq!(codes(&fragment, "AA"), vec!["custom.AA"]);
        // A chunk that is not a code disqualifies the whole output.
        assert!(codes(&fragment, "ABZZ").is_empty());
        // An odd length cannot be chunked.
        assert!(codes(&fragment, "ABA").is_empty());
        // Mixed widths disable the fallback.
        let mixed = build(true, 0, &[("A", "A", "A")]);
        assert!(codes(&mixed, "AA").is_empty());
    }

    #[test]
    fn names_mode_accepts_case_labels_tokens_and_digits() {
        let fragment = names_fragment();
        assert_eq!(codes(&fragment, "AlphaGo"), vec!["custom.AlphaGo"]);
        assert_eq!(codes(&fragment, "alphago"), vec!["custom.AlphaGo"]);
        assert_eq!(codes(&fragment, "Beta Test"), vec!["custom.BetaTest"]);
        assert_eq!(codes(&fragment, "BetaTest"), vec!["custom.BetaTest"]);
        assert_eq!(codes(&fragment, "Gamma"), vec!["custom.Gamma"]);
        // Plain digits: a matching numeric code first, then a 1-based position.
        assert_eq!(codes(&fragment, "2"), vec!["custom.BetaTest"]);
        assert_eq!(codes(&fragment, "3"), vec!["custom.Gamma"]);
        assert!(codes(&fragment, "4").is_empty());
        assert!(codes(&fragment, "0").is_empty());
        // A copied names-mode line.
        assert_eq!(
            codes(&fragment, "- Candidate Gamma: some description"),
            vec!["custom.Gamma"]
        );
        // Multi-token code lists.
        assert_eq!(
            codes(&fragment, "AlphaGo, Gamma"),
            vec!["custom.AlphaGo", "custom.Gamma"]
        );
    }

    #[test]
    fn names_mode_digit_matches_a_numeric_code_first() {
        let numeric = build(false, 1, &[("1", "One", "One"), ("2", "Two", "Two")]);
        assert_eq!(codes(&numeric, "2"), vec!["custom.2"]);
        assert_eq!(codes(&numeric, "1"), vec!["custom.1"]);
    }

    #[test]
    fn compact_mode_never_applies_the_names_fallbacks() {
        let fragment = build(true, 2, &[("AA", "Summary", "Summary")]);
        assert!(codes(&fragment, "summary").is_empty());
        assert!(codes(&fragment, "1").is_empty());
    }

    #[test]
    fn duplicates_keep_the_first_occurrence_and_order_is_preserved() {
        let fragment = compact_fragment();
        assert_eq!(
            codes(&fragment, "AC\nAA\nAC\nAB"),
            vec!["custom.AC", "custom.AA", "custom.AB"]
        );
        assert_eq!(
            codes(&fragment, "Summary [id: AA]: x\n[id: AA] again"),
            vec!["custom.AA"]
        );
    }

    #[test]
    fn the_parser_returns_resolutions_not_just_ids() {
        let fragment = compact_fragment();
        let parsed = parse_selected_codes(&fragment, "AB");
        assert_eq!(parsed.len(), 1);
        let resolution = &parsed[0];
        assert_eq!(resolution.code, "AB");
        assert_eq!(resolution.display_name, "Translate");
        assert!(resolution.is_terminal);
        assert_eq!(resolution, fragment.resolution("AB").unwrap());
    }

    #[test]
    fn camel_tokens_mirror_the_reference_tokenizer() {
        assert_eq!(camel_tokens("SkillsAlpha"), vec!["skills", "alpha"]);
        assert_eq!(camel_tokens("ABCdef"), vec!["ab", "cdef"]);
        assert_eq!(camel_tokens("mySkill2"), vec!["my", "skill", "2"]);
        assert_eq!(camel_tokens("能力树"), Vec::<String>::new());
        assert_eq!(camel_tokens("a-b_c"), vec!["a", "b", "c"]);
    }
}
