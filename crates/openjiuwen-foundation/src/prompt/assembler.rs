//! Private placeholder scanning and substitution engine for [`crate::prompt::template`].
//!
//! Ports the string handling of `openjiuwen/core/foundation/prompt/assemble/variables/`
//! (`textable.py`, `dictable.py`) without a regular expression: Python compiles
//! `re.escape(prefix) + "([^{}]*?)" + re.escape(suffix)`, whose behavior is
//! reproduced by a hand-written left-to-right scan over byte indices.
//!
//! The contract (see `prompt/CONTEXT.md`, "Design Decisions"):
//! - A placeholder is `prefix + name + suffix`; `name` is whitespace-trimmed and
//!   may not contain `{` or `}`. Empty names (`{{}}`) and unmatched delimiters
//!   stay literal text, exactly as in the Python implementation.
//! - Rendering is single-pass: literal text is copied verbatim and substituted
//!   values are appended, never re-scanned, so a value containing `{{x}}` is
//!   emitted literally (no injection cascade).
//! - A placeholder that cannot be resolved (missing head key, missing
//!   intermediate object key, non-object intermediate, or `null` leaf) keeps its
//!   original text verbatim — including interior whitespace — and logs a warning
//!   naming the placeholder only, never its value.

use std::collections::BTreeMap;

use serde_json::Value;

/// One placeholder occurrence found in a string.
///
/// `start`/`end` are byte offsets of the whole occurrence (delimiters included),
/// so `&text[start..end]` is the original placeholder text and both offsets are
/// `char` boundaries. `name` is the trimmed name as written.
pub(crate) struct Placeholder<'a> {
    /// Byte offset of the opening delimiter.
    pub start: usize,
    /// Byte offset one past the closing delimiter.
    pub end: usize,
    /// Trimmed placeholder name (without delimiters).
    pub name: &'a str,
}

/// One `tracing` target for every event emitted by this module.
const LOG_TARGET: &str = "openjiuwen_foundation::prompt::assembler";

/// Length in bytes of the first `char` of `s` (1 for empty input).
///
/// Used to advance a rejected match by a whole character so the cursor always
/// lands on a `char` boundary, even for multi-byte delimiters.
fn first_char_len(s: &str) -> usize {
    s.chars().next().map_or(1, char::len_utf8)
}

/// Scan `text` left to right and return every placeholder occurrence in order.
///
/// The scan mirrors the Python non-greedy regex `prefix([^{}]*?)suffix`:
/// - the earliest possible opening delimiter wins;
/// - the name ends at the *first* suffix at or after the name start;
/// - an empty name or a name containing `{`/`}` rejects that match only, and the
///   search resumes at the next character after the opening delimiter rather
///   than after the whole rejected match.
///
/// That recovery is what makes overlapping input behave naturally: `"{{a{{b}}"`
/// yields the inner placeholder at `{{b}}` (rendered as `"{{a"` + the value), and
/// `"{{{b}}"` yields `{{b}}` starting at byte 1.
///
/// Empty delimiters are treated as "no placeholders": with no prefix every
/// position would match, and with no suffix no non-empty name can be delimited.
/// The scan always terminates and never panics.
pub(crate) fn scan<'a>(text: &'a str, prefix: &str, suffix: &str) -> Vec<Placeholder<'a>> {
    let mut found = Vec::new();
    if prefix.is_empty() || suffix.is_empty() {
        return found;
    }

    let mut cursor = 0;
    while cursor < text.len() {
        let Some(offset) = text[cursor..].find(prefix) else {
            break;
        };
        let start = cursor + offset;
        let name_start = start + prefix.len();
        // A missing suffix after the name start cannot exist for any later
        // prefix either, because both searches would begin at or after this
        // position, so the whole scan is done.
        let Some(offset) = text[name_start..].find(suffix) else {
            break;
        };
        let suffix_start = name_start + offset;
        let name = &text[name_start..suffix_start];

        if name.trim().is_empty() || name.contains('{') || name.contains('}') {
            // Rejected match: retry from the next character so that a valid
            // placeholder nested inside the rejected span is still found.
            cursor = start + first_char_len(&text[start..]);
            continue;
        }

        let end = suffix_start + suffix.len();
        found.push(Placeholder {
            start,
            end,
            name: name.trim(),
        });
        cursor = end;
    }

    found
}

/// The distinct placeholder names of `text`, in first-seen order.
pub(crate) fn placeholders(text: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for placeholder in scan(text, prefix, suffix) {
        if !names.iter().any(|known| known == placeholder.name) {
            names.push(placeholder.name.to_string());
        }
    }
    names
}

/// The distinct head segments of the placeholder names of `text`, in first-seen order.
///
/// The head is the part before the first `.`, so `"user.name"` contributes
/// `"user"`: that is the key a caller must supply to fill the placeholder.
pub(crate) fn input_keys(text: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for name in placeholders(text, prefix, suffix) {
        let head = name.split('.').next().unwrap_or(name.as_str()).to_string();
        if !keys.contains(&head) {
            keys.push(head);
        }
    }
    keys
}

/// Render `text`, substituting every resolvable placeholder from `values`.
///
/// Single pass: literal segments are copied verbatim, substituted values are
/// appended as-is and never re-scanned. Unresolvable placeholders are kept
/// verbatim (including interior whitespace) and reported with a `warn` event
/// that names the placeholder but never logs a value.
pub(crate) fn render(
    text: &str,
    values: &BTreeMap<String, Value>,
    prefix: &str,
    suffix: &str,
) -> String {
    let found = scan(text, prefix, suffix);
    if found.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for placeholder in found {
        out.push_str(&text[cursor..placeholder.start]);
        if let Some(value) = resolve(values, placeholder.name) {
            out.push_str(&stringify(value));
        } else {
            tracing::warn!(
                target: LOG_TARGET,
                placeholder = placeholder.name,
                "placeholder could not be resolved; keeping it verbatim"
            );
            out.push_str(&text[placeholder.start..placeholder.end]);
        }
        cursor = placeholder.end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Resolve a dotted `path` (`"a"`, `"a.b.c"`) through nested JSON objects.
///
/// Returns `Some` only for a present, non-`null` leaf. A missing head key, a
/// missing intermediate key, a non-object intermediate (array or scalar), and a
/// `null` leaf all return `None`. An empty path is a miss.
pub(crate) fn resolve<'a>(values: &'a BTreeMap<String, Value>, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }

    let mut segments = path.split('.');
    let mut current = values.get(segments.next()?)?;
    for segment in segments {
        current = current.as_object()?.get(segment)?;
    }

    if current.is_null() {
        None
    } else {
        Some(current)
    }
}

/// Render a resolved JSON value as template text.
///
/// `String` values render as their raw text (no JSON quotes); numbers, booleans,
/// arrays, and objects render as compact JSON. `null` never reaches a template
/// (it is treated as unresolved) but renders as `"null"` if called directly.
pub(crate) fn stringify(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "null".to_string(),
        // `Display` for `Value` is the compact JSON text, which is what numbers,
        // booleans, arrays, and objects render as.
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! Ports the template-rendering behaviors of
    //! `tests/unit_tests/core/foundation/prompt/` (the `TextableVariable` and
    //! `DictableVariable` cases), with the crate's documented deviations:
    //! unresolvable dotted paths and `None` values keep the placeholder verbatim
    //! instead of raising, and non-string values render as JSON rather than
    //! Python `str()`.

    use super::*;
    use serde_json::json;

    const PREFIX: &str = "{{";
    const SUFFIX: &str = "}}";

    fn map(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    fn fill(text: &str, pairs: &[(&str, Value)]) -> String {
        render(text, &map(pairs), PREFIX, SUFFIX)
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(fill("hello world", &[]), "hello world");
        assert_eq!(fill("", &[("a", json!(1))]), "");
    }

    #[test]
    fn single_placeholder_is_replaced() {
        assert_eq!(fill("hi {{name}}!", &[("name", json!("bob"))]), "hi bob!");
    }

    #[test]
    fn repeated_placeholder_is_replaced_everywhere() {
        assert_eq!(fill("{{x}}-{{x}}-{{x}}", &[("x", json!("a"))]), "a-a-a");
    }

    #[test]
    fn partially_formatted_template_keeps_unresolved_placeholder_verbatim() {
        assert_eq!(fill("{{a}} and {{b}}", &[("a", json!(1))]), "1 and {{b}}");
        assert_eq!(
            fill("{{ user.name }}", &[]),
            "{{ user.name }}",
            "interior whitespace must survive an unresolved placeholder"
        );
    }

    #[test]
    fn extra_values_are_ignored() {
        assert_eq!(fill("{{a}}", &[("a", json!(1)), ("b", json!(2))]), "1");
        assert_eq!(
            placeholders("no slots here", PREFIX, SUFFIX),
            Vec::<String>::new()
        );
    }

    #[test]
    fn custom_angle_delimiters() {
        assert_eq!(
            render("hi <<name>>", &map(&[("name", json!("bob"))]), "<<", ">>"),
            "hi bob"
        );
        assert_eq!(placeholders("<<a>> <<b>>", "<<", ">>"), vec!["a", "b"]);
    }

    #[test]
    fn shell_style_delimiters() {
        assert_eq!(
            render(
                "${HOME}$/bin",
                &map(&[("HOME", json!("/root"))]),
                "${",
                "}$"
            ),
            "/root/bin"
        );
        // The suffix must actually appear: `${HOME}/bin` has no `}$` and stays literal.
        assert_eq!(
            render("${HOME}/bin", &map(&[("HOME", json!("/root"))]), "${", "}$"),
            "${HOME}/bin"
        );
        // The default delimiters are inert under the custom ones.
        assert_eq!(
            render("{{a}} ${b}$", &map(&[("b", json!(2))]), "${", "}$"),
            "{{a}} 2"
        );
    }

    #[test]
    fn single_char_braces_delimiters() {
        assert_eq!(
            render(
                "{a} {b}",
                &map(&[("a", json!(1)), ("b", json!(2))]),
                "{",
                "}"
            ),
            "1 2"
        );
        // `{{a}}` under single-brace delimiters: the outer `{` is literal because
        // the inner name would contain a brace.
        assert_eq!(render("{{a}}", &map(&[("a", json!(1))]), "{", "}"), "{1}");
    }

    #[test]
    fn whitespace_inside_delimiters_is_trimmed() {
        assert_eq!(fill("{{ name }}", &[("name", json!("bob"))]), "bob");
        assert_eq!(fill("{{\tname\n}}", &[("name", json!("bob"))]), "bob");
    }

    #[test]
    fn empty_placeholder_is_literal() {
        assert_eq!(fill("a{{}}b", &[("", json!("x"))]), "a{{}}b");
        assert_eq!(fill("{{ }}", &[]), "{{ }}");
        assert!(scan("{{}}", PREFIX, SUFFIX).is_empty());
    }

    #[test]
    fn single_braces_are_literal() {
        assert_eq!(
            fill("{name} and {a.b}", &[("name", json!("bob"))]),
            "{name} and {a.b}"
        );
        assert!(scan("a { b } c", PREFIX, SUFFIX).is_empty());
    }

    #[test]
    fn unterminated_placeholder_is_literal() {
        assert_eq!(fill("{{name", &[("name", json!("bob"))]), "{{name");
        assert_eq!(fill("{{name} }}", &[("name", json!("bob"))]), "{{name} }}");
    }

    #[test]
    fn dotted_path_resolves_through_objects() {
        let values = map(&[("user", json!({"name": "bob", "tags": ["a"]}))]);
        assert_eq!(render("{{user.name}}", &values, PREFIX, SUFFIX), "bob");
        assert_eq!(
            resolve(&values, "user.name").and_then(Value::as_str),
            Some("bob")
        );
    }

    #[test]
    fn missing_head_keeps_placeholder() {
        assert_eq!(fill("{{user.name}}", &[]), "{{user.name}}");
        assert_eq!(
            fill("{{user.name}}", &[("other", json!({"name": "bob"}))]),
            "{{user.name}}"
        );
    }

    #[test]
    fn missing_intermediate_keeps_placeholder() {
        assert_eq!(
            fill("{{user.profile.name}}", &[("user", json!({"name": "bob"}))]),
            "{{user.profile.name}}"
        );
    }

    #[test]
    fn non_object_intermediate_keeps_placeholder() {
        assert_eq!(
            fill("{{user.name}}", &[("user", json!("bob"))]),
            "{{user.name}}"
        );
        assert_eq!(
            fill("{{user.0}}", &[("user", json!(["bob"]))]),
            "{{user.0}}",
            "arrays are not traversed by dotted paths"
        );
    }

    #[test]
    fn null_leaf_keeps_placeholder_and_never_renders_null() {
        assert_eq!(fill("{{a}}", &[("a", Value::Null)]), "{{a}}");
        assert_eq!(
            fill("{{user.name}}", &[("user", json!({"name": null}))]),
            "{{user.name}}"
        );
        assert!(resolve(&map(&[("a", Value::Null)]), "a").is_none());
    }

    #[test]
    fn non_string_values_render_as_documented() {
        assert_eq!(fill("{{n}}", &[("n", json!(42))]), "42");
        assert_eq!(fill("{{n}}", &[("n", json!(-1.5))]), "-1.5");
        assert_eq!(fill("{{b}}", &[("b", json!(true))]), "true");
        assert_eq!(fill("{{b}}", &[("b", json!(false))]), "false");
        assert_eq!(
            fill("{{o}}", &[("o", json!({"k": 1, "s": "v"}))]),
            r#"{"k":1,"s":"v"}"#
        );
        assert_eq!(
            fill("{{a}}", &[("a", json!([1, "two", null]))]),
            r#"[1,"two",null]"#
        );
        assert_eq!(stringify(&Value::Null), "null");
    }

    #[test]
    fn string_values_are_not_json_quoted() {
        assert_eq!(stringify(&json!("a \"b\"")), "a \"b\"");
    }

    #[test]
    fn inserted_values_are_not_re_expanded() {
        assert_eq!(
            fill("{{a}}", &[("a", json!("{{x}}")), ("x", json!("boom"))]),
            "{{x}}"
        );
        assert_eq!(
            fill("pre {{a}} post", &[("a", json!("{{b}}"))]),
            "pre {{b}} post"
        );
    }

    #[test]
    fn empty_string_value_renders_empty() {
        assert_eq!(fill("[{{a}}]", &[("a", json!(""))]), "[]");
    }

    #[test]
    fn placeholders_are_deduped_in_first_seen_order() {
        assert_eq!(
            placeholders("{{b}} {{a}} {{b}} {{a.b}}", PREFIX, SUFFIX),
            vec!["b", "a", "a.b"]
        );
        assert_eq!(
            input_keys("{{b}} {{a}} {{b}} {{a.b}}", PREFIX, SUFFIX),
            vec!["b", "a"]
        );
        assert_eq!(input_keys("plain", PREFIX, SUFFIX), Vec::<String>::new());
    }

    #[test]
    fn scan_reports_positions_and_trimmed_names() {
        let text = "x{{ a }}y";
        let found = scan(text, PREFIX, SUFFIX);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].start, 1);
        assert_eq!(found[0].end, 8);
        assert_eq!(found[0].name, "a");
        assert_eq!(&text[found[0].start..found[0].end], "{{ a }}");
    }

    #[test]
    fn rejected_match_recovers_to_nested_placeholder() {
        // The outer `{{` cannot be closed by the first `}}` because the enclosed
        // name would contain `{`; the inner `{{b}}` is the placeholder.
        assert_eq!(fill("{{a{{b}}", &[("b", json!("B"))]), "{{aB");
        assert_eq!(placeholders("{{a{{b}}", PREFIX, SUFFIX), vec!["b"]);
        // Triple opening delimiter: `{{b}}` starts at byte 1.
        let found = scan("{{{b}}", PREFIX, SUFFIX);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].start, found[0].end), (1, 6));
    }

    #[test]
    fn prefix_equal_to_suffix() {
        assert_eq!(render("a#b#c", &map(&[("b", json!(2))]), "#", "#"), "a2c");
        // `##` is an empty name here, so the text stays literal.
        assert_eq!(
            render("a####c", &map(&[("", json!(9))]), "##", "##"),
            "a####c"
        );
    }

    #[test]
    fn empty_delimiters_find_nothing() {
        assert!(scan("abc", "", "").is_empty());
        assert!(scan("abc}}", "", "}}").is_empty());
        assert!(scan("abc{{", "{{", "").is_empty());
        assert_eq!(
            render("abc{{d}}", &map(&[("d", json!(1))]), "", ""),
            "abc{{d}}"
        );
    }

    #[test]
    fn multibyte_delimiters_and_values() {
        assert_eq!(
            render("«名前»!", &map(&[("名前", json!("太郎"))]), "«", "»"),
            "太郎!"
        );
        // No brace rule rejects `««a»`: the name is the brace-free `«a`, so the
        // outermost match wins (Python regex parity) and stays verbatim.
        assert_eq!(render("««a»", &map(&[("a", json!("A"))]), "«", "»"), "««a»");
        // A rejected match advances by whole characters, never splitting one:
        // the name `{a«b` contains a brace, so the inner `«b»` is the placeholder.
        assert_eq!(
            render("«{a«b»", &map(&[("b", json!("B"))]), "«", "»"),
            "«{aB"
        );
    }

    #[test]
    fn resolve_handles_empty_path_and_nesting() {
        let values = map(&[("a", json!({"b": {"c": 1}}))]);
        assert!(resolve(&values, "").is_none());
        assert_eq!(resolve(&values, "a.b.c"), Some(&json!(1)));
        assert!(resolve(&values, "a.b.d").is_none());
    }
}
