//! Strict `{{name}}` placeholder rendering for skill instruction templates.
//!
//! Python's `str.format` raises on a missing key, and the skill creator surfaces
//! that as a validation failure rather than shipping a half-rendered document.
//! This module reproduces that strictness for the `{{name}}` syntax used by
//! skill templates, with two deliberate rules:
//!
//! - Rendering is a **single pass**: an inserted value is never re-scanned, so a
//!   variable whose value contains `{{other}}` is copied verbatim.
//! - Empty or whitespace-only placeholders (`{{}}`, `{{   }}`) are not
//!   placeholders at all; they stay literal instead of failing.
//!
//! Rendering never panics: every failure is an
//! [`Error::validation`](openjiuwen_common::error::Error::validation).

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};

/// One piece of a scanned template.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece<'a> {
    /// Text copied verbatim, including unresolvable brace runs.
    Literal(&'a str),
    /// A `{{name}}` placeholder whose trimmed name is non-empty.
    Placeholder(&'a str),
}

/// Splits `template` into literal text and placeholders.
///
/// An opening `{{` with no closing `}}` after it is literal text; so is a
/// placeholder whose trimmed name is empty, and an opening `{{` whose candidate
/// name contains another `{{`.
fn pieces(template: &str) -> Vec<Piece<'_>> {
    let mut result = Vec::new();
    let mut position = 0usize;
    while let Some(offset) = template[position..].find("{{") {
        let open = position + offset;
        let Some(close_offset) = template[open + 2..].find("}}") else {
            break;
        };
        let close = open + 2 + close_offset;
        if template[open + 2..close].contains("{{") {
            // A nested opening brace means the outer braces are literal text.
            result.push(Piece::Literal(&template[position..open + 2]));
            position = open + 2;
            continue;
        }
        if open > position {
            result.push(Piece::Literal(&template[position..open]));
        }
        let name = template[open + 2..close].trim();
        if name.is_empty() {
            result.push(Piece::Literal(&template[open..close + 2]));
        } else {
            result.push(Piece::Placeholder(name));
        }
        position = close + 2;
    }
    if position < template.len() {
        result.push(Piece::Literal(&template[position..]));
    }
    result
}

/// Renders `template`, replacing every `{{name}}` with its variable value.
///
/// Names are trimmed before lookup, so `{{ name }}` and `{{name}}` are the same
/// placeholder. Inserted values are not re-scanned.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
/// listing every placeholder name that has no entry in `variables`, in order of
/// first appearance.
///
/// # Examples
///
/// ```
/// # use std::collections::BTreeMap;
/// # use openjiuwen_dev_tools::skill_creator::template::render;
/// let variables = BTreeMap::from([("topic".to_owned(), "rust".to_owned())]);
/// assert_eq!(
///     render("Write about {{ topic }}.", &variables).unwrap(),
///     "Write about rust."
/// );
/// ```
pub fn render(template: &str, variables: &BTreeMap<String, String>) -> Result<String> {
    let mut rendered = String::with_capacity(template.len());
    let mut missing: Vec<&str> = Vec::new();
    for piece in pieces(template) {
        match piece {
            Piece::Literal(text) => rendered.push_str(text),
            Piece::Placeholder(name) => match variables.get(name) {
                Some(value) => rendered.push_str(value),
                None => {
                    if !missing.contains(&name) {
                        missing.push(name);
                    }
                }
            },
        }
    }
    if missing.is_empty() {
        Ok(rendered)
    } else {
        Err(Error::validation(format!(
            "unknown template variable(s): {}",
            missing.join(", ")
        )))
    }
}

/// Lists the placeholder names `template` uses, deduplicated in order of first
/// appearance.
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::skill_creator::template::placeholders;
/// assert_eq!(placeholders("{{a}} {{b}} {{a}}"), vec!["a", "b"]);
/// ```
#[must_use]
pub fn placeholders(template: &str) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for piece in pieces(template) {
        if let Piece::Placeholder(name) = piece {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variables(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn renders_every_placeholder() {
        let template = "Hello {{name}}, welcome to {{place}}!";
        let rendered =
            render(template, &variables(&[("name", "Ada"), ("place", "Rust")])).expect("renders");
        assert_eq!(rendered, "Hello Ada, welcome to Rust!");
    }

    #[test]
    fn missing_variable_lists_its_name() {
        let error = render("{{alpha}} and {{beta}}", &variables(&[("beta", "b")]))
            .expect_err("missing variable");
        let message = error.to_string();
        assert!(message.contains("alpha"), "{message}");
        assert!(!message.contains("beta,"), "{message}");
        assert!(message.starts_with("validation:"), "{message}");
    }

    #[test]
    fn whitespace_inside_braces_is_trimmed() {
        let rendered = render("{{  name  }}", &variables(&[("name", "ok")])).expect("renders");
        assert_eq!(rendered, "ok");
    }

    #[test]
    fn empty_placeholders_stay_literal() {
        let rendered = render("a {{}} b {{   }} c", &variables(&[])).expect("renders");
        assert_eq!(rendered, "a {{}} b {{   }} c");
    }

    #[test]
    fn rendering_is_single_pass() {
        let rendered = render(
            "{{outer}} and {{inner}}",
            &variables(&[("outer", "{{inner}}"), ("inner", "done")]),
        )
        .expect("renders");
        assert_eq!(rendered, "{{inner}} and done");
    }

    #[test]
    fn unterminated_opening_braces_are_literal() {
        let rendered =
            render("prefix {{name and {{name}}", &variables(&[("name", "x")])).expect("renders");
        assert_eq!(rendered, "prefix {{name and x");
    }

    #[test]
    fn unicode_and_newlines_are_preserved() {
        let rendered = render(
            "第一行 {{名}}\n第二行 {{value}}",
            &variables(&[("名", "汉字"), ("value", "值")]),
        )
        .expect("renders");
        assert_eq!(rendered, "第一行 汉字\n第二行 值");
    }

    #[test]
    fn template_without_placeholders_is_unchanged() {
        let template = "plain text with } braces { and no placeholders";
        assert_eq!(
            render(template, &variables(&[])).expect("renders"),
            template
        );
        assert_eq!(placeholders(template).len(), 0);
    }

    #[test]
    fn placeholders_lists_names_in_order() {
        assert_eq!(
            placeholders("{{a}} text {{b}} {{a}} {{  c  }} {{}}"),
            vec!["a", "b", "c"]
        );
    }
}
