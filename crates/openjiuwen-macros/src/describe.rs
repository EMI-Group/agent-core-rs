//! Doc-comment → model-facing description.
//!
//! Ports Python's `CallableSchemaExtractor.extract_function_description`: take
//! the first paragraph of the doc comment, collapse it to one line, and stop at
//! a blank line, a rustdoc heading, or a section marker such as `Args:`.

use syn::{Attribute, Expr, ExprLit, Lit, Meta};

/// Returns the first doc-comment paragraph as a single line, or [`None`] when
/// the doc comment has none.
pub(crate) fn description_from_docs(attrs: &[Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for attr in attrs {
        let Some(text) = doc_text(attr) else {
            continue;
        };
        for line in text.split('\n') {
            let line = line.strip_prefix(' ').unwrap_or(line);
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || is_section_marker(trimmed) {
                return finish(&lines);
            }
            lines.push(trimmed.to_owned());
        }
    }
    finish(&lines)
}

/// Returns the string value of a `#[doc = "..."]` attribute.
fn doc_text(attr: &Attribute) -> Option<String> {
    if !attr.path().is_ident("doc") {
        return None;
    }
    let Meta::NameValue(name_value) = &attr.meta else {
        return None;
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Str(text),
        ..
    }) = &name_value.value
    else {
        return None;
    };
    Some(text.value())
}

/// Joins the collected lines, or [`None`] when nothing survived.
fn finish(lines: &[String]) -> Option<String> {
    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" ").trim().to_owned())
    }
}

/// Matches Python's `^\s*\w+\s*:` section marker (`Args:`, `Returns:`,
/// `Examples:`, ...).
fn is_section_marker(line: &str) -> bool {
    let mut chars = line.chars().peekable();
    let mut saw_word = false;
    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '_' {
            saw_word = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_word {
        return false;
    }
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
    chars.peek() == Some(&':')
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use syn::parse_quote;

    use super::*;

    fn describe(attrs: &[Attribute]) -> Option<String> {
        description_from_docs(attrs)
    }

    #[test]
    fn first_paragraph_is_collapsed_to_one_line() {
        let attrs: Vec<Attribute> = vec![
            parse_quote!(#[doc = " Add two integers."]),
            parse_quote!(#[doc = " Returns the sum."]),
        ];
        assert_eq!(
            describe(&attrs).as_deref(),
            Some("Add two integers. Returns the sum.")
        );
    }

    #[test]
    fn blank_line_ends_the_description() {
        let attrs: Vec<Attribute> = vec![
            parse_quote!(#[doc = " Echo text."]),
            parse_quote!(#[doc = ""]),
            parse_quote!(#[doc = " Args:"]),
            parse_quote!(#[doc = "     text: the text"]),
        ];
        assert_eq!(describe(&attrs).as_deref(), Some("Echo text."));
    }

    #[test]
    fn headings_and_section_markers_end_the_description() {
        let heading: Vec<Attribute> = vec![parse_quote!(#[doc = " # Panics"])];
        assert_eq!(describe(&heading), None);
        let section: Vec<Attribute> = vec![
            parse_quote!(#[doc = " Read a file."]),
            parse_quote!(#[doc = " Returns:"]),
        ];
        assert_eq!(describe(&section).as_deref(), Some("Read a file."));
    }

    #[test]
    fn no_doc_comment_yields_none() {
        assert_eq!(describe(&[]), None);
    }
}
