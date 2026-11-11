//! Identifier surgery: wrapper-struct names, constructor names, and the
//! humanized description fallback.
//!
//! All names derive from the annotated function's identifier, never from the
//! `name` attribute: `name` only reaches the model-facing [`ToolCard`] name.

use proc_macro2::Ident;

/// Words that are kept upper-case in a humanized name, like Python's
/// `_humanize_name`.
const ABBREVIATIONS: &[&str] = &[
    "id", "url", "uri", "api", "json", "http", "https", "llm", "mcp", "sql", "uuid", "html", "xml",
    "csv", "pdf", "cli", "ui", "os", "ip", "tcp", "udp", "tls", "ast", "ocr", "rag",
];

/// Returns the identifier text without a leading `r#` marker.
pub(crate) fn bare_ident(ident: &Ident) -> String {
    let text = ident.to_string();
    match text.strip_prefix("r#") {
        Some(bare) => bare.to_owned(),
        None => text,
    }
}

/// Returns the wrapper struct name: `PascalCase(function) + "Tool"`.
pub(crate) fn struct_ident(function: &Ident) -> Ident {
    let pascal = pascal_case(&bare_ident(function));
    Ident::new(&format!("{pascal}Tool"), function.span())
}

/// Returns the constructor name: `{function}_tool`.
pub(crate) fn ctor_ident(function: &Ident) -> Ident {
    Ident::new(&format!("{}_tool", bare_ident(function)), function.span())
}

/// Converts a `snake_case` name to `PascalCase` (`read_write_tool` →
/// `ReadWriteTool`).
fn pascal_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for word in name.split('_').filter(|word| !word.is_empty()) {
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

/// Humanizes an identifier for the model (`read_write_tool` → `read write
/// tool`), keeping abbreviations upper-case (`get_url` → `get URL`).
pub(crate) fn humanized_name(function: &Ident) -> String {
    let name = bare_ident(function);
    name.split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let lowered = word.to_lowercase();
            if ABBREVIATIONS.contains(&lowered.as_str()) {
                lowered.to_uppercase()
            } else {
                lowered
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proc_macro2::Span;

    use super::*;

    #[test]
    fn struct_and_constructor_names_follow_the_function_ident() {
        let function = Ident::new("add", Span::call_site());
        assert_eq!(struct_ident(&function).to_string(), "AddTool");
        assert_eq!(ctor_ident(&function).to_string(), "add_tool");
    }

    #[test]
    fn pascal_case_handles_snake_case_and_runs_of_underscores() {
        assert_eq!(pascal_case("read_write_tool"), "ReadWriteTool");
        assert_eq!(pascal_case("__odd__name_"), "OddName");
        assert_eq!(pascal_case("add"), "Add");
    }

    #[test]
    fn raw_identifiers_lose_their_marker() {
        let function = Ident::new_raw("type", Span::call_site());
        assert_eq!(bare_ident(&function), "type");
        assert_eq!(struct_ident(&function).to_string(), "TypeTool");
        assert_eq!(humanized_name(&function), "type");
    }

    #[test]
    fn humanized_names_lowercase_words_and_keep_abbreviations() {
        let name = |text| humanized_name(&Ident::new(text, Span::call_site()));
        assert_eq!(name("read_write_tool"), "read write tool");
        assert_eq!(name("get_url"), "get URL");
        assert_eq!(name("parse_json_ast"), "parse JSON AST");
        assert_eq!(name("ping"), "ping");
    }
}
