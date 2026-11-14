//! Structure-preserving digest of files read by path.
//!
//! Python's `source_file_compressor.py` deliberately declines (`modified =
//! false`) so the offloader can fall back to a head/tail preview. This port
//! implements the structure-preserving digest instead: comment-only lines and
//! blank-line runs are dropped first, and — only when the result still exceeds
//! the budget — the text is reduced to imports, signatures, headings and the
//! lines that mention a query term, with `[N lines omitted]` markers where text
//! was dropped.
//!
//! Both stages are gated by [`fits_budget_and_saves`], so a file that gains
//! nothing from the digest is returned verbatim. Dropped content stays
//! retrievable when the offloader then offloads the original.
//!
//! [`fits_budget_and_saves`]: crate::processor::rule_compression::common::fits_budget_and_saves
use super::{RuleCompressionResult, RuleCompressor};
use crate::processor::rule_compression::common::{
    file_name, fits_budget_and_saves, meets_savings_ratio,
};
use crate::processor::rule_compression::types::{ContentType, RuleContext};
use serde_json::json;
/// Which comment syntax a file uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommentStyle {
    /// `#`-to-end-of-line (Python, Ruby, shell, YAML, TOML, ...).
    Hash,
    /// `//`, `/* */` and `*` continuation (C-like sources).
    Slash,
    /// `--` (SQL, Lua, Haskell).
    DoubleDash,
    /// `<!-- -->` (HTML, XML, Markdown).
    Html,
    /// `;` (Clojure).
    Semicolon,
    /// Unknown file type: no line is treated as a comment.
    None,
}
/// Extensions whose `#` starts a comment.
const HASH_EXTENSIONS: &[&str] = &[
    ".py",
    ".pyi",
    ".pyx",
    ".rb",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".env",
    ".pl",
    ".pm",
    ".r",
    ".jl",
    ".ex",
    ".exs",
    ".ps1",
    ".psm1",
    ".nix",
    ".cmake",
    ".mk",
    ".bzl",
    ".tf",
    ".hcl",
];
/// Extensions whose `//` starts a comment.
const SLASH_EXTENSIONS: &[&str] = &[
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".go", ".rs", ".java", ".kt",
    ".kts", ".scala", ".groovy", ".gradle", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp",
    ".hxx", ".m", ".mm", ".cs", ".fs", ".vb", ".swift", ".dart", ".php", ".proto", ".sol", ".zig",
    ".nim", ".v", ".css", ".scss", ".sass", ".less", ".styl",
];
/// Extensions whose `--` starts a comment.
const DOUBLE_DASH_EXTENSIONS: &[&str] = &[".sql", ".lua", ".hs"];
/// Extensions whose `<!--` starts a comment.
const HTML_EXTENSIONS: &[&str] = &[
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".xsd",
    ".xsl",
    ".svg",
    ".vue",
    ".svelte",
    ".astro",
    ".jinja",
    ".j2",
    ".hbs",
    ".ejs",
    ".erb",
    ".md",
    ".mdx",
    ".markdown",
    ".rst",
    ".adoc",
];
/// Extensions whose `;` starts a comment.
const SEMICOLON_EXTENSIONS: &[&str] = &[".clj"];
/// The prefixes that make a line an import or include.
const IMPORT_PREFIXES: &[&str] = &[
    "import ",
    "from ",
    "use ",
    "require(",
    "require ",
    "include ",
    "#include",
    "extern crate ",
    "mod ",
    "package ",
    "using ",
    "namespace ",
    "@import",
];
/// The words that make a line a declaration or signature.
const SIGNATURE_WORDS: &[&str] = &[
    "fn ",
    "def ",
    "class ",
    "struct ",
    "enum ",
    "trait ",
    "impl ",
    "interface ",
    "func ",
    "function ",
    "type ",
    "export ",
    "public ",
    "private ",
    "protected ",
    "pub ",
    "async ",
    "module ",
    "object ",
    "record ",
    "abstract ",
];
/// Digests source, markup, config and prose files.
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceFileCompressor;
impl RuleCompressor for SourceFileCompressor {
    fn compress(&self, content: &str, ctx: &RuleContext) -> RuleCompressionResult {
        if content.trim().is_empty() {
            return RuleCompressionResult::unchanged(content, ContentType::SourceFile);
        }
        let style = comment_style(ctx.file_path.as_deref());
        let original_lines = content.split('\n').count();
        let (without_comments, comment_lines, blank_lines) = strip_comments(content, style);
        if fits_budget_and_saves(content, &without_comments, ctx) {
            let details = json!({
                "digest_stage": "comments",
                "original_line_count": original_lines,
                "compressed_line_count": without_comments.split('\n').count(),
                "omitted_comment_lines": comment_lines,
                "omitted_blank_lines": blank_lines,
                "query_term_hits": 0,
            });
            return applied(without_comments, details);
        }

        let (structural, omitted, hits) = keep_structure(&without_comments, ctx);
        if !fits_budget_and_saves(content, &structural, ctx) {
            // The comment pass may still have saved enough on its own.
            if without_comments != content && meets_savings_ratio(content, &without_comments, ctx) {
                let details = json!({
                    "digest_stage": "comments",
                    "original_line_count": original_lines,
                    "compressed_line_count": without_comments.split('\n').count(),
                    "omitted_comment_lines": comment_lines,
                    "omitted_blank_lines": blank_lines,
                    "query_term_hits": 0,
                });
                return applied(without_comments, details);
            }
            return RuleCompressionResult::unchanged(content, ContentType::SourceFile);
        }
        let details = json!({
            "digest_stage": "structure",
            "original_line_count": original_lines,
            "compressed_line_count": structural.split('\n').count(),
            "omitted_comment_lines": comment_lines,
            "omitted_blank_lines": blank_lines,
            "omitted_structure_lines": omitted,
            "query_term_hits": hits,
        });
        applied(structural, details)
    }
}
/// Builds an accepted result with `details`.
fn applied(content: String, details: serde_json::Value) -> RuleCompressionResult {
    RuleCompressionResult::applied(content, ContentType::SourceFile, true, Some(details))
}
/// The comment syntax `file_path` uses, by extension.
fn comment_style(file_path: Option<&str>) -> CommentStyle {
    let Some(path) = file_path else {
        return CommentStyle::None;
    };
    let name = file_name(path).to_lowercase();
    let extension = match name.rfind('.') {
        Some(0) | None => {
            return match name.as_str() {
                "makefile" | "gnumakefile" | "rakefile" | "gemfile" | "podfile" | "justfile" => {
                    CommentStyle::Hash
                }
                _ => CommentStyle::None,
            };
        }
        Some(index) => &name[index..],
    };
    if HASH_EXTENSIONS.contains(&extension) {
        CommentStyle::Hash
    } else if SLASH_EXTENSIONS.contains(&extension) {
        CommentStyle::Slash
    } else if DOUBLE_DASH_EXTENSIONS.contains(&extension) {
        CommentStyle::DoubleDash
    } else if HTML_EXTENSIONS.contains(&extension) {
        CommentStyle::Html
    } else if SEMICOLON_EXTENSIONS.contains(&extension) {
        CommentStyle::Semicolon
    } else {
        CommentStyle::None
    }
}
/// Drops comment-only lines and collapses blank runs.
///
/// Returns the rewritten text, the number of comment lines dropped and the
/// number of blank lines dropped.
fn strip_comments(content: &str, style: CommentStyle) -> (String, usize, usize) {
    if style == CommentStyle::None {
        return (content.to_owned(), 0, 0);
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut comment_lines = 0_usize;
    let mut blank_lines = 0_usize;
    let mut in_block = false;
    for line in content.split('\n') {
        let trimmed = line.trim_start();
        let is_comment = if in_block {
            if trimmed.contains("-->") || trimmed.contains("*/") {
                in_block = false;
            }
            true
        } else {
            classify_comment(trimmed, style, &mut in_block)
        };
        if is_comment {
            comment_lines += 1;
            continue;
        }
        if trimmed.is_empty() {
            if kept.last().is_some_and(|last| last.trim().is_empty()) {
                blank_lines += 1;
                continue;
            }
            kept.push("");
            continue;
        }
        kept.push(line);
    }
    let text = kept.join("\n").trim().to_owned();
    (text, comment_lines, blank_lines)
}
/// Whether `trimmed` opens or is an entire comment line for `style`.
fn classify_comment(trimmed: &str, style: CommentStyle, in_block: &mut bool) -> bool {
    match style {
        CommentStyle::Hash => trimmed.starts_with('#'),
        CommentStyle::DoubleDash => trimmed.starts_with("--"),
        CommentStyle::Semicolon => trimmed.starts_with(';'),
        CommentStyle::Slash => {
            if trimmed.starts_with("/*") {
                *in_block = !trimmed.contains("*/");
                return true;
            }
            trimmed.starts_with("//") || trimmed.starts_with('*')
        }
        CommentStyle::Html => {
            if trimmed.contains("<!--") {
                *in_block = !trimmed.contains("-->");
                return true;
            }
            trimmed.starts_with("-->")
        }
        CommentStyle::None => false,
    }
}
/// Keeps imports, signatures, headings and query-term hits.
///
/// Returns the digest, the number of content lines dropped and the number of
/// structural lines that mentioned a query term.
fn keep_structure(content: &str, ctx: &RuleContext) -> (String, usize, usize) {
    let mut kept: Vec<String> = Vec::new();
    let mut dropped_run = 0_usize;
    let mut dropped_total = 0_usize;
    let mut hits = 0_usize;
    for (index, line) in content.split('\n').enumerate() {
        let trimmed = line.trim();
        let mentions = mentions_query_term(line, ctx);
        if mentions {
            hits += 1;
        }
        let interesting = index == 0 || is_structural(trimmed) || mentions;
        if interesting {
            if dropped_run > 0 {
                kept.push(format!("[{dropped_run} lines omitted]"));
                dropped_run = 0;
            }
            kept.push(line.to_owned());
        } else if !trimmed.is_empty() {
            dropped_run += 1;
            dropped_total += 1;
        }
    }
    if dropped_run > 0 {
        kept.push(format!("[{dropped_run} lines omitted]"));
    }
    let text = kept.join("\n");
    let text = head_tail(text.trim(), ctx);
    (text, dropped_total, hits)
}
/// Trims `text` to the head and tail budgets when it is still too long.
fn head_tail(text: &str, ctx: &RuleContext) -> String {
    if ctx.token_count(text) <= ctx.max_tokens {
        return text.to_owned();
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let mut head: Vec<String> = Vec::new();
    let mut used = 0_usize;
    for line in &lines {
        used += ctx.token_count(line);
        if used > ctx.head_tokens.max(1) {
            break;
        }
        head.push((*line).to_owned());
    }
    let mut tail: Vec<String> = Vec::new();
    let mut used = 0_usize;
    for line in lines.iter().rev() {
        used += ctx.token_count(line);
        if used > ctx.tail_tokens.max(1) {
            break;
        }
        tail.push((*line).to_owned());
    }
    tail.reverse();
    let omitted = lines.len().saturating_sub(head.len() + tail.len());
    head.push(format!("[{omitted} lines omitted]"));
    head.extend(tail);
    head.join("\n")
}
/// Whether `trimmed` is an import, a signature or a heading.
fn is_structural(trimmed: &str) -> bool {
    if trimmed.starts_with('#') && !trimmed.starts_with("#[") {
        // Markdown heading; a Rust attribute is not a heading and `.rs` uses the
        // slash comment style anyway.
        return true;
    }
    if is_underline(trimmed) {
        return true;
    }
    let lowered = trimmed.to_lowercase();
    IMPORT_PREFIXES
        .iter()
        .any(|prefix| lowered.starts_with(prefix))
        || SIGNATURE_WORDS.iter().any(|word| {
            crate::processor::rule_compression::common::find_word(&lowered, word).is_some()
        })
}
/// Whether `trimmed` is a setext/RST heading underline.
fn is_underline(trimmed: &str) -> bool {
    let mut seen = false;
    for value in trimmed.chars() {
        match value {
            '=' | '~' | '^' | '-' => seen = true,
            _ => return false,
        }
    }
    seen && trimmed.chars().count() >= 3
}
/// Whether `line` mentions one of the query terms.
fn mentions_query_term(line: &str, ctx: &RuleContext) -> bool {
    if ctx.query_terms.is_empty() {
        return false;
    }
    let lowered = line.to_lowercase();
    ctx.query_terms.iter().any(|term| lowered.contains(term))
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    fn ctx_for(path: &str) -> RuleContext {
        RuleContext {
            file_path: Some(path.to_owned()),
            max_tokens: 60,
            ..RuleContext::new(60)
        }
    }
    #[test]
    fn comments_and_blank_runs_are_dropped_first() {
        let source = "\
// header comment
/// doc comment
fn main() {
    let value = 1;


    println!(\"{value}\");
}
// trailing
";
        let ctx = ctx_for("src/main.rs");
        let result = SourceFileCompressor.compress(source, &ctx);
        assert!(result.modified);
        assert!(!result.content.contains("header comment"));
        assert!(result.content.contains("fn main()"));
        assert!(result.content.contains("println!"));
        assert_eq!(result.content_type, ContentType::SourceFile);
    }
    #[test]
    fn structure_pass_keeps_signatures_and_query_hits() {
        let mut source = String::from("use std::io;\n\nfn payout_handler() {\n");
        for index in 0..80 {
            let _ = writeln!(source, "    let noise_{index} = {index};");
        }
        source.push_str("    audit_the_payout();\n}\n");
        let mut ctx = ctx_for("src/payout.rs");
        ctx.max_tokens = 60;
        ctx.query_terms = ["payout".to_owned()].into_iter().collect();
        let result = SourceFileCompressor.compress(&source, &ctx);
        assert!(result.modified);
        assert!(result.content.contains("use std::io;"));
        assert!(result.content.contains("fn payout_handler() {"));
        assert!(result.content.contains("lines omitted"));
        assert_eq!(result.content_type, ContentType::SourceFile);
        assert!(result.details.is_some());
    }
    #[test]
    fn unknown_file_types_are_declined() {
        let result = SourceFileCompressor.compress("plain text\nlines\n", &RuleContext::new(100));
        assert!(!result.modified);
    }
}
