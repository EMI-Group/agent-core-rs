//! The dependency-free readability extractor behind [`super::HtmlCompressor`].
//!
//! Parsing, tag stripping, metadata reading, entity decoding and block rendering live here; the compressor above only gates and renders the report.
//! Two passes decide what the model sees:
//!
//! 1. the renderer emits one markdown block per heading, paragraph, list item,
//!    table and code fence (`script`/`style`/`nav`/`footer`/… subtrees are gone
//!    before rendering, which is the ladder's secondary filter),
//! 2. [`super::ladder`] scores those blocks and keeps the highest-scoring
//!    contiguous run, so a link menu, a sidebar or a repeated footer line is
//!    dropped even when no tag marks it as chrome.
//!
//! A selection that keeps fewer than
//! [`MIN_REGION_VISIBLE_CHARS`](super::ladder::MIN_REGION_VISIBLE_CHARS) visible
//! characters falls back to the whole document (`ladder_fallback`), and an
//! extraction that would come out empty falls back to the plain-text strip, so a
//! page with text never extracts to nothing.
//!
//! Report-shaped HTML is pruned before the ladder runs: see [`super::report`].
use super::ladder::select_blocks;
use super::report::{looks_like_report_html, prune_report_rows};
use std::collections::{BTreeMap, BTreeSet};

/// Elements whose whole subtree is discarded.
const NOISE_TAGS: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form", "iframe",
    "head",
];
/// What [`HTMLExtractor`] renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HtmlOutputFormat {
    /// Markdown with headings, lists, links, code fences and tables.
    Markdown,
    /// The same text with the markdown markers removed.
    Text,
}
/// Knobs of [`HTMLExtractor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HTMLExtractorConfig {
    /// Markdown or plain text.
    pub output_format: HtmlOutputFormat,
    /// Render `<a href>` as `[label](href)`.
    pub include_links: bool,
    /// Render `<table>` as a markdown table.
    pub include_tables: bool,
    /// Keep `<!-- … -->` comments in the output.
    pub include_comments: bool,
    /// Keep `#` headings and backtick code spans.
    pub include_formatting: bool,
    /// Read `<title>` and the `meta` tags.
    pub extract_metadata: bool,
}
impl Default for HTMLExtractorConfig {
    fn default() -> Self {
        Self {
            output_format: HtmlOutputFormat::Markdown,
            include_links: true,
            include_tables: true,
            include_comments: false,
            include_formatting: true,
            extract_metadata: true,
        }
    }
}
/// What one extraction produced.
#[derive(Debug, Clone, PartialEq)]
pub struct HTMLExtractionResult {
    /// The extracted markdown/text.
    pub extracted: String,
    /// Length of the input HTML.
    pub original_length: usize,
    /// Length of [`HTMLExtractionResult::extracted`].
    pub extracted_length: usize,
    /// `extracted_length / original_length`.
    pub compression_ratio: f64,
    /// `<title>` or `og:title`, when present.
    pub title: Option<String>,
    /// `meta[name=author]`, when present.
    pub author: Option<String>,
    /// `meta[name=date]` or `article:published_time`, when present.
    pub date: Option<String>,
    /// Every `meta` pair that was read.
    pub metadata: BTreeMap<String, String>,
    /// Elements dropped as markup noise.
    pub removed_node_count: usize,
    /// `<a>` elements kept.
    pub link_count: usize,
    /// `<table>` elements kept.
    pub table_count: usize,
    /// `<pre>` blocks kept.
    pub code_block_count: usize,
    /// Rendered blocks kept.
    pub output_block_count: usize,
    /// Blocks dropped as duplicates.
    pub duplicate_block_count: usize,
    /// Report rows dropped.
    pub report_rows_removed: usize,
    /// Blocks the readability ladder kept.
    pub selected_block_count: usize,
    /// Blocks the ladder dropped as page chrome.
    pub boilerplate_block_count: usize,
    /// The score of the ladder's selected run.
    pub best_region_score: f64,
    /// Why the ladder kept the whole document, when it did.
    pub ladder_fallback: Option<String>,
}
impl HTMLExtractionResult {
    /// `(1 - compression_ratio) * 100`.
    #[must_use]
    pub fn reduction_percent(&self) -> f64 {
        (1.0 - self.compression_ratio) * 100.0
    }
}
/// Reads the readable text out of an HTML document.
#[derive(Debug, Clone, Copy, Default)]
pub struct HTMLExtractor {
    /// Extraction knobs.
    pub config: HTMLExtractorConfig,
}
impl HTMLExtractor {
    /// An extractor with `config`.
    #[must_use]
    pub fn new(config: HTMLExtractorConfig) -> Self {
        Self { config }
    }
    /// Extracts `html` into markdown or plain text.
    #[must_use]
    pub fn extract(&self, html: &str) -> HTMLExtractionResult {
        self.extract_with_report(html, looks_like_report_html(html))
    }
    /// Extracts `html`, treating it as a report when `report` is set.
    #[must_use]
    pub fn extract_with_report(&self, html: &str, report: bool) -> HTMLExtractionResult {
        let metadata = if self.config.extract_metadata {
            read_metadata(html)
        } else {
            BTreeMap::new()
        };
        let title = metadata
            .get("title")
            .cloned()
            .filter(|value| !value.is_empty());
        let mut noise: Vec<&str> = NOISE_TAGS.to_vec();
        if !self.config.include_tables {
            noise.push("table");
        }
        let mut removed_node_count = 0_usize;
        let mut body = if self.config.include_comments {
            html.to_owned()
        } else {
            let (stripped, removed) = remove_comments(html);
            removed_node_count += removed;
            stripped
        };
        let (stripped, removed) = remove_tag_subtrees(&body, &noise);
        removed_node_count += removed;
        body = stripped;

        let mut renderer = Renderer::new(self.config);
        renderer.render(&body);
        let mut blocks = renderer.finish();
        let mut report_rows_removed = 0_usize;
        if report {
            report_rows_removed = prune_report_rows(&mut blocks);
        }
        let ladder = select_blocks(&blocks);
        let mut ladder_fallback = ladder.fallback.map(str::to_owned);
        let mut boilerplate_block_count = ladder.dropped;
        let mut selected_block_count = ladder.blocks.len();
        let best_region_score = ladder.best_run_score;
        let mut outcome = self.render_blocks(ladder.blocks, title.as_deref());
        if outcome.extracted.is_empty() && !blocks.is_empty() {
            // A page that carried text must never extract to nothing: strip the
            // whole document and say that the ladder was bypassed.
            let whole = self.render_blocks(blocks, title.as_deref());
            if !whole.extracted.is_empty() {
                ladder_fallback = Some("empty_selection".to_owned());
                boilerplate_block_count = 0;
                selected_block_count = whole.blocks.len();
                outcome = whole;
            }
        }
        let original_length = html.chars().count();
        let extracted_length = outcome.extracted.chars().count();
        HTMLExtractionResult {
            compression_ratio: extracted_length as f64 / original_length.max(1) as f64,
            extracted: outcome.extracted,
            original_length,
            extracted_length,
            title,
            author: metadata.get("author").cloned(),
            date: metadata.get("date").cloned(),
            metadata,
            removed_node_count,
            link_count: renderer.link_count,
            table_count: renderer.table_count,
            code_block_count: renderer.code_block_count,
            duplicate_block_count: outcome.duplicate_block_count,
            output_block_count: outcome.blocks.len(),
            report_rows_removed,
            selected_block_count,
            boilerplate_block_count,
            best_region_score,
            ladder_fallback,
        }
    }

    /// Inserts the title heading, deduplicates and renders `blocks`.
    fn render_blocks(self, mut blocks: Vec<String>, title: Option<&str>) -> RenderOutcome {
        if let Some(title) = title {
            if !blocks.iter().any(|block| block_headline(block, title)) {
                blocks.insert(0, format!("# {title}"));
            }
        }
        let (blocks, duplicate_block_count) = deduplicate_blocks(blocks);
        let joined = blocks.join("\n\n");
        let extracted = match self.config.output_format {
            HtmlOutputFormat::Markdown => joined,
            HtmlOutputFormat::Text => plain_text(&joined),
        }
        .trim()
        .to_owned();
        RenderOutcome {
            extracted,
            blocks,
            duplicate_block_count,
        }
    }
}

/// The rendered output of one block list.
struct RenderOutcome {
    /// The rendered markdown or plain text.
    extracted: String,
    /// The blocks the rendering kept, title heading included.
    blocks: Vec<String>,
    /// Blocks dropped as duplicates.
    duplicate_block_count: usize,
}

/// Drops `<!-- … -->` comments.
fn remove_comments(html: &str) -> (String, usize) {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let mut removed = 0_usize;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("-->") {
            rest = &rest[start + end + 3..];
        } else {
            rest = "";
        }
        removed += 1;
    }
    out.push_str(rest);
    (out, removed)
}
/// Drops the whole subtree of every element named in `tags`.
fn remove_tag_subtrees(html: &str, tags: &[&str]) -> (String, usize) {
    let mut current = html.to_owned();
    let mut removed = 0_usize;
    for tag in tags {
        while let Some((start, open_end, self_closing)) = find_open_tag(&current, tag) {
            let end = if self_closing {
                open_end
            } else {
                match find_close_tag(&current, tag, open_end) {
                    Some(close_end) => close_end,
                    None => current.len(),
                }
            };
            current.replace_range(start..end, "");
            removed += 1;
        }
    }
    (current, removed)
}
/// Finds the first `needle` in `haystack` at or after `from`, ignoring ASCII case.
///
/// The scan runs over the bytes of the original string instead of a lower-cased
/// copy, so every offset it returns can slice `haystack` directly. That is the
/// whole point: [`str::to_lowercase`] can change the byte length of a character
/// (`İ` U+0130 becomes `i̇`, `ẞ` becomes `ß`), which shifts every later offset in
/// the copy away from the original.
///
/// A match always starts on a char boundary, because `needle` starts with an
/// ASCII byte and a byte holding an ASCII value can never be a UTF-8
/// continuation byte.
fn find_ascii_case_insensitive(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    let last_start = haystack.len().checked_sub(needle.len())?;
    (from..=last_start)
        .find(|start| haystack[*start..*start + needle.len()].eq_ignore_ascii_case(needle))
}
/// Finds the next `<tag …>` and reports its span and whether it self-closes.
fn find_open_tag(html: &str, tag: &str) -> Option<(usize, usize, bool)> {
    let needle = format!("<{tag}");
    let mut from = 0;
    while let Some(start) = find_ascii_case_insensitive(html, &needle, from) {
        let after = html.as_bytes().get(start + needle.len()).copied();
        if matches!(after, Some(b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/')) {
            let open_end = find_tag_end(html, start)?;
            let self_closing = html.as_bytes().get(open_end.saturating_sub(2)) == Some(&b'/');
            return Some((start, open_end, self_closing));
        }
        from = start + needle.len();
    }
    None
}
/// Finds the `>` that closes the tag opened at `start`.
fn find_tag_end(html: &str, start: usize) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (offset, value) in html[start..].char_indices() {
        match quote {
            Some(active) if value == active => quote = None,
            Some(_) => {}
            None => match value {
                '"' | '\'' => quote = Some(value),
                '>' => return Some(start + offset + 1),
                _ => {}
            },
        }
    }
    None
}
/// Finds the end of `</tag>` after `from`.
fn find_close_tag(html: &str, tag: &str, from: usize) -> Option<usize> {
    let needle = format!("</{tag}");
    let close_start = find_ascii_case_insensitive(html, &needle, from)?;
    find_tag_end(html, close_start)
}
/// Reads `<title>`, `og:title`, author, date and description.
fn read_metadata(html: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    if let Some(title) = element_text(html, "title") {
        let title = collapse_whitespace(&title);
        if !title.is_empty() {
            metadata.insert("title".to_owned(), title);
        }
    }
    let mut from = 0;
    while let Some(start) = find_ascii_case_insensitive(html, "<meta", from) {
        let Some(end) = find_tag_end(html, start) else {
            break;
        };
        let attrs = &html[start..end];
        let key = attribute(attrs, "property")
            .or_else(|| attribute(attrs, "name"))
            .map(|value| value.to_lowercase());
        if let (Some(key), Some(content)) = (key, attribute(attrs, "content")) {
            let slot = match key.as_str() {
                "og:title" => Some("title"),
                "author" => Some("author"),
                "article:published_time" | "date" | "datepublished" => Some("date"),
                "description" | "og:description" => Some("description"),
                _ => None,
            };
            if let Some(slot) = slot {
                metadata.entry(slot.to_owned()).or_insert(content);
            }
        }
        from = end;
    }
    metadata
}
/// The collapsed text of the first `tag` element.
fn element_text(html: &str, tag: &str) -> Option<String> {
    let (_start, open_end, self_closing) = find_open_tag(html, tag)?;
    if self_closing {
        return None;
    }
    let close = find_close_tag(html, tag, open_end)?;
    let body = &html[open_end..close];
    let close_start = body.rfind("</").unwrap_or(body.len());
    Some(decode_entities(&body[..close_start]))
}
/// The value of `name` in a tag's attribute text.
fn attribute(attrs: &str, name: &str) -> Option<String> {
    let mut from = 0;
    while let Some(start) = find_ascii_case_insensitive(attrs, name, from) {
        let before_ok = start == 0
            || attrs[..start]
                .chars()
                .next_back()
                .is_none_or(char::is_whitespace);
        let mut rest = &attrs[start + name.len()..];
        if !before_ok {
            from = start + name.len();
            continue;
        }
        rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            from = start + name.len();
            continue;
        };
        let rest = rest.trim_start();
        let value = match rest.chars().next() {
            Some(quote @ ('"' | '\'')) => rest[1..].split(quote).next().unwrap_or(""),
            _ => rest.split_whitespace().next().unwrap_or(""),
        };
        return Some(decode_entities(value).trim().to_owned());
    }
    None
}
/// Decodes the common named and numeric HTML entities.
fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(semicolon) = tail.find(';').filter(|index| *index <= 12) else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let entity = &tail[1..semicolon];
        if let Some(value) = decode_entity(entity) {
            out.push_str(&value);
            rest = &tail[semicolon + 1..];
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}
/// Decodes one entity body (without `&` and `;`).
fn decode_entity(entity: &str) -> Option<String> {
    let named = match entity {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => " ",
        "ndash" => "–",
        "mdash" => "—",
        "hellip" => "…",
        "copy" => "©",
        "reg" => "®",
        "trade" => "™",
        "laquo" => "«",
        "raquo" => "»",
        "middot" => "·",
        "times" => "×",
        "euro" => "€",
        "pound" => "£",
        "yen" => "¥",
        "sect" => "§",
        "para" => "¶",
        "deg" => "°",
        "plusmn" => "±",
        "frac12" => "½",
        _ => "",
    };
    if !named.is_empty() {
        return Some(named.to_owned());
    }
    let digits = entity.strip_prefix('#')?;
    let value = match digits.strip_prefix(['x', 'X']) {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None => digits.parse::<u32>().ok()?,
    };
    char::from_u32(value).map(String::from)
}
/// Collapses every whitespace run to one space and trims.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
/// Strips the markdown markers a `Text` extraction does not keep.
fn plain_text(markdown: &str) -> String {
    let stripped: String = markdown
        .chars()
        .map(|value| {
            if "#>*_`[]()|-".contains(value) {
                ' '
            } else {
                value
            }
        })
        .collect();
    stripped
        .lines()
        .map(collapse_whitespace)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
/// Deduplicates identical blocks (compared whitespace-collapsed, lower-cased).
fn deduplicate_blocks(blocks: Vec<String>) -> (Vec<String>, usize) {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<String> = Vec::new();
    let mut duplicates = 0_usize;
    for block in blocks {
        let key = collapse_whitespace(&block).to_lowercase();
        if key.is_empty() {
            continue;
        }
        if !seen.insert(key) {
            duplicates += 1;
            continue;
        }
        kept.push(block.trim().to_owned());
    }
    (kept, duplicates)
}
/// Whether a block already carries `title` as its headline.
fn block_headline(block: &str, title: &str) -> bool {
    let headline = block.trim_start_matches(['#', ' ']);
    collapse_whitespace(headline).to_lowercase() == collapse_whitespace(title).to_lowercase()
}
/// One tokenizer token.
enum Token<'a> {
    /// Raw text between tags.
    Text(&'a str),
    /// A start tag.
    Open {
        /// Lower-cased tag name.
        name: String,
        /// The raw attribute text.
        attrs: String,
        /// Whether the tag self-closed.
        self_closing: bool,
    },
    /// An end tag.
    Close {
        /// Lower-cased tag name.
        name: String,
    },
    /// A comment body.
    Comment(&'a str),
}
/// Splits `html` into tags and text.
fn tokenize(html: &str) -> Vec<Token<'_>> {
    let bytes = html.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut text_start = 0;
    while index < bytes.len() {
        if bytes[index] != b'<' {
            index += 1;
            continue;
        }
        let next = bytes.get(index + 1).copied();
        if next == Some(b'!') && html[index..].starts_with("<!--") {
            let Some(end) = html[index + 4..].find("-->") else {
                break;
            };
            if text_start < index {
                tokens.push(Token::Text(&html[text_start..index]));
            }
            tokens.push(Token::Comment(&html[index + 4..index + 4 + end]));
            index += 4 + end + 3;
            text_start = index;
            continue;
        }
        if matches!(next, Some(b'!' | b'?')) {
            let Some(end) = html[index..].find('>') else {
                break;
            };
            index += end + 1;
            text_start = index;
            continue;
        }
        if next == Some(b'/') {
            let Some(end) = html[index..].find('>') else {
                break;
            };
            if text_start < index {
                tokens.push(Token::Text(&html[text_start..index]));
            }
            let name = html[index + 2..index + end].trim().to_lowercase();
            tokens.push(Token::Close {
                name: name.split_whitespace().next().unwrap_or("").to_owned(),
            });
            index += end + 1;
            text_start = index;
            continue;
        }
        if !next.is_some_and(|value| value.is_ascii_alphabetic()) {
            index += 1;
            continue;
        }
        let Some(end) = find_tag_end(html, index) else {
            break;
        };
        if text_start < index {
            tokens.push(Token::Text(&html[text_start..index]));
        }
        let inner = html[index + 1..end - 1].trim_end();
        let self_closing = inner.ends_with('/');
        let inner = inner.trim_end_matches('/');
        let name_end = inner
            .bytes()
            .position(|value| value.is_ascii_whitespace())
            .unwrap_or(inner.len());
        let name = inner[..name_end].to_lowercase();
        let attrs = inner[name_end..].to_owned();
        tokens.push(Token::Open {
            name,
            attrs,
            self_closing,
        });
        index = end;
        text_start = index;
    }
    if text_start < html.len() {
        tokens.push(Token::Text(&html[text_start..]));
    }
    tokens
}
/// The block structure of a rendered document.
struct Renderer {
    config: HTMLExtractorConfig,
    blocks: Vec<String>,
    buffer: String,
    heading: Option<(u8, usize)>,
    preformatted: Option<String>,
    link: Option<(usize, String)>,
    code: Option<usize>,
    ordered_list: usize,
    list_stack: Vec<bool>,
    pending_bullet: Option<String>,
    table: Option<TableState>,
    /// `<a>` elements kept.
    link_count: usize,
    /// `<table>` elements kept.
    table_count: usize,
    /// `<pre>` blocks kept.
    code_block_count: usize,
}
/// The rows collected from one `<table>`.
struct TableState {
    rows: Vec<Vec<String>>,
    row: Vec<String>,
    cell: Option<String>,
}
impl Renderer {
    /// A renderer over `config`.
    fn new(config: HTMLExtractorConfig) -> Self {
        Self {
            config,
            blocks: Vec::new(),
            buffer: String::new(),
            heading: None,
            preformatted: None,
            link: None,
            code: None,
            ordered_list: 0,
            list_stack: Vec::new(),
            pending_bullet: None,
            table: None,
            link_count: 0,
            table_count: 0,
            code_block_count: 0,
        }
    }
    /// Renders every token in `html`.
    fn render(&mut self, html: &str) {
        for token in tokenize(html) {
            match token {
                Token::Text(text) => self.text(text),
                Token::Comment(text) => {
                    if self.config.include_comments {
                        self.blocks.push(collapse_whitespace(text));
                    }
                }
                Token::Open {
                    name,
                    attrs,
                    self_closing,
                } => self.open(&name, &attrs, self_closing),
                Token::Close { name } => self.close(&name),
            }
        }
        self.flush();
    }
    /// Returns the rendered blocks after flushing anything pending.
    fn finish(&mut self) -> Vec<String> {
        self.flush();
        std::mem::take(&mut self.blocks)
    }
    /// Handles raw text.
    fn text(&mut self, text: &str) {
        if let Some(preformatted) = self.preformatted.as_mut() {
            preformatted.push_str(&decode_entities(text));
            return;
        }
        let decoded = decode_entities(text);
        if let Some(table) = self.table.as_mut() {
            if let Some(cell) = table.cell.as_mut() {
                cell.push_str(&decoded);
            }
            return;
        }
        self.buffer.push_str(&decoded);
    }
    /// Handles a start tag.
    fn open(&mut self, name: &str, attrs: &str, self_closing: bool) {
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.flush();
                let level = name[1..].parse::<u8>().unwrap_or(1);
                self.heading = Some((level, self.buffer.len()));
            }
            "p" | "div" | "section" | "article" | "main" | "blockquote" | "header" | "figure"
            | "figcaption" | "dl" | "dd" | "dt" => self.flush(),
            "hr" => {
                self.flush();
                self.blocks.push("---".to_owned());
            }
            "ul" | "ol" => {
                self.flush();
                self.list_stack.push(name == "ol");
                self.ordered_list = 0;
            }
            "li" => {
                self.flush();
                let ordered = *self.list_stack.last().unwrap_or(&false);
                self.pending_bullet = Some(if ordered {
                    self.ordered_list += 1;
                    format!("{}. ", self.ordered_list)
                } else {
                    "- ".to_owned()
                });
            }
            "pre" => {
                self.flush();
                self.preformatted = Some(String::new());
                self.code_block_count += 1;
            }
            "code" => {
                if self.preformatted.is_none() && self.config.include_formatting {
                    self.code = Some(self.buffer.len());
                }
            }
            "a" => {
                if self.config.include_links {
                    self.link_count += 1;
                    let href = attribute(attrs, "href").unwrap_or_default();
                    self.link = Some((self.buffer.len(), href));
                }
            }
            "br" => match self.preformatted.as_mut() {
                Some(preformatted) => preformatted.push('\n'),
                None => self.buffer.push('\n'),
            },
            "table" => {
                if self.config.include_tables {
                    self.flush();
                    self.table = Some(TableState {
                        rows: Vec::new(),
                        row: Vec::new(),
                        cell: None,
                    });
                    self.table_count += 1;
                }
            }
            "tr" => {
                if let Some(table) = self.table.as_mut() {
                    finish_row(table);
                }
            }
            "td" | "th" => {
                if let Some(table) = self.table.as_mut() {
                    table.cell = Some(String::new());
                }
            }
            _ => {}
        }
        if self_closing {
            self.close(name);
        }
    }
    /// Handles an end tag.
    fn close(&mut self, name: &str) {
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                if let Some((level, start)) = self.heading.take() {
                    let text = collapse_whitespace(&self.buffer[start.min(self.buffer.len())..]);
                    self.buffer.truncate(start.min(self.buffer.len()));
                    if !text.is_empty() {
                        let prefix = if self.config.include_formatting {
                            format!("{} ", "#".repeat(usize::from(level)))
                        } else {
                            String::new()
                        };
                        self.blocks.push(format!("{prefix}{text}"));
                    }
                }
            }
            "p" | "div" | "section" | "article" | "main" | "blockquote" | "header" | "figure"
            | "figcaption" | "dl" | "dd" | "dt" | "li" => self.flush(),
            "ul" | "ol" => {
                self.flush();
                self.list_stack.pop();
                self.ordered_list = 0;
            }
            "pre" => {
                if let Some(preformatted) = self.preformatted.take() {
                    let body = preformatted.trim_matches('\n');
                    if !body.trim().is_empty() {
                        self.blocks.push(format!("```\n{body}\n```"));
                    }
                }
            }
            "code" => {
                if let Some(start) = self.code.take() {
                    let start = start.min(self.buffer.len());
                    let text = collapse_whitespace(&self.buffer[start..]);
                    if !text.is_empty() {
                        self.buffer.truncate(start);
                        self.buffer.push('`');
                        self.buffer.push_str(&text);
                        self.buffer.push('`');
                    }
                }
            }
            "a" => {
                if let Some((start, href)) = self.link.take() {
                    let start = start.min(self.buffer.len());
                    let label = collapse_whitespace(&self.buffer[start..]);
                    if !href.is_empty() && !label.is_empty() {
                        self.buffer.truncate(start);
                        self.buffer.push('[');
                        self.buffer.push_str(&label);
                        self.buffer.push_str("](");
                        self.buffer.push_str(&href);
                        self.buffer.push(')');
                    } else if !label.is_empty() {
                        self.buffer.truncate(start);
                        self.buffer.push_str(&label);
                    }
                }
            }
            "table" => {
                if let Some(mut table) = self.table.take() {
                    finish_row(&mut table);
                    let rendered = render_table(&table.rows);
                    if !rendered.is_empty() {
                        self.blocks.push(rendered);
                    }
                }
            }
            "tr" => {
                if let Some(table) = self.table.as_mut() {
                    finish_row(table);
                }
            }
            "td" | "th" => {
                if let Some(table) = self.table.as_mut() {
                    if let Some(cell) = table.cell.take() {
                        table.row.push(collapse_whitespace(&cell));
                    }
                }
            }
            _ => {}
        }
    }
    /// Flushes the current inline buffer as one block.
    fn flush(&mut self) {
        if self.heading.is_some() {
            return;
        }
        let text = collapse_whitespace(&self.buffer);
        self.buffer.clear();
        if text.is_empty() {
            return;
        }
        let text = match self.pending_bullet.take() {
            Some(bullet) => format!("{bullet}{text}"),
            None => text,
        };
        self.blocks.push(text);
    }
}
/// Pushes the pending row when it holds cells.
fn finish_row(table: &mut TableState) {
    if !table.row.is_empty() {
        table.rows.push(std::mem::take(&mut table.row));
    }
}
/// Renders collected rows as a markdown table.
fn render_table(rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return String::new();
    }
    let mut normalized: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            let mut cells = row.clone();
            cells.resize(width, String::new());
            cells
        })
        .collect();
    let header = normalized.remove(0);
    let mut lines = vec![
        format!("| {} |", header.join(" | ")),
        format!("| {} |", vec!["---"; width].join(" | ")),
    ];
    for row in normalized {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    lines.join("\n")
}
