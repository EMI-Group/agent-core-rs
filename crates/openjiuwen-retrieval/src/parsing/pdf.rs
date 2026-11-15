//! `PDF` text extraction from content streams.
//!
//! Port of the text half of the Python `PDFParser`
//! (`indexing/processor/parser/pdf_parser.py`) without `pdfplumber`: instead of
//! a full PDF object model this module reads the content streams directly.
//!
//! Every stream is run through the filter pipeline named by its dictionary
//! (`/Filter` plus `/DecodeParms`), in the order the dictionary lists it:
//!
//! - the decodable filters — `ASCIIHexDecode`, `ASCII85Decode`,
//!   `RunLengthDecode`, and `FlateDecode` (through [`zlib_decompress`]), long and
//!   abbreviated (`/AHx`, `/A85`, `/RL`, `/Fl`) forms included — with predictors
//!   `2` (`TIFF` differencing) and `10`..`15` (PNG row filters) undone after the
//!   filter that names them;
//! - an empty chain means the stream bytes are already the content stream.
//!
//! The decoded bytes are then read with the text operators `Tj`, `TJ` (string and
//! kerning-number arrays, a space is inserted when a negative adjustment exceeds
//! [`KERNING_SPACE_THRESHOLD`]), `'` and `"` (line break before the string), and
//! the positioning operators `Td`, `TD`, `T*`, `Tm`, `BT`, `ET` (line break).
//! PDF literal strings are decoded with their escapes (`\n \r \t \b \f \( \)
//! \\`), balanced parentheses, line continuations, and octal escapes (`\ddd`),
//! plus hex strings (`<48656C6C6F>`); a UTF-16BE byte-order mark switches either
//! form to UTF-16BE, everything else is UTF-8 with a Latin-1 fallback.
//! Text-only extraction: there is no OCR, font-metric reconstruction, or layout
//! analysis, matching the feature set the reference gets from
//! `page.extract_text()`.
//!
//! Two kinds of stream are reported instead of silently disappearing:
//!
//! - image payload filters (`CCITTFaxDecode`, `DCTDecode`, `JPXDecode`,
//!   `JBIG2Decode`), an unknown filter name, and a stream whose filter chain or
//!   predictor cannot be undone are skipped and recorded in
//!   [`PdfDocument::incomplete`], [`PdfDocument::skipped_filters`], and
//!   [`PdfDocument::skipped_streams`] with a `warn` log, so the caller learns the
//!   text is partial instead of trusting an empty page;
//! - documents that cannot be read at all are [`Error::Unsupported`]: encrypted
//!   documents (`/Encrypt`), compressed object streams (`/Type /ObjStm`, which
//!   would need xref parsing), `/LZWDecode` and `/Crypt` streams (the extraction
//!   cannot continue past them), and an unsupported predictor value or bit depth.
//!
//! Page order is stream order, since no cross-reference table is parsed: one
//! [`PdfDocument::pages`] entry per decoded content stream, in file order. A
//! scanned or otherwise text-free document is not an error — it parses to an
//! empty text with [`PdfDocument::has_text`] set to `false`.

use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::warn;
use serde::{Deserialize, Serialize};

use super::inflate::DEFAULT_MAX_OUTPUT_BYTES;

mod dictionary;
mod filters;
mod predictor;

/// A `TJ` kerning adjustment more negative than this (thousandths of a text-space
/// unit) is rendered as a space.
pub const KERNING_SPACE_THRESHOLD: f64 = -100.0;

/// How far before a `stream` keyword the stream dictionary is inspected.
const STREAM_DICTIONARY_WINDOW: usize = 1024;

/// Operand stack depth kept between operators (only the tail is ever used).
const MAX_OPERANDS: usize = 8;

/// Maximum nesting of `TJ` arrays.
const MAX_ARRAY_DEPTH: usize = 8;

/// Maximum number of elements read from one `TJ` array.
const MAX_ARRAY_ITEMS: usize = 65_536;

/// Text extracted from a PDF file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PdfDocument {
    /// Non-empty page texts joined by `\n` (the reference joins page strings the
    /// same way).
    pub text: String,
    /// One entry per decoded content stream, in file order, trimmed of
    /// surrounding whitespace.
    pub pages: Vec<String>,
    /// Number of decoded content streams (`pages.len()`).
    pub page_count: usize,
    /// `true` when at least one page yielded text.
    pub has_text: bool,
    /// `true` when at least one stream was skipped, so `text` is partial.
    ///
    /// Skipped streams are image payloads, unknown filters, and payloads whose
    /// filter chain or predictor could not be undone.
    pub incomplete: bool,
    /// Filters of the skipped streams, deduplicated in first-seen order.
    pub skipped_filters: Vec<String>,
    /// One record per skipped stream, naming its 0-based ordinal and the reason.
    pub skipped_streams: Vec<String>,
}

impl PdfDocument {
    /// Returns `true` when no page yielded text (for example a scanned PDF).
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Reads text out of PDF files.
#[derive(Debug, Clone, Copy, Default)]
pub struct PdfParser;

impl PdfParser {
    /// Creates a parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a whole PDF file from its bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for encrypted documents, object streams, the
    /// undecodable `/LZWDecode` and `/Crypt` filters, and a predictor value or
    /// bit depth this crate does not implement. Streams that are skipped instead
    /// (image payloads, unknown filters, undecodable payloads) are reported
    /// through [`PdfDocument::incomplete`] and its two record fields. A text-free
    /// document is not an error.
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<PdfDocument> {
        if contains_token(bytes, b"/Encrypt") {
            return Err(Error::Unsupported(
                "encrypted PDF documents are not supported".to_owned(),
            ));
        }
        if has_object_stream(bytes) {
            return Err(Error::Unsupported(
                "PDF object streams (/Type /ObjStm) are not supported: \
                 cross-reference streams need a full PDF object model"
                    .to_owned(),
            ));
        }
        let extracted = Self::extract_streams(bytes)?;
        let pages: Vec<String> = extracted
            .pages
            .iter()
            .map(|page| page.trim().to_owned())
            .collect();
        let text = pages
            .iter()
            .filter(|page| !page.is_empty())
            .cloned()
            .collect::<Vec<String>>()
            .join("\n");
        Ok(PdfDocument {
            has_text: !text.is_empty(),
            page_count: pages.len(),
            text,
            pages,
            incomplete: !extracted.skipped_streams.is_empty(),
            skipped_filters: extracted.skipped_filters,
            skipped_streams: extracted.skipped_streams,
        })
    }

    /// Parses the PDF file at `path` (read with `tokio::fs`).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] for a missing file, [`Error::Transport`] for any other
    /// I/O failure, plus every [`PdfParser::parse_bytes`] error.
    pub async fn parse_path(&self, path: impl AsRef<Path>) -> Result<PdfDocument> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => {
                    Error::NotFound(format!("File {} does not exist", path.display()))
                }
                _ => Error::Transport(format!("reading {}: {error}", path.display())),
            })?;
        self.parse_bytes(&bytes)
    }

    /// Walks every `stream … endstream` region in file order and decodes it.
    ///
    /// # Errors
    ///
    /// The document-level errors of [`PdfParser::parse_bytes`]: an LZW or `Crypt`
    /// filter, an unsupported predictor, or a predictor whose `validate` rejects
    /// its parameters.
    fn extract_streams(bytes: &[u8]) -> Result<ExtractedStreams> {
        let mut extracted = ExtractedStreams::default();
        let mut cursor = 0usize;
        let mut ordinal = 0usize;
        while let Some(stream_start) = find_subslice(bytes, b"stream", cursor) {
            let Some(data_start) = stream_data_start(bytes, stream_start) else {
                cursor = stream_start + b"stream".len();
                continue;
            };
            let Some(relative_end) = find_subslice(bytes, b"endstream", data_start) else {
                break;
            };
            let data_end = trim_stream_end(bytes, data_start, relative_end);
            let dictionary = stream_dictionary(bytes, stream_start);
            cursor = relative_end + b"endstream".len();
            let index = ordinal;
            ordinal += 1;

            let chain = dictionary::parse_filter_chain(dictionary);
            let parms = dictionary::parse_decode_parms(dictionary);
            for params in parms.iter().flatten() {
                params.validate()?;
            }
            if let Some(filter) = chain.iter().find(|filter| filter.aborts_extraction()) {
                return Err(Error::Unsupported(format!(
                    "PDF stream filter {} cannot be decoded: text extraction cannot continue",
                    filter.name()
                )));
            }
            if let Some(filter) = chain.iter().find(|filter| filter.is_image_payload()) {
                let reason = format!(
                    "image payload filter {} cannot carry text operators",
                    filter.name()
                );
                extracted.record_skipped(index, &chain, &reason);
                continue;
            }

            let data = &bytes[data_start..data_end];
            match decode_stream(data, &chain, &parms) {
                Ok(decoded) => extracted.pages.push(extract_text(&decoded)),
                Err(StreamFailure { filter, error }) => {
                    let reason = format!("{filter} decode failed: {error}");
                    extracted.record_skipped(index, &chain, &reason);
                }
            }
        }
        Ok(extracted)
    }
}

/// Streams decoded by one document pass, plus the skipped-stream accounting.
#[derive(Debug, Default)]
struct ExtractedStreams {
    /// Decoded content-stream texts, in file order.
    pages: Vec<String>,
    /// Filters of the skipped streams, deduplicated in first-seen order.
    skipped_filters: Vec<String>,
    /// One human-readable record per skipped stream.
    skipped_streams: Vec<String>,
}

impl ExtractedStreams {
    /// Records a skipped stream and logs it.
    fn record_skipped(&mut self, index: usize, chain: &[filters::Filter], reason: &str) {
        for filter in chain {
            let name = filter.name().to_owned();
            if !self.skipped_filters.contains(&name) {
                self.skipped_filters.push(name);
            }
        }
        warn!(stream = index, reason, "skipping PDF stream");
        self.skipped_streams
            .push(format!("stream {index}: {reason}"));
    }
}

/// One failed filter step, naming the filter that could not be undone.
struct StreamFailure {
    /// Display name of the filter whose step failed.
    filter: String,
    /// Error that made the step fail.
    error: Error,
}

/// Folds the filter chain and its predictors over one stream payload.
fn decode_stream(
    data: &[u8],
    chain: &[filters::Filter],
    parms: &[Option<predictor::PredictorParams>],
) -> std::result::Result<Vec<u8>, StreamFailure> {
    let mut current: Option<Vec<u8>> = None;
    for (index, filter) in chain.iter().enumerate() {
        let input = current.as_deref().unwrap_or(data);
        let mut decoded =
            filters::decode(filter, input, DEFAULT_MAX_OUTPUT_BYTES).map_err(|error| {
                StreamFailure {
                    filter: filter.name().to_owned(),
                    error,
                }
            })?;
        let params = parms.get(index).copied().flatten().unwrap_or_default();
        if !params.is_identity() {
            decoded =
                predictor::apply(&decoded, &params, DEFAULT_MAX_OUTPUT_BYTES).map_err(|error| {
                    StreamFailure {
                        filter: filter.name().to_owned(),
                        error,
                    }
                })?;
        }
        current = Some(decoded);
    }
    Ok(current.unwrap_or_else(|| data.to_vec()))
}

/// Returns `true` when the file declares a compressed object stream.
///
/// A `/Type` key within the 64 bytes before `/ObjStm` is what marks the
/// dictionary; the file is rejected because resolving those objects needs
/// cross-reference parsing.
fn has_object_stream(bytes: &[u8]) -> bool {
    let mut cursor = 0usize;
    while let Some(found) = find_subslice(bytes, b"/ObjStm", cursor) {
        let start = found.saturating_sub(64);
        if find_subslice(&bytes[start..found], b"/Type", 0).is_some() {
            return true;
        }
        cursor = found + b"/ObjStm".len();
    }
    false
}

/// Returns the offset of the first byte of a stream's data, if `stream_start`
/// really begins a `stream` keyword.
fn stream_data_start(bytes: &[u8], stream_start: usize) -> Option<usize> {
    if let Some(previous) = stream_start.checked_sub(1).map(|index| bytes[index]) {
        if is_regular(previous) {
            return None;
        }
    }
    let mut position = stream_start + b"stream".len();
    match *bytes.get(position)? {
        b'\r' => {
            position += 1;
            if bytes.get(position) == Some(&b'\n') {
                position += 1;
            }
        }
        b'\n' => position += 1,
        _ => return None,
    }
    Some(position)
}

/// Returns the stream dictionary that precedes `stream_start`.
///
/// The window looks back at most [`STREAM_DICTIONARY_WINDOW`] bytes and never
/// past the previous stream's `endstream`, so a filter mentioned by an earlier
/// object cannot be attributed to this one.
fn stream_dictionary(bytes: &[u8], stream_start: usize) -> &[u8] {
    let window_start = stream_start.saturating_sub(STREAM_DICTIONARY_WINDOW);
    let lower = match rfind_subslice(&bytes[..stream_start], b"endstream") {
        Some(end) => (end + b"endstream".len()).max(window_start),
        None => window_start,
    };
    &bytes[lower..stream_start]
}

/// Drops the end-of-line that precedes `endstream`.
fn trim_stream_end(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut end = end;
    while end > start && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    end
}

/// Interprets the text operators of one content stream.
fn extract_text(data: &[u8]) -> String {
    let mut lexer = Lexer::new(data);
    let mut operands: Vec<Operand> = Vec::new();
    let mut text = String::new();
    while let Some(token) = lexer.next_token() {
        match token {
            Token::Number(value) => push_operand(&mut operands, Operand::Number(value)),
            Token::Bytes(bytes) => push_operand(&mut operands, Operand::Bytes(bytes)),
            Token::Array(items) => push_operand(&mut operands, Operand::Array(items)),
            Token::Operator(operator) => {
                apply_operator(operator, &operands, &mut text);
                operands.clear();
            }
            Token::Ignored => {}
        }
    }
    text.trim().to_owned()
}

/// One operand pushed by the content-stream lexer.
enum Operand {
    /// A number (kerning adjustments in `TJ` arrays).
    Number(f64),
    /// A decoded string (literal or hex).
    Bytes(Vec<u8>),
    /// A `TJ` array.
    Array(Vec<Operand>),
}

/// Adds `operand` to the tail of the stack, dropping the oldest entry when full.
fn push_operand(operands: &mut Vec<Operand>, operand: Operand) {
    if operands.len() >= MAX_OPERANDS {
        operands.remove(0);
    }
    operands.push(operand);
}

/// Applies one text operator to `text`.
fn apply_operator(operator: &str, operands: &[Operand], text: &mut String) {
    match operator {
        "Tj" => append_last_string(operands, text),
        "TJ" => {
            if let Some(Operand::Array(items)) = operands.last() {
                for item in items {
                    match item {
                        Operand::Bytes(bytes) => append_text(text, bytes),
                        Operand::Number(value) if *value < KERNING_SPACE_THRESHOLD => {
                            append_space(text);
                        }
                        _ => {}
                    }
                }
            }
        }
        "'" | "\"" => {
            line_break(text);
            append_last_string(operands, text);
        }
        "Td" | "TD" | "T*" | "Tm" | "BT" | "ET" => line_break(text),
        _ => {}
    }
}

/// Appends the most recent string operand, if there is one.
fn append_last_string(operands: &[Operand], text: &mut String) {
    if let Some(Operand::Bytes(bytes)) = operands.last() {
        append_text(text, bytes);
    }
}

/// Appends a decoded PDF string.
fn append_text(text: &mut String, bytes: &[u8]) {
    text.push_str(&decode_string(bytes));
}

/// Appends a separating space unless the text already ends in whitespace.
fn append_space(text: &mut String) {
    if text
        .chars()
        .last()
        .is_some_and(|last| !last.is_whitespace())
    {
        text.push(' ');
    }
}

/// Ends the current line, unless the text is empty or already ends with a newline.
fn line_break(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// Decodes a PDF string: UTF-16BE behind a byte-order mark, otherwise UTF-8 with
/// a Latin-1 fallback.
fn decode_string(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = rest
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_owned(),
        Err(_) => bytes.iter().map(|&byte| char::from(byte)).collect(),
    }
}

/// One token of a content stream.
enum Token<'a> {
    /// A number.
    Number(f64),
    /// A literal (`( … )`) or hex (`< … >`) string.
    Bytes(Vec<u8>),
    /// A `[ … ]` array.
    Array(Vec<Operand>),
    /// An operator keyword.
    Operator(&'a str),
    /// A name, dictionary delimiter, or other byte with no text meaning.
    Ignored,
}

/// Byte lexer for content streams.
struct Lexer<'a> {
    /// Content stream bytes.
    data: &'a [u8],
    /// Current offset.
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Starts a lexer at the beginning of `data`.
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Returns the byte at the cursor without consuming it.
    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Advances past whitespace and `%` comments.
    fn skip_ignorable(&mut self) {
        loop {
            match self.peek() {
                Some(byte) if is_whitespace(byte) => self.pos += 1,
                Some(b'%') => {
                    while let Some(byte) = self.peek() {
                        self.pos += 1;
                        if byte == b'\n' || byte == b'\r' {
                            break;
                        }
                    }
                }
                _ => return,
            }
        }
    }

    /// Returns the next token, or `None` at the end of the stream.
    fn next_token(&mut self) -> Option<Token<'a>> {
        self.skip_ignorable();
        match self.peek()? {
            b'(' => {
                self.pos += 1;
                Some(Token::Bytes(self.read_literal()))
            }
            b'<' => {
                if self.data.get(self.pos + 1) == Some(&b'<') {
                    self.pos += 2;
                    Some(Token::Ignored)
                } else {
                    self.pos += 1;
                    Some(Token::Bytes(self.read_hex()))
                }
            }
            b'>' => {
                self.pos += if self.data.get(self.pos + 1) == Some(&b'>') {
                    2
                } else {
                    1
                };
                Some(Token::Ignored)
            }
            b'[' => {
                self.pos += 1;
                Some(Token::Array(self.read_array(1)))
            }
            b']' | b')' | b'{' | b'}' => {
                self.pos += 1;
                Some(Token::Ignored)
            }
            b'/' => {
                self.pos += 1;
                self.read_name();
                Some(Token::Ignored)
            }
            b'\'' => {
                self.pos += 1;
                Some(Token::Operator("'"))
            }
            b'"' => {
                self.pos += 1;
                Some(Token::Operator("\""))
            }
            b'+' | b'-' | b'.' | b'0'..=b'9' => Some(self.read_number()),
            _ => {
                let start = self.pos;
                while let Some(byte) = self.peek() {
                    if is_whitespace(byte) || is_delimiter(byte) {
                        break;
                    }
                    self.pos += 1;
                }
                if start == self.pos {
                    self.pos += 1;
                    return Some(Token::Ignored);
                }
                Some(Token::Operator(
                    std::str::from_utf8(&self.data[start..self.pos]).unwrap_or(""),
                ))
            }
        }
    }

    /// Reads a number, or an ignored token when the bytes are not one.
    fn read_number(&mut self) -> Token<'a> {
        let start = self.pos;
        while let Some(byte) = self.peek() {
            let continues =
                byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.' | b'e' | b'E');
            if !continues {
                break;
            }
            self.pos += 1;
        }
        std::str::from_utf8(&self.data[start..self.pos])
            .ok()
            .and_then(|raw| raw.parse::<f64>().ok())
            .map_or(Token::Ignored, Token::Number)
    }

    /// Reads a literal string, honouring escapes and balanced parentheses.
    ///
    /// A string truncated by the end of the stream yields what was read, so one
    /// damaged stream cannot hide the rest of the page.
    fn read_literal(&mut self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut depth = 1usize;
        while let Some(byte) = self.peek() {
            self.pos += 1;
            match byte {
                b'\\' => {
                    let Some(escaped) = self.peek() else {
                        break;
                    };
                    self.pos += 1;
                    self.read_escape(escaped, &mut bytes);
                }
                b'(' => {
                    depth += 1;
                    bytes.push(b'(');
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    bytes.push(b')');
                }
                other => bytes.push(other),
            }
        }
        bytes
    }

    /// Appends the effect of one backslash escape.
    fn read_escape(&mut self, escaped: u8, bytes: &mut Vec<u8>) {
        match escaped {
            b'n' => bytes.push(b'\n'),
            b'r' => bytes.push(b'\r'),
            b't' => bytes.push(b'\t'),
            b'b' => bytes.push(0x08),
            b'f' => bytes.push(0x0c),
            b'(' | b')' | b'\\' => bytes.push(escaped),
            // Line continuations produce nothing.
            b'\r' => {
                if self.peek() == Some(b'\n') {
                    self.pos += 1;
                }
            }
            b'\n' => {}
            b'0'..=b'7' => {
                let mut value = u32::from(escaped - b'0');
                let mut digits = 1;
                while digits < 3 {
                    match self.peek() {
                        Some(next @ b'0'..=b'7') => {
                            value = value * 8 + u32::from(next - b'0');
                            self.pos += 1;
                            digits += 1;
                        }
                        _ => break,
                    }
                }
                bytes.push(u8::try_from(value & 0xFF).unwrap_or(0));
            }
            other => bytes.push(other),
        }
    }

    /// Reads a hex string; whitespace is ignored and a trailing half byte is
    /// completed with `0`.
    fn read_hex(&mut self) -> Vec<u8> {
        let mut digits = Vec::new();
        while let Some(byte) = self.peek() {
            self.pos += 1;
            if byte == b'>' {
                break;
            }
            if let Some(value) = hex_value(byte) {
                digits.push(value);
            }
        }
        digits
            .chunks(2)
            .map(|pair| {
                let high = pair[0];
                let low = pair.get(1).copied().unwrap_or(0);
                (high << 4) | low
            })
            .collect()
    }

    /// Reads an array, including nested arrays up to [`MAX_ARRAY_DEPTH`].
    fn read_array(&mut self, depth: usize) -> Vec<Operand> {
        let mut items = Vec::new();
        while items.len() < MAX_ARRAY_ITEMS {
            self.skip_ignorable();
            let Some(byte) = self.peek() else {
                break;
            };
            match byte {
                b']' => {
                    self.pos += 1;
                    break;
                }
                b'(' => {
                    self.pos += 1;
                    items.push(Operand::Bytes(self.read_literal()));
                }
                b'<' => {
                    if self.data.get(self.pos + 1) == Some(&b'<') {
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                        items.push(Operand::Bytes(self.read_hex()));
                    }
                }
                b'>' => {
                    self.pos += if self.data.get(self.pos + 1) == Some(&b'>') {
                        2
                    } else {
                        1
                    };
                }
                b'[' if depth < MAX_ARRAY_DEPTH => {
                    self.pos += 1;
                    items.push(Operand::Array(self.read_array(depth + 1)));
                }
                b'/' => {
                    self.pos += 1;
                    self.read_name();
                }
                b'+' | b'-' | b'.' | b'0'..=b'9' => {
                    if let Token::Number(value) = self.read_number() {
                        items.push(Operand::Number(value));
                    }
                }
                _ => self.pos += 1,
            }
        }
        items
    }

    /// Skips a `/Name` token.
    fn read_name(&mut self) {
        while let Some(byte) = self.peek() {
            if is_whitespace(byte) || is_delimiter(byte) {
                break;
            }
            self.pos += 1;
        }
    }
}

/// Returns `true` for PDF whitespace (including the null byte).
fn is_whitespace(byte: u8) -> bool {
    matches!(byte, 0 | 9 | 10 | 12 | 13 | 32)
}

/// Returns `true` for PDF delimiter bytes.
fn is_delimiter(byte: u8) -> bool {
    matches!(
        byte,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

/// Returns `true` for bytes that continue a keyword or number.
fn is_regular(byte: u8) -> bool {
    !is_whitespace(byte) && !is_delimiter(byte)
}

/// Returns the numeric value of one hexadecimal digit.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Returns the offset of `needle` in `haystack`, starting at `start`.
fn find_subslice(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if needle.is_empty() || start >= haystack.len() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| offset + start)
}

/// Returns the offset of the last `needle` in `haystack`.
fn rfind_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .rposition(|window| window == needle)
}

/// Returns `true` when `haystack` contains `token` as a complete PDF token.
///
/// The byte after the token must not continue a name, so `/Fl` matches `/Fl` but
/// not `/FlateDecode`.
fn contains_token(haystack: &[u8], token: &[u8]) -> bool {
    let mut cursor = 0usize;
    while let Some(found) = find_subslice(haystack, token, cursor) {
        let end = found + token.len();
        let terminated = match haystack.get(end) {
            None => true,
            Some(byte) => !is_regular(*byte),
        };
        if terminated {
            return true;
        }
        cursor = end;
    }
    false
}

#[cfg(test)]
mod tests;
