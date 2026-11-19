//! Port of `openjiuwen/harness/tools/filesystem.py`: the `read_file` tool.
//!
//! The tool reads a window of a text file, renders it with `cat -n` style line numbers and
//! records the observed lines in the read-before-write registry so `write_file`/`edit_file`
//! can refuse to clobber content the model has never seen.
//!
//! PDF files take a second path. `pages` selects 1-based pages — a single page (`"3"`), an
//! inclusive range (`"1-5"`) or a comma-separated union (`"1-3,7"`), at most
//! [`PDF_MAX_PAGES_PER_READ`] pages — and every selected page is rendered with its own
//! `## Page {n}` marker followed by that page's numbered text. Page selection is deliberately
//! *not* a text read: a page read records the file fingerprint with **no** coverage, so
//! `write_file` still refuses to overwrite a PDF whose text the model has only seen page by
//! page. Only a whole-text read registers coverage.
//!
//! `caption` (Python's multimodal caption for images under `skills/`) is rejected with a typed
//! failure: this build ships no vision model, and silently ignoring the request would let a
//! caller believe the image had been described.
//!
//! Size note: this file overruns the 1000-line baseline only because the text reader, the PDF
//! reader and the page-selection grammar keep their unit tests beside the code; the code above
//! the test module is well under the limit.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_retrieval::parsing::{PdfDocument, PdfParser};
use openjiuwen_sysop::fs::types::{
    FileContent, LineRange, ReadBytesParams, ReadFileParams, ReadMode,
};
use openjiuwen_sysop::operation::FsOperation;

use super::descriptions;
use super::read_state::{file_fingerprint, ReadCoverage, ReadStateRegistry};
use super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};

/// Maximum number of lines a single read returns (`ReadFileTool.MAX_LINES_TO_READ`).
const MAX_LINES_TO_READ: usize = 2000;

/// Maximum file size, in bytes, accepted by a read without an explicit `limit`.
const MAX_SIZE_BYTES: u64 = 256 * 1024;

/// Maximum number of pages a single PDF read selects
/// (`ReadFileTool.PDF_MAX_PAGES_PER_READ`).
const PDF_MAX_PAGES_PER_READ: usize = 20;

/// Maximum PDF size, in bytes, a single read will load.
///
/// Python has no such ceiling, but a PDF is parsed from memory, so an unbounded read of a
/// multi-gigabyte file would allocate the whole file. The cap is far above any document the
/// page-selection limit allows a caller to render.
const PDF_MAX_SIZE_BYTES: u64 = 64 * 1024 * 1024;

/// Bytes fetched per bounded read while scanning for a line window.
const READ_CHUNK_BYTES: u64 = 64 * 1024;

/// Maximum bytes scanned to reach a requested line offset in a large file.
///
/// A text read with an explicit `limit` is bounded by this budget instead of the file size:
/// the first lines of a multi-gigabyte log cost the budget, never the file. When the budget is
/// exhausted the lines found so far are returned; a request whose offset lies beyond it is
/// reported as a typed failure.
const TEXT_SCAN_MAX_BYTES: u64 = 8 * 1024 * 1024;

/// The warning returned when a whole-file read finds no content.
const EMPTY_FILE_WARNING: &str = "Warning: the file exists but the contents are empty.";

/// The failure returned for a `caption` request.
const CAPTION_UNSUPPORTED: &str = "caption is not supported: image captioning needs a vision \
model and this build does not ship one. Use image_metadata for the image container facts \
(format, dimensions, animation) and call read_file without caption to read text files.";

/// The failure returned for `offset`/`limit` on a PDF read.
const PDF_WINDOW_UNSUPPORTED: &str = "offset/limit do not apply to PDF reads: PDFs are read \
page by page, so select 1-based pages instead (e.g. pages='1-5').";

/// The hint appended to every page-range rejection.
const PAGE_RANGE_HINT: &str = "Use formats like '3', '1-5', or '1-3,7'. Pages are 1-indexed.";

/// Arguments of the `read_file` tool.
///
/// Python parity: the `file_path`/`offset`/`limit`/`pages`/`caption` parameters of
/// `ReadFileTool.invoke`. `pages` selects PDF pages; `caption` is rejected because image
/// captioning needs a vision model that is not part of this build.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileArgs {
    /// The absolute path to the file to read.
    pub file_path: String,
    /// The line number to start reading from. Only provide if the file is too large to read at once or you already know the needed location.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub offset: Option<u64>,
    /// The number of lines to read. Only provide if the file is too large to read at once or only a specific range is needed.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub limit: Option<u64>,
    /// PDF-only page range, e.g. '1-5', '3', '1-3,7'. Maximum 20 pages per request
    #[serde(default)]
    #[schemars(with = "String")]
    pub pages: Option<String>,
    /// Image caption request; rejected, because image captioning needs a vision model.
    #[serde(default)]
    #[schemars(with = "String")]
    pub caption: Option<String>,
}

/// Reads a text or PDF file, one window (or one page selection) at a time.
pub struct ReadFileTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// Read-before-write bookkeeping shared with the other filesystem tools.
    state: Arc<ReadStateRegistry>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl ReadFileTool {
    /// Builds the tool over `env`, reading through `fs` and recording into `state`.
    #[must_use]
    pub fn new(
        env: ToolEnv,
        fs: Option<Arc<dyn FsOperation>>,
        state: Arc<ReadStateRegistry>,
    ) -> Self {
        let card = Self::card_for(env.language());
        Self {
            env,
            fs,
            state,
            card,
        }
    }

    /// Builds the model-facing card in `language`.
    ///
    /// Unknown languages fall back to the Python default (`cn`) inside
    /// [`descriptions::description`].
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "read_file",
            descriptions::description("read_file", language),
            schema_for::<ReadFileArgs>(),
            true,
        )
    }

    /// Renders `text` with 1-based, six-column line numbers (`cat -n` / Python `_cat_n`).
    ///
    /// A trailing newline does not add a phantom line, and the empty string stays empty.
    #[must_use]
    pub fn number_lines(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        text.lines()
            .enumerate()
            .map(|(index, line)| format!("{:>6}\t{line}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Validates the arguments, resolves the path and dispatches to the text or PDF reader.
    async fn read(&self, inputs: &Value) -> ToolPayload {
        // Python reads `file_path` from the raw arguments and reports its absence with the
        // plain validation text rather than a schema error, so check it before parsing.
        if inputs.is_object()
            && inputs
                .get("file_path")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return ToolPayload::fail("file_path is required");
        }

        let args: ReadFileArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.file_path.is_empty() {
            return ToolPayload::fail("file_path is required");
        }
        // The caption check runs before any I/O so the reason is deterministic: the argument is
        // not a hint to be dropped, it asks for a capability this build does not have.
        if args.caption.is_some() {
            return ToolPayload::fail(CAPTION_UNSUPPORTED);
        }

        let resolved = match self
            .env
            .resolve_path("read_file", "file_path", &args.file_path, inputs)
            .await
        {
            Ok(path) => path,
            Err(payload) => return payload,
        };

        let Some(fs) = self.fs.as_deref() else {
            return ToolPayload::fail("filesystem operation is not configured");
        };

        if is_pdf(&resolved) {
            self.read_pdf(&resolved, &args, fs).await
        } else {
            self.read_text(&resolved, &args, fs).await
        }
    }

    /// Reads a window of a text file and registers the coverage it observed.
    async fn read_text(
        &self,
        resolved: &Path,
        args: &ReadFileArgs,
        fs: &dyn FsOperation,
    ) -> ToolPayload {
        if args.pages.is_some() {
            return ToolPayload::fail(format!(
                "pages only applies to PDF files: '{}' is not a PDF. Use offset/limit for text files.",
                resolved.display()
            ));
        }

        let user_supplied_limit = args.limit.is_some();
        let offset = usize::try_from(args.offset.unwrap_or(0)).unwrap_or(usize::MAX);
        let max_lines = u64::try_from(MAX_LINES_TO_READ).unwrap_or(u64::MAX);
        let limit = usize::try_from(args.limit.unwrap_or(max_lines).clamp(1, max_lines))
            .unwrap_or(MAX_LINES_TO_READ)
            .clamp(1, MAX_LINES_TO_READ);
        let display = resolved.to_string_lossy().into_owned();

        // Stat first: an oversized file is rejected before any of it is read, and the whole-file
        // ceiling only applies when the caller did not ask for a bounded window.
        let bounded_size = match bounded_file_size(fs, &display).await {
            Ok(size) => size,
            Err(message) => return ToolPayload::fail(message),
        };
        if let Some(total_bytes) = bounded_size {
            if !user_supplied_limit && total_bytes > MAX_SIZE_BYTES {
                return ToolPayload::fail(oversized_message(total_bytes));
            }
        }

        let (raw_text, _total_bytes) = if bounded_size.is_some() {
            match read_text_window(fs, &display, offset, limit).await {
                Ok((text, actual_total)) => (text, actual_total),
                Err(message) => return ToolPayload::fail(message),
            }
        } else {
            // A backend without a bounded read keeps the whole-file path; its own policy bounds
            // the read, and the size guard above cannot see the file size in advance.
            let params = ReadFileParams {
                path: display.clone(),
                mode: ReadMode::Text,
                line_range: Some(LineRange {
                    start: offset.saturating_add(1),
                    end: offset.saturating_add(limit),
                }),
                ..ReadFileParams::default()
            };

            let result = fs.read_file(params).await;
            if result.is_failure() {
                return ToolPayload::fail(result.message);
            }
            let Some(data) = result.data else {
                return ToolPayload::fail(
                    "read_file returned no data from the filesystem operation",
                );
            };
            let total_bytes = data.total_bytes;
            let text = match data.content {
                FileContent::Text(text) => text,
                FileContent::Bytes(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            };
            if !user_supplied_limit && total_bytes > MAX_SIZE_BYTES {
                return ToolPayload::fail(oversized_message(total_bytes));
            }
            (text, total_bytes)
        };

        let (displayed, line_count) = if raw_text.trim().is_empty() {
            let content = if offset == 0 {
                EMPTY_FILE_WARNING.to_owned()
            } else {
                format!(
                    "Error: The file exists but no content was returned at offset {offset}. Use a smaller offset or read from the beginning."
                )
            };
            (content, 0)
        } else {
            let displayed = Self::number_lines(&raw_text);
            let count = displayed.lines().count();
            (displayed, count)
        };

        let payload = ToolPayload::ok(json!({
            "content": displayed,
            "file_path": resolved.to_string_lossy(),
            "line_count": line_count,
        }));

        // A metadata failure only skips registration: the read itself already succeeded.
        match tokio::fs::metadata(resolved).await {
            Ok(metadata) => {
                let coverage = (line_count > 0).then(|| ReadCoverage {
                    start: offset.saturating_add(1),
                    end: offset.saturating_add(line_count),
                });
                self.state.record_read(
                    resolved,
                    file_fingerprint(&metadata),
                    coverage,
                    Some(raw_text),
                );
            }
            Err(error) => {
                tracing::debug!(
                    path = %resolved.display(),
                    %error,
                    "skipping read-state registration: metadata unavailable"
                );
            }
        }

        payload
    }

    /// Extracts the selected text pages of a PDF and renders them with page markers.
    ///
    /// The bytes come from the configured [`FsOperation`] in `bytes` mode, so a sandboxed
    /// backend keeps enforcing its own policy; there is no raw-`std::fs` bypass.
    async fn read_pdf(
        &self,
        resolved: &Path,
        args: &ReadFileArgs,
        fs: &dyn FsOperation,
    ) -> ToolPayload {
        let displayed = resolved.to_string_lossy().into_owned();
        if args.offset.unwrap_or(0) != 0 || args.limit.is_some() {
            return ToolPayload::fail(PDF_WINDOW_UNSUPPORTED);
        }
        let range = match args.pages.as_deref() {
            Some(raw) => match PageRange::parse(raw) {
                Ok(range) => Some(range),
                Err(message) => return ToolPayload::fail(message),
            },
            None => None,
        };

        let bytes = match read_pdf_bytes(fs, resolved).await {
            Ok(bytes) => bytes,
            Err(message) => return ToolPayload::fail(message),
        };
        let document = match PdfParser::new().parse_bytes(&bytes) {
            Ok(document) => document,
            Err(error) => {
                return ToolPayload::fail(format!("Failed to read PDF '{displayed}': {error}"));
            }
        };
        let total_pages = document.page_count;
        if !document.has_text {
            return ToolPayload::fail(no_text_message(&displayed, total_pages));
        }
        let selected = match range {
            None => {
                if total_pages > PDF_MAX_PAGES_PER_READ {
                    return ToolPayload::fail(format!(
                        "This PDF has {total_pages} pages, which is too many to read at once. \
                         Use the pages parameter to specify a range (e.g. pages='1-10'). \
                         Maximum {PDF_MAX_PAGES_PER_READ} pages per request."
                    ));
                }
                (1..=total_pages).collect()
            }
            Some(range) => {
                let selected = range.resolve(total_pages);
                if selected.is_empty() {
                    return ToolPayload::fail(format!(
                        "Page range '{}' selects no page of '{displayed}': the document has \
                         {total_pages} page(s).",
                        range.raw()
                    ));
                }
                selected
            }
        };

        let page_range = normalized_range(&selected);
        let (content, line_count) = render_pages(&document, &selected);
        self.record_pdf_read(resolved).await;
        ToolPayload::ok(json!({
            "content": content,
            "file_path": displayed,
            "line_count": line_count,
            "total_pages": total_pages,
            "selected_pages": selected,
            "page_range": page_range,
        }))
    }

    /// Records the fingerprint of a PDF page read without registering any coverage.
    ///
    /// A page read is never a full text read: coverage would let `write_file` overwrite a PDF
    /// whose text the model has only seen one page at a time. Recording the fingerprint alone
    /// still answers "was this file read?" and still detects an external change afterwards.
    async fn record_pdf_read(&self, resolved: &Path) {
        match tokio::fs::metadata(resolved).await {
            Ok(metadata) => {
                self.state
                    .record_read(resolved, file_fingerprint(&metadata), None, None);
            }
            Err(error) => {
                tracing::debug!(
                    path = %resolved.display(),
                    %error,
                    "skipping read-state registration: metadata unavailable"
                );
            }
        }
    }
}

/// A validated `pages` selection.
///
/// Python's `_parse_pdf_page_range` accepts one page or one inclusive range; this grammar adds
/// the comma-separated union (`"1-3,7"`) and rejects everything it cannot honour instead of
/// falling back to "the whole document".
#[derive(Debug, Clone, PartialEq, Eq)]
struct PageRange {
    /// The selection as the model wrote it, trimmed.
    raw: String,
    /// 1-based page numbers, ascending and unique.
    pages: Vec<usize>,
}

impl PageRange {
    /// Parses `"3"`, `"1-5"` or `"1-3,7"` into ascending, unique page numbers.
    ///
    /// `Err` carries the model-facing message of the first rejection: an empty selection, an
    /// empty list element, a non-numeric or zero page, a reversed range, or a selection that
    /// exceeds [`PDF_MAX_PAGES_PER_READ`] pages.
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "Invalid PDF page range: the selection is empty. {PAGE_RANGE_HINT}"
            ));
        }
        let mut pages: Vec<usize> = Vec::new();
        for element in trimmed.split(',') {
            let element = element.trim();
            if element.is_empty() {
                return Err(format!(
                    "Invalid PDF page range '{trimmed}': empty element in the comma-separated \
                     list. {PAGE_RANGE_HINT}"
                ));
            }
            let (first, last) = match element.split_once('-') {
                Some((first, last)) => (first.trim(), last.trim()),
                None => (element, element),
            };
            let first = parse_page_number(first, element, trimmed)?;
            let last = parse_page_number(last, element, trimmed)?;
            if first > last {
                return Err(format!(
                    "Invalid PDF page range '{trimmed}': range '{element}' is reversed (the first \
                     page must not exceed the last). {PAGE_RANGE_HINT}"
                ));
            }
            if last - first + 1 > PDF_MAX_PAGES_PER_READ {
                return Err(too_many_pages_message(trimmed, last - first + 1));
            }
            pages.extend(first..=last);
        }
        pages.sort_unstable();
        pages.dedup();
        if pages.len() > PDF_MAX_PAGES_PER_READ {
            return Err(too_many_pages_message(trimmed, pages.len()));
        }
        Ok(Self {
            raw: trimmed.to_owned(),
            pages,
        })
    }

    /// The selection as the model wrote it.
    fn raw(&self) -> &str {
        &self.raw
    }

    /// Intersects the selection with the pages `total_pages` actually has.
    ///
    /// Python clamps an out-of-range page to the document end; keeping only the pages that
    /// exist has the same effect for a single page and a range, and it never invents a page for
    /// a union such as `"1,9"` on a three-page document. The payload always reports
    /// `selected_pages` and `total_pages`, so a clamped or dropped page is visible to the model
    /// instead of being silently ignored.
    fn resolve(&self, total_pages: usize) -> Vec<usize> {
        let mut pages: Vec<usize> = self
            .pages
            .iter()
            .copied()
            .filter(|page| *page <= total_pages)
            .collect();
        pages.sort_unstable();
        pages.dedup();
        pages
    }
}

/// Parses one page number of a range element.
fn parse_page_number(text: &str, element: &str, raw: &str) -> std::result::Result<usize, String> {
    if text.is_empty() {
        return Err(format!(
            "Invalid PDF page range '{raw}': range '{element}' is missing a page number. \
             {PAGE_RANGE_HINT}"
        ));
    }
    let page: usize = text.parse().map_err(|_| {
        format!("Invalid PDF page range '{raw}': '{text}' is not a page number. {PAGE_RANGE_HINT}")
    })?;
    if page == 0 {
        return Err(format!(
            "Invalid PDF page range '{raw}': page 0 is not a page (pages are 1-indexed). \
             {PAGE_RANGE_HINT}"
        ));
    }
    Ok(page)
}

/// Builds the "too many pages" rejection.
fn too_many_pages_message(raw: &str, count: usize) -> String {
    format!(
        "Page range '{raw}' selects {count} pages, which exceeds the maximum of \
         {PDF_MAX_PAGES_PER_READ} pages per request. Use a smaller or fully-bounded range \
         (e.g. '1-20')."
    )
}

/// Builds the "no extractable text" rejection of a scanned PDF.
fn no_text_message(path: &str, total_pages: usize) -> String {
    format!(
        "PDF '{path}' has no extractable text ({total_pages} page(s)): it is probably a scanned \
         document. OCR is not available in this build, so its page text cannot be read."
    )
}

/// Returns `true` when `path` carries the PDF extension (Python `_is_pdf`).
fn is_pdf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
}

/// Returns the size of `path` through the bounded read, or `None` when the backend cannot bound
/// a read (it keeps its own whole-file policy in that case).
///
/// A zero-byte window asks the backend only for the metadata facts, so the file is never loaded
/// just to learn how large it is.
async fn bounded_file_size(
    fs: &dyn FsOperation,
    path: &str,
) -> std::result::Result<Option<u64>, String> {
    match fs
        .read_bytes_limited(ReadBytesParams::new(path, 0, 0))
        .await
    {
        Ok(data) => Ok(Some(data.total_bytes)),
        Err(error) if error.kind() == ErrorKind::Unsupported => Ok(None),
        Err(error) => Err(error.message().into_owned()),
    }
}

/// Reads the line window `[offset, offset + limit)` without loading the whole file.
///
/// The scan is bounded by [`TEXT_SCAN_MAX_BYTES`] and by [`READ_CHUNK_BYTES`] per read, so the
/// memory a read needs is a function of the window, not of the file. Lines found before the
/// budget runs out are returned; a window whose offset lies beyond the budget is a typed
/// failure instead of an empty answer.
async fn read_text_window(
    fs: &dyn FsOperation,
    path: &str,
    offset: usize,
    limit: usize,
) -> std::result::Result<(String, u64), String> {
    let mut collected: Vec<String> = Vec::with_capacity(limit.min(64));
    let mut pending: Vec<u8> = Vec::new();
    // 1-based number of the line `pending` starts.
    let mut next_line = 1usize;
    let mut position = 0u64;
    // Assigned by the first iteration, which every path enters before it can break.
    let mut total_bytes: u64;
    let mut scanned = 0u64;
    let mut eof = false;
    let mut budget_exhausted = false;
    loop {
        let chunk = fs
            .read_bytes_limited(ReadBytesParams::new(path, position, READ_CHUNK_BYTES))
            .await
            .map_err(|error| error.message().into_owned())?;
        total_bytes = chunk.total_bytes;
        if chunk.bytes.is_empty() {
            eof = true;
            break;
        }
        position = position.saturating_add(chunk.bytes.len() as u64);
        scanned = scanned.saturating_add(chunk.bytes.len() as u64);
        pending.extend_from_slice(&chunk.bytes);

        let mut range_start = 0usize;
        let mut done = false;
        for (index, byte) in pending.iter().enumerate() {
            if *byte != b'\n' {
                continue;
            }
            let line = strip_line_ending(&pending[range_start..index]);
            if next_line > offset && collected.len() < limit {
                collected.push(String::from_utf8_lossy(line).into_owned());
            }
            next_line = next_line.saturating_add(1);
            range_start = index + 1;
            if collected.len() >= limit {
                done = true;
                break;
            }
        }
        if range_start > 0 {
            pending.drain(..range_start);
        }
        if done || !chunk.truncated {
            eof = !done;
            break;
        }
        if scanned >= TEXT_SCAN_MAX_BYTES
            || u64::try_from(pending.len()).unwrap_or(u64::MAX) >= TEXT_SCAN_MAX_BYTES
        {
            budget_exhausted = true;
            break;
        }
    }

    if eof && !pending.is_empty() && collected.len() < limit && next_line > offset {
        collected.push(String::from_utf8_lossy(&pending).into_owned());
    }
    if budget_exhausted && collected.is_empty() {
        return Err(offset_beyond_budget_message(offset));
    }
    Ok((collected.join("\n"), total_bytes))
}

/// Strips one trailing `\r` so a CRLF line matches what `str::lines` reports.
fn strip_line_ending(line: &[u8]) -> &[u8] {
    match line.last() {
        Some(b'\r') => &line[..line.len() - 1],
        _ => line,
    }
}

/// Builds the rejection of an offset beyond the bounded scan budget.
fn offset_beyond_budget_message(offset: usize) -> String {
    format!(
        "Error: The requested offset {offset} lies more than {} MiB into the file. Use grep to \
         locate the content and read a bounded window around it.",
        TEXT_SCAN_MAX_BYTES / (1024 * 1024)
    )
}

/// Builds the rejection of a PDF larger than [`PDF_MAX_SIZE_BYTES`].
#[allow(clippy::cast_precision_loss)]
fn pdf_oversized_message(total_bytes: u64) -> String {
    format!(
        "Error: PDF content ({:.1}MB) exceeds the maximum allowed PDF size ({}MB). Split the \
         document or extract the pages you need before reading it.",
        total_bytes as f64 / (1024.0 * 1024.0),
        PDF_MAX_SIZE_BYTES / (1024 * 1024)
    )
}

/// Reads a PDF's bytes with a hard size ceiling.
///
/// The bounded read caps the allocation at [`PDF_MAX_SIZE_BYTES`] even when the file is far
/// larger; a backend without a bounded read falls back to its own whole-file policy.
async fn read_pdf_bytes(fs: &dyn FsOperation, path: &Path) -> std::result::Result<Vec<u8>, String> {
    let display = path.to_string_lossy().into_owned();
    match fs
        .read_bytes_limited(ReadBytesParams::new(display.clone(), 0, PDF_MAX_SIZE_BYTES))
        .await
    {
        Ok(data) if data.truncated => Err(pdf_oversized_message(data.total_bytes)),
        Ok(data) => Ok(data.bytes),
        Err(error) if error.kind() == ErrorKind::Unsupported => read_bytes(fs, path).await,
        Err(error) => Err(error.message().into_owned()),
    }
}

/// Reads the whole file in byte mode through the filesystem backend.
///
/// `Err` carries the model-facing failure message, so the caller keeps ownership of the
/// payload envelope (and this helper avoids a 150-byte `Err` variant).
async fn read_bytes(fs: &dyn FsOperation, path: &Path) -> std::result::Result<Vec<u8>, String> {
    let params = ReadFileParams {
        path: path.to_string_lossy().into_owned(),
        mode: ReadMode::Bytes,
        ..ReadFileParams::default()
    };
    let result = fs.read_file(params).await;
    if result.is_failure() {
        return Err(result.message);
    }
    let Some(data) = result.data else {
        return Err("read_file returned no data from the filesystem operation".to_owned());
    };
    Ok(match data.content {
        FileContent::Bytes(bytes) => bytes,
        FileContent::Text(text) => text.into_bytes(),
    })
}

/// Renders the `pages` of `document` as `## Page {n}` blocks with per-page line numbers.
///
/// Returns the rendered content and the number of text lines it contains (page markers and the
/// no-text note do not count).
fn render_pages(document: &PdfDocument, pages: &[usize]) -> (String, usize) {
    let mut blocks = Vec::with_capacity(pages.len());
    let mut line_count = 0usize;
    for page in pages {
        let text = document.pages.get(page - 1).map_or("", String::as_str);
        if text.is_empty() {
            blocks.push(format!(
                "## Page {page}\n(no extractable text on this page)"
            ));
        } else {
            let numbered = ReadFileTool::number_lines(text);
            line_count += numbered.lines().count();
            blocks.push(format!("## Page {page}\n{numbered}"));
        }
    }
    (blocks.join("\n\n"), line_count)
}

/// Renders ascending page numbers as a compact range string (`[1, 2, 3, 7]` -> `"1-3,7"`).
fn normalized_range(pages: &[usize]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut remaining = pages.iter().copied().peekable();
    while let Some(start) = remaining.next() {
        let mut end = start;
        while remaining
            .peek()
            .is_some_and(|next| *next == end.saturating_add(1))
        {
            end = end.saturating_add(1);
            remaining.next();
        }
        if start == end {
            parts.push(start.to_string());
        } else {
            parts.push(format!("{start}-{end}"));
        }
    }
    parts.join(",")
}

/// Builds the Python "file too large" message for a `total_bytes` file.
#[allow(clippy::cast_precision_loss)]
fn oversized_message(total_bytes: u64) -> String {
    format!(
        "Error: File content ({:.1}KB) exceeds maximum allowed size ({}KB). Use grep for specific content or read a smaller offset/limit range.",
        total_bytes as f64 / 1024.0,
        MAX_SIZE_BYTES / 1024
    )
}

#[async_trait]
impl Tool for ReadFileTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.read(&inputs).await.to_json())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};
    use tempfile::TempDir;

    use super::*;
    use crate::tools::read_state::WriteGate;
    use crate::workspace::Workspace;
    use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

    /// Builds a tool over a real local backend rooted at a fresh temp directory.
    fn setup() -> (ReadFileTool, Arc<ReadStateRegistry>, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let operation = SysOperation::local(LocalWorkConfig::unrestricted(
            dir.path().to_string_lossy().to_string(),
        ))
        .unwrap();
        let fs = operation.fs();
        let state = ReadStateRegistry::new();
        let tool = ReadFileTool::new(
            ToolEnv::new(Workspace::new(dir.path())),
            fs,
            Arc::clone(&state),
        );
        (tool, state, dir)
    }

    /// Invokes the tool with a fresh context and unwraps the payload envelope.
    async fn invoke(tool: &ReadFileTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// Builds a minimal PDF whose content streams hold `pages`, one stream per page.
    ///
    /// The retrieval parser walks `stream … endstream` regions in file order, so a plain
    /// `BT (text) Tj ET` stream is a complete page as far as text extraction is concerned.
    fn pdf_bytes(pages: &[&str]) -> Vec<u8> {
        let mut document = String::from("%PDF-1.4\n");
        for text in pages {
            let stream = format!("BT ({text}) Tj ET\n");
            let block = format!(
                "<< /Length {} >>\nstream\n{stream}endstream\n",
                stream.len()
            );
            document.push_str(&block);
        }
        document.push_str("%%EOF\n");
        document.into_bytes()
    }

    /// Writes `pages` as a PDF at `name` inside `dir` and returns its payload path.
    fn write_pdf(dir: &TempDir, name: &str, pages: &[&str]) {
        std::fs::write(dir.path().join(name), pdf_bytes(pages)).unwrap();
    }

    #[test]
    fn card_exposes_python_arguments() {
        let card = ReadFileTool::card_for("en");
        assert_eq!(card.name, "read_file");
        assert!(card.parallel_safe);
        assert_eq!(card.input_params["required"], json!(["file_path"]));

        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["caption", "file_path", "limit", "offset", "pages"],
            "unexpected property set: {keys:?}"
        );
        assert_eq!(properties["offset"]["type"], json!("integer"));
        assert_eq!(properties["limit"]["type"], json!("integer"));
    }

    #[test]
    fn number_lines_formats_six_wide_numbers_and_tabs() {
        assert_eq!(
            ReadFileTool::number_lines("alpha\nbeta"),
            "     1\talpha\n     2\tbeta"
        );
        assert_eq!(ReadFileTool::number_lines(""), "");
    }

    #[test]
    fn page_range_parses_single_pages_ranges_and_unions() {
        assert_eq!(PageRange::parse("3").unwrap().pages, vec![3]);
        assert_eq!(
            PageRange::parse(" 1-5 ").unwrap().pages,
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(PageRange::parse("1-3,7").unwrap().pages, vec![1, 2, 3, 7]);
        assert_eq!(PageRange::parse("7,1-3").unwrap().pages, vec![1, 2, 3, 7]);
        assert_eq!(PageRange::parse("2,2,3").unwrap().pages, vec![2, 3]);
        assert_eq!(PageRange::parse("1-3,2-4").unwrap().pages, vec![1, 2, 3, 4]);
        // Exactly the cap is fine; deduplication happens before the cap is checked.
        assert_eq!(PageRange::parse("1-20").unwrap().pages.len(), 20);
        assert_eq!(PageRange::parse("1-15,5-10").unwrap().pages.len(), 15);
    }

    #[test]
    fn page_range_rejects_every_malformed_selection() {
        let cases = [
            ("", "empty"),
            ("   ", "empty"),
            ("abc", "is not a page number"),
            ("1,", "empty element"),
            ("1,,2", "empty element"),
            ("-", "missing a page number"),
            ("1-", "missing a page number"),
            ("-5", "missing a page number"),
            ("0", "page 0 is not a page"),
            ("0-2", "page 0 is not a page"),
            ("1-0", "page 0 is not a page"),
            ("5-2", "reversed"),
            ("1-3,-2", "missing a page number"),
            ("1.5", "is not a page number"),
            ("1-21", "exceeds the maximum of 20 pages"),
        ];
        for (raw, expected) in cases {
            let error = PageRange::parse(raw).unwrap_err();
            assert!(
                error.contains(expected),
                "selection {raw:?} produced {error:?}, expected {expected:?}"
            );
            assert!(
                error.contains("1-indexed") || error.contains("20 pages"),
                "selection {raw:?} lost its hint: {error:?}"
            );
        }
    }

    #[test]
    fn page_range_resolution_keeps_only_pages_the_document_has() {
        let range = PageRange::parse("1,9").unwrap();
        assert_eq!(range.resolve(3), vec![1]);
        let range = PageRange::parse("2-9").unwrap();
        assert_eq!(range.resolve(3), vec![2, 3]);
        let range = PageRange::parse("5,9").unwrap();
        assert_eq!(range.resolve(3), Vec::<usize>::new());
        assert_eq!(range.raw(), "5,9");
    }

    #[test]
    fn normalized_range_groups_consecutive_pages() {
        assert_eq!(normalized_range(&[1, 2, 3, 7]), "1-3,7");
        assert_eq!(normalized_range(&[4]), "4");
        assert_eq!(normalized_range(&[1, 2]), "1-2");
        assert_eq!(normalized_range(&[1, 3, 5]), "1,3,5");
        assert_eq!(normalized_range(&[]), "");
    }

    #[tokio::test]
    async fn reads_a_five_line_file_with_line_numbers() {
        let (tool, _state, dir) = setup();
        std::fs::write(
            dir.path().join("notes.txt"),
            "one\ntwo\nthree\nfour\nfive\n",
        )
        .unwrap();

        let out = invoke(&tool, json!({"file_path": "notes.txt"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("     1\tone\n     2\ttwo\n     3\tthree\n     4\tfour\n     5\tfive")
        );
        assert_eq!(out["data"]["line_count"], json!(5));
        let path = out["data"]["file_path"].as_str().unwrap();
        assert!(path.ends_with("notes.txt"), "unexpected path: {path}");
        assert_eq!(out["error"], Value::Null);
    }

    #[tokio::test]
    async fn offset_and_limit_read_a_window_and_register_partial_coverage() {
        let (tool, state, dir) = setup();
        let full_content = "one\ntwo\nthree\nfour\nfive\n";
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, full_content).unwrap();

        let out = invoke(
            &tool,
            json!({"file_path": "notes.txt", "offset": 2, "limit": 2}),
        )
        .await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"]["content"], json!("     1\tthree\n     2\tfour"));
        assert_eq!(out["data"]["line_count"], json!(2));

        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        assert_eq!(
            state.write_gate(&path, fingerprint, full_content),
            WriteGate::Partial {
                lines_read: 2,
                total_lines: 0,
                resume_offset: 0,
            }
        );
    }

    #[tokio::test]
    async fn pdf_single_page_selection_renders_its_marker_and_line_numbers() {
        let (tool, _state, dir) = setup();
        write_pdf(
            &dir,
            "doc.pdf",
            &["first page", "second page", "third page"],
        );

        let out = invoke(&tool, json!({"file_path": "doc.pdf", "pages": "2"})).await;

        assert_eq!(out["success"], json!(true), "payload: {out}");
        assert_eq!(out["data"]["total_pages"], json!(3));
        assert_eq!(out["data"]["selected_pages"], json!([2]));
        assert_eq!(out["data"]["page_range"], json!("2"));
        assert_eq!(out["data"]["line_count"], json!(1));
        assert_eq!(
            out["data"]["content"],
            json!("## Page 2\n     1\tsecond page")
        );
        let path = out["data"]["file_path"].as_str().unwrap();
        assert!(path.ends_with("doc.pdf"), "unexpected path: {path}");
    }

    #[tokio::test]
    async fn pdf_range_and_comma_union_select_exactly_those_pages() {
        let (tool, _state, dir) = setup();
        write_pdf(&dir, "doc.pdf", &["alpha", "beta", "gamma", "delta"]);

        let range = invoke(&tool, json!({"file_path": "doc.pdf", "pages": "1-2"})).await;
        assert_eq!(range["success"], json!(true), "payload: {range}");
        assert_eq!(range["data"]["selected_pages"], json!([1, 2]));
        assert_eq!(
            range["data"]["content"],
            json!("## Page 1\n     1\talpha\n\n## Page 2\n     1\tbeta")
        );
        assert_eq!(range["data"]["line_count"], json!(2));

        let union = invoke(&tool, json!({"file_path": "doc.pdf", "pages": "1,4"})).await;
        assert_eq!(union["success"], json!(true), "payload: {union}");
        assert_eq!(union["data"]["selected_pages"], json!([1, 4]));
        assert_eq!(union["data"]["page_range"], json!("1,4"));
        assert!(!union["data"]["content"].as_str().unwrap().contains("beta"));
        assert!(union["data"]["content"]
            .as_str()
            .unwrap()
            .contains("## Page 4\n     1\tdelta"));
    }

    #[tokio::test]
    async fn pdf_read_without_pages_reads_the_whole_document() {
        let (tool, _state, dir) = setup();
        write_pdf(&dir, "doc.pdf", &["one page", "two page"]);

        let out = invoke(&tool, json!({"file_path": "doc.pdf"})).await;

        assert_eq!(out["success"], json!(true), "payload: {out}");
        assert_eq!(out["data"]["total_pages"], json!(2));
        assert_eq!(out["data"]["selected_pages"], json!([1, 2]));
        assert_eq!(out["data"]["page_range"], json!("1-2"));
        assert_eq!(out["data"]["line_count"], json!(2));
    }

    #[tokio::test]
    async fn malformed_page_specs_are_typed_failures() {
        let (tool, _state, dir) = setup();
        write_pdf(&dir, "doc.pdf", &["one", "two", "three"]);

        for (spec, expected) in [
            ("", "the selection is empty"),
            ("abc", "is not a page number"),
            ("0", "page 0 is not a page"),
            ("3-1", "reversed"),
            ("1-21", "exceeds the maximum of 20 pages"),
        ] {
            let out = invoke(&tool, json!({"file_path": "doc.pdf", "pages": spec})).await;
            assert_eq!(out["success"], json!(false), "spec {spec:?}: {out}");
            let error = out["error"].as_str().unwrap();
            assert!(
                error.contains(expected),
                "spec {spec:?} produced {error:?}, expected {expected:?}"
            );
            assert!(
                error.contains("20 pages") || error.contains("1-indexed"),
                "spec {spec:?} lost its hint: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn more_than_twenty_pages_without_a_selection_names_the_cap() {
        let (tool, _state, dir) = setup();
        let pages: Vec<String> = (1..=21).map(|index| format!("page {index}")).collect();
        let borrowed: Vec<&str> = pages.iter().map(String::as_str).collect();
        write_pdf(&dir, "long.pdf", &borrowed);

        let out = invoke(&tool, json!({"file_path": "long.pdf"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(error.contains("21 pages"), "unexpected error: {error}");
        assert!(
            error.contains("Maximum 20 pages per request"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn pages_outside_the_document_are_rejected() {
        let (tool, _state, dir) = setup();
        write_pdf(&dir, "doc.pdf", &["one", "two"]);

        let out = invoke(&tool, json!({"file_path": "doc.pdf", "pages": "9"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(
            error.contains("selects no page"),
            "unexpected error: {error}"
        );
        assert!(error.contains("2 page(s)"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn pages_on_a_text_file_is_rejected() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("notes.txt"), "one\ntwo\n").unwrap();

        let out = invoke(&tool, json!({"file_path": "notes.txt", "pages": "1"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(error.contains("pages only applies to PDF"), "{error}");
        assert!(error.contains("notes.txt"), "{error}");
    }

    #[tokio::test]
    async fn caption_is_rejected_with_the_unsupported_capability() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("notes.txt"), "one\n").unwrap();
        write_pdf(&dir, "doc.pdf", &["one"]);

        for path in ["notes.txt", "doc.pdf"] {
            let out = invoke(
                &tool,
                json!({"file_path": path, "caption": "a red bicycle"}),
            )
            .await;
            assert_eq!(out["success"], json!(false), "path {path}: {out}");
            let error = out["error"].as_str().unwrap();
            assert!(error.contains("caption is not supported"), "{error}");
            assert!(error.contains("image_metadata"), "{error}");
            assert!(error.contains("vision model"), "{error}");
        }
    }

    #[tokio::test]
    async fn pdf_without_extractable_text_reports_the_missing_ocr() {
        let (tool, _state, dir) = setup();
        // No `pages`, no streams: a scanned document as far as the parser is concerned.
        std::fs::write(dir.path().join("scan.pdf"), b"%PDF-1.4\n%%EOF\n").unwrap();

        let out = invoke(&tool, json!({"file_path": "scan.pdf", "pages": "1"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(error.contains("no extractable text"), "{error}");
        assert!(error.contains("OCR is not available"), "{error}");
        assert!(error.contains("0 page(s)"), "{error}");
    }

    #[tokio::test]
    async fn pdf_offset_and_limit_are_rejected_instead_of_dropped() {
        let (tool, _state, dir) = setup();
        write_pdf(&dir, "doc.pdf", &["one", "two"]);

        for inputs in [
            json!({"file_path": "doc.pdf", "offset": 2}),
            json!({"file_path": "doc.pdf", "limit": 10}),
        ] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false), "payload: {out}");
            let error = out["error"].as_str().unwrap();
            assert!(error.contains("offset/limit do not apply"), "{error}");
            assert!(error.contains("pages"), "{error}");
        }
    }

    #[tokio::test]
    async fn pdf_page_read_does_not_register_full_text_coverage() {
        let (tool, state, dir) = setup();
        let path = dir.path().join("doc.pdf");
        write_pdf(&dir, "doc.pdf", &["alpha", "beta"]);

        let out = invoke(&tool, json!({"file_path": "doc.pdf", "pages": "1"})).await;
        assert_eq!(out["success"], json!(true), "payload: {out}");

        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        // The file is known (not `NotRead`) but nothing was covered, so a whole-file write is
        // still refused: the model has only seen page 1.
        assert_eq!(
            state.write_gate(&path, fingerprint, "alpha\nbeta\n"),
            WriteGate::Partial {
                lines_read: 0,
                total_lines: 0,
                resume_offset: 0,
            }
        );
    }

    #[tokio::test]
    async fn missing_file_path_is_a_failure_payload() {
        let (tool, _state, _dir) = setup();

        for inputs in [json!({}), json!({"file_path": ""})] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false));
            assert_eq!(out["error"], json!("file_path is required"));
        }
    }

    #[tokio::test]
    async fn nonexistent_file_returns_failure_without_panicking() {
        let (tool, _state, _dir) = setup();

        let out = invoke(&tool, json!({"file_path": "missing.txt"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(!error.is_empty(), "expected a failure message");
    }

    #[tokio::test]
    async fn nonexistent_pdf_returns_the_backend_failure() {
        let (tool, _state, _dir) = setup();

        let out = invoke(&tool, json!({"file_path": "missing.pdf", "pages": "1"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(!error.is_empty(), "expected a failure message");
    }

    #[tokio::test]
    async fn oversized_file_requires_an_explicit_limit() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("big.txt"), "x".repeat(300 * 1024)).unwrap();

        let out = invoke(&tool, json!({"file_path": "big.txt"})).await;
        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(
            error.contains("exceeds maximum allowed size (256KB)"),
            "unexpected error: {error}"
        );

        let limited = invoke(&tool, json!({"file_path": "big.txt", "limit": 10})).await;
        assert_eq!(limited["success"], json!(true));
    }

    #[tokio::test]
    async fn a_bounded_read_of_a_huge_sparse_file_returns_the_first_lines() {
        let (tool, _state, dir) = setup();
        let path = dir.path().join("huge.log");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"alpha\nbeta\ngamma\n").unwrap();
            file.flush().unwrap();
            // 2 GiB is larger than any allocation a read may make; the tail is a hole.
            file.set_len(2 * 1024 * 1024 * 1024).unwrap();
        }

        let out = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            invoke(&tool, json!({"file_path": "huge.log", "limit": 10})),
        )
        .await
        .expect("a bounded read must not load the file");

        assert_eq!(out["success"], json!(true), "unexpected payload: {out}");
        assert_eq!(out["data"]["line_count"], json!(3));
        assert_eq!(
            out["data"]["content"],
            json!("     1\talpha\n     2\tbeta\n     3\tgamma")
        );
    }

    #[tokio::test]
    async fn an_oversized_pdf_is_refused_before_it_is_parsed() {
        let (tool, _state, dir) = setup();
        let path = dir.path().join("huge.pdf");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"%PDF-1.4\n").unwrap();
            file.flush().unwrap();
            file.set_len(PDF_MAX_SIZE_BYTES + 1).unwrap();
        }

        let out = invoke(&tool, json!({"file_path": "huge.pdf"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(
            error.contains("exceeds the maximum allowed PDF size"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn empty_file_returns_the_python_warning() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();

        let out = invoke(&tool, json!({"file_path": "empty.txt"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("Warning: the file exists but the contents are empty.")
        );
        assert_eq!(out["data"]["line_count"], json!(0));
    }

    #[tokio::test]
    async fn offset_past_end_of_file_returns_python_message() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("short.txt"), "a\nb\nc\n").unwrap();

        let out = invoke(&tool, json!({"file_path": "short.txt", "offset": 99})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("Error: The file exists but no content was returned at offset 99. Use a smaller offset or read from the beginning.")
        );
        assert_eq!(out["data"]["line_count"], json!(0));
    }

    #[tokio::test]
    async fn full_read_is_visible_through_the_write_gate() {
        let (tool, state, dir) = setup();
        let full_content = "one\ntwo\nthree\nfour\nfive\n";
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, full_content).unwrap();

        let out = invoke(&tool, json!({"file_path": "notes.txt"})).await;
        assert_eq!(out["success"], json!(true));

        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        assert_eq!(
            state.write_gate(&path, fingerprint, full_content),
            WriteGate::Ok
        );
    }

    #[tokio::test]
    async fn missing_filesystem_backend_is_a_failure_payload() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(
            ToolEnv::new(Workspace::new(dir.path())),
            None,
            ReadStateRegistry::new(),
        );

        let out = invoke(&tool, json!({"file_path": "a.txt"})).await;

        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("filesystem operation is not configured")
        );
    }
}
