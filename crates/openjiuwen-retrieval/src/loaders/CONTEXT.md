# loaders — document sources

## Intent

Port the Python `retrieval/indexing/processor/parser/` subset that can run offline: plain text, Markdown, JSON, CSV/TSV, local HTML, HTTP(S) URLs, DOCX, XLSX, PDF, and images, plus the auto-dispatch/registry behavior of `AutoFileParser`/`AutoLinkParser`/`AutoParser`.
The binary formats are read in-crate by `crate::parsing` (`DocxParser`, `XlsxParser`, `PdfParser`) and images by `crate::multimodal::ImageLoader`, so no external office/PDF crate or vision model is needed.
WeChat articles and article-extraction services remain out of scope; an extension with no registered loader returns `Error::Unsupported`.

## Formats and dispatch

| Extension | Loader | Output |
| --- | --- | --- |
| `.txt` | `TextLoader` | one document, trimmed text |
| `.md`, `.markdown` | `MarkdownLoader` | one document, trimmed text |
| `.json` | `JsonLoader` | one document, pretty-printed JSON (raw text when malformed) |
| `.csv`, `.tsv` | `CsvLoader` | row documents then column documents |
| `.html`, `.htm` | `HtmlLoader` | one web-page document |
| `.docx` | `DocxLoader` | one document, rendered Word body |
| `.xlsx` | `XlsxLoader` | row documents then column documents per sheet |
| `.pdf` | `PdfLoader` | one document, extracted page text |
| `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.jfif` | `ImageLoader` | one captioned document |
| `http(s)://…` | `UrlLoader` | one web-page or plain-text document |
| anything else | — | `Error::Unsupported("Unsupported format: .ext, only [...] are supported")` |

- Registry lookup uses the lowercase final extension (`file_extension`), so `REPORT.DOCX` and `report.docx` share one loader.
- `LoaderRegistry::with_defaults()` registers all rows above; the first registration for an extension wins, so later calls cannot shadow a default.

## API Surface

### `mod.rs`
- `#[async_trait] pub trait DocumentLoader: Send + Sync { fn supports(&self, source: &str) -> bool; async fn load(&self, source: &str) -> Result<Vec<Document>>; }`.
- Re-exports every loader plus `html_to_text`/`extract_title`.
- Shared helpers: `generate_document_id()`, `file_extension(source)`, `is_url(source)`, `file_stem(source)`, `read_source(source)` (lossy UTF-8).
- `#[cfg(test)] mod fixtures` holds `build_zip`, `pdf_document`, and `plain_stream` for the loader unit tests.

### `text.rs`
- `TextLoader` (`.txt`) and `MarkdownLoader` (`.md`, `.markdown`); both read UTF-8 (lossy fallback) and return one `Document` with `text.trim()`; empty content → `Error::Validation`; missing file → `Error::NotFound`.

### `json.rs`
- `JsonLoader` (`.json`): parse then `serde_json::to_string_pretty`; malformed JSON returns the raw file text (Python parity); one document.

### `csv.rs`
- `CsvLoader` (`.csv`, `.tsv`) with `CsvLoader::new()` / `with_include_header(bool)`; delimiter from extension or `with_delimiter(u8)`.
- RFC-4180-style parser (quoted fields, escaped quotes, CRLF) implemented locally.
- Header = first row (or `column_{i}` when `include_header=false`); `sheet_name` = file stem.
- Row documents: `text = "H: v, H2: v2"`, `id = "{base}_s{sheet}_r{row}"` (row 1-based including header; first data row = 2), metadata `{sheet_name, row_index, source_type: "row"}`.
- Column documents: `text = "Column name: X. Values: a, b"`, `id = "{base}_s{sheet}_c{col}"` (0-based), metadata adds `{column_name, source_type: "column"}`.
- `base` is the `source` path as passed to `load`; empty file → `[]`; a header-only file still yields column documents.

### `html.rs`
- `pub fn html_to_text(html: &str) -> String` — strips `<script>`, `<style>`, comments and tags, decodes named/numeric entities, collapses whitespace; `extract_title(html) -> Option<String>` prefers `og:title` then `<title>`.
- `HtmlLoader` (`.html`, `.htm`): one document with metadata `{title, source_type: "web_page"}`; no extractable text → `Error::Validation`.

### `url.rs`
- `UrlLoader` using `openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest}`; defaults: 30s timeout, Chrome-120 `User-Agent`, `verify_ssl=true`.
- `UrlLoader::new()`, `with_client(HttpClient)`, `with_timeout(Duration)`.
- Non-HTTP(S) scheme → `Error::Unsupported`; non-2xx → `Error::Retrieval`; body converted with `html_to_text` when the content type is HTML or the body starts with `<`; otherwise plain text.
- Document id = provided id or the URL; metadata `{source_url, title, source_type: "web_page"}`.

### `docx.rs`
- `DocxLoader` (`.docx`) parses with `DocxParser::parse_path` (async `tokio::fs`) and returns exactly one `Document`.
- `text` is `DocxDocument::text` (Markdown-flavoured headings, paragraphs, and tables joined by newlines).
- Metadata: `title` from `dc:title` when present, `paragraph_count`, `table_count`, `source_type: "document"`.
- Missing file → `Error::NotFound`; corrupt container → `Error::Validation`; archive without `word/document.xml` → `Error::Retrieval`.

### `xlsx.rs`
- `XlsxLoader` (`.xlsx`) parses with `XlsxParser::parse_path` and returns `XlsxWorkbook::to_documents(file_stem(source))`.
- Ids are `{stem}_s{sheet}_r{row}` (1-based, header included) and `{stem}_s{sheet}_c{column}` (0-based); metadata keeps `{sheet_name, row_index?, column_name?, source_type}`.
- An empty workbook (no sheets) returns `Ok(vec![])`; missing file → `Error::NotFound`; missing `xl/workbook.xml` → `Error::Retrieval`.

### `pdf.rs`
- `PdfLoader` (`.pdf`) parses with `PdfParser::parse_path` and returns one `Document` whose text is `PdfDocument::text`.
- Metadata: `page_count`, `has_text`, `source_type: "document"`.
- A PDF with no extractable text (scanned/image-only) → `Error::Unsupported("PDF has no extractable text: <path>")`.
- Encrypted PDFs and object-stream PDFs propagate the parser's `Error::Unsupported` messages; missing file → `Error::NotFound`.

### `auto.rs`
- `LoaderRegistry`: normalized extension map (with leading dot); `with_defaults()` registers the text/Markdown/JSON/CSV/TSV/HTML/DOCX/XLSX/PDF/image loaders (17 extensions); `register(extensions, loader)` first registration wins (Python parity); `get_for_path(path) -> Option<Arc<dyn DocumentLoader>>`; `extensions()` returns the sorted list.
- `AutoLoader { registry, url_loader }`: `new()`/`Default`, `with_url_loader`, `register`, `registry`; `load(&source)` and `load_as(&source, doc_id, title)`.
- Dispatch: URL → `UrlLoader`; known extension → registered loader; missing path → `Error::NotFound("File X does not exist")`; unregistered extension (for example `.xls`, `.zip`, `.xyz`) → `Error::Unsupported("Unsupported format: .ext, only [...] are supported")`.
- File results are stamped (overwriting child keys): `metadata.doc_id = document.id`, `metadata.title = title or file stem`, `metadata.file_path = source`, `metadata.file_ext = lowercase extension with dot`; empty ids are replaced by a generated uuid4.
- `load_as` semantics: single-document loaders (text, Markdown, JSON, HTML, DOCX, PDF, image) take `doc_id` verbatim; multi-document loaders (CSV/TSV keyed by the full path, XLSX keyed by the file stem) use it as the id base prefix, so every generated id keeps its `_s{sheet}_r{row}`/`_s{sheet}_c{col}` suffix and stays distinct.
- `impl DocumentLoader for AutoLoader` with `supports` true for URLs and registered extensions.

## Constraints

- All loaders are async and use `tokio::fs` for files; the URL loader uses the shared async HTTP client.
- Never panic on malformed input; return `Error::Validation`/`NotFound`/`Unsupported`/`Retrieval`.
- Keep each loader ≤300 lines; shared parsing helpers stay private in their module and test-only binary fixtures live in `mod.rs`'s `fixtures`.
- Tests use `tempfile` and `wiremock`; no network.

## Test Strategy

`TextLoader` missing/empty; JSON valid (pretty) and malformed (raw); CSV row/column text, ids, metadata, header-only and `include_header=false`; HTML title precedence, script/style removal, entity decoding, empty content error; URL success (HTML and plain text) and non-2xx; auto dispatch, metadata stamping, unknown extension, missing files per format.
Each loader additionally has in-module unit tests over in-memory fixtures: DOCX renders headings/paragraphs/tables with `dc:title`, `paragraph_count`, and `table_count`; XLSX emits row/column documents keyed by the file stem and returns `[]` for an empty workbook; PDF returns page metadata and rejects scanned and encrypted files; corrupt archives propagate the parser's typed errors.
`tests/loaders.rs` builds minimal stored-entry ZIP (DOCX/XLSX) and uncompressed-content-stream PDF fixtures in-test, so it covers `AutoLoader` dispatch for all binary formats plus image captioning, XLSX `doc_id` base-prefix handling, scanned-PDF rejection, and the 17-extension default registry.
Registry first-wins and case-insensitive extensions are covered in both the unit and integration suites.

## Routing Table

- `./mod.rs` → `DocumentLoader` trait, shared helpers, re-exports, test fixtures
- `./text.rs` → text and Markdown loaders
- `./json.rs` → JSON loader
- `./csv.rs` → CSV/TSV loader (row + column documents)
- `./html.rs` → HTML-to-text conversion and HTML file loader
- `./url.rs` → HTTP(S) loader
- `./docx.rs` → Word loader over `parsing::DocxParser`
- `./xlsx.rs` → Excel loader over `parsing::XlsxParser`
- `./pdf.rs` → PDF loader over `parsing::PdfParser`
- `./auto.rs` → extension/URL dispatch, registry, metadata stamping
