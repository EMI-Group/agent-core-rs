# loaders — document sources

## Intent

Port the Python `retrieval/indexing/processor/parser/` subset that can run offline: plain text, Markdown, JSON, CSV/TSV, local HTML, and HTTP(S) URLs, plus the auto-dispatch/registry behavior of `AutoFileParser`/`AutoLinkParser`/`AutoParser`.
PDF, DOCX, XLSX, images, WeChat articles, and article-extraction services are out of scope and return `Error::Unsupported`.

## API Surface

### `mod.rs`
- `#[async_trait] pub trait DocumentLoader: Send + Sync { fn supports(&self, source: &str) -> bool; async fn load(&self, source: &str) -> Result<Vec<Document>>; }`.
- Re-exports all loaders and `html_to_text`.
- Shared helpers: `generate_document_id()`, `file_extension(source)`, `is_url(source)`.

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
- Empty file → `[]`; a header-only file still yields column documents.

### `html.rs`
- `pub fn html_to_text(html: &str) -> String` — strips `<script>`, `<style>`, comments and tags, decodes named/numeric entities, collapses whitespace; `extract_title(html) -> Option<String>` prefers `og:title` then `<title>`.
- `HtmlLoader` (`.html`, `.htm`): one document with metadata `{title, source_type: "web_page"}`; no extractable text → `Error::Validation`.

### `url.rs`
- `UrlLoader` using `openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest}`; defaults: 30s timeout, Chrome-120 `User-Agent`, `verify_ssl=true`.
- `UrlLoader::new()`, `with_client(HttpClient)`, `with_timeout(Duration)`.
- Non-HTTP(S) scheme → `Error::Unsupported`; non-2xx → `Error::Retrieval`; body converted with `html_to_text` when the content type is HTML or the body starts with `<`; otherwise plain text.
- Document id = provided id or the URL; metadata `{source_url, title, source_type: "web_page"}`.

### `auto.rs`
- `LoaderRegistry`: normalized extension map (with leading dot), `with_defaults()` registers `.txt/.md/.markdown/.json/.csv/.tsv/.html/.htm`; `register(extensions, loader)` first registration wins (Python parity); `get_for_path(path) -> Option<Arc<dyn DocumentLoader>>`.
- `AutoLoader { registry, url_loader }`: `new()`/`Default`, `with_url_loader`, `register`; `load(&source)` and `load_as(&source, doc_id, title)`.
- Dispatch: URL → `UrlLoader`; known extension → loader; missing path → `Error::NotFound("File X does not exist")`; unsupported extension (including `.pdf`, `.docx`, `.xls`, `.xlsx`) → `Error::Unsupported("Unsupported format: .pdf, only [...] are supported")`.
- File results are stamped (overwriting child keys): `metadata.doc_id = document.id`, `metadata.title = title or file stem`, `metadata.file_path = source`, `metadata.file_ext = lowercase extension with dot`; empty ids are replaced by a generated uuid4.
- `impl DocumentLoader for AutoLoader` with `supports` true for URLs and registered extensions.

## Constraints

- All loaders are async and use `tokio::fs` for files; the URL loader uses the shared async HTTP client.
- Never panic on malformed input; return `Error::Validation`/`NotFound`/`Unsupported`/`Retrieval`.
- Keep each loader ≤300 lines; shared parsing helpers stay private in their module.
- Tests use `tempfile` and `wiremock`; no network.

## Test Strategy

`TextLoader` missing/empty; JSON valid (pretty) and malformed (raw); CSV row/column text, ids, metadata, header-only and `include_header=false`; HTML title precedence, script/style removal, entity decoding, empty content error; URL success (HTML and plain text) and non-2xx; auto dispatch, metadata stamping, unknown extension, PDF/DOCX unsupported; registry first-wins and case-insensitive extensions.

## Routing Table

- `./mod.rs` → `DocumentLoader` trait, constants, shared helpers, re-exports
- `./text.rs` → text and Markdown loaders
- `./json.rs` → JSON loader
- `./csv.rs` → CSV/TSV loader (row + column documents)
- `./html.rs` → HTML-to-text conversion and HTML file loader
- `./url.rs` → HTTP(S) loader
- `./auto.rs` → extension/URL dispatch, registry, metadata stamping
