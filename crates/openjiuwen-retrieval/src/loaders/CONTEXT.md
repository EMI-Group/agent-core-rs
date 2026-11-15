# loaders — document sources

## Intent

Port the Python `retrieval/indexing/processor/parser/` subset that can run offline: plain text, Markdown, JSON, CSV/TSV, local HTML, HTTP(S) URLs, DOCX, XLSX, PPTX, ODT, PDF, and images, plus the auto-dispatch/registry behavior of `AutoFileParser`/`AutoLinkParser`/`AutoParser`.
The binary formats are read in-crate by `crate::parsing` (`DocxParser`, `XlsxParser`, `PptxParser`, `OdtParser`, `PdfParser`) and images by `crate::multimodal::ImageLoader`, so no external office/PDF crate and no vision model is needed.
Sources can also be loaded from memory: every crate loader accepts `DocumentBytes` through `DocumentLoader::load_bytes`, which parses the bytes with the same parser and applies the same `LoadOptions`.
WeChat articles, article-extraction services, and LLM/OCR document understanding remain out of scope; an extension with no registered loader returns `Error::Unsupported`.

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
| `.pptx` | `PptxLoader` | one document, rendered deck (`## Slide N` blocks) |
| `.odt` | `OdtLoader` | one document, rendered body |
| `.pdf` | `PdfLoader` | one document, extracted page text |
| `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.jfif` | `ImageLoader` | one captioned document |
| `http(s)://…` | `UrlLoader` | one web-page or plain-text document |
| anything else | — | `Error::Unsupported("Unsupported format: .ext, only [...] are supported")` |

- Registry lookup uses the lowercase final extension (`file_extension`), so `REPORT.DOCX` and `report.docx` share one loader.
- `LoaderRegistry::with_defaults()` registers all rows above — 19 extensions: `.csv .docx .gif .htm .html .jfif .jpeg .jpg .json .markdown .md .odt .pdf .png .pptx .tsv .txt .webp .xlsx`.
- The first registration for an extension wins, so later calls cannot shadow a default.
- Byte dispatch (`AutoLoader::load_bytes`) is extension-only: a URL name has no registered loader and fails with the same `Unsupported` message.

## API Surface

### `mod.rs`
- `#[async_trait] pub trait DocumentLoader: Send + Sync` with `fn supports(&self, source: &str) -> bool`, `async fn load(&self, source: &str) -> Result<Vec<Document>>`, and two defaulted methods:
  - `async fn load_with_options(&self, source: &str, options: &LoadOptions) -> Result<Vec<Document>>` — `load` then `apply_load_options`.
  - `async fn load_bytes(&self, source: &DocumentBytes, options: &LoadOptions) -> Result<Vec<Document>>` — in-memory parse; the default returns `Error::Unsupported("<loader> does not support in-memory bytes: <name>")`, which keeps external implementors (and `ImageLoader`, which has no byte path) compiling.
- `pub struct LoadOptions { pub doc_id: Option<String>, pub title: Option<String> }` — serde snake_case, both fields defaulted and unknown fields tolerated; `new()`, `with_doc_id(impl Into<String>)`, `with_title(impl Into<String>)`; `Default + PartialEq + Eq`.
- `pub struct DocumentBytes { pub name: String, pub bytes: Vec<u8> }` — `new(name, impl Into<Vec<u8>>)`, `extension()` (lowercase, leading dot, `""` when absent), `source() -> &str`, `len()`, `is_empty()`; serde round-trips as `{"name":…,"bytes":[…]}`.
- Shared helpers (crate-private): `generate_document_id()`, `file_extension(source)`, `is_url(source)`, `file_stem(source)`, `read_source(source)` (lossy UTF-8), `compile_pattern(pattern)`, `apply_load_options(source, documents, options)`, `apply_doc_id(source, doc_id, id)`.
- Re-exports every loader plus `html_to_text`/`extract_title`; `fixtures` is public.

### Options semantics (shared by every loader)
- `doc_id` replaces a single-document id verbatim; for multi-document ids it replaces the base prefix and keeps the `_s{sheet}_r{row}`/`_s{sheet}_c{col}` suffix, so ids stay distinct.
- Every document touched by `apply_load_options` gets `metadata["doc_id"] = <final id>`, an empty id is replaced by a generated uuid4, and `title` (trimmed) overwrites `metadata["title"]` when provided.
- Path loading through `AutoLoader::load_as` additionally stamps `title` (provided title or file stem), `file_path`, and `file_ext` on every document, and leaves URL results untouched.
- Byte loading stamps the same provenance with `file_path` = the supplied name.

### `text.rs`
- `TextLoader` (`.txt`) and `MarkdownLoader` (`.md`, `.markdown`); both read UTF-8 (lossy fallback) or lossy-decoded bytes and return one `Document` with `text.trim()`; empty content → `Error::Validation("File <name> is empty")`; missing file → `Error::NotFound`.

### `json.rs`
- `JsonLoader` (`.json`): parse then `serde_json::to_string_pretty`; malformed JSON returns the raw text (Python parity); empty/whitespace content → `Error::Validation`; one document.

### `csv.rs`
- `CsvLoader` (`.csv`, `.tsv`) with `CsvLoader::new()` / `with_include_header(bool)` / `with_delimiter(u8)`; delimiter from extension (or override), leading BOM dropped.
- RFC-4180-style parser (quoted fields, escaped quotes, CRLF) implemented locally.
- Header = first row (or `column_{i}` when `include_header=false`); `sheet_name` = file stem.
- Row documents: `text = "H: v, H2: v2"`, `id = "{base}_s0_r{row}"` (row 1-based including header; first data row = 2), metadata `{sheet_name, row_index, source_type: "row"}`.
- Column documents: `text = "Column name: X. Values: a, b"`, `id = "{base}_s0_c{col}"` (0-based), metadata adds `{column_name, source_type: "column"}`.
- `base` is the source string as passed to `load`/`load_bytes` (the full path or full name — `people.csv_s0_r2`), matching Python's `base_id = doc_id or doc`; empty input → `[]`.

### `html.rs`
- `pub fn html_to_text(html: &str) -> String` — strips `<script>`, `<style>`, comments and tags, decodes named/numeric entities, collapses whitespace; `extract_title(html) -> Option<String>` prefers `og:title` then `<title>`.
- `HtmlLoader` (`.html`, `.htm`): one document with metadata `{title, source_type: "web_page"}`; no extractable text → `Error::Validation("No text content found in HTML: <name>")`.

### `url.rs`
- `UrlLoader` using `openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest}`; defaults: 30s timeout, Chrome-120 `User-Agent`, `verify_ssl=true`; `new()`, `with_client(HttpClient)`, `with_timeout(Duration)`.
- Non-HTTP(S) scheme → `Error::Unsupported`; non-2xx → `Error::Retrieval`; body converted with `html_to_text` when the content type is HTML or the body starts with `<`; otherwise plain text.
- Document id = provided id or the URL; metadata `{source_url, title, source_type: "web_page"}`.
- `load_bytes` always fails with `Error::Unsupported("URL loader needs an HTTP(S) source; use load(): <name>")` — a URL has no byte form.

### `docx.rs`
- `DocxLoader` (`.docx`) parses with `DocxParser::parse_path` / `parse_bytes` and returns exactly one `Document`.
- `text` is `DocxDocument::text` (Markdown-flavoured headings, paragraphs, and tables joined by newlines).
- Metadata: `title` from `dc:title` when present, `paragraph_count`, `table_count`, `source_type: "document"`.
- Missing file → `Error::NotFound`; corrupt container → `Error::Validation`; archive without `word/document.xml` → `Error::Retrieval`.

### `xlsx.rs`
- `XlsxLoader` (`.xlsx`) parses with `XlsxParser::parse_path` / `parse_bytes` and returns `XlsxWorkbook::to_documents(file_stem(source))`.
- Ids are `{stem}_s{sheet}_r{row}` (1-based, header included) and `{stem}_s{sheet}_c{column}` (0-based); metadata keeps `{sheet_name, row_index?, column_name?, source_type}`.
- An empty workbook (no sheets) returns `Ok(vec![])`; missing file → `Error::NotFound`; missing `xl/workbook.xml` → `Error::Retrieval`; an unresolvable `t="s"` cell propagates the parser's `Error::Retrieval`, naming the worksheet part, the cell and the shared-string table size instead of indexing empty text.

### `pptx.rs`
- `PptxLoader` (`.pptx`) parses with `PptxParser::parse_path` / `parse_bytes` and returns exactly one `Document`.
- `text` is `PptxDocument::text` — one `## Slide N` heading block per slide (plus `### Notes` when a slide has speaker notes); a slide whose part could not be resolved keeps its heading with a `! missing slide part: …` body line.
- Metadata: `title` (only when `docProps/core.xml` carries a non-empty `dc:title`), `slide_count` (placeholder slides included), `has_notes` (any slide has a notes part), `source_type: "document"`, `pptx_problems` (array of the parser's unresolved-slide messages, always present and empty for a clean deck), `pptx_incomplete` (bool, always present, `true` iff `pptx_problems` is non-empty).
- The two `pptx_*` keys are the loader's completeness markers: a deck that lost slides to dangling relationships is still indexed, but says so, mirroring `pdf_incomplete`/`pdf_skipped_filters`.
- A deck with no slide at all propagates `PptxParser`'s `Error::Validation`.

### `odt.rs`
- `OdtLoader` (`.odt`) parses with `OdtParser::parse_path` / `parse_bytes` and returns exactly one `Document`.
- `text` is `OdtDocument::text()` — headings as `#`-prefixed lines, body paragraphs, Markdown tables.
- Metadata: `title` (only when `meta.xml` carries a non-empty `dc:title`), `paragraph_count`, `table_count`, `source_type: "document"`, `odt_unsupported_elements` (array of the parser's `<name> (×<count>)` entries, sorted, always present and empty for a clean document), `odt_incomplete` (bool, always present, `true` iff the list is non-empty).
- The two `odt_*` keys name what contributed no text (drawings, footnotes, tables of contents, tracked changes, annotations, unknown blocks) instead of dropping it silently.
- An archive without `content.xml` → `Error::Validation`.

### `pdf.rs`
- `PdfLoader` (`.pdf`) parses with `PdfParser::parse_path` / `parse_bytes` and returns exactly one `Document`.
- `text` is the decoded page text, and `page_count` / `has_text` / `source_type: "document"` carry the document shape.
- Metadata also carries the parser's completeness markers: `pdf_incomplete` (bool, `true` when at least one stream was skipped, so `text` is partial) and `pdf_skipped_filters` (array of the display names of the filters on skipped streams, deduplicated in first-seen order); both are always present.
- An image-only document is not an error (`has_text == false`); encrypted PDFs and object streams propagate the parser's `Error::Unsupported`.

### `auto.rs`
- `LoaderRegistry`: normalized extension map (with leading dot); `with_defaults()` registers the 19 extensions above; `register(extensions, loader)` first registration wins (Python parity); `get_for_path(path) -> Option<Arc<dyn DocumentLoader>>`; `extensions()` returns the sorted list.
- `AutoLoader { registry, url_loader }`: `new()`/`Default`, `with_url_loader`, `register`, `registry`, `supports(&self, source) -> bool`.
- `load(&self, source)` = `load_as(source, None, None)`; URL → `UrlLoader`; missing path → `Error::NotFound("File X does not exist")`; unregistered extension → `Error::Unsupported("Unsupported format: .ext, only [...] are supported")`.
- `load_as(&self, source, doc_id: Option<&str>, title: Option<&str>)` keeps its Python-parity behaviour: file sources go through `DocumentLoader::load_with_options` and then `stamp_provenance`, which overwrites `doc_id`, `title` (given title, else the file stem), `file_path`, and `file_ext`.
- `load_bytes(&self, name: &str, bytes: impl Into<Vec<u8>>, options: &LoadOptions)` and `load_document_bytes(&self, source: &DocumentBytes, options: &LoadOptions)` dispatch by `name` extension, pass the options to the child loader (which applies them once), then stamp the same provenance with `file_path = name`.
- `impl DocumentLoader for AutoLoader` overrides `supports`, `load`, and `load_bytes` (the last delegates to `load_document_bytes`), so `Arc<dyn DocumentLoader>` callers get the byte path too.

### `fixtures.rs`
- Public byte builders for downstream crates, `openjiuwen-e2e`, and examples — no hand-rolled ZIP bytes anywhere else.
- `build_zip(entries: &[(&str, &str)]) -> Result<Vec<u8>>` delegates to `crate::parsing::ZipWriter`; entries are stored (never deflated) and keep their order.
- `pdf_document(body: &str) -> Vec<u8>` (file-level wrapper) and `plain_stream(content: &str) -> String` (one uncompressed content-stream object).
- `docx_document(paragraphs: &[&str]) -> Result<Vec<u8>>` — `word/document.xml` only, so no title and no headings/tables.
- `xlsx_workbook(sheets: &[(&str, &[&str])]) -> Result<Vec<u8>>` — `(sheet name, rows)`; a row string splits on commas and trims cells, empty cells leave gaps, text is indexed through `xl/sharedStrings.xml`, first non-empty row is the header; an empty slice yields a workbook with no sheets.
- `pptx_slides(slides: &[(&str, &[&str])]) -> Result<Vec<u8>>` — `(title, paragraphs)` per slide, linked from `ppt/presentation.xml` in order, no notes parts.
- `odt_document(paragraphs: &[&str]) -> Result<Vec<u8>>` — `content.xml` only, so no title and no headings/tables.
- Text is XML-escaped; the builders emit minimal valid archives and never re-implement parsing.

## Constraints

- All loaders are async and use `tokio::fs` for files; `load_bytes` parses in memory with the `parsing` `*Parser::parse_bytes` entry points (no temp files, no `tokio::fs`), and the URL loader uses the shared async HTTP client.
- Never panic on malformed input; return `Error::Validation`/`NotFound`/`Unsupported`/`Retrieval`/`Transport`.
- Error messages name the source (`source.name` for bytes), never the bytes themselves.
- Keep each loader ≤300 lines and `auto.rs`/`fixtures.rs` under ~1000; shared parsing helpers stay private in their module, and `apply_doc_id`/`apply_load_options` live only in `mod.rs` (no private copies).
- Tests use `tempfile` and `wiremock`; no network, no compression library, no wall clock.
- `tracing` only — no `println!`/`eprintln!`; `unwrap`/`expect` are test-only.

## Test Strategy

- Unit tests live inline (`#[cfg(test)] mod tests`) beside each loader, and cover both paths with the same fixtures: path `load` and `load_bytes` parity, metadata keys, `LoadOptions` stamping (`doc_id` verbatim, base-prefix for CSV/XLSX, `title` override, `metadata["doc_id"]`), typed errors per format (missing → `NotFound`, corrupt bytes → `Validation`, scanned/encrypted PDF and `UrlLoader::load_bytes` → `Unsupported`), `supports` case-insensitivity, `AutoLoader` byte dispatch for text/CSV/DOCX/XLSX/PDF/PPTX/ODT plus unknown/extension-less names, and the completeness markers on both the clean and the lossy path (`pptx_problems`/`pptx_incomplete`, `odt_unsupported_elements`/`odt_incomplete`, `pdf_incomplete`/`pdf_skipped_filters`).
- `fixtures.rs` unit tests parse each builder back through the matching parser, assert the ZIP entries are stored, and check shared strings, gapped cells, 28-column sheets, slide ordering/titles, and XML escaping.
- Path-level coverage of every loader and of `AutoLoader` dispatch/provenance lives in `tests/loaders.rs`, which belongs to the parent node; it also asserts the `pptx_*` and `odt_*` completeness markers through `AutoLoader` and the bare loaders.

## Notes for Agents

- Minimal offline stack: `SimpleKnowledgeBase::from_config(...)` for the pipeline, or `AutoLoader::new().load_bytes("notes.txt", bytes, &LoadOptions::new())` for a single source.
- Use `fixtures::{docx_document, xlsx_workbook, pptx_slides, odt_document, pdf_document, plain_stream, build_zip}` instead of writing ZIP/PDF bytes by hand; `build_zip` is the only ZIP writer path and it returns `Result` (`?` in libraries, `.unwrap()` in tests).
- `AutoLoader::load_bytes` and `DocumentLoader::load_bytes` differ in argument shape; method-call syntax on an `AutoLoader` picks the inherent one, so trait-object/generic callers use `DocumentLoader::load_bytes` (which is overridden and works).
- `ImageLoader` lives in `crate::multimodal` and deliberately keeps the trait's default `load_bytes` (`Unsupported`); its path is `load`.
- CSV ids are keyed by the full source name while XLSX ids use the file stem — pass a bare file name (`people.csv`) to `load_bytes` for readable ids.
- The default registry size is part of the contract: `tests/loaders.rs` (parent-owned) asserts it, so any newly registered extension must be reflected there.
- `crate::parsing` also owns `ZipWriter`, the tolerant XML reader, and the DOCX/XLSX/PPTX/ODT/PDF parsers — loader files never parse formats themselves.

## Routing Table

- `./mod.rs` → `DocumentLoader` trait (path + byte + options), `LoadOptions`, `DocumentBytes`, shared id/option helpers, re-exports
- `./text.rs` → text and Markdown loaders
- `./json.rs` → JSON loader
- `./csv.rs` → CSV/TSV loader (row + column documents)
- `./html.rs` → HTML-to-text conversion and HTML file loader
- `./url.rs` → HTTP(S) loader
- `./docx.rs` → Word loader over `parsing::DocxParser`
- `./xlsx.rs` → Excel loader over `parsing::XlsxParser`
- `./pptx.rs` → PowerPoint loader over `parsing::PptxParser`
- `./odt.rs` → OpenDocument Text loader over `parsing::OdtParser`
- `./pdf.rs` → PDF loader over `parsing::PdfParser`
- `./auto.rs` → extension/URL/byte dispatch, registry, provenance stamping
- `./fixtures.rs` → public byte builders for DOCX/XLSX/PPTX/ODT/PDF and stored-entry ZIP archives
