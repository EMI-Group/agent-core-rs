# multimodal — multimodal documents, image loading, table conversion

## Intent

Multimodal document representation plus the offline image and table ingestion paths of the Python `openjiuwen/core/retrieval/` package.
Covers `common/document.py` (`MultimodalDocument`), `indexing/processor/parser/captioner.py` (caption seam), `image_parser.py` (image loader), and the table half of `excel_parser.py` (row/column documents and Markdown tables).
Everything here runs offline: the default captioner never calls a model, and tables arrive as a `TableData` value instead of an `XLSX` reader.
Model-backed captioning, PDF/DOCX/XLSX parsing, and `LoaderRegistry` wiring are out of scope for this module and are handled by upper crates or a later round.

## API Surface

### `modality.rs`

`Modality { Text, Image, Audio, Video }` is serde `snake_case` with `Default = Text`, `Display`, `FromStr` (validation error with Python's wording), `mime_prefix()` (`"text/"`, `"image/"`, `"audio/"`, `"video/"`), `from_extension(&str)`, and the `ALL` constant.
`guess_mime_type(path) -> Option<&'static str>` replaces Python's `mimetypes.guess_type` for the images, audio, video, and text extensions the crate understands (`.jfif` maps to `image/jpeg`).

### `document.rs`

`ContentPart { modality, data, data_id }` is a serializable part with serde defaults.
`MultimodalDocument { id, text, metadata, parts }` serializes `id` as `id_` (alias `id`), defaults every field, and tolerates unknown keys.
Constructors: `new(text)` (generated uuid4 id) and `with_id(id, text)`.
Part builders: `add_text(text)`, `add_part(modality, data, data_id)`, `add_field(modality, data)` returning `Result<()>`, and async `add_file(modality, path)` returning `Result<()>`.
`add_field` accepts any text payload, a `data:<modality>/...;base64,` URI, or an `http(s)://` URL for image/video only (audio stays base64-only); violations are `Error::Validation` and non-text parts get a 32-character hex `data_id`.
`add_file` reads bytes with `tokio::fs`, derives the MIME type from the extension, base64-encodes non-text payloads, and reports `NotFound`/`Transport`/`Validation` (undeterminable MIME type); text parts keep an empty `data_id` (Python parity).
`content()` returns the OpenAI-compatible content array (`text`, `image_url`, `video_url`, `input_audio`, plus `uuid` when `data_id` is set) and rejects non-data-URI audio with `Error::Validation`.
`strip()` returns `None` only when both `text` and `parts` are empty, which keeps text-only documents alive (documented deviation from Python's `_data`-only check).
`to_document()` projects identity/text/metadata into a plain `Document` (parts are dropped) and `from_document(&Document)` builds a part-less multimodal document.
`modality_summary()` counts parts per modality name in a `BTreeMap`.
Free helpers: `to_data_uri(modality, mime, bytes)` (repairs a foreign top-level media type) and `parse_data_uri(uri)` (returns `(mime, bytes)` or `None`; percent-encoded URIs are not decoded).

### `caption.rs`

`#[async_trait] ImageCaptioner::caption(&self, image_paths: &[String]) -> Result<Vec<String>>` mirrors `ImageCaptioner.caption_images`; the provided `caption_status()` defaults to `"custom"`.
`OfflineCaptioner { prefix }` is the deterministic fallback: it returns `"{prefix}{file_name}"` per path, performs no I/O, and reports `caption_status() == "offline"`; the default prefix is `"[image] "` (`DEFAULT_CAPTION_PREFIX`).
`IMAGE_CAPTION_PROMPT` carries the reference prompt for an upper-crate model-backed captioner.

### `image.rs`

`ImageLoader` implements `loaders::DocumentLoader` for `.png .jpg .jpeg .webp .gif .jfif` and returns exactly one `Document` per image.
`ImageLoader::new()` uses the offline captioner; `with_captioner(Arc<dyn ImageCaptioner>)` injects another one and `with_data_uri(bool)` additionally stamps the base64 payload.
Document text is the captions joined by newlines (empty for a zero-byte file); metadata holds `image_path` (the caller's path), `mime_type`, `byte_size`, `modality: "image"`, and `caption_status` from the captioner, plus `data_uri` when enabled.
Missing file → `Error::NotFound("File X does not exist")`; unsupported extension → `Error::Unsupported("Unsupported image format: ...")`.
`IMAGE_EXTENSIONS` and `ImageLoader::is_supported(path)` expose the accepted extension set.

### `table.rs`

`TableData { headers, rows }` is serializable with serde defaults and built with `new(headers, rows)`.
`to_markdown()` renders `| a | b |` plus a `| --- | --- |` separator and escapes `\` as `\\`, `|` as `\|`, and line breaks as `<br>`; `from_markdown(&str)` parses it back, tolerating a missing separator, alignment markers, rows without outer pipes, and blank lines (`Error::Validation` when no row is present).
`column(index)`, `column_name(index)` (`Column {n}`, 1-based, when the header is empty), `column_text(index)` (`"Column name: X. Values: a, b"`, `(empty)` when blank), `row_text(index)` (`"H: v, H2: v2"`, empty headers skipped), and `row_texts()` reproduce the Python `ExcelParser` text conventions.
`to_documents(base_id, sheet_name)` emits row documents (`{base_id}_s0_r{row}`, 1-based including the header so the first data row is 2, metadata `{sheet_name, row_index, source_type: "row"}`) followed by `to_column_documents(base_id, sheet_name)` (`{base_id}_s0_c{column}`, 0-based, metadata adds `column_name`, `source_type: "column"`).
`sheet_name` is always present and `null` when unnamed.
`to_chunks(chunk_prefix)` returns one Markdown table chunk (`{chunk_prefix}_table`) plus one chunk per row (`{chunk_prefix}_r{row}`), all carrying `modality: "table"`, `source_type`, `sheet_name: null`, and `row_index` for row chunks.

### `mod.rs`

Re-exports `Modality`, `guess_mime_type`, `ContentPart`, `MultimodalDocument`, `to_data_uri`, `parse_data_uri`, `ImageCaptioner`, `OfflineCaptioner`, `DEFAULT_CAPTION_PREFIX`, `IMAGE_CAPTION_PROMPT`, `ImageLoader`, `IMAGE_EXTENSIONS`, and `TableData`.
The module is exposed as `pub mod multimodal;` from `lib.rs` and is not part of the crate-root `pub use` block.

## Constraints

Depends only on `openjiuwen-common`, `serde`, `serde_json`, `tokio`, `async-trait`, `uuid`, and the workspace `base64` crate.
No dependency on `openjiuwen-foundation`, so a model-backed captioner lives in an upper crate.
`#![forbid(unsafe_code)]` at the crate root; all I/O is async via `tokio::fs`.
No panics on user input: invalid payloads are `Error::Validation`, missing paths `Error::NotFound`, unsupported extensions `Error::Unsupported`, I/O failures `Error::Transport`.
Data-URI encoding is deterministic (`base64` standard alphabet with padding); only the generated `data_id`/document id are random.
No `println!`/`eprintln!`; no logging side effects in the offline fallback.
Files stay under 900 lines and one responsibility per file.

## Test Strategy

Unit tests beside each module cover the wire names and MIME table, data-URI round trips and malformed input, `content()` shapes per modality, `add_field` validation, `add_file` success/missing/unknown-extension paths, offline caption determinism, Markdown round trips with escaped pipes and newlines, row/column text parity, document ids/metadata, and chunk metadata.
`tests/multimodal.rs` exercises the same surface through `openjiuwen_retrieval::multimodal` end to end with `tempfile` images only.
Everything is offline and deterministic; generated ids are checked for shape, never for a value.

## Routing Table

- `./modality.rs` → `Modality`, MIME/extension helpers
- `./document.rs` → `ContentPart`, `MultimodalDocument`, data-URI helpers
- `./caption.rs` → `ImageCaptioner` seam, `OfflineCaptioner`, caption prompt
- `./image.rs` → `ImageLoader` implementing `DocumentLoader`
- `./table.rs` → `TableData`, Markdown conversion, row/column documents, table chunks
