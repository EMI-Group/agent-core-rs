# skill_creator/omni

## Intent

The `skill_omni_creation` pipeline: turn a documentation URL into a portable skill bundle (`SKILL.md` + text references + persisted images).
Ports the scripted Python pipeline in `openjiuwen/dev_tools/skill_creator/skills/skill_omni_creation/scripts/` (`common.py`, `stage_01_scrape.py` … `stage_05_generate.py`) as typed, deterministic, offline-testable Rust.
Python's Playwright scrapes, Pillow checks, and multimodal image judge are replaced by an injected fetcher, rule-based HTML extraction, header-only image dimension sniffing, byte/format filters, and an optional single-round `Model` call.
Archives (tar/zip/gzip, classic and ZIP64, ZipCrypto), DEFLATE, CRC-32, and SHA-256 are implemented in-crate: the workspace dependency set stays closed and offline.
Binary document formats (PDF/DOCX/XLSX/PPTX/ODT) are read through the kernel's retrieval loaders (`openjiuwen_core::retrieval::loaders::AutoLoader`), which are already a workspace dependency.
Network access happens only through [`fetch::Fetcher`]; everything else is pure and testable without a network.

## API Surface

Crate path `openjiuwen_dev_tools::skill_creator::omni`, with `omni` re-exporting `detect`, `detect_named`, `ArchiveEntry`, `ArchiveEntryKind`, `ArchiveFormat`, `ArchiveUnpacker`, `UnpackOptions`, `digest`, `hex_digest`, `content_hash`, `dedupe_links`, `filter_images`, `filter_images_with`, `image_dimensions`, `is_probable_icon`, `slugify`, `sniff_image`, `ImageCandidate`, `ResourceFilter`, `resolve_url`, `Heading`, `HtmlDocument`, `Image`, `Link`, `DownloadedResource`, `GeneratedSkill`, `GenerationProvenance`, `OmniConfig`, `OmniCreation`, `OmniOutcome`, `ScrapedPage`, `FetchedResource`, `Fetcher`, `HttpFetcher`, `ScriptedFetcher`.

### `fetch`

- `FetchedResource` — `url`, `final_url`, `status: u16`, `content_type: Option<String>`, `body: Vec<u8>`; `text()`, `is_html()`, `is_success()`.
- `#[async_trait] Fetcher: Send + Sync` — `async fn fetch(&self, url: &str) -> Result<FetchedResource>`; object-safe (`Arc<dyn Fetcher>`).
- `HttpFetcher` — one shared `Arc<HttpClient>`; `new()`/`Default`, `with_max_bytes(u64)` (default 8 MiB), `with_timeout(Duration)`; `http`/`https` only, non-2xx → `Error::transport`, oversized body → `Error::transport`.
- `ScriptedFetcher` — `new()`/`Default`, `insert`, `with_page`, `with_html`, `calls()`; unknown URL → `Error::transport`.

### `html`

- `HtmlDocument { title, description, headings, links, images, code_blocks, text }`, `Heading { level, text }`, `Link { href, text }`, `Image { src, alt }`.
- `HtmlDocument::parse(html, base_url)` — total, never fails or hangs; hrefs are not resolved here.
- `resolve_url(base, href) -> Option<String>` — absolute, protocol-relative, root-relative, `.`/`..` normalization; `None` for `mailto:`/`javascript:`/`data:`/`tel:`/`#fragment`/empty/non-http base.

### `archive` (see `./archive/CONTEXT.md`)

- `ArchiveFormat::{Tar, Zip, Gzip, Bzip2, Xz, Zstd}` (+ `as_str`/`Display`), `detect(bytes)`, `detect_named(name, bytes)` (magic bytes first, then a `.tar.bz2`/`.txz`/`.zstd`-style suffix fallback).
- `ArchiveEntry { path, bytes, is_dir, kind, link_target, mode }`, `ArchiveEntryKind::{File, Directory, Symlink, HardLink}` — serde-defaulted, so a pre-existing three-field payload still deserializes.
- `ArchiveUnpacker::{unpack, unpack_auto, unpack_with_options, unpack_auto_with_options}`; `UnpackOptions { password }` enables traditional PKWARE (ZipCrypto) decryption.
- `ArchiveWriter::write_stored(entries)` — deterministic stored zip writer (writes file/directory/symlink modes).
- Caps: `MAX_ENTRY_UNCOMPRESSED_BYTES` (8 MiB) and `MAX_ARCHIVE_UNCOMPRESSED_BYTES` (32 MiB).

### `sha256`

- `digest(bytes) -> [u8; 32]` and `hex_digest(bytes) -> String` — FIPS 180-4, no dependencies; the payload identity used for dedupe and provenance.

### `filter` (see `./filter/CONTEXT.md` for the module split)

- `ResourceFilter { min_image_bytes, max_image_bytes, allow_cross_host }` (defaults 1 KiB / 8 MiB / `false`) + builders + `validate()`.
- `ImageCandidate { url, alt, bytes, content_type }`; `filter_images(images)`, `filter_images_with(images, filter)`.
- `image_dimensions(bytes) -> Option<(u32, u32)>` — header-only dimensions for PNG (IHDR), JPEG (SOFn), GIF (logical screen descriptor), WebP (`VP8 `/`VP8L`/`VP8X`) and PNM (`P1`–`P6`); `None` for anything else or a malformed header, never a panic.
- `is_probable_icon(url, width, height)` — spacer rule (any known axis ≤ 2 px), Python's `< 80 && < 80` pair rule, and URL hints.
- `SUPPORTED_EXTS` (`png`, `jpg`, `jpeg`, `webp`, `gif`) and `SUPPORTED_MIMES` (`image/jpeg`, `image/png`, `image/gif`, `image/webp`) mirror Python `common.py:24-25`; SVG is classified by `sniff_image` but never kept.
- `image_extension(url, content_type) -> &'static str` — Python `stage_04_save.py` naming rule (URL suffix when supported, else MIME map, else `png`).
- `sniff_image`, `content_hash` (FNV-1a, the URL/identity dedupe helper), `dedupe_links`, `slugify`, `host`, `same_host`.
- Filter order: non-http → url-hint icons → declared MIME/format (magic bytes, SHA-256 dedupe) → size bounds → decoded dimension pair → dedupe; first-seen order, idempotent.

### `pipeline`

- `OmniConfig { max_links: 8, max_images: 16, max_resources: 32, min_image_bytes: 1 KiB, max_image_bytes: 8 MiB, max_resource_bytes: 2 MiB, allow_cross_host: false }` + builders + `validate()` + `resource_filter()`.
- `ScrapedPage { url, title, description, text, headings, links, images, code_blocks }` + `is_blank()`.
- `DownloadedResource { url, content_type, bytes, alt, sha256, saved_path }` — `sha256` is the lowercase hex payload digest, `saved_path` the bundle-relative asset path (`references/img_00.png`) for a persisted image and `None` for a followed subpage.
- `GenerationProvenance::{Deterministic, Model}` (+ `as_str`/`Display`, `Default` = `Deterministic`) and `GeneratedSkill { definition, provenance }`.
- `OmniOutcome { definition, resources, bundle_path, provenance }` — `resources` lists text resource paths first, then saved image paths; the field-level default keeps old JSON readable.
- `OmniCreation` — `new(Arc<dyn Fetcher>)`, `with_model(Arc<dyn Model>)`, `with_config(OmniConfig)`, `model()`, `config()`.
- Stages: `scrape(url)`, `download(&page)`, `generate(&page, &resources)`, `generate_profiled(...)`, `generate_with_model(...)`, `generate_with_model_profiled(...)`, `text_resources(&page, &resources)`, `save(&definition, &resources, root)`, `save_with_images(&definition, &resources, &images, root)`, `run(url, root)`.
- `pipeline::assets` (crate-internal) — `IMAGE_RESOURCE_DIR`, `IMAGE_RESOURCE_PREFIX`, `image_resource_path`, `image_resource`, `image_target` (path-safety check for a bundle-relative image path).
- `pipeline::page` (crate-internal) — `archive_page`, `html_page`, and the markdown heading/fence collectors the scrape stage shares with the document route.
- `pipeline::documents` (crate-internal) — the retrieval-loader bridge: suffix/content-type dispatch, `AutoLoader::load_bytes`, `page_from_documents`.

### `render` (crate-internal)

`skill_name`, `skill_description`, `instructions_markdown(page, name, description, references, images, config)`, `reference_files_section(images, references)`, `sources_markdown(page, images, config)`, `render_page_resource`, `model_user_prompt(page, images, config)`, `strip_hallucinated_images(markdown, valid_paths)`, `same_page`, `ModelSkillPayload`, `MODEL_SYSTEM_PROMPT`, `SOURCES_RESOURCE`, `PAGE_RESOURCE_PREFIX`.
They are `pub(crate)` so the module stays documentable without widening the public API.

## Stage Behaviour

`scrape` — fetch, then route by content in this order: an archive detected by `detect_named` (magic bytes, then the URL suffix) is unpacked (tar/zip/gzip, gzip-wrapped tar/zip; only `kind == File` entries contribute text, so directories, symlinks and hardlinks are never read as content); a resource whose URL suffix or content type names a supported document (PDF, DOCX, XLSX, PPTX, ODT, text, Markdown, CSV, TSV, JSON) goes through `AutoLoader` and contributes its text, its metadata title or first `#` heading, and markdown-derived headings/code blocks; an HTML body goes through `HtmlDocument::parse` with links and images resolved to absolute URLs; anything else is a plain text page.
A blank page is `Error::validation` naming the URL; a container the reader cannot decode (bzip2/xz/zstd, AES, strong encryption) propagates its typed error.

`download` — images: same-host (unless `allow_cross_host`), url-deduped, capped at `max_images`, fetched sequentially, per-resource failures and oversized bodies logged and skipped, then filtered once by `filter_images_with` (which now derives each payload's dimensions from its header, so a real 79x79 icon is dropped); every kept image gets `alt`, `sha256`, and `saved_path = references/img_NN.<ext>` numbered from `00` in kept order.
Subpages: `dedupe_links` of the page's links, self excluded, same-host, capped at `max_links`, HTML responses only, bounded by `max_resource_bytes`, with `saved_path: None`.
The combined list keeps image order then link order and is truncated to `max_resources`.

`generate` / `generate_profiled` — deterministic definition: name from `slugify(title)` → `slugify(host/path)` → `scraped-skill`; description from the meta description, else the first text paragraph (≤300 chars), else a URL-derived sentence.
Instructions are `# title`, description, then the non-empty sections `## Steps`, `## Details`, `## Code Examples`, `## Reference Images` (one `![alt](references/img_NN.ext)` per persisted image, `- alt — url` for an image that was not persisted), `## Further Reading`, and exactly one `## Reference Files` (persisted images first as ``- `path` — alt-or-screenshot``, then the text resources via `describe_resource`).
`generate_profiled` reports `GenerationProvenance::Deterministic`; `generate` is its definition-only shorthand.

`text_resources` — `references/sources.md` (source URL, title/description, deduped subpage links, every image URL with alt text and ` (saved as references/img_NN.ext)` when persisted) and one `references/page-NN.md` per non-blank HTML resource, capped at `max_resources`.

`generate_with_model` / `generate_with_model_profiled(&page, &resources)` — one `Model::invoke` round trip asking for a fenced ```json``` object (`name`, `description`, `instructions`, optional `metadata`), parsed with `crate::util::extract_json_fence`.
The prompt lists persisted images by bundle-relative path (Python `blocks_for_llm`) and other images by URL.
The answer is then `strip_hallucinated_images`-ed against the persisted paths, the `## Reference Files` section is appended unless the body already contains that heading, resources come from `text_resources`, and the definition is validated.
Any failure after the model is attached (model error, no fence, malformed payload, blank-after-strip instructions, invalid definition) logs `tracing::warn` with `provenance = Deterministic` and falls back to the deterministic generator, which the profiled variant reports; only a missing model is a hard `Error::validation` naming `with_model`.

`save` / `save_with_images` — clones the definition, appends non-duplicate validated text resources, delegates to `SkillBundle::write`, then writes every image payload to `<bundle>/<saved_path>` with `tokio::fs` after creating the parent directory.
An image `saved_path` must be relative, `/`-separated, rooted at `references/`, free of `.`/`..`/empty components and backslashes, else `Error::validation` and nothing is written.
`SkillBundle::read` skips non-UTF-8 files, so persisted images never reappear as text resources.

`run` — `config.validate()` → `scrape` → `download` → `generate_with_model_profiled` (model attached) or `generate_profiled` → `save_with_images`.
The outcome's `provenance` is `Model` only when the model's answer was accepted; taking the deterministic path because no model is attached emits exactly one `openjiuwen_common::logging::warn!` naming the URL and pointing at `OmniCreation::with_model`.
`OmniOutcome.resources` lists the text resource paths then the saved image paths.

## Constraints

- Only files under `omni/` are edited here; `creator.rs`, `bundle.rs`, `definition.rs`, `template.rs`, `skill_creator/mod.rs`, the crate manifest, `src/lib.rs`, `tests/`, and `docs/` belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for unusable input (bad config, blank page, unsafe archive or image path, malformed archive, an encrypted entry with no password), `Error::transport` for HTTP failures and size caps, `Error::unsupported` for a codec or encryption scheme this offline reader does not ship (bzip2/xz/zstd, `WinZip` AES, PKWARE strong encryption).
- No new dependencies: HTTP is `openjiuwen_common::http`; HTML, tar, zip (classic + ZIP64 + ZipCrypto), gzip, DEFLATE, CRC-32, SHA-256, image header sniffing, and URL resolution are implemented here; the office/PDF loaders come from `openjiuwen_core::retrieval` (already a dependency); `regex` is available but the extractors are hand-written scanners.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing (`tracing` only), no environment reads, no blocking calls on async tasks.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`); every string truncation happens on a char boundary; output order never depends on hash iteration.
- Work is bounded by `OmniConfig` and the archive caps; fetching is sequential so ordering is deterministic.
- Provenance is never silent: a deterministic rendering is reported as `GenerationProvenance::Deterministic` and logged, so a caller can tell template output from model output.

## Deferrals

- Playwright/JS rendering, stealth user agents, cookie-banner dismissal, and scroll/lazy-load simulation: no browser stack is reimplemented; static HTML plus the fetcher seam covers the ported behaviour.
- Pillow image decoding beyond the header: `image_dimensions` reads the size fields of PNG/JPEG/GIF/WebP/PNM, but no pixels are decoded, so crop/pixel-statistics checks and the `80 px` gate work only on the declared size.
- The LLM image-relevance judge (Python stage 3 `FILTER_PROMPT`): replaced by the deterministic format/size/dimension/duplicate filters; a multimodal judge would need a vision model and per-image cost.
- Video URLs and frame extraction, the platform-URL detection set, and the LLM "web plugin" scrape fallback: not ported.
- `bzip2`/`xz`/`zstd` containers, `WinZip` AES, and PKWARE strong encryption stay typed `Error::unsupported` naming the codec or scheme: no such crate is in the workspace or the offline registry (see `./archive/CONTEXT.md`).
- Python's `append_reference_files` lead-in sentence ("For visual reference, …") and its `blocks_for_llm` structured block payload are not reproduced: the section is heading plus ``- `path` — description`` lines, and the prompt stays plain text with bundle-relative image paths.

## Known Issues

- `filter/mod.rs`'s `has_supported_format` accepts a payload whose declared `content-type` is a supported MIME without sniffing its magic bytes, so a mislabelled or corrupt body is persisted as an image.
- `is_dir` on `ArchiveEntry` is redundant with `kind == ArchiveEntryKind::Directory`; it is kept for backward compatibility and no longer drives the pipeline's entry selection.

## Test Strategy

- Unit tests live beside each module: `fetch.rs` (loopback TCP server, size cap, schemes, scripted fetcher), `html/tests.rs` (fixtures, entities, malformed markup, url resolution), `archive/{tar,zip,gzip,inflate}` + their `tests.rs` siblings (archive matrix, ZIP64 records and extra fields, ZipCrypto round trips and rejections, PAX/GNU extension records, symlink entries, DEFLATE golden/dynamic streams, gzip members, zip round-trip and caps), `crc.rs`/`sha256.rs` (published vectors), `filter/tests.rs` + `filter/dimensions/tests.rs` (sniffing, keep policy, extension mapping, icon heuristics, the dimension parser matrix, SHA-256 dedupe, slugify), `render.rs` (`strip_hallucinated_images` matrix, section rendering), `pipeline/tests.rs` + `pipeline/documents/tests.rs` (end-to-end URL→bundle read back through `SkillBundle::read`, archive and document page inputs, codec-suffix refusals, caps, blank page, model/fallback with provenance, saved-image provenance and persistence, determinism, config validation).
- `SkillCreator::create_from_url` is covered in `creator.rs`'s own test module; the crate-root `tests/omni_bundle.rs` covers the public seam.
- Everything is offline and deterministic: `ScriptedFetcher` for HTTP, `TempWorkspace` for bundles, `ScriptedModel` for the model path, in-crate byte builders for archives and image headers, no wall-clock ordering.

## File Size Notes

`pipeline.rs` (~1100 lines) owns the stage orchestration while its stage-free helpers live in `pipeline/assets.rs` (asset paths), `pipeline/page.rs` (page construction) and `pipeline/documents.rs` (the retrieval-loader bridge).
`render.rs` (~737 lines) owns document and prompt rendering; `filter/mod.rs` (~658 lines) owns the classification/dedupe policy, with the image-header parser in `filter/dimensions.rs`.
`archive/zip/mod.rs` (~653), `archive/zip/zip64.rs` (~472), `archive/zip/zipcrypto.rs` (~258), `archive/zip/writer.rs` (~301), `archive/zip/payload.rs` (~142), `archive/inflate.rs` (~572) and `archive/tar/mod.rs` (~498) are the archive implementation files; their unit tests live in sibling `tests.rs` files to keep each implementation readable.

## Routing Table

- `./mod.rs` → module tree, module-level docs, re-exports of every public omni item
- `./fetch.rs` → `Fetcher` seam, `FetchedResource`, `HttpFetcher`, `ScriptedFetcher`
- `./html.rs` → `HtmlDocument::parse`, `Heading`/`Link`/`Image`, `resolve_url` (`./html/tests.rs`)
- `./archive/` → archive sniffing, tar/zip/gzip unpacking, ZIP64 and ZipCrypto, stored zip writer, DEFLATE and CRC-32 (`archive/CONTEXT.md`)
- `./sha256.rs` → in-crate FIPS 180-4 SHA-256 (`digest`, `hex_digest`)
- `./filter/` → `ResourceFilter`, `ImageCandidate`, `sniff_image`, `image_dimensions`, `SUPPORTED_EXTS`/`SUPPORTED_MIMES`, `image_extension`, `filter_images*`, `dedupe_links`, `slugify`, `content_hash`, `host`/`same_host` (`./filter/tests.rs`, `./filter/dimensions/tests.rs`)
- `./pipeline.rs` → `OmniConfig`, `ScrapedPage`, `DownloadedResource`, `GenerationProvenance`, `GeneratedSkill`, `OmniOutcome`, `OmniCreation` and its stages (`./pipeline/tests.rs`)
- `./pipeline/assets.rs` → bundle-relative image paths, extension choice, image-path safety
- `./pipeline/page.rs` → archive/HTML/text page construction and the markdown collectors
- `./pipeline/documents.rs` → retrieval-loader dispatch and document page construction (`./pipeline/documents/tests.rs`)
- `./render.rs` → crate-internal deterministic document and prompt rendering, `strip_hallucinated_images`
