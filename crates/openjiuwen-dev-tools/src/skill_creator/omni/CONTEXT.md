# skill_creator/omni

## Intent

The `skill_omni_creation` pipeline: turn a documentation URL into a portable skill bundle (`SKILL.md` + text references + persisted images).
Ports the scripted Python pipeline in `openjiuwen/dev_tools/skill_creator/skills/skill_omni_creation/scripts/` (`common.py`, `stage_01_scrape.py` … `stage_05_generate.py`) as typed, deterministic, offline-testable Rust.
Python's Playwright scrapes, Pillow checks, and multimodal image judge are replaced by an injected fetcher, rule-based HTML extraction, byte/format filters, and an optional single-round `Model` call.
Archives, DEFLATE, CRC-32, and SHA-256 are implemented in-crate: the workspace dependency set stays closed and offline.
Network access happens only through [`fetch::Fetcher`]; everything else is pure and testable without a network.

## API Surface

Crate path `openjiuwen_dev_tools::skill_creator::omni`, with `omni` re-exporting `detect`, `ArchiveEntry`, `ArchiveFormat`, `ArchiveUnpacker`, `digest`, `hex_digest`, `content_hash`, `dedupe_links`, `filter_images`, `filter_images_with`, `is_probable_icon`, `slugify`, `sniff_image`, `ImageCandidate`, `ResourceFilter`, `resolve_url`, `Heading`, `HtmlDocument`, `Image`, `Link`, `DownloadedResource`, `OmniConfig`, `OmniCreation`, `OmniOutcome`, `ScrapedPage`, `FetchedResource`, `Fetcher`, `HttpFetcher`, `ScriptedFetcher`.

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

- `ArchiveFormat::{Tar, Zip, Gzip}` (+ `as_str`/`Display`), `detect(bytes)`, `ArchiveEntry { path, bytes, is_dir }`.
- `ArchiveUnpacker::{unpack, unpack_auto}` — `tar` (ustar/GNU), `zip` (stored + DEFLATE), `gzip` (RFC 1952, including gzip-wrapped `tar`/`zip`).
- `ArchiveWriter::write_stored(entries) -> Result<Vec<u8>>` — deterministic stored zip writer.
- Caps: `MAX_ENTRY_UNCOMPRESSED_BYTES` (8 MiB) and `MAX_ARCHIVE_UNCOMPRESSED_BYTES` (32 MiB).

### `sha256`

- `digest(bytes) -> [u8; 32]` and `hex_digest(bytes) -> String` — FIPS 180-4, no dependencies; the payload identity used for dedupe and provenance.

### `filter`

- `ResourceFilter { min_image_bytes, max_image_bytes, allow_cross_host }` (defaults 1 KiB / 8 MiB / `false`) + builders + `validate()`.
- `ImageCandidate { url, alt, bytes, content_type }`; `filter_images(images)`, `filter_images_with(images, filter)`.
- `SUPPORTED_EXTS` (`png`, `jpg`, `jpeg`, `webp`, `gif`) and `SUPPORTED_MIMES` (`image/jpeg`, `image/png`, `image/gif`, `image/webp`) mirror Python `common.py:24-25`; SVG is classified by `sniff_image` but never kept.
- `image_extension(url, content_type) -> &'static str` — Python `stage_04_save.py` naming rule (URL suffix when supported, else MIME map, else `png`).
- `sniff_image`, `is_probable_icon`, `content_hash` (FNV-1a, the URL/identity dedupe helper), `dedupe_links`, `slugify`, `host`, `same_host`.
- Filter order: non-http → url-hint icons → declared MIME/format (magic bytes, SHA-256 dedupe) → size bounds → dedupe; first-seen order, idempotent.

### `pipeline`

- `OmniConfig { max_links: 8, max_images: 16, max_resources: 32, min_image_bytes: 1 KiB, max_image_bytes: 8 MiB, max_resource_bytes: 2 MiB, allow_cross_host: false }` + builders + `validate()` + `resource_filter()`.
- `ScrapedPage { url, title, description, text, headings, links, images, code_blocks }` + `is_blank()`.
- `DownloadedResource { url, content_type, bytes, alt, sha256, saved_path }` — `alt` is the image alt text, `sha256` the lowercase hex payload digest, `saved_path` the bundle-relative asset path (`references/img_00.png`) for a persisted image and `None` for a followed subpage.
- `OmniOutcome { definition, resources, bundle_path }` — `resources` lists text resource paths first, then saved image paths.
- `OmniCreation` — `new(Arc<dyn Fetcher>)`, `with_model(Arc<dyn Model>)`, `with_config(OmniConfig)`, `model()`, `config()`.
- Stages: `scrape(url)`, `download(&page)`, `generate(&page, &resources)`, `text_resources(&page, &resources)`, `save(&definition, &resources, root)`, `save_with_images(&definition, &resources, &images, root)`, `generate_with_model(&page, &resources)`, `run(url, root)`.
- `pipeline::assets` (crate-internal) — `IMAGE_RESOURCE_DIR`, `IMAGE_RESOURCE_PREFIX`, `image_resource_path`, `image_resource`, `image_target` (path-safety check for a bundle-relative image path).
- `pipeline::page` (crate-internal) — `archive_page`, `html_page`, and the markdown heading/fence collectors used by the scrape stage.

### `render` (crate-internal)

`skill_name`, `skill_description`, `instructions_markdown(page, name, description, references, images, config)`, `reference_files_section(images, references)`, `sources_markdown(page, images, config)`, `render_page_resource`, `model_user_prompt(page, images, config)`, `strip_hallucinated_images(markdown, valid_paths)`, `same_page`, `ModelSkillPayload`, `MODEL_SYSTEM_PROMPT`, `SOURCES_RESOURCE`, `PAGE_RESOURCE_PREFIX`.
They are `pub(crate)` so the module stays documentable without widening the public API.

## Stage Behaviour

`scrape` — fetch, then: a detected archive is unpacked (`tar`, `zip`, gzip-wrapped `tar`/`zip`); tar text entries become the page (markdown ATX headings and fenced code blocks parsed; directories, non-UTF-8 entries, and entries with a NUL in the first 512 bytes are skipped); HTML goes through `HtmlDocument::parse` with links and images resolved to absolute URLs; any other body is a plain text page.
A page with no text, headings, links, or images is `Error::validation` naming the URL; a ZIP64/encrypted/unknown-method archive or a gzip payload that is not an archive propagates its typed error.

`download` — images: same-host (unless `allow_cross_host`), url-deduped, capped at `max_images`, fetched sequentially, per-resource failures and oversized bodies logged and skipped, then filtered once by `filter_images_with`; every kept image gets `alt`, `sha256`, and `saved_path = references/img_NN.<ext>` numbered from `00` in kept order.
Subpages: `dedupe_links` of the page's links, self excluded, same-host, capped at `max_links`, HTML responses only, bounded by `max_resource_bytes`, with `saved_path: None`.
The combined list keeps image order then link order and is truncated to `max_resources`.

`generate` — deterministic definition: name from `slugify(title)` → `slugify(host/path)` → `scraped-skill`; description from the meta description, else the first text paragraph (≤300 chars), else a URL-derived sentence.
Instructions are `# title`, description, then the non-empty sections `## Steps`, `## Details`, `## Code Examples`, `## Reference Images` (one `![alt](references/img_NN.ext)` per persisted image, `- alt — url` for an image that was not persisted), `## Further Reading`, and exactly one `## Reference Files` (persisted images first as ``- `path` — alt-or-screenshot``, then the text resources via `describe_resource`).

`text_resources` — `references/sources.md` (source URL, title/description, deduped subpage links, every image URL with alt text and ` (saved as references/img_NN.ext)` when persisted) and one `references/page-NN.md` per non-blank HTML resource, capped at `max_resources`.

`generate_with_model(&page, &resources)` — one `Model::invoke` round trip asking for a fenced ```json``` object (`name`, `description`, `instructions`, optional `metadata`), parsed with `crate::util::extract_json_fence`.
The prompt lists persisted images by bundle-relative path (Python `blocks_for_llm`) and other images by URL.
The answer is then `strip_hallucinated_images`-ed against the persisted paths, the `## Reference Files` section is appended unless the body already contains that heading, resources come from `text_resources`, and the definition is validated.
Any failure after the model is attached (model error, no fence, malformed payload, blank-after-strip instructions, invalid definition) logs `tracing::warn` and falls back to `generate`; only a missing model is a hard `Error::validation` naming `with_model`.

`save` / `save_with_images` — clones the definition, appends non-duplicate validated text resources, delegates to `SkillBundle::write`, then writes every image payload to `<bundle>/<saved_path>` with `tokio::fs` after creating the parent directory.
An image `saved_path` must be relative, `/`-separated, rooted at `references/`, free of `.`/`..`/empty components and backslashes, else `Error::validation` and nothing is written.
`SkillBundle::read` skips non-UTF-8 files, so persisted images never reappear as text resources.

`run` — `config.validate()` → `scrape` → `download` → `generate_with_model` (model attached) or `generate` → `save_with_images`; `OmniOutcome.resources` lists the text resource paths then the saved image paths.

## Constraints

- Only files under `omni/` are edited here; `creator.rs`, `bundle.rs`, `definition.rs`, `template.rs`, `skill_creator/mod.rs`, the crate manifest, `src/lib.rs`, `tests/`, and `docs/` belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for unusable input (bad config, blank page, unsafe archive or image path, malformed archive), `Error::transport` for HTTP failures and size caps, `Error::unsupported` for ZIP64, encrypted entries, and unsupported compression methods.
- No new dependencies: HTTP is `openjiuwen_common::http`; HTML, tar, zip, gzip, DEFLATE, CRC-32, SHA-256, image sniffing, and URL resolution are implemented here; `regex` is available but the extractors are hand-written scanners.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing (`tracing` only), no environment reads, no blocking calls on async tasks.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`); every string truncation happens on a char boundary; output order never depends on hash iteration.
- Work is bounded by `OmniConfig` and the archive caps; fetching is sequential so ordering is deterministic.

## Deferrals

- ZIP64 archives, encrypted zip entries, zip compression methods other than stored/DEFLATE, gzip methods other than DEFLATE: `Error::unsupported` naming the reason; the zip writer is stored-only.
- Playwright/JS rendering, stealth user agents, cookie-banner dismissal, and scroll/lazy-load simulation: no browser stack is reimplemented; static HTML plus the fetcher seam covers the ported behaviour.
- Pillow image decoding and the 80 px dimension gate: replaced by magic sniffing plus byte bounds and URL-hint icon heuristics.
- The LLM image-relevance judge (Python stage 3 `FILTER_PROMPT`): replaced by the deterministic format/size/duplicate filters; a multimodal judge would need a vision model and per-image cost.
- Video URLs and frame extraction, the platform-URL detection set, and the LLM "web plugin" scrape fallback: not ported.
- Python's `append_reference_files` lead-in sentence ("For visual reference, …") and its `blocks_for_llm` structured block payload are not reproduced: the section is heading plus ``- `path` — description`` lines, and the prompt stays plain text with bundle-relative image paths.

## Test Strategy

- Unit tests live beside each module: `fetch.rs` (loopback TCP server, size cap, schemes, scripted fetcher), `html/tests.rs` (fixtures, entities, malformed markup, url resolution), `archive/{tar,zip,gzip,inflate}` + their `tests.rs` siblings (archive matrix, DEFLATE golden/dynamic streams, gzip members, zip round-trip and rejections, caps), `crc.rs`/`sha256.rs` (published vectors), `filter/tests.rs` (sniffing, keep policy, extension mapping, icon heuristics, SHA-256 dedupe, slugify), `render.rs` (`strip_hallucinated_images` matrix, section rendering), `pipeline/tests.rs` (end-to-end URL→bundle read back through `SkillBundle::read`, archive page inputs, caps, blank page, model/fallback, saved-image provenance and persistence, determinism, config validation).
- `SkillCreator::create_from_url` is covered in `creator.rs`'s own test module; the crate-root `tests/omni_bundle.rs` covers the public seam.
- Everything is offline and deterministic: `ScriptedFetcher` for HTTP, `TempWorkspace` for bundles, `ScriptedModel` for the model path, no wall-clock ordering.

## File Size Notes

`pipeline.rs` (~913 lines) owns the stage orchestration while its stage-free helpers live in `pipeline/assets.rs` (asset paths) and `pipeline/page.rs` (page construction).
`render.rs` (~737 lines) owns document and prompt rendering; `filter.rs` (~618 lines) owns the classification/dedupe policy with `filter/tests.rs` beside it.
`archive/zip.rs` (~788) and `archive/inflate.rs` (~572) are the largest archive files; their unit tests live in sibling `tests.rs` files to keep the implementation readable.

## Routing Table

- `./mod.rs` → module tree, module-level docs, re-exports of every public omni item
- `./fetch.rs` → `Fetcher` seam, `FetchedResource`, `HttpFetcher`, `ScriptedFetcher`
- `./html.rs` → `HtmlDocument::parse`, `Heading`/`Link`/`Image`, `resolve_url` (`./html/tests.rs`)
- `./archive/` → archive sniffing, tar/zip/gzip unpacking, stored zip writer, DEFLATE and CRC-32 (`archive/CONTEXT.md`)
- `./sha256.rs` → in-crate FIPS 180-4 SHA-256 (`digest`, `hex_digest`)
- `./filter.rs` → `ResourceFilter`, `ImageCandidate`, `sniff_image`, `SUPPORTED_EXTS`/`SUPPORTED_MIMES`, `image_extension`, `filter_images*`, `dedupe_links`, `slugify`, `content_hash`, `host`/`same_host` (`./filter/tests.rs`)
- `./pipeline.rs` → `OmniConfig`, `ScrapedPage`, `DownloadedResource`, `OmniOutcome`, `OmniCreation` and its stages (`./pipeline/tests.rs`)
- `./pipeline/assets.rs` → bundle-relative image paths, extension choice, image-path safety
- `./pipeline/page.rs` → archive/HTML/text page construction and the markdown collectors
- `./render.rs` → crate-internal deterministic document and prompt rendering, `strip_hallucinated_images`
