# skill_creator/omni

## Intent

The `skill_omni_creation` pipeline: turn a documentation URL into a portable skill bundle (`SKILL.md` + text references).
Ports the scripted Python pipeline in `openjiuwen/dev_tools/skill_creator/skills/skill_omni_creation/scripts/` (`common.py`, `stage_01_scrape.py` … `stage_05_generate.py`) as typed, deterministic, offline-testable Rust.
Python's Playwright scrapes, Pillow checks, and multimodal image judge are replaced by an injected fetcher, rule-based HTML extraction, byte/format filters, and an optional single-round `Model` call.
Network access happens only through [`fetch::Fetcher`]; everything else is pure and testable without a network.

## API Surface

Crate path `openjiuwen_dev_tools::skill_creator::omni`, with `omni` re-exporting the public types of all six submodules.

### `fetch`

- `FetchedResource` — `url`, `final_url`, `status: u16`, `content_type: Option<String>`, `body: Vec<u8>`; `text()` (lossy UTF-8), `is_html()` (content type, else a 1 KiB body sniff), `is_success()`.
- `#[async_trait] Fetcher: Send + Sync` — `async fn fetch(&self, url: &str) -> Result<FetchedResource>`; object-safe (`Arc<dyn Fetcher>`).
- `HttpFetcher` — one shared `Arc<HttpClient>`; `new()`/`Default`, `with_max_bytes(u64)` (default 8 MiB), `with_timeout(Duration)`; `http`/`https` only (`Error::validation`), non-2xx → `Error::transport("HTTP {status} for {url}")`, oversized body → `Error::transport` naming the cap.
- `ScriptedFetcher` — `new()`/`Default`, `insert(url, FetchedResource)`, `with_page(url, content_type, body)`, `with_html(url, body)`, `calls() -> Vec<String>`; unknown URL → `Error::transport`.

### `html`

- `HtmlDocument { title, description, headings, links, images, code_blocks, text }`, `Heading { level, text }`, `Link { href, text }`, `Image { src, alt }` — serde snake_case, `Debug/Clone/Default/PartialEq`.
- `HtmlDocument::parse(html: &str, base_url: &str) -> HtmlDocument` — total function (never fails or hangs); `base_url` is accepted for signature parity and does not resolve hrefs (callers use `resolve_url`).
- `resolve_url(base, href) -> Option<String>` — absolute, protocol-relative, root-relative, relative with `.`/`..` normalization; `None` for `mailto:`/`javascript:`/`data:`/`tel:`/`#fragment`/empty/non-http base.

### `archive`

- `ArchiveFormat { Tar, Zip, Gzip }` with `as_str()`/`Display`, `detect(bytes) -> Option<ArchiveFormat>`.
- `ArchiveEntry { path, bytes, is_dir }`; `ArchiveUnpacker::unpack(format, bytes)` and `unpack_auto(bytes)`.
- Tar: full ustar parsing (512-byte headers, `prefix/name`, octal and GNU base-256 size, unsigned and GNU signed checksums, typeflags file/dir/longname/PAX/link, two zero blocks terminate), rejecting absolute, `..`, and backslash paths with `Error::validation`.
- Zip and gzip return `Error::unsupported` naming the format and the missing compression dependency.

### `filter`

- `ResourceFilter { min_image_bytes, max_image_bytes, allow_cross_host }` (defaults 1 KiB / 8 MiB / `false`) with builders and `validate()`.
- `ImageCandidate { url, alt, bytes, content_type }`; `filter_images(images)` (default policy) and `filter_images_with(images, filter)`.
- `sniff_image(bytes) -> Option<"png"|"jpeg"|"gif"|"webp"|"svg">`, `is_probable_icon(url, width, height)`, `content_hash(bytes) -> u64` (FNV-1a 64), `dedupe_links(&[Link]) -> Vec<Link>`, `slugify(&str) -> String`, `host(url)`, `same_host(a, b)`.
- Filter order: non-http → url-hint icons → format (magic, else content type) → size bounds → dedupe (bytes hash, else normalized url); first-seen order, idempotent.

### `pipeline`

- `OmniConfig { max_links: 8, max_images: 16, max_resources: 32, min_image_bytes: 1 KiB, max_image_bytes: 8 MiB, max_resource_bytes: 2 MiB, allow_cross_host: false }` + builders + `validate()` (byte bounds non-zero and ordered; zero count caps disable that class) + `resource_filter()`.
- `ScrapedPage { url, title, description, text, headings, links, images, code_blocks }` + `is_blank()`; `DownloadedResource { url, content_type, bytes }` + `is_html()`; `OmniOutcome { definition, resources, bundle_path }`.
- `OmniCreation` — `new(Arc<dyn Fetcher>)`, `with_model(Arc<dyn Model>)`, `with_config(OmniConfig)`, `model()`, `config()`.
- Stages: `scrape(url)`, `download(&page)`, `generate(&page, &resources)`, `text_resources(&page, &resources)`, `save(&definition, &resources, root)`, `generate_with_model(&page)`, `run(url, root)`.

### `render` (crate-internal)

`pub(crate)` helpers only: `skill_name`, `skill_description`, `instructions_markdown`, `sources_markdown`, `render_page_resource`, `model_user_prompt`, `MODEL_SYSTEM_PROMPT`, `ModelSkillPayload`, `SOURCES_RESOURCE`, `PAGE_RESOURCE_PREFIX`.
They are `pub(crate)` so the module stays documentable without widening the public API.

### Creator integration

`SkillCreator::create_from_url(&self, fetcher: Arc<dyn Fetcher>, url: &str, root: &Path) -> Result<OmniOutcome>` builds an `OmniCreation`, attaches the creator's model when present, and returns `OmniCreation::run`.

## Stage Behaviour

`scrape` — fetch, then: a detected tar is unpacked and its text entries become the page (markdown ATX headings and fenced code blocks parsed; directories, non-UTF-8 entries, and entries with a NUL in the first 512 bytes are skipped); a detected zip/gzip propagates the typed `Unsupported` error; HTML goes through `HtmlDocument::parse` with links and images resolved to absolute URLs; any other body is a plain text page.
A page with no text, headings, links, or images is `Error::validation` naming the URL.

`download` — images: same-host (unless `allow_cross_host`), url-deduped, capped at `max_images`, fetched sequentially, per-resource failures and oversized bodies logged and skipped, then filtered once by `filter_images_with` (there is no separate public filter stage).
Subpages: `dedupe_links` of the page's links, self excluded, same-host, capped at `max_links`, HTML responses only, bounded by `max_resource_bytes`.
The combined list keeps image order then link order and is truncated to `max_resources`.

`generate` — deterministic definition: name from `slugify(title)` → `slugify(host/path)` → `scraped-skill`; description from the meta description, else the first text paragraph (≤300 chars), else a URL-derived sentence.
Instructions are `# title`, description, `## Steps` (page headings), `## Details` (text paragraphs), and, when available, `## Code Examples`, `## Reference Images` (URLs, never invented paths), `## Further Reading` (subpage links), `## Reference Files` (the text resources written into the bundle).

`text_resources` — `references/sources.md` (source URL, title/description, deduped subpage links, every image URL with alt text) and one `references/page-NN.md` per non-blank HTML resource, capped at `max_resources`.

`generate_with_model` — one `Model::invoke` round trip asking for a fenced ```json``` object (`name`, `description`, `instructions`, optional `metadata`), parsed with `crate::util::extract_json_fence`.
Any failure after the model is attached (model error, no fence, malformed payload, invalid definition) logs `tracing::warn` and falls back to `generate`; only a missing model is a hard `Error::validation` naming `with_model`.

`save` — clones the definition, appends non-duplicate validated resources, and writes through `SkillBundle::write`, returning the bundle directory.
Binary assets are never written: `SkillResource::content` is text and `SkillBundle::read` skips non-UTF-8 files, so images remain URLs in the document and `references/sources.md`.

`run` — `config.validate()` → `scrape` → `download` → `generate_with_model` (model attached) or `generate` → `save`; `OmniOutcome.resources` lists the written resource paths.

## Constraints

- Only files under `omni/` are edited here; `creator.rs`, `bundle.rs`, `definition.rs`, `template.rs`, `skill_creator/mod.rs`, the crate manifest, and `docs/` belong to other nodes.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for unusable input (bad config, blank page, unsafe archive path, bad scheme), `Error::transport` for HTTP failures and size caps, `Error::unsupported` for zip/gzip.
- No new dependencies: HTTP is `openjiuwen_common::http`, HTML parsing, tar parsing, image sniffing, hashing (FNV-1a), and URL resolution are implemented here; `regex` is available but the extractors are hand-written scanners.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing (`tracing` only), no environment reads, no blocking calls on async tasks.
- Every public item is documented (workspace `missing_docs`, clippy pedantic, `-D warnings`); every string truncation happens on a char boundary; output order never depends on hash iteration.
- Work is bounded by `OmniConfig`: caps on images, links, resources, and bytes, with sequential (deterministic) fetching.

## Deferrals

- Zip and gzip decompression: `Error::unsupported` naming the format; the offline workspace has no compression crate and adding one would break the pinned, offline-verified dependency set.
- Playwright/JS rendering, stealth user agents, cookie-banner dismissal, and scroll/lazy-load simulation: no browser stack is reimplemented; static HTML plus the fetcher seam covers the ported behaviour.
- Pillow image decoding and the 80 px dimension gate: replaced by magic sniffing plus byte bounds and URL-hint icon heuristics, since no image-decoding crate is available.
- The LLM image-relevance judge (Python stage 3 `FILTER_PROMPT`): replaced by the deterministic format/size/duplicate filters; a multimodal judge would need a vision model and per-image cost.
- Binary image storage: images are recorded as URLs in the document and `references/sources.md`; `SkillResource` and `SkillBundle` are text-only by design.
- Video URLs and frame extraction (`video_urls`, `VIDEO_FRAMES`), the platform-URL detection set, and the LLM "web plugin" scrape fallback: not ported; they depend on video tooling and a vendor plugin.

## Test Strategy

- Unit tests live beside each module: `fetch.rs` (loopback `std::net::TcpListener` server in a thread with bounded reads and tokio timeouts, size cap, schemes, scripted fetcher), `html/tests.rs` (fixtures, entities, malformed markup, url resolution), `archive.rs` (in-test ustar builder, checksums, traversal, zip/gzip unsupported), `filter.rs` (sniffing, icon heuristics, dedupe, slugify, FNV vectors), `pipeline/tests.rs` (end-to-end URL→bundle read back through `SkillBundle::read`, tar page input, zip/gzip errors, caps, blank page, model/fallback, determinism, config validation).
- `SkillCreator::create_from_url` is covered in `creator.rs`'s own test module.
- Everything is offline and deterministic: `ScriptedFetcher` for HTTP, `TempWorkspace` for bundles, `ScriptedModel` for the model path, no wall-clock ordering.

## File Size Notes

`filter.rs` is the largest file (~960 lines, of which ~440 are tests) because it owns five small, individually-documented concerns that share the FNV/normalization helpers; splitting it would duplicate those helpers.
`html.rs` (~810 lines) keeps a single tolerant scanner plus its own `html/tests.rs`; `pipeline.rs` (~890 lines) owns the stage orchestration while `render.rs` (~440 lines) owns document/prompt rendering.

## Routing Table

- `./mod.rs` → module tree, module-level docs, re-exports of every public omni item
- `./fetch.rs` → `Fetcher` seam, `FetchedResource`, `HttpFetcher`, `ScriptedFetcher`
- `./html.rs` → `HtmlDocument::parse`, `Heading`/`Link`/`Image`, `resolve_url` (`./html/tests.rs` holds its unit tests)
- `./archive.rs` → `ArchiveFormat`, `detect`, `ArchiveEntry`, `ArchiveUnpacker` (ustar; zip/gzip deferred)
- `./filter.rs` → `ResourceFilter`, `ImageCandidate`, `sniff_image`, `is_probable_icon`, `filter_images*`, `dedupe_links`, `slugify`, `content_hash`, `host`/`same_host`
- `./pipeline.rs` → `OmniConfig`, `ScrapedPage`, `DownloadedResource`, `OmniOutcome`, `OmniCreation` and its stages (`./pipeline/tests.rs` holds its unit tests)
- `./render.rs` → crate-internal deterministic document and prompt rendering
