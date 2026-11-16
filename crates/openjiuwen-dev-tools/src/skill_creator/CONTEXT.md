# skill_creator

## Intent

Author, validate, template, and persist portable skill bundles for the `openjiuwen` harness, and derive them from a documentation URL.
Ports `openjiuwen.dev_tools.skill_creator`: a typed skill definition, `SKILL.md` frontmatter rendering/parsing, text resources, model-assisted drafting, and the `skill_omni_creation` script pipeline (scrape → download → generate → save).
Python's scrape/download/filter stages are replaced by an injected fetcher, rule-based HTML extraction, in-process archive unpacking, the kernel's retrieval loaders for office/PDF documents, and deterministic filters; a `SkillSpec` (deterministic) or a natural-language `SkillRequest`/URL (model-assisted or rule-based) supplies the content.
Everything here is deterministic and offline-safe except the single model round trips in `SkillCreator::create_with_model`, `SkillCreator::create_from_url` (through `OmniCreation::generate_with_model_profiled`), and `OmniCreation::generate_with_model`.
Every URL-derived skill reports whether the model or the deterministic template renderer produced it, via `GenerationProvenance`.

## API Surface

- `SkillDefinition` — `name`, `description`, `instructions`, `version: Option<String>`, `metadata: BTreeMap<String, Value>`, `resources: Vec<SkillResource>`.
- `SkillDefinition::new/validate/skill_dir_name/with_resource/with_metadata/with_version/frontmatter/to_markdown/from_markdown`.
- `SkillResource` — `path` (relative POSIX), `content` (text), `media_type: Option<String>` (informational, never written).
- `SkillResource::new/with_media_type/validate`.
- `SkillFrontmatter` — `name`, `description`, `version: Option<String>`, `#[serde(flatten)] extra: BTreeMap<String, Value>` (the metadata keys).
- `parse_markdown(text) -> Result<(SkillFrontmatter, String)>` and `render_markdown(&SkillFrontmatter, body) -> Result<String>`.
- `SkillBundle::write(&SkillDefinition, root) -> Result<PathBuf>` and `SkillBundle::read(directory) -> Result<SkillDefinition>` (associated functions on a never-instantiated type).
- `template::render(template, &BTreeMap<String, String>) -> Result<String>` strict `{{name}}` substitution.
- `template::placeholders(template) -> Vec<&str>` deduplicated placeholder names in first-appearance order.
- `SkillSpec` — deterministic input: identity, instruction template, `variables`, `resources`, `version`, `metadata`; builders `new/with_variable/with_resource/with_version`.
- `SkillRequest` — natural-language input: `request`, `resource_hints`, `language` (`"zh-CN"` default, `"en-US"` alternative); builders `new/with_language/with_resource_hints`.
- `SkillCreator::new/with_model/has_model/create/create_with_model/write/read`, plus `Default`.
- `SkillCreator::create_from_url(fetcher: Arc<dyn Fetcher>, url, root) -> Result<OmniOutcome>` — the URL → bundle path, reusing the creator's model when attached; the outcome's `provenance` says whether the model or the deterministic template renderer drafted the document.
- `omni::GenerationProvenance` (`Deterministic`/`Model`, `as_str`/`Display`) and `omni::GeneratedSkill { definition, provenance }` — which generator produced a document.
- `omni::{OmniConfig, OmniCreation, OmniOutcome}` re-exported here; every omni type/function is re-exported from `skill_creator::omni` (see `omni/CONTEXT.md`).
- Module re-exports at `openjiuwen_dev_tools::skill_creator`: `SkillBundle`, `SkillCreator`, `SkillRequest`, `SkillSpec`, `SkillDefinition`, `SkillFrontmatter`, `SkillResource`, `parse_markdown`, `render_markdown`, `OmniConfig`, `OmniCreation`, `OmniOutcome`.
- Submodules `definition`, `template`, `bundle`, `creator`, `omni` are public; `NAME_PATTERN`, `MAX_NAME_LENGTH`, `SKILL_DOCUMENT` are public constants.

## On-Disk Format

`<root>/<name>/SKILL.md` plus every resource at its relative path beneath `<root>/<name>/`.
`SKILL.md` is `---\n<YAML mapping>---\n\n<instructions>`: the body starts after one blank line and is trimmed on parse.
The YAML mapping is `name`, `description`, optional `version`, then every `metadata` key flattened at the top level (sorted, since `metadata` is a `BTreeMap`); `version` is omitted when `None`.
Skill names must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` (the harness `_name` rule), which makes the name a safe single directory component.
Resource paths must be relative, `/`-separated, free of empty/`.`/`..` components and backslashes, and must not name `SKILL.md`.
`SkillBundle::read` skips symlinks and non-UTF-8 files (with a `tracing::debug` event) and returns resources sorted by path; `write` preserves the definition's resource order.

## Constraints

- Only files under `skill_creator/` are edited here; the crate manifest, `src/lib.rs`, `src/util.rs`, and `docs/` belong to other nodes.
- Errors are `openjiuwen_common::error::Error`: `Error::validation` for bad input, malformed documents, missing template variables, unsupported languages, and a page or document that yields nothing; `Error::unsupported` for a `bzip2`/`xz`/`zstd` page input or an archive capability the in-crate reader does not ship (the message names the codec or encryption scheme); `From<std::io::Error>` (`Error::external`) for filesystem failures; model failures propagate unchanged (`Error::llm`) from `create_with_model`, while `create_from_url` falls back to the deterministic renderer.
- Public data types derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize`, use `#[serde(rename_all = "snake_case", default)]`, and tolerate unknown fields; `SkillDefinition`, `SkillResource`, and `SkillFrontmatter` also derive `schemars::JsonSchema`.
- No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests, no stdout printing; `tracing` only.
- Bundle IO uses `tokio::fs`; both entry points are `async`.
- Rustdoc is required on every public item (workspace `missing_docs`, clippy pedantic + `-D warnings`).
- The `omni` subtree adds no dependencies: HTTP is `openjiuwen_common::http`, HTML/tar/ZIP/gzip parsing, image sniffing and header-only dimension parsing, hashing, and URL resolution are implemented in-crate, and non-text documents (PDF/DOCX/XLSX/PPTX/ODT plus text/Markdown/CSV/TSV/JSON) are extracted by `openjiuwen_core::retrieval::loaders::AutoLoader` from the crate's existing `openjiuwen-core` dependency.

## Design Decisions

- Text-only resources: `SkillResource::content` is a `String`, and a non-UTF-8 file found on disk is skipped rather than base64-wrapped; the harness only consumes text assets.
- Strict instead of Python's warn-and-degrade: a blank description, an invalid name, an unterminated frontmatter, or an unresolved template variable is a validation error, never a silently degraded document.
- `{{name}}` rendering is single-pass (an inserted value is never re-scanned), trims the name inside the braces, and leaves `{{}}`/`{{   }}`, unterminated `{{`, and non-placeholder brace runs literal.
- A nested opening brace inside a candidate placeholder (`{{a {{b}}`) makes the outer braces literal so scanning recovers at the inner placeholder instead of failing on a synthetic name.
- `render_markdown` rejects metadata keys named `name`, `description`, or `version` to avoid duplicate YAML keys, and `parse_markdown` folds them back into the structured fields.
- A YAML numeric `version` (`version: 2.0`) is stringified on parse, because YAML typing would otherwise turn a human-authored version into a float; other non-scalar `version` values are rejected.
- `SkillCreator::create_with_model` is one model round trip that must answer with a ```json fence holding `name`, `description`, `instructions` (optional `version`, `metadata`); a plain-text answer is a validation error instead of the Python multi-stage clarifier/designer loop.
- `SkillRequest::resource_hints` become resources with empty content, so the drafted bundle has the declared file layout ready for the author to fill in.
- The prompt language is chosen from `language` (`zh-CN`/`zh`/empty → Chinese wording, `en-US`/`en` → English wording); any other value is a validation error raised before the model is called.
- `SkillCreator` stores `Option<Arc<dyn Model>>` so the caller owns provider configuration and no builder reads the environment.
- The URL path (`omni`) reuses these same definition/bundle seams instead of a parallel writer: `OmniCreation::save` appends validated text resources and delegates to `SkillBundle::write`.
- URL-derived skills are deterministic first: name from `slugify(title)`, description from the meta description or first paragraph, instructions rendered from headings/text/code blocks; the model, when attached, only re-drafts the document and a deterministic fallback always exists.
- Provenance is never silent: a model answer is reported as `GenerationProvenance::Model` and the template path as `GenerationProvenance::Deterministic`, on both `GeneratedSkill` and `OmniOutcome::provenance`.
- The no-model run logs exactly one `openjiuwen_common::logging::warn!` naming the URL and pointing at `OmniCreation::with_model`, and an unusable model answer logs its cause before falling back, so a caller can check the provenance instead of reading logs.
- A model failure never fails `create_from_url`: an attached model is optional drafting help, and the deterministic renderer always produces the bundle.
- Filtering happens inside `OmniCreation::download`, so there is no public filter stage that could be skipped accidentally.
- Images are recorded as URLs, never written as binary resources, because the bundle format is text-only.

## Deferred

- Archive decoding stops at the in-crate codecs: `tar` (including PAX and GNU long-name/link records), `zip` (classic and ZIP64, stored/DEFLATE, `ZipCrypto` via `UnpackOptions`), and `gzip` (DEFLATE, including gzip-wrapped `tar`/`zip`); `bzip2`/`xz`/`zstd`, `WinZip` AES, PKWARE strong encryption, an encrypted central directory, and every other compression method are `Error::unsupported` naming the codec or method (`omni/archive`).
- The ZIP writer stays stored-only, so nothing here writes a compressed or encrypted ZIP entry.
- Playwright browser automation, JS rendering, stealth user agents, and cookie-banner handling for scraping: replaced by the `Fetcher` seam plus static rule-based HTML extraction.
- Pixel-level image decoding (Python's Pillow pipeline) and the LLM image-relevance judge (stage 3): image pixels are never decoded, dimensions come from header parsing only, and relevance is decided by the deterministic filter rules plus url-hint icon heuristics.
- OCR for scanned PDFs, video URLs and frame extraction, and Python's LLM "web plugin" scrape fallback: not ported (office/PDF text extraction itself is implemented through the retrieval loaders).
- Binary resources, per-resource media-type persistence, and asset naming conventions are out of scope.
- The Python multi-stage LLM clarifier/designer/reflector pipeline and cycle checker are replaced by a single fenced-JSON round trip.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`); async paths use `#[tokio::test]`.
- `definition` covers the name matrix (valid, empty, punctuation start, 128 vs 129 bytes, spaces, unicode), blank fields, the resource-path matrix, duplicate paths, markdown round-trip with version/metadata/multi-line bodies, CRLF+BOM input, structural parse errors, JSON round-trip with unknown fields, and schema generation.
- `template` covers strict missing-variable errors, whitespace trimming, literal empty/brace runs, single-pass substitution, nested braces, unicode, and placeholder listing.
- `bundle` round-trips through `openjiuwen_test_support::TempWorkspace` with a nested `references/guide.md`, asserts the exact on-disk `SKILL.md`, and checks that non-UTF-8 files and symlinks are skipped.
- `creator` drives `openjiuwen_test_support::ScriptedModel` (fenced JSON, plain text, incomplete payloads, language selection, `call_count` assertions), covers the no-model, unsupported-language, and invalid-hint errors, and exercises `create_from_url` against a `ScriptedFetcher` bundle.
- `creator`'s `create_from_url` tests assert the provenance matrix: no model → `Deterministic`, an accepted model answer → `Model`, an unusable answer → `Deterministic` after one call, and an `OmniOutcome` JSON without the `provenance` field → `Deterministic` (legacy default).
- `omni` tests are documented in `omni/CONTEXT.md`: loopback TCP HTTP tests, HTML fixtures, an in-test ustar builder, archive/document page inputs, filter/dedupe matrices, and URL → bundle end-to-end runs through `ScriptedFetcher`, `TempWorkspace`, and `ScriptedModel`.
- Everything is offline and deterministic: no network, no environment reads, no wall-clock dependence.

## Routing Table

- `./definition.rs` → `SkillDefinition`, `SkillResource`, `SkillFrontmatter`, name/path validation, markdown codec
- `./template.rs` → strict `{{name}}` rendering and placeholder listing
- `./bundle.rs` → `SkillBundle::write`/`read`, `SKILL.md` + resource tree on disk
- `./creator.rs` → `SkillSpec`, `SkillRequest`, `SkillCreator`, model-assisted drafting, prompt wording, `create_from_url` (returning the provenance-bearing `OmniOutcome`)
- `./omni/` → the URL → bundle pipeline: fetch seam, HTML extraction, archives, filters, stages, rendering (`omni/CONTEXT.md`)
- `./mod.rs` → module tree, crate-level re-exports, usage example
