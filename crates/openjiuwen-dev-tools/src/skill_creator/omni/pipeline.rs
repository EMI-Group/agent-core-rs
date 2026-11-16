//! The staged omni-creation pipeline: URL → scraped page → assets → skill bundle.
//!
//! Ports stages 1–5 of the Python `skill_omni_creation/scripts/` without browser
//! automation, image decoding, or binary storage: [`OmniCreation::scrape`] turns
//! an HTML, plain-text, `tar`, or binary-document page into a [`ScrapedPage`]
//! (the document route lives in `documents`);
//! [`OmniCreation::download`] fetches the bounded image and subpage candidates
//! and applies the deterministic [`super::filter`] rules there (there is no
//! separate filtering stage that could be skipped); [`OmniCreation::generate`]
//! renders a `SKILL.md` deterministically while [`OmniCreation::generate_with_model`]
//! asks a model once and falls back on any failure, and the profiled variants
//! ([`OmniCreation::generate_profiled`],
//! [`OmniCreation::generate_with_model_profiled`]) report the
//! [`GenerationProvenance`] that produced the document; and
//! [`OmniCreation::text_resources`] / [`OmniCreation::save`] write the text
//! assets beside the document.
//!
//! The document bodies themselves — the skill name and description, the
//! `SKILL.md` instruction body, the `references/` documents, and the model
//! prompt — are rendered by [`super::render`]; this module owns the caps, the
//! fetching, and the stage orchestration.
//!
//! Kept images are persisted beside the text assets as
//! `references/img_NN.<ext>` by [`OmniCreation::save_with_images`], exactly like
//! Python's `stage_04_save.py`; the instruction body references a kept image by
//! that bundle-relative path, while `references/sources.md` keeps its source URL
//! and notes the saved path. The model-assisted path is hallucination-safe: the
//! drafted body may only cite the saved paths it was given, its invented image
//! references are stripped, and the single `## Reference Files` section is
//! rendered from the same helper as the deterministic path.
//! Every [`DownloadedResource`] carries the payload's SHA-256 digest and, for a
//! kept image, the bundle-relative
//! [`saved_path`](DownloadedResource::saved_path) it is written to; because
//! [`SkillResource`] content is text and a bundle reader skips non-UTF-8 files,
//! a written image never comes back as a text resource. Playwright, Pillow, the
//! LLM image-relevance judge, and concurrent fetching are deliberately not ported:
//! resources are fetched sequentially in first-seen order. Every cap comes from
//! [`OmniConfig`] (a count cap of zero disables that class), and every stage is
//! deterministic given its inputs — no timestamps, no environment reads, no
//! global state, and network access only through the injected [`Fetcher`].

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use crate::skill_creator::{SkillBundle, SkillDefinition, SkillResource};
use crate::util::extract_json_fence;

use self::assets::{image_resource, image_target};
use self::documents::{document_page, is_document};
use self::page::{archive_page, html_page};
use super::archive::{self, ArchiveFormat, ArchiveUnpacker};
use super::fetch::Fetcher;
use super::filter::{
    content_hash, dedupe_links, filter_images_with, same_host, ImageCandidate, ResourceFilter,
};
use super::html::{resolve_url, Heading, Image, Link};
use super::render::{
    instructions_markdown, model_user_prompt, reference_files_section, render_page_resource,
    same_page, skill_description, skill_name, sources_markdown, strip_hallucinated_images,
    ModelSkillPayload, MODEL_SYSTEM_PROMPT, PAGE_RESOURCE_PREFIX, SOURCES_RESOURCE,
};
use super::sha256;

/// Default cap on followed subpages.
const DEFAULT_MAX_LINKS: usize = 8;
/// Default cap on downloaded images.
const DEFAULT_MAX_IMAGES: usize = 16;
/// Default cap on the combined downloaded (`images` + subpages) list.
const DEFAULT_MAX_RESOURCES: usize = 32;
/// Default lower bound for a downloaded image payload: 1 KiB.
const DEFAULT_MIN_IMAGE_BYTES: usize = 1024;
/// Default upper bound for a downloaded image payload: 8 MiB.
const DEFAULT_MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
/// Default upper bound for a downloaded subpage: 2 MiB.
const DEFAULT_MAX_RESOURCE_BYTES: usize = 2 * 1024 * 1024;
/// How many leading bytes are inspected when sniffing a resource for HTML.
const HTML_SNIFF_BYTES: usize = 1024;
/// Caps and policy for one omni-creation run.
///
/// The defaults mirror the Python script constants: eight subpages, sixteen
/// images, thirty-two combined resources, a 1 KiB image floor, an 8 MiB image
/// ceiling, and a 2 MiB per-subpage ceiling. Count caps of zero are valid and
/// disable that class of work; byte bounds must be non-zero and ordered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OmniConfig {
    /// Maximum number of subpages followed from the source page.
    pub max_links: usize,
    /// Maximum number of images downloaded from the source page.
    pub max_images: usize,
    /// Maximum size of the combined downloaded image and subpage list.
    pub max_resources: usize,
    /// Payloads smaller than this many bytes are dropped.
    pub min_image_bytes: usize,
    /// Image responses larger than this many bytes are ignored.
    pub max_image_bytes: usize,
    /// Subpage responses larger than this many bytes are ignored.
    pub max_resource_bytes: usize,
    /// Whether resources on hosts other than the source page are followed.
    pub allow_cross_host: bool,
}

impl Default for OmniConfig {
    fn default() -> Self {
        Self {
            max_links: DEFAULT_MAX_LINKS,
            max_images: DEFAULT_MAX_IMAGES,
            max_resources: DEFAULT_MAX_RESOURCES,
            min_image_bytes: DEFAULT_MIN_IMAGE_BYTES,
            max_image_bytes: DEFAULT_MAX_IMAGE_BYTES,
            max_resource_bytes: DEFAULT_MAX_RESOURCE_BYTES,
            allow_cross_host: false,
        }
    }
}

impl OmniConfig {
    /// Returns the configuration with a different subpage cap.
    #[must_use]
    pub fn with_max_links(mut self, max_links: usize) -> Self {
        self.max_links = max_links;
        self
    }

    /// Returns the configuration with a different image cap.
    #[must_use]
    pub fn with_max_images(mut self, max_images: usize) -> Self {
        self.max_images = max_images;
        self
    }

    /// Returns the configuration with a different combined-resource cap.
    #[must_use]
    pub fn with_max_resources(mut self, max_resources: usize) -> Self {
        self.max_resources = max_resources;
        self
    }

    /// Returns the configuration with a different image-size floor.
    #[must_use]
    pub fn with_min_image_bytes(mut self, min_image_bytes: usize) -> Self {
        self.min_image_bytes = min_image_bytes;
        self
    }

    /// Returns the configuration with a different image-response ceiling.
    #[must_use]
    pub fn with_max_image_bytes(mut self, max_image_bytes: usize) -> Self {
        self.max_image_bytes = max_image_bytes;
        self
    }

    /// Returns the configuration with a different subpage-response ceiling.
    #[must_use]
    pub fn with_max_resource_bytes(mut self, max_resource_bytes: usize) -> Self {
        self.max_resource_bytes = max_resource_bytes;
        self
    }

    /// Returns the configuration with a different cross-host policy.
    #[must_use]
    pub fn with_allow_cross_host(mut self, allow_cross_host: bool) -> Self {
        self.allow_cross_host = allow_cross_host;
        self
    }

    /// Rejects a configuration whose byte bounds cannot describe a usable range.
    ///
    /// Count caps are allowed to be zero: they mean "disable that class".
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// naming the offending field (`min_image_bytes`, `max_image_bytes`, or
    /// `max_resource_bytes`) when it is zero, or naming both image bounds when
    /// `min_image_bytes` exceeds `max_image_bytes`.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("min_image_bytes", self.min_image_bytes),
            ("max_image_bytes", self.max_image_bytes),
            ("max_resource_bytes", self.max_resource_bytes),
        ] {
            if value == 0 {
                return Err(Error::validation(format!(
                    "{field} must be greater than zero, got 0"
                )));
            }
        }
        if self.min_image_bytes > self.max_image_bytes {
            return Err(Error::validation(format!(
                "min_image_bytes ({}) must not exceed max_image_bytes ({})",
                self.min_image_bytes, self.max_image_bytes
            )));
        }
        self.resource_filter().validate()
    }

    /// The image filter implied by this configuration.
    #[must_use]
    pub fn resource_filter(&self) -> ResourceFilter {
        ResourceFilter {
            min_image_bytes: self.min_image_bytes,
            max_image_bytes: self.max_image_bytes,
            allow_cross_host: self.allow_cross_host,
        }
    }
}

/// The information the pipeline needs from one scraped source.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ScrapedPage {
    /// The final URL that answered, or the requested URL when none was reported.
    pub url: String,
    /// The page title, empty for text and archive inputs without a level-1 heading.
    pub title: String,
    /// The page description, empty for text and archive inputs.
    pub description: String,
    /// The tag-free page text.
    pub text: String,
    /// Headings in document or archive order.
    pub headings: Vec<Heading>,
    /// Absolute `http(s)` anchors in document order.
    pub links: Vec<Link>,
    /// Absolute `http(s)` images in document order.
    pub images: Vec<Image>,
    /// Fenced or `<pre>`/`<code>` code blocks in document order.
    pub code_blocks: Vec<String>,
}

impl ScrapedPage {
    /// Reports whether the page yielded nothing worth turning into a skill.
    ///
    /// The check mirrors the Python scrape guard: text, headings, links, and
    /// images must all be empty. A title or description alone does not make a
    /// page usable, and code blocks are covered by the text.
    #[must_use]
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
            && self.headings.is_empty()
            && self.links.is_empty()
            && self.images.is_empty()
    }
}

/// One downloaded resource: an image payload or a followed HTML subpage.
///
/// The payload is identified by its SHA-256 digest, the same
/// `hashlib.sha256(data).hexdigest()` identity the download filter deduplicates
/// on, and an image that survives filtering also carries `saved_path`, the
/// bundle-relative asset path [`OmniCreation::save_with_images`] writes it to
/// (`references/img_NN.<ext>`). Subpages have no `saved_path` and no `alt`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DownloadedResource {
    /// The URL that answered.
    pub url: String,
    /// The `content-type` reported by the server, verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// The raw response body.
    pub bytes: Vec<u8>,
    /// The scraped alt text of an image, empty for a subpage.
    pub alt: String,
    /// The lowercase hex SHA-256 digest of [`bytes`](Self::bytes).
    ///
    /// This is the payload identity: two URLs serving the same bytes have the
    /// same digest, so the value is provenance as well as a dedupe key.
    pub sha256: String,
    /// The bundle-relative path a kept image is written to, if any.
    ///
    /// `Some("references/img_00.png")` for a kept image, `None` for a subpage
    /// and for an image that is never persisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saved_path: Option<String>,
}

impl DownloadedResource {
    /// Reports whether the payload is an HTML document.
    ///
    /// The `content-type` header wins when it names `text/html` or
    /// `application/xhtml…`; otherwise the first 1 KiB of the body is inspected
    /// for a `<!doctype html` prefix or an `<html` tag. The same rule as
    /// [`super::fetch::FetchedResource::is_html`], applied to a buffered resource.
    #[must_use]
    pub fn is_html(&self) -> bool {
        if let Some(content_type) = self.content_type.as_deref() {
            let mime = content_type
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if mime.starts_with("text/html") || mime.starts_with("application/xhtml") {
                return true;
            }
        }
        let head = &self.bytes[..self.bytes.len().min(HTML_SNIFF_BYTES)];
        let lowered = String::from_utf8_lossy(head)
            .trim_start()
            .to_ascii_lowercase();
        lowered.starts_with("<!doctype html") || lowered.contains("<html")
    }
}

/// Which generator produced a skill document.
///
/// The deterministic template renderer always succeeds for a usable page, while
/// the model-assisted generator may be unusable (model error, no fenced JSON,
/// malformed payload, instructions that are blank once hallucinated image
/// references are removed, or an invalid definition) and then falls back to the
/// deterministic renderer. The provenance makes that choice observable instead
/// of only being logged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationProvenance {
    /// The deterministic template renderer produced the document.
    #[default]
    Deterministic,
    /// The attached drafting model produced the document.
    Model,
}

impl GenerationProvenance {
    /// The stable `snake_case` token of this provenance (`"deterministic"` or `"model"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Model => "model",
        }
    }
}

impl fmt::Display for GenerationProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A generated skill document together with the generator that produced it.
///
/// Returned by [`OmniCreation::generate_profiled`] and
/// [`OmniCreation::generate_with_model_profiled`], so a caller of the generation
/// stage can tell a model-drafted document from the deterministic template
/// fallback without inspecting the log stream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GeneratedSkill {
    /// The generated skill, including every text resource of the run.
    pub definition: SkillDefinition,
    /// Which generator produced [`definition`](Self::definition).
    pub provenance: GenerationProvenance,
}

/// The artifact of one successful [`OmniCreation::run`].
///
/// [`provenance`](Self::provenance) records whether the drafting model produced
/// the document or the deterministic template renderer did — check it instead of
/// assuming the attached model was used, because the model-assisted generator
/// falls back to the deterministic renderer on any unusable answer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OmniOutcome {
    /// The generated skill, including every text resource the pipeline wrote.
    pub definition: SkillDefinition,
    /// The relative paths of the written assets, in write order.
    ///
    /// The text resources of the definition come first, then the bundle-relative
    /// paths of the persisted images ([`DownloadedResource::saved_path`] in
    /// download order).
    pub resources: Vec<String>,
    /// The skill directory under the requested root.
    pub bundle_path: PathBuf,
    /// Which generator produced [`definition`](Self::definition).
    ///
    /// [`GenerationProvenance::Model`] means the attached drafting model drafted
    /// the document; [`GenerationProvenance::Deterministic`] means the
    /// deterministic template renderer produced it — either because no model was
    /// attached or because the model answer was unusable.
    pub provenance: GenerationProvenance,
}

/// The staged URL-to-skill pipeline.
///
/// One value carries the fetching seam, the optional drafting model, and the
/// run policy, so every stage reads the same configuration. Construction needs
/// a [`Fetcher`]; [`OmniCreation::with_model`] enables the model-assisted
/// generator, and [`OmniCreation::with_config`] overrides [`OmniConfig`].
#[derive(Clone)]
pub struct OmniCreation {
    /// Where pages and resources are fetched from.
    fetcher: Arc<dyn Fetcher>,
    /// The optional drafting model.
    model: Option<Arc<dyn Model>>,
    /// The caps and policy of this run.
    config: OmniConfig,
}

impl fmt::Debug for OmniCreation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OmniCreation")
            .field("fetcher", &"dyn Fetcher")
            .field("has_model", &self.model.is_some())
            .field("config", &self.config)
            .finish()
    }
}

impl OmniCreation {
    /// Creates a deterministic pipeline over `fetcher` with [`OmniConfig::default`].
    #[must_use]
    pub fn new(fetcher: Arc<dyn Fetcher>) -> Self {
        Self {
            fetcher,
            model: None,
            config: OmniConfig::default(),
        }
    }

    /// Attaches the model used by [`OmniCreation::generate_with_model`].
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the run configuration.
    #[must_use]
    pub fn with_config(mut self, config: OmniConfig) -> Self {
        self.config = config;
        self
    }

    /// The attached drafting model, when one is configured.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// The run configuration.
    #[must_use]
    pub fn config(&self) -> &OmniConfig {
        &self.config
    }

    /// Fetches `url` and extracts a [`ScrapedPage`].
    ///
    /// HTML pages become title, description, text, headings, links, images, and
    /// code blocks, with every link and image resolved to an absolute `http(s)`
    /// URL. A `tar` body is unpacked and read as markdown: only file entries
    /// (never directory, symlink, or hardlink entries, whose link targets are
    /// not followed) with valid UTF-8 and no NUL byte in their first 512 bytes
    /// contribute text, ATX headings (`#`…`######` plus a space), and fenced
    /// code blocks; the title is the first level-1 heading. The archive format
    /// is sniffed by magic bytes first and by the URL suffix second, so a
    /// `bzip2`/`xz`/`zstd` container is recognized (and refused for its missing
    /// decoder) even when its body carries no usable signature. A resource that
    /// names a document the kernel's retrieval loaders read (`PDF`, `DOCX`,
    /// `XLSX`, `PPTX`, `ODT`, and the text, Markdown, `CSV`/`TSV`, and `JSON`
    /// formats — by URL suffix, else by `content-type`) is extracted by
    /// `documents` instead, so a usable page can come from a
    /// binary document. An office document is a `ZIP` container, so such a
    /// resource is dispatched to its loader before the archive reader unpacks the
    /// package parts as text. Any other body is treated as plain text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid configuration, a page with no text, headings, links, or
    /// images, or a declared document format whose loader and plain-text
    /// fallback both carry nothing (the message names the URL and the format);
    /// [`Error::unsupported`](openjiuwen_common::error::Error::unsupported) for a
    /// `bzip2`, `xz`, or `zstd` page input, whose decoder is not bundled (the
    /// message names the codec); and whatever the [`Fetcher`] reports.
    pub async fn scrape(&self, url: &str) -> Result<ScrapedPage> {
        self.config.validate()?;
        let resource = self.fetcher.fetch(url).await?;
        let final_url = if resource.final_url.is_empty() {
            url.to_owned()
        } else {
            resource.final_url.clone()
        };
        let page = match archive::detect_named(&final_url, &resource.body) {
            // An office document (`DOCX`/`XLSX`/`PPTX`/`ODT`) is itself a ZIP
            // container, so a resource that names one is handed to its document
            // loader instead of having the package parts unpacked as text. The
            // archive route below stays the fallback when the name is ambiguous.
            Some(ArchiveFormat::Zip)
                if is_document(&final_url, resource.content_type.as_deref()) =>
            {
                if let Some(page) = document_page(&resource, &final_url).await? {
                    page
                } else {
                    let entries = ArchiveUnpacker::unpack(ArchiveFormat::Zip, &resource.body)?;
                    archive_page(&final_url, &entries)
                }
            }
            // `tar` is unpacked in-process; `zip`/`gzip` propagate their typed
            // validation errors from `unpack`, and `bzip2`/`xz`/`zstd`
            // propagate its typed `Unsupported` error naming the codec instead
            // of being read as plain text. The format is the one the sniff
            // settled on, so a body whose signature is unrecognized but whose
            // URL names a `bzip2`/`xz`/`zstd` container still reports that
            // codec's capability error instead of a generic parse failure.
            Some(
                format @ (ArchiveFormat::Tar
                | ArchiveFormat::Zip
                | ArchiveFormat::Gzip
                | ArchiveFormat::Bzip2
                | ArchiveFormat::Xz
                | ArchiveFormat::Zstd),
            ) => {
                let entries = ArchiveUnpacker::unpack(format, &resource.body)?;
                archive_page(&final_url, &entries)
            }
            None if resource.is_html() => html_page(&resource, &final_url),
            None => {
                if let Some(page) = document_page(&resource, &final_url).await? {
                    page
                } else {
                    ScrapedPage {
                        url: final_url,
                        text: resource.text(),
                        ..ScrapedPage::default()
                    }
                }
            }
        };
        if page.is_blank() {
            return Err(Error::validation(format!(
                "scrape of {url} extracted no text, links, or images"
            )));
        }
        Ok(page)
    }

    /// Downloads the bounded image and subpage candidates of `page`.
    ///
    /// Images are considered in document order, same-host only unless
    /// [`OmniConfig::allow_cross_host`] is set, deduplicated by the [`content_hash`]
    /// of their URL, and capped at [`OmniConfig::max_images`]. Subpages come from
    /// [`dedupe_links`] on the page anchors, exclude the page itself, and are
    /// capped at [`OmniConfig::max_links`] fetch attempts. Everything is fetched
    /// sequentially so request order is reproducible: a page with `n` candidates
    /// issues exactly `n` fetches in list order, never a concurrent swarm.
    ///
    /// Per-resource failures, non-success statuses, oversized bodies, and
    /// non-HTML subpages are logged with [`tracing::warn`] and skipped instead of
    /// failing the stage. Images are filtered once by
    /// [`ResourceFilter`] after their payloads are known: the keep policy is
    /// PNG, JPEG, GIF, or WebP, so an `svg` payload is rejected by
    /// [`filter_images_with`] even though [`sniff_image`](super::filter::sniff_image)
    /// classifies it. Every kept image is given its `references/img_NN.<ext>`
    /// [`saved_path`](DownloadedResource::saved_path), numbered from `00` in kept
    /// order, plus the SHA-256 of its payload; a first-seen duplicate payload is
    /// dropped by the filter, so the numbering follows the surviving order.
    /// The result is accepted images first (display order), then subpages (link
    /// order), truncated to [`OmniConfig::max_resources`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the configuration is invalid. Fetch failures never escape.
    pub async fn download(&self, page: &ScrapedPage) -> Result<Vec<DownloadedResource>> {
        self.config.validate()?;
        let mut fetched = Vec::new();
        for mut candidate in self.image_candidates(page) {
            match self.fetcher.fetch(&candidate.url).await {
                Ok(response) if response.is_success() => {
                    if response.body.len() > self.config.max_image_bytes {
                        tracing::warn!(
                            url = %candidate.url,
                            bytes = response.body.len(),
                            "skipping an oversized image response"
                        );
                        continue;
                    }
                    candidate.bytes = Some(response.body);
                    candidate.content_type = response.content_type;
                    fetched.push(candidate);
                }
                Ok(response) => tracing::warn!(
                    url = %candidate.url,
                    status = response.status,
                    "skipping a non-success image response"
                ),
                Err(error) => tracing::warn!(
                    url = %candidate.url,
                    %error,
                    "skipping a failed image fetch"
                ),
            }
        }
        let mut resources = Vec::new();
        for (index, candidate) in filter_images_with(&fetched, &self.config.resource_filter())
            .into_iter()
            .enumerate()
        {
            resources.push(image_resource(index, candidate));
        }

        let mut attempted = 0;
        for link in dedupe_links(&page.links) {
            if attempted >= self.config.max_links {
                break;
            }
            let Some(target) = resolve_url(&page.url, &link.href) else {
                continue;
            };
            if !self.config.allow_cross_host && !same_host(&target, &page.url) {
                continue;
            }
            if same_page(&target, &page.url) {
                continue;
            }
            attempted += 1;
            match self.fetcher.fetch(&target).await {
                Ok(response) if response.is_success() && response.is_html() => {
                    if response.body.len() > self.config.max_resource_bytes {
                        tracing::warn!(
                            url = %target,
                            bytes = response.body.len(),
                            "skipping an oversized subpage response"
                        );
                        continue;
                    }
                    let digest = sha256::hex_digest(&response.body);
                    resources.push(DownloadedResource {
                        url: target,
                        content_type: Some("text/html".to_owned()),
                        bytes: response.body,
                        alt: String::new(),
                        sha256: digest,
                        saved_path: None,
                    });
                }
                Ok(response) => tracing::warn!(
                    url = %target,
                    status = response.status,
                    "skipping a non-HTML subpage response"
                ),
                Err(error) => tracing::warn!(
                    url = %target,
                    %error,
                    "skipping a failed subpage fetch"
                ),
            }
        }
        resources.truncate(self.config.max_resources);
        Ok(resources)
    }

    /// Renders the deterministic skill for `page` and `resources`.
    ///
    /// This is [`OmniCreation::generate_profiled`] without the provenance: the
    /// result is exactly the [`GeneratedSkill::definition`] of the deterministic
    /// rendering, which always reports
    /// [`GenerationProvenance::Deterministic`]. Use the profiled variant when the
    /// caller must distinguish a model-drafted document from this fallback.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the rendered definition does not validate (an empty name, a blank
    /// description or body, or an unsafe resource path).
    pub fn generate(
        &self,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Result<SkillDefinition> {
        self.generate_profiled(page, resources)
            .map(|generated| generated.definition)
    }

    /// Renders the deterministic skill for `page` and `resources`, with provenance.
    ///
    /// The name is [`slugify`](super::filter::slugify) of the title, then of the URL host (or the URL
    /// itself), then `scraped-skill`. The description is the page description,
    /// else the first non-empty text line, capped at 300 characters on a
    /// character boundary, else a URL-based sentence. The instruction body is an
    /// `#` title, the description, and the `## Steps`, `## Details`,
    /// `## Code Examples`, `## Reference Images`, `## Further Reading`, and
    /// `## Reference Files` sections that have content. A kept image (one with a
    /// [`saved_path`](DownloadedResource::saved_path)) is referenced by that
    /// bundle-relative path, an image that did not survive filtering stays a
    /// URL line, and `## Reference Files` lists the saved images beside the text
    /// resources. Link references in the body stay URLs, and the definition
    /// carries every text resource of [`OmniCreation::text_resources`] —
    /// persisted images are not text resources, they are written beside them by
    /// [`OmniCreation::save_with_images`].
    ///
    /// The returned provenance is always
    /// [`GenerationProvenance::Deterministic`], because no model is consulted
    /// here; this is the path a caller can trust to work offline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when the rendered definition does not validate (an empty name, a blank
    /// description or body, or an unsafe resource path).
    pub fn generate_profiled(
        &self,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Result<GeneratedSkill> {
        let name = skill_name(page);
        let description = skill_description(page);
        let text_resources = self.text_resources(page, resources);
        let instructions = instructions_markdown(
            page,
            &name,
            &description,
            &text_resources,
            resources,
            &self.config,
        );
        let mut definition = SkillDefinition::new(name, description, instructions);
        definition.resources = text_resources;
        definition.validate()?;
        Ok(GeneratedSkill {
            definition,
            provenance: GenerationProvenance::Deterministic,
        })
    }

    /// Writes `definition` plus `resources` as a bundle under `root`.
    ///
    /// Resources already present in the definition are skipped, every appended
    /// resource is validated, and the directory returned by [`SkillBundle::write`]
    /// is handed back. No image payload is written; use
    /// [`OmniCreation::save_with_images`] when downloaded images must land in the
    /// bundle too.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid resource path or definition, plus every filesystem failure
    /// [`SkillBundle::write`] reports.
    pub async fn save(
        &self,
        definition: &SkillDefinition,
        resources: &[SkillResource],
        root: &Path,
    ) -> Result<PathBuf> {
        self.save_with_images(definition, resources, &[], root)
            .await
    }

    /// Writes `definition`, `resources`, and every kept image as one bundle.
    ///
    /// The text part is exactly [`OmniCreation::save`]: resources already present
    /// in the definition are skipped, every appended resource is validated, and
    /// [`SkillBundle::write`] produces `<root>/<name>/SKILL.md` plus the text
    /// assets. Then every image whose
    /// [`saved_path`](DownloadedResource::saved_path) is `Some` is written
    /// byte-for-byte to that path beneath the bundle directory (its parent is
    /// created first), mirroring Python's `stage_04_save.py`.
    ///
    /// The `saved_path` invariant is checked before anything is written: it must
    /// be a relative, `/`-separated path whose first component is `references`
    /// and whose other components are non-empty, free of `.`/`..`, and free of
    /// backslashes, so an image can never escape its bundle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid resource path, definition, or image `saved_path` (the
    /// message names the path), plus every filesystem failure of
    /// [`SkillBundle::write`] or of an image write.
    pub async fn save_with_images(
        &self,
        definition: &SkillDefinition,
        resources: &[SkillResource],
        images: &[DownloadedResource],
        root: &Path,
    ) -> Result<PathBuf> {
        let mut definition = definition.clone();
        for resource in resources {
            resource.validate()?;
            if definition
                .resources
                .iter()
                .any(|existing| existing.path == resource.path)
            {
                continue;
            }
            definition.resources.push(resource.clone());
        }
        // Validate every image path up front so a rejected path writes nothing.
        let mut image_targets = Vec::new();
        for image in images {
            let Some(saved_path) = image.saved_path.as_deref() else {
                continue;
            };
            image_targets.push((image_target(saved_path)?, image.bytes.as_slice()));
        }
        let bundle_path = SkillBundle::write(&definition, root).await?;
        for (relative, bytes) in image_targets {
            let target = bundle_path.join(relative);
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&target, bytes).await?;
        }
        Ok(bundle_path)
    }

    /// The deterministic text assets of a bundle for `page` and `resources`.
    ///
    /// `references/sources.md` is always first when the cap allows it: source URL,
    /// title, description, the bounded deduplicated subpage links, and every
    /// image URL with its alt text — plus the saved path of an image that was
    /// persisted, as `(saved as references/img_NN.<ext>)`. Then each HTML
    /// [`DownloadedResource`] with a non-blank document becomes
    /// `references/page-NN.md` (two-digit ordinal among the written reference
    /// pages), rendering the page title, its headings as `##`/`###`, and its
    /// text. The list is truncated to [`OmniConfig::max_resources`], and a cap
    /// of zero returns nothing.
    #[must_use]
    pub fn text_resources(
        &self,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Vec<SkillResource> {
        let mut assets = Vec::new();
        if self.config.max_resources == 0 {
            return assets;
        }
        assets.push(SkillResource::new(
            SOURCES_RESOURCE,
            sources_markdown(page, resources, &self.config),
        ));
        let mut page_index = 0_usize;
        for resource in resources {
            if assets.len() >= self.config.max_resources {
                break;
            }
            if !resource.is_html() {
                continue;
            }
            let Some(content) = render_page_resource(page, resource) else {
                continue;
            };
            let path = format!("{PAGE_RESOURCE_PREFIX}{page_index:02}.md");
            page_index += 1;
            assets.push(SkillResource::new(path, content));
        }
        assets
    }

    /// Drafts the skill with one model call, falling back to [`OmniCreation::generate`].
    ///
    /// The model receives the page title, source URL, headings, text (truncated
    /// to 8 000 characters on a character boundary), code blocks, and the
    /// images with their alt text — a kept image as its bundle-relative
    /// `saved_path`, never the raw URL — and must answer with one fenced `json`
    /// object holding `name`, `description`, `instructions`, and optionally
    /// `metadata`.
    ///
    /// The drafted instructions are cleaned before they are accepted: every
    /// `![…](…)` reference whose target is not one of the kept images'
    /// `saved_path`s is dropped (see `strip_hallucinated_images`), and the
    /// single `## Reference Files` section of the run
    /// (`reference_files_section`) is appended unless the cleaned body already
    /// carries a `## Reference Files` heading — an existing section is kept
    /// as-is rather than merged. The definition then carries the provenance and
    /// page resources of [`OmniCreation::text_resources`].
    ///
    /// Any failure after a model is attached — a model error, an answer without
    /// a JSON fence, a malformed or incomplete payload, instructions that are
    /// blank once the hallucinated image references are removed, or a definition
    /// that does not validate — is logged with
    /// `openjiuwen_common::logging::warn!` (recording
    /// `provenance = Deterministic` beside the cause) and replaced by the
    /// deterministic rendering, so the stage stays productive offline. The
    /// returned definition does not say which path produced it: use
    /// [`OmniCreation::generate_with_model_profiled`] (or
    /// [`OmniCreation::run`], whose [`OmniOutcome::provenance`] is set) when that
    /// distinction matters.
    ///
    /// # Errors
    ///
    /// The only error that escapes is
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when no
    /// model is attached (the message names `OmniCreation::with_model`).
    pub async fn generate_with_model(
        &self,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Result<SkillDefinition> {
        self.generate_with_model_profiled(page, resources)
            .await
            .map(|generated| generated.definition)
    }

    /// Drafts the skill with one model call and reports which path produced it.
    ///
    /// The drafting rules are exactly [`OmniCreation::generate_with_model`]. The
    /// returned provenance is [`GenerationProvenance::Model`] when the model
    /// answer was usable and [`GenerationProvenance::Deterministic`] when the
    /// answer was unusable and the deterministic renderer took over; that
    /// fallback warns with `openjiuwen_common::logging::warn!`, recording the
    /// cause and `provenance = Deterministic`, so a caller can either observe
    /// the returned value or the log stream.
    ///
    /// # Errors
    ///
    /// The only error that escapes is
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when no
    /// model is attached (the message names `OmniCreation::with_model`).
    pub async fn generate_with_model_profiled(
        &self,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Result<GeneratedSkill> {
        let Some(model) = self.model.as_ref() else {
            return Err(Error::validation(
                "model-assisted omni creation requires a pipeline built with \
                 OmniCreation::with_model",
            ));
        };
        match self.model_definition(model, page, resources).await {
            Ok(definition) => Ok(GeneratedSkill {
                definition,
                provenance: GenerationProvenance::Model,
            }),
            Err(error) => {
                openjiuwen_common::logging::warn!(
                    %error,
                    provenance = %GenerationProvenance::Deterministic,
                    "model-assisted skill generation failed; using the deterministic generator"
                );
                self.generate_profiled(page, resources)
            }
        }
    }

    /// Runs scrape → download → generate → save and reports the artifact.
    ///
    /// The drafting model is used when attached (with its deterministic
    /// fallback); otherwise the deterministic generator consumes the downloaded
    /// resources and exactly one `openjiuwen_common::logging::warn!` names `url`
    /// and states that no drafting model is attached, so the deterministic
    /// template generator produced the skill (and that
    /// [`OmniCreation::with_model`] enables model drafting). The bundle keeps
    /// both the text resources and every downloaded image,
    /// [`OmniOutcome::resources`] lists the text paths first and then the image
    /// paths, and [`OmniOutcome::provenance`] names the generator that produced
    /// the document.
    ///
    /// # Errors
    ///
    /// Returns every error of [`OmniCreation::scrape`],
    /// [`OmniCreation::download`], [`OmniCreation::generate_profiled`], and
    /// [`OmniCreation::save_with_images`]; a model failure is never one of them
    /// because the model-assisted generator falls back.
    pub async fn run(&self, url: &str, root: &Path) -> Result<OmniOutcome> {
        self.config.validate()?;
        let page = self.scrape(url).await?;
        let downloaded = self.download(&page).await?;
        let generated = if self.model.is_some() {
            self.generate_with_model_profiled(&page, &downloaded)
                .await?
        } else {
            openjiuwen_common::logging::warn!(
                %url,
                provenance = %GenerationProvenance::Deterministic,
                "no drafting model is attached, so the deterministic template generator \
                 produced the skill; build the pipeline with OmniCreation::with_model \
                 to enable model drafting"
            );
            self.generate_profiled(&page, &downloaded)?
        };
        let bundle_path = self
            .save_with_images(&generated.definition, &[], &downloaded, root)
            .await?;
        let mut resources: Vec<String> = generated
            .definition
            .resources
            .iter()
            .map(|resource| resource.path.clone())
            .collect();
        resources.extend(
            downloaded
                .iter()
                .filter_map(|image| image.saved_path.clone()),
        );
        Ok(OmniOutcome {
            definition: generated.definition,
            resources,
            bundle_path,
            provenance: generated.provenance,
        })
    }

    /// One model round trip that must answer with a fenced skill definition.
    ///
    /// The drafted body is made hallucination-safe before validation: the
    /// `saved_path`s of the kept images are the only accepted image targets, so
    /// any other `![…](…)` reference is removed, and a body that only held such
    /// references is rejected so the caller falls back to the deterministic
    /// renderer. A body without a `## Reference Files` heading then receives the
    /// run's single reference-files section, which lists the saved images and
    /// the text resources; a body that already carries the heading is left
    /// untouched (the section is not merged or replaced — a documented deviation
    /// from Python, whose model answers never contain the section).
    async fn model_definition(
        &self,
        model: &Arc<dyn Model>,
        page: &ScrapedPage,
        resources: &[DownloadedResource],
    ) -> Result<SkillDefinition> {
        let completion = model
            .invoke(
                &[
                    BaseMessage::system(MODEL_SYSTEM_PROMPT),
                    BaseMessage::user(model_user_prompt(page, resources, &self.config)),
                ],
                &ModelRequestConfig::new(model.model_name()),
            )
            .await?;
        let answer = completion
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        let Some(payload) = extract_json_fence(&answer) else {
            return Err(Error::validation(
                "model answer contained no ```json fenced skill definition",
            ));
        };
        let payload: ModelSkillPayload = serde_json::from_value(payload).map_err(|error| {
            Error::validation(format!(
                "invalid omni skill definition JSON from model: {error}"
            ))
        })?;
        let valid_paths: Vec<String> = resources
            .iter()
            .filter_map(|resource| resource.saved_path.clone())
            .collect();
        let instructions =
            strip_hallucinated_images(&payload.required("instructions")?, &valid_paths);
        if instructions.trim().is_empty() {
            return Err(Error::validation(
                "model instructions were empty once hallucinated image references were removed",
            ));
        }
        let text_resources = self.text_resources(page, resources);
        let instructions = if instructions.contains("## Reference Files") {
            instructions
        } else {
            match reference_files_section(resources, &text_resources) {
                Some(section) => format!("{instructions}\n\n{section}"),
                None => instructions,
            }
        };
        let mut definition = SkillDefinition::new(
            payload.required("name")?,
            payload.required("description")?,
            instructions,
        );
        definition.metadata = payload.metadata;
        definition.resources = text_resources;
        definition.validate()?;
        Ok(definition)
    }

    /// Builds the image candidates worth fetching for `page`.
    fn image_candidates(&self, page: &ScrapedPage) -> Vec<ImageCandidate> {
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for image in &page.images {
            if candidates.len() >= self.config.max_images {
                break;
            }
            let Some(target) = resolve_url(&page.url, &image.src) else {
                continue;
            };
            if !self.config.allow_cross_host && !same_host(&target, &page.url) {
                continue;
            }
            if !seen.insert(content_hash(target.as_bytes())) {
                continue;
            }
            candidates.push(ImageCandidate::new(target, image.alt.clone()));
        }
        candidates
    }
}

pub(crate) mod assets;
pub(crate) mod documents;
pub(crate) mod page;

#[cfg(test)]
mod tests;
