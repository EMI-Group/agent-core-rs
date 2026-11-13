//! The staged omni-creation pipeline: URL → scraped page → assets → skill bundle.
//!
//! Ports stages 1–5 of the Python `skill_omni_creation/scripts/` without browser
//! automation, image decoding, or binary storage: [`OmniCreation::scrape`] turns
//! an HTML, plain-text, or `tar` page into a [`ScrapedPage`];
//! [`OmniCreation::download`] fetches the bounded image and subpage candidates
//! and applies the deterministic [`super::filter`] rules there (there is no
//! separate filtering stage that could be skipped); [`OmniCreation::generate`]
//! renders a `SKILL.md` deterministically while [`OmniCreation::generate_with_model`]
//! asks a model once and falls back on any failure; and
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
//! Python's `stage_04_save.py`, while the document keeps their source URLs.
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
use super::archive::{self, ArchiveEntry, ArchiveFormat, ArchiveUnpacker};
use super::fetch::{FetchedResource, Fetcher};
use super::filter::{
    content_hash, dedupe_links, filter_images_with, same_host, ImageCandidate, ResourceFilter,
};
use super::html::{resolve_url, Heading, HtmlDocument, Image, Link};
use super::render::{
    instructions_markdown, model_user_prompt, render_page_resource, same_page, skill_description,
    skill_name, sources_markdown, ModelSkillPayload, MODEL_SYSTEM_PROMPT, PAGE_RESOURCE_PREFIX,
    SOURCES_RESOURCE,
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
/// How many leading bytes are inspected when sniffing an archive entry for NULs.
const BINARY_SNIFF_BYTES: usize = 512;
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
    /// [`FetchedResource::is_html`], applied to a buffered resource.
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

/// The artifact of one successful [`OmniCreation::run`].
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
    /// URL. A `tar` body is unpacked and read as markdown: file entries with
    /// valid UTF-8 and no NUL byte in their first 512 bytes contribute text,
    /// ATX headings (`#`…`######` plus a space), and fenced code blocks; the
    /// title is the first level-1 heading. Any other body is treated as plain
    /// text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for an invalid configuration or a page with no text, headings, links, or
    /// images; [`Error::unsupported`](openjiuwen_common::error::Error::unsupported)
    /// for a `zip` or `gzip` page input (the message names the format); and
    /// whatever the [`Fetcher`] reports.
    pub async fn scrape(&self, url: &str) -> Result<ScrapedPage> {
        self.config.validate()?;
        let resource = self.fetcher.fetch(url).await?;
        let final_url = if resource.final_url.is_empty() {
            url.to_owned()
        } else {
            resource.final_url.clone()
        };
        let page = match archive::detect(&resource.body) {
            // `tar` is unpacked in-process; `zip`/`gzip` propagate the typed
            // `Unsupported` error naming the format from `unpack_auto`.
            Some(ArchiveFormat::Tar | ArchiveFormat::Zip | ArchiveFormat::Gzip) => {
                let entries = ArchiveUnpacker::unpack_auto(&resource.body)?;
                archive_page(&final_url, &entries)
            }
            None if resource.is_html() => html_page(&resource, &final_url),
            None => ScrapedPage {
                url: final_url,
                text: resource.text(),
                ..ScrapedPage::default()
            },
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
    /// The name is [`slugify`](super::filter::slugify) of the title, then of the URL host (or the URL
    /// itself), then `scraped-skill`. The description is the page description,
    /// else the first non-empty text line, capped at 300 characters on a
    /// character boundary, else a URL-based sentence. The instruction body is an
    /// `#` title, the description, and the `## Steps`, `## Details`,
    /// `## Code Examples`, `## Reference Images`, `## Further Reading`, and
    /// `## Reference Files` sections that have content. Image and link
    /// references in the body stay URLs, and the definition carries every text
    /// resource of [`OmniCreation::text_resources`] — persisted images are not
    /// text resources, they are written beside them by
    /// [`OmniCreation::save_with_images`].
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
        let name = skill_name(page);
        let description = skill_description(page);
        let text_resources = self.text_resources(page, resources);
        let instructions =
            instructions_markdown(page, &name, &description, &text_resources, &self.config);
        let mut definition = SkillDefinition::new(name, description, instructions);
        definition.resources = text_resources;
        definition.validate()?;
        Ok(definition)
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
    /// image URL with its alt text. Then each HTML [`DownloadedResource`] with a
    /// non-blank document becomes `references/page-NN.md` (two-digit ordinal
    /// among the written reference pages), rendering the page title, its
    /// headings as `##`/`###`, and its text. The list is truncated to
    /// [`OmniConfig::max_resources`], and a cap of zero returns nothing.
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
            sources_markdown(page, &self.config),
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
    /// to 8 000 characters on a character boundary), code blocks, and image
    /// URLs with alt text, and must answer with one fenced `json` object holding
    /// `name`, `description`, `instructions`, and optionally `metadata`. The
    /// resulting definition carries the provenance resource of
    /// [`OmniCreation::text_resources`].
    ///
    /// Any failure after a model is attached — a model error, an answer without
    /// a JSON fence, a malformed or incomplete payload, or a definition that
    /// does not validate — is logged with [`tracing::warn`] and replaced by the
    /// deterministic rendering, so the stage stays productive offline.
    ///
    /// # Errors
    ///
    /// The only error that escapes is
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when no
    /// model is attached (the message names `OmniCreation::with_model`).
    pub async fn generate_with_model(&self, page: &ScrapedPage) -> Result<SkillDefinition> {
        let Some(model) = self.model.as_ref() else {
            return Err(Error::validation(
                "model-assisted omni creation requires a pipeline built with \
                 OmniCreation::with_model",
            ));
        };
        match self.model_definition(model, page).await {
            Ok(definition) => Ok(definition),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "model-assisted skill generation failed; using the deterministic generator"
                );
                self.generate(page, &[])
            }
        }
    }

    /// Runs scrape → download → generate → save and reports the artifact.
    ///
    /// The drafting model is used when attached (with its deterministic
    /// fallback); otherwise the deterministic generator consumes the downloaded
    /// resources. The bundle keeps both the text resources and every downloaded
    /// image, and [`OmniOutcome::resources`] lists the text paths first, then the
    /// image paths.
    ///
    /// # Errors
    ///
    /// Returns every error of [`OmniCreation::scrape`],
    /// [`OmniCreation::download`], [`OmniCreation::generate`], and
    /// [`OmniCreation::save_with_images`]; a model failure is never one of them
    /// because the model-assisted generator falls back.
    pub async fn run(&self, url: &str, root: &Path) -> Result<OmniOutcome> {
        self.config.validate()?;
        let page = self.scrape(url).await?;
        let downloaded = self.download(&page).await?;
        let definition = if self.model.is_some() {
            self.generate_with_model(&page).await?
        } else {
            self.generate(&page, &downloaded)?
        };
        let bundle_path = self
            .save_with_images(&definition, &[], &downloaded, root)
            .await?;
        let mut resources: Vec<String> = definition
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
            definition,
            resources,
            bundle_path,
        })
    }

    /// One model round trip that must answer with a fenced skill definition.
    async fn model_definition(
        &self,
        model: &Arc<dyn Model>,
        page: &ScrapedPage,
    ) -> Result<SkillDefinition> {
        let completion = model
            .invoke(
                &[
                    BaseMessage::system(MODEL_SYSTEM_PROMPT),
                    BaseMessage::user(model_user_prompt(page, &self.config)),
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
        let mut definition = SkillDefinition::new(
            payload.required("name")?,
            payload.required("description")?,
            payload.required("instructions")?,
        );
        definition.metadata = payload.metadata;
        definition.resources = self.text_resources(page, &[]);
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

/// Builds a page from a `tar` archive's text entries.
fn archive_page(url: &str, entries: &[ArchiveEntry]) -> ScrapedPage {
    let mut texts = Vec::new();
    for entry in entries {
        if entry.is_dir {
            continue;
        }
        let text = match std::str::from_utf8(&entry.bytes) {
            Ok(text) => text.to_owned(),
            Err(error) => {
                tracing::debug!(
                    path = %entry.path,
                    %error,
                    "skipping a non-UTF-8 archive entry"
                );
                continue;
            }
        };
        if looks_binary(&entry.bytes) {
            tracing::debug!(
                path = %entry.path,
                "skipping a binary-looking archive entry"
            );
            continue;
        }
        texts.push(text);
    }
    let mut headings = Vec::new();
    let mut code_blocks = Vec::new();
    for text in &texts {
        collect_markdown_headings(text, &mut headings);
        collect_fenced_blocks(text, &mut code_blocks);
    }
    let title = headings
        .iter()
        .find(|heading| heading.level == 1)
        .map_or_else(String::new, |heading| heading.text.clone());
    ScrapedPage {
        url: url.to_owned(),
        title,
        text: texts.join("\n\n"),
        headings,
        code_blocks,
        ..ScrapedPage::default()
    }
}

/// Whether an archive entry looks binary: a NUL byte in its first 512 bytes.
fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(BINARY_SNIFF_BYTES).any(|byte| *byte == 0)
}

/// Collects markdown ATX headings (`#`…`######` plus a space) from `text`.
fn collect_markdown_headings(text: &str, headings: &mut Vec<Heading>) {
    for line in text.lines() {
        let trimmed = line.trim_start();
        let hashes = trimmed.bytes().take_while(|byte| *byte == b'#').count();
        if !(1..=6).contains(&hashes) {
            continue;
        }
        let Some(rest) = trimmed.get(hashes..) else {
            continue;
        };
        let Some(rest) = rest.strip_prefix(' ') else {
            continue;
        };
        let heading = rest.trim();
        if heading.is_empty() {
            continue;
        }
        headings.push(Heading {
            level: hashes as u8,
            text: heading.to_owned(),
        });
    }
}

/// Collects triple-backtick fenced code blocks from `text`, dropping the info string.
fn collect_fenced_blocks(text: &str, blocks: &mut Vec<String>) {
    let mut open: Option<Vec<&str>> = None;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if let Some(body) = open.take() {
                let content = body.join("\n");
                let content = content.trim();
                if !content.is_empty() {
                    blocks.push(content.to_owned());
                }
            } else {
                open = Some(Vec::new());
            }
            continue;
        }
        if let Some(body) = open.as_mut() {
            body.push(line);
        }
    }
}

/// Builds a page from an HTML resource, resolving links and images absolutely.
fn html_page(resource: &FetchedResource, url: &str) -> ScrapedPage {
    let document = HtmlDocument::parse(&resource.text(), url);
    let links = document
        .links
        .iter()
        .filter_map(|link| {
            resolve_url(url, &link.href).map(|href| Link {
                href,
                text: link.text.clone(),
            })
        })
        .collect();
    let images = document
        .images
        .iter()
        .filter_map(|image| {
            resolve_url(url, &image.src).map(|src| Image {
                src,
                alt: image.alt.clone(),
            })
        })
        .collect();
    ScrapedPage {
        url: url.to_owned(),
        title: document.title,
        description: document.description,
        text: document.text,
        headings: document.headings,
        links,
        images,
        code_blocks: document.code_blocks,
    }
}

pub(crate) mod assets;

#[cfg(test)]
mod tests;
