//! Resource filtering for the omni pipeline: which scraped assets survive.
//!
//! Ports the deterministic checks of the Python
//! `skill_omni_creation/scripts/common.py` (`SUPPORTED_EXTS`, `SUPPORTED_MIMES`,
//! `MIME_TO_EXT`, `MAX_IMAGE_BYTES`), `stage_02_download.py` (media-type gate,
//! SHA-256 payload dedupe) and `stage_04_save.py` (the saved file extension).
//!
//! Python decides relevance with an `OpenAI` vision judge and measures image
//! dimensions with Pillow; this port replaces both with pure functions of the
//! URL, the declared `content-type`, the payload bytes and the configured size
//! bounds ([`ResourceFilter`]). No model, no image decoder, no network.
//!
//! - [`sniff_image`] identifies `png`/`jpeg`/`gif`/`webp`/`svg` from magic
//!   bytes without decoding the image; `is_supported_image_format` is the keep
//!   policy layered on top of that classification.
//! - [`image_extension`] picks the saved file extension, exactly like Python's
//!   `stage_04_save.py`.
//! - [`image_dimensions`] reads a payload's pixel size out of its container
//!   header (PNG, JPEG, GIF, WebP, PNM), the dependency-free stand-in for
//!   Pillow's `Image.open(...).size` (see the [`dimensions`] submodule).
//! - [`is_probable_icon`] drops spacers, tracking pixels and 1x1/2x2 sprites,
//!   plus — mirroring Python's `MIN_DIMENSION` rule — any image whose known
//!   width *and* height are below 80 px.
//! - [`filter_images_with`] applies scheme, icon, format, size and dedupe rules
//!   in that order, keeping the first occurrence and the input order. The keep
//!   policy is Python's: only PNG, JPEG, GIF and WebP survive, decided by the
//!   declared media type when there is one and by magic bytes otherwise.
//!   Payload duplicates collapse on their SHA-256 hex digest, as in Python.
//!   A downloaded payload's own header supplies the dimension gate, so a real
//!   1x1 spacer or a 79x79 bullet icon is dropped without any caller input.
//! - [`dedupe_links`] collapses equivalent anchors while preserving the first
//!   original `href` text.
//! - [`slugify`] produces the skill-name slug used by later stages.
//! - [`content_hash`] is the `FNV-1a` *URL identity* helper used by the
//!   download stage; it is no longer the payload dedupe key.
//!
//! Every function is total: malformed URLs, truncated payloads and non-UTF-8
//! bytes are classified, never fatal.
use super::sha256;
use crate::skill_creator::omni::html::Link;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
pub mod dimensions;
pub use dimensions::image_dimensions;
/// Default lower bound for a downloaded image payload: 1 KiB.
const DEFAULT_MIN_IMAGE_BYTES: usize = 1024;
/// Default upper bound for a downloaded image payload: 8 MiB.
const DEFAULT_MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
/// Python's `common.MIN_DIMENSION`: an image narrower *and* shorter than this
/// many pixels is decorative.
const MIN_DIMENSION: u32 = 80;
/// How many leading bytes are inspected when sniffing for an `svg` document.
const SVG_SNIFF_BYTES: usize = 512;
/// Maximum slug length; the value doubles as a conservative skill-name budget.
const MAX_SLUG_LENGTH: usize = 60;
/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Lowercased URL fragments that mark a decorative or tracking image.
const ICON_HINTS: [&str; 6] = [
    "spacer",
    "pixel",
    "blank.gif",
    "1x1",
    "2x2",
    "transparent.gif",
];
/// The image extensions the omni save stage accepts, without the leading dot.
///
/// Python's `common.SUPPORTED_EXTS` is the dotted set
/// `{".png", ".jpg", ".jpeg", ".webp", ".gif"}`; this constant holds the
/// extension-less names so a lowercased URL suffix can be compared directly.
/// It is the whitelist behind [`image_extension`].
pub const SUPPORTED_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "gif"];
/// The image media types the omni download stage accepts.
///
/// Mirrors Python's `common.SUPPORTED_MIMES` exactly. `image/svg+xml` is absent
/// on purpose: [`sniff_image`] still classifies SVG documents, but the keep
/// policy rejects them, just like Python's media-type gate does.
pub const SUPPORTED_MIMES: [&str; 4] = ["image/jpeg", "image/png", "image/gif", "image/webp"];
/// Python's `common.MIME_TO_EXT`: declared media type to saved extension.
const MIME_TO_EXT: [(&str, &str); 4] = [
    ("image/jpeg", "jpg"),
    ("image/png", "png"),
    ("image/gif", "gif"),
    ("image/webp", "webp"),
];
/// The extension used when neither the URL suffix nor the media type is known;
/// Python's `MIME_TO_EXT.get(mime, ".png")` default.
const DEFAULT_IMAGE_EXTENSION: &str = "png";
/// Policy controlling which downloaded images are kept in a skill bundle.
///
/// The defaults mirror the Python constants: a 1 KiB floor (anything smaller is
/// a placeholder or a broken download) and an 8 MiB ceiling (the download
/// stage's cap per response). Note Python's `MAX_IMAGE_BYTES` is 5 MiB and its
/// `MIN_DIMENSION` is 80 px; this port keeps the byte bounds here and derives
/// the pixel size from the payload header instead of decoding it, so
/// [`is_probable_icon`] applies the same 80 px rule through
/// [`image_dimensions`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ResourceFilter {
    /// Payloads smaller than this many bytes are dropped.
    pub min_image_bytes: usize,
    /// Payloads larger than this many bytes are dropped.
    pub max_image_bytes: usize,
    /// Whether later stages may follow links to hosts other than the source
    /// page; owned here so one policy object describes a whole run.
    pub allow_cross_host: bool,
}
impl Default for ResourceFilter {
    fn default() -> Self {
        Self {
            min_image_bytes: DEFAULT_MIN_IMAGE_BYTES,
            max_image_bytes: DEFAULT_MAX_IMAGE_BYTES,
            allow_cross_host: false,
        }
    }
}
impl ResourceFilter {
    /// Returns the filter with a different lower size bound.
    #[must_use]
    pub fn with_min_image_bytes(mut self, min_image_bytes: usize) -> Self {
        self.min_image_bytes = min_image_bytes;
        self
    }
    /// Returns the filter with a different upper size bound.
    #[must_use]
    pub fn with_max_image_bytes(mut self, max_image_bytes: usize) -> Self {
        self.max_image_bytes = max_image_bytes;
        self
    }
    /// Returns the filter with a different cross-host policy.
    #[must_use]
    pub fn with_allow_cross_host(mut self, allow_cross_host: bool) -> Self {
        self.allow_cross_host = allow_cross_host;
        self
    }
    /// Rejects a filter whose size bounds cannot describe a usable range.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`]
    /// naming the offending bound (`min_image_bytes` or `max_image_bytes`) when
    /// either is zero, or when `min_image_bytes` exceeds `max_image_bytes`.
    pub fn validate(&self) -> Result<()> {
        if self.min_image_bytes == 0 {
            return Err(Error::validation(format!(
                "min_image_bytes must be greater than zero, got {}",
                self.min_image_bytes
            )));
        }
        if self.max_image_bytes == 0 {
            return Err(Error::validation(format!(
                "max_image_bytes must be greater than zero, got {}",
                self.max_image_bytes
            )));
        }
        if self.min_image_bytes > self.max_image_bytes {
            return Err(Error::validation(format!(
                "min_image_bytes ({}) must not exceed max_image_bytes ({})",
                self.min_image_bytes, self.max_image_bytes
            )));
        }
        Ok(())
    }
}
/// One scraped image, with its payload when it has already been downloaded.
///
/// A candidate without `bytes` is a reference only: it survives filtering when
/// its `content_type` is absent or names an `image/*` type, so `svg` and
/// deferred downloads can still be recorded as links.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ImageCandidate {
    /// The raw or resolved image URL.
    pub url: String,
    /// The `alt` text, possibly empty.
    pub alt: String,
    /// The downloaded payload, absent when the image was never fetched.
    pub bytes: Option<Vec<u8>>,
    /// The `content-type` reported by the server, verbatim.
    pub content_type: Option<String>,
}
impl ImageCandidate {
    /// Builds a reference candidate without a payload.
    #[must_use]
    pub fn new(url: impl Into<String>, alt: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            alt: alt.into(),
            ..Self::default()
        }
    }
    /// Returns the candidate with a downloaded payload.
    #[must_use]
    pub fn with_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.bytes = Some(bytes);
        self
    }
    /// Returns the candidate with a declared `content-type`.
    #[must_use]
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }
}
/// Identifies an image format from its magic bytes.
///
/// Recognized values, in match order: `"png"` (`\x89PNG`), `"jpeg"`
/// (`FF D8 FF`), `"gif"` (`GIF8`), `"webp"` (`RIFF` with `WEBP` at offset 8),
/// and `"svg"` (an optional UTF-8 BOM, then whitespace, an XML declaration, a
/// comment and a doctype may precede `<svg`, all case-insensitively, within the
/// first 512 bytes).
///
/// Returns `None` for anything else, including truncated payloads, so the
/// function never panics whatever the byte slice looks like.
#[must_use]
pub fn sniff_image(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG") {
        return Some("png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("jpeg");
    }
    if bytes.starts_with(b"GIF8") {
        return Some("gif");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12).is_some_and(|tag| tag == b"WEBP") {
        return Some("webp");
    }
    if looks_like_svg(bytes) {
        return Some("svg");
    }
    None
}
/// Chooses the saved file extension for one downloaded image.
///
/// Ports `stage_04_save.py`: the lowercased suffix of the URL path wins when it
/// is one of [`SUPPORTED_EXTS`] (so `…/a.PNG` yields `"png"` and `…/a.jpeg`
/// stays `"jpeg"`); otherwise the declared `content-type` is mapped through
/// Python's `MIME_TO_EXT` (`image/jpeg` → `"jpg"`, `image/png` → `"png"`,
/// `image/gif` → `"gif"`, `image/webp` → `"webp"`); and anything else — no
/// suffix or an unsupported one, no media type or an unmapped one — falls back
/// to `"png"`, exactly like `MIME_TO_EXT.get(mime, ".png")`.
///
/// The query string and fragment are ignored, as `urlparse(url).path` ignores
/// them, and the returned name never carries the leading dot.
#[must_use]
pub fn image_extension(url: &str, content_type: Option<&str>) -> &'static str {
    if let Some(extension) = url_path_extension(url) {
        if let Some(supported) = SUPPORTED_EXTS.iter().find(|known| **known == extension) {
            return supported;
        }
    }
    content_type
        .map(mime_of)
        .as_deref()
        .and_then(extension_for_mime)
        .unwrap_or(DEFAULT_IMAGE_EXTENSION)
}
/// Reports whether a URL alone or a known dimension marks a decorative image.
///
/// Two rules apply, either of which makes the image decorative:
///
/// 1. **Spacer rule** — any known axis at or below 2 pixels (so `1x1`, `2x1` and
///    `2x2` spacers fail, and so does a 1-pixel-high layout graphic).
/// 2. **Python's `MIN_DIMENSION` rule** — a *complete* known dimension pair with
///    `width < 80 && height < 80`, mirroring
///    `if width < MIN_DIMENSION and height < MIN_DIMENSION` in Python's
///    `stage_02_download.py`. It needs both axes, so a lone known width cannot
///    trigger it, and it excludes the boundary: a known 79x79 image is
///    decorative, a known 80x80 image is not, and a known 1200x60 banner is not
///    (only one axis is short, and neither is at or below 2).
///
/// Independently of both, a lowercased URL containing one of `spacer`, `pixel`,
/// `blank.gif`, `1x1`, `2x2`, `transparent.gif` is decorative.
///
/// The dimensions are the caller's — [`image_dimensions`] derives them from a
/// downloaded payload's header, and [`filter_images_with`] passes them here — so
/// a payload whose header cannot be parsed simply has no dimension evidence and
/// falls back to the URL hints.
#[must_use]
pub fn is_probable_icon(url: &str, width: Option<u32>, height: Option<u32>) -> bool {
    if width.is_some_and(|value| value <= 2) || height.is_some_and(|value| value <= 2) {
        return true;
    }
    if let (Some(width), Some(height)) = (width, height) {
        if width < MIN_DIMENSION && height < MIN_DIMENSION {
            return true;
        }
    }
    let lowered = url.to_ascii_lowercase();
    ICON_HINTS.iter().any(|hint| lowered.contains(hint))
}
/// Filters images with the default [`ResourceFilter`].
///
/// Equivalent to [`filter_images_with`] with [`ResourceFilter::default`]; see
/// that function for the rules.
#[must_use]
pub fn filter_images(images: &[ImageCandidate]) -> Vec<ImageCandidate> {
    filter_images_with(images, &ResourceFilter::default())
}
/// Filters images against an explicit policy.
///
/// Rules, applied in order, with the first matching rule deciding:
///
/// 1. the URL must be absolute `http`/`https` (relative, `data:` and `ftp:`
///    references are dropped);
/// 2. [`is_probable_icon`] must not fire, checked against the URL *and* the
///    payload's own header dimensions: [`image_dimensions`] reads the pixel size
///    out of a downloaded PNG/JPEG/GIF/WebP/PNM header, so Python's decorative
///    rule (`width < 80 && height < 80`) applies to real bytes and a genuine 1x1
///    spacer is dropped even when its URL looks innocuous. A payload whose
///    header cannot be parsed contributes no dimensions, leaving the URL hints
///    as the only evidence;
/// 3. the format must be in Python's keep policy, i.e. PNG, JPEG, GIF or WebP:
///    a declared `content-type` must be empty or one of [`SUPPORTED_MIMES`]
///    (`image/svg+xml`, `image/bmp` and `text/html` are rejected even when the
///    payload is a raster), otherwise magic bytes must classify a supported
///    raster. [`sniff_image`] still *classifies* `svg`, but `svg` is never kept;
/// 4. a downloaded payload must lie within `filter.min_image_bytes` and
///    `filter.max_image_bytes`, inclusive;
/// 5. a candidate is a duplicate when the SHA-256 hex digest of its payload has
///    been seen (bytes present) or its normalized URL has been seen (bytes
///    absent), matching Python's `hashlib.sha256(data).hexdigest()` key.
///
/// Output order is the input order with the first occurrence of each duplicate
/// kept, so the function is deterministic and idempotent.
#[must_use]
pub fn filter_images_with(
    images: &[ImageCandidate],
    filter: &ResourceFilter,
) -> Vec<ImageCandidate> {
    let mut seen: HashSet<DedupeKey> = HashSet::new();
    let mut kept = Vec::new();
    for candidate in images {
        if !is_http_url(&candidate.url) {
            continue;
        }
        let (width, height) = candidate
            .bytes
            .as_deref()
            .and_then(image_dimensions)
            .map_or((None, None), |(width, height)| (Some(width), Some(height)));
        if is_probable_icon(&candidate.url, width, height) {
            continue;
        }
        if !has_supported_format(candidate) {
            continue;
        }
        if let Some(bytes) = candidate.bytes.as_ref() {
            if bytes.len() < filter.min_image_bytes || bytes.len() > filter.max_image_bytes {
                continue;
            }
        }
        let key = match candidate.bytes.as_ref() {
            Some(bytes) => DedupeKey::Content(sha256::hex_digest(bytes)),
            None => match normalize_url(&candidate.url) {
                Some(normalized) => DedupeKey::Url(normalized),
                None => continue,
            },
        };
        if seen.insert(key) {
            kept.push(candidate.clone());
        }
    }
    kept
}
/// Deduplicates anchors for the reference-links section of a skill.
///
/// The key is the `href` with the fragment removed, a single trailing `/`
/// dropped and, for absolute `http(s)` URLs, the scheme and host lowercased;
/// the query string is kept. Relative links are compared verbatim after that
/// same trimming. The first occurrence wins and keeps its original `href` and
/// text; links whose `href` is empty after normalization are skipped.
#[must_use]
pub fn dedupe_links(links: &[Link]) -> Vec<Link> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept = Vec::new();
    for link in links {
        let Some(key) = link_key(&link.href) else {
            continue;
        };
        if seen.insert(key) {
            kept.push(link.clone());
        }
    }
    kept
}
/// Turns free text into a skill-name slug.
///
/// Lowercases ASCII runs of letters and digits, joins them with a single `-`,
/// drops every other character (punctuation, whitespace, `_` and non-ASCII
/// characters act as separators), and caps the result at 60 characters, cutting
/// on a `-` boundary when one exists so the slug never ends with `-`.
///
/// The result is either empty or matches `^[A-Za-z0-9][A-Za-z0-9._-]*$`; the
/// caller supplies the fallback name when it is empty.
#[must_use]
pub fn slugify(text: &str) -> String {
    let mut slug = String::new();
    let mut separator_pending = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            if separator_pending && !slug.is_empty() {
                slug.push('-');
            }
            separator_pending = false;
            slug.push(character.to_ascii_lowercase());
        } else {
            separator_pending = true;
        }
    }
    if slug.len() > MAX_SLUG_LENGTH {
        slug.truncate(MAX_SLUG_LENGTH);
        if let Some(index) = slug.rfind('-') {
            slug.truncate(index);
        }
    }
    slug
}
/// The `FNV-1a` 64-bit hash of a payload.
///
/// This is the *URL/identity* dedupe helper: the download stage keys its
/// candidate list on the hash of the target URL text, so a page that repeats the
/// same link fetches it once. It is deliberately not the payload key — content
/// dedupe uses the SHA-256 hex digest ([`sha256::hex_digest`]), matching Python's
/// `hashlib.sha256(data).hexdigest()` in `stage_02_download.py`.
/// `content_hash(&[])` is the offset basis `0xcbf29ce484222325`.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
/// The lowercase host of an absolute `http(s)` URL, including the port.
///
/// User information is stripped, the host is ASCII-lowercased, and a `www.`
/// prefix is kept literally. Relative, malformed and non-`http(s)` URLs yield
/// `None`.
#[must_use]
pub fn host(url: &str) -> Option<String> {
    let (scheme, authority, _) = split_url(url)?;
    if !is_http_scheme(scheme) {
        return None;
    }
    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    if authority.is_empty() {
        return None;
    }
    Some(authority.to_ascii_lowercase())
}
/// Reports whether two URLs name the same host.
///
/// The comparison is case-insensitive because [`host`] lowercases, and `www.`
/// is not special: `www.example.com` and `example.com` differ. Either side
/// failing to yield a host makes the answer `false`.
#[must_use]
pub fn same_host(left: &str, right: &str) -> bool {
    match (host(left), host(right)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}
/// The dedupe key of one filtered candidate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DedupeKey {
    /// The SHA-256 hex digest of a downloaded payload.
    Content(String),
    /// The normalized URL of a reference-only candidate.
    Url(String),
}
/// Detects an `svg` document behind a BOM, whitespace, a declaration or a doctype.
fn looks_like_svg(bytes: &[u8]) -> bool {
    let window = bytes
        .get(..bytes.len().min(SVG_SNIFF_BYTES))
        .unwrap_or(bytes);
    let text = String::from_utf8_lossy(window);
    let mut rest = text.trim_start_matches('\u{feff}').trim_start();
    loop {
        let lowered = rest.to_ascii_lowercase();
        if lowered.starts_with("<?xml") {
            let Some(end) = rest.find("?>") else {
                return false;
            };
            rest = rest.get(end + 2..).unwrap_or_default().trim_start();
            continue;
        }
        if lowered.starts_with("<!--") {
            let Some(end) = rest.find("-->") else {
                return false;
            };
            rest = rest.get(end + 3..).unwrap_or_default().trim_start();
            continue;
        }
        if lowered.starts_with("<!doctype") {
            let Some(end) = rest.find('>') else {
                return false;
            };
            rest = rest.get(end + 1..).unwrap_or_default().trim_start();
            continue;
        }
        return lowered.starts_with("<svg");
    }
}
/// Reports whether a [`sniff_image`] format name survives the keep policy.
///
/// The policy is Python's `SUPPORTED_EXTS`/`SUPPORTED_MIMES` pair: `png`,
/// `jpeg`, `gif` and `webp` are kept, and everything else — notably `svg`,
/// which [`sniff_image`] still classifies — is rejected. Kept crate-internal
/// because it is a policy predicate, not a classifier: callers use
/// [`sniff_image`] to learn what a payload is and this to learn whether it may
/// be kept.
pub(crate) fn is_supported_image_format(sniffed: &str) -> bool {
    matches!(sniffed, "png" | "jpeg" | "gif" | "webp")
}
/// Whether the candidate's format survives the keep policy.
///
/// A declared `content-type` is the gate, mirroring Python's `stage_02_download`
/// media-type check against `SUPPORTED_MIMES`: a candidate declaring
/// `image/svg+xml`, `image/bmp` or `text/html` is rejected even when its payload
/// would sniff as a supported format, and an empty declaration counts as no
/// declaration. Without a declaration the payload magic decides through
/// [`sniff_image`] and `is_supported_image_format`; a reference that has neither
/// bytes nor a declaration is kept, because nothing contradicts it (the
/// download stage records such images as URLs).
fn has_supported_format(candidate: &ImageCandidate) -> bool {
    match candidate.content_type.as_deref() {
        Some(header) => {
            let mime = mime_of(header);
            mime.is_empty() || SUPPORTED_MIMES.contains(&mime.as_str())
        }
        None => match candidate.bytes.as_ref() {
            Some(bytes) => sniff_image(bytes).is_some_and(is_supported_image_format),
            None => true,
        },
    }
}
/// Python's `MIME_TO_EXT` lookup: declared media type to saved extension.
fn extension_for_mime(mime: &str) -> Option<&'static str> {
    MIME_TO_EXT
        .iter()
        .find_map(|(name, extension)| (*name == mime).then_some(*extension))
}
/// The lowercased extension of the last URL path segment, without the dot.
///
/// The query string and fragment are dropped first, as `urlparse(url).path`
/// does; a hidden file such as `.png`, a trailing dot and a path without any dot
/// all yield `None`.
fn url_path_extension(url: &str) -> Option<String> {
    let without_fragment = strip_fragment(url.trim());
    let path = match without_fragment.find('?') {
        Some(index) => without_fragment.get(..index)?,
        None => without_fragment,
    };
    let file = path.rsplit('/').next()?;
    let (stem, extension) = file.rsplit_once('.')?;
    if stem.is_empty() || extension.is_empty() {
        return None;
    }
    Some(extension.to_ascii_lowercase())
}
/// The lowercased media type of a `content-type` header, without parameters.
fn mime_of(header: &str) -> String {
    header
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}
/// Whether `url` is an absolute `http`/`https` URL with a non-empty authority.
fn is_http_url(url: &str) -> bool {
    split_url(url).is_some_and(|(scheme, _, _)| is_http_scheme(scheme))
}
/// Whether a URL scheme names HTTP or HTTPS.
fn is_http_scheme(scheme: &str) -> bool {
    scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
}
/// Splits an absolute URL into scheme, authority and tail (path, query, fragment).
///
/// Returns `None` for relative URLs, for URLs without `//`, for invalid schemes
/// and for an empty authority.
fn split_url(url: &str) -> Option<(&str, &str, &str)> {
    let url = url.trim();
    let colon = url.find(':')?;
    let scheme = url.get(..colon)?;
    let mut characters = scheme.chars();
    let first = characters.next()?;
    if !first.is_ascii_alphabetic()
        || !characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
    {
        return None;
    }
    let rest = url.get(colon + 1..)?.strip_prefix("//")?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = rest.get(..end)?;
    if authority.is_empty() {
        return None;
    }
    Some((scheme, authority, rest.get(end..)?))
}
/// Removes the fragment from a URL or reference.
fn strip_fragment(url: &str) -> &str {
    match url.find('#') {
        Some(index) => url.get(..index).unwrap_or_default(),
        None => url,
    }
}
/// Normalizes an absolute `http(s)` URL for dedupe: lowercase scheme and host,
/// no fragment, no trailing `/` on the path, query kept.
fn normalize_url(url: &str) -> Option<String> {
    let (scheme, authority, tail) = split_url(url)?;
    if !is_http_scheme(scheme) {
        return None;
    }
    let tail = strip_fragment(tail);
    let (path, query) = match tail.find('?') {
        Some(index) => (tail.get(..index).unwrap_or_default(), tail.get(index..)?),
        None => (tail, ""),
    };
    let path = path.strip_suffix('/').unwrap_or(path);
    Some(format!(
        "{}://{}{}{}",
        scheme.to_ascii_lowercase(),
        authority.to_ascii_lowercase(),
        path,
        query
    ))
}
/// The dedupe key of one link, or `None` when it normalizes to nothing.
fn link_key(href: &str) -> Option<String> {
    let trimmed = strip_fragment(href.trim());
    let trimmed = trimmed.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(normalized) = normalize_url(trimmed) {
        return Some(normalized);
    }
    let without_slash = trimmed.strip_suffix('/').unwrap_or(trimmed);
    if without_slash.is_empty() {
        return None;
    }
    Some(without_slash.to_string())
}
#[cfg(test)]
mod tests;
