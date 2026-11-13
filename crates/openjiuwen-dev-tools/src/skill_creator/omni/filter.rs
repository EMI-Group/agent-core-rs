//! Resource filtering for the omni pipeline: which scraped assets survive.
//!
//! Ports the deterministic checks of the Python
//! `skill_omni_creation/scripts/stage_02_download.py` (`SUPPORTED_MIMES`,
//! `MAX_IMAGE_BYTES`, content-hash dedupe) and the deterministic helpers of
//! `stage_03_filter.py`/`common.py` (`MIN_DIMENSION`, `slugify`).
//!
//! Python decides relevance with an `OpenAI` vision judge and measures image
//! dimensions with Pillow; this port replaces both with pure functions of the
//! URL, the declared `content-type`, the payload bytes and the configured size
//! bounds ([`ResourceFilter`]). No model, no image decoder, no network.
//!
//! - [`sniff_image`] identifies `png`/`jpeg`/`gif`/`webp`/`svg` from magic
//!   bytes without decoding the image.
//! - [`is_probable_icon`] drops spacers, tracking pixels and 1x1/2x2 sprites.
//! - [`filter_images_with`] applies scheme, icon, format, size and dedupe rules
//!   in that order, keeping the first occurrence and the input order.
//! - [`dedupe_links`] collapses equivalent anchors while preserving the first
//!   original `href` text.
//! - [`slugify`] produces the skill-name slug used by later stages.
//!
//! Every function is total: malformed URLs, truncated payloads and non-UTF-8
//! bytes are classified, never fatal.
use crate::skill_creator::omni::html::Link;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
/// Default lower bound for a downloaded image payload: 1 KiB.
const DEFAULT_MIN_IMAGE_BYTES: usize = 1024;
/// Default upper bound for a downloaded image payload: 8 MiB.
const DEFAULT_MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
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
/// Policy controlling which downloaded images are kept in a skill bundle.
///
/// The defaults mirror the Python constants: a 1 KiB floor (anything smaller is
/// a placeholder or a broken download) and an 8 MiB ceiling (the download
/// stage's cap per response). Note Python's `MAX_IMAGE_BYTES` is 5 MiB and its
/// `MIN_DIMENSION` is 80 px; this port uses byte bounds only, because it never
/// decodes the image, and drops dimension-less icons through
/// [`is_probable_icon`] instead.
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
/// Reports whether a URL alone or a known dimension marks a decorative image.
///
/// The rule is: any known axis at or below 2 pixels (so `1x1`, `2x1` and `2x2`
/// spacers fail, and so does a 1-pixel-high layout graphic), or a lowercased URL
/// containing one of `spacer`, `pixel`, `blank.gif`, `1x1`, `2x2`,
/// `transparent.gif`.
///
/// Python compares `width < 80 and height < 80` on the decoded image; because
/// this port never decodes pixels, the only dimensions available are the ones a
/// caller supplies, and the URL hints carry the rest of the decision.
#[must_use]
pub fn is_probable_icon(url: &str, width: Option<u32>, height: Option<u32>) -> bool {
    if width.is_some_and(|value| value <= 2) || height.is_some_and(|value| value <= 2) {
        return true;
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
/// 2. [`is_probable_icon`] must not fire on the URL;
/// 3. a downloaded payload must be a recognized format ([`sniff_image`],
///    falling back to the declared `content-type`); a reference without bytes
///    must have no `content-type` or an `image/*` one;
/// 4. a downloaded payload must lie within `filter.min_image_bytes` and
///    `filter.max_image_bytes`, inclusive;
/// 5. a candidate is a duplicate when its [`content_hash`] has been seen (bytes
///    present) or its normalized URL has been seen (bytes absent).
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
        if is_probable_icon(&candidate.url, None, None) {
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
            Some(bytes) => DedupeKey::Content(content_hash(bytes)),
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
/// Used as the content-dedupe key, so two downloads of the same image collapse
/// regardless of their URLs. `content_hash(&[])` is the offset basis
/// `0xcbf29ce484222325`.
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
    /// The `FNV-1a` hash of a downloaded payload.
    Content(u64),
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
/// Whether the candidate names a supported format, by bytes or by `content-type`.
fn has_supported_format(candidate: &ImageCandidate) -> bool {
    match candidate.bytes.as_ref() {
        Some(bytes) => {
            sniff_image(bytes).is_some()
                || format_from_content_type(candidate.content_type.as_deref()).is_some()
        }
        None => match candidate.content_type.as_deref() {
            None => true,
            Some(header) => {
                let mime = mime_of(header);
                mime.is_empty() || mime.starts_with("image/")
            }
        },
    }
}
/// Maps an `image/*` `content-type` onto a [`sniff_image`] format name.
fn format_from_content_type(header: Option<&str>) -> Option<&'static str> {
    let mime = mime_of(header?);
    match mime.as_str() {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "image/svg+xml" => Some("svg"),
        _ => None,
    }
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
mod tests {
    use super::*;

    /// A minimal `png` payload of exactly `size` bytes (at least 8).
    fn png(size: usize) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.resize(size.max(bytes.len()), 0);
        bytes
    }

    /// A minimal `jpeg` payload of exactly `size` bytes (at least 3).
    fn jpeg(size: usize) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0xD8, 0xFF];
        bytes.resize(size.max(bytes.len()), 0);
        bytes
    }

    /// A candidate carrying a downloaded payload.
    fn downloaded(url: &str, bytes: Vec<u8>) -> ImageCandidate {
        ImageCandidate::new(url, "").with_bytes(bytes)
    }

    /// A reference-only candidate without a payload.
    fn reference(url: &str) -> ImageCandidate {
        ImageCandidate::new(url, "")
    }

    /// An anchor with `href` and `text`.
    fn link(href: &str, text: &str) -> Link {
        Link {
            href: href.to_string(),
            text: text.to_string(),
        }
    }

    /// The URLs of `images`, in order.
    fn urls(images: &[ImageCandidate]) -> Vec<&str> {
        images.iter().map(|image| image.url.as_str()).collect()
    }

    #[test]
    fn sniff_image_detects_binary_formats() {
        assert_eq!(sniff_image(b"\x89PNG\r\n\x1a\npayload"), Some("png"));
        assert_eq!(sniff_image(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpeg"));
        assert_eq!(sniff_image(b"GIF89a...."), Some("gif"));
        assert_eq!(sniff_image(b"GIF87a...."), Some("gif"));
        assert_eq!(sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some("webp"));
    }

    #[test]
    fn sniff_image_detects_svg_documents() {
        let declared =
            br#"<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://example.com"/>"#;
        let with_bom = b"\xEF\xBB\xBF<svg xmlns=\"http://example.com\"/>";
        assert_eq!(sniff_image(declared), Some("svg"));
        assert_eq!(sniff_image(with_bom), Some("svg"));
        assert_eq!(sniff_image(b"\n\n  <SVG width=\"10\"></SVG>"), Some("svg"));
        assert_eq!(sniff_image(b"<!-- note --><svg></svg>"), Some("svg"));
        assert_eq!(
            sniff_image(b"<?xml version=\"1.0\"?><!DOCTYPE svg><svg></svg>"),
            Some("svg")
        );
    }

    #[test]
    fn sniff_image_rejects_other_payloads() {
        for payload in [
            b"".as_slice(),
            &[0xFF, 0xD8],
            b"\x89P",
            b"RIFF\x00\x00\x00\x00WAVEfmt ",
            b"RIFF\x00\x00\x00\x00WEB",
            b"hello world",
            b"<?xml version=\"1.0\"?><html></html>",
            b"<!-- unterminated <svg",
        ] {
            assert_eq!(sniff_image(payload), None, "{payload:?}");
        }
    }

    #[test]
    fn icon_heuristics_by_dimension() {
        let url = "https://example.com/assets/chart.png";
        assert!(is_probable_icon(url, Some(2), Some(2)));
        assert!(is_probable_icon(url, Some(1), Some(1)));
        assert!(is_probable_icon(url, Some(2), Some(8)));
        assert!(is_probable_icon(url, Some(100), Some(1)));
        assert!(!is_probable_icon(url, Some(80), Some(80)));
        assert!(!is_probable_icon(url, None, None));
    }

    #[test]
    fn icon_heuristics_by_url() {
        for url in [
            "https://example.com/spacer.gif",
            "https://example.com/1x1.png",
            "https://example.com/img/pixel.gif",
            "https://example.com/blank.gif",
            "https://example.com/transparent.gif",
            "https://example.com/SPACER.GIF",
        ] {
            assert!(is_probable_icon(url, None, None), "{url}");
        }
        assert!(!is_probable_icon(
            "https://example.com/images/pivot-chart.png",
            None,
            None
        ));
    }

    #[test]
    fn filter_drops_non_http_urls() {
        let images = [
            downloaded("/relative.png", png(2048)),
            downloaded("ftp://example.com/image.png", png(2048)),
            downloaded("data:image/png;base64,AAAA", png(2048)),
            downloaded("https://example.com/ok.png", png(2048)),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            ["https://example.com/ok.png"]
        );
    }

    #[test]
    fn filter_drops_icons() {
        let images = [
            downloaded("https://example.com/spacer.gif", png(2048)),
            downloaded("https://example.com/1x1.png", png(2048)),
            downloaded("https://example.com/screen.png", png(2048)),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            ["https://example.com/screen.png"]
        );
    }

    #[test]
    fn filter_drops_unknown_binary_formats() {
        let images = [
            downloaded(
                "https://example.com/bitmap.bmp",
                b"BM\x00\x00\x00\x00".repeat(256),
            ),
            downloaded("https://example.com/text.bin", b"not an image".repeat(128)),
        ];
        assert_eq!(filter_images(&images), Vec::<ImageCandidate>::new());
    }

    #[test]
    fn filter_respects_size_bounds() {
        let filter = ResourceFilter::default()
            .with_min_image_bytes(8)
            .with_max_image_bytes(16);
        let images = [
            downloaded("https://example.com/small.png", jpeg(3)),
            downloaded("https://example.com/min.png", png(8)),
            downloaded("https://example.com/max.png", png(16)),
            downloaded("https://example.com/big.png", png(17)),
        ];
        let kept = filter_images_with(&images, &filter);
        assert_eq!(
            urls(&kept),
            ["https://example.com/min.png", "https://example.com/max.png"]
        );
    }

    #[test]
    fn filter_uses_default_size_floor() {
        let images = [
            downloaded("https://example.com/tiny.png", png(512)),
            downloaded("https://example.com/ok.png", png(2048)),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            ["https://example.com/ok.png"]
        );
    }

    #[test]
    fn filter_keeps_valid_png_and_jpeg() {
        let images = [
            downloaded("https://example.com/a.png", png(2048)),
            downloaded("https://example.com/b.jpg", jpeg(4096)),
            reference("https://example.com/c.svg")
                .with_content_type("image/svg+xml; charset=utf-8"),
            reference("https://example.com/d.png").with_content_type("image/png"),
            reference("https://example.com/e.png"),
        ];
        assert_eq!(filter_images(&images).len(), 5);
    }

    #[test]
    fn filter_drops_non_image_content_types_without_bytes() {
        let images = [
            reference("https://example.com/page.html").with_content_type("text/html"),
            reference("https://example.com/other.png"),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            ["https://example.com/other.png"]
        );
    }

    #[test]
    fn filter_falls_back_to_content_type_for_bytes() {
        let payload = b"definitely not sniffed".repeat(128);
        let images = [
            downloaded("https://example.com/a.png", payload.clone()).with_content_type("image/png"),
            downloaded("https://example.com/b.png", payload).with_content_type("image/bmp"),
        ];
        assert_eq!(urls(&filter_images(&images)), ["https://example.com/a.png"]);
    }

    #[test]
    fn filter_dedupes_identical_bytes() {
        let payload = png(2048);
        let images = [
            downloaded("https://example.com/first.png", payload.clone()),
            downloaded("https://example.com/second.png", payload),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            ["https://example.com/first.png"]
        );
    }

    #[test]
    fn filter_dedupes_normalized_urls() {
        let images = [
            reference("https://Example.com/chart/#top"),
            reference("https://example.com/chart/"),
            reference("https://example.com/chart"),
            reference("https://example.com/chart?page=2"),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            [
                "https://Example.com/chart/#top",
                "https://example.com/chart?page=2"
            ]
        );
    }

    #[test]
    fn filter_preserves_first_seen_order() {
        let images = [
            downloaded("https://example.com/a.png", png(2048)),
            downloaded("https://example.com/icon/2x2.png", png(2048)),
            downloaded("https://example.com/b.jpg", jpeg(2048)),
            downloaded("https://example.com/a-copy.png", png(2048)),
            downloaded("https://example.com/c.gif", b"GIF89a".repeat(512)),
        ];
        assert_eq!(
            urls(&filter_images(&images)),
            [
                "https://example.com/a.png",
                "https://example.com/b.jpg",
                "https://example.com/c.gif"
            ]
        );
    }

    #[test]
    fn filter_is_idempotent() {
        let images = [
            downloaded("https://example.com/a.png", png(2048)),
            downloaded("https://example.com/b.jpg", jpeg(2048)),
            reference("https://example.com/c.png"),
        ];
        let once = filter_images(&images);
        assert_eq!(filter_images(&once), once);
    }

    #[test]
    fn dedupe_links_collapses_equivalent_hrefs() {
        let links = [
            link("https://Example.COM/a/#section", "one"),
            link("https://example.com/a/", "two"),
            link("https://example.com/a", "three"),
            link("https://example.com/a?page=2", "four"),
            link("https://example.com/a/?page=2", "five"),
        ];
        let kept = dedupe_links(&links);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], link("https://Example.COM/a/#section", "one"));
        assert_eq!(kept[1], link("https://example.com/a?page=2", "four"));
    }

    #[test]
    fn dedupe_links_drops_empty_and_preserves_order() {
        let links = [
            link("", "empty"),
            link("   ", "blank"),
            link("#anchor", "fragment"),
            link("/guide", "relative"),
            link("/guide/", "relative slash"),
            link("https://example.com/later", "later"),
        ];
        let kept = dedupe_links(&links);
        assert_eq!(
            kept,
            [
                link("/guide", "relative"),
                link("https://example.com/later", "later")
            ]
        );
    }

    #[test]
    fn slugify_matrix() {
        assert_eq!(slugify("Create a PivotChart!"), "create-a-pivotchart");
        assert_eq!(slugify("创建图表"), "");
        assert_eq!(slugify("!!!"), "");
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("  --Hello   World--  "), "hello-world");
        assert_eq!(slugify("_foo_bar_"), "foo-bar");
        assert_eq!(slugify("123 guide"), "123-guide");
        assert_eq!(slugify("A/B.C"), "a-b-c");
    }

    #[test]
    fn slugify_caps_length_without_trailing_separator() {
        let slug = slugify(&"word ".repeat(40));
        assert!(slug.len() <= 60, "{slug}");
        assert!(!slug.ends_with('-'), "{slug}");
        assert_eq!(slug.matches('-').count(), 11);
        let dense = slugify(&"a".repeat(200));
        assert_eq!(dense.len(), 60);
        assert!(!dense.ends_with('-'));
    }

    #[test]
    fn slugify_output_is_a_valid_skill_name() {
        for input in ["Create a PivotChart!", "A/B.C", "123 guide", "foo__bar"] {
            let slug = slugify(input);
            assert_ne!(slug, "");
            assert!(slug
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
            assert!(slug
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric()));
        }
    }

    #[test]
    fn content_hash_matches_fnv1a_vectors() {
        assert_eq!(content_hash(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(content_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(content_hash(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(content_hash(b"payload"), content_hash(b"payload"));
        assert_ne!(content_hash(b"one"), content_hash(b"two"));
    }

    #[test]
    fn host_extraction() {
        assert_eq!(
            host("https://Example.COM:8443/a/b?c=1"),
            Some("example.com:8443".to_string())
        );
        assert_eq!(host("http://example.com"), Some("example.com".to_string()));
        assert_eq!(
            host("http://user:secret@Example.com/a"),
            Some("example.com".to_string())
        );
        assert_eq!(host("/relative/path"), None);
        assert_eq!(host("mailto:someone@example.com"), None);
        assert_eq!(host("ftp://example.com/file"), None);
        assert_eq!(host("http:///no-authority"), None);
        assert_eq!(host("not a url"), None);
        assert_eq!(host(""), None);
    }

    #[test]
    fn same_host_comparison() {
        assert!(same_host("https://Example.com/a", "http://example.COM/b"));
        assert!(same_host(
            "http://example.com:8080/a",
            "http://example.com:8080/b"
        ));
        assert!(!same_host(
            "http://example.com:8080",
            "http://example.com:9090"
        ));
        assert!(!same_host("http://www.example.com", "http://example.com"));
        assert!(!same_host("/relative", "/relative"));
        assert!(!same_host("mailto:a@example.com", "http://example.com"));
    }

    #[test]
    fn resource_filter_defaults_and_validate() {
        let filter = ResourceFilter::default();
        assert_eq!(filter.min_image_bytes, 1024);
        assert_eq!(filter.max_image_bytes, 8 * 1024 * 1024);
        assert!(!filter.allow_cross_host);
        assert!(filter.validate().is_ok());
        let error = ResourceFilter::default()
            .with_min_image_bytes(0)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("min_image_bytes"), "{error}");
        let error = ResourceFilter::default()
            .with_max_image_bytes(0)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("max_image_bytes"), "{error}");
        let inverted = ResourceFilter::default().with_min_image_bytes(4096);
        let error = inverted.with_max_image_bytes(1024).validate().unwrap_err();
        assert!(error.to_string().contains("4096"), "{error}");
        assert!(error.to_string().contains("1024"), "{error}");
    }

    #[test]
    fn resource_filter_serde_defaults_and_round_trip() {
        let parsed: ResourceFilter = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, ResourceFilter::default());
        let filter = ResourceFilter::default()
            .with_min_image_bytes(16)
            .with_allow_cross_host(true);
        let json = serde_json::to_string(&filter).unwrap();
        assert_eq!(
            serde_json::from_str::<ResourceFilter>(&json).unwrap(),
            filter
        );
        let candidate = reference("https://example.com/a.png")
            .with_bytes(vec![1, 2, 3])
            .with_content_type("image/png");
        let json = serde_json::to_string(&candidate).unwrap();
        assert_eq!(
            serde_json::from_str::<ImageCandidate>(&json).unwrap(),
            candidate
        );
    }
}
