//! Deterministic rendering and model-prompt helpers of the omni pipeline.
//!
//! Everything the pipeline turns into text lives here: the skill name and
//! description derived from a [`ScrapedPage`], the `SKILL.md` instruction body,
//! the `references/sources.md` and `references/page-NN.md` documents, and the
//! prompt of the model-assisted generator. [`super::pipeline`] keeps the caps,
//! the fetching, and the stage orchestration; these helpers are pure functions
//! of their inputs, so they are deterministic, offline-testable, and free of
//! I/O. The helpers are crate-internal; the public surface of this stage stays
//! [`super::pipeline`].
//!
//! The cap-driven choices stay with the caller: `sources_markdown`,
//! `instructions_markdown`, and `model_user_prompt` read
//! [`OmniConfig::max_links`], [`OmniConfig::max_images`], and
//! [`OmniConfig::allow_cross_host`], while the `references/` resource paths are
//! `SOURCES_RESOURCE` and `PAGE_RESOURCE_PREFIX` so the pipeline and the
//! document bodies cannot drift apart.
//!
//! Persisted images are rendered from their
//! [`saved_path`](DownloadedResource::saved_path): the instruction body points
//! at the bundle-relative asset instead of the source URL, and the single
//! `## Reference Files` section — `reference_files_section` — lists the saved
//! images beside the text resources for both the deterministic and the
//! model-assisted path.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use openjiuwen_common::error::{Error, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::skill_creator::SkillResource;

use super::filter::{self, dedupe_links, slugify};
use super::html::{resolve_url, Heading, HtmlDocument, Image, Link};
use super::pipeline::{DownloadedResource, OmniConfig, ScrapedPage};

/// Longest description written into the generated document, in characters.
const MAX_DESCRIPTION_CHARS: usize = 300;
/// How much scraped text is offered to the model, in characters.
const MAX_MODEL_TEXT_CHARS: usize = 8_000;
/// How many text paragraphs the deterministic document lists.
const MAX_DETAIL_PARAGRAPHS: usize = 20;
/// How many fenced code blocks the deterministic document lists.
const MAX_CODE_EXAMPLES: usize = 5;
/// The name used when neither the title nor the URL yields a slug.
const FALLBACK_NAME: &str = "scraped-skill";
/// The provenance resource every generated bundle carries.
pub(crate) const SOURCES_RESOURCE: &str = "references/sources.md";
/// The prefix of the per-subpage reference resources.
pub(crate) const PAGE_RESOURCE_PREFIX: &str = "references/page-";
/// The system prompt of the model-assisted generator.
pub(crate) const MODEL_SYSTEM_PROMPT: &str = "\
You turn a scraped documentation page into a portable agent skill. Answer with exactly one ```json code block whose object has:
- name: ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit
- description: one to three sentences describing the skill and when to use it
- instructions: markdown body with numbered steps
- metadata (optional): a string-keyed object of extra values

Ground every statement in the supplied title, headings, text, code blocks, and images.
Never invent steps, commands, or image paths; an image may only be referenced by the relative bundle path listed for it.";

/// The skill-name slug of a page: title, then URL host, then the raw URL.
pub(crate) fn skill_name(page: &ScrapedPage) -> String {
    let mut name = slugify(&page.title);
    if name.is_empty() {
        let source = filter::host(&page.url).unwrap_or_else(|| page.url.clone());
        name = slugify(&source);
    }
    if name.is_empty() {
        FALLBACK_NAME.to_owned()
    } else {
        name
    }
}

/// The one-line description of a page, capped on a character boundary.
pub(crate) fn skill_description(page: &ScrapedPage) -> String {
    if !page.description.trim().is_empty() {
        return truncate_chars(page.description.trim(), MAX_DESCRIPTION_CHARS);
    }
    if let Some(line) = page
        .text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
    {
        return truncate_chars(line, MAX_DESCRIPTION_CHARS);
    }
    format!("Usage steps captured from {}", page.url)
}

/// Renders the deterministic `SKILL.md` instruction body.
///
/// The body opens with the page title (or `name` when the page has none) and
/// `description`, then lists the `## Steps`, `## Details`, `## Code Examples`,
/// `## Reference Images`, `## Further Reading`, and `## Reference Files`
/// sections that have content. Link references stay URLs.
///
/// `images` is the downloaded resource list of the run; its entries that carry
/// a [`saved_path`](DownloadedResource::saved_path) are the kept images, in keep
/// order. `## Reference Images` renders one markdown image per kept image —
/// `![{alt}]({saved_path})`, with the trimmed alt text falling back to
/// `screenshot` — followed by a `- {alt} — {src}` URL line for every page image
/// that was not persisted. A page image is considered persisted when its
/// resolved URL equals the `url` of a kept image literally (no path
/// normalisation). `## Reference Files` appears exactly once, saved images
/// first as `- {path} — {description}` lines, then the text `references`; it is
/// built by [`reference_files_section`], the single renderer of that section,
/// and is omitted when both lists are empty.
pub(crate) fn instructions_markdown(
    page: &ScrapedPage,
    name: &str,
    description: &str,
    references: &[SkillResource],
    images: &[DownloadedResource],
    config: &OmniConfig,
) -> String {
    let title = if page.title.trim().is_empty() {
        name
    } else {
        page.title.trim()
    };
    let mut out = String::new();
    let _ = writeln!(out, "# {title}\n");
    let _ = writeln!(out, "{description}\n");

    let steps = step_headings(page, title);
    if !steps.is_empty() {
        let _ = writeln!(out, "## Steps\n");
        for heading in steps {
            let _ = writeln!(out, "- **{}**", heading.text.trim());
        }
        out.push('\n');
    }

    let paragraphs = paragraphs(&page.text);
    if !paragraphs.is_empty() {
        let _ = writeln!(out, "## Details\n");
        for paragraph in paragraphs.iter().take(MAX_DETAIL_PARAGRAPHS) {
            let _ = writeln!(out, "{paragraph}\n");
        }
    }

    let code_blocks: Vec<&String> = page
        .code_blocks
        .iter()
        .filter(|block| !block.trim().is_empty())
        .take(MAX_CODE_EXAMPLES)
        .collect();
    if !code_blocks.is_empty() {
        let _ = writeln!(out, "## Code Examples\n");
        for block in code_blocks {
            let _ = writeln!(out, "```\n{}\n```\n", block.trim());
        }
    }

    let page_images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    let mut image_lines: Vec<String> = Vec::new();
    for saved in images.iter().filter(|image| image.saved_path.is_some()) {
        if let Some(path) = saved.saved_path.as_deref() {
            image_lines.push(format!("![{}]({path})", saved_image_alt(saved)));
        }
    }
    for image in &page_images {
        if saved_image(page, image, images).is_some() {
            continue;
        }
        let alt = image_alt(image);
        image_lines.push(format!("- {alt} — {}", image.src));
    }
    if !image_lines.is_empty() {
        let _ = writeln!(out, "## Reference Images\n");
        for line in image_lines {
            let _ = writeln!(out, "{line}");
        }
        out.push('\n');
    }

    let further = further_reading(page, config);
    if !further.is_empty() {
        let _ = writeln!(out, "## Further Reading\n");
        for link in &further {
            let label = link_label(link);
            let _ = writeln!(out, "- [{label}]({})", link.href);
        }
        out.push('\n');
    }

    if let Some(section) = reference_files_section(images, references) {
        let _ = writeln!(out, "{section}");
    }
    out.trim_end().to_owned()
}

/// Renders the single `## Reference Files` section of an instruction body.
///
/// The section lists, one per line, the bundle-relative paths of the saved
/// images with their description (the trimmed alt text, else `screenshot`),
/// then the text `references` with their [`describe_resource`] description. The
/// line shape follows Python's `append_reference_files`: an ASCII hyphen-minus,
/// the backtick-quoted path, an em dash, then the description. `None` means both
/// lists are empty, so no section is rendered; the section text itself carries
/// no trailing newline, which lets both the deterministic and the model-assisted
/// paths append it verbatim.
pub(crate) fn reference_files_section(
    images: &[DownloadedResource],
    references: &[SkillResource],
) -> Option<String> {
    let mut out = String::new();
    for image in images {
        let Some(path) = image.saved_path.as_deref() else {
            continue;
        };
        let _ = writeln!(out, "- `{path}` — {}", saved_image_alt(image));
    }
    for resource in references {
        let description = describe_resource(&resource.path);
        let _ = writeln!(out, "- `{}` — {description}", resource.path);
    }
    if out.is_empty() {
        return None;
    }
    Some(format!("## Reference Files\n\n{}", out.trim_end()))
}

/// Builds the `references/sources.md` body for `page`.
///
/// The document holds the source URL, the title and description when present,
/// the bounded deduplicated subpage links, and every image URL with its alt
/// text, all reading the caps from `config`. An image that was persisted to the
/// bundle keeps its URL and gains the saved path, in the shape
/// `- {alt} — {url} (saved as references/img_NN.<ext>)`, so the provenance
/// document points at both the origin and the local asset.
pub(crate) fn sources_markdown(
    page: &ScrapedPage,
    images: &[DownloadedResource],
    config: &OmniConfig,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Sources\n");
    let _ = writeln!(out, "- Source: {}", page.url);
    if !page.title.trim().is_empty() {
        let _ = writeln!(out, "- Title: {}", page.title.trim());
    }
    if !page.description.trim().is_empty() {
        let _ = writeln!(out, "- Description: {}", page.description.trim());
    }
    let links = further_reading(page, config);
    if !links.is_empty() {
        let _ = writeln!(out, "\n## Links\n");
        for link in &links {
            let label = link_label(link);
            let _ = writeln!(out, "- [{label}]({})", link.href);
        }
    }
    let page_images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    if !page_images.is_empty() {
        let _ = writeln!(out, "\n## Images\n");
        for image in page_images {
            let alt = image_alt(image);
            match saved_image(page, image, images).and_then(|saved| saved.saved_path.as_deref()) {
                Some(path) => {
                    let _ = writeln!(out, "- {alt} — {} (saved as {path})", image.src);
                }
                None => {
                    let _ = writeln!(out, "- {alt} — {}", image.src);
                }
            }
        }
    }
    out
}

/// Renders one HTML subpage as a markdown reference document, or `None` when
/// the payload is not UTF-8 or holds nothing worth writing.
pub(crate) fn render_page_resource(
    page: &ScrapedPage,
    resource: &DownloadedResource,
) -> Option<String> {
    let text = match std::str::from_utf8(&resource.bytes) {
        Ok(text) => text,
        Err(error) => {
            tracing::debug!(
                url = %resource.url,
                %error,
                "skipping a non-UTF-8 reference page"
            );
            return None;
        }
    };
    let document = HtmlDocument::parse(text, &resource.url);
    if document.title.trim().is_empty()
        && document.headings.is_empty()
        && document.text.trim().is_empty()
    {
        return None;
    }
    let mut out = String::new();
    let title = if document.title.trim().is_empty() {
        page.title.trim()
    } else {
        document.title.trim()
    };
    let title = if title.is_empty() { "Reference" } else { title };
    let _ = writeln!(out, "# {title}");
    for heading in &document.headings {
        let text = heading.text.trim();
        if text.is_empty() {
            continue;
        }
        let marker = if heading.level <= 2 { "##" } else { "###" };
        let _ = writeln!(out, "\n{marker} {text}");
    }
    if !document.text.trim().is_empty() {
        let _ = writeln!(out, "\n{}", document.text.trim());
    }
    Some(out)
}

/// The user message of the model-assisted generator.
///
/// It carries the title, source URL, the optional description, the headings,
/// the text truncated to 8 000 characters on a character boundary, the fenced
/// code blocks, and the images with their alt text. A kept image (one with a
/// [`saved_path`](DownloadedResource::saved_path)) is listed as
/// `- {alt} — {saved_path}`, mirroring Python's `blocks_for_llm`, which drops
/// the raw URL so the model can only cite the bundle-relative asset it was
/// given; an image without a saved path keeps its absolute URL.
pub(crate) fn model_user_prompt(
    page: &ScrapedPage,
    images: &[DownloadedResource],
    config: &OmniConfig,
) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "Title: {}", page.title.trim());
    let _ = writeln!(prompt, "Source: {}", page.url);
    if !page.description.trim().is_empty() {
        let _ = writeln!(prompt, "Description: {}", page.description.trim());
    }
    if !page.headings.is_empty() {
        prompt.push_str("\nHeadings:\n");
        for heading in &page.headings {
            let _ = writeln!(prompt, "- h{} {}", heading.level, heading.text.trim());
        }
    }
    let text = truncate_chars(page.text.trim(), MAX_MODEL_TEXT_CHARS);
    if !text.is_empty() {
        let _ = writeln!(prompt, "\nText:\n{text}");
    }
    let code_blocks: Vec<&String> = page
        .code_blocks
        .iter()
        .filter(|block| !block.trim().is_empty())
        .take(MAX_CODE_EXAMPLES)
        .collect();
    if !code_blocks.is_empty() {
        prompt.push_str("\nCode blocks:\n");
        for block in code_blocks {
            let _ = writeln!(prompt, "```\n{}\n```", block.trim());
        }
    }
    let page_images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    if !page_images.is_empty() {
        prompt.push_str("\nImages:\n");
        for image in page_images {
            let alt = image_alt(image);
            match saved_image(page, image, images).and_then(|saved| saved.saved_path.as_deref()) {
                Some(path) => {
                    let _ = writeln!(prompt, "- {alt} — {path}");
                }
                None => {
                    let _ = writeln!(prompt, "- {alt} — {}", image.src);
                }
            }
        }
    }
    prompt
}

/// The JSON payload a drafting model is expected to return.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ModelSkillPayload {
    /// Skill name; required once parsed.
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// Skill description; required once parsed.
    #[serde(default)]
    pub(crate) description: Option<String>,
    /// Instruction body; required once parsed.
    #[serde(default)]
    pub(crate) instructions: Option<String>,
    /// Optional metadata object.
    #[serde(default)]
    pub(crate) metadata: BTreeMap<String, Value>,
}

impl ModelSkillPayload {
    /// Reads a required, non-blank field, naming it in the failure.
    pub(crate) fn required(&self, field: &str) -> Result<String> {
        let value = match field {
            "name" => self.name.as_deref(),
            "description" => self.description.as_deref(),
            "instructions" => self.instructions.as_deref(),
            other => {
                return Err(Error::validation(format!(
                    "unknown omni skill payload field '{other}'"
                )));
            }
        };
        match value {
            Some(value) if !value.trim().is_empty() => Ok(value.to_owned()),
            _ => Err(Error::validation(format!(
                "omni skill definition JSON from model is missing a non-blank '{field}'"
            ))),
        }
    }
}

/// The bounded, same-host, non-self subpage links of `page`.
fn further_reading(page: &ScrapedPage, config: &OmniConfig) -> Vec<Link> {
    let mut links = Vec::new();
    for link in dedupe_links(&page.links) {
        if links.len() >= config.max_links {
            break;
        }
        let Some(target) = resolve_url(&page.url, &link.href) else {
            continue;
        };
        if !config.allow_cross_host && !filter::same_host(&target, &page.url) {
            continue;
        }
        if same_page(&target, &page.url) {
            continue;
        }
        links.push(Link {
            href: target,
            text: link.text,
        });
    }
    links
}

/// Whether two URLs name the same page modulo a trailing slash.
pub(crate) fn same_page(left: &str, right: &str) -> bool {
    left.trim_end_matches('/') == right.trim_end_matches('/')
}

/// The headings that become steps: no empty text, no title duplicate, and not
/// the leading level-1 heading that already titles the document.
fn step_headings<'a>(page: &'a ScrapedPage, title: &str) -> Vec<&'a Heading> {
    page.headings
        .iter()
        .enumerate()
        .filter(|(index, heading)| {
            let text = heading.text.trim();
            !(text.is_empty()
                || text.eq_ignore_ascii_case(title)
                || (*index == 0 && heading.level == 1))
        })
        .map(|(_, heading)| heading)
        .collect()
}

/// Splits text into paragraphs of blank-line separated, space-joined lines.
fn paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(trimmed);
        }
    }
    if !current.is_empty() {
        paragraphs.push(current.join(" "));
    }
    paragraphs
}

/// Truncates `text` to at most `max_chars` characters without splitting one.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut end = 0;
    for (count, (index, character)) in text.char_indices().enumerate() {
        if count == max_chars {
            break;
        }
        end = index + character.len_utf8();
    }
    text.get(..end).unwrap_or_default().to_owned()
}

/// The label of a link: its text, else its `href`.
fn link_label(link: &Link) -> &str {
    let text = link.text.trim();
    if text.is_empty() {
        link.href.trim()
    } else {
        text
    }
}

/// The display text of an image: its alt text, else `"image"`.
fn image_alt(image: &Image) -> &str {
    let alt = image.alt.trim();
    if alt.is_empty() {
        "image"
    } else {
        alt
    }
}

/// The description of a persisted image: its trimmed alt text, else
/// `"screenshot"` (Python's `b.get("alt") or "screenshot"`).
fn saved_image_alt(image: &DownloadedResource) -> &str {
    let alt = image.alt.trim();
    if alt.is_empty() {
        "screenshot"
    } else {
        alt
    }
}

/// The kept image whose `url` is the resolved URL of `image`, if any.
///
/// The comparison is literal, like Python's `valid_paths` set membership: no
/// path normalisation and no query-string or fragment stripping. A page image
/// that no saved resource matches was filtered out, so it stays a URL
/// reference.
fn saved_image<'a>(
    page: &ScrapedPage,
    image: &Image,
    images: &'a [DownloadedResource],
) -> Option<&'a DownloadedResource> {
    let target = resolve_url(&page.url, &image.src).unwrap_or_else(|| image.src.trim().to_owned());
    images
        .iter()
        .find(|saved| saved.saved_path.is_some() && saved.url == target)
}

/// The one-line description of a written text resource.
fn describe_resource(path: &str) -> &'static str {
    if path == SOURCES_RESOURCE {
        "source page provenance, links, and image references"
    } else if path.starts_with(PAGE_RESOURCE_PREFIX) {
        "captured text of a followed subpage"
    } else {
        "captured reference text"
    }
}

/// Removes the image references whose target is not one of `valid_paths`.
///
/// Ports `strip_hallucinated_images` from the Python `common.py` of the
/// `skill_omni_creation` scripts: a drafting model likes to invent image paths,
/// so a `![alt](target)` reference is kept only when its trimmed `target` equals
/// one of `valid_paths` exactly (no path normalisation, no query-string or
/// fragment stripping). Every other reference is deleted in place, leaving the
/// surrounding text and any enclosing link untouched, and the blank-line runs
/// the deletion leaves behind collapse back to at most one blank line before
/// the result is trimmed.
///
/// The scanner is hand-written, like the rest of this module, and mirrors the
/// Python regular expression `!\[([^\]]*)\]\(([^)]+)\)`: it looks for `![`, takes
/// the first `]` as the end of the alt text, requires the `](` pair immediately,
/// and takes the first `)` as the end of a non-empty target. An attempt that
/// does not complete (no `]`, no `](`, or an empty `()`) is left literal and
/// scanning resumes after the `![`, so a reference nested inside the unfinished
/// text still matches.
///
/// Line boundaries are `\n` with an optional trailing `\r`, the convention of
/// the crate's other scanners, rather than Python's wider `str.splitlines`.
///
/// The model-assisted generation path calls this before the drafted
/// instructions are validated, with the `saved_path`s of the kept images as
/// `valid_paths`.
pub(crate) fn strip_hallucinated_images(markdown: &str, valid_paths: &[String]) -> String {
    let mut cleaned = String::with_capacity(markdown.len());
    let mut rest = markdown;
    while let Some(start) = rest.find("![") {
        cleaned.push_str(&rest[..start]);
        if let Some((target, length)) = image_reference(&rest[start + 1..]) {
            if valid_paths.iter().any(|path| path == target.trim()) {
                cleaned.push_str(&rest[start..start + 1 + length]);
            }
            rest = &rest[start + 1 + length..];
        } else {
            cleaned.push_str("![");
            rest = &rest[start + 2..];
        }
    }
    cleaned.push_str(rest);
    collapse_blank_lines(&cleaned)
}

/// The target and byte length (from `[` through `)`) of the image reference a
/// `![` just opened, or `None` when the reference does not complete.
fn image_reference(after_bang: &str) -> Option<(&str, usize)> {
    let alt_end = after_bang.find(']')?;
    if after_bang.as_bytes().get(alt_end + 1) != Some(&b'(') {
        return None;
    }
    let target_start = alt_end + 2;
    let target_len = after_bang.get(target_start..)?.find(')')?;
    if target_len == 0 {
        return None;
    }
    let target = after_bang.get(target_start..target_start + target_len)?;
    Some((target, target_start + target_len + 1))
}

/// Drops every blank line that starts the document or follows another blank
/// line, then trims the joined result.
fn collapse_blank_lines(text: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        let content = !line.trim().is_empty();
        let previous_is_content = kept.last().is_some_and(|last| !last.trim().is_empty());
        if content || previous_is_content {
            kept.push(line);
        }
    }
    kept.join("\n").trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::strip_hallucinated_images;

    /// The valid-path list handed to [`strip_hallucinated_images`].
    fn paths(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    #[test]
    fn text_without_images_keeps_its_content_and_loses_stray_blank_lines() {
        assert_eq!(
            strip_hallucinated_images("# Title\n\nJust text\n\n\n", &[]),
            "# Title\n\nJust text"
        );
        assert_eq!(strip_hallucinated_images("", &[]), "");
        assert_eq!(
            strip_hallucinated_images("\n\nText ![x](y.png)\n\n\n", &[]),
            "Text"
        );
    }

    #[test]
    fn a_valid_reference_survives_with_alt_spaces_and_unicode() {
        let document = "Before ![a diagram with spaces — ünïcode](references/img_00.png) after";
        assert_eq!(
            strip_hallucinated_images(document, &paths(&["references/img_00.png"])),
            document
        );
        assert_eq!(
            strip_hallucinated_images(document, &paths(&["references/other.png"])),
            "Before  after"
        );
    }

    #[test]
    fn targets_are_trimmed_before_the_exact_comparison() {
        let document = "![logo]( references/img_00.png )";
        assert_eq!(
            strip_hallucinated_images(document, &paths(&["references/img_00.png"])),
            document
        );
        assert_eq!(
            strip_hallucinated_images(document, &paths(&["./references/img_00.png"])),
            ""
        );
    }

    #[test]
    fn a_query_string_target_is_not_the_valid_path() {
        assert_eq!(
            strip_hallucinated_images(
                "![shot](references/img_00.png?x=1)",
                &paths(&["references/img_00.png"]),
            ),
            ""
        );
    }

    #[test]
    fn a_reference_nested_in_a_link_loses_only_the_reference() {
        assert_eq!(
            strip_hallucinated_images("[![a](x.png)](https://example.com/docs)", &[]),
            "[](https://example.com/docs)"
        );
        let kept = "[![a](x.png)](https://example.com/docs)";
        assert_eq!(strip_hallucinated_images(kept, &paths(&["x.png"])), kept);
    }

    #[test]
    fn an_empty_alt_still_matches() {
        assert_eq!(
            strip_hallucinated_images("![](x.png)", &paths(&["x.png"])),
            "![](x.png)"
        );
        assert_eq!(strip_hallucinated_images("![](x.png)", &[]), "");
    }

    #[test]
    fn incomplete_references_stay_literal() {
        for document in [
            "Text ![unterminated alt\nmore text",
            "Text ![alt](missing-close",
            "Text ![alt] (not-an-image)",
            "Text ![alt]()",
        ] {
            assert_eq!(strip_hallucinated_images(document, &[]), document);
        }
    }

    #[test]
    fn two_references_on_one_line_keep_the_surviving_line() {
        assert_eq!(
            strip_hallucinated_images("![keep](a.png) and ![drop](b.png)", &paths(&["a.png"])),
            "![keep](a.png) and"
        );
    }

    #[test]
    fn a_line_emptied_by_a_stripped_reference_disappears() {
        assert_eq!(
            strip_hallucinated_images("# Title\n\n![drop](b.png)\n\nAfter", &[]),
            "# Title\n\nAfter"
        );
    }
}
