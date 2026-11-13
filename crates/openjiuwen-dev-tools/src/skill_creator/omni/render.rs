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
Never invent steps, commands, or image paths; refer to an image by its URL when it matters.";

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
/// sections that have content. Image and link references always stay URLs;
/// `references` contributes the `## Reference Files` listing.
pub(crate) fn instructions_markdown(
    page: &ScrapedPage,
    name: &str,
    description: &str,
    references: &[SkillResource],
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

    let images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    if !images.is_empty() {
        let _ = writeln!(out, "## Reference Images\n");
        for image in images {
            let alt = image_alt(image);
            let _ = writeln!(out, "- {alt} — {}", image.src);
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

    if !references.is_empty() {
        let _ = writeln!(out, "## Reference Files\n");
        for resource in references {
            let description = describe_resource(&resource.path);
            let _ = writeln!(out, "- `{}` — {description}", resource.path);
        }
    }
    out.trim_end().to_owned()
}

/// Builds the `references/sources.md` body for `page`.
///
/// The document holds the source URL, the title and description when present,
/// the bounded deduplicated subpage links, and every image URL with its alt
/// text, all reading the caps from `config`.
pub(crate) fn sources_markdown(page: &ScrapedPage, config: &OmniConfig) -> String {
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
    let images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    if !images.is_empty() {
        let _ = writeln!(out, "\n## Images\n");
        for image in images {
            let alt = image_alt(image);
            let _ = writeln!(out, "- {alt} — {}", image.src);
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
/// code blocks, and the image URLs with their alt text.
pub(crate) fn model_user_prompt(page: &ScrapedPage, config: &OmniConfig) -> String {
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
    let images: Vec<&Image> = page.images.iter().take(config.max_images).collect();
    if !images.is_empty() {
        prompt.push_str("\nImages:\n");
        for image in images {
            let alt = image_alt(image);
            let _ = writeln!(prompt, "- {alt} — {}", image.src);
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
