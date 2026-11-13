//! Source-to-[`ScrapedPage`] builders: HTML extraction and `tar` archive text.
//!
//! The fetch, the format detection, and the blank-page guard stay in
//! [`super::OmniCreation::scrape`]; these helpers turn the already-fetched
//! payload into the page the rest of the pipeline consumes, so the stage module
//! can stay focused on orchestration. Everything here is a pure function of its
//! inputs: no I/O, no configuration, and no ordering beyond document order.

use super::super::archive::ArchiveEntry;
use super::super::fetch::FetchedResource;
use super::super::html::{resolve_url, Heading, HtmlDocument, Image, Link};
use super::ScrapedPage;

/// How many leading bytes are inspected when sniffing an archive entry for NULs.
const BINARY_SNIFF_BYTES: usize = 512;

/// Builds a page from a `tar` archive's text entries.
///
/// Directory entries, non-UTF-8 entries, and entries with a NUL in their first
/// 512 bytes are skipped (the skips are logged at `debug`); the remaining text
/// is joined with blank lines, and the first level-1 ATX heading becomes the
/// title.
pub(super) fn archive_page(url: &str, entries: &[ArchiveEntry]) -> ScrapedPage {
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
pub(super) fn html_page(resource: &FetchedResource, url: &str) -> ScrapedPage {
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
