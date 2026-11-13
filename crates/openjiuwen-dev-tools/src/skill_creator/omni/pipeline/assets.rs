//! Bundle asset paths: where a kept image is written and how it is named.
//!
//! Ports the naming half of Python's `stage_04_save.py`: every approved image is
//! written into the skill's `references/` directory as `img_{counter:02d}{ext}`,
//! numbered from `00` in kept order, with the extension taken from the URL
//! suffix when it is a supported one and from the declared media type otherwise
//! ([`image_extension`]).
//!
//! The helpers here are stage-free so the renderer can describe the very same
//! asset paths, and so the save stage has one place to enforce that a
//! [`saved_path`](super::DownloadedResource::saved_path) stays inside its bundle.

use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};

use super::super::filter::{image_extension, ImageCandidate};
use super::super::sha256;
use super::DownloadedResource;

/// The bundle-relative directory holding persisted image assets.
pub(crate) const IMAGE_RESOURCE_DIR: &str = "references";

/// The bundle-relative prefix of a persisted image asset (`references/img_`).
pub(crate) const IMAGE_RESOURCE_PREFIX: &str = "references/img_";

/// Builds the bundle-relative path of the `index`-th kept image.
///
/// Mirrors Python's `img_dir / f"img_{counter:02d}{ext}"`: `index` is the
/// 0-based position among the images that survived filtering, so the first kept
/// image is `references/img_00.png`. The extension comes from
/// [`image_extension`], i.e. the lowercased URL suffix when it is one of
/// `png`/`jpg`/`jpeg`/`webp`/`gif`, else the media type, else `png`.
#[must_use]
pub(crate) fn image_resource_path(index: usize, url: &str, content_type: Option<&str>) -> String {
    format!(
        "{IMAGE_RESOURCE_PREFIX}{index:02}.{}",
        image_extension(url, content_type)
    )
}

/// Converts one filtered image candidate into a persisted-asset resource.
///
/// `index` is the candidate's position among the kept images, so the resource
/// carries its `references/img_NN.<ext>` saved path, its payload digest, and the
/// scraped alt text.
pub(crate) fn image_resource(index: usize, candidate: ImageCandidate) -> DownloadedResource {
    let bytes = candidate.bytes.unwrap_or_default();
    let content_type = candidate.content_type;
    let saved_path = image_resource_path(index, &candidate.url, content_type.as_deref());
    DownloadedResource {
        url: candidate.url,
        content_type,
        sha256: sha256::hex_digest(&bytes),
        bytes,
        alt: candidate.alt,
        saved_path: Some(saved_path),
    }
}

/// Resolves a validated image `saved_path` to a path relative to the bundle.
///
/// The path must be relative, `/`-separated, start with [`IMAGE_RESOURCE_DIR`],
/// and hold no empty, `.`, or `..` component and no backslash — the accepted
/// grammar is `references/img_NN.<ext>` and nothing that could leave the bundle.
///
/// # Errors
///
/// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
/// naming the offending path when it does not match that grammar, so a
/// hand-crafted [`saved_path`](super::DownloadedResource::saved_path) can never
/// write outside the skill directory.
pub(crate) fn image_target(saved_path: &str) -> Result<PathBuf> {
    if saved_path.contains('\\') {
        return Err(unsafe_path(saved_path));
    }
    let mut components = saved_path.split('/');
    let mut well_formed = components.next() == Some(IMAGE_RESOURCE_DIR);
    let mut parts = Vec::new();
    for component in components {
        well_formed &= !component.is_empty() && component != "." && component != "..";
        parts.push(component);
    }
    if !well_formed || parts.is_empty() {
        return Err(unsafe_path(saved_path));
    }
    let mut path = PathBuf::from(IMAGE_RESOURCE_DIR);
    for part in parts {
        path.push(part);
    }
    Ok(path)
}

/// The validation error naming an offending image asset path.
fn unsafe_path(saved_path: &str) -> Error {
    Error::validation(format!(
        "unsafe image asset path {saved_path:?}: expected \
         {IMAGE_RESOURCE_DIR}/img_NN.<ext>"
    ))
}
