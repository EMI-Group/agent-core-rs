//! The `skill_omni_creation` pipeline: URL → scraped page → assets → skill bundle.
//!
//! Ports the scripted pipeline of the Python
//! `openjiuwen/dev_tools/skill_creator/skills/skill_omni_creation/scripts/`
//! (`common.py`, `stage_01_scrape.py` … `stage_05_generate.py`) as a typed,
//! offline-testable Rust pipeline with no browser automation and no vendor SDK:
//!
//! - [`fetch`] — the [`fetch::Fetcher`] seam, [`fetch::HttpFetcher`] over
//!   `openjiuwen_common::http`, and [`fetch::ScriptedFetcher`] for tests.
//! - [`html`] — rule-based HTML extraction into headings, links, images, code
//!   blocks, title, description, and text ([`html::HtmlDocument`]).
//! - [`archive`] — archive sniffing plus ustar unpacking; zip/gzip are deferred
//!   with a typed [`Error::unsupported`](openjiuwen_common::error::Error::unsupported).
//! - [`filter`] — image sniffing, icon heuristics, content/url deduplication,
//!   link deduplication, and [`filter::slugify`].
//! - [`pipeline`] — [`pipeline::OmniCreation`]: scrape → download → generate →
//!   save, plus model-assisted generation.
//!
//! Stages 1–5 are implemented; Playwright browser automation, Pillow image
//! decoding, the LLM image-relevance judge (Python stage 3), and binary image
//! storage are deliberately not ported. Every stage is deterministic and
//! bounded: caps come from [`pipeline::OmniConfig`], ordering is first-seen, and
//! network access happens only through the injected [`fetch::Fetcher`].
pub mod archive;
pub mod fetch;
pub mod filter;
pub mod html;
pub mod pipeline;

pub use archive::{detect, ArchiveEntry, ArchiveFormat, ArchiveUnpacker};
pub use fetch::{FetchedResource, Fetcher, HttpFetcher, ScriptedFetcher};
pub use filter::{
    content_hash, dedupe_links, filter_images, filter_images_with, is_probable_icon, slugify,
    sniff_image, ImageCandidate, ResourceFilter,
};
pub use html::{resolve_url, Heading, HtmlDocument, Image, Link};
pub use pipeline::{DownloadedResource, OmniConfig, OmniCreation, OmniOutcome, ScrapedPage};
