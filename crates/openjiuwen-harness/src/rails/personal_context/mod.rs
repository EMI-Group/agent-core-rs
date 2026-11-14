//! Portable, offline subset of the Python `personal_context` package.
//!
//! Python's package is dominated by vendor fetchers and two very large
//! pipeline modules; this subtree keeps only the parts that run offline and
//! deterministically: status codes, portable-path safety, source metadata, the
//! source-link book, local-file ingestion, BM25F keyword recall, and the rail
//! that injects a bounded root description into the prompt.
//!
//! Every remote provider stays behind [`provider::SourceProvider`] and answers
//! with a typed `Error::Unsupported` naming the vendor, so nothing degrades
//! silently.
//!
//! Module layout:
//!
//! - `status` — the seven `${PACKAGE}`-owned status codes and their `Error` mapping
//! - `path` — portable segment checks and safe root-relative resolution
//! - `atomic` — the shared `<path>.tmp` + rename JSON writer/reader
//! - `source` — source metadata, locator normalisation, stable `src_<32 hex>` ids
//! - `link_book` — persisted source-link book and target resolution
//! - `ingest` — local-directory scanning and text snippet extraction
//! - `search` — BM25F keyword ranking and snippet rendering
//! - `provider` — source-provider seam, vendor `Unsupported` providers, registry
//! - `store` — persisted entry index and ingest/search facade
//! - `rail` — [`PersonalContextRail`] prompt-section injection
// The subtree is reached through `crate::rails`, whose module declaration is
// private until that module re-exports these items; the allows keep
// `-D warnings` green during that window.
#![allow(dead_code, unreachable_pub, unused_imports)]

mod atomic;
mod ingest;
mod link_book;
mod path;
mod provider;
mod rail;
mod search;
mod source;
mod status;
mod store;

pub use rail::{
    messages_are_contiguous, render_content, PersonalContextRail, PersonalContextRailConfig,
    DEFAULT_CONFIG_FILENAME, DEFAULT_LANGUAGE, DEFAULT_MAX_DESCRIPTION_CHARS, MAX_CONFIG_BYTES,
    PERSONAL_CONTEXT_PRIORITY, PERSONAL_CONTEXT_SECTION, TRUNCATION_NOTICE,
};
pub use status::{build_error, PersonalContextStatus};

pub use path::{
    portable_context_segment_is_safe, resolve_context_relative_path,
    semantic_context_segment_is_safe, service_storage_segment, service_storage_segment_is_safe,
    validate_service_id, MAX_SEGMENT_CHARS, MAX_SEGMENT_UTF8_BYTES, PORTABLE_FORBIDDEN,
    SEMANTIC_NAME_MAX_CHARS, WINDOWS_DEVICE_NAMES,
};
pub use source::{
    display_title, is_source_id, normalize_source_locator, observed_at_string,
    source_id_for_locator, SourceMetadata, SourceMetadataStore, DEFAULT_SOURCE_TITLE,
    MAX_METADATA_BYTES, MAX_TITLE_CHARS, SOURCE_ID_HEX_CHARS,
};

pub use ingest::{
    ingest_local_directory, IngestReport, IngestedDocument, LocalDirectoryConfig,
    DEFAULT_MAX_FILES, DEFAULT_MAX_FILE_BYTES, DEFAULT_SNIPPET_CHARS, TEXT_EXTENSIONS,
};
pub use link_book::{
    locator_key, resolve_target, SourceLink, SourceLinkBook, IDENTITY_HEX_CHARS, LINK_BOOK_VERSION,
    LINK_TOKEN_PREFIX,
};
pub use provider::{
    unsupported_message, FetchRequest, FetchedSource, LocalFilesProvider, SourceKind,
    SourceProvider, SourceProviderRegistry, UnsupportedSourceProvider,
};
pub use search::{
    page_heading_fields, search, snippet, tokenize, SearchDocument, SearchHit, BM25_B, BM25_K1,
    FIELD_WEIGHTS, MAX_SEARCH_RESULTS, MAX_SNIPPET_CHARS, SNIPPET_LEAD_CHARS,
};
pub use store::{
    ContextEntry, ContextIndex, PersonalContextStore, DEFAULT_SEARCH_LIMIT, DESCRIPTION_FILENAME,
    INDEX_FILENAME, INDEX_VERSION, LINK_BOOK_FILENAME, SOURCE_ROOT_DIRNAME,
};
