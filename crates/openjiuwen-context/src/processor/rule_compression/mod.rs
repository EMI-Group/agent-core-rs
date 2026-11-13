//! Deterministic, model-free compression of large tool output.
//!
//! Ports `processor/forked/offloader/rule_compression/`: content-type routing,
//! the per-type compressors (HTML, JSON array, log, diff, search results,
//! source file, plain text), query-term extraction, the compression pipeline
//! and its optional debug dump.
