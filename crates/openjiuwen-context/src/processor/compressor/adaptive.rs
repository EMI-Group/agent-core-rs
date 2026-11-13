//! Adaptive compression strategy selection.
//!
//! Ports the model-chosen branch of Python's compressor family: a strategy
//! request that can pick the compression pass and task/JSON-schema memory
//! block shape, with a deterministic ladder whenever no model is configured
//! or the model reply cannot be parsed.
