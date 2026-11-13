//! Parsing of tip candidates out of model or markdown output.
//!
//! Ports the offline grammar of Python `openjiuwen.agent_evolving.ttse.tip_parse`:
//! a JSON array or `{"tips": [...]}` payload, markdown bullets and numbered
//! lines are all accepted, and a malformed payload falls back to line parsing
//! instead of failing.

/// The default longest rule kept by the parser, in characters.
pub const DEFAULT_MAX_RULE_CHARS: usize = 240;

/// Turns free-form text into deduplicated, classified tip records.
pub struct TipParser;
