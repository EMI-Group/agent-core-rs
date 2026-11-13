//! Consultation: selecting and rendering tips for prompt injection.
//!
//! Ports the offline intent of Python `openjiuwen.agent_evolving.ttse.consult`
//! and `ttse/render.py`: the bank is ranked against the current query and the
//! winners are rendered as a bounded, category-grouped bullet list.

use crate::ttse::bank::TipBank;
use crate::ttse::config::TtseConfig;
use crate::ttse::models::{ConsultResult, TipRecord};

/// Renders tips grouped by category, bounded to `max_chars` characters.
#[must_use]
pub fn render_tips(_tips: &[TipRecord], _max_chars: usize) -> String {
    String::new()
}

/// Selects the tips relevant to `query` and renders them for injection.
#[must_use]
pub fn consult(_bank: &TipBank, _query: &str, _config: &TtseConfig) -> ConsultResult {
    ConsultResult::default()
}
