//! The inputs, signatures and options of an analysis call.
//!
//! These types are what a caller passes to
//! [`ContextUsageAnalyzer::analyze`](super::ContextUsageAnalyzer::analyze),
//! [`ContextUsageAnalyzer::build_report`](super::ContextUsageAnalyzer::build_report)
//! and
//! [`ContextUsageAnalyzer::analyze_from_report`](super::ContextUsageAnalyzer::analyze_from_report);
//! they carry no measurement logic of their own.
use std::collections::BTreeMap;

use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use crate::stats::ContextStats;

use super::models::{Attribution, ContextCategory, KvCacheUsage, PHASE_PRE_CALL};

/// One declared logical section of the system prompt.
///
/// A section list replaces the measured `system_messages` entirely; the two are
/// never summed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptSection<'a> {
    /// The category the section belongs to; only `system_prompt` and `skills`
    /// are accepted, and anything else is folded into `system_prompt`.
    pub category: ContextCategory,
    /// The section text to measure.
    pub text: &'a str,
    /// The carrier name; defaults to `system_message`.
    pub carrier: Option<&'a str>,
}

impl<'a> PromptSection<'a> {
    /// Builds a section for `category` carrying `text`.
    #[must_use]
    pub fn new(category: ContextCategory, text: &'a str) -> Self {
        Self {
            category,
            text,
            carrier: None,
        }
    }

    /// Names the carrier of the section.
    #[must_use]
    pub fn with_carrier(mut self, carrier: &'a str) -> Self {
        self.carrier = Some(carrier);
        self
    }
}

/// The revision and signature metadata frozen into a window report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowSignatures<'a> {
    /// The message-list revision that was measured.
    pub message_revision: u64,
    /// The prompt revision that was measured.
    pub prompt_revision: u64,
    /// The skills revision that was measured.
    pub skills_revision: u64,
    /// The attachment revision that was measured.
    pub attachment_revision: u64,
    /// A stable signature of the tool definitions that were sent.
    pub tools_signature: Option<&'a str>,
    /// A stable identity of the tokenizer that produced the counts.
    pub tokenizer_identity: Option<&'a str>,
    /// The measurement contract version; defaults to [`super::models::MEASUREMENT_VERSION`].
    pub measurement_version: Option<&'a str>,
    /// Additional per-component signatures the caller wants recorded.
    pub component_signatures: Option<&'a BTreeMap<String, String>>,
}

/// The request identity and environment of a snapshot rebuilt from a report.
#[derive(Debug, Clone)]
pub struct SnapshotOptions<'a> {
    /// The id of the request the snapshot describes.
    pub request_id: &'a str,
    /// The session id, kept for compatibility.
    pub session_id: &'a str,
    /// `pre_call`, `post_call` or `post_compact`.
    pub phase: &'a str,
    /// The provider id, when known.
    pub provider: Option<&'a str>,
    /// The id of the agent that emits the snapshot.
    pub agent_id: Option<&'a str>,
    /// The emission sequence number of the request.
    pub sequence: u64,
    /// The provider-reported input tokens, when known.
    pub provider_input_tokens: Option<u64>,
    /// The per-request and per-session KV-cache blocks.
    pub kv_cache: Option<&'a KvCacheUsage>,
    /// The deployment (`api_base`-level) id, when known.
    pub deployment: Option<&'a str>,
    /// The owner/lineage attribution; the report's own values apply when unset.
    pub attribution: Option<&'a Attribution>,
}

impl<'a> SnapshotOptions<'a> {
    /// Builds options for one request.
    #[must_use]
    pub fn new(request_id: &'a str, session_id: &'a str) -> Self {
        Self {
            request_id,
            session_id,
            phase: PHASE_PRE_CALL,
            provider: None,
            agent_id: None,
            sequence: 0,
            provider_input_tokens: None,
            kv_cache: None,
            deployment: None,
            attribution: None,
        }
    }
}

/// Everything [`super::ContextUsageAnalyzer::analyze`] and
/// [`super::ContextUsageAnalyzer::build_report`] need about one window.
#[derive(Debug, Clone)]
pub struct AnalysisInput<'a> {
    /// The id of the request the measurement belongs to.
    pub request_id: &'a str,
    /// The session id, kept for compatibility.
    pub session_id: &'a str,
    /// The context id, when the measurement belongs to one context.
    pub context_id: Option<&'a str>,
    /// `pre_call`, `post_call` or `post_compact`.
    pub phase: &'a str,
    /// The provider id, when known.
    pub provider: Option<&'a str>,
    /// The id of the agent that emits the snapshot.
    pub agent_id: Option<&'a str>,
    /// The emission sequence number of the request.
    pub sequence: u64,
    /// The provider-reported input tokens, when known.
    pub provider_input_tokens: Option<u64>,
    /// The system messages of the window.
    pub system_messages: Option<&'a [BaseMessage]>,
    /// The non-system messages of the window.
    pub context_messages: &'a [BaseMessage],
    /// The serialized tool definitions sent with the request.
    pub tools: &'a [Value],
    /// Logical system-prompt sections; they replace `system_messages`.
    pub system_prompt_sections: Option<&'a [PromptSection<'a>]>,
    /// The per-request and per-session KV-cache blocks.
    pub kv_cache: Option<&'a KvCacheUsage>,
    /// The deployment (`api_base`-level) id, when known.
    pub deployment: Option<&'a str>,
    /// The id of the context snapshot that was measured.
    pub context_snapshot_id: Option<&'a str>,
    /// Free-form description of what changed in the window.
    pub context_changes: Option<&'a BTreeMap<String, Value>>,
    /// The owner/lineage attribution of the emitting agent.
    pub attribution: Option<&'a Attribution>,
    /// The revision and signature metadata of the window.
    pub signatures: Option<&'a WindowSignatures<'a>>,
    /// The conversation statistics of the measured window.
    pub context_stats: Option<&'a ContextStats>,
}

impl<'a> AnalysisInput<'a> {
    /// Builds an input for `request_id` in `session_id` with an empty window.
    #[must_use]
    pub fn new(request_id: &'a str, session_id: &'a str) -> Self {
        Self {
            request_id,
            session_id,
            context_id: None,
            phase: PHASE_PRE_CALL,
            provider: None,
            agent_id: None,
            sequence: 0,
            provider_input_tokens: None,
            system_messages: None,
            context_messages: &[],
            tools: &[],
            system_prompt_sections: None,
            kv_cache: None,
            deployment: None,
            context_snapshot_id: None,
            context_changes: None,
            attribution: None,
            signatures: None,
            context_stats: None,
        }
    }
}
