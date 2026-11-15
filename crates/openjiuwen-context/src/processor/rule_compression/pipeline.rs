//! The rule-compression pass: budget, route, stamp metadata, dump.
//!
//! Ports `rule_compression/pipeline.py`. A pass runs on one tool message:
//!
//! 1. only `Role::Tool` messages with text content are considered — a message
//!    whose content is a part list is returned unchanged with one
//!    `tracing::debug!` line naming the message index, the content kind and the
//!    `non_text_content` reason, and an index the conversation does not own is
//!    [`Error::Validation`] naming the index and the message count,
//! 2. the message is skipped when its text is not longer than `max_chars`,
//!    unless `force` is set (the TTL pass compresses regardless of size),
//! 3. a [`RuleContext`] is derived from the character budget — `max_tokens =
//!    max_chars / 3`, head and tail `max_chars / 6` — and from the conversation:
//!    query terms come from the latest user message plus the triggering tool call,
//!    and `tool_name`/`file_path` come from that call,
//! 4. the router compresses, and a modified result is returned as a copy of the
//!    message with the new text, the `rule_compressed_*` metadata keys, and the
//!    candidate's honesty flag (`rule_compression_lossy`),
//! 5. the optional dump writes one record — a pretty JSON file per application or
//!    one JSONL line per application — and one `tracing::info!` line records the
//!    pass.
//!
//! The dump carries the real session and context ids: the caller passes the
//! [`crate::offload::OffloadScope`] it runs under (see
//! [`RuleCompressionPipeline::compress_in_messages_with_scope`]), and the
//! placeholders [`crate::processor::rule_compression::dump::UNKNOWN_SESSION_ID`] and
//! [`crate::processor::rule_compression::dump::UNKNOWN_CONTEXT_ID`] appear only
//! when that scope genuinely has none.
//!
//! The pass is synchronous and deterministic; only the dump touches the
//! filesystem, and a failed dump never fails the pass. A dump directory that
//! cannot be written is reported once per directory, naming the path and the
//! error, instead of repeating the warning on every application.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{BaseMessage, MessageContent, Role};
use serde_json::{json, Value};

use super::common::extract_file_path_argument;
use super::dump::{
    resolve_dump_dir, write_record, DumpRecord, DumpTemplateVars, RuleCompressionDumpFormat,
};
use super::query_terms::extract_query_terms;
use super::router::RuleContentRouter;
use super::types::{
    RuleCompressionResult, RuleCompressionStrategy, RuleContext, CHARACTERS_PER_TOKEN,
};
use crate::offload::OffloadScope;
use crate::processor::offloader::resolve_tool_call;

/// Metadata key: wall-clock time of the compression.
pub const RULE_COMPRESSED_AT_KEY: &str = "rule_compressed_at";
/// Metadata key: detected content type.
pub const RULE_COMPRESSION_TYPE_KEY: &str = "rule_compression_type";
/// Metadata key: pipeline pass that compressed the message.
pub const RULE_COMPRESSION_PASS_KEY: &str = "rule_compression_pass";
/// Metadata key: the compressor's details payload.
pub const RULE_COMPRESSION_DETAILS_KEY: &str = "rule_compression_details";
/// Metadata key: path of the debug dump, when one was written.
pub const RULE_COMPRESSION_DUMP_PATH_KEY: &str = "rule_compression_dump_path";
/// Metadata key: whether the accepted candidate dropped information.
pub const RULE_COMPRESSION_LOSSY_KEY: &str = "rule_compression_lossy";
/// Pass name of the pass that runs when a tool result is added.
pub const ADD_PASS: &str = "add";
/// Pass name of the pass that compresses whatever survived its time to live.
pub const TTL_PASS: &str = "ttl";
/// A clock the pipeline reads instead of the wall clock.
pub type RuleClock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Runs deterministic rule compression over tool messages.
#[derive(Clone)]
pub struct RuleCompressionPipeline {
    router: RuleContentRouter,
    clock: RuleClock,
    enable_dump: bool,
    dump_dir: Option<PathBuf>,
    dump_format: RuleCompressionDumpFormat,
    strategy: RuleCompressionStrategy,
    /// Last dump directory a write failure was reported for, so the warning is
    /// emitted once per directory rather than once per application.
    warned_dump_dir: Arc<Mutex<Option<String>>>,
}

impl std::fmt::Debug for RuleCompressionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleCompressionPipeline")
            .field("enable_dump", &self.enable_dump)
            .field("dump_dir", &self.dump_dir)
            .field("dump_format", &self.dump_format)
            .field("strategy", &self.strategy)
            .finish_non_exhaustive()
    }
}

impl Default for RuleCompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleCompressionPipeline {
    /// A pipeline that reads the system clock and dumps nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            router: RuleContentRouter::new(),
            clock: Arc::new(system_now),
            enable_dump: false,
            dump_dir: None,
            dump_format: RuleCompressionDumpFormat::default(),
            strategy: RuleCompressionStrategy::default(),
            warned_dump_dir: Arc::new(Mutex::new(None)),
        }
    }

    /// A pipeline that reads `clock` instead of the system clock.
    #[must_use]
    pub fn with_clock(clock: impl Fn() -> f64 + Send + Sync + 'static) -> Self {
        Self {
            clock: Arc::new(clock),
            ..Self::new()
        }
    }

    /// Enables dumps into `dir` (or the environment variable, else the workspace
    /// default [`super::dump::DEFAULT_DUMP_DIR`], when `dir` is `None`).
    ///
    /// `dir` may carry the `{session_id}`, `{context_id}`, `{date}` and
    /// `{timestamp}` templates, which are expanded per application.
    #[must_use]
    pub fn with_dump(mut self, enable: bool, dir: Option<String>) -> Self {
        self.enable_dump = enable;
        self.dump_dir = dir.as_deref().map(PathBuf::from);
        self
    }

    /// Selects how a dump is written (one JSON file per application, the default,
    /// or one appended JSONL line per application).
    #[must_use]
    pub fn with_dump_format(mut self, format: RuleCompressionDumpFormat) -> Self {
        self.dump_format = format;
        self
    }

    /// Selects which tier this pipeline belongs to.
    ///
    /// The pipeline *is* the rule tier, so [`RuleCompressionStrategy::TruncateOnly`]
    /// makes every pass a no-op; the caller keys its own summary/offload tier off
    /// the same value.
    #[must_use]
    pub fn with_strategy(mut self, strategy: RuleCompressionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Whether a modified message is dumped to disk.
    #[must_use]
    pub fn dump_enabled(&self) -> bool {
        self.enable_dump
    }

    /// The configured dump directory, before the environment fallback.
    #[must_use]
    pub fn dump_dir(&self) -> Option<&std::path::Path> {
        self.dump_dir.as_deref()
    }

    /// How a dump is written.
    #[must_use]
    pub fn dump_format(&self) -> RuleCompressionDumpFormat {
        self.dump_format
    }

    /// The strategy this pipeline runs under.
    #[must_use]
    pub fn strategy(&self) -> RuleCompressionStrategy {
        self.strategy
    }

    /// The router this pipeline dispatches through.
    #[must_use]
    pub fn router(&self) -> &RuleContentRouter {
        &self.router
    }

    /// The current time according to the injected clock.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        (self.clock)()
    }

    /// The character budget of a context window of `context_window_tokens`.
    #[must_use]
    pub fn context_character_capacity(context_window_tokens: usize) -> usize {
        context_window_tokens
            .saturating_mul(CHARACTERS_PER_TOKEN)
            .max(1)
    }

    /// Builds the [`RuleContext`] one pass runs with.
    ///
    /// `max_tokens` is `max_chars / 3` and the head/tail budgets are
    /// `max_chars / 6` (both at least one); the routing hints come from the
    /// latest user message and from `tool_name`/`tool_arguments`.
    #[must_use]
    pub fn context_for(
        &self,
        max_chars: usize,
        messages: &[BaseMessage],
        tool_name: Option<&str>,
        tool_arguments: Option<&str>,
    ) -> RuleContext {
        let mut ctx = RuleContext::new((max_chars / CHARACTERS_PER_TOKEN).max(1));
        let leg = (max_chars / (CHARACTERS_PER_TOKEN * 2)).max(1);
        ctx.head_tokens = leg;
        ctx.tail_tokens = leg;
        ctx.query_terms = Self::query_terms(messages, tool_name, tool_arguments);
        ctx.tool_name = tool_name.map(str::to_owned);
        ctx.file_path = extract_file_path_argument(tool_arguments);
        ctx
    }

    /// Compresses the tool message at `index` in `messages`.
    ///
    /// The routing hints and query terms are resolved from the surrounding
    /// messages, so this is the entry point a processor uses. The dump this pass
    /// may write carries no session or context id; use
    /// [`RuleCompressionPipeline::compress_in_messages_with_scope`] when the
    /// caller knows them.
    ///
    /// An `index` the conversation does not own is [`Error::Validation`] naming
    /// the index and the message count, never a silent no-op.
    pub fn compress_in_messages(
        &self,
        messages: &[BaseMessage],
        index: usize,
        pass_name: &str,
        max_chars: usize,
        force: bool,
    ) -> Result<BaseMessage> {
        self.compress_in_messages_with_scope(
            messages,
            index,
            pass_name,
            max_chars,
            force,
            &OffloadScope::default(),
        )
    }

    /// Compresses the tool message at `index` in `messages` under `scope`.
    ///
    /// `scope` supplies the real session and context ids a dump records and the
    /// `{session_id}`/`{context_id}` templates it expands; when the scope carries
    /// neither, the dump falls back to [`super::dump::UNKNOWN_SESSION_ID`] and
    /// [`super::dump::UNKNOWN_CONTEXT_ID`] (a processor always has a scope, so the
    /// placeholders only appear for a standalone caller).
    ///
    /// An `index` the conversation does not own is [`Error::Validation`] naming
    /// the index and the message count: the pass never fabricates a message, so
    /// a caller cannot mistake an out-of-range index for a declined body.
    pub fn compress_in_messages_with_scope(
        &self,
        messages: &[BaseMessage],
        index: usize,
        pass_name: &str,
        max_chars: usize,
        force: bool,
        scope: &OffloadScope,
    ) -> Result<BaseMessage> {
        let Some(message) = messages.get(index) else {
            return Err(Error::validation(format!(
                "rule compression: message index {index} is out of range for a conversation \
                 of {} messages",
                messages.len()
            )));
        };
        let tool_call = resolve_tool_call(messages, index);
        let tool_name = tool_call.map(|call| call.name.as_str());
        let tool_arguments = tool_call.map(|call| call.arguments.as_str());
        let ctx = self.context_for(max_chars, messages, tool_name, tool_arguments);
        Ok(self.compress_message(
            message,
            Some(index),
            &ctx,
            pass_name,
            max_chars,
            force,
            scope,
        ))
    }

    /// Compresses `message` with `ctx`, or returns it unchanged.
    ///
    /// `max_chars` re-bases the token budget of the pass (`ctx`'s own
    /// `max_tokens`/`head_tokens`/`tail_tokens` are overridden), matching Python,
    /// where the caller passes only the character budget.
    pub fn compress(
        &self,
        message: &BaseMessage,
        ctx: &RuleContext,
        pass_name: &str,
        max_chars: usize,
        force: bool,
    ) -> Result<BaseMessage> {
        self.compress_with_scope(
            message,
            ctx,
            pass_name,
            max_chars,
            force,
            &OffloadScope::default(),
        )
    }

    /// Compresses `message` with `ctx` under `scope`, or returns it unchanged.
    pub fn compress_with_scope(
        &self,
        message: &BaseMessage,
        ctx: &RuleContext,
        pass_name: &str,
        max_chars: usize,
        force: bool,
        scope: &OffloadScope,
    ) -> Result<BaseMessage> {
        Ok(self.compress_message(message, None, ctx, pass_name, max_chars, force, scope))
    }

    /// The one compression body: `index` is the message's position when known.
    ///
    /// A tool message whose content is not text is returned unchanged, with one
    /// `tracing::debug!` line naming the message index (or `unknown` for a
    /// standalone [`RuleCompressionPipeline::compress_with_scope`] call), the
    /// content kind and the `non_text_content` reason, because "nothing to do"
    /// and "cannot do it" must not look identical.
    fn compress_message(
        &self,
        message: &BaseMessage,
        index: Option<usize>,
        ctx: &RuleContext,
        pass_name: &str,
        max_chars: usize,
        force: bool,
        scope: &OffloadScope,
    ) -> BaseMessage {
        if message.role != Role::Tool || !self.strategy.runs_rule_compression() {
            return message.clone();
        }
        let Some(text) = message.content.as_text() else {
            let content_kind = content_kind(&message.content);
            let label = index.map_or_else(|| "unknown".to_owned(), |value| value.to_string());
            tracing::debug!(
                reason = "non_text_content",
                content_kind,
                message_index = %label,
                details = %json!({
                    "reason": "non_text_content",
                    "content_kind": content_kind,
                    "message_index": index,
                }),
                "rule compression: message content is not text"
            );
            return message.clone();
        };
        let text = text.to_owned();
        if !force && text.chars().count() <= max_chars {
            return message.clone();
        }
        let mut budget = ctx.clone();
        budget.max_tokens = (max_chars / CHARACTERS_PER_TOKEN).max(1);
        let leg = (max_chars / (CHARACTERS_PER_TOKEN * 2)).max(1);
        budget.head_tokens = leg;
        budget.tail_tokens = leg;

        let result = self.router.compress(&text, &budget);
        if !result.modified {
            // The compressor's own `details` are the only record of why it
            // declined, and an unchanged message carries no metadata, so the
            // reason is logged here rather than dropped.
            if let Some(reason) = result
                .details
                .as_ref()
                .and_then(|details| details.get("reason"))
                .and_then(Value::as_str)
            {
                let details = result.details.clone().unwrap_or(Value::Null);
                tracing::debug!(
                    content_type = result.content_type.as_str(),
                    reason,
                    message_index = %index.map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                    details = %details,
                    "rule compression: compressor declined the body"
                );
            }
            return message.clone();
        }
        let tool_call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
        let mut compressed = message.clone();
        compressed.content = MessageContent::Text(result.content.clone());
        compressed.metadata.insert(
            RULE_COMPRESSED_AT_KEY.to_owned(),
            json!(self.current_time()),
        );
        compressed.metadata.insert(
            RULE_COMPRESSION_TYPE_KEY.to_owned(),
            json!(result.content_type.as_str()),
        );
        compressed
            .metadata
            .insert(RULE_COMPRESSION_PASS_KEY.to_owned(), json!(pass_name));
        compressed
            .metadata
            .insert(RULE_COMPRESSION_LOSSY_KEY.to_owned(), json!(result.lossy));
        if let Some(details) = result.details.as_ref().filter(|details| is_truthy(details)) {
            compressed
                .metadata
                .insert(RULE_COMPRESSION_DETAILS_KEY.to_owned(), details.clone());
        }
        if let Some(path) = self.dump(&text, &result, pass_name, tool_call_id, scope) {
            compressed.metadata.insert(
                RULE_COMPRESSION_DUMP_PATH_KEY.to_owned(),
                json!(path.to_string_lossy()),
            );
        }
        tracing::info!(
            "[RuleCompression] applied tool_call_id={} pass={} type={} lossy={} chars={}->{}",
            tool_call_id,
            pass_name,
            result.content_type.as_str(),
            result.lossy,
            text.chars().count(),
            result.content.chars().count(),
        );
        compressed
    }

    /// Query terms of the latest user message plus the triggering tool call.
    ///
    /// An empty user contribution is attributed with one `tracing::debug!` line
    /// naming which case it was (`no_user_message`, `non_text_user_message` or
    /// `empty_user_message`), so a relevance-free pass is explainable.
    fn query_terms(
        messages: &[BaseMessage],
        tool_name: Option<&str>,
        tool_arguments: Option<&str>,
    ) -> BTreeSet<String> {
        let Some((index, message)) = messages
            .iter()
            .enumerate()
            .rev()
            .find(|(_, message)| message.role == Role::User)
        else {
            tracing::debug!(
                reason = "no_user_message",
                messages = messages.len(),
                "rule compression: no query terms, the conversation has no user message"
            );
            return extract_query_terms("", tool_name, tool_arguments);
        };
        let Some(text) = message.content.as_text() else {
            tracing::debug!(
                reason = "non_text_user_message",
                content_kind = content_kind(&message.content),
                message_index = index,
                "rule compression: no query terms, the latest user message is not text"
            );
            return extract_query_terms("", tool_name, tool_arguments);
        };
        if text.trim().is_empty() {
            tracing::debug!(
                reason = "empty_user_message",
                message_index = index,
                "rule compression: the latest user message is empty"
            );
        }
        extract_query_terms(text, tool_name, tool_arguments)
    }

    /// Writes the dump of one application, returning its path on success.
    ///
    /// The record carries the scope's real ids, the resolved template directory,
    /// and the format this pipeline was configured with. A failure is reported
    /// once per directory (`warn` naming the path and the error) and never fails
    /// the pass.
    fn dump(
        &self,
        original: &str,
        result: &RuleCompressionResult,
        pass_name: &str,
        tool_call_id: &str,
        scope: &OffloadScope,
    ) -> Option<PathBuf> {
        if !self.enable_dump {
            return None;
        }
        let timestamp = self.current_time();
        let vars = DumpTemplateVars::from_scope(scope, timestamp);
        let configured = self
            .dump_dir
            .as_ref()
            .map(|dir| dir.to_string_lossy().into_owned());
        let dir = resolve_dump_dir(configured.as_deref(), &vars);
        let record = DumpRecord {
            timestamp,
            session_id: vars.session_or_unknown(),
            context_id: vars.context_or_unknown(),
            tool_call_id,
            content_type: result.content_type.as_str(),
            pass_name,
            lossy: result.lossy,
            original,
            compressed: &result.content,
            details: result.details.as_ref(),
        };
        match write_record(&record, &dir, self.dump_format) {
            Ok(path) => {
                tracing::info!(
                    "[RuleCompression] dumped compression payload path={}",
                    path.display()
                );
                Some(path)
            }
            Err(error) => {
                self.warn_dump_failure(&dir, &error);
                None
            }
        }
    }

    /// Warns about a dump failure once per directory.
    ///
    /// Repeating the same message on every application would drown the log, and
    /// guessing another location would hide the failure, so the first failure
    /// under a directory is reported with its path and error and later ones stay
    /// silent.
    fn warn_dump_failure(&self, dir: &Path, error: &Error) {
        let key = dir.to_string_lossy().into_owned();
        let mut last = self
            .warned_dump_dir
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if last.as_deref() == Some(key.as_str()) {
            return;
        }
        tracing::warn!(
            "[RuleCompression] cannot write compression dumps under {}: {error}",
            dir.display()
        );
        *last = Some(key);
    }
}

/// The kind of a message content, for attribution in a log line.
fn content_kind(content: &MessageContent) -> &'static str {
    match content {
        MessageContent::Text(_) => "text",
        MessageContent::Parts(parts) if parts.is_empty() => "empty_parts",
        MessageContent::Parts(_) => "parts",
    }
}

/// Whether a details payload counts as present (Python's dict truthiness).
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Seconds since the Unix epoch, or `0.0` when the clock is before it.
fn system_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::ToolCall;
    use std::fmt::Write as _;

    fn tool_message(text: &str, call_id: &str) -> BaseMessage {
        BaseMessage::tool(call_id, text)
    }

    fn conversation(user: &str, call: ToolCall, result: &str) -> Vec<BaseMessage> {
        let mut assistant = BaseMessage::assistant("");
        assistant.tool_calls = vec![call];
        vec![
            BaseMessage::user(user),
            assistant,
            tool_message(result, "call_1"),
        ]
    }

    #[test]
    fn capability_and_budget_follow_the_three_characters_rule() {
        assert_eq!(
            RuleCompressionPipeline::context_character_capacity(1_000),
            3_000
        );
        assert_eq!(RuleCompressionPipeline::context_character_capacity(0), 1);
        let pipeline = RuleCompressionPipeline::new();
        let ctx = pipeline.context_for(900, &[], None, Some("{\"file_path\": \"a.rs\"}"));
        assert_eq!(ctx.max_tokens, 300);
        assert_eq!(ctx.head_tokens, 150);
        assert_eq!(ctx.file_path.as_deref(), Some("a.rs"));
    }

    #[test]
    fn small_or_non_tool_messages_are_returned_unchanged() {
        let pipeline = RuleCompressionPipeline::new();
        let ctx = RuleContext::new(1_000);
        let small = tool_message("tiny", "call_1");
        let unchanged = pipeline.compress(&small, &ctx, ADD_PASS, 1_000, false);
        assert_eq!(
            unchanged.expect("ok").content.as_text(),
            Some("tiny"),
            "a message below the budget is not compressed"
        );
        let user = BaseMessage::user("x".repeat(10_000));
        let unchanged = pipeline
            .compress(&user, &ctx, ADD_PASS, 10, true)
            .expect("ok");
        assert_eq!(unchanged.content.as_text().map(str::len), Some(10_000));
    }

    #[test]
    fn an_oversized_tool_result_is_compressed_and_stamped() {
        let mut lines = Vec::new();
        for index in 0..400 {
            lines.push(format!("src/file{}.rs:{}: duplicate hit", index % 3, index));
        }
        let content = lines.join("\n");
        let call = ToolCall {
            id: Some("call_1".to_owned()),
            name: "grep".to_owned(),
            arguments: "{\"pattern\": \"duplicate\"}".to_owned(),
            ..ToolCall::default()
        };
        let messages = conversation("find duplicate hits", call, &content);
        let pipeline = RuleCompressionPipeline::with_clock(|| 7.5);
        let compressed = pipeline
            .compress_in_messages(&messages, 2, ADD_PASS, 1_000, false)
            .expect("ok");
        assert!(compressed.content.as_text().expect("text").len() < content.len());
        assert_eq!(
            compressed.metadata[RULE_COMPRESSION_TYPE_KEY],
            json!("SEARCH_RESULTS")
        );
        assert_eq!(
            compressed.metadata[RULE_COMPRESSION_PASS_KEY],
            json!(ADD_PASS)
        );
        assert_eq!(compressed.metadata[RULE_COMPRESSED_AT_KEY], json!(7.5));
        assert!(compressed
            .metadata
            .contains_key(RULE_COMPRESSION_DETAILS_KEY));
        assert_eq!(compressed.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn force_bypasses_the_size_gate() {
        let call = ToolCall {
            id: Some("call_1".to_owned()),
            name: "read_file".to_owned(),
            arguments: "{}".to_owned(),
            ..ToolCall::default()
        };
        let mut content = String::new();
        for index in 0..50 {
            let _ = writeln!(content, "duplicate line about failures error {}", index % 2);
        }
        let messages = conversation("look at it", call, &content);
        let pipeline = RuleCompressionPipeline::new();
        let skipped = pipeline
            .compress_in_messages(&messages, 2, TTL_PASS, 100_000, false)
            .expect("ok");
        assert!(skipped.metadata.is_empty());
        let forced = pipeline
            .compress_in_messages(&messages, 2, TTL_PASS, 100_000, true)
            .expect("ok");
        assert!(!forced.metadata.is_empty());
    }

    #[test]
    fn the_accepted_candidate_bench_stamps_the_lossy_flag() {
        let call = ToolCall {
            id: Some("call_1".to_owned()),
            name: "grep".to_owned(),
            arguments: "{\"pattern\": \"duplicate\"}".to_owned(),
            ..ToolCall::default()
        };
        let mut content = String::new();
        for index in 0..200 {
            let _ = writeln!(content, "the same progress line {}", index % 3);
        }
        let messages = conversation("progress", call, &content);
        let pipeline = RuleCompressionPipeline::with_clock(|| 7.5);
        let compressed = pipeline
            .compress_in_messages(&messages, 2, ADD_PASS, 500, false)
            .expect("ok");
        assert_eq!(compressed.metadata[RULE_COMPRESSION_LOSSY_KEY], json!(true));
    }

    #[test]
    fn truncate_only_makes_the_rule_tier_a_no_op() {
        let call = ToolCall {
            id: Some("call_1".to_owned()),
            name: "grep".to_owned(),
            arguments: "{}".to_owned(),
            ..ToolCall::default()
        };
        let mut content = String::new();
        for index in 0..200 {
            let _ = writeln!(content, "the same progress line {}", index % 3);
        }
        let messages = conversation("progress", call, &content);
        let pipeline =
            RuleCompressionPipeline::new().with_strategy(RuleCompressionStrategy::TruncateOnly);
        assert_eq!(pipeline.strategy(), RuleCompressionStrategy::TruncateOnly);
        let untouched = pipeline
            .compress_in_messages(&messages, 2, ADD_PASS, 500, false)
            .expect("ok");
        assert!(untouched.metadata.is_empty());
        assert_eq!(untouched.content.as_text(), messages[2].content.as_text());
    }

    #[test]
    fn a_failed_dump_is_remembered_per_directory() {
        let pipeline = RuleCompressionPipeline::new();
        let first = Path::new("/this/path/does/not/exist/for/the/test");
        let second = Path::new("/another/missing/directory");
        let error = Error::sys_op("create dump dir: boom");
        pipeline.warn_dump_failure(first, &error);
        // Same directory again: the warning is not repeated.
        pipeline.warn_dump_failure(first, &error);
        assert_eq!(
            warned_dir(&pipeline).as_deref(),
            Some(first.to_string_lossy().as_ref())
        );
        // A different directory is reported again.
        pipeline.warn_dump_failure(second, &error);
        assert_eq!(
            warned_dir(&pipeline).as_deref(),
            Some(second.to_string_lossy().as_ref())
        );
    }

    /// The directory the last dump failure was reported for.
    fn warned_dir(pipeline: &RuleCompressionPipeline) -> Option<String> {
        pipeline
            .warned_dump_dir
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    #[test]
    fn dumps_are_written_only_when_enabled() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut content = String::new();
        for index in 0..200 {
            let _ = writeln!(content, "src/a.rs:{index}: repeated error");
        }
        let call = ToolCall {
            id: Some("call_9".to_owned()),
            name: "grep".to_owned(),
            arguments: "{}".to_owned(),
            ..ToolCall::default()
        };
        let messages = conversation("error", call, &content);
        let pipeline = RuleCompressionPipeline::new()
            .with_dump(true, Some(dir.path().to_string_lossy().into_owned()));
        let compressed = pipeline
            .compress_in_messages(&messages, 2, ADD_PASS, 500, false)
            .expect("ok");
        let path = compressed.metadata[RULE_COMPRESSION_DUMP_PATH_KEY]
            .as_str()
            .expect("dump path")
            .to_owned();
        assert!(PathBuf::from(&path).exists());
        let disabled = RuleCompressionPipeline::new();
        let plain = disabled
            .compress_in_messages(&messages, 2, ADD_PASS, 500, false)
            .expect("ok");
        assert!(!plain.metadata.contains_key(RULE_COMPRESSION_DUMP_PATH_KEY));
    }
}
