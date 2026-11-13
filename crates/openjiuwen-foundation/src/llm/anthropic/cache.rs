//! Anthropic prompt-caching anchors.
//!
//! The Messages API caches a request prefix at explicit
//! `cache_control: {"type": "ephemeral"}` breakpoints; only the block carrying
//! the marker and everything before it is cached. This module owns the three
//! anchors the client always writes (there is no beta header and no request
//! flag to enable them):
//!
//! 1. the last entry of the top-level `tools` array;
//! 2. the last cacheable block of the hoisted `system` block array;
//! 3. the last cacheable block of the last *stable* message, walking back past
//!    a transient tail message (a per-call runtime reminder must stay outside
//!    the cached prefix, otherwise every turn misses the cache) and past
//!    messages that carry no block array of their own.
//!
//! Cacheable block types are `text`, `image`, `document`, `tool_use` and
//! `tool_result`. Two per-request overrides are honoured through
//! [`ModelRequestConfig::extra`]: `disable_cache_control: true` skips all three
//! anchors, and `cache_control_ttl: "1h"` writes the long ephemeral lifetime
//! (`{"type": "ephemeral", "ttl": "1h"}`); the default marker carries no `ttl`
//! key.
//!
//! [`ModelRequestConfig::extra`]: crate::llm::config::ModelRequestConfig::extra

use serde_json::{json, Value};

use crate::prompt::BaseMessage;

/// Block types that can carry a cache breakpoint.
const CACHEABLE_BLOCK_TYPES: [&str; 5] = ["text", "image", "document", "tool_use", "tool_result"];

/// `extra` key that turns every cache anchor off.
const DISABLE_KEY: &str = "disable_cache_control";

/// `extra` key that selects the ephemeral cache lifetime.
const TTL_KEY: &str = "cache_control_ttl";

/// Cacheable lifetime longer than the 5-minute default.
const LONG_TTL: &str = "1h";

/// Cache-breakpoint settings derived from a request's `extra` map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheControl {
    /// Whether the three anchors are written at all.
    enabled: bool,
    /// Ephemeral lifetime to put on the marker, when the caller set one.
    ttl: Option<String>,
}

impl CacheControl {
    /// Reads the opt-out/ttl overrides out of a request's `extra` value.
    ///
    /// Only `disable_cache_control == true` disables the anchors; any other
    /// value (including a missing key) keeps them on. `cache_control_ttl` is
    /// taken as a string and trimmed; an unrecognized lifetime falls back to
    /// the default 5-minute marker rather than being forwarded to the API.
    pub(crate) fn from_extra(extra: &Value) -> Self {
        let enabled = !matches!(extra.get(DISABLE_KEY), Some(Value::Bool(true)));
        let ttl = extra
            .get(TTL_KEY)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|ttl| !ttl.is_empty())
            .map(str::to_owned);
        Self { enabled, ttl }
    }

    /// Whether cache anchors should be written.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The `cache_control` marker to attach to an anchored block.
    pub(crate) fn marker(&self) -> Value {
        match self.ttl.as_deref() {
            Some(LONG_TTL) => json!({"type": "ephemeral", "ttl": LONG_TTL}),
            _ => json!({"type": "ephemeral"}),
        }
    }
}

/// Reports whether the final *input* message is flagged `metadata["transient"]`.
///
/// Transient messages are stripped after the call, so anchoring on them would
/// move the breakpoint on every turn; the caller uses this to anchor one
/// message earlier instead.
pub(crate) fn last_input_is_transient(messages: &[BaseMessage]) -> bool {
    messages
        .last()
        .is_some_and(|message| matches!(message.metadata.get("transient"), Some(Value::Bool(true))))
}

/// Whether an `extra` key is a client-side cache-control knob.
///
/// These keys tune the anchors and have no wire representation, so the body
/// builder consumes them instead of forwarding them to the API.
pub(crate) fn is_control_key(key: &str) -> bool {
    key == DISABLE_KEY || key == TTL_KEY
}

/// Attaches `marker` to the last cacheable block of `blocks`.
///
/// Returns `false` when the slice holds no cacheable block, so the caller can
/// keep walking back through the conversation.
pub(crate) fn mark_last_cacheable_block(blocks: &mut [Value], marker: &Value) -> bool {
    for block in blocks.iter_mut().rev() {
        if is_cacheable(block) {
            if let Value::Object(fields) = block {
                fields.insert("cache_control".to_owned(), marker.clone());
            }
            return true;
        }
    }
    false
}

/// Anchors the last converted tool definition, when caching is enabled.
pub(crate) fn apply_tool_anchor(tools: &mut Value, cache: &CacheControl) {
    if !cache.is_enabled() {
        return;
    }
    if let Value::Array(items) = tools {
        if let Some(Value::Object(last)) = items.last_mut() {
            last.insert("cache_control".to_owned(), cache.marker());
        }
    }
}

/// Anchors the last cacheable block of the hoisted system block array.
pub(crate) fn apply_system_anchor(system: &mut [Value], cache: &CacheControl) {
    if !cache.is_enabled() {
        return;
    }
    mark_last_cacheable_block(system, &cache.marker());
}

/// Anchors the conversation cache prefix on the last stable message.
///
/// `exclude_tail` skips a trailing transient message (when another message
/// precedes it). A mid-conversation `system` turn carries plain text rather
/// than blocks, so the walk continues backwards to the nearest message that
/// does have a block array instead of dropping the breakpoint. A message whose
/// `content` is a plain, non-empty string is materialized as a one-element
/// `text` block array so the marker has somewhere to live; an empty string is
/// not an anchor and the walk continues.
pub(crate) fn apply_messages_anchor(
    messages: &mut [Value],
    exclude_tail: bool,
    cache: &CacheControl,
) {
    if !cache.is_enabled() || messages.is_empty() {
        return;
    }
    let marker = cache.marker();
    let mut index = messages.len() - 1;
    if exclude_tail && index >= 1 {
        index -= 1;
    }

    loop {
        let is_inline_system = message_role(&messages[index]) == Some("system");
        if !is_inline_system {
            if let Some(Value::Array(blocks)) = messages[index].get_mut("content") {
                if mark_last_cacheable_block(blocks, &marker) {
                    return;
                }
            }
            let plain = messages[index]
                .get("content")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_owned);
            if let Some(text) = plain {
                messages[index]["content"] =
                    json!([{"type": "text", "text": text, "cache_control": marker}]);
                return;
            }
        }
        if index == 0 {
            return;
        }
        index -= 1;
    }
}

/// Whether a converted block type can carry a cache breakpoint.
fn is_cacheable(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| CACHEABLE_BLOCK_TYPES.contains(&kind))
}

/// The `role` of a converted message, when it has one.
fn message_role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> CacheControl {
        CacheControl::from_extra(&Value::Null)
    }

    #[test]
    fn the_default_marker_has_no_ttl_and_the_opt_in_long_one_does() {
        let mut blocks = vec![json!({"type": "text", "text": "x"})];
        apply_system_anchor(&mut blocks, &cache());
        assert_eq!(blocks[0]["cache_control"], json!({"type": "ephemeral"}));

        let long = CacheControl::from_extra(&json!({"cache_control_ttl": "1h"}));
        let mut blocks = vec![json!({"type": "text", "text": "x"})];
        apply_system_anchor(&mut blocks, &long);
        assert_eq!(
            blocks[0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[test]
    fn an_unknown_ttl_falls_back_to_the_default_marker() {
        let cache = CacheControl::from_extra(&json!({"cache_control_ttl": "2h"}));
        assert_eq!(cache.marker(), json!({"type": "ephemeral"}));
        let blank = CacheControl::from_extra(&json!({"cache_control_ttl": "  "}));
        assert_eq!(blank.marker(), json!({"type": "ephemeral"}));
    }

    #[test]
    fn only_a_true_flag_disables_the_anchors() {
        assert!(CacheControl::from_extra(&json!({"disable_cache_control": false})).is_enabled());
        assert!(!CacheControl::from_extra(&json!({"disable_cache_control": true})).is_enabled());
        assert!(CacheControl::from_extra(&json!({"disable_cache_control": "true"})).is_enabled());
        assert!(cache().is_enabled());
    }

    #[test]
    fn the_marker_lands_on_the_last_cacheable_block_only() {
        let mut blocks = vec![
            json!({"type": "text", "text": "a"}),
            json!({"type": "audio", "audio": {"data": "x"}}),
            json!({"type": "tool_use", "id": "call_1"}),
        ];
        assert!(mark_last_cacheable_block(
            &mut blocks,
            &json!({"type": "ephemeral"})
        ));
        assert!(blocks[0].get("cache_control").is_none());
        assert!(blocks[1].get("cache_control").is_none());
        assert_eq!(
            blocks[2]["cache_control"],
            json!({"type": "ephemeral"}),
            "the uncacheable audio block must be skipped"
        );
    }

    #[test]
    fn a_block_list_without_cacheable_types_reports_no_anchor() {
        let mut blocks = vec![json!({"type": "audio", "audio": {"data": "x"}})];
        assert!(!mark_last_cacheable_block(
            &mut blocks,
            &json!({"type": "ephemeral"})
        ));
    }

    #[test]
    fn the_tool_anchor_lands_on_the_last_definition() {
        let mut tools = json!([
            {"name": "a", "input_schema": {}},
            {"name": "b", "input_schema": {}},
        ]);
        apply_tool_anchor(&mut tools, &cache());
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn a_disabled_cache_writes_no_marker_anywhere() {
        let disabled = CacheControl::from_extra(&json!({"disable_cache_control": true}));
        let mut tools = json!([{"name": "a"}]);
        let mut system = vec![json!({"type": "text", "text": "s"})];
        let mut messages =
            vec![json!({"role": "user", "content": [{"type": "text", "text": "u"}]})];
        apply_tool_anchor(&mut tools, &disabled);
        apply_system_anchor(&mut system, &disabled);
        apply_messages_anchor(&mut messages, false, &disabled);
        assert!(tools[0].get("cache_control").is_none());
        assert!(system[0].get("cache_control").is_none());
        assert!(messages[0]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn a_transient_tail_is_walked_past() {
        let mut messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "stable"}]}),
            json!({"role": "user", "content": [{"type": "text", "text": "transient"}]}),
        ];
        apply_messages_anchor(&mut messages, true, &cache());
        assert_eq!(
            messages[0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(messages[1]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn a_single_transient_message_still_anchors_itself() {
        let mut messages =
            vec![json!({"role": "user", "content": [{"type": "text", "text": "only"}]})];
        apply_messages_anchor(&mut messages, true, &cache());
        assert_eq!(
            messages[0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn an_inline_system_message_is_not_an_anchor() {
        let mut messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "stable"}]}),
            json!({"role": "system", "content": "instruction"}),
        ];
        apply_messages_anchor(&mut messages, false, &cache());
        assert_eq!(
            messages[0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(messages[1]["content"], json!("instruction"));
    }

    #[test]
    fn a_plain_string_anchor_becomes_a_one_block_array() {
        let mut messages = vec![json!({"role": "assistant", "content": "done"})];
        apply_messages_anchor(&mut messages, false, &cache());
        assert_eq!(
            messages[0]["content"],
            json!([{
                "type": "text",
                "text": "done",
                "cache_control": {"type": "ephemeral"},
            }])
        );
    }

    #[test]
    fn an_empty_string_anchor_is_walked_past() {
        let mut messages = vec![
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "call_1"}]}),
            json!({"role": "user", "content": ""}),
        ];
        apply_messages_anchor(&mut messages, false, &cache());
        assert_eq!(
            messages[0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(messages[1]["content"], json!(""));
    }

    #[test]
    fn only_the_two_cache_knobs_are_consumed_by_the_body_builder() {
        assert!(is_control_key("disable_cache_control"));
        assert!(is_control_key("cache_control_ttl"));
        assert!(!is_control_key("cache_control"));
        assert!(!is_control_key("custom_flag"));
    }

    #[test]
    fn the_transient_flag_is_read_from_the_original_messages() {
        let mut message = BaseMessage::user("hi");
        assert!(!last_input_is_transient(std::slice::from_ref(&message)));
        message.metadata.insert("transient".to_owned(), json!(true));
        assert!(last_input_is_transient(std::slice::from_ref(&message)));
        message
            .metadata
            .insert("transient".to_owned(), json!("yes"));
        assert!(!last_input_is_transient(std::slice::from_ref(&message)));
        assert!(!last_input_is_transient(&[]));
    }
}
