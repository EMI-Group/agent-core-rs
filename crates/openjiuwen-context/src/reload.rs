//! The harness "reload" prompt section.
//!
//! Ports the harness hint that explains how to recover content hidden behind an
//! `[[OFFLOAD: …]]` marker. The hint is injected as a system message only when
//! the history actually carries a marker, and injection is idempotent so a
//! context can call it before every request:
//!
//! ```
//! use openjiuwen_foundation::prompt::BaseMessage;
//! use openjiuwen_context::reload::{inject_reload_hint, ReloadLanguage};
//!
//! let mut messages = vec![
//!     BaseMessage::system("persona"),
//!     BaseMessage::tool("call_1", "[[OFFLOAD: handle=x, type=in_memory]]"),
//! ];
//! assert!(inject_reload_hint(&mut messages, ReloadLanguage::En));
//! assert_eq!(messages.len(), 3);
//! assert!(!inject_reload_hint(&mut messages, ReloadLanguage::En));
//! ```
use openjiuwen_foundation::prompt::{BaseMessage, Role};

use crate::offload::OFFLOAD_MARKER_PREFIX;

/// Language of the reload hint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReloadLanguage {
    /// English hint ([`RELOAD_HINT_EN`]).
    #[default]
    En,
    /// Chinese hint ([`RELOAD_HINT_CN`]).
    Cn,
}

/// The English reload hint.
pub const RELOAD_HINT_EN: &str = "\
# Context Compression and Recovery

When the context becomes too long, it may be compressed and marked with:

```text
[[OFFLOAD: handle=<id>, type=<type>, path=<path>]]
```

When hidden original content is needed:

- If the marker contains `path`, use its exact value as `read_file.file_path`.
- `handle` identifies content and is not a file path; do not use it as `file_path`.
- If the marker has no `path`, do not guess or construct one; explain that `read_file` cannot recover it precisely.
- When only specific content is needed, search for relevant portions instead of reading the entire file.

`filesystem` means the content is persisted at `path`; `in_memory` means it exists only in the session cache and cannot be recovered with `read_file` when no `path` is present.";

/// The Chinese reload hint.
pub const RELOAD_HINT_CN: &str = "\
# 上下文压缩与恢复

上下文过长时会被自动压缩，并使用以下 marker 标记：

```text
[[OFFLOAD: handle=<id>, type=<type>, path=<path>]]
```

需要被隐藏的原始内容时：

- marker 包含 `path` 时，使用其精确值作为 `read_file.file_path`。
- `handle` 是内容标识，不是文件路径；不要把它当成 `file_path`。
- marker 没有 `path` 时，不要猜测或拼接路径，应说明无法通过 `read_file` 精确恢复。
- 只需定位特定内容时，优先搜索相关片段，不要盲目读取整个文件。

`filesystem` 表示内容已经持久化到 `path`；`in_memory` 表示内容只在会话缓存中，没有 `path` 时无法通过 `read_file` 恢复。";

/// Returns the reload hint for `language`.
#[must_use]
pub fn reload_hint(language: ReloadLanguage) -> &'static str {
    match language {
        ReloadLanguage::En => RELOAD_HINT_EN,
        ReloadLanguage::Cn => RELOAD_HINT_CN,
    }
}

/// Whether any message text carries an offload marker.
#[must_use]
pub fn has_offload_markers(messages: &[BaseMessage]) -> bool {
    messages
        .iter()
        .any(|message| message.text().contains(OFFLOAD_MARKER_PREFIX))
}

/// The reload hint as a system message, only when a marker is present.
#[must_use]
pub fn reload_system_message(
    messages: &[BaseMessage],
    language: ReloadLanguage,
) -> Option<BaseMessage> {
    has_offload_markers(messages).then(|| BaseMessage::system(reload_hint(language)))
}

/// Inserts the reload hint after the leading system messages when needed.
///
/// The hint is inserted only when the history carries an offload marker and no
/// reload section is present yet (detected by the known hint headlines, so a
/// later call with the other language does not duplicate the section). Returns
/// whether a message was inserted.
pub fn inject_reload_hint(messages: &mut Vec<BaseMessage>, language: ReloadLanguage) -> bool {
    if !has_offload_markers(messages) {
        return false;
    }
    if messages.iter().any(|message| {
        let text = message.text();
        reload_headlines().any(|headline| text.starts_with(headline))
    }) {
        return false;
    }
    let position = messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    messages.insert(position, BaseMessage::system(reload_hint(language)));
    true
}

/// The first line of each shipped reload hint.
fn reload_headlines() -> impl Iterator<Item = &'static str> {
    [RELOAD_HINT_EN, RELOAD_HINT_CN]
        .into_iter()
        .filter_map(|hint| hint.lines().next())
}
