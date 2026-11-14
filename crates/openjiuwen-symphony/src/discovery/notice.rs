//! The incremental skill-notice cursor.
//!
//! Ports Python `openjiuwen.symphony.discovery.toolkit`'s session notice state
//! (`initialize_incremental_skill_notice_state`, `_NoticeState`,
//! `consume_incremental_skill_reminder`) onto a plain, persistable Rust value.
//!
//! Python keeps one process-global state per `(session_scope, tool_name)` and only
//! announces *new* skills. [`SkillNoticeState`] keeps the same per-skill baselines but
//! is owned by the caller, so it can be stored with a session instead of a global map;
//! `consume` additionally reports removed and changed skills.
//!
//! The reminder text is exactly Python's `_render_reminder`: a `<system-reminder>`
//! block listing the added skills as `- <name>: <description>` (sorted by
//! `name.casefold()`), falling back to the one-line "`N` new skills are available"
//! form when the block exceeds `max_chars`. Because Python's reminder only describes
//! added skills, `reminder` is `None` when nothing was added even if skills were
//! removed or changed; those deltas stay visible in the notice fields.
//!
//! ```
//! # use openjiuwen_symphony::discovery::{SkillInventory, SkillNoticeState};
//! # async fn example(inventory: &SkillInventory) {
//! let mut state = SkillNoticeState::new(inventory);
//! let notice = state.consume(inventory).await;
//! assert!(notice.added.is_empty());
//! assert!(notice.reminder.is_none());
//! # }
//! ```

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::discovery::models::{inventory_fingerprint, SkillInventory, SkillRecord};

/// The discovery tool name Python uses in the compact reminder fallback.
pub const SKILL_INDEX_TOOL_NAME: &str = "skill_index";

/// The default character budget of one reminder (`consume_incremental_skill_reminder`).
pub const DEFAULT_NOTICE_MAX_CHARS: usize = 4_000;

/// The maximum description length Python's `textwrap.shorten` keeps in a reminder row.
pub const NOTICE_DESCRIPTION_CHARS: usize = 300;

/// The per-session cursor used to announce newly installed skills.
///
/// `seen` maps each worker id to the [`inventory_fingerprint`] of its record at the
/// last consumed inventory. Everything is plain data: serialize the state to persist
/// it across processes, then call [`SkillNoticeState::consume`] with the next
/// inventory. `max_chars` and `discovery_tool_name` control the rendered reminder and
/// default to Python's `4_000` and `"skill_index"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillNoticeState {
    /// The last-seen fingerprint of every worker id.
    pub seen: BTreeMap<String, String>,
    /// The character budget of the rendered reminder.
    pub max_chars: usize,
    /// The discovery tool name used by the compact reminder fallback.
    pub discovery_tool_name: String,
}

impl Default for SkillNoticeState {
    fn default() -> Self {
        Self {
            seen: BTreeMap::new(),
            max_chars: DEFAULT_NOTICE_MAX_CHARS,
            discovery_tool_name: SKILL_INDEX_TOOL_NAME.to_string(),
        }
    }
}

impl SkillNoticeState {
    /// Seeds a cursor with `inventory` as the baseline, consuming no delta.
    ///
    /// This is Python's `initialize_incremental_skill_notice_state`: a later
    /// [`consume`](Self::consume) reports only skills that were not in this inventory.
    #[must_use]
    pub fn new(inventory: &SkillInventory) -> Self {
        Self {
            seen: snapshot(inventory),
            ..Self::default()
        }
    }

    /// Sets the character budget of the rendered reminder.
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Sets the discovery tool name used by the compact reminder fallback.
    #[must_use]
    pub fn with_discovery_tool_name(mut self, tool_name: impl Into<String>) -> Self {
        self.discovery_tool_name = tool_name.into();
        self
    }

    /// Consumes the delta between the last-seen inventory and `inventory`.
    ///
    /// `added`, `removed`, and `changed` are sorted by `(worker_id.casefold(),
    /// worker_id)`. The cursor is updated to `inventory`, so consuming the same
    /// inventory twice is a no-op. `reminder` follows Python's `_render_reminder` and
    /// is `Some` only when at least one skill was added; it is `None` when the
    /// catalog is unchanged.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    #[must_use]
    pub async fn consume(&mut self, inventory: &SkillInventory) -> SkillNotice {
        let current = snapshot(inventory);
        let mut added: Vec<String> = current
            .keys()
            .filter(|worker_id| !self.seen.contains_key(*worker_id))
            .cloned()
            .collect();
        let mut changed: Vec<String> = current
            .iter()
            .filter(|(worker_id, fingerprint)| {
                self.seen
                    .get(*worker_id)
                    .is_some_and(|previous| previous != *fingerprint)
            })
            .map(|(worker_id, _)| worker_id.clone())
            .collect();
        let mut removed: Vec<String> = self
            .seen
            .keys()
            .filter(|worker_id| !current.contains_key(*worker_id))
            .cloned()
            .collect();
        sort_notice_names(&mut added);
        sort_notice_names(&mut removed);
        sort_notice_names(&mut changed);

        let reminder =
            render_reminder(&added, inventory, self.max_chars, &self.discovery_tool_name);
        self.seen = current;
        SkillNotice {
            added,
            removed,
            changed,
            reminder,
        }
    }
}

/// The delta reported by one [`SkillNoticeState::consume`] call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillNotice {
    /// Worker ids present now but absent from the baseline, sorted by name.
    pub added: Vec<String>,
    /// Worker ids absent now but present in the baseline, sorted by name.
    pub removed: Vec<String>,
    /// Worker ids whose per-skill fingerprint changed, sorted by name.
    pub changed: Vec<String>,
    /// Python's `<system-reminder>` block for the added skills; `None` when none were
    /// added.
    pub reminder: Option<String>,
}

/// The per-record fingerprint map of one inventory.
fn snapshot(inventory: &SkillInventory) -> BTreeMap<String, String> {
    inventory
        .items
        .iter()
        .map(|record| (record.worker_id.clone(), record_fingerprint(record)))
        .collect()
}

/// The content-addressed fingerprint of one record.
fn record_fingerprint(record: &SkillRecord) -> String {
    inventory_fingerprint(std::slice::from_ref(record))
}

/// Sorts notice names by Python's `(name.casefold(), name)` key.
fn sort_notice_names(names: &mut [String]) {
    names.sort_by_cached_key(|name| (name.to_lowercase(), name.clone()));
}

/// Renders Python's `_render_reminder` for the added worker ids.
fn render_reminder(
    added: &[String],
    inventory: &SkillInventory,
    max_chars: usize,
    tool_name: &str,
) -> Option<String> {
    if added.is_empty() {
        return None;
    }
    let mut rows: Vec<(String, String)> = added
        .iter()
        .map(|worker_id| reminder_row(worker_id, inventory))
        .collect();
    rows.sort_by_cached_key(|(name, _)| (name.to_lowercase(), name.clone()));
    let rendered_rows = rows
        .iter()
        .map(|(name, description)| format!("- {name}: {description}"))
        .collect::<Vec<_>>()
        .join("\n");
    let full = format!(
        "<system-reminder>\n\
         The following skills are newly available for use with the Skill tool:\n\n\
         {rendered_rows}\n\
         </system-reminder>"
    );
    if max_chars > 0 && full.chars().count() <= max_chars {
        return Some(full);
    }
    Some(format!(
        "<system-reminder>\n\
         {} new skills are available. Use `{tool_name}` to search the current catalog.\n\
         </system-reminder>",
        added.len()
    ))
}

/// Builds one `(name, description)` reminder row exactly as Python's card
/// normalization does.
fn reminder_row(worker_id: &str, inventory: &SkillInventory) -> (String, String) {
    let name = safe_text(worker_id);
    let raw_description = inventory.get(worker_id).map_or(worker_id, |record| {
        if !record.description.trim().is_empty() {
            record.description.as_str()
        } else if !record.name.trim().is_empty() {
            record.name.as_str()
        } else {
            record.worker_id.as_str()
        }
    });
    let description = shorten_description(&safe_text(raw_description), NOTICE_DESCRIPTION_CHARS);
    (name, description)
}

/// Python's `_safe_text`: collapse whitespace and escape angle brackets.
fn safe_text(value: &str) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.replace('<', "&lt;").replace('>', "&gt;")
}

/// Python's `textwrap.shorten(text, width, placeholder="...")`.
fn shorten_description(text: &str, width: usize) -> String {
    let placeholder = "...";
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width <= placeholder.chars().count() {
        return placeholder.chars().take(width).collect();
    }
    let available = width - placeholder.chars().count();
    let mut result = String::new();
    let mut used = 0usize;
    for word in text.split_whitespace() {
        let word_chars = word.chars().count();
        let additional = if result.is_empty() {
            word_chars
        } else {
            1 + word_chars
        };
        if used + additional > available {
            break;
        }
        if !result.is_empty() {
            result.push(' ');
            used += 1;
        }
        result.push_str(word);
        used += word_chars;
    }
    result.push_str(placeholder);
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn inventory(records: Vec<SkillRecord>) -> SkillInventory {
        SkillInventory::new(records)
    }

    #[test]
    fn reminder_pins_pythons_exact_wording() {
        let inventory = inventory(vec![
            SkillRecord::new("beta", "Beta").with_description("Second <skill>"),
            SkillRecord::new("alpha", "Alpha").with_description("  Adds   numbers "),
        ]);
        let reminder = render_reminder(
            &["beta".to_string(), "alpha".to_string()],
            &inventory,
            4_000,
            SKILL_INDEX_TOOL_NAME,
        )
        .unwrap();
        assert_eq!(
            reminder,
            "<system-reminder>\n\
             The following skills are newly available for use with the Skill tool:\n\n\
             - alpha: Adds numbers\n\
             - beta: Second &lt;skill&gt;\n\
             </system-reminder>"
        );
    }

    #[test]
    fn reminder_falls_back_when_the_budget_is_tight() {
        let inventory = inventory(vec![
            SkillRecord::new("alpha", "Alpha").with_description("Adds")
        ]);
        let reminder = render_reminder(
            &["alpha".to_string()],
            &inventory,
            40,
            SKILL_INDEX_TOOL_NAME,
        )
        .unwrap();
        assert_eq!(
            reminder,
            "<system-reminder>\n\
             1 new skills are available. Use `skill_index` to search the current catalog.\n\
             </system-reminder>"
        );
        assert!(render_reminder(&[], &inventory, 40, SKILL_INDEX_TOOL_NAME).is_none());
    }

    #[test]
    fn descriptions_shorten_like_textwrap() {
        assert_eq!(shorten_description("one two", 300), "one two");
        assert_eq!(
            shorten_description("alpha beta gamma delta", 15),
            "alpha beta..."
        );
        assert_eq!(shorten_description("alpha", 3), "...");
    }

    #[test]
    fn state_round_trips_and_updates() {
        let baseline = inventory(vec![SkillRecord::new("alpha", "Alpha")]);
        let state = SkillNoticeState::new(&baseline);
        let decoded: SkillNoticeState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(decoded, state);
        assert_eq!(decoded.max_chars, DEFAULT_NOTICE_MAX_CHARS);
        assert_eq!(decoded.discovery_tool_name, SKILL_INDEX_TOOL_NAME);
    }
}
