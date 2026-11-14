//! Typed event model of the workspace watcher.
//!
//! A [`WatchEvent`] is the pure data record the debouncer emits: which kind of
//! harness root changed, at which path, in which way, and under which stable
//! target label. It carries no behaviour (`EffectKind` is deliberately not
//! introduced) so it can be logged, serialized and replayed without touching
//! the filesystem.
//!
//! [`WatchUpdate`] is the envelope a `Subscription` publishes: an event, a
//! failed reload handler, a notice that the watcher had to drop notifications
//! because one of its bounded queues was full, or a runtime failure of the OS
//! watcher itself.
//!
//! Wire form is `snake_case` throughout and unknown fields are ignored, so a
//! record written by a newer release still deserializes.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The kind of harness root a watched path belongs to.
///
/// Python parity: none — the watcher is a Rust-native addition, so the names
/// are chosen to match the harness directories they cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WatchTargetKind {
    /// A skill library root (`<workspace>/skills`); wire name `skills`.
    Skills,
    /// An agent-template root (`<root>/.openjiuwen/agents` or a plugin root);
    /// wire name `spec`.
    Spec,
    /// An instruction document such as `IDENTITY.md` or `OPENJIUWEN.md`; wire
    /// name `instruction_file`.
    InstructionFile,
    /// An MCP server configuration file; wire name `mcp_config`.
    McpConfig,
    /// Any path a caller watches for its own reasons; wire name `other`.
    Other,
}

impl WatchTargetKind {
    /// Returns the serde wire name of the kind (`"skills"`, `"spec"`,
    /// `"instruction_file"`, `"mcp_config"` or `"other"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Skills => "skills",
            Self::Spec => "spec",
            Self::InstructionFile => "instruction_file",
            Self::McpConfig => "mcp_config",
            Self::Other => "other",
        }
    }
}

/// How a watched path changed inside one debounce window.
///
/// Several raw notifications for one path collapse into one event through
/// [`ChangeKind::merge`], which implements this table
/// (`existing \ incoming`):
///
/// | existing \ incoming | Created | Modified | Removed | Renamed |
/// | --- | --- | --- | --- | --- |
/// | Created | Created | Created | Removed | Renamed |
/// | Modified | Modified | Modified | Removed | Renamed |
/// | Removed | Modified | Removed | Removed | Renamed |
/// | Renamed | Renamed | Renamed | Removed | Renamed |
///
/// `Removed + Created` nets to `Modified` (an in-place rewrite), while
/// `Created + Removed` nets to `Removed` (a transient file).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// The path appeared; wire name `created`.
    Created,
    /// The path content or metadata changed; wire name `modified`.
    Modified,
    /// The path disappeared; wire name `removed`.
    Removed,
    /// The path moved; wire name `renamed`.
    Renamed,
}

impl ChangeKind {
    /// Returns the serde wire name of the change (`"created"`, `"modified"`,
    /// `"removed"` or `"renamed"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Modified => "modified",
            Self::Removed => "removed",
            Self::Renamed => "renamed",
        }
    }

    /// Coalesces an earlier change kind with a later one for the same path.
    ///
    /// Implements the coalescing table in this type's documentation; see also
    /// [`Debouncer::push`](crate::watcher::Debouncer::push), the only caller.
    #[must_use]
    pub fn merge(self, incoming: Self) -> Self {
        match incoming {
            Self::Created => match self {
                Self::Removed => Self::Modified,
                other => other,
            },
            Self::Modified => self,
            Self::Removed => Self::Removed,
            Self::Renamed => Self::Renamed,
        }
    }
}

/// One debounced, coalesced change of a watched path.
///
/// `sequence` is monotonic per watcher and starts at `1`; `previous_path` is
/// set only for [`ChangeKind::Renamed`]; `label` names the concrete target the
/// path belongs to (`"skills:workspace"`, `"mcp:home"`, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchEvent {
    /// Monotonic sequence number assigned by the debouncer, starting at `1`.
    pub sequence: u64,
    /// The kind of harness root the path belongs to.
    pub kind: WatchTargetKind,
    /// The path that changed.
    pub path: PathBuf,
    /// How the path changed.
    pub change: ChangeKind,
    /// The original path of a rename; `None` for every other change kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_path: Option<PathBuf>,
    /// Stable label of the target the path matched, when it matched one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// A reload handler that failed for one event.
///
/// `sequence` is the sequence number of the triggering [`WatchEvent`] and
/// `error_kind` is the wire name of the handler's error kind
/// ([`ErrorKind::as_str`](openjiuwen_common::error::ErrorKind::as_str)).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReloadFailure {
    /// Sequence number of the event whose reload failed.
    pub sequence: u64,
    /// Name of the handler that failed.
    pub handler: String,
    /// The kind of harness root the event belonged to.
    pub kind: WatchTargetKind,
    /// The path that triggered the reload.
    pub path: PathBuf,
    /// Wire name of the error kind.
    pub error_kind: String,
    /// Human-readable error message.
    pub message: String,
}

/// A runtime failure of the underlying OS watcher.
///
/// `error_kind` is the wire name of the harness error kind derived from the
/// backend error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchFailure {
    /// Wire name of the error kind.
    pub error_kind: String,
    /// Human-readable error message.
    pub message: String,
}

/// The stage of the watcher pipeline that had to drop notifications.
///
/// Both stages use bounded queues, so a producer that outruns a consumer loses
/// notifications instead of growing without limit; the loss is never silent and
/// is reported as a [`WatchUpdate::Dropped`] notice plus a counter on
/// [`WatcherHandle`](crate::watcher::WatcherHandle).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropStage {
    /// The bounded queue between the OS callback and the debounce loop; wire
    /// name `intake`.
    Intake,
    /// The bounded queue between the debounce loop and the reload dispatcher;
    /// wire name `dispatch`.
    Dispatch,
}

impl DropStage {
    /// Returns the serde wire name of the stage (`"intake"` or `"dispatch"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Intake => "intake",
            Self::Dispatch => "dispatch",
        }
    }
}

/// Notifications one stage of the watcher dropped because its queue was full.
///
/// `dropped` counts the notifications this notice covers (the ones lost since
/// the previous notice for the same stage), `total` counts everything that
/// stage lost since the watcher started. A consumer should treat a drop notice
/// as "reload from disk" rather than "apply these deltas", because the lost
/// notifications may have been the only ones for a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedUpdates {
    /// The stage that dropped the notifications.
    pub stage: DropStage,
    /// Notifications this notice covers.
    pub dropped: u64,
    /// Notifications this stage dropped since the watcher started.
    pub total: u64,
}

/// The envelope a subscription yields.
///
/// Tagged with `"type"` on the wire, so the variants are
/// `{"type": "event", …}`, `{"type": "reload_failed", …}`,
/// `{"type": "dropped", …}` and `{"type": "watcher_failed", …}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WatchUpdate {
    /// A debounced change event.
    Event(WatchEvent),
    /// A reload handler rejected an event.
    ReloadFailed(ReloadFailure),
    /// The watcher dropped notifications because a bounded queue was full.
    Dropped(DroppedUpdates),
    /// The OS watcher reported a runtime failure.
    WatcherFailed(WatchFailure),
}

impl WatchUpdate {
    /// Returns the sequence number this update belongs to.
    ///
    /// [`Self::Event`] and [`Self::ReloadFailed`] carry one; [`Self::Dropped`]
    /// and [`Self::WatcherFailed`] do not describe a single event and return
    /// `None`.
    #[must_use]
    pub fn sequence(&self) -> Option<u64> {
        match self {
            Self::Event(event) => Some(event.sequence),
            Self::ReloadFailed(failure) => Some(failure.sequence),
            Self::Dropped(_) | Self::WatcherFailed(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    const ALL_KINDS: [WatchTargetKind; 5] = [
        WatchTargetKind::Skills,
        WatchTargetKind::Spec,
        WatchTargetKind::InstructionFile,
        WatchTargetKind::McpConfig,
        WatchTargetKind::Other,
    ];

    const ALL_CHANGES: [ChangeKind; 4] = [
        ChangeKind::Created,
        ChangeKind::Modified,
        ChangeKind::Removed,
        ChangeKind::Renamed,
    ];

    fn sample_event() -> WatchEvent {
        WatchEvent {
            sequence: 7,
            kind: WatchTargetKind::McpConfig,
            path: PathBuf::from("/w/.openjiuwen/mcp.json"),
            change: ChangeKind::Renamed,
            previous_path: Some(PathBuf::from("/w/mcp.old.json")),
            label: Some("mcp:workspace".to_owned()),
        }
    }

    #[test]
    fn target_kind_wire_names_are_snake_case() {
        let expected = [
            (WatchTargetKind::Skills, "skills"),
            (WatchTargetKind::Spec, "spec"),
            (WatchTargetKind::InstructionFile, "instruction_file"),
            (WatchTargetKind::McpConfig, "mcp_config"),
            (WatchTargetKind::Other, "other"),
        ];
        for (kind, wire) in expected {
            assert_eq!(kind.as_str(), wire);
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(wire));
        }
        for kind in ALL_KINDS {
            let encoded = serde_json::to_string(&kind).unwrap();
            let decoded: WatchTargetKind = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, kind);
        }
    }

    #[test]
    fn change_kind_wire_names_round_trip() {
        let expected = [
            (ChangeKind::Created, "created"),
            (ChangeKind::Modified, "modified"),
            (ChangeKind::Removed, "removed"),
            (ChangeKind::Renamed, "renamed"),
        ];
        for (change, wire) in expected {
            assert_eq!(change.as_str(), wire);
            assert_eq!(serde_json::to_value(change).unwrap(), json!(wire));
        }
        for change in ALL_CHANGES {
            let encoded = serde_json::to_string(&change).unwrap();
            let decoded: ChangeKind = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, change);
        }
    }

    #[test]
    fn merge_matches_the_documented_table() {
        // Rows are `existing`, columns are `incoming`.
        let expected = [
            // Created row
            [
                ChangeKind::Created,
                ChangeKind::Created,
                ChangeKind::Removed,
                ChangeKind::Renamed,
            ],
            // Modified row
            [
                ChangeKind::Modified,
                ChangeKind::Modified,
                ChangeKind::Removed,
                ChangeKind::Renamed,
            ],
            // Removed row
            [
                ChangeKind::Modified,
                ChangeKind::Removed,
                ChangeKind::Removed,
                ChangeKind::Renamed,
            ],
            // Renamed row
            [
                ChangeKind::Renamed,
                ChangeKind::Renamed,
                ChangeKind::Removed,
                ChangeKind::Renamed,
            ],
        ];
        for (row, existing) in ALL_CHANGES.iter().enumerate() {
            for (column, incoming) in ALL_CHANGES.iter().enumerate() {
                assert_eq!(
                    existing.merge(*incoming),
                    expected[row][column],
                    "merge({existing:?}, {incoming:?})"
                );
            }
        }
    }

    #[test]
    fn merge_is_idempotent_for_repeated_identical_changes() {
        for change in ALL_CHANGES {
            assert_eq!(change.merge(change), change);
        }
    }

    #[test]
    fn watch_event_round_trips_through_json() {
        let event = sample_event();
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["sequence"], json!(7));
        assert_eq!(value["kind"], json!("mcp_config"));
        assert_eq!(value["change"], json!("renamed"));
        assert_eq!(value["path"], json!("/w/.openjiuwen/mcp.json"));
        assert_eq!(value["previous_path"], json!("/w/mcp.old.json"));
        assert_eq!(value["label"], json!("mcp:workspace"));

        let decoded: WatchEvent = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn watch_event_omits_absent_optional_fields() {
        let event = WatchEvent {
            sequence: 1,
            kind: WatchTargetKind::Skills,
            path: PathBuf::from("/w/skills"),
            change: ChangeKind::Created,
            previous_path: None,
            label: None,
        };
        let value = serde_json::to_value(&event).unwrap();
        assert!(value.get("previous_path").is_none());
        assert!(value.get("label").is_none());

        let decoded: WatchEvent = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn watch_event_tolerates_unknown_fields() {
        let raw = json!({
            "sequence": 3,
            "kind": "spec",
            "path": "/w/.openjiuwen/agents/a.yaml",
            "change": "created",
            "future_field": {"nested": true}
        });
        let decoded: WatchEvent = serde_json::from_value(raw).unwrap();
        assert_eq!(decoded.sequence, 3);
        assert_eq!(decoded.kind, WatchTargetKind::Spec);
        assert_eq!(decoded.change, ChangeKind::Created);
        assert!(decoded.previous_path.is_none());
    }

    #[test]
    fn updates_are_tagged_and_carry_sequences() {
        let event = sample_event();
        let event_update = WatchUpdate::Event(event.clone());
        let value = serde_json::to_value(&event_update).unwrap();
        assert_eq!(value["type"], json!("event"));
        assert_eq!(event_update.sequence(), Some(7));
        let decoded: WatchUpdate = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, event_update);

        let failure = ReloadFailure {
            sequence: 9,
            handler: "skills".to_owned(),
            kind: WatchTargetKind::Skills,
            path: PathBuf::from("/w/skills/CONTEXT.md"),
            error_kind: "sys_op".to_owned(),
            message: "cannot read".to_owned(),
        };
        let reload_update = WatchUpdate::ReloadFailed(failure.clone());
        let value = serde_json::to_value(&reload_update).unwrap();
        assert_eq!(value["type"], json!("reload_failed"));
        assert_eq!(value["handler"], json!("skills"));
        assert_eq!(reload_update.sequence(), Some(9));
        let decoded: WatchUpdate = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, reload_update);

        let dropped_update = WatchUpdate::Dropped(DroppedUpdates {
            stage: DropStage::Intake,
            dropped: 3,
            total: 11,
        });
        let value = serde_json::to_value(&dropped_update).unwrap();
        assert_eq!(value["type"], json!("dropped"));
        assert_eq!(value["stage"], json!("intake"));
        assert_eq!(value["total"], json!(11));
        assert_eq!(dropped_update.sequence(), None);
        let decoded: WatchUpdate = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, dropped_update);

        let watcher_update = WatchUpdate::WatcherFailed(WatchFailure {
            error_kind: "config".to_owned(),
            message: "backend rejected the plan".to_owned(),
        });
        let value = serde_json::to_value(&watcher_update).unwrap();
        assert_eq!(value["type"], json!("watcher_failed"));
        assert_eq!(watcher_update.sequence(), None);
        let decoded: WatchUpdate = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, watcher_update);
    }

    #[test]
    fn drop_stage_wire_names_round_trip() {
        let expected = [
            (DropStage::Intake, "intake"),
            (DropStage::Dispatch, "dispatch"),
        ];
        for (stage, wire) in expected {
            assert_eq!(stage.as_str(), wire);
            assert_eq!(serde_json::to_value(stage).unwrap(), json!(wire));
            let decoded: DropStage =
                serde_json::from_str(&serde_json::to_string(&stage).unwrap()).unwrap();
            assert_eq!(decoded, stage);
        }
    }
}
