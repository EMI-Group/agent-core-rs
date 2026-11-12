//! Permission levels, decisions, confirmation payloads and the `permissions:` config sections.
//!
//! Ports `openjiuwen/harness/security/permission_engine/models.py` (v0.1.20): the
//! [`PermissionLevel`] enum, the [`PermissionResult`] decision record, the frozen
//! [`PermissionConfirmResponse`] HITL answer, and the `TypedDict` configuration
//! structures that back the `permissions:` section of the agent YAML
//! ([`PermissionsSection`], [`PermissionRule`], [`ApprovalOverrideEntry`],
//! [`FileGuardSection`], [`NetGuardSection`], [`ShellGuardSection`]).
//!
//! This is the data layer of the permission pipeline: it performs no IO and makes
//! no decisions. The tiered policy, the file/network guards, the engine fold and
//! the interrupt rail are built on top of these types.
//!
//! Two tolerance rules keep legacy YAML loadable:
//!
//! - [`RuleMatchType`] and [`PathMatch`] fall back to their first variant for any
//!   unrecognised spelling instead of failing the whole configuration.
//! - [`PermissionsSection::tools`] accepts both the scalar form
//!   (`bash: ask`) and the legacy mapping form (`bash: { "*": ask }`), and always
//!   serializes as the scalar form.

// Size note: ~720 lines of implementation plus an inline test module; the tests stay here so the module contract and its verification live together.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Permission decision level; wire form `allow`/`ask`/`deny`.
///
/// Python parity: `PermissionLevel(str, Enum)`. The default is [`Self::Ask`] so a
/// missing or malformed level can never widen to an implicit allow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PermissionLevel {
    /// Prompt the user before running the tool call.
    #[default]
    Ask,
    /// Run the tool call without asking.
    Allow,
    /// Reject the tool call with a permission error.
    Deny,
}

impl PermissionLevel {
    /// The lowercase Python value of the level (`"ask"` … `"deny"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Allow => "allow",
            Self::Deny => "deny",
        }
    }

    /// Parses a level from its Python value, ignoring surrounding whitespace and
    /// ASCII case. Anything but `allow`/`ask`/`deny` yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "allow" => Some(Self::Allow),
            "ask" => Some(Self::Ask),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }
}

impl fmt::Display for PermissionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of a permission check plus the labels that explain it.
///
/// Python parity: `PermissionResult` (`matched_rule`/`reason` are `None` when the
/// pipeline produced no explanation).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PermissionResult {
    /// The decision; [`PermissionLevel::Ask`] when unset.
    pub permission: PermissionLevel,
    /// Behavioural label of the winning rule, e.g. `tools.bash` or
    /// `tiered_policy:builtin:deny:<ids>`.
    pub matched_rule: Option<String>,
    /// Human-readable reason for the decision.
    pub reason: Option<String>,
    /// Paths outside the workspace that the call would touch.
    pub external_paths: Option<Vec<String>>,
}

impl PermissionResult {
    /// Creates a result carrying only a decision.
    #[must_use]
    pub fn new(permission: PermissionLevel) -> Self {
        Self {
            permission,
            ..Self::default()
        }
    }

    /// Sets the `matched_rule` label.
    #[must_use]
    pub fn with_matched_rule(mut self, rule: impl Into<String>) -> Self {
        self.matched_rule = Some(rule.into());
        self
    }

    /// Sets the human-readable reason.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Sets the external paths that triggered the decision.
    #[must_use]
    pub fn with_external_paths(mut self, paths: Vec<String>) -> Self {
        self.external_paths = Some(paths);
        self
    }

    /// Whether the decision allows the call outright.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.permission == PermissionLevel::Allow
    }

    /// Whether the decision rejects the call.
    #[must_use]
    pub fn is_denied(&self) -> bool {
        self.permission == PermissionLevel::Deny
    }

    /// Whether the decision requires a human confirmation.
    #[must_use]
    pub fn needs_approval(&self) -> bool {
        self.permission == PermissionLevel::Ask
    }
}

/// User answer to a tool-permission ASK prompt.
///
/// Python parity: the frozen dataclass `PermissionConfirmResponse`. The four
/// user-facing choices map onto the three flags:
///
/// - allow once: `approved`, `auto_confirm = false`;
/// - remember for the session: `approved`, `auto_confirm`, `persist_allow = false`;
/// - remember permanently: `approved`, `auto_confirm`, `persist_allow = true` or
///   `None` (legacy hosts omit the field, which means permanent);
/// - reject: `approved = false`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PermissionConfirmResponse {
    /// Whether the user approved the call.
    pub approved: bool,
    /// Free-form feedback shown to the model when rejected.
    pub feedback: String,
    /// Whether the approval should be remembered at all.
    pub auto_confirm: bool,
    /// `Some(true)` = permanent, `Some(false)` = session-scoped, `None` = legacy
    /// host (treated as permanent).
    pub persist_allow: Option<bool>,
}

/// Maximum number of nested JSON-string layers [`PermissionConfirmResponse::parse`]
/// unwraps before giving up; bounds the recursion on hostile payloads.
const MAX_STRING_UNWRAP_DEPTH: u8 = 8;

impl PermissionConfirmResponse {
    /// Whether the answer asks for a permanent allow rule.
    ///
    /// Only an approved, auto-confirmed answer can persist; a missing
    /// `persist_allow` means a legacy host and therefore a permanent rule.
    #[must_use]
    pub fn wants_permanent_persist(&self) -> bool {
        if !self.approved || !self.auto_confirm {
            return false;
        }
        self.persist_allow.unwrap_or(true)
    }

    /// Whether the answer asks for a session-scoped allow rule
    /// (`persist_allow = Some(false)`).
    #[must_use]
    pub fn wants_session_persist(&self) -> bool {
        self.approved && self.auto_confirm && self.persist_allow == Some(false)
    }

    /// Parses a confirm answer of any shape the kernel may deliver.
    ///
    /// - a JSON bool is the `approved` flag;
    /// - a JSON object is read field by field (`approved` must be a bool, the
    ///   other fields default, unknown keys are ignored);
    /// - a JSON *string* is first re-parsed as JSON and re-entered when it holds an
    ///   object or a bool, and otherwise falls back to
    ///   [`openjiuwen_agent::interrupt::parse_approval`] so answer words such as
    ///   `yes`, `allow_once` or `deny` keep working;
    /// - anything else yields `None`.
    #[must_use]
    pub fn parse(value: &Value) -> Option<Self> {
        Self::parse_depth(value, 0)
    }

    /// The JSON schema advertised for the confirmation payload, mirroring
    /// Python's `ConfirmPayload.to_schema()`.
    #[must_use]
    pub fn payload_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "approved": { "type": "boolean" },
                "feedback": { "type": "string" },
                "auto_confirm": { "type": "boolean" },
                "persist_allow": { "type": ["boolean", "null"] },
            },
            "required": ["approved"],
        })
    }

    fn parse_depth(value: &Value, depth: u8) -> Option<Self> {
        match value {
            Value::Bool(approved) => Some(Self {
                approved: *approved,
                ..Self::default()
            }),
            Value::Object(map) => {
                let approved = map.get("approved").and_then(Value::as_bool)?;
                Some(Self {
                    approved,
                    feedback: map
                        .get("feedback")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    auto_confirm: map
                        .get("auto_confirm")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    persist_allow: map.get("persist_allow").and_then(Value::as_bool),
                })
            }
            Value::String(text) => {
                if depth < MAX_STRING_UNWRAP_DEPTH {
                    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                        if matches!(parsed, Value::Bool(_) | Value::Object(_)) {
                            return Self::parse_depth(&parsed, depth + 1);
                        }
                    }
                }
                openjiuwen_agent::interrupt::parse_approval(value).map(|approved| Self {
                    approved,
                    ..Self::default()
                })
            }
            _ => None,
        }
    }
}

/// What a rule or path entry matches against.
///
/// Python parity: `match_type` values `command` (default) and `path`. Any other
/// spelling degrades to [`Self::Command`] so one bad YAML value cannot make the
/// whole `permissions:` section unloadable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuleMatchType {
    /// Match the command text of a shell tool.
    #[default]
    Command,
    /// Match a path argument of a file tool.
    Path,
}

impl RuleMatchType {
    /// The wire value of the match type (`"command"` / `"path"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::Path => "path",
        }
    }

    /// Parses a match type, ignoring whitespace and ASCII case; unknown values
    /// yield [`Self::Command`].
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "path" => Self::Path,
            _ => Self::Command,
        }
    }
}

impl<'de> Deserialize<'de> for RuleMatchType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Option::<String>::deserialize(deserializer)?.unwrap_or_default();
        Ok(Self::parse(&raw))
    }
}

/// How a [`FileGuardPathEntry`] path is compared.
///
/// Python parity: `match` values `prefix` (default) and `glob`. Unknown values
/// degrade to [`Self::Prefix`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PathMatch {
    /// The entry covers everything below `path`.
    #[default]
    Prefix,
    /// The entry matches `path` as a `**`-aware glob.
    Glob,
}

impl PathMatch {
    /// The wire value of the matcher (`"prefix"` / `"glob"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Glob => "glob",
        }
    }

    /// Parses a matcher, ignoring whitespace and ASCII case; anything but `glob`
    /// yields [`Self::Prefix`].
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "glob" => Self::Glob,
            _ => Self::Prefix,
        }
    }
}

impl<'de> Deserialize<'de> for PathMatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Option::<String>::deserialize(deserializer)?.unwrap_or_default();
        Ok(Self::parse(&raw))
    }
}

/// Action a path access asks for.
///
/// Python parity: the `read`/`write`/`exec` axes of the file guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileAction {
    /// Read the path.
    Read,
    /// Create or modify the path.
    Write,
    /// Execute the path.
    Exec,
}

impl FileAction {
    /// The wire value of the action (`"read"` / `"write"` / `"exec"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Exec => "exec",
        }
    }
}

/// One path a tool call would touch, and how.
///
/// Python parity: the `(path, action)` tuples returned by the path extractor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathAccess {
    /// Resolved path the call would touch.
    pub path: PathBuf,
    /// What the call would do with it.
    pub action: FileAction,
}

impl PathAccess {
    /// Creates an access record.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, action: FileAction) -> Self {
        Self {
            path: path.into(),
            action,
        }
    }
}

/// A user or packaged parameter rule.
///
/// Python parity: one entry of `permissions.rules` / the packaged
/// `builtin_rules.yaml`. `tools` names the tools (or tool categories such as
/// `shell`) the rule applies to; `pattern` is matched against the command text
/// ([`RuleMatchType::Command`]) or a path ([`RuleMatchType::Path`]), with a `re:`
/// prefix switching to a regex.
///
/// `action` is filled from the legacy `severity` label at ingest time when the
/// entry does not carry one; `layer: "builtin"` marks packaged rules so that
/// inlining them into the section stays idempotent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PermissionRule {
    /// Stable identifier used in `matched_rule` labels.
    pub id: String,
    /// Tools or tool categories the rule applies to, e.g. `["shell"]`.
    pub tools: Vec<String>,
    /// Command (or path) pattern; a `re:` prefix means a regular expression.
    pub pattern: String,
    /// Which input `pattern` is matched against.
    pub match_type: RuleMatchType,
    /// Resulting decision; filled from `severity` when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<PermissionLevel>,
    /// Legacy severity label (`LOW`/`MEDIUM`/`HIGH`/`CRITICAL`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
    /// Human-readable description of the rule.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Rule provenance; `Some("builtin")` marks packaged rules.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
}

/// A remembered "allow" decision from `permissions.approval_overrides`.
///
/// Python parity: `ApprovalOverrideEntry`. `match_type` selects which input
/// `pattern` applies to, exactly like [`PermissionRule`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ApprovalOverrideEntry {
    /// Stable identifier used in `matched_rule` labels.
    pub id: String,
    /// Tools or tool categories the override applies to.
    pub tools: Vec<String>,
    /// Which input `pattern` is matched against.
    pub match_type: RuleMatchType,
    /// Command (or path) pattern; a `re:` prefix means a regular expression.
    pub pattern: String,
    /// Decision the override yields (always [`PermissionLevel::Allow`] in
    /// practice).
    pub action: PermissionLevel,
}

impl Default for ApprovalOverrideEntry {
    fn default() -> Self {
        Self {
            id: String::new(),
            tools: Vec::new(),
            match_type: RuleMatchType::Command,
            pattern: String::new(),
            action: PermissionLevel::Allow,
        }
    }
}

/// Default read/write/exec decisions of the file guard.
///
/// Python parity: `file_guard.defaults` — reads and writes are allowed inside the
/// workspace, execution still asks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct FileGuardDefaults {
    /// Decision for read accesses.
    pub read: PermissionLevel,
    /// Decision for write accesses.
    pub write: PermissionLevel,
    /// Decision for execute accesses.
    pub exec: PermissionLevel,
}

impl Default for FileGuardDefaults {
    fn default() -> Self {
        Self {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Allow,
            exec: PermissionLevel::Ask,
        }
    }
}

/// One path policy entry of the file guard.
///
/// Python parity: `file_guard.paths[]`. Unset axes default to
/// [`PermissionLevel::Ask`], so a path entry only opens what it names explicitly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct FileGuardPathEntry {
    /// Optional identifier carried through to `matched_rule` labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Prefix or glob the entry covers (see `match`).
    pub path: String,
    /// Decision for read accesses below `path`.
    pub read: PermissionLevel,
    /// Decision for write accesses below `path`.
    pub write: PermissionLevel,
    /// Decision for execute accesses below `path`.
    pub exec: PermissionLevel,
    /// How `path` is compared.
    #[serde(rename = "match")]
    pub r#match: PathMatch,
    /// Rule provenance; `Some("builtin")` marks packaged entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
}

impl Default for FileGuardPathEntry {
    fn default() -> Self {
        Self {
            id: None,
            path: String::new(),
            read: PermissionLevel::Ask,
            write: PermissionLevel::Ask,
            exec: PermissionLevel::Ask,
            r#match: PathMatch::Prefix,
            layer: None,
        }
    }
}

/// Path guard configuration (pipeline B); can be enabled independently of the
/// tool-policy switch.
///
/// Python parity: `permissions.file_guard`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct FileGuardSection {
    /// Whether the file guard runs at all.
    pub enabled: bool,
    /// Decisions for paths no `paths` entry covers.
    pub defaults: FileGuardDefaults,
    /// Decisions applied to the workspace root, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<FileGuardDefaults>,
    /// Per-path policies.
    pub paths: Vec<FileGuardPathEntry>,
}

/// Network guard configuration (pipeline C); allow/deny only, never ask.
///
/// Python parity: `permissions.net_guard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct NetGuardSection {
    /// Whether the network guard runs at all.
    pub enabled: bool,
    /// Decision for URLs no `urls` entry covers; defaults to
    /// [`PermissionLevel::Allow`] (the guard already narrowed the candidate set).
    pub defaults: PermissionLevel,
    /// Per-URL-pattern decisions.
    pub urls: BTreeMap<String, PermissionLevel>,
}

impl Default for NetGuardSection {
    fn default() -> Self {
        Self {
            enabled: false,
            defaults: PermissionLevel::Allow,
            urls: BTreeMap::new(),
        }
    }
}

/// Extra shell-structure checks; `true` widens a decision to ask.
///
/// Python parity: `permissions.shell_guard`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ShellGuardSection {
    /// Ask when the command structure cannot be analysed confidently.
    pub unknown_structure: bool,
    /// Ask when the command pipes into an interpreter (`bash -c`, `python -c`, …).
    pub interpreter_sink: bool,
}

impl Default for ShellGuardSection {
    fn default() -> Self {
        Self {
            unknown_structure: true,
            interpreter_sink: true,
        }
    }
}

/// The `permissions:` section of the agent YAML.
///
/// Python parity: the `PermissionsSection` `TypedDict`, modelled as typed data
/// because it is embedded in [`crate::schema::DeepAgentConfig`]. Defaults are
/// fail-closed: every tool falls back to [`PermissionLevel::Ask`] unless an
/// explicit rule, override, baseline or `defaults` entry says otherwise.
///
/// `tools` accepts both the scalar form (`bash: ask`) and the legacy mapping form
/// (`bash: { "*": ask }`); it always serializes as the scalar form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PermissionsSection {
    /// Whether the permission pipeline runs at all.
    pub enabled: bool,
    /// Whether the packaged builtin rules, sensitive paths and metadata URLs are
    /// inlined into this section before evaluation.
    pub package_builtin_rules: bool,
    /// `normal` (default) or `strict`; `None` means `normal`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<String>,
    /// Fallback decisions keyed by tool name or `"*"`.
    #[serde(default = "default_rule_levels")]
    pub defaults: BTreeMap<String, PermissionLevel>,
    /// Per-tool baselines, keyed by tool name.
    #[serde(default, deserialize_with = "deserialize_tool_levels")]
    #[schemars(with = "BTreeMap<String, String>")]
    pub tools: BTreeMap<String, PermissionLevel>,
    /// Tool-category members, e.g. `categories["shell"]` replacing the default
    /// shell tool set.
    pub categories: BTreeMap<String, Vec<String>>,
    /// Parameter rules, evaluated before the tool baselines.
    pub rules: Vec<PermissionRule>,
    /// Remembered allow decisions from earlier confirmations.
    pub approval_overrides: Vec<ApprovalOverrideEntry>,
    /// Path guard configuration.
    pub file_guard: FileGuardSection,
    /// Network guard configuration.
    pub net_guard: NetGuardSection,
    /// Shell-structure guard configuration.
    pub shell_guard: ShellGuardSection,
}

impl Default for PermissionsSection {
    fn default() -> Self {
        Self {
            enabled: true,
            package_builtin_rules: true,
            permission_mode: None,
            defaults: default_rule_levels(),
            tools: BTreeMap::new(),
            categories: BTreeMap::new(),
            rules: Vec::new(),
            approval_overrides: Vec::new(),
            file_guard: FileGuardSection::default(),
            net_guard: NetGuardSection::default(),
            shell_guard: ShellGuardSection::default(),
        }
    }
}

impl PermissionsSection {
    /// The configured baseline of exactly `tool`, if any.
    ///
    /// Only an exact tool-name entry counts; the `"*"` fallback is reported by
    /// [`Self::default_level`]. Names are not normalized here — callers fold them
    /// through the tool-name aliases first.
    #[must_use]
    pub fn level_for_tool(&self, tool: &str) -> Option<PermissionLevel> {
        self.tools.get(tool).copied()
    }

    /// The `defaults["*"]` fallback decision, or [`PermissionLevel::Ask`].
    #[must_use]
    pub fn default_level(&self) -> PermissionLevel {
        self.defaults
            .get("*")
            .copied()
            .unwrap_or(PermissionLevel::Ask)
    }

    /// Whether the packaged builtin rules should be inlined before evaluation.
    #[must_use]
    pub fn package_builtin_rules_enabled(&self) -> bool {
        self.package_builtin_rules
    }
}

/// Default `defaults` map: everything falls back to ask.
fn default_rule_levels() -> BTreeMap<String, PermissionLevel> {
    BTreeMap::from([("*".to_owned(), PermissionLevel::Ask)])
}

/// Deserializes per-tool baselines, accepting the legacy `{"*": level}` mapping.
///
/// Entries whose value is not a known level (neither directly nor under `"*"`)
/// are dropped, mirroring Python's warn-and-ignore behaviour; the tool then falls
/// back to `defaults`, which is ask, never allow.
fn deserialize_tool_levels<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, PermissionLevel>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = BTreeMap::<String, Value>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|(tool, value)| tool_level(&value).map(|level| (tool, level)))
        .collect())
}

/// Reads a tool baseline value: a level string, or a legacy object with a `"*"`
/// level string.
fn tool_level(value: &Value) -> Option<PermissionLevel> {
    match value {
        Value::String(raw) => PermissionLevel::parse(raw),
        Value::Object(map) => map.get("*").and_then(|inner| match inner {
            Value::String(raw) => PermissionLevel::parse(raw),
            _ => None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    #[test]
    fn permission_level_wire_form_and_parse() {
        assert_eq!(
            serde_json::to_value(PermissionLevel::Ask).unwrap(),
            json!("ask")
        );
        assert_eq!(
            serde_json::to_value(PermissionLevel::Allow).unwrap(),
            json!("allow")
        );
        assert_eq!(
            serde_json::to_value(PermissionLevel::Deny).unwrap(),
            json!("deny")
        );
        for level in [
            PermissionLevel::Ask,
            PermissionLevel::Allow,
            PermissionLevel::Deny,
        ] {
            assert_eq!(PermissionLevel::parse(level.as_str()), Some(level));
            assert_eq!(level.to_string(), level.as_str());
            let round_trip: PermissionLevel =
                serde_json::from_value(serde_json::to_value(level).unwrap()).unwrap();
            assert_eq!(round_trip, level);
        }
        assert_eq!(
            PermissionLevel::parse("  ALLOW "),
            Some(PermissionLevel::Allow)
        );
        assert_eq!(PermissionLevel::parse("Deny"), Some(PermissionLevel::Deny));
        assert_eq!(PermissionLevel::parse(""), None);
        assert_eq!(PermissionLevel::parse("always"), None);
        assert_eq!(PermissionLevel::default(), PermissionLevel::Ask);
        assert_eq!(PermissionLevel::default().as_str(), "ask");
    }

    #[test]
    fn rule_match_type_wire_form_and_tolerance() {
        assert_eq!(
            serde_json::to_value(RuleMatchType::Command).unwrap(),
            json!("command")
        );
        assert_eq!(
            serde_json::to_value(RuleMatchType::Path).unwrap(),
            json!("path")
        );
        assert_eq!(RuleMatchType::default(), RuleMatchType::Command);

        let parsed: RuleMatchType = serde_json::from_str("\"path\"").unwrap();
        assert_eq!(parsed, RuleMatchType::Path);
        let parsed: RuleMatchType = serde_json::from_str("\"PATH\"").unwrap();
        assert_eq!(parsed, RuleMatchType::Path);
        let unknown: RuleMatchType = serde_json::from_str("\"regex\"").unwrap();
        assert_eq!(unknown, RuleMatchType::Command);
        let null: RuleMatchType = serde_json::from_str("null").unwrap();
        assert_eq!(null, RuleMatchType::Command);
        assert_eq!(RuleMatchType::parse(" command "), RuleMatchType::Command);
        assert_eq!(RuleMatchType::Path.as_str(), "path");
    }

    #[test]
    fn path_match_wire_form_and_tolerance() {
        assert_eq!(
            serde_json::to_value(PathMatch::Prefix).unwrap(),
            json!("prefix")
        );
        assert_eq!(
            serde_json::to_value(PathMatch::Glob).unwrap(),
            json!("glob")
        );
        assert_eq!(PathMatch::default(), PathMatch::Prefix);

        let parsed: PathMatch = serde_json::from_str("\"glob\"").unwrap();
        assert_eq!(parsed, PathMatch::Glob);
        let unknown: PathMatch = serde_json::from_str("\"recursive\"").unwrap();
        assert_eq!(unknown, PathMatch::Prefix);
        let null: PathMatch = serde_json::from_str("null").unwrap();
        assert_eq!(null, PathMatch::Prefix);
        assert_eq!(PathMatch::Glob.as_str(), "glob");
    }

    #[test]
    fn file_action_and_path_access_wire_form() {
        assert_eq!(
            serde_json::to_value(FileAction::Read).unwrap(),
            json!("read")
        );
        assert_eq!(
            serde_json::to_value(FileAction::Write).unwrap(),
            json!("write")
        );
        assert_eq!(
            serde_json::to_value(FileAction::Exec).unwrap(),
            json!("exec")
        );
        assert_eq!(FileAction::Exec.as_str(), "exec");

        let access = PathAccess::new("/tmp/file.txt", FileAction::Write);
        assert_eq!(access.path, PathBuf::from("/tmp/file.txt"));
        assert_eq!(
            serde_json::to_value(&access).unwrap(),
            json!({"path": "/tmp/file.txt", "action": "write"})
        );
        let round_trip: PathAccess = serde_json::from_value(json!({
            "path": "/tmp/file.txt",
            "action": "write",
            "extra": 1,
        }))
        .unwrap();
        assert_eq!(round_trip, access);
    }

    #[test]
    fn permission_result_helpers() {
        let ask = PermissionResult::default();
        assert_eq!(ask.permission, PermissionLevel::Ask);
        assert!(ask.needs_approval());
        assert!(!ask.is_allowed());
        assert!(!ask.is_denied());
        assert_eq!(ask.matched_rule, None);

        let allow = PermissionResult::new(PermissionLevel::Allow)
            .with_matched_rule("tools.bash")
            .with_reason("baseline")
            .with_external_paths(vec!["/etc".to_owned()]);
        assert!(allow.is_allowed());
        assert!(!allow.needs_approval());
        assert_eq!(allow.matched_rule.as_deref(), Some("tools.bash"));
        assert_eq!(allow.reason.as_deref(), Some("baseline"));
        assert_eq!(allow.external_paths, Some(vec!["/etc".to_owned()]));

        let deny = PermissionResult::new(PermissionLevel::Deny);
        assert!(deny.is_denied());
        assert!(!deny.is_allowed());
        assert!(!deny.needs_approval());

        let round_trip: PermissionResult =
            serde_json::from_value(serde_json::to_value(&allow).unwrap()).unwrap();
        assert_eq!(round_trip, allow);
        assert_eq!(
            serde_json::from_str::<PermissionResult>("{}").unwrap(),
            PermissionResult::default()
        );
    }

    #[test]
    fn confirm_response_persist_truth_table() {
        let permanent_legacy = PermissionConfirmResponse {
            approved: true,
            auto_confirm: true,
            ..PermissionConfirmResponse::default()
        };
        assert!(permanent_legacy.wants_permanent_persist());
        assert!(!permanent_legacy.wants_session_persist());

        let permanent_explicit = PermissionConfirmResponse {
            persist_allow: Some(true),
            ..permanent_legacy.clone()
        };
        assert!(permanent_explicit.wants_permanent_persist());
        assert!(!permanent_explicit.wants_session_persist());

        let session = PermissionConfirmResponse {
            persist_allow: Some(false),
            ..permanent_legacy.clone()
        };
        assert!(!session.wants_permanent_persist());
        assert!(session.wants_session_persist());

        let once = PermissionConfirmResponse {
            approved: true,
            auto_confirm: false,
            persist_allow: None,
            feedback: String::new(),
        };
        assert!(!once.wants_permanent_persist());
        assert!(!once.wants_session_persist());

        let rejected = PermissionConfirmResponse {
            approved: false,
            auto_confirm: true,
            ..PermissionConfirmResponse::default()
        };
        assert!(!rejected.wants_permanent_persist());
        assert!(!rejected.wants_session_persist());

        let rejected_session = PermissionConfirmResponse {
            approved: false,
            auto_confirm: true,
            persist_allow: Some(false),
            feedback: "no".to_owned(),
        };
        assert!(!rejected_session.wants_permanent_persist());
        assert!(!rejected_session.wants_session_persist());
    }

    #[test]
    fn confirm_response_parse_shapes() {
        assert!(
            PermissionConfirmResponse::parse(&json!(true))
                .unwrap()
                .approved
        );
        assert_eq!(
            PermissionConfirmResponse::parse(&json!(false)).unwrap(),
            PermissionConfirmResponse::default()
        );

        let full = PermissionConfirmResponse::parse(&json!({
            "approved": true,
            "feedback": "ok",
            "auto_confirm": true,
            "persist_allow": false,
            "unknown": 42,
        }))
        .unwrap();
        assert_eq!(
            full,
            PermissionConfirmResponse {
                approved: true,
                feedback: "ok".to_owned(),
                auto_confirm: true,
                persist_allow: Some(false),
            }
        );
        assert!(full.wants_session_persist());

        let missing = PermissionConfirmResponse::parse(&json!({"approved": true})).unwrap();
        assert_eq!(missing.feedback, "");
        assert!(!missing.auto_confirm);
        assert_eq!(missing.persist_allow, None);
        assert!(!missing.wants_permanent_persist());

        assert_eq!(PermissionConfirmResponse::parse(&json!({})), None);
        assert_eq!(
            PermissionConfirmResponse::parse(&json!({"approved": "yes"})),
            None
        );

        let from_json_string = PermissionConfirmResponse::parse(&json!(
            "{\"approved\": true, \"auto_confirm\": true}"
        ))
        .unwrap();
        assert!(from_json_string.approved);
        assert!(from_json_string.auto_confirm);
        assert!(from_json_string.wants_permanent_persist());

        let from_bool_string = PermissionConfirmResponse::parse(&json!("true")).unwrap();
        assert!(from_bool_string.approved);
        assert!(!from_bool_string.auto_confirm);

        let from_word = PermissionConfirmResponse::parse(&json!("YES")).unwrap();
        assert!(from_word.approved);
        assert!(
            !PermissionConfirmResponse::parse(&json!("deny"))
                .unwrap()
                .approved
        );
        assert_eq!(PermissionConfirmResponse::parse(&json!("maybe")), None);
        assert_eq!(PermissionConfirmResponse::parse(&json!(1)), None);
        assert_eq!(PermissionConfirmResponse::parse(&Value::Null), None);
        assert_eq!(PermissionConfirmResponse::parse(&json!([true])), None);

        let round_trip: PermissionConfirmResponse =
            serde_json::from_value(serde_json::to_value(&full).unwrap()).unwrap();
        assert_eq!(round_trip, full);
        assert_eq!(
            serde_json::from_str::<PermissionConfirmResponse>("{}").unwrap(),
            PermissionConfirmResponse::default()
        );
    }

    #[test]
    fn confirm_response_payload_schema() {
        let schema = PermissionConfirmResponse::payload_schema();
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["required"], json!(["approved"]));
        assert_eq!(schema["properties"]["approved"], json!({"type": "boolean"}));
        assert_eq!(schema["properties"]["feedback"], json!({"type": "string"}));
        assert_eq!(
            schema["properties"]["auto_confirm"],
            json!({"type": "boolean"})
        );
        assert_eq!(
            schema["properties"]["persist_allow"],
            json!({"type": ["boolean", "null"]})
        );
    }

    #[test]
    fn permissions_section_defaults_are_fail_closed() {
        let section = PermissionsSection::default();
        assert!(section.enabled);
        assert!(section.package_builtin_rules);
        assert!(section.package_builtin_rules_enabled());
        assert_eq!(section.permission_mode, None);
        assert_eq!(
            section.defaults,
            BTreeMap::from([("*".to_owned(), PermissionLevel::Ask)])
        );
        assert_eq!(section.default_level(), PermissionLevel::Ask);
        assert!(section.tools.is_empty());
        assert_eq!(section.level_for_tool("bash"), None);
        assert!(section.categories.is_empty());
        assert_eq!(section.rules, Vec::<PermissionRule>::new());
        assert_eq!(
            section.approval_overrides,
            Vec::<ApprovalOverrideEntry>::new()
        );

        assert!(!section.file_guard.enabled);
        assert_eq!(section.file_guard.paths, Vec::<FileGuardPathEntry>::new());
        assert_eq!(section.file_guard.workspace, None);
        assert_eq!(section.file_guard.defaults.read, PermissionLevel::Allow);
        assert_eq!(section.file_guard.defaults.write, PermissionLevel::Allow);
        assert_eq!(section.file_guard.defaults.exec, PermissionLevel::Ask);

        assert!(!section.net_guard.enabled);
        assert_eq!(section.net_guard.defaults, PermissionLevel::Allow);
        assert!(section.net_guard.urls.is_empty());

        assert!(section.shell_guard.unknown_structure);
        assert!(section.shell_guard.interpreter_sink);

        let parsed: PermissionsSection = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, section);
    }

    #[test]
    fn permissions_section_tool_baselines_accept_legacy_form() {
        let section: PermissionsSection = serde_json::from_str(
            r#"{
                "tools": {
                    "bash": {"*": "ask"},
                    "read_file": "allow",
                    "write_file": {"*": "deny"},
                    "edit_file": {"command": "allow"},
                    "glob": "sometimes"
                }
            }"#,
        )
        .unwrap();
        assert_eq!(section.level_for_tool("bash"), Some(PermissionLevel::Ask));
        assert_eq!(
            section.level_for_tool("read_file"),
            Some(PermissionLevel::Allow)
        );
        assert_eq!(
            section.level_for_tool("write_file"),
            Some(PermissionLevel::Deny)
        );
        // Unknown shapes are dropped instead of failing the whole section.
        assert_eq!(section.level_for_tool("edit_file"), None);
        assert_eq!(section.level_for_tool("glob"), None);
        // `"*"` is never reported as a tool baseline.
        assert_eq!(section.level_for_tool("*"), None);

        let value = serde_json::to_value(&section).unwrap();
        assert_eq!(value["tools"]["bash"], json!("ask"));
        assert_eq!(value["tools"]["write_file"], json!("deny"));
    }

    #[test]
    fn permissions_section_defaults_map_handling() {
        let missing: PermissionsSection = serde_json::from_str(r#"{"tools": {}}"#).unwrap();
        assert_eq!(missing.default_level(), PermissionLevel::Ask);

        let explicit: PermissionsSection =
            serde_json::from_str(r#"{"defaults": {"bash": "allow"}}"#).unwrap();
        assert_eq!(explicit.defaults.len(), 1);
        assert_eq!(explicit.default_level(), PermissionLevel::Ask);

        let wildcard: PermissionsSection =
            serde_json::from_str(r#"{"defaults": {"*": "deny"}, "permission_mode": "strict"}"#)
                .unwrap();
        assert_eq!(wildcard.default_level(), PermissionLevel::Deny);
        assert_eq!(wildcard.permission_mode.as_deref(), Some("strict"));
    }

    #[test]
    fn permission_rule_and_override_defaults() {
        let rule: PermissionRule = serde_json::from_str("{}").unwrap();
        assert_eq!(rule, PermissionRule::default());
        assert_eq!(rule.match_type, RuleMatchType::Command);
        assert_eq!(rule.action, None);
        let value = serde_json::to_value(&rule).unwrap();
        assert_eq!(
            value,
            json!({"id": "", "tools": [], "pattern": "", "match_type": "command"})
        );

        let rule: PermissionRule = serde_json::from_str(
            r#"{
                "id": "deny-rm",
                "tools": ["shell"],
                "pattern": "re:^rm -rf",
                "match_type": "command",
                "severity": "HIGH",
                "description": "no recursive deletes",
                "layer": "builtin",
                "unknown": true
            }"#,
        )
        .unwrap();
        assert_eq!(rule.id, "deny-rm");
        assert_eq!(rule.severity.as_deref(), Some("HIGH"));
        assert_eq!(rule.layer.as_deref(), Some("builtin"));
        let round_trip: PermissionRule =
            serde_json::from_value(serde_json::to_value(&rule).unwrap()).unwrap();
        assert_eq!(round_trip, rule);

        let entry: ApprovalOverrideEntry = serde_json::from_str(r#"{"pattern": "ls*"}"#).unwrap();
        assert_eq!(entry.action, PermissionLevel::Allow);
        assert_eq!(entry.match_type, RuleMatchType::Command);
        assert_eq!(entry.id, "");
        assert_eq!(
            serde_json::to_value(&entry).unwrap(),
            json!({
                "id": "",
                "tools": [],
                "match_type": "command",
                "pattern": "ls*",
                "action": "allow",
            })
        );
    }

    #[test]
    fn file_guard_section_defaults_and_entries() {
        let path_entry: FileGuardPathEntry = serde_json::from_str(r#"{"path": "/data"}"#).unwrap();
        assert_eq!(path_entry.read, PermissionLevel::Ask);
        assert_eq!(path_entry.write, PermissionLevel::Ask);
        assert_eq!(path_entry.exec, PermissionLevel::Ask);
        assert_eq!(path_entry.r#match, PathMatch::Prefix);
        assert_eq!(path_entry.id, None);
        assert_eq!(path_entry.layer, None);

        let value = serde_json::to_value(&path_entry).unwrap();
        assert_eq!(value["match"], json!("prefix"));
        assert_eq!(value.get("id"), None);

        let section: FileGuardSection = serde_json::from_str(
            r#"{
                "enabled": true,
                "workspace": {"read": "allow", "write": "ask", "exec": "deny"},
                "paths": [
                    {"path": "/data/public", "read": "allow", "write": "ask", "exec": "deny"},
                    {"path": "/logs/**", "match": "glob", "read": "allow"}
                ]
            }"#,
        )
        .unwrap();
        assert!(section.enabled);
        assert_eq!(
            section.workspace.unwrap(),
            FileGuardDefaults {
                read: PermissionLevel::Allow,
                write: PermissionLevel::Ask,
                exec: PermissionLevel::Deny,
            }
        );
        assert_eq!(section.paths.len(), 2);
        assert_eq!(section.paths[0].exec, PermissionLevel::Deny);
        assert_eq!(section.paths[1].r#match, PathMatch::Glob);
        assert_eq!(section.paths[1].read, PermissionLevel::Allow);
        assert_eq!(section.paths[1].write, PermissionLevel::Ask);
    }

    #[test]
    fn net_and_shell_guard_defaults() {
        let net: NetGuardSection = serde_json::from_str("{}").unwrap();
        assert_eq!(net, NetGuardSection::default());
        assert!(!net.enabled);
        assert_eq!(net.defaults, PermissionLevel::Allow);

        let net: NetGuardSection = serde_json::from_str(
            r#"{"enabled": true, "defaults": "deny", "urls": {"https://*": "allow"}}"#,
        )
        .unwrap();
        assert!(net.enabled);
        assert_eq!(net.defaults, PermissionLevel::Deny);
        assert_eq!(net.urls.get("https://*"), Some(&PermissionLevel::Allow));

        let shell: ShellGuardSection = serde_json::from_str("{}").unwrap();
        assert_eq!(shell, ShellGuardSection::default());
        let shell: ShellGuardSection =
            serde_json::from_str(r#"{"unknown_structure": false, "interpreter_sink": false}"#)
                .unwrap();
        assert!(!shell.unknown_structure);
        assert!(!shell.interpreter_sink);
    }

    #[test]
    fn permissions_section_round_trips_through_json() {
        let raw = json!({
            "enabled": true,
            "package_builtin_rules": false,
            "permission_mode": "strict",
            "defaults": {"*": "ask"},
            "tools": {"bash": "ask", "read_file": "allow"},
            "categories": {"shell": ["bash", "sh"]},
            "rules": [
                {
                    "id": "deny-curl-pipe",
                    "tools": ["shell"],
                    "pattern": "re:curl .*\\|.*sh",
                    "match_type": "command",
                    "action": "deny",
                    "severity": "CRITICAL",
                    "layer": "builtin"
                }
            ],
            "approval_overrides": [
                {
                    "id": "allow-ls",
                    "tools": ["bash"],
                    "match_type": "command",
                    "pattern": "ls*",
                    "action": "allow"
                }
            ],
            "file_guard": {
                "enabled": true,
                "defaults": {"read": "allow", "write": "ask", "exec": "ask"},
                "paths": [
                    {
                        "path": "/data",
                        "read": "allow",
                        "write": "allow",
                        "exec": "deny",
                        "match": "prefix"
                    }
                ]
            },
            "net_guard": {"enabled": true, "defaults": "allow", "urls": {}},
            "shell_guard": {"unknown_structure": true, "interpreter_sink": false}
        });
        let section: PermissionsSection = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(section.rules[0].action, Some(PermissionLevel::Deny));
        assert_eq!(
            section.categories.get("shell"),
            Some(&vec!["bash".to_owned(), "sh".to_owned()])
        );
        assert_eq!(section.default_level(), PermissionLevel::Ask);

        let serialized = serde_json::to_value(&section).unwrap();
        assert_eq!(serialized, raw);
        let round_trip: PermissionsSection = serde_json::from_value(serialized).unwrap();
        assert_eq!(round_trip, section);
    }

    #[test]
    fn permissions_section_is_json_schema_derivable() {
        let schema = serde_json::to_value(schemars::schema_for!(PermissionsSection)).unwrap();
        assert_eq!(schema["type"], json!("object"));
        assert!(schema["properties"].get("file_guard").is_some());
        assert!(
            schema["properties"].get("net_guard").is_some(),
            "every section must appear in the schema"
        );
    }
}
