//! `MemberMemoryToolkit`: the model-facing team memory tools (schemas plus a JSON dispatcher).
//!
//! The Python `openjiuwen.agent_teams.memory.member_memory_toolkit.MemberMemoryToolkit`
//! hands every team member a set of lite-memory *file* tools (`memory_search`,
//! `memory_get`, `read_memory`, `write_memory`, `edit_memory`, plus the coding-scenario
//! variants) over per-member memory directories, and keeps the shared team summary in
//! `TEAM_MEMORY.md`.
//! This port targets the crate's team memory store instead: [`TeamMemory`] rows scoped to
//! the calling member, exposed as four transport-neutral tools that a model may call
//! through a JSON dispatcher.
//! Nothing here knows about a model provider, a prompt, or a tool registry — the caller
//! (the agent-team runtime) advertises [`MemberMemoryToolkit::get_tool_schemas`] and
//! routes tool calls to [`MemberMemoryToolkit::handle_tool_call`].
//!
//! # Tools
//!
//! | Tool | Required | Optional |
//! | --- | --- | --- |
//! | [`TOOL_REMEMBER`] | `content` | `title`, `kind`, `tags`, `importance`, `member_id`, `visibility`, `expires_at` |
//! | [`TOOL_SEARCH`] | `query` | `max_results`, `min_score`, `kinds`, `tags`, `include_expired` |
//! | [`TOOL_FORGET`] | `id` | — |
//! | [`TOOL_STATS`] | — | — |
//!
//! `content`/`query`/`id` are trimmed and must not be blank.
//! `kind` is parsed tolerantly ([`TeamMemoryKind::parse`], so an unknown kind becomes
//! [`TeamMemoryKind::Custom`]); `tags`/`kinds` are arrays of strings and `tags` are
//! normalized (trimmed, lowercased, deduplicated) by the store.
//! `importance` and `min_score` are finite numbers inside `0.0..=1.0` (a numeric string is
//! accepted, so a model that emits `"0.8"` still works); `max_results` is an integer in
//! `1..=100` and defaults to [`DEFAULT_SEARCH_TOP_K`]; `min_score` defaults to
//! [`DEFAULT_SEARCH_THRESHOLD`].
//! `visibility` is `"team"`, `"private"`, or `{"members": [...]}`.
//! `expires_at` is an RFC 3339 timestamp.
//! Every argument name is checked: an unknown name is a validation error rather than a
//! silently ignored typo.
//!
//! # Envelope
//!
//! A successful call returns `{"ok": true, "tool": <name>, "result": <payload>}` where the
//! payload is the stored [`TeamMemoryRecord`] (remember), `{"count": n, "results":
//! [<TeamSearchHit>]}` (search), `{"deleted": bool, "id": <id>}` (forget), or the
//! [`TeamMemoryStats`] aggregate (stats).
//! A failed call is a typed [`Error`]; [`MemberMemoryToolkit::error_envelope`] renders that
//! same failure as `{"ok": false, "tool": <name>, "error": {"kind": <snake_case>, "message":
//! <text>}}` for a protocol layer that reports errors as data instead of `Err`. The
//! dispatcher never returns an `ok: false` envelope itself, so a caller that propagates
//! `Err` and a caller that renders the envelope observe the same failures.
//!
//! # Member and scope rules
//!
//! The context names the acting member ([`MemberMemoryContext::member_id`]; `None` is the
//! team-wide/leader view) and, optionally, the coordination session.
//! Reads go through [`TeamMemoryScope`], whose `allows` rule is the authority: a member
//! sees team-visible records, its own private records, and member-list records that name
//! it (or that it authored); a team-wide context sees everything.
//! Deletes are stricter than reads: a member may delete team-visible records and its own
//! records, but another member's private or member-list record is reported as
//! [`Error::NotFound`] without revealing that it exists, and an unknown id is
//! `deleted: false` (never an error).
//! Writes are authored by the context member (unless `member_id` overrides it) and carry
//! the context session, so every tool call of one agent lands in one scope.
//!
//! # Read-only contexts
//!
//! A context built with [`MemberMemoryContext::read_only`] refuses remember and forget with
//! [`Error::Unsupported`] and still serves search and stats — the review/observer variant
//! of the Python toolkit's `read_only` flag.
//!
//! # Deliberately not ported
//!
//! The Python file read/write/edit tools (`memory_get`, `read_memory`, `write_memory`,
//! `edit_memory`) and the coding-scenario tool set (`coding_memory_*`) do not exist here:
//! this crate stores team memory as records, not as per-member Markdown files, so there is
//! no path, line range, or old/new text to address.
//! The `MEMORY_ENABLED` environment switch is also absent — the caller decides whether to
//! build a toolkit at all — as is the Python global tool registry and the
//! `MemoryIndexManager` lifecycle; the toolkit is a plain value over an injected
//! [`TeamMemory`].
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Map, Value};

use crate::record::CallOptions;
use crate::team::model::{
    NewTeamMemory, TeamMemoryFilter, TeamMemoryKind, TeamMemoryRecord, TeamMemoryScope,
    TeamMemoryStats, TeamVisibility, TEAM_WIDE_MEMBER_KEY,
};
use crate::team::store::{
    TeamMemory, TeamSearchOptions, DEFAULT_SEARCH_THRESHOLD, DEFAULT_SEARCH_TOP_K,
};

/// The tool that stores one team memory.
pub const TOOL_REMEMBER: &str = "team_memory_remember";
/// The tool that searches the team memories the caller may read.
pub const TOOL_SEARCH: &str = "team_memory_search";
/// The tool that deletes one team memory.
pub const TOOL_FORGET: &str = "team_memory_forget";
/// The tool that aggregates the team memories the caller may read.
pub const TOOL_STATS: &str = "team_memory_stats";

/// The id segment used by the tool schemas when the context names no member.
const TEAM_WIDE_ID_LABEL: &str = "team";
/// The message of every read-only refusal.
const READ_ONLY_REASON: &str = "member memory toolkit is read-only for this member";

const ARG_CONTENT: &str = "content";
const ARG_TITLE: &str = "title";
const ARG_KIND: &str = "kind";
const ARG_TAGS: &str = "tags";
const ARG_IMPORTANCE: &str = "importance";
const ARG_MEMBER_ID: &str = "member_id";
const ARG_VISIBILITY: &str = "visibility";
const ARG_EXPIRES_AT: &str = "expires_at";
const ARG_QUERY: &str = "query";
const ARG_MAX_RESULTS: &str = "max_results";
const ARG_MIN_SCORE: &str = "min_score";
const ARG_KINDS: &str = "kinds";
const ARG_INCLUDE_EXPIRED: &str = "include_expired";
const ARG_ID: &str = "id";

/// The largest `max_results` a search may request.
const MAX_SEARCH_RESULTS: usize = 100;

const REMEMBER_DESCRIPTION: &str = "Store one team memory (a decision, task, artifact, \
handoff note, member profile, or shared fact) in the team memory store. The record is \
authored by the calling member unless `member_id` says otherwise.";
const SEARCH_DESCRIPTION: &str = "Search the team memories this member may read: \
team-visible records, its own private records, and member-list records that name it. Hits \
are ranked by keyword relevance, then importance, then recency.";
const FORGET_DESCRIPTION: &str = "Delete one team memory by id. A member may delete \
team-visible records and its own records; another member's private record is reported as \
not found.";
const STATS_DESCRIPTION: &str = "Report how many team memories this member may read, \
broken down by kind and author, together with how many of them have expired.";

/// Who is calling the team memory tools, and whether it may write.
///
/// `team_id` is the Python team scope id, `member_id` the acting member (`None` is the
/// team-wide/leader view, as in [`TeamMemoryScope`]), and `session_id` the coordination
/// session stamped onto written records and narrowed by searches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberMemoryContext {
    /// The team whose memory is addressed.
    pub team_id: String,
    /// The acting member; `None` is the team-wide view.
    pub member_id: Option<String>,
    /// The coordination session of the caller, when there is one.
    pub session_id: Option<String>,
    /// Whether remember/forget are refused.
    pub read_only: bool,
}

impl MemberMemoryContext {
    /// A writable team-wide context for `team_id` (no member, no session).
    #[must_use]
    pub fn new(team_id: impl Into<String>) -> Self {
        Self {
            team_id: team_id.into(),
            member_id: None,
            session_id: None,
            read_only: false,
        }
    }
    /// Returns the same context bound to the acting member.
    #[must_use]
    pub fn with_member(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = Some(member_id.into());
        self
    }
    /// Returns the same context bound to the coordination session.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Returns the same context with writes refused.
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
    /// The scope this context reads and writes through.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the context ids break the shared component rules (blank,
    /// NUL, `/`, `\`, control characters, or longer than 128 characters).
    pub fn scope(&self) -> Result<TeamMemoryScope> {
        let mut scope = TeamMemoryScope::new(self.team_id.as_str());
        if let Some(member) = self.member_id.as_deref() {
            scope = scope.with_member(member);
        }
        if let Some(session) = self.session_id.as_deref() {
            scope = scope.with_session(session);
        }
        scope.validate()?;
        Ok(scope)
    }
    /// The member label used by the tool schema ids.
    fn member_label(&self) -> &str {
        self.member_id.as_deref().unwrap_or(TEAM_WIDE_ID_LABEL)
    }
}

/// The model-facing team memory tools of one member.
///
/// Construct it with the team's [`TeamMemory`] and the calling member's
/// [`MemberMemoryContext`]; the toolkit owns no state beyond those two handles, so a
/// runtime may cache one per member and rebuild it freely.
pub struct MemberMemoryToolkit {
    memory: Arc<TeamMemory>,
    context: MemberMemoryContext,
}

impl MemberMemoryToolkit {
    /// Binds the tool surface to `memory` for the caller described by `context`.
    #[must_use]
    pub fn new(memory: Arc<TeamMemory>, context: MemberMemoryContext) -> Self {
        Self { memory, context }
    }
    /// The team memory facade the tools call.
    #[must_use]
    pub fn memory(&self) -> &Arc<TeamMemory> {
        &self.memory
    }
    /// The calling member and its scope.
    #[must_use]
    pub fn context(&self) -> &MemberMemoryContext {
        &self.context
    }
    /// The tool names, in the order [`MemberMemoryToolkit::get_tool_schemas`] returns them.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&'static str> {
        vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
    }
    /// The Python-shaped tool schemas advertised to a model.
    ///
    /// Each object is `{"id", "name", "description", "input_schema"}` where `id` is
    /// `team_memory.<team>.<member>.<name>` (`team` replaces an absent member, mirroring
    /// the Python `memory.<team>.<member>.<tool>` prefix) and `input_schema` is a JSON
    /// Schema object with a `description` on every property.
    #[must_use]
    pub fn get_tool_schemas(&self) -> Vec<Value> {
        let prefix = format!(
            "team_memory.{}.{}",
            self.context.team_id,
            self.context.member_label()
        );
        vec![
            remember_schema(&prefix),
            search_schema(&prefix),
            forget_schema(&prefix),
            stats_schema(&prefix),
        ]
    }
    /// Dispatches one tool call and returns the success envelope.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unknown tool name or a bad argument,
    /// [`Error::Unsupported`] when a read-only context calls remember/forget,
    /// [`Error::NotFound`] when forget targets a record the caller may not see or delete,
    /// and any store/model failure unchanged.
    pub async fn handle_tool_call(&self, name: &str, args: Value) -> Result<Value> {
        match name {
            TOOL_REMEMBER => self.remember(args).await,
            TOOL_SEARCH => self.search(args).await,
            TOOL_FORGET => self.forget(args).await,
            TOOL_STATS => self.stats(args).await,
            other => Err(Error::validation(format!(
                "unknown team memory tool `{other}` (expected one of `{TOOL_REMEMBER}`, \
                 `{TOOL_SEARCH}`, `{TOOL_FORGET}`, `{TOOL_STATS}`)"
            ))),
        }
    }
    /// Renders a failure as the `ok: false` envelope of the tool protocol.
    ///
    /// `kind` is the [`openjiuwen_common::error::ErrorKind`] name (`"validation"`,
    /// `"not_found"`, `"unsupported"`, ...) and `message` is the error payload without the
    /// kind prefix, so a protocol layer can report the same failure it would otherwise
    /// return as `Err`.
    #[must_use]
    pub fn error_envelope(name: &str, error: &Error) -> Value {
        json!({
            "ok": false,
            "tool": name,
            "error": {
                "kind": error.kind().as_str(),
                "message": error.message(),
            },
        })
    }
    /// `team_memory_remember`: stores one team memory.
    async fn remember(&self, raw: Value) -> Result<Value> {
        self.require_writable()?;
        let mut args = Args::new(TOOL_REMEMBER, raw)?;
        let content = args.required_string(ARG_CONTENT)?;
        let title = args.optional_string(ARG_TITLE)?;
        let kind = args
            .optional_string(ARG_KIND)?
            .map_or_else(TeamMemoryKind::default, |raw| TeamMemoryKind::parse(&raw));
        let tags = args.optional_strings(ARG_TAGS)?.unwrap_or_default();
        let importance = args
            .optional_float(ARG_IMPORTANCE)?
            .map(|value| check_score(&args, ARG_IMPORTANCE, value))
            .transpose()?;
        let member_id = args
            .optional_string(ARG_MEMBER_ID)?
            .or_else(|| self.context.member_id.clone());
        let visibility = args
            .take(ARG_VISIBILITY)
            .map(|value| parse_visibility(&args, &value))
            .transpose()?
            .unwrap_or_default();
        let expires_at = args
            .optional_string(ARG_EXPIRES_AT)?
            .map(|raw| parse_timestamp(&args, ARG_EXPIRES_AT, &raw))
            .transpose()?;
        args.finish()?;
        let mut input = NewTeamMemory::new(self.context.team_id.as_str(), content, kind);
        input.title = title;
        input.tags = tags;
        input.importance = importance;
        input.member_id = member_id;
        input.session_id = self.context.session_id.clone();
        input.visibility = visibility;
        input.expires_at = expires_at;
        let record = self.memory.remember(input, &CallOptions::default()).await?;
        Ok(success(TOOL_REMEMBER, to_value(&record)?))
    }
    /// `team_memory_search`: scores the records the caller may read.
    async fn search(&self, raw: Value) -> Result<Value> {
        let mut args = Args::new(TOOL_SEARCH, raw)?;
        let query = args.required_string(ARG_QUERY)?;
        let requested = args.optional_usize(ARG_MAX_RESULTS)?;
        let top_k = resolve_top_k(&args, requested)?;
        let min_score = args
            .optional_float(ARG_MIN_SCORE)?
            .map(|value| check_score(&args, ARG_MIN_SCORE, value))
            .transpose()?
            .unwrap_or(DEFAULT_SEARCH_THRESHOLD);
        let kinds = args.optional_strings(ARG_KINDS)?;
        let tags = args.optional_strings(ARG_TAGS)?;
        let include_expired = args.optional_bool(ARG_INCLUDE_EXPIRED)?.unwrap_or(false);
        args.finish()?;
        let mut options = TeamSearchOptions::default()
            .with_top_k(top_k)
            .with_threshold(min_score);
        if let Some(kinds) = kinds {
            options = options.with_kinds(kinds.iter().map(|kind| TeamMemoryKind::parse(kind)));
        }
        if let Some(tags) = tags {
            options = options.with_tags(tags);
        }
        if include_expired {
            options = options.include_expired();
        }
        let scope = self.context.scope()?;
        let hits = self.memory.search(&scope, &query, &options).await?;
        Ok(success(
            TOOL_SEARCH,
            json!({"count": hits.len(), "results": to_value(&hits)?}),
        ))
    }
    /// `team_memory_forget`: deletes one record the caller may see and delete.
    async fn forget(&self, raw: Value) -> Result<Value> {
        self.require_writable()?;
        let mut args = Args::new(TOOL_FORGET, raw)?;
        let id = args.required_string(ARG_ID)?;
        args.finish()?;
        let team_id = self.context.team_id.as_str();
        let scope = self.context.scope()?;
        let Some(record) = self.memory.get(team_id, &id).await? else {
            return Ok(success(TOOL_FORGET, json!({"deleted": false, "id": id})));
        };
        if !self.may_delete(&scope, &record) {
            return Err(Error::not_found(format!(
                "team memory `{id}` not found in team `{team_id}`"
            )));
        }
        let deleted = self.memory.forget(team_id, &id).await?;
        Ok(success(TOOL_FORGET, json!({"deleted": deleted, "id": id})))
    }
    /// `team_memory_stats`: aggregates the records the caller may read.
    async fn stats(&self, raw: Value) -> Result<Value> {
        let args = Args::new(TOOL_STATS, raw)?;
        args.finish()?;
        let scope = self.context.scope()?;
        let mut filter = TeamMemoryFilter::for_team(self.context.team_id.as_str());
        filter.include_expired = true;
        let records = self.memory.list(&filter).await?;
        let stats = visible_stats(&records, &scope);
        Ok(success(TOOL_STATS, to_value(&stats)?))
    }
    /// Refuses a write on a read-only context.
    fn require_writable(&self) -> Result<()> {
        if self.context.read_only {
            Err(Error::unsupported(READ_ONLY_REASON))
        } else {
            Ok(())
        }
    }
    /// Whether the caller may delete `record`.
    ///
    /// The record must be readable by `scope`; a member-scoped caller may then delete
    /// team-visible records and its own records only, so another member's private or
    /// member-list record stays invisible and undeletable.
    fn may_delete(&self, scope: &TeamMemoryScope, record: &TeamMemoryRecord) -> bool {
        if !scope.allows(record) {
            return false;
        }
        self.context.member_id.as_deref().is_none_or(|member| {
            record.visibility == TeamVisibility::Team || record.member_id.as_deref() == Some(member)
        })
    }
}

/// The success envelope of the tool protocol.
fn success(tool: &str, result: Value) -> Value {
    json!({"ok": true, "tool": tool, "result": result})
}

/// Serializes a payload into the envelope, mapping a serde failure to [`Error::External`].
fn to_value<T: serde::Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(Error::external)
}

/// Aggregates the records of `records` that `scope` allows.
///
/// [`TeamMemoryFilter`] cannot express the visibility union a member sees, so the caller's
/// view is selected with [`TeamMemoryScope::allows`] and aggregated here.
/// `total`, `by_kind`, `by_member`, `oldest`, and `newest` count the live records; expired
/// records are counted in `expired` only, so `total` is the number of memories the member
/// can still recall.
fn visible_stats(records: &[TeamMemoryRecord], scope: &TeamMemoryScope) -> TeamMemoryStats {
    let now = Utc::now();
    let mut stats = TeamMemoryStats::default();
    for record in records.iter().filter(|record| scope.allows(record)) {
        if record.is_expired(now) {
            stats.expired += 1;
            continue;
        }
        stats.total += 1;
        *stats
            .by_kind
            .entry(record.kind.as_str().to_owned())
            .or_insert(0) += 1;
        let member = record
            .member_id
            .clone()
            .unwrap_or_else(|| TEAM_WIDE_MEMBER_KEY.to_owned());
        *stats.by_member.entry(member).or_insert(0) += 1;
        stats.oldest = Some(
            stats
                .oldest
                .map_or(record.created_at, |oldest| oldest.min(record.created_at)),
        );
        stats.newest = Some(
            stats
                .newest
                .map_or(record.created_at, |newest| newest.max(record.created_at)),
        );
    }
    stats
}

/// Rejects a non-finite or out-of-range score and narrows it to `f32`.
fn check_score(args: &Args, field: &str, value: f64) -> Result<f32> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(args.fail(
            field,
            format!("must be a finite number within 0.0..=1.0 (got {value})"),
        ));
    }
    Ok(value as f32)
}

/// Resolves the `max_results` argument into a search `top_k`.
fn resolve_top_k(args: &Args, requested: Option<usize>) -> Result<usize> {
    match requested {
        None => Ok(DEFAULT_SEARCH_TOP_K),
        Some(value) if (1..=MAX_SEARCH_RESULTS).contains(&value) => Ok(value),
        Some(value) => Err(args.fail(
            ARG_MAX_RESULTS,
            format!("must be within 1..={MAX_SEARCH_RESULTS} (got {value})"),
        )),
    }
}

/// Parses the `visibility` argument.
fn parse_visibility(args: &Args, value: &Value) -> Result<TeamVisibility> {
    match value {
        Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "team" => Ok(TeamVisibility::Team),
            "private" => Ok(TeamVisibility::Private),
            _ => Err(args.fail(
                ARG_VISIBILITY,
                format!(
                    "must be \"team\", \"private\", or an object like \
                     {{\"members\": [\"alice\"]}} (got \"{raw}\")"
                ),
            )),
        },
        Value::Object(map) => {
            let Some(members) = map.get("members") else {
                return Err(args.fail(
                    ARG_VISIBILITY,
                    "object must hold a `members` array of member ids",
                ));
            };
            let Some(members) = members.as_array() else {
                return Err(args.fail(
                    ARG_VISIBILITY,
                    format!(
                        "`members` must be an array of member ids (got {})",
                        type_name(members)
                    ),
                ));
            };
            let mut ids = Vec::with_capacity(members.len());
            for member in members {
                match member {
                    Value::String(raw) => ids.push(raw.clone()),
                    other => {
                        return Err(args.fail(
                            ARG_VISIBILITY,
                            format!(
                                "`members` must be an array of member ids (got an entry {})",
                                type_name(other)
                            ),
                        ));
                    }
                }
            }
            let visibility = TeamVisibility::members(ids);
            visibility.validate()?;
            Ok(visibility)
        }
        other => Err(args.fail(
            ARG_VISIBILITY,
            format!(
                "must be \"team\", \"private\", or a `members` object (got {})",
                type_name(other)
            ),
        )),
    }
}

/// Parses the `expires_at` argument as an RFC 3339 timestamp.
fn parse_timestamp(args: &Args, field: &str, raw: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|parsed| parsed.with_timezone(&Utc))
        .map_err(|error| {
            args.fail(
                field,
                format!("must be an RFC 3339 timestamp (got \"{raw}\": {error})"),
            )
        })
}

/// The JSON type name used in argument errors.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The JSON arguments of one tool call, consumed field by field.
///
/// Every accessor removes its key, so [`Args::finish`] can report a name the tool does not
/// accept instead of silently dropping a misspelled argument.
struct Args {
    tool: &'static str,
    values: Map<String, Value>,
}

impl Args {
    /// Wraps the raw arguments, which must be a JSON object.
    fn new(tool: &'static str, raw: Value) -> Result<Self> {
        match raw {
            Value::Object(values) => Ok(Self { tool, values }),
            other => Err(Error::validation(format!(
                "team memory tool `{tool}` arguments must be a JSON object (got {})",
                type_name(&other)
            ))),
        }
    }
    /// Builds a validation error naming `field`.
    fn fail(&self, field: &str, detail: impl std::fmt::Display) -> Error {
        Error::validation(format!(
            "team memory tool `{}` `{field}` {detail}",
            self.tool
        ))
    }
    /// Removes the raw value of `field`.
    fn take(&mut self, field: &str) -> Option<Value> {
        self.values.remove(field)
    }
    /// Reads a required, non-blank string (whitespace is trimmed).
    fn required_string(&mut self, field: &str) -> Result<String> {
        let Some(value) = self.take(field) else {
            return Err(self.fail(field, "is required"));
        };
        match value {
            Value::String(raw) if !raw.trim().is_empty() => Ok(raw.trim().to_owned()),
            Value::String(_) => Err(self.fail(field, "must not be blank")),
            other => Err(self.fail(
                field,
                format!("must be a string (got {})", type_name(&other)),
            )),
        }
    }
    /// Reads an optional string; `null` and blank strings mean "absent".
    fn optional_string(&mut self, field: &str) -> Result<Option<String>> {
        match self.take(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(raw)) if raw.trim().is_empty() => Ok(None),
            Some(Value::String(raw)) => Ok(Some(raw.trim().to_owned())),
            Some(other) => Err(self.fail(
                field,
                format!("must be a string (got {})", type_name(&other)),
            )),
        }
    }
    /// Reads an optional array of strings.
    fn optional_strings(&mut self, field: &str) -> Result<Option<Vec<String>>> {
        match self.take(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Array(items)) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::String(raw) => values.push(raw.trim().to_owned()),
                        other => {
                            return Err(self.fail(
                                field,
                                format!(
                                    "must be an array of strings (got an entry {})",
                                    type_name(&other)
                                ),
                            ));
                        }
                    }
                }
                Ok(Some(values))
            }
            Some(other) => Err(self.fail(
                field,
                format!("must be an array of strings (got {})", type_name(&other)),
            )),
        }
    }
    /// Reads an optional number; a numeric string is accepted for model leniency.
    fn optional_float(&mut self, field: &str) -> Result<Option<f64>> {
        match self.take(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => Ok(number.as_f64()),
            Some(Value::String(raw)) => raw
                .trim()
                .parse::<f64>()
                .map(Some)
                .map_err(|_| self.fail(field, format!("must be a number (got \"{raw}\")"))),
            Some(other) => Err(self.fail(
                field,
                format!("must be a number (got {})", type_name(&other)),
            )),
        }
    }
    /// Reads an optional non-negative integer.
    fn optional_usize(&mut self, field: &str) -> Result<Option<usize>> {
        match self.take(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => number
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .map(Some)
                .ok_or_else(|| self.fail(field, "must be a non-negative integer")),
            Some(Value::String(raw)) => raw.trim().parse::<usize>().map(Some).map_err(|_| {
                self.fail(
                    field,
                    format!("must be a non-negative integer (got \"{raw}\")"),
                )
            }),
            Some(other) => Err(self.fail(
                field,
                format!("must be a non-negative integer (got {})", type_name(&other)),
            )),
        }
    }
    /// Reads an optional boolean.
    fn optional_bool(&mut self, field: &str) -> Result<Option<bool>> {
        match self.take(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(value)) => Ok(Some(value)),
            Some(other) => Err(self.fail(
                field,
                format!("must be a boolean (got {})", type_name(&other)),
            )),
        }
    }
    /// Rejects every argument name the tool does not accept.
    fn finish(self) -> Result<()> {
        if let Some(field) = self.values.keys().next() {
            return Err(Error::validation(format!(
                "team memory tool `{}` does not accept the `{field}` argument",
                self.tool
            )));
        }
        Ok(())
    }
}

/// Builds one tool schema object.
fn schema(
    prefix: &str,
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
) -> Value {
    json!({
        "id": format!("{prefix}.{name}"),
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}

/// The `team_memory_remember` schema.
fn remember_schema(prefix: &str) -> Value {
    schema(
        prefix,
        TOOL_REMEMBER,
        REMEMBER_DESCRIPTION,
        json!({
            ARG_CONTENT: {
                "type": "string",
                "description": "The memory text to store (required, non-blank).",
            },
            ARG_TITLE: {
                "type": "string",
                "description": "Optional short title; titles are searchable too.",
            },
            ARG_KIND: {
                "type": "string",
                "description": "One of `shared_fact` (default), `member_profile`, `decision`, \
                                `task`, `artifact`, `handoff_note`, or a custom kind.",
            },
            ARG_TAGS: {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for later filtering and search.",
            },
            ARG_IMPORTANCE: {
                "type": "number",
                "description": "Importance inside 0.0..=1.0; defaults to the store default.",
            },
            ARG_MEMBER_ID: {
                "type": "string",
                "description": "Author member id; defaults to the calling member.",
            },
            ARG_VISIBILITY: {
                "type": ["string", "object"],
                "description": "`\"team\"` (default), `\"private\"`, or \
                                `{\"members\": [\"alice\", \"bob\"]}`.",
            },
            ARG_EXPIRES_AT: {
                "type": "string",
                "description": "Optional RFC 3339 expiry; expired records leave search and stats.",
            },
        }),
        &[ARG_CONTENT],
    )
}

/// The `team_memory_search` schema.
fn search_schema(prefix: &str) -> Value {
    schema(
        prefix,
        TOOL_SEARCH,
        SEARCH_DESCRIPTION,
        json!({
            ARG_QUERY: {
                "type": "string",
                "description": "Keyword query text (required, non-blank).",
            },
            ARG_MAX_RESULTS: {
                "type": "integer",
                "description": "Maximum number of hits, 1..=100 (default 5).",
            },
            ARG_MIN_SCORE: {
                "type": "number",
                "description": "Minimum blended score inside 0.0..=1.0 (default 0.3).",
            },
            ARG_KINDS: {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict the search to these record kinds.",
            },
            ARG_TAGS: {
                "type": "array",
                "items": {"type": "string"},
                "description": "Require every one of these tags (case-insensitive).",
            },
            ARG_INCLUDE_EXPIRED: {
                "type": "boolean",
                "description": "Include records whose expiry has passed (default false).",
            },
        }),
        &[ARG_QUERY],
    )
}

/// The `team_memory_forget` schema.
fn forget_schema(prefix: &str) -> Value {
    schema(
        prefix,
        TOOL_FORGET,
        FORGET_DESCRIPTION,
        json!({
            ARG_ID: {
                "type": "string",
                "description": "Id of the team memory record to delete (required).",
            },
        }),
        &[ARG_ID],
    )
}

/// The `team_memory_stats` schema.
fn stats_schema(prefix: &str) -> Value {
    schema(prefix, TOOL_STATS, STATS_DESCRIPTION, json!({}), &[])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::team::model::TeamSearchHit;

    const TEAM: &str = "review-team";
    const ALICE: &str = "alice";
    const BOB: &str = "bob";

    fn shared_memory() -> Arc<TeamMemory> {
        Arc::new(TeamMemory::in_memory())
    }

    fn toolkit_on(memory: &Arc<TeamMemory>, member: Option<&str>) -> MemberMemoryToolkit {
        let context = match member {
            Some(member) => MemberMemoryContext::new(TEAM).with_member(member),
            None => MemberMemoryContext::new(TEAM),
        };
        MemberMemoryToolkit::new(Arc::clone(memory), context)
    }

    fn alice_on(memory: &Arc<TeamMemory>) -> MemberMemoryToolkit {
        toolkit_on(memory, Some(ALICE))
    }

    fn bob_on(memory: &Arc<TeamMemory>) -> MemberMemoryToolkit {
        toolkit_on(memory, Some(BOB))
    }

    async fn call(toolkit: &MemberMemoryToolkit, tool: &str, args: Value) -> Value {
        toolkit
            .handle_tool_call(tool, args)
            .await
            .expect("the tool call succeeds")
    }

    async fn call_err(toolkit: &MemberMemoryToolkit, tool: &str, args: Value) -> Error {
        toolkit
            .handle_tool_call(tool, args)
            .await
            .expect_err("the tool call fails")
    }

    fn record_of(envelope: &Value) -> TeamMemoryRecord {
        serde_json::from_value(envelope["result"].clone()).expect("the stored record")
    }

    fn stats_of(envelope: &Value) -> TeamMemoryStats {
        serde_json::from_value(envelope["result"].clone()).expect("the stats")
    }

    fn required_of(schema: &Value) -> Vec<String> {
        schema["input_schema"]["required"]
            .as_array()
            .expect("a required array")
            .iter()
            .map(|item| item.as_str().expect("a required name").to_owned())
            .collect()
    }

    #[test]
    fn schemas_are_python_shaped_and_ordered() {
        let memory = shared_memory();
        let toolkit = alice_on(&memory);
        let schemas = toolkit.get_tool_schemas();
        let names = toolkit.tool_names();
        assert_eq!(
            names,
            vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
        );
        assert_eq!(schemas.len(), names.len());
        for (schema, name) in schemas.iter().zip(names.iter()) {
            assert_eq!(schema["name"], json!(name));
            assert_eq!(
                schema["id"],
                json!(format!("team_memory.{TEAM}.{ALICE}.{name}"))
            );
            assert_eq!(schema["input_schema"]["type"], json!("object"));
            assert!(schema["description"]
                .as_str()
                .is_some_and(|text| !text.is_empty()));
            let properties = schema["input_schema"]["properties"]
                .as_object()
                .expect("a properties object");
            for (field, property) in properties {
                assert!(
                    property["description"]
                        .as_str()
                        .is_some_and(|text| !text.is_empty()),
                    "property `{field}` of `{name}` needs a description"
                );
            }
        }
        assert_eq!(required_of(&schemas[0]), vec![ARG_CONTENT.to_owned()]);
        assert_eq!(required_of(&schemas[1]), vec![ARG_QUERY.to_owned()]);
        assert_eq!(required_of(&schemas[2]), vec![ARG_ID.to_owned()]);
        assert!(required_of(&schemas[3]).is_empty());
        let remember = schemas[0]["input_schema"]["properties"]
            .as_object()
            .expect("remember properties");
        for field in [
            ARG_CONTENT,
            ARG_TITLE,
            ARG_KIND,
            ARG_TAGS,
            ARG_IMPORTANCE,
            ARG_MEMBER_ID,
            ARG_VISIBILITY,
            ARG_EXPIRES_AT,
        ] {
            assert!(remember.contains_key(field), "`{field}` is missing");
        }
        let team_wide = toolkit_on(&memory, None);
        assert_eq!(
            team_wide.get_tool_schemas()[0]["id"],
            json!("team_memory.review-team.team.team_memory_remember")
        );
    }

    #[tokio::test]
    async fn remember_stores_member_scoped_records() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let envelope = call(
            &alice,
            TOOL_REMEMBER,
            json!({
                "content": "Use SQLite for the shared store",
                "title": "Storage choice",
                "kind": "decision",
                "tags": ["Storage", "sqlite"],
                "importance": 0.8,
                "visibility": "team",
            }),
        )
        .await;
        assert_eq!(envelope["ok"], json!(true));
        assert_eq!(envelope["tool"], json!(TOOL_REMEMBER));
        let record = record_of(&envelope);
        assert_eq!(record.team_id, TEAM);
        assert_eq!(record.member_id.as_deref(), Some(ALICE));
        assert_eq!(record.kind, TeamMemoryKind::Decision);
        assert_eq!(record.title.as_deref(), Some("Storage choice"));
        assert_eq!(record.tags, vec!["storage".to_owned(), "sqlite".to_owned()]);
        assert!((record.importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(record.visibility, TeamVisibility::Team);
        assert_eq!(record.session_id, None);
        let search = call(&alice, TOOL_SEARCH, json!({"query": "sqlite storage"})).await;
        assert_eq!(search["result"]["count"], json!(1));
        assert_eq!(search["result"]["results"][0]["id"], json!(record.id));
    }

    #[tokio::test]
    async fn remember_defaults_and_argument_validation() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let record =
            record_of(&call(&alice, TOOL_REMEMBER, json!({"content": "  ship it  "})).await);
        assert_eq!(record.content, "ship it");
        assert_eq!(record.member_id.as_deref(), Some(ALICE));
        assert_eq!(record.kind, TeamMemoryKind::SharedFact);
        assert!(record.tags.is_empty());
        assert_eq!(record.visibility, TeamVisibility::Team);
        assert!(record.title.is_none());
        let session = MemberMemoryToolkit::new(
            Arc::clone(&memory),
            MemberMemoryContext::new(TEAM)
                .with_member(ALICE)
                .with_session("s-1"),
        );
        let record =
            record_of(&call(&session, TOOL_REMEMBER, json!({"content": "session note"})).await);
        assert_eq!(record.session_id.as_deref(), Some("s-1"));
        let listed = call(
            &alice,
            TOOL_REMEMBER,
            json!({"content": "for bob", "member_id": BOB, "kind": "task"}),
        )
        .await;
        let record = record_of(&listed);
        assert_eq!(record.member_id.as_deref(), Some(BOB));
        assert_eq!(record.kind, TeamMemoryKind::Task);

        for args in [
            json!({"title": "no content"}),
            json!({"content": ""}),
            json!({"content": "   "}),
            json!({"content": 7}),
        ] {
            let error = call_err(&alice, TOOL_REMEMBER, args).await;
            assert!(matches!(error, Error::Validation(_)), "{error}");
            assert!(error.to_string().contains("content"), "{error}");
        }
        for importance in [json!(1.5), json!(-0.1), json!("NaN")] {
            let error = call_err(
                &alice,
                TOOL_REMEMBER,
                json!({"content": "x", "importance": importance}),
            )
            .await;
            assert!(matches!(error, Error::Validation(_)), "{error}");
            assert!(error.to_string().contains("importance"), "{error}");
        }
        for (args, field) in [
            (json!({"content": "x", "tags": [1, 2]}), ARG_TAGS),
            (
                json!({"content": "x", "visibility": "everyone"}),
                ARG_VISIBILITY,
            ),
            (
                json!({"content": "x", "expires_at": "tomorrow"}),
                ARG_EXPIRES_AT,
            ),
            (json!({"content": "x", "conttent": "typo"}), "conttent"),
        ] {
            let error = call_err(&alice, TOOL_REMEMBER, args).await;
            assert!(matches!(error, Error::Validation(_)), "{error}");
            assert!(error.to_string().contains(field), "{error}");
        }
        let error = call_err(
            &alice,
            TOOL_REMEMBER,
            json!({"content": "x", "visibility": {"members": []}}),
        )
        .await;
        assert!(error.to_string().contains("visibility"), "{error}");
    }

    #[test]
    fn score_validation_rejects_non_finite_values() {
        let args = Args::new(TOOL_REMEMBER, json!({})).unwrap();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.5] {
            let error = check_score(&args, ARG_IMPORTANCE, value).unwrap_err();
            assert!(error.to_string().contains("importance"), "{error}");
        }
        assert!((check_score(&args, ARG_IMPORTANCE, 0.4).unwrap() - 0.4).abs() < f32::EPSILON);
        let error = Args::new(TOOL_STATS, json!([1])).err().expect("an error");
        assert!(error.to_string().contains("JSON object"), "{error}");
    }

    #[tokio::test]
    async fn search_filters_and_validates_arguments() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let decision = record_of(
            &call(
                &alice,
                TOOL_REMEMBER,
                json!({"content": "use sqlite for the shared store", "kind": "decision", "tags": ["storage"], "importance": 0.9}),
            )
            .await,
        );
        let task = record_of(
            &call(
                &alice,
                TOOL_REMEMBER,
                json!({"content": "migrate the sqlite schema", "kind": "task"}),
            )
            .await,
        );
        let search = call(&alice, TOOL_SEARCH, json!({"query": "sqlite"})).await;
        assert_eq!(search["ok"], json!(true));
        assert_eq!(search["tool"], json!(TOOL_SEARCH));
        assert_eq!(search["result"]["count"], json!(2));
        let results = search["result"]["results"].as_array().expect("results");
        assert_eq!(results.len(), 2);
        let mut found: Vec<String> = Vec::new();
        for result in results {
            let score = result["score"].as_f64().expect("a numeric score");
            assert!((0.0..=1.0).contains(&score), "{score}");
            let hit: TeamSearchHit = serde_json::from_value(result.clone()).expect("a hit");
            found.push(hit.record.id);
        }
        found.sort();
        let mut expected = vec![decision.id.clone(), task.id.clone()];
        expected.sort();
        assert_eq!(found, expected);

        let narrowed = call(
            &alice,
            TOOL_SEARCH,
            json!({"query": "sqlite", "kinds": ["task"]}),
        )
        .await;
        assert_eq!(narrowed["result"]["count"], json!(1));
        assert_eq!(narrowed["result"]["results"][0]["id"], json!(task.id));
        let tagged = call(
            &alice,
            TOOL_SEARCH,
            json!({"query": "sqlite", "tags": ["Storage"]}),
        )
        .await;
        assert_eq!(tagged["result"]["count"], json!(1));
        assert_eq!(tagged["result"]["results"][0]["id"], json!(decision.id));
        let limited = call(
            &alice,
            TOOL_SEARCH,
            json!({"query": "sqlite", "max_results": 1}),
        )
        .await;
        assert_eq!(limited["result"]["count"], json!(1));

        for args in [
            json!({}),
            json!({"query": ""}),
            json!({"query": "sqlite", "max_results": 0}),
            json!({"query": "sqlite", "max_results": 1000}),
            json!({"query": "sqlite", "max_results": "two"}),
            json!({"query": "sqlite", "min_score": 2.0}),
            json!({"query": "sqlite", "kinds": "task"}),
            json!({"query": "sqlite", "include_expired": "yes"}),
        ] {
            let error = call_err(&alice, TOOL_SEARCH, args.clone()).await;
            assert!(matches!(error, Error::Validation(_)), "{args}: {error}");
        }
        let error = call_err(
            &alice,
            TOOL_SEARCH,
            json!({"query": "sqlite", "max_results": 0}),
        )
        .await;
        assert!(error.to_string().contains("max_results"), "{error}");
        let error = call_err(
            &alice,
            TOOL_SEARCH,
            json!({"query": "sqlite", "min_score": 2.0}),
        )
        .await;
        assert!(error.to_string().contains("min_score"), "{error}");
    }

    #[tokio::test]
    async fn private_records_of_other_members_stay_invisible() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let bob = bob_on(&memory);
        let bob_private = record_of(
            &call(
                &bob,
                TOOL_REMEMBER,
                json!({"content": "bob sqlite secret", "visibility": "private"}),
            )
            .await,
        );
        let bob_listed = record_of(
            &call(
                &bob,
                TOOL_REMEMBER,
                json!({"content": "bob sqlite note for carol", "visibility": {"members": ["carol"]}}),
            )
            .await,
        );
        let shared = record_of(
            &call(
                &bob,
                TOOL_REMEMBER,
                json!({"content": "sqlite for the shared store"}),
            )
            .await,
        );
        let search = call(&alice, TOOL_SEARCH, json!({"query": "sqlite"})).await;
        assert_eq!(search["result"]["count"], json!(1));
        assert_eq!(search["result"]["results"][0]["id"], json!(shared.id));
        for hidden in [&bob_private, &bob_listed] {
            let error = call_err(&alice, TOOL_FORGET, json!({"id": hidden.id})).await;
            assert!(matches!(error, Error::NotFound(_)), "{error}");
            assert!(error.to_string().contains(&hidden.id), "{error}");
            assert!(
                memory.get(TEAM, &hidden.id).await.unwrap().is_some(),
                "a refused delete leaves the record in place"
            );
        }
        let deleted = call(&alice, TOOL_FORGET, json!({"id": shared.id})).await;
        assert_eq!(deleted["result"]["deleted"], json!(true));
        assert_eq!(deleted["result"]["id"], json!(shared.id));
        assert!(memory.get(TEAM, &shared.id).await.unwrap().is_none());
        let own = record_of(
            &call(
                &alice,
                TOOL_REMEMBER,
                json!({"content": "alice private note", "visibility": "private"}),
            )
            .await,
        );
        let deleted = call(&alice, TOOL_FORGET, json!({"id": own.id})).await;
        assert_eq!(deleted["result"]["deleted"], json!(true));
    }

    #[tokio::test]
    async fn forget_unknown_ids_and_stats_stay_scoped() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let bob = bob_on(&memory);
        call(
            &bob,
            TOOL_REMEMBER,
            json!({"content": "bob sqlite decision", "kind": "decision", "visibility": "private"}),
        )
        .await;
        call(
            &bob,
            TOOL_REMEMBER,
            json!({"content": "shared sqlite decision", "kind": "decision"}),
        )
        .await;
        let missing = "0123456789abcdef01234567";
        let envelope = call(&alice, TOOL_FORGET, json!({"id": missing})).await;
        assert_eq!(envelope["result"], json!({"deleted": false, "id": missing}));
        let stats = stats_of(&call(&alice, TOOL_STATS, json!({})).await);
        assert_eq!(stats.total, 1);
        assert_eq!(stats.by_kind.get("decision"), Some(&1));
        assert_eq!(stats.by_member.get(BOB), Some(&1));
        assert_eq!(stats.expired, 0);
        assert!(stats.oldest.is_some());
        assert_eq!(stats.oldest, stats.newest);
        let team_wide = toolkit_on(&memory, None);
        assert_eq!(
            stats_of(&call(&team_wide, TOOL_STATS, json!({})).await).total,
            2
        );
        for args in [json!({"limit": 5}), json!({"kind": "decision"})] {
            let error = call_err(&alice, TOOL_STATS, args.clone()).await;
            assert!(matches!(error, Error::Validation(_)), "{error}");
            let field = args.as_object().unwrap().keys().next().unwrap().clone();
            assert!(error.to_string().contains(&field), "{error}");
        }
    }

    #[tokio::test]
    async fn unknown_tools_and_error_envelopes() {
        let memory = shared_memory();
        let alice = alice_on(&memory);
        let error = call_err(&alice, "team_memory_drop", json!({})).await;
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(error.to_string().contains("team_memory_drop"), "{error}");
        let envelope = MemberMemoryToolkit::error_envelope("team_memory_drop", &error);
        assert_eq!(envelope["ok"], json!(false));
        assert_eq!(envelope["tool"], json!("team_memory_drop"));
        assert_eq!(envelope["error"]["kind"], json!("validation"));
        assert!(envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("team_memory_drop")));
        let error = call_err(&alice, TOOL_SEARCH, json!(["sqlite"])).await;
        assert!(error.to_string().contains("JSON object"), "{error}");
        let duplicated = MemberMemoryToolkit::error_envelope(
            TOOL_FORGET,
            &Error::not_found("team memory `x` not found in team `review-team`"),
        );
        assert_eq!(duplicated["error"]["kind"], json!("not_found"));
        assert_eq!(
            duplicated["error"]["message"],
            json!("team memory `x` not found in team `review-team`")
        );
        let read_only = Error::unsupported(READ_ONLY_REASON);
        let envelope = MemberMemoryToolkit::error_envelope(TOOL_REMEMBER, &read_only);
        assert_eq!(envelope["error"]["kind"], json!("unsupported"));
        assert!(envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("read-only")));
    }

    #[tokio::test]
    async fn read_only_contexts_serve_reads_only() {
        let memory = shared_memory();
        let writer = alice_on(&memory);
        call(
            &writer,
            TOOL_REMEMBER,
            json!({"content": "sqlite for the shared store"}),
        )
        .await;
        let reader = MemberMemoryToolkit::new(
            Arc::clone(&memory),
            MemberMemoryContext::new(TEAM)
                .with_member(ALICE)
                .read_only(),
        );
        assert!(reader.context().read_only);
        assert_eq!(reader.context().team_id, TEAM);
        assert_eq!(
            reader.memory().default_importance(),
            TeamMemory::in_memory().default_importance()
        );
        let error = call_err(&reader, TOOL_REMEMBER, json!({"content": "anything"})).await;
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("read-only"), "{error}");
        let error = call_err(&reader, TOOL_FORGET, json!({"id": "unknown-id"})).await;
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        let search = call(&reader, TOOL_SEARCH, json!({"query": "sqlite"})).await;
        assert_eq!(search["result"]["count"], json!(1));
        assert_eq!(
            stats_of(&call(&reader, TOOL_STATS, json!({})).await).total,
            1
        );
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_toolkits_persist_and_hide_private_records() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("team-memory.db");
        let (bob_private, shared) = {
            let memory = Arc::new(TeamMemory::sqlite(&path).await.unwrap());
            let alice = alice_on(&memory);
            let bob = bob_on(&memory);
            let bob_private = record_of(
                &call(
                    &bob,
                    TOOL_REMEMBER,
                    json!({"content": "bob sqlite secret", "visibility": "private"}),
                )
                .await,
            );
            let shared = record_of(
                &call(
                    &bob,
                    TOOL_REMEMBER,
                    json!({"content": "sqlite for the shared store"}),
                )
                .await,
            );
            let search = call(&alice, TOOL_SEARCH, json!({"query": "sqlite"})).await;
            assert_eq!(search["result"]["count"], json!(1));
            assert_eq!(search["result"]["results"][0]["id"], json!(shared.id));
            let error = call_err(&alice, TOOL_FORGET, json!({"id": bob_private.id})).await;
            assert!(matches!(error, Error::NotFound(_)), "{error}");
            let deleted = call(&alice, TOOL_FORGET, json!({"id": shared.id})).await;
            assert_eq!(deleted["result"]["deleted"], json!(true));
            (bob_private, shared)
        };
        let reopened = Arc::new(TeamMemory::sqlite(&path).await.unwrap());
        let alice = alice_on(&reopened);
        let bob = bob_on(&reopened);
        assert!(reopened.get(TEAM, &shared.id).await.unwrap().is_none());
        assert!(reopened.get(TEAM, &bob_private.id).await.unwrap().is_some());
        assert_eq!(
            call(&alice, TOOL_SEARCH, json!({"query": "sqlite"})).await["result"]["count"],
            json!(0)
        );
        let search = call(&bob, TOOL_SEARCH, json!({"query": "sqlite"})).await;
        assert_eq!(search["result"]["count"], json!(1));
        assert_eq!(search["result"]["results"][0]["id"], json!(bob_private.id));
        assert_eq!(stats_of(&call(&bob, TOOL_STATS, json!({})).await).total, 1);
    }
}
