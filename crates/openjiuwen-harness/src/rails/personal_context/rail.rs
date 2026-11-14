//! The `PersonalContext` rail: attaches the caller-published root description.
//!
//! Port of the rail half of
//! `openjiuwen/harness/rails/personal_context.py`. On every model call the rail
//! refreshes the fixed `agent_use_enabled` switch, reads the published
//! `workspace/context/description.md` (regular file only, leading symlinks
//! rejected, bounded read) and publishes the rendered wrapper as the
//! `personal_context` prompt section. The section is cleared before each model
//! call and again after the model call and after the invocation, so one turn's
//! description never leaks into the next one.
//!
//! The rail never aborts a run: a missing, blank or unreadable description, a
//! disabled switch and a non-contiguous message window all simply publish
//! nothing (failures are logged with `tracing::warn!`). State lives per firing
//! point in the [`HarnessRailContext`] — the prompt-section channel and the
//! message window — plus one cached switch flag that is refreshed from disk on
//! every firing point, never a per-turn rail field. The short filesystem reads
//! in the hook bodies (one control document, one bounded description) are
//! synchronous by design and stay under caller-chosen paths.
//!
//! Scope: Python's `pcs_search` tool registration (`init`, `_sync_search_tool`,
//! `_owns_search_tool` and the ability-manager plumbing) is not ported; the
//! [`PersonalContextRail::search`] method is the equivalent capability a host
//! can wrap in a model-facing tool.
//! Size note: the ported rail body plus its behavioural test suite make this file long;
//! production code stays under ~1000 lines.
// Size note: ~620 lines of implementation plus an inline test module; the
// switch, description, contiguity and wrapper parity assertions stay beside the
// code they verify.
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, Role};
use serde::{Deserialize, Serialize};
use serde_yaml_ng::Value as YamlValue;

use crate::prompts::{PromptCategory, PromptSection};
use crate::rails::{
    publish_prompt_section, HarnessRail, HarnessRailContext, PROMPT_SECTIONS_EXTRA_KEY,
};

use super::atomic::read_bounded;
use super::search::SearchHit;
use super::status::{build_error, PersonalContextStatus};
use super::store::{PersonalContextStore, DEFAULT_SEARCH_LIMIT};

/// Dispatch priority of [`PersonalContextRail`].
///
/// The Python rail declares `priority = 40`, i.e. below
/// [`crate::rails::rail_priority::DEFAULT`] (50): the description is an
/// additional runtime attachment, not part of the base system prompt.
pub const PERSONAL_CONTEXT_PRIORITY: i32 = 40;

/// Name of the prompt section the rail publishes (Python `_SECTION`).
pub const PERSONAL_CONTEXT_SECTION: &str = "personal_context";

/// File name of the Agent-use switch inside the configured home.
pub const DEFAULT_CONFIG_FILENAME: &str = "personal_context.yaml";

/// Language key the rendered section is published under.
pub const DEFAULT_LANGUAGE: &str = "cn";

/// Characters of the description kept in the prompt wrapper.
pub const DEFAULT_MAX_DESCRIPTION_CHARS: usize = 3000;

/// Byte cap of the Agent-use switch document.
pub const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

/// Truncation notice of the default character cap ([`DEFAULT_MAX_DESCRIPTION_CHARS`]).
///
/// [`render_content`] interpolates the *configured* cap into the same sentence,
/// so a rail with a different `max_description_chars` renders a matching notice.
pub const TRUNCATION_NOTICE: &str =
    "本次仅载入前 3000 个字符；根说明文件更大，请按 description_path 继续读取。";

/// Document label used in switch error messages.
const CONFIG_LABEL: &str = "personal context config";

/// Document label used in description error messages.
const DESCRIPTION_LABEL: &str = "personal context description";

/// Reason reported when the switch is off or the search capability is detached.
const DISABLED_REASON: &str =
    "PersonalContext Agent use is disabled or the search tool is detached";

/// Reason reported for an empty search query.
const BLANK_QUERY_REASON: &str = "query must be a non-empty string";

/// Tunables of [`PersonalContextRail`].
///
/// The defaults mirror the Python module constants: the switch lives at
/// `<home>/personal_context.yaml`, the description at
/// `<home>/workspace/context/description.md`, and the section is published
/// under the `cn` language key with priority [`PERSONAL_CONTEXT_PRIORITY`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PersonalContextRailConfig {
    /// Home directory every derived path hangs off.
    pub home: PathBuf,
    /// File name of the Agent-use switch, relative to `home`.
    pub config_filename: String,
    /// Language key the rendered section is published under.
    pub language: String,
    /// Characters of the description kept in the prompt wrapper.
    pub max_description_chars: usize,
    /// Priority of the published prompt section.
    pub section_priority: i32,
    /// Name of the published prompt section.
    pub section_name: String,
}

impl Default for PersonalContextRailConfig {
    fn default() -> Self {
        Self {
            home: PathBuf::new(),
            config_filename: DEFAULT_CONFIG_FILENAME.to_string(),
            language: DEFAULT_LANGUAGE.to_string(),
            max_description_chars: DEFAULT_MAX_DESCRIPTION_CHARS,
            section_priority: PERSONAL_CONTEXT_PRIORITY,
            section_name: PERSONAL_CONTEXT_SECTION.to_string(),
        }
    }
}

impl PersonalContextRailConfig {
    /// Builds the default configuration rooted at `home`.
    #[must_use]
    pub fn new(home: impl Into<PathBuf>) -> Self {
        Self {
            home: home.into(),
            ..Self::default()
        }
    }

    /// Returns this configuration rooted at `home`.
    #[must_use]
    pub fn with_home(mut self, home: impl Into<PathBuf>) -> Self {
        self.home = home.into();
        self
    }

    /// Overrides the switch file name relative to `home`.
    #[must_use]
    pub fn with_config_filename(mut self, config_filename: impl Into<String>) -> Self {
        self.config_filename = config_filename.into();
        self
    }

    /// Overrides the language key of the published section.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Overrides the number of description characters kept in the wrapper.
    #[must_use]
    pub fn with_max_description_chars(mut self, max_description_chars: usize) -> Self {
        self.max_description_chars = max_description_chars;
        self
    }

    /// Overrides the priority of the published section.
    #[must_use]
    pub fn with_section_priority(mut self, section_priority: i32) -> Self {
        self.section_priority = section_priority;
        self
    }

    /// Overrides the name of the published section.
    #[must_use]
    pub fn with_section_name(mut self, section_name: impl Into<String>) -> Self {
        self.section_name = section_name.into();
        self
    }

    /// Returns the Agent-use switch path: `<home>/<config_filename>`.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.home.join(&self.config_filename)
    }
}

/// Injects the published `PersonalContext` root description into the prompt.
///
/// The rail owns a [`PersonalContextStore`] (and therefore the description and
/// entry paths) plus the switch configuration. One rail instance can serve
/// concurrent invocations: everything but the cached switch flag lives per
/// firing point.
pub struct PersonalContextRail {
    store: Arc<PersonalContextStore>,
    config: PersonalContextRailConfig,
    enabled: AtomicBool,
}

impl PersonalContextRail {
    /// Builds a rail over the store derived from `config.home`.
    #[must_use]
    pub fn new(config: PersonalContextRailConfig) -> Self {
        let store = Arc::new(PersonalContextStore::new(config.home.clone()));
        Self {
            store,
            config,
            enabled: AtomicBool::new(false),
        }
    }

    /// Replaces the store; the caller owns the `home` the paths hang off.
    #[must_use]
    pub fn with_store(mut self, store: Arc<PersonalContextStore>) -> Self {
        self.store = store;
        self
    }

    /// Returns the store the rail reads its description and entries from.
    #[must_use]
    pub fn store(&self) -> &PersonalContextStore {
        &self.store
    }

    /// Returns the rail configuration.
    #[must_use]
    pub fn config(&self) -> &PersonalContextRailConfig {
        &self.config
    }

    /// Returns `<home>/workspace/context`, the directory the wrapper points at.
    #[must_use]
    pub fn context_root(&self) -> PathBuf {
        self.store.context_root()
    }

    /// Returns `<home>/workspace/context/description.md`.
    #[must_use]
    pub fn description_path(&self) -> PathBuf {
        self.store.description_path()
    }

    /// Returns the Agent-use flag, re-reading the switch document first.
    ///
    /// The switch is only enabled when the document parses as a YAML mapping
    /// whose `agent_use_enabled` value is the boolean `true`; a quoted string
    /// `"true"`, a missing value, a non-mapping document, malformed YAML, any
    /// symlink in the path chain, a non-regular file and a document larger than
    /// [`MAX_CONFIG_BYTES`] all yield `false`. Failures are never errors: the
    /// switch simply stays off and a bounded `tracing::debug!` record is
    /// written (never the file contents).
    pub fn agent_use_enabled(&self) -> bool {
        self.refresh_switch()
    }

    /// Re-reads the switch document, stores the flag and returns it.
    ///
    /// Used by the `before_*` hooks so the cached flag and the returned value
    /// never disagree.
    pub fn refresh_switch(&self) -> bool {
        let path = self.config.config_path();
        let enabled = match Self::read_switch_file(&path) {
            Ok(enabled) => enabled,
            Err(error) => {
                tracing::debug!(
                    error = %error,
                    path = %path.display(),
                    "personal context switch is unreadable; staying disabled"
                );
                false
            }
        };
        self.enabled.store(enabled, Ordering::Relaxed);
        enabled
    }

    /// Runs a keyword query over the published entries.
    ///
    /// # Errors
    ///
    /// `Validation` with the Python message when Agent use is disabled or the
    /// query is blank, plus the index errors of
    /// [`PersonalContextStore::search`].
    pub fn search(&self, query: &str) -> Result<Vec<SearchHit>> {
        if !self.agent_use_enabled() {
            return Err(build_error(
                PersonalContextStatus::StateInvalid,
                DISABLED_REASON,
            ));
        }
        let query = query.trim();
        if query.is_empty() {
            return Err(build_error(
                PersonalContextStatus::StateInvalid,
                BLANK_QUERY_REASON,
            ));
        }
        self.store.search(query, DEFAULT_SEARCH_LIMIT)
    }

    /// Reads the published description, bounded and truncated.
    ///
    /// At most `(max_description_chars + 1) * 4` bytes are read through
    /// [`Read::take`], so a huge file is never loaded. One character beyond the
    /// cap is retained so [`render_content`] can tell truncation from an exact
    /// fit; the returned text is already cut to `max_description_chars + 1`
    /// characters.
    ///
    /// # Errors
    ///
    /// `Validation` when the path is a symlink or not a regular file, or when
    /// the content is not valid UTF-8; `Config` when the file cannot be read.
    pub fn description(&self) -> Result<Option<String>> {
        Ok(self
            .read_description()?
            .map(|(description, _size_bytes)| description))
    }

    /// Returns `true` when every tool-call group is adjacent to its results.
    ///
    /// Port of Python's `_messages_are_contiguous`: while a call id is pending,
    /// the next message must be a tool message answering that id; a tool
    /// message without a pending call, a blank or unknown id, a repeated id and
    /// a dangling call at the end all make the window non-contiguous.
    #[must_use]
    pub fn messages_are_contiguous(messages: &[BaseMessage]) -> bool {
        messages_are_contiguous(messages)
    }

    /// Reads the description together with the file's real byte length.
    ///
    /// The length comes from `symlink_metadata`, so it is the on-disk size and
    /// not the length of the bounded, truncated read.
    fn read_description(&self) -> Result<Option<(String, u64)>> {
        let path = self.description_path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(read_error(DESCRIPTION_LABEL, &path, &error)),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Error::validation("description path is not a regular file"));
        }

        let byte_limit = self
            .config
            .max_description_chars
            .saturating_add(1)
            .saturating_mul(4);
        let file = std::fs::File::open(&path)
            .map_err(|error| read_error(DESCRIPTION_LABEL, &path, &error))?;
        let mut bytes = Vec::new();
        file.take(u64::try_from(byte_limit).unwrap_or(u64::MAX))
            .read_to_end(&mut bytes)
            .map_err(|error| read_error(DESCRIPTION_LABEL, &path, &error))?;

        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            // The bounded read may cut a multi-byte character in half; the
            // valid prefix is still an honest description.
            Err(error) if error.utf8_error().error_len().is_none() => {
                let valid_up_to = error.utf8_error().valid_up_to();
                let mut bytes = error.into_bytes();
                bytes.truncate(valid_up_to);
                String::from_utf8(bytes)
                    .map_err(|_| Error::validation("description is not valid UTF-8"))?
            }
            Err(_) => return Err(Error::validation("description is not valid UTF-8")),
        };

        let text: String = text
            .chars()
            .take(self.config.max_description_chars.saturating_add(1))
            .collect();
        if text.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some((text, metadata.len())))
    }

    /// Reads and classifies the Agent-use switch document at `path`.
    fn read_switch_file(path: &Path) -> Result<bool> {
        if path_chain_has_symlink(path) {
            return Err(Error::validation(format!(
                "{CONFIG_LABEL} {} is a symlink",
                path.display()
            )));
        }
        let limit = usize::try_from(MAX_CONFIG_BYTES).unwrap_or(usize::MAX);
        let Some(bytes) = read_bounded(path, limit, CONFIG_LABEL)? else {
            return Ok(false);
        };
        let Ok(YamlValue::Mapping(mapping)) = serde_yaml_ng::from_slice::<YamlValue>(&bytes) else {
            return Err(Error::validation(format!(
                "{CONFIG_LABEL} {} is not a YAML mapping",
                path.display()
            )));
        };
        let key = YamlValue::String("agent_use_enabled".to_string());
        Ok(matches!(mapping.get(&key), Some(YamlValue::Bool(true))))
    }
}

impl Default for PersonalContextRail {
    fn default() -> Self {
        Self::new(PersonalContextRailConfig::default())
    }
}

impl fmt::Debug for PersonalContextRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersonalContextRail")
            .field("config", &self.config)
            .field("context_root", &self.store.context_root())
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for PersonalContextRail {
    fn name(&self) -> &str {
        PERSONAL_CONTEXT_SECTION
    }

    fn priority(&self) -> i32 {
        PERSONAL_CONTEXT_PRIORITY
    }

    async fn before_invoke(&self, _ctx: &mut HarnessRailContext) -> Result<()> {
        self.refresh_switch();
        Ok(())
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        clear_section(ctx, &self.config.section_name);
        if !self.refresh_switch() {
            return Ok(());
        }
        if ctx.base.messages.is_empty() || !messages_are_contiguous(&ctx.base.messages) {
            return Ok(());
        }

        let (description, description_size_bytes) = match self.read_description() {
            Ok(Some(read)) => read,
            Ok(None) => return Ok(()),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "personal context description read failed; publishing no section"
                );
                return Ok(());
            }
        };
        let content = render_content(
            &self.context_root(),
            &self.description_path(),
            description_size_bytes,
            &description,
            self.config.max_description_chars,
        );
        let mut languages = BTreeMap::new();
        languages.insert(self.config.language.clone(), content);
        let section = PromptSection::new(
            self.config.section_name.clone(),
            languages,
            self.config.section_priority,
        )
        .with_category(PromptCategory::Memory);
        publish_prompt_section(ctx, &section);
        Ok(())
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        clear_section(ctx, &self.config.section_name);
        Ok(())
    }

    async fn after_invoke(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        clear_section(ctx, &self.config.section_name);
        Ok(())
    }
}

/// Removes only `name` from the prompt-section channel of `ctx`.
///
/// The channel is a JSON object keyed by section name, so the rail can drop its
/// own entry without touching sections published by other rails; a missing or
/// non-object channel is left alone.
fn clear_section(ctx: &mut HarnessRailContext, name: &str) {
    let Some(value) = ctx.extra_mut().get_mut(PROMPT_SECTIONS_EXTRA_KEY) else {
        return;
    };
    let Some(entries) = value.as_object_mut() else {
        return;
    };
    entries.remove(name);
}

/// Builds the `Config` error of a failed read of `label` at `path`.
fn read_error(label: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::config(format!(
        "failed to read {label} {}: {error}",
        path.display()
    ))
}

/// Returns `true` when `path` or any of its ancestors is a symbolic link.
///
/// Mirrors Python's `is_symlink()` walk up to the filesystem root: a missing
/// component is not a symlink, and the walk never follows a link.
fn path_chain_has_symlink(path: &Path) -> bool {
    let mut current = Some(path);
    while let Some(candidate) = current {
        let is_symlink = std::fs::symlink_metadata(candidate)
            .is_ok_and(|metadata| metadata.file_type().is_symlink());
        if is_symlink {
            return true;
        }
        current = candidate
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
    }
    false
}

/// Returns `true` when every tool-call group is adjacent to its results.
///
/// Free-function form of [`PersonalContextRail::messages_are_contiguous`] and
/// the public entry point of the guard: the rail only publishes the description
/// for a window the model accepted, because a description inserted between a
/// call and its result would corrupt the transcript.
#[must_use]
pub fn messages_are_contiguous(messages: &[BaseMessage]) -> bool {
    let mut seen: HashSet<String> = HashSet::new();
    let mut pending: HashSet<String> = HashSet::new();

    for message in messages {
        if !pending.is_empty() {
            if message.role != Role::Tool {
                return false;
            }
            let Some(tool_call_id) = message.tool_call_id.as_deref() else {
                return false;
            };
            if tool_call_id.trim().is_empty() || !pending.remove(tool_call_id) {
                return false;
            }
            continue;
        }

        if message.role == Role::Tool {
            return false;
        }
        if message.role != Role::Assistant || message.tool_calls.is_empty() {
            continue;
        }

        let mut current: Vec<&str> = Vec::with_capacity(message.tool_calls.len());
        for call in &message.tool_calls {
            let Some(tool_call_id) = call.id.as_deref() else {
                return false;
            };
            if tool_call_id.trim().is_empty()
                || seen.contains(tool_call_id)
                || current.contains(&tool_call_id)
            {
                return false;
            }
            current.push(tool_call_id);
        }
        for tool_call_id in current {
            seen.insert(tool_call_id.to_string());
            pending.insert(tool_call_id.to_string());
        }
    }

    pending.is_empty()
}

/// Renders the fixed, non-user-request attachment wrapper around `description`.
///
/// `description_size_bytes` is the file's real byte length (not the truncated
/// read length) and `description` is cut to `max_description_chars` characters;
/// when the supplied text is longer than that cap, the truncation notice for
/// `max_description_chars` is appended in brackets.
#[must_use]
pub fn render_content(
    context_root: &Path,
    description_path: &Path,
    description_size_bytes: u64,
    description: &str,
    max_description_chars: usize,
) -> String {
    let mut body: String = description.chars().take(max_description_chars).collect();
    if description.chars().count() > max_description_chars {
        body.push_str("\n\n[");
        body.push_str(&truncation_notice(max_description_chars));
        body.push(']');
    }
    format!(
        "# 主动上下文\n\n\
         这是当前模型调用的临时运行时附件，不是新的用户请求；仅在与当前任务相关时使用。\n\n\
         - context_root: `{}`\n\
         - description_path: `{}`\n\
         - description_size_bytes: `{}`\n\
         - filesystem access: 从顶层 description.md 开始，按其中相对链接继续读取。\n\n\
         ## 当前上下文说明\n\n\
         {}",
        context_root.display(),
        description_path.display(),
        description_size_bytes,
        body
    )
}

/// Returns the truncation notice for a `max_description_chars` cap.
///
/// [`TRUNCATION_NOTICE`] is this sentence rendered for
/// [`DEFAULT_MAX_DESCRIPTION_CHARS`].
fn truncation_notice(max_description_chars: usize) -> String {
    format!("本次仅载入前 {max_description_chars} 个字符；根说明文件更大，请按 description_path 继续读取。")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::super::ingest::LocalDirectoryConfig;
    use super::*;
    use crate::rails::{prompt_sections, rail_priority};
    use openjiuwen_agent::RailEvent;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::ToolCall;

    const SWITCH_ON: &str = "agent_use_enabled: true\n";

    fn write_switch(home: &Path, text: &str) {
        std::fs::write(home.join(DEFAULT_CONFIG_FILENAME), text).unwrap();
    }

    fn rail_at(home: &Path) -> PersonalContextRail {
        PersonalContextRail::new(PersonalContextRailConfig::new(home))
    }

    fn rail_with(home: &Path, config: PersonalContextRailConfig) -> PersonalContextRail {
        PersonalContextRail::new(config.with_home(home))
    }

    fn enabled_rail(home: &Path) -> PersonalContextRail {
        write_switch(home, SWITCH_ON);
        rail_at(home)
    }

    fn write_description(home: &Path, text: &str) {
        let store = PersonalContextStore::new(home);
        std::fs::create_dir_all(store.context_root()).unwrap();
        std::fs::write(store.description_path(), text).unwrap();
    }

    fn call_message(ids: &[&str]) -> BaseMessage {
        let calls = ids
            .iter()
            .map(|id| ToolCall::new("read_file").with_id(*id))
            .collect();
        BaseMessage::assistant("").with_tool_calls(calls)
    }

    fn ctx_with(messages: Vec<BaseMessage>) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages = messages;
        ctx
    }

    /// Asserts exactly one section is published and returns its text.
    fn section_content<'a>(sections: &'a [PromptSection], language: &str) -> &'a str {
        assert_eq!(sections.len(), 1, "exactly one section is published");
        sections[0].content.get(language).unwrap()
    }

    #[test]
    fn construction_surface_matches_the_python_rail() {
        let rail = PersonalContextRail::default();
        assert_eq!(rail.name(), "personal_context");
        assert_eq!(rail.priority(), 40);
        assert_eq!(rail.priority(), PERSONAL_CONTEXT_PRIORITY);
        assert!(rail.priority() < rail_priority::DEFAULT);

        let defaults = PersonalContextRailConfig::default();
        assert!(defaults.home.as_os_str().is_empty());
        assert_eq!(defaults.config_filename, DEFAULT_CONFIG_FILENAME);
        assert_eq!(defaults.language, DEFAULT_LANGUAGE);
        assert_eq!(
            defaults.max_description_chars,
            DEFAULT_MAX_DESCRIPTION_CHARS
        );
        assert_eq!(defaults.section_priority, PERSONAL_CONTEXT_PRIORITY);
        assert_eq!(defaults.section_name, PERSONAL_CONTEXT_SECTION);
        assert_eq!(
            defaults.config_path(),
            PathBuf::from(DEFAULT_CONFIG_FILENAME)
        );

        let config = PersonalContextRailConfig::new("/ignored")
            .with_home("/tmp/home")
            .with_config_filename("switch.yaml")
            .with_language("en")
            .with_max_description_chars(7)
            .with_section_priority(3)
            .with_section_name("ctx");
        let rail = PersonalContextRail::new(config.clone());
        assert_eq!(rail.config(), &config);
        assert_eq!(config.config_path(), PathBuf::from("/tmp/home/switch.yaml"));
        assert_eq!(
            rail.context_root(),
            PathBuf::from("/tmp/home/workspace/context")
        );
        assert_eq!(
            rail.description_path(),
            PathBuf::from("/tmp/home/workspace/context/description.md")
        );
        assert_eq!(rail.store().home(), Path::new("/tmp/home"));
        assert!(format!("{rail:?}").contains("PersonalContextRail"));
    }

    #[test]
    fn render_content_matches_the_wrapper_verbatim() {
        let rendered = render_content(
            Path::new("/home/u/workspace/context"),
            Path::new("/home/u/workspace/context/description.md"),
            12,
            "hello",
            DEFAULT_MAX_DESCRIPTION_CHARS,
        );
        let expected = "# 主动上下文\n\n\
             这是当前模型调用的临时运行时附件，不是新的用户请求；仅在与当前任务相关时使用。\n\n\
             - context_root: `/home/u/workspace/context`\n\
             - description_path: `/home/u/workspace/context/description.md`\n\
             - description_size_bytes: `12`\n\
             - filesystem access: 从顶层 description.md 开始，按其中相对链接继续读取。\n\n\
             ## 当前上下文说明\n\n\
             hello";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn render_content_cuts_at_the_cap_and_appends_the_configured_notice() {
        assert_eq!(
            truncation_notice(DEFAULT_MAX_DESCRIPTION_CHARS),
            TRUNCATION_NOTICE
        );
        assert_eq!(
            truncation_notice(4),
            "本次仅载入前 4 个字符；根说明文件更大，请按 description_path 继续读取。"
        );

        let rendered = render_content(Path::new("/r"), Path::new("/r/d.md"), 10, "abcdefghij", 4);
        assert!(rendered.ends_with(
            "## 当前上下文说明\n\nabcd\n\n[本次仅载入前 4 个字符；根说明文件更大，请按 description_path 继续读取。]"
        ));
        assert!(!rendered.contains("abcde"), "body stops at the cap");

        let exact = render_content(Path::new("/r"), Path::new("/r/d.md"), 4, "abcd", 4);
        assert!(exact.ends_with("## 当前上下文说明\n\nabcd"));
        assert!(!exact.contains("本次仅载入前"));
    }

    #[test]
    fn description_is_bounded_and_rejects_unusable_files() {
        let home = tempfile::tempdir().unwrap();
        let rail = rail_with(
            home.path(),
            PersonalContextRailConfig::default().with_max_description_chars(4),
        );
        assert!(rail.description().unwrap().is_none(), "missing file");

        write_description(home.path(), "  \n\t ");
        assert!(rail.description().unwrap().is_none(), "blank file");

        write_description(home.path(), "abcdef");
        // One extra character is retained so render_content detects truncation,
        // while the reported size stays the real file length.
        assert_eq!(rail.description().unwrap().as_deref(), Some("abcde"));
        assert_eq!(std::fs::metadata(rail.description_path()).unwrap().len(), 6);

        std::fs::write(rail.description_path(), [0xFF_u8, b'a']).unwrap();
        let error = rail.description().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("description is not valid UTF-8"));
    }

    #[cfg(unix)]
    #[test]
    fn description_rejects_a_symlink() {
        let home = tempfile::tempdir().unwrap();
        let rail = rail_at(home.path());
        write_description(home.path(), "real text");
        let target = rail.description_path().with_file_name("real.md");
        std::fs::rename(rail.description_path(), &target).unwrap();
        std::os::unix::fs::symlink(&target, rail.description_path()).unwrap();

        let error = rail.description().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("description path is not a regular file"));
    }

    #[test]
    fn switch_only_honours_a_boolean_true_mapping() {
        let home = tempfile::tempdir().unwrap();
        let rail = rail_at(home.path());
        assert!(!rail.agent_use_enabled(), "missing file");

        let cases = [
            (SWITCH_ON, true),
            ("agent_use_enabled: false\n", false),
            ("agent_use_enabled: \"true\"\n", false),
            ("agent_use_enabled: true\nother: 1\n", true),
            ("other: true\n", false),
            ("- a\n- b\n", false),
            ("agent_use_enabled: [unclosed\n", false),
            ("", false),
        ];
        for (text, expected) in cases {
            write_switch(home.path(), text);
            assert_eq!(rail.agent_use_enabled(), expected, "switch {text:?}");
        }

        // A document over the byte cap is rejected before it is parsed.
        write_switch(
            home.path(),
            &"a".repeat(usize::try_from(MAX_CONFIG_BYTES).unwrap() + 1),
        );
        assert!(!rail.agent_use_enabled(), "oversized document");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_switch_is_disabled() {
        let home = tempfile::tempdir().unwrap();
        let target = home.path().join("real.yaml");
        std::fs::write(&target, SWITCH_ON).unwrap();
        std::os::unix::fs::symlink(&target, home.path().join(DEFAULT_CONFIG_FILENAME)).unwrap();

        let rail = rail_at(home.path());
        assert!(!rail.agent_use_enabled());
    }

    #[tokio::test]
    async fn before_model_call_publishes_the_wrapper_section() {
        let home = tempfile::tempdir().unwrap();
        let rail = enabled_rail(home.path());
        write_description(home.path(), "# Root\n\nSummary.\n");

        let mut ctx = ctx_with(vec![BaseMessage::user("what do my notes say?")]);
        rail.before_model_call(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections[0].name, PERSONAL_CONTEXT_SECTION);
        assert_eq!(sections[0].priority, PERSONAL_CONTEXT_PRIORITY);
        assert_eq!(sections[0].category, Some(PromptCategory::Memory));
        let content = section_content(&sections, "cn");
        assert!(content.starts_with("# 主动上下文\n\n"));
        assert!(content.contains(&format!(
            "- context_root: `{}`",
            rail.context_root().display()
        )));
        assert!(content.contains(&format!(
            "- description_path: `{}`",
            rail.description_path().display()
        )));
        assert!(content.contains("- filesystem access: 从顶层 description.md 开始"));
        assert!(content.ends_with("## 当前上下文说明\n\n# Root\n\nSummary.\n"));
    }

    #[tokio::test]
    async fn before_model_call_uses_the_configured_language_and_cap() {
        let home = tempfile::tempdir().unwrap();
        write_switch(home.path(), SWITCH_ON);
        let rail = rail_with(
            home.path(),
            PersonalContextRailConfig::default()
                .with_language("en")
                .with_max_description_chars(4),
        );
        write_description(home.path(), "abcdef");

        let mut ctx = ctx_with(vec![BaseMessage::user("hi")]);
        rail.before_model_call(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections[0].content.keys().collect::<Vec<_>>(), ["en"]);
        let content = section_content(&sections, "en");
        assert!(content.contains("- description_size_bytes: `6`"));
        assert!(content.ends_with(
            "## 当前上下文说明\n\nabcd\n\n[本次仅载入前 4 个字符；根说明文件更大，请按 description_path 继续读取。]"
        ));
    }

    #[tokio::test]
    async fn before_model_call_publishes_nothing_unless_everything_lines_up() {
        let home = tempfile::tempdir().unwrap();
        let disabled = rail_at(home.path());
        write_description(home.path(), "# Root\n");
        let mut stale = ctx_with(vec![BaseMessage::user("hi")]);
        publish_prompt_section(
            &mut stale,
            &PromptSection::bilingual(PERSONAL_CONTEXT_SECTION, "stale", "stale", 1),
        );
        disabled.before_model_call(&mut stale).await.unwrap();
        assert!(
            prompt_sections(&stale).is_empty(),
            "disabled clears the stale section"
        );

        let rail = enabled_rail(home.path());
        let mut empty = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut empty).await.unwrap();
        assert!(prompt_sections(&empty).is_empty(), "empty window");

        let mut dangling = ctx_with(vec![call_message(&["a"]), BaseMessage::user("hi")]);
        rail.before_model_call(&mut dangling).await.unwrap();
        assert!(
            prompt_sections(&dangling).is_empty(),
            "non-contiguous window"
        );

        std::fs::remove_file(rail.description_path()).unwrap();
        let mut missing = ctx_with(vec![BaseMessage::user("hi")]);
        rail.before_model_call(&mut missing).await.unwrap();
        assert!(prompt_sections(&missing).is_empty(), "missing description");

        write_description(home.path(), "# Root\n");
        let mut contiguous = ctx_with(vec![
            BaseMessage::user("read"),
            call_message(&["a"]),
            BaseMessage::tool("a", "out"),
        ]);
        rail.before_model_call(&mut contiguous).await.unwrap();
        assert_eq!(
            prompt_sections(&contiguous).len(),
            1,
            "contiguous tool window"
        );
    }

    #[tokio::test]
    async fn after_hooks_clear_only_this_section() {
        let home = tempfile::tempdir().unwrap();
        let rail = enabled_rail(home.path());
        for event in [RailEvent::AfterModelCall, RailEvent::AfterInvoke] {
            let mut ctx = HarnessRailContext::for_event(event);
            publish_prompt_section(
                &mut ctx,
                &PromptSection::bilingual(PERSONAL_CONTEXT_SECTION, "body", "body", 40),
            );
            publish_prompt_section(
                &mut ctx,
                &PromptSection::bilingual("other_rail", "keep", "keep", 10),
            );
            if event == RailEvent::AfterInvoke {
                rail.after_invoke(&mut ctx).await.unwrap();
            } else {
                rail.after_model_call(&mut ctx).await.unwrap();
            }
            let sections = prompt_sections(&ctx);
            assert_eq!(sections.len(), 1, "at {event:?}");
            assert_eq!(sections[0].name, "other_rail");
        }

        // A missing or non-object channel is left alone.
        let mut empty = HarnessRailContext::for_event(RailEvent::AfterInvoke);
        rail.after_invoke(&mut empty).await.unwrap();
        assert!(empty.extra().get(PROMPT_SECTIONS_EXTRA_KEY).is_none());
        empty
            .extra_mut()
            .insert(PROMPT_SECTIONS_EXTRA_KEY.to_string(), serde_json::json!([]));
        rail.after_invoke(&mut empty).await.unwrap();
        assert_eq!(
            empty.extra().get(PROMPT_SECTIONS_EXTRA_KEY),
            Some(&serde_json::json!([]))
        );
    }

    #[tokio::test]
    async fn before_invoke_refreshes_the_switch() {
        let home = tempfile::tempdir().unwrap();
        let rail = rail_at(home.path());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        rail.before_invoke(&mut ctx).await.unwrap();
        assert!(!rail.agent_use_enabled());

        write_switch(home.path(), SWITCH_ON);
        rail.before_invoke(&mut ctx).await.unwrap();
        assert!(rail.agent_use_enabled());
    }

    #[test]
    fn search_requires_the_switch_and_a_query() {
        let home = tempfile::tempdir().unwrap();
        let rail = rail_at(home.path());

        let error = rail.search("notes").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: context proactive_state is invalid, reason: \
             PersonalContext Agent use is disabled or the search tool is detached"
        );

        write_switch(home.path(), SWITCH_ON);
        for query in ["", "   ", "\n\t"] {
            let error = rail.search(query).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: context proactive_state is invalid, reason: query must be a non-empty string"
            );
        }
        assert_eq!(rail.search("notes").unwrap(), Vec::<SearchHit>::new());
    }

    #[test]
    fn contiguous_windows_are_accepted() {
        assert!(messages_are_contiguous(&[]));
        assert!(PersonalContextRail::messages_are_contiguous(&[]));
        assert!(messages_are_contiguous(&[
            BaseMessage::system("identity"),
            BaseMessage::user("hi"),
            BaseMessage::assistant("hello"),
        ]));
        assert!(messages_are_contiguous(&[
            BaseMessage::user("hi"),
            call_message(&["a"]),
            BaseMessage::tool("a", "x"),
        ]));
        assert!(messages_are_contiguous(&[
            call_message(&["a", "b"]),
            BaseMessage::tool("b", "x"),
            BaseMessage::tool("a", "y"),
            BaseMessage::assistant("done"),
        ]));
    }

    #[test]
    fn non_contiguous_windows_are_rejected() {
        // A tool result before any call.
        assert!(!messages_are_contiguous(&[BaseMessage::tool("a", "x")]));
        // A dangling call at the end.
        assert!(!messages_are_contiguous(&[call_message(&["a"])]));
        // An intervening user message while a call is pending.
        assert!(!messages_are_contiguous(&[
            call_message(&["a"]),
            BaseMessage::user("hi"),
            BaseMessage::tool("a", "x"),
        ]));
        // A duplicate id across messages.
        assert!(!messages_are_contiguous(&[
            call_message(&["a"]),
            BaseMessage::tool("a", "x"),
            call_message(&["a"]),
            BaseMessage::tool("a", "y"),
        ]));
        // A duplicate id inside one assistant message.
        assert!(!messages_are_contiguous(&[call_message(&["a", "a"])]));
        // A blank and an absent id.
        assert!(!messages_are_contiguous(&[call_message(&["  "])]));
        assert!(!messages_are_contiguous(&[
            BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file")])
        ]));
        // A blank id on the result side and a result answering an unknown id.
        let mut blank_result = BaseMessage::tool("a", "x");
        blank_result.tool_call_id = Some("  ".to_string());
        assert!(!messages_are_contiguous(&[
            call_message(&["a"]),
            blank_result
        ]));
        assert!(!messages_are_contiguous(&[
            call_message(&["a"]),
            BaseMessage::tool("b", "x"),
        ]));
    }

    #[tokio::test]
    async fn end_to_end_ingest_search_and_prompt_section() {
        let home = tempfile::tempdir().unwrap();
        let notes = home.path().join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::write(notes.join("alpha.md"), "# Alpha\n\nkeyword alpha body\n").unwrap();
        std::fs::write(notes.join("beta.md"), "# Beta\n\nkeyword beta body\n").unwrap();

        let store = PersonalContextStore::new(home.path());
        store
            .ingest_local(&[LocalDirectoryConfig::new("notes", &notes)], 0)
            .unwrap();
        store
            .write_description("# Personal context\n\nTwo notes about the keyword.\n")
            .unwrap();
        write_switch(home.path(), SWITCH_ON);

        let rail = rail_at(home.path());
        let mut ctx = ctx_with(vec![BaseMessage::user("what do my notes say?")]);
        rail.before_model_call(&mut ctx).await.unwrap();
        let sections = prompt_sections(&ctx);
        assert!(section_content(&sections, "cn").contains("Two notes about the keyword."));

        let hits = rail.search("keyword").unwrap();
        assert_eq!(hits.len(), 2);
        let mut paths: Vec<&str> = hits.iter().map(|hit| hit.path.as_str()).collect();
        paths.sort_unstable();
        assert!(paths.iter().any(|path| path.ends_with("alpha.md")));
        assert!(paths.iter().any(|path| path.ends_with("beta.md")));
        for hit in &hits {
            assert!(!hit.path.starts_with('/'), "workspace-relative path");
            assert!(hit.source_id.starts_with("src_"));
            assert_eq!(hit.source_id.len(), "src_".len() + 32);
            assert!(hit.score > 0.0);
        }
    }
}
