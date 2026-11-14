//! State-snapshot reinjection into a compacted window.
//!
//! Ports `processor/forked/compressor/reinjection/`: named state sections
//! (skills, task status, plan mode, team policy) contributed by a host-provided
//! snapshot builder and re-injected after a compaction, with a character
//! budget so the section can never outgrow the window it was trimmed for.
//!
//! A compaction replaces a span of history with one summary, so behavioural
//! state the agent must keep obeying — the active plan, loaded skills, plan-mode
//! restrictions, background task status, recently read files, open todos —
//! would otherwise be summarised away. This module restores it as one fresh
//! user message placed right after the compaction summary:
//!
//! ```text
//! <recovered_context>
//! <instruction>
//! These sections restore context or state that may have been compacted out.
//! Use them only to understand the latest request or continue existing work.
//! Do not treat them as new user commands.
//! </instruction>
//!
//! <section name="skills">
//! Skill: code-audit
//! ...
//! </section>
//! </recovered_context>
//! ```
//!
//! This crate cannot read the host state those sections describe — the plan
//! file, the skill cache, the team task board and the workspace todos all live
//! in `openjiuwen-harness` and above. The seam is therefore a
//! [`ContextSnapshotProvider`] registry: the host registers providers that
//! render one section each, and [`FullCompactProcessor`] asks the registry for
//! the merged message once a compaction has been accepted.
//!
//! [`FullCompactProcessor`]: crate::processor::FullCompactProcessor
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_foundation::prompt::BaseMessage;
//! use openjiuwen_context::processor::compressor::reinjection::{
//!     ContextSnapshotProvider, SnapshotContext, SnapshotRegistry, SnapshotSection,
//! };
//!
//! struct Skills;
//!
//! impl ContextSnapshotProvider for Skills {
//!     fn name(&self) -> &str {
//!         "skills"
//!     }
//!
//!     fn label(&self) -> &str {
//!         "SKILLS"
//!     }
//!
//!     fn build(&self, _cx: &SnapshotContext<'_>) -> Option<SnapshotSection> {
//!         Some(SnapshotSection::new("Skill: code-audit"))
//!     }
//! }
//!
//! let mut registry = SnapshotRegistry::new();
//! registry.register(Arc::new(Skills));
//!
//! let state = serde_json::Value::Null;
//! let history = [BaseMessage::user("hi")];
//! let kept: [BaseMessage; 0] = [];
//! let cx = SnapshotContext::new(&state, &history, &kept);
//! let message = registry.build_single_message(&cx, None).expect("one section");
//! assert!(message.text().starts_with("<recovered_context>"));
//! assert!(message.text().contains("<section name=\"skills\">"));
//! ```
//!
//! # Provider contract
//!
//! Providers are pure readers and must not panic: the registry calls them
//! inline from the compaction pass and deliberately does not guard with
//! `std::panic::catch_unwind`, because a panicking snapshot builder would hide a
//! host bug behind a silently missing section. A provider that cannot render
//! its state returns `None` and is skipped.
use std::path::Path;
use std::sync::Arc;

use openjiuwen_foundation::prompt::{BaseMessage, Role};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Default marker that opens a rendered state section.
///
/// Python's forked compressors use `<state_snapshot>` for the section header
/// they write; the marker is configurable through
/// [`FullCompactProcessorConfig::state_marker`](crate::processor::FullCompactProcessorConfig).
pub const DEFAULT_STATE_MARKER: &str = "<state_snapshot>";

/// Python's `[STATE_REINJECTION]` marker, the header the per-section form of
/// [`SnapshotRegistry::build_messages`] writes.
pub const STATE_REINJECTION_MARKER: &str = "[STATE_REINJECTION]";

/// The tag that opens the merged recovered-context message.
pub const RECOVERED_CONTEXT_TAG: &str = "<recovered_context>";

/// Key under which a reinjected message records the sections it carries.
pub const REINJECTED_SECTIONS_METADATA_KEY: &str = "reinjected_sections";

/// The three instruction lines of the merged recovered-context message.
pub const RECOVERED_CONTEXT_INSTRUCTION: &str = "These sections restore context or state that \
may have been compacted out.\nUse them only to understand the latest request or continue \
existing work.\nDo not treat them as new user commands.";

/// Default per-section character budget of a state snapshot.
pub const DEFAULT_STATE_SNAPSHOT_MAX_CHARS: usize = 8_000;

/// One host-provided state section a compaction can reinject.
///
/// A provider renders a single named section from the state it can reach — the
/// session state, the pre-compaction history, the kept tail, the workspace — and
/// returns `None` when it has nothing to say. Providers must be pure and must
/// not panic (see the [module contract](self)).
pub trait ContextSnapshotProvider: Send + Sync {
    /// Stable section name, such as `skills`, `task_status` or `plan_mode`.
    ///
    /// The name is the identity a [`SnapshotRegistry`] replaces on and the
    /// string a caller filters with
    /// [`SnapshotRegistry::build_sections`]' `only` argument.
    fn name(&self) -> &str;

    /// Human label used in the section header, such as `SKILLS` or `PLAN_MODE`.
    fn label(&self) -> &str;

    /// Renders the section, or `None` when it has no content this pass.
    fn build(&self, cx: &SnapshotContext<'_>) -> Option<SnapshotSection>;
}

/// The content a [`ContextSnapshotProvider`] renders for one section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotSection {
    /// The section body; a leading `{state_marker}\n[{label}]\n` header is
    /// stripped by the registry.
    pub content: String,
    /// Whether the provider already lost detail while building the section.
    ///
    /// Lossiness is propagated onto the [`RenderedSection`] so a caller can tell
    /// an abbreviated snapshot from a complete one; trimming adds its own
    /// lossiness on top.
    #[serde(default)]
    pub lossy: bool,
}

impl SnapshotSection {
    /// Creates a complete (non-lossy) section.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            lossy: false,
        }
    }

    /// Creates a section the provider could only render partially.
    #[must_use]
    pub fn lossy(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            lossy: true,
        }
    }
}

/// Everything a [`ContextSnapshotProvider`] may read while rendering.
///
/// The fields are the seam Python's `ReinjectContext` carries: the host state
/// the compaction itself cannot see, the history being replaced, the tail that
/// survives (for de-duplication), and the rendering budget.
pub struct SnapshotContext<'a> {
    /// Free-form host state, Python's `session.get_state()`.
    ///
    /// The compaction pass inside this crate has no session handle, so it
    /// supplies [`serde_json::Value::Null`]; a host that keeps state in session
    /// state builds its own [`SnapshotContext`] (or captures the state in the
    /// provider) and calls the registry directly.
    pub session_state: &'a Value,
    /// The pre-compaction history, including the part being summarised away.
    pub source_messages: &'a [BaseMessage],
    /// The tail that survives the compaction, for de-duplication.
    pub kept_messages: &'a [BaseMessage],
    /// Workspace root, when the host has one; `None` at the compaction seam.
    pub workspace_root: Option<&'a Path>,
    /// Marker that opens a section header, [`DEFAULT_STATE_MARKER`] by default.
    pub state_marker: &'a str,
    /// Per-section character budget; [`DEFAULT_STATE_SNAPSHOT_MAX_CHARS`] by
    /// default.
    pub max_chars: usize,
}

impl<'a> SnapshotContext<'a> {
    /// Creates a context over `session_state`, `source_messages` and
    /// `kept_messages`, with the default marker and budget.
    #[must_use]
    pub fn new(
        session_state: &'a Value,
        source_messages: &'a [BaseMessage],
        kept_messages: &'a [BaseMessage],
    ) -> Self {
        Self {
            session_state,
            source_messages,
            kept_messages,
            workspace_root: None,
            state_marker: DEFAULT_STATE_MARKER,
            max_chars: DEFAULT_STATE_SNAPSHOT_MAX_CHARS,
        }
    }

    /// Sets the workspace root providers may read.
    #[must_use]
    pub fn with_workspace_root(mut self, workspace_root: Option<&'a Path>) -> Self {
        self.workspace_root = workspace_root;
        self
    }

    /// Sets the section header marker.
    #[must_use]
    pub fn with_state_marker(mut self, state_marker: &'a str) -> Self {
        self.state_marker = state_marker;
        self
    }

    /// Sets the per-section character budget.
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }
}

/// A section after header stripping and trimming, ready to be rendered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedSection {
    /// The provider's section name.
    pub name: String,
    /// The provider's human label.
    pub label: String,
    /// The section body, with any leading state header removed and trimmed to
    /// the budget.
    pub content: String,
    /// Whether the section lost detail, either in the provider or to trimming.
    pub lossy: bool,
    /// Characters the trimming pass removed from the body.
    #[serde(default)]
    pub truncated_chars: usize,
}

impl RenderedSection {
    /// Renders the section as Python's `<section name="…">…</section>` block.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "<section name=\"{}\">\n{}\n</section>",
            self.name, self.content
        )
    }
}

/// The ordered registry of state-snapshot providers, Python's
/// `StateReinjector`.
///
/// Registration order fixes output order: a `only` filter selects a subset but
/// never reorders, so a caller that lists sections in its own preference order
/// still gets them in registration order.
#[derive(Default)]
pub struct SnapshotRegistry {
    providers: Vec<Arc<dyn ContextSnapshotProvider>>,
    total_max_chars: Option<usize>,
}

impl std::fmt::Debug for SnapshotRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotRegistry")
            .field("providers", &self.names())
            .field("total_max_chars", &self.total_max_chars)
            .finish()
    }
}

impl SnapshotRegistry {
    /// Creates an empty registry; every provider has to be registered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `provider`.
    ///
    /// A provider with an existing name replaces that one **in place**, keeping
    /// its position, which is Python's `StateReinjector.register` behaviour. A
    /// new name is appended.
    pub fn register(&mut self, provider: Arc<dyn ContextSnapshotProvider>) {
        if let Some(existing) = self
            .providers
            .iter_mut()
            .find(|existing| existing.name() == provider.name())
        {
            *existing = provider;
            return;
        }
        self.providers.push(provider);
    }

    /// Registers every provider of `providers`, in iteration order.
    pub fn register_all(
        &mut self,
        providers: impl IntoIterator<Item = Arc<dyn ContextSnapshotProvider>>,
    ) {
        for provider in providers {
            self.register(provider);
        }
    }

    /// The registered providers, in registration (and therefore output) order.
    #[must_use]
    pub fn providers(&self) -> &[Arc<dyn ContextSnapshotProvider>] {
        &self.providers
    }

    /// Iterates the registered providers, in registration order.
    pub fn iter(&self) -> std::slice::Iter<'_, Arc<dyn ContextSnapshotProvider>> {
        self.providers.iter()
    }

    /// The registered section names, in registration order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.providers
            .iter()
            .map(|provider| provider.name())
            .collect()
    }

    /// The number of registered providers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether no provider is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Caps the total characters of all rendered section bodies.
    ///
    /// The cap is applied on top of the per-section budget: when the sections
    /// together exceed it, the larger sections absorb the cut first (the small
    /// ones keep their whole body) and any remainder is shared proportionally,
    /// so the recovered-context message cannot outgrow the budget it was sized
    /// for. `0` disables the total cap, matching Python's `max_chars <= 0`
    /// convention for a disabled budget.
    #[must_use]
    pub fn with_total_max_chars(mut self, max_chars: usize) -> Self {
        self.total_max_chars = Some(max_chars).filter(|cap| *cap > 0);
        self
    }

    /// The total character cap, `None` when it is disabled.
    #[must_use]
    pub fn total_max_chars(&self) -> Option<usize> {
        self.total_max_chars
    }

    /// Renders every registered section, in registration order.
    ///
    /// `only` selects a subset by name (`None` renders every provider). A
    /// provider returning `None` or blank content is skipped, the leading
    /// `{state_marker}\n[{label}]\n` header (or the generic
    /// `{state_marker}\n[…]` one) is stripped, and the body is trimmed to
    /// `cx.max_chars` and then to
    /// [`SnapshotRegistry::with_total_max_chars`] when that cap is set.
    #[must_use]
    pub fn build_sections(
        &self,
        cx: &SnapshotContext<'_>,
        only: Option<&[&str]>,
    ) -> Vec<RenderedSection> {
        let mut sections: Vec<RenderedSection> = Vec::with_capacity(self.providers.len());
        for provider in &self.providers {
            if only.is_some_and(|names| !names.contains(&provider.name())) {
                continue;
            }
            let Some(section) = provider.build(cx) else {
                continue;
            };
            let stripped = strip_state_header(&section.content, cx.state_marker, provider.label());
            if stripped.is_empty() {
                continue;
            }
            let (content, truncated_chars) = truncate_state_text(&stripped, cx.max_chars);
            sections.push(RenderedSection {
                name: provider.name().to_owned(),
                label: provider.label().to_owned(),
                content,
                lossy: section.lossy || truncated_chars > 0,
                truncated_chars,
            });
        }
        if let Some(cap) = self.total_max_chars {
            enforce_total_cap(&mut sections, cap);
        }
        sections
    }

    /// Renders one user message per section, Python's `build_messages` shape:
    /// `"{state_marker}\n[{label}]\n{trimmed content}"`.
    ///
    /// Each message records its section name under
    /// [`REINJECTED_SECTIONS_METADATA_KEY`], so a resumed session can tell the
    /// message apart from real user input.
    #[must_use]
    pub fn build_messages(
        &self,
        cx: &SnapshotContext<'_>,
        only: Option<&[&str]>,
    ) -> Vec<BaseMessage> {
        self.build_sections(cx, only)
            .into_iter()
            .map(|section| {
                let mut message = BaseMessage::user(format!(
                    "{}\n[{}]\n{}",
                    cx.state_marker, section.label, section.content
                ));
                message.metadata.insert(
                    REINJECTED_SECTIONS_METADATA_KEY.to_owned(),
                    json!([section.name]),
                );
                message
            })
            .collect()
    }

    /// Renders every selected section into one recovered-context message,
    /// Python's `build_single_reinjected_state_message`.
    ///
    /// The message carries a `<recovered_context>` wrapper with the fixed
    /// instruction block and one `<section name="…">` block per non-empty
    /// section, joined by a blank line. `None` is returned when no section
    /// renders, so a caller can skip the insertion entirely.
    #[must_use]
    pub fn build_single_message(
        &self,
        cx: &SnapshotContext<'_>,
        only: Option<&[&str]>,
    ) -> Option<BaseMessage> {
        let sections = self.build_sections(cx, only);
        if sections.is_empty() {
            return None;
        }
        let mut content = String::with_capacity(
            RECOVERED_CONTEXT_INSTRUCTION.len()
                + sections
                    .iter()
                    .map(|section| section.content.len() + 40)
                    .sum::<usize>()
                + 64,
        );
        content.push_str(RECOVERED_CONTEXT_TAG);
        content.push('\n');
        content.push_str("<instruction>\n");
        content.push_str(RECOVERED_CONTEXT_INSTRUCTION);
        content.push_str("\n</instruction>\n\n");
        let blocks: Vec<String> = sections.iter().map(RenderedSection::render).collect();
        content.push_str(&blocks.join("\n\n"));
        content.push('\n');
        content.push_str("</recovered_context>");

        let mut message = BaseMessage::user(content);
        let names: Vec<&str> = sections
            .iter()
            .map(|section| section.name.as_str())
            .collect();
        message
            .metadata
            .insert(REINJECTED_SECTIONS_METADATA_KEY.to_owned(), json!(names));
        Some(message)
    }
}

impl<'a> IntoIterator for &'a SnapshotRegistry {
    type Item = &'a Arc<dyn ContextSnapshotProvider>;
    type IntoIter = std::slice::Iter<'a, Arc<dyn ContextSnapshotProvider>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Whether `message` is a state snapshot a previous compaction reinjected.
///
/// A snapshot written by this crate carries
/// [`REINJECTED_SECTIONS_METADATA_KEY`] in its metadata; a message written by
/// Python (which records no metadata) is recognised by its
/// [`RECOVERED_CONTEXT_TAG`] prefix instead, so a resumed session still
/// replaces a stale snapshot rather than stacking a second one.
#[must_use]
pub fn is_reinjected_state_message(message: &BaseMessage) -> bool {
    message
        .metadata
        .contains_key(REINJECTED_SECTIONS_METADATA_KEY)
        || (message.role == Role::User && message.text().starts_with(RECOVERED_CONTEXT_TAG))
}

/// Strips a leading state header from `content`, Python's `_strip_state_header`.
///
/// The exact `"{state_marker}\n[{label}]\n"` prefix is removed first; otherwise a
/// generic `"{state_marker}\n[…]…]\n"` header is removed by dropping everything up
/// to the first line break after the closing bracket. Content without a header
/// is returned trimmed.
#[must_use]
pub fn strip_state_header(content: &str, state_marker: &str, label: &str) -> String {
    let text = content.trim();
    let prefix = format!("{state_marker}\n[{label}]\n");
    if let Some(rest) = text.strip_prefix(&prefix) {
        return rest.trim().to_owned();
    }
    if let Some(rest) = text.strip_prefix(&format!("{state_marker}\n")) {
        if rest.starts_with('[') {
            if let Some(end) = rest.find("]\n") {
                return rest[end + 2..].trim().to_owned();
            }
        }
    }
    text.to_owned()
}

/// Truncates `text` head/tail-wise to `max_chars`, Python's
/// `truncate_state_text`.
///
/// Roughly 60 % of the budget is kept from the head and 40 % from the tail,
/// with a `\n[... truncated N characters ...]\n` marker between them, where `N`
/// is the number of source characters dropped. A budget of `0` disables
/// truncation. The returned string can exceed `max_chars` by the marker's own
/// length; the second value is the number of characters removed (`0` when the
/// text fit).
#[must_use]
pub fn truncate_state_text(text: &str, max_chars: usize) -> (String, usize) {
    let length = text.chars().count();
    if max_chars == 0 || length <= max_chars {
        return (text.to_owned(), 0);
    }
    let head_chars = max_chars.saturating_mul(3) / 5;
    let tail_chars = max_chars.saturating_sub(head_chars);
    let head: String = text.chars().take(head_chars).collect();
    let tail: String = text.chars().skip(length - tail_chars).collect();
    let removed = length - head_chars - tail_chars;
    (
        format!("{head}{}{tail}", truncation_marker(removed)),
        removed,
    )
}

/// The marker placed between the head and the tail of a trimmed section.
fn truncation_marker(removed: usize) -> String {
    format!("\n[... truncated {removed} characters ...]\n")
}

/// Trims `sections` so their combined bodies fit `cap` characters.
///
/// Two deterministic passes: a water-filling pass lets the smaller sections keep
/// their body and trims the larger ones to what is left of the budget (with the
/// usual truncation marker), then a final proportional pass cuts whole
/// characters from the bodies that are still over their equal share, so the cap
/// always holds. Every step marks the section lossy and accumulates its removed
/// characters.
fn enforce_total_cap(sections: &mut [RenderedSection], cap: usize) {
    if char_sum(sections) <= cap {
        return;
    }
    water_fill(sections, cap);
    if char_sum(sections) <= cap {
        return;
    }
    let count = sections
        .iter()
        .filter(|section| !section.content.is_empty())
        .count();
    if count == 0 {
        return;
    }
    let share = cap / count;
    let protected: usize = sections
        .iter()
        .filter(|section| {
            let length = section.content.chars().count();
            length > 0 && length <= share
        })
        .map(|section| section.content.chars().count())
        .sum();
    // Sections at or below their equal share keep their body; the rest share
    // whatever budget is left, proportionally to their size. When even the
    // small sections do not fit, every body is scaled down instead.
    let (budget, over_share_only) = if protected >= cap {
        (cap, false)
    } else {
        (cap - protected, true)
    };
    let sources: Vec<usize> = (0..sections.len())
        .filter(|index| {
            let length = sections[*index].content.chars().count();
            length > 0 && (!over_share_only || length > share)
        })
        .collect();
    let source_total: usize = sources
        .iter()
        .map(|index| sections[*index].content.chars().count())
        .sum();
    if source_total == 0 {
        return;
    }
    for index in sources {
        let length = sections[index].content.chars().count();
        hard_cut(
            &mut sections[index],
            length.saturating_mul(budget) / source_total,
        );
    }
}

/// Gives the smaller sections their full body and trims the larger ones to the
/// remaining budget, smallest first.
fn water_fill(sections: &mut [RenderedSection], cap: usize) {
    let mut order: Vec<usize> = (0..sections.len())
        .filter(|index| !sections[*index].content.is_empty())
        .collect();
    order.sort_by_key(|index| sections[*index].content.chars().count());
    let count = order.len();
    let mut remaining = cap;
    for (position, &index) in order.iter().enumerate() {
        let length = sections[index].content.chars().count();
        let share = remaining / (count - position).max(1);
        if length <= share {
            remaining -= length;
            continue;
        }
        if share == 0 {
            continue;
        }
        trim_section(&mut sections[index], share);
        remaining = remaining.saturating_sub(sections[index].content.chars().count());
    }
}

/// Cuts a section body to `allowed` characters, accumulating its lossiness.
fn hard_cut(section: &mut RenderedSection, allowed: usize) {
    let length = section.content.chars().count();
    if length <= allowed {
        return;
    }
    section.content = section.content.chars().take(allowed).collect();
    section.lossy = true;
    section.truncated_chars += length - allowed;
}

/// Trims one section body to `budget` characters, accumulating its lossiness.
fn trim_section(section: &mut RenderedSection, budget: usize) {
    let (content, removed) = truncate_state_text(&section.content, budget);
    if removed == 0 {
        return;
    }
    section.content = content;
    section.lossy = true;
    section.truncated_chars += removed;
}

/// The total character count of every section body.
fn char_sum(sections: &[RenderedSection]) -> usize {
    sections
        .iter()
        .map(|section| section.content.chars().count())
        .sum()
}

#[cfg(test)]
mod tests;
