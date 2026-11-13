//! The tag index: a two-way, per-kind mapping from resources to tags and back.
//!
//! Ports Python `openjiuwen/core/runner/resources_manager/tag_manager.py`
//! (v0.1.20). Python keys both directions on the bare resource id; the Rust port
//! partitions both indexes by [`ResourceKind`], so a tool, an agent, a workflow,
//! a team, and an MCP server may share an id without sharing tags. Tag values
//! stay one flat namespace: the kind is the index partition, never part of the
//! tag itself.
//!
//! Semantics preserved from Python:
//!
//! - two bi-directional indexes (resource to tags, tag to resources);
//! - [`GLOBAL_TAG`] is exclusive: tagging a resource with it drops that
//!   resource's other tags, and merging more tags into an already-global
//!   resource adds nothing;
//! - a tag that loses its last resource is deleted from the index (garbage
//!   collection), so [`TagMgr::list_tags`] and [`TagMgr::has_tag`] only report
//!   live tags;
//! - [`TagUpdateStrategy::Merge`] unions, [`TagUpdateStrategy::Replace`] swaps,
//!   and [`TagMatchStrategy::Any`] / [`TagMatchStrategy::All`] are set union and
//!   intersection over the ids of one kind.
//!
//! Deliberate differences, each documented on the method that exhibits it:
//! failures are [`Error`] values instead of Python's
//! `build_error(StatusCode.RESOURCE_TAG_*)` exceptions; every mutation is
//! revision-counted; [`GLOBAL_TAG`] is garbage-collected like any other tag
//! (Python pre-seeds it, so `has_tag(GLOBAL)` is always true there); and every
//! accessor is deterministic (tags in registration order, ids sorted).
//!
//! There are no `.await`s here; at most one `parking_lot::RwLock` guard is held,
//! for a purely synchronous section.
//!
//! ```
//! use openjiuwen_runner::resources::tag::{ResourceKind, Tag, TagMgr};
//!
//! let tags = TagMgr::new();
//! tags.tag(ResourceKind::Agent, "assistant", [Tag::new("chat")]).unwrap();
//! assert_eq!(
//!     tags.tags_of(ResourceKind::Agent, "assistant"),
//!     vec![Tag::new("chat")]
//! );
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::Deref;

use openjiuwen_common::error::{Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{invalid_id, not_found};

/// The tag that makes a resource globally addressable and exclusive.
///
/// Python parity: `GLOBAL` in
/// `openjiuwen/core/runner/resources_manager/base.py`. Tagging a resource with
/// it drops that resource's other tags.
pub const GLOBAL_TAG: &str = "__global__";

/// The message used for an empty or whitespace-only tag value.
const EMPTY_TAG: &str = "tag cannot be empty or whitespace only";

/// The message used for a tag operation with no tags at all.
const NO_TAGS: &str = "at least one tag is required";

/// A tag value: an arbitrary non-empty string the manager does not interpret.
///
/// Serializes as a bare JSON string (`#[serde(transparent)]`), matching Python's
/// `Tag = str` alias.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tag(String);

impl Tag {
    /// Wraps `value` as a tag.
    ///
    /// Validation happens at the [`TagMgr`] boundary, so a tag can be built for
    /// comparison or as a lookup key without a fallible call.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the exclusive global tag ([`GLOBAL_TAG`]).
    #[must_use]
    pub fn global() -> Self {
        Self(GLOBAL_TAG.to_owned())
    }

    /// Returns whether this is the global tag.
    #[must_use]
    pub fn is_global(&self) -> bool {
        self.0 == GLOBAL_TAG
    }

    /// Borrows the tag as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Tag {
    /// Prints the bare tag value.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Deref for Tag {
    type Target = str;

    /// Dereferences to the bare tag value.
    fn deref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Tag {
    /// Wraps a borrowed tag value.
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for Tag {
    /// Wraps an owned tag value.
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// The resource partition a tag association belongs to.
///
/// Python has one index per resource manager; the Rust port keeps one [`TagMgr`]
/// and partitions it by kind instead. Serializes as `"tool"`, `"agent"`,
/// `"workflow"`, `"team"`, `"mcp_server"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// A tool registered in the runner.
    Tool,
    /// A single agent.
    Agent,
    /// A workflow.
    Workflow,
    /// An agent team.
    Team,
    /// An MCP server registration.
    McpServer,
}

impl ResourceKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 5] = [
        Self::Tool,
        Self::Agent,
        Self::Workflow,
        Self::Team,
        Self::McpServer,
    ];

    /// Returns the lower-case name used in error messages and serialization.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Agent => "agent",
            Self::Workflow => "workflow",
            Self::Team => "team",
            Self::McpServer => "mcp_server",
        }
    }
}

impl fmt::Display for ResourceKind {
    /// Prints the lower-case kind name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How [`TagMgr::resources_by_tags`] combines several tags (`ANY` / `ALL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagMatchStrategy {
    /// A resource matches when it carries **any** of the requested tags (union).
    Any,
    /// A resource matches when it carries **all** the requested tags
    /// (intersection).
    All,
}

/// How [`TagMgr::update_tags`] treats a resource's existing tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagUpdateStrategy {
    /// Union the existing tags with the new ones (`MERGE`).
    Merge,
    /// Swap the existing tags for the new ones (`REPLACE`).
    Replace,
}

/// The lock-guarded state of a [`TagMgr`].
struct Inner {
    /// Tag to kind to resource ids. Only live tags have an entry, so
    /// `by_tag.len()` is the number of distinct tags in use.
    by_tag: BTreeMap<Tag, HashMap<ResourceKind, BTreeSet<String>>>,
    /// Resource to tags in registration order. Only tagged resources have an
    /// entry; a resource that loses its last tag disappears.
    by_resource: HashMap<(ResourceKind, String), Vec<Tag>>,
    /// Mutation counter; `+1` on every successful mutation.
    revision: u64,
}

impl Inner {
    /// Returns the ids of `kind` that carry `tag` (empty when either is absent).
    fn resource_ids(&self, kind: ResourceKind, tag: &Tag) -> BTreeSet<String> {
        match self.by_tag.get(tag).and_then(|kinds| kinds.get(&kind)) {
            Some(ids) => ids.clone(),
            None => BTreeSet::new(),
        }
    }
}

/// The tag index of the resource manager: tag values on one side, resources on
/// the other, partitioned by [`ResourceKind`].
///
/// Interior mutability is a `parking_lot::RwLock` held only for synchronous map
/// updates, so the manager can live inside the runner without `&mut`.
///
/// Python parity: `TagMgr` in
/// `openjiuwen/core/runner/resources_manager/tag_manager.py`, extended with a
/// resource-kind partition, a revision counter, and an explicit
/// [`TagUpdateStrategy`] argument instead of a keyword.
pub struct TagMgr {
    inner: RwLock<Inner>,
}

impl TagMgr {
    /// Builds an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                by_tag: BTreeMap::new(),
                by_resource: HashMap::new(),
                revision: 0,
            }),
        }
    }

    /// Adds `tags` to one resource and returns its resulting full tag list.
    ///
    /// An unknown `(kind, id)` pair is just a new key (Python's `tag_resource`
    /// also creates the entry). Duplicate tags are recorded once and new tags are
    /// appended in input order. [`GLOBAL_TAG`] is exclusive: if it appears in
    /// `tags` the resource ends up carrying only that tag, and if the resource is
    /// already global this call adds nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty or whitespace-only `id` or tag value,
    /// and for an empty `tags` input.
    pub fn tag(
        &self,
        kind: ResourceKind,
        id: &str,
        tags: impl IntoIterator<Item = Tag>,
    ) -> Result<Vec<Tag>> {
        validate_id(id)?;
        let requested = normalize(tags)?;

        let mut inner = self.inner.write();
        if requested.iter().any(Tag::is_global) {
            return Ok(Self::set_global(&mut inner, kind, id));
        }
        let key = (kind, id.to_owned());
        let mut current = inner.by_resource.get(&key).cloned().unwrap_or_default();
        if current.iter().any(Tag::is_global) {
            return Ok(current);
        }

        let mut changed = false;
        for tag in requested {
            if !current.contains(&tag) {
                Self::attach(&mut inner, kind, id, &tag);
                current.push(tag);
                changed = true;
            }
        }
        if changed {
            inner.by_resource.insert(key, current.clone());
            Self::bump(&mut inner);
        }
        Ok(current)
    }

    /// Updates one resource's tags under `strategy`, returning the resulting
    /// tag list.
    ///
    /// [`TagUpdateStrategy::Merge`] is [`TagMgr::tag`] by another name: it unions
    /// the existing tags with `tags` and leaves an already-global resource
    /// global. [`TagUpdateStrategy::Replace`] makes `tags` the complete set and
    /// therefore swaps an existing [`GLOBAL_TAG`] out (Python's
    /// `_replace_resource_tags` does the same). Unlike Python's
    /// `update_resource_tags`, an untagged resource is a new key, not an error.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty or whitespace-only `id` or tag value,
    /// and for an empty `tags` input.
    pub fn update_tags(
        &self,
        kind: ResourceKind,
        id: &str,
        tags: impl IntoIterator<Item = Tag>,
        strategy: TagUpdateStrategy,
    ) -> Result<Vec<Tag>> {
        validate_id(id)?;
        let requested = normalize(tags)?;

        let mut inner = self.inner.write();
        if requested.iter().any(Tag::is_global) {
            return Ok(Self::set_global(&mut inner, kind, id));
        }
        let key = (kind, id.to_owned());
        let previous = inner.by_resource.get(&key).cloned().unwrap_or_default();
        if previous.iter().any(Tag::is_global) && strategy == TagUpdateStrategy::Merge {
            return Ok(previous);
        }

        let result = match strategy {
            TagUpdateStrategy::Merge => {
                let mut merged = previous.clone();
                for tag in requested {
                    if !merged.contains(&tag) {
                        merged.push(tag);
                    }
                }
                merged
            }
            TagUpdateStrategy::Replace => requested,
        };
        if previous == result {
            return Ok(result);
        }
        for tag in &previous {
            Self::detach(&mut inner, kind, id, tag);
        }
        for tag in &result {
            Self::attach(&mut inner, kind, id, tag);
        }
        inner.by_resource.insert(key, result.clone());
        Self::bump(&mut inner);
        Ok(result)
    }

    /// Returns the tags of one resource in registration order.
    ///
    /// An untagged or unknown resource returns an empty list and never errors
    /// (Python's `get_resources_tags` likewise returns `[]`).
    #[must_use]
    pub fn tags_of(&self, kind: ResourceKind, id: &str) -> Vec<Tag> {
        self.inner
            .read()
            .by_resource
            .get(&(kind, id.to_owned()))
            .cloned()
            .unwrap_or_default()
    }

    /// Returns whether one resource carries `tag` (Python's `has_resource_tag`).
    #[must_use]
    pub fn has_resource_tag(&self, kind: ResourceKind, id: &str, tag: &Tag) -> bool {
        self.inner
            .read()
            .by_resource
            .get(&(kind, id.to_owned()))
            .is_some_and(|tags| tags.contains(tag))
    }

    /// Returns whether `tag` is live (at least one resource carries it).
    ///
    /// Tags are garbage-collected, so this flips back to `false` once the last
    /// carrier is untagged or removed; that includes [`GLOBAL_TAG`].
    #[must_use]
    pub fn has_tag(&self, tag: &Tag) -> bool {
        self.inner.read().by_tag.contains_key(tag)
    }

    /// Returns the sorted ids of `kind` that carry `tag` (Python's
    /// `get_tag_resources`, plus the kind partition and a sorted result).
    #[must_use]
    pub fn resources_with_tag(&self, kind: ResourceKind, tag: &Tag) -> Vec<String> {
        self.inner
            .read()
            .resource_ids(kind, tag)
            .into_iter()
            .collect()
    }

    /// Returns the sorted ids of `kind` matching `tags` under `strategy`.
    ///
    /// [`TagMatchStrategy::Any`] is the union of the carriers of each tag;
    /// [`TagMatchStrategy::All`] is the intersection, so a tag no resource of that
    /// kind carries yields an empty result rather than an error (Python's
    /// `find_resources_by_tags` has a `skip_if_not_exists` flag for exactly that;
    /// the Rust port always skips).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty `tags` slice and for an empty or
    /// whitespace-only tag value.
    pub fn resources_by_tags(
        &self,
        kind: ResourceKind,
        tags: &[Tag],
        strategy: TagMatchStrategy,
    ) -> Result<Vec<String>> {
        let requested = normalize(tags.iter().cloned())?;
        let inner = self.inner.read();

        let found: BTreeSet<String> = match strategy {
            TagMatchStrategy::Any => requested
                .iter()
                .flat_map(|tag| inner.resource_ids(kind, tag))
                .collect(),
            TagMatchStrategy::All => {
                let mut iter = requested.iter();
                let Some(first) = iter.next() else {
                    return Ok(Vec::new());
                };
                let mut intersection = inner.resource_ids(kind, first);
                for tag in iter {
                    let ids = inner.resource_ids(kind, tag);
                    intersection = intersection.intersection(&ids).cloned().collect();
                }
                intersection
            }
        };
        Ok(found.into_iter().collect())
    }

    /// Removes `tags` from one resource and returns the remaining tags.
    ///
    /// Python parity: `remove_resource_tags`. A resource the manager never tagged
    /// has no entry, so this is [`Error::NotFound`] (the crate's
    /// `not_found(kind, id)` wording) unless `skip_if_missing` is `true`, in which
    /// case it returns an empty list. With `skip_if_missing == false` every
    /// requested tag must be present: Python raised
    /// `RESOURCE_TAG_REMOVE_RESOURCE_TAG_ERROR` naming the resource and the tag,
    /// which maps to [`Error::Validation`] here. A call that removes nothing is a
    /// no-op (the revision does not change).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty or whitespace-only `id` or tag value,
    /// and for a requested tag the resource does not carry when
    /// `skip_if_missing` is `false`. [`Error::NotFound`] when the resource has no
    /// tags recorded and `skip_if_missing` is `false`.
    pub fn remove_tags(
        &self,
        kind: ResourceKind,
        id: &str,
        tags: &[Tag],
        skip_if_missing: bool,
    ) -> Result<Vec<Tag>> {
        validate_id(id)?;
        for tag in tags {
            validate_tag(tag)?;
        }

        let mut inner = self.inner.write();
        let key = (kind, id.to_owned());
        let Some(current) = inner.by_resource.get(&key).cloned() else {
            if skip_if_missing {
                return Ok(Vec::new());
            }
            return Err(not_found(kind.as_str(), id));
        };
        if let Some(missing) = tags.iter().find(|tag| !current.contains(tag)) {
            if !skip_if_missing {
                return Err(Error::Validation(format!(
                    "{} '{id}' does not have tag '{missing}'",
                    kind.as_str()
                )));
            }
        }

        let mut remaining = current;
        let mut changed = false;
        for tag in tags {
            if remaining.contains(tag) {
                remaining.retain(|current| current != tag);
                Self::detach(&mut inner, kind, id, tag);
                changed = true;
            }
        }
        if !changed {
            return Ok(remaining);
        }
        if remaining.is_empty() {
            inner.by_resource.remove(&key);
        } else {
            inner.by_resource.insert(key, remaining.clone());
        }
        Self::bump(&mut inner);
        Ok(remaining)
    }

    /// Removes every tag of one resource and returns the removed tags.
    ///
    /// Python parity: `remove_resource_tags` without a tag list. An unknown
    /// resource returns an empty list and does not change the revision.
    #[must_use]
    pub fn untag(&self, kind: ResourceKind, id: &str) -> Vec<Tag> {
        self.remove_resource(kind, id)
    }

    /// Removes one tag from every resource and returns the ids that carried it.
    ///
    /// The result is sorted and de-duplicated, so an id shared by several kinds
    /// appears once, and tags that lose their last resource are collected. An
    /// unknown tag returns an empty list and does not change the revision
    /// (Python's `remove_tag` raised `RESOURCE_TAG_REMOVE_TAG_ERROR` unless
    /// `skip_if_not_exists` was set; this infallible signature always skips).
    #[must_use]
    pub fn remove_tag(&self, tag: &Tag) -> Vec<String> {
        let mut inner = self.inner.write();
        let Some(per_kind) = inner.by_tag.remove(tag) else {
            return Vec::new();
        };

        let mut ids = Vec::new();
        for (kind, kind_ids) in per_kind {
            for id in kind_ids {
                let key = (kind, id.clone());
                if let Some(tags) = inner.by_resource.get_mut(&key) {
                    tags.retain(|current| current != tag);
                }
                if inner.by_resource.get(&key).is_some_and(Vec::is_empty) {
                    inner.by_resource.remove(&key);
                }
                ids.push(id);
            }
        }
        ids.sort();
        ids.dedup();
        if !ids.is_empty() {
            Self::bump(&mut inner);
        }
        ids
    }

    /// Removes one resource and every tag association it holds, returning the
    /// tags it carried.
    ///
    /// This is the cleanup hook the resource manager calls when a resource is
    /// removed or its registry is cleared. An unknown resource returns an empty
    /// list and does not change the revision.
    #[must_use]
    pub fn remove_resource(&self, kind: ResourceKind, id: &str) -> Vec<Tag> {
        let mut inner = self.inner.write();
        let removed = Self::detach_tags(&mut inner, kind, id);
        if !removed.is_empty() {
            Self::bump(&mut inner);
        }
        removed
    }

    /// Returns every live tag, sorted.
    ///
    /// Python parity: `list_tags` (which also filters empty tags; this index
    /// deletes them eagerly). [`GLOBAL_TAG`] appears only while a resource is
    /// global.
    #[must_use]
    pub fn list_tags(&self) -> Vec<Tag> {
        self.inner.read().by_tag.keys().cloned().collect()
    }

    /// Returns the sorted ids of `kind` that carry at least one tag.
    #[must_use]
    pub fn resources(&self, kind: ResourceKind) -> Vec<String> {
        let inner = self.inner.read();
        let mut ids = BTreeSet::new();
        for per_kind in inner.by_tag.values() {
            if let Some(kind_ids) = per_kind.get(&kind) {
                ids.extend(kind_ids.iter().cloned());
            }
        }
        ids.into_iter().collect()
    }

    /// Returns the number of distinct live tags (see [`TagMgr::list_tags`]).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.read().by_tag.len()
    }

    /// Returns whether no tag is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.read().by_tag.is_empty()
    }

    /// Returns the mutation counter.
    ///
    /// It starts at zero and increments once per successful mutation (a tag, an
    /// update, a removal, a clear); validation failures and no-ops such as
    /// re-tagging an already-tagged resource leave it untouched.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.inner.read().revision
    }

    /// Removes the whole index and returns the number of distinct tags removed.
    ///
    /// A non-empty clear counts as one mutation; clearing an empty index is a
    /// no-op.
    #[must_use]
    pub fn clear(&self) -> usize {
        let mut inner = self.inner.write();
        let removed = inner.by_tag.len();
        inner.by_tag.clear();
        inner.by_resource.clear();
        if removed > 0 {
            Self::bump(&mut inner);
        }
        removed
    }

    /// Removes the whole index and resets the revision counter to zero.
    ///
    /// Used by the resource manager's release path, which returns the runner to a
    /// pristine state. Returns the number of distinct tags removed.
    // The resource manager's `release` wires this up; it stays `pub(crate)` so it
    // is not part of the public surface, which is why it may look unused here.
    #[allow(dead_code)]
    pub(crate) fn reset(&self) -> usize {
        let mut inner = self.inner.write();
        let removed = inner.by_tag.len();
        inner.by_tag.clear();
        inner.by_resource.clear();
        inner.revision = 0;
        removed
    }

    /// Makes `id` global: drops every other tag and records [`GLOBAL_TAG`].
    fn set_global(inner: &mut Inner, kind: ResourceKind, id: &str) -> Vec<Tag> {
        let previous = Self::detach_tags(inner, kind, id);
        let global = Tag::global();
        let changed = previous.len() != 1 || !previous.first().is_some_and(Tag::is_global);
        Self::attach(inner, kind, id, &global);
        inner
            .by_resource
            .insert((kind, id.to_owned()), vec![global.clone()]);
        if changed {
            Self::bump(inner);
        }
        vec![global]
    }

    /// Records that `id` of `kind` carries `tag`.
    fn attach(inner: &mut Inner, kind: ResourceKind, id: &str, tag: &Tag) {
        inner
            .by_tag
            .entry(tag.clone())
            .or_default()
            .entry(kind)
            .or_default()
            .insert(id.to_owned());
    }

    /// Removes one association, collecting the tag when it loses its last id.
    fn detach(inner: &mut Inner, kind: ResourceKind, id: &str, tag: &Tag) {
        if let Some(per_kind) = inner.by_tag.get_mut(tag) {
            if let Some(ids) = per_kind.get_mut(&kind) {
                ids.remove(id);
                if ids.is_empty() {
                    per_kind.remove(&kind);
                }
            }
            if per_kind.is_empty() {
                inner.by_tag.remove(tag);
            }
        }
    }

    /// Forgets every tag of one resource, returning the tags it carried.
    fn detach_tags(inner: &mut Inner, kind: ResourceKind, id: &str) -> Vec<Tag> {
        let Some(tags) = inner.by_resource.remove(&(kind, id.to_owned())) else {
            return Vec::new();
        };
        for tag in &tags {
            Self::detach(inner, kind, id, tag);
        }
        tags
    }

    /// Increments the mutation counter.
    fn bump(inner: &mut Inner) {
        inner.revision = inner.revision.saturating_add(1);
    }
}

impl Default for TagMgr {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TagMgr {
    /// Prints per-kind resource counts, the live tag count, and the revision.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read();
        let mut counts = BTreeMap::new();
        for (kind, _) in inner.by_resource.keys() {
            *counts.entry(*kind).or_insert(0_usize) += 1;
        }
        let counts: Vec<(&str, usize)> = counts
            .iter()
            .map(|(kind, count)| (kind.as_str(), *count))
            .collect();
        formatter
            .debug_struct("TagMgr")
            .field("resources", &counts)
            .field("tags", &inner.by_tag.len())
            .field("revision", &inner.revision)
            .finish()
    }
}

/// Rejects an empty or whitespace-only resource id.
fn validate_id(id: &str) -> Result<()> {
    if id.trim().is_empty() {
        return Err(invalid_id("resource"));
    }
    Ok(())
}

/// Rejects an empty or whitespace-only tag value.
fn validate_tag(tag: &Tag) -> Result<()> {
    if tag.as_str().trim().is_empty() {
        return Err(Error::Validation(EMPTY_TAG.to_owned()));
    }
    Ok(())
}

/// Validates a tag input and returns it de-duplicated in input order.
fn normalize(tags: impl IntoIterator<Item = Tag>) -> Result<Vec<Tag>> {
    let mut normalized = Vec::new();
    for tag in tags {
        validate_tag(&tag)?;
        if !normalized.contains(&tag) {
            normalized.push(tag);
        }
    }
    if normalized.is_empty() {
        return Err(Error::Validation(NO_TAGS.to_owned()));
    }
    Ok(normalized)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use ResourceKind::{Agent, McpServer, Team, Tool, Workflow};
    use TagMatchStrategy::{All, Any};
    use TagUpdateStrategy::{Merge, Replace};

    fn tag(value: &str) -> Tag {
        Tag::new(value)
    }

    fn tags(values: &[&str]) -> Vec<Tag> {
        values.iter().map(|value| tag(value)).collect()
    }

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn mgr() -> TagMgr {
        TagMgr::new()
    }

    fn tagged(manager: &TagMgr, kind: ResourceKind, id: &str, values: &[&str]) -> Vec<Tag> {
        manager.tag(kind, id, tags(values)).unwrap()
    }

    fn updated(
        manager: &TagMgr,
        kind: ResourceKind,
        id: &str,
        values: &[&str],
        strategy: TagUpdateStrategy,
    ) -> Vec<Tag> {
        manager
            .update_tags(kind, id, tags(values), strategy)
            .unwrap()
    }

    fn found(
        manager: &TagMgr,
        kind: ResourceKind,
        values: &[&str],
        strategy: TagMatchStrategy,
    ) -> Vec<String> {
        manager
            .resources_by_tags(kind, &tags(values), strategy)
            .unwrap()
    }

    fn removed(
        manager: &TagMgr,
        kind: ResourceKind,
        id: &str,
        values: &[&str],
        skip: bool,
    ) -> Vec<Tag> {
        manager.remove_tags(kind, id, &tags(values), skip).unwrap()
    }

    #[test]
    fn tag_serializes_as_a_bare_json_string() {
        assert_eq!(serde_json::to_value(tag("chat")).unwrap(), json!("chat"));
        let parsed: Tag = serde_json::from_value(json!("chat")).unwrap();
        assert_eq!(parsed, tag("chat"));
        let global: Tag = serde_json::from_value(json!(GLOBAL_TAG)).unwrap();
        assert_eq!(global, Tag::global());
    }

    #[test]
    fn resource_kind_round_trips_through_serde() {
        let expected = [
            (Tool, "tool"),
            (Agent, "agent"),
            (Workflow, "workflow"),
            (Team, "team"),
            (McpServer, "mcp_server"),
        ];
        assert_eq!(
            ResourceKind::ALL.map(|kind| kind.as_str()),
            expected.map(|(_, name)| name)
        );
        for (kind, name) in expected {
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(name));
            let parsed: ResourceKind = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(parsed, kind);
            assert_eq!(kind.as_str(), name);
            assert_eq!(kind.to_string(), name);
        }
    }

    #[test]
    fn strategies_round_trip_through_serde() {
        for (strategy, name) in [(Any, "any"), (All, "all")] {
            assert_eq!(serde_json::to_value(strategy).unwrap(), json!(name));
            let parsed: TagMatchStrategy = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(parsed, strategy);
        }
        for (strategy, name) in [(Merge, "merge"), (Replace, "replace")] {
            assert_eq!(serde_json::to_value(strategy).unwrap(), json!(name));
            let parsed: TagUpdateStrategy = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(parsed, strategy);
        }
    }

    #[test]
    fn tag_conversions_and_display_expose_the_bare_value() {
        let borrowed: Tag = "chat".into();
        let owned: Tag = String::from("chat").into();
        assert_eq!(borrowed, owned);
        assert_eq!(borrowed, Tag::new("chat"));
        assert_eq!(borrowed.as_str(), "chat");
        assert_eq!(borrowed.to_string(), "chat");
        assert_eq!(&*borrowed, "chat");
        assert!(!borrowed.is_global());

        let mut sorted = vec![tag("b"), Tag::global(), tag("a")];
        sorted.sort();
        assert_eq!(sorted, vec![Tag::global(), tag("a"), tag("b")]);

        let global = Tag::global();
        assert!(global.is_global());
        assert_eq!(global.as_str(), GLOBAL_TAG);
        assert_eq!(global.to_string(), GLOBAL_TAG);
    }

    #[test]
    fn tagging_records_and_reads_back_the_tags() {
        let manager = mgr();
        let result = tagged(&manager, Agent, "a1", &["chat", "beta"]);
        assert_eq!(result, tags(&["chat", "beta"]));
        assert_eq!(manager.tags_of(Agent, "a1"), result);
        assert!(manager.has_resource_tag(Agent, "a1", &tag("chat")));
        assert!(!manager.has_resource_tag(Agent, "a1", &tag("gamma")));
        assert!(manager.has_tag(&tag("chat")));
        assert!(!manager.has_tag(&tag("gamma")));
        // `list_tags` is sorted, not registration-ordered.
        assert_eq!(manager.list_tags(), tags(&["beta", "chat"]));
        assert_eq!((manager.len(), manager.is_empty()), (2, false));
        assert_eq!(manager.revision(), 1);
        assert_eq!(manager.resources(Agent), ids(&["a1"]));

        // The kind is the index partition, not part of the tag.
        assert_eq!(manager.tags_of(Tool, "a1"), Vec::new());
        assert!(!manager.has_resource_tag(Tool, "a1", &tag("chat")));
        assert_eq!(manager.resources(Tool), Vec::<String>::new());
    }

    #[test]
    fn duplicate_and_already_present_tags_are_recorded_once() {
        let manager = mgr();
        assert_eq!(
            tagged(&manager, Tool, "t1", &["x", "x", "y"]),
            tags(&["x", "y"])
        );
        assert_eq!(manager.revision(), 1);

        assert_eq!(
            tagged(&manager, Tool, "t1", &["x"]),
            tags(&["x", "y"]),
            "a known tag is neither duplicated nor reordered"
        );
        assert_eq!(manager.revision(), 1, "re-tagging is not a mutation");
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn the_global_tag_drops_every_other_tag_of_the_resource() {
        let manager = mgr();
        tagged(&manager, Agent, "a1", &["chat", "beta"]);
        tagged(&manager, Agent, "a2", &["chat"]);

        assert_eq!(
            manager.tag(Agent, "a1", [Tag::global()]).unwrap(),
            vec![Tag::global()]
        );
        assert_eq!(manager.tags_of(Agent, "a1"), vec![Tag::global()]);
        assert!(
            !manager.has_tag(&tag("beta")),
            "the orphaned tag is collected"
        );
        assert!(manager.has_tag(&tag("chat")), "a2 still carries it");
        assert!(manager.has_resource_tag(Agent, "a1", &Tag::global()));
        assert_eq!(
            manager.resources_with_tag(Agent, &Tag::global()),
            ids(&["a1"])
        );

        // A list that also names GLOBAL still ends up global-only.
        assert_eq!(
            manager
                .tag(Agent, "a2", tags(&["other", GLOBAL_TAG]))
                .unwrap(),
            vec![Tag::global()]
        );
        assert!(!manager.has_tag(&tag("other")));
    }

    #[test]
    fn adding_tags_to_a_global_resource_adds_nothing() {
        let manager = mgr();
        manager.tag(Tool, "t1", [Tag::global()]).unwrap();
        let revision = manager.revision();

        assert_eq!(tagged(&manager, Tool, "t1", &["chat"]), vec![Tag::global()]);
        assert_eq!(
            updated(&manager, Tool, "t1", &["chat"], Merge),
            vec![Tag::global()]
        );
        assert_eq!(manager.tags_of(Tool, "t1"), vec![Tag::global()]);
        assert!(!manager.has_tag(&tag("chat")));
        assert_eq!(manager.revision(), revision, "both calls are no-ops");
    }

    #[test]
    fn replace_on_a_global_resource_swaps_the_global_tag_out() {
        let manager = mgr();
        manager.tag(Tool, "t1", [Tag::global()]).unwrap();

        assert_eq!(
            updated(&manager, Tool, "t1", &["chat"], Replace),
            tags(&["chat"])
        );
        assert_eq!(manager.tags_of(Tool, "t1"), tags(&["chat"]));
        assert!(!manager.has_tag(&Tag::global()));
    }

    #[test]
    fn merge_unions_and_replace_swaps_the_tag_lists() {
        let manager = mgr();
        tagged(&manager, Agent, "a1", &["a", "b"]);

        assert_eq!(
            updated(&manager, Agent, "a1", &["b", "c"], Merge),
            tags(&["a", "b", "c"])
        );
        assert_eq!(
            updated(&manager, Agent, "a1", &["z"], Replace),
            tags(&["z"])
        );
        assert_eq!(manager.tags_of(Agent, "a1"), tags(&["z"]));
        assert_eq!(
            manager.list_tags(),
            tags(&["z"]),
            "a, b, and c are collected"
        );
        assert_eq!(manager.revision(), 3);
    }

    #[test]
    fn update_tags_creates_the_entry_for_an_untagged_resource() {
        let manager = mgr();
        assert_eq!(
            updated(&manager, Workflow, "w1", &["x"], Replace),
            tags(&["x"])
        );
        assert_eq!(
            updated(&manager, Team, "team1", &["y", "x"], Merge),
            tags(&["y", "x"])
        );
        assert_eq!(manager.tags_of(Workflow, "w1"), tags(&["x"]));
        assert_eq!(manager.tags_of(Team, "team1"), tags(&["y", "x"]));
        assert_eq!(manager.revision(), 2);
    }

    #[test]
    fn any_matches_the_union_and_all_matches_the_intersection() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);
        tagged(&manager, Tool, "t2", &["b", "c"]);
        tagged(&manager, Tool, "t3", &["d"]);

        assert_eq!(found(&manager, Tool, &["a", "c"], Any), ids(&["t1", "t2"]));
        assert_eq!(found(&manager, Tool, &["a", "b"], All), ids(&["t1"]));
        // A tag only some resources carry narrows ALL to nothing.
        assert_eq!(
            found(&manager, Tool, &["a", "d"], All),
            Vec::<String>::new()
        );
        // A single-tag ALL query is the carrier list.
        assert_eq!(found(&manager, Tool, &["b"], All), ids(&["t1", "t2"]));
        // Unknown tags are skipped rather than rejected.
        assert_eq!(found(&manager, Tool, &["ghost"], Any), Vec::<String>::new());
        assert_eq!(
            found(&manager, Tool, &["a", "ghost"], All),
            Vec::<String>::new()
        );
    }

    #[test]
    fn resource_lookups_are_sorted_and_partitioned_by_kind() {
        let manager = mgr();
        tagged(&manager, Tool, "t2", &["x"]);
        tagged(&manager, Agent, "t2", &["x"]);
        tagged(&manager, Tool, "t1", &["y"]);

        assert_eq!(manager.resources_with_tag(Tool, &tag("x")), ids(&["t2"]));
        assert_eq!(manager.resources_with_tag(Agent, &tag("x")), ids(&["t2"]));
        assert_eq!(manager.resources(Tool), ids(&["t1", "t2"]));
        assert_eq!(manager.resources(Agent), ids(&["t2"]));

        // Removing one kind's association leaves the other intact.
        assert_eq!(manager.remove_resource(Tool, "t2"), tags(&["x"]));
        assert!(manager.has_tag(&tag("x")));
        assert_eq!(
            manager.resources_with_tag(Tool, &tag("x")),
            Vec::<String>::new()
        );
        assert_eq!(manager.tags_of(Agent, "t2"), tags(&["x"]));
    }

    #[test]
    fn a_tag_is_collected_when_its_last_resource_loses_it() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);

        assert_eq!(removed(&manager, Tool, "t1", &["a"], false), tags(&["b"]));
        assert!(!manager.has_tag(&tag("a")));
        assert!(manager.has_tag(&tag("b")));

        assert_eq!(removed(&manager, Tool, "t1", &["b"], false), Vec::new());
        assert!(!manager.has_tag(&tag("b")));
        assert!(manager.is_empty());
        assert_eq!((manager.len(), manager.list_tags().len()), (0, 0));
        assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
        assert_eq!(manager.revision(), 3);
    }

    #[test]
    fn untag_removes_every_tag_of_one_resource() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);
        tagged(&manager, Tool, "t2", &["a"]);

        assert_eq!(manager.untag(Tool, "t1"), tags(&["a", "b"]));
        assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
        assert!(manager.has_tag(&tag("a")), "t2 still carries it");
        assert!(!manager.has_tag(&tag("b")));

        assert_eq!(manager.untag(Tool, "t1"), Vec::new());
        assert_eq!(manager.untag(Tool, "ghost"), Vec::new());
        assert_eq!(
            manager.revision(),
            3,
            "an unknown resource is not a mutation"
        );
    }

    #[test]
    fn remove_tag_strips_the_tag_from_every_resource() {
        let manager = mgr();
        tagged(&manager, Tool, "t2", &["x", "y"]);
        tagged(&manager, Agent, "t1", &["x"]);

        assert_eq!(manager.remove_tag(&tag("x")), ids(&["t1", "t2"]));
        assert!(!manager.has_tag(&tag("x")));
        assert_eq!(manager.tags_of(Tool, "t2"), tags(&["y"]));
        assert_eq!(manager.tags_of(Agent, "t1"), Vec::new());

        let revision = manager.revision();
        assert_eq!(manager.remove_tag(&tag("x")), Vec::<String>::new());
        assert_eq!(manager.remove_tag(&tag("ghost")), Vec::<String>::new());
        assert_eq!(
            manager.revision(),
            revision,
            "unknown tags are not mutations"
        );
    }

    #[test]
    fn remove_resource_cleans_up_every_tag_of_the_resource() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);
        tagged(&manager, McpServer, "t1", &["a"]);

        assert_eq!(manager.remove_resource(Tool, "t1"), tags(&["a", "b"]));
        assert!(!manager.has_resource_tag(Tool, "t1", &tag("a")));
        assert!(
            manager.has_tag(&tag("a")),
            "the MCP server still carries it"
        );
        assert!(!manager.has_tag(&tag("b")));
        assert_eq!(manager.remove_resource(Tool, "t1"), Vec::new());
    }

    #[test]
    fn remove_tags_reports_missing_tags_and_leaves_the_index_alone() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a"]);

        let error = manager
            .remove_tags(Tool, "t1", &tags(&["a", "b"]), false)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: tool 't1' does not have tag 'b'"
        );
        assert_eq!(manager.tags_of(Tool, "t1"), tags(&["a"]));
        assert_eq!(
            manager.revision(),
            1,
            "a rejected removal is not a mutation"
        );
    }

    #[test]
    fn remove_tags_skips_missing_tags_when_asked() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);

        assert_eq!(
            removed(&manager, Tool, "t1", &["b", "ghost"], true),
            tags(&["a"])
        );
        assert_eq!(manager.revision(), 2);
        assert_eq!(
            removed(&manager, Tool, "t1", &["ghost"], true),
            tags(&["a"]),
            "removing only absent tags is a no-op"
        );
        assert_eq!(manager.revision(), 2);
    }

    #[test]
    fn removing_tags_from_an_unknown_resource_is_not_found_or_a_no_op() {
        let manager = mgr();

        let error = manager
            .remove_tags(Tool, "ghost", &tags(&["a"]), false)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: tool 'ghost' is not registered"
        );
        assert_eq!(removed(&manager, Tool, "ghost", &["a"], true), Vec::new());
        assert_eq!(manager.revision(), 0);
        assert_eq!(manager.tags_of(Tool, "ghost"), Vec::new());
    }

    #[test]
    fn empty_ids_tags_and_lists_are_rejected() {
        let manager = mgr();
        let no_tags = "validation: at least one tag is required";

        assert_eq!(
            manager.tag(Tool, "t1", Vec::new()).unwrap_err().to_string(),
            no_tags
        );
        assert_eq!(
            manager
                .update_tags(Tool, "t1", Vec::new(), Merge)
                .unwrap_err()
                .to_string(),
            no_tags
        );
        assert_eq!(
            manager
                .resources_by_tags(Tool, &[], Any)
                .unwrap_err()
                .to_string(),
            no_tags
        );

        for blank in ["", "   ", "\t\n"] {
            let error = manager.tag(Tool, "t1", [tag(blank)]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: tag cannot be empty or whitespace only"
            );
            assert!(manager
                .remove_tags(Tool, "t1", &[tag(blank)], true)
                .is_err());
        }

        for blank in ["", "  "] {
            let error = manager.tag(Tool, blank, tags(&["a"])).unwrap_err();
            assert_eq!(
                error.to_string(),
                "validation: resource id cannot be empty or whitespace only"
            );
            assert_eq!(
                manager
                    .remove_tags(Tool, blank, &tags(&["a"]), true)
                    .unwrap_err()
                    .kind(),
                ErrorKind::Validation
            );
        }

        assert!(manager.is_empty());
        assert_eq!(
            manager.revision(),
            0,
            "validation failures are not mutations"
        );
    }

    #[test]
    fn revision_counts_mutations_and_reset_zeroes_it() {
        let manager = mgr();
        assert_eq!(manager.revision(), 0);

        tagged(&manager, Tool, "t1", &["a"]);
        tagged(&manager, Tool, "t1", &["a"]);
        assert_eq!(manager.revision(), 1, "the second tag is a no-op");
        tagged(&manager, Tool, "t2", &["a"]);
        assert_eq!(manager.revision(), 2);
        updated(&manager, Tool, "t1", &["a", "b"], Merge);
        assert_eq!(manager.revision(), 3);
        removed(&manager, Tool, "t2", &["a"], false);
        assert_eq!(manager.revision(), 4);
        assert_eq!(manager.remove_resource(Tool, "t1"), tags(&["a", "b"]));
        assert_eq!(manager.revision(), 5);

        tagged(&manager, Tool, "t1", &["a", "b"]);
        assert_eq!(manager.reset(), 2);
        assert_eq!(manager.revision(), 0);
        assert!(manager.is_empty());
        assert_eq!(manager.list_tags(), Vec::new());
    }

    #[test]
    fn clear_removes_the_index_as_one_mutation() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a", "b"]);
        tagged(&manager, Agent, "a1", &["c"]);
        assert_eq!(manager.revision(), 2);

        assert_eq!(manager.clear(), 3);
        assert!(manager.is_empty());
        assert_eq!(manager.revision(), 3);
        assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
        assert_eq!(manager.resources(Agent), Vec::<String>::new());

        assert_eq!(manager.clear(), 0);
        assert_eq!(manager.revision(), 3, "clearing an empty index is a no-op");
    }

    #[test]
    fn debug_output_reports_counts_tags_and_revision() {
        let manager = mgr();
        tagged(&manager, Tool, "t1", &["a"]);
        tagged(&manager, Agent, "a1", &["b"]);

        let rendered = format!("{manager:?}");
        assert!(rendered.contains("TagMgr"), "{rendered}");
        assert!(rendered.contains("(\"agent\", 1)"), "{rendered}");
        assert!(rendered.contains("(\"tool\", 1)"), "{rendered}");
        assert!(rendered.contains("tags: 2"), "{rendered}");
        assert!(rendered.contains("revision: 2"), "{rendered}");
        // The dump never lists tag values or ids.
        assert!(!rendered.contains("\"a\""), "{rendered}");
        assert!(!rendered.contains("t1"), "{rendered}");
    }

    #[test]
    fn default_builds_an_empty_index() {
        let manager = TagMgr::default();
        assert!(manager.is_empty());
        assert_eq!((manager.len(), manager.revision()), (0, 0));
        assert_eq!(manager.list_tags(), Vec::new());
    }
}
