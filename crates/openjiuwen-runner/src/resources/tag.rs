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
//! The module also carries the thin [`crate::resources::ResourceMgr`] facade over
//! the index (`tag`, `update_tags`, `tags_of`, `resources_by_tags`, `untag`,
//! `remove_tag`, `tag_revision`). It adds only the registry existence check:
//! [`TagMgr`] treats an id it has never seen as untagged, while the manager
//! rejects an unregistered id with [`Error::NotFound`] because it owns the
//! registries.
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

use super::ResourceMgr;

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

/// The tag operations a host reaches through [`ResourceMgr`] rather than
/// [`TagMgr`] directly.
///
/// [`TagMgr`] has no registries, so it treats an id it has never seen as simply
/// untagged. [`ResourceMgr`] owns the registries, so the same operations reject
/// an id that is not registered for that [`ResourceKind`] up front with
/// [`Error::NotFound`] (via `crate::error::not_found`). Apart from that check
/// every method is a single delegation: the index semantics live in [`TagMgr`],
/// and the tag index is shared by all five kinds.
///
/// Reads stay infallible (`tags_of`, `untag`, `remove_tag`), and the removal
/// helpers are pure delegations as well, so removing tags from an id the runner
/// no longer knows is a no-op rather than an error: removal races registry
/// cleanup by design.
impl ResourceMgr {
    /// Tags a registered resource and returns its resulting full tag list.
    ///
    /// The tag list is appended in input order and duplicate values are recorded
    /// once; [`GLOBAL_TAG`] is exclusive (see [`TagMgr::tag`]).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `id` is not registered for `kind`, and the
    /// validation failures [`TagMgr::tag`] reports (empty id, empty or
    /// whitespace-only tag value, empty `tags` input).
    pub fn tag(
        &self,
        kind: ResourceKind,
        id: &str,
        tags: impl IntoIterator<Item = Tag>,
    ) -> Result<Vec<Tag>> {
        self.ensure_tagged_resource(kind, id)?;
        self.tags().tag(kind, id, tags)
    }

    /// Updates a registered resource's tags under `strategy`, returning the
    /// resulting full tag list.
    ///
    /// [`TagUpdateStrategy::Merge`] unions and [`TagUpdateStrategy::Replace`]
    /// swaps the set (see [`TagMgr::update_tags`]).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `id` is not registered for `kind`, and the
    /// validation failures [`TagMgr::update_tags`] reports.
    pub fn update_tags(
        &self,
        kind: ResourceKind,
        id: &str,
        tags: impl IntoIterator<Item = Tag>,
        strategy: TagUpdateStrategy,
    ) -> Result<Vec<Tag>> {
        self.ensure_tagged_resource(kind, id)?;
        self.tags().update_tags(kind, id, tags, strategy)
    }

    /// Returns the tags of one resource in registration order.
    ///
    /// This never errors: an untagged or entirely unknown id yields an empty
    /// list (delegating to [`TagMgr::tags_of`]), which keeps read paths free of
    /// existence checks.
    #[must_use]
    pub fn tags_of(&self, kind: ResourceKind, id: &str) -> Vec<Tag> {
        self.tags().tags_of(kind, id)
    }

    /// Returns the sorted ids of `kind` carrying `tags` under `strategy`.
    ///
    /// [`TagMatchStrategy::Any`] is the union of the carriers and
    /// [`TagMatchStrategy::All`] the intersection (see
    /// [`TagMgr::resources_by_tags`]). A tag no resource of `kind` carries
    /// simply contributes nothing, so a search without hits is `Ok(vec![])`
    /// rather than an error.
    ///
    /// # Errors
    ///
    /// The validation failures [`TagMgr::resources_by_tags`] reports for an
    /// empty `tags` slice and for an empty or whitespace-only tag value.
    ///
    /// # Note
    ///
    /// This is the one tag operation with no resource id to validate: it already
    /// spans every registered id of `kind`, so there is nothing for
    /// [`ResourceMgr`] to reject beyond the tag values themselves.
    pub fn resources_by_tags(
        &self,
        kind: ResourceKind,
        tags: &[Tag],
        strategy: TagMatchStrategy,
    ) -> Result<Vec<String>> {
        self.tags().resources_by_tags(kind, tags, strategy)
    }

    /// Removes every tag of one resource and returns the removed tags.
    ///
    /// A pure delegation to [`TagMgr::untag`]: an id with no recorded tags (an
    /// untagged or already-removed resource) is a no-op returning an empty list,
    /// because removal races registry cleanup by design.
    #[must_use]
    pub fn untag(&self, kind: ResourceKind, id: &str) -> Vec<Tag> {
        self.tags().untag(kind, id)
    }

    /// Removes one tag from every resource and returns the ids that carried it.
    ///
    /// A pure delegation to [`TagMgr::remove_tag`]: the result is sorted and
    /// de-duplicated, and an unknown tag is a no-op returning an empty list.
    #[must_use]
    pub fn remove_tag(&self, tag: &Tag) -> Vec<String> {
        self.tags().remove_tag(tag)
    }

    /// Returns the tag index's mutation counter (see [`TagMgr::revision`]).
    #[must_use]
    pub fn tag_revision(&self) -> u64 {
        self.tags().revision()
    }

    /// Rejects an id that is not registered for `kind`.
    ///
    /// The check reads the public card lookups (or, for
    /// [`ResourceKind::McpServer`], the MCP-server manager's configuration), so
    /// no registry internals leak into this module.
    fn ensure_tagged_resource(&self, kind: ResourceKind, id: &str) -> Result<()> {
        let registered = match kind {
            ResourceKind::Tool => self.tool_card(id).is_some(),
            ResourceKind::Agent => self.agent_card(id).is_some(),
            ResourceKind::Workflow => self.workflow_card(id).is_some(),
            ResourceKind::Team => self.team_card(id).is_some(),
            ResourceKind::McpServer => self.mcp_servers().config(id).is_some(),
        };
        if registered {
            return Ok(());
        }
        Err(not_found(kind.as_str(), id))
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
#[path = "tag/tests.rs"]
mod tests;
