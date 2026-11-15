//! The session link graph: how sessions descend from and fork each other.
//!
//! Ports the relationship half of Python's
//! `openjiuwen/core/session/session_controller/chain_session.py` (v0.1.20): the
//! downstream links a `ChainSession` tracks so a session's lineage is visible.
//! Python keeps those links as `.link` files next to a data container and pairs
//! them with a `ChainSession` that fans writes downstream; here the graph is
//! plain metadata — one `links.json` document per agent at
//! `<base>/<agent_id>/links.json`, read and written by
//! [`DiskSessionStore`](super::DiskSessionStore) — and traversed on demand.
//!
//! A node carries at most one lineage pointer: [`LinkKind::Parent`] for a
//! session created inside or spawned from another session, [`LinkKind::Fork`]
//! for a session branched from another session's state. Children are recorded on
//! the parent's entry (in insertion order); forks are discovered by scanning
//! `forked_from`, so they never appear in `children`.
//!
//! # Why there is no `ChainSession`
//!
//! Python's `ChainSession` is a data container that pushes writes to every
//! downstream session of the same agent. In this port a session is an
//! `Arc<dyn Session>` that the caller already owns, state writes go through
//! [`Session::update_state`](crate::Session::update_state) on that handle, and
//! run loops decide what a fork inherits. Duplicating that fan-out here would
//! need a second, shadow state store, so the port keeps the graph as metadata
//! and leaves propagation to the crates that own a run loop. Use
//! [`SessionLinks::chain`] plus the [`DiskSessionStore`](super::DiskSessionStore)
//! wrappers to walk a root→leaf path, and copy state yourself.
//!
//! ```
//! use openjiuwen_session::session_controller::SessionLinks;
//!
//! let mut links = SessionLinks::new();
//! links.record_parent("child", "root").unwrap();
//! links.record_fork("branch", "child").unwrap();
//! assert_eq!(links.chain("branch").unwrap(), ["root", "child", "branch"]);
//! assert_eq!(links.depth("branch").unwrap(), 2);
//! ```
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use crate::checkpoint::validate_id;

use super::schema::now_seconds;
use super::store::{read_json_optional, write_json_atomic};

/// The file name of the per-agent link index.
pub(crate) const LINKS_FILE: &str = "links.json";

/// How a session link was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// A child session created inside, or spawned from, another session.
    Parent,
    /// A session branched from another session's state.
    Fork,
}

impl LinkKind {
    /// The wire name of the kind (`"parent"` / `"fork"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parent => "parent",
            Self::Fork => "fork",
        }
    }
}

impl fmt::Display for LinkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One node of the session link graph.
///
/// Field names are the JSON contract (`snake_case`), unknown fields are
/// tolerated, and every field defaults, so a partial hand-written document still
/// deserializes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionLink {
    /// The session this entry describes.
    pub session_id: String,
    /// The session this one was created from ([`LinkKind::Parent`]).
    pub parent: Option<String>,
    /// The session whose state this one was branched from ([`LinkKind::Fork`]).
    pub forked_from: Option<String>,
    /// Sessions created from this one, in insertion order.
    pub children: Vec<String>,
    /// Creation time as an RFC 3339 string.
    pub created_at: String,
}

impl SessionLink {
    /// Creates a link from `session_id` back to `origin`.
    ///
    /// `created_at` is stamped with the current RFC 3339 time; `children` starts
    /// empty (a later child of `session_id` records itself on this entry).
    #[must_use]
    pub fn new(session_id: impl Into<String>, kind: LinkKind, origin: impl Into<String>) -> Self {
        let origin = origin.into();
        let mut link = Self {
            session_id: session_id.into(),
            created_at: now_rfc3339(),
            ..Self::default()
        };
        match kind {
            LinkKind::Parent => link.parent = Some(origin),
            LinkKind::Fork => link.forked_from = Some(origin),
        }
        link
    }

    /// The session this one descends from, preferring `parent` over
    /// `forked_from`.
    #[must_use]
    pub fn origin(&self) -> Option<&str> {
        self.parent.as_deref().or(self.forked_from.as_deref())
    }

    /// The kind of link recorded on this entry (`None` for a graph root and for
    /// a placeholder created for an unknown parent).
    #[must_use]
    pub fn kind(&self) -> Option<LinkKind> {
        if self.parent.is_some() {
            Some(LinkKind::Parent)
        } else if self.forked_from.is_some() {
            Some(LinkKind::Fork)
        } else {
            None
        }
    }
}

/// The session link graph of one agent.
///
/// The value is the deserialized `links` object of the agent's `links.json`;
/// [`LinksIndex`] adds the document's agent id and timestamp.
///
/// Traversal is deterministic (entries are keyed by session id) and
/// **cycle-safe**: a stored cycle is an [`Error::Session`] naming the node
/// instead of an endless walk, and traversal of an unknown session id yields an
/// empty result. Adding a link that would create a cycle is
/// [`Error::Validation`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionLinks {
    links: BTreeMap<String, SessionLink>,
}

impl SessionLinks {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Wraps an existing map without validating it.
    #[must_use]
    pub fn from_map(links: BTreeMap<String, SessionLink>) -> Self {
        Self { links }
    }

    /// The graph as a map keyed by session id.
    #[must_use]
    pub fn as_map(&self) -> &BTreeMap<String, SessionLink> {
        &self.links
    }

    /// Consumes the graph and returns its map.
    #[must_use]
    pub fn into_map(self) -> BTreeMap<String, SessionLink> {
        self.links
    }

    /// Iterates over `(session_id, link)` pairs in session-id order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SessionLink)> {
        self.links
            .iter()
            .map(|(session_id, link)| (session_id.as_str(), link))
    }

    /// The number of linked sessions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Returns whether the graph holds no sessions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Looks up one session's link entry.
    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<&SessionLink> {
        self.links.get(session_id)
    }

    /// Returns whether `session_id` has a link entry.
    #[must_use]
    pub fn contains(&self, session_id: &str) -> bool {
        self.links.contains_key(session_id)
    }

    /// Records `link`, merging it with an existing entry for the same session.
    ///
    /// The entry's `children` are preserved (and any child listed on `link` is
    /// added), a previously recorded `created_at` survives an empty one, and a
    /// child already listed under another parent is detached from that parent's
    /// `children` so re-linking stays consistent. A parent that has no entry yet
    /// gets a placeholder entry (no lineage pointer) so `children_of` works.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed session id, a link whose
    /// parent (or fork origin) is the session itself, or a link that would make
    /// the session its own ancestor. Returns [`Error::Session`] when the stored
    /// graph already contains an unrelated cycle on the path being walked.
    pub fn link(&mut self, link: SessionLink) -> Result<()> {
        validate_id("session", &link.session_id)?;
        let mut link = link;
        for (kind, origin) in [
            (LinkKind::Parent, link.parent.clone()),
            (LinkKind::Fork, link.forked_from.clone()),
        ] {
            let Some(origin) = origin else { continue };
            validate_id("session", &origin)?;
            if origin == link.session_id {
                return Err(Error::validation(format!(
                    "session {} cannot be its own {}: a link must name a different session",
                    link.session_id, kind
                )));
            }
        }

        if let Some(existing) = self.links.get(&link.session_id).cloned() {
            if link.created_at.is_empty() {
                link.created_at.clone_from(&existing.created_at);
            }
            for child in existing.children {
                if !link.children.contains(&child) {
                    link.children.push(child);
                }
            }
        }
        if link.created_at.is_empty() {
            link.created_at = now_rfc3339();
        }

        if let Some(origin) = link.origin() {
            self.assert_reachable(origin, &link.session_id)?;
        }

        let session_id = link.session_id.clone();
        let parent = link.parent.clone();
        // Every referenced session gets an entry, so a chain that starts or ends
        // at a hand-recorded origin still exists in the graph.
        for origin in [link.parent.as_ref(), link.forked_from.as_ref()]
            .into_iter()
            .flatten()
        {
            self.links
                .entry(origin.clone())
                .or_insert_with(|| SessionLink {
                    session_id: origin.clone(),
                    created_at: now_rfc3339(),
                    ..SessionLink::default()
                });
        }
        for (id, entry) in &mut self.links {
            if id == &session_id || Some(id) == parent.as_ref() {
                continue;
            }
            entry.children.retain(|child| child != &session_id);
        }
        self.links.insert(session_id.clone(), link);
        if let Some(parent) = parent {
            let entry = self
                .links
                .entry(parent.clone())
                .or_insert_with(|| SessionLink {
                    session_id: parent,
                    created_at: now_rfc3339(),
                    ..SessionLink::default()
                });
            if !entry.children.iter().any(|child| child == &session_id) {
                entry.children.push(session_id);
            }
        }
        Ok(())
    }

    /// Records `child` as created from `parent` ([`LinkKind::Parent`]).
    ///
    /// # Errors
    ///
    /// See [`SessionLinks::link`].
    pub fn record_parent(&mut self, child: &str, parent: &str) -> Result<()> {
        self.link(SessionLink::new(child, LinkKind::Parent, parent))
    }

    /// Records `child` as branched from `origin`'s state ([`LinkKind::Fork`]).
    ///
    /// A fork keeps its own entry and is reported by [`SessionLinks::forks_of`]
    /// and by [`SessionLinks::descendants`], but it is never listed in its
    /// origin's `children`.
    ///
    /// # Errors
    ///
    /// See [`SessionLinks::link`].
    pub fn record_fork(&mut self, child: &str, origin: &str) -> Result<()> {
        self.link(SessionLink::new(child, LinkKind::Fork, origin))
    }

    /// The session `session_id` was created from, or `None` for an unknown
    /// session, a root, or a fork (use [`SessionLinks::forks_of`] for forks).
    #[must_use]
    pub fn parent_of(&self, session_id: &str) -> Option<&str> {
        self.links.get(session_id)?.parent.as_deref()
    }

    /// The sessions created from `session_id`, in insertion order.
    ///
    /// An unknown session yields an empty slice; forks are not included.
    #[must_use]
    pub fn children_of(&self, session_id: &str) -> &[String] {
        self.links
            .get(session_id)
            .map_or(&[], |link| link.children.as_slice())
    }

    /// The sessions forked from `session_id`, in session-id order.
    #[must_use]
    pub fn forks_of(&self, session_id: &str) -> Vec<String> {
        self.links
            .iter()
            .filter(|(_, link)| link.forked_from.as_deref() == Some(session_id))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Removes `session_id`'s entry.
    ///
    /// The entry is unlinked from its parent's `children` and dropped. **Its
    /// children keep their own entries but lose the parent pointer**, so they
    /// become graph roots; the same applies to the `forked_from` pointer of its
    /// forks. Returns the dropped entry, or `None` when the session was unknown.
    pub fn remove(&mut self, session_id: &str) -> Option<SessionLink> {
        let removed = self.links.remove(session_id)?;
        if let Some(parent) = removed.parent.as_ref() {
            if let Some(entry) = self.links.get_mut(parent) {
                entry.children.retain(|child| child != session_id);
            }
        }
        for child in &removed.children {
            if let Some(entry) = self.links.get_mut(child) {
                if entry.parent.as_deref() == Some(session_id) {
                    entry.parent = None;
                }
            }
        }
        for (id, entry) in &mut self.links {
            if id != session_id && entry.forked_from.as_deref() == Some(session_id) {
                entry.forked_from = None;
            }
        }
        Some(removed)
    }

    /// The sessions with no lineage pointer (neither a parent nor a fork
    /// origin), in session-id order.
    ///
    /// A placeholder created for an unknown parent counts as a root.
    #[must_use]
    pub fn roots(&self) -> Vec<String> {
        self.links
            .iter()
            .filter(|(_, link)| link.origin().is_none())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// The sessions with no children and no forks, in session-id order.
    #[must_use]
    pub fn leaves(&self) -> Vec<String> {
        let forks = self.forks_by_origin();
        self.links
            .iter()
            .filter(|(id, link)| link.children.is_empty() && !forks.contains_key(id.as_str()))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// The root→session path (inclusive) through the parent/fork chain.
    ///
    /// An unknown session yields an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the stored graph contains a cycle on the
    /// path, naming the repeated node.
    pub fn chain(&self, session_id: &str) -> Result<Vec<String>> {
        let mut path = self.lineage_walk(session_id)?;
        path.reverse();
        Ok(path)
    }

    /// The root→parent path of `session_id` (the chain without the session).
    ///
    /// An unknown session yields an empty vector.
    ///
    /// # Errors
    ///
    /// See [`SessionLinks::chain`].
    pub fn ancestors(&self, session_id: &str) -> Result<Vec<String>> {
        let chain = self.chain(session_id)?;
        let keep = chain.len().saturating_sub(1);
        Ok(chain[..keep].to_vec())
    }

    /// Every session downstream of `session_id`, breadth-first, excluding the
    /// session itself.
    ///
    /// Both children and forks are downstream; each level is emitted children
    /// first, then forks. An unknown session yields an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the stored graph contains a cycle.
    pub fn descendants(&self, session_id: &str) -> Result<Vec<String>> {
        if !self.contains(session_id) {
            return Ok(Vec::new());
        }
        self.check_acyclic()?;
        let forks = self.forks_by_origin();
        let mut visited = BTreeSet::from([session_id.to_string()]);
        let mut queue = VecDeque::from([session_id.to_string()]);
        let mut out = Vec::new();
        while let Some(node) = queue.pop_front() {
            let mut next: Vec<String> = self.children_of(&node).to_vec();
            if let Some(forks) = forks.get(node.as_str()) {
                next.extend(forks.iter().map(|fork| (*fork).to_string()));
            }
            for child in next {
                if visited.insert(child.clone()) {
                    out.push(child.clone());
                    queue.push_back(child);
                }
            }
        }
        Ok(out)
    }

    /// The number of lineage hops from a root to `session_id`; a root (and an
    /// unknown session) is `0`.
    ///
    /// # Errors
    ///
    /// See [`SessionLinks::chain`].
    pub fn depth(&self, session_id: &str) -> Result<usize> {
        Ok(self.lineage_walk(session_id)?.len().saturating_sub(1))
    }

    /// Verifies that no lineage chain in the graph loops.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] naming the repeated node when a stored chain
    /// is cyclic.
    pub fn check_acyclic(&self) -> Result<()> {
        for session_id in self.links.keys() {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            let mut cursor = Some(session_id.as_str());
            while let Some(node) = cursor {
                let Some(entry) = self.links.get(node) else {
                    break;
                };
                if !seen.insert(node) {
                    return Err(cycle_error(node));
                }
                cursor = entry.origin();
            }
        }
        Ok(())
    }

    /// Walks the lineage pointers from `session_id` towards a root, returning
    /// `[session_id, …, root]`.
    fn lineage_walk(&self, session_id: &str) -> Result<Vec<String>> {
        let mut path = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut cursor = Some(session_id.to_string());
        while let Some(node) = cursor {
            let Some(entry) = self.links.get(&node) else {
                // A named origin with no entry of its own is still part of the
                // lineage; an unknown starting session yields nothing.
                if !path.is_empty() {
                    path.push(node);
                }
                break;
            };
            if !seen.insert(node.clone()) {
                return Err(cycle_error(&node));
            }
            cursor = entry.origin().map(str::to_string);
            path.push(node);
        }
        Ok(path)
    }

    /// Fails when the lineage that leads up to `origin` already passes through
    /// `session_id` (the link would create a cycle).
    fn assert_reachable(&self, origin: &str, session_id: &str) -> Result<()> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut cursor = Some(origin);
        while let Some(node) = cursor {
            if node == session_id {
                return Err(Error::validation(format!(
                    "session link for {session_id} would create a cycle through {origin}"
                )));
            }
            let Some(entry) = self.links.get(node) else {
                break;
            };
            if !seen.insert(node) {
                return Err(cycle_error(node));
            }
            cursor = entry.origin();
        }
        Ok(())
    }

    /// Maps every session to the sessions forked from it.
    fn forks_by_origin(&self) -> BTreeMap<&str, Vec<&str>> {
        let mut forks: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (session_id, link) in &self.links {
            if let Some(origin) = link.forked_from.as_deref() {
                forks.entry(origin).or_default().push(session_id.as_str());
            }
        }
        forks
    }
}

/// The persisted per-agent link document (an agent's `links.json`).
///
/// The wire shape is
/// `{"agent_id": …, "updated_at": <epoch seconds>, "links": {<session_id>: …}}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LinksIndex {
    /// The agent this link document belongs to.
    pub agent_id: String,
    /// Last write time, as UTC seconds since the Unix epoch.
    pub updated_at: f64,
    /// The link entries keyed by session id.
    pub links: BTreeMap<String, SessionLink>,
}

impl LinksIndex {
    /// Builds a document for `agent_id` from `links`, stamped with the current
    /// time.
    #[must_use]
    pub fn new(agent_id: impl Into<String>, links: SessionLinks) -> Self {
        Self {
            agent_id: agent_id.into(),
            updated_at: now_seconds(),
            links: links.into_map(),
        }
    }

    /// The graph this document carries.
    #[must_use]
    pub fn into_links(self) -> SessionLinks {
        SessionLinks::from_map(self.links)
    }
}

/// Builds the [`Error::Session`] reported for a stored cycle.
fn cycle_error(session_id: &str) -> Error {
    Error::session(format!(
        "session link graph contains a cycle at session {session_id}"
    ))
}

/// `<root>/<agent_id>/links.json`, the agent's link document.
pub(crate) fn links_path(root: &Path, agent_id: &str) -> PathBuf {
    root.join(agent_id).join(LINKS_FILE)
}

/// Reads `<root>/<agent_id>/links.json`; a missing document yields an empty
/// graph.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a malformed agent id and [`Error::Session`]
/// when the document is unreadable or malformed.
pub(crate) async fn read_links(root: &Path, agent_id: &str) -> Result<SessionLinks> {
    validate_id("agent", agent_id)?;
    let index: Option<LinksIndex> = read_json_optional(&links_path(root, agent_id)).await?;
    Ok(index.map_or_else(SessionLinks::new, LinksIndex::into_links))
}

/// Atomically writes the graph to `<root>/<agent_id>/links.json`.
///
/// Every recorded session id is validated, so a graph assembled by hand cannot
/// smuggle a path-traversal id into the document.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a malformed agent or session id and
/// [`Error::Session`] for I/O or serialization failures.
pub(crate) async fn write_links(root: &Path, agent_id: &str, links: &SessionLinks) -> Result<()> {
    validate_id("agent", agent_id)?;
    for (session_id, link) in links.iter() {
        validate_id("session", session_id)?;
        if let Some(origin) = link.origin() {
            validate_id("session", origin)?;
        }
        for child in &link.children {
            validate_id("session", child)?;
        }
    }
    write_json_atomic(
        &links_path(root, agent_id),
        &LinksIndex::new(agent_id, links.clone()),
    )
    .await
}

/// Applies `mutate` to the persisted graph under `lock`.
///
/// The read-modify-write cycle is serialized across store clones, so concurrent
/// link updates never lose an entry.
///
/// # Errors
///
/// See [`read_links`] and [`write_links`], plus whatever `mutate` reports.
pub(crate) async fn update_links<F>(
    root: &Path,
    lock: &AsyncMutex<()>,
    agent_id: &str,
    mutate: F,
) -> Result<()>
where
    F: FnOnce(&mut SessionLinks) -> Result<()>,
{
    validate_id("agent", agent_id)?;
    let _guard = lock.lock().await;
    let mut links = read_links(root, agent_id).await?;
    mutate(&mut links)?;
    write_links(root, agent_id, &links).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{LinkKind, LinksIndex, SessionLink, SessionLinks};

    #[test]
    fn record_parent_links_both_directions() {
        let mut links = SessionLinks::new();
        links.record_parent("child", "root").unwrap();
        assert_eq!(links.parent_of("child"), Some("root"));
        assert_eq!(links.children_of("root"), vec!["child".to_string()]);
        assert_eq!(links.get("child").unwrap().kind(), Some(LinkKind::Parent));
        assert!(links.get("root").unwrap().kind().is_none());
        assert_ne!(links.get("child").unwrap().created_at, "");
    }

    #[test]
    fn a_fork_is_not_a_child() {
        let mut links = SessionLinks::new();
        links.record_fork("branch", "root").unwrap();
        assert_eq!(links.forks_of("root"), ["branch"]);
        assert_eq!(links.children_of("root").len(), 0);
        assert_eq!(links.get("branch").unwrap().kind(), Some(LinkKind::Fork));
        assert_eq!(links.chain("branch").unwrap(), ["root", "branch"]);
    }

    #[test]
    fn relinking_moves_the_child_between_parents() {
        let mut links = SessionLinks::new();
        links.record_parent("child", "a").unwrap();
        links.record_parent("child", "b").unwrap();
        assert_eq!(links.children_of("a").len(), 0);
        assert_eq!(links.children_of("b"), vec!["child".to_string()]);
    }

    #[test]
    fn remove_detaches_the_entry_and_orphans_children() {
        let mut links = SessionLinks::new();
        links.record_parent("mid", "root").unwrap();
        links.record_parent("leaf", "mid").unwrap();
        links.record_fork("branch", "mid").unwrap();
        let removed = links.remove("mid").unwrap();
        assert_eq!(removed.session_id, "mid");
        assert_eq!(links.children_of("root").len(), 0);
        assert!(links.parent_of("leaf").is_none());
        assert!(links.get("leaf").unwrap().forked_from.is_none());
        assert_eq!(links.roots(), ["branch", "leaf", "root"]);
        assert!(links.remove("mid").is_none());
    }

    #[test]
    fn unknown_sessions_traverse_to_empty_results() {
        let links = SessionLinks::new();
        assert_eq!(links.chain("nope").unwrap().len(), 0);
        assert_eq!(links.ancestors("nope").unwrap().len(), 0);
        assert_eq!(links.descendants("nope").unwrap().len(), 0);
        assert_eq!(links.children_of("nope").len(), 0);
        assert_eq!(links.depth("nope").unwrap(), 0);
    }

    #[test]
    fn self_links_and_cycles_are_validation_errors() {
        let mut links = SessionLinks::new();
        let error = links
            .link(SessionLink::new("a", LinkKind::Parent, "a"))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        links.record_parent("b", "a").unwrap();
        links.record_parent("c", "b").unwrap();
        let error = links.record_parent("a", "c").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("cycle"));
    }

    #[test]
    fn link_document_round_trips_through_json() {
        let mut links = SessionLinks::new();
        links.record_parent("child", "root").unwrap();
        let document = LinksIndex::new("agent", links.clone());
        assert_eq!(document.agent_id, "agent");
        let json = serde_json::to_value(&document).unwrap();
        assert_eq!(json["agent_id"], "agent");
        assert_eq!(json["links"]["child"]["parent"], "root");
        assert_eq!(
            json["links"]["root"]["children"],
            serde_json::json!(["child"])
        );
        let restored: LinksIndex = serde_json::from_value(json).unwrap();
        assert_eq!(restored.into_links(), links);
    }

    #[test]
    fn unknown_document_fields_are_tolerated() {
        let restored: LinksIndex = serde_json::from_value(serde_json::json!({
            "agent_id": "agent",
            "future": 1,
            "links": {"s1": {"session_id": "s1", "extra": "kept-lenient"}},
        }))
        .unwrap();
        let links = restored.into_links();
        assert_eq!(links.len(), 1);
        assert!(links.parent_of("s1").is_none());
    }
}
