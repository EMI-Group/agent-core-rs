//! The default current-subtree provider and the progressive-root catalog helpers.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.subtree` (v0.1.20).
//!
//! # Deviation: the fragment builder is injected
//!
//! Python's provider calls `render.disclosure.build_exposed_fragment` directly. Rendering
//! is a separate module here, so the provider takes a
//! [`FragmentBuildFn`] instead and returns
//! [`Error::Unsupported`](crate::error::Error) until one is supplied — it never panics
//! and never falls back to a silently empty fragment. A caller that only needs the
//! catalog helpers (`build_progressive_root` and friends) can construct the provider and
//! ignore the build path entirely.
//!
//! # Cache invariant
//!
//! The cache is keyed by `(generation, node_id, branch_path, top_k)`. `generation`
//! belongs to the *binding* — one immutable tree plus a counter — so an entry can only
//! ever describe the tree it was built from:
//!
//! - [`DefaultCurrentSubtreeProvider::rebind`] swaps the tree and bumps the generation in
//!   one write lock, then clears the cache;
//! - a reader that snapshotted the old generation may still insert an old-generation
//!   entry after that clear, but such an entry is unreachable (no later reader sees the
//!   old generation) and [`DefaultCurrentSubtreeProvider::clear_cache`] frees it;
//! - a stale entry can therefore never be served, which is what Python's `id(node)` key
//!   achieves only implicitly (it can never invalidate, because a rebuilt tree simply
//!   misses the cache).
//!
//! The only locks are two `std::sync::RwLock`s, both taken for a snapshot or a mutation
//! and never across an `.await` — every method here is synchronous by design.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use serde_json::{json, Value};

use crate::error::{unsupported, validation, Error, Result};
use crate::retrieval::search::config::ProgressiveSearchConfig;
use crate::retrieval::search::contracts::{CurrentSubtreeProvider, FragmentBuildFn};
use crate::retrieval::search::flow::{
    CurrentSubtree, ExposedFragment, SearchCursor, SelectableTarget,
};
use crate::retrieval::types::{RetrieverChoice, RetrieverItem, RetrieverNode, ROOT_NODE_ID};

/// Counts the items one subtree holds; the disclosure threshold's input.
pub type ItemCounterFn = Arc<dyn Fn(&RetrieverNode) -> usize + Send + Sync>;

/// The default [`CurrentSubtreeProvider`]: render, cache, and rebuild per cursor.
///
/// Mirrors Python `DefaultCurrentSubtreeProvider`. The provider is bound to exactly one
/// tree; [`DefaultCurrentSubtreeProvider::rebind`] is the only way to move it to another
/// one.
pub struct DefaultCurrentSubtreeProvider {
    /// The bound tree and its generation counter, swapped together.
    binding: RwLock<TreeBinding>,
    /// The traversal configuration; its [`DisclosureConfig`] is what a build renders with.
    ///
    /// [`DisclosureConfig`]: crate::retrieval::search::config::DisclosureConfig
    config: ProgressiveSearchConfig,
    /// The injected renderer; `None` makes [`CurrentSubtreeProvider::get_current_subtree`]
    /// answer [`Error::Unsupported`](crate::error::Error).
    builder: Option<FragmentBuildFn>,
    /// How many items a subtree holds, as the renderer's exposure threshold sees it.
    item_counter: ItemCounterFn,
    /// The fragments built so far, keyed by generation and cursor position.
    cache: RwLock<BTreeMap<SubtreeCacheKey, CachedSubtree>>,
    /// How many entries [`DefaultCurrentSubtreeProvider::clear_cache`] has dropped.
    evictions: AtomicU64,
}

/// One immutable tree plus the generation that identifies it in the cache.
struct TreeBinding {
    /// The tree a cursor's `node_id` is resolved against.
    root: Arc<RetrieverNode>,
    /// Bumped by every [`DefaultCurrentSubtreeProvider::rebind`].
    generation: u64,
}

/// The cache key of one exposed subtree.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SubtreeCacheKey {
    /// The binding generation the subtree was built from.
    generation: u64,
    /// The node the cursor pointed at.
    node_id: String,
    /// The id chain that reached the node.
    branch_path: Vec<String>,
    /// The `top_k` the subtree was exposed for.
    top_k: usize,
}

/// One cached exposure: the fragment and the targets derived from it.
///
/// The cursor is deliberately **not** cached: a hit pairs the cached exposure with the
/// caller's own cursor, so the emitted [`CurrentSubtree`] always describes the request
/// that asked for it.
#[derive(Debug, Clone)]
struct CachedSubtree {
    /// The rendered disclosure.
    fragment: ExposedFragment,
    /// The targets, in candidate-code order.
    selectable_targets: Vec<SelectableTarget>,
}

impl DefaultCurrentSubtreeProvider {
    /// Binds `root` with `config`, without a fragment builder.
    ///
    /// The provider is usable for planning immediately, but
    /// [`CurrentSubtreeProvider::get_current_subtree`] answers
    /// [`Error::Unsupported`](crate::error::Error) until a builder is supplied through
    /// [`DefaultCurrentSubtreeProvider::with_builder`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `root.node_id` when the root has
    /// no id — a tree nothing can be resolved against.
    pub fn new(root: Arc<RetrieverNode>, config: ProgressiveSearchConfig) -> Result<Self> {
        Self::bind(root, config, None)
    }

    /// Binds `root` with `config` and the renderer `builder`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `root.node_id` when the root has
    /// no id.
    pub fn with_builder(
        root: Arc<RetrieverNode>,
        config: ProgressiveSearchConfig,
        builder: FragmentBuildFn,
    ) -> Result<Self> {
        Self::bind(root, config, Some(builder))
    }

    /// Shared constructor of the two entry points.
    fn bind(
        root: Arc<RetrieverNode>,
        config: ProgressiveSearchConfig,
        builder: Option<FragmentBuildFn>,
    ) -> Result<Self> {
        if root.node_id.trim().is_empty() {
            return Err(validation("root.node_id", "must not be blank"));
        }
        Ok(Self {
            binding: RwLock::new(TreeBinding {
                root,
                generation: 0,
            }),
            config,
            builder,
            item_counter: Arc::new(RetrieverNode::item_count),
            cache: RwLock::new(BTreeMap::new()),
            evictions: AtomicU64::new(0),
        })
    }

    /// Replaces the item counter (Python's `subtree_item_count` callable).
    #[must_use]
    pub fn with_item_counter(self, counter: ItemCounterFn) -> Self {
        Self {
            item_counter: counter,
            ..self
        }
    }

    /// Swaps the bound tree, bumping the generation so no cached entry can be served.
    ///
    /// The swap and the generation bump happen inside one write lock, so no reader can
    /// observe the new tree with the old generation (or the reverse). The cache is then
    /// cleared; entries an in-flight reader inserts afterwards are tagged with the old
    /// generation and are unreachable.
    pub fn rebind(&self, root: Arc<RetrieverNode>) {
        let mut binding = self.binding.write().unwrap_or_else(PoisonError::into_inner);
        binding.root = root;
        binding.generation = binding.generation.wrapping_add(1);
        drop(binding);
        self.clear_cache();
    }

    /// Drops every cached exposure.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap_or_else(PoisonError::into_inner);
        let dropped = cache.len() as u64;
        cache.clear();
        drop(cache);
        self.evictions.fetch_add(dropped, Ordering::SeqCst);
    }

    /// How many exposures are cached right now.
    #[must_use]
    pub fn cached_len(&self) -> usize {
        self.cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// The generation of the current binding (bumped by every
    /// [`DefaultCurrentSubtreeProvider::rebind`]).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.binding
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .generation
    }

    /// How many cached exposures were dropped by
    /// [`DefaultCurrentSubtreeProvider::clear_cache`] (directly or through
    /// [`DefaultCurrentSubtreeProvider::rebind`]).
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::SeqCst)
    }
}

impl CurrentSubtreeProvider for DefaultCurrentSubtreeProvider {
    /// Exposes the subtree `cursor` points at.
    ///
    /// A cache hit rebuilds the [`CurrentSubtree`] around the **caller's** cursor, so the
    /// emitted cursor always matches the request even when the exposure is shared.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`](crate::error::Error) naming `cursor.node_id` when the bound
    ///   tree does not hold that node;
    /// - [`Error::Unsupported`](crate::error::Error) when no fragment builder is
    ///   configured;
    /// - whatever the injected builder reports.
    fn get_current_subtree(&self, cursor: &SearchCursor) -> Result<CurrentSubtree> {
        let (root, generation) = {
            let binding = self.binding.read().unwrap_or_else(PoisonError::into_inner);
            (Arc::clone(&binding.root), binding.generation)
        };
        let key = SubtreeCacheKey {
            generation,
            node_id: cursor.node_id.clone(),
            branch_path: cursor.branch_path.clone(),
            top_k: cursor.top_k,
        };
        let cached = {
            let cache = self.cache.read().unwrap_or_else(PoisonError::into_inner);
            cache.get(&key).cloned()
        };
        if let Some(cached) = cached {
            return Ok(cached.with_cursor(cursor));
        }

        let node = root.find(&cursor.node_id).ok_or_else(|| {
            Error::not_found(format!("search node not found: {}", cursor.node_id))
        })?;
        let builder = self.builder.as_ref().ok_or_else(|| {
            unsupported(
                "no fragment builder configured: progressive subtree exposure needs a renderer, \
                 supplied with DefaultCurrentSubtreeProvider::with_builder",
            )
        })?;
        let fragment = builder(
            node,
            &cursor.branch_path,
            &self.config.disclosure,
            (self.item_counter)(node),
        )?;
        let cached = CachedSubtree {
            selectable_targets: selectable_targets(&fragment),
            fragment,
        };
        {
            let mut cache = self.cache.write().unwrap_or_else(PoisonError::into_inner);
            cache.insert(key, cached.clone());
        }
        Ok(cached.with_cursor(cursor))
    }
}

impl CachedSubtree {
    /// Pairs the cached exposure with `cursor`, cloning the cached pieces.
    fn with_cursor(&self, cursor: &SearchCursor) -> CurrentSubtree {
        CurrentSubtree {
            cursor: cursor.clone(),
            fragment: self.fragment.clone(),
            selectable_targets: self.selectable_targets.clone(),
        }
    }
}

impl std::fmt::Debug for DefaultCurrentSubtreeProvider {
    /// Renders the provider's bookkeeping without the unprintable builder and counter.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let generation = self.generation();
        formatter
            .debug_struct("DefaultCurrentSubtreeProvider")
            .field("generation", &generation)
            .field("has_builder", &self.builder.is_some())
            .field("cached_len", &self.cached_len())
            .finish_non_exhaustive()
    }
}

/// Derives the selectable targets of `fragment` in candidate-code order.
///
/// A candidate code without a resolution is skipped and logged: the fragment's own code
/// assignment is authoritative, and a missing entry means the renderer assigned a code it
/// never described.
fn selectable_targets(fragment: &ExposedFragment) -> Vec<SelectableTarget> {
    let mut targets = Vec::with_capacity(fragment.candidate_codes.len());
    for code in &fragment.candidate_codes {
        if let Some(resolution) = fragment.code_to_resolution.get(code) {
            targets.push(SelectableTarget::new(resolution.clone()));
        } else {
            tracing::warn!(
                code = %code,
                fragment_fingerprint = %fragment.fragment_fingerprint,
                "candidate code has no resolution; skipping it"
            );
        }
    }
    targets
}

/// The cache key of one `RetrieverChoice` list: the SHA-256 of its canonical rendering.
///
/// Mirrors Python `choices_cache_key`: three string fields per choice (`choice_id`,
/// `payload`, `description`), rendered as canonical JSON (sorted keys, no whitespace) and
/// hashed with SHA-256. Order matters (the list is not sorted), so two catalogs that
/// differ only in choice order hash differently.
#[must_use]
pub fn choices_cache_key(choices: &[RetrieverChoice]) -> String {
    let payload: Vec<Value> = choices
        .iter()
        .map(|choice| {
            json!({
                "choice_id": choice.choice_id,
                "payload": choice.payload,
                "description": choice.description,
            })
        })
        .collect();
    crate::llm::stable_metadata_sha256(&Value::Array(payload))
}

/// Builds the synthetic progressive root of a `RetrieverChoice` catalog.
///
/// Mirrors Python `build_progressive_root` / `freeze_progressive_root`:
///
/// - a choice whose `payload` has no `.` becomes an item directly on the root;
/// - a dotted payload (`Skills.alpha.one`) creates one branch node per proper prefix
///   (`Skills`, `Skills.alpha`) and one item under the deepest branch;
/// - blank `choice_id` or blank `payload` choices are skipped;
/// - children are sorted by node id and items by `(label, item_id)`, both lowercased;
/// - every branch description is
///   [`build_progressive_branch_description`], which returns `""` by design.
///
/// The root is always [`ROOT_NODE_ID`]/`"ROOT"`; an empty or fully skipped catalog yields
/// an empty root (Python returns `None` there — this signature returns the node instead,
/// because a caller of a tree type should not have to unwrap).
#[must_use]
pub fn build_progressive_root(choices: &[RetrieverChoice]) -> RetrieverNode {
    let mut root = RootBuilder::new(ROOT_NODE_ID, ROOT_NODE_ID);
    for choice in choices {
        let choice_id = choice.choice_id.trim();
        let payload = choice.payload.trim();
        if choice_id.is_empty() || payload.is_empty() {
            continue;
        }
        let parts: Vec<&str> = payload
            .split('.')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        let item = RetrieverItem::new(choice_id, payload)
            .with_label(build_progressive_item_label(choice_id, payload))
            .with_description(choice.description.trim());
        if parts.len() <= 1 {
            root.items.push(item);
            continue;
        }
        let mut current = &mut root;
        for depth in 1..parts.len() {
            let branch_id = parts[..depth].join(".");
            current = current.child(&branch_id, parts[depth - 1]);
        }
        current.items.push(item);
    }
    root.freeze(&[])
}

/// One branch node under construction.
///
/// Python's `_NodeBuilder`: a child map keyed by the dotted node id plus the items
/// attached directly to the node.
struct RootBuilder {
    /// The node id (the dotted cid of the branch).
    node_id: String,
    /// The rendered label (the last path segment).
    label: String,
    /// The branch's children, keyed by node id.
    children: BTreeMap<String, Self>,
    /// The items attached directly to the branch.
    items: Vec<RetrieverItem>,
}

impl RootBuilder {
    /// Builds a childless, itemless branch.
    fn new(node_id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            label: label.into(),
            children: BTreeMap::new(),
            items: Vec::new(),
        }
    }

    /// Returns the child branch `node_id`, creating it (labelled `label`) when absent.
    fn child(&mut self, node_id: &str, label: &str) -> &mut Self {
        self.children
            .entry(node_id.to_owned())
            .or_insert_with(|| Self::new(node_id, label))
    }

    /// Freezes the branch and its descendants into a [`RetrieverNode`].
    ///
    /// `ancestors` is the id chain that reached this branch (root first, excluding the
    /// branch itself); it becomes the branch path handed to
    /// [`build_progressive_branch_description`].
    fn freeze(self, ancestors: &[String]) -> RetrieverNode {
        let mut path: Vec<String> = ancestors.to_vec();
        path.push(self.node_id.clone());
        let children = self
            .children
            .into_values()
            .map(|child| child.freeze(&path))
            .collect();
        let mut items: Vec<RetrieverItem> = self.items;
        items.sort_by(|left, right| {
            let left_label = if left.label.is_empty() {
                &left.item_id
            } else {
                &left.label
            };
            let right_label = if right.label.is_empty() {
                &right.item_id
            } else {
                &right.label
            };
            left_label
                .to_lowercase()
                .cmp(&right_label.to_lowercase())
                .then_with(|| {
                    left.item_id
                        .to_lowercase()
                        .cmp(&right.item_id.to_lowercase())
                })
        });
        RetrieverNode {
            description: build_progressive_branch_description(&path),
            node_id: self.node_id,
            label: self.label,
            children,
            items,
        }
    }
}

/// The display label of one progressive item.
///
/// Mirrors Python `build_progressive_item_label`, extended by the crate's combined
/// form: both parts non-blank yields `"<choice_id> — <last payload segment>"`, otherwise
/// the non-blank one, otherwise the payload verbatim.
#[must_use]
pub fn build_progressive_item_label(choice_id: &str, payload: &str) -> String {
    let display_name = choice_id.trim();
    let leaf_term = payload
        .split('.')
        .next_back()
        .map(str::trim)
        .unwrap_or_default();
    match (display_name.is_empty(), leaf_term.is_empty()) {
        (false, false) => format!("{display_name} — {leaf_term}"),
        (false, true) => display_name.to_owned(),
        (true, false) => leaf_term.to_owned(),
        (true, true) => payload.to_owned(),
    }
}

/// The description of one progressive branch: always `""`.
///
/// Python `build_progressive_branch_description` returns an empty string by design —
/// branch descriptions are never generated for a progressive root, because the payload
/// carries no text to describe a branch with. The signature keeps `branch_path` so the
/// port documents what the caller knows and Python discards, and so a later wave can
/// derive a description without changing call sites.
#[must_use]
pub fn build_progressive_branch_description(branch_path: &[String]) -> String {
    let _ = branch_path;
    String::new()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::DisclosureConfig;
    use crate::retrieval::search::flow::SelectableResolution;
    use std::sync::atomic::AtomicUsize;

    /// A tiny tree: ROOT -> Skills -> {alpha(item), beta(item)} plus a root item.
    fn tree() -> Arc<RetrieverNode> {
        Arc::new(RetrieverNode::new(ROOT_NODE_ID, "Root").with_children(vec![
            RetrieverNode::new("Skills", "Skills").with_children(vec![
                RetrieverNode::new("Skills.alpha", "Alpha").with_items(vec![RetrieverItem::new(
                    "alpha-one",
                    "Skills.alpha.one",
                )]),
                RetrieverNode::new("Skills.beta", "Beta"),
            ]),
        ]))
    }

    /// A resolution for `code`.
    fn resolution(code: &str, canonical_id: &str) -> SelectableResolution {
        SelectableResolution {
            code: code.to_owned(),
            canonical_id: canonical_id.to_owned(),
            display_name: canonical_id.to_owned(),
            label: canonical_id.to_owned(),
            description: String::new(),
            is_terminal: false,
            branch_path: vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()],
            node_id: Some("Skills.alpha".to_owned()),
            item: None,
        }
    }

    /// A builder that exposes one code per child of the resolved node.
    ///
    /// The exposure carries one deliberately unresolvable code (`"99"`) so the
    /// skip-and-warn path of the target derivation runs on every build.
    fn builder(seen: Arc<AtomicUsize>) -> FragmentBuildFn {
        Arc::new(
            move |node: &RetrieverNode,
                  branch_path: &[String],
                  _config: &DisclosureConfig,
                  item_count: usize|
                  -> Result<ExposedFragment> {
                assert_eq!(branch_path.first().map(String::as_str), Some(ROOT_NODE_ID));
                assert_eq!(item_count, node.item_count());
                seen.fetch_add(1, Ordering::SeqCst);
                let mut fragment = ExposedFragment {
                    rendered_tree: node.node_id.clone(),
                    ..ExposedFragment::default()
                };
                for (index, child) in node.children().into_iter().enumerate() {
                    let code = (index + 1).to_string();
                    fragment.candidate_codes.push(code.clone());
                    fragment.code_to_resolution.insert(
                        code,
                        SelectableResolution {
                            branch_path: branch_path.to_vec(),
                            ..resolution("", &format!("node::{}", child.node_id))
                        },
                    );
                }
                fragment.candidate_codes.push("99".to_owned());
                Ok(fragment)
            },
        )
    }

    #[test]
    fn missing_node_is_not_found() {
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            tree(),
            ProgressiveSearchConfig::default(),
            builder(Arc::new(AtomicUsize::new(0))),
        )
        .unwrap();
        let cursor = SearchCursor::at("nope", 1, vec![ROOT_NODE_ID.to_owned()], 3);
        let error = provider.get_current_subtree(&cursor).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
        assert!(error.message().contains("nope"));
    }

    #[test]
    fn without_a_builder_exposure_is_unsupported() {
        let provider =
            DefaultCurrentSubtreeProvider::new(tree(), ProgressiveSearchConfig::default()).unwrap();
        let cursor = SearchCursor::new(&tree(), 3);
        let error = provider.get_current_subtree(&cursor).unwrap_err();
        assert!(crate::error::is_unsupported(&error));
        assert!(error.message().contains("fragment builder"));
    }

    #[test]
    fn a_blank_root_id_is_rejected() {
        let error = DefaultCurrentSubtreeProvider::new(
            Arc::new(RetrieverNode::new("  ", "Root")),
            ProgressiveSearchConfig::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("root.node_id"));
    }

    #[test]
    fn cache_hits_rebuild_the_callers_cursor() {
        let seen = Arc::new(AtomicUsize::new(0));
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            tree(),
            ProgressiveSearchConfig::default(),
            builder(Arc::clone(&seen)),
        )
        .unwrap();
        let first = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], 4);
        let subtree = provider.get_current_subtree(&first).unwrap();
        assert_eq!(subtree.cursor, first);
        assert_eq!(subtree.selectable_targets.len(), 1);
        assert_eq!(subtree.selectable_targets[0].canonical_id(), "node::Skills");

        let second = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], 4);
        let cached = provider.get_current_subtree(&second).unwrap();
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "the second call hits the cache"
        );
        assert_eq!(cached.cursor, second);
        assert_eq!(cached.fragment, subtree.fragment);
        assert_eq!(provider.cached_len(), 1);
    }

    #[test]
    fn the_cache_grows_once_per_position_and_clears_on_rebind() {
        let seen = Arc::new(AtomicUsize::new(0));
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            tree(),
            ProgressiveSearchConfig::default(),
            builder(Arc::clone(&seen)),
        )
        .unwrap();
        for top_k in [1_usize, 1, 2] {
            let cursor = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], top_k);
            provider.get_current_subtree(&cursor).unwrap();
        }
        assert_eq!(
            provider.cached_len(),
            2,
            "one entry per (node, path, top_k)"
        );
        assert_eq!(seen.load(Ordering::SeqCst), 2);

        let before = provider.generation();
        provider.rebind(tree());
        assert_eq!(provider.generation(), before + 1);
        assert_eq!(provider.cached_len(), 0);
        assert_eq!(provider.evictions(), 2);

        let cursor = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], 1);
        provider.get_current_subtree(&cursor).unwrap();
        assert_eq!(
            seen.load(Ordering::SeqCst),
            3,
            "a rebind makes the next call rebuild"
        );
    }

    #[test]
    fn clearing_the_cache_manually_forces_a_rebuild() {
        let seen = Arc::new(AtomicUsize::new(0));
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            tree(),
            ProgressiveSearchConfig::default(),
            builder(Arc::clone(&seen)),
        )
        .unwrap();
        let cursor = SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], 1);
        provider.get_current_subtree(&cursor).unwrap();
        provider.clear_cache();
        assert_eq!(provider.cached_len(), 0);
        provider.get_current_subtree(&cursor).unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_custom_item_counter_is_used() {
        let counted = Arc::new(AtomicUsize::new(0));
        let counted_in_closure = Arc::clone(&counted);
        let observed = Arc::new(AtomicUsize::new(0));
        let counter_observed = Arc::clone(&observed);
        let provider = DefaultCurrentSubtreeProvider::with_builder(
            tree(),
            ProgressiveSearchConfig::default(),
            Arc::new(
                move |node: &RetrieverNode,
                      _path: &[String],
                      _config: &DisclosureConfig,
                      items: usize| {
                    counted_in_closure.fetch_add(items, Ordering::SeqCst);
                    Ok(ExposedFragment {
                        rendered_tree: node.node_id.clone(),
                        ..ExposedFragment::default()
                    })
                },
            ),
        )
        .unwrap()
        .with_item_counter(Arc::new(move |node: &RetrieverNode| {
            counter_observed.fetch_add(1, Ordering::SeqCst);
            node.item_count() + 7
        }));
        let cursor = SearchCursor::new(&tree(), 2);
        provider.get_current_subtree(&cursor).unwrap();
        assert_eq!(observed.load(Ordering::SeqCst), 1);
        // The root carries one item, so the counter reports 1 + 7.
        assert_eq!(counted.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn progressive_root_nests_dotted_payloads() {
        let root = build_progressive_root(&[
            RetrieverChoice {
                choice_id: "one".to_owned(),
                payload: "Skills.alpha.one".to_owned(),
                description: "first".to_owned(),
            },
            RetrieverChoice {
                choice_id: "two".to_owned(),
                payload: "Skills.beta.two".to_owned(),
                description: String::new(),
            },
            RetrieverChoice {
                choice_id: "flat".to_owned(),
                payload: "misc".to_owned(),
                description: String::new(),
            },
        ]);
        assert_eq!(root.node_id, ROOT_NODE_ID);
        assert_eq!(root.label, ROOT_NODE_ID);
        let root_items: Vec<&str> = root
            .items()
            .iter()
            .map(|item| item.item_id.as_str())
            .collect();
        assert_eq!(root_items, vec!["flat"]);
        let children = root.children();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].node_id, "Skills");
        let skills = children[0];
        let branches: Vec<&str> = skills
            .children()
            .iter()
            .map(|child| child.node_id.as_str())
            .collect();
        assert_eq!(branches, vec!["Skills.alpha", "Skills.beta"]);
        let alpha = skills.children()[0];
        assert_eq!(alpha.items().len(), 1);
        assert_eq!(alpha.items()[0].item_id, "one");
        assert_eq!(alpha.items()[0].payload, "Skills.alpha.one");
        assert_eq!(alpha.items()[0].label, "one — one");
        assert_eq!(alpha.items()[0].description, "first");
        assert!(
            alpha.description.is_empty(),
            "branch descriptions stay empty"
        );
    }

    #[test]
    fn progressive_root_sorts_items_and_skips_incomplete_choices() {
        let root = build_progressive_root(&[
            RetrieverChoice::new("Beta", "Group.x"),
            RetrieverChoice::new("alpha", "Group.y"),
            RetrieverChoice::new("", "Group.z"),
            RetrieverChoice::new("blank", "  "),
        ]);
        let group = root.children()[0];
        // Stored order is Python's (label casefolds first); the `items()` accessor applies
        // the crate's own item-id ordering on top of whatever order is stored.
        let stored: Vec<&str> = group.items.iter().map(|item| item.label.as_str()).collect();
        assert_eq!(stored, vec!["alpha — y", "Beta — x"]);
        let read: Vec<&str> = group
            .items()
            .iter()
            .map(|item| item.label.as_str())
            .collect();
        assert_eq!(read, vec!["Beta — x", "alpha — y"]);
    }

    #[test]
    fn an_empty_catalog_yields_an_empty_root() {
        let root = build_progressive_root(&[]);
        assert_eq!(root.node_id, ROOT_NODE_ID);
        assert!(root.children.is_empty());
        assert!(root.items.is_empty());
    }

    #[test]
    fn item_labels_cover_every_combination() {
        assert_eq!(build_progressive_item_label("id", "a.b.c"), "id — c");
        assert_eq!(build_progressive_item_label(" id ", " a.b.c "), "id — c");
        assert_eq!(build_progressive_item_label("id", "   "), "id");
        assert_eq!(build_progressive_item_label("  ", "a.b"), "b");
        assert_eq!(build_progressive_item_label("", ""), "");
        assert_eq!(build_progressive_item_label("", "  raw  "), "raw");
    }

    #[test]
    fn branch_descriptions_stay_empty() {
        assert_eq!(
            build_progressive_branch_description(&["ROOT".to_owned(), "Skills".to_owned()]),
            String::new()
        );
        assert_eq!(build_progressive_branch_description(&[]), String::new());
    }

    #[test]
    fn choices_cache_key_is_stable_and_sensitive() {
        let choices = vec![RetrieverChoice {
            choice_id: "a".to_owned(),
            payload: "p".to_owned(),
            description: "d".to_owned(),
        }];
        let first = choices_cache_key(&choices);
        assert_eq!(first.len(), 64);
        assert_eq!(first, choices_cache_key(&choices.clone()));

        let mut changed = choices.clone();
        changed[0].description = "other".to_owned();
        assert_ne!(first, choices_cache_key(&changed));

        let mut reordered = choices.clone();
        reordered.push(RetrieverChoice::new("b", "q"));
        assert_ne!(first, choices_cache_key(&reordered));

        assert_eq!(choices_cache_key(&[]).len(), 64);
        assert_ne!(first, choices_cache_key(&[]));
    }
}
