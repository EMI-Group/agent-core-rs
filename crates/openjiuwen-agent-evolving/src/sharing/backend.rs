//! The sharing backend seam and its two offline implementations.
//!
//! Python's sharing backends speak to a remote hub. Here the trait stays
//! backend-neutral (ids are strings, not paths): the local-file store is used for
//! single-machine sharing, the in-memory store for tests, and
//! [`RemoteHubSharingBackend`](crate::sharing::RemoteHubSharingBackend)
//! (module `remote`) implements the same trait over HTTP.
//!
//! The two local implementations share one ranking rule:
//! `score = 0.7 * keyword_ratio + 0.3 * intent_ratio` — see [`score_bundle`].

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};

use crate::experience::{jaccard, tokens};
use crate::sharing::types::{QueryKeywords, SharedSkillBundle, SkillSearchResult};

/// The weight of the keyword-overlap term in a search score.
const KEYWORD_WEIGHT: f64 = 0.7;
/// The weight of the free-text intent term in a search score.
const INTENT_WEIGHT: f64 = 0.3;

/// A pluggable store for published skill bundles.
///
/// Implementations are consumed through `&dyn SharingBackend`, so every method
/// must be safe to call concurrently: the trait requires `Send + Sync`, and no
/// implementation may hold a lock across an `.await`.
#[async_trait]
pub trait SharingBackend: Send + Sync {
    /// Persists `bundle` and returns the id it is stored under.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O or serialization failures.
    async fn publish(&self, bundle: &SharedSkillBundle) -> Result<String>;

    /// Returns every stored bundle, ordered by `bundle_id` ascending.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O or deserialization failures.
    async fn list(&self) -> Result<Vec<SharedSkillBundle>>;

    /// Returns the bundle stored under `bundle_id`, or `None` when absent.
    ///
    /// Absence is a normal outcome, never an error.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O failures, and deserialization failures for a
    /// malformed stored payload.
    async fn get(&self, bundle_id: &str) -> Result<Option<SharedSkillBundle>>;

    /// Removes `bundle_id` and reports whether it existed.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O failures.
    async fn remove(&self, bundle_id: &str) -> Result<bool>;

    /// Returns up to `limit` bundles ranked against `query`.
    ///
    /// `limit == 0` means unlimited; results with a score of `0.0` are
    /// dropped, and an empty query returns no results.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O or deserialization failures.
    async fn search(&self, query: &QueryKeywords, limit: usize) -> Result<Vec<SkillSearchResult>>;
}

/// A local directory of `<bundle_id>.json` files.
///
/// The backend is suitable for offline demos and tests: each bundle is one
/// document, `publish` writes atomically through a temporary file plus rename,
/// and a missing directory reads as empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalFileSharingBackend {
    /// The directory bundles are stored in.
    dir: PathBuf,
}

impl LocalFileSharingBackend {
    /// Creates a backend rooted at `dir`; the directory is created on first
    /// publish.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory bundles are stored in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The path of `bundle_id`'s document.
    fn path_for(&self, bundle_id: &str) -> PathBuf {
        self.dir.join(format!("{bundle_id}.json"))
    }
}

#[async_trait]
impl SharingBackend for LocalFileSharingBackend {
    async fn publish(&self, bundle: &SharedSkillBundle) -> Result<String> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let target = self.path_for(&bundle.bundle_id);
        let temporary = self.dir.join(format!(
            ".{}.json.tmp-{}",
            bundle.bundle_id,
            next_temporary_id()
        ));
        let payload = serde_json::to_string_pretty(bundle)?;
        tokio::fs::write(&temporary, payload).await?;
        if let Err(error) = tokio::fs::rename(&temporary, &target).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error.into());
        }
        Ok(bundle.bundle_id.clone())
    }

    async fn list(&self) -> Result<Vec<SharedSkillBundle>> {
        let mut bundles = Vec::new();
        let mut entries = match tokio::fs::read_dir(&self.dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(bundles),
            Err(error) => return Err(error.into()),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
                continue;
            }
            let content = tokio::fs::read_to_string(&path).await?;
            let bundle: SharedSkillBundle = serde_json::from_str(&content)?;
            bundles.push(bundle);
        }
        bundles.sort_by(|left, right| left.bundle_id.cmp(&right.bundle_id));
        Ok(bundles)
    }

    async fn get(&self, bundle_id: &str) -> Result<Option<SharedSkillBundle>> {
        match tokio::fs::read_to_string(self.path_for(bundle_id)).await {
            Ok(content) => Ok(Some(serde_json::from_str(&content)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn remove(&self, bundle_id: &str) -> Result<bool> {
        match tokio::fs::remove_file(self.path_for(bundle_id)).await {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    async fn search(&self, query: &QueryKeywords, limit: usize) -> Result<Vec<SkillSearchResult>> {
        let bundles = self.list().await?;
        Ok(rank_bundles(bundles, query, limit))
    }
}

/// An in-process store used by tests and ephemeral callers.
///
/// Semantics match [`LocalFileSharingBackend`]; bundles live in a
/// `BTreeMap` so `list` is already ordered by `bundle_id`.
#[derive(Debug, Default)]
pub struct MemorySharingBackend {
    /// The stored bundles keyed by `bundle_id`.
    bundles: Mutex<BTreeMap<String, SharedSkillBundle>>,
}

impl MemorySharingBackend {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Locks the map, mapping a poisoned lock to an external error.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, SharedSkillBundle>>> {
        self.bundles.lock().map_err(|error| {
            Error::external(std::io::Error::other(format!(
                "memory sharing backend lock poisoned: {error}"
            )))
        })
    }
}

#[async_trait]
impl SharingBackend for MemorySharingBackend {
    async fn publish(&self, bundle: &SharedSkillBundle) -> Result<String> {
        let bundle_id = bundle.bundle_id.clone();
        self.lock()?.insert(bundle_id.clone(), bundle.clone());
        Ok(bundle_id)
    }

    async fn list(&self) -> Result<Vec<SharedSkillBundle>> {
        Ok(self.lock()?.values().cloned().collect())
    }

    async fn get(&self, bundle_id: &str) -> Result<Option<SharedSkillBundle>> {
        Ok(self.lock()?.get(bundle_id).cloned())
    }

    async fn remove(&self, bundle_id: &str) -> Result<bool> {
        Ok(self.lock()?.remove(bundle_id).is_some())
    }

    async fn search(&self, query: &QueryKeywords, limit: usize) -> Result<Vec<SkillSearchResult>> {
        let bundles: Vec<SharedSkillBundle> = self.lock()?.values().cloned().collect();
        Ok(rank_bundles(bundles, query, limit))
    }
}

/// A process-local counter that keeps temporary file names distinct.
fn next_temporary_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Scores, filters and ranks `bundles` against `query`.
fn rank_bundles(
    bundles: impl IntoIterator<Item = SharedSkillBundle>,
    query: &QueryKeywords,
    limit: usize,
) -> Vec<SkillSearchResult> {
    let query_keywords: Vec<String> = query
        .keywords
        .iter()
        .map(|keyword| keyword.trim().to_lowercase())
        .filter(|keyword| !keyword.is_empty())
        .collect();
    let intent = query.intent.trim();
    if query_keywords.is_empty() && intent.is_empty() {
        return Vec::new();
    }
    let mut results: Vec<SkillSearchResult> = bundles
        .into_iter()
        .filter_map(|bundle| {
            let score = score_bundle(&bundle, &query_keywords, intent);
            if score <= 0.0 {
                return None;
            }
            Some(SkillSearchResult {
                bundle_id: bundle.bundle_id,
                skill_id: bundle.skill_id,
                skill_name: bundle.skill_name,
                description: bundle.summary_aggregate,
                experience_count: bundle.experiences.len(),
                keywords: bundle.keywords_aggregate,
                score,
            })
        })
        .collect();
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.bundle_id.cmp(&right.bundle_id))
    });
    if limit > 0 {
        results.truncate(limit);
    }
    results
}

/// The relevance score of one bundle against a normalised query.
///
/// `keyword_ratio` is the fraction of query keywords whose lowercased form is
/// present in the bundle's aggregated keywords, skill name or skill id.
/// `intent_ratio` is the token Jaccard similarity between the intent and the
/// bundle's skill name, summary and aggregated keywords.
fn score_bundle(bundle: &SharedSkillBundle, query_keywords: &[String], intent: &str) -> f64 {
    let keyword_ratio = if query_keywords.is_empty() {
        0.0
    } else {
        let mut haystack: BTreeSet<String> = bundle
            .keywords_aggregate
            .iter()
            .map(|keyword| keyword.trim().to_lowercase())
            .filter(|keyword| !keyword.is_empty())
            .collect();
        haystack.insert(bundle.skill_name.trim().to_lowercase());
        haystack.insert(bundle.skill_id.trim().to_lowercase());
        let matched = query_keywords
            .iter()
            .filter(|keyword| haystack.contains(*keyword))
            .count();
        matched as f64 / query_keywords.len() as f64
    };
    let intent_ratio = if intent.is_empty() {
        0.0
    } else {
        let haystack = format!(
            "{} {} {}",
            bundle.skill_name,
            bundle.summary_aggregate,
            bundle.keywords_aggregate.join(" ")
        );
        jaccard(&tokens(intent), &tokens(&haystack))
    };
    KEYWORD_WEIGHT.mul_add(keyword_ratio, INTENT_WEIGHT * intent_ratio)
}
