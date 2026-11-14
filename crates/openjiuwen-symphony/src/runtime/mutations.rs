//! The graph-mutation path: request digests, the durable idempotency journal,
//! compare-and-swap on the source revision, and the recorded results.
//!
//! [`SymphonyRuntime::mutate`](crate::runtime::SymphonyRuntime::mutate) is the one entry
//! point every capability mutation funnels through; the three positional methods
//! ([`add_capabilities`](crate::runtime::SymphonyRuntime::add_capabilities),
//! [`update_capabilities`](crate::runtime::SymphonyRuntime::update_capabilities),
//! [`remove_capabilities`](crate::runtime::SymphonyRuntime::remove_capabilities)) are thin
//! delegations that build a [`MutationRequest`] with no `source_revision`.
//!
//! # Ordering
//!
//! One call runs, under the runtime's mutation lock:
//!
//! 1. **journal replay** — a known `request_id` answers from the recorded result (with
//!    [`MutationStatus::Replayed`]) instead of re-applying the batch, and a reuse of that
//!    id with a different [digest](crate::orchestration::mutation::request_digest) is
//!    [`Error::Validation`] naming `request_id`;
//! 2. **source-revision compare-and-swap** — a declared `source_revision` must equal the
//!    current artifact revision, otherwise the batch is stale and rejected naming
//!    `source_revision`;
//! 3. **apply** — the catalog is mutated, with `add`/`update` validating first so a bad
//!    batch changes nothing, and `delete` skipping unregistered keys;
//! 4. **change-set check** — the ids the catalog actually changed must match the request
//!    (see [`validate_mutation_against_inventory`] and the delete subset rule below);
//! 5. **publish** — the graph is re-derived and published as a new immutable revision;
//! 6. **record** — the result is kept in memory and, when a journal is configured, written
//!    through atomically so a rebuilt runtime replays it too.
//!
//! A rejected step returns its error before anything is published, so the revision chain
//! never advances for a request that was not fully accepted. The one exception is a
//! *journal write* failure (step 6): the revision is already published, so the error is
//! returned while the in-memory record keeps answering replays for this process.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use openjiuwen_common::utils::id::now_rfc3339;

use crate::error::{validation, Error, Result};
use crate::model::{CapabilityDescriptor, CapabilityKey};
use crate::orchestration::mutation::{
    validate_mutation_against_inventory, JournalRecord, MutationJournal, MutationOperation,
    MutationRequest,
};
use crate::runtime::{qualified_ids, SymphonyRuntime};

/// What one mutation call did to the published artifacts.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MutationStatus {
    /// The batch changed the catalog and published a new revision.
    #[default]
    Published,
    /// The batch changed nothing in the catalog; the revision was re-published unchanged.
    Noop,
    /// The request id was already recorded: the recorded result was answered instead.
    Replayed,
}

impl MutationStatus {
    /// The lowercase name used in serialized results and logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::Noop => "noop",
            Self::Replayed => "replayed",
        }
    }
}

/// What one capability mutation changed.
///
/// `changed` holds the sorted qualified ids the mutation touched (inserted/replaced for
/// add and update, actually removed for delete), and `added_edges`/`removed_edges` count
/// the `(source, target)` graph pairs that appeared or disappeared. A replay of a
/// recorded `request_id` returns the recorded result with [`Self::replayed`] set and
/// [`Self::status`] [`MutationStatus::Replayed`].
///
/// The remaining fields are additive bookkeeping: `status` distinguishes a real
/// publication from a no-op batch and a replay, `digest` is the replay identity of the
/// request, `source_revision` echoes the compare-and-swap revision the caller declared, and
/// `diagnostics` carries non-fatal notes (empty for the built-in path).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphMutationResult {
    /// Which mutation ran.
    pub operation: MutationOperation,
    /// The caller's idempotency key.
    pub request_id: String,
    /// The sorted qualified ids that changed.
    pub changed: Vec<String>,
    /// The number of `(source, target)` edge pairs the mutation added.
    pub added_edges: usize,
    /// The number of `(source, target)` edge pairs the mutation removed.
    pub removed_edges: usize,
    /// The version of the revision published by this mutation.
    pub revision: String,
    /// Whether this result is a replay of an earlier mutation with the same id.
    pub replayed: bool,
    /// What the call did to the artifacts.
    pub status: MutationStatus,
    /// The SHA-256 request digest this result is recorded under.
    pub digest: String,
    /// The graph revision the caller declared, when it declared one.
    pub source_revision: Option<String>,
    /// Non-fatal notes recorded with the result.
    pub diagnostics: Vec<String>,
}

/// One recorded result, plus the digest that produced it.
#[derive(Debug, Clone)]
struct RecordedMutation {
    /// The [`MutationRequest::digest`] of the recorded request.
    digest: String,
    /// The recorded public result (with `replayed = false`).
    result: GraphMutationResult,
}

/// The recorded mutation results, keyed by caller `request_id`, with the optional
/// write-through journal.
#[derive(Debug, Default)]
pub(super) struct MutationLedger {
    /// The durable journal, when one is configured.
    journal: Option<MutationJournal>,
    /// Results recorded in this process, in `request_id` order.
    recorded: BTreeMap<String, RecordedMutation>,
}

impl MutationLedger {
    /// Builds a ledger over an optional journal.
    pub(super) fn with_journal(journal: Option<MutationJournal>) -> Self {
        Self {
            journal,
            recorded: BTreeMap::new(),
        }
    }

    /// The journal's recorded entry for `request_id`, when a journal is configured.
    async fn journal_lookup(&self, request_id: &str) -> Option<JournalRecord> {
        match &self.journal {
            Some(journal) => journal.lookup(request_id).await,
            None => None,
        }
    }

    /// The path of the configured journal, for error messages.
    fn journal_path(&self) -> Option<&Path> {
        self.journal.as_ref().map(MutationJournal::path)
    }

    /// Keeps `result` in memory and writes it through the journal when one is configured.
    async fn remember(
        &mut self,
        request: &MutationRequest,
        digest: &str,
        result: &GraphMutationResult,
    ) -> Result<()> {
        self.recorded.insert(
            request.request_id.clone(),
            RecordedMutation {
                digest: digest.to_string(),
                result: result.clone(),
            },
        );
        let Some(journal) = self.journal.as_mut() else {
            return Ok(());
        };
        let payload = serde_json::to_value(result).map_err(|error| {
            Error::external(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("mutation result serialize: {error}"),
            ))
        })?;
        let record = JournalRecord::new(
            request.operation,
            result.revision.clone(),
            digest.to_string(),
        )
        .with_inserted_at(now_rfc3339())
        .with_result(payload);
        journal.record(request.request_id.clone(), record).await
    }
}

impl SymphonyRuntime {
    /// Applies one fully specified mutation batch and publishes the resulting revision.
    ///
    /// Requires [`RuntimeState::Running`](crate::runtime::RuntimeState::Running). The
    /// request is the contract: [`MutationRequest::changed_skill_ids`] must name the batch
    /// exactly (`add`/`update` take the descriptors to upsert, `delete` takes the keys to
    /// remove), a declared
    /// [`source_revision`](crate::orchestration::mutation::MutationRequest::source_revision)
    /// must equal the current artifact revision, and a repeated `request_id` answers the
    /// recorded result instead of re-applying the batch.
    ///
    /// `descriptors` is ignored for [`MutationOperation::Delete`], whose batch is the
    /// parsed qualified ids of the request; an id that is not `type:id` is
    /// [`Error::Validation`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `runtime.state`, `request_id`, `source_revision`,
    /// or `changed_skill_ids`, [`Error::NotFound`] for an update of an unregistered
    /// capability, and the store's and artifact backend's errors.
    pub async fn mutate(
        &self,
        request: MutationRequest,
        descriptors: Vec<CapabilityDescriptor>,
    ) -> Result<GraphMutationResult> {
        self.require_running()?;
        let digest = request.digest();
        let mut ledger = self.ledger.lock().await;

        if let Some(recorded) = ledger.recorded.get(&request.request_id).cloned() {
            return replay_recorded(recorded, &request, &digest);
        }
        if let Some(record) = ledger.journal_lookup(&request.request_id).await {
            let journal_path = ledger.journal_path().map_or_else(
                || "<mutation journal>".to_string(),
                |path| path.display().to_string(),
            );
            let result = replay_journal_record(&journal_path, record, &request, &digest)?;
            ledger.recorded.insert(
                request.request_id.clone(),
                RecordedMutation {
                    digest: digest.clone(),
                    result: result.clone(),
                },
            );
            debug!(
                request_id = %request.request_id,
                revision = %result.revision,
                "replayed a journaled capability mutation"
            );
            return Ok(result);
        }

        self.check_source_revision(request.source_revision.as_deref())
            .await?;
        let previous = self.edge_set().await?;
        let changed = self.apply_operation(&request, descriptors).await?;
        let capabilities = self.registry.list().await;
        validate_applied(&request, &changed, &qualified_ids(&capabilities))?;
        let revision = self.publish_graph(&capabilities).await?;
        let current = self.edge_set().await?;
        let status = if changed.is_empty() {
            MutationStatus::Noop
        } else {
            MutationStatus::Published
        };
        let result = GraphMutationResult {
            operation: request.operation,
            request_id: request.request_id.clone(),
            changed,
            added_edges: current.difference(&previous).count(),
            removed_edges: previous.difference(&current).count(),
            revision: revision.version,
            replayed: false,
            status,
            digest: digest.clone(),
            source_revision: request.source_revision.clone(),
            diagnostics: Vec::new(),
        };
        ledger.remember(&request, &digest, &result).await?;
        info!(
            operation = %result.operation,
            request_id = %result.request_id,
            status = result.status.as_str(),
            changed = result.changed.len(),
            added_edges = result.added_edges,
            removed_edges = result.removed_edges,
            revision = %result.revision,
            "applied a capability mutation"
        );
        Ok(result)
    }

    /// Requires a declared `source_revision` to be the current artifact revision.
    ///
    /// This is Python's `stale_graph_version` check: a caller that built its batch against
    /// a revision the engine has since replaced must re-read the graph and retry instead
    /// of overwriting the newer revision's edges.
    async fn check_source_revision(&self, expected: Option<&str>) -> Result<()> {
        let Some(expected) = expected else {
            return Ok(());
        };
        let current = self.artifacts.status().await?.current_version;
        if current.as_deref() == Some(expected) {
            return Ok(());
        }
        Err(validation(
            "source_revision",
            format!(
                "`{expected}` is not the current graph revision ({}); rebuild before retrying",
                current.as_deref().unwrap_or("<none>")
            ),
        ))
    }

    /// Applies `request` to the catalog and reports the sorted changed qualified ids.
    async fn apply_operation(
        &self,
        request: &MutationRequest,
        descriptors: Vec<CapabilityDescriptor>,
    ) -> Result<Vec<String>> {
        match request.operation {
            MutationOperation::Add => {
                check_batch_matches_descriptors(request, &descriptors)?;
                for descriptor in &descriptors {
                    descriptor.validate()?;
                }
                let changed = qualified_ids(&descriptors);
                self.registry.register_all(descriptors).await?;
                Ok(changed)
            }
            MutationOperation::Update => {
                check_batch_matches_descriptors(request, &descriptors)?;
                for descriptor in &descriptors {
                    descriptor.validate()?;
                    if !self.registry.contains(&descriptor.key()).await {
                        return Err(Error::not_found(format!(
                            "capability `{}` is not registered",
                            descriptor.qualified_id()
                        )));
                    }
                }
                let changed = qualified_ids(&descriptors);
                self.registry.register_all(descriptors).await?;
                Ok(changed)
            }
            MutationOperation::Delete => {
                let mut removed = BTreeSet::new();
                for key in parse_keys(request)? {
                    if self.registry.remove(&key).await? {
                        removed.insert(key.qualified());
                    }
                }
                Ok(removed.into_iter().collect())
            }
        }
    }
}

/// Answers a replay from a result recorded in this process.
fn replay_recorded(
    recorded: RecordedMutation,
    request: &MutationRequest,
    digest: &str,
) -> Result<GraphMutationResult> {
    if recorded.digest != digest {
        return Err(conflict(request));
    }
    let mut result = recorded.result;
    result.replayed = true;
    result.status = MutationStatus::Replayed;
    debug!(
        request_id = %request.request_id,
        revision = %result.revision,
        "replayed a recorded capability mutation"
    );
    Ok(result)
}

/// Answers a replay from a journal record, upgrading it into the in-memory ledger.
fn replay_journal_record(
    journal_path: &str,
    record: JournalRecord,
    request: &MutationRequest,
    digest: &str,
) -> Result<GraphMutationResult> {
    if record.digest != digest {
        return Err(conflict(request));
    }
    let mut result = match record.result {
        Some(payload) => {
            serde_json::from_value::<GraphMutationResult>(payload).map_err(|error| {
                validation(
                    "mutation_journal_path",
                    format!(
                        "{journal_path} holds an invalid result for `{}`: {error}",
                        request.request_id
                    ),
                )
            })?
        }
        None => GraphMutationResult {
            operation: record.operation,
            request_id: request.request_id.clone(),
            changed: request.changed_skill_ids.clone(),
            revision: record.revision,
            digest: record.digest,
            source_revision: request.source_revision.clone(),
            ..GraphMutationResult::default()
        },
    };
    result.request_id.clone_from(&request.request_id);
    result.replayed = true;
    result.status = MutationStatus::Replayed;
    Ok(result)
}

/// The error a reused request id with a different digest produces.
fn conflict(request: &MutationRequest) -> Error {
    validation(
        "request_id",
        format!(
            "`{}` already refers to a different graph mutation (digest mismatch)",
            request.request_id
        ),
    )
}

/// Requires the request's declared batch to be exactly the descriptors it carries.
fn check_batch_matches_descriptors(
    request: &MutationRequest,
    descriptors: &[CapabilityDescriptor],
) -> Result<()> {
    let provided: BTreeSet<String> = descriptors
        .iter()
        .map(CapabilityDescriptor::qualified_id)
        .collect();
    let declared = request.declared_ids();
    let missing: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|id| !provided.contains(*id))
        .collect();
    let extra: Vec<&str> = provided
        .iter()
        .filter(|id| !declared.contains(id.as_str()))
        .map(String::as_str)
        .collect();
    if missing.is_empty() && extra.is_empty() {
        return Ok(());
    }
    Err(validation(
        "changed_skill_ids",
        format!(
            "the request declares [{}] but the payload provides [{}]",
            missing.join(", "),
            extra.join(", ")
        ),
    ))
}

/// Requires what the catalog changed to match what the request declared.
///
/// `add` and `update` reproduce the declared batch exactly, so they are checked against
/// the inventory with [`validate_mutation_against_inventory`]. `delete` deliberately skips
/// unregistered keys (a documented no-op), so its actual removals are only required to be
/// a subset of the declared batch.
fn validate_applied(
    request: &MutationRequest,
    changed: &[String],
    inventory_ids: &[String],
) -> Result<()> {
    if request.operation != MutationOperation::Delete {
        return validate_mutation_against_inventory(request, changed, inventory_ids);
    }
    let declared = request.declared_ids();
    match changed.iter().find(|id| !declared.contains(id.as_str())) {
        Some(id) => Err(validation(
            "changed_skill_ids",
            format!("`{id}` changed without being declared by the request"),
        )),
        None => Ok(()),
    }
}

/// Parses the qualified ids of a `delete` request into registry keys.
fn parse_keys(request: &MutationRequest) -> Result<Vec<CapabilityKey>> {
    let mut keys = Vec::with_capacity(request.changed_skill_ids.len());
    for id in &request.changed_skill_ids {
        let key: CapabilityKey = id.parse()?;
        keys.push(key);
    }
    Ok(keys)
}
