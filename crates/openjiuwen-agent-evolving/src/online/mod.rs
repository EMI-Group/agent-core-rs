//! Online RL service state machine: task registry, sample ledger, training runs.
//!
//! Python parity: `openjiuwen/agent_evolving/agent_rl/online/` —
//! `task_registry.py` and the registry-owned halves of `capture_pipeline.py`
//! (the task/capture machine), `storage/trajectory_store.py` (the sample ledger),
//! `online/training_runner.py` (the training-run machine), `abstract/trainer.py`
//! plus `scheduler/ppo_executor.py` (the executor seam) and
//! `storage/lora_repo.py` (the artifact repository).
//!
//! # Store-agnostic by construction
//!
//! [`TaskRegistry`](crate::online::TaskRegistry), [`SampleLedger`](crate::online::SampleLedger) and
//! [`TrainingRunner`](crate::online::TrainingRunner) hold no state of their own: every operation is a
//! sequence of [`OnlineStore`](crate::online::OnlineStore) primitives (string get/set, conditional set,
//! set add/remove/members). That is deliberate — the in-memory store and the
//! Redis store share one implementation of every rule, so a behaviour proven
//! offline holds on Redis, and no rule lives in two places. Python gets the same
//! effect with WATCH/MULTI pipelines; Rust gets it with a linearizable per-key
//! seam plus [`OnlineStore::set_if_absent`](crate::online::OnlineStore::set_if_absent) on the create and claim paths.
//!
//! # State machines
//!
//! Task lifecycle (reserved for the capture pipeline):
//!
//! ```text
//!            start (new id)                   finish (no open captures)
//!   (absent) ───────────────▶ active ────────────────────────────▶ finalized
//!                                │                                    ▲
//!                                │ abort (drops open captures         │ finish/abort
//!                                │        and unpublished samples)    │ are idempotent
//!                                ▼                                    │
//!                             aborted ────────────────────────────────┘
//! ```
//!
//! Capture lifecycle and turn publication are documented beside their
//! implementations in [`capture`](crate::online::capture); the sample ledger's status machine in
//! [`sample`](crate::online::sample); and the training-run lifecycle — claim → training → activating →
//! succeeded/failed/canceled plus the restart rule — in [`training`](crate::online::training). Those
//! transition tables are the authority on which transitions are legal and which
//! error they raise.
//!
//! # Key layout
//!
//! One versioned namespace, `rl:v1`, written by the `keys` module:
//!
//! | Key | Kind | Contents |
//! |---|---|---|
//! | `rl:v1:task:{id}` | string | serialized [`TaskRecord`](crate::online::TaskRecord) |
//! | `rl:v1:active_session:{session}` | string | the session's active task id |
//! | `rl:v1:tasks:active` | set | task ids that may still be active (recovery index) |
//! | `rl:v1:task:{id}:captures` | set | every capture id of the task |
//! | `rl:v1:task:{id}:captures:open` | set | captures waiting for their response |
//! | `rl:v1:task:{id}:captures:committed` | set | captures with a staged sample |
//! | `rl:v1:task:{id}:capture:{cid}` | string | serialized [`CaptureMeta`](crate::online::CaptureMeta) |
//! | `rl:v1:task:{id}:turn:{turn}:captures` | set | captures committed in one turn |
//! | `rl:v1:task:{id}:turn:current` | string | the turn accepting captures |
//! | `rl:v1:task:{id}:turn:transition` | string | the turn waiting to become current |
//! | `rl:v1:task:{id}:turns:closed` | set | turns whose samples were published |
//! | `rl:v1:task:{id}:turn:{turn}:publish` | string | `publishing` / `published` / `failed` |
//! | `rl:v1:task:{id}:pending:{cid}` | string | staged sample JSON awaiting publication |
//! | `rl:v1:task:{id}:reward` | string | `{reward, status, sample_count}` projection record |
//! | `rl:v1:sample:{id}` | string | serialized [`SampleRecord`](crate::online::SampleRecord) |
//! | `rl:v1:samples:{status}` | set | sample ids in one status index |
//! | `rl:v1:sample:seq` | string | monotonic sample creation counter |
//! | `rl:v1:training_run:{id}` | string | serialized [`TrainingRunRecord`](crate::online::TrainingRunRecord) |
//! | `rl:v1:training_run:active` | string | the single run the service may execute |
//! | `rl:v1:training_runs` | set | every claimed run id (recovery and stats index) |
//! | `rl:v1:training_run:seq` | string | monotonic run-id counter |
//! | `rl:v1:training_run:{id}:artifact` | string | serialized [`TrainingArtifact`](crate::online::TrainingArtifact) |
//! | `rl:v1:training_run:{id}:activation` | string | activation recorded by [`NoopLoraActivator`](crate::online::NoopLoraActivator) |
//!
//! `:captures:open`, `:captures:committed` and `:turns:closed` are the indexes
//! the guards read, so they are never derived by scanning another set. Deleting
//! a set's last member deletes nothing in Redis but drops the key in the
//! in-memory store; both mean "empty" to every reader. A training-run id must
//! never be the literal `active`, because the marker shares its prefix; run ids
//! are allocated as `run-{seq:012x}`. The LoRA repository is filesystem-backed
//! and writes no store key at all.
//!
//! # Determinism
//!
//! - Timestamps come from the injected [`Clock`](crate::online::Clock); [`FixedClock`](crate::online::FixedClock) is the default,
//!   and nothing in this module calls `SystemTime`, `Instant` or `rand`.
//! - Set reads are sorted, so a scan is ordered on both backends.
//! - Fingerprints are caller-supplied strings; this module hashes nothing.
//! - Ids are caller-supplied for tasks and samples, and allocated from a store
//!   counter (`run-{seq:012x}`) for training runs, so a test can script an
//!   entire session.
//! - Queue order is creation order (`created_seq`, then the id), never a ZSET
//!   score derived from the wall clock.
//!
//! # Errors
//!
//! Every failure of the state machines is an [`OnlineError`](crate::online::OnlineError), which carries
//! the service envelope's `code()` and `http_status()`; `From<OnlineError> for
//! openjiuwen_common::error::Error` bridges into the kernel, and
//! `From<openjiuwen_common::error::Error> for OnlineError` classifies the errors
//! raised at the executor and activator seams. Registry methods return
//! [`Result`](crate::online::Result) (an alias of `Result<_, OnlineError>`) so the HTTP layer can
//! answer with the Python envelope without inspecting messages.

pub mod activation;
pub mod capture;
pub mod checkpoint;
pub mod error;
pub mod executor;
pub mod lora;
pub mod routes;
pub mod sample;
pub mod store;
pub mod task;
#[cfg(test)]
mod tests;
pub mod training;

/// Serializes a public record into its store representation.
pub(crate) mod json {
    use serde::de::DeserializeOwned;
    use serde::Serialize;

    use super::error::{OnlineError, Result};

    pub(crate) fn encode<T: Serialize>(value: &T) -> Result<String> {
        serde_json::to_string(value)
            .map_err(|error| OnlineError::store(openjiuwen_common::error::Error::external(error)))
    }

    /// Decodes a stored record.
    ///
    /// A payload this revision cannot read (a downgrade, a truncated write) is a
    /// storage fault, not a client error, so it surfaces as [`OnlineError::Store`]
    /// and keeps the serde error as its source.
    pub(crate) fn decode<T: DeserializeOwned>(payload: &str) -> Result<T> {
        serde_json::from_str(payload)
            .map_err(|error| OnlineError::store(openjiuwen_common::error::Error::external(error)))
    }
}

pub use activation::{LoraActivation, LoraActivator, NoopLoraActivator, UnsupportedLoraActivator};
pub use capture::{CaptureMeta, CaptureStatus, PublishState, TurnTransition};
pub use checkpoint::{TrainingCheckpoint, TRAINING_CHECKPOINT_VERSION};
pub use error::{OnlineError, Result};
pub use executor::{
    artifact_name, artifact_version, is_artifact_name, LocalArtifactDescriptor,
    LocalTrainingExecutor, PpoTrainingExecutor, TrainingArtifact, TrainingExecutor,
    TrainingRequest,
};
pub use lora::{
    parse_version_literal, version_literal, LoraAvailability, LoraRepository, LoraVersion,
};
pub use routes::RlServiceState;
pub use sample::{SampleLedger, SampleRecord, SampleStatus, SampleStore};
pub use store::{MemoryOnlineStore, OnlineStore};
pub use task::{
    Clock, FinishReason, FixedClock, RewardMode, TaskRecord, TaskRegistry, TaskSpec, TaskStatus,
    EPOCH_STAMP,
};
pub use training::{
    RunStage, RunStatus, TrainingRunRecord, TrainingRunner, TrainingRunnerConfig, SERVICE_RESTARTED,
};
