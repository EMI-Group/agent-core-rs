//! Online RL task registry: the durable state machine behind the RL service.
//!
//! Python parity: `openjiuwen/agent_evolving/agent_rl/online/task_registry.py`
//! plus the registry-owned halves of `capture_pipeline.py`. This step ports the
//! **state machine only** — the training runner, the executors and the LoRA
//! repository, the Redis backend, and the HTTP routes are later steps.
//!
//! # Store-agnostic by construction
//!
//! [`TaskRegistry`] holds no state of its own: every operation is a sequence of
//! [`OnlineStore`] primitives (string get/set, conditional set, set add/remove/
//! members). That is deliberate — the in-memory store and the Redis store share
//! one implementation of the state machine, so a behaviour proven offline holds
//! on Redis, and no rule lives in two places. Python gets the same effect with
//! WATCH/MULTI pipelines; Rust gets it with a linearizable per-key seam plus
//! [`OnlineStore::set_if_absent`] on the create and claim paths.
//!
//! # State machines
//!
//! Task lifecycle (reserved for the service layer):
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
//! implementations in [`capture`]; the transition tables there are the
//! authority on which transitions are legal and which error they raise.
//!
//! # Key layout
//!
//! One versioned namespace, `rl:v1`, written by the `keys` module:
//!
//! | Key | Kind | Contents |
//! |---|---|---|
//! | `rl:v1:task:{id}` | string | serialized [`TaskRecord`] |
//! | `rl:v1:active_session:{session}` | string | the session's active task id |
//! | `rl:v1:tasks:active` | set | task ids that may still be active (recovery index) |
//! | `rl:v1:task:{id}:captures` | set | every capture id of the task |
//! | `rl:v1:task:{id}:captures:open` | set | captures waiting for their response |
//! | `rl:v1:task:{id}:captures:committed` | set | captures with a staged sample |
//! | `rl:v1:task:{id}:capture:{cid}` | string | serialized [`CaptureMeta`] |
//! | `rl:v1:task:{id}:turn:{turn}:captures` | set | captures committed in one turn |
//! | `rl:v1:task:{id}:turn:current` | string | the turn accepting captures |
//! | `rl:v1:task:{id}:turn:transition` | string | the turn waiting to become current |
//! | `rl:v1:task:{id}:turns:closed` | set | turns whose samples were published |
//! | `rl:v1:task:{id}:turn:{turn}:publish` | string | `publishing` / `published` / `failed` |
//! | `rl:v1:task:{id}:pending:{cid}` | string | staged sample JSON awaiting publication |
//! | `rl:v1:task:{id}:reward` | string | `{reward, status, sample_count}` projection record |
//!
//! `:captures:open`, `:captures:committed` and `:turns:closed` are the indexes
//! the guards read, so they are never derived by scanning another set. Deleting
//! a set's last member deletes nothing in Redis but drops the key in the
//! in-memory store; both mean "empty" to every reader.
//!
//! # Determinism
//!
//! - Timestamps come from the injected [`Clock`]; [`FixedClock`] is the default,
//!   and nothing in this module calls `SystemTime`, `Instant` or `rand`.
//! - Set reads are sorted, so a scan is ordered on both backends.
//! - Fingerprints are caller-supplied strings; this module hashes nothing.
//! - Ids are caller-supplied, so a test can script an entire session.
//!
//! # Errors
//!
//! Every failure is an [`OnlineError`], which carries the service envelope's
//! `code()` and `http_status()`; `From<OnlineError> for
//! openjiuwen_common::error::Error` bridges into the kernel. Registry methods
//! return [`Result`] (an alias of `Result<_, OnlineError>`) so the HTTP layer can
//! answer with the Python envelope without inspecting messages.

pub mod capture;
pub mod error;
pub mod store;
pub mod task;
#[cfg(test)]
mod tests;

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

pub use capture::{CaptureMeta, CaptureStatus, PublishState, TurnTransition};
pub use error::{OnlineError, Result};
pub use store::{MemoryOnlineStore, OnlineStore};
pub use task::{
    Clock, FinishReason, FixedClock, RewardMode, TaskRecord, TaskRegistry, TaskSpec, TaskStatus,
    EPOCH_STAMP,
};
