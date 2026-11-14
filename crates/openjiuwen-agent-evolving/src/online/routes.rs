//! HTTP transport of the online RL state machines.
//!
//! Python parity: `openjiuwen/agent_evolving/agent_rl/online/service.py` — the
//! loopback RL Service's task, capture, training, upload and trajectory routes,
//! served over the state machines in [`crate::online`].
//!
//! This module owns three things:
//!
//! - [`RlServiceState`] — the in-process dependencies every handler needs: one
//!   shared [`MemoryOnlineStore`](crate::online::MemoryOnlineStore) behind a
//!   [`TaskRegistry`](crate::online::TaskRegistry), a
//!   [`SampleLedger`](crate::online::SampleLedger) and a
//!   [`TrainingRunner`](crate::online::TrainingRunner), plus the optional
//!   [`LoraRepository`](crate::online::LoraRepository) rooted at
//!   [`RlServiceConfig::lora_root`](crate::service::RlServiceConfig::lora_root).
//! - The route table itself, split by concern:
//!   [`tasks`] (task lifecycle and the internal finish/abort actions),
//!   [`capture`] (the completions before/after/discard hooks),
//!   [`training`] (training runs) and [`samples`] (batch upload and the
//!   trajectory read model).
//! - The helpers those handlers share: body decoding with a caller-chosen error
//!   code, required/optional field access, request fingerprinting, the
//!   [`OnlineError`](crate::online::OnlineError) → envelope mapping and delayed
//!   turn publication.
//!
//! # Transport, not policy
//!
//! Every rule — what "active" means, when a capture may commit, when a training
//! run may claim a batch — lives in the state machine modules. A handler only
//! decodes the wire shape, calls one state-machine method, and renders the
//! result (or the error) into Python's envelope. That keeps the offline unit
//! suites of [`crate::online`] authoritative and makes the HTTP layer auditable
//! route by route.
//!
//! # Deviations from Python, all deliberate
//!
//! - `POST /v1/rl/training/runs` awaits the run instead of detaching it into an
//!   executor thread pool: the local executor finishes inline, so the response
//!   body is the terminal record rather than a queued one. The status code
//!   contract (201 created, 200 active-reuse) is unchanged.
//! - A capture failure does not abort the whole task the way Python's
//!   `_abort_capture_failure` does; the conflict is reported and the caller may
//!   `POST /internal/v1/rl/tasks/{id}:abort` explicitly.
//! - The response `usage` counters are checked for integrality and for
//!   `total = prompt + completion`; Python additionally compares them against
//!   parsed token ids, which needs the tokenizer half of the gateway that this
//!   crate does not port.
//! - `POST /v1/gateway/upload/batch` always persists through the sample ledger
//!   and always uses the `invalid_rail_batch` code, because the rail-ingest and
//!   trajectory-store variants of Python share one ledger here.
//!
//! ## Route table
//!
//! | Method | Path | State machine call |
//! |---|---|---|
//! | `POST` | `/v1/rl/tasks/start` | `TaskRegistry::start` |
//! | `GET` | `/v1/rl/tasks/{rl_task_id}` | `TaskRegistry::get` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/stop` | `TaskRegistry::finish(UserStopped)` |
//! | `POST` | `/v1/rl/tasks/{rl_task_id}/reward` | `TaskRegistry::record_reward` |
//! | `POST` | `/v1/rl/training/runs` | `TrainingRunner::claim` + `run` |
//! | `GET` | `/v1/rl/training/runs/{id}` | `TrainingRunner::get` |
//! | `POST` | `/v1/rl/training/runs/{id}/stop` | `TrainingRunner::stop` |
//! | `GET` | `/v1/rl/trajectories/stats` | `SampleLedger::stats` |
//! | `GET` | `/v1/rl/trajectories` | `SampleLedger::list_by_status` |
//! | `GET` | `/v1/rl/trajectories/{id}` | `SampleLedger::get` |
//! | `POST` | `/internal/v1/completions:before` | `TaskRegistry::begin_capture` |
//! | `POST` | `/internal/v1/completions:after` | `TaskRegistry::commit_capture` |
//! | `POST` | `/internal/v1/completions:discard` | `TaskRegistry::discard_capture` |
//! | `POST` | `/internal/v1/rl/tasks/{*operation}` | `TaskRegistry::finish` / `abort` |
//! | `POST` | `/v1/gateway/upload/batch` | `SampleLedger::save_once` |

pub(crate) mod capture;
pub(crate) mod samples;
pub(crate) mod tasks;
pub(crate) mod training;

use std::path::PathBuf;
use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use openjiuwen_common::error::{Error, Result as KernelResult};
use serde_json::{Map, Value};

use super::activation::NoopLoraActivator;
use super::error::OnlineError;
use super::executor::{LocalTrainingExecutor, PpoTrainingExecutor, TrainingExecutor};
use super::lora::LoraRepository;
use super::sample::SampleLedger;
use super::store::MemoryOnlineStore;
use super::task::TaskRegistry;
use super::training::{TrainingRunner, TrainingRunnerConfig};
use crate::service::{ApiError, RlServiceConfig};

/// Directory name the local executor writes under when no `lora_root` is set.
const DEFAULT_ARTIFACT_DIR: &str = "openjiuwen-rl-service";

/// How often `completions:before` publishes a preceding turn before it gives up.
const MAX_TURN_TRANSITIONS: usize = 8;

/// The in-process dependencies of the online RL routes.
///
/// One [`MemoryOnlineStore`] backs all three machines, so a task started through
/// the HTTP layer is visible to the sample ledger and to the training runner
/// without any further wiring. The struct is built once per router and shared
/// behind an [`Arc`], and is [`Clone`] so a caller can keep a handle (checkpoint
/// exports, state inspection) after handing the router its own copy.
#[derive(Clone)]
pub struct RlServiceState {
    store: Arc<MemoryOnlineStore>,
    registry: Arc<TaskRegistry<Arc<MemoryOnlineStore>>>,
    ledger: Arc<SampleLedger<Arc<MemoryOnlineStore>>>,
    runner: Arc<TrainingRunner<Arc<MemoryOnlineStore>>>,
    lora: Option<LoraRepository>,
    executor: &'static str,
}

impl RlServiceState {
    /// Builds the state described by `config`.
    ///
    /// The executor is `"local"` (the deterministic offline
    /// [`LocalTrainingExecutor`](crate::online::LocalTrainingExecutor)) or
    /// `"ppo"` (the typed
    /// [`PpoTrainingExecutor`](crate::online::PpoTrainingExecutor) stub, which
    /// reports the missing GPU stack). Artifacts are written under `lora_root`,
    /// or a per-user temporary directory when the config leaves it unset.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for an unknown executor name, a zero
    /// `min_samples_for_training`, a zero `upload_max_samples`, or a
    /// `max_samples_per_run` below `min_samples_for_training`.
    pub fn new(config: &RlServiceConfig) -> KernelResult<Self> {
        if config.min_samples_for_training == 0 {
            return Err(Error::validation(
                "min_samples_for_training must be at least 1",
            ));
        }
        if config.max_samples_per_run < config.min_samples_for_training {
            return Err(Error::validation(
                "max_samples_per_run must be >= min_samples_for_training",
            ));
        }
        if config.upload_max_samples == 0 {
            return Err(Error::validation("upload_max_samples must be at least 1"));
        }
        let root = config
            .lora_root
            .as_deref()
            .map_or_else(default_artifact_root, PathBuf::from);
        let executor: Arc<dyn TrainingExecutor> = match config.executor.as_str() {
            "local" => Arc::new(LocalTrainingExecutor::new(&root)),
            "ppo" => Arc::new(PpoTrainingExecutor::new()),
            other => {
                return Err(Error::validation(format!(
                    "unknown executor: {other} (expected local or ppo)"
                )))
            }
        };
        let executor_name = executor.name();
        let store = Arc::new(MemoryOnlineStore::new());
        let registry = Arc::new(TaskRegistry::new(Arc::clone(&store)));
        let ledger = Arc::new(SampleLedger::new(Arc::clone(&store)));
        let runner_config = TrainingRunnerConfig {
            model_id: config.model_id.clone(),
            min_samples_for_training: config.min_samples_for_training,
            max_samples_per_run: config.max_samples_per_run,
            auto_activate_lora: config.auto_activate_lora,
            ..TrainingRunnerConfig::default()
        };
        let activator = Arc::new(NoopLoraActivator::new(Arc::clone(&store)));
        let runner = Arc::new(TrainingRunner::new(
            Arc::clone(&store),
            executor,
            activator,
            runner_config,
        ));
        Ok(Self {
            store,
            registry,
            ledger,
            runner,
            lora: config.lora_root.as_deref().map(LoraRepository::new),
            executor: executor_name,
        })
    }

    /// The store every machine writes through.
    #[must_use]
    pub fn store(&self) -> &Arc<MemoryOnlineStore> {
        &self.store
    }

    /// The task/capture registry.
    #[must_use]
    pub fn registry(&self) -> &Arc<TaskRegistry<Arc<MemoryOnlineStore>>> {
        &self.registry
    }

    /// The sample ledger (Python's `TrajectorySampleStore`).
    #[must_use]
    pub fn ledger(&self) -> &Arc<SampleLedger<Arc<MemoryOnlineStore>>> {
        &self.ledger
    }

    /// The training runner.
    #[must_use]
    pub fn runner(&self) -> &Arc<TrainingRunner<Arc<MemoryOnlineStore>>> {
        &self.runner
    }

    /// The `LoRA` artifact repository, present when `lora_root` is set.
    #[must_use]
    pub fn lora(&self) -> Option<&LoraRepository> {
        self.lora.as_ref()
    }

    /// The configured executor name (`"local"` or `"ppo"`).
    #[must_use]
    pub fn executor_name(&self) -> &'static str {
        self.executor
    }

    /// Probes the two dependencies `/health` reports on: the sample ledger and
    /// the training executor.
    ///
    /// # Errors
    ///
    /// The store failure, or the executor's typed
    /// [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) —
    /// which is what makes a `ppo` service answer `503` naming the missing GPU
    /// stack instead of claiming it can train.
    pub async fn check_health(&self) -> KernelResult<()> {
        self.ledger.stats().await?;
        self.runner.executor().check_health().await
    }
}

/// The per-user artifact directory used when `lora_root` is unset.
fn default_artifact_root() -> PathBuf {
    std::env::temp_dir().join(DEFAULT_ARTIFACT_DIR)
}

/// State of the router: the service config plus the (possibly unavailable) state.
///
/// A config the state cannot be built from is kept as a typed [`ApiError`], so
/// every RL route answers `503` naming the reason instead of panicking at
/// construction time.
#[derive(Clone)]
pub(crate) struct RouteState {
    config: Arc<RlServiceConfig>,
    online: std::result::Result<Arc<RlServiceState>, Arc<ApiError>>,
}

impl RouteState {
    /// Wraps a ready state.
    pub(crate) fn ready(config: RlServiceConfig, state: RlServiceState) -> Self {
        Self {
            config: Arc::new(config),
            online: Ok(Arc::new(state)),
        }
    }

    /// Wraps the reason the state could not be built.
    pub(crate) fn unavailable(config: RlServiceConfig, error: ApiError) -> Self {
        Self {
            config: Arc::new(config),
            online: Err(Arc::new(error)),
        }
    }

    /// The service configuration.
    pub(crate) fn config(&self) -> &RlServiceConfig {
        &self.config
    }

    /// The ready state, or the construction failure as a `503`.
    pub(crate) fn online(&self) -> ApiResult<&Arc<RlServiceState>> {
        self.online.as_ref().map_err(|error| (**error).clone())
    }
}

/// Result of a route handler.
pub(crate) type ApiResult<T> = std::result::Result<T, ApiError>;

/// The online RL routes, to be merged into the service router.
pub(crate) fn rl_router() -> Router<RouteState> {
    Router::new()
        .route("/v1/rl/tasks/start", post(tasks::start_task))
        .route("/v1/rl/tasks/{rl_task_id}", get(tasks::get_task))
        .route("/v1/rl/tasks/{rl_task_id}/stop", post(tasks::stop_task))
        .route("/v1/rl/tasks/{rl_task_id}/reward", post(tasks::reward_task))
        .route("/v1/rl/training/runs", post(training::start_run))
        .route(
            "/v1/rl/training/runs/{training_run_id}",
            get(training::get_run),
        )
        .route(
            "/v1/rl/training/runs/{training_run_id}/stop",
            post(training::stop_run),
        )
        // The static route is registered before the dynamic one so `stats` can
        // never be captured as a trajectory id.
        .route("/v1/rl/trajectories/stats", get(samples::trajectory_stats))
        .route("/v1/rl/trajectories", get(samples::list_trajectories))
        .route(
            "/v1/rl/trajectories/{trajectory_id}",
            get(samples::get_trajectory),
        )
        .route(
            "/internal/v1/completions:before",
            post(capture::before_completion),
        )
        .route(
            "/internal/v1/completions:after",
            post(capture::after_completion),
        )
        .route(
            "/internal/v1/completions:discard",
            post(capture::discard_completion),
        )
        // axum 0.8 cannot express a parameter that only covers part of a path
        // segment (`{id}:finish`), so the internal task actions are served by
        // one catch-all that splits the requested operation itself.
        .route(
            "/internal/v1/rl/tasks/{*operation}",
            post(tasks::internal_task_operation),
        )
        .route("/v1/gateway/upload/batch", post(samples::upload_batch))
}

/// Builds `400`/`409`/... envelope errors from a literal code.
pub(crate) fn fail(code: &str, status: StatusCode, message: impl Into<String>) -> ApiError {
    ApiError::with_status(code, message, status)
}

/// The generic envelope of a state-machine failure.
pub(crate) fn online_failure(error: OnlineError) -> ApiError {
    ApiError::from_online(error)
}

/// Maps a capture-pipeline failure onto Python's codes: conflicts and closed
/// turns are `409 completion_conflict` / `409 turn_closed`, everything else keeps
/// the state-machine code.
pub(crate) fn capture_failure(error: OnlineError) -> ApiError {
    match &error {
        OnlineError::TaskConflict(_) => ApiError::with_status(
            "completion_conflict",
            error.to_string(),
            StatusCode::CONFLICT,
        ),
        OnlineError::InvalidRequest(_) => ApiError::with_status(
            "invalid_completion",
            error.to_string(),
            StatusCode::BAD_REQUEST,
        ),
        _ => ApiError::from_online(error),
    }
}

/// Maps a task-route failure onto the task codes (`invalid_task_request`).
pub(crate) fn task_failure(error: OnlineError) -> ApiError {
    match &error {
        OnlineError::InvalidRequest(_) => ApiError::with_status(
            "invalid_task_request",
            error.to_string(),
            StatusCode::BAD_REQUEST,
        ),
        _ => ApiError::from_online(error),
    }
}

/// Decodes a JSON body; an absent body decodes as [`Value::Null`].
pub(crate) fn json_body(body: &[u8], code: &str) -> ApiResult<Value> {
    if body.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(body).map_err(|error| {
        fail(
            code,
            StatusCode::BAD_REQUEST,
            format!("body is not valid JSON: {error}"),
        )
    })
}

/// Decodes a JSON object body.
pub(crate) fn object_body(body: &[u8], code: &str) -> ApiResult<Map<String, Value>> {
    match json_body(body, code)? {
        Value::Object(fields) => Ok(fields),
        _ => Err(fail(
            code,
            StatusCode::BAD_REQUEST,
            "body must be an object",
        )),
    }
}

/// Reads a required string field.
pub(crate) fn required_str(
    fields: &Map<String, Value>,
    field: &str,
    code: &str,
) -> ApiResult<String> {
    match fields.get(field) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(other) => Err(fail(
            code,
            StatusCode::BAD_REQUEST,
            format!("{field} must be a string: {other}"),
        )),
        None => Err(fail(
            code,
            StatusCode::BAD_REQUEST,
            format!("missing field: {field}"),
        )),
    }
}

/// Reads an optional string field; an explicit `null`, a blank string and an
/// absent field all read as `None`.
pub(crate) fn optional_str(
    fields: &Map<String, Value>,
    field: &str,
    code: &str,
) -> ApiResult<Option<String>> {
    match fields.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(fail(
            code,
            StatusCode::BAD_REQUEST,
            format!("{field} must be a string: {other}"),
        )),
    }
}

/// Reads a required field of any JSON shape.
pub(crate) fn required_value(
    fields: &Map<String, Value>,
    field: &str,
    code: &str,
) -> ApiResult<Value> {
    fields.get(field).cloned().ok_or_else(|| {
        fail(
            code,
            StatusCode::BAD_REQUEST,
            format!("missing field: {field}"),
        )
    })
}

/// Rejects a body carrying any key outside `allowed`.
pub(crate) fn require_only(
    fields: &Map<String, Value>,
    allowed: &[&str],
    code: &str,
    message: &str,
) -> ApiResult<()> {
    if fields.len() == allowed.len() && fields.keys().all(|key| allowed.contains(&key.as_str())) {
        return Ok(());
    }
    Err(fail(code, StatusCode::BAD_REQUEST, message.to_owned()))
}

/// Reads a request header, trimmed; a blank value reads as absent.
pub(crate) fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// The deterministic fingerprint of a normalized request or response.
///
/// Keys are sorted at every level before hashing (FNV-1a, 64 bit), so a replay
/// matches regardless of the key order the client used on the wire.
pub(crate) fn fingerprint(value: &Value) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical_json(value).bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Serializes `value` with every object's keys sorted.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(fields) => {
            let mut sorted: Vec<(&String, &Value)> = fields.iter().collect();
            sorted.sort_by(|left, right| left.0.cmp(right.0));
            let rendered: Vec<String> = sorted
                .into_iter()
                .map(|(key, item)| {
                    format!("{}:{}", Value::String(key.clone()), canonical_json(item))
                })
                .collect();
            format!("{{{}}}", rendered.join(","))
        }
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", rendered.join(","))
        }
        other => other.to_string(),
    }
}

/// Publishes one delayed-feedback turn before it stops accepting captures.
///
/// Python retries while the turn is `waiting`/`publishing`; the port makes one
/// attempt and reports a conflict instead, because there is no scheduler here to
/// make another attempt progress.
pub(crate) async fn publish_turn(
    state: &RlServiceState,
    rl_task_id: &str,
    agent_turn_id: &str,
) -> ApiResult<()> {
    use super::capture::PublishState;
    let claim = state
        .registry()
        .claim_turn_publish(rl_task_id, agent_turn_id)
        .await
        .map_err(capture_failure)?;
    match claim {
        PublishState::Claimed => state
            .registry()
            .complete_turn_publish(rl_task_id, agent_turn_id)
            .await
            .map_err(capture_failure),
        PublishState::Published => Ok(()),
        PublishState::Waiting => Err(fail(
            "completion_conflict",
            StatusCode::CONFLICT,
            "RL Task still has open captures",
        )),
        PublishState::Publishing => Err(fail(
            "completion_conflict",
            StatusCode::CONFLICT,
            format!("turn is already being published: {agent_turn_id}"),
        )),
        PublishState::Failed => Err(fail(
            "completion_conflict",
            StatusCode::CONFLICT,
            format!("turn publication failed: {agent_turn_id}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{canonical_json, fingerprint};

    #[test]
    fn fingerprint_ignores_key_order() {
        let left = json!({"b": 1, "a": {"d": 2, "c": [3, {"f": 4, "e": 5}]}});
        let right = json!({"a": {"c": [3, {"e": 5, "f": 4}], "d": 2}, "b": 1});
        assert_eq!(fingerprint(&left), fingerprint(&right));
        assert_ne!(fingerprint(&left), fingerprint(&json!({"b": 1, "a": 2})));
    }

    #[test]
    fn canonical_json_sorts_nested_objects() {
        assert_eq!(
            canonical_json(&json!({"z": 1, "a": {"y": 2, "b": 3}})),
            r#"{"a":{"b":3,"y":2},"z":1}"#
        );
    }
}
