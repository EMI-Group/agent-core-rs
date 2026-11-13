//! The serde configuration of the subprocess backend: what to launch, how long
//! to wait, and how many bytes one frame may be.
//!
//! [`SubprocessConfig`] holds the launch line (`program`, `args`, `cwd`, `env`),
//! the request and shutdown deadlines, the background health-check policy, and
//! the per-frame cap. It serializes (`snake_case`, unknown fields ignored, every
//! field defaulted) so a host can keep it beside the rest of its runner
//! configuration.
//!
//! Defaults: a 30 s request deadline, a 10 s graceful shutdown, a 5 s health
//! interval with a 5 s probe deadline, three consecutive failed probes before
//! the child is killed, and the 8 MiB cap of
//! [`crate::spawn::frames::DEFAULT_MAX_FRAME_BYTES`]. A `0.0` health interval
//! disables the background health task (explicit probes still work), and a `0`
//! failure count never kills on health.
//!
//! This is the private sibling of the `subprocess` module: the supervisor there
//! owns the child, this module owns its settings. All configured timeouts are
//! clamped to a representable range, so nonsense configuration means
//! "effectively no deadline", never a panic.

use crate::spawn::frames::DEFAULT_MAX_FRAME_BYTES;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// Default deadline for one request/reply exchange, in seconds.
pub(super) const DEFAULT_REQUEST_TIMEOUT_SECS: f64 = 30.0;
/// Default deadline for a graceful shutdown before a kill, in seconds.
pub(super) const DEFAULT_SHUTDOWN_TIMEOUT_SECS: f64 = 10.0;
/// Default interval between background health probes, in seconds.
pub(super) const DEFAULT_HEALTH_CHECK_INTERVAL_SECS: f64 = 5.0;
/// Default deadline for one health probe, in seconds.
pub(super) const DEFAULT_HEALTH_CHECK_TIMEOUT_SECS: f64 = 5.0;
/// Consecutive failed health probes before the child is killed.
pub(super) const DEFAULT_MAX_HEALTH_FAILURES: u32 = 3;
/// Upper bound on any configured timeout: one day.
///
/// Clamping keeps `Instant + Duration` (which panics on overflow) infallible and
/// makes nonsense configuration mean "effectively no deadline", not a panic.
pub(super) const MAX_TIMEOUT_SECS: f64 = 86_400.0;

/// How a [`SubprocessSpawner`](crate::spawn::SubprocessSpawner) launches and
/// supervises its children.
///
/// Serializes (`snake_case`, unknown fields ignored, every field defaulted) so a
/// host can keep it beside the rest of its runner configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubprocessConfig {
    /// Program to launch (an interpreter or an agent binary); empty is an error.
    pub program: String,
    /// Arguments passed to [`SubprocessConfig::program`].
    pub args: Vec<String>,
    /// Working directory; the parent's is inherited when `None`.
    pub cwd: Option<PathBuf>,
    /// Environment overrides added on top of the parent's environment.
    pub env: BTreeMap<String, String>,
    /// Deadline for one request/reply exchange, in seconds.
    pub request_timeout_secs: f64,
    /// Deadline for a graceful shutdown before a kill, in seconds.
    pub shutdown_timeout_secs: f64,
    /// Interval between background health probes, in seconds.
    ///
    /// `0.0` disables the background task; explicit
    /// [`SpawnedProcess::health`](crate::spawn::SpawnedProcess::health) calls
    /// still work.
    pub health_check_interval_secs: f64,
    /// Deadline for one health probe, in seconds.
    pub health_check_timeout_secs: f64,
    /// Consecutive failed health probes before the child is killed.
    ///
    /// `0` never kills on health, whatever the failure count.
    pub max_health_failures: u32,
    /// Cap on one frame line read from the child, in bytes.
    pub max_frame_bytes: usize,
}

impl Default for SubprocessConfig {
    fn default() -> Self {
        Self {
            program: String::new(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            shutdown_timeout_secs: DEFAULT_SHUTDOWN_TIMEOUT_SECS,
            health_check_interval_secs: DEFAULT_HEALTH_CHECK_INTERVAL_SECS,
            health_check_timeout_secs: DEFAULT_HEALTH_CHECK_TIMEOUT_SECS,
            max_health_failures: DEFAULT_MAX_HEALTH_FAILURES,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }
}

impl SubprocessConfig {
    /// A configuration for `program`.
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Appends one argument.
    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Replaces the argument list.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the child's working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Adds one environment override (below a
    /// [`SpawnSpec`](crate::spawn::SpawnSpec)'s own overrides).
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the deadline for one request/reply exchange.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the deadline for a graceful shutdown before a kill.
    #[must_use]
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the background health-check policy.
    ///
    /// A zero `interval` disables the background task; a zero `max_failures`
    /// disables killing on health.
    #[must_use]
    pub fn with_health_check(
        mut self,
        interval: Duration,
        timeout: Duration,
        max_failures: u32,
    ) -> Self {
        self.health_check_interval_secs = interval.as_secs_f64();
        self.health_check_timeout_secs = timeout.as_secs_f64();
        self.max_health_failures = max_failures;
        self
    }

    /// Sets the cap on one frame line read from the child.
    #[must_use]
    pub fn with_max_frame_bytes(mut self, max_frame_bytes: usize) -> Self {
        self.max_frame_bytes = max_frame_bytes;
        self
    }

    /// The configured request deadline, clamped to a representable range.
    pub(super) fn request_timeout(&self) -> Duration {
        secs(self.request_timeout_secs)
    }

    /// The configured health-probe deadline, clamped to a representable range.
    pub(super) fn health_timeout(&self) -> Duration {
        secs(self.health_check_timeout_secs)
    }

    /// The configured background health interval, clamped to a representable
    /// range.
    pub(super) fn health_interval(&self) -> Duration {
        secs(self.health_check_interval_secs)
    }
}

/// A configured number of seconds as a [`Duration`], clamped to
/// `[0, MAX_TIMEOUT_SECS]`.
///
/// Negative and NaN values become zero (`f64::max` ignores NaN) and huge values
/// become one day, so `Instant + Duration` cannot overflow.
pub(super) fn secs(value: f64) -> Duration {
    // `clamp` panics on a NaN bound but returns NaN for a NaN input, and
    // `Duration::from_secs_f64` panics on NaN, so handle it first.
    let clamped = if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, MAX_TIMEOUT_SECS)
    };
    Duration::from_secs_f64(clamped)
}
