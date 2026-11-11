//! Logging facade built on [`tracing`].
//!
//! Port of the Python package `openjiuwen/core/common/logging`, with one
//! deliberate architectural deviation: Python's `LogManager`, `LoggerProtocol`,
//! `LazyLogger`, and `LogConfig` singleton machinery is **not** reproduced.
//! Rust's `tracing` already provides a global, structured, filterable pipeline, so
//! the SDK keeps only the user-visible concepts Python exposed and maps them onto
//! it:
//!
//! | Python concept | Rust replacement |
//! | --- | --- |
//! | logger names (`openjiuwen.agent`, ...) | `tracing` targets in [`targets`] |
//! | level names and aliases | [`normalize_level`] / [`parse_level`] |
//! | `ContextVar` trace ids | task-local [`LogScope`] plus scope spans |
//! | `BaseLogEvent` records | [`LogEvent`] and [`emit`] |
//! | `LogConfig` + `LogManager.setup` | [`LoggingConfig`] and [`init`] |
//!
//! Initialization is idempotent, so a library may call [`init`] defensively without
//! disturbing a host application that already installed its own subscriber.
//! Library code never writes to `stdout`/`stderr` directly; it emits through the
//! macros re-exported here.
//!
//! ```
//! use openjiuwen_common::logging::{info, targets};
//!
//! info!(target: targets::SESSION, session_id = "s-1", "session resumed");
//! ```
pub mod events;
pub mod levels;
pub mod scope;
pub mod subscriber;
pub mod targets;

pub use events::{emit, event_types, EventStatus, EventType, LogEvent, LogLevel, ModuleType};
pub use levels::{
    normalize_level, parse_level, Level, LevelFilter, DEFAULT_CONFIG_LEVEL, DEFAULT_FALLBACK_LEVEL,
};
pub use scope::{
    current, current_member_id, current_trace_id, with_scope, LogScope, DEFAULT_TRACE_ID,
};
pub use subscriber::{init, init_with, is_initialized, LogFormat, LogWriter, LoggingConfig};
pub use tracing::{debug, error, info, instrument, trace, warn};
