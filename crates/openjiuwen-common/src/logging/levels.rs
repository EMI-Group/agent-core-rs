//! Log levels, Python aliases, and fallbacks.
//!
//! Port of Python `openjiuwen/core/common/logging/log_levels.py`: the Python SDK
//! exposes both `logging` level numbers (`logging.WARNING`) and a
//! `normalize_log_level` helper that maps user-supplied strings to those levels.
//! The Rust SDK maps the same strings onto [`tracing`] levels, which are the only
//! levels the rest of the workspace uses.
pub use tracing::level_filters::LevelFilter;
pub use tracing::Level;

/// Level used when the SDK configures its own subscriber (`INFO`).
///
/// Python `LogConfig.level` default (`logging.INFO`).
pub const DEFAULT_CONFIG_LEVEL: LevelFilter = LevelFilter::INFO;

/// Level used when a level string cannot be interpreted (`WARN`).
///
/// Python `normalize_log_level` defaults unknown values to `WARNING`, which is
/// deliberately conservative: an unparseable `OPENJIUWEN_LOG_LEVEL` must not
/// silently enable verbose logging in production.
pub const DEFAULT_FALLBACK_LEVEL: LevelFilter = LevelFilter::WARN;

/// Normalizes a Python-style level name to a [`LevelFilter`].
///
/// Matching is case-insensitive and ignores surrounding whitespace. The Python
/// aliases are preserved: `CRITICAL` and `FATAL` both mean `ERROR`, `WARN` means
/// `WARNING`, and `NOTSET`/`OFF` disable logging. An empty or unrecognized value
/// falls back to [`DEFAULT_FALLBACK_LEVEL`].
///
/// ```
/// use openjiuwen_common::logging::{normalize_level, LevelFilter};
///
/// assert_eq!(normalize_level(" fatal "), LevelFilter::ERROR);
/// assert_eq!(normalize_level("nonsense"), LevelFilter::WARN);
/// assert_eq!(normalize_level("off"), LevelFilter::OFF);
/// ```
#[must_use]
pub fn normalize_level(value: &str) -> LevelFilter {
    match value.trim().to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARNING" | "WARN" => LevelFilter::WARN,
        "ERROR" | "CRITICAL" | "FATAL" => LevelFilter::ERROR,
        "NOTSET" | "OFF" => LevelFilter::OFF,
        _ => DEFAULT_FALLBACK_LEVEL,
    }
}

/// Parses a Python-style level name into a [`tracing::Level`].
///
/// Like [`normalize_level`] this accepts the Python aliases, but returns `None`
/// for `NOTSET`, `OFF`, empty values, and unknown names, because those do not
/// correspond to an emitting level.
///
/// ```
/// use openjiuwen_common::logging::{parse_level, Level};
///
/// assert_eq!(parse_level("Critical"), Some(Level::ERROR));
/// assert_eq!(parse_level("notset"), None);
/// ```
#[must_use]
pub fn parse_level(value: &str) -> Option<Level> {
    match value.trim().to_ascii_uppercase().as_str() {
        "TRACE" => Some(Level::TRACE),
        "DEBUG" => Some(Level::DEBUG),
        "INFO" => Some(Level::INFO),
        "WARNING" | "WARN" => Some(Level::WARN),
        "ERROR" | "CRITICAL" | "FATAL" => Some(Level::ERROR),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    // port of openjiuwen/core/common/logging/log_levels.py
    #[test]
    fn normalize_level_accepts_every_alias() {
        assert_eq!(normalize_level("TRACE"), LevelFilter::TRACE);
        assert_eq!(normalize_level("trace"), LevelFilter::TRACE);
        assert_eq!(normalize_level("DEBUG"), LevelFilter::DEBUG);
        assert_eq!(normalize_level("debug"), LevelFilter::DEBUG);
        assert_eq!(normalize_level("INFO"), LevelFilter::INFO);
        assert_eq!(normalize_level(" info "), LevelFilter::INFO);
        assert_eq!(normalize_level("WARNING"), LevelFilter::WARN);
        assert_eq!(normalize_level("warn"), LevelFilter::WARN);
        assert_eq!(normalize_level("Warn"), LevelFilter::WARN);
        assert_eq!(normalize_level("ERROR"), LevelFilter::ERROR);
        assert_eq!(normalize_level("critical"), LevelFilter::ERROR);
        assert_eq!(normalize_level("FATAL"), LevelFilter::ERROR);
        assert_eq!(normalize_level("notset"), LevelFilter::OFF);
        assert_eq!(normalize_level("OFF"), LevelFilter::OFF);
    }
    #[test]
    fn normalize_level_falls_back_for_unknown_values() {
        assert_eq!(normalize_level(""), DEFAULT_FALLBACK_LEVEL);
        assert_eq!(normalize_level("   "), DEFAULT_FALLBACK_LEVEL);
        assert_eq!(normalize_level("verbose"), DEFAULT_FALLBACK_LEVEL);
        assert_eq!(DEFAULT_FALLBACK_LEVEL, LevelFilter::WARN);
        assert_eq!(DEFAULT_CONFIG_LEVEL, LevelFilter::INFO);
    }
    #[test]
    fn parse_level_returns_none_for_non_emitting_levels() {
        assert_eq!(parse_level("TRACE"), Some(Level::TRACE));
        assert_eq!(parse_level("debug"), Some(Level::DEBUG));
        assert_eq!(parse_level("Info"), Some(Level::INFO));
        assert_eq!(parse_level("warning"), Some(Level::WARN));
        assert_eq!(parse_level("warn"), Some(Level::WARN));
        assert_eq!(parse_level("error"), Some(Level::ERROR));
        assert_eq!(parse_level("CRITICAL"), Some(Level::ERROR));
        assert_eq!(parse_level("fatal"), Some(Level::ERROR));
        assert_eq!(parse_level("notset"), None);
        assert_eq!(parse_level("off"), None);
        assert_eq!(parse_level(""), None);
        assert_eq!(parse_level("unknown"), None);
    }
}
