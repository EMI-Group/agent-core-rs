//! Pure helpers shared by the fake server's modules.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use std::time::Duration;
use tokio::time::Instant;

use super::Data;
use crate::redis::codec::Reply;

/// Upper-cases a byte argument for command/option matching.
pub(super) fn upper(value: &[u8]) -> String {
    String::from_utf8_lossy(value).to_ascii_uppercase()
}
/// Parses a byte argument as an `i64`.
pub(super) fn parse_i64(value: &[u8]) -> Option<i64> {
    std::str::from_utf8(value).ok()?.parse().ok()
}
/// Parses a byte argument as a finite `f64`.
pub(super) fn parse_f64(value: &[u8]) -> Option<f64> {
    let parsed = std::str::from_utf8(value).ok()?.parse::<f64>().ok()?;
    parsed.is_finite().then_some(parsed)
}
/// Converts a count to `i64` without panicking.
pub(super) fn count_to_i64(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}
/// The `-ERR wrong number of arguments` reply.
pub(super) fn wrong_arity(name: &str) -> Reply {
    Reply::Error(format!(
        "ERR wrong number of arguments for '{name}' command"
    ))
}
/// The `-WRONGTYPE` reply.
pub(super) fn wrong_type() -> Reply {
    Reply::Error("WRONGTYPE Operation against a key holding the wrong kind of value".to_owned())
}
/// The `-ERR syntax error` reply.
pub(super) fn syntax_error() -> Reply {
    Reply::Error("ERR syntax error".to_owned())
}
/// The `-ERR Protocol error` reply.
pub(super) fn protocol_error(message: &str) -> Reply {
    Reply::Error(format!("ERR Protocol error: {message}"))
}
/// Computes a relative expiry instant.
pub(super) fn relative_expiry(duration: Duration) -> Instant {
    Instant::now() + duration
}
/// Computes an absolute expiry from a Unix timestamp in seconds.
pub(super) fn absolute_expiry_seconds(seconds: i64) -> Instant {
    absolute_expiry(seconds as f64 * 1_000.0)
}
/// Computes an absolute expiry from a Unix timestamp in milliseconds.
pub(super) fn absolute_expiry_millis(millis: i64) -> Instant {
    absolute_expiry(millis as f64)
}
/// Computes an absolute expiry from a Unix timestamp in milliseconds.
pub(super) fn absolute_expiry(target_millis: f64) -> Instant {
    let now_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1_000.0);
    let remaining = target_millis - now_millis;
    if remaining <= 0.0 {
        Instant::now()
    } else {
        Instant::now() + Duration::from_secs_f64(remaining / 1_000.0)
    }
}
/// Formats a sorted-set score the way `Redis` does (`1`, `1.5`, ...).
pub(super) fn format_score(score: f64) -> Vec<u8> {
    if score.fract().abs() < f64::EPSILON && score.abs() < 1e17 {
        format!("{}", score as i64).into_bytes()
    } else {
        format!("{score}").into_bytes()
    }
}
/// The `-2`/`-1`/remaining TTL reply.
pub(super) fn ttl_reply(data: &Data, key: &[u8], milliseconds: bool) -> Reply {
    if !data.exists(key) {
        return Reply::Integer(-2);
    }
    let Some(at) = data.expiries.get(key) else {
        return Reply::Integer(-1);
    };
    let remaining = at.saturating_duration_since(Instant::now());
    let value = if milliseconds {
        remaining.as_millis()
    } else {
        u128::from(remaining.as_secs())
    };
    Reply::Integer(i64::try_from(value).unwrap_or(i64::MAX))
}
/// The live keys matching an optional glob pattern.
pub(super) fn matching_keys(data: &Data, pattern: Option<&str>) -> Vec<Vec<u8>> {
    data.keys()
        .into_iter()
        .filter(|key| pattern.is_none_or(|glob| glob_match(glob.as_bytes(), key)))
        .collect()
}
/// Minimal glob matcher supporting `*` and `?`.
pub(super) fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(b'*') => {
            let rest = &pattern[1..];
            let mut index = 0;
            loop {
                if glob_match(rest, &text[index..]) {
                    return true;
                }
                if index == text.len() {
                    return false;
                }
                index += 1;
            }
        }
        Some(b'?') => !text.is_empty() && glob_match(&pattern[1..], &text[1..]),
        Some(&expected) => text.first() == Some(&expected) && glob_match(&pattern[1..], &text[1..]),
    }
}
