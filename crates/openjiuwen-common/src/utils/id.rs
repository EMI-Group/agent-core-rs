//! Identifiers and timestamps ported from Python `openjiuwen.core.common.utils.id_utils`.

use chrono::{DateTime, Local, SecondsFormat, Utc};
use uuid::Uuid;

/// Generates a random UUID v4 in its hyphenated canonical form.
#[must_use]
pub fn uuid_v4() -> String {
    Uuid::new_v4().to_string()
}

/// Generates a random UUID v4 as 32 lowercase hex characters without hyphens.
///
/// This matches Python `uuid.uuid4().hex`, the form used for card and event ids.
#[must_use]
pub fn uuid_hex() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Returns the current time in UTC.
#[must_use]
pub fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

/// Returns the current time in the system's local timezone.
#[must_use]
pub fn now_local() -> DateTime<Local> {
    Local::now()
}

/// Formats a UTC timestamp as RFC 3339 with millisecond precision and a `Z` suffix.
///
/// ```
/// use chrono::{DateTime, Utc};
/// use openjiuwen_common::utils::id::format_iso8601;
///
/// let dt: DateTime<Utc> = DateTime::parse_from_rfc3339("2024-01-02T03:04:05.678Z")
///     .unwrap()
///     .with_timezone(&Utc);
/// assert_eq!(format_iso8601(&dt), "2024-01-02T03:04:05.678Z");
/// ```
#[must_use]
pub fn format_iso8601(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Returns the current UTC time formatted as RFC 3339 with millisecond precision.
#[must_use]
pub fn now_rfc3339() -> String {
    format_iso8601(&now_utc())
}

/// Returns the current time as milliseconds since the Unix epoch.
#[must_use]
pub fn unix_millis() -> i64 {
    now_utc().timestamp_millis()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn uuid_hex_is_32_lowercase_hex_chars() {
        let id = uuid_hex();
        assert_eq!(id.len(), 32);
        assert!(id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn uuid_hex_is_unique_across_calls() {
        let ids: HashSet<String> = (0..100).map(|_| uuid_hex()).collect();
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn uuid_v4_parses_as_a_uuid() {
        let id = uuid_v4();
        let parsed = Uuid::parse_str(&id).expect("uuid_v4 must produce a parseable uuid");
        assert_eq!(parsed.get_version_num(), 4);
        assert_eq!(parsed.to_string(), id);
    }

    #[test]
    fn format_iso8601_uses_millis_and_a_z_suffix() {
        let dt = DateTime::parse_from_rfc3339("2024-01-02T03:04:05.6Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(format_iso8601(&dt), "2024-01-02T03:04:05.600Z");
    }

    #[test]
    fn now_rfc3339_round_trips() {
        let formatted = now_rfc3339();
        assert!(
            formatted.ends_with('Z'),
            "unexpected timestamp: {formatted}"
        );
        assert!(DateTime::parse_from_rfc3339(&formatted).is_ok());
    }

    #[test]
    fn timestamps_are_sensible() {
        // 2024-01-01T00:00:00Z, far enough in the past for any plausible run.
        let lower_bound = 1_704_067_200_000_i64;
        let millis = unix_millis();
        assert!(millis > lower_bound, "unix_millis out of range: {millis}");
        assert!(now_utc().timestamp_millis() >= millis);
        assert!(now_local().timestamp_millis() >= lower_bound);
    }
}
