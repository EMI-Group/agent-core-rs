//! Tolerant RFC 3339 parsing and UTC formatting for observed timestamps.
//!
//! The observation contract carries `observed_at` as RFC 3339 *text* (Python stores a
//! tz-aware `datetime`), so the crate needs a parser and a formatter that agree with each
//! other and never panic. Both are hand rolled on integer arithmetic: the workspace has no
//! `chrono` dependency, and this module must stay dependency free.
//!
//! Accepted shapes: `YYYY-MM-DD[Tt ]HH:MM:SS` with an optional fraction and a required
//! offset — `Z`, `z`, or `±HH:MM` / `±HHMM`. The
//! fractional digits are carried verbatim (not converted to nanoseconds), so a caller can
//! round-trip `...T00:00:00.123456Z` without losing the written precision. Calendar days
//! are validated against the proleptic Gregorian month length, including leap years.
//!
//! Everything here is `pub(crate)`: the public surface is
//! `GraphEvolutionInput::normalized`, which rewrites `observed_at` to the canonical `...Z`
//! form and reports an `Error::Validation` naming the field when the text is unusable.

use crate::error::{validation, Result};

/// Parses an RFC 3339 timestamp into epoch seconds plus the fractional digits as written.
///
/// Returns [`None`] for text that is not a well-formed timestamp **or** that carries no
/// timezone at all (two naive timestamps cannot be ordered, so the parser refuses them
/// instead of guessing UTC). A complete instant is at least 20 bytes long
/// (`2024-03-01T00:30:00Z`).
#[must_use]
pub(crate) fn parse_epoch(raw: &str) -> Option<(i64, Option<String>)> {
    match scan(raw) {
        Scan::Parsed {
            epoch_seconds,
            fraction,
        } => Some((epoch_seconds, fraction)),
        Scan::MissingTimezone | Scan::Malformed => None,
    }
}

/// Formats epoch seconds (plus optional fractional digits) as `YYYY-MM-DDTHH:MM:SS[.frac]Z`.
///
/// No fraction is printed when `fraction` is [`None`]; otherwise `.` plus the digits
/// exactly as supplied. The output is always UTC.
#[must_use]
pub(crate) fn format_epoch(epoch_seconds: i64, fraction: Option<&str>) -> String {
    let days = epoch_seconds.div_euclid(86_400);
    let second_of_day = epoch_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = second_of_day % 3_600 / 60;
    let second = second_of_day % 60;
    let mut rendered = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
    if let Some(digits) = fraction {
        rendered.push('.');
        rendered.push_str(digits);
    }
    rendered.push('Z');
    rendered
}

/// Normalises an RFC 3339 timestamp to UTC (`...Z`) text.
///
/// # Errors
///
/// Returns `Error::Validation` naming `observed_at`
/// when the text has no timezone (`must include a timezone`) or is not a well-formed
/// RFC 3339 timestamp (the message echoes the offending text).
pub(crate) fn normalize_observed_at(raw: &str) -> Result<String> {
    if let Some((epoch_seconds, fraction)) = parse_epoch(raw) {
        return Ok(format_epoch(epoch_seconds, fraction.as_deref()));
    }
    Err(refusal(raw))
}

/// Builds the validation error for text [`parse_epoch`] refused.
///
/// The happy path parses once; this diagnosis re-scans only to tell a missing timezone
/// apart from malformed text, which is what makes the message actionable.
fn refusal(raw: &str) -> crate::error::Error {
    match scan(raw) {
        Scan::MissingTimezone => validation("observed_at", "must include a timezone"),
        Scan::Parsed { .. } | Scan::Malformed => validation(
            "observed_at",
            format!("must be an RFC 3339 timestamp with a timezone, got `{raw}`"),
        ),
    }
}

/// The outcome of scanning one timestamp, including *why* it was refused.
enum Scan {
    /// A fully parsed instant: seconds since the epoch (UTC) plus the written fraction.
    Parsed {
        /// Seconds since the Unix epoch, offset already applied.
        epoch_seconds: i64,
        /// The fractional digits exactly as written, when the text carried any.
        fraction: Option<String>,
    },
    /// The date and time parsed, but no `Z`/`z`/`±HH:MM` offset followed.
    MissingTimezone,
    /// The text is not an RFC 3339 timestamp.
    Malformed,
}

/// The calendar fields of a timestamp, before any offset is applied.
struct WallClock {
    /// Proleptic Gregorian year as written (four digits).
    year: i64,
    /// Month, `1..=12`.
    month: i64,
    /// Day of month, `1..=days_in_month(year, month)`.
    day: i64,
    /// Hour, `0..=23`.
    hour: i64,
    /// Minute, `0..=59`.
    minute: i64,
    /// Second, `0..=59` (no leap-second form: an RFC 3339 `:60` is rejected).
    second: i64,
}

/// Scans `raw`, keeping the reason a timestamp is unusable.
fn scan(raw: &str) -> Scan {
    let text = raw.trim();
    let bytes = text.as_bytes();
    // `YYYY-MM-DDTHH:MM:SS` is 19 bytes; a complete instant also needs an offset.
    if bytes.len() < 19 {
        return Scan::Malformed;
    }
    let Some(clock) = parse_wall_clock(bytes) else {
        return Scan::Malformed;
    };
    let mut index = 19;
    let mut fraction = None;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while index < bytes.len() && bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == start {
            return Scan::Malformed;
        }
        let Some(digits) = text.get(start..index) else {
            return Scan::Malformed;
        };
        fraction = Some(digits.to_string());
    }
    let Some(separator) = bytes.get(index) else {
        return Scan::MissingTimezone;
    };
    let offset_seconds = match *separator {
        b'Z' | b'z' => {
            index += 1;
            0
        }
        sign @ (b'+' | b'-') => {
            index += 1;
            let Some((offset_hour, offset_minute, next)) = parse_offset(bytes, index) else {
                return Scan::Malformed;
            };
            if offset_hour > 23 || offset_minute > 59 {
                return Scan::Malformed;
            }
            index = next;
            let magnitude = offset_hour * 3_600 + offset_minute * 60;
            if sign == b'-' {
                -magnitude
            } else {
                magnitude
            }
        }
        _ => return Scan::Malformed,
    };
    if index != bytes.len() {
        return Scan::Malformed;
    }
    if !(1..=12).contains(&clock.month)
        || clock.day < 1
        || clock.day > days_in_month(clock.year, clock.month)
        || !(0..=23).contains(&clock.hour)
        || !(0..=59).contains(&clock.minute)
        || !(0..=59).contains(&clock.second)
    {
        return Scan::Malformed;
    }
    let days = days_from_civil(clock.year, clock.month, clock.day);
    let epoch_seconds =
        days * 86_400 + clock.hour * 3_600 + clock.minute * 60 + clock.second - offset_seconds;
    Scan::Parsed {
        epoch_seconds,
        fraction,
    }
}

/// Parses `YYYY-MM-DD[Tt ]HH:MM:SS` from the first 19 bytes of `bytes`.
fn parse_wall_clock(bytes: &[u8]) -> Option<WallClock> {
    let year = parse_digits(bytes, 0, 4)?;
    if bytes.get(4) != Some(&b'-') {
        return None;
    }
    let month = parse_digits(bytes, 5, 2)?;
    if bytes.get(7) != Some(&b'-') {
        return None;
    }
    let day = parse_digits(bytes, 8, 2)?;
    if !matches!(bytes.get(10), Some(b'T' | b't' | b' ')) {
        return None;
    }
    let hour = parse_digits(bytes, 11, 2)?;
    if bytes.get(13) != Some(&b':') {
        return None;
    }
    let minute = parse_digits(bytes, 14, 2)?;
    if bytes.get(16) != Some(&b':') {
        return None;
    }
    let second = parse_digits(bytes, 17, 2)?;
    Some(WallClock {
        year,
        month,
        day,
        hour,
        minute,
        second,
    })
}

/// Parses `HH:MM` or `HHMM` starting at `start`, returning the fields and the next index.
fn parse_offset(bytes: &[u8], start: usize) -> Option<(i64, i64, usize)> {
    let hour = parse_digits(bytes, start, 2)?;
    let mut index = start + 2;
    if bytes.get(index) == Some(&b':') {
        index += 1;
    }
    let minute = parse_digits(bytes, index, 2)?;
    Some((hour, minute, index + 2))
}

/// Parses exactly `count` ASCII digits starting at `start`.
fn parse_digits(bytes: &[u8], start: usize, count: usize) -> Option<i64> {
    if start + count > bytes.len() {
        return None;
    }
    let mut value: i64 = 0;
    for offset in start..start + count {
        let byte = *bytes.get(offset)?;
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + i64::from(byte - b'0');
    }
    Some(value)
}

/// Number of days in `month` of `year`, honoring the Gregorian leap-year rule.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Reports whether `year` is a leap year in the proleptic Gregorian calendar.
fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The proleptic Gregorian date for a count of days since 1970-01-01 (Hinnant, inverse).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_leap_day() {
        let (seconds, fraction) = parse_epoch("2024-02-29T00:00:00Z").expect("leap day");
        assert_eq!(
            format_epoch(seconds, fraction.as_deref()),
            "2024-02-29T00:00:00Z"
        );
        assert_eq!(seconds, 1_709_164_800);
    }

    #[test]
    fn rejects_a_non_leap_february_29() {
        assert_eq!(parse_epoch("2023-02-29T00:00:00Z"), None);
        assert!(normalize_observed_at("2023-02-29T00:00:00Z").is_err());
        assert!(parse_epoch("2023-02-28T00:00:00Z").is_some());
    }

    #[test]
    fn applies_a_positive_offset_backwards_across_midnight() {
        assert_eq!(
            normalize_observed_at("2024-03-01T00:30:00+08:00").unwrap(),
            "2024-02-29T16:30:00Z"
        );
    }

    #[test]
    fn accepts_an_offset_without_a_colon() {
        assert_eq!(
            normalize_observed_at("2024-03-01T08:30:00+0830").unwrap(),
            "2024-03-01T00:00:00Z"
        );
        assert_eq!(
            normalize_observed_at("2024-03-01T08:30:00-0830").unwrap(),
            "2024-03-01T17:00:00Z"
        );
    }

    #[test]
    fn accepts_a_lowercase_z_and_a_space_separator() {
        assert_eq!(
            normalize_observed_at("2024-03-01t08:30:00z").unwrap(),
            "2024-03-01T08:30:00Z"
        );
        assert_eq!(
            normalize_observed_at("2024-03-01 08:30:00Z").unwrap(),
            "2024-03-01T08:30:00Z"
        );
    }

    #[test]
    fn keeps_six_fractional_digits_verbatim() {
        let (seconds, fraction) =
            parse_epoch("2024-03-01T08:30:00.123456+08:00").expect("fraction");
        assert_eq!(fraction.as_deref(), Some("123456"));
        assert_eq!(
            format_epoch(seconds, fraction.as_deref()),
            "2024-03-01T00:30:00.123456Z"
        );
        assert_eq!(
            normalize_observed_at("2024-03-01T08:30:00.123456+08:00").unwrap(),
            "2024-03-01T00:30:00.123456Z"
        );
    }

    #[test]
    fn formats_without_a_fraction_by_default() {
        assert_eq!(format_epoch(0, None), "1970-01-01T00:00:00Z");
        assert_eq!(format_epoch(0, Some("5")), "1970-01-01T00:00:00.5Z");
    }

    #[test]
    fn rejects_a_missing_timezone() {
        assert_eq!(parse_epoch("2024-03-01T00:30:00"), None);
        let error = normalize_observed_at("2024-03-01T00:30:00").unwrap_err();
        assert!(error.message().contains("observed_at"));
        assert!(error.message().contains("must include a timezone"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert_eq!(parse_epoch("2024-03-01T00:30:00Zx"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:00Zjunk"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:00+08:00x"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:00."), None);
        assert!(normalize_observed_at("2024-03-01T00:30:00Zx").is_err());
    }

    #[test]
    fn rejects_out_of_range_fields() {
        assert_eq!(parse_epoch("2024-13-01T00:30:00Z"), None);
        assert_eq!(parse_epoch("2024-00-01T00:30:00Z"), None);
        assert_eq!(parse_epoch("2024-04-31T00:30:00Z"), None);
        assert_eq!(parse_epoch("2024-03-01T24:30:00Z"), None);
        assert_eq!(parse_epoch("2024-03-01T00:60:00Z"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:60Z"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:00+24:00"), None);
        assert_eq!(parse_epoch("2024-03-01T00:30:0Z"), None);
        assert_eq!(parse_epoch("not-a-timestamp"), None);
    }

    #[test]
    fn handles_a_negative_epoch() {
        let (seconds, fraction) = parse_epoch("1969-12-31T23:59:59Z").expect("negative epoch");
        assert_eq!(seconds, -1);
        assert_eq!(fraction, None);
        assert_eq!(format_epoch(seconds, None), "1969-12-31T23:59:59Z");
        let (seconds, _) = parse_epoch("1960-02-29T12:00:00Z").expect("leap day pre-epoch");
        assert_eq!(format_epoch(seconds, None), "1960-02-29T12:00:00Z");
    }
}
