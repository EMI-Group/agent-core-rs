//! Pure rendering helpers for compression-state records.
//!
//! Timestamps, compact numbers, saved deltas and the one-line summary live here so the recorder itself stays about building, keeping and delivering records.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::telemetry::state::{
    CompressionStateStatus, ContextCompressionMetric, ContextCompressionSaved,
};

/// Renders `time` as an ISO-8601 UTC timestamp with millisecond precision.
///
/// Python formats the observation time in the local timezone; the Rust port
/// always writes UTC with a `Z` suffix so records compare as strings.
#[must_use]
pub fn format_timestamp(time: SystemTime) -> String {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs();
    let millis = elapsed.subsec_millis();
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        seconds_of_day / 3_600,
        seconds_of_day % 3_600 / 60,
        seconds_of_day % 60
    )
}

/// Converts days since the unix epoch into a proleptic Gregorian date.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

/// The wall-clock duration between two observations, in milliseconds.
pub(super) fn duration_ms(
    started_at: Option<SystemTime>,
    ended_at: Option<SystemTime>,
) -> Option<u64> {
    let elapsed = ended_at?.duration_since(started_at?).ok()?;
    Some(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
}

/// `tokens / context_max` as a rounded percentage, clamped to 0–100.
pub(super) fn context_percent(tokens: usize, context_max: Option<u32>) -> Option<u32> {
    let max = context_max.filter(|max| *max > 0)?;
    let max = u64::from(max);
    let tokens = u64::try_from(tokens).unwrap_or(u64::MAX);
    let percent = (tokens.saturating_mul(100).saturating_add(max / 2)) / max;
    Some(u32::try_from(percent.min(100)).unwrap_or(100))
}

/// `before - after` as a signed delta; `None` until the pass produced a history.
///
/// The saving inherits estimation from both sides, so a delta of two estimates is
/// never published as an exact one.
pub(super) fn saved_delta(
    before: &ContextCompressionMetric,
    after: Option<&ContextCompressionMetric>,
) -> Option<ContextCompressionSaved> {
    let after = after?;
    let before_messages = i64::try_from(before.messages).unwrap_or(i64::MAX);
    let after_messages = i64::try_from(after.messages).unwrap_or(i64::MAX);
    let before_tokens = i64::try_from(before.tokens).unwrap_or(i64::MAX);
    let after_tokens = i64::try_from(after.tokens).unwrap_or(i64::MAX);
    let saved_tokens = before_tokens.saturating_sub(after_tokens);
    let percent = if before.tokens == 0 {
        0.0
    } else {
        round_tenths(saved_tokens as f64 / before_tokens as f64 * 100.0)
    };
    Some(ContextCompressionSaved {
        messages: before_messages.saturating_sub(after_messages),
        tokens: saved_tokens,
        percent,
        estimated: before.estimated || after.estimated,
    })
}

/// Rounds to one decimal, Python's `round(value, 1)`.
fn round_tenths(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// Renders a count the way Python's `_compact_number` does (`1.5m`, `12.3k`).
pub(super) fn compact_number(value: usize) -> String {
    compact_signed(i64::try_from(value).unwrap_or(i64::MAX))
}

/// [`compact_number`] for a signed count, so a negative saving keeps its sign.
fn compact_signed(value: i64) -> String {
    let with_suffix = |divisor: f64, suffix: &str| {
        let tenths = format!("{:.1}", value as f64 / divisor);
        match tenths.strip_suffix(".0") {
            Some(whole) => format!("{whole}{suffix}"),
            None => format!("{tenths}{suffix}"),
        }
    };
    let absolute = value.unsigned_abs();
    if absolute >= 1_000_000 {
        with_suffix(1_000_000.0, "m")
    } else if absolute >= 1_000 {
        with_suffix(1_000.0, "k")
    } else {
        value.to_string()
    }
}

/// The one-line summary of a record, matching Python's `_build_summary`.
pub(super) fn build_summary(
    status: CompressionStateStatus,
    before: &ContextCompressionMetric,
    after: Option<&ContextCompressionMetric>,
    saved: Option<&ContextCompressionSaved>,
    reason: &str,
    modified: usize,
) -> String {
    match status {
        CompressionStateStatus::Started => format!(
            "Compressing {} messages, ~{} tokens",
            before.messages,
            compact_number(before.tokens)
        ),
        CompressionStateStatus::Paused => format!(
            "Context compression paused; context remains ~{} tokens. \
             It will be re-evaluated when the task resumes.",
            compact_number(before.tokens)
        ),
        CompressionStateStatus::Cancelled => format!(
            "Context compression cancelled; context remains ~{} tokens. \
             It will be re-evaluated on the next request.",
            compact_number(before.tokens)
        ),
        CompressionStateStatus::Failed => format!(
            "Context processor failed; context remains ~{} tokens",
            compact_number(before.tokens)
        ),
        CompressionStateStatus::Completed
        | CompressionStateStatus::Noop
        | CompressionStateStatus::Skipped => {
            let (Some(after), Some(saved)) = (after, saved) else {
                return format!("Context processor skipped: {reason}");
            };
            if status == CompressionStateStatus::Noop {
                return format!(
                    "Context unchanged at ~{} tokens (saved {:.1}%)",
                    compact_number(after.tokens),
                    saved.percent
                );
            }
            let modified_text = if modified > 0 {
                format!(", modified {modified} messages")
            } else {
                String::new()
            };
            format!(
                "Compressed {} -> {} messages, ~{} -> ~{} tokens, saved ~{} tokens ({:.1}%){}",
                before.messages,
                after.messages,
                compact_number(before.tokens),
                compact_number(after.tokens),
                compact_signed(saved.tokens),
                saved.percent,
                modified_text
            )
        }
    }
}
