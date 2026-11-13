//! Five-field cron expression parsing and next-occurrence computation.
//!
//! The supported grammar is the classic 5-field form
//! `minute hour day-of-month month day-of-week`:
//!
//! - `*` — every value of the field,
//! - `*/n` — every `n`-th value starting at the field minimum,
//! - `a` — a single value,
//! - `a-b` — an inclusive range,
//! - `a-b/n` (and `a/n`, which Vixie cron reads as `a-max/n`) — a stepped range,
//! - `1,15,30` — a comma-separated list of any of the above.
//!
//! Day-of-week accepts the three-letter names `SUN`..`SAT` (case-insensitive) besides the
//! numbers `0..=7`, where both `0` and `7` mean Sunday. The shorthands `@hourly`, `@daily`,
//! `@midnight`, `@weekly`, `@monthly`, `@yearly` and `@annually` expand to that grammar.
//!
//! # Day-of-month / day-of-week parity choice
//!
//! Python's cron backend runs a 7-field Quartz engine, where the two day fields must not both
//! be specified and the unspecified one is written `?`. Standard (Vixie) cron instead treats
//! them with **OR semantics**: when *both* fields are restricted, a day matches if either the
//! day-of-month or the day-of-week matches; when one is unrestricted (`*`), only the other
//! decides. This module ports the standard semantics because the public expression grammar
//! here is the 5-field one; `?` and the seconds/year fields are not accepted.
//!
//! A field counts as unrestricted when its normalized value set covers the whole range
//! (`*`, `*/1`, or `0-7` for day-of-week).
//!
//! # Bounded search
//!
//! [`CronSchedule::next_after`] scans at most [`SEARCH_HORIZON_DAYS`] days ahead and at most
//! [`MAX_SEARCH_STEPS`] loop iterations, so a schedule that can never match (for example
//! `0 0 31 2 *`) returns `None` instead of looping forever. The horizon is a documented
//! limitation: a leap-day schedule never needs more than four years, but the eight-year gap
//! around a skipped century leap day (for example 2096 → 2104) exceeds it.

use std::collections::BTreeSet;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Timelike, Utc};

use openjiuwen_common::error::{Error, Result};

/// Number of fields a cron expression must have.
const FIELD_COUNT: usize = 5;

/// How far ahead [`CronSchedule::next_after`] searches, in days (five years plus slack).
const SEARCH_HORIZON_DAYS: i64 = 5 * 366 + 1;

/// Safety cap on the scan iterations of [`CronSchedule::next_after`].
///
/// Every rejected candidate jumps to the next minute, hour, day or month, so the cap is only
/// a guard against a pathological loop; the five-year horizon is the semantic bound.
const MAX_SEARCH_STEPS: usize = 200_000;

/// The five cron fields, each with its range, universe size and error label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldKind {
    /// `minute`, `0..=59`.
    Minute,
    /// `hour`, `0..=23`.
    Hour,
    /// `day-of-month`, `1..=31`.
    DayOfMonth,
    /// `month`, `1..=12`.
    Month,
    /// `day-of-week`, `0..=7` with `7` normalized to Sunday.
    DayOfWeek,
}

impl FieldKind {
    /// The field name used in validation messages.
    const fn label(self) -> &'static str {
        match self {
            Self::Minute => "minute",
            Self::Hour => "hour",
            Self::DayOfMonth => "day-of-month",
            Self::Month => "month",
            Self::DayOfWeek => "day-of-week",
        }
    }

    /// The lowest value the grammar accepts.
    const fn min(self) -> u32 {
        match self {
            Self::DayOfMonth | Self::Month => 1,
            Self::Minute | Self::Hour | Self::DayOfWeek => 0,
        }
    }

    /// The highest value the grammar accepts (before normalization).
    const fn max(self) -> u32 {
        match self {
            Self::Minute => 59,
            Self::Hour => 23,
            Self::DayOfMonth => 31,
            Self::Month => 12,
            Self::DayOfWeek => 7,
        }
    }

    /// How many distinct values a fully covered field holds after normalization.
    const fn universe(self) -> usize {
        match self {
            Self::Minute => 60,
            Self::Hour => 24,
            Self::DayOfMonth => 31,
            Self::Month => 12,
            Self::DayOfWeek => 7,
        }
    }

    /// Maps a raw accepted value to its normalized form: day-of-week `7` is Sunday (`0`).
    const fn normalize(self, value: u32) -> u32 {
        if matches!(self, Self::DayOfWeek) && value == 7 {
            0
        } else {
            value
        }
    }

    /// Parses one value of this field: a number, or a day-of-week name.
    fn parse_value(self, text: &str, raw: &str) -> Result<u32> {
        let value = if matches!(self, Self::DayOfWeek) {
            match weekday_from_name(text) {
                Some(value) => value,
                None => parse_number(text, self, raw)?,
            }
        } else {
            parse_number(text, self, raw)?
        };
        if value < self.min() || value > self.max() {
            return Err(field_error(self, raw));
        }
        Ok(value)
    }
}

/// Builds the validation error of an offending field, echoing the field text verbatim.
fn field_error(kind: FieldKind, raw: &str) -> Error {
    Error::validation(format!("invalid {} field: {raw:?}", kind.label()))
}

/// Parses a non-negative integer, reporting a malformed field on failure.
fn parse_number(text: &str, kind: FieldKind, raw: &str) -> Result<u32> {
    text.parse::<u32>().map_err(|_| field_error(kind, raw))
}

/// Maps a three-letter day-of-week name (case-insensitive) to `0` (Sunday) .. `6` (Saturday).
fn weekday_from_name(text: &str) -> Option<u32> {
    const NAMES: [(&str, u32); 7] = [
        ("SUN", 0),
        ("MON", 1),
        ("TUE", 2),
        ("WED", 3),
        ("THU", 4),
        ("FRI", 5),
        ("SAT", 6),
    ];
    NAMES
        .iter()
        .find(|(name, _)| text.eq_ignore_ascii_case(name))
        .map(|(_, value)| *value)
}

/// Expands an `@`-shorthand into the 5-field grammar; a non-shorthand expression passes through.
fn expand_shorthand(expression: &str) -> Result<&str> {
    if !expression.starts_with('@') {
        return Ok(expression);
    }
    let expanded = match expression {
        "@hourly" => "0 * * * *",
        "@daily" | "@midnight" => "0 0 * * *",
        "@weekly" => "0 0 * * 0",
        "@monthly" => "0 0 1 * *",
        "@yearly" | "@annually" => "0 0 1 1 *",
        other => {
            return Err(Error::validation(format!(
                "unknown cron shorthand: {other:?}"
            )));
        }
    };
    Ok(expanded)
}

/// Parses one comma-separated part of a field into `values`.
fn parse_part(kind: FieldKind, part: &str, raw: &str, values: &mut BTreeSet<u32>) -> Result<()> {
    let (base, step) = match part.split_once('/') {
        Some((base, step_text)) => {
            let step = step_text
                .parse::<u32>()
                .map_err(|_| field_error(kind, raw))?;
            if step == 0 {
                return Err(field_error(kind, raw));
            }
            (base, Some(step))
        }
        None => (part, None),
    };

    let (start, end) = if base == "*" {
        (kind.min(), kind.max())
    } else if let Some((start_text, end_text)) = base.split_once('-') {
        let start = kind.parse_value(start_text, raw)?;
        let end = kind.parse_value(end_text, raw)?;
        if start > end {
            return Err(field_error(kind, raw));
        }
        (start, end)
    } else {
        let value = kind.parse_value(base, raw)?;
        // `a/n` is the Vixie shorthand for `a-max/n`.
        (value, if step.is_some() { kind.max() } else { value })
    };

    let step = step.unwrap_or(1);
    let mut current = start;
    while current <= end {
        values.insert(kind.normalize(current));
        current += step;
    }
    Ok(())
}

/// Parses one whole field (a comma-separated list of parts) into a sorted, deduplicated set.
fn parse_field(kind: FieldKind, raw: &str) -> Result<Vec<u32>> {
    let mut values = BTreeSet::new();
    for part in raw.split(',') {
        parse_part(kind, part, raw, &mut values)?;
    }
    if values.is_empty() {
        return Err(field_error(kind, raw));
    }
    Ok(values.into_iter().collect())
}

/// A parsed cron expression, reusable for any number of next-occurrence queries.
///
/// Equality compares the parsed value sets and the source text, so two spellings of the same
/// schedule (`0 0 * * 0` and `@weekly`) are distinct values with identical behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronSchedule {
    /// The expression as supplied (trimmed), kept for persisted jobs and messages.
    expression: String,
    /// Minutes of the hour, sorted and deduplicated.
    minutes: Vec<u32>,
    /// Hours of the day, sorted and deduplicated.
    hours: Vec<u32>,
    /// Days of the month, sorted and deduplicated.
    days_of_month: Vec<u32>,
    /// Months of the year, sorted and deduplicated.
    months: Vec<u32>,
    /// Days of the week (`0` = Sunday), sorted and deduplicated.
    weekdays: Vec<u32>,
    /// Whether the day-of-month field restricts the days of a month.
    day_of_month_restricted: bool,
    /// Whether the day-of-week field restricts the days of a week.
    day_of_week_restricted: bool,
}

impl CronSchedule {
    /// Parses a 5-field cron expression or one of the supported `@` shorthands.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the
    /// expression is empty, has the wrong number of fields, uses an unknown shorthand, or
    /// holds a field value, range or step outside the grammar. Messages name the offending
    /// field and echo its text, for example `invalid minute field: "75"`.
    pub fn parse(expression: &str) -> Result<Self> {
        let trimmed = expression.trim();
        if trimmed.is_empty() {
            return Err(Error::validation("cron expression cannot be empty"));
        }
        let expanded = expand_shorthand(trimmed)?;
        let fields: Vec<&str> = expanded.split_whitespace().collect();
        if fields.len() != FIELD_COUNT {
            return Err(Error::validation(format!(
                "invalid cron expression: expected {FIELD_COUNT} fields (minute hour day-of-month month day-of-week), got {}: {trimmed:?}",
                fields.len()
            )));
        }

        let minutes = parse_field(FieldKind::Minute, fields[0])?;
        let hours = parse_field(FieldKind::Hour, fields[1])?;
        let days_of_month = parse_field(FieldKind::DayOfMonth, fields[2])?;
        let months = parse_field(FieldKind::Month, fields[3])?;
        let weekdays = parse_field(FieldKind::DayOfWeek, fields[4])?;

        let day_of_month_restricted = days_of_month.len() < FieldKind::DayOfMonth.universe();
        let day_of_week_restricted = weekdays.len() < FieldKind::DayOfWeek.universe();

        Ok(Self {
            expression: trimmed.to_owned(),
            minutes,
            hours,
            days_of_month,
            months,
            weekdays,
            day_of_month_restricted,
            day_of_week_restricted,
        })
    }

    /// The expression as supplied (trimmed), which is what a persisted job stores.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Returns the first minute strictly after `now` that satisfies the whole expression.
    ///
    /// The search works at minute granularity in UTC, ignores the seconds and sub-second part
    /// of `now`, and never returns `now` itself — a query at `09:00:00` for `0 9 * * *` yields
    /// the next day's `09:00`. Returns `None` when the expression can never match (for example
    /// `0 0 31 2 *`) or when the next match lies beyond the five-year search horizon.
    #[must_use]
    pub fn next_after(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        let horizon = now.checked_add_signed(Duration::days(SEARCH_HORIZON_DAYS))?;
        let mut candidate = now
            .with_second(0)?
            .with_nanosecond(0)?
            .checked_add_signed(Duration::minutes(1))?;
        let mut steps = 0usize;
        loop {
            if candidate > horizon || steps >= MAX_SEARCH_STEPS {
                return None;
            }
            steps += 1;
            if !self.months.contains(&candidate.month()) {
                candidate = next_month_start(candidate)?;
                continue;
            }
            if !self.day_matches(&candidate) {
                candidate = next_day_start(candidate)?;
                continue;
            }
            if !self.hours.contains(&candidate.hour()) {
                candidate = next_hour_start(candidate)?;
                continue;
            }
            if !self.minutes.contains(&candidate.minute()) {
                candidate = candidate.checked_add_signed(Duration::minutes(1))?;
                continue;
            }
            return Some(candidate);
        }
    }

    /// Whether `candidate`'s date satisfies the day-of-month/day-of-week pair.
    ///
    /// Both restricted means OR (standard cron); one unrestricted means it alone decides.
    fn day_matches(&self, candidate: &DateTime<Utc>) -> bool {
        let dom = self.days_of_month.contains(&candidate.day());
        let dow = self
            .weekdays
            .contains(&candidate.weekday().num_days_from_sunday());
        match (self.day_of_month_restricted, self.day_of_week_restricted) {
            (false, false) => true,
            (false, true) => dow,
            (true, false) => dom,
            (true, true) => dom || dow,
        }
    }
}

/// Advances to the first minute of the next hour.
fn next_hour_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let minutes = 60 - candidate.minute();
    candidate.checked_add_signed(Duration::minutes(i64::from(minutes)))
}

/// Advances to midnight of the next day.
fn next_day_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = candidate.date_naive().succ_opt()?;
    Some(next.and_hms_opt(0, 0, 0)?.and_utc())
}

/// Advances to the first minute of the next month.
fn next_month_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let (year, month) = if candidate.month() == 12 {
        (candidate.year().checked_add(1)?, 1)
    } else {
        (candidate.year(), candidate.month() + 1)
    };
    Some(
        NaiveDate::from_ymd_opt(year, month, 1)?
            .and_hms_opt(0, 0, 0)?
            .and_utc(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    /// Builds a UTC instant at the given wall-clock minute.
    fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .unwrap()
    }

    /// Parses `expression`, returning its validation message.
    fn parse_error(expression: &str) -> String {
        CronSchedule::parse(expression)
            .unwrap_err()
            .message()
            .into_owned()
    }

    #[test]
    fn parses_five_field_expression() {
        let schedule = CronSchedule::parse("*/15 9 1,15 6 MON-FRI").unwrap();
        assert_eq!(schedule.expression(), "*/15 9 1,15 6 MON-FRI");
        assert_eq!(schedule.minutes, vec![0, 15, 30, 45]);
        assert_eq!(schedule.hours, vec![9]);
        assert_eq!(schedule.days_of_month, vec![1, 15]);
        assert_eq!(schedule.months, vec![6]);
        assert_eq!(schedule.weekdays, vec![1, 2, 3, 4, 5]);
        assert!(schedule.day_of_month_restricted);
        assert!(schedule.day_of_week_restricted);
    }

    #[test]
    fn leading_and_trailing_whitespace_is_ignored() {
        let schedule = CronSchedule::parse("  @daily  ").unwrap();
        assert_eq!(schedule.expression(), "@daily");
        assert_eq!(schedule.hours, vec![0]);
        assert_eq!(schedule.minutes, vec![0]);
    }

    #[test]
    fn expands_shorthands() {
        let hourly = CronSchedule::parse("@hourly").unwrap();
        assert_eq!(hourly.minutes, vec![0]);
        assert_eq!(hourly.hours.len(), 24);
        assert!(!hourly.day_of_month_restricted);
        assert!(!hourly.day_of_week_restricted);

        let daily = CronSchedule::parse("@daily").unwrap();
        assert_eq!(daily.minutes, vec![0]);
        assert_eq!(daily.hours, vec![0]);
        assert!(!daily.day_of_month_restricted);
        assert_eq!(CronSchedule::parse("@midnight").unwrap().hours, vec![0]);

        let weekly = CronSchedule::parse("@weekly").unwrap();
        assert_eq!(weekly.minutes, vec![0]);
        assert_eq!(weekly.hours, vec![0]);
        assert_eq!(weekly.weekdays, vec![0]);
        assert!(weekly.day_of_week_restricted);

        let monthly = CronSchedule::parse("@monthly").unwrap();
        assert_eq!(monthly.days_of_month, vec![1]);
        assert_eq!(monthly.months.len(), 12);

        let yearly = CronSchedule::parse("@yearly").unwrap();
        assert_eq!(yearly.days_of_month, vec![1]);
        assert_eq!(yearly.months, vec![1]);
        assert_eq!(CronSchedule::parse("@annually").unwrap().months, vec![1]);
    }

    #[test]
    fn supports_steps_ranges_and_lists() {
        let schedule = CronSchedule::parse("0-30/10 8,20 * * *").unwrap();
        assert_eq!(schedule.minutes, vec![0, 10, 20, 30]);
        assert_eq!(schedule.hours, vec![8, 20]);

        // `a/n` is `a-max/n`, the Vixie reading of a start-only step.
        let open_step = CronSchedule::parse("5/20 * * * *").unwrap();
        assert_eq!(open_step.minutes, vec![5, 25, 45]);
    }

    #[test]
    fn normalizes_and_deduplicates_values() {
        let schedule = CronSchedule::parse("0,0,0 9 * * 7,7,0").unwrap();
        assert_eq!(schedule.minutes, vec![0]);
        assert_eq!(schedule.weekdays, vec![0]);
        assert!(schedule.day_of_week_restricted);
    }

    #[test]
    fn full_day_of_week_range_is_unrestricted() {
        let schedule = CronSchedule::parse("0 0 * * 0-7").unwrap();
        assert_eq!(schedule.weekdays, vec![0, 1, 2, 3, 4, 5, 6]);
        assert!(!schedule.day_of_week_restricted);
        assert!(!schedule.day_of_month_restricted);
    }

    #[test]
    fn rejects_wrong_field_count() {
        assert_eq!(
            parse_error("* * * *"),
            "invalid cron expression: expected 5 fields (minute hour day-of-month month day-of-week), got 4: \"* * * *\""
        );
        assert_eq!(
            parse_error("0 0 0 1 1 1"),
            "invalid cron expression: expected 5 fields (minute hour day-of-month month day-of-week), got 6: \"0 0 0 1 1 1\""
        );
        assert_eq!(parse_error("   "), "cron expression cannot be empty");
    }

    #[test]
    fn rejects_unknown_shorthands() {
        assert_eq!(parse_error("@every"), "unknown cron shorthand: \"@every\"");
    }

    #[test]
    fn rejects_out_of_range_values() {
        assert_eq!(parse_error("75 * * * *"), "invalid minute field: \"75\"");
        assert_eq!(parse_error("0 24 * * *"), "invalid hour field: \"24\"");
        assert_eq!(
            parse_error("0 0 0 * *"),
            "invalid day-of-month field: \"0\""
        );
        assert_eq!(
            parse_error("0 0 32 * *"),
            "invalid day-of-month field: \"32\""
        );
        assert_eq!(parse_error("0 0 * 13 *"), "invalid month field: \"13\"");
        assert_eq!(parse_error("0 0 * * 8"), "invalid day-of-week field: \"8\"");
        assert_eq!(
            parse_error("0 0 * * MONDAY"),
            "invalid day-of-week field: \"MONDAY\""
        );
    }

    #[test]
    fn rejects_malformed_fields() {
        assert_eq!(parse_error("*/0 * * * *"), "invalid minute field: \"*/0\"");
        assert_eq!(parse_error("5-1 * * * *"), "invalid minute field: \"5-1\"");
        assert_eq!(
            parse_error("1,,2 * * * *"),
            "invalid minute field: \"1,,2\""
        );
        assert_eq!(
            parse_error("*/-1 * * * *"),
            "invalid minute field: \"*/-1\""
        );
        assert_eq!(
            parse_error("1/2/3 * * * *"),
            "invalid minute field: \"1/2/3\""
        );
        assert_eq!(parse_error("a * * * *"), "invalid minute field: \"a\"");
        assert_eq!(
            parse_error("0 0 * * FRI-MON"),
            "invalid day-of-week field: \"FRI-MON\""
        );
    }

    #[test]
    fn next_after_is_strictly_after_an_exact_minute() {
        let schedule = CronSchedule::parse("0 9 * * *").unwrap();
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 9, 0)),
            Some(at(2024, 1, 2, 9, 0))
        );
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 8, 59)),
            Some(at(2024, 1, 1, 9, 0))
        );
    }

    #[test]
    fn next_after_ignores_seconds_and_nanoseconds() {
        let schedule = CronSchedule::parse("* * * * *").unwrap();
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 30).unwrap() + Duration::nanoseconds(500);
        assert_eq!(schedule.next_after(now), Some(at(2024, 1, 1, 0, 1)));
    }

    #[test]
    fn next_after_steps_by_quarter_hour() {
        let schedule = CronSchedule::parse("*/15 * * * *").unwrap();
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 0, 0)),
            Some(at(2024, 1, 1, 0, 15))
        );
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 0, 7)),
            Some(at(2024, 1, 1, 0, 15))
        );
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 0, 15)),
            Some(at(2024, 1, 1, 0, 30))
        );
    }

    #[test]
    fn next_after_crosses_hours_days_and_months() {
        let hourly = CronSchedule::parse("@hourly").unwrap();
        assert_eq!(
            hourly.next_after(at(2024, 1, 1, 0, 30)),
            Some(at(2024, 1, 1, 1, 0))
        );
        assert_eq!(
            hourly.next_after(at(2024, 1, 1, 23, 59)),
            Some(at(2024, 1, 2, 0, 0))
        );

        let monthly = CronSchedule::parse("0 0 1 * *").unwrap();
        assert_eq!(
            monthly.next_after(at(2024, 1, 15, 0, 0)),
            Some(at(2024, 2, 1, 0, 0))
        );
        assert_eq!(
            monthly.next_after(at(2024, 12, 1, 0, 0)),
            Some(at(2025, 1, 1, 0, 0))
        );

        let yearly = CronSchedule::parse("@yearly").unwrap();
        assert_eq!(
            yearly.next_after(at(2024, 6, 1, 0, 0)),
            Some(at(2025, 1, 1, 0, 0))
        );
    }

    #[test]
    fn next_after_rolls_over_a_year_boundary() {
        let schedule = CronSchedule::parse("0 0 31 12 *").unwrap();
        assert_eq!(
            schedule.next_after(at(2024, 12, 31, 0, 0)),
            Some(at(2025, 12, 31, 0, 0))
        );
    }

    #[test]
    fn next_after_selects_weekdays_by_number_and_name() {
        // 2024-01-01 is a Monday.
        let weekdays = CronSchedule::parse("0 9 * * MON-FRI").unwrap();
        assert_eq!(
            weekdays.next_after(at(2024, 1, 1, 0, 0)),
            Some(at(2024, 1, 1, 9, 0))
        );
        assert_eq!(
            weekdays.next_after(at(2024, 1, 5, 10, 0)),
            Some(at(2024, 1, 8, 9, 0)),
            "Friday after 09:00 rolls over the weekend"
        );
        assert_eq!(
            weekdays.next_after(at(2024, 1, 6, 10, 0)),
            Some(at(2024, 1, 8, 9, 0)),
            "Saturday rolls to Monday"
        );

        let sunday = CronSchedule::parse("0 0 * * SUN").unwrap();
        assert_eq!(
            sunday.next_after(at(2024, 1, 1, 0, 0)),
            Some(at(2024, 1, 7, 0, 0))
        );

        let sunday_numeric = CronSchedule::parse("0 0 * * 7").unwrap();
        assert_eq!(
            sunday_numeric.next_after(at(2024, 1, 1, 0, 0)),
            Some(at(2024, 1, 7, 0, 0))
        );
    }

    #[test]
    fn next_after_hits_the_leap_day() {
        let schedule = CronSchedule::parse("0 0 29 2 *").unwrap();
        assert_eq!(
            schedule.next_after(at(2024, 2, 28, 0, 0)),
            Some(at(2024, 2, 29, 0, 0))
        );
        assert_eq!(
            schedule.next_after(at(2024, 3, 1, 0, 0)),
            Some(at(2028, 2, 29, 0, 0)),
            "the next leap day after March 2024 is in 2028"
        );
    }

    #[test]
    fn next_after_uses_or_semantics_for_both_day_fields() {
        // 2024-01-05 is a Friday, 2024-01-13 is a Saturday: the Friday matches first.
        let schedule = CronSchedule::parse("0 0 13 * FRI").unwrap();
        assert_eq!(
            schedule.next_after(at(2024, 1, 1, 0, 0)),
            Some(at(2024, 1, 5, 0, 0))
        );
        assert_eq!(
            schedule.next_after(at(2024, 1, 6, 0, 0)),
            Some(at(2024, 1, 12, 0, 0)),
            "the next Friday after 2024-01-05"
        );
        assert_eq!(
            schedule.next_after(at(2024, 1, 12, 0, 1)),
            Some(at(2024, 1, 13, 0, 0)),
            "the 13th matches through the day-of-month branch"
        );
    }

    #[test]
    fn next_after_returns_none_for_impossible_expressions() {
        assert_eq!(
            CronSchedule::parse("0 0 31 2 *")
                .unwrap()
                .next_after(at(2024, 1, 1, 0, 0)),
            None
        );
        assert_eq!(
            CronSchedule::parse("0 0 30 2 *")
                .unwrap()
                .next_after(at(2024, 1, 1, 0, 0)),
            None
        );
    }

    #[test]
    fn next_after_handles_a_far_future_year() {
        let schedule = CronSchedule::parse("0 0 1 1 *").unwrap();
        // Near the top of the representable calendar the five-year horizon itself overflows,
        // so the search reports "no match" instead of panicking.
        assert_eq!(schedule.next_after(at(262_140, 6, 1, 0, 0)), None);
    }

    #[test]
    fn parses_weekday_names_case_insensitively() {
        let lower = CronSchedule::parse("0 0 * * mon-fri").unwrap();
        assert_eq!(lower.weekdays, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn equality_includes_the_source_spelling() {
        let numeric = CronSchedule::parse("0 0 * * 0").unwrap();
        let named = CronSchedule::parse("0 0 * * SUN").unwrap();
        assert_eq!(numeric.weekdays, named.weekdays);
        assert_ne!(numeric, named);
    }
}
