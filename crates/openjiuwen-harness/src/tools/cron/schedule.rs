//! Cron expression parsing and next-occurrence computation.
//!
//! Two expression forms are accepted:
//!
//! - **5-field classic cron** — `minute hour day-of-month month day-of-week`,
//! - **6- or 7-field Quartz** — `second minute hour day-of-month month day-of-week [year]`.
//!
//! Both forms share the value grammar: `*` (every value), `*/n` (every `n`-th value starting
//! at the field minimum), `a` (a single value), `a-b` (an inclusive range), `a-b/n` and `a/n`
//! (which reads as `a-max/n`) — all combinable in comma lists. Month names `JAN`..`DEC` and
//! day-of-week names `SUN`..`SAT` are accepted case-insensitively wherever a number is; the
//! numeric day-of-week range is `0..=7`, where both `0` and `7` mean Sunday. The shorthands
//! `@hourly`, `@daily`, `@midnight`, `@weekly`, `@monthly`, `@yearly` and `@annually` expand to
//! the 5-field form.
//!
//! # The seconds and year fields
//!
//! A six-field expression starts with a seconds field (`0..=59`) and matches at second
//! granularity; a five-field expression keeps its minute granularity. The optional seventh
//! field restricts the year to `1970..=2099`.
//!
//! # Day-of-month / day-of-week rules
//!
//! Classic cron combines the two day fields with **OR semantics**: when *both* are restricted,
//! a day matches if either the day-of-month or the day-of-week matches; when one is
//! unrestricted (`*`), only the other decides. A field counts as unrestricted when its
//! normalized value set covers the whole range (`*`, `*/1`, or `0-7` for day-of-week).
//!
//! Quartz requires exactly one of the two day fields to be `?` ("no specific value"): the
//! unspecified field is ignored and the specified one decides. Two `?` fields, or none, are a
//! typed parse error. Because the rule leaves exactly one restricted field, the OR table above
//! reduces to "the specified field decides".
//!
//! # Quartz extensions
//!
//! - `L` in day-of-month — the last day of the month; `L-n` — `n` days before it (`0..=30`).
//! - `nL` in day-of-week — the last weekday `n` of the month; a bare `L` is Saturday (`6L`).
//! - `d#n` in day-of-week — the `n`-th weekday `d` of the month (`n` in `1..=5`).
//! - wrap-around ranges — `22-2` for hours (22, 23, 0, 1) and `FRI-MON` for weekdays.
//! - `W` (nearest weekday, written `15W` or `LW`) is **not** implemented: it is rejected with
//!   `unsupported cron token: "15W": "W" (nearest weekday) is not supported`, so a caller
//!   never gets a silently different schedule.
//! - every Quartz-only token is rejected in the 5-field form, which keeps its classic
//!   contract (`0 0 * * FRI-MON` stays `invalid day-of-week field: "FRI-MON"`).
//!
//! # Bounded search
//!
//! [`CronSchedule::next_after`] scans at most [`SEARCH_HORIZON_DAYS`] days ahead (extended to
//! the end of the last accepted year when the expression names one) and at most
//! [`MAX_SEARCH_STEPS`] loop iterations, so a schedule that can never match (for example
//! `0 0 31 2 *`) returns `None` instead of looping forever. The horizon is a documented
//! limitation: a leap-day schedule never needs more than four years, but the eight-year gap
//! around a skipped century leap day (for example 2096 → 2104) exceeds it.

use std::collections::BTreeSet;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Timelike, Utc};

use openjiuwen_common::error::{Error, Result};

/// Field count of the classic form (`minute hour day-of-month month day-of-week`).
const CLASSIC_FIELDS: usize = 5;

/// Field count of the six-field Quartz form (leading seconds, no year).
const QUARTZ_FIELDS: usize = 6;

/// Field count of the seven-field Quartz form (leading seconds plus a year).
const QUARTZ_YEAR_FIELDS: usize = 7;

/// How far ahead [`CronSchedule::next_after`] searches, in days (five years plus slack).
const SEARCH_HORIZON_DAYS: i64 = 5 * 366 + 1;

/// Safety cap on the scan iterations of [`CronSchedule::next_after`].
///
/// Every rejected candidate jumps to the next matching second, minute, hour, day, month or
/// year, so the cap is only a guard against a pathological loop; the horizon is the semantic
/// bound.
const MAX_SEARCH_STEPS: usize = 200_000;

/// Lowest year the Quartz `year` field accepts.
const MIN_YEAR: u32 = 1970;

/// Highest year the Quartz `year` field accepts.
const MAX_YEAR: u32 = 2099;

/// Largest `n` accepted by the `L-n` day-of-month token (a month has at most 31 days).
const MAX_LAST_DAY_OFFSET: u32 = 30;

/// Largest `n` accepted by the `d#n` day-of-week token (a month holds at most five of a day).
const MAX_NTH_WEEKDAY: u32 = 5;

/// The expression form, which decides the field layout and the extension rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Five fields, minute granularity, Vixie day-field OR semantics, no extensions.
    Classic,
    /// Six or seven fields, second granularity, Quartz day-field rule and extensions.
    Quartz,
}

impl Form {
    /// Whether this form accepts the Quartz-only tokens.
    const fn is_quartz(self) -> bool {
        matches!(self, Self::Quartz)
    }
}

/// The cron fields, each with its range, universe size and error label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldKind {
    /// `second`, `0..=59` (Quartz only).
    Second,
    /// `minute`, `0..=59`.
    Minute,
    /// `hour`, `0..=23`.
    Hour,
    /// `day-of-month`, `1..=31`.
    DayOfMonth,
    /// `month`, `1..=12`, or `JAN`..`DEC`.
    Month,
    /// `day-of-week`, `0..=7` with `7` normalized to Sunday, or `SUN`..`SAT`.
    DayOfWeek,
    /// `year`, `1970..=2099` (Quartz only).
    Year,
}

impl FieldKind {
    /// The field name used in validation messages.
    const fn label(self) -> &'static str {
        match self {
            Self::Second => "second",
            Self::Minute => "minute",
            Self::Hour => "hour",
            Self::DayOfMonth => "day-of-month",
            Self::Month => "month",
            Self::DayOfWeek => "day-of-week",
            Self::Year => "year",
        }
    }

    /// The lowest value the grammar accepts.
    const fn min(self) -> u32 {
        match self {
            Self::DayOfMonth | Self::Month => 1,
            Self::Year => MIN_YEAR,
            Self::Second | Self::Minute | Self::Hour | Self::DayOfWeek => 0,
        }
    }

    /// The highest value the grammar accepts (before normalization).
    const fn max(self) -> u32 {
        match self {
            Self::Second | Self::Minute => 59,
            Self::Hour => 23,
            Self::DayOfMonth => 31,
            Self::Month => 12,
            Self::DayOfWeek => 7,
            Self::Year => MAX_YEAR,
        }
    }

    /// How many distinct values a fully covered field holds after normalization.
    const fn universe(self) -> usize {
        match self {
            Self::Second | Self::Minute => 60,
            Self::Hour => 24,
            Self::DayOfMonth => 31,
            Self::Month => 12,
            Self::DayOfWeek => 7,
            Self::Year => (MAX_YEAR - MIN_YEAR + 1) as usize,
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

    /// Parses one value of this field: a number, or a month / day-of-week name.
    fn parse_value(self, text: &str, raw: &str) -> Result<u32> {
        let named = match self {
            Self::DayOfWeek => weekday_from_name(text),
            Self::Month => month_from_name(text),
            _ => None,
        };
        let value = match named {
            Some(value) => value,
            None => parse_number(text, self, raw)?,
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

/// Builds the error of a recognised-but-unsupported Quartz token.
fn unsupported_token(part: &str, what: &str) -> Error {
    Error::validation(format!(
        "unsupported cron token: {part:?}: {what} is not supported"
    ))
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

/// Maps a three-letter month name (case-insensitive) to `1` (January) .. `12` (December).
fn month_from_name(text: &str) -> Option<u32> {
    const NAMES: [(&str, u32); 12] = [
        ("JAN", 1),
        ("FEB", 2),
        ("MAR", 3),
        ("APR", 4),
        ("MAY", 5),
        ("JUN", 6),
        ("JUL", 7),
        ("AUG", 8),
        ("SEP", 9),
        ("OCT", 10),
        ("NOV", 11),
        ("DEC", 12),
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

/// One parsed field: its plain values plus the Quartz specials of a day field.
#[derive(Debug, Clone, Default)]
struct ParsedField {
    /// Plain values, normalized and deduplicated.
    values: BTreeSet<u32>,
    /// Day-of-month `L` / `L-n` tokens, as offsets from the last day of the month.
    last_day_offsets: BTreeSet<u32>,
    /// Day-of-week `nL` tokens, as weekday numbers.
    last_weekdays: BTreeSet<u32>,
    /// Day-of-week `d#n` tokens, as `(weekday, nth)` pairs.
    nth_weekdays: BTreeSet<(u32, u32)>,
}

impl ParsedField {
    /// Whether no value and no special token was found, which is always an error.
    fn is_empty(&self) -> bool {
        self.values.is_empty()
            && self.last_day_offsets.is_empty()
            && self.last_weekdays.is_empty()
            && self.nth_weekdays.is_empty()
    }

    /// Whether the field selects every value of its range and holds no special token.
    fn covers(&self, kind: FieldKind) -> bool {
        self.last_day_offsets.is_empty()
            && self.last_weekdays.is_empty()
            && self.nth_weekdays.is_empty()
            && self.values.len() == kind.universe()
    }

    /// Consumes the field into its owned, sorted value vectors.
    fn into_parts(self) -> FieldParts {
        FieldParts {
            values: self.values.into_iter().collect(),
            last_day_offsets: self.last_day_offsets.into_iter().collect(),
            last_weekdays: self.last_weekdays.into_iter().collect(),
            nth_weekdays: self.nth_weekdays.into_iter().collect(),
        }
    }
}

/// The owned, sorted form of a [`ParsedField`] — what [`CronSchedule::parse`] stores.
///
/// The parser works on [`BTreeSet`]s so duplicates and ordering are handled while parts are
/// being added; this struct is the plain `Vec` view the schedule keeps afterwards.
#[derive(Debug, Default)]
struct FieldParts {
    /// Plain values, normalized and deduplicated.
    values: Vec<u32>,
    /// Day-of-month `L` / `L-n` tokens, as offsets from the last day of the month.
    last_day_offsets: Vec<u32>,
    /// Day-of-week `nL` tokens, as weekday numbers.
    last_weekdays: Vec<u32>,
    /// Day-of-week `d#n` tokens, as `(weekday, nth)` pairs.
    nth_weekdays: Vec<(u32, u32)>,
}

/// Parses one comma-separated part of a field into `out`.
fn parse_part(
    kind: FieldKind,
    part: &str,
    raw: &str,
    form: Form,
    out: &mut ParsedField,
) -> Result<()> {
    if part.is_empty() {
        return Err(field_error(kind, raw));
    }
    let upper = part.to_ascii_uppercase();
    if upper.contains('W') {
        return Err(unsupported_token(part, "\"W\" (nearest weekday)"));
    }
    if part.contains('?') {
        return Err(unsupported_token(raw, "\"?\" (must be the whole field)"));
    }
    match kind {
        FieldKind::DayOfWeek if upper.contains('#') => {
            if form.is_quartz() {
                return parse_nth_weekday(part, raw, out);
            }
            return Err(field_error(kind, raw));
        }
        FieldKind::DayOfWeek if upper.ends_with('L') => {
            if form.is_quartz() {
                return parse_last_weekday(part, raw, out);
            }
            return Err(field_error(kind, raw));
        }
        FieldKind::DayOfMonth if upper.starts_with('L') => {
            if form.is_quartz() {
                return parse_last_day(part, raw, out);
            }
            return Err(field_error(kind, raw));
        }
        _ => {}
    }
    parse_clock_part(kind, part, raw, form, out)
}

/// Parses the `d#n` day-of-week token: the `n`-th weekday `d` of the month.
fn parse_nth_weekday(part: &str, raw: &str, out: &mut ParsedField) -> Result<()> {
    let Some((day_text, nth_text)) = part.split_once('#') else {
        return Err(field_error(FieldKind::DayOfWeek, raw));
    };
    let weekday = FieldKind::DayOfWeek.parse_value(day_text, raw)?;
    let nth = parse_number(nth_text, FieldKind::DayOfWeek, raw)?;
    if !(1..=MAX_NTH_WEEKDAY).contains(&nth) {
        return Err(field_error(FieldKind::DayOfWeek, raw));
    }
    out.nth_weekdays
        .insert((FieldKind::DayOfWeek.normalize(weekday), nth));
    Ok(())
}

/// Parses the `nL` day-of-week token: the last weekday `n` of the month.
///
/// A bare `L` is Saturday, which is the Quartz reading of "the last day of the week".
fn parse_last_weekday(part: &str, raw: &str, out: &mut ParsedField) -> Result<()> {
    let prefix = &part[..part.len() - 1];
    let weekday = if prefix.is_empty() {
        6
    } else {
        FieldKind::DayOfWeek.normalize(FieldKind::DayOfWeek.parse_value(prefix, raw)?)
    };
    out.last_weekdays.insert(weekday);
    Ok(())
}

/// Parses the `L` / `L-n` day-of-month token: the last day of the month, minus `n`.
fn parse_last_day(part: &str, raw: &str, out: &mut ParsedField) -> Result<()> {
    let rest = &part[1..];
    let offset = if rest.is_empty() {
        0
    } else {
        let Some(digits) = rest.strip_prefix('-') else {
            return Err(field_error(FieldKind::DayOfMonth, raw));
        };
        let offset = parse_number(digits, FieldKind::DayOfMonth, raw)?;
        if offset > MAX_LAST_DAY_OFFSET {
            return Err(field_error(FieldKind::DayOfMonth, raw));
        }
        offset
    };
    out.last_day_offsets.insert(offset);
    Ok(())
}

/// Parses one comma-separated numeric part of a field into `out`.
fn parse_clock_part(
    kind: FieldKind,
    part: &str,
    raw: &str,
    form: Form,
    out: &mut ParsedField,
) -> Result<()> {
    let (base, step) = match part.split_once('/') {
        Some((base, step_text)) => {
            let step = parse_number(step_text, kind, raw)?;
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
            if !form.is_quartz() {
                return Err(field_error(kind, raw));
            }
            // Quartz wrap-around range: `22-2` is 22, 23, 0, 1.
            insert_wrapped(kind, start, end, step.unwrap_or(1), out);
            return Ok(());
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
        out.values.insert(kind.normalize(current));
        current += step;
    }
    Ok(())
}

/// Inserts a wrap-around range (`start > end`) into `out`, stepping along the unrolled values.
fn insert_wrapped(kind: FieldKind, start: u32, end: u32, step: u32, out: &mut ParsedField) {
    let unrolled: Vec<u32> = (start..=kind.max()).chain(kind.min()..=end).collect();
    let stride = usize::try_from(step).unwrap_or(1);
    for value in unrolled.into_iter().step_by(stride) {
        out.values.insert(kind.normalize(value));
    }
}

/// Parses one whole field (a comma-separated list of parts).
fn parse_field(kind: FieldKind, raw: &str, form: Form) -> Result<ParsedField> {
    let mut parsed = ParsedField::default();
    for part in raw.split(',') {
        parse_part(kind, part, raw, form, &mut parsed)?;
    }
    if parsed.is_empty() {
        return Err(field_error(kind, raw));
    }
    Ok(parsed)
}

/// Parses an optional day field, mapping Quartz `?` to "no specific value".
fn parse_day_field(kind: FieldKind, raw: &str, form: Form) -> Result<Option<ParsedField>> {
    if form.is_quartz() && raw == "?" {
        return Ok(None);
    }
    parse_field(kind, raw, form).map(Some)
}

/// Parses a required field into its sorted, deduplicated values.
fn parse_values(kind: FieldKind, text: &str, form: Form) -> Result<Vec<u32>> {
    Ok(parse_field(kind, text, form)?.values.into_iter().collect())
}

/// Parses an optional field (`second`, `year`) into its sorted, deduplicated values.
fn parse_optional_values(
    kind: FieldKind,
    text: Option<&str>,
    form: Form,
) -> Result<Option<Vec<u32>>> {
    text.map(|text| parse_values(kind, text, form)).transpose()
}

/// The raw texts of one expression's fields, in their named slots.
#[derive(Debug, Clone, Copy)]
struct FieldTexts<'a> {
    /// The expression form implied by the field count.
    form: Form,
    /// The Quartz `second` field, absent in the classic form.
    second: Option<&'a str>,
    /// The `minute` field.
    minute: &'a str,
    /// The `hour` field.
    hour: &'a str,
    /// The `day-of-month` field, which may be `?` in a Quartz expression.
    day_of_month: &'a str,
    /// The `month` field.
    month: &'a str,
    /// The `day-of-week` field, which may be `?` in a Quartz expression.
    day_of_week: &'a str,
    /// The Quartz `year` field, present only in the seven-field form.
    year: Option<&'a str>,
}

impl<'a> FieldTexts<'a> {
    /// Splits `expanded` into its named slots, rejecting an unsupported field count.
    ///
    /// `expression` is the trimmed source text, echoed verbatim in the error message.
    fn split(expanded: &'a str, expression: &str) -> Result<Self> {
        let fields: Vec<&'a str> = expanded.split_whitespace().collect();
        match fields.as_slice() {
            [minute, hour, day_of_month, month, day_of_week] => Ok(Self {
                form: Form::Classic,
                second: None,
                minute: *minute,
                hour: *hour,
                day_of_month: *day_of_month,
                month: *month,
                day_of_week: *day_of_week,
                year: None,
            }),
            [second, minute, hour, day_of_month, month, day_of_week] => Ok(Self {
                form: Form::Quartz,
                second: Some(*second),
                minute: *minute,
                hour: *hour,
                day_of_month: *day_of_month,
                month: *month,
                day_of_week: *day_of_week,
                year: None,
            }),
            [second, minute, hour, day_of_month, month, day_of_week, year] => Ok(Self {
                form: Form::Quartz,
                second: Some(*second),
                minute: *minute,
                hour: *hour,
                day_of_month: *day_of_month,
                month: *month,
                day_of_week: *day_of_week,
                year: Some(*year),
            }),
            _ => Err(Error::validation(format!(
                "invalid cron expression: expected {CLASSIC_FIELDS} fields \
                 (minute hour day-of-month month day-of-week) or {QUARTZ_FIELDS}/\
                 {QUARTZ_YEAR_FIELDS} Quartz fields (second minute hour day-of-month month \
                 day-of-week [year]), got {}: {expression:?}",
                fields.len()
            ))),
        }
    }
}

/// Enforces the Quartz rule that exactly one of the two day fields is `?`.
///
/// `expression` is the trimmed source text, echoed verbatim in the error message.
fn require_single_day_field(
    day_of_month: Option<&ParsedField>,
    day_of_week: Option<&ParsedField>,
    expression: &str,
) -> Result<()> {
    match (day_of_month.is_some(), day_of_week.is_some()) {
        (true, true) => Err(Error::validation(format!(
            "invalid cron expression: both day-of-month and day-of-week are \
             specified (one must be \"?\" in a Quartz expression): {expression:?}"
        ))),
        (false, false) => Err(Error::validation(format!(
            "invalid cron expression: both day-of-month and day-of-week are \"?\" \
             (exactly one must be specified in a Quartz expression): {expression:?}"
        ))),
        _ => Ok(()),
    }
}

/// A parsed cron expression, reusable for any number of next-occurrence queries.
///
/// Equality compares the parsed value sets and the source text, so two spellings of the same
/// schedule (`0 0 * * 0` and `@weekly`) are distinct values with identical behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronSchedule {
    /// The expression as supplied (trimmed), kept for persisted jobs and messages.
    expression: String,
    /// Seconds of the minute; `Some` for the Quartz forms, `None` for minute granularity.
    seconds: Option<Vec<u32>>,
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
    /// Accepted years, sorted and deduplicated; `None` means every year.
    years: Option<Vec<u32>>,
    /// Day-of-month `L` / `L-n` tokens, as offsets from the last day of the month.
    last_day_offsets: Vec<u32>,
    /// Day-of-week `nL` tokens: the last such weekday of the month.
    last_weekdays: Vec<u32>,
    /// Day-of-week `d#n` tokens: the `n`-th such weekday of the month.
    nth_weekdays: Vec<(u32, u32)>,
    /// Whether the day-of-month field restricts the days of a month.
    day_of_month_restricted: bool,
    /// Whether the day-of-week field restricts the days of a week.
    day_of_week_restricted: bool,
}

impl CronSchedule {
    /// Parses a 5-field classic or 6/7-field Quartz expression, or a supported `@` shorthand.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the
    /// expression is empty, has the wrong number of fields, uses an unknown shorthand, breaks
    /// the Quartz `?` rule, or holds a value, range, step or token outside the grammar.
    /// Messages name the offending field and echo its text, for example
    /// `invalid minute field: "75"`, or name an unsupported token, for example
    /// `unsupported cron token: "15W": "W" (nearest weekday) is not supported`.
    pub fn parse(expression: &str) -> Result<Self> {
        let trimmed = expression.trim();
        if trimmed.is_empty() {
            return Err(Error::validation("cron expression cannot be empty"));
        }
        let expanded = expand_shorthand(trimmed)?;
        let fields = FieldTexts::split(expanded, trimmed)?;
        let form = fields.form;

        let day_of_month = parse_day_field(FieldKind::DayOfMonth, fields.day_of_month, form)?;
        let day_of_week = parse_day_field(FieldKind::DayOfWeek, fields.day_of_week, form)?;
        if form.is_quartz() {
            require_single_day_field(day_of_month.as_ref(), day_of_week.as_ref(), trimmed)?;
        }

        let seconds = parse_optional_values(FieldKind::Second, fields.second, form)?;
        let minutes = parse_values(FieldKind::Minute, fields.minute, form)?;
        let hours = parse_values(FieldKind::Hour, fields.hour, form)?;
        let months = parse_values(FieldKind::Month, fields.month, form)?;
        let years = parse_optional_values(FieldKind::Year, fields.year, form)?;

        let day_of_month_restricted = day_of_month
            .as_ref()
            .is_some_and(|field| !field.covers(FieldKind::DayOfMonth));
        let day_of_week_restricted = day_of_week
            .as_ref()
            .is_some_and(|field| !field.covers(FieldKind::DayOfWeek));
        let parts = day_of_month.unwrap_or_default().into_parts();
        let weekday_parts = day_of_week.unwrap_or_default().into_parts();

        Ok(Self {
            expression: trimmed.to_owned(),
            seconds,
            minutes,
            hours,
            days_of_month: parts.values,
            months,
            weekdays: weekday_parts.values,
            years,
            last_day_offsets: parts.last_day_offsets,
            last_weekdays: weekday_parts.last_weekdays,
            nth_weekdays: weekday_parts.nth_weekdays,
            day_of_month_restricted,
            day_of_week_restricted,
        })
    }

    /// The expression as supplied (trimmed), which is what a persisted job stores.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Whether the expression carries a seconds field, i.e. matches at second granularity.
    #[must_use]
    pub fn has_seconds(&self) -> bool {
        self.seconds.is_some()
    }

    /// Returns the first instant strictly after `now` that satisfies the whole expression.
    ///
    /// A five-field expression searches at minute granularity in UTC and ignores the seconds
    /// and sub-second part of `now`, so a query at `09:00:00` for `0 9 * * *` yields the next
    /// day's `09:00`. A Quartz expression with a seconds field searches at second granularity.
    /// The result is never `now` itself. Returns `None` when the expression can never match
    /// (for example `0 0 31 2 *`) or when the next match lies beyond the search horizon.
    #[must_use]
    pub fn next_after(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        let horizon = self.horizon(now)?;
        let mut candidate = match self.seconds {
            Some(_) => now
                .with_nanosecond(0)?
                .checked_add_signed(Duration::seconds(1))?,
            None => now
                .with_second(0)?
                .with_nanosecond(0)?
                .checked_add_signed(Duration::minutes(1))?,
        };
        let mut steps = 0usize;
        loop {
            if candidate > horizon || steps >= MAX_SEARCH_STEPS {
                return None;
            }
            steps += 1;
            if !self.year_matches(&candidate) {
                candidate = next_year_start(candidate)?;
                continue;
            }
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
                candidate = next_minute_start(candidate)?;
                continue;
            }
            if let Some(seconds) = &self.seconds {
                if !seconds.contains(&candidate.second()) {
                    candidate = candidate.checked_add_signed(Duration::seconds(1))?;
                    continue;
                }
            }
            return Some(candidate);
        }
    }

    /// Returns up to `count` occurrences strictly after `now`, in ascending order.
    ///
    /// The list stops early when the expression can never match again, so a caller asking for
    /// more occurrences than the schedule produces gets a shorter list. Nothing is mutated, and
    /// callers clamp `count` (the `cron` tool stops at `PREVIEW_LIMIT`).
    #[must_use]
    pub fn next_occurrences(&self, now: DateTime<Utc>, count: usize) -> Vec<DateTime<Utc>> {
        let mut occurrences = Vec::with_capacity(count.min(PREVIEW_LIMIT));
        let mut cursor = now;
        for _ in 0..count {
            let Some(next) = self.next_after(cursor) else {
                break;
            };
            occurrences.push(next);
            cursor = next;
        }
        occurrences
    }

    /// The last instant [`Self::next_after`] is willing to consider.
    fn horizon(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        let base = now.checked_add_signed(Duration::days(SEARCH_HORIZON_DAYS))?;
        let Some(years) = &self.years else {
            return Some(base);
        };
        let last = i32::try_from(*years.last()?).ok()?;
        let end = NaiveDate::from_ymd_opt(last, 12, 31)?
            .and_hms_opt(23, 59, 59)?
            .and_utc();
        Some(base.max(end))
    }

    /// Whether the candidate's year is accepted.
    fn year_matches(&self, candidate: &DateTime<Utc>) -> bool {
        let Some(years) = &self.years else {
            return true;
        };
        u32::try_from(candidate.year()).is_ok_and(|year| years.contains(&year))
    }

    /// Whether `candidate`'s date satisfies the day-of-month/day-of-week pair.
    ///
    /// Both restricted means OR (standard cron); one unrestricted means it alone decides.
    /// Quartz leaves exactly one of the two unrestricted (`?`).
    fn day_matches(&self, candidate: &DateTime<Utc>) -> bool {
        let dom = self.day_of_month_matches(candidate);
        let dow = self.day_of_week_matches(candidate);
        match (self.day_of_month_restricted, self.day_of_week_restricted) {
            (false, false) => true,
            (false, true) => dow,
            (true, false) => dom,
            (true, true) => dom || dow,
        }
    }

    /// Whether the candidate's day-of-month is selected, including the `L` / `L-n` tokens.
    fn day_of_month_matches(&self, candidate: &DateTime<Utc>) -> bool {
        if self.days_of_month.contains(&candidate.day()) {
            return true;
        }
        if self.last_day_offsets.is_empty() {
            return false;
        }
        let last = days_in_month(candidate.year(), candidate.month());
        self.last_day_offsets
            .contains(&(last.saturating_sub(candidate.day())))
    }

    /// Whether the candidate's weekday is selected, including the `nL` and `d#n` tokens.
    fn day_of_week_matches(&self, candidate: &DateTime<Utc>) -> bool {
        let weekday = candidate.weekday().num_days_from_sunday();
        if self.weekdays.contains(&weekday) {
            return true;
        }
        if self
            .nth_weekdays
            .iter()
            .any(|(day, nth)| *day == weekday && nth_weekday_of_month(candidate.day()) == *nth)
        {
            return true;
        }
        if self.last_weekdays.contains(&weekday) {
            let last = days_in_month(candidate.year(), candidate.month());
            return candidate.day() + 7 > last;
        }
        false
    }
}

/// Longest occurrence list a single call may ask for.
///
/// Kept here so [`CronSchedule::next_occurrences`] can size its output; the `cron` tool owns
/// the model-facing clamp.
pub(crate) const PREVIEW_LIMIT: usize = 20;

/// Number of days in `month` of `year`, with the Gregorian leap rule.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if NaiveDate::from_ymd_opt(year, 2, 29).is_some() => 29,
        2 => 28,
        _ => 0,
    }
}

/// The 1-based ordinal of `day` within its weekday for the month (`1..=5`).
fn nth_weekday_of_month(day: u32) -> u32 {
    (day.saturating_sub(1)) / 7 + 1
}

/// Advances to the first second of the next minute.
fn next_minute_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let seconds = 60 - candidate.second();
    candidate.checked_add_signed(Duration::seconds(i64::from(seconds)))
}

/// Advances to the first second of the next hour.
fn next_hour_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let seconds = (60 - candidate.minute()) * 60 - candidate.second();
    candidate.checked_add_signed(Duration::seconds(i64::from(seconds)))
}

/// Advances to midnight of the next day.
fn next_day_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = candidate.date_naive().succ_opt()?;
    Some(next.and_hms_opt(0, 0, 0)?.and_utc())
}

/// Advances to the first second of the next month.
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

/// Advances to the first second of the next year.
fn next_year_start(candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    Some(
        NaiveDate::from_ymd_opt(candidate.year().checked_add(1)?, 1, 1)?
            .and_hms_opt(0, 0, 0)?
            .and_utc(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "schedule_tests.rs"]
mod tests;
