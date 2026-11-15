//! ISO-8601 UTC rendering of the caller-supplied report timestamp.
//!
//! [`iso8601_utc`] renders an epoch-millisecond value in Python's
//! `"YYYY-MM-DDTHH:MM:SSZ"` form: UTC, second precision, a zero-padded
//! four-digit year, and no fractional seconds.
//!
//! The conversion is implemented here from Howard Hinnant's `civil_from_days`
//! algorithm over days-since-`1970-01-01`, so the crate keeps its current
//! dependency set (no `chrono`, no `time`) and stays offline. It is exact for
//! every Gregorian leap year (`2000` is leap, `1900` is not, `2024` is leap,
//! `2100` is not), for dates before 1970 (negative epochs), and across the whole
//! `i64` millisecond range: a year that needs more than four digits renders all
//! of them (with a leading `-` for negative years) instead of overflowing.
//!
//! Sub-second milliseconds are truncated toward negative infinity, which is a
//! floor division of the millisecond value by 1000: `1_700_000_000_999` and
//! `1_700_000_000_000` both render `2023-11-14T22:13:20Z`, and `-1` (one
//! millisecond before the epoch) renders `1969-12-31T23:59:59Z`, never
//! `...:59.999`.
//!
//! Nothing here reads the clock or any environment value: the epoch
//! milliseconds are always supplied by the caller, as documented on
//! [`FunctionalReport::timestamp`].
//!
//! [`FunctionalReport::timestamp`]: super::report::FunctionalReport::timestamp

/// Milliseconds in one second.
const MILLIS_PER_SECOND: i64 = 1_000;

/// Seconds in one (UTC) day.
const SECONDS_PER_DAY: i64 = 86_400;

/// Seconds in one hour.
const SECONDS_PER_HOUR: i64 = 3_600;

/// Seconds in one minute.
const SECONDS_PER_MINUTE: i64 = 60;

/// Days from `0000-03-01` (the start of the proleptic Gregorian era clock used
/// by `civil_from_days`) to `1970-01-01`.
const EPOCH_SHIFT_IN_DAYS: i64 = 719_468;

/// Days in one 400-year Gregorian era.
const DAYS_PER_ERA: i64 = 146_097;

/// Renders `epoch_ms` as `YYYY-MM-DDTHH:MM:SSZ` in UTC.
///
/// Milliseconds below the second are truncated toward negative infinity, so the
/// rendering is the floor of `epoch_ms / 1000` seconds (see the module docs).
/// Years wider than four digits render in full, and negative years carry a
/// leading `-`; nothing panics for any `i64` input.
#[must_use]
pub fn iso8601_utc(epoch_ms: i64) -> String {
    let seconds = epoch_ms.div_euclid(MILLIS_PER_SECOND);
    let (year, month, day) = civil_from_days(seconds.div_euclid(SECONDS_PER_DAY));
    let second_of_day = seconds.rem_euclid(SECONDS_PER_DAY);
    let hour = second_of_day / SECONDS_PER_HOUR;
    let minute = (second_of_day % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let second = second_of_day % SECONDS_PER_MINUTE;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Converts a day count relative to `1970-01-01` into a civil `(year, month,
/// day)`.
///
/// This is Howard Hinnant's `civil_from_days`: shift into an era starting on
/// `0000-03-01` so the leap day lands at the end of the year, then decompose
/// with floor divisions. The arithmetic is exact over the whole `i64` day range
/// the millisecond input can produce (`|days| <= i64::MAX / 86_400_000`), so no
/// operation can overflow.
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let shifted = days_since_epoch + EPOCH_SHIFT_IN_DAYS;
    let era = shifted.div_euclid(DAYS_PER_ERA);
    let day_of_era = shifted.rem_euclid(DAYS_PER_ERA);
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
    // January and February belong to the previous civil year, because the era
    // starts in March.
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One UTC day rendered from its epoch milliseconds.
    struct Case {
        epoch_ms: i64,
        year: i64,
        month: i64,
        day: i64,
        hour: i64,
        minute: i64,
        second: i64,
        rendered: &'static str,
    }

    const CASES: [Case; 12] = [
        Case {
            epoch_ms: 0,
            year: 1970,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "1970-01-01T00:00:00Z",
        },
        Case {
            epoch_ms: 1_700_000_000_000,
            year: 2023,
            month: 11,
            day: 14,
            hour: 22,
            minute: 13,
            second: 20,
            rendered: "2023-11-14T22:13:20Z",
        },
        Case {
            epoch_ms: 1_700_000_000_999,
            year: 2023,
            month: 11,
            day: 14,
            hour: 22,
            minute: 13,
            second: 20,
            rendered: "2023-11-14T22:13:20Z",
        },
        Case {
            epoch_ms: -1,
            year: 1969,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            rendered: "1969-12-31T23:59:59Z",
        },
        Case {
            epoch_ms: -1_000,
            year: 1969,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            rendered: "1969-12-31T23:59:59Z",
        },
        Case {
            epoch_ms: 1_709_164_800_000,
            year: 2024,
            month: 2,
            day: 29,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "2024-02-29T00:00:00Z",
        },
        Case {
            epoch_ms: 951_782_400_000,
            year: 2000,
            month: 2,
            day: 29,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "2000-02-29T00:00:00Z",
        },
        Case {
            epoch_ms: 4_107_542_399_999,
            year: 2100,
            month: 2,
            day: 28,
            hour: 23,
            minute: 59,
            second: 59,
            rendered: "2100-02-28T23:59:59Z",
        },
        Case {
            epoch_ms: 4_107_542_400_000,
            year: 2100,
            month: 3,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "2100-03-01T00:00:00Z",
        },
        Case {
            epoch_ms: 1_704_067_199_999,
            year: 2023,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            rendered: "2023-12-31T23:59:59Z",
        },
        Case {
            epoch_ms: 1_704_067_200_000,
            year: 2024,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "2024-01-01T00:00:00Z",
        },
        Case {
            epoch_ms: 32_503_680_000_000,
            year: 3000,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            rendered: "3000-01-01T00:00:00Z",
        },
    ];

    /// `true` when `year` is a Gregorian leap year.
    fn is_leap_year(year: i64) -> bool {
        (year.rem_euclid(4) == 0 && year.rem_euclid(100) != 0) || year.rem_euclid(400) == 0
    }

    /// The number of days in `month` of `year`, in `1..=12`.
    fn days_in_month(year: i64, month: i64) -> i64 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if is_leap_year(year) => 29,
            2 => 28,
            _ => 0,
        }
    }

    /// Days from `1970-01-01` to `year-01-01`, counted one year at a time.
    ///
    /// This deliberately avoids the era arithmetic under test so the table's
    /// epoch values are checked against an independent day count.
    fn days_before_year(year: i64) -> i64 {
        let mut days = 0;
        if year >= 1970 {
            for y in 1970..year {
                days += if is_leap_year(y) { 366 } else { 365 };
            }
        } else {
            for y in year..1970 {
                days -= if is_leap_year(y) { 366 } else { 365 };
            }
        }
        days
    }

    /// The epoch **seconds** of one civil UTC timestamp.
    fn epoch_seconds(case: &Case) -> i64 {
        if !(1..=12).contains(&case.month)
            || case.day < 1
            || case.day > days_in_month(case.year, case.month)
        {
            return i64::MIN;
        }
        let mut days = days_before_year(case.year);
        for month in 1..case.month {
            days += days_in_month(case.year, month);
        }
        days += case.day - 1;
        days * SECONDS_PER_DAY
            + case.hour * SECONDS_PER_HOUR
            + case.minute * SECONDS_PER_MINUTE
            + case.second
    }

    #[test]
    fn renders_the_known_value_table() {
        for case in &CASES {
            assert_eq!(
                iso8601_utc(case.epoch_ms),
                case.rendered,
                "epoch_ms {}",
                case.epoch_ms
            );
        }
    }

    #[test]
    fn the_table_epochs_match_an_independent_day_count() {
        for case in &CASES {
            let seconds = epoch_seconds(case);
            assert_ne!(seconds, i64::MIN, "invalid civil date in the table");
            // Within one second of the table epoch: the table may carry
            // sub-second milliseconds on purpose.
            let delta = case.epoch_ms.div_euclid(MILLIS_PER_SECOND) - seconds;
            assert!(
                (0..=1).contains(&delta),
                "{}: independent day count says {seconds}s, table says {}ms",
                case.rendered,
                case.epoch_ms
            );
        }
    }

    #[test]
    fn sub_second_milliseconds_truncate_toward_negative_infinity() {
        assert_eq!(
            iso8601_utc(1_700_000_000_999),
            iso8601_utc(1_700_000_000_000)
        );
        assert_eq!(iso8601_utc(-1), "1969-12-31T23:59:59Z");
        assert_eq!(iso8601_utc(-999), "1969-12-31T23:59:59Z");
        assert_eq!(iso8601_utc(-1_000), "1969-12-31T23:59:59Z");
        assert_eq!(iso8601_utc(-1_001), "1969-12-31T23:59:58Z");
        assert_eq!(iso8601_utc(-2_000), "1969-12-31T23:59:58Z");
    }

    #[test]
    fn century_leap_rules_are_exact() {
        // 2000 is a leap year (divisible by 400): the leap day exists.
        assert_eq!(iso8601_utc(951_782_400_000), "2000-02-29T00:00:00Z");
        // 1900 is not a leap year: the day after 1900-02-28 is 1900-03-01.
        assert_eq!(iso8601_utc(-2_203_891_201_000), "1900-02-28T23:59:59Z");
        assert_eq!(iso8601_utc(-2_203_891_200_000), "1900-03-01T00:00:00Z");
        // 2024 is a leap year.
        assert_eq!(iso8601_utc(1_709_164_800_000), "2024-02-29T00:00:00Z");
        // 2100 is not a leap year: 2100-02-28 is the last day of the month.
        assert_eq!(iso8601_utc(4_107_542_399_999), "2100-02-28T23:59:59Z");
        assert_eq!(iso8601_utc(4_107_542_400_000), "2100-03-01T00:00:00Z");
    }

    #[test]
    fn dates_before_1960_and_the_epoch_are_exact() {
        assert_eq!(iso8601_utc(-315_619_200_000), "1960-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1_234_567_890_123), "2009-02-13T23:31:30Z");
    }

    #[test]
    fn extreme_years_render_in_full_without_panicking() {
        // Cross-checked against an independent `civil_from_days`
        // implementation (Python) over the whole `i64` range.
        assert_eq!(iso8601_utc(i64::MAX), "292278994-08-17T07:12:55Z");
        assert_eq!(iso8601_utc(i64::MIN), "-292275055-05-16T16:47:04Z");
    }

    #[test]
    fn rendered_values_always_keep_the_wire_shape() {
        for epoch_ms in [i64::MIN, -62_135_596_800_000, -1, 0, 1, i64::MAX] {
            let rendered = iso8601_utc(epoch_ms);
            assert!(rendered.ends_with('Z'), "{rendered}");
            assert_eq!(rendered.matches('T').count(), 1, "{rendered}");
            assert_eq!(rendered.matches('.').count(), 0, "{rendered}");
            let (date, time) = rendered.split_once('T').expect("timestamp has a T");
            assert!(date.len() >= 8, "{rendered}");
            assert_eq!(time.len(), 9, "{rendered}");
        }
    }

    #[test]
    fn every_day_of_a_leap_year_round_trips() {
        // 2024-01-01T00:00:00Z through 2024-12-31T23:59:59Z, one day at a time.
        let start = 1_704_067_200_i64;
        let mut expected_day = 1;
        let mut expected_month = 1;
        for offset_days in 0..366 {
            let epoch_ms = (start + offset_days * SECONDS_PER_DAY) * MILLIS_PER_SECOND;
            let rendered = iso8601_utc(epoch_ms);
            let expected = format!("2024-{expected_month:02}-{expected_day:02}T00:00:00Z");
            assert_eq!(rendered, expected, "day offset {offset_days}");
            expected_day += 1;
            if expected_day > days_in_month(2024, expected_month) {
                expected_day = 1;
                expected_month += 1;
            }
        }
        assert_eq!(expected_month, 13);
    }
}
