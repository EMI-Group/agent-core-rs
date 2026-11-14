//! Unit tests of the cron expression engine: both grammars, their rejections and `next_after`.
//!
//! Kept beside `schedule.rs` (through a `#[path]` module) because the assertions inspect the
//! parsed field sets directly, which are private to the `schedule` module.

use chrono::TimeZone;

use super::*;

/// Builds a UTC instant at the given wall-clock minute.
fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .unwrap()
}

/// Builds a UTC instant at the given wall-clock second.
fn at_second(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
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
    assert!(schedule.seconds.is_none());
    assert!(!schedule.has_seconds());
    assert!(schedule.years.is_none());
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
fn parses_month_names_in_both_forms() {
    let classic = CronSchedule::parse("0 0 1 JAN-MAR *").unwrap();
    assert_eq!(classic.months, vec![1, 2, 3]);

    let names = CronSchedule::parse("0 0 0 1 jan,Feb,DEC ?").unwrap();
    assert_eq!(names.months, vec![1, 2, 12]);

    assert_eq!(parse_error("0 0 * FOO *"), "invalid month field: \"FOO\"");
}

#[test]
fn rejects_wrong_field_count() {
    let expected = |count: usize, text: &str| {
        format!(
            "invalid cron expression: expected 5 fields (minute hour day-of-month month \
             day-of-week) or 6/7 Quartz fields (second minute hour day-of-month month \
             day-of-week [year]), got {count}: {text:?}"
        )
    };
    assert_eq!(
        parse_error("* * * *"),
        expected(4, "* * * *"),
        "the message names both accepted layouts"
    );
    assert_eq!(
        parse_error("0 0 0 1 1 1 1 1"),
        expected(8, "0 0 0 1 1 1 1 1")
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
    // The classic form keeps Vixie's "start must not exceed end" rule.
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

// --- Quartz grammar --------------------------------------------------------

#[test]
fn parses_six_field_quartz_expression() {
    let schedule = CronSchedule::parse("*/15 0 9 ? * MON-FRI").unwrap();
    assert_eq!(schedule.expression(), "*/15 0 9 ? * MON-FRI");
    assert_eq!(schedule.seconds, Some(vec![0, 15, 30, 45]));
    assert!(schedule.has_seconds());
    assert_eq!(schedule.minutes, vec![0]);
    assert_eq!(schedule.hours, vec![9]);
    assert!(schedule.days_of_month.is_empty());
    assert!(!schedule.day_of_month_restricted, "? never restricts");
    assert_eq!(schedule.weekdays, vec![1, 2, 3, 4, 5]);
    assert!(schedule.day_of_week_restricted);
    assert!(schedule.years.is_none());
}

#[test]
fn parses_seven_field_quartz_expression_with_a_year() {
    let schedule = CronSchedule::parse("0 30 9 1 * ? 2026,2027").unwrap();
    assert_eq!(schedule.years, Some(vec![2026, 2027]));
    assert_eq!(schedule.seconds, Some(vec![0]));
    assert_eq!(schedule.days_of_month, vec![1]);
    assert!(!schedule.day_of_week_restricted);

    let star = CronSchedule::parse("0 0 0 ? * * 2024-2099").unwrap();
    assert_eq!(star.years.as_ref().map(Vec::len), Some(76));
}

#[test]
fn quartz_requires_exactly_one_question_mark() {
    assert_eq!(
        parse_error("0 0 12 1 * MON"),
        "invalid cron expression: both day-of-month and day-of-week are specified \
         (one must be \"?\" in a Quartz expression): \"0 0 12 1 * MON\""
    );
    assert_eq!(
        parse_error("0 0 12 ? * ?"),
        "invalid cron expression: both day-of-month and day-of-week are \"?\" \
         (exactly one must be specified in a Quartz expression): \"0 0 12 ? * ?\""
    );
}

#[test]
fn question_mark_selects_the_other_day_field() {
    // Only the day-of-week field decides.
    let weekly = CronSchedule::parse("0 0 12 ? * MON").unwrap();
    assert_eq!(
        weekly.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 1, 12, 0))
    );
    assert_eq!(
        weekly.next_after(at(2024, 1, 1, 12, 0)),
        Some(at(2024, 1, 8, 12, 0))
    );

    // Only the day-of-month field decides.
    let monthly = CronSchedule::parse("0 0 12 15 * ?").unwrap();
    assert_eq!(
        monthly.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 15, 12, 0))
    );

    // Both wildcards: every day.
    let daily = CronSchedule::parse("0 0 12 * * ?").unwrap();
    assert_eq!(
        daily.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 1, 12, 0))
    );
}

#[test]
fn quartz_searches_at_second_granularity() {
    let every_ten_seconds = CronSchedule::parse("*/10 * * * * ?").unwrap();
    assert_eq!(
        every_ten_seconds.next_after(at_second(2024, 1, 1, 0, 0, 5)),
        Some(at_second(2024, 1, 1, 0, 0, 10))
    );
    assert_eq!(
        every_ten_seconds.next_after(at_second(2024, 1, 1, 0, 0, 10)),
        Some(at_second(2024, 1, 1, 0, 0, 20)),
        "the result is strictly after the input"
    );
    assert_eq!(
        every_ten_seconds.next_after(at_second(2024, 1, 1, 0, 0, 58)),
        Some(at_second(2024, 1, 1, 0, 1, 0)),
        "the scan crosses the minute boundary"
    );

    let once_a_minute = CronSchedule::parse("0 * * * * ?").unwrap();
    assert_eq!(
        once_a_minute.next_after(at_second(2024, 1, 31, 23, 59, 30)),
        Some(at_second(2024, 2, 1, 0, 0, 0))
    );
}

#[test]
fn quartz_rejects_a_bad_seconds_or_year_field() {
    assert_eq!(parse_error("60 * * * * ?"), "invalid second field: \"60\"");
    assert_eq!(
        parse_error("0 0 0 ? * * 1969"),
        "invalid year field: \"1969\""
    );
    assert_eq!(
        parse_error("0 0 0 ? * * 2100"),
        "invalid year field: \"2100\""
    );
}

#[test]
fn quartz_accepts_wrap_around_ranges() {
    let hours = CronSchedule::parse("0 0 22-2 * * ?").unwrap();
    assert_eq!(hours.hours, vec![0, 1, 2, 22, 23]);

    let weekdays = CronSchedule::parse("0 0 0 ? * FRI-MON").unwrap();
    assert_eq!(weekdays.weekdays, vec![0, 1, 5, 6]);

    let stepped = CronSchedule::parse("0 0 22-2/2 * * ?").unwrap();
    assert_eq!(
        stepped.hours,
        vec![0, 2, 22],
        "the step runs along the wrap"
    );

    // 2024-01-05 is a Friday; the next wrap match is Saturday the 6th.
    let weekend = CronSchedule::parse("0 0 0 ? * SAT-SUN").unwrap();
    assert_eq!(
        weekend.next_after(at(2024, 1, 5, 0, 0)),
        Some(at(2024, 1, 6, 0, 0))
    );
}

#[test]
fn quartz_last_day_of_the_month() {
    let last_day = CronSchedule::parse("0 0 0 L * ?").unwrap();
    assert_eq!(
        last_day.next_after(at(2024, 1, 15, 0, 0)),
        Some(at(2024, 1, 31, 0, 0))
    );
    assert_eq!(
        last_day.next_after(at(2024, 1, 31, 0, 0)),
        Some(at(2024, 2, 29, 0, 0)),
        "a leap February ends on the 29th"
    );
    assert_eq!(
        last_day.next_after(at(2023, 2, 1, 0, 0)),
        Some(at(2023, 2, 28, 0, 0))
    );

    let offset = CronSchedule::parse("0 0 0 L-3 * ?").unwrap();
    assert_eq!(
        offset.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 28, 0, 0))
    );
}

#[test]
fn quartz_last_weekday_of_the_month() {
    // 2024-01-26 is the last Friday of January 2024.
    let last_friday = CronSchedule::parse("0 0 0 ? * 5L").unwrap();
    assert_eq!(
        last_friday.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 26, 0, 0))
    );

    // A bare `L` in the day-of-week field is Saturday, the last day of the week.
    let last_saturday = CronSchedule::parse("0 0 0 ? * L").unwrap();
    assert_eq!(
        last_saturday.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 27, 0, 0))
    );
}

#[test]
fn quartz_nth_weekday_of_the_month() {
    // The first Monday of January 2024 is the 1st, the third Saturday the 20th.
    let first_monday = CronSchedule::parse("0 0 0 ? * 1#1").unwrap();
    assert_eq!(
        first_monday.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 2, 5, 0, 0)),
        "strictly after the first occurrence"
    );
    assert_eq!(
        first_monday.next_after(at(2023, 12, 31, 0, 0)),
        Some(at(2024, 1, 1, 0, 0))
    );

    let third_saturday = CronSchedule::parse("0 0 0 ? * SAT#3").unwrap();
    assert_eq!(
        third_saturday.next_after(at(2024, 1, 1, 0, 0)),
        Some(at(2024, 1, 20, 0, 0))
    );
}

#[test]
fn quartz_rejects_an_out_of_range_nth_weekday() {
    assert_eq!(
        parse_error("0 0 0 ? * 1#6"),
        "invalid day-of-week field: \"1#6\""
    );
    assert_eq!(
        parse_error("0 0 0 ? * 1#0"),
        "invalid day-of-week field: \"1#0\""
    );
    assert_eq!(
        parse_error("0 0 0 ? * 1#x"),
        "invalid day-of-week field: \"1#x\""
    );
}

#[test]
fn rejects_the_nearest_weekday_token() {
    assert_eq!(
        parse_error("0 0 0 15W * ?"),
        "unsupported cron token: \"15W\": \"W\" (nearest weekday) is not supported"
    );
    assert_eq!(
        parse_error("0 0 0 LW * ?"),
        "unsupported cron token: \"LW\": \"W\" (nearest weekday) is not supported"
    );
}

#[test]
fn rejects_a_question_mark_inside_a_list() {
    assert_eq!(
        parse_error("0 0 12 ?,1 * ?"),
        "unsupported cron token: \"?,1\": \"?\" (must be the whole field) is not supported"
    );
    assert_eq!(
        parse_error("0 0 ? * *"),
        "unsupported cron token: \"?\": \"?\" (must be the whole field) is not supported"
    );
}

#[test]
fn rejects_quartz_tokens_in_the_classic_form() {
    assert_eq!(
        parse_error("0 0 L * *"),
        "invalid day-of-month field: \"L\""
    );
    assert_eq!(
        parse_error("0 0 * * 5#3"),
        "invalid day-of-week field: \"5#3\""
    );
    assert_eq!(parse_error("0 22-2 * * *"), "invalid hour field: \"22-2\"");
}

#[test]
fn rejects_an_out_of_range_last_day_offset() {
    assert_eq!(
        parse_error("0 0 0 L-31 * ?"),
        "invalid day-of-month field: \"L-31\""
    );
    assert_eq!(
        parse_error("0 0 0 L+1 * ?"),
        "invalid day-of-month field: \"L+1\""
    );
}

#[test]
fn the_year_field_bounds_the_search() {
    let once = CronSchedule::parse("0 0 12 1 1 ? 2030").unwrap();
    assert_eq!(
        once.next_after(at(2024, 6, 1, 0, 0)),
        Some(at(2030, 1, 1, 12, 0)),
        "the horizon stretches to the named year"
    );
    assert_eq!(
        once.next_after(at(2030, 1, 1, 12, 0)),
        None,
        "no later occurrence exists"
    );
}

#[test]
fn quartz_reports_impossible_expressions() {
    assert_eq!(
        CronSchedule::parse("0 0 0 31 2 ?")
            .unwrap()
            .next_after(at(2024, 1, 1, 0, 0)),
        None
    );
    assert_eq!(
        CronSchedule::parse("0 0 0 30 2 ?")
            .unwrap()
            .next_after(at(2024, 1, 1, 0, 0)),
        None
    );
}

#[test]
fn next_occurrences_lists_a_bounded_window() {
    let schedule = CronSchedule::parse("0 0 12 * * ?").unwrap();
    assert_eq!(
        schedule.next_occurrences(at(2024, 1, 1, 0, 0), 3),
        vec![
            at(2024, 1, 1, 12, 0),
            at(2024, 1, 2, 12, 0),
            at(2024, 1, 3, 12, 0)
        ]
    );
    assert!(schedule
        .next_occurrences(at(2024, 1, 1, 0, 0), 0)
        .is_empty());
    assert_eq!(
        CronSchedule::parse("0 0 0 31 2 ?")
            .unwrap()
            .next_occurrences(at(2024, 1, 1, 0, 0), 5),
        Vec::<DateTime<Utc>>::new(),
        "an impossible schedule yields no occurrences"
    );
}
