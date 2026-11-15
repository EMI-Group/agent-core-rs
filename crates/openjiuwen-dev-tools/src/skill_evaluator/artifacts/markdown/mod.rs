//! Human-readable Markdown renderings of an [`EvaluationReport`].
//!
//! The Python evaluation skills write four prose reports beside the JSON
//! artifacts; the renderers here produce the same documents as deterministic
//! strings, so a caller can persist them (see
//! [`EvaluationReport::write_markdown_artifacts`]) or print them.
//!
//! | Function | Document | Headings |
//! |---|---|---|
//! | [`llm_judge_report`] | `llm_judge_report.md` | `# Skill Evaluation Report: <skill>`, `## Summary`, `## Dimension Scores`, `## Critical Issues`, `## Detailed Analysis` |
//! | [`skill_safety_report`] | `skill_safety_report.md` | `# Safety Evaluation Report: <skill>`, `## Verdict`, `## Dimension Scores`, `## Findings`, `## Cleared Dimensions` |
//! | [`final_report`] | `final_report.md` | `# Eval Pipeline Report: <skill>`, `## Final Verdict`, `## Evaluator Breakdown`, `## Key Issues to Fix`, `## Full Reports` |
//! | [`pipeline_report`] | `pipeline_report.md` | the same four pipeline sections for a blocked run |
//!
//! The headings are the Python contract and are reproduced verbatim; no extra
//! section is invented. Every bullet label (`**Total Score**`, `**Gate**`,
//! `**Ship-ready**`, `**Run timestamp**`, …) is likewise Python's.
//!
//! # Units and formatting
//!
//! Points (0–100) stay points where Python shows points — the design
//! `**Total Score**`, the safety `**Score**`, and every dimension `Score`/`Max`
//! cell. Fractions become percentages in the tables and verdicts
//! (the `format_percent` helper, e.g. `0.875` reads `87.5%`) and stay decimals for the
//! weighted pipeline score (`0.818`) and the evaluator contributions (`0.296`).
//! Numbers are trimmed of trailing zeros, so whole points read `7`, not `7.00`.
//!
//! # Determinism
//!
//! Rendering is a pure function of the report: no I/O, no `async`, no clock and
//! no environment reads. The one date in the documents is
//! `FunctionalReport::timestamp_iso`, which renders the caller-supplied
//! `timestamp` field. Lines never carry trailing whitespace, sections are
//! separated by exactly one blank line, and every document ends with a single
//! newline.
//!
//! # Deviations from Python
//!
//! The Python design summary also carries a `**Pattern**` line describing the
//! skill's instruction pattern; [`DesignReport`] has no pattern field, so that
//! line is omitted rather than guessed.
//!
//! A dimension the report does not carry (a partially filled `dimensions` map)
//! renders `-` in its `Score` cell instead of being invented, and the canonical
//! rubric maximum is shown in `Max`. On an auto-fail the safety report carries
//! no dimension scores at all, so its whole table renders `-`.

#[cfg(test)]
mod tests;

use std::fmt::Write as _;

use crate::skill_evaluator::report::{
    grade_for, DesignReport, EvaluationReport, SafetyReport, PIPELINE_BLOCKED_AT_E1,
    PIPELINE_BLOCKED_AT_E3, SAFETY_DIMENSIONS,
};

use super::{DESIGN_ARTIFACT, FINAL_SCORE_ARTIFACT, FUNCTIONAL_ARTIFACT, SAFETY_ARTIFACT};

/// The seven design rubric dimensions in report order: id, humanised label, and
/// the points available to the dimension.
///
/// The maxima mirror the rubric of
/// [`score_design`](crate::skill_evaluator::score_design) (10 + 25 + 15 + 10 +
/// 15 + 15 + 10 = 100) and are only used for a dimension the report omitted.
const DESIGN_DIMENSIONS: [(&str, &str, f64); 7] = [
    ("d1_knowledge_ratio", "D1: Knowledge Ratio", 10.0),
    (
        "d2_expert_knowledge_craft",
        "D2: Expert Knowledge Craft",
        25.0,
    ),
    ("d3_spec_compliance", "D3: Specification Compliance", 15.0),
    (
        "d4_progressive_disclosure",
        "D4: Progressive Disclosure",
        10.0,
    ),
    ("d5_pattern_freedom_fit", "D5: Pattern + Freedom Fit", 15.0),
    ("d6_practical_usability", "D6: Practical Usability", 15.0),
    ("d7_output_specification", "D7: Output Specification", 10.0),
];

/// The humanised labels of the five safety dimensions, in
/// [`SAFETY_DIMENSIONS`] order.
const SAFETY_LABELS: [&str; 5] = [
    "D1: Harmful Instruction Potential",
    "D2: Scope Containment",
    "D3: Data Handling Safety",
    "D4: Injection Resilience",
    "D5: Guardrail Presence",
];

/// The release-ready threshold of the pipeline verdict (`final_score_pct`).
const SHIP_READY_THRESHOLD: f64 = 0.80;

/// Renders the design review report (`llm_judge_report.md`).
///
/// Sections: the `# Skill Evaluation Report: <skill>` title; `## Summary` with
/// `**Total Score**`, `**Grade**`, `**Knowledge Delta**`, and
/// `**Ready for functional testing**`; `## Dimension Scores` with the seven
/// humanised design dimensions; `## Critical Issues` (a closed design gate and
/// every dimension awarded no points); and `## Detailed Analysis` (every
/// dimension below 80% of its maximum). An empty section reads `None.`
#[must_use]
pub fn llm_judge_report(design: &DesignReport) -> String {
    let mut out = String::new();

    let name = design.skill_name.trim();
    push_title(&mut out, "# Skill Evaluation Report:", name);
    blank(&mut out);

    let _ = writeln!(out, "## Summary");
    let total = format_points(design.score_points());
    let _ = writeln!(out, "- **Total Score**: {total}/100");
    let grade = grade_text(&design.grade, design.score_fraction());
    let _ = writeln!(out, "- **Grade**: {grade}");
    let _ = writeln!(
        out,
        "- **Knowledge Delta**: {}",
        design.knowledge_delta_label()
    );
    let readiness = yes_no(design.ready_for_functional_testing);
    let _ = writeln!(out, "- **Ready for functional testing**: {readiness}");
    blank(&mut out);

    let _ = writeln!(out, "## Dimension Scores");
    let rows = DESIGN_DIMENSIONS
        .iter()
        .map(|&(id, label, max)| design_row(design, id, label, max))
        .collect::<Vec<_>>();
    push_table(&mut out, &["Dimension", "Score", "Max"], &rows);
    blank(&mut out);

    let _ = writeln!(out, "## Critical Issues");
    push_bullets(&mut out, &critical_issues(design));
    blank(&mut out);

    let _ = writeln!(out, "## Detailed Analysis");
    push_bullets(&mut out, &detailed_analysis(design));

    out
}

/// Renders the safety review report (`skill_safety_report.md`).
///
/// Sections: the `# Safety Evaluation Report: <skill>` title; `## Verdict` with
/// `**Score**`, `**Gate**`, `**Blocked**`, and `**Auto-fail triggered**`;
/// `## Dimension Scores` with the five canonical safety dimensions;
/// `## Findings` (one bullet per finding, `None.` when clean); and
/// `## Cleared Dimensions` (the dimensions that kept their full maximum).
#[must_use]
pub fn skill_safety_report(safety: &SafetyReport) -> String {
    let mut out = String::new();

    let name = safety.skill_name.trim();
    push_title(&mut out, "# Safety Evaluation Report:", name);
    blank(&mut out);

    let _ = writeln!(out, "## Verdict");
    let points = format_points(safety.score_fraction() * 100.0);
    let percent = format_percent(safety.score_fraction());
    let _ = writeln!(out, "- **Score**: {points}/100 ({percent})");
    let gate = if safety.passes_gate() { "PASS" } else { "FAIL" };
    let _ = writeln!(out, "- **Gate**: {gate}");
    let _ = writeln!(out, "- **Blocked**: {}", yes_no(safety.blocked));
    let auto_fail = safety
        .auto_fail
        .map_or_else(|| "None".to_owned(), |code| code.to_string());
    let _ = writeln!(out, "- **Auto-fail triggered**: {auto_fail}");
    blank(&mut out);

    let _ = writeln!(out, "## Dimension Scores");
    let rows = SAFETY_DIMENSIONS
        .iter()
        .zip(SAFETY_LABELS)
        .map(|(&(id, max, _), label)| safety_row(safety, id, label, max))
        .collect::<Vec<_>>();
    push_table(&mut out, &["Dimension", "Score", "Max"], &rows);
    blank(&mut out);

    let _ = writeln!(out, "## Findings");
    push_bullets(&mut out, &finding_bullets(safety));
    blank(&mut out);

    let _ = writeln!(out, "## Cleared Dimensions");
    push_bullets(&mut out, &cleared_dimensions(safety));

    out
}

/// Renders the pipeline verdict of a completed run (`final_report.md`).
///
/// Sections: the `# Eval Pipeline Report: <skill>` title; `## Final Verdict`
/// with `**Score**`, `**Status**`, and `**Ship-ready**`; `## Evaluator
/// Breakdown` with the E1 design and E2 functional weighted rows plus the E3
/// safety gate row; `## Key Issues to Fix`; `## Full Reports` linking the four
/// JSON artifacts; and the `**Run timestamp**` line.
///
/// The document is only meaningful for a completed pipeline
/// ([`FinalScore::is_completed`](crate::skill_evaluator::FinalScore::is_completed));
/// a blocked run belongs in [`pipeline_report`]. A caller that renders this for
/// a blocked report still gets a well-formed document whose status names the
/// blocking gate.
#[must_use]
pub fn final_report(report: &EvaluationReport) -> String {
    let mut out = String::new();

    let name = report_name(report);
    push_title(&mut out, "# Eval Pipeline Report:", name);
    blank(&mut out);

    let _ = writeln!(out, "## Final Verdict");
    let score = format_fraction(report.final_score.score_fraction());
    let grade = grade_text(
        &report.final_score.grade,
        report.final_score.score_fraction(),
    );
    let _ = writeln!(out, "- **Score**: {score} ({grade})");
    let status = or_placeholder(report.final_score.pipeline_status.trim());
    let _ = writeln!(out, "- **Status**: {status}");
    let _ = writeln!(out, "- {}", ship_ready_line(report));
    blank(&mut out);

    let _ = writeln!(out, "## Evaluator Breakdown");
    push_table(
        &mut out,
        &["Evaluator", "Score", "Weight", "Contribution"],
        &evaluator_rows(report),
    );
    blank(&mut out);

    let _ = writeln!(out, "## Key Issues to Fix");
    push_bullets(&mut out, &key_issues(report));
    blank(&mut out);

    let _ = writeln!(out, "## Full Reports");
    push_bullets(&mut out, &full_reports(report.final_score.is_completed()));
    blank(&mut out);

    let _ = writeln!(
        out,
        "**Run timestamp**: {}",
        report.functional.timestamp_iso()
    );

    out
}

/// Renders the pipeline verdict of a blocked run (`pipeline_report.md`).
///
/// Sections: the `# Eval Pipeline Report: <skill>` title; `## Final Verdict`
/// with `**Status**`, `**Blocked by**` (the gate that closed), and the
/// `**Design score**`/`**Safety score**` evidence bullets; `## Key Issues to
/// Fix` naming the blocking gate first; `## Full Reports` linking the three JSON
/// artifacts a blocked run writes; and the `**Run timestamp**` line.
///
/// The document is only meaningful for a blocked pipeline; a completed run
/// belongs in [`final_report`]. A caller that renders this for a completed
/// report still gets a well-formed document whose `**Blocked by**` line falls
/// back to `pipeline gate`.
#[must_use]
pub fn pipeline_report(report: &EvaluationReport) -> String {
    let mut out = String::new();

    let name = report_name(report);
    push_title(&mut out, "# Eval Pipeline Report:", name);
    blank(&mut out);

    let _ = writeln!(out, "## Final Verdict");
    let status = or_placeholder(report.final_score.pipeline_status.trim());
    let _ = writeln!(out, "- **Status**: {status}");
    let _ = writeln!(out, "- **Blocked by**: {}", gate_description(status));
    let design_points = format_points(report.design.score_points());
    let design_percent = format_percent(report.design.score_fraction());
    let _ = writeln!(
        out,
        "- **Design score**: {design_points}/100 ({design_percent})"
    );
    let safety_points = format_points(report.safety.score_fraction() * 100.0);
    let safety_percent = format_percent(report.safety.score_fraction());
    let gate = if report.safety.passes_gate() {
        "PASS"
    } else {
        "FAIL"
    };
    let _ = writeln!(
        out,
        "- **Safety score**: {safety_points}/100 ({safety_percent}), gate {gate}"
    );
    blank(&mut out);

    let _ = writeln!(out, "## Key Issues to Fix");
    push_bullets(&mut out, &key_issues(report));
    blank(&mut out);

    let _ = writeln!(out, "## Full Reports");
    push_bullets(&mut out, &full_reports(false));
    blank(&mut out);

    let _ = writeln!(
        out,
        "**Run timestamp**: {}",
        report.functional.timestamp_iso()
    );

    out
}

// ---------------------------------------------------------------------------
// Section builders
// ---------------------------------------------------------------------------

/// Appends a document title, leaving no trailing space when the name is blank.
fn push_title(out: &mut String, prefix: &str, name: &str) {
    let name = name.trim();
    if name.is_empty() {
        let _ = writeln!(out, "{prefix}");
    } else {
        let _ = writeln!(out, "{prefix} {name}");
    }
}

/// Appends a blank line separating two sections.
fn blank(out: &mut String) {
    out.push('\n');
}

/// Appends a Markdown table with `headers` and `rows`.
fn push_table(out: &mut String, headers: &[&str], rows: &[Vec<String>]) {
    let _ = writeln!(out, "| {} |", headers.join(" | "));
    let separators = headers.iter().map(|_| "---").collect::<Vec<_>>();
    let _ = writeln!(out, "| {} |", separators.join(" | "));
    for row in rows {
        let _ = writeln!(out, "| {} |", row.join(" | "));
    }
}

/// Appends `bullets`, or the Python empty-state line `None.` when there are none.
fn push_bullets(out: &mut String, bullets: &[String]) {
    if bullets.is_empty() {
        let _ = writeln!(out, "None.");
        return;
    }
    for bullet in bullets {
        let _ = writeln!(out, "{bullet}");
    }
}

/// One `| Dimension | Score | Max |` row of the design rubric.
fn design_row(design: &DesignReport, id: &str, label: &str, canonical_max: f64) -> Vec<String> {
    let dimension = design.dimensions.get(id);
    let score = dimension.map_or_else(|| "-".to_owned(), |value| format_points(value.score));
    let max = dimension.map_or_else(
        || format_points(canonical_max),
        |value| format_points(value.max),
    );
    vec![label.to_owned(), score, max]
}

/// One `| Dimension | Score | Max |` row of the safety rubric.
fn safety_row(safety: &SafetyReport, id: &str, label: &str, canonical_max: f64) -> Vec<String> {
    let dimension = safety
        .dimensions
        .as_ref()
        .and_then(|dimensions| dimensions.get(id));
    let score = dimension.map_or_else(|| "-".to_owned(), |value| format_points(value.score));
    let max = dimension.map_or_else(
        || format_points(canonical_max),
        |value| format_points(value.max),
    );
    vec![label.to_owned(), score, max]
}

/// The design dimensions that lost more than 20% of their points, as
/// `(label, score, max)` in rubric order.
fn weak_design_dimensions(design: &DesignReport) -> Vec<(&'static str, f64, f64)> {
    let mut weak = Vec::new();
    for (id, label, canonical_max) in DESIGN_DIMENSIONS {
        let Some(dimension) = design.dimensions.get(id) else {
            continue;
        };
        let max = if dimension.max > 0.0 {
            dimension.max
        } else {
            canonical_max
        };
        // `score < 0.8 * max`, cross-multiplied so the 80% boundary is exact.
        if dimension.score * 5.0 >= max * 4.0 {
            continue;
        }
        weak.push((label, dimension.score, max));
    }
    weak
}

/// The `## Critical Issues` bullets of the design report.
fn critical_issues(design: &DesignReport) -> Vec<String> {
    let mut issues = Vec::new();
    if !design.ready_for_functional_testing {
        let points = format_points(design.score_points());
        issues.push(format!(
            "- Design gate closed: {points} of 100 points is below the 60-point readiness threshold."
        ));
    }
    for (id, label, _) in DESIGN_DIMENSIONS {
        let Some(dimension) = design.dimensions.get(id) else {
            continue;
        };
        if dimension.score > 0.0 {
            continue;
        }
        let max = format_points(dimension.max);
        issues.push(format!("- {label}: 0/{max} points — no rubric credit."));
    }
    issues
}

/// The `## Detailed Analysis` bullets of the design report.
fn detailed_analysis(design: &DesignReport) -> Vec<String> {
    weak_design_dimensions(design)
        .into_iter()
        .map(|(label, score, max)| {
            let points = format_points(score);
            let points_max = format_points(max);
            let percent = format_ratio_percent(score, max);
            format!("- {label}: {points}/{points_max} points ({percent}).")
        })
        .collect()
}

/// The `## Findings` bullets of the safety report.
fn finding_bullets(safety: &SafetyReport) -> Vec<String> {
    safety
        .findings
        .iter()
        .map(|finding| {
            let label = safety_label(&finding.dimension)
                .unwrap_or(finding.dimension.as_str())
                .to_owned();
            let severity = finding.severity.trim();
            let message = finding.message.trim();
            format!("- {label} ({severity}): {message}")
        })
        .collect()
}

/// The `## Cleared Dimensions` bullets of the safety report.
fn cleared_dimensions(safety: &SafetyReport) -> Vec<String> {
    let Some(dimensions) = safety.dimensions.as_ref() else {
        return Vec::new();
    };
    let mut cleared = Vec::new();
    for (&(id, canonical_max, _), label) in SAFETY_DIMENSIONS.iter().zip(SAFETY_LABELS) {
        let Some(dimension) = dimensions.get(id) else {
            continue;
        };
        let max = if dimension.max > 0.0 {
            dimension.max
        } else {
            canonical_max
        };
        if dimension.score < max {
            continue;
        }
        let score = format_points(dimension.score);
        let max = format_points(max);
        cleared.push(format!("- {label} ({score}/{max})"));
    }
    cleared
}

/// The `## Evaluator Breakdown` rows.
fn evaluator_rows(report: &EvaluationReport) -> Vec<Vec<String>> {
    let evaluators = &report.final_score.evaluators;
    let design_row = evaluators.e1_llm_judge.as_ref().map_or_else(
        || vec!["-".to_owned(); 3],
        |score| {
            vec![
                format_percent(score.score_pct),
                format_weight(score.weight),
                format_fraction(score.contribution),
            ]
        },
    );
    let functional = &evaluators.e2_functional;
    let functional_row = vec![
        format_percent(functional.score_pct),
        format_weight(functional.weight),
        format_fraction(functional.contribution),
    ];
    let gate = if report.safety.passes_gate() {
        "PASS"
    } else {
        "FAIL"
    };
    vec![
        std::iter::once("E1 Design (skill_llm_judge)".to_owned())
            .chain(design_row)
            .collect(),
        std::iter::once("E2 Functional (skill_tester)".to_owned())
            .chain(functional_row)
            .collect(),
        vec![
            "E3 Safety (skill_safety_judge)".to_owned(),
            gate.to_owned(),
            "gate".to_owned(),
            "-".to_owned(),
        ],
    ]
}

/// The `## Key Issues to Fix` bullets, identical for both pipeline reports.
///
/// A blocked run leads with the gate that closed it, then lists the functional
/// recommendations, the design dimensions below 80% of their maximum, and every
/// safety finding.
fn key_issues(report: &EvaluationReport) -> Vec<String> {
    let mut issues = Vec::new();
    let status = or_placeholder(report.final_score.pipeline_status.trim());
    if !report.final_score.is_completed() {
        issues.push(format!(
            "- Gate: stopped at {status} — {}.",
            gate_description(status)
        ));
    }
    for recommendation in &report.functional.recommendations {
        let recommendation = recommendation.trim();
        if recommendation.is_empty() {
            continue;
        }
        issues.push(format!("- Functional: {recommendation}"));
    }
    for (label, score, max) in weak_design_dimensions(&report.design) {
        let points = format_points(score);
        let points_max = format_points(max);
        let percent = format_ratio_percent(score, max);
        issues.push(format!(
            "- Design: {label} — {points}/{points_max} points ({percent})"
        ));
    }
    for finding in &report.safety.findings {
        let label = safety_label(&finding.dimension).unwrap_or(finding.dimension.as_str());
        let severity = finding.severity.trim();
        let message = finding.message.trim();
        issues.push(format!("- Safety: {label} ({severity}): {message}"));
    }
    issues
}

/// The `## Full Reports` links, in the order the JSON writers produce them.
fn full_reports(completed: bool) -> Vec<String> {
    let mut files = vec![DESIGN_ARTIFACT, SAFETY_ARTIFACT, FUNCTIONAL_ARTIFACT];
    if completed {
        files.push(FINAL_SCORE_ARTIFACT);
    }
    files
        .into_iter()
        .map(|file| format!("- [{file}]({file})"))
        .collect()
}

/// The `**Ship-ready**` bullet of a completed run.
fn ship_ready_line(report: &EvaluationReport) -> String {
    let ready = report.final_score.score_fraction() >= SHIP_READY_THRESHOLD;
    let verdict = yes_no(ready);
    let recommendation = report.final_score.recommendation.trim();
    if recommendation.is_empty() {
        format!("**Ship-ready**: {verdict}")
    } else {
        format!("**Ship-ready**: {verdict} — {recommendation}")
    }
}

/// Python's prose for the gate a blocked pipeline stopped at.
fn gate_description(status: &str) -> &'static str {
    match status {
        PIPELINE_BLOCKED_AT_E1 => "E1 design gate (design score below 60 points)",
        PIPELINE_BLOCKED_AT_E3 => {
            "E3 safety gate (safety score below its gate threshold or a critical finding)"
        }
        _ => "pipeline gate",
    }
}

/// The humanised label of one of the five canonical safety dimensions.
fn safety_label(id: &str) -> Option<&'static str> {
    SAFETY_DIMENSIONS
        .iter()
        .position(|(candidate, _, _)| *candidate == id)
        .map(|index| SAFETY_LABELS[index])
}

/// The first non-blank skill name of a report, or the empty string.
fn report_name(report: &EvaluationReport) -> &str {
    [
        report.final_score.skill_name.as_str(),
        report.design.skill_name.as_str(),
        report.safety.skill_name.as_str(),
        report.functional.skill_name.as_str(),
    ]
    .into_iter()
    .find(|name| !name.trim().is_empty())
    .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------

/// A verdict bullet value, `-` when the report field is blank.
fn or_placeholder(text: &str) -> &str {
    if text.is_empty() {
        "-"
    } else {
        text
    }
}

/// Renders `Yes` or `No` for a boolean verdict.
fn yes_no(value: bool) -> &'static str {
    if value {
        "Yes"
    } else {
        "No"
    }
}

/// The report's grade, falling back to [`grade_for`] when it is blank.
fn grade_text(grade: &str, fraction: f64) -> String {
    let grade = grade.trim();
    if grade.is_empty() {
        grade_for(fraction)
    } else {
        grade.to_owned()
    }
}

/// Renders a fraction as a percentage with at most two decimals, e.g. `87.5%`.
fn format_percent(fraction: f64) -> String {
    let percent = trim_zeros(format!("{:.2}", fraction * 100.0));
    format!("{percent}%")
}

/// Renders `score / max` as a percentage, or `-` when `max` is not positive.
fn format_ratio_percent(score: f64, max: f64) -> String {
    if max <= 0.0 {
        return "-".to_owned();
    }
    format_percent(score / max)
}

/// Renders a fraction with at most four decimals, e.g. `0.818` or `0.296`.
fn format_fraction(fraction: f64) -> String {
    trim_zeros(format!("{fraction:.4}"))
}

/// Renders points (0–100) with at most two decimals, e.g. `7` or `12.5`.
fn format_points(points: f64) -> String {
    trim_zeros(format!("{points:.2}"))
}

/// Renders an evaluator weight with exactly two decimals, e.g. `0.40`.
fn format_weight(weight: f64) -> String {
    format!("{weight:.2}")
}

/// Strips the trailing zeros (and a bare trailing point) from a decimal string.
fn trim_zeros(text: String) -> String {
    if !text.contains('.') {
        return text;
    }
    let trimmed = text.trim_end_matches('0');
    trimmed.strip_suffix('.').unwrap_or(trimmed).to_owned()
}
