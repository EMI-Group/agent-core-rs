//! The stdout narrative helpers of the examples.
//!
//! These four functions are the only place an example writes to standard output:
//! diagnostics go through [`tracing`] to stderr (see [`super::logging`]), so a
//! reader can follow the narrative even when logging is at `debug`, and piping an
//! example's stdout yields just the story. The formatters are kept separate from
//! the printing functions so they can be unit-tested without capturing stdout.

/// Prefix of a section title.
const TITLE_PREFIX: &str = "====";

/// Prefix of a numbered step.
const STEP_PREFIX: &str = "  -";

/// Prefix of an informational note.
const NOTE_PREFIX: &str = "note:";

/// Prefix of a final answer.
const ANSWER_PREFIX: &str = "answer:";

/// Prints a section title, for example `==== openjiuwen examples ====`.
pub fn title(text: &str) {
    println!("{}", format_title(text));
}

/// Prints a numbered step, for example `  - build the model`.
pub fn step(text: &str) {
    println!("{}", format_step(text));
}

/// Prints an informational note, for example `note: run mode: offline`.
pub fn note(text: &str) {
    println!("{}", format_note(text));
}

/// Prints the final answer of an example, for example `answer: Paris`.
pub fn answer(text: &str) {
    println!("{}", format_answer(text));
}

/// Formats a section title.
fn format_title(text: &str) -> String {
    format!("{TITLE_PREFIX} {text} {TITLE_PREFIX}")
}

/// Formats a step line.
fn format_step(text: &str) -> String {
    format!("{STEP_PREFIX} {text}")
}

/// Formats a note line.
fn format_note(text: &str) -> String {
    format!("{NOTE_PREFIX} {text}")
}

/// Formats an answer line.
fn format_answer(text: &str) -> String {
    format!("{ANSWER_PREFIX} {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_is_delimited_by_equals() {
        assert_eq!(
            format_title("openjiuwen examples"),
            "==== openjiuwen examples ===="
        );
    }

    #[test]
    fn step_is_indented_with_a_dash() {
        assert_eq!(format_step("build the model"), "  - build the model");
    }

    #[test]
    fn note_and_answer_are_prefixed() {
        assert_eq!(
            format_note("run mode: offline (scripted model)"),
            "note: run mode: offline (scripted model)"
        );
        assert_eq!(format_answer("Paris"), "answer: Paris");
    }

    #[test]
    fn empty_text_leaves_the_prefix_readable() {
        assert_eq!(format_title(""), "====  ====");
        assert_eq!(format_step(""), "  - ");
        assert_eq!(format_note(""), "note: ");
        assert_eq!(format_answer(""), "answer: ");
    }
}
