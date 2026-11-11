//! Resolution of the interpreter that runs a command line, and extraction of the absolute
//! paths a command references.
//!
//! Python parity: `_build_shell_command` / `_extract_abs_paths` in
//! `openjiuwen/core/sys_operation/local/shell_operation.py`.
//!
//! [`resolve_plan`] never interpolates anything but the command text itself, and the
//! command is always the last argument of the returned plan, so a caller cannot smuggle
//! extra interpreter flags through a shell selection. [`extract_absolute_paths`] feeds the
//! sandbox check: every absolute path literal a command mentions must live inside the
//! sandbox roots. Device and pseudo-filesystem prefixes (`/dev/`, `/proc/`, `/sys/`) are
//! extracted as well and can be filtered with [`is_ignored_absolute_path`].
//!
//! ```
//! use openjiuwen_sysop::shell::plan::extract_absolute_paths;
//!
//! assert_eq!(
//!     extract_absolute_paths("cat /etc/passwd | grep root && ls /tmp"),
//!     vec!["/etc/passwd".to_string(), "/tmp".to_string()],
//! );
//! assert_eq!(extract_absolute_paths("echo -n hi"), Vec::<String>::new());
//! ```

use crate::shell::types::ShellType;
use crate::util::format_path;

/// How a command line is handed to a shell or interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    /// Interpreter to launch, for example `/bin/sh` or `cmd`.
    pub program: String,
    /// Interpreter arguments; the command text is always the last element.
    pub args: Vec<String>,
    /// The shell this plan implements (`"sh"`, `"bash"`, `"cmd"`, `"powershell"`).
    pub shell_name: &'static str,
}

impl ExecutionPlan {
    /// The program followed by its arguments, ready for `Command::new(argv[0])`.
    #[must_use]
    pub fn to_argv(&self) -> Vec<String> {
        let mut argv = Vec::with_capacity(self.args.len() + 1);
        argv.push(self.program.clone());
        argv.extend_from_slice(&self.args);
        argv
    }
}

/// Resolves the interpreter used to run `command` for `shell_type`.
///
/// [`ShellType::Auto`] picks `sh` on POSIX and `cmd` on Windows.
///
/// # Errors
///
/// Returns Python's rejection text when the requested interpreter is unavailable on this
/// system (`"shell 'bash' is not available on this system"`) or is not supported by the
/// host OS (`"shell_type 'cmd' is only supported on Windows"`).
#[cfg(unix)]
pub fn resolve_plan(shell_type: ShellType, command: &str) -> Result<ExecutionPlan, String> {
    match shell_type {
        ShellType::Auto | ShellType::Sh => {
            if !std::path::Path::new("/bin/sh").exists() {
                return Err("shell 'sh' is not available on this system".to_string());
            }
            Ok(plan("/bin/sh", "-c", command, "sh"))
        }
        ShellType::Bash => match which::which("bash") {
            Ok(program) => Ok(plan(format_path(&program), "-lc", command, "bash")),
            Err(_) if std::path::Path::new("/bin/bash").exists() => {
                Ok(plan("/bin/bash", "-lc", command, "bash"))
            }
            Err(_) => Err("shell 'bash' is not available on this system".to_string()),
        },
        ShellType::Powershell => powershell_plan(command),
        ShellType::Cmd => Err("shell_type 'cmd' is only supported on Windows".to_string()),
    }
}

/// Resolves the interpreter used to run `command` for `shell_type`.
///
/// [`ShellType::Auto`] picks `cmd` on Windows.
///
/// # Errors
///
/// Returns Python's rejection text when the requested interpreter is unavailable on this
/// system (`"shell 'bash' is not available on this system"`).
#[cfg(windows)]
pub fn resolve_plan(shell_type: ShellType, command: &str) -> Result<ExecutionPlan, String> {
    match shell_type {
        ShellType::Auto | ShellType::Cmd => Ok(ExecutionPlan {
            program: "cmd".to_string(),
            args: vec!["/C".to_string(), command.to_string()],
            shell_name: "cmd",
        }),
        ShellType::Bash => match which::which("bash") {
            Ok(program) => Ok(plan(format_path(&program), "-lc", command, "bash")),
            Err(_) => Err("shell 'bash' is not available on this system".to_string()),
        },
        ShellType::Sh => match which::which("sh") {
            Ok(program) => Ok(plan(format_path(&program), "-c", command, "sh")),
            Err(_) => Err("shell 'sh' is not available on this system".to_string()),
        },
        ShellType::Powershell => powershell_plan(command),
    }
}

/// Resolves a PowerShell plan, preferring `pwsh` over the legacy `powershell` host.
fn powershell_plan(command: &str) -> Result<ExecutionPlan, String> {
    let program = which::which("pwsh")
        .or_else(|_| which::which("powershell"))
        .map_err(|_| "shell 'powershell' is not available on this system".to_string())?;
    Ok(ExecutionPlan {
        program: format_path(&program),
        args: vec![
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-Command".to_string(),
            command.to_string(),
        ],
        shell_name: "powershell",
    })
}

/// Builds a plan whose last argument is the command text.
fn plan(
    program: impl Into<String>,
    flag: &str,
    command: &str,
    shell_name: &'static str,
) -> ExecutionPlan {
    ExecutionPlan {
        program: program.into(),
        args: vec![flag.to_string(), command.to_string()],
        shell_name,
    }
}

/// Best-effort extraction of the absolute path literals a command line mentions.
///
/// The scan is quote-aware: a quoted run may contain spaces. Shell operators (`|`, `&`,
/// `;`, `>`, `<`, `(`, `)`) and whitespace separate tokens. A token counts as absolute
/// when it starts with `/` (but not `//`), with `\\`, or with a drive letter followed by
/// `\` or `/`; drive-letter detection is platform-independent so the same analysis runs
/// on every host. One layer of surrounding quotes and trailing punctuation is stripped.
/// Variable-interpolated paths (`$var`) are intentionally not detected.
#[must_use]
pub fn extract_absolute_paths(command: &str) -> Vec<String> {
    split_words(command)
        .into_iter()
        .filter_map(|word| {
            let cleaned = trim_trailing_punctuation(&word);
            let cleaned = strip_surrounding_quotes(cleaned);
            if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
                return None;
            }
            if is_absolute_literal(cleaned) {
                Some(cleaned.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Returns `true` when `path` points into a device or pseudo filesystem that callers skip.
///
/// The prefixes are `/dev/`, `/proc/`, and `/sys/`; the check is an exact prefix test, so
/// `/devices` is not ignored.
#[must_use]
pub fn is_ignored_absolute_path(path: &str) -> bool {
    ["/dev/", "/proc/", "/sys/"]
        .iter()
        .any(|prefix| path.starts_with(prefix))
}

/// Splits `text` into whitespace/operator-delimited words, keeping quotes part of the word.
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for character in text.chars() {
        match quote {
            Some(delimiter) => {
                current.push(character);
                if character == delimiter {
                    quote = None;
                }
            }
            None => {
                if character == '\'' || character == '"' {
                    quote = Some(character);
                    current.push(character);
                } else if character.is_whitespace() || is_operator(character) {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push(character);
                }
            }
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Whether `character` separates shell words.
fn is_operator(character: char) -> bool {
    matches!(character, '|' | '&' | ';' | '>' | '<' | '(' | ')')
}

/// Removes trailing punctuation Python's path matchers exclude.
fn trim_trailing_punctuation(word: &str) -> &str {
    word.trim_end_matches([',', ';', '|', '&', ')', '>'])
}

/// Removes one layer of matching surrounding quotes.
fn strip_surrounding_quotes(word: &str) -> &str {
    for delimiter in ['"', '\''] {
        if word.len() >= 2 && word.starts_with(delimiter) && word.ends_with(delimiter) {
            return &word[delimiter.len_utf8()..word.len() - delimiter.len_utf8()];
        }
    }
    word
}

/// Whether `word` looks like an absolute path literal.
///
/// A bare `/` is not a literal: Python's matcher requires at least one non-separator
/// character after the leading separator.
fn is_absolute_literal(word: &str) -> bool {
    let posix = word.len() > 1 && word.starts_with('/') && !word.starts_with("//");
    let windows_unc = word.starts_with("\\\\");
    let windows_drive = {
        let mut characters = word.chars();
        matches!(
            (characters.next(), characters.next(), characters.next()),
            (Some(letter), Some(':'), Some(separator))
                if letter.is_ascii_alphabetic() && (separator == '\\' || separator == '/')
        )
    };
    posix || windows_unc || windows_drive
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn plan_appends_command_last() {
        let plan = ExecutionPlan {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), "echo hi".to_string()],
            shell_name: "sh",
        };
        assert_eq!(plan.to_argv(), vec!["/bin/sh", "-c", "echo hi"]);
        assert_eq!(plan.shell_name, "sh");
    }

    #[cfg(unix)]
    #[test]
    fn posix_auto_and_sh_use_bin_sh() {
        if !std::path::Path::new("/bin/sh").exists() {
            return;
        }
        let plan = resolve_plan(ShellType::Auto, "echo hi").unwrap();
        assert_eq!(plan.program, "/bin/sh");
        assert_eq!(plan.args, vec!["-c", "echo hi"]);
        assert_eq!(plan.shell_name, "sh");

        let plan = resolve_plan(ShellType::Sh, "id").unwrap();
        assert_eq!(plan.program, "/bin/sh");
        assert_eq!(plan.to_argv().last().map(String::as_str), Some("id"));
    }

    #[cfg(unix)]
    #[test]
    fn posix_cmd_is_rejected() {
        assert_eq!(
            resolve_plan(ShellType::Cmd, "dir").unwrap_err(),
            "shell_type 'cmd' is only supported on Windows"
        );
    }

    #[cfg(unix)]
    #[test]
    fn bash_plan_uses_login_shell_flag_when_available() {
        if which::which("bash").is_err() && !std::path::Path::new("/bin/bash").exists() {
            // bash is not installed on this host; nothing to verify.
            return;
        }
        let plan = resolve_plan(ShellType::Bash, "echo hi").unwrap();
        assert_eq!(plan.args, vec!["-lc", "echo hi"]);
        assert_eq!(plan.shell_name, "bash");
        assert_eq!(plan.to_argv().last().map(String::as_str), Some("echo hi"));
    }

    #[test]
    fn missing_powershell_is_reported() {
        // Only assert the error shape when PowerShell really is absent.
        if which::which("pwsh").is_ok() || which::which("powershell").is_ok() {
            let plan = resolve_plan(ShellType::Powershell, "Get-ChildItem").unwrap();
            assert_eq!(plan.shell_name, "powershell");
            assert_eq!(plan.args.last().map(String::as_str), Some("Get-ChildItem"));
        } else {
            assert_eq!(
                resolve_plan(ShellType::Powershell, "Get-ChildItem").unwrap_err(),
                "shell 'powershell' is not available on this system"
            );
        }
    }

    #[test]
    fn extractor_handles_operators_and_quotes() {
        assert_eq!(
            extract_absolute_paths("cat /etc/passwd | grep root && ls /tmp"),
            vec!["/etc/passwd".to_string(), "/tmp".to_string()]
        );
        assert_eq!(
            extract_absolute_paths("cat \"/a b/c\""),
            vec!["/a b/c".to_string()]
        );
        assert_eq!(extract_absolute_paths("echo -n hi"), Vec::<String>::new());
        assert_eq!(
            extract_absolute_paths("relative/path other"),
            Vec::<String>::new()
        );
        assert_eq!(
            extract_absolute_paths("//host/share/x"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn extractor_handles_windows_literals_on_every_platform() {
        assert_eq!(
            extract_absolute_paths("copy D:\\secret\\file.txt x"),
            vec!["D:\\secret\\file.txt".to_string()]
        );
        assert_eq!(
            extract_absolute_paths(r#"copy "C:\Program Files\a.txt" x"#),
            vec![r"C:\Program Files\a.txt".to_string()]
        );
        assert_eq!(
            extract_absolute_paths("dir \\\\server\\share\\x"),
            vec!["\\\\server\\share\\x".to_string()]
        );
    }

    #[test]
    fn extractor_strips_punctuation_and_skips_dots() {
        assert_eq!(
            extract_absolute_paths("ls /tmp; /var,"),
            vec!["/tmp".to_string(), "/var".to_string()]
        );
        assert_eq!(
            extract_absolute_paths("cd /tmp/../other"),
            vec!["/tmp/../other".to_string()]
        );
        assert_eq!(extract_absolute_paths("open '.'"), Vec::<String>::new());
        assert_eq!(extract_absolute_paths("open '/'"), Vec::<String>::new());
        assert_eq!(extract_absolute_paths("open ''"), Vec::<String>::new());
    }

    #[test]
    fn device_paths_are_extracted_but_marked_ignorable() {
        let paths = extract_absolute_paths("cat /dev/null /proc/cpuinfo /sys/x /devices/y");
        assert_eq!(
            paths,
            vec![
                "/dev/null".to_string(),
                "/proc/cpuinfo".to_string(),
                "/sys/x".to_string(),
                "/devices/y".to_string(),
            ]
        );
        assert!(is_ignored_absolute_path("/dev/null"));
        assert!(is_ignored_absolute_path("/proc/cpuinfo"));
        assert!(is_ignored_absolute_path("/sys/x"));
        assert!(!is_ignored_absolute_path("/devices/y"));
        assert!(!is_ignored_absolute_path("/dev"));
    }
}
