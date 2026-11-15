//! String parsing and rendering for the `lsp` tool's public types.
//!
//! A host (the CLI, or a wrapper around the `lsp` tool) can drive the tool from text instead of
//! duplicating [`LspToolConfig`], [`LspAction`] and [`LspArgs`]:
//!
//! - [`LspToolConfig`] parses a comma-separated `key=value` spec over the scalar knobs. The
//!   accepted keys are [`LspToolConfig::SPEC_KEYS`]: `workspace` (the workspace root path),
//!   `language`, `language_id`, `request_timeout` (a whole number with an `ms`, `s` or `ns`
//!   suffix), `max_file_size` (bytes, as a plain integer) and `context_lines`. Missing keys take
//!   their defaults, so an empty spec is the default configuration. Server definitions
//!   (`server`, `servers`) are deliberately *not* part of the spec: they describe a process
//!   (command, arguments, environment) rather than a scalar, so they stay builder-only.
//! - [`LspAction`] parses the `snake_case` wire names in [`LspAction::ALL`], case-insensitively,
//!   accepting `-` in place of `_` so a kebab-case CLI flag value works.
//! - [`LspArgs`] parses one CLI token of the form `<action> [path][:line[:column]] [query...]`.
//!
//! Every failure is [`Error::Validation`] naming the offending token, and nothing panics.
//!
//! # `LspArgs` grammar
//!
//! ```text
//! <action> [path][:line[:column]] [query...]
//! ```
//!
//! - The first whitespace-separated token is the action.
//! - An action that [`LspAction::requires_path`] takes the next token as its path, optionally
//!   followed by `:line` or `:line:column` (both one-based, `1..=4294967295`). Positions split
//!   from the right and only trailing all-digit segments count, so `C:\dir\main.rs:10:5` still
//!   parses; a non-digit segment that follows a numeric one (`src/lib.rs:10:x`) is a malformed
//!   position and is rejected, as is a third numeric segment.
//! - `workspace_symbols` takes every remaining token as its `query`, joined with single spaces
//!   (`workspace_symbols Foo`).
//! - `diagnostics` takes the next token as its path, and no path means "drain every buffered
//!   document" (`diagnostics src/main.rs`); a position suffix is rejected because that action
//!   ignores it.
//! - Any other trailing token is rejected, so a typo cannot be silently dropped.
//!
//! Tokens are whitespace-separated, so a *path* containing whitespace cannot be parsed; build
//! such arguments with [`LspArgs::with_path`]. A query may contain whitespace.
//!
//! [`Display`](std::fmt::Display) renders the canonical form of each type, which
//! [`FromStr`](std::str::FromStr) parses back: an unset field is omitted, and
//! [`LspToolConfig`] emits only the keys that differ from the defaults (so the default
//! configuration renders as the empty string). `Display` renders only the fields the grammar
//! carries for that action, so a `path` on a `workspace_symbols` value (or a `query` on a
//! position action) is not rendered, and a path whose final `:`-separated segment is numeric is
//! indistinguishable from a position.
use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};

use super::tool::{LspAction, LspArgs, LspToolConfig};

impl LspToolConfig {
    /// The spec keys accepted by [`LspToolConfig::from_str`], in `Display` order.
    pub const SPEC_KEYS: [&'static str; 6] = [
        "workspace",
        "language",
        "language_id",
        "request_timeout",
        "max_file_size",
        "context_lines",
    ];

    /// Applies one parsed `key=value` item, naming the whole `item` in every error.
    fn apply_spec_item(&mut self, key: &str, value: &str, item: &str) -> Result<()> {
        match key {
            "workspace" => {
                if value.is_empty() {
                    return Err(empty_value(key, item));
                }
                self.workspace_root = Some(PathBuf::from(value));
            }
            "language" => {
                if value.is_empty() {
                    return Err(empty_value(key, item));
                }
                value.clone_into(&mut self.language);
            }
            "language_id" => {
                if value.is_empty() {
                    return Err(empty_value(key, item));
                }
                value.clone_into(&mut self.language_id);
            }
            "request_timeout" => self.request_timeout = parse_timeout(value, item)?,
            "max_file_size" => self.max_file_size = parse_number("max_file_size", value, item)?,
            "context_lines" => self.context_lines = parse_number("context_lines", value, item)?,
            other => {
                return Err(Error::validation(format!(
                    "unknown key '{other}' in lsp config item '{item}'; expected one of: {}",
                    Self::SPEC_KEYS.join(", ")
                )))
            }
        }
        Ok(())
    }
}

impl FromStr for LspToolConfig {
    type Err = Error;

    /// Parses a comma-separated `key=value` spec over the scalar knobs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty item, a token without `=`, a duplicate key, an
    /// unknown key, an empty string value, or a value that is not a number/duration; every
    /// message names the offending token.
    fn from_str(spec: &str) -> Result<Self> {
        let mut config = Self::default();
        if spec.trim().is_empty() {
            return Ok(config);
        }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for item in spec.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(Error::validation(format!(
                    "empty item in lsp config spec; expected a comma-separated '<key>=<value>' list of: {}",
                    Self::SPEC_KEYS.join(", ")
                )));
            }
            let Some((key, value)) = item.split_once('=') else {
                return Err(Error::validation(format!(
                    "invalid lsp config item '{item}'; expected '<key>=<value>' with a key from: {}",
                    Self::SPEC_KEYS.join(", ")
                )));
            };
            let key = key.trim();
            if !seen.insert(key.to_owned()) {
                return Err(Error::validation(format!(
                    "duplicate key '{key}' in lsp config spec"
                )));
            }
            config.apply_spec_item(key, value.trim(), item)?;
        }
        Ok(config)
    }
}

impl fmt::Display for LspToolConfig {
    /// Renders the knobs that differ from the defaults as a `key=value` spec.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let default = Self::default();
        let mut items: Vec<String> = Vec::new();
        if let Some(root) = &self.workspace_root {
            items.push(format!("workspace={}", root.to_string_lossy()));
        }
        if self.language != default.language {
            items.push(format!("language={}", self.language));
        }
        if self.language_id != default.language_id {
            items.push(format!("language_id={}", self.language_id));
        }
        if self.request_timeout != default.request_timeout {
            items.push(format!(
                "request_timeout={}",
                format_duration(self.request_timeout)
            ));
        }
        if self.max_file_size != default.max_file_size {
            items.push(format!("max_file_size={}", self.max_file_size));
        }
        if self.context_lines != default.context_lines {
            items.push(format!("context_lines={}", self.context_lines));
        }
        formatter.write_str(&items.join(","))
    }
}

impl FromStr for LspAction {
    type Err = Error;

    /// Parses a wire action name, case-insensitively and with `-` for `_`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] listing every accepted name for an unknown action.
    fn from_str(action: &str) -> Result<Self> {
        let normalized = action.trim().to_ascii_lowercase().replace('-', "_");
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == normalized)
            .ok_or_else(|| {
                Error::validation(format!(
                    "unknown lsp action '{action}'; expected one of: {}",
                    accepted_actions()
                ))
            })
    }
}

impl fmt::Display for LspAction {
    /// Renders the `snake_case` wire name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for LspArgs {
    type Err = Error;

    /// Parses `<action> [path][:line[:column]] [query...]`; see the module docs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty spec, an unknown action, a missing path for an
    /// action that requires one, a non-numeric or out-of-range position, a position on
    /// `diagnostics`, or an unexpected trailing token.
    fn from_str(spec: &str) -> Result<Self> {
        let mut tokens = spec.split_whitespace();
        let Some(action_token) = tokens.next() else {
            return Err(Error::validation(format!(
                "empty lsp tool spec; expected '<action> [path][:line[:column]] [query...]' with an action from: {}",
                accepted_actions()
            )));
        };
        let action: LspAction = action_token.parse()?;
        let mut args = Self::new(action);
        if action.requires_path() {
            let Some(token) = tokens.next() else {
                return Err(Error::validation(format!(
                    "action '{}' requires a path; expected '{} <path>[:line[:column]]'",
                    action.as_str(),
                    action.as_str()
                )));
            };
            let (path, line, column) = split_position(token)?;
            args.path = Some(path);
            args.line = line;
            args.column = column;
            reject_trailing(tokens.next(), action)?;
        } else if action == LspAction::WorkspaceSymbols {
            let query: Vec<&str> = tokens.collect();
            if !query.is_empty() {
                args.query = Some(query.join(" "));
            }
        } else {
            // `diagnostics`: an optional path, and no path drains every buffered document.
            let Some(token) = tokens.next() else {
                return Ok(args);
            };
            let (path, line, column) = split_position(token)?;
            if line.is_some() || column.is_some() {
                return Err(Error::validation(format!(
                    "action '{}' does not take a line/column; got '{token}'",
                    action.as_str()
                )));
            }
            args.path = Some(path);
            reject_trailing(tokens.next(), action)?;
        }
        Ok(args)
    }
}

impl fmt::Display for LspArgs {
    /// Renders the canonical `<action> ...` form the parser accepts.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.action.as_str())?;
        match self.action {
            LspAction::WorkspaceSymbols => {
                if let Some(query) = self.query.as_deref().filter(|query| !query.is_empty()) {
                    write!(formatter, " {query}")?;
                }
            }
            LspAction::Diagnostics => {
                if let Some(path) = &self.path {
                    write!(formatter, " {path}")?;
                }
            }
            _ => {
                if let Some(path) = &self.path {
                    write!(formatter, " {path}")?;
                }
                if let Some(line) = self.line {
                    write!(formatter, ":{line}")?;
                    if let Some(column) = self.column {
                        write!(formatter, ":{column}")?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The accepted action names, comma-separated, for error messages.
fn accepted_actions() -> String {
    LspAction::ALL
        .iter()
        .map(|action| action.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The error for an empty `key=` value.
fn empty_value(key: &str, item: &str) -> Error {
    Error::validation(format!(
        "empty value for '{key}' in lsp config item '{item}'"
    ))
}

/// Parses a plain non-negative integer for `key`.
fn parse_number<T>(key: &str, value: &str, item: &str) -> Result<T>
where
    T: FromStr,
{
    value.parse().map_err(|_| {
        Error::validation(format!(
            "invalid value '{value}' for '{key}' in lsp config item '{item}'; expected a non-negative whole number"
        ))
    })
}

/// Parses a duration with an `ms`, `s` or `ns` suffix.
fn parse_timeout(value: &str, item: &str) -> Result<Duration> {
    let (digits, unit) = if let Some(rest) = value.strip_suffix("ms") {
        (rest, "ms")
    } else if let Some(rest) = value.strip_suffix("ns") {
        (rest, "ns")
    } else if let Some(rest) = value.strip_suffix('s') {
        (rest, "s")
    } else {
        return Err(Error::validation(format!(
            "invalid duration '{value}' in lsp config item '{item}'; expected a whole number with an 'ms', 's' or 'ns' suffix (for example '1500ms' or '15s')"
        )));
    };
    let amount: u64 = digits.parse().map_err(|_| {
        Error::validation(format!(
            "invalid duration '{value}' in lsp config item '{item}'; '{digits}' is not a whole number"
        ))
    })?;
    if amount == 0 {
        return Err(Error::validation(format!(
            "invalid duration '{value}' in lsp config item '{item}'; a timeout must be greater than zero"
        )));
    }
    Ok(match unit {
        "ms" => Duration::from_millis(amount),
        "s" => Duration::from_secs(amount),
        _ => Duration::from_nanos(amount),
    })
}

/// Renders `duration` with the shortest exact suffix (`s`, `ms` or `ns`).
fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds > 0 && duration.subsec_nanos() == 0 {
        return format!("{seconds}s");
    }
    let millis = duration.as_millis();
    if duration.subsec_nanos() % 1_000_000 == 0 && u64::try_from(millis).is_ok() {
        return format!("{millis}ms");
    }
    format!("{}ns", duration.as_nanos())
}

/// Whether `segment` is a non-empty run of ASCII digits.
fn is_digits(segment: &str) -> bool {
    !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit())
}

/// Splits `<path>[:line[:column]]` into the path and its one-based position parts.
fn split_position(token: &str) -> Result<(String, Option<u32>, Option<u32>)> {
    let parts: Vec<&str> = token.split(':').collect();
    if parts.len() == 1 {
        return Ok((token.to_owned(), None, None));
    }
    let numeric_tail = parts
        .iter()
        .rev()
        .take_while(|part| is_digits(part))
        .count();
    if numeric_tail > 2 {
        return Err(Error::validation(format!(
            "too many numeric position components in '{token}'; expected '<path>[:line[:column]]'"
        )));
    }
    match numeric_tail {
        0 => {
            let last = parts[parts.len() - 1];
            let previous = parts[parts.len() - 2];
            if is_digits(previous) {
                return Err(Error::validation(format!(
                    "invalid column '{last}' in '{token}'; a column must be a whole number"
                )));
            }
            Ok((token.to_owned(), None, None))
        }
        1 => {
            let path = parts[..parts.len() - 1].join(":");
            ensure_path(&path, token)?;
            let line = parse_position("line", parts[parts.len() - 1], token)?;
            Ok((path, Some(line), None))
        }
        _ => {
            let path = parts[..parts.len() - 2].join(":");
            ensure_path(&path, token)?;
            let line = parse_position("line", parts[parts.len() - 2], token)?;
            let column = parse_position("column", parts[parts.len() - 1], token)?;
            Ok((path, Some(line), Some(column)))
        }
    }
}

/// Rejects a path that is empty or only whitespace.
fn ensure_path(path: &str, token: &str) -> Result<()> {
    if path.trim().is_empty() {
        return Err(Error::validation(format!(
            "missing path in '{token}'; expected '<path>[:line[:column]]'"
        )));
    }
    Ok(())
}

/// Parses one one-based position component.
fn parse_position(kind: &str, digits: &str, token: &str) -> Result<u32> {
    let value: u32 = digits.parse().map_err(|_| {
        Error::validation(format!(
            "invalid {kind} '{digits}' in '{token}'; a {kind} must be between 1 and 4294967295"
        ))
    })?;
    if value == 0 {
        return Err(Error::validation(format!(
            "invalid {kind} '0' in '{token}'; {kind}s are 1-based and must be at least 1"
        )));
    }
    Ok(value)
}

/// Rejects an extra token instead of dropping it silently.
fn reject_trailing(token: Option<&str>, action: LspAction) -> Result<()> {
    if let Some(token) = token {
        return Err(Error::validation(format!(
            "unexpected trailing token '{token}' for action '{}'",
            action.as_str()
        )));
    }
    Ok(())
}
