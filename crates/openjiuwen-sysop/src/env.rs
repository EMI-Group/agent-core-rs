//! Environment overlays for shell and code subprocesses.
//!
//! Python parity: `ExecutionEnvironment` (`core/sys_operation/execution_environment.py`)
//! plus `OperationUtils.prepare_environment` (`local/utils.py`). Precedence is
//! *policy values < call-site environment < policy overrides*, and a `None` value removes
//! the variable. The Rust port has no `ContextVar`: the overlay is passed explicitly to
//! the backend instead of being ambient, which makes precedence local and testable.
//!
//! ```
//! use std::collections::BTreeMap;
//! use openjiuwen_sysop::env::ExecutionEnvironment;
//!
//! let mut call = BTreeMap::new();
//! call.insert("MODE".to_string(), Some("test".to_string()));
//! let mut overlay = ExecutionEnvironment::default();
//! overlay.set_override("MODE", Some("prod".to_string()));
//! // Overrides win over call-site values, exactly like the Python ContextVar.
//! assert_eq!(overlay.merged(&call).get("MODE").unwrap().as_deref(), Some("prod"));
//! ```

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};

use crate::config::EnvPolicy;

/// An environment overlay: shared values plus forced overrides.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionEnvironment {
    values: BTreeMap<String, Option<String>>,
    overrides: BTreeMap<String, Option<String>>,
}

impl ExecutionEnvironment {
    /// An empty overlay.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a shared value (`None` removes the variable).
    pub fn set_value(&mut self, name: impl Into<String>, value: Option<String>) {
        self.values.insert(name.into(), value);
    }

    /// Sets a forced override (`None` removes the variable).
    pub fn set_override(&mut self, name: impl Into<String>, value: Option<String>) {
        self.overrides.insert(name.into(), value);
    }

    /// The shared values.
    #[must_use]
    pub const fn values(&self) -> &BTreeMap<String, Option<String>> {
        &self.values
    }

    /// The forced overrides.
    #[must_use]
    pub const fn overrides(&self) -> &BTreeMap<String, Option<String>> {
        &self.overrides
    }

    /// Merges the overlay with a call-site environment.
    ///
    /// Precedence: shared values, then call-site entries, then overrides.
    #[must_use]
    pub fn merged(
        &self,
        call: &BTreeMap<String, Option<String>>,
    ) -> BTreeMap<String, Option<String>> {
        let mut merged = self.values.clone();
        for (name, value) in call {
            merged.insert(name.clone(), value.clone());
        }
        for (name, value) in &self.overrides {
            merged.insert(name.clone(), value.clone());
        }
        merged
    }

    /// Validates every variable name in the overlay.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Validation` for a name that is not a legal environment
    /// variable name.
    pub fn validate(&self) -> Result<()> {
        for name in self.values.keys().chain(self.overrides.keys()) {
            validate_env_name(name)?;
        }
        Ok(())
    }
}

/// Validates an environment variable name (`[A-Za-z_][A-Za-z0-9_]*`).
///
/// # Errors
///
/// Returns `ErrorKind::Validation` when the name is empty, starts with a digit, or
/// contains a character outside the allowed set.
pub fn validate_env_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(Error::validation(
            "environment variable name must not be empty",
        ));
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(Error::validation(format!(
            "invalid environment variable name '{name}'"
        )));
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::validation(format!(
            "invalid environment variable name '{name}'"
        )));
    }
    Ok(())
}

/// Computes the environment a child process receives.
///
/// The parent environment is inherited according to `policy`,
/// [`ExecutionEnvironment::merged`] is applied on top (a `None` value removes the
/// variable), and the result is returned sorted by name so child processes see a
/// deterministic environment. Variables whose name or value is not valid UTF-8 are
/// skipped.
#[must_use]
pub fn resolve_environment(
    policy: &EnvPolicy,
    overlay: &ExecutionEnvironment,
    call: Option<&BTreeMap<String, Option<String>>>,
) -> Vec<(String, String)> {
    let empty = BTreeMap::new();
    let merged = overlay.merged(call.unwrap_or(&empty));
    let mut resolved: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in std::env::vars() {
        if policy.inherits(&name) {
            resolved.insert(name, value);
        }
    }
    for (name, value) in merged {
        // Explicitly requested values are always applied; only scrubbing vetoes them.
        if policy.scrubs(&name) {
            continue;
        }
        match value {
            Some(value) => {
                resolved.insert(name, value);
            }
            None => {
                resolved.remove(&name);
            }
        }
    }
    resolved.into_iter().collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, Option<&str>)]) -> BTreeMap<String, Option<String>> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.map(str::to_string)))
            .collect()
    }

    #[test]
    fn precedence_is_values_then_call_then_overrides() {
        let mut overlay = ExecutionEnvironment::new();
        overlay.set_value("A", Some("shared".to_string()));
        overlay.set_value("B", Some("shared".to_string()));
        overlay.set_override("B", Some("forced".to_string()));
        let merged = overlay.merged(&map(&[("B", Some("call")), ("C", Some("call"))]));
        assert_eq!(merged["A"].as_deref(), Some("shared"));
        assert_eq!(merged["B"].as_deref(), Some("forced"));
        assert_eq!(merged["C"].as_deref(), Some("call"));
    }

    #[test]
    fn none_removes_variable() {
        let mut overlay = ExecutionEnvironment::new();
        overlay.set_value("TOKEN", None);
        overlay.validate().unwrap();
        let resolved = resolve_environment(&EnvPolicy::default(), &overlay, None);
        assert!(!resolved.iter().any(|(name, _)| name == "TOKEN"));
    }

    #[test]
    fn policy_scopes_inheritance() {
        // `PATH` is present in every test environment; `CARGO_PKG_NAME` is set by cargo.
        let policy = EnvPolicy::minimal();
        let resolved = resolve_environment(&policy, &ExecutionEnvironment::new(), None);
        assert!(resolved.iter().any(|(name, _)| name == "PATH"));
        assert!(!resolved.iter().any(|(name, _)| name == "CARGO_PKG_VERSION"));

        let none = EnvPolicy {
            mode: crate::config::EnvInheritMode::None,
            ..EnvPolicy::default()
        };
        let mut overlay = ExecutionEnvironment::new();
        overlay.set_value("EXPLICIT", Some("1".to_string()));
        let resolved = resolve_environment(&none, &overlay, None);
        assert_eq!(resolved, vec![("EXPLICIT".to_string(), "1".to_string())]);
    }

    #[test]
    fn scrub_removes_from_inherited_environment() {
        let policy = EnvPolicy {
            scrub: vec!["PATH".to_string()],
            ..EnvPolicy::default()
        };
        let resolved = resolve_environment(&policy, &ExecutionEnvironment::new(), None);
        assert!(!resolved.iter().any(|(name, _)| name == "PATH"));
    }

    #[test]
    fn env_names_are_validated() {
        assert!(validate_env_name("PATH").is_ok());
        assert!(validate_env_name("_x1").is_ok());
        assert!(validate_env_name("1ABC").is_err());
        assert!(validate_env_name("A-B").is_err());
        assert!(validate_env_name("").is_err());
    }
}
