//! The embedded built-in agent-template catalog.
//!
//! [`BUILTIN_RULES_YAML`] is compiled into the binary with `include_str!`; the
//! first call to [`builtin_specs`] parses it into the same [`DeepAgentSpec`]
//! type a workspace template uses and caches the result for the process.
//!
//! The catalog is pure data. Parsing never executes a template, and a broken
//! catalog is reported as [`Error::config`] (it is a packaging bug, not user
//! input).

use std::sync::OnceLock;

use openjiuwen_common::error::{Error, Result};

use super::spec::DeepAgentSpec;

/// The embedded YAML catalog of built-in templates and rules.
pub const BUILTIN_RULES_YAML: &str = include_str!("resources/builtin_rules.yaml");

/// Parsed catalog cache; the inner `String` is the invariant-violation message.
static BUILTIN_SPECS: OnceLock<std::result::Result<Vec<DeepAgentSpec>, String>> = OnceLock::new();

/// Returns the built-in agent templates, parsed on first use.
///
/// The returned vector is a clone of the cached catalog, so callers may mutate
/// or inspect it freely. Order is the declaration order in
/// [`BUILTIN_RULES_YAML`].
///
/// # Errors
///
/// Returns [`Error::config`] when the embedded catalog cannot be parsed or
/// contains no template. The catalog is compile-time data, so this indicates a
/// broken build rather than bad user input.
pub fn builtin_specs() -> Result<Vec<DeepAgentSpec>> {
    let cached = BUILTIN_SPECS.get_or_init(|| {
        let specs: Vec<DeepAgentSpec> = serde_yaml_ng::from_str(BUILTIN_RULES_YAML)
            .map_err(|error| format!("built-in agent templates are not valid YAML: {error}"))?;
        if specs.is_empty() {
            return Err("built-in agent template catalog is empty".to_string());
        }
        for spec in &specs {
            spec.validate()
                .map_err(|error| format!("built-in agent template is invalid: {error}"))?;
        }
        Ok(specs)
    });
    match cached {
        Ok(specs) => Ok(specs.clone()),
        Err(message) => Err(Error::config(message.clone())),
    }
}
