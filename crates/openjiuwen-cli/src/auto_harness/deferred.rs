//! The two `auto-harness` commands whose Python machinery has no Rust port.
//!
//! - `verify-ext` validates a harness extension package: Python builds a
//!   `RuntimeExtensionArtifact`, hot-loads the rails/tools/skills through the
//!   runtime extension loader and runs `ruff`. Hot-loading needs synthetic
//!   `sys.modules` entries, so the loader is deferred
//!   ([`openjiuwen_rsi::deferred::RuntimeExtensionLoader`]) and this command
//!   reports exactly that instead of a fake "PASSED" line.
//! - `gap-analyze` is a placeholder in Python too (`_run_gap_analyze` calls
//!   `run_gap_analysis`, which returns no gaps and prints "phase 1 placeholder"),
//!   so the Rust port keeps the same placeholder answer and exits `0`.
use openjiuwen_common::error::Result;
use openjiuwen_rsi::auto_harness::Gap;
use openjiuwen_rsi::deferred::RuntimeExtensionLoader;
use serde_json::Value;

/// The placeholder text Python prints for `gap-analyze`.
const GAP_ANALYZE_NOTE: &str =
    "phase 1 placeholder: gap analysis is not wired to a model in this port";

/// Options of `auto-harness verify-ext`.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct VerifyExtCommand {
    /// Extension package root (the directory holding `harness_config.yaml`).
    #[arg(long = "ext-path", value_name = "DIR")]
    pub(crate) ext_path: Option<std::path::PathBuf>,
}

/// Options of `auto-harness gap-analyze` (none: Python takes no flags either).
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GapAnalyzeCommand {}

/// Runs `auto-harness verify-ext`.
///
/// # Errors
///
/// Always returns [`openjiuwen_common::error::Error::Unsupported`] today: the
/// runtime extension loader is a typed deferred surface. If the loader ever
/// gains an implementation, the loaded artifact is printed and the command
/// succeeds.
pub(crate) fn verify_ext(command: &VerifyExtCommand) -> Result<()> {
    RuntimeExtensionLoader::new()?;
    let ext_path = command
        .ext_path
        .as_ref()
        .map(|path| path.display().to_string());
    let loaded = RuntimeExtensionLoader::load(serde_json::json!({ "ext_path": ext_path }))?;
    // Unreachable while the loader is deferred; kept so a future implementation
    // has the user-visible behaviour Python has.
    println!("{loaded}");
    Ok(())
}

/// Runs `auto-harness gap-analyze`, which is a placeholder in both ports.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::Config`] only if the JSON form
/// cannot be serialised, which the fixed payload cannot trigger.
pub(crate) fn gap_analyze(_command: &GapAnalyzeCommand, json: bool) -> Result<()> {
    if json {
        let payload = gap_payload();
        let text = serde_json::to_string_pretty(&payload).map_err(|err| {
            openjiuwen_common::error::Error::config(format!("cannot serialise gap analysis: {err}"))
        })?;
        println!("{text}");
        return Ok(());
    }
    println!("{GAP_ANALYZE_NOTE}");
    Ok(())
}

/// The placeholder answer of `gap-analyze --json`.
///
/// `gaps` stays an empty `Vec<Gap>` (rather than an untyped `[]`) so the JSON
/// shape is the one a wired-in analyser will produce.
fn gap_payload() -> Value {
    serde_json::json!({
        "gaps": Vec::<Gap>::new(),
        "placeholder": true,
        "note": GAP_ANALYZE_NOTE,
    })
}

#[cfg(test)]
mod tests {
    //! Deferred-surface tests: the typed error and the gap-analyze JSON shape.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_rsi::deferred::RuntimeExtensionLoader;

    use super::{gap_payload, verify_ext, VerifyExtCommand};

    #[test]
    fn gap_payload_is_the_documented_placeholder_shape() {
        let payload = gap_payload();
        assert_eq!(
            payload,
            serde_json::json!({
                "gaps": [],
                "placeholder": true,
                "note": "phase 1 placeholder: gap analysis is not wired to a model in this port",
            })
        );
        assert!(
            payload["gaps"].as_array().unwrap().is_empty(),
            "the placeholder has no gaps"
        );
    }

    #[test]
    fn verify_ext_reports_the_deferred_runtime_extension_loader() {
        let command = VerifyExtCommand { ext_path: None };
        let error = verify_ext(&command).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("runtime extension loader"),
            "{error}"
        );
    }

    #[test]
    fn the_runtime_extension_loader_is_the_surface_that_fails() {
        let error = RuntimeExtensionLoader::new().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error
            .to_string()
            .contains("not implemented in the Rust port"));
    }
}
