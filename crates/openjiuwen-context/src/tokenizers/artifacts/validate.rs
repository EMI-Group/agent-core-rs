//! Artefact validation: existence, declared digest and body decoding.
//!
//! Every artefact path the manager hands back passes through here, so a caller
//! never receives a file this crate cannot decode: the download path validates
//! the bytes before they are cached, and both resolution seams ([`resolve`](super::TokenizerArtifactManager::resolve) and
//! [`resolve_cached`](super::TokenizerArtifactManager::resolve_cached)) validate the cached or hand-placed file before returning it.
//!
//! The body parse costs one read of a local file per resolution. That is not a
//! hot path (a resolution runs once per context/model binding, then counting is
//! in memory), and it is what stops a hand-placed artefact naming a model type
//! this loader does not decode from being silently downgraded to a `tiktoken`
//! guess or a characters-per-token estimate.
//!
//! Two dispatch modes are deliberate. A downloaded body is validated against
//! the engine the spec declares, because the target file name is that engine's
//! artefact name by construction ([`artifact_name`]). A resolved path is validated by the
//! shared [`native::load_artifact_file`] dispatch, so the bytes are decoded
//! exactly the way the selector will decode them later.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use super::cache::{artifact_name, sha256_file, sha256_hex};
use super::manager::{
    TIKTOKEN_MODEL_MISSING_CODE, TOKENIZER_DOWNLOAD_FAILED_CODE, TOKENIZER_JSON_MISSING_CODE,
    TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE, TOKENIZER_SHA256_MISMATCH_CODE,
};
use crate::tokenizers::native::{self, NativeTokenizer};
use crate::tokenizers::{TokenizerEngine, TokenizerSpec};

/// The declared digest, normalized to lowercase hex.
pub(super) fn expected_digest(spec: &TokenizerSpec) -> Option<String> {
    spec.sha256
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// The compact code reported when the expected artefact is absent.
pub(super) fn missing_artifact_code(engine: TokenizerEngine) -> &'static str {
    if engine == TokenizerEngine::Tiktoken {
        TIKTOKEN_MODEL_MISSING_CODE
    } else {
        TOKENIZER_JSON_MISSING_CODE
    }
}

/// Validates existence, digest and body of the artefact a spec names.
pub(super) fn validate_file(path: &Path, spec: &TokenizerSpec) -> Result<PathBuf> {
    validate_file_with(
        path.to_path_buf(),
        expected_digest(spec).as_deref(),
        missing_artifact_code(spec.engine),
    )
}

/// Validates one artefact path against an already-normalized digest.
///
/// The order is existence, declared digest, then the body: a tampered file
/// reports [`TOKENIZER_SHA256_MISMATCH_CODE`] even when its body is also
/// undecodable, so a digest mismatch never hides behind a parse error.
///
/// # Errors
///
/// - [`Error::Validation`] with the engine's missing code when `path` is not a file, with [`TOKENIZER_SHA256_MISMATCH_CODE`] when the digest differs, and with [`TOKENIZER_DOWNLOAD_FAILED_CODE`] when the body cannot be read or decoded;
/// - [`Error::Unsupported`] with [`TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE`] when the body declares a model type this crate does not decode.
pub(super) fn validate_file_with(
    path: PathBuf,
    expected: Option<&str>,
    missing: &'static str,
) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::Validation(missing.to_owned()));
    }
    if let Some(expected) = expected {
        let actual = sha256_file(&path).map_err(|error| {
            Error::Validation(format!(
                "{TOKENIZER_DOWNLOAD_FAILED_CODE}: tokenizer artefact {} could not be read: {error}",
                path.display()
            ))
        })?;
        if actual != expected {
            return Err(Error::Validation(TOKENIZER_SHA256_MISMATCH_CODE.to_owned()));
        }
    }
    validate_file_body(&path)?;
    Ok(path)
}

/// Parses one artefact file without the digest check.
///
/// This is the check the cached-hit probes use: a cached file whose body cannot
/// be decoded is not a cache entry, so it is fetched (or refused) again instead
/// of being reused.
pub(super) fn validate_file_body(path: &Path) -> Result<()> {
    let display = path.display().to_string();
    match native::load_artifact_file(path, &display) {
        Ok(_) => Ok(()),
        Err(Error::Unsupported(detail)) => Err(undecodable_error(&display, &detail)),
        Err(error) => Err(Error::Validation(format!(
            "{TOKENIZER_DOWNLOAD_FAILED_CODE}: {display} is not a usable tokenizer: {error}"
        ))),
    }
}

/// Validates an artefact file off the async task.
///
/// The read, the digest and the parse run inside `spawn_blocking`, so a
/// resolution on a tokio task never blocks the reactor for the length of a file
/// read.
///
/// # Errors
///
/// Same as [`validate_file_with`], plus [`Error::Validation`] with
/// [`TOKENIZER_DOWNLOAD_FAILED_CODE`] when the blocking task itself fails.
pub(super) async fn validate_async(path: PathBuf, spec: &TokenizerSpec) -> Result<PathBuf> {
    let expected = expected_digest(spec);
    let missing = missing_artifact_code(spec.engine);
    let validated =
        tokio::task::spawn_blocking(move || validate_file_with(path, expected.as_deref(), missing))
            .await;
    match validated {
        Ok(result) => result,
        Err(error) => {
            tracing::warn!(%error, "tokenizer artefact validation task failed");
            Err(Error::Validation(format!(
                "{TOKENIZER_DOWNLOAD_FAILED_CODE}: tokenizer artefact validation task failed: {error}"
            )))
        }
    }
}

/// Refuses a body that cannot be the artefact its name promises.
///
/// A well-formed document of a model type this crate does not decode is a typed
/// [`Error::Unsupported`] carrying [`TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE`] and the declared type, so the caller learns why counting falls back to `tiktoken` or the estimate instead of getting a silent downgrade.
/// The bytes are not cached: an artefact this loader cannot decode would be handed straight back by [`TokenizerArtifactManager::resolve_cached`](super::TokenizerArtifactManager::resolve_cached) for the selector to downgrade again, so the precedence is reject-before-write and a later loader that decodes the type downloads it afresh.
/// The decode follows the engine the spec declares, matching the name a download writes; a resolved path is validated by [`validate_file_body`] instead, which dispatches on the path.
/// Malformed text is rejected with [`Error::Validation`] and [`TOKENIZER_DOWNLOAD_FAILED_CODE`], and the SHA-256 check is unchanged.
///
/// # Errors
///
/// See the description above: [`Error::Validation`] for a non-UTF-8 or
/// undecodable body, [`Error::Unsupported`] for an undecoded model type.
pub(super) fn validate_body(spec: &TokenizerSpec, bytes: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        Error::Validation(format!(
            "{TOKENIZER_DOWNLOAD_FAILED_CODE}: artefact body is not UTF-8: {error}"
        ))
    })?;
    let decoded = match spec.engine {
        TokenizerEngine::Tiktoken => NativeTokenizer::from_tiktoken_model(text).map(|_| ()),
        TokenizerEngine::Auto | TokenizerEngine::Native => {
            NativeTokenizer::from_json_str(text).map(|_| ())
        }
    };
    match decoded {
        Ok(()) => Ok(()),
        Err(Error::Unsupported(detail)) => {
            Err(undecodable_error(&declared_body_name(spec), &detail))
        }
        Err(error) => Err(Error::Validation(format!(
            "{TOKENIZER_DOWNLOAD_FAILED_CODE}: artefact body is not a usable tokenizer: {error}"
        ))),
    }
}

/// Compares the downloaded bytes with the declared digest.
///
/// # Errors
///
/// [`Error::Validation`] carrying [`TOKENIZER_SHA256_MISMATCH_CODE`] when the
/// bytes do not match the declared digest.
pub(super) fn verify_digest(spec: &TokenizerSpec, bytes: &[u8]) -> Result<()> {
    let Some(expected) = expected_digest(spec) else {
        return Ok(());
    };
    if sha256_hex(bytes) == expected {
        return Ok(());
    }
    Err(Error::Validation(format!(
        "{TOKENIZER_SHA256_MISMATCH_CODE}: downloaded artefact does not match sha256 {expected}"
    )))
}

/// The typed refusal of an artefact declaring an undecoded model type.
fn undecodable_error(artifact: &str, detail: &str) -> Error {
    Error::Unsupported(format!(
        "{TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE}: {artifact} declares a model type this loader does not decode: {detail}"
    ))
}

/// The artefact a body belongs to, as a message names it.
///
/// The tokenizer id is preferred over the engine's file name, so a download
/// failure tells the reader which artifact to look at.
fn declared_body_name(spec: &TokenizerSpec) -> String {
    let id = spec
        .tokenizer_id
        .as_deref()
        .map(str::trim)
        .unwrap_or_default();
    if id.is_empty() {
        format!("the {} artefact", artifact_name(spec.engine))
    } else {
        format!("the tokenizer artefact {id:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny BPE document the native decoder understands.
    const BPE: &str = r#"{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}"#;

    /// A well-formed document whose model type this crate does not decode.
    const UNDECODED: &str = r#"{"model": {"type": "UnigramV2", "vocab": [["a", 0.0]]}}"#;

    /// A tiny `tiktoken.model` table: the token `a` at rank 0.
    const MODEL: &str = "YQ== 0\n";

    fn spec(engine: TokenizerEngine) -> TokenizerSpec {
        TokenizerSpec {
            provider: "acme".to_owned(),
            model: "acme-mini".to_owned(),
            tokenizer_id: Some("acme/tok".to_owned()),
            engine,
            ..TokenizerSpec::default()
        }
    }

    fn write(name: &str, content: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent");
        }
        std::fs::write(&path, content).expect("fixture");
        (dir, path)
    }

    #[test]
    fn a_decodable_file_passes_and_keeps_its_path() {
        let (_dir, path) = write("tokenizer.json", BPE);
        let validated = validate_file(&path, &spec(TokenizerEngine::Native)).expect("valid");
        assert_eq!(validated, path);
    }

    #[test]
    fn a_missing_file_reports_the_engine_code() {
        let dir = tempfile::tempdir().expect("temp dir");
        let error = validate_file(
            &dir.path().join("absent.json"),
            &spec(TokenizerEngine::Native),
        )
        .expect_err("missing");
        assert_eq!(
            error.message(),
            missing_artifact_code(TokenizerEngine::Native)
        );

        let error = validate_file(
            &dir.path().join("absent.model"),
            &spec(TokenizerEngine::Tiktoken),
        )
        .expect_err("missing");
        assert_eq!(
            error.message(),
            missing_artifact_code(TokenizerEngine::Tiktoken)
        );
    }

    #[test]
    fn a_hand_placed_undecoded_model_type_is_the_typed_refusal() {
        let (_dir, path) = write("tokenizer.json", UNDECODED);
        let error = validate_file(&path, &spec(TokenizerEngine::Native)).expect_err("undecoded");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        let message = error.message();
        assert!(
            message.starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
            "{message}"
        );
        assert!(message.contains("UnigramV2"), "{message}");
        assert!(message.contains(&path.display().to_string()), "{message}");
    }

    #[test]
    fn a_malformed_body_is_a_validation_error() {
        let (_dir, path) = write("tokenizer.json", "not a tokenizer");
        let error = validate_file(&path, &spec(TokenizerEngine::Native)).expect_err("malformed");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(
            error.message().starts_with(TOKENIZER_DOWNLOAD_FAILED_CODE),
            "{error}"
        );
        assert!(
            error.message().contains("not a usable tokenizer"),
            "{error}"
        );
    }

    #[test]
    fn the_declared_digest_is_checked_before_the_body() {
        let (_dir, path) = write("tokenizer.json", UNDECODED);
        let mut strict = spec(TokenizerEngine::Native);
        strict.sha256 = Some("0".repeat(64));
        let error = validate_file(&path, &strict).expect_err("digest mismatch");
        assert_eq!(error.message(), TOKENIZER_SHA256_MISMATCH_CODE);

        // The matching digest lets the body check run and report the real cause.
        let mut matching = spec(TokenizerEngine::Native);
        matching.sha256 = Some(sha256_hex(UNDECODED.as_bytes()));
        let error = validate_file(&path, &matching).expect_err("undecoded body");
        assert!(
            error
                .message()
                .starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
            "{error}"
        );
    }

    #[test]
    fn a_path_with_a_custom_name_is_sniffed_like_the_selector_does() {
        let (_dir, json) = write("custom-name", BPE);
        assert!(validate_file_body(&json).is_ok());
        let (_dir, model) = write("vocab.bin", MODEL);
        assert!(
            validate_file_body(&model).is_ok(),
            "a tiktoken table under any name is a usable artefact"
        );
    }

    #[test]
    fn downloaded_bytes_are_validated_against_the_declared_engine() {
        let tiktoken = spec(TokenizerEngine::Tiktoken);
        validate_body(&tiktoken, MODEL.as_bytes()).expect("a model table");
        let json = spec(TokenizerEngine::Native);
        validate_body(&json, BPE.as_bytes()).expect("a document");

        // The engine decides: a model table is not a `tokenizer.json` body.
        let error = validate_body(&json, MODEL.as_bytes()).expect_err("engine dispatch");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let error = validate_body(&tiktoken, UNDECODED.as_bytes()).expect_err("not a table");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn a_downloaded_body_of_an_undecoded_model_type_names_the_artefact() {
        let error = validate_body(&spec(TokenizerEngine::Native), UNDECODED.as_bytes())
            .expect_err("undecoded");
        let message = error.message();
        assert!(
            message.starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
            "{message}"
        );
        assert!(message.contains("acme/tok"), "{message}");
        assert!(message.contains("UnigramV2"), "{message}");
    }

    #[test]
    fn a_body_that_is_not_utf8_is_refused() {
        let error =
            validate_body(&spec(TokenizerEngine::Native), &[0xff, 0xfe]).expect_err("not utf-8");
        assert!(error.message().contains("not UTF-8"), "{error}");
    }

    #[test]
    fn a_declared_digest_must_match_the_downloaded_bytes() {
        let mut strict = spec(TokenizerEngine::Native);
        strict.sha256 = Some(sha256_hex(BPE.as_bytes()).to_uppercase());
        verify_digest(&strict, BPE.as_bytes()).expect("the normalized digest matches");

        strict.sha256 = Some("0".repeat(64));
        let error = verify_digest(&strict, BPE.as_bytes()).expect_err("mismatch");
        assert!(error.message().starts_with(TOKENIZER_SHA256_MISMATCH_CODE));
    }

    #[tokio::test]
    async fn the_async_seam_validates_the_same_way() {
        let (_dir, path) = write("tokenizer.json", UNDECODED);
        let error = validate_async(path, &spec(TokenizerEngine::Native))
            .await
            .expect_err("undecoded");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
    }
}
