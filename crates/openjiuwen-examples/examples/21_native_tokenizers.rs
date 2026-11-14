//! `21_native_tokenizers` — native tokenizers end to end: load, count, select, cache.
//!
//! Python counterpart: `openjiuwen/context_engine/token/` (`tokenizer_manager.py`
//! plus the tokenizer registry and its spec ladder). The crate decodes both
//! native artifact kinds in pure Rust, so every seam below runs offline:
//!
//! 1. **`tokenizer.json`** — generate a five-entry BPE document in a temp dir,
//!    load it with `load_native_tokenizer`, and count one text and one chat
//!    history through the `TokenCounter` seam. An exact load reports
//!    `source() == NativeTokenizer`, `estimated() == false`, and no fallback
//!    reason.
//! 2. **`tiktoken` `.model`** — generate a base64 rank table, let
//!    `load_native_tokenizer` sniff the artifact kind, then encode and decode a
//!    sample and print the vocabulary size (the selection only counts, so the
//!    ids come from `NativeTokenizer` itself).
//! 3. **the selection ladder** — five rungs, each printing `source()`,
//!    `estimated()`, and `fallback_reason()`: an explicit spec whose artifact
//!    resolves, a `Native` engine with no artifact, an unregistered model, a
//!    family registry match, and a spec whose first `compatible_fallbacks`
//!    entry resolves.
//! 4. **the artifact cache** — `cache_target` computes the path without creating
//!    it, `resolve_cached` misses twice with two compact codes, the generated
//!    document is written into the cache, and the next resolve plus selection
//!    hits it.
//!
//! Offline and deterministic: every artifact is generated inside a
//! `tempfile::TempDir`, nothing is downloaded, the ids and counts are exact for
//! the fixtures, and the temp path itself never reaches stdout.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use openjiuwen::core::context_engine::tokenizers::{NativeTokenizer, NativeTokenizerKind};
use openjiuwen::core::context_engine::{
    load_native_tokenizer, CompatibleTokenizerSpec, SelectedTokenizer, TokenMeasurement,
    TokenizerArtifactManager, TokenizerEngine, TokenizerMatchKind, TokenizerRegistry,
    TokenizerSelector, TokenizerSource, TokenizerSpec, REMOTE_DOWNLOAD_DISABLED_CODE,
    TOKENIZER_NOT_IN_LOCAL_CACHE_CODE,
};
use openjiuwen::core::foundation::prompt::{BaseMessage, TokenCounter};

use common::banner;

/// A minimal valid BPE `tokenizer.json`: `a`, `b`, `c` merge through `ab` into
/// `abc`, and nothing else is known.
const BPE_JSON: &str = r#"{
    "name": "tiny-bpe",
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
        "merges": ["a b", "ab c"]
    },
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

/// A second, deliberately distinct BPE document: the same vocabulary with
/// another name, so a selection that named `tiny-bpe-2` would prove the wrong
/// fallback had been read.
const BPE_JSON_SECOND: &str = r#"{
    "name": "tiny-bpe-2",
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
        "merges": ["a b", "ab c"]
    },
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

/// A `tiktoken` `.model` rank table: one `<base64-token> <rank>` line per entry,
/// with a `#` comment and a blank line the loader has to skip.
///
/// Every token is spelled by hand so the example needs no base64 dependency:
/// `YQ==` is `a`, `Yg==` is `b`, `Yw==` is `c`, `YWI=` is `ab`, and `YWJj` is
/// `abc`. Ranks are merged lowest first, so `a b` then `ab c` rebuild `abc`.
const TIKTOKEN_MODEL: &str = "\
# tiny tiktoken vocabulary
YQ== 0
Yg== 1
Yw== 2
YWI= 3
YWJj 4
";

/// The sentence both artifacts encode; it is built from the fixture tokens only.
const SAMPLE: &str = "abcabc";

/// A twelve-character string for the selector's characters-per-token estimate.
const ESTIMATE_SAMPLE: &str = "abcdefghijkl";

fn main() -> Result<()> {
    let mode = common::setup()?;
    banner::title("21 native tokenizers");
    banner::note(&format!(
        "run mode: {}; every artifact is generated in a temp dir and decoded in pure Rust, so \
         nothing downloads",
        mode.describe()
    ));

    let dir = common::tempdir()?;
    let root = dir.path();
    let json_path = native_json(root)?;
    tiktoken_model(root)?;
    ladder(&json_path, root)?;
    artifact_cache(root)?;

    banner::title("what this showed");
    banner::note(
        "`load_native_tokenizer` decodes a `tokenizer.json` and a `tiktoken` `.model` from bytes \
         already on disk; the counts are exact for the loaded artifact, and the `TokenCounter` \
         seam adds four framing tokens per chat message",
    );
    banner::note(
        "the selector is a fallback ladder, not a guess: an unusable rung reports `source`, \
         `estimated`, and `fallback_reason` instead of failing",
    );
    banner::note(
        "the artifact manager computes cache paths without creating them, only reads the local \
         cache on the synchronous selector seam, and reports two distinct miss codes",
    );
    banner::answer(
        "both native artifacts counted exactly, five ladder rungs with typed fallbacks, and a \
         cache hit after one computed miss",
    );
    Ok(())
}

/// Section 1: generate a `tokenizer.json`, load it, and count text and a chat
/// history. Returns the path so later rungs can point at the same artifact.
fn native_json(root: &Path) -> Result<PathBuf> {
    banner::title("a native tokenizer.json");
    let path = root.join("tiny-bpe").join("tokenizer.json");
    write_file(&path, BPE_JSON)?;
    banner::step(&format!(
        "wrote {} — a five-entry BPE with a whitespace pre-tokenizer",
        relative(&path, root)?
    ));

    let selection =
        load_native_tokenizer(&path).context("decoding the generated tokenizer.json")?;
    ensure!(
        selection.source() == TokenizerSource::NativeTokenizer,
        "a loadable tokenizer.json must report the native source"
    );
    ensure!(
        !selection.estimated(),
        "a native artifact is an exact count, not an estimate"
    );
    ensure!(
        selection.fallback_reason().is_none(),
        "an exact load records no fallback reason"
    );
    banner::step(&format!(
        "load_native_tokenizer: source {:?}, estimated {}, fallback_reason {:?}, tokenizer {:?}",
        selection.source(),
        selection.estimated(),
        selection.fallback_reason(),
        selection.tokenizer()
    ));

    let single = selection.count_text("abc");
    let pair = selection.count_text("abc abc");
    ensure!(
        single == 1 && pair == 2,
        "the fixture merges `abc` into one id and the whitespace rule splits two pieces"
    );
    ensure!(
        selection.count_text("") == 0,
        "the empty string always counts as zero"
    );
    banner::step(&format!(
        "count_text: \"abc\" -> {single} id, \"abc abc\" -> {pair} ids; the empty string -> 0"
    ));

    let messages = vec![BaseMessage::system("abc abc"), BaseMessage::user("abc")];
    let framed = selection.count_messages(&messages);
    ensure!(
        framed == 11,
        "two messages cost 4 + 2 and 4 + 1: the chat framing is four tokens per message"
    );
    banner::step(&format!(
        "count_messages: {framed} = 2 messages x 4 framing + 3 payload ids"
    ));

    provenance("measure(\"abc abc\")", &selection.measure("abc abc"));
    provenance(
        "measure_messages(2 messages)",
        &selection.measure_messages(&messages),
    );
    banner::note(
        "the count is exact for this artifact, so `estimated` is false and no fallback is \
         recorded; an unknown model would instead report where its count came from",
    );
    Ok(path)
}

/// Section 2: generate a `.model` rank table, sniff it through
/// `load_native_tokenizer`, and show encode/decode ids plus the vocabulary size.
fn tiktoken_model(root: &Path) -> Result<()> {
    banner::title("a tiktoken .model rank table");
    let path = root.join("tiny.model");
    write_file(&path, TIKTOKEN_MODEL)?;
    banner::step(&format!(
        "wrote {} — five base64 rank lines plus one comment and one blank line",
        relative(&path, root)?
    ));

    // The selection is a counter, so the ids and the vocabulary size come from
    // the `NativeTokenizer` the loader built from the same bytes.
    let native =
        NativeTokenizer::from_tiktoken_model_file(&path).context("decoding the .model table")?;
    ensure!(
        native.kind() == NativeTokenizerKind::Tiktoken,
        "a .model table is the tiktoken kind, not a tokenizer.json model type"
    );
    let specials = native.special_tokens().len();
    let vocab_size = native.vocab_size();
    banner::step(&format!(
        "NativeTokenizer: kind {:?}, vocab_size {vocab_size} ({} mergeable + {specials} \
         special/reserved ids)",
        native.kind(),
        vocab_size - specials
    ));

    let ids = native.encode(SAMPLE);
    ensure!(
        ids == [4, 4],
        "`abcabc` must reuse the fully merged `abc` id"
    );
    let decoded = native.decode(&ids);
    ensure!(
        decoded == SAMPLE,
        "the decoded ids must reproduce the sample sentence"
    );
    banner::step(&format!(
        "encode(\"{SAMPLE}\") -> {ids:?}, decode -> {decoded:?}"
    ));

    let selection =
        load_native_tokenizer(&path).context("loading the artifact through the selection seam")?;
    ensure!(
        selection.source() == TokenizerSource::NativeTokenizer
            && !selection.estimated()
            && selection.fallback_reason().is_none(),
        "the extension/sniffing path must also produce an exact native selection"
    );
    let count = selection.count_text(SAMPLE);
    ensure!(
        count == ids.len(),
        "the selection and the direct tokenizer count the same ids"
    );
    banner::step(&format!(
        "load_native_tokenizer sniffed the `.model` kind: count_text(\"{SAMPLE}\") -> {count} \
         tokens, estimated {}",
        selection.estimated()
    ));
    banner::note(
        "a `.model` table carries no name, so the selection falls back to the artifact path as \
         its tokenizer id; that path is not printed here",
    );
    Ok(())
}

/// Section 3: the selection ladder, one function per rung.
fn ladder(json_path: &Path, root: &Path) -> Result<()> {
    banner::title("the selection ladder");
    banner::note("every rung is synchronous and cache-only: no rung touches the network");
    explicit_spec(json_path, root)?;
    native_without_artifact()?;
    unregistered_model()?;
    family_match()?;
    artifact_chain(json_path, root)?;
    Ok(())
}

/// Rung 1: an explicit spec whose artifact path resolves through the manager.
fn explicit_spec(json_path: &Path, root: &Path) -> Result<()> {
    let spec = TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-tok".to_owned(),
        engine: TokenizerEngine::Native,
        tokenizer_id: Some("tiny-bpe".to_owned()),
        artifact_path: Some(json_path.display().to_string()),
        ..TokenizerSpec::default()
    };
    let manager = TokenizerArtifactManager::new().with_cache_dir(root.join("cache"));
    let selection = TokenizerSelector::new("acme", "acme-tok", Some(spec), None)
        .with_artifact_manager(manager)
        .select();
    ensure!(
        selection.source() == TokenizerSource::NativeTokenizer,
        "rung 1: a resolvable spec artifact is the native source"
    );
    ensure!(
        !selection.estimated() && selection.fallback_reason().is_none(),
        "rung 1: an exact artifact is not an estimate and has no reason"
    );
    selected(
        "rung 1 — an explicit spec with `artifact_path`, manager attached",
        &selection,
    );
    provenance(
        "measure(\"abc abc\") on rung 1",
        &selection.measure("abc abc"),
    );
    Ok(())
}

/// Rung 2: a `Native` engine with no artifact and no manager, for a known and
/// an unknown model.
fn native_without_artifact() -> Result<()> {
    let spec = TokenizerSpec {
        provider: "openai".to_owned(),
        model: "gpt-4o".to_owned(),
        engine: TokenizerEngine::Native,
        ..TokenizerSpec::default()
    };
    let selection = TokenizerSelector::new("openai", "gpt-4o", Some(spec), None).select();
    ensure!(
        selection.source() == TokenizerSource::TiktokenFallback
            && selection.fallback_reason() == Some("native_tokenizer_unavailable"),
        "rung 2a: a missing native artifact degrades to the model's tiktoken vocabulary"
    );
    selected(
        "rung 2a — `Native` engine, no artifact, a model with an embedded vocabulary",
        &selection,
    );

    let spec = TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-unlisted".to_owned(),
        engine: TokenizerEngine::Native,
        ..TokenizerSpec::default()
    };
    let selection = TokenizerSelector::new("acme", "acme-unlisted", Some(spec), None).select();
    ensure!(
        selection.source() == TokenizerSource::StringLengthFallback
            && selection.fallback_reason() == Some("native_tokenizer_unavailable"),
        "rung 2b: the same missing artifact degrades to the estimate for an unknown model"
    );
    selected(
        "rung 2b — `Native` engine, no artifact, an unknown model",
        &selection,
    );
    banner::note(
        "the `Native` engine never fails: it reports `native_tokenizer_unavailable` and continues \
         down the ladder",
    );
    Ok(())
}

/// Rung 3: no spec, no registry, an unknown model.
fn unregistered_model() -> Result<()> {
    let selection = TokenizerSelector::new("acme", "mystery-model", None, None).select();
    ensure!(
        selection.source() == TokenizerSource::StringLengthFallback
            && selection.fallback_reason() == Some("model_not_registered"),
        "rung 3: an unknown model falls to the three-characters-per-token estimate"
    );
    ensure!(
        selection.tokenizer().is_none(),
        "rung 3: the estimate names no tokenizer"
    );
    let estimate = selection.count_text(ESTIMATE_SAMPLE);
    ensure!(
        estimate == 4,
        "rung 3: twelve characters at three characters per token is four tokens"
    );
    selected(
        "rung 3 — no spec, no registry, an unregistered model",
        &selection,
    );
    banner::step(&format!(
        "the estimate is 3 characters per token: \"{ESTIMATE_SAMPLE}\" (12 chars) -> {estimate} \
         tokens"
    ));
    Ok(())
}

/// Rung 4: a family registry match. The requested variant must itself resolve to
/// an embedded vocabulary (here `gpt-4-turbo` -> `cl100k_base`), because the
/// selector can only build the family rung from a vocabulary it knows.
fn family_match() -> Result<()> {
    let registry = TokenizerRegistry::new(vec![TokenizerSpec::new("openai", "gpt-4")]);
    let hit = registry
        .resolve("openai", "gpt-4-turbo")
        .context("the registry matches the variant against its base")?;
    ensure!(
        hit.kind == TokenizerMatchKind::Family,
        "rung 4: `gpt-4-turbo` continues the registered `gpt-4` base"
    );
    banner::step(&format!(
        "registry.resolve(\"openai\", \"gpt-4-turbo\") -> {:?} for the base `gpt-4`",
        hit.kind
    ));
    let selection = TokenizerSelector::new("openai", "gpt-4-turbo", None, Some(registry)).select();
    ensure!(
        selection.source() == TokenizerSource::FamilyTokenizerFallback
            && selection.fallback_reason() == Some("model_variant_tokenizer_family"),
        "rung 4: a family match keeps the family source and reason"
    );
    ensure!(
        selection.tokenizer() == Some("cl100k_base") && selection.estimated(),
        "rung 4: the sibling vocabulary counts, but the result stays an estimate"
    );
    selected("rung 4 — a family registry match", &selection);
    banner::note(
        "`estimated` stays true because the vocabulary is a sibling's, not the variant's own \
         artifact; with a manager attached the same rung would use a cached family artifact",
    );
    Ok(())
}

/// Rung 5: the spec's own artifact is missing, its first family fallback
/// resolves, and the second fallback is never read.
fn artifact_chain(json_path: &Path, root: &Path) -> Result<()> {
    let second = root.join("tiny-bpe-2").join("tokenizer.json");
    write_file(&second, BPE_JSON_SECOND)?;
    let spec = TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-tok".to_owned(),
        engine: TokenizerEngine::Auto,
        artifact_path: Some(
            root.join("not-installed")
                .join("tokenizer.json")
                .display()
                .to_string(),
        ),
        compatible_fallbacks: vec![
            CompatibleTokenizerSpec {
                model: "tiny-bpe".to_owned(),
                engine: TokenizerEngine::Native,
                tokenizer_id: Some("tiny-bpe".to_owned()),
                artifact_path: Some(json_path.display().to_string()),
                ..CompatibleTokenizerSpec::default()
            },
            CompatibleTokenizerSpec {
                model: "tiny-bpe-2".to_owned(),
                engine: TokenizerEngine::Native,
                tokenizer_id: Some("tiny-bpe-2".to_owned()),
                artifact_path: Some(second.display().to_string()),
                ..CompatibleTokenizerSpec::default()
            },
        ],
        ..TokenizerSpec::default()
    };
    let manager = TokenizerArtifactManager::new().with_cache_dir(root.join("cache"));
    let selection = TokenizerSelector::new("acme", "acme-tok", Some(spec), None)
        .with_artifact_manager(manager)
        .select();
    ensure!(
        selection.source() == TokenizerSource::FamilyTokenizerFallback
            && selection.fallback_reason() == Some("target_tokenizer_unavailable"),
        "rung 5: a missing target artifact reports the one-hop family reason"
    );
    ensure!(
        selection.tokenizer() == Some("tiny-bpe"),
        "rung 5: the first fallback is the one that resolved"
    );
    ensure!(
        selection.count_text("abc") == 1,
        "rung 5: the fallback artifact really counts"
    );
    selected(
        "rung 5 — target artifact missing, `compatible_fallbacks[0]` resolves",
        &selection,
    );
    banner::note(
        "the chain is one hop: `compatible_fallbacks[1]` also points at a decodable \
         tokenizer.json, but it is never read — the selected id stays `tiny-bpe`",
    );
    Ok(())
}

/// Section 4: the artifact manager's cache target, two misses, and one hit.
fn artifact_cache(root: &Path) -> Result<()> {
    banner::title("the artifact cache");
    let cache = root.join("cache");
    let spec = cache_spec();
    let target = cache_misses(&spec, &cache, root)?;
    cache_hit(&spec, &target, &cache, root)?;
    banner::note(
        "the layout is `<cache>/huggingface/<tokenizer_id>/<revision>/tokenizer.json`; with no \
         revision declared the `main` branch is assumed",
    );
    banner::note("the example points the cache at its temp dir, so nothing is shared between runs");
    Ok(())
}

/// The hub spec whose artifact the cache section fills.
fn cache_spec() -> TokenizerSpec {
    TokenizerSpec {
        provider: "huggingface".to_owned(),
        model: "acme/tiny-bpe".to_owned(),
        engine: TokenizerEngine::Native,
        tokenizer_id: Some("acme/tiny-bpe".to_owned()),
        source: Some("huggingface".to_owned()),
        ..TokenizerSpec::default()
    }
}

/// Computes the cache target and shows both miss codes; returns that path.
fn cache_misses(spec: &TokenizerSpec, cache: &Path, root: &Path) -> Result<PathBuf> {
    let manager = TokenizerArtifactManager::new()
        .with_cache_dir(cache.to_path_buf())
        .with_enable_download(false);
    let target = manager
        .cache_target(spec)
        .context("a hub spec has a computed cache target")?;
    ensure!(
        !target.exists(),
        "`cache_target` computes the path, it does not create it"
    );
    banner::step(&format!(
        "cache_target(spec) -> {} (computed, not created)",
        relative(&target, root)?
    ));

    // First miss: downloads enabled, but the synchronous seam never waits for one.
    let eager = TokenizerArtifactManager::new()
        .with_cache_dir(cache.to_path_buf())
        .with_enable_download(true);
    ensure!(
        eager.resolve_cached(spec).is_none(),
        "an uncached artifact cannot be resolved by the cache-only seam"
    );
    let code = eager.last_error().unwrap_or_default();
    ensure!(
        code == TOKENIZER_NOT_IN_LOCAL_CACHE_CODE,
        "with downloads enabled the miss is reported as `{TOKENIZER_NOT_IN_LOCAL_CACHE_CODE}`"
    );
    banner::step(&format!(
        "resolve_cached with downloads enabled -> None, last_error() `{code}`"
    ));

    // Second miss: the cache-only manager has downloads off.
    ensure!(
        manager.resolve_cached(spec).is_none(),
        "the cache is still empty"
    );
    let code = manager.last_error().unwrap_or_default();
    ensure!(
        code == REMOTE_DOWNLOAD_DISABLED_CODE,
        "with downloads disabled the miss is reported as `{REMOTE_DOWNLOAD_DISABLED_CODE}`"
    );
    banner::step(&format!(
        "resolve_cached with downloads disabled -> None, last_error() `{code}`"
    ));

    // `with_offline(true)` is the stricter switch: it refuses a download even
    // when downloads are enabled.
    let strict = TokenizerArtifactManager::new()
        .with_cache_dir(cache.to_path_buf())
        .with_enable_download(true)
        .with_offline(true);
    ensure!(
        strict.resolve_cached(spec).is_none(),
        "the cache is still empty"
    );
    let code = strict.last_error().unwrap_or_default();
    ensure!(
        code == REMOTE_DOWNLOAD_DISABLED_CODE,
        "offline mode reports the stricter code even with downloads enabled"
    );
    banner::step(&format!(
        "with_offline(true) with downloads enabled -> None, last_error() `{code}`"
    ));
    Ok(target)
}

/// Writes the artifact into the computed target, then resolves and selects
/// through the same manager.
fn cache_hit(spec: &TokenizerSpec, target: &Path, cache: &Path, root: &Path) -> Result<()> {
    write_file(target, BPE_JSON)?;
    banner::step(&format!("wrote {} into the cache", relative(target, root)?));
    let manager = TokenizerArtifactManager::new()
        .with_cache_dir(cache.to_path_buf())
        .with_enable_download(false);
    let hit = manager
        .resolve_cached(spec)
        .context("the cached artifact validates")?;
    ensure!(
        hit.as_path() == target,
        "the hit is the computed cache path"
    );
    ensure!(
        manager.last_error().is_none(),
        "a successful resolve clears the last error"
    );
    banner::step(&format!(
        "resolve_cached -> Some({}), last_error() None",
        relative(&hit, root)?
    ));

    let selection =
        TokenizerSelector::new("huggingface", "acme/tiny-bpe", Some(spec.clone()), None)
            .with_artifact_manager(manager)
            .select();
    ensure!(
        selection.source() == TokenizerSource::NativeTokenizer
            && !selection.estimated()
            && selection.fallback_reason().is_none(),
        "the selector uses the cached native artifact and reports no fallback"
    );
    ensure!(
        selection.count_text("abc") == 1,
        "the cached artifact counts like the artifact from section 1"
    );
    selected(
        "the selector with the manager attached uses the cached artifact",
        &selection,
    );
    provenance(
        "measure(\"abc abc\") from the cache",
        &selection.measure("abc abc"),
    );
    Ok(())
}

/// Prints one selection's provenance fields.
fn selected(label: &str, selection: &SelectedTokenizer) {
    banner::step(&format!(
        "{label}: source {:?}, estimated {}, fallback_reason {:?}, tokenizer {:?}, model {:?}",
        selection.source(),
        selection.estimated(),
        selection.fallback_reason(),
        selection.tokenizer(),
        selection.model()
    ));
}

/// Prints one measurement's provenance fields: count, source, estimate flag,
/// tokenizer, model, and fallback reason.
fn provenance(label: &str, measurement: &TokenMeasurement) {
    banner::step(&format!(
        "{label}: {} tokens, source {:?}, estimated {}, tokenizer {:?}, model {:?}, \
         fallback_reason {:?}",
        measurement.tokens,
        measurement.source,
        measurement.estimated,
        measurement.tokenizer,
        measurement.model,
        measurement.fallback_reason
    ));
}

/// Writes `contents` to `path`, creating the parent directories.
fn write_file(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// The path below `root`, with `/` separators, so no temp path reaches stdout.
fn relative(path: &Path, root: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("{} is outside the example's temp directory", path.display()))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}
