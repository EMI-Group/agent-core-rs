# tokenizers

## Intent

Owns tokenizer resolution and the token-counting seam of the context engine: model names map to a `tiktoken` vocabulary, declared specs resolve through a registry, and two native artefact kinds are decoded in pure Rust — a `HuggingFace` `tokenizer.json` and a `tiktoken` `.model` BPE vocabulary.
A `tokenizer.json` is decoded into its whole pipeline — normalizer, added tokens, pre-tokenizer, model, single-sequence post-processor and decoder — so encoding and decoding follow the artefact rather than a fixed rule.
The node families are implemented offline: the `regex` crate compiles a `tokenizer.json` pattern and `unicode-normalization` supplies the `NFC`/`NFD`/`NFKC`/`NFKD` forms.
`TokenizerArtifactManager` resolves, validates, caches and (only when explicitly enabled) downloads those artefacts, and `TokenizerSelector` walks the ladder that joins the two.
Ports the token selection of the Python `core/context_engine/token/` package to Rust without its optional runtime dependencies (`tiktoken`, `tokenizers`, `huggingface_hub`, `modelscope`).
Everything here is offline by construction: BPE vocabularies come from `tiktoken-rs`'s embedded data, native artefacts are decoded from bytes already on disk, and the manager's two download switches both default to off.

## API Surface

`tokenizers.rs` (the parent module) declares `artifacts`, `native`, `selector` (public) plus `counter` and `spec` (private), and re-exports the public surface:

- `spec.rs` — `ModelEncoding` (`name`, `from_model`), `TokenizerEngine` (`Auto` (default) | `Tiktoken` | `Native`), `TokenizerSpec` (`provider`, `model`, `family`, `engine`, `tokenizer_id` with the `id` alias, `source`, `revision`, `artifact_path`, `sha256`, `compatible_fallbacks`; `new`, `validate`), `CompatibleTokenizerSpec` (the same shape minus `provider`/`family`/`compatible_fallbacks`, `From<TokenizerSpec>`), `TokenizerMatchKind` (`Exact` | `Family`), `TokenizerMatch` (`spec`, `kind`), `TokenizerRegistry` (`new`, `register`, `is_empty`, `len`, `resolve(provider, model)`).
- `counter.rs` — `TokenizerSource` (`NativeTokenizer` | `FamilyTokenizerFallback` | `Tiktoken` | `TiktokenFallback` | `StringLengthFallback`; `is_native`, `estimation_note`), `TokenMeasurement` (`tokens`, `source`, `estimated`, `tokenizer`, `model`, `fallback_reason`, `estimation_note`), `ModelTokenCounter` (`new`, `model`, `encoding`, `tokenizer`, `source`, `estimated`, `fallback_reason`, `measure`, `measure_messages`, `estimation_note`), `SelectedTokenizer` (`from_native`, `from_native_with`, `model`, `tokenizer`, `source`, `estimated`, `fallback_reason`, `measure`, `measure_messages`, `estimation_note`, plus `impl TokenCounter`).
- `selector.rs` — `TokenizerSelector` (`new`, `with_spec`, `with_registry`, `with_artifact_manager`, `provider`, `model`, `artifact_manager`, `select`), `load_native_tokenizer(path)`, and the public `NATIVE_ARTIFACT_UNDECODABLE_REASON`; the other reason spellings are private string constants.
- `native/` — `NativeTokenizer` (`from_json_str`, `load`, `from_tiktoken_model`, `from_tiktoken_model_file`, `from_tiktoken_model_file_with_model`, `from_tiktoken`, `encode`, `decode`, `decode_model_level`, `vocab_size`, `kind`, `warnings`, `tokenizer_id`, `special_tokens`, plus `impl TokenCounter`), `NativeTokenizerKind` (`Bpe`, `WordPiece`, `Unigram`, `WordLevel`, `Tiktoken`, where `as_str` is the artefact's `"type"` tag), `byte_level_alphabet()`, and the public `tiktoken_model` module (`TiktokenModel`, `PretokenizePattern` (`Kimi` default, `Gpt2`), `TiktokenSpecials`).
- `native::load_artifact_file` is the one crate-visible file-decode dispatch: extension first (`.model` / `.json`), then content sniffing, so the artefact manager validates a file exactly the way the selector later decodes it.
- `SelectedTokenizer::estimated()` is `false` only for `TokenizerSource::NativeTokenizer`, so a family or one-hop fallback artefact stays `estimated() == true`.
- `load_native_tokenizer(path)` builds an exact native selection from a file and reports `source() == NativeTokenizer`, `estimated() == false` and no fallback reason; it goes through the same extension-then-content dispatch and leaves a failed decode to the typed error.

Provenance of a measurement is three facts, not one:

- `estimated` is the broad flag: `true` for every source but `NativeTokenizer`, so an exact `tiktoken` vocabulary is an estimate too.
- `fallback_reason` is the narrow one: it names a degraded strategy (`tiktoken_unavailable`, `tiktoken_disabled`, `model_not_registered`, `native_tokenizer_unavailable`, `native_artifact_undecodable`, `target_tokenizer_unavailable`, `model_variant_tokenizer_family`) and is empty when a known strategy resolved.
- `estimation_note()` is the prose explanation, so an estimate stays explained even when no reason was recorded, and a source that cannot describe itself gets a generic sentence.
- `TokenizerSelector::select` emits exactly one `tracing::warn!` per degraded selection, naming the model, the source, the reason and the note, and a refused artefact emits one `warn!` naming the artefact and the typed error.

`artifacts/` (the `openjiuwen_context::tokenizers::artifacts` module) exposes:

- `TokenizerArtifactManager` — `new`, `with_cache_dir`, `with_enable_download`, `with_offline`, `with_endpoint`, `with_request_timeout`, `cache_dir`, `enable_download`, `offline`, `endpoint`, `modelscope_endpoint`, `request_timeout`, `last_error`, `last_error_detail`, `cache_target`, `async resolve`, `resolve_cached`, `async download`.
- `last_error()` is always a compact code from `ARTIFACT_ERROR_CODES` (`None` until a resolution fails), while `last_error_detail()` returns the whole typed message — still prefixed with that code — or `None` for a gate the manager decided itself (a disabled download, a missing local artefact, a refused URL).
- Cache helpers re-exported from `artifacts::cache`: `default_cache_dir`, `cache_dir_from`, `expand_home`, `artifact_name`, `find_artifact`, `relative_artifact_id`, `sha256_file`, `sha256_hex`, `CACHE_DIR_ENV`, `DEFAULT_CACHE_SUBDIR`, `HF_ENDPOINT_ENV`, `MAX_ARTIFACT_SEARCH_DEPTH`.
- Lock: `ArtifactLock` (`acquire`, `acquire_within`, `path`), `LOCK_WAIT_LIMIT`, `STALE_LOCK_AFTER`.
- Endpoint defaults: `DEFAULT_HUGGINGFACE_ENDPOINT` (`https://hf-mirror.com`), `DEFAULT_MODELSCOPE_ENDPOINT` (`https://modelscope.cn`), `DEFAULT_TOKENIZER_REQUEST_TIMEOUT` (10 s), `MAX_ARTIFACT_BYTES` (64 MiB), `HUGGINGFACE_SOURCE`, `MODELSCOPE_SOURCE`, `PROVIDER_OFFICIAL_SOURCE`, `HUGGINGFACE_DEFAULT_REVISION` (`main`), `MODELSCOPE_DEFAULT_REVISION` (`master`).
- Compact codes: `ARTIFACT_ERROR_CODES` holds thirteen — `local_tokenizer_artifact_missing`, `tokenizer_json_missing`, `tiktoken_model_missing`, `tokenizer_id_missing`, `remote_download_disabled`, `tokenizer_network_unavailable`, `tokenizer_not_in_local_cache`, `tokenizer_revision_unavailable`, `tokenizer_download_failed`, `tokenizer_sha256_mismatch`, `remote_url_not_allowed`, `provider_official_adapter_not_configured`, `tokenizer_model_type_unsupported` — each with its own exported constant.
- The validation helpers live in the private `artifacts::validate` module (existence, declared digest, body decode, the async `spawn_blocking` wrapper, digest verification).

## Constraints

- Dependencies are `serde_json`, `base64` (a `.model` vocabulary is base64 token bytes), `sha2` (artefact validation), `regex` (pattern nodes), `unicode-normalization` (the Unicode forms, already locked transitively and pinned inline in `Cargo.toml`), `tracing`, `uuid` and the standard library; HTTP goes through `openjiuwen_common::http::HttpClient`.
- The manager is network-free unless `with_enable_download(true)` and `!with_offline(true)` are both set; `TokenizerSelector::select` never downloads because it is synchronous and cache-only.
- `resolve`/`download` never panic: `resolve` returns `Option` and records a compact code plus the typed detail, `download` returns a typed `Error` whose message starts with the same code.
- A `://` tokenizer id is refused before any source is read; a download id must stay repository-relative (`artifacts::relative_artifact_id`), which closes the SSRF/escape surface Python guards the same way.
- A load is a one-off per context/model binding; `encode`, `decode` and `count_text` are pure in-memory computation and never touch the filesystem or the network.
- `tokenizer.json` model types outside BPE/WordPiece/Unigram/WordLevel are `Error::Unsupported` naming the type; malformed documents and missing vocabularies are `Error::Validation`; a missing file is `Error::NotFound`.
- Artefact validation is a refusal, not a downgrade: a downloaded or hand-placed body declaring an undecoded model type is `Error::Unsupported` carrying `tokenizer_model_type_unsupported` and the declared type, such a body is never written to the cache, and malformed bytes are `Error::Validation` carrying `tokenizer_download_failed`.
- `provider_official` is refused with `Error::Unsupported` whose message is `provider_official_adapter_not_configured: <vendor>:<tokenizer_id> needs a <vendor> tokenizer adapter (no provider-hosted tokenizer client is shipped)`; the vendor comes from the spec's `provider`, then the `vendor/` prefix of the id, then `provider-official`, and an empty id renders `<unnamed>`.
- Everything a load cannot reproduce exactly (`Nmt`/`Precompiled` normalization, an unknown normalizer/pre-tokenizer/decoder node, an uncompilable pattern, a post-processor template piece whose special token declares no ids) is recorded in `NativeTokenizer::warnings()` and degrades to a coarser behaviour — never a failed load, because a token count stays useful.
- `TokenCounter::count_text` is `encode().len()`, so the framing a declared post-processor adds is part of the count; message framing uses the workspace default (a fixed per-message overhead) rather than Python's `<|start|>role` convention, so a native count stays comparable with every other counter.
- No `unsafe`, no `println!`, no `unwrap`/`expect` in library code; errors are typed `openjiuwen_common::error::Error`.
- Decoding includes special tokens (unlike `HuggingFace`'s `skip_special_tokens = true` default) and skips ids the artefact does not declare; a `.model` vocabulary allows special tokens by default (`allowed_special="all"` parity).
- Each file keeps one responsibility: `spec.rs`, `counter.rs` and `selector.rs` hold the split public surface, `artifacts/` splits cache, lock, validation and the manager, and `native/` keeps one pipeline stage (plus `tiktoken_model/`) per module.

## Design Decisions

- The artefact is decoded into immutable tables once; `NativeModel` dispatches BPE, WordPiece, Unigram and WordLevel behind one `encode_piece` seam, and a `Tiktoken` artefact bypasses that pipeline with its own pre-tokenization and merge loop.
- BPE applies the lowest-ranked merge first, left to right, until no ranked pair remains; `continuing_subword_prefix` lives inside the initial symbols and `end_of_word_suffix` is appended before the final lookup.
- Model-level BPE decoding undoes the merge-time spellings: every piece loses its `continuing_subword_prefix` and `end_of_word_suffix`, a `<0xXX>` token contributes its byte when the model declares `byte_fallback` (consecutive byte tokens form one UTF-8 run decoded with replacement characters, so a character cut mid-codepoint still decodes), and with `fuse_unk` a run of unknown-token spellings collapses to the first spelling of that run.
- Unigram model-level decoding concatenates spellings and reassembles `<0xXX>` runs under `byte_fallback`, leaving metaspace (`▁`) handling to the declared decoder node; WordLevel concatenates whole pieces; WordPiece has no model-level rule (its `##` prefix is only unambiguous through the decoder node), so `decode_model_level` falls back to the declared decoder there.
- Unigram runs a forward dynamic program over the declared log-probabilities and prices the unknown edge at `min_score - unk_penalty`, reading the penalty from `unk_penalty` or `unigram_unk_penalty` and defaulting to `10.0`.
- The unknown edge is inserted only where no single-character vocabulary token covers the position, so a declared multi-character token is not automatically worse than a run of unknown characters; a vocabulary edge wins a score tie against it, and between two equally scored vocabulary edges the shorter one keeps the position.
- Unknown runs respect `fuse_unk` (default `true`: one unknown id per run; `false`: one per character) or become `<0xXX>` byte tokens under `byte_fallback`.
- Two Unigram approximations remain deliberate: the lattice walks characters instead of the reference's byte trie (a token is matched by scalar count, not a shared prefix walk), and the unknown run is emitted by scanning the chosen path instead of from the reference's node graph.
- Byte-level pre-tokenization follows `HuggingFace`'s order: the GPT-2 pattern splits the raw text first and every piece is byte-mapped afterwards, so a space stays a delimiter and `hi there!` becomes `hi` + `Ġthere` + `!`; the pattern itself is reproduced by hand in `pretokenizer/gpt2.rs` (contractions, ` ?<letters>`, ` ?<numbers>`, ` ?<other>`, `\s+(?!\S)`, `\s+`).
- `byte_fallback` spells an unresolvable piece as `<0xXX>` tokens over its UTF-8 bytes; it is never combined with a byte-level pre-tokenizer in real artefacts.
- WordPiece is greedy longest-match with `##`-prefixed continuations and `max_input_chars_per_word` as a hard unknown cut-off.
- Added tokens are matched longest-first after normalization, honouring `single_word`, `lstrip` and `rsw`; special tokens are ordinary added tokens, so they always encode as one id.
- The decoder reproduces `Plain` (join with spaces), `Fuse`, `ByteLevel`, `Metaspace` (`replacement`/`prepend_scheme`/`add_prefix_space`), `WordPiece`, literal and regex `Replace`, `Strip` (`content`/`start`/`stop`), `ByteFallback`, `BPEDecoder`, `CTC` (`pad_token`/`word_delimiter_token`/`cleanup`), the four Unicode normalizers and `Sequence`; an uncompilable regex or an unknown node warns and leaves that step out.
- A declared decoder node always wins, and an artefact that declares none (or only `Plain`) joins the token spellings with a single space, which is what `HuggingFace`'s `tokenizers` — and therefore the Python reference — does.
- `NativeTokenizer::decode_model_level` is the explicit, documented opt-in for the byte-aware model-level reassembly a byte-level artefact without a decoder node needs to round-trip; `decode` never takes that shortcut, so a caller sees the artefact's own pipeline unless it asks otherwise.
- `encode` applies the declared single-sequence post-processor: `TemplateProcessing` contributes its `single` template, `BertProcessing`/`RobertaProcessing` their cls/sep pair, `ByteLevel` contributes nothing (it only trims offsets in the reference), and `Sequence` chains the stages; any other shape warns instead of guessing, and a template piece whose special token declares no ids — including a malformed `ids` entry — is warned and left out.
- An empty string encodes to no ids and therefore no framing, and an artefact that declares no post-processor gains no framing at all.
- The normalizer reproduces `Identity`, `Sequence`, `Lowercase`/`LowerCase`, `NFC`/`NFD`/`NFKC`/`NFKD`, `BertNormalizer` (`clean_text`, `handle_chinese_chars`, `strip_accents`, `lowercase`), `Strip`, `Prepend`, a literal `Replace` (`{"String": …}`) and a regex `Replace` (`{"Regex": …}` or a bare string, compiled with the `regex` crate so Rust's `$1`/`${name}` replacement syntax applies).
- `Nmt` and `Precompiled` still warn by name and leave the text unnormalized, because they need the `SentencePiece` compiled character map; an unknown node type warns with its own name as well, and a `Replace` regex that does not compile warns with the pattern source and the `regex` error and leaves that step out.
- The pre-tokenizer reproduces `BERT`, `Whitespace`/`WhitespaceSplit`, `Punctuation`, `Digits`, `ByteLevel`, `Metaspace`, a literal or regex `Split` across all five `SplitBehavior` values (`Removed`, `Isolated`, `MergedWithPrevious`, `MergedWithNext`, `Contiguous`) plus `invert`, `Replace` (the rewritten text stays one piece), `CharDelimiterSplit` (the delimiter is dropped, an empty or missing delimiter warns) and the `NFC`/`NFD`/`NFKC`/`NFKD` wrappers (normalize, then split on whitespace), plus `Sequence`.
- A `Split`/`Replace` regex that does not compile warns with the pattern and the error and degrades to whitespace splitting (or to no pre-tokenization), and any other unknown pre-tokenizer node warns with its name and leaves the text unsplit.
- `is_punctuation` is ASCII punctuation plus the Unicode `P` category, and the `\p{P}` check is compiled once in a `OnceLock`.
- A `.model` line is `<base64 token bytes> <rank>`; blank lines and `#` comments are ignored, ranks order the merges, and encoding turns each pre-tokenized piece into UTF-8 bytes and repeatedly merges the adjacent pair with the lowest rank.
- The Kimi pre-tokenization rule is the default for every `.model` vocabulary (matching Python's `TiktokenModelCounter`), with `PretokenizePattern::Gpt2` available; Unicode classes come from `char::is_alphabetic`/`is_numeric`/`is_lowercase` plus a combining-mark range table instead of the full `\p{Han}`/`\p{L}`/`\p{M}` tables.
- A `.model` byte with no single-byte vocabulary entry is skipped and reported as a warning, where Python's `tiktoken` would raise, and decoding a vocabulary whose ids are not declared returns no text for that id.
- `.model` special tokens come from the sibling `tokenizer_config.json` when it declares `added_tokens_decoder`; otherwise the built-in Kimi map is used, with K3 selected when the name contains `kimi-k3` and K2 for anything else.
- Cache-dir precedence is explicit argument → `OPENJIUWEN_TOKENIZER_CACHE_DIR` → `$HOME/.cache/openjiuwen/tokenizers` → the relative `.cache/openjiuwen/tokenizers` when `HOME` is unset (documented fallback, never a temp dir: a cache should survive the process).
- The artefact name derives from the engine (`tiktoken.model` for `Tiktoken`, else `tokenizer.json`); an `artifact_path` may name the file or a directory, and a directory is searched deterministically — direct `<dir>/<artifact name>`, then a sorted depth-first walk (≤4 levels, ≤64 hits) taking the lexicographically first absolute match, instead of Python's non-deterministic `rglob`.
- Downloads land at `<cache>/huggingface/<id>/<revision>/<artifact>` and `<cache>/models/<id>/<revision>/<artifact>`; the `models` prefix is also one of the three offline `modelscope` snapshot layouts (`<cache>/<id>`, `<cache>/models/<id>`, `<cache>/hub/models/<id>`), so a downloaded artefact is found again without a second request.
- Hub URLs are `{endpoint}/<id>/resolve/<revision>/<artifact>` (`huggingface`, revision default `main`) and `{endpoint}/models/<id>/resolve/<revision>/<artifact>` (`modelscope`, default `master`); the single-artifact GET replaces Python's `snapshot_download(allow_patterns=...)`, so no sibling config file is fetched unless it is needed for specials.
- One download at a time is guaranteed by an advisory lock marker named `sha256(source, tokenizer_id, revision, sha256, engine).lock` in the cache dir: `create_new` acquisition, `tokio::time::sleep` exponential backoff (10 ms → 250 ms, 10 s budget), stale removal after 120 s of untouched mtime, and removal on drop.
- A marker whose mtime lies in the future is clock skew rather than staleness: it is logged as such and still treated as live, so a skewed writer's lock is not yanked from under it.
- Bytes are validated before anything is written (UTF-8, then the engine's decode, then the declared sha256), written to a `.<uuid>.tmp` file in the target directory and renamed into place, so a reader never sees a partial artefact and a failed download leaves nothing behind.
- Every path the manager hands back is validated in the order existence → declared digest → body, so a tampered file reports `tokenizer_sha256_mismatch` even when its body is also undecodable; the body check uses `native::load_artifact_file`, the same dispatch the selector uses later.
- A cached artefact that cannot be read or decoded is logged with its path and the error and treated as a cache miss, so the caller downloads it again instead of reusing bytes nobody could digest; a validation task whose blocking read panics is logged with the join error and mapped to `tokenizer_download_failed`.
- `resolve`/`resolve_cached`/`download` share one resolution order: `artifact_path` (file or dir) → on-disk `tokenizer_id` → engine-specific missing code for a declared-but-unusable `artifact_path` → `://`/relative-id rejection → `huggingface`/`modelscope` cache → gate (`remote_download_disabled` when downloads are off, `tokenizer_not_in_local_cache` when offline or when a synchronous caller cannot wait) → download.
- `provider_official` always answers the adapter refusal, for `resolve`, `resolve_cached` and `download` alike, and it records both the compact code and the vendor/id detail.
- The selector ladder is spec artefact → spec's first `compatible_fallbacks` entry (one hop, reporting the primary refusal when there is one, else `target_tokenizer_unavailable`) → for a registry hit, its artefact then its first fallback (`family_tokenizer_fallback`/`model_variant_tokenizer_family`) → tiktoken → estimate; `Auto` therefore prefers a native artefact, and a `Native` engine with no resolvable artefact reports `native_tokenizer_unavailable` rather than failing.
- An artefact that resolves but cannot be decoded is refused with `native_artifact_undecodable` instead of a silent downgrade: the rung the ladder lands on (a `tiktoken` vocabulary, the estimate, or a one-hop sibling) carries that reason, so the degradation names its real cause.
- `ContextConfig` carries `tokenizer_cache_dir`, `enable_tokenizer_download` (default `false`) and `tokenizer_offline` (default `false`), builds the manager in `tokenizer_artifact_manager()`, and rejects the contradictory `enable_tokenizer_download && tokenizer_offline` pair in `validate()`.

## Routing Table

- `./tokenizers.rs` → module declarations and the public re-exports of the surface above
- `./spec.rs` → `ModelEncoding`, `TokenizerEngine`, `TokenizerSpec`, `CompatibleTokenizerSpec`, registry and match kinds
- `./counter.rs` → `TokenizerSource`, `TokenMeasurement`, `ModelTokenCounter`, `SelectedTokenizer` (including the native backing), estimation notes
- `./selector.rs` → `TokenizerSelector` and its artefact/tiktoken/estimate ladder, refusal reasons, `load_native_tokenizer`
- `./artifacts.rs` → module docs, cache-layout contract, `mod` declarations and the public re-exports
- `./artifacts/manager.rs` → `TokenizerArtifactManager` (resolution order, downloads, codes, recorded failures), endpoint/code constants, inline tests
- `./artifacts/validate.rs` → existence/digest/body validation, the `spawn_blocking` wrapper, undecodable-model refusal
- `./artifacts/cache.rs` → cache-dir precedence, artefact naming, deterministic directory search, relative ids, sha256
- `./artifacts/lock.rs` → advisory download lock (`create_new` marker, async backoff, stale expiry, clock-skew reporting)
- `./native/mod.rs` → `NativeTokenizer` pipeline (normalize → added tokens → pre-tokenize → model → post-processor), decode entry points, the post-processor module, `load_artifact_file`, `NativeTokenizerKind`, `byte_level_alphabet`, both `from_tiktoken*` constructors, counter impl
- `./native/hf_json.rs` → `tokenizer.json` serde shapes and polymorphic-node helpers (kind, literal/regex pattern resolution, pattern descriptions)
- `./native/model.rs` → model dispatch and model-level decode, `WordLevel`, vocabulary tables, `<0xXX>` helpers
- `./native/bpe.rs` → byte-pair encoding, merge lists, continuation prefix, byte fallback, model-level decode
- `./native/wordpiece.rs` → WordPiece greedy segmentation
- `./native/unigram.rs` → Unigram-lite dynamic program, unknown pricing, `fuse_unk`, model-level decode
- `./native/normalizer.rs` → `Identity`/`Sequence`/lowercase/`BERT`/`Strip`/`Prepend`/literal and regex `Replace`/`NFC`–`NFKD` normalization and its warnings
- `./native/pretokenizer.rs` → `BERT`, whitespace, punctuation, digits, byte-level, metaspace, literal/regex `Split`, `Replace`, `CharDelimiterSplit`, the Unicode wrappers, `Sequence`
- `./native/pretokenizer/gpt2.rs` → the hand-reproduced GPT-2 split pattern and the split-then-byte-map byte-level rule
- `./native/added_tokens.rs` → added/special token matching and chunking
- `./native/decoder.rs` → plain, `Fuse`, byte-level, metaspace, word-piece, BPE, CTC, `ByteFallback`, `Strip`, literal and regex `Replace`, the Unicode normalizers, `Sequence`
- `./native/bytes.rs` → the byte↔unicode alphabet
- `./native/tiktoken_model/` → `.model` ranks/merges and encode/decode (`mod.rs`), Kimi and GPT-2 pre-tokenization (`pretokenize.rs`), special-token maps and `tokenizer_config.json` parsing (`specials.rs`)
- `./native/modeltests.rs` → shared test fixtures for the native loaders
- `../../tests/tokenizers*.rs` → the integration suites listed under Test Strategy

## Test Strategy

- `cargo test -p openjiuwen-context --offline` runs 374 library unit tests, 34 integration binaries and 16 doc tests — 1024 tests, green offline.
- This node's unit tests live beside the modules (`native/*` and `tiktoken_model/*` hold the decoder, normalization and pre-tokenizer node tests); the integration suites are `tokenizers` (26 tests: spec/registry/counter/selector), `tokenizers_native` (18), `tokenizers_native_models` (9: model-level decode and post-processing), `tokenizers_native_normalization` (6: the `NFC` family and regex `Replace`), `tokenizers_native_pretokenizer` (6: regex `Split`, `Replace`, `CharDelimiterSplit`), `tokenizers_tiktoken_model` (33) and `tokenizers_artifacts` (47).
- `estimate_honesty` (23 tests, crate-wide) pins that every non-native `TokenizerSource` reports `estimated = true` with a named reason and note while a native artefact counts exactly.
- Deterministic and offline: artefacts are built in memory with `serde_json::json!`, caches are `tempfile` directories, the download and endpoint paths use `wiremock`, and there is no live network test.
