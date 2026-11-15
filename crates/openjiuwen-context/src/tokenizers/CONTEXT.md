# tokenizers

## Intent

Owns tokenizer resolution and the token-counting seam of the context engine: model names map to a `tiktoken` vocabulary, declared specs resolve through a registry, and two native artefact kinds are decoded in pure Rust — a `HuggingFace` `tokenizer.json` and a `tiktoken` `.model` BPE vocabulary.
`TokenizerArtifactManager` resolves, validates, caches and (only when explicitly enabled) downloads those artefacts, and `TokenizerSelector` walks the ladder that joins the two.
Ports the token selection of the Python `core/context_engine/token/` package to Rust without its optional runtime dependencies (`tiktoken`, `tokenizers`, `huggingface_hub`, `modelscope`).
Everything here is offline by construction: BPE vocabularies come from `tiktoken-rs`'s embedded data, native artefacts are decoded from bytes already on disk, and the manager's two download switches both default to off.

## API Surface

`tokenizers.rs` (the parent module) declares `artifacts`, `native`, `selector` (public) plus `counter` and `spec` (private), and re-exports the public surface:

- `spec.rs` — `ModelEncoding` (`name`, `from_model`), `TokenizerEngine` (`Auto` (default) | `Tiktoken` | `Native`), `TokenizerSpec` (`provider`, `model`, `family`, `engine`, `tokenizer_id` with the `id` alias, `source`, `revision`, `artifact_path`, `sha256`, `compatible_fallbacks`; `new`, `validate`), `CompatibleTokenizerSpec` (the same shape minus `provider`/`family`/`compatible_fallbacks`, `From<TokenizerSpec>`), `TokenizerMatchKind` (`Exact` | `Family`), `TokenizerMatch` (`spec`, `kind`), `TokenizerRegistry` (`new`, `register`, `is_empty`, `len`, `resolve(provider, model)`).
- `counter.rs` — `TokenizerSource` (`NativeTokenizer` | `FamilyTokenizerFallback` | `Tiktoken` | `TiktokenFallback` | `StringLengthFallback`), `TokenMeasurement` (`tokens`, `source`, `estimated`, `tokenizer`, `model`, `fallback_reason`), `ModelTokenCounter` (`new`, `model`, `encoding`, `tokenizer`, `source`, `estimated`, `fallback_reason`, `measure`, `measure_messages`), `SelectedTokenizer` (`from_native`, `from_native_with`, `model`, `tokenizer`, `source`, `estimated`, `fallback_reason`, `measure`, `measure_messages`, plus `impl TokenCounter`).
- `selector.rs` — `TokenizerSelector` (`new`, `with_spec`, `with_registry`, `with_artifact_manager`, `provider`, `model`, `artifact_manager`, `select`) and `load_native_tokenizer(path)`.
- `native/` — `NativeTokenizer`, `NativeTokenizerKind`, `byte_level_alphabet`, and the public `tiktoken_model` module (`TiktokenModel`, `PretokenizePattern`, `TiktokenSpecials`).
- `SelectedTokenizer::estimated()` is `false` only for `TokenizerSource::NativeTokenizer`, so a family or one-hop fallback artefact stays `estimated() == true`.
- `load_native_tokenizer(path)` builds an exact native selection from a file and reports `source() == NativeTokenizer`, `estimated() == false` and no fallback reason; it picks the loader by extension (`.model` / `.json`) and then by content sniffing (a first meaningful line of `<base64-token> <rank>` is a `.model` vocabulary).

`artifacts/` (the `openjiuwen_context::tokenizers::artifacts` module) exposes:

- `TokenizerArtifactManager` — `new`, `with_cache_dir`, `with_enable_download`, `with_offline`, `with_endpoint`, `with_request_timeout`, `cache_dir`, `enable_download`, `offline`, `endpoint`, `modelscope_endpoint`, `request_timeout`, `last_error`, `cache_target`, `async resolve`, `resolve_cached`, `async download`.
- Cache helpers re-exported from `artifacts::cache`: `default_cache_dir`, `cache_dir_from`, `expand_home`, `artifact_name`, `find_artifact`, `relative_artifact_id`, `sha256_file`, `sha256_hex`, `CACHE_DIR_ENV`, `DEFAULT_CACHE_SUBDIR`, `HF_ENDPOINT_ENV`, `MAX_ARTIFACT_SEARCH_DEPTH`.
- Lock: `ArtifactLock` (`acquire`, `acquire_within`, `path`), `LOCK_WAIT_LIMIT`, `STALE_LOCK_AFTER`.
- Endpoint defaults: `DEFAULT_HUGGINGFACE_ENDPOINT` (`https://hf-mirror.com`), `DEFAULT_MODELSCOPE_ENDPOINT` (`https://modelscope.cn`), `DEFAULT_TOKENIZER_REQUEST_TIMEOUT` (10 s), `MAX_ARTIFACT_BYTES` (64 MiB), `HUGGINGFACE_SOURCE`, `MODELSCOPE_SOURCE`, `PROVIDER_OFFICIAL_SOURCE`, `HUGGINGFACE_DEFAULT_REVISION` (`main`), `MODELSCOPE_DEFAULT_REVISION` (`master`).
- Compact codes (`ARTIFACT_ERROR_CODES` plus the individual constants): `local_tokenizer_artifact_missing`, `tokenizer_json_missing`, `tiktoken_model_missing`, `tokenizer_id_missing`, `remote_download_disabled`, `tokenizer_network_unavailable`, `tokenizer_not_in_local_cache`, `tokenizer_revision_unavailable`, `tokenizer_download_failed`, `tokenizer_sha256_mismatch`, `remote_url_not_allowed`, `provider_official_adapter_not_configured`.

`native/` (the `openjiuwen_context::tokenizers::native` module) exposes exactly:

- `NativeTokenizer` — `from_json_str`, `load`, `from_tiktoken_model`, `from_tiktoken_model_file`, `from_tiktoken_model_file_with_model`, `from_tiktoken`, `encode`, `decode`, `vocab_size`, `kind`, `warnings`, `tokenizer_id`, `special_tokens`, plus `impl TokenCounter`.
- `NativeTokenizerKind` — `Bpe`, `WordPiece`, `Unigram`, `WordLevel`, `Tiktoken` (`as_str` is the artefact's `"type"` tag).
- `byte_level_alphabet()` — the 256-entry GPT-2/Llama byte↔unicode table.
- `tiktoken_model` — `TiktokenModel` (`from_text`, `from_file`, `from_file_with_model`, `with_pattern`, `with_special_tokens`, `special_token_id`, `mergeable_count`, `vocab_size`, `id_to_token`, `warnings`, `encode`, `decode`, `pretokenize`), `PretokenizePattern` (`Kimi` — the default — and `Gpt2`), `TiktokenSpecials` (`new`, `for_model`, `from_config_json`, `len`, `is_empty`, `token_id`, `text`, `texts`).

## Constraints

- Dependencies are `serde_json`, `base64` (a `.model` vocabulary is base64 token bytes), `sha2` (artefact validation), `tracing`, `uuid` and the standard library; HTTP goes through `openjiuwen_common::http::HttpClient`.
- The manager is network-free unless `with_enable_download(true)` and `!with_offline(true)` are both set; `TokenizerSelector::select` never downloads because it is synchronous.
- `resolve`/`download` never panic: `resolve` returns `Option` and records a compact code in `last_error`, `download` returns a typed `Error` whose message starts with the same code.
- A `://` tokenizer id is refused before any source is read; a download id must stay repository-relative (`artifacts::relative_artifact_id`), which closes the SSRF/escape surface Python guards the same way.
- A load is a one-off per context/model binding; `encode`, `decode` and `count_text` are pure in-memory computation and never touch the filesystem or the network.
- `tokenizer.json` model types outside BPE/WordPiece/Unigram/WordLevel are `Error::Unsupported` naming the type; malformed documents and missing vocabularies are `Error::Validation`; a missing file is `Error::NotFound`.
- `validate_body` tolerates that `Error::Unsupported`: a downloaded artefact declaring an undecoded model type is logged at `debug` and written to the cache (`Ok(())`), and the selector then logs the decode failure at `debug` and falls down its ladder to an estimate — only a non-UTF-8 body or an `Error::Validation` decode failure fails the download.
- `provider_official` surfaces as the compact code alone: `resolve`/`resolve_cached` return `None` with `last_error = provider_official_adapter_not_configured`, `download` returns `Error::Unsupported("provider_official_adapter_not_configured: provider-official tokenizers need a provider adapter")`, and the message names no vendor, provider id or `tokenizer_id`.
- Everything a load cannot reproduce exactly (regex splits, the NFC/NFD/NFKC/NFKD normalizer families, unknown pipeline nodes, BPE dropout) is recorded in `NativeTokenizer::warnings()` and degrades to a coarser behaviour — never a failed load, because a token count stays useful.
- `TokenCounter::count_text` is `encode().len()`; message framing uses the workspace default (a fixed per-message overhead) rather than Python's `<|start|>role` convention, so a native count stays comparable with every other counter.
- No `unsafe`, no `println!`, no `unwrap`/`expect` in library code; errors are typed `openjiuwen_common::error::Error`.
- Decoding includes special tokens (unlike `HuggingFace`'s `skip_special_tokens = true` default) and skips ids the artefact does not declare; a `.model` vocabulary allows special tokens by default (`allowed_special="all"` parity).
- Files stay around 1000 lines: `spec.rs`, `counter.rs` and `selector.rs` hold the split public surface, and `native/tiktoken_model/` splits the rank table and merge loop (`mod.rs`) from the Kimi/GPT-2 pre-tokenizers (`pretokenize.rs`) and the special-token maps (`specials.rs`).

## Design Decisions

- The artefact is decoded into immutable tables once; `NativeModel` dispatches BPE, WordPiece, Unigram and WordLevel behind one `encode_piece` seam, and a `Tiktoken` artefact bypasses that pipeline with its own pre-tokenization and merge loop.
- BPE applies the lowest-ranked merge first, left to right, until no ranked pair remains; `continuing_subword_prefix` lives inside the initial symbols and `end_of_word_suffix` is appended before the final lookup.
- Byte-level pre-tokenization maps bytes to printable scalars *before* applying the GPT-2 split, matching `HuggingFace`'s order (a mapped space is the letter `Ġ`, so it joins the following word).
- `byte_fallback` spells an unresolvable piece as `<0xXX>` tokens over its UTF-8 bytes; it is never combined with a byte-level pre-tokenizer in real artefacts.
- WordPiece is greedy longest-match with `##`-prefixed continuations and `max_input_chars_per_word` as a hard unknown cut-off.
- Unigram runs a forward dynamic program over the declared log-probabilities with an unknown penalty of `-100` per uncovered character; the best-scoring tiling wins, ties keep the first edge found.
- Added tokens are matched longest-first after normalization, honouring `single_word`, `lstrip` and `rsw`; special tokens are ordinary added tokens, so they always encode as one id.
- A `.model` line is `<base64 token bytes> <rank>`; blank lines and `#` comments are ignored, ranks order the merges, and encoding turns each pre-tokenized piece into UTF-8 bytes and repeatedly merges the adjacent pair with the lowest rank.
- The Kimi pre-tokenization rule is the default for every `.model` vocabulary (matching Python's `TiktokenModelCounter`), with `PretokenizePattern::Gpt2` available; Unicode classes come from `char::is_alphabetic`/`is_numeric`/`is_lowercase` plus a combining-mark range table instead of the full `\p{Han}`/`\p{L}`/`\p{M}` tables.
- A `.model` byte with no single-byte vocabulary entry is skipped and reported as a warning, where Python's `tiktoken` would raise, and decoding a vocabulary whose ids are not declared returns no text for that id.
- `.model` special tokens come from the sibling `tokenizer_config.json` when it declares `added_tokens_decoder`; otherwise the built-in Kimi map is used, with K3 selected when the name contains `kimi-k3` and K2 for anything else.
- Cache-dir precedence is explicit argument → `OPENJIUWEN_TOKENIZER_CACHE_DIR` → `$HOME/.cache/openjiuwen/tokenizers` → the relative `.cache/openjiuwen/tokenizers` when `HOME` is unset (documented fallback, never a temp dir: a cache should survive the process).
- The artefact name derives from the engine (`tiktoken.model` for `Tiktoken`, else `tokenizer.json`); an `artifact_path` may name the file or a directory, and a directory is searched deterministically — direct `<dir>/<artifact name>`, then a sorted depth-first walk (≤4 levels, ≤64 hits) taking the lexicographically first absolute match, instead of Python's non-deterministic `rglob`.
- Downloads land at `<cache>/huggingface/<id>/<revision>/<artifact>` and `<cache>/models/<id>/<revision>/<artifact>`; the `models` prefix is also one of the three offline `modelscope` snapshot layouts (`<cache>/<id>`, `<cache>/models/<id>`, `<cache>/hub/models/<id>`), so a downloaded artefact is found again without a second request.
- Hub URLs are `{endpoint}/<id>/resolve/<revision>/<artifact>` (`huggingface`, revision default `main`) and `{endpoint}/models/<id>/resolve/<revision>/<artifact>` (`modelscope`, default `master`); the single-artifact GET replaces Python's `snapshot_download(allow_patterns=...)`, so no sibling config file is fetched unless it is needed for specials.
- One download at a time is guaranteed by an advisory lock marker named `sha256(source, tokenizer_id, revision, sha256, engine).lock` in the cache dir: `create_new` acquisition, `tokio::time::sleep` exponential backoff (10 ms → 250 ms, 10 s budget), stale removal after 120 s of untouched mtime, and removal on drop.
- Bytes are validated before anything is written (UTF-8, then `NativeTokenizer::from_json_str` or `from_tiktoken_model`, then the declared sha256), written to a `.<uuid>.tmp` file in the target directory and renamed into place, so a reader never sees a partial artefact and a failed download leaves nothing behind.
- A well-formed artefact of a model type this crate does not decode is cached (`Error::Unsupported` is downgraded to a debug log); malformed text is `Error::Validation`.
- `resolve`/`resolve_cached` order: `artifact_path` (file or dir) → on-disk `tokenizer_id` → engine-specific missing code for a declared-but-unusable `artifact_path` → `://` rejection → `huggingface`/`modelscope` cache → gate (`remote_download_disabled` when downloads are off, `tokenizer_not_in_local_cache` when offline) → download; `provider_official` always answers `provider_official_adapter_not_configured`.
- The selector ladder is spec artefact → spec's first `compatible_fallbacks` entry (one hop, `target_tokenizer_unavailable`) → for a registry hit, its artefact then its first fallback (`family_tokenizer_fallback`/`model_variant_tokenizer_family`) → tiktoken → estimate; `Auto` therefore prefers a native artefact, and a `Native` engine with no resolvable artefact reports `native_tokenizer_unavailable` rather than failing.
- `ContextConfig` carries `tokenizer_cache_dir`, `enable_tokenizer_download` (default `false`) and `tokenizer_offline` (default `false`), builds the manager in `tokenizer_artifact_manager()`, and rejects the contradictory `enable_tokenizer_download && tokenizer_offline` pair in `validate()`.

## Routing Table

- `./tokenizers.rs` → module declarations and the public re-exports of the surface above
- `./spec.rs` → `ModelEncoding`, `TokenizerEngine`, `TokenizerSpec`, `CompatibleTokenizerSpec`, registry and match kinds
- `./counter.rs` → `TokenizerSource`, `TokenMeasurement`, `ModelTokenCounter`, `SelectedTokenizer` (including the native backing)
- `./selector.rs` → `TokenizerSelector` and its artefact/tiktoken/estimate ladder, `load_native_tokenizer`
- `./artifacts.rs` → module docs, cache-layout contract, `mod` declarations and the public re-exports
- `./artifacts/manager.rs` → `TokenizerArtifactManager` (resolution order, downloads, codes), endpoint/code constants, inline tests
- `./artifacts/cache.rs` → cache-dir precedence, artefact naming, deterministic directory search, relative ids, sha256
- `./artifacts/lock.rs` → advisory download lock (`create_new` marker, async backoff, stale expiry)
- `./native/mod.rs` → `NativeTokenizer` pipeline (normalize → added tokens → pre-tokenize → model, or the `.model` branch), `NativeTokenizerKind`, `byte_level_alphabet`, both `from_tiktoken*` constructors, counter impl
- `./native/hf_json.rs` → `tokenizer.json` serde shapes and polymorphic-node helpers
- `./native/model.rs` → model dispatch, `WordLevel`, vocabulary tables
- `./native/bpe.rs` → byte-pair encoding, merge lists, continuation prefix, byte fallback
- `./native/wordpiece.rs` → WordPiece greedy segmentation
- `./native/unigram.rs` → Unigram-lite dynamic program
- `./native/normalizer.rs` → `BERT`/lowercase/`Strip`/`Prepend`/`Replace` normalization and its warnings
- `./native/pretokenizer.rs` → `BERT`, whitespace, punctuation, digits, byte-level, metaspace, literal split, `Sequence`
- `./native/added_tokens.rs` → added/special token matching and chunking
- `./native/decoder.rs` → byte-level, metaspace, word-piece, `Fuse`, `Replace`, `Sequence` decoding
- `./native/bytes.rs` → the byte↔unicode alphabet
- `./native/tiktoken_model/` → `.model` ranks/merges and encode/decode (`mod.rs`), Kimi and GPT-2 pre-tokenization (`pretokenize.rs`), special-token maps and `tokenizer_config.json` parsing (`specials.rs`)
- `./native/modeltests.rs` → shared test fixtures for the native loaders
