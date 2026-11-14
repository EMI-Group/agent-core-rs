# tokenizers

## Intent

Owns tokenizer resolution and the token-counting seam of the context engine: model names map to a `tiktoken` vocabulary, declared specs resolve through a registry, and a local `HuggingFace` `tokenizer.json` artifact can be decoded and used directly.
Ports the counting and selection behaviour of the Python `core/context_engine/token/` package to Rust without its optional runtime dependencies.
Everything here is offline: BPE vocabularies come from `tiktoken-rs`'s embedded data, and native artifacts are decoded with `serde_json` alone.

## API Surface

`tokenizers.rs` (the parent module) exposes:

- `ModelEncoding`, `ModelTokenCounter`, `TokenizerSpec`, `TokenizerRegistry`, `TokenizerMatch`, `TokenizerMatchKind`, `TokenizerEngine`, `TokenizerSource`, `TokenizerSelector`, `SelectedTokenizer`, `TokenMeasurement`, `load_native_tokenizer`.
- `TokenizerSource::NativeTokenizer` is the measured source of an exact native count; `SelectedTokenizer::estimated()` is `false` only then.
- `SelectedTokenizer::from_native(Arc<NativeTokenizer>, model, tokenizer)` is the only public native constructor; `load_native_tokenizer(path)` builds it from a file and reports `source() == NativeTokenizer`, `estimated() == false` and no fallback reason.
- Re-exported from `native`: `NativeTokenizer`, `NativeTokenizerKind`, `byte_level_alphabet`.

`native/` (the `openjiuwen_context::tokenizers::native` module) exposes exactly:

- `NativeTokenizer` — `from_json_str`, `load`, `encode`, `decode`, `vocab_size`, `kind`, `warnings`, `tokenizer_id`, `special_tokens`, plus `impl TokenCounter`.
- `NativeTokenizerKind` — `Bpe`, `WordPiece`, `Unigram`, `WordLevel` (its `as_str` is the artifact's `"type"` tag).
- `byte_level_alphabet()` — the 256-entry GPT-2/Llama byte↔unicode table.

## Constraints

- No new dependencies: decoding, quantization of merges, byte mapping and determinism all come from `serde_json`, `tracing` and the standard library.
- A load is a one-off per context/model binding; `encode`, `decode` and `count_text` are pure in-memory computation and never touch the filesystem or the network.
- Model types outside BPE/WordPiece/Unigram/WordLevel are `Error::Unsupported` naming the type; malformed documents and missing vocabularies are `Error::Validation`; a missing file is `Error::NotFound`.
- Everything a load cannot reproduce exactly (regex splits, the NFC/NFD/NFKC/NFKD normalizer families, unknown pipeline nodes, BPE dropout) is recorded in `NativeTokenizer::warnings()` and degrades to a coarser behaviour — never a failed load, because a token count stays useful.
- `TokenCounter::count_text` is `encode().len()`; message framing uses the workspace default (a fixed per-message overhead) rather than Python's `<|start|>role` convention, so a native count stays comparable with every other counter.
- No `unsafe`, no `println!`, no `unwrap`/`expect` in library code; errors are typed `openjiuwen_common::error::Error`; files stay well under 1000 lines.
- Decoding includes special tokens (unlike `HuggingFace`'s `skip_special_tokens = true` default) and skips ids the artifact does not declare.

## Design Decisions

- The artifact is decoded into immutable tables once; `NativeModel` dispatches BPE, WordPiece, Unigram and WordLevel behind one `encode_piece` seam.
- BPE applies the lowest-ranked merge first, left to right, until no ranked pair remains; `continuing_subword_prefix` lives inside the initial symbols and `end_of_word_suffix` is appended before the final lookup.
- Byte-level pre-tokenization maps bytes to printable scalars *before* applying the GPT-2 split, matching `HuggingFace`'s order (a mapped space is the letter `Ġ`, so it joins the following word).
- `byte_fallback` spells an unresolvable piece as `<0xXX>` tokens over its UTF-8 bytes; it is never combined with a byte-level pre-tokenizer in real artifacts.
- WordPiece is greedy longest-match with `##`-prefixed continuations and `max_input_chars_per_word` as a hard unknown cut-off.
- Unigram runs a forward dynamic program over the declared log-probabilities with an unknown penalty of `-100` per uncovered character; the best-scoring tiling wins, ties keep the first edge found.
- Added tokens are matched longest-first after normalization, honouring `single_word`, `lstrip` and `rsw`; special tokens are ordinary added tokens, so they always encode as one id.

## Routing Table

- `./tokenizers.rs` → selector ladder, `tiktoken` counters, `SelectedTokenizer` (including the native backing), `load_native_tokenizer`
- `./artifacts.rs` → artifact discovery/caching/download deferral (`tokenizer_manager.py`)
- `./native/mod.rs` → `NativeTokenizer` pipeline (normalize → added tokens → pre-tokenize → model), `NativeTokenizerKind`, `byte_level_alphabet`, counter impl
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
