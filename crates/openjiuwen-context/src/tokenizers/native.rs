//! Native tokenizer artefacts decoded in pure Rust.
//!
//! Reads a `HuggingFace` `tokenizer.json` (BPE, WordPiece and a Unigram-lite
//! branch, added and special tokens, normalizer/pre-tokenizer/decoder shapes)
//! and a `tiktoken`-style `.model` file (base64 ranks), so a model-native
//! count works with no Python, no network and no `tokenizers` runtime.
