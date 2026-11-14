//! Model-aware tokenizer registry and token accounting.
//!
//! Python keeps a registry of tokenizer artifacts in
//! `core.context_engine.token` so a context engine can pick the exact counter a
//! model needs. The Rust port keeps the same seam:
//!
//! - [`ModelEncoding`] maps well-known model names to one of the four `OpenAI`
//!   BPE vocabularies embedded in `tiktoken-rs`.
//! - [`TokenizerRegistry`] holds [`TokenizerSpec`] entries so deployments can
//!   declare a provider/family tokenizer without changing code.
//! - [`artifacts::TokenizerArtifactManager`] resolves, validates, caches and
//!   (only when explicitly enabled) downloads native artifacts, so selection
//!   stays network-free by default.
//! - [`TokenizerSelector`] walks the selection ladder: explicit spec, registry
//!   match, native artifact, `tiktoken` encoding, then a conservative
//!   string-length estimate.
//! - [`ModelTokenCounter`] and [`SelectedTokenizer`] implement the
//!   [`TokenCounter`](openjiuwen_foundation::prompt::TokenCounter) seam used by
//!   the rest of the workspace.
//!
//! Native artifacts are decoded in pure Rust by [`NativeTokenizer`], so
//! [`load_native_tokenizer`] needs neither the `HuggingFace` `tokenizers`
//! runtime nor network access. Both artifact kinds are supported: a
//! `tokenizer.json` document and a `tiktoken` `.model` BPE vocabulary (see
//! [`TiktokenModel`]).
pub mod artifacts;
mod counter;
pub mod native;
pub mod selector;
mod spec;

pub use counter::{ModelTokenCounter, SelectedTokenizer, TokenMeasurement, TokenizerSource};
pub use native::tiktoken_model::{PretokenizePattern, TiktokenModel, TiktokenSpecials};
pub use native::{byte_level_alphabet, NativeTokenizer, NativeTokenizerKind};
pub use selector::{load_native_tokenizer, TokenizerSelector};
pub use spec::{
    CompatibleTokenizerSpec, ModelEncoding, TokenizerEngine, TokenizerMatch, TokenizerMatchKind,
    TokenizerRegistry, TokenizerSpec,
};
