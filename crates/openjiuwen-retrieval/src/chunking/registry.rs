//! Named chunker factories.
//!
//! Port of the Python chunker registry (`register_chunker` / `get_chunker`) plus
//! the extra `"recursive"` and `"token"` built-ins of this crate.
//! `"hybrid"` mirrors Python's `_hybrid_factory`: it takes the same size/overlap
//! pair as every other factory and builds a [`HybridChunker`] over a
//! [`CharChunker`] with the default no-split predicate.
use std::collections::HashMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use super::{CharChunker, Chunker, HybridChunker, RecursiveCharChunker, TokenChunker};

/// Factory that builds a chunker from a size and an overlap.
pub type ChunkerFactory = Arc<dyn Fn(usize, usize) -> Result<Arc<dyn Chunker>> + Send + Sync>;

/// Registry mapping chunker names to factories.
///
/// Names are trimmed and lowercased, so lookups are case-insensitive. The
/// built-ins are `"char"` (alias `"fixed"`), `"recursive"`, `"token"`, and
/// `"hybrid"`.
#[derive(Clone, Default)]
pub struct ChunkerRegistry {
    /// Normalized name to factory.
    factories: HashMap<String, ChunkerFactory>,
}

impl ChunkerRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry pre-populated with the built-in chunkers.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.insert("char", Arc::new(create_char));
        registry.insert("fixed", Arc::new(create_char));
        registry.insert("recursive", Arc::new(create_recursive));
        registry.insert("token", Arc::new(create_token));
        registry.insert("hybrid", Arc::new(create_hybrid));
        registry
    }

    /// Registers a factory under `name`.
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when the
    /// normalized name is empty or already registered.
    pub fn register(&mut self, name: &str, factory: ChunkerFactory) -> Result<()> {
        let name = normalize(name);
        if name.is_empty() {
            return Err(Error::Validation(
                "chunker name must be a non-empty string".to_owned(),
            ));
        }
        if self.factories.contains_key(&name) {
            return Err(Error::Validation(format!(
                "chunker '{name}' is already registered"
            )));
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    /// Builds the chunker registered under `name`.
    ///
    /// Returns [`openjiuwen_common::error::Error::Unsupported`] naming the
    /// unknown chunker when the name is not registered; factory validation
    /// errors are propagated unchanged.
    pub fn create(
        &self,
        name: &str,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Result<Arc<dyn Chunker>> {
        let name = normalize(name);
        let Some(factory) = self.factories.get(&name) else {
            let mut known: Vec<&str> = self.factories.keys().map(String::as_str).collect();
            known.sort_unstable();
            return Err(Error::Unsupported(format!(
                "unknown chunker: {name}; registered: {}",
                known.join(", ")
            )));
        };
        factory(chunk_size, chunk_overlap)
    }

    /// Returns whether a chunker is registered under `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(&normalize(name))
    }

    /// Inserts without validation; used for the built-in registrations only.
    fn insert(&mut self, name: &str, factory: ChunkerFactory) {
        self.factories.insert(normalize(name), factory);
    }
}

/// Normalizes a registry key (trimmed, lowercased).
fn normalize(name: &str) -> String {
    name.trim().to_lowercase()
}

/// Builds a [`CharChunker`], backing both `"char"` and `"fixed"`.
fn create_char(chunk_size: usize, chunk_overlap: usize) -> Result<Arc<dyn Chunker>> {
    Ok(Arc::new(CharChunker::new(chunk_size, chunk_overlap)?))
}

/// Builds a [`RecursiveCharChunker`].
fn create_recursive(chunk_size: usize, chunk_overlap: usize) -> Result<Arc<dyn Chunker>> {
    Ok(Arc::new(RecursiveCharChunker::new(
        chunk_size,
        chunk_overlap,
    )?))
}

/// Builds a [`TokenChunker`].
fn create_token(chunk_size: usize, chunk_overlap: usize) -> Result<Arc<dyn Chunker>> {
    Ok(Arc::new(TokenChunker::new(chunk_size, chunk_overlap)?))
}

/// Builds a [`HybridChunker`] over a [`CharChunker`] (Python's default factory).
fn create_hybrid(chunk_size: usize, chunk_overlap: usize) -> Result<Arc<dyn Chunker>> {
    let inner = Arc::new(CharChunker::new(chunk_size, chunk_overlap)?);
    Ok(Arc::new(HybridChunker::new(inner)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn defaults_resolve_every_builtin() {
        let registry = ChunkerRegistry::with_defaults();
        assert_eq!(registry.create("char", 8, 2).unwrap().chunk_size(), 8);
        assert_eq!(registry.create("recursive", 9, 3).unwrap().chunk_size(), 9);
        assert_eq!(registry.create("token", 7, 1).unwrap().chunk_size(), 7);
        assert!(registry.contains("char"));
        assert!(registry.contains("fixed"));
        assert!(registry.contains("hybrid"));
    }

    #[test]
    fn builtin_factories_apply_the_requested_parameters() {
        let registry = ChunkerRegistry::with_defaults();
        let char_chunker = registry.create("char", 5, 2).unwrap();
        assert_eq!(
            char_chunker.chunk_text("abcdefgh").unwrap(),
            ["abcde", "defgh", "gh"]
        );
        assert_eq!(char_chunker.chunk_overlap(), 2);
        let token_chunker = registry.create("token", 2, 0).unwrap();
        assert_eq!(token_chunker.chunk_text("a b c").unwrap(), ["a b", "c"]);
        // The hybrid factory wraps a char chunker with the same parameters.
        let hybrid_chunker = registry.create("hybrid", 5, 2).unwrap();
        assert_eq!(hybrid_chunker.chunk_size(), 5);
        assert_eq!(hybrid_chunker.chunk_overlap(), 2);
        assert_eq!(
            hybrid_chunker.chunk_text("abcdefgh").unwrap(),
            ["abcde", "defgh", "gh"]
        );
    }

    #[test]
    fn the_fixed_alias_resolves_to_the_char_chunker() {
        let registry = ChunkerRegistry::with_defaults();
        let alias = registry.create("  FIXED ", 4, 0).unwrap();
        assert_eq!(alias.chunk_text("abcdefgh").unwrap(), ["abcd", "efgh"]);
        assert_eq!(alias.chunk_size(), 4);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = ChunkerRegistry::with_defaults();
        let error = registry
            .register("char", Arc::new(create_char))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("already registered"));
    }

    #[test]
    fn blank_names_are_rejected() {
        let mut registry = ChunkerRegistry::new();
        let error = registry
            .register(" \t ", Arc::new(create_char))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("non-empty"));
        assert!(!registry.contains(""));
    }

    #[test]
    fn custom_chunkers_can_be_registered() {
        let mut registry = ChunkerRegistry::new();
        registry.register("Char2", Arc::new(create_char)).unwrap();
        let chunker = registry.create("char2", 3, 0).unwrap();
        assert_eq!(chunker.chunk_text("abcdef").unwrap(), ["abc", "def"]);
    }

    #[test]
    fn unknown_names_are_unsupported() {
        let registry = ChunkerRegistry::with_defaults();
        let error = registry.create("semantic", 8, 2).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("unknown chunker: semantic"));
    }

    #[test]
    fn factory_validation_errors_are_propagated() {
        let registry = ChunkerRegistry::with_defaults();
        let error = registry.create("char", 0, 0).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
