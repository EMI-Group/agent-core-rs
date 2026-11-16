//! The private pagination-cursor machinery of the `skill_index` toolkit.
//!
//! The cursor state and its least-recently-used store live here so
//! [`SkillIndexToolkit`](super::toolkit::SkillIndexToolkit) stays focused on request
//! handling. Nothing in this module is exported: the token format and the store's bounds
//! are an implementation detail of the toolkit, and a caller only ever sees the opaque
//! token on a [`SkillIndexResult`](super::toolkit::SkillIndexResult) and its `[next]` footer.

use base64::Engine as _;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::toolkit::MAX_CURSORS;
use super::validate::Operation;

/// The schema version the cursor token hashes.
pub(super) const CURSOR_STATE_VERSION: u64 = 1;

/// The bytes of the SHA-256 digest a cursor token keeps.
pub(super) const CURSOR_TOKEN_BYTES: usize = 16;

/// One page cursor's continuation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CursorState {
    /// The operation the page belongs to.
    pub(super) operation: Operation,
    /// The category scope, when one was named.
    pub(super) category: Option<String>,
    /// The validated search queries; empty for a list.
    pub(super) queries: Vec<String>,
    /// The offset the next page starts at.
    pub(super) offset: usize,
    /// The page size the original request used.
    pub(super) page_size: usize,
    /// The catalog fingerprint the page was minted against.
    pub(super) fingerprint: String,
}

impl CursorState {
    /// The deterministic token for this state.
    ///
    /// The URL-safe unpadded base64 of the first [`CURSOR_TOKEN_BYTES`] of SHA-256 over
    /// the canonical JSON of `{v, operation, category, queries, offset, page_size,
    /// fingerprint}`. The field order is fixed and the document holds no maps, so the
    /// bytes — and therefore the token — are identical for identical states.
    pub(super) fn token(&self) -> String {
        let document = json!({
            "v": CURSOR_STATE_VERSION,
            "operation": self.operation.as_str(),
            "category": self.category,
            "queries": self.queries,
            "offset": self.offset,
            "page_size": self.page_size,
            "fingerprint": self.fingerprint,
        });
        let canonical = document.to_string();
        let digest = Sha256::digest(canonical.as_bytes());
        let head = digest
            .get(..CURSOR_TOKEN_BYTES)
            .unwrap_or(digest.as_slice());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(head)
    }
}

/// The page cursors, ordered least recently used first.
#[derive(Debug, Default, Clone)]
pub(super) struct CursorStore {
    /// `(token, state)` pairs, the most recently used last.
    entries: Vec<(String, CursorState)>,
}

impl CursorStore {
    /// The state behind `token`, when it is still held.
    pub(super) fn get(&self, token: &str) -> Option<&CursorState> {
        self.entries
            .iter()
            .find(|(held, _)| held == token)
            .map(|(_, state)| state)
    }

    /// Stores `state` under `token`, evicting the least recently used past
    /// [`MAX_CURSORS`].
    pub(super) fn push(&mut self, token: String, state: CursorState) {
        self.entries.retain(|(held, _)| held != &token);
        self.entries.push((token, state));
        while self.entries.len() > MAX_CURSORS {
            self.entries.remove(0);
        }
    }

    /// Drops `token` from the store.
    pub(super) fn remove(&mut self, token: &str) {
        self.entries.retain(|(held, _)| held != token);
    }

    /// How many cursors the store holds.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drops every cursor.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }
}
