//! [`StorageBackend`]: the storage backend the memory stores are built from.
//!
//! The value is plain data; `crate::store::MemoryStores` turns it into concrete
//! store instances. JSON uses an internally tagged shape, e.g.
//! `{"kind":"in_memory"}` or `{"kind":"local_file","root":"/var/lib/memory"}`.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Which storage backend the memory stores should use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageBackend {
    /// Ephemeral in-process stores; the default.
    InMemory,
    /// File-backed stores rooted at `root`.
    LocalFile {
        /// Directory that holds the store files.
        root: PathBuf,
    },
    /// SQLite-backed stores (requires the `sqlite` feature).
    Sqlite {
        /// Path of the SQLite database file.
        path: PathBuf,
    },
}

impl Default for StorageBackend {
    /// Returns [`StorageBackend::InMemory`].
    fn default() -> Self {
        Self::InMemory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_in_memory() {
        assert_eq!(StorageBackend::default(), StorageBackend::InMemory);
    }

    #[test]
    fn json_is_internally_tagged_by_kind() {
        assert_eq!(
            serde_json::to_value(StorageBackend::InMemory).unwrap(),
            serde_json::json!({"kind": "in_memory"})
        );
        assert_eq!(
            serde_json::to_value(StorageBackend::LocalFile {
                root: PathBuf::from("/data/memory"),
            })
            .unwrap(),
            serde_json::json!({"kind": "local_file", "root": "/data/memory"})
        );
        assert_eq!(
            serde_json::to_value(StorageBackend::Sqlite {
                path: PathBuf::from("/data/memory.db"),
            })
            .unwrap(),
            serde_json::json!({"kind": "sqlite", "path": "/data/memory.db"})
        );
    }

    #[test]
    fn json_round_trip_preserves_every_variant() {
        for backend in [
            StorageBackend::InMemory,
            StorageBackend::LocalFile {
                root: PathBuf::from("/data/memory"),
            },
            StorageBackend::Sqlite {
                path: PathBuf::from("/data/memory.db"),
            },
        ] {
            let json = serde_json::to_value(&backend).unwrap();
            let restored: StorageBackend = serde_json::from_value(json).unwrap();
            assert_eq!(restored, backend);
        }
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let backend: StorageBackend =
            serde_json::from_value(serde_json::json!({"kind": "in_memory", "extra": true}))
                .unwrap();
        assert_eq!(backend, StorageBackend::InMemory);
        let backend: StorageBackend = serde_json::from_value(
            serde_json::json!({"kind": "local_file", "root": "/data", "extra": 1}),
        )
        .unwrap();
        assert_eq!(
            backend,
            StorageBackend::LocalFile {
                root: PathBuf::from("/data")
            }
        );
    }

    #[test]
    fn unknown_variants_are_rejected() {
        let result = serde_json::from_value::<StorageBackend>(serde_json::json!({"kind": "s3"}));
        assert!(result.is_err());
    }
}
