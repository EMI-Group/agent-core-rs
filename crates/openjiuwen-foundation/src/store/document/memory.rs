//! In-memory document store.
//!
//! The behavioral reference every other backend is checked against: collections
//! and their documents live in one `BTreeMap`, so iteration is ordered by
//! collection name and then by document id. Nothing is persisted.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::Result;

use super::{matching, run_query, validate_collection, Document, DocumentStore, Query};

/// The collection map: collection name → document id → document.
type Collections = BTreeMap<String, BTreeMap<String, Document>>;

/// A [`DocumentStore`] that keeps every collection in process memory.
///
/// Clones share the same map, so a clone observes writes made through another
/// clone. Dropping every clone discards the data.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDocumentStore {
    /// All collections behind one lock.
    collections: Arc<Mutex<Collections>>,
}

impl InMemoryDocumentStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            collections: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Number of collections currently held.
    pub fn len(&self) -> usize {
        self.collections.lock().len()
    }

    /// Whether the store holds no collection.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl DocumentStore for InMemoryDocumentStore {
    async fn create_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        self.collections
            .lock()
            .entry(collection.to_string())
            .or_default();
        Ok(())
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self.collections.lock().keys().cloned().collect())
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        validate_collection(collection)?;
        self.collections.lock().remove(collection);
        Ok(())
    }

    async fn upsert(&self, collection: &str, documents: Vec<Document>) -> Result<usize> {
        validate_collection(collection)?;
        let written = documents.len();
        let mut collections = self.collections.lock();
        let stored = collections.entry(collection.to_string()).or_default();
        for document in documents {
            stored.insert(document.id.clone(), document);
        }
        Ok(written)
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Document>> {
        validate_collection(collection)?;
        Ok(self
            .collections
            .lock()
            .get(collection)
            .and_then(|documents| documents.get(id).cloned()))
    }

    async fn query(&self, collection: &str, query: &Query) -> Result<Vec<Document>> {
        validate_collection(collection)?;
        let collections = self.collections.lock();
        let Some(documents) = collections.get(collection) else {
            return Ok(Vec::new());
        };
        Ok(run_query(documents.values(), query)
            .into_iter()
            .cloned()
            .collect())
    }

    async fn delete(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let mut collections = self.collections.lock();
        let Some(documents) = collections.get_mut(collection) else {
            return Ok(0);
        };
        let doomed: Vec<String> = matching(documents.values(), &query.filter)
            .into_iter()
            .map(|document| document.id.clone())
            .collect();
        for id in &doomed {
            documents.remove(id);
        }
        Ok(doomed.len())
    }

    async fn count(&self, collection: &str, query: &Query) -> Result<usize> {
        validate_collection(collection)?;
        let collections = self.collections.lock();
        let Some(documents) = collections.get(collection) else {
            return Ok(0);
        };
        Ok(matching(documents.values(), &query.filter).len())
    }
}

#[cfg(test)]
mod tests {
    use super::super::contract;
    use super::*;
    use serde_json::json;

    async fn store() -> InMemoryDocumentStore {
        InMemoryDocumentStore::new()
    }

    #[tokio::test]
    async fn satisfies_the_shared_contract() {
        contract::run(&store().await).await;
    }

    #[tokio::test]
    async fn default_and_new_are_empty() {
        let store = InMemoryDocumentStore::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(
            store.list_collections().await.unwrap(),
            Vec::<String>::new()
        );

        store.create_collection("docs").await.unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[tokio::test]
    async fn clones_share_state() {
        let store = store().await;
        let clone = store.clone();
        clone
            .upsert("docs", vec![Document::new("a").with_field("n", 1)])
            .await
            .unwrap();

        let document = store.get("docs", "a").await.unwrap().expect("shared");
        assert_eq!(document.field("n"), Some(&json!(1)));
    }

    #[tokio::test]
    async fn ties_keep_id_order() {
        // The `BTreeMap` iteration order makes equal documents come back by id.
        let store = store().await;
        store
            .upsert(
                "docs",
                vec![
                    Document::new("c").with_field("age", 36),
                    Document::new("a").with_field("age", 36),
                    Document::new("b").with_field("age", 36),
                ],
            )
            .await
            .unwrap();

        let query = Query {
            order_by: vec![("age".to_string(), super::super::SortOrder::Asc)],
            ..Query::default()
        };
        let ids: Vec<String> = store
            .query("docs", &query)
            .await
            .unwrap()
            .into_iter()
            .map(|document| document.id)
            .collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn dropping_a_collection_forgets_its_documents() {
        let store = store().await;
        store
            .upsert("docs", vec![Document::new("a")])
            .await
            .unwrap();
        store.drop_collection("docs").await.unwrap();
        assert!(store.get("docs", "a").await.unwrap().is_none());

        // Writing again recreates the collection.
        store
            .upsert("docs", vec![Document::new("b")])
            .await
            .unwrap();
        assert_eq!(store.list_collections().await.unwrap(), vec!["docs"]);
        assert!(store.get("docs", "a").await.unwrap().is_none());
        assert!(store.get("docs", "b").await.unwrap().is_some());
    }
}
