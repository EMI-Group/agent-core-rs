//! Backend parity: the same contract over the in-memory store and the SQLite store.
//!
//! Both tests call [`exercise_contract`](super::contract::exercise_contract), so any observable
//! answer that differs between `InMemoryTeamStore` and `SqliteTeamStore` fails one of them.

use crate::storage::{InMemoryTeamStore, TeamStore};

use super::contract::exercise_contract;
use crate::storage::SqliteTeamStore;

#[tokio::test]
async fn contract_holds_for_the_in_memory_store() {
    let store = InMemoryTeamStore::new();
    assert_eq!(store.name(), "memory");
    exercise_contract(&store).await;
}

#[tokio::test]
async fn contract_holds_for_the_sqlite_store() {
    let store = SqliteTeamStore::in_memory().await.unwrap();
    assert_eq!(store.name(), "sqlite");
    exercise_contract(&store).await;
}
