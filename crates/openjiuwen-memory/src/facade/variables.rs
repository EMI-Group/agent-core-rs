//! Variables and per-scope configuration.
//!
//! Variables are plain text KV entries owned by a user scope, or by a session when the
//! scope carries a `session_id`; scope configuration is JSON stored under
//! `memory_scope_config/{scope_id}`. Both live in [`crate::store::MemoryStorage`], so
//! the facade only validates the ids and forwards the call.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};

use crate::config::MemoryScopeConfig;
use crate::record::{MemoryScope, DEFAULT_ID};

use super::LongTermMemory;

/// Validates a variable name: non-empty after trimming, without `/` or NUL.
pub(super) fn validate_variable_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(Error::validation(format!(
            "memory variable `name` must not be empty (got \"{name}\")"
        )));
    }
    if name.contains('/') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain '/' (got \"{name}\")"
        )));
    }
    if name.contains('\0') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain NUL (got \"{name}\")"
        )));
    }
    Ok(())
}

/// Validates a bare `scope_id` with the [`MemoryScope`] rules (non-empty, no `/`, no
/// backslash, no NUL, at most 128 characters).
pub(super) fn validate_scope_id(scope_id: &str) -> Result<()> {
    MemoryScope::new(DEFAULT_ID, scope_id).validate()
}

impl LongTermMemory {
    /// Stores `value` as the text variable `name` of `scope`.
    ///
    /// A scope with a `session_id` writes a session variable; otherwise the variable is
    /// user-scoped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the scope or the variable name is invalid and
    /// the storage error when the write fails.
    pub async fn set_variable(&self, scope: &MemoryScope, name: &str, value: &str) -> Result<()> {
        scope.validate()?;
        validate_variable_name(name)?;
        self.engine
            .storage()
            .set_variable(scope, scope.session_id.as_deref(), name, value)
            .await
    }

    /// Reads variables of `scope`: every stored variable when `names` is `None`, or the
    /// named ones that exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when a
    /// read fails.
    pub async fn get_variables(
        &self,
        scope: &MemoryScope,
        names: Option<&[String]>,
    ) -> Result<BTreeMap<String, String>> {
        scope.validate()?;
        let storage = self.engine.storage();
        let session_id = scope.session_id.as_deref();
        let Some(names) = names else {
            return storage.list_variables(scope, session_id).await;
        };
        let mut variables = BTreeMap::new();
        for name in names {
            if let Some(value) = storage.get_variable(scope, session_id, name).await? {
                variables.insert(name.clone(), value);
            }
        }
        Ok(variables)
    }

    /// Deletes `names` from `scope`, returning how many variables existed.
    ///
    /// An empty `names` slice deletes nothing and returns `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or variable name and the
    /// storage error when a delete fails.
    pub async fn delete_variables(&self, scope: &MemoryScope, names: &[String]) -> Result<usize> {
        scope.validate()?;
        let session_id = scope.session_id.as_deref();
        let mut removed = 0;
        for name in names {
            validate_variable_name(name)?;
            if self
                .engine
                .storage()
                .delete_variable(scope, session_id, name)
                .await?
            {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Stores the configuration of `scope_id`, returning whether it replaced one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and the storage error
    /// when the write fails.
    pub async fn set_scope_config(
        &self,
        scope_id: &str,
        config: MemoryScopeConfig,
    ) -> Result<bool> {
        validate_scope_id(scope_id)?;
        self.engine
            .storage()
            .set_scope_config(scope_id, &config)
            .await
    }

    /// Reads the configuration of `scope_id`, or `None` when it is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` or malformed stored JSON
    /// and the storage error when the read fails.
    pub async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>> {
        validate_scope_id(scope_id)?;
        self.engine.storage().get_scope_config(scope_id).await
    }

    /// Deletes the configuration of `scope_id`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and the storage error
    /// when the delete fails.
    pub async fn delete_scope_config(&self, scope_id: &str) -> Result<bool> {
        validate_scope_id(scope_id)?;
        self.engine.storage().delete_scope_config(scope_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::MemoryEngineConfig;

    async fn facade() -> std::sync::Arc<LongTermMemory> {
        LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    #[test]
    fn variable_names_are_validated() {
        assert!(validate_variable_name("theme").is_ok());
        for name in ["", "   ", "a/b", "nul\0byte"] {
            let error = validate_variable_name(name).expect_err("invalid name");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
    }

    #[test]
    fn scope_ids_are_validated() {
        assert!(validate_scope_id("ws").is_ok());
        for scope_id in ["", "a/b", &"x".repeat(129)] {
            let error = validate_scope_id(scope_id).expect_err("invalid scope id");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
        assert!(validate_scope_id(&"x".repeat(128)).is_ok());
    }

    #[tokio::test]
    async fn variables_round_trip_through_the_facade() {
        let ltm = facade().await;
        ltm.set_variable(&scope(), "theme", "dark")
            .await
            .expect("set");
        ltm.set_variable(&scope(), "city", "Berlin")
            .await
            .expect("set");
        let all = ltm.get_variables(&scope(), None).await.expect("list");
        assert_eq!(all.len(), 2);
        assert_eq!(all.get("theme").map(String::as_str), Some("dark"));

        let named = ltm
            .get_variables(&scope(), Some(&["theme".to_owned(), "missing".to_owned()]))
            .await
            .expect("named");
        assert_eq!(named.len(), 1);
        assert_eq!(named.get("theme").map(String::as_str), Some("dark"));

        assert_eq!(
            ltm.delete_variables(&scope(), &["theme".to_owned()])
                .await
                .expect("delete"),
            1
        );
        assert_eq!(
            ltm.delete_variables(&scope(), &["theme".to_owned()])
                .await
                .expect("delete again"),
            0
        );
        assert_eq!(
            ltm.delete_variables(&scope(), &[])
                .await
                .expect("delete none"),
            0
        );
        assert_eq!(
            ltm.get_variables(&scope(), None).await.expect("list").len(),
            1
        );
    }

    #[tokio::test]
    async fn session_variables_are_separate_from_user_variables() {
        let ltm = facade().await;
        ltm.set_variable(&scope(), "theme", "dark")
            .await
            .expect("user variable");
        let session = scope().session("s1");
        ltm.set_variable(&session, "theme", "light")
            .await
            .expect("session variable");
        assert_eq!(
            ltm.get_variables(&scope(), None)
                .await
                .expect("user")
                .get("theme")
                .map(String::as_str),
            Some("dark")
        );
        assert_eq!(
            ltm.get_variables(&session, None)
                .await
                .expect("session")
                .get("theme")
                .map(String::as_str),
            Some("light")
        );
    }

    #[tokio::test]
    async fn invalid_variables_and_scopes_are_rejected() {
        let ltm = facade().await;
        let error = ltm
            .set_variable(&scope(), "a/b", "value")
            .await
            .expect_err("invalid name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = ltm
            .set_variable(&MemoryScope::new("alice", "bad/scope"), "theme", "dark")
            .await
            .expect_err("invalid scope");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn scope_config_round_trips_and_is_deleted() {
        let ltm = facade().await;
        let config = MemoryScopeConfig {
            user_profile_definition: Some("identity facts".to_owned()),
            semantic_memory_definition: Some("durable facts".to_owned()),
            episodic_memory_definition: None,
        };
        assert!(!ltm
            .set_scope_config("ws", config.clone())
            .await
            .expect("create"));
        assert_eq!(
            ltm.get_scope_config("ws").await.expect("get"),
            Some(config.clone())
        );
        assert!(ltm
            .set_scope_config("ws", config.clone())
            .await
            .expect("replace"));
        assert_eq!(ltm.get_scope_config("other").await.expect("absent"), None);
        assert!(ltm.delete_scope_config("ws").await.expect("delete"));
        assert!(!ltm.delete_scope_config("ws").await.expect("delete again"));
        assert_eq!(ltm.get_scope_config("ws").await.expect("absent"), None);
        let error = ltm
            .get_scope_config("bad/scope")
            .await
            .expect_err("invalid scope id");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }
}
