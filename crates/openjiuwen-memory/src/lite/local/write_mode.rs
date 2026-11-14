//! Write modes and bounded conflict retries for the local coding-memory store.
//!
//! Wires [`crate::lite::write`] into the Markdown source of truth: one call applies a
//! [`WriteMode`] to a named fact, computing the outcome from the stored snapshot, writing
//! only what the mode implies, and retrying a lost race under the store's own locks.
//!
//! # Sequence
//!
//! [`LocalCodingMemory::write_mode`] validates the incoming fact, resolves its type against
//! [`CodingMemoryConfig::fact_types`](crate::config::CodingMemoryConfig::fact_types), forces
//! the store's scope, and hands the actual work to a
//! [`MemUpdateChecker`](crate::lite::write::MemUpdateChecker) with the caller's
//! [`WritePolicy`]. Each attempt is one full read-decide-write cycle on the current file:
//!
//! 1. acquire the cross-process [`FileLock`](crate::lite::lock::FileLock), then the
//!    in-process state guard (always in that order);
//! 2. reload the document when its stamp changed
//!    ([`ensure_fresh_locked`](super::facts_mode)) so another process's committed edit is
//!    part of the decision;
//! 3. look up the existing same-named fact and decide with
//!    [`MemUpdateChecker::plan`](crate::lite::write::MemUpdateChecker::plan);
//! 4. upsert the planned fact, persist the document (which also refreshes the per-fact file
//!    projection when that layout is enabled), update the derived full-text index, drop both
//!    guards, and only then refresh the embedding index — the state guard is never held
//!    across a model call.
//!
//! # Conflicts
//!
//! Two conditions count as a conflict rather than a decision:
//!
//! - [`WriteMode::Create`] meeting an existing fact. The attempt reports "no commit", so
//!   the checker retries with exponential backoff and, when the budget is exhausted, the
//!   call fails with a typed [`Error::Memory`](openjiuwen_common::error::Error::Memory)
//!   naming the fact and the attempt count. The store is left untouched: a `create`
//!   collision never overwrites the existing fact and is never reported as a skip.
//! - A document that changed between the attempt's read and its commit. The local store
//!   serializes writers, so this is rare; when it happens the same bounded retry applies.
//!
//! A successful write reports `attempts` (1 when it committed first time) and
//! `conflict = attempts > 1`.
//!
//! # Skips
//!
//! [`WriteMode::Skip`] on an existing fact and an [`WriteMode::Append`] whose body would add
//! nothing both decide on [`WritePlan::Skipped`](crate::lite::write::WritePlan): nothing is
//! written, the file, its stamps, and the derived indices are untouched, and the returned
//! [`WriteOutcome`] reports `mode = WriteMode::Skip`, `created = false`, the stored fact, and
//! a note explaining why. An append no-op is therefore observable without inspecting the
//! document.
//!
//! # Errors
//!
//! Validation and scope errors are [`Error::Validation`], a missing capability is never
//! reached here, storage/lock/filesystem errors propagate unchanged (filesystem failures
//! carry the failing path), and an exhausted retry budget is [`Error::Memory`]. Nothing
//! panics on user input.
use chrono::Utc;

use openjiuwen_common::error::Result;

use crate::lite::fact::CodingFact;
use crate::lite::write::{MemUpdateChecker, WriteMode, WriteOutcome, WritePlan, WritePolicy};

use super::state::LocalCodingMemory;
use super::TARGET;

impl LocalCodingMemory {
    /// Applies `mode` to `fact` under `policy`, retrying a conflict with backoff.
    ///
    /// The stored fact's type is resolved against the configured type list (an unconfigured
    /// type is replaced by the configured default) and the scope is always the store's scope,
    /// so a caller cannot write into another scope. What the mode does to an existing fact is
    /// [`WriteMode`]'s documented table; a write stamps `updated_at = now`, and a skip keeps
    /// the stored fact's timestamps.
    ///
    /// Returns the outcome: the stored fact, the applied mode, whether a fact was created,
    /// the attempt count, whether an attempt conflicted, and a note for a skip or a retried
    /// write. A conflicting `create` that never clears returns
    /// [`Error::Memory`](openjiuwen_common::error::Error::Memory) naming the fact and the
    /// attempt budget rather than overwriting or skipping.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) for an
    /// invalid fact or policy, the lock, filesystem, or index error when the attempt cannot
    /// complete, and [`Error::Memory`](openjiuwen_common::error::Error::Memory) when every
    /// attempt conflicted.
    pub async fn write_mode(
        &self,
        fact: CodingFact,
        mode: WriteMode,
        policy: WritePolicy,
    ) -> Result<WriteOutcome> {
        fact.validate()?;
        let mut fact = fact;
        fact.fact_type = self.resolve_fact_type(fact.fact_type);
        fact.scope = self.inner.scope.clone();
        let checker = MemUpdateChecker::new(policy)?;
        let what = format!("coding fact `{}`", fact.name);
        let checker_ref = &checker;
        let (mut outcome, attempts) = checker
            .run(&what, |_attempt| {
                let fact = fact.clone();
                async move { self.write_mode_once(&fact, mode, checker_ref).await }
            })
            .await?;
        outcome.attempts = attempts;
        outcome.conflict = attempts > 1;
        if outcome.conflict {
            outcome.note = Some(format!(
                "a conflicting write of `{}` succeeded on attempt {attempts}",
                fact.name
            ));
        }
        Ok(outcome)
    }

    /// One read-decide-write attempt of [`LocalCodingMemory::write_mode`].
    ///
    /// Returns `Ok(None)` when the attempt observed a conflict and wrote nothing, so the
    /// checker retries; every committed, skipped, or failed attempt returns from here
    /// directly.
    async fn write_mode_once(
        &self,
        fact: &CodingFact,
        mode: WriteMode,
        checker: &MemUpdateChecker,
    ) -> Result<Option<WriteOutcome>> {
        let (guard, mut state) = self.lock_for_write().await?;
        self.ensure_fresh_locked(&mut state).await?;
        let existing = state.document.fact(&fact.name);
        if mode == WriteMode::Create && existing.is_some() {
            tracing::debug!(
                target: TARGET,
                name = %fact.name,
                "lite write mode `create` found an existing fact; reporting a conflict"
            );
            return Ok(None);
        }
        let plan = checker.plan(mode, fact, existing.as_ref(), Utc::now())?;
        let next = match plan {
            WritePlan::Skipped(stored) => {
                let stored = stored.unwrap_or_else(|| fact.clone());
                let note = skip_note(mode, &fact.name);
                tracing::debug!(
                    target: TARGET,
                    name = %fact.name,
                    mode = mode.as_str(),
                    "lite write changed nothing"
                );
                return Ok(Some(WriteOutcome {
                    fact: stored,
                    mode: WriteMode::Skip,
                    created: false,
                    attempts: 0,
                    conflict: false,
                    note: Some(note),
                }));
            }
            WritePlan::Created(next)
            | WritePlan::Replaced(next)
            | WritePlan::Merged(next)
            | WritePlan::Appended(next) => next,
        };
        let created = existing.is_none();
        let _replaced = state.document.upsert_fact(&next);
        self.persist(&mut state).await?;
        #[cfg(feature = "sqlite")]
        self.fts.upsert(&next).await?;
        drop(state);
        drop(guard);
        self.refresh_embedding_index().await?;
        Ok(Some(WriteOutcome {
            fact: next,
            mode,
            created,
            attempts: 0,
            conflict: false,
            note: None,
        }))
    }
}

/// The note explaining why a write decided on [`WritePlan::Skipped`](crate::lite::write::WritePlan).
fn skip_note(mode: WriteMode, name: &str) -> String {
    match mode {
        WriteMode::Append => format!(
            "coding fact `{name}`: appending the incoming body would not change the stored body"
        ),
        _ => format!("coding fact `{name}` already exists; write mode `{mode}` left it unchanged"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use crate::config::CodingMemoryConfig;
    use crate::lite::CodingFactType;

    use super::super::embedding::test_support::{open_keyword, scope};
    use super::*;

    /// A short, deterministic retry budget for the exhaustion test.
    fn tight_policy(max_attempts: usize) -> WritePolicy {
        WritePolicy {
            max_attempts,
            backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(1),
        }
    }

    /// Opens a keyword-only store on a fresh directory.
    async fn store() -> (TempDir, LocalCodingMemory) {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        (temp, store)
    }

    /// A fact of the store's scope named `name` with `content` as its body.
    fn fact(name: &str, content: &str) -> CodingFact {
        CodingFact::new(&scope(), name, content)
    }

    #[tokio::test]
    async fn merge_unions_tags_keeps_the_stored_identity_and_takes_incoming_fields() {
        let (_temp, store) = store().await;
        store
            .write(
                fact("build", "cargo build")
                    .with_description("how to build")
                    .with_tags(vec!["build".to_owned()]),
            )
            .await
            .unwrap();

        let outcome = store
            .write_mode(
                fact("build", "cargo build --release")
                    .with_description("build the workspace")
                    .with_type(CodingFactType::Reference)
                    .with_tags(vec!["ci".to_owned(), "build".to_owned()]),
                WriteMode::Merge,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(outcome.mode, WriteMode::Merge);
        assert!(!outcome.created);
        assert!(!outcome.conflict);
        assert_eq!(outcome.attempts, 1);
        assert_eq!(outcome.note, None);
        assert_eq!(outcome.fact.name, "build");
        assert_eq!(outcome.fact.fact_type, CodingFactType::Project);
        assert_eq!(outcome.fact.scope, scope());
        assert_eq!(outcome.fact.content, "cargo build --release");
        assert_eq!(outcome.fact.description, "build the workspace");
        assert_eq!(
            outcome.fact.tags,
            vec!["build".to_owned(), "ci".to_owned()],
            "stored tags first, then the unseen incoming ones"
        );

        let stored = store.read("build").await.unwrap().unwrap();
        assert_eq!(stored.content, "cargo build --release");
        assert_eq!(stored.tags, outcome.fact.tags);
        assert_eq!(stored.updated_at, outcome.fact.updated_at);
    }

    #[tokio::test]
    async fn replace_overwrites_the_body_and_a_none_path_clears_it() {
        let (_temp, store) = store().await;
        store
            .write(fact("build", "cargo build").with_path("docs/BUILD.md"))
            .await
            .unwrap();

        let outcome = store
            .write_mode(
                fact("build", "totally different"),
                WriteMode::Replace,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(outcome.mode, WriteMode::Replace);
        assert!(!outcome.created);
        assert_eq!(outcome.fact.content, "totally different");
        assert_eq!(outcome.fact.path, None);

        let stored = store.read("build").await.unwrap().unwrap();
        assert_eq!(stored.content, "totally different");
        assert_eq!(stored.path, None);
    }

    #[tokio::test]
    async fn append_joins_the_bodies_and_a_no_op_append_is_a_skip() {
        let (_temp, store) = store().await;
        store.write(fact("build", "first body")).await.unwrap();

        let outcome = store
            .write_mode(
                fact("build", "second body"),
                WriteMode::Append,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(outcome.mode, WriteMode::Append);
        assert!(!outcome.created);
        assert_eq!(outcome.attempts, 1);
        assert_eq!(outcome.fact.content, "first body\n\nsecond body");
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "first body\n\nsecond body"
        );

        let reloads = store.reload_count();
        let skipped = store
            .write_mode(
                fact("build", "second body"),
                WriteMode::Append,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(skipped.mode, WriteMode::Skip);
        assert!(!skipped.created);
        assert_eq!(skipped.attempts, 1);
        assert_eq!(skipped.fact.content, "first body\n\nsecond body");
        assert!(
            skipped
                .note
                .as_deref()
                .is_some_and(|note| note.contains("append")),
            "{:?}",
            skipped.note
        );
        assert_eq!(store.reload_count(), reloads, "a skip touches no file");
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "first body\n\nsecond body"
        );
    }

    #[tokio::test]
    async fn skip_leaves_an_existing_fact_and_create_writes_a_new_one() {
        let (_temp, store) = store().await;
        store.write(fact("build", "cargo build")).await.unwrap();

        let skipped = store
            .write_mode(
                fact("build", "ignored"),
                WriteMode::Skip,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(skipped.mode, WriteMode::Skip);
        assert!(!skipped.created);
        assert_eq!(skipped.fact.content, "cargo build");
        assert!(skipped.note.is_some());
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "cargo build"
        );

        let created = store
            .write_mode(
                fact("fresh", "new body"),
                WriteMode::Create,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(created.mode, WriteMode::Create);
        assert!(created.created);
        assert!(!created.conflict);
        assert_eq!(created.attempts, 1);
        assert_eq!(created.fact.content, "new body");
        assert_eq!(
            store.read("fresh").await.unwrap().unwrap().content,
            "new body"
        );
    }

    #[tokio::test]
    async fn an_exhausted_create_conflict_is_a_typed_error_that_leaves_the_store_alone() {
        let (_temp, store) = store().await;
        store.write(fact("build", "cargo build")).await.unwrap();

        let error = store
            .write_mode(
                fact("build", "other body"),
                WriteMode::Create,
                tight_policy(2),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Memory(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("build"), "{message}");
        assert!(message.contains('2'), "{message}");
        assert!(
            message.contains("conflict"),
            "the message explains the conflict: {message}"
        );
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "cargo build",
            "an exhausted create never overwrites the stored fact"
        );
    }

    #[tokio::test]
    async fn a_merge_updates_the_fact_file_and_both_index_documents() {
        let temp = TempDir::new().unwrap();
        let store =
            LocalCodingMemory::builder(temp.path().to_path_buf(), CodingMemoryConfig::default())
                .scope(scope())
                .fact_files(true)
                .open()
                .await
                .unwrap();
        assert!(store.fact_files_enabled());
        store
            .write(fact("build", "cargo build").with_description("how to build"))
            .await
            .unwrap();

        let outcome = store
            .write_mode(
                fact("build", "cargo build --release"),
                WriteMode::Merge,
                WritePolicy::default(),
            )
            .await
            .unwrap();
        assert_eq!(outcome.mode, WriteMode::Merge);
        assert_eq!(outcome.fact.content, "cargo build --release");

        let fact_files: Vec<_> = std::fs::read_dir(store.facts_dir())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(fact_files.len(), 1, "{fact_files:?}");
        let fact_file = std::fs::read_to_string(&fact_files[0]).unwrap();
        assert!(fact_file.contains("cargo build --release"), "{fact_file}");
        assert!(
            fact_file.contains("description: how to build"),
            "{fact_file}"
        );

        let memory = std::fs::read_to_string(store.memory_path()).unwrap();
        assert!(memory.contains("cargo build --release"), "{memory}");
        let user = std::fs::read_to_string(store.user_path()).unwrap();
        assert!(
            !user.contains("cargo build --release"),
            "a project fact stays out of USER.md: {user}"
        );
        assert_eq!(
            store.read("build").await.unwrap().unwrap().content,
            "cargo build --release"
        );
    }
}
