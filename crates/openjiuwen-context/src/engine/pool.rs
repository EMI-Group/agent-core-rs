//! The context pool: creation, lookup and removal of contexts.
//!
//! A context's pool identity is `(session_id, normalized context_id)`, where
//! dots in the context id become underscores, matching Python's pool identity.
//! This module also resolves what a new context is built with: a per-context
//! [`ContextOptions`](super::ContextOptions) value wins over the engine default,
//! which in turn wins over the configuration-derived fallback.
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_session::Session;

use crate::config::{ContextConfig, TokenCounterKind};
use crate::context::{ConversationContext, ModelContext};
use crate::processor::{ContextPipeline, PipelineDeps, ProcessorConfig};
use crate::recall::{InMemoryRecallArchiveStore, RecallArchiveStore};
use crate::session_memory::{InMemorySessionMemoryStore, SessionMemoryStore};
use crate::telemetry::recorder::TokenProvenance;
use crate::tokenizers::TokenizerSource;

use super::{ContextEngine, ContextOptions, DEFAULT_SESSION_ID};

/// Python's `context_id.replace(".", "_")`: the pool identity of a context.
fn normalize_context_id(context_id: &str) -> String {
    context_id.replace('.', "_")
}

/// How the configuration's own token counter produces its numbers.
///
/// The kinds this crate builds itself are known exactly, so the record can name
/// them; [`TokenCounterKind::Model`] resolves through the tokenizer selector,
/// whose outcome (a native artefact, a `tiktoken` vocabulary, or the string-length
/// estimate) is not visible through a boxed counter, so it is reported as an
/// unnamed estimate instead of being guessed exact.
fn config_counter_provenance(config: &ContextConfig) -> TokenProvenance {
    match config.token_counter.kind {
        TokenCounterKind::Approx => TokenProvenance::estimated(
            "approx_token_counter",
            "estimated by the approximate characters-per-token counter",
        ),
        TokenCounterKind::Characters => TokenProvenance::estimated(
            "characters_token_counter",
            "estimated from the character count of every message",
        ),
        TokenCounterKind::Tiktoken => TokenProvenance::from_source(TokenizerSource::Tiktoken),
        TokenCounterKind::Model => TokenProvenance::unverified(
            "the model-aware tokenizer selection is not introspectable through a boxed counter",
        ),
    }
}

impl ContextEngine {
    /// Creates or returns the context `context_id` of `session`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) when
    /// the configuration is invalid and the pipeline's own error (for example a
    /// summarisation processor without a model) when it cannot be built.
    pub async fn create_context(
        &self,
        context_id: impl Into<String>,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Arc<ConversationContext>> {
        self.create_context_with(
            context_id,
            ContextOptions {
                session,
                ..ContextOptions::default()
            },
        )
        .await
    }

    /// Creates or returns a context with explicit options.
    ///
    /// The call is idempotent: when the pool already holds the
    /// `(session, context)` pair, the existing context is returned and
    /// `options` is ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error::Config) when
    /// [`ContextConfig::validate`](crate::config::ContextConfig::validate) fails.
    pub async fn create_context_with(
        &self,
        context_id: impl Into<String>,
        options: ContextOptions,
    ) -> Result<Arc<ConversationContext>> {
        self.config.validate()?;
        let context_id = context_id.into();
        let session_id = options.session.as_ref().map_or_else(
            || DEFAULT_SESSION_ID.to_owned(),
            |session| session.id().to_owned(),
        );
        self.ensure_context(&context_id, &session_id, options).await
    }

    /// Inserts `context_id` into the pool under `session_id`, or returns the
    /// existing entry.
    pub(super) async fn ensure_context(
        &self,
        context_id: &str,
        session_id: &str,
        options: ContextOptions,
    ) -> Result<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        if let Some(existing) = self.contexts.read().await.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let model = options.model;
        let caller_counter = options.token_counter;
        let counter_supplied = caller_counter.is_some();
        let counter = caller_counter.unwrap_or_else(|| self.config.token_counter());
        let configs = options
            .processors
            .unwrap_or_else(|| self.config.processors.clone());
        // An explicit per-context store wins over the engine default, which in
        // turn wins over the configuration-derived fallback.
        let recall = options.recall.or_else(|| self.recall.clone()).or_else(|| {
            self.config
                .compression_recall
                .enabled
                .then(|| Arc::new(InMemoryRecallArchiveStore::new()) as Arc<dyn RecallArchiveStore>)
        });
        let needs_session_memory = configs.iter().any(|config| {
            matches!(
                config,
                ProcessorConfig::FullCompact(_) | ProcessorConfig::MessageSummaryOffloader(_)
            )
        });
        let session_memory = options
            .session_memory
            .or_else(|| self.session_memory.clone())
            .or_else(|| {
                needs_session_memory.then(|| {
                    Arc::new(InMemorySessionMemoryStore::new()) as Arc<dyn SessionMemoryStore>
                })
            });
        // A per-context recorder/analyzer wins over the engine default; when
        // both are unset the context stays uninstrumented.
        let recorder = options.recorder.or_else(|| self.recorder.clone());
        // The record's `before`/`after` counts are measured by the recorder while
        // the per-role statistic comes from the pipeline, which counts with this
        // same counter; installing it here keeps the two totals equal by
        // construction. A counter the caller set on the recorder wins and is
        // never replaced, and a counter supplied per context cannot describe
        // itself, so its provenance is reported as an unnamed estimate.
        if let Some(recorder) = recorder.as_ref() {
            let provenance = if counter_supplied {
                TokenProvenance::unverified(
                    "the counter was supplied for this context and does not declare how it counts",
                )
            } else {
                config_counter_provenance(&self.config)
            };
            recorder.install_token_counter(Arc::clone(&counter), provenance);
        }
        let usage_analyzer = options
            .usage_analyzer
            .or_else(|| self.usage_analyzer.clone());
        let pipeline = ContextPipeline::from_configs_with(
            &configs,
            &PipelineDeps::new(Arc::clone(&self.store))
                .with_model(model.clone())
                .with_recall(recall.clone())
                .with_session_memory(session_memory.clone())
                .with_recorder(recorder.clone()),
        )?;
        let context = Arc::new(ConversationContext::new(
            context_id,
            session_id,
            Arc::clone(&self.config),
            counter,
            pipeline,
            Arc::clone(&self.store),
            model,
            options.session,
            recall,
            session_memory,
            recorder,
            usage_analyzer,
        ));
        if !options.history.is_empty() {
            context.set_messages(options.history).await?;
        }

        // A concurrent creation may have won the race while the pipeline was
        // being built; the first entry wins.
        let mut pool = self.contexts.write().await;
        if let Some(existing) = pool.get(&key) {
            return Ok(Arc::clone(existing));
        }
        pool.insert(key, Arc::clone(&context));
        Ok(context)
    }

    /// Returns the context `context_id` of `session_id`, when it exists.
    pub async fn get_context(
        &self,
        session_id: &str,
        context_id: &str,
    ) -> Option<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        self.contexts.read().await.get(&key).map(Arc::clone)
    }

    /// Removes and returns the context `context_id` of `session_id`.
    pub async fn remove_context(
        &self,
        session_id: &str,
        context_id: &str,
    ) -> Option<Arc<ConversationContext>> {
        let key = (session_id.to_owned(), normalize_context_id(context_id));
        self.contexts.write().await.remove(&key)
    }
}
