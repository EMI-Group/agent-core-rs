//! The toolkit: construction, the model-facing `invoke`, the typed `execute` and paging.
//!
//! [`RetrievalToolkit::invoke`] is the model-facing shape: exactly `operation`, `cid`,
//! `query`, `cursor`, `page_size` and `top_k`, with an unknown key, a wrong JSON type and a
//! cursor that cannot be resumed all rejected. [`RetrievalToolkit::execute`] takes a typed
//! [`super::RetrievalRequest`]. Both share one execution path: resolve the scope, run the
//! operation, page the rows, render them, and mint the `[next]` cursor when entries remain.
use serde_json::Value;

use super::ops::{self, OperationOutcome};
use super::render::{self, RenderContext};
use super::validate::{self, CursorState, CursorStore, ValidatedRequest};
use super::{
    RetrievalOperation, RetrievalPromptSnapshot, RetrievalRequest, RetrievalResponse,
    RetrievalToolCard, RetrievalToolkitConfig, ABSTAIN_BLANK_QUERY, ABSTAIN_NO_RESULTS,
    CURSOR_FOOTER_CHARS, SKILL_RETRIEVAL_TOOL_NAME, TOOL_DESCRIPTION,
};
use crate::error::{validation, Result};
use crate::retrieval::search::service::{Retriever, RetrieverConfig, SearchRequest};
use crate::retrieval::types::LoadedIndex;

/// The suggestion attached to an abstain caused by a blank search query.
pub const BLANK_QUERY_SUGGESTION: &str = "Provide a short task phrase, or use operation=list or \
operation=browse to walk the categories.";

/// The suggestion attached to an abstain caused by a search with zero hits.
pub const NO_RESULTS_SUGGESTION: &str = "Rephrase with task-specific terms, or use operation=list \
or operation=browse to find the right branch.";

/// One retrieval toolkit over one loaded index and one session's page cursors.
pub struct RetrievalToolkit {
    /// The query-time runtime every operation reads.
    retriever: Retriever,
    /// The validated budgets and the session scope.
    config: RetrievalToolkitConfig,
    /// The per-session tool id.
    tool_id: String,
    /// The structural index fingerprint every minted cursor pins.
    fingerprint: String,
    /// The page cursors, least recently used first.
    cursors: CursorStore,
}

impl RetrievalToolkit {
    /// Opens a toolkit over `retriever` for one session.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error::Validation) naming `session_scope`,
    /// `max_output_chars`, `max_page_size`, `max_top_k` or `default_page_size` when the
    /// configuration is unusable.
    pub fn new(retriever: Retriever, config: RetrievalToolkitConfig) -> Result<Self> {
        validate::validate_config(&config)?;
        let tool_id = crate::discovery::tool_id(&config.session_scope, SKILL_RETRIEVAL_TOOL_NAME)?;
        let fingerprint = validate::index_fingerprint(retriever.index());
        Ok(Self {
            retriever,
            config,
            tool_id,
            fingerprint,
            cursors: CursorStore::default(),
        })
    }

    /// Opens a toolkit over an already-loaded index.
    ///
    /// The retriever is built with [`RetrieverConfig::top_k`] equal to
    /// [`RetrievalToolkitConfig::max_top_k`] (and the other retriever defaults), so one
    /// search can page over every hit up to the toolkit's own bound.
    ///
    /// # Errors
    ///
    /// The retriever's config validation, then [`RetrievalToolkit::new`]'s checks.
    pub async fn from_loaded_index(
        index: LoadedIndex,
        config: RetrievalToolkitConfig,
    ) -> Result<Self> {
        let retriever_config = RetrieverConfig {
            top_k: config.max_top_k,
            ..RetrieverConfig::default()
        };
        let retriever = Retriever::from_loaded_index(index, retriever_config).await?;
        Self::new(retriever, config)
    }

    /// The card the host publishes for this tool.
    #[must_use]
    pub fn card(&self) -> RetrievalToolCard {
        RetrievalToolCard {
            id: self.tool_id.clone(),
            name: SKILL_RETRIEVAL_TOOL_NAME.to_string(),
            description: TOOL_DESCRIPTION.to_string(),
            input_schema: super::tool_input_schema(),
            parallel_safe: false,
            stateless: false,
        }
    }

    /// The per-session tool id (`skill_retrieval__<sha256 hex>`).
    #[must_use]
    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    /// The loaded index every operation reads.
    #[must_use]
    pub fn index(&self) -> &LoadedIndex {
        self.retriever.index()
    }

    /// The configuration the toolkit was built with.
    #[must_use]
    pub fn config(&self) -> &RetrievalToolkitConfig {
        &self.config
    }

    /// The retriever behind the toolkit.
    #[must_use]
    pub fn retriever(&self) -> &Retriever {
        &self.retriever
    }

    /// How many page cursors the toolkit holds.
    #[must_use]
    pub fn cursor_count(&self) -> usize {
        self.cursors.len()
    }

    /// Drops every page cursor.
    pub fn close(&mut self) {
        self.cursors.clear();
    }

    /// Renders the retrieval prompt over this toolkit's index into `max_chars`.
    #[must_use]
    pub fn prompt_snapshot(&self, max_chars: usize) -> RetrievalPromptSnapshot {
        render::render_retrieval_prompt(self.index(), max_chars)
    }

    /// Runs one model-facing call.
    ///
    /// Accepts exactly `operation`, `cid`, `query`, `cursor`, `page_size` and `top_k`;
    /// `cursor` resumes an earlier page and rejects an operation, cid, query, page size or
    /// `top_k` that differs from the original request, or an index that changed since the
    /// page was minted.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error::Validation) for an unknown argument key,
    /// a value of the wrong JSON type, a misused field, or a cursor that cannot be
    /// resumed; [`Error::NotFound`](crate::error::Error::NotFound) for an unknown branch or
    /// item cid.
    pub async fn invoke(&mut self, arguments: &Value) -> Result<RetrievalResponse> {
        let request = validate::parse_invoke_arguments(arguments)?;
        self.execute(&request).await
    }

    /// Runs one typed request.
    ///
    /// An empty or missing search query and a zero-hit search return an abstain response
    /// rather than an error.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error::Validation) naming the first rejected
    /// field, [`Error::NotFound`](crate::error::Error::NotFound) for an unknown branch or
    /// item cid, and whatever the retriever reports for a search.
    pub async fn execute(&mut self, request: &RetrievalRequest) -> Result<RetrievalResponse> {
        let validated = validate::validate_request(request, &self.config)?;
        if validated.cursor.is_some() {
            return self.resume(&validated).await;
        }
        self.run(&validated, 0).await
    }

    /// Resumes the page the cursor names.
    async fn resume(&mut self, request: &ValidatedRequest) -> Result<RetrievalResponse> {
        let token = request.cursor.clone().unwrap_or_default();
        validate::validate_cursor_token(&token)?;
        let Some(state) = self.cursors.get(&token).cloned() else {
            return Err(validation(
                "cursor",
                "unknown or expired retrieval cursor; omit cursor to start a new request",
            ));
        };
        validate::validate_resume(&state, request, &self.fingerprint)?;
        self.cursors.remove(&token);
        let resumed = ValidatedRequest {
            operation: state.operation,
            cid: state.cid.clone(),
            cid_given: true,
            query: state.query.clone(),
            query_given: true,
            cursor: None,
            page_size: state.page_size,
            page_size_given: Some(state.page_size),
            top_k: state.top_k,
            top_k_given: Some(state.top_k),
        };
        self.run(&resumed, state.offset).await
    }

    /// Runs one validated request at `offset`.
    async fn run(
        &mut self,
        request: &ValidatedRequest,
        offset: usize,
    ) -> Result<RetrievalResponse> {
        let method = self.retriever.method().to_string();
        let outcome = match request.operation {
            RetrievalOperation::List => ops::list(self.index(), &request.cid)?,
            RetrievalOperation::Browse => ops::browse(self.index(), &request.cid)?,
            RetrievalOperation::Read => ops::read(self.index(), &request.cid)?,
            RetrievalOperation::Search => {
                if request.query.is_empty() {
                    return Ok(Self::abstain(
                        ABSTAIN_BLANK_QUERY,
                        BLANK_QUERY_SUGGESTION,
                        &method,
                        request.top_k,
                    ));
                }
                let result = self
                    .retriever
                    .search_details_with(
                        &request.query,
                        &SearchRequest {
                            top_k: Some(request.top_k),
                        },
                    )
                    .await?;
                if result.candidates.is_empty() {
                    return Ok(Self::abstain(
                        ABSTAIN_NO_RESULTS,
                        NO_RESULTS_SUGGESTION,
                        &result.method,
                        request.top_k,
                    ));
                }
                ops::search(&result)
            }
        };
        let outcome = OperationOutcome {
            method: if outcome.method.is_empty() {
                method
            } else {
                outcome.method
            },
            ..outcome
        };
        Ok(self.finish(request, outcome, offset))
    }

    /// Builds the abstain response for one pinned reason.
    fn abstain(reason: &str, suggestion: &str, method: &str, top_k: usize) -> RetrievalResponse {
        let (content, model_content) = render::render_abstain(reason, suggestion, method);
        RetrievalResponse {
            operation: RetrievalOperation::Search,
            content,
            model_content,
            entries: Vec::new(),
            cid: String::new(),
            path: Vec::new(),
            total_count: 0,
            shown_count: 0,
            truncated: false,
            truncation_reason: None,
            next_cursor: None,
            result_complete: true,
            abstained: true,
            abstain_reason: Some(reason.to_string()),
            suggestion: Some(suggestion.to_string()),
            method: method.to_string(),
            effective_top_k: top_k,
            page_size: 0,
        }
    }

    /// Pages, renders and mints the cursor for one operation outcome.
    #[allow(clippy::too_many_lines)]
    fn finish(
        &mut self,
        request: &ValidatedRequest,
        outcome: OperationOutcome,
        offset: usize,
    ) -> RetrievalResponse {
        let method = self.retriever.method().to_string();
        if let Some(read) = &outcome.read {
            let page = render::render_read(read, self.config.max_output_chars);
            return RetrievalResponse {
                operation: RetrievalOperation::Read,
                content: page.content,
                model_content: page.model_content,
                entries: outcome.entries,
                cid: outcome.cid,
                path: outcome.path,
                total_count: 1,
                shown_count: 1,
                truncated: page.shortened,
                truncation_reason: page.shortened.then(|| "max_output_chars".to_string()),
                next_cursor: None,
                result_complete: !page.shortened,
                abstained: false,
                abstain_reason: None,
                suggestion: None,
                method,
                effective_top_k: 0,
                page_size: 0,
            };
        }
        let total = outcome.entries.len();
        let page_size = request.page_size;
        let offset = offset.min(total);
        let end = (offset + page_size).min(total);
        let window = outcome.entries.get(offset..end).unwrap_or(&[]);
        let remaining_after_page = end < total;
        let footer_reserve = if remaining_after_page {
            CURSOR_FOOTER_CHARS
        } else {
            0
        };
        let context = RenderContext {
            operation: request.operation,
            cid: &outcome.cid,
            label: &outcome.label,
            description: &outcome.description,
            query: &request.query,
            method: &outcome.method,
            total,
            offset,
        };
        let page = render::render_page(
            &context,
            window,
            self.config.max_output_chars,
            footer_reserve,
        );
        let shown = page.shown_count;
        let remaining = offset + shown < total;
        let mut body = page.content;
        let mut model_content = page.model_content;
        let mut next_cursor = None;
        if remaining && shown > 0 {
            let state = CursorState {
                operation: request.operation,
                cid: outcome.cid.clone(),
                query: if request.operation == RetrievalOperation::Search {
                    request.query.clone()
                } else {
                    String::new()
                },
                offset: offset + shown,
                page_size,
                top_k: request.top_k,
                fingerprint: self.fingerprint.clone(),
            };
            let token = state.token();
            self.cursors.push(token.clone(), state);
            let footer = format!(
                "[next] {{\"operation\":\"{}\",\"cursor\":\"{token}\"}}",
                request.operation.as_str()
            );
            body = render::append_note(&body, &footer, None);
            model_content = render::append_note(&model_content, &footer, None);
            next_cursor = Some(token);
        }
        let budget_limited = page.shortened || (shown == 0 && offset < total);
        let truncated = budget_limited || remaining;
        let truncation_reason = if budget_limited {
            Some("max_output_chars".to_string())
        } else if remaining {
            Some("page".to_string())
        } else {
            None
        };
        let entries = window.iter().take(shown).cloned().collect();
        RetrievalResponse {
            operation: request.operation,
            content: body,
            model_content,
            entries,
            cid: outcome.cid,
            path: outcome.path,
            total_count: total,
            shown_count: shown,
            truncated,
            truncation_reason,
            next_cursor,
            result_complete: !remaining,
            abstained: false,
            abstain_reason: None,
            suggestion: None,
            method: outcome.method,
            effective_top_k: if request.operation == RetrievalOperation::Search {
                request.top_k
            } else {
                0
            },
            page_size,
        }
    }
}
