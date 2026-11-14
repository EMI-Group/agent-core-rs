# rule_compression

## Intent

Deterministic, model-free compression of large tool output, ported from Python's `processor/forked/offloader/rule_compression/`.
It is tier 1 of `MessageSummaryOffloader`: an oversized tool result is typed, compressed by one focused rule compressor, and only a still-oversized remainder falls through to the model-backed summary/offload path.
Every path is synchronous, offline and reproducible: no model, no network, no shared mutable state, no `unsafe`, and the only I/O is the explicit debug dump.
The two acceptance gates are Python's: `meets_savings_ratio` (at least `RuleContext::min_savings_ratio` = 10 % of `len // 3` tokens) and, for JSON arrays and source files, `fits_budget_and_saves` (the candidate must also fit `max_tokens`).

## API Surface

- `types` — `ContentType` (`JsonArray`, `GitDiff`, `Html`, `SearchResults`, `Log`, `PlainText`, `SourceFile`, plus the `BUILD_OUTPUT` alias of `Log`; `as_str()` is Python's `ContentType.value`), `RuleContext` (budget, token policy, `min_savings_ratio`, `query_terms`, `tool_name`, `file_path`, every per-compressor knob with Python's defaults), `RuleCompressionResult { content, content_type, modified, lossy, details }`, `TokenCountPolicy`, `CHARACTERS_PER_TOKEN = 3`.
- `common` — `count_tokens`, `meets_savings_ratio`, `fits_budget_and_saves`, `is_error_text`, `contains_word`, `find_word`, `extract_file_path_argument`, `is_source_file_path`, `file_name`, `strip_display_line_prefixes`, `strip_display_line_prefixes_preserving_body_whitespace`.
- `query_terms` — `extract_query_terms(user_content, tool_name, tool_arguments) -> BTreeSet<String>`; keys contribute terms, path-valued and non-user-input *values* do not.
- `router` — `RuleContentRouter::{new, detect, compress}`, the `ContentRouter` alias, and `repair_json_array` (the bounded stand-in for `json_repair`).
- `compressors` — the `RuleCompressor` trait plus `JsonArrayCompressor`, `DiffCompressor`, `HtmlCompressor` (`HTMLExtractor`, `HTMLExtractorConfig`, `HTMLExtractionResult`, `HtmlOutputFormat`), `LogCompressor` (`LogLevel`, `LogLine`), `SearchResultsCompressor` (`SearchMatch`, `parse_search_match`), `PlainTextCompressor`, `SourceFileCompressor`.
- `pipeline` — `RuleCompressionPipeline::{new, with_clock, with_dump, current_time, context_for, compress, compress_in_messages, dump_enabled, dump_dir, router}`, `RuleClock`, `context_character_capacity(tokens) = tokens * 3`, the `ADD_PASS`/`TTL_PASS` names and the five `RULE_COMPRESSION_*` metadata keys.
- `dump` — `DumpRecord`, `write_dump`, `resolve_dump_dir`, `dump_file_name`, `safe_filename_part`, `RULE_COMPRESSION_DUMP_DIR_ENV`.
- `MessageSummaryOffloaderConfig` (in `summary_offloader`) — `enable_rule_compression` (default `true`), `rule_compression_max_chars` (default 20 000), `enable_rule_compression_dump`, `rule_compression_dump_dir`; all serde-defaulted so old configs load.

## Constraints

- Every compressor is a pure function of `(content, ctx)` and must never panic on malformed input: parse failures and unusable candidates return the input with `modified = false`.
- Detection precedence is fixed: named source file → JSON array → HTML → log → git diff → search results (≥ 30 % of non-empty lines) → plain text.
- Display line-number prefixes are stripped before sniffing; git diffs keep their body whitespace, plain text and source files are routed verbatim.
- Matching is ASCII and hand-rolled (no `regex` crate): log-line, timestamp, search-line and HTML-tag sniffing fold ASCII case only.
- No `unwrap`/`expect` in library code, no `println!`, `tracing` only, files stay well under 1 000 lines.
- `details` are emitted for rejected candidates where Python emits them (diff, log, HTML, search); the diff's two early exits (too short, unparsable) emit none.

## Design Decisions

- `count_tokens(text, ctx)` is `max(len // 3, 1)`; `TokenCountPolicy::CharactersPerToken(n)` lets a caller re-measure without a tokenizer.
- Only JSON-array and source-file compression enforce `max_tokens`; every other compressor enforces only the savings ratio, matching Python's `meets_savings_ratio` calls.
- The pipeline re-bases `max_tokens`/`head_tokens`/`tail_tokens` of the supplied context from `max_chars` (`/3` and `/6`, floored at one) because Python's entry point takes only the character budget.
- `force` (the TTL pass) compresses regardless of size; `max_chars` still becomes the pass budget, so a forced pass is a real compression, not a no-op.
- `compress_in_messages` resolves the triggering tool call through `offloader::resolve_tool_call`, so `tool_name`, `file_path` and the query terms come from the conversation; a message with no resolvable call still compresses with an empty hint set.
- Query terms come from the latest `Role::User` message (its text only), the tool name, and the tool-call arguments; CJK text contributes only its ASCII runs, documented in `query_terms`.
- The HTML extractor is a tokenizer plus a block renderer: noise subtrees (`script`, `style`, `noscript`, `template`, `svg`, `nav`, `footer`, `aside`, `form`, `iframe`, `head`) are removed, entities decoded, and title/headings/links/code blocks/lists/tables/paragraphs rendered; duplicates are dropped and report tables prune "passed" rows once a failure row exists.
- The diff compressor parses unified diffs into files and hunks, keeps `diff_max_files` files and `diff_max_hunks_per_file` hunks by score, retains `diff_max_changed_lines_per_hunk`/`diff_max_changed_lines_total` changed lines and `diff_max_context_lines` context lines, and always reports `should_offload_original` (`compressed_lines < 0.8 * original_lines`) when it parsed.
- The log compressor classifies levels, keeps the first/last/top errors and failures, deduplicates warnings after normalising addresses, numbers and paths, keeps up to `log_max_stack_traces` traces of `log_stack_trace_max_lines` lines, widens by `log_error_context_lines`, and caps at `log_max_total_lines`.
- The search compressor groups `path:line:content` hits per file, scores query-term and error/warning/importance hits, caps files, an adaptive total and `search_max_matches_per_file`, and marks every omitted group.
- `SourceFileCompressor` strips comment-only lines and blank runs when the file type has a comment syntax, then keeps imports, signatures, query-term hits and the head/tail; Python's compressor is a no-op, so this is a documented extension rather than a parity requirement.
- The pipeline stamps `rule_compressed_at`, `rule_compression_type`, `rule_compression_pass`, `rule_compression_details` (only when the payload is truthy) and `rule_compression_dump_path`, and logs one `[RuleCompression] applied …` line per application.
- Dump names are `{epoch_ms}_{pass}_{type}_{tool_call_id}.json`; the directory is the configured value, else `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR`, else no dump (the Python workspace fallback needs a `ModelContext` this pipeline does not hold); a failed dump logs a warning and keeps the compressed message.
- `MessageSummaryOffloader` runs tier 1 only on its existing candidates, never on a message that already carries `rule_compressed_at`, and only when the text exceeds `rule_compression_max_chars`; a result that fits is kept as is, otherwise the summary/offload path runs on the compressed body and stores that body.

## Deferred

- `json_repair`'s wider grammar (unquoted keys, comments, partial literals): the bounded repair handles a code fence, trailing prose, trailing commas and single quotes only.
- Python's trafilatura/BeautifulSoup readability ladder (`_json_repair`-style fallbacks, `readability-lxml` scoring): the built-in extractor is a heuristic block renderer instead.
- Python's `regex`-level Unicode matching in the log/timestamp/search sniffers (non-ASCII digits and spaces are not recognised).
- The Python workspace dump directory (`<workspace>/context/<session>_context/rule_compression_logs`) and the `{session_id}`/`{context_id}` template expansion, which need a context handle.
- Python's strategy switch (`rule_then_truncate`/`rule_only`/`truncate_only`), the debug JSONL trace, and the TTL pass scheduling stay in the host processor; the Rust pipeline exposes one explicit `compress` call per pass.

## Routing Table

- `./mod.rs` → module docs, declarations and the public re-exports
- `./types.rs` → `ContentType`, `RuleContext`, `RuleCompressionResult`, `TokenCountPolicy`
- `./common.rs` → token accounting, acceptance gates, path lookup, display-prefix stripping, word matching
- `./query_terms.rs` → relevance-term extraction
- `./router.rs` → detection precedence, dispatch, `repair_json_array`
- `./pipeline.rs` → per-pass budgets, metadata stamping, the `tracing` line
- `./dump.rs` → dump naming, directory resolution, JSON writing
- `./compressors/mod.rs` → the `RuleCompressor` trait and re-exports
- `./compressors/html.rs` → dependency-free extractor and renderer
- `./compressors/json_array.rs` → CSV/JSONL rendering and salient-row digests
- `./compressors/log.rs` → level classification, dedupe, context widening, caps
- `./compressors/diff.rs` → unified-diff parsing, hunk selection, offload decision
- `./compressors/search_results.rs` → hit parsing, scoring, adaptive selection
- `./compressors/plain_text.rs` → line dedupe and blank-run collapse
- `./compressors/source_file.rs` → comment stripping and structure digest
