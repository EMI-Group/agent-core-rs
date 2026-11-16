# rule_compression

## Intent

Deterministic, model-free compression of large tool output, ported from Python's `processor/forked/offloader/rule_compression/`.
It is tier 1 of `MessageSummaryOffloader`: an oversized tool result is sniffed, typed, compressed by one focused rule compressor, and only a still-oversized remainder falls through to the model-backed summary/offload path.
Every path is synchronous, offline and reproducible: no model, no network, no shared mutable state, no `unsafe`, and the only I/O is the explicit debug dump and its reader.
The two acceptance gates are Python's: `meets_savings_ratio` (at least `RuleContext::min_savings_ratio` = 10 % of `len // 3` tokens) and, for JSON arrays and source files, `fits_budget_and_saves` (the candidate must also fit `max_tokens`).
Three subsystems replace Python's libraries: byte-level sniffing (`router::sniff`) for a mis-decoded payload, a typed JSON-repair ladder (`router::repair`) for `json_repair`, and a deterministic readability scoring ladder (`compressors::html::ladder`) for `BeautifulSoup` plus `trafilatura`.
Matching stays hand-rolled inside this node — no `Regex` is compiled here, although the crate depends on `regex` for its tokenizer nodes — and it is Unicode-aware: log lines fold full Unicode lowercasing and `is_timestamp_line` folds `NFKC` plus the Arabic-Indic decimal blocks.

## API Surface

- `types` — `ContentType` (`JsonArray`, `GitDiff`, `Html`, `SearchResults`, `Log`, `PlainText`, `SourceFile`, plus the `BUILD_OUTPUT` alias of `Log`; `as_str()` is Python's `ContentType.value`), `RuleContext` (budget, token policy, `min_savings_ratio`, `query_terms`, `tool_name`, `file_path`, every per-compressor knob with Python's defaults, `new`, `token_count`), `RuleCompressionResult` (`content`, `content_type`, `modified`, `lossy`, `details`; `unchanged`/`declined`/`applied`, and `with_details(self, Option<Value>)` as the attribution seam every constructor shares), `TokenCountPolicy`, `CHARACTERS_PER_TOKEN = 3`.
- `types` — `RuleCompressionStrategy` (`RuleThenTruncate` default, `RuleOnly`, `TruncateOnly`; serde `snake_case`, `as_str`, `runs_rule_compression`, `runs_fallback_offload`, `truncates_rule_preview`, `parse`); an unknown spelling is `Error::Config` from `parse` and a serde refusal on deserialize, never a silent default.
- `RuleCompressionResult::lossy` is meaningful only for an accepted candidate: `true` means information was dropped, `false` means a meaning-preserving rewrite, and the pipeline stamps the value into `rule_compression_lossy`.
- `common` — `count_tokens`, `meets_savings_ratio`, `fits_budget_and_saves`, `is_error_text`, `contains_word`, `find_word`, `extract_file_path_argument`, `is_source_file_path`, `file_name`, `strip_display_line_prefixes`, `strip_display_line_prefixes_preserving_body_whitespace`.
- `query_terms` — `extract_query_terms(user_content, tool_name, tool_arguments) -> BTreeSet<String>`; keys contribute terms, path-valued and non-user-input *values* do not.
- `router` — `RuleContentRouter::{new, detect, compress}`, the `ContentRouter` alias, `is_timestamp_line`, the crate-internal `is_log_line`, and the re-exports of the `repair` and `sniff` submodules.
- `RuleContentRouter::detect` logs the same `content_kind`/`reason` pair `compress` records (`binary_payload` / `replacement_damaged`, plus the source encoding and the damage counters), so detection alone explains a pass-through; a clean UTF-8 body logs nothing.
- `router::repair` — `repair_json(&str) -> Result<Value, JsonRepairFailure>`, `repair_json_array_checked` (the same plus the array check), `repair_json_array(&str) -> Option<Value>` (the router and JSON-array compressor entry point: one `tracing::debug!` per failure), the crate-internal `strip_code_fence`, and `JsonRepairFailure` (`NoJsonBody`, `UnbalancedBrackets`, `Unparsable`, `NotAnArray`; `as_str` labels `no_json_body`, `unbalanced_brackets`, `unparsable`, `not_an_array`, plus `Display` and `std::error::Error`).
- `router::sniff` — `sniff_content(&str)`, `sniff_bytes(&[u8])`, `SniffKind` (`Utf8`, `Utf8Bom`, `Utf16Le`, `Utf16Be`, `Binary`, `ReplacementHeavy`; `as_str`, `source_encoding`, `is_passthrough`), `SniffedContent` (`text`, `kind`, `converted`, `nul_bytes`, `control_ratio`, `replacement_characters`, `decode_replacements`, `dropped_bytes`; `text()`, `is_passthrough()`, `source_encoding()`, `details()`), and the thresholds `BINARY_SNIFF_BYTES` (8 KiB), `MAX_CONTROL_RATIO_PERCENT` (5), `REPLACEMENT_CHAR_MIN` (8), `REPLACEMENT_CHAR_RATIO_PERCENT` (5).
- `SniffedContent::details()` is `None` only for a clean UTF-8 body (`kind = Utf8` with `decode_replacements` and `dropped_bytes` both `0`); otherwise it carries `content_kind`, `source_encoding`, `recoded`, `nul_bytes`, `control_ratio`, `replacement_characters`, `decode_replacement_characters` and `dropped_bytes`.
- `compressors` — the `RuleCompressor` trait plus `JsonArrayCompressor`, `DiffCompressor`, `HtmlCompressor`, `LogCompressor` (`LogLevel`, `LogLine`), `SearchResultsCompressor` (`SearchMatch`, `parse_search_match`), `PlainTextCompressor`, `SourceFileCompressor`.
- `compressors::html` — `HtmlCompressor` (gating and the `details` report), `HTMLExtractor`/`HTMLExtractorConfig`/`HtmlOutputFormat`, `HTMLExtractionResult` (`extracted`, lengths and ratio, `title`/`author`/`date`/`metadata`, `removed_node_count`, `link_count`, `table_count`, `code_block_count`, `output_block_count`, `duplicate_block_count`, `report_rows_removed`, `selected_block_count`, `boilerplate_block_count`, `best_region_score`, `ladder_fallback`, `reduction_percent`).
- `compressors::html::ladder` (module-private) — `BlockMetrics`, `BlockScore`, `BlockAnalysis`, `LadderOutcome` (`blocks`, `dropped`, `best_run_score`, `fallback`), `analyse_blocks`, `select_blocks`, `measure`, `best_run`, `KEEP_THRESHOLD = 1.0`, `MIN_REGION_VISIBLE_CHARS = 80`.
- `compressors::html::report` (module-private) — `looks_like_report_html`, `prune_report_rows`.
- `pipeline` — `RuleCompressionPipeline::{new, with_clock, with_dump, with_dump_format, with_strategy, current_time, context_for, compress, compress_with_scope, compress_in_messages, compress_in_messages_with_scope, dump_enabled, dump_dir, dump_format, strategy, router, context_character_capacity}`, `RuleClock`, `ADD_PASS`/`TTL_PASS`, and the six metadata keys `RULE_COMPRESSED_AT_KEY`, `RULE_COMPRESSION_TYPE_KEY`, `RULE_COMPRESSION_PASS_KEY`, `RULE_COMPRESSION_LOSSY_KEY`, `RULE_COMPRESSION_DETAILS_KEY`, `RULE_COMPRESSION_DUMP_PATH_KEY`.
- `dump` — `DumpRecord` (with `lossy`), `RuleCompressionDumpRecord`, `DumpTemplateVars` (`new`, `from_scope`, `session_or_unknown`, `context_or_unknown`), `RuleCompressionDumpFormat` (`Json` default, `Jsonl`; `as_str`, `parse`), `write_dump`, `write_record`, `dump_file_name`, `jsonl_file_name`, `resolve_dump_dir`, `expand_dump_dir_template`, `utc_date_stamp`, `safe_filename_part`, `safe_filename_part_or`, and the constants `DEFAULT_DUMP_DIR`, `RULE_COMPRESSION_DUMP_DIR_ENV`, `UNKNOWN_SESSION_ID`, `UNKNOWN_CONTEXT_ID`.
- `dump::jsonl` — `append_jsonl`, `read_dump_file`, `read_dump_dir`, `read_dump_text` (re-exported from `dump`).
- `summary_offloader` (the tier-1 host) — `RuleCompressionTierConfig` (`strategy`, `dump_format`; `new`, `from_json_value`, `runs_rule_compression`, `runs_fallback_offload`, `truncates_rule_preview`) reached through `MessageSummaryOffloader::{with_rule_compression_tier, rule_compression_tier}`.
- `MessageSummaryOffloaderConfig` carries the four rule-compression knobs — `enable_rule_compression` (default `true`), `rule_compression_max_chars` (default 20 000), `enable_rule_compression_dump`, `rule_compression_dump_dir` — all serde-defaulted so old configs load; the strategy and the dump format live on `RuleCompressionTierConfig` instead, because `tests/rule_compression.rs` constructs the config struct exhaustively.

## Constraints

- Every compressor is a pure function of `(content, ctx)` and must never panic on malformed input: parse failures and unusable candidates return the input with `modified = false`.
- Detection precedence is fixed: named source file → a pass-through sniffed body (binary or replacement-heavy) short-circuits to `PlainText` → JSON array → HTML → log → git diff → search results (≥ 30 % of non-empty lines, timestamp-shaped lines excluded) → plain text.
- Display line-number prefixes are stripped before sniffing; git diffs keep their body whitespace, plain text and source files are routed verbatim.
- The sniff order is fixed: a `FF FE`/`FE FF` byte-order mark re-decodes the body from UTF-16, a leading UTF-8 BOM is stripped, and a NUL/control-heavy or replacement-heavy body is passed through untouched.
- A latin-1 projection of a UTF-16 BOM is only reinterpreted when it is faithful: an odd byte count or a character above `U+00FF` refuses it with a debug reason and keeps the text.
- Decoding damage is attributed rather than hidden: a UTF-8 decode that substituted `U+FFFD` is reported as `ReplacementHeavy` (a `U+FFFD` the payload genuinely encoded is not counted as damage), and UTF-16 damage is reported as `decode_replacements`/`dropped_bytes`.
- A pass-through body is returned unchanged as `ContentType::PlainText` with the reason in `details.content_kind`, because `ContentType` has no binary variant; `sniff.details()` is merged into a compressor's own `details` only when the body was re-encoded or passed through.
- A body the repair ladder cannot fix is a typed `JsonRepairFailure` plus one `tracing::debug!` line; nothing is silently truncated into something that parses.
- No `unwrap`/`expect` in library code, no `println!`, `tracing` only.
- Files stay under ~1 000 lines — the largest are `compressors/html/extract.rs` (908) and `router/repair.rs` (901) — so new behaviour goes into a new submodule.
- `details` are emitted for rejected candidates where Python emits them (diff, log, HTML, search); a declined diff that parsed no file reports `reason: "no_files_parsed"` with `original_line_count`/`preamble_line_count`, while the too-short exit still emits no `details`.
- The honesty contract is that every decline a compressor can explain carries a `reason` (or a `digest_stage`).
- A short log reports `below_min_lines`; a log past `log_stack_trace_max_lines` reports `stack_traces_truncated`, `stack_trace_truncated_lines` and `stack_trace_dropped_lines`; every JSON-array exit carries a `reason` (`unparsable_array`, `empty_array`, `no_object_keys`, `too_few_rows`, `candidate_does_not_fit`, `candidate_unchanged`) and reports `unmeasurable_rows`.
- A search-result body with no parsable hit reports `no_parsable_hits` with `unparsed_line_count`, and blank source content reports `digest_stage: "none"` plus `reason: "blank_content"`.
- `compress_in_messages`/`compress_in_messages_with_scope` return `Error::Validation` naming the index and the message count for an index the conversation does not own, never a fabricated empty user message.
- A non-text content kind, a compressor decline and an empty query-term set each log their reason (`non_text_content`, the compressor's `reason`, `no_user_message`/`non_text_user_message`/`empty_user_message`), because "nothing to do" must not look like "cannot do it".

## Design Decisions

- `count_tokens(text, ctx)` is `max(len // 3, 1)`; `TokenCountPolicy::CharactersPerToken(n)` lets a caller re-measure without a tokenizer.
- Only JSON-array and source-file compression enforce `max_tokens`; every other compressor enforces only the savings ratio, matching Python's `meets_savings_ratio` calls.
- The pipeline re-bases `max_tokens`/`head_tokens`/`tail_tokens` of the supplied context from `max_chars` (`/3` and `/6`, floored at one) because Python's entry point takes only the character budget.
- `force` (the TTL pass) compresses regardless of size; `max_chars` still becomes the pass budget, so a forced pass is a real compression, not a no-op.
- `with_strategy` gates the whole pass: `TruncateOnly` makes every pipeline call a no-op, because this pipeline *is* the rule tier.
- `compress_in_messages` resolves the triggering tool call through `offloader::resolve_tool_call`, so `tool_name`, `file_path` and the query terms come from the conversation; a message with no resolvable call still compresses with an empty hint set.
- The `*_with_scope` entry points take the caller's `OffloadScope`, so a dump records the real session and context ids; the plain `compress`/`compress_in_messages` pass `OffloadScope::default()`, which is why `unknown_session`/`unknown_context` only appear for a scopeless standalone caller.
- Query terms come from the latest `Role::User` message (its text only), the tool name, and the tool-call arguments; CJK text contributes only its ASCII runs, documented in `query_terms`.
- The repair ladder is a fixed ten-step sequence — code fence, JSON span, smart quotes, single quotes, Python literals, unquoted keys, trailing commas, close truncated, drop incomplete tail, wrap sequence/bare members — and each step only runs when the earlier one left something to fix.
- Single quotes are rewritten before Python literals, so `'True'` stays a string; `first_parsable` refuses a closed or cut-back candidate that would turn a one-element body into an empty container.
- `is_timestamp_line` requires a 19-character ISO `YYYY-MM-DD[T ]HH:MM:SS` prefix at fixed character positions with 14 numeric positions, after `NFKC` folding and mapping of the Arabic-Indic blocks (`U+0660`–`U+0669`, `U+06F0`–`U+06F9`) that `NFKC` leaves alone.
- The HTML extractor is a tokenizer plus a block renderer: noise subtrees (`script`, `style`, `noscript`, `template`, `svg`, `nav`, `footer`, `aside`, `form`, `iframe`, `head`) are removed, entities decoded, and title/headings/links/code blocks/lists/tables/paragraphs rendered; duplicates are dropped and report tables prune "passed" rows once a failure row exists.
- The tag block list is only the ladder's secondary filter: the readability ladder scores every rendered block and keeps the best contiguous region, so an unmarked menu, sidebar or footer is dropped as well.
- Ladder measurements are `visible`/`links`/`markup`/`digits`/`punctuation`, then `text_density`, `link_density`, a repetition-or-marker `boilerplate_ratio`, `signal_density = 0.5 * min(digit_ratio / 0.05, 1) + 0.5 * min(punctuation_ratio / 0.10, 1)` and a binary `structural_bonus`.
- The ladder score is `1.0 * ln(1 + visible) / 5 + 2.0 * text_density + 1.5 * (1 - 2 * link_density) + 1.0 * signal_density + 0.5 * structural_bonus - 5.0 * boilerplate_ratio - 1.0 * link_penalty`, with the link penalty applied over 80 % links or to a block under 120 characters that is over half links.
- `best_run` maximises the sum of `score - KEEP_THRESHOLD` over a contiguous run, preferring the longer run and then the earlier one, so a region has to earn its length; blocks inside that run still scoring below `KEEP_THRESHOLD = 1.0` are dropped.
- A selection that keeps fewer than `MIN_REGION_VISIBLE_CHARS = 80` visible characters — or nothing at all — is replaced by the whole document and reported through `ladder_fallback` (`empty_region`/`region_too_small`), so text is never lost to the ladder.
- `HtmlCompressor` accepts a candidate only when it is at least `html_min_content_chars` long and passes the savings ratio, marks every accepted candidate `lossy = true`, and its `details` carry the extraction counters plus `selected_block_count`, `boilerplate_block_count`, `best_region_score` and `ladder_fallback`.
- The diff compressor parses unified diffs into files and hunks, keeps `diff_max_files` files and `diff_max_hunks_per_file` hunks by score, retains `diff_max_changed_lines_per_hunk`/`diff_max_changed_lines_total` changed lines and `diff_max_context_lines` context lines, and always reports `should_offload_original` (`compressed_lines < 0.8 * original_lines`) when it parsed.
- The log compressor classifies levels, keeps the first/last/top errors and failures, deduplicates warnings after normalising addresses, numbers and paths, keeps up to `log_max_stack_traces` traces of `log_stack_trace_max_lines` lines, widens by `log_error_context_lines`, and caps at `log_max_total_lines`.
- The search compressor groups `path:line:content` hits per file, scores query-term and error/warning/importance hits, caps files, an adaptive total and `search_max_matches_per_file`, and marks every omitted group.
- `SourceFileCompressor` strips comment-only lines and blank runs when the file type has a comment syntax, then keeps imports, signatures, query-term hits and the head/tail; Python's compressor is a no-op, so this is a documented extension rather than a parity requirement.
- `PlainTextCompressor` reports `lossy = true` when it dropped a duplicate line, collapsed a blank run or rewrote a line's internal spacing, and `false` only for a faithful rewrite; trimming leading and trailing blank lines alone does not make a result lossy.
- The pipeline stamps `rule_compressed_at`, `rule_compression_type`, `rule_compression_pass`, `rule_compression_lossy` (always), `rule_compression_details` (only when the payload is truthy) and `rule_compression_dump_path` (only when a dump was written), and logs one `[RuleCompression] applied …` line per application.
- Dump directory precedence is the configured value → `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR` → `DEFAULT_DUMP_DIR = ".openjiuwen/rule_compression_dumps"`, so a dump always has somewhere to go; every candidate expands the `{session_id}`, `{context_id}`, `{date}` and `{timestamp}` templates, and the date templates are expanded first so an id containing `{date}` cannot re-trigger expansion.
- The JSON format (the default) writes one pretty-printed object per application as `{epoch_ms}_{pass}_{type}_{tool_call_id}.json`; the JSONL format appends one compact object per line to `{session or "session"}-{context or "context"}.jsonl`, so a conversation accumulates in one file.
- A dump carries the real `OffloadScope` ids (sanitized like any filename part) and its `lossy` flag; a failed write is reported once per directory through `tracing::warn!` naming the path and the error, and never fails the pass.
- Dumps are readable: `read_dump_file`/`read_dump_dir`/`read_dump_text` accept both formats, skip blank lines, sort a directory read by file name, and report a malformed record as `Error::Validation` naming `source:line`.
- `MessageSummaryOffloader` runs tier 1 only on its existing candidates, never on a message that already carries `rule_compressed_at`, and only when the text exceeds `rule_compression_max_chars`; it passes `ADD_PASS` and the request's `OffloadScope`.
- The tier's strategy decides the fallthrough: `rule_then_truncate` keeps a rule-compressed body that fits the budget and summarises/offloads a larger one, `rule_only` keeps an oversized rule body and never summarises or offloads a candidate the rule tier declined, and `truncate_only` skips the rule tier entirely.

## Routing Table

- `./mod.rs` → module docs, declarations and the public re-exports
- `./types.rs` → `ContentType`, `RuleContext`, `RuleCompressionResult`, `TokenCountPolicy`, `RuleCompressionStrategy`
- `./common.rs` → token accounting, acceptance gates, path lookup, display-prefix stripping, word matching
- `./query_terms.rs` → relevance-term extraction
- `./router.rs` → detection precedence, dispatch, `is_timestamp_line`, log-line patterns, sniff integration
- `./router/repair.rs` → the ten-step JSON-repair ladder and `JsonRepairFailure`
- `./router/sniff.rs` → `sniff_content`/`sniff_bytes`, `SniffKind`, `SniffedContent` and the sniff thresholds
- `./pipeline.rs` → per-pass budgets, strategy gating, metadata stamping, dump dispatch, the `tracing` line
- `./dump/mod.rs` → dump records, naming, directory resolution and template expansion, the JSON writer
- `./dump/jsonl.rs` → the JSONL writer and the reader for both formats
- `./compressors/mod.rs` → the `RuleCompressor` trait and re-exports
- `./compressors/html/mod.rs` → `HtmlCompressor` gating and the `details` report
- `./compressors/html/extract.rs` → the dependency-free tokenizer, block renderer and `HTMLExtractionResult`
- `./compressors/html/ladder.rs` → the readability scoring ladder (`best_run`, `KEEP_THRESHOLD`, the whole-document fallback)
- `./compressors/html/report.rs` → report-page detection and row pruning
- `./compressors/html/tests.rs` → extraction, entity/table rendering and compressor-gate tests
- `./compressors/json_array.rs` → CSV/JSONL rendering and salient-row digests
- `./compressors/log.rs` → level classification, dedupe, context widening, caps
- `./compressors/diff.rs` → unified-diff parsing, hunk selection, offload decision
- `./compressors/search_results.rs` → hit parsing, scoring, adaptive selection
- `./compressors/plain_text.rs` → line dedupe, blank-run collapse, the honest `lossy` flag
- `./compressors/source_file.rs` → comment stripping and structure digest

## Test Strategy

- `cargo test -p openjiuwen-context --offline` is green: 379 library unit tests, 651 tests across 34 integration binaries, and 16 doc tests (1 046 in total).
- Rule-compression suites: `tests/rule_compression.rs` (17 tests — detection, the compressor gates, token accounting, metadata stamping, query terms, the tier host), `tests/rule_compression_operators.rs` (45 — the repair ladder, sniffing and encodings, the pinned sniff thresholds, localized timestamps, the readability ladder, and the decline-attribution reasons) and `tests/rule_compression_dump.rs` (13 — dump naming, templates, both formats, replay, malformed records, the three tier strategies, the `lossy` metadata and typed strategy/format errors); `tests/offloaders.rs` (28) covers the offloader family that hosts the tier.
- Unit tests sit beside every module: `types` (strategy parsing and gates), `common`, `query_terms`, `router` (detection precedence, localized timestamps, binary pass-through), `router::repair`, `router::sniff`, `pipeline` (size gate, force, `lossy` stamping, `truncate_only`, dump enabling and the per-directory warning), `dump` (filename sanitizing, template expansion), `dump::jsonl` (append/replay, malformed line), `compressors/html/ladder` (scoring and `best_run` ties), `compressors/html/report`, and every compressor.
- Deterministic offline: real temp directories for dumps (`tempfile`), an injected `RuleClock` where a timestamp matters, no network and no wall-clock ordering.
- The suites assert the honesty contract directly — a declined candidate returns the input with `modified = false`, an accepted plain-text digest reports `lossy = true`, a repair failure leaves the body untouched, and every decline a compressor can explain carries a `reason` (or a `digest_stage`).

## Deferred

- The rest of `json_repair`'s grammar: block comments, schema-anchored repairs, and a scalar-only JSON Lines body (`1\n2\n3`), which still fails to repair because comma insertion only separates bracketed or quoted values.
- Python's trafilatura/readability producer scores (`content_score`, tag-ratio heuristics): the built-in ladder is a transparent fixed-weight model instead of a tuned corpus.
- TTL pass scheduling in the host processor: the pipeline exposes `TTL_PASS` and `force`, but `MessageSummaryOffloader` only ever runs the `add` pass.
