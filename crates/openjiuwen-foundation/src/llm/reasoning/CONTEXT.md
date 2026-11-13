# reasoning — provider-neutral reasoning intent and per-provider wire encoding

## Intent

Turn a provider-neutral "thinking" request into the exact parameters a vendor expects, and write them into an OpenAI-compatible request body.
Ports `openjiuwen/core/foundation/llm/reasoning_profiles.py` (the declarative provider/model capability table) and `openjiuwen/core/foundation/llm/reasoning.py` (the intent model, the resolver, and the wire encoders).
The layer has three stages, each usable on its own: intent (`ReasoningConfig`), resolution (profile lookup plus encoder → `ReasoningPlan`), and application (`apply_reasoning_plan` into a `serde_json::Value`).
Everything is pure data plus pure functions: no I/O, no clock, no global mutable state, and the whole capability table lives in read-only static memory.

## API Surface

- `ReasoningMode` — `Auto` (default), `Enabled`, `Disabled`; snake_case serde.
- `ReasoningConfig` — the intent: `mode`, `effort: Option<String>`, `budget_tokens: Option<u32>`; builders `new`, `with_mode`, `with_effort`, `with_budget_tokens`, `disabled`; predicates `is_no_intent`, `is_disabled`, `has_disabled_effort`, `effective_effort`; `from_value(&Value) -> Result<Option<Self>>`.
- `ReasoningCapability` — owned `options` plus `recommended`, `standard(protocol)`, `supports(effort)`; `Default` is the chat-completions set.
- `CapabilitySpec` — `const`-constructible capability used in the static table: `new`, `is_empty`, `supports`, `to_capability`, `options_list`.
- `ReasoningProtocol` — `ChatCompletions` (default), `Responses`, `AnthropicMessages`; snake_case serde.
- `ReasoningProfile` — one provider-scoped model family: `provider`, `model_patterns`, `capability`, `anthropic_capability`, `wire`, `anthropic_wire`, plus `new`, `with_anthropic_capability`, `wire_for`, `capability_for`, `capability_spec_for`.
- `ReasoningPlan` — the resolved output: `sdk_params` (top-level body keys), `extra_body` (vendor extensions), `warnings`; `new`, `is_empty`, `has_warnings`, `sdk_param_names`, `extra_body_names`.
- `provider_key_for(provider, api_base) -> String` — the table's provider vocabulary for a configured provider name.
- `matches_pattern(pattern, model) -> bool` — case-insensitive `*`/`?` glob match.
- `find_reasoning_profile(provider, model) -> Option<&'static ReasoningProfile>` — first matching table entry.
- `reasoning_capability(provider, model)`, `reasoning_capability_for(protocol, provider, model)`, `standard_reasoning_capability(protocol)` — user-selectable levels and their recommendation.
- `resolve_reasoning_plan(provider, api_base, model, &intent)` — resolve for chat completions; `resolve_reasoning_plan_for(protocol, provider, api_base, model, &intent)` for an explicit wire protocol.
- `apply_reasoning_plan(&mut body, &plan)` and `apply_reasoning_plan_with(&mut body, &plan, override_existing)` — write a plan into a JSON body.
- Helpers `is_disabled_effort(effort)` and `normalize_effort(effort)` (`"none"` folds onto `"off"`).
- Tables `REASONING_PROFILES` (provider-scoped, specific to broad) and `MODEL_REASONING_FALLBACKS` (cross-provider, model-name only).

## Wire Contract

- `provider_key_for` normalizes the provider name with the `llm::provider` vocabulary (so `qwen` becomes `dashscope`) and consults the `api_base` host hint only for a generic name (`""`, `openai`, `openaicompatible`), so an explicit vendor name wins over the host.
- Known hosts map to `openrouter`, `dashscope`, `deepseek`, `moonshot`, `minimax`, `modelarts`, `qianfan`, `zhipu`, `volcengine`, and `mimo`.
- `find_reasoning_profile` scans `REASONING_PROFILES` in table order (provider equality plus model glob), then `MODEL_REASONING_FALLBACKS` against both the full model id and its last `/` segment.
- Chat completions treats a `deepseek` key, spelling, or `deepseek.com` base as `deepseek`, falls back to the OpenRouter encoder for an unprofiled `openrouter` model, and otherwise encodes a generic effort.
- `standard_openai` sends `reasoning_effort` (disable → `"none"`); `standard_anthropic` sends `thinking` (adaptive) plus `output_config.effort`, adding `budget_tokens` when given.
- `anthropic_manual` maps a level onto `thinking.budget_tokens` (minimal 1024, low 2048, medium 8192, high/xhigh/max 16384; plain `enabled` uses the medium budget because the API minimum is 1024).
- `dashscope_budget` sends `thinking: enabled` with an optional budget; `qwen38` sends `enable_thinking` plus `thinking_budget` or a normalized `reasoning_effort`.
- `qwen38_anthropic` and `dashscope_glm_anthropic` send `thinking` plus `output_config.effort`, where an explicit numeric budget wins and travels alone.
- `enable_thinking` (also used by `qianfan_toggle`) sends `enable_thinking` plus an optional `thinking_budget`.
- `thinking_toggle` puts `thinking` in the extension body, `anthropic_toggle` puts it at the top level, and both answer only an explicit mode.
- `deepseek` sends `thinking` in the extension body plus a top-level `reasoning_effort` (`xhigh` becomes `max` only on `pro` variants); `deepseek_anthropic` sends both at the top level.
- `glm` and `modelarts_deepseek` send `thinking` plus `reasoning_effort` (GLM: low/medium/high → `high`, xhigh/max → `max`; ModelArts collapses every level onto `high`); the `*_anthropic` variants put the effort in the extension body.
- `kimi_k3` sends `reasoning_effort` from K3 levels (ultra/xhigh/max → `max`, medium/high → `high`, the rest → `low`); the Kimi K2 toggle families send `thinking`.
- `minimax_m2` always sends `reasoning_split: true`; `minimax_m3` sends `thinking` (`adaptive`) in the extension body and `minimax_m3_anthropic` at the top level.
- `volcengine_seed` sends `reasoning_effort` and encodes disable as `minimal`; `openrouter` sends one normalized `reasoning` object (`effort`, `enabled`, `max_tokens`) in the extension body, leaving model ids with an unknown vendor prefix alone.
- `unsupported` and `always_on` send nothing; `ReasoningProtocol::Responses` sends a nested `reasoning: {effort}` object, and `null` for an explicit `disabled`.
- `../openai/request.rs` reads `ModelRequestConfig::extra["reasoning"]` through `ReasoningConfig::from_value`, resolves and applies the plan, consumes the intent key, and merges `extra` last so caller keys still win.

## Constraints

- Resolution depends only on its arguments and the static tables: no I/O, no environment, no randomness, so bodies are reproducible offline.
- Invalid intent JSON is `Error::Validation`; an unknown effort, unknown model, or unknown encoder name degrades to "send nothing" instead of panicking.
- `apply_reasoning_plan` never removes a key and never reorders the body; non-object bodies and empty plans are a no-op.
- No new dependencies: only `serde`, `serde_json`, and `openjiuwen_common::error` on top of what the crate already uses.
- Every public item is documented; `missing_docs`, `unreachable_pub`, and `clippy --all-targets -- -D warnings` stay clean.
- Logging comes from the request builder with target `openjiuwen_foundation::llm::reasoning`: warnings at debug, applied parameter names at trace; model names are logged, credentials never (this module handles none).
- `profiles.rs` is the largest file at ~967 lines and stays under the ~1000-line rule; the type definitions live in `mod.rs`, the encoders in `plan.rs`.

## Design Decisions

- The plan's `extra_body` parameters are merged at the **top level** of the request body because Python nests them under `extra_body` only for the `OpenAI` SDK to fold in; this crate builds the final body directly.
- `sdk_params` fill only missing keys, `extra_body` merges one object level with the plan winning, so a caller's unrelated fields inside `thinking` survive and an explicit caller key still overrides the plan through `ModelRequestConfig::extra`.
- Capability checking drops only the unsupported effort (or both effort and budget for a family with no controls) and keeps at most one warning per plan; an explicit mode still encodes, so `enabled` is never silently discarded.
- Warnings are carried on the plan rather than logged by the resolver, which keeps resolution pure and lets the caller decide how to surface them.
- An intent that asks for nothing resolves to an empty plan, so bodies from callers that never opted in stay byte-identical.
- `ReasoningConfig::from_value` accepts shorthands (`true`/`false`, mode keywords, a bare level) and treats an object with a key outside `mode`/`effort`/`budget_tokens` as a legacy raw payload to forward verbatim.
- `wire_for`/`capability_for` take the protocol as a parameter instead of branching at every call site; the profile keeps Python's `openai_wire`/`anthropic_wire` pair as `wire`/`anthropic_wire`.
- `CapabilitySpec` is `const`-constructible so the table lives in read-only memory; `ReasoningProfile` and `CapabilitySpec` are `Serialize` only, never deserialized.
- The `api_base` host hint is deliberately stricter than Python: an explicit non-generic provider name is authoritative, because in this crate the provider name *is* the endpoint profile.
- Per-family normalizers (`normalize_deepseek_effort`, `normalize_glm_effort`, `normalize_kimi_k3_effort`, `normalize_qwen38_effort`, `normalize_volcengine_seed_effort`, `normalize_modelarts_effort`) mirror the Python tables, so a vendor change is a table edit plus at most one normalizer.

## Test Strategy

- Unit tests in `mod.rs` (`inline_tests`) cover mode serde, partial/unknown-field intents, `from_value` objects, legacy payloads, shorthands, invalid intents, capability defaults, plan helpers, and plan round-trips.
- `plan_tests.rs` (declared from `plan.rs` with `#[path]`) drives every encoder through the public `resolve_reasoning_plan`/`resolve_reasoning_plan_for`, plus each normalizer table, provider resolution, warnings, Responses encoding, and Moonshot family classification.
- `tests.rs` iterates the whole table: every entry uses a known encoder (`KNOWN_WIRES`, 25 names), every capability is lower case, unique and self-consistent, every entry is reachable by a sample model id, no provider/pattern pair is duplicated, and resolution never panics.
- `tests.rs` also asserts end-to-end body effects: a resolved plan fills the body, an empty plan or a warning-only plan leaves it byte-identical, and caller `extra` overrides a plan key.
- `profiles_parity_tests.rs` pins the tables to the Python reference: every row of `REASONING_PROFILES` and `MODEL_REASONING_FALLBACKS` (provider, patterns, capability options/recommended, `wire`, `anthropic_wire`) is compared in Python order against the hard-coded transcription in `profiles_python_rows.rs`, and the audited families are resolved through the public lookup API; Python is never read at test time.
- `plan_tests.rs` proves encoder coverage: every table `wire`/`anthropic_wire` outside the guard set (`always_on`, `minimax_m2`, `unsupported`) dispatches to a real `encode_wire` arm and produces a non-empty plan for a level its capability offers.
- Everything is offline and deterministic — pure functions, no `wiremock`, no environment mutation; `cargo test -p openjiuwen-foundation reasoning` runs the module.

## Known Gaps

- `build_chat_completion_body` calls `resolve_reasoning_plan` with `api_base = None`, so an api-base host hint is reachable only through `provider_key_for` and the public `resolve_reasoning_plan*` functions.
- No client speaks the Responses API, so `ReasoningProtocol::Responses` encoding is reachable only via `resolve_reasoning_plan_for`.
- The Anthropic Messages request builder does not read `extra["reasoning"]`; its reasoning wiring is owned by `../anthropic/`.
- The capability tables are static: no live capability negotiation, endpoint discovery, or per-account level list.
- No reasoning-content capture or replay here; assistant `reasoning_content` replay is `openai::replays_reasoning_content`.
- `apply_reasoning_plan` can only add or merge keys — it cannot delete a key or force a `null` outside an encoded `sdk_params` value.

## Routing Table

- `./mod.rs` → intent, capability, protocol, profile, and plan types plus their helpers and inline tests
- `./profiles.rs` → static profile/fallback tables, provider-key resolution, host hints, glob matching, profile lookup, capability lookup
- `./profiles_parity_tests.rs` → Python-parity row pins, audited-family resolution, provider-vs-fallback precedence (compiled as `profiles.rs`'s test module)
- `./profiles_python_rows.rs` → hard-coded transcription of the Python profile and fallback tables, used only by the parity tests
- `./plan.rs` → resolution entry points, the encoder dispatch, per-family normalizers, and intent sanitizing
- `./apply.rs` → writing a plan into a request body without clobbering caller keys
- `./plan_tests.rs` → per-encoder and normalizer suites (compiled as `plan.rs`'s test module)
- `./tests.rs` → whole-table invariants and end-to-end intent/plan/body tests
