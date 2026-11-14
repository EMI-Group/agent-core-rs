# sharing

## Intent
Packaging experience records into repeatable skill bundles: the wire types, deterministic keyword extraction, the approval-gated stager, and the pluggable backend seam with two offline stores plus a remote-hub client.
Ports the offline part of Python `openjiuwen.agent_evolving.sharing`: `types.py`, `keyword_extractor.py`, `share_stager.py` and `backends/local_file.py`.
The remote hub client is **new design, not a Python port** — the Python package only knows the local filesystem — so its HTTP hub protocol v1 is defined by `remote.rs` and is documented there.

## API Surface
- `SharingMeta`, `SharedExperience`, `SharedSkillBundle`, `SkillPackageMeta`, `SkillSearchResult`, `QueryKeywords` — pure serde wire data.
- `SharedSkillBundle::new` derives `bundle_id` as `bundle-{16 hex}` of an FNV-1a hash over the skill id and the sorted distinct record ids; `SharedExperience` carries the record payload plus its provenance.
- `KeywordExtractor` — deterministic keyword and summary extraction (`extract`), with the shared `experience::stopwords` vocabulary.
- `ShareStager` — `min_confidence`, `require_approval`, `max_experiences`; `build_bundle(records, meta)` filters, dedupes, wraps and keywords; `publish`/`share` enforce the approval gate; `StagedBundle` is the pending result.
- `SharingMeta::default()` uses `upload_trigger = "user_approval"` and `confidence = 0.7`; `SharedSkillBundle::new` requires a non-blank `skill_id` and derives `bundle_id` itself.
- `SharingBackend` (async trait) — `publish`, `list`, `get`, `remove`, `search`; implementations `LocalFileSharingBackend` (one JSON file per bundle id), `MemorySharingBackend` and `RemoteHubSharingBackend`.
- Both offline backends share one ranking rule, `score = 0.7 * keyword_ratio + 0.3 * intent_ratio`.
- `RemoteHubOptions` — `base_url`, `api_key`, `headers`, `timeout` (30 s), `max_retries` (2); `new`, `with_api_key`, `with_header(s)`, `with_timeout`, `with_max_retries`, `from_env`, `validate` (normalizes a trailing `/` in place).
- `RemoteHubSharingBackend` — `new(options)`, `from_url(url)`, `options()`, `health()`, plus the `SharingBackend` implementation.
- `HUB_URL_ENV` (`OPENJIUWEN_HUB_URL`) and `HUB_API_KEY_ENV` (`OPENJIUWEN_HUB_API_KEY`) name the two variables `RemoteHubOptions::from_env` reads; the URL is required, the key is optional.

## Hub protocol v1
| Method | Path | Request body | Success reply |
| --- | --- | --- | --- |
| `POST` | `/v1/hub/bundles` | the `SharedSkillBundle` JSON | `{"bundle_id": "..."}` |
| `GET` | `/v1/hub/bundles` | — | `{"bundles": [<bundle>, ...]}` |
| `GET` | `/v1/hub/bundles/{bundle_id}` | — | the bundle JSON |
| `DELETE` | `/v1/hub/bundles/{bundle_id}` | — | `{"removed": bool}` |
| `POST` | `/v1/hub/bundles/search` | `{"query": <QueryKeywords>, "limit": n}` | `{"results": [<SkillSearchResult>, ...]}` |
| `GET` | `/health` | — | any `2xx`; body ignored |

All paths hang off the validated `base_url` (no trailing `/`, path prefixes preserved); `bundle_id` segments are percent-encoded by the client.
`GET`/`DELETE` answer `404` for an unknown id, which the client maps to `Ok(None)`/`Ok(false)`.
The client sends `accept: application/json`, every `headers` entry, and `Authorization: Bearer <api_key>` when a key is set — a static `Authorization` header is overridden by the bearer token.
`RemoteHubOptions`' manual `Debug` redacts the key and prints header names only, because a static header may carry its own credential.

## Error mapping
- Connection failure and any non-4xx non-2xx status (including a `5xx` whose retry budget is spent) → `Error::transport`.
- A spent request deadline → `Error::timeout` (the shared `HttpClient` draws that line).
- Any `4xx` (including a `429` whose retry budget is spent) → `Error::validation`, quoting the hub envelope `{"error":{"code","message"}}` or a flattened, 200-character body snippet.
- A body that is not the expected JSON → `Error::validation` naming the endpoint label (`POST /v1/hub/bundles`, `GET /health`, …).
- Retries cover `429`/`500`/`502`/`503`/`504` only, `max_retries` additional attempts after the first, 50 ms initial backoff capped at 500 ms; `4xx` is never retried.

## Constraints
- Deterministic: bundle ids are derived from content, never from `uuid4` or a clock, so re-publishing the same records yields the same id.
- The stager never mutates or deletes input records; a rejected record stays in the local archive untouched.
- `build_bundle` drops records that fail `ExperienceRecord::validate` or fall below `min_confidence`, and rejects a bundle with no surviving record; `max_experiences` `0` means unlimited.
- Backends are async and return `Error::Io` (or `Error::External` for a poisoned lock); the in-memory backend guards its map with a `Mutex` that is never held across an `.await`.
- The remote backend normalises a sloppy hub client-side: `list` sorts by `bundle_id`, `search` drops `0.0` scores, sorts by descending score then ascending `bundle_id`, applies `limit` itself and short-circuits an empty query without a request, and `publish` falls back to the bundle's content-derived id when the hub echoes a blank one.
- `RemoteHubSharingBackend::new` validates once, so an invalid base URL is rejected before any request; only `http`/`https` with a non-empty host is accepted.
- `remote.rs` keeps its tests in the sibling `remote/tests.rs` (`#[cfg(test)] mod tests`), so the transport is covered without a `tests/` integration binary; the rest of the module's coverage lives in `crates/openjiuwen-agent-evolving/tests/sharing.rs`.

## Design Decisions
- A content-derived `bundle_id` replaces Python's random one, which keeps tests, re-publishes and file names reproducible.
- The offline port collapses Python's stager-plus-upload-queue into one `build_bundle` step and leaves approval explicit in `publish`/`share`.
- The trait takes string ids rather than paths so a remote hub or database slots in beside the two offline backends.
- Keyword extraction is local tokenising and ranking instead of a model call, so bundles carry usable retrieval metadata with no network.
- The hub client is built on `openjiuwen_common::http` rather than `reqwest` directly, so timeouts, retries and error kinds stay consistent with every other transport in the workspace.
- The `api_key` lives as a plain `Option<String>` in the options (no `SecretString` plumbing reaches this crate yet) and is turned into a bearer token per request, because `HttpClientConfig` deliberately drops `Authorization` from its default headers.
- `RemoteHubOptions::validate` takes `&mut self` so the normalised URL is what the backend stores and what callers can inspect through `options()`.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./types.rs` → wire types, `SharedSkillBundle`, bundle-id derivation
- `./keywords.rs` → `KeywordExtractor`
- `./stager.rs` → `ShareStager`, `StagedBundle`, approval and quality gates
- `./backend.rs` → `SharingBackend`, `LocalFileSharingBackend`, `MemorySharingBackend` and their private `score_bundle` ranking
- `./remote.rs` → `RemoteHubOptions`, `RemoteHubSharingBackend`, hub protocol v1, error mapping, client-side normalisation
- `./remote/tests.rs` → loopback `wiremock` tests for the hub client (module unit tests)
