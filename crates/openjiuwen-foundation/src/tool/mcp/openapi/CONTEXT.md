# openapi — OpenAPI→tools MCP transport

## Intent
Expose one MCP tool per operation of an `OpenAPI` document, called over plain HTTP against `servers[0].url` (or the `params.base_url` override) instead of JSON-RPC.
Ports the spec-facing half of `openjiuwen/core/foundation/tool/mcp/client/openapi_client.py` (`load_conf`, the `servers[0].url` handling, `parse_openapi_to_http_routes`) without the Python `fastmcp`/`jsonschema-path` dependencies.
A spec loads from a filesystem path (`.json`, `.yaml`, `.yml`, or sniffed) or an `http(s)` URL, and every document its `$ref`s reach is loaded eagerly at that point and cached with the spec, so a broken reference fails the load instead of the first call.
The Playwright sibling of the Python transport stays deferred in `../client.rs` (`Error::Unsupported`, browser runtime absent).
Compiled only behind the crate's default `mcp` feature; nothing here reaches outside that gate.

## API Surface
`spec`:
- `OpenApiSpec` — `load(source)` / `load_with(&HttpClient, source)` (async), `from_str(text)`, `from_value(Value)`, `base_url() -> &str`, `operations() -> &[OpenApiOperation]`, `document() -> &Value` (the root document), `resolve(&str) -> Result<Value>` (any `$ref`, including cross-document ones), `source_url() -> Option<&str>`.
- `from_str`/`from_value` build an *inline* spec: it has no file or URL base, so only local `#/...` pointers resolve and an external `$ref` is an `Error::Validation`.
- `OpenApiOperation { name, description, method /* upper-case */, path, parameters, body_schema, body_required, input_schema }` and `OpenApiParameter { name, location, required, schema, description }`, both `Debug + Clone + PartialEq + Serialize + Deserialize` with `#[serde(default)]`.
- `OpenApiParameter::location` is one of `path`, `query`, `header`, `cookie` (lower-case), and every other declared location is skipped with a `tracing::debug!`.
`refs` (crate-private):
- `RefContext` — the root `CachedDocument` plus a `BTreeMap` of externally loaded ones: `inline(Value)`, `load(Value, DocumentBase, &HttpClient)` (async prefetch), `root_document()`, `resolve(&str)`, `resolve_schema(&Value)`, `maybe_ref(&Value)`.
- `DocumentBase { Inline, File(PathBuf), Url(String) }` with `key()` (cache key and diagnostic label) and `resolve_document(document_part, reference)` (relative file paths join the declaring document's directory, relative URLs join the declaring URL, absolute `http(s)` URLs are used as written).
- Limits: `MAX_REF_DEPTH = 32` hops per `$ref` chain, `MAX_REMOTE_DOCUMENTS = 32` external documents per spec; a fetch uses a 30 s timeout.
`call`:
- `execute` (only I/O), `build_url`, `query_pairs`, `header_pairs -> Result<Vec<(String, String)>>`, `cookie_header -> Result<Option<String>>`, `json_body`, `MAX_ERROR_BODY_CHARS = 512`.
`mod`:
- `OpenApiClient` — `new(&McpServerConfig)`, `with_http_client(&McpServerConfig, HttpClient)`, `config()`, `spec() -> Option<Arc<OpenApiSpec>>`, and the `McpClient` impl (`connect` loads the cached spec, `disconnect` drops it, `list_tools` maps operations, `call_tool_with` executes the matching operation; unknown names are `Error::Mcp`).
- Naming, description assembly, flattened input schema, and response mapping are documented in `../CONTEXT.md` and unchanged here.

## `$ref` resolution rules
A reference is split into a document part and a JSON pointer fragment (`#/a/b`, `#`, or empty for the whole document).
Local `#/...` pointers resolve inside the document the reference appears in, and nested references inside `properties` stay untouched (shallow, documented policy).
A relative file path resolves against the directory of the declaring document, an absolute file path is used as written, and paths are normalised lexically (no filesystem path resolution).
A relative reference made by a URL-loaded document is itself a URL: it is joined RFC 3986-style (`Url::join`) and fetched over HTTP, so a remote document never reaches the local filesystem.
Only `http(s)` URLs and local file paths are followed: any other scheme (`file:`, `ftp://`, …) is a validation error.
Referenced documents are read with the same strict-by-extension parser as the root (`.json` strict JSON, `.yaml`/`.yml` strict YAML, everything else JSON then YAML), and a referenced file must exist, be a regular file, and not be a symbolic link.
Every failure is an `Error::Validation` naming the reference as written, the document it was resolved in, and the JSON pointer; a chain that revisits a document/pointer pair is reported as a cycle, and a chain longer than `MAX_REF_DEPTH` names the depth bound.

## Cookie handling
`in: cookie` parameters are collected like any other parameter and appear in the flattened input schema, but their values travel only in one `Cookie` request header.
`cookie_header` renders them in declaration order as `name=value`, percent-encodes value bytes outside the RFC 6265 `cookie-octet` set, and joins pairs with `; `.
A missing optional cookie parameter is omitted; a missing required one is an `Error::Validation` naming the tool and the parameter.
`header_pairs` merges the assembled header into a caller-supplied `Cookie` header (declared header parameter or `auth_headers` entry) as `existing; ours`, so callers never see a duplicate `Cookie` header.
Cookie parameters never enter the query string or the JSON body.

## Constraints
Request bodies: only `application/json` is mapped onto tool arguments; `multipart/form-data` and `application/x-www-form-urlencoded` bodies leave `body_schema = None` and log one `tracing::warn!` per operation naming the ignored media types.
All HTTP goes through `openjiuwen_common::http::HttpClient` (`HttpRequest::get`/`execute`); no `reqwest` use, no `unsafe`, `tracing` only, doc comment on every public item.
No new crate dependencies: `serde_json`, `url`, `tokio::fs`, `std::path`, and `std::fs::symlink_metadata` cover loading, joining, and caching.
Files stay around ≤1000 lines: the resolution engine lives in `refs.rs` so `spec.rs` stays at the operation walk.
The load path performs no writes; the only filesystem effects are reads of the spec and of the documents it references.

## Tests
`../openapi_tests.rs` (22 tests, inline specs plus `tempfile` files and loopback `wiremock` only, nothing reaches the network) covers naming, descriptions, flattened schemas (path/query/header/cookie + merged JSON body), local `$ref` resolution/cycles/missing targets, YAML↔JSON parity, strict `.json` parsing, GET/POST calls, non-2xx results, unknown tools, a URL-loaded spec through `connect_mcp_server`, the Playwright `Unsupported` dispatch, client-type round-trips, an empty `server_path`, `build_url` encoding, the ignored multipart body, remote `./common.yaml#/…` / `../common.yaml#/…` / absolute file refs, `http(s)` refs with RFC 3986 relative joins, cyclic/dangling/`file:`-scheme ref errors, and cookie headers (joined, omitted when optional, percent-encoded, merged into a caller `Cookie`, required-missing error).

## Routing Table
- `./spec.rs` → document loading, base-URL resolution, operation collection, shallow `$ref` inlining, request-body/media-type handling
- `./refs.rs` → `RefContext`, document bases, eager prefetch with cache, cross-document `$ref` resolution, cycle/depth/limit errors
- `./call.rs` → URL/query/header/cookie/body request building and response mapping
- `./mod.rs` → `OpenApiClient` and its `McpClient` implementation
- `../openapi_tests.rs` → the transport's test suite (inline specs, `tempfile`, loopback wiremock)
