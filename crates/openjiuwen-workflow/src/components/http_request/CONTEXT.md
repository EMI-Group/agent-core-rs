# `http_request` component

## Intent

The built-in `http_request` workflow component: a Python `HttpComponentConfig` tree is turned into one request, executed through `openjiuwen_common::http::HttpClient`, and projected into Python's `statusCode` / `headers` / `body` / `url` / `ok` wire object.
Every request kind — JSON, text, form, binary, multipart — travels through `HttpClient`, so retries, cancellation, and the response size cap apply uniformly and are never reimplemented here.
Digest and AWS SigV4 authentication are implemented locally (Python only declares the configuration); MD5 is hand-rolled and SigV4 is hand-rolled over `sha2`, mirroring `openjiuwen_foundation::store::external::s3::sigv4`.
The module is split by responsibility so each file stays small and testable; `mod.rs` owns the component, the input merge, the request build, and the send/replay flow.

## API Surface

- `mod.rs` — `HttpRequestComponent` (`new(config)`, `config()`), the `Component` impl (`invoke`/`stream`/`collect`/`transform`), `PreparedRequest { request, digest_uri }`, and `pub const COMPONENT_HTTP_ERROR: u32 = 102_002`.
- `config.rs` — the public `Http*` configuration tree (`HttpComponentConfig`, `HttpRequestParamConfig`, `HttpHeadersConfig`, `HttpRequestBodyConfig`, `HttpContentType`, `HttpAuthConfig`, `HttpAuthType`, `HttpAuthIn`, `HttpResponseHandlingConfig`, `HttpResponseFormat`, `HttpResponseMode`, `HttpRetryConfig`, `HttpBackoffType`, `HttpAdvancedOptionsConfig`, `HttpRateLimitConfig`, `HttpRateLimitUnit`); field names and wire values match Python and are unchanged by this module.
- `template.rs` — `resolve_template(value, inputs)` (embedded `${path}` first, then `{{key}}` placeholders) and `stringify(value)` (compact JSON text); `pub(crate)` inside the crate.
- `body.rs` — `RequestBody { Json, Text, Bytes }` with `payload()`/`payload_len()`, `prepare_body(params, inputs) -> (Option<RequestBody>, Option<String>)` (body plus the content type it wants), and `enforce_max_body_length(max, body)`.
- `multipart.rs` — `encode(fields, inputs) -> EncodedMultipart { boundary, body }` with `content_type()` = `multipart/form-data; boundary=…`.
- `auth.rs` — `apply_authentication` (basic/bearer/api_key plus digest credential validation), `sign_aws(...)` (pre-flight SigV4 headers), and the `AWS_*` env constants.
- `digest.rs` — `DigestChallenge::parse`, `Algorithm { Md5, Md5Sess, Sha256, Sha256Sess }`, `authorization_with(challenge, user, password, method, uri, cnonce, nonce_count)`, `client_nonce()`, `md5_hex`/`md5`.
- `sigv4.rs` — `Credentials`, `SigningRequest`, `Signed { authorization, amz_date, payload_hash, session_token }`, `sign`, `canonical_request`, `string_to_sign`, `signing_key`, `hmac_sha256`, `rfc3986_encode`.
- `response.rs` — `ResponseBody { Json, Text, Base64, Buffer }` with `into_wire()`, `ResponseView::new(&HttpResponse)`, `resolved_format`, `is_failure`, `handle_response(params, &HttpResponse) -> Result<Value>`, `http_error`.
- `rate_limit.rs` — `RateLimiter` (`new(&HttpRateLimitConfig) -> Option<Arc<Self>>`, pure `wait_for(now) -> Duration`, async `acquire(&CancellationToken)`).

## Wire shapes and codes

- Response object: `{"statusCode": u16, "headers": {…}, "body": <body>, "url": str, "ok": bool}` (`ok` = status not in `response_code_failure_codes` and `< 400`).
- `response_format = binary` → `body` is the base64 string of the bytes (Python parity, lossy through JSON).
- `response_format = buffer` → `body` is `{"type": "Buffer", "data": [..u8..], "length": n}`, the Node-style lossless byte form; `autodetect` still maps non-text content types to the base64 `binary` form.
- `response_data_property` extracts one key from an object body (both the JSON object and the buffer object), otherwise the whole body is returned.
- Multipart requests send `Content-Type: multipart/form-data; boundary=----openjiuwen-<uuid>` and CRLF-framed `Content-Disposition: form-data; name="…"` parts, closing with `--<boundary>--\r\n`; file parts add `; filename="…"` and a `Content-Type` (object `content_type`, extension guess, or `application/octet-stream`).
- Digest replay sets `Authorization: Digest username=…, realm=…, nonce=…, uri=…, qop=auth, nc=00000001, cnonce=…, response=…, algorithm=…[, opaque=…]`; `qop` is omitted entirely for legacy (no-`qop`) challenges and `uri` is the path plus the sorted query string.
- SigV4 adds `X-Amz-Date`, `X-Amz-Content-Sha256`, `X-Amz-Security-Token` (when present), and `Authorization: AWS4-HMAC-SHA256 Credential=…/date/region/service/aws4_request, SignedHeaders=…, Signature=…`; the signed set is every header actually sent (always including `host`).
- `COMPONENT_HTTP_ERROR = 102002` — `throw_on_http_error` turned a failed status into `"[102002] the HTTP request failed, statusCode=…, ok=…, url=…, body=<512-char preview>"`. Python never raises on a status, so the numeric slot is borrowed from `COMPONENT_TOOL_INIT_FAILED`; the constant lives here, not in `crate::error`.
- `[102001]` (`component_tool_input_param_error`) — the prepared body exceeds `advanced_options.max_body_length`, naming both sizes; `0` means unlimited.
- `[102000]` (`component_tool_execution_error`) — transport, timeout, cancellation, size, and client-build failures.

## Environment variables

| Value | Source | Default |
| --- | --- | --- |
| access key | `authentication.username`, else `AWS_ACCESS_KEY_ID` | required |
| secret key | `authentication.password`, else `AWS_SECRET_ACCESS_KEY` | required |
| session token | `AWS_SESSION_TOKEN` | none |
| region | `AWS_REGION`, else `AWS_DEFAULT_REGION` | `us-east-1` |
| service | `AWS_SERVICE` | `execute-api` |

`execute-api` is the default because API Gateway is the service an HTTP component most often signs for; `AWS_SERVICE` overrides it for any other AWS API.
Missing access/secret keys are a typed `[102001]` input-parameter error naming the environment variables — never an unsigned request.
Digest credentials come only from `authentication.username`/`password`; a missing pair is the same typed error, and an unanswerable challenge (another scheme, an unsupported algorithm, or a replay that is still `401`) is returned as ordinary response data.

## Constraints

- Every send goes through `HttpClient` (`execute_cancellable`) so retries, cancellation, and the response cap keep their single implementation.
- No process-global state: the rate limiter is per component instance and shared only by that component's clones; locks are never held across `.await`.
- `openjiuwen_common::error::{Error, Result}` only; no `unsafe`, no stdout, `tracing` for diagnostics.
- `sha2` (workspace dependency) is the only added dependency; MD5 and HMAC-SHA256 are hand-rolled (no `md-5`/`hmac` crate).
- The public `HttpComponentConfig` field names and wire values are frozen: `HttpAuthType::Digest`/`Aws` and `HttpContentType::MultipartForm` must keep deserializing even when their backends change.

## Test Strategy

- Unit tests beside each module: `template` (references, placeholders, stringify), `body` (auto/form/text/json/binary selection, `max_body_length`), `multipart` (boundary presence/collision, part framing, binary round-trip, unicode, sanitization), `auth` (basic/bearer/api-key headers, `AWS_*` env resolution, missing credentials), `digest` (RFC 1321 MD5 vectors, RFC 2617 MD5/MD5-sess/SHA-256 responses, challenge parsing, `cnonce` shape), `sigv4` (AWS `get-vanilla` vector, fixed signed-GET/signed-POST vectors, signing key, canonical URI/query normalization), `response` (format resolution, buffer wire object, `response_data_property`, `[102002]`), `rate_limit` (pure `wait_for` schedule, disabled config, `acquire` cancellation, one clearly marked 1-second timing case).
- `tests/http_component.rs` covers the component end to end against a loopback `wiremock` server: request encoding, response projection, `throw_on_http_error` in both directions, retries, `[102000]` transport failures, multipart round-trip, digest challenge/replay (and a `401` without a challenge), SigV4 signing, `max_body_length` rejection, per-component rate limiting with cancellation, a generous limiter, and `binary` versus `buffer` bodies.

## Routing Table

- `./mod.rs` → component, input merge, request build, send, digest replay
- `./config.rs` → the public `Http*` configuration tree
- `./template.rs` → `{{key}}` / `${path}` templating and value stringification
- `./body.rs` → body kinds, encoding, and the `max_body_length` check
- `./multipart.rs` → `multipart/form-data` encoder and boundary generation
- `./auth.rs` → basic/bearer/api-key headers and pre-flight SigV4 signing
- `./digest.rs` → MD5 and the digest challenge/replay computation
- `./sigv4.rs` → AWS SigV4 canonical request, string to sign, signature
- `./response.rs` → response projection, formats, and the `[102002]` error
- `./rate_limit.rs` → per-component sliding-window limiter
