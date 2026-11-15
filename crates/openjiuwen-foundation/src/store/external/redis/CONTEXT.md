# redis — RESP2 key-value backend

## Intent

Backs the parent module's [`KvStore`](../CONTEXT.md) contract with a Redis server over a hand-written RESP2 connection, with no vendor SDK.
Speaks cleartext `redis://` and TLS `rediss://`; the TLS leg is a `tokio-rustls` client over the same TCP socket, so every command path is shared.
The in-memory backend in `../../kv/` remains the behavioral reference.

## API Surface

- `RedisConfig` (`url`, `addr`, `username`, `password`, `db`, `tls`, `timeout`, `key_prefix`) with `new`, `from_url`, `from_env`, the `with_*` builders (including `with_tls`), and `connect()`.
- `RedisKvStore` (`KvStore`), with `connect`/`from_connection`/`config`.
- `client` — `RedisConnection`, `RedisServerError`, `DEFAULT_TIMEOUT`, `READ_ONLY_COMMANDS`, `RedisAuth`.
  `connect`/`connect_with_timeout` dial cleartext; `connect_tls`/`connect_tls_with_timeout` dial TLS; `is_tls()` reports which.
- `codec` — `RespValue`, `encode_command`, `encode`, `Decoder`, `protocol_error`.

## URL and TLS Rules

- `redis://[user:password@]host[:port][/db]` and `rediss://…` parse identically except for `tls`; the default port is 6379 for both.
- An explicit `addr`, `username`, `password` or `db` overrides the matching URL component; an explicit `addr` never clears TLS, so a `rediss://` URL is never silently downgraded to cleartext.
- `with_tls(true)` is how an explicit `addr` reaches a TLS endpoint; `with_tls(false)` cannot disable a `rediss://` URL.
- Any other scheme is `Error::Config` listing `redis://` and `rediss://`; nothing is `Error::Unsupported` here any more.

## TLS Details

- Trust anchors are `webpki_roots::TLS_SERVER_ROOTS` only, loaded into a fresh `rustls::RootCertStore` per connection attempt.
- The crypto provider is `rustls::crypto::ring`, selected explicitly through `ClientConfig::builder_with_provider`, so no process-global provider installation is relied on.
- SNI is the host of `addr`: a literal IP (IPv6 brackets stripped) becomes `ServerName::IpAddress` and sends no SNI extension, a DNS name goes through `ServerName::try_from`, and an unusable host is `Error::Config` reported before any socket is opened.
- The configuration's `timeout` wraps the TCP connect and the TLS handshake together; a stalled handshake is `Error::Timeout`, a refused/aborted one is `Error::External`.
- Not supported, by design: custom CA bundles, certificate pinning, `insecure-skip-verify`, and a real offline TLS round trip (that needs either a trusted chain or a PEM/certificate-generation crate, neither of which the workspace ships).

## Constraints

- No vendor SDK and no extra crate: RESP is hand-written over `tokio::net::TcpStream`, and TLS uses the already-declared `tokio-rustls`/`webpki-roots` dependencies (`rustls` types are reached through `tokio_rustls::rustls`).
- The TLS code is unconditional: both dependencies are non-optional, so `--no-default-features` builds exercise the same surface.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Config` for an unusable URL/host or failed `AUTH`/`SELECT`, `Timeout` for a stalled connect/handshake/read/write, `External` for socket and server failures, protocol errors for malformed frames.
- Exactly one transparent reconnect-and-retry per command: failures before or while writing retry for every command, failures while reading only for `READ_ONLY_COMMANDS`; every reconnect replays `AUTH`/`SELECT` (and the TLS handshake).
- No cluster or Sentinel support, one serialized connection per store, no `unsafe`, no blocking I/O on async tasks.

## Design Decisions

- The transport is an internal `RedisStream` enum (`Plain(TcpStream)` / `Tls(Pin<Box<TlsStream<TcpStream>>>)`) implementing `AsyncRead`/`AsyncWrite`, so `RedisConnection`'s public surface and every command path are transport-agnostic and source-compatible.
- TLS pieces (server name and client configuration) are built before the TCP connect so an unusable host fails as `Error::Config` instead of a DNS error.
- `tls` is a plain `bool` on `RedisConfig` combined with the URL scheme by OR, because a "disable TLS" switch would permit the silent downgrade the backend forbids.

## Known Issues and Gaps

- No custom CA, no certificate pinning, no `insecure-skip-verify`, no client certificates; the platform trust store is not read either (`rustls-native-certs` is not a dependency).
- No `rediss://` cluster/Sentinel topologies, no pipelining, and the reconnect retry is a single best-effort retry rather than a backoff loop.
- `RedisConfig::tls` cannot express "cleartext even though the URL is `rediss://`", by design.

## Test Strategy

- `tests.rs` runs a scripted loopback `TcpListener` that decodes commands with this module's own `Decoder` and answers hand-built RESP frames; nothing leaves the loopback interface.
- `RawProbe`, a protocol-blind loopback server, proves the TLS path without completing a handshake: the first bytes the server reads must be a TLS handshake record (`0x16 0x03`) and must contain no cleartext `AUTH`/`PING`/`SELECT`; a probe that closes mid-handshake and one that stays silent both yield typed errors inside a 5 s bound; the same probe still receives plain RESP from a `redis://` connection.
- URL resolution (`resolved()`) is asserted directly: `rediss://user:pass@host:6390/2`, the default port, cleartext `redis://`, explicit-field override, `with_tls` on an explicit `addr`, and the rejected schemes.
