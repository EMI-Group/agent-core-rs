# otel/grafana

## Intent

Typed presets for the two Grafana OTLP paths: the **Grafana Cloud gateway** (`https://otlp-gateway-<zone>.grafana.net/otlp`, HTTP Basic `base64(instance_id:token)`) and the **self-hosted Grafana Agent / Grafana Alloy push** (a local OTLP/HTTP collector, `http://localhost:4318` by default).
Both presets compile in every feature mode and hold no SDK types.
Maps the Grafana side of `openjiuwen/extensions/tracer_otel/` (v0.1.20); Python has no dedicated Grafana module, so the presets own the derivation and validation rules themselves.
Each preset only builds configuration: nothing contacts the gateway or the collector until a span batch is exported.
Neither preset installs, starts, discovers, or supervises a backend: a Grafana Cloud stack and a self-hosted Agent/Alloy process have to exist and listen on their own.

## API Surface

- `GrafanaCloudConfig` — `new(instance_id, token, zone)`, `with_service_name`, `with_zone`, `with_otlp_endpoint`, `zone()`, `service_name()`, `endpoint()`, `authorization()`, `validate()`, `otel_config()`, `from_env()`, and `exporter()` behind `feature = "otel"`.
- `GrafanaAgentConfig` — `new(endpoint)`, `with_service_name`, `with_header(name, value)`, `with_headers(map)`, `endpoint()`, `service_name()`, `headers()`, `otel_config()`, and `exporter()` behind `feature = "otel"`.
- `DEFAULT_LOCAL_OTLP_ENDPOINT` — public const `http://localhost:4318`, re-using `otel::env::DEFAULT_OTLP_HTTP_ENDPOINT`; the `/v1/traces` suffix is appended by the exporter.
- `GrafanaAgentConfig::otel_config()` selects `ExporterType::Otlp` with `OtlpProtocol::Http`, resolves a blank endpoint to `DEFAULT_LOCAL_OTLP_ENDPOINT`, applies the preset headers, and reports `service.name` (preset value, else `openjiuwen`) with no `service.version`.
- `GrafanaAgentConfig::exporter()` builds the OTLP/HTTP exporter for the resolved endpoint, headers, and `export_timeout_ms`; construction starts the worker thread and performs no I/O.
- Env constants: `GRAFANA_INSTANCE_ID`, `GRAFANA_API_TOKEN`, `GRAFANA_ZONE`, `GRAFANA_OTLP_ZONE`, `GRAFANA_OTLP_ENDPOINT`.
- `Debug` for `GrafanaCloudConfig` redacts the token; `GrafanaAgentConfig` holds no credentials and derives `Debug`.

## Constraints

- `endpoint()` is infallible: an unparsable zone yields an empty string and `validate()` reports the reason with `Error::validation` naming the field.
- `GRAFANA_ZONE` wins over `GRAFANA_OTLP_ZONE`; `GRAFANA_OTLP_ENDPOINT` wins over both and removes the zone requirement.
- A zone may be a slug (`prod-us-east-0`), a host label, or a `*.grafana.net` URL with an optional `/otlp` path; paths, queries, fragments, whitespace, foreign hosts, and empty `-` segments are rejected.
- Both presets validate endpoints through the same absolute `http`/`https` URL check and drop trailing slashes; anything else is `Error::Validation` naming the requirement.
- `GrafanaAgentConfig` header pairs are validated in `otel_config()` with `HeaderName`/`HeaderValue`, so an invalid name, an empty name, or a control character is `Error::Validation` naming the header; the builders themselves stay infallible.
- `with_header` overwrites an existing name and `with_headers` overwrites per key, so the final map is insertion-independent.
- Serialization is `snake_case` with `#[serde(default)]`, so a JSON document written before the `headers` field existed still deserializes.
- No panics on user input; `tracing` only; no `unsafe`.

## Test Strategy

`tests.rs` (declared with `#[cfg(test)] mod tests;`) covers endpoint derivation for all accepted zone forms, the exact base64 Basic credentials, both `otel_config()` shapes field-by-field, endpoint/service-name/header validation, `from_env()` precedence, the Agent/Alloy default endpoint, the serde round-trip (including a document without `headers`), and `exporter().traces_url()` for both presets under `feature = "otel"`.
Environment tests hold a `static Mutex` and clear `GRAFANA_*` on entry and on drop; no test contacts the network.
