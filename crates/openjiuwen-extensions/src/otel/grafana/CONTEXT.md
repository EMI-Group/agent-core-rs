# otel/grafana

## Intent

Typed preset for the **Grafana Cloud OTLP gateway** (`https://otlp-gateway-<zone>.grafana.net/otlp`, HTTP Basic `base64(instance_id:token)`), compiled in every feature mode and holding no SDK types.
Maps the Grafana side of `openjiuwen/extensions/tracer_otel/` (v0.1.20); Python has no dedicated Grafana module, so the preset owns the derivation and validation rules itself.
Everything here needs an external Grafana Cloud stack (instance id, `traces:write` token, zone) plus network access; nothing contacts Grafana at configuration time.
Only OTLP/HTTP **JSON** is available offline (`gRPC` needs `tonic`/`prost`), so `otel_config()` always selects `OtlpProtocol::Http`.

## API Surface

- `GrafanaCloudConfig` — `new(instance_id, token, zone)`, `with_service_name`, `with_zone`, `with_otlp_endpoint`, `zone()`, `service_name()`, `endpoint()`, `authorization()`, `validate()`, `otel_config()`, `from_env()`, and `exporter()` behind `feature = "otel"`.
- `GrafanaAgentConfig` — `new(endpoint)`, `with_service_name`, `endpoint()`, `service_name()`, `otel_config()`/`exporter()`, both `Error::Unsupported` naming the external collector requirement.
- Env constants: `GRAFANA_INSTANCE_ID`, `GRAFANA_API_TOKEN`, `GRAFANA_ZONE`, `GRAFANA_OTLP_ZONE`, `GRAFANA_OTLP_ENDPOINT`.
- `Debug` for `GrafanaCloudConfig` redacts the token; `GrafanaAgentConfig` holds no credentials and derives `Debug`.

## Constraints

- `endpoint()` is infallible: an unparsable zone yields an empty string and `validate()` reports the reason with `Error::validation` naming the field.
- `GRAFANA_ZONE` wins over `GRAFANA_OTLP_ZONE`; `GRAFANA_OTLP_ENDPOINT` wins over both and removes the zone requirement.
- A zone may be a slug (`prod-us-east-0`), a host label, or a `*.grafana.net` URL with an optional `/otlp` path; paths, queries, fragments, whitespace, foreign hosts, and empty `-` segments are rejected.
- No panics on user input; `tracing` only; no `unsafe`.

## Test Strategy

`tests.rs` (declared with `#[cfg(test)] mod tests;`) covers endpoint derivation for all accepted zone forms, the exact base64 Basic credentials, `otel_config()` field-by-field, every validation error, `from_env()` precedence, the collector deferral, and `exporter().traces_url()` under `feature = "otel"`.
Environment tests hold a `static Mutex` and clear `GRAFANA_*` on entry and on drop; no test contacts the network.
