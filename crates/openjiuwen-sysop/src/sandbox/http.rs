//! HTTP sandbox gateway client (feature `sandbox`).
//!
//! The reference implementation does **not** speak HTTP inside
//! `openjiuwen/core/sys_operation/`: `SandboxGateway` is an in-process async router over
//! provider instances, and the real transports live in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/{aio,jiuwenbox,yuanrong}.py`.
//! The wire contract of those providers (the AIO container service) is an external
//! service protocol that this rewrite does not have a specification for.
//!
//! This module therefore implements the card/config surface and the request/response
//! envelope ([`crate::sandbox::config::GatewayInvokeRequest`] →
//! [`crate::sandbox::config::GatewayResponse`]) and reports
//! [`openjiuwen_common::error::Error::Unsupported`] for every remote operation, so a
//! sandbox card compiles, validates, and fails loudly instead of silently doing nothing.
//! Deferred work: launcher lifecycle (create/pause/resume/delete), endpoint discovery,
//! and the provider-specific parameter mapping.
