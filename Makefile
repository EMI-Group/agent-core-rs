.PHONY: build test test-all check fmt fmt-check clippy doc examples verify clean

# Build every workspace crate and binary.
build:
	cargo build --workspace

# Default test gate: the whole workspace, offline (no network or credentials).
test:
	cargo test --workspace --offline --no-fail-fast

# Feature-gated suites: integrations and optional backends.
test-all:
	cargo test -p openjiuwen-e2e --features a2a
	cargo test -p openjiuwen-extensions --features a2a,otel
	cargo test -p openjiuwen-harness --no-default-features
	cargo test -p openjiuwen-agent-teams --no-default-features
	cargo test -p openjiuwen-symphony --features sqlite
	cargo test -p openjiuwen-sysop --features sandbox
	cargo test -p openjiuwen-harness-providers --features vendor

check:
	cargo check --workspace --all-targets

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Compile all runnable examples (offline set + A2A/MCP-gated set).
examples:
	cargo build -p openjiuwen-examples --examples
	cargo build -p openjiuwen-examples --examples --features a2a,mcp

# Full local verification, mirroring the documented gates.
verify: fmt-check clippy test doc test-all
	cargo build -p openjiuwen-examples --examples
	cargo build -p openjiuwen-examples --examples --features a2a,mcp

clean:
	cargo clean
