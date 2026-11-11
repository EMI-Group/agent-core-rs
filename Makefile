.PHONY: build test check fmt clippy doc clean

build:
	cargo build --workspace

test:
	cargo test --workspace

check:
	cargo check --workspace --all-targets

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

doc:
	cargo doc --workspace --no-deps

clean:
	cargo clean
