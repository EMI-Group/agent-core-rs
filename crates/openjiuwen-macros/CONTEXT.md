# openjiuwen-macros

## Intent

Proc-macro crate exporting `#[tool]`: the Rust equivalent of the Python `@tool` decorator (`openjiuwen/core/foundation/tool/tool.py`).
One attribute turns an `async fn` plus a typed argument struct into a model-facing `ToolCard`, a `Tool` implementation, and an `Arc<dyn Tool>` constructor, so a tool author writes a plain async function and never hand-builds a card or a JSON Schema.
Compile-time only: `syn` parses, `quote` emits; the emitted code targets `openjiuwen-foundation` and `openjiuwen-common` at runtime, so the crate itself has no runtime dependency.
It sits in layer 0 and depends on nothing in the workspace; `openjiuwen-foundation` depends on it, and every crate above consumes tools through `openjiuwen_foundation::tool::Tool`.

## API Surface

- `#[openjiuwen_macros::tool]` — attribute macro on a free `async fn`; every argument is optional (see **Macro Contract**).
- Nothing else is exported. `#[derive(ToolArgs)]` is deliberately *not* provided: `#[derive(serde::Deserialize, schemars::JsonSchema)]` on the argument struct already supplies deserialization and the schema, and a second derive would only restate `schemars` with weaker expressiveness.
- The macro exports no items of its own beyond the wrapper struct/constructor; the generated names are `PascalCase(fn) + "Tool"` and `{fn}_tool()`.

## Usage

```rust
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolContext};
use openjiuwen_macros::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs { pub query: String, pub limit: Option<u32> }

/// Search the web.
#[tool(name = "search")]
pub async fn search(args: SearchArgs, ctx: &ToolContext) -> Result<serde_json::Value> { .. }

let tool: std::sync::Arc<dyn Tool> = search_tool();
```

Agent, workflow, and harness crates register the `Arc<dyn Tool>` from `{fn}_tool()` in `ResourceMgr` or an agent's ability list; they never hand-build a `FunctionTool` for a local Rust function.
`search(args, &ctx)` stays directly callable, so the tool logic is unit-testable without a tool card.

## Macro Contract

### Accepted forms

```rust
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolContext;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddArgs { pub a: i64, pub b: i64 }

/// Add two integers.
#[tool(name = "add")]
pub async fn add(args: AddArgs) -> Result<i64> { Ok(args.a + args.b) }          // unary

#[tool(name = "search", description = "Search the web")]
pub async fn search(args: SearchArgs, ctx: &ToolContext) -> Result<Value> { .. } // borrowed context

#[tool]
pub async fn ping(args: PingArgs, ctx: ToolContext) -> PingOut { .. }            // owned context, infallible return
```

- The function must be a free `async fn` (not a method, not `const`/`unsafe`/`extern`), non-generic, taking one or two parameters by value.
- Parameter 1 is the argument struct (`Args`): `serde::de::DeserializeOwned`, plus `schemars::JsonSchema` unless `input_params` is supplied.
- Parameter 2 is optional and must be `&ToolContext` (`&ctx` is passed) or `ToolContext` (`ctx` is moved in).
- The return type is `openjiuwen_common::error::Result<T>` (the error is propagated unchanged) or a plain `T` (wrapped in `Ok`); `T: serde::Serialize` (unit/no return type serializes to `null`).
- Everything else (`self` receiver, generics, three parameters, `&Args`, a `ToolContext` first parameter, non-async) is a `compile_error!` that names the supported shapes.

### Attributes

| attribute | literal kind | default | sets |
|---|---|---|---|
| `name = "..."` | `&str` | the function ident, verbatim | `ToolCard.name` |
| `description = "..."` | `&str` | the doc-comment description (below), else the humanized ident | `ToolCard.description` |
| `input_params = <expr>` | any expression of type `serde_json::Value` (e.g. `json!({..})`, `custom_schema()`) | `schema_for::<Args>()` | `ToolCard.input_params` (stored raw, no normalization) |
| `stateless = <bool>` | `true`/`false` | `false` | `ToolCard.stateless` |
| `parallel_safe = <bool>` | `true`/`false` | `true` | `ToolCard.parallel_safe` |
| `idempotent = <bool>` | `true`/`false` | `false` | `ToolCard.idempotent` |
| `exposure = "direct" \| "deferred"` | `&str` | `"direct"` | `ToolCard.exposure` |
| `execution = "server" \| "client"` | `&str` | `"server"` | `ToolCard.execution` |
| `crate = "path"` (aliases `crate_path`, `krate`) | string literal or bare path | `::openjiuwen_foundation` | root path of the foundation surface the generated code calls |
| `common_crate = "path"` | string literal or bare path | `::openjiuwen_common` | root path of the common surface (`Error`, `Result`) |

Aliases collapse to one key for duplicate detection (`crate` + `crate_path` is a duplicate).

Unknown keys, duplicate keys, and wrong literal kinds are `compile_error!`s naming the offending key and the accepted set.
`name`/`description`/`input_params` mirror the Python decorator; the policy keys cover what Python sets through a pre-built `card=`; `crate`/`common_crate` are the escape hatch for crates that re-export the foundation surface under a different path (the generated code also needs `::serde_json` and `::async_trait` resolvable in the user crate).

### Description from doc comments

With no `description` attribute, the description is the first paragraph of the doc comment, mirroring Python's `CallableSchemaExtractor.extract_function_description`:
take the `#[doc = " ..."]` lines, strip one leading space per line, then stop at the first blank line, at a rustdoc heading (`# ...`, `## ...`), or at a line matching `^\s*\w+\s*:` (which covers `Args:`, `Returns:`, `Examples:`, `Errors:`, ...); join the surviving lines with a single space and trim.
When that is empty (no doc comment, or a doc comment that starts with a heading) the description is the humanized function ident: `read_write_tool` → `read write tool` (words lowercased, a documented abbreviation set — `id`, `url`, `api`, `json`, `http`, `llm`, `mcp`, `sql`, `uuid`, `html`, `xml`, `csv`, `pdf`, `cli`, `ui`, `os`, `ip`, `tcp`, `udp`, `tls`, `ast`, `ocr`, `rag` — kept upper-case).

### Generated items

`#[tool(name = "add")] pub async fn add(args: AddArgs, ctx: &ToolContext) -> Result<Value>` expands to the original function **unchanged** (same name, visibility, doc comment, and attributes, so Rust callers and unit tests can call the logic directly) plus:

```rust
#[derive(Debug)]
pub struct AddTool { inner: ::openjiuwen_foundation::tool::FunctionTool }

impl AddTool {
    pub const NAME: &'static str = "add";
    pub const DESCRIPTION: &'static str = "Add two integers.";
    #[must_use]
    pub fn new() -> Self {
        let card = ::openjiuwen_foundation::tool::ToolCard::new(
            Self::NAME, Self::DESCRIPTION, ::openjiuwen_foundation::tool::schema_for::<AddArgs>(),
        );                                              // + .with_* calls for the policy keys
        let handler: ::openjiuwen_foundation::tool::ToolHandler = ::std::sync::Arc::new(
            |inputs: ::serde_json::Value, ctx: ::openjiuwen_foundation::tool::ToolContext| {
                ::std::boxed::Box::pin(async move {
                    let args: AddArgs = ::serde_json::from_value(inputs)
                        .map_err(|error| ::openjiuwen_common::error::Error::validation(format!("invalid tool input: {error}")))?;
                    let output = add(args, &ctx).await?;      // `?` only for a Result return type
                    ::serde_json::to_value(output)
                        .map_err(|error| ::openjiuwen_common::error::Error::tool(format!("failed to serialize tool output: {error}")))
                })
            },
        );
        Self { inner: ::openjiuwen_foundation::tool::FunctionTool::new(card, handler) }
    }
}

impl Default for AddTool { fn default() -> Self { Self::new() } }

#[::async_trait::async_trait]
impl ::openjiuwen_foundation::tool::Tool for AddTool {
    fn card(&self) -> &::openjiuwen_foundation::tool::ToolCard { ::openjiuwen_foundation::tool::Tool::card(&self.inner) }
    async fn invoke(&self, inputs: ::serde_json::Value, ctx: &::openjiuwen_foundation::tool::ToolContext)
        -> ::openjiuwen_common::error::Result<::serde_json::Value> {
        ::openjiuwen_foundation::tool::Tool::invoke(&self.inner, inputs, ctx).await
    }
}

#[must_use]
pub fn add_tool() -> ::std::sync::Arc<dyn ::openjiuwen_foundation::tool::Tool> { ::std::sync::Arc::new(AddTool::new()) }
```

- The struct, `NAME`, `DESCRIPTION`, `new`, and `{fn}_tool` carry the visibility of the annotated function (`pub`, `pub(crate)`, or inherited/private) so `unreachable_pub`/`missing_docs` stay quiet; the inherent, `Default`, and `Tool` impls are bare `impl` blocks because Rust forbids a visibility qualifier on an impl (E0449).
- `#[cfg(...)]`/`#[cfg_attr(...)]` attributes written *after* `#[tool]` are copied to every generated item, so a feature-gated tool does not leave orphan items behind (attributes before `#[tool]` are consumed by `cfg` expansion and never reach the macro).
- `stream`, `render_for_llm`, and `is_parallel_safe` are inherited from `Tool`'s defaults through the delegation, so the generated tool behaves exactly like a hand-built `FunctionTool`.
- The constructor is infallible: `FunctionTool::new` takes the card and handler directly, so no `Result`, no `unwrap`, and no `must_use`-ignoring fallback path exists. The tool validates inputs, fills schema defaults, and honours cancellation through `FunctionTool::invoke`, so the macro never re-implements foundation's validation.

## Constraints

- `proc-macro = true`; runtime dependencies are exactly `proc-macro2`, `quote`, `syn` (workspace versions, `syn` with `full`). No runtime dependency may be added.
- Dev-dependencies exist only for this crate's tests: `openjiuwen-common` + `openjiuwen-foundation` (the generated code's targets; the cycle runs through dev-dependencies only), `async-trait`, `schemars`, `serde`, `serde_json`, `tokio`, `futures` (`StreamExt` for the `Tool::stream` test), and the pinned `trybuild` (`=1.0.119`, matching the workspace MSRV because `.stderr` output is compiler-version sensitive).
- No `unsafe`; workspace lints are inherited and `cargo clippy -p openjiuwen-macros --all-targets -- -D warnings` must pass.
- Generated code must be lint-clean in *downstream* crates: doc comments on every generated public item, no `unwrap`/`expect`/`panic!`, no unused imports, no `println!`.
- Every error path goes through `syn::Error` → `compile_error!`; the macro must never panic (no `unwrap`/`expect` in the library either), and it must report at the most specific span available (`name = ...` literal, the offending parameter, the return type).
- Files stay around ≤1000 lines (largest module is ~280 lines); parsing, signature validation, description extraction, and code generation live in separate modules (`src/tool/`).
- `cargo fmt -p openjiuwen-macros`, `cargo test -p openjiuwen-macros`, and the clippy gate above must all be green offline once `trybuild` is cached.
- User crates applying `#[tool]` need `openjiuwen-foundation`, `openjiuwen-common`, `serde_json` (generated code), `async-trait` (generated `impl Tool`), `serde` + `schemars` (argument struct, unless `input_params` supplies the schema).

## Design Decisions

- **Keep the annotated function.** Python replaces the function with a `LocalFunction` object; Rust keeps the typed `async fn` and *adds* the tool around it, so the logic stays directly callable and testable and no name is shadowed.
- **Infallible constructor, no builder chain.** `docs/ARCHITECTURE.md` §2.3 and `openjiuwen-foundation`'s `tool/function.rs` doc comment describe the macro as expanding to `FunctionTool::builder()...build()`, which returns `Result`. The macro instead emits `FunctionTool::new(card, handler)`: `build()` can only fail on a missing handler, which generated code always sets, and `unwrap`/`expect` are forbidden, so the builder path would force every call site to handle an impossible error. Foundation's doc comment should be updated to match (flagged, not edited: cross-crate change).
- **`AddTool` wraps a `FunctionTool` and delegates `Tool`.** Deserializing arguments and serializing results could be emitted inline, but then schema-default filling and `validate_inputs` would be re-implemented; delegation keeps exactly one copy of the invocation semantics (validation → defaults → cancellation → handler) in foundation. The only duplicated text is the two error messages, quoted from `FunctionToolBuilder::typed_handler`.
- **`input_params` is an expression, not a JSON string.** A literal schema would need a JSON parser at macro time and could not reference a schema built at runtime; an `Expr` accepts `json!({..})`, a `const`, or a function call, and skips the `JsonSchema` requirement on `Args`.
- **Description rules follow Python's extractor, not rustdoc.** First paragraph only, collapsed to one line, stopping at section markers, with the humanized ident as the final fallback so the model never receives an empty description (Python's `_humanize_name` behavior).
- **`ToolContext` is explicit, never injected implicitly.** Python drops `session`/`kwargs` instead of forwarding them; Rust's second parameter is opt-in and typed, which gives the same capability without hidden parameter-name magic.
- **No `derive(ToolArgs)`** — see API Surface.

## Python Mapping

| Python | Rust |
|---|---|
| `@tool` / `@tool(...)` (`tool.py`) | `#[openjiuwen_macros::tool]` / `#[tool(...)]` |
| `name` argument / `func.__name__` | `name = "..."` / the function ident |
| `description` / `extract_function_description` / `_humanize_name` | `description = "..."` / first doc paragraph / humanized ident |
| `input_params` argument / `CallableSchemaExtractor.generate_schema` | `input_params = <expr>` / `schema_for::<Args>()` (`schemars`) |
| `card=` with `exposure`/`execution`/`parallel_safe`/`stateless`/`idempotent` | the same keys on `#[tool(...)]` |
| returned `LocalFunction` (`.invoke`, `.stream`, `.card`) | `AddTool` + `impl Tool`, built by `add_tool()` |
| sync `def` / generators | not ported: `async fn` only, single-shot `invoke` (`stream` is the trait's one-item default) |
| `additionalProperties: false` on generated schemas | inherited from `schemars`/`schema_for` (permissive, foundation's documented deviation) |

## Known Issues and Gaps

- No `Args:`-block parameter descriptions: Python copies `name: description` doc lines into `properties.<name>.description`; Rust takes parameter docs from `schemars`(`#[schemars(description = "...")]`), and a doc-block scraper would duplicate `schemars` badly.
- The macro accepts a single argument struct instead of Python's flat keyword parameters, so `Args` must be a named struct (or `Value` with `input_params`) — Python's `*args` positional expansion has no Rust equivalent.
- Sync and generator handlers are not supported (foundation documents async-only handlers); streaming tools must implement `Tool::stream` by hand.
- An alias of `Result` (e.g. `type MyResult<T> = Result<T>`) is not detected as fallible: only a return type whose last path segment is `Result` propagates errors; anything else is wrapped in `Ok`, which surfaces as a `Serialize` bound error on the alias.
- Generated names are mechanical, so a function already named `read_write_tool` yields the struct `ReadWriteToolTool` and the constructor `read_write_tool_tool()`; the constructor never collides with the function itself because the suffix is always appended.

## Test Strategy

- `tests/tool_macro.rs` — 28 runtime tests on generated tools (`#[tokio::test]`): unary and context forms, owned vs borrowed `ToolContext`, `card()` shape (`name`, `description`, `input_params` properties/required), `NAME`/`DESCRIPTION`, policy keys, `input_params` override, plain-`T` and unit returns, error propagation from the handler, `Error::Validation` on invalid input, cancellation short-circuit, `Arc<dyn Tool>` coercion, `Default`, `stream` yielding one item, `render_for_llm` of a `ToolOutput`, and that the annotated function stays callable.
- `tests/compile.rs` — `trybuild` driver: `pass("tests/ui/pass/*.rs")` (12 fixtures) and `compile_fail("tests/ui/fail/*.rs")` (12 fixtures).
- `tests/ui/pass/` — every supported form: unary, borrowed context, owned context, infallible return, unit return, all attribute keys, doc-comment description, `input_params` override, private and `pub(crate)` visibility, raw identifier, and the `crate`/`common_crate` escape hatch.
- `tests/ui/fail/` — non-async fn, method (impl-block placement), zero/three parameters, `&Args` first parameter, `ToolContext` first parameter, generic fn, unknown key, duplicate key, non-string `name`, bad `exposure`, bad `execution`; each pins a message the macro itself emits (never compiler trait-resolution output, which changes with dependency and toolchain versions) and has a committed `.stderr` (regenerate with `TRYBUILD=overwrite cargo test -p openjiuwen-macros --test compile`, inspect, then re-run without the variable).
- The crate-level doctest in `src/lib.rs` is a real, running example (defines a tool, asserts the card shape and one `invoke` round-trip), so the documented usage cannot rot.
- Gates: `cargo fmt -p openjiuwen-macros --check`, `cargo test -p openjiuwen-macros` (8 unit + 28 runtime + 1 trybuild + 1 doctest), `cargo clippy -p openjiuwen-macros --all-targets -- -D warnings`, all offline.
- Test crates start with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; `tests/tool_macro.rs` also allows `clippy::unused_async` (the macro requires `async fn`, and thin test handlers have nothing to await). The library itself keeps every lint enabled.

## Routing Table

- `./src/lib.rs` → crate docs, the `#[proc_macro_attribute]` entry points, module wiring
- `./src/error.rs` → `Fail`/`fail!` helpers that turn every problem into a spanned `compile_error!`
- `./src/naming.rs` → `PascalCase` struct/constructor names, humanized-name description fallback
- `./src/describe.rs` → doc-comment → model-facing description (first-paragraph rules)
- `./src/tool/` → attribute parsing, signature validation, code generation (`src/tool/CONTEXT.md`)
- `./tests/tool_macro.rs` → runtime behavior of generated tools
- `./tests/compile.rs` → `trybuild` driver
- `./tests/ui/` → compile-pass / compile-fail fixtures and their `.stderr` expectations
