# tool — `#[tool]` attribute implementation

## Intent

Turn the parsed attribute and annotated function into the expansion described in `../CONTEXT.md` ("Macro Contract").
Parsing, signature validation, and code generation are separate modules so each stays small and testable.
The module never panics: every problem is a spanned `syn::Error` returned to `expand`, which keeps the original item and appends a `compile_error!`.

## API Surface

- `expand(attr: TokenStream, item: TokenStream) -> TokenStream` (`mod.rs`) — the only entry point; parses the attribute, validates the function through `signature`, generates through `expand::expand`, and falls back to `original + compile_error!`.
- `attrs::parse(TokenStream) -> Result<ToolAttrs>` — the typed attribute arguments (`name`, `description`, `input_params`, `stateless`, `parallel_safe`, `idempotent`, `exposure`, `execution`, `crate`/`crate_path`/`krate`, `common_crate`), with unknown/duplicate/wrong-kind errors.
- `signature::analyze(&ItemFn) -> Result<ToolFn>` — the argument type, the optional `&ToolContext`/`ToolContext` parameter, and whether the return type is `Result`.
- `expand::expand(ItemFn, ToolAttrs) -> Result<TokenStream>` — emits the original function plus the wrapper struct, inherent impl, `Default` impl, `Tool` impl, and `{fn}_tool` constructor.

## Constraints

- Attribute values are parsed as `syn::Expr`, so `input_params` accepts `json!({..})` or any runtime expression and `crate` accepts a string literal or a bare path.
- Keys are parsed with `IdentExt::parse_any` because `crate` is a keyword.
- `crate`/`common_crate` default to `::openjiuwen_foundation` / `::openjiuwen_common` and replace those roots everywhere in the generated code.
- Generated identifiers come from the function ident (`PascalCase(fn) + "Tool"`, `{fn}_tool`), never from `name`; raw idents lose their `r#` marker.
- `#[cfg]`/`#[cfg_attr]` attributes on the function are copied to every generated item.
- Every generated public item carries a `#[doc = "..."]`; the original function is re-emitted with its own attributes untouched.
- No `unwrap`/`expect`/`panic!`/`unreachable!`; no panicking `format_ident!`/`parse_quote!` on user input (only on the fixed default paths).

## Routing Table

- `./mod.rs` → `expand` entry point and error fallback
- `./attrs.rs` → attribute parsing and the `ToolAttrs` model
- `./signature.rs` → function-shape validation and `ToolFn`
- `./expand.rs` → generated tokens
