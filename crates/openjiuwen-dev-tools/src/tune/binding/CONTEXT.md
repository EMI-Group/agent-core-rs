# openjiuwen-dev-tools / tune / binding

## Intent

Port Python's `LLMCall`/`TextualParameter` binding layer (`tune/optimizer/base.py`, `tune/utils.py`) as typed values, supplying the glue the Rust optimizers lack.
Python's optimizers mutate an agent's prompt parameters in place through a mutable `LLMCall`; the Rust optimizers take prompt strings and return an `OptimizationResult`, so this module turns that result back into a parameter.
The parameter owns its template text rather than a live call, so a binding is a pure function: validate a candidate against the template's placeholders, return the change, mutate nothing.
The tool half is the join between what a `Case` declares (`ToolInfo` descriptors) and what is runnable (`ToolStubs`), reusing the crate's one tool model instead of defining another descriptor.
Everything here is offline and deterministic: no model, network, clock, environment read, or filesystem.

## API Surface

- `TextualParameter` — `name`, `template`, `description`, `gradients: BTreeMap<String, String>`, `candidates: BTreeMap<String, Vec<String>>`; `Debug + Clone + Default + PartialEq + Serialize + Deserialize` with `snake_case`, every field defaulted, unknown fields ignored.
  `new(name, template)` (infallible, like Python's), `with_description`, `with_gradient(name, gradient)` / `gradient(name) -> Option<&str>`, `with_candidates(name, Vec<String>)` / `candidates(name) -> &[String]` (empty when absent), `placeholder_names() -> Vec<String>` (sorted, delegates to `crate::tune::optimizer::placeholders`), `validate()`, `render(&BTreeMap<String, Value>) -> Result<String>`.
  `validate` rejects a blank `name` (`TextualParameter.name must not be empty`) and a blank `template` (naming the parameter); `render` checks only the template.
  `render` substitutes every `{{name}}` found in `values`, renders JSON strings bare and every other value as compact JSON, leaves a placeholder with no value verbatim, leaves an empty placeholder name untouched, preserves a malformed `{{` run by resuming just past the opener, and preserves an unclosed opener to the end of the text.
- `ParameterBinding` — one `TextualParameter` plus the rules for accepting a new prompt; `Debug + Clone + PartialEq` (a live handle, not a serialized payload).
  `new(parameter)`, `from_template(name, template)`, `parameter() -> &TextualParameter`, `with_description`, `validate(&str)`, `apply(&str) -> Result<ParameterUpdate>`, `apply_result(&OptimizationResult) -> Result<ParameterUpdate>`, `prompt_values() -> BTreeMap<String, Value>`.
  `validate` checks, in order: the parameter itself, a blank candidate, the declared placeholders the candidate dropped (sorted, rendered `{{name}}`), and the placeholders the candidate introduced (sorted, rendered `{{name}}`).
  `apply` validates and returns the update; `apply_result` applies `result.best_prompt` and sets `gain`.
  `prompt_values` is a skeleton keyed by the declared placeholders whose values are the literal placeholder text, so `parameter().render(&binding.prompt_values())` reproduces the template.
- `ParameterUpdate` — `parameter_name`, `previous_prompt`, `new_prompt`, `preserved_placeholders: Vec<String>`, `gain: Option<f64>`; `Debug + Clone + Default + PartialEq + Serialize + Deserialize` (`snake_case`, defaulted, unknown-field tolerant) with `changed()`.
  `changed()` is false exactly when the candidate equals the prompt the parameter already held; `preserved_placeholders` is every declared placeholder (sorted), because validation guarantees each one survives.
- `ToolBinding` — `ToolStubs` plus the join to a case's declarations; `Debug + Clone + Default`.
  `new(stubs)`, `tool_stubs() -> &ToolStubs`, `declare_only(&Case) -> Vec<String>` (the case's `tool_names()`, declaration order, deduplicated), `bind_case(&Case) -> Result<ToolBindingReport>`, `validate_case(&Case) -> Result<()>`, `usage(&ToolTrace) -> BTreeMap<String, usize>` (delegates to `ToolTrace::usage`).
- `ToolBindingReport` — `declared` (declaration order), `bound` (declaration order), `missing` (sorted); `Debug + Clone + Default + PartialEq + Eq + Serialize + Deserialize` (`snake_case`, defaulted) with `is_complete()` and `missing_count()`.
  A tool-free case reports three empty vectors and is not an error; `bind_case` is permissive about unbound names, `validate_case` is strict.
- `BoundAgent` — a `Vec<ParameterBinding>` sorted by name; `Debug + Clone + Default + PartialEq`.
  `new(Vec<ParameterBinding>) -> Result<Self>`, `parameters() -> &[ParameterBinding]`, `parameter(name) -> Option<&ParameterBinding>`, `validate_all(&BTreeMap<String, String>) -> Result<Vec<ParameterUpdate>>`, and `apply(&BTreeMap<String, String>) -> Result<Vec<ParameterUpdate>>` (the same operation under the name a tuning driver reaches for).
  `validate_all` requires a candidate for every bound parameter, returns updates in parameter-name order, ignores candidate entries for parameters the agent does not declare, and mutates nothing.
- Crate root `tune` re-exports all five types (`binding` itself stays private), so `openjiuwen_dev_tools::tune::{TextualParameter, ParameterBinding, ParameterUpdate, ToolBinding, ToolBindingReport, BoundAgent}` is the public path.

## Constraints

- Errors come only from `openjiuwen_common::error`: blank name/template, blank candidate, dropped/introduced/renamed placeholder, blank declared tool name, missing candidate prompt, and an unbound tool via `validate_case` are all `Error::validation`; no path in this module returns `Error::Unsupported`, `Error::NotFound`, or panics.
- `ToolBinding::validate_case` deliberately reports `Error::validation` (naming the first unbound tool in declaration order plus the unbound count) rather than `Error::not_found`: an unbound declaration is a configuration mismatch between the case and the binding, while `ToolStubs::call` on an unknown name still reports `Error::not_found`.
- Names are matched exactly: only an all-whitespace name is blank, so `system` and `" system"` are two distinct parameters and `BoundAgent::parameter` does no trimming.
- Serialization is limited to the payload types (`ParameterUpdate`, `ToolBindingReport`) and the parameter (`TextualParameter`); `ParameterBinding`, `ToolBinding` and `BoundAgent` own live handles (a stub handler is an opaque `Arc` closure) and are not serialized.
- No new dependencies and no `unsafe`, printing, env reads, clock reads, sleeps, or network; the only I/O is none at all.
- Files stay well under 1000 lines: `mod.rs` holds docs, `mod` declarations and re-exports only.

## Design Decisions

- The parameter *is* the template: Python's `TextualParameter` wraps a mutable `LLMCall`, whereas here the text is the value an optimizer rewrites, so a binding can be applied and validated repeatedly without shared mutable state and `apply` never touches the template it read.
- Rendering and binding are separate operations: `render` fills placeholders from a value map (leaving unknown ones verbatim so a partial map renders a partial prompt), while `validate`/`apply` decide whether a whole new template is acceptable; neither implies the other.
- Placeholder validation is set equality in both directions — a dropped or renamed placeholder is reported as missing, a new one as introduced — because a downstream parameter map has no value for a name the template never declared.
- `preserved_placeholders` is the declared set rather than an intersection recomputed after the fact: validation already proves every declared name survives, so recomputing could only disagree with the check that just passed.
- `gain` is the optimizer's absolute score, not a delta, and only when it is finite and strictly positive: the binding holds no baseline, so a zero, negative, or non-finite score reports `None` instead of a fabricated improvement.
- `prompt_values` returns the placeholders themselves as values, which makes the skeleton a render round-trip fixture: filling in a few entries demonstrates exactly what `render` substitutes.
- `ToolBinding` is a join, never a factory: a `Case` declares `ToolInfo` descriptors and cannot carry handlers, so the binding only partitions declared names and the caller supplies the `ToolStubs`, keeping one tool model for the whole crate.
- Permissive and strict entry points coexist on purpose: `bind_case` lets a caller run with a partial binding and inspect what is missing, while `validate_case` names the first unbound declaration (plus the unbound count) for a caller that wants to fail before running anything.
- `missing` is sorted while `declared`/`bound` keep declaration order: the report compares equal regardless of authoring order, and `bound` still reads in the order the author wrote the tools.
- `BoundAgent` sorts by name and rejects duplicates so an update list is deterministic and a caller can index it by parameter; it ignores candidates for undeclared parameters so one candidate map can serve a supervisor and its sub-agents.
- `BoundAgent::apply` is an alias of `validate_all` (not a second implementation) because "apply" here means validate-then-report: nothing in this module writes back into shared state.
- The usage lookup lives on the binding although it never consults the stubs, so a caller keeps binding, validation, and call accounting on the same value; the `&self` receiver is kept for that uniformity.

## Test Strategy

- Unit tests live in each file's `#[cfg(test)]` module; `binding/tests.rs` holds the cross-file suites. Everything is offline and deterministic: no model, no network, no env, no clock.
- `parameter.rs` covers construction and defaults, sorted/deduplicated `placeholder_names` (including an unclosed opener and spaced placeholders), blank-name/blank-template validation, rendering (substitution, spaced placeholders, non-string values as compact JSON, unknown placeholders verbatim, empty placeholder name, malformed and unclosed braces, skeleton round trip, blank-template rejection), gradient/candidate bookkeeping including replacement, and the serde round trip plus per-field defaulting and unknown-field tolerance.
- `apply.rs` covers constructor/`with_description` wiring, validation accept/reject for every rule (dropped names sorted, rename reported as missing-then-introduced, added placeholder, blank candidate, invalid parameter, empty declared set), `apply` update content and non-mutation (including repeat application and the unchanged case), `apply_result` gain for positive, zero, negative, `NaN` and infinite scores plus a best prompt that fails validation, `prompt_values` as a renderable skeleton with partial filling, and the `ParameterUpdate` JSON round trip plus defaulting.
- `tools.rs` covers the declared/bound/missing partition in all orders (declaration order preserved, missing sorted), the no-tools case, every-tool-missing, duplicate declarations, blank declared names, strict `validate_case` (first declared missing name and the count), `declare_only`, `usage` delegation on a multi-call trace and an empty one, direct stub calls through `tool_stubs()` (bound call succeeds, unknown name is `NotFound`), and the report's JSON round trip plus defaulting.
- `agent.rs` covers name ordering, exact-match lookup, blank and duplicate name rejection, whitespace-distinct names, the empty agent (empty update list), `validate_all` order/updates/missing-candidate error/propagated placeholder error/ignored undeclared candidates, and `apply` as the same result as `validate_all`, including an all-unchanged map.
- `tests.rs` proves the optimizer seam end to end: a real `InstructionOptimizer` run over three cases binds its winner back through `apply_result` (unchanged declared placeholder set, no introduced placeholder, gain equal to the reported score), a two-parameter `BoundAgent` applies per-parameter optimizer winners in name order, a rendered parameter stays independent of an applied update, a declared tool binds and runs through the tool layer with the trace usage counted, and a partial declaration reports the unbound name without failing `bind_case`.
- The binding module's two doctests (apply a candidate map through `BoundAgent`; render from the placeholder skeleton) are part of the crate's doctest suite.

## Deferred

- Python's `LLMCall` operator itself is not ported: there is no mutable system/user prompt pair, no `optimizer_callback`, no frozen-prompt accessor, and no `get_freeze_*` provenance.
- `OptimizeHistory`/`TraceNode` and the traced-node gradient attribution built on them (`adopt/`) are out of scope here; a binding never observes a run.
- There is no per-parameter `OptimizationResult` map: applying several optimizer results means feeding each `best_prompt` into `BoundAgent::apply` or one `apply_result` per `ParameterBinding`.
- Python's "cannot optimize empty parameters" guard is not reproduced: an empty `BoundAgent` validates to an empty update list instead of raising.
- Tool binding matches declarations by name only; it does not validate a declared `ToolInfo` argument schema against a stub's handler, and it does not build stubs from declarations.
- No executor, runner, or agent object is bound here: the module validates and reports updates, and the caller remains responsible for storing prompts and running agents.

## Routing Table

- `./mod.rs` → module docs (including the deferred note), `mod` declarations, re-exports
- `./parameter.rs` → `TextualParameter`, placeholder-name delegation, template rendering and value rendering, blank name/template validation
- `./apply.rs` → `ParameterBinding`, `ParameterUpdate`, placeholder-set validation, `apply`/`apply_result`, the prompt-value skeleton
- `./tools.rs` → `ToolBinding`, `ToolBindingReport`, declared/bound/missing partitioning, strict `validate_case`, tool-usage delegation
- `./agent.rs` → `BoundAgent`, name-keyed parameter set, `validate_all`/`apply`
- `./tests.rs` → cross-file suites: optimizer output bound back, rendered parameters, runnable tool bindings
- `../optimizer/` → `placeholders`, `OptimizationResult` and the optimizers whose output a binding applies (sibling — read-only here)
- `../tool.rs` → `ToolStubs`, `ToolStub`, `ToolTrace`, `ToolCallRecord` (sibling — read-only here)
- `../base.rs` → `Case`, `EvaluatedCase` and the `tools`/`tool_names`/`has_tools` accessors (sibling — read-only here)
- `../CONTEXT.md` → the `tune` layer's own intent, surface and deferred list
