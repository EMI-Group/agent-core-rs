# skills

## Intent

Discover, validate, and budget agent skills: directories containing a `SKILL.md` with YAML front matter, discovered under workspace/global roots and optionally disabled through `skills_state.json`.
Ports `openjiuwen/harness/skills/{library_state.py,mount.py}` and the discovery/loading half of `rails/skills/skill_use_rail.py`.
The module is a leaf of the harness: it depends only on `openjiuwen_common`, `openjiuwen_sysop`, serde and tokio — never on another harness module.
Workspace paths arrive as `PathBuf`; the caller decides which roots are visible to the agent.

## API Surface

Everything is re-exported flat from this module (`crate::skills::*`).

### Library (`mod.rs`)

- `Skill { name: String, description: String, directory: PathBuf }` — `Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema`.
  `name` is the bare directory name (a `name:` front-matter field never renames a skill); first-wins across roots.
- `SkillBudget { max_skills: Option<usize>, max_total_chars: Option<usize> }` — `Default`, `unlimited()`, `is_unlimited()`, builder setters.
  `max_total_chars` sums `description.chars().count()` (Python `len(description)` ≈ chars) of whole skills, never mid-description.
- `SkillLibrary { roots: Vec<PathBuf>, disabled: BTreeSet<String> }` — public fields (callers build it with a struct literal), `Default`.
  - `new()`, `with_roots(iter)`, `add_root(path) -> bool` (true when newly appended), `remove_root(&Path) -> bool`, `is_disabled(name)`, `set_disabled(name, bool)`, `disabled_names()`.
  - `discover() -> Result<Vec<Skill>>` and `discover_with_language(Option<&str>)` — sorted DFS, first-wins dedupe, descriptions loaded from front matter; disabled skills are **not** filtered here.
  - `select(&self, skills: &[Skill], budget: SkillBudget) -> Vec<Skill>` — drops disabled names, then applies `max_skills` (hard cap) and `max_total_chars` (soft cap: drop lowest-ranked whole skills, keep at least one, warn if the survivor alone exceeds).
  - `load_state(root)`, `load_all_state()`, `save_state(root)` — merge/write `skills_state.json` (see below).

### State (`library_state.rs`)

- `SKILLS_STATE_FILENAME = "skills_state.json"`.
- `collect_disabled_skills(roots: &[PathBuf]) -> BTreeSet<String>` — async, fail-open: only the exact JSON value `{"skill_configs": {"<name>": {"enabled": false}}}` disables; missing file, unreadable/invalid JSON, non-object root or `skill_configs`, and non-object entries contribute nothing (one `tracing::warn` per unreadable file). Names are sorted/deduped across roots.
- `save_disabled_skills(root, &BTreeSet<String>)` — async, writes `{"skill_configs": {"<name>": {"enabled": false}}}` (pretty, trailing newline), creating the root if needed.
- `parse_disabled_skills(&str) -> BTreeSet<String>` — pure, unit-tested JSON parser.

### Mount (`mount.rs`)

- `split_skill_front_matter(text) -> (BTreeMap<String, Value>, String)` — tolerant: empty map + original text when there is no front matter or YAML is malformed.
- `parse_skill_front_matter(text) -> Result<Option<(BTreeMap<String, Value>, String)>>` — Python-faithful: `None` when the text does not start with `---` (no BOM/whitespace tolerance) or the closing fence is missing; `Err` on malformed YAML; non-object YAML yields `None`. Body is `lstrip`ped. Splitting is the naive first-two-`---` split Python performs.
- `SkillMetadata { name, description }` — `Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema`.
- `SkillRootMount { root: PathBuf, owned: bool }` — `mount(&mut Vec<PathBuf>, root)` appends and owns only a root it introduced, `root()`, `owned()`, `is_mounted(&[PathBuf])`, `ensure_mounted(&mut Vec<PathBuf>) -> bool` (re-add, OR ownership), `unmount(&mut Vec<PathBuf>) -> bool` (release only an owned root, idempotent). Paths compare absolute/lexically normalized.
- `SkillResourceValidator { name, directory }` — `new(name, directory)`, `name()`, `directory()`, `skill_path()`, `validate(&dyn FsOperation) -> Result<SkillMetadata>`, `validate_local() -> Result<SkillMetadata>`, `clear()`.
  Requires `<directory>/SKILL.md` to exist, front matter `name == validator name`, and a non-empty string `description`; reads only the first 64 lines unless the front matter is unterminated. Revalidates when the backend identity or the file `(mtime_ns, size)` changes; only successes are cached. Errors are `Error::validation` with Python's text `is not readable: <path>: <inner>`.

### Discovery (`discovery.rs`)

- `discover_skills(roots, Option<&str>) -> Result<Vec<Skill>>` (crate-internal).
- Rules: roots in caller order; DFS pre-order with children sorted by file name; a directory containing a `SKILL.md` **is** a skill and prunes its subtree; dot-directories and exactly `output`, `temp`, `assets`, `node_modules` are skipped at every depth; missing/non-directory roots and unreadable directories are skipped with a `tracing` warning; visited directories are deduped by canonical path (symlink cycles terminate); names are deduped first-wins with a warning.
- Description: front matter `description_<language>` (lowercased/trimmed language) first, then `description`; missing front matter or key falls back to `Skill located in <directory>`; IO errors never fail discovery.

## Constraints

- No panics or `unwrap` outside tests; IO failures are logged and skipped during discovery and fail-open for state reading.
- Async: `tokio::fs` only; directory walks are iterative (explicit stack) so no `Box::pin` recursion and no blocking IO.
- `openjiuwen_common::error::{Error, Result}`; `Error::validation` for malformed skills, `Error::sys_op` for state write failures.
- `tracing` only, no stdout; dot-directory and skip-set checks are pure string comparisons.
- Files ≤1000 lines; tests live beside each module in `#[cfg(test)] mod tests` (tempfile workspaces, `#[tokio::test]`).
- Secrets never appear here; nothing in this module touches model or network state.

## Design decisions

- `SkillLibrary` keeps no discovery cache: `discover()` is explicit and `select()` is a pure filter over already-discovered skills, so callers control refresh timing. This is why `select` takes `&[Skill]` rather than only a budget.
- Python's `SkillUseRail` both discovers and renders prompts; the Rust port splits that: discovery/state/mount here, prompt rendering in `crate::rails`.
- `SkillRootMount` is decoupled from the rail: it mutates a plain `Vec<PathBuf>` instead of borrowing a rail object, keeping this module a leaf.
- The validator takes a `&dyn FsOperation` (not the `SysOperation` facade) because the harness only needs the filesystem view; `validate_local` covers callers without a sysop backend.

## Deferred

- Prompt-section rendering, `auto_list`/`all` skill modes, keyword ranking with a query, front-matter `name` conflict rejection across roots, env-string root parsing (`a,b;c`), and tilde expansion — all belong to `crate::rails` / the agent factory.
- Language-aware descriptions are exposed through `discover_with_language`; the language itself comes from the prompt builder, not this module.

## Routing Table

- `./mod.rs` → `Skill`, `SkillBudget`, `SkillLibrary`, flat re-exports
- `./library_state.rs` → `skills_state.json` read/write/parse (disabled skills)
- `./mount.rs` → front-matter parsing, `SkillRootMount`, `SkillResourceValidator`, `SkillMetadata`, shared head reads
- `./discovery.rs` → recursive `SKILL.md` walk, skip/prune rules, description loading

## Test Strategy

- State: only `enabled:false` disables, multi-root merge/sorted dedupe, fail-open on missing/invalid/top-level-array/`skill_configs`-array/string-entry JSON, save→collect round trip.
- Discovery: grouped order (`a-group/first` before `z-last`), pruning under a skill, dot-dir + skip-set at every depth, first-wins across roots, root-is-a-skill, missing root, description precedence/fallback, symlink-cycle dedupe (Unix).
- Mount: ownership/borrowed unmount/idempotence/ensure-remount, front-matter splitting (no front matter, unterminated fence, body lstrip), validator success, name mismatch, empty description, missing `SKILL.md`, backend-error propagation, and cache-hit via a counting `FsOperation` fake.
- Parity references: `tests/unit_tests/harness/test_skill_library_state.py`, `test_skill_mount.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`.
