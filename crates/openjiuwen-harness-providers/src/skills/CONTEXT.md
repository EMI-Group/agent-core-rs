# skills

## Intent

Portable skill-bundle normalization, installation and per-vendor policy.
Ports Python `openjiuwen.harness_providers.skills` (v0.1.20): a provider configuration names skill sources — one bundle, or a library directory of bundles — and the installer copies the complete bundles into the vendor CLI's project-local discovery directory (`claudecode` → `.claude/skills`, `codex` → `.agents/skills`, `dsh` → `.dsh/skills`).
A bundle is a directory containing `SKILL.md`; the skill name comes from the document's front-matter `name:` entry, falling back to the bundle directory name.
The three vendor drivers call `install_for_provider` from their `open` (on a blocking task) and fail the start when installation fails; nothing installs at `create` time.

## API Surface

- `SkillMode` — `All` (every bundle) / `AutoList` (only `SkillSource::enabled_skills`), serde `snake_case`.
- `SkillConflict` — `Skip` / `Replace` / `Append` (install under the next free `name-N` and rewrite the copied front-matter `name:`), serde `snake_case`, plus `FromStr` for the Python wire values.
- `SkillSource { dir, mode, enabled_skills }` with `new(dir)`, `with_mode`, `with_enabled_skills`, `from_json` (a bare string is a directory with the defaults).
- `normalize_skills(&[JsonValue], SkillConflict) -> Result<Vec<SkillSource>>` — the `conflict` argument exists for Python signature parity only and is not consulted.
- `skill_scan_dir(provider) -> Option<&'static str>` — the built-in provider's discovery directory, `None` for any other name.
- `install_skills(&[SkillSource], provider, cwd: &Path, conflict) -> Result<Vec<PathBuf>>` — the portable installer; returns the installed destinations sorted by path.
- `SkillLayout` — `Directory` (`<scan>/<name>/SKILL.md`) / `DirectoryOrFlat` (additionally a flat `<scan>/<name>.md`), with `as_str()` giving `"directory"` / `"directory_or_flat"`.
- `SkillInstaller` (`for_provider(provider, conflict)`, `provider()`, `scan_dir()`, `layout()`, `install(sources, cwd: Option<&Path>)`, `validate_name(name)`) and the wrapper `install_for_provider(sources, provider, cwd, conflict)`.

## Constraints

- Discovery happens before any write: the bundle or library is scanned, every name validated, and only then is a destination touched — an invalid source or a rejected name leaves no discovery directory behind.
- Libraries are walked recursively and sorted by path; a directory that is itself a bundle stops the walk, and symlinked directories are not traversed.
- Internal symlinks are materialized by copying through them; escaping links, directory cycles and non-regular files are rejected, and the canonical discovery directory must stay inside the project (a symlinked scan dir that escapes it is rejected).
- Installation is staged inside the project as `.openjiuwen-skill-<uuid>/new` with the previous bundle moved to `.openjiuwen-skill-<uuid>/previous`, so the swap stays on one filesystem and a failed rename restores it; the stage directory is always removed.
- `install_skills` creates the discovery directory before canonicalizing it (Python resolves the not-yet-existing path non-strictly).
- Per-vendor rules: `claudecode` and `codex` share the portable `[A-Za-z0-9._-]+` name rule, `dsh` requires kebab-case (`^[a-z0-9]+(-[a-z0-9]+)*$`, message `"DSH skill names must be kebab-case: '<name>'"`).
- A flat `*.md` source root is accepted only by `DirectoryOrFlat`: it is copied into a temporary directory as `<name>/SKILL.md` and then installed like a directory bundle; a `Directory` provider rejects it with `"the {provider} skill layout requires a SKILL.md directory bundle"`.
- `SkillSource` is part of every vendor create mapping (`skills`, `skill_conflict`), decoded through the crate-internal `parse_skills_value` (`"skills must be an array"` for a non-null, non-array value).
- Errors are `Error::Validation` for shape/name/path rejections and `Error::External` for filesystem failures; no panics on user input, `tracing` only, files ≤1000 lines.

## Design Decisions

- `normalize_skills` keeps Python's `conflict` parameter but ignores it: `SkillConflict` is a validated enum here, so the conflict is decided at the call site (`install_skills`) instead of being re-parsed during normalization.
- The portable-name rule is deliberately laxer than Python's (Python also requires an alphanumeric first character and at most 128 bytes); the per-vendor `dsh` rule is stricter than both.
- `SkillInstaller::install` defaults `cwd` to the process working directory, while the drivers pass the configured project directory so a skill install cannot escape the workspace.
- An appended copy has only its front-matter `name:` line rewritten in place; the document is not round-tripped through YAML (a comment/ordering-preserving choice, documented deviation).

## Deferred

- Remote/marketplace skill sources and content search are not ported — sources are local directories only.
- Python's skill *evaluation* and prompt-builder integration live in `openjiuwen-dev-tools`, not here.

## Test Strategy

- `tests/skills.rs` — 33 offline tests: normalize modes and errors, `SkillSource` serde round-trip, skip/replace/append conflict policies, enabled-skill selection, traversal/cycle/non-regular-file rejection, symlink materialization, sorted destinations, scan-directory mapping, `SkillInstaller` name rules, the unknown-provider and flat-layout rejections, and the `dsh` flat-`*.md` normalization. Every case uses `tempfile::tempdir`.
- Unit tests beside the module for the pure helpers (front-matter name extraction, portable-name validation, mode parsing).

## Routing Table

- `./mod.rs` → `SkillMode`, `SkillConflict`, `SkillSource`, `normalize_skills`, `parse_skills_value`, `skill_scan_dir`, `install_skills` and its discovery/staging helpers
- `./installer.rs` → the per-vendor policy: `SkillLayout`, `SkillInstaller`, `install_for_provider`, flat-`*.md` normalization
