# skills

## Intent

Discover, validate, and budget agent skills: directories containing a `SKILL.md` with YAML front matter, discovered under workspace/global roots and optionally disabled through `skills_state.json`.
Ports `openjiuwen/harness/skills/{library_state.py,mount.py}` and the discovery half of `rails/skills/skill_use_rail.py`.

## API Surface

- `Skill { name, description, directory: PathBuf }` — name is the bare directory name, first-wins across roots.
- `SkillLibrary { roots: Vec<PathBuf>, disabled: BTreeSet<String> }`: `discover() -> Result<Vec<Skill>>` (sorted recursive walk; a directory containing `SKILL.md` **is** a skill and prunes the walk; skips dot-directories and `output`/`temp`/`assets`/`node_modules`), `select(budget: SkillBudget) -> Vec<Skill>` (max skills, max total characters), `load_state(root)`/`save_state(root)` for `skills_state.json`.
- `SkillBudget { max_skills: Option<usize>, max_total_chars: Option<usize> }`.
- `library_state`: `SKILLS_STATE_FILENAME = "skills_state.json"`, `collect_disabled_skills(roots) -> BTreeSet<String>` (only `enabled: false` entries; fail-open on unreadable/invalid JSON).
- `mount.rs`: `split_skill_front_matter(text) -> (BTreeMap<String, Value>, String)`, `SkillRootMount { root, owned }` with `ensure_mounted(&mut Vec<PathBuf>)`/`unmount`, `SkillResourceValidator { name, directory }` with `validate(fs) -> Result<SkillMetadata>` (requires `SKILL.md`, frontmatter `name == name`, non-empty `description`).
- `SkillMetadata { name, description }`.

## Constraints

- Discovery never panics or fails the agent on IO errors; unreadable roots are skipped with a `tracing::warn`.
- Only front matter head (first ~64 lines) is read for validation.
- Mounting appends a root to the library; it never copies or symlinks skill content.
- Uses `openjiuwen_sysop::FsOperation` when supplied, falling back to `tokio::fs`.
- Errors: `Error::validation` for malformed skills, `Error::not_found` for missing `SKILL.md`.

## Routing Table

- `./mod.rs` → `Skill`, `SkillLibrary`, budget, re-exports
- `./library_state.rs` → `skills_state.json` handling (disabled skills)
- `./mount.rs` → front matter parsing, `SkillRootMount`, `SkillResourceValidator`
- `./discovery.rs` → recursive `SKILL.md` walk and root conventions

## Test Strategy

- Unit tests: only `enabled:false` entries disable skills, multi-root merge, fail-open on invalid state JSON, directory-with-`SKILL.md` pruning, dot-directory skipping, mount/unmount ownership, validation rejects a mismatched front-matter name.
- Parity references: `tests/unit_tests/harness/test_skill_library_state.py`, `test_skill_mount.py`.
