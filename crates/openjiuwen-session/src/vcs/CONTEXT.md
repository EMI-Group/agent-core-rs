# vcs

## Intent

Local, JSON-only session version history on top of `session_controller::DiskSessionStore`: immutable versions, local refs (`HEAD`, `main`, caller-defined), rewinds, forks, and pure snapshot diffing.
Ports the local subset of `openjiuwen/core/session/vcs/` (v0.1.20) — the `VersioningManager` restore/rewind behaviour and snapshot diffing — without git, a WAL, or codecs.
Refs are metadata over the version files, not git objects: names are flat, `HEAD` is the mutable tip and `main` tracks the latest commit.

## API Surface

### `version` — ids and documents (`src/vcs/version.rs`)

- `VersionId(u64)` — `Display` renders `v000001` (six zero-padded digits), `FromStr` accepts `v000001` or `1` and rejects `0` and non-numeric input; `as_u64()` and `From<u64>`.
- `Version { id, label: Option<String>, created_at /* RFC 3339 */, parent: Option<u64>, snapshot: SessionSnapshot }` and `VersionMeta { id, label, created_at, parent }` with `Version::meta()`.
- Documents are `snake_case`, tolerant of unknown fields; a version file is immutable once written.

### `diff` — pure JSON diffing (`src/vcs/diff.rs`)

- `ChangeKind { Added, Removed, Changed }` and `JsonChange { path, kind, before, after }`.
- `diff_values(before, after) -> Vec<JsonChange>` walks objects by sorted key and arrays by ascending index (an appended tail is one `Added` per index, a removed tail one `Removed` per index) and reports any other inequality as `Changed` at that path; paths use the crate's nested-path syntax (`a.b[0].c`, root `""`) and the output is deterministic.
- `diff_snapshots(before, after)` diffs the persisted snapshot shape, so a difference surfaces as `kind`, `global.<path>`, `components.<id>.<path>`, `trace.<field>`, or `metadata.<key>`.

### `refs` — local refs (`src/vcs/refs.rs`)

- `HEAD_REF = "HEAD"` (the mutable tip, advanced by every commit and reset by a rewind), `MAIN_REF = "main"` (the latest committed version), `MAX_REF_NAME_LEN = 128`.
- `RefName` — validated name (`new`, `head()`, `main()`, `as_str()`, `is_head()`); non-empty, at most 128 characters, and free of whitespace, control characters, path separators and `..`.
- `RefEntry { name, target: u64, created_at /* RFC 3339 */, updated_at }` with a tolerant `Default`.
- `Refs` — `new`, `get`, `set`, `set_head`, `set_main`, `remove` (reports whether the ref existed), `list`, `len`, `is_empty`, `head`, `main`; the document is `snake_case`, tolerant of missing and unknown fields, with monotonic `updated_at` per entry.

### `history` — history, rewinds and forks (`src/vcs/history.rs`)

- Layout under a session directory: `history/versions/v000001.json` (one `Version`), `history/index.json` (an optional version metadata cache, rescanned whenever absent or corrupt), `history/refs.json` (the `Refs` document, removed when it becomes empty).
- `SessionHistory::new(DiskSessionStore)` (or `store.history()`) with path helpers `store`, `history_dir`, `versions_dir`, `version_path`, `index_path`, `refs_path`.
- Version ids are sequential and allocated under the store-level `tokio::sync::Mutex`, so concurrent commits — even through cloned stores or histories — never reuse an id; a new version's `parent` is the previous latest and the history directory is created lazily; every write is atomic (temp file + rename).
- Version methods: `commit(agent_id, session_id, &SessionSnapshot, label) -> VersionMeta` (also advances `HEAD` and `main`), `commit_from_session`, `list` (ascending id), `load`, `latest`, `delete` (returns `false` when missing), `clear` (removes the history directory), `diff(left, right) -> Vec<JsonChange>` (`Error::Session` when either version is missing), `restore(agent_id, session_id, id, &dyn Session) -> bool` (applies an old snapshot without committing).
- Ref methods: `refs`, `set_ref` (rejects a version that does not exist), `delete_ref`, `resolve_ref`, `list_refs`, `head` (falls back to `latest()` when the document or its target is missing).
- `RewindOptions { discard_newer }` with `SOFT` (`false`) and `HARD` (`true`); `rewind`/`rewind_hard` are the two recipes and `rewind_with` takes the options explicitly.
- A rewind restores the live session first (so a failed restore leaves the history untouched) and then resets `HEAD` to the target; the soft form keeps every newer version and leaves `main` alone, the hard form deletes them, rewrites the version index and re-points `main`. Rewinding to the current `HEAD` is idempotent, a missing target version (or an empty history without a target) is `Error::Session`, and a malformed id is `Error::Validation`.
- `fork(agent_id, source_session_id, from, new_session_id, title)` copies the chosen version's snapshot (default: the source's `HEAD`) into a brand-new session record in the **source session's scope** and commits it as the fork's first version, labelled `fork of <source>@v000003`; that version has no `parent` (a new lineage) and the fork's `HEAD`/`main` point at it.
- Fork errors: a malformed id, a `new_session_id` equal to the source or already stored for the agent, and a source session the store does not know are `Error::Validation`; a missing source version (or an empty source history without `from`) is `Error::Session`. The source session, its history and its refs stay untouched.
- An unreadable or malformed refs document is logged at warn level and treated as empty, so a damaged refs file never blocks commits or restores.

## Constraints

- Offline and deterministic: `tokio::fs` only, atomic temp-file-plus-rename writes, no network, no wall-clock ordering assumptions beyond `created_at` strings.
- Errors: `Error::Validation` for ids/names/user input, `Error::Session` for I/O, corrupt documents and missing versions; never panic on user input.
- Every public type is serde-serializable, `snake_case`, tolerant of unknown fields, with `#[serde(default)]`.
- The history owns version allocation and ref updates; `DiskSessionStore` owns the directories and the atomic-write helpers, and `Session::restore`/`Session::snapshot` are the only session operations used.
- Deferred from Python's `vcs/`: git and other external backends, event/WAL (`e<N>`) references, codecs, and context-engine adapters; the local history, refs, rewinds and forks are self-contained.
- No `unsafe`, `tracing` only, no stdout logging, files ≤ ~1000 lines.

## Design Decisions

- **One JSON file per version with sequential ids**, not git or a WAL: ids are allocated under one lock, each version records its `parent`, and diffing is a pure function over JSON, so history stays offline, inspectable, and dependency-free.
- **Refs are plain metadata.** `HEAD` and `main` are entries in one small document with the same atomic write path as versions, so a rewind is two writes (refs, optionally truncation) and a missing or corrupt document degrades to "no refs" instead of failing the history.
- **Soft by default, hard on request.** `rewind` matches Python's read-only replay (the live session and `HEAD` move, the newer versions stay), while `rewind_hard` mirrors Python's overwrite `rewind` that truncates after the target; keeping both names makes the destructive choice explicit at the call site.
- **A fork is a new session plus a one-version history.** Copying the snapshot into a session record in the source's scope (rather than sharing files) keeps both sessions independently resumable, and the labelled first version records the lineage without needing the link graph.

## Routing Table

- `./version.rs` → `VersionId`, `Version`, `VersionMeta`
- `./diff.rs` → `diff_values`, `diff_snapshots`, `ChangeKind`, `JsonChange`
- `./refs.rs` → `RefName`, `RefEntry`, `Refs`, `HEAD_REF`, `MAIN_REF`, `MAX_REF_NAME_LEN`
- `./history.rs` → `SessionHistory` (commit/list/load/latest/delete/clear/diff/restore), ref management, `RewindOptions`/`rewind`/`rewind_hard`/`rewind_with`, `fork`
- `./mod.rs` → module wiring and re-exports (`vcs::{…}`)
