//! Per-member checkpoint persistence for the team harness adapter.
//!
//! This is the persistence seam between the team runtime and the provider-neutral checkpoint SPI.
//! A [`MemberCheckpointPolicy`] decides whether a member captures at all and how it consumes the
//! checkpoint supplied at start, a [`MemberCheckpointStore`] keeps the latest record per
//! `(session_id, card_id)`, and a [`MemberCheckpointSink`] adapts that store to
//! [`openjiuwen_harness_protocol::HarnessCheckpointSink`].
//!
//! Every stored [`MemberCheckpointRecord`] carries two documents, because the two consumers need
//! different halves:
//!
//! * the **payload** is a
//!   [`NativeCheckpoint`](crate::harness::checkpoint::NativeCheckpoint) — the portable,
//!   Python-parity snapshot a team restore or an operator inspection reads; and
//! * the **envelope** is the provider's own [`openjiuwen_harness_protocol::HarnessCheckpoint`],
//!   opaque to this layer and the only thing a native harness can actually resume from.
//!
//! Storage is JSON only — never pickle (see
//! [`NativeCheckpoint`](crate::harness::checkpoint::NativeCheckpoint) for the decode rules) — and
//! last-write-wins: a save replaces the record for that identity and returns a fresh revision, so
//! the protocol's `expected_storage_revision` is advisory here.
//!
//! The file backend stores one document per identity:
//!
//! ```text
//! <root>/<sanitised session>/<sanitised card>/checkpoint.json
//! ```
//!
//! A path component keeps ASCII alphanumerics, `-` and `_`, replaces every other byte (including
//! `/`, `\`, `:` and `.`) with `_`, and becomes `_` when nothing survives. `..` therefore turns
//! into `__` and no identity can escape the root.
//!
//! Module map:
//!
//! | File | Contents |
//! | --- | --- |
//! | `policy` | [`CheckpointResume`], [`MemberCheckpointPolicy`] |
//! | `store` | [`MemberCheckpointRecord`], [`MemberCheckpointStore`], in-memory and file backends |
//! | `sink` | [`MemberCheckpointSink`] over the harness-protocol sink SPI |

mod policy;
mod sink;
mod store;

pub use policy::{CheckpointResume, MemberCheckpointPolicy};
pub use sink::MemberCheckpointSink;
pub use store::{
    FileMemberCheckpointStore, InMemoryMemberCheckpointStore, MemberCheckpointRecord,
    MemberCheckpointStore,
};
