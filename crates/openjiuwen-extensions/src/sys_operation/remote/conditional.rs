//! Host-side conditional-write support shared by the remote sandbox providers.
//!
//! Python parity: the reference providers under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` do not implement
//! `expected_content_sha256` / `create_if_absent` at all — only the local
//! provider does — so this module is an additive Rust capability rather than a
//! port. It keeps the kernel's option semantics
//! (`openjiuwen-sysop/src/fs/write.rs`): the two options are mutually
//! exclusive, neither combines with `append`, `create_if_absent` implies
//! `create_if_not_exist`, and a rejected write leaves the target untouched.
//!
//! # Guarantee
//!
//! A conditional write is evaluated as *read the current remote bytes, check
//! the precondition, then write*, all while holding [`lock_path`] for that
//! `(provider, path)` pair. The lock is process-local, so the guarantee is:
//!
//! - two conditional writes to the same path **from this process** never
//!   interleave, and a rejected write issues no write request at all;
//! - a writer in another process, or one that bypasses this crate, is not
//!   prevented from racing with the comparison;
//! - the final write is one `upload` / `file.write` / gateway `write_file`
//!   request, so a concurrent reader can observe the container mid-write. The
//!   providers offer no atomic rename over their wire protocol, and the
//!   transports that could (`flock` + `sha256sum` + `mv` over exec) accept a
//!   different payload shape per provider, so no container-side primitive is
//!   used here.
//!
//! Locks are dropped with their guard and the registry prunes entries that are
//! no longer reachable, so a long-lived process writing many distinct paths
//! does not grow the map without bound.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use openjiuwen_core::sys_operation::fs::types::WriteFileParams;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Registry size at which unreachable lock entries are pruned.
const PRUNE_THRESHOLD: usize = 256;

/// Lower-case hex digits of [`sha256_hex`].
const HEX_DIGITS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// The process-global registry of per-path write locks.
///
/// Entries are weak so the map only ever holds locks a live guard or a
/// concurrent acquisition still references.
fn path_locks() -> &'static Mutex<BTreeMap<String, Weak<AsyncMutex<()>>>> {
    static LOCKS: OnceLock<Mutex<BTreeMap<String, Weak<AsyncMutex<()>>>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Takes the exclusive lock of one remote path in this process.
///
/// Writers of different paths never block each other. The returned guard
/// releases the lock when dropped, so the lock is held for exactly the
/// read-check-write sequence of one conditional write.
pub(crate) async fn lock_path(provider: &str, path: &str) -> OwnedMutexGuard<()> {
    let key = format!("{provider}|{path}");
    let lock = {
        let mut registry = path_locks().lock().unwrap_or_else(PoisonError::into_inner);
        match registry.get(&key).and_then(Weak::upgrade) {
            Some(lock) => lock,
            None => {
                if registry.len() >= PRUNE_THRESHOLD {
                    registry.retain(|_, entry| entry.strong_count() > 0);
                }
                let lock = Arc::new(AsyncMutex::new(()));
                registry.insert(key, Arc::downgrade(&lock));
                lock
            }
        }
    };
    lock.lock_owned().await
}

/// The SHA-256 of `bytes` as lower-case hex, the form `expected_content_sha256`
/// carries.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(HEX_DIGITS[usize::from(byte >> 4)]);
        hex.push(HEX_DIGITS[usize::from(byte & 0x0f)]);
    }
    hex
}

/// The conditional-write options of one kernel `write_file` request, reduced to
/// what the precondition check needs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Precondition {
    /// The write may only create a file that does not exist yet.
    create_if_absent: bool,
    /// The current remote bytes must hash to this hex digest.
    expected_sha256: Option<String>,
    /// The request appends to the current content.
    append: bool,
    /// The request may create a missing file.
    create_if_not_exist: bool,
}

impl Precondition {
    /// Reads the conditional options out of a kernel write request.
    #[must_use]
    pub(crate) fn from_params(params: &WriteFileParams) -> Self {
        Self {
            create_if_absent: params.create_if_absent,
            expected_sha256: params.expected_content_sha256.clone(),
            append: params.append,
            create_if_not_exist: params.create_if_not_exist,
        }
    }

    /// Whether the request carries a conditional option at all.
    #[must_use]
    pub(crate) fn is_conditional(&self) -> bool {
        self.create_if_absent || self.expected_sha256.is_some()
    }

    /// The option name a rejection message should name first.
    #[must_use]
    pub(crate) fn field(&self) -> Option<&'static str> {
        if self.expected_sha256.is_some() {
            Some("expected_content_sha256")
        } else if self.create_if_absent {
            Some("create_if_absent")
        } else {
            None
        }
    }

    /// Rejects the flag combinations the kernel refuses, before any I/O.
    ///
    /// Mirrors `validate_write` in `openjiuwen-sysop/src/fs/write.rs`: the two
    /// conditional options are mutually exclusive, neither combines with
    /// `append`, and `create_if_absent` requires `create_if_not_exist`.
    ///
    /// # Errors
    ///
    /// Returns the kernel's reason, which the caller renders as a `199003`
    /// filesystem failure.
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        let conditional = self.is_conditional();
        let incompatible = (self.expected_sha256.is_some() && self.create_if_absent)
            || (conditional && self.append)
            || (self.create_if_absent && !self.create_if_not_exist);
        if incompatible {
            return Err("conditional write options are incompatible".to_string());
        }
        Ok(())
    }

    /// Evaluates the precondition against the current remote content.
    ///
    /// `current` is the target's bytes, or `None` when the remote path does not
    /// exist. Both failures name the path, so a caller can act on the message.
    ///
    /// # Errors
    ///
    /// Returns a reason naming the path when the file already exists although
    /// the caller asked for `create_if_absent`, when the content hashes to a
    /// different digest than `expected_content_sha256`, or when an expected
    /// file is missing.
    pub(crate) fn check(
        &self,
        path: &str,
        current: Option<&[u8]>,
    ) -> std::result::Result<(), String> {
        if self.create_if_absent && current.is_some() {
            return Err(format!(
                "create_if_absent rejected the write: '{path}' already exists"
            ));
        }
        let Some(expected) = self.expected_sha256.as_deref() else {
            return Ok(());
        };
        let Some(current) = current else {
            return Err(format!(
                "expected_content_sha256 mismatch for '{path}': the remote file does not exist"
            ));
        };
        let actual = sha256_hex(current);
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "expected_content_sha256 mismatch for '{path}': the remote file hashes to {actual}, not {expected}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A write request with the kernel's defaults plus the given options.
    fn params(create_if_absent: bool, expected: Option<&str>) -> WriteFileParams {
        let mut params = WriteFileParams::raw("/work/a.txt", "x");
        params.create_if_absent = create_if_absent;
        params.expected_content_sha256 = expected.map(str::to_string);
        params
    }

    #[test]
    fn a_plain_write_has_no_precondition() {
        let precondition = Precondition::from_params(&WriteFileParams::raw("/work/a.txt", "x"));
        assert!(!precondition.is_conditional());
        assert_eq!(precondition.field(), None);
        assert_eq!(precondition.validate(), Ok(()));
        assert_eq!(precondition.check("/work/a.txt", Some(b"anything")), Ok(()));
    }

    #[test]
    fn the_two_conditional_options_are_mutually_exclusive() {
        let mut both = params(true, Some("abc"));
        both.create_if_not_exist = true;
        let error = Precondition::from_params(&both).validate().unwrap_err();
        assert_eq!(error, "conditional write options are incompatible");
    }

    #[test]
    fn a_conditional_option_never_combines_with_append() {
        for mut request in [params(true, None), params(false, Some("abc"))] {
            request.create_if_not_exist = true;
            request.append = true;
            let error = Precondition::from_params(&request).validate().unwrap_err();
            assert_eq!(error, "conditional write options are incompatible");
        }
    }

    #[test]
    fn create_if_absent_requires_create_if_not_exist() {
        let mut request = params(true, None);
        request.create_if_not_exist = false;
        let error = Precondition::from_params(&request).validate().unwrap_err();
        assert_eq!(error, "conditional write options are incompatible");

        request.create_if_not_exist = true;
        assert_eq!(Precondition::from_params(&request).validate(), Ok(()));
    }

    #[test]
    fn create_if_absent_accepts_a_missing_file_and_rejects_an_existing_one() {
        let precondition = Precondition::from_params(&params(true, None));
        assert_eq!(precondition.field(), Some("create_if_absent"));
        assert_eq!(precondition.check("/work/a.txt", None), Ok(()));
        let error = precondition.check("/work/a.txt", Some(b"")).unwrap_err();
        assert_eq!(
            error,
            "create_if_absent rejected the write: '/work/a.txt' already exists"
        );
    }

    #[test]
    fn expected_sha256_compares_the_remote_digest() {
        let digest = sha256_hex(b"hello");
        let precondition = Precondition::from_params(&params(false, Some(&digest)));
        assert_eq!(precondition.field(), Some("expected_content_sha256"));
        assert_eq!(
            precondition.check("/work/a.txt", Some(b"hello")),
            Ok(()),
            "an empty expectation must match an empty file"
        );
        let error = precondition
            .check("/work/a.txt", Some(b"goodbye"))
            .unwrap_err();
        assert!(error.contains("/work/a.txt"), "{error}");
        assert!(error.contains(&digest), "{error}");
        assert!(error.contains(&sha256_hex(b"goodbye")), "{error}");
        let error = precondition.check("/work/a.txt", None).unwrap_err();
        assert_eq!(
            error,
            "expected_content_sha256 mismatch for '/work/a.txt': the remote file does not exist"
        );
    }

    #[test]
    fn an_uppercase_digest_is_accepted() {
        let digest = sha256_hex(b"hello").to_ascii_uppercase();
        let precondition = Precondition::from_params(&params(false, Some(&digest)));
        assert_eq!(precondition.check("/work/a.txt", Some(b"hello")), Ok(()));
    }

    #[test]
    fn sha256_hex_is_the_lowercase_hex_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[tokio::test]
    async fn the_lock_serialises_one_path_and_leaves_other_paths_free() {
        let first = lock_path("fake", "/work/shared.txt").await;
        assert!(path_locks()
            .lock()
            .unwrap()
            .contains_key("fake|/work/shared.txt"));
        let other = lock_path("fake", "/work/other.txt").await;
        drop(other);
        drop(first);
        // The lock is reacquirable after the guard was dropped.
        let recovered = lock_path("fake", "/work/shared.txt").await;
        drop(recovered);
    }

    #[tokio::test]
    async fn a_second_acquisition_waits_for_the_first_guard() {
        let held = lock_path("fake", "/work/serialised.txt").await;
        let pending = tokio::spawn(async move {
            let _guard = lock_path("fake", "/work/serialised.txt").await;
        });
        tokio::task::yield_now().await;
        assert!(
            !pending.is_finished(),
            "the second acquisition must wait for the first guard"
        );
        drop(held);
        pending.await.unwrap();
    }
}
