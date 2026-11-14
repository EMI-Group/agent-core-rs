//! Validation and mounting of a consultation skill root into a skill library.
//!
//! Ports `openjiuwen/harness/moa/skill_mount.py`. Python's `ConsultationSkillMount` borrows a
//! `SkillUseRail` from a live `DeepAgent`, validates the skill through `SysOperation`, and
//! attaches the root to that rail; the Rust port has no agent-owned rail, so it validates the
//! local path directly and mounts a leaf discovery root into a caller-supplied
//! [`SkillLibrary`]. Only MoA-owned roots are tracked, so unmounting never disturbs roots the
//! host configured itself.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use crate::skills::{Skill, SkillLibrary, SkillRootMount};

use super::builtin_skills::{consult_skill, CONSULT_SKILL_FILE, CONSULT_SKILL_NAME};

/// Error message used when the configured candidates are not a consultation skill directory.
const INVALID_DIRECTORY: &str =
    "consultation_skill_dir must be a consult-reference-models directory with SKILL.md";

/// Error message used when no candidate directory was configured at all.
const MISSING_DIRECTORY: &str =
    "MoA consultation skill requires a skill directory; materialize the embedded skill first";

/// Error message used when the recorded mount no longer covers the configured directory.
const STALE_MOUNT: &str =
    "the recorded MoA consultation skill mount does not cover consultation_skill_dir";

/// Tracks whether the consultation skill root is mounted, and who introduced it.
///
/// A handle is created with an explicit skill directory ([`Self::from_dir`]) or without one
/// ([`Self::embedded`]); the latter only satisfies the type and must be given a directory
/// before it can be validated, because Python's packaged directory has no Rust counterpart
/// until [`super::builtin_skills::materialize_builtin_skill`] writes one.
#[derive(Debug, Clone, Default)]
pub struct ConsultationSkillMount {
    skill_dir: Option<PathBuf>,
    mount: Option<SkillRootMount>,
}

impl ConsultationSkillMount {
    /// Creates a handle for an optional explicit skill directory.
    #[must_use]
    pub fn new(skill_dir: Option<PathBuf>) -> Self {
        Self {
            skill_dir,
            mount: None,
        }
    }

    /// Creates a handle without a directory, matching Python's packaged-skill default.
    ///
    /// Validation and [`Self::skill`] fail with [`Error::config`] until the embedded skill is
    /// materialized and supplied through [`Self::from_dir`].
    #[must_use]
    pub fn embedded() -> Self {
        Self::new(None)
    }

    /// Creates a handle for `directory`.
    #[must_use]
    pub fn from_dir(directory: impl Into<PathBuf>) -> Self {
        Self::new(Some(directory.into()))
    }

    /// Returns the configured skill directory, when there is one.
    #[must_use]
    pub fn skill_dir(&self) -> Option<&Path> {
        self.skill_dir.as_deref()
    }

    /// Returns `true` when this handle currently holds a mount of its skill root.
    ///
    /// A held mount does not imply ownership: a root that was already present in the library
    /// stays mounted after [`Self::unmount`] and the call reports no removal.
    #[must_use]
    pub fn is_mounted(&self) -> bool {
        self.mount.is_some()
    }

    /// Returns the consultation [`Skill`] descriptor for the configured directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no directory was configured.
    pub fn skill(&self) -> Result<Skill> {
        Ok(consult_skill(self.require_dir()?))
    }

    /// Checks that the configured directory holds this skill's `SKILL.md`.
    ///
    /// The directory must be named `consult-reference-models` and contain a regular
    /// `SKILL.md`, so a task-loop turn can never read a sibling skill or an unrelated tree.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] with `consultation_skill_dir must be a
    /// consult-reference-models directory with SKILL.md` when the directory is missing, has a
    /// different leaf name, or has no `SKILL.md`; without a configured directory it returns
    /// the materialize-first message.
    pub fn validate(&self) -> Result<()> {
        let directory = self.require_dir()?.to_path_buf();
        let named_correctly = directory
            .file_name()
            .is_some_and(|name| name == CONSULT_SKILL_NAME);
        if !named_correctly || !directory.join(CONSULT_SKILL_FILE).is_file() {
            return Err(Error::config(INVALID_DIRECTORY));
        }
        Ok(())
    }

    /// Validates the skill and mounts its root into `library`.
    ///
    /// The root is attached through [`SkillRootMount`], so only a root this call introduced is
    /// released again by [`Self::unmount`]. Mounting twice is idempotent: the stored handle
    /// re-adds the root when it is absent, keeps its prior ownership and logs the re-add.
    ///
    /// # Errors
    ///
    /// Returns the [`Self::validate`] error when the directory is absent or is not a
    /// consultation skill directory, and [`Error::Config`] when the recorded mount no longer
    /// covers the configured directory: re-adding that root would leave the configured skill
    /// undiscoverable while `mount` reported success.
    pub fn mount(&mut self, library: &mut SkillLibrary) -> Result<Skill> {
        self.validate()?;
        let skill = self.skill()?;
        match &mut self.mount {
            Some(mount) => {
                // Keep prior ownership: the stored handle re-adds the root only when it went
                // missing, and the re-add is verified because a recorded root that drifted
                // away from the configured directory would silently mount the wrong tree.
                if !same_skill_root(mount.root(), &skill.directory) {
                    tracing::warn!(
                        recorded_root = %mount.root().display(),
                        skill_dir = %skill.directory.display(),
                        "MoA consultation skill mount does not cover the configured directory"
                    );
                    return Err(Error::config(STALE_MOUNT));
                }
                let owned = mount.ensure_mounted(&mut library.roots);
                tracing::debug!(
                    root = %mount.root().display(),
                    owned,
                    "reused the recorded MoA consultation skill mount"
                );
            }
            None => {
                self.mount = Some(SkillRootMount::mount(&mut library.roots, &skill.directory));
            }
        }
        Ok(skill)
    }

    /// Releases the root through the stored mount and drops the handle.
    ///
    /// Returns whether a root was removed, which is `false` for a handle that was never
    /// mounted, for a root the library already contained, and for a second call.
    pub fn unmount(&mut self, library: &mut SkillLibrary) -> bool {
        let Some(mut mount) = self.mount.take() else {
            return false;
        };
        mount.unmount(&mut library.roots)
    }

    /// Returns the configured directory or the materialize-first error.
    fn require_dir(&self) -> Result<&Path> {
        self.skill_dir
            .as_deref()
            .ok_or_else(|| Error::config(MISSING_DIRECTORY))
    }
}

/// Returns `true` when `recorded` and `directory` name the same skill root.
///
/// The library owns the path-normalization rules, so the comparison borrows them from
/// [`SkillRootMount`] instead of re-deriving them here.
fn same_skill_root(recorded: &Path, directory: &Path) -> bool {
    SkillRootMount::mount(&mut Vec::new(), recorded).is_mounted(&[directory.to_path_buf()])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        same_skill_root, ConsultationSkillMount, INVALID_DIRECTORY, MISSING_DIRECTORY, STALE_MOUNT,
    };
    use crate::moa::builtin_skills::{
        materialize_builtin_skill, CONSULT_SKILL_FILE, CONSULT_SKILL_NAME,
    };
    use crate::skills::{SkillLibrary, SkillRootMount};
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn embedded_handle_requires_a_materialized_directory() {
        let mount = ConsultationSkillMount::embedded();
        assert!(mount.skill_dir().is_none());
        assert!(!mount.is_mounted());

        let error = mount.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {MISSING_DIRECTORY}"));

        let error = mount.skill().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {MISSING_DIRECTORY}"));

        assert_eq!(ConsultationSkillMount::default().skill_dir(), None);
    }

    #[test]
    fn from_dir_rejects_wrong_names_and_missing_files() {
        let temp = tempfile::tempdir().unwrap();

        // Right file, wrong leaf name.
        let other = temp.path().join("other-skill");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join(CONSULT_SKILL_FILE), b"---\nname: x\n---\n").unwrap();
        let mount = ConsultationSkillMount::from_dir(&other);
        assert_eq!(mount.skill_dir(), Some(other.as_path()));
        let error = mount.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {INVALID_DIRECTORY}"));

        // Right name, no SKILL.md.
        let empty = temp.path().join(CONSULT_SKILL_NAME);
        std::fs::create_dir_all(&empty).unwrap();
        let mount = ConsultationSkillMount::from_dir(&empty);
        let error = mount.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {INVALID_DIRECTORY}"));

        // A directory holding SKILL.md as a directory is not a skill either.
        let bogus = temp.path().join("bogus").join(CONSULT_SKILL_NAME);
        std::fs::create_dir_all(bogus.join(CONSULT_SKILL_FILE)).unwrap();
        let mount = ConsultationSkillMount::from_dir(&bogus);
        assert_eq!(
            mount.validate().unwrap_err().to_string(),
            format!("config: {INVALID_DIRECTORY}")
        );
    }

    #[tokio::test]
    async fn mount_discovers_the_skill_and_unmount_keeps_foreign_roots() {
        let temp = tempfile::tempdir().unwrap();
        let foreign = temp.path().join("foreign-root");
        std::fs::create_dir_all(&foreign).unwrap();

        let mut library = SkillLibrary::with_roots([foreign.clone()]);
        let skill = materialize_builtin_skill(temp.path()).await.unwrap();

        let mut mount = ConsultationSkillMount::from_dir(&skill.directory);
        mount.validate().unwrap();
        let mounted = mount.mount(&mut library).unwrap();
        assert_eq!(mounted, skill);
        assert!(mount.is_mounted());
        assert_eq!(
            library.roots,
            vec![foreign.clone(), skill.directory.clone()]
        );

        let discovered = library.discover().await.unwrap();
        assert!(
            discovered.iter().any(|found| found.name == CONSULT_SKILL_NAME
                && found.directory == skill.directory),
            "discovered: {discovered:?}"
        );

        // Mounting again is idempotent.
        assert_eq!(mount.mount(&mut library).unwrap(), skill);
        assert_eq!(library.roots.len(), 2);

        assert!(mount.unmount(&mut library));
        assert!(!mount.is_mounted());
        assert_eq!(library.roots, vec![foreign]);
        assert!(!mount.unmount(&mut library));
    }

    #[tokio::test]
    async fn mounting_again_restores_a_root_that_went_missing() {
        let temp = tempfile::tempdir().unwrap();
        let skill = materialize_builtin_skill(temp.path()).await.unwrap();
        let mut library = SkillLibrary::new();

        let mut mount = ConsultationSkillMount::from_dir(&skill.directory);
        mount.mount(&mut library).unwrap();
        assert_eq!(library.roots, vec![skill.directory.clone()]);

        // The root disappears behind the handle's back.
        library.roots.clear();
        assert_eq!(mount.mount(&mut library).unwrap(), skill);
        assert_eq!(library.roots, vec![skill.directory.clone()]);

        // Prior ownership survives the re-add, so the root is still released on unmount.
        assert!(mount.unmount(&mut library));
        assert_eq!(library.roots, Vec::<std::path::PathBuf>::new());
    }

    #[tokio::test]
    async fn mount_rejects_a_recorded_root_that_drifted_from_the_directory() {
        let temp = tempfile::tempdir().unwrap();
        let skill = materialize_builtin_skill(temp.path()).await.unwrap();
        let mut library = SkillLibrary::new();

        // A handle that recorded a mount for a different root: re-adding it would mount the
        // wrong tree and leave the configured consultation skill undiscoverable.
        let stale = temp.path().join("stale-root");
        let mut mount = ConsultationSkillMount::from_dir(&skill.directory);
        mount.mount = Some(SkillRootMount::mount(&mut library.roots, &stale));

        let error = mount.mount(&mut library).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {STALE_MOUNT}"));
        // The failed call leaves the library untouched: the configured skill root is not
        // mounted, which is exactly what the error reports.
        assert_eq!(library.roots, vec![stale]);
        assert!(!library.roots.contains(&skill.directory));
    }

    #[test]
    fn same_skill_root_follows_the_library_path_rules() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join(CONSULT_SKILL_NAME);
        assert!(same_skill_root(&root, &root));
        assert!(same_skill_root(&root.join("."), &root));
        assert!(!same_skill_root(&root, &temp.path().join("other-skill")));
    }

    #[tokio::test]
    async fn unmount_does_not_remove_a_preexisting_root() {
        let temp = tempfile::tempdir().unwrap();
        let skill = materialize_builtin_skill(temp.path()).await.unwrap();
        let mut library = SkillLibrary::with_roots([skill.directory.clone()]);

        let mut mount = ConsultationSkillMount::from_dir(&skill.directory);
        mount.mount(&mut library).unwrap();
        assert!(mount.is_mounted());

        assert!(!mount.unmount(&mut library));
        assert_eq!(library.roots, vec![skill.directory]);
    }
}
