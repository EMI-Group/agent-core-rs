//! Per-vendor skill installation policy on top of the portable installer.
//!
//! [`install_skills`](super::install_skills) copies portable bundles and applies the conflict
//! policy, but each vendor runtime discovers skills in its own layout and enforces its own
//! skill-name rule. [`SkillInstaller`] carries that per-provider policy:
//!
//! | Provider | Discovery directory | Layout | Name rule |
//! |---|---|---|---|
//! | `claudecode` | `.claude/skills` | [`SkillLayout::Directory`] | portable `[A-Za-z0-9._-]` |
//! | `codex` | `.agents/skills` | [`SkillLayout::Directory`] | portable `[A-Za-z0-9._-]` |
//! | `dsh` | `.dsh/skills` | [`SkillLayout::DirectoryOrFlat`] | kebab-case |
//!
//! Every bundle name is validated before the destination directory is touched, so a rejected
//! name never leaves a discovery directory behind. A flat `*.md` source root is only accepted by a
//! [`SkillLayout::DirectoryOrFlat`] provider and is normalized into the portable
//! `<scan>/<name>/SKILL.md` directory form, because the directory form is what every vendor
//! runtime discovers.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use super::{
    document_front_matter_name, install_skills, is_enabled, plan_install, skill_scan_dir,
    validate_portable_name, SkillConflict, SkillSource,
};

/// On-disk bundle layout a vendor runtime discovers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillLayout {
    /// `<scan>/<name>/SKILL.md` only (Claude Code, Codex).
    Directory,
    /// `<scan>/<name>/SKILL.md` or a flat `<scan>/<name>.md` (DSH).
    DirectoryOrFlat,
}

impl SkillLayout {
    /// Returns the stable wire name of the layout (`"directory"` or `"directory_or_flat"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Directory => "directory",
            Self::DirectoryOrFlat => "directory_or_flat",
        }
    }
}

/// Layouts of the providers that declare a skill discovery directory.
const LAYOUTS: &[(&str, SkillLayout)] = &[
    ("claudecode", SkillLayout::Directory),
    ("codex", SkillLayout::Directory),
    ("dsh", SkillLayout::DirectoryOrFlat),
];

/// Per-vendor skill installation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkillInstaller {
    provider: &'static str,
    scan_dir: &'static str,
    layout: SkillLayout,
    conflict: SkillConflict,
}

impl SkillInstaller {
    /// Builds the installer for a registry provider name (`claudecode`, `codex`, `dsh`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with
    /// `"unknown skill provider '<x>'; expected one of claudecode, codex, dsh"` for any other
    /// name.
    pub fn for_provider(provider: &str, conflict: SkillConflict) -> Result<Self> {
        let Some(&(name, layout)) = LAYOUTS.iter().find(|(name, _)| *name == provider) else {
            return Err(Error::validation(format!(
                "unknown skill provider '{provider}'; expected one of claudecode, codex, dsh"
            )));
        };
        let scan_dir = skill_scan_dir(name).ok_or_else(|| {
            Error::validation(format!(
                "provider '{name}' declares no skill discovery directory"
            ))
        })?;
        Ok(Self {
            provider: name,
            scan_dir,
            layout,
            conflict,
        })
    }

    /// Returns the registry name of the provider the installer targets.
    #[must_use]
    pub fn provider(&self) -> &str {
        self.provider
    }

    /// Returns the project-relative discovery directory of the provider.
    #[must_use]
    pub fn scan_dir(&self) -> &str {
        self.scan_dir
    }

    /// Returns the on-disk bundle layout the provider discovers.
    #[must_use]
    pub fn layout(&self) -> SkillLayout {
        self.layout
    }

    /// Installs `sources` into `<cwd or process cwd>/<scan_dir>` under `self.conflict`.
    ///
    /// Every bundle is discovered and its name validated against the vendor rule before the
    /// discovery directory is created or written to. A [`SkillLayout::DirectoryOrFlat`] source
    /// root that is itself a `*.md` file is installed as the portable `<scan>/<name>/SKILL.md`
    /// directory bundle.
    ///
    /// # Errors
    ///
    /// Returns the [`Error::Validation`] of an unknown name or rejected layout, plus every error
    /// documented by [`install_skills`](super::install_skills); returns [`Error::External`] when
    /// the process working directory cannot be read and `cwd` is `None`.
    pub fn install(&self, sources: &[SkillSource], cwd: Option<&Path>) -> Result<Vec<PathBuf>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let project = match cwd {
            Some(path) => path.to_path_buf(),
            None => std::env::current_dir()?,
        };
        let prepared = self.prepare(sources)?;
        install_skills(&prepared.sources, self.provider, &project, self.conflict)
    }

    /// Validates one bundle name against the vendor's rule.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `"DSH skill names must be kebab-case: '<name>'"`
    /// (DSH requires `^[a-z0-9]+(-[a-z0-9]+)*$`) or with the shared
    /// `"invalid portable skill name: '<name>'"` message of the Claude Code and Codex layouts.
    pub fn validate_name(&self, name: &str) -> Result<()> {
        if self.provider == "dsh" {
            if is_kebab_name(name) {
                Ok(())
            } else {
                Err(Error::validation(format!(
                    "DSH skill names must be kebab-case: '{name}'"
                )))
            }
        } else {
            validate_portable_name(name)
        }
    }

    /// Validates every bundle name and normalizes flat `*.md` sources into bundle directories.
    ///
    /// Only reads the sources and writes into a private temporary directory, so the project's
    /// discovery directory stays untouched until the caller copies the prepared sources.
    fn prepare(&self, sources: &[SkillSource]) -> Result<Prepared> {
        let mut stage = FlatStage::default();
        let mut prepared = Vec::with_capacity(sources.len());
        for source in sources {
            if !is_flat_bundle(&source.dir)? {
                for (name, _bundle) in plan_install(std::slice::from_ref(source))? {
                    self.validate_name(&name)?;
                }
                prepared.push(source.clone());
                continue;
            }
            if self.layout == SkillLayout::Directory {
                return Err(Error::validation(format!(
                    "the {} skill layout requires a SKILL.md directory bundle",
                    self.provider
                )));
            }
            let name = flat_bundle_name(&source.dir)?;
            self.validate_name(&name)?;
            if !is_enabled(source.enabled_skills.as_deref(), &name) {
                continue;
            }
            prepared.push(SkillSource {
                dir: stage.add(&name, &source.dir)?,
                mode: source.mode,
                enabled_skills: source.enabled_skills.clone(),
            });
        }
        Ok(Prepared {
            sources: prepared,
            _stage: stage,
        })
    }
}

/// Convenience wrapper: `SkillInstaller::for_provider(provider, conflict)?.install(sources, cwd)`.
///
/// # Errors
///
/// Returns every error of [`SkillInstaller::for_provider`] and [`SkillInstaller::install`].
pub fn install_for_provider(
    sources: &[SkillSource],
    provider: &str,
    cwd: Option<&Path>,
    conflict: SkillConflict,
) -> Result<Vec<PathBuf>> {
    SkillInstaller::for_provider(provider, conflict)?.install(sources, cwd)
}

/// The validated, layout-normalized sources of one installation.
struct Prepared {
    /// Sources ready for [`install_skills`](super::install_skills).
    sources: Vec<SkillSource>,
    /// Kept alive so the normalized flat bundles exist while they are copied.
    _stage: FlatStage,
}

/// Temporary directory holding the directory-form copies of flat `*.md` bundles.
#[derive(Default)]
struct FlatStage {
    /// Root of the temporary directory; created on the first flat bundle.
    root: Option<PathBuf>,
}

impl FlatStage {
    /// Materializes `document` as `<root>/<name>/SKILL.md` and returns the bundle directory.
    fn add(&mut self, name: &str, document: &Path) -> Result<PathBuf> {
        let root = if let Some(root) = &self.root {
            root.clone()
        } else {
            let root = std::env::temp_dir()
                .join(format!("openjiuwen-skill-flat-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root)?;
            self.root = Some(root.clone());
            root
        };
        let bundle = root.join(name);
        std::fs::create_dir_all(&bundle)?;
        std::fs::copy(document, bundle.join("SKILL.md"))?;
        Ok(bundle)
    }
}

impl Drop for FlatStage {
    fn drop(&mut self) {
        if let Some(root) = &self.root {
            // Best-effort cleanup: the installed bundles are the project copies, so a failure
            // here must not mask the install result.
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

/// Returns `true` when `path` is a flat `*.md` skill document rather than a bundle directory.
fn is_flat_bundle(path: &Path) -> Result<bool> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("md")))
}

/// Returns the skill name of a flat `*.md` bundle: its front-matter `name:`, else its file stem.
fn flat_bundle_name(document: &Path) -> Result<String> {
    let resolved = document.canonicalize()?;
    let text = std::fs::read_to_string(&resolved)?;
    if let Some(name) = document_front_matter_name(&text, document)? {
        return Ok(name);
    }
    document
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| Error::validation(format!("skill file has no name: {}", document.display())))
}

/// Returns `true` when `name` matches `^[a-z0-9]+(-[a-z0-9]+)*$`.
fn is_kebab_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('-').all(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit())
        })
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::path::{Path, PathBuf};

    use openjiuwen_common::error::Error;
    use tempfile::tempdir;

    use super::{install_for_provider, SkillInstaller, SkillLayout};
    use crate::skills::{SkillConflict, SkillSource};

    /// Writes a complete bundle declaring `name` and returns its directory.
    fn write_bundle(root: &Path, name: &str) -> PathBuf {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill\n---\nbody\n"),
        )
        .unwrap();
        root.to_path_buf()
    }

    /// Creates and canonicalizes a project directory.
    fn project_dir(path: &Path) -> PathBuf {
        std::fs::create_dir_all(path).unwrap();
        path.canonicalize().unwrap()
    }

    #[test]
    fn for_provider_resolves_the_builtin_vendors() {
        let claude = SkillInstaller::for_provider("claudecode", SkillConflict::Skip).unwrap();
        assert_eq!(claude.provider(), "claudecode");
        assert_eq!(claude.scan_dir(), ".claude/skills");
        assert_eq!(claude.layout(), SkillLayout::Directory);
        let codex = SkillInstaller::for_provider("codex", SkillConflict::Append).unwrap();
        assert_eq!(codex.provider(), "codex");
        assert_eq!(codex.scan_dir(), ".agents/skills");
        assert_eq!(codex.layout(), SkillLayout::Directory);
        let dsh = SkillInstaller::for_provider("dsh", SkillConflict::Replace).unwrap();
        assert_eq!(dsh.provider(), "dsh");
        assert_eq!(dsh.scan_dir(), ".dsh/skills");
        assert_eq!(dsh.layout(), SkillLayout::DirectoryOrFlat);
        assert_eq!(SkillLayout::Directory.as_str(), "directory");
        assert_eq!(SkillLayout::DirectoryOrFlat.as_str(), "directory_or_flat");
    }

    #[test]
    fn for_provider_rejects_unknown_names() {
        for provider in ["native", "native_v2", "", "ClaudeCode"] {
            let error =
                SkillInstaller::for_provider(provider, SkillConflict::Skip).expect_err("unknown");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert_eq!(
                error.message().as_ref(),
                format!(
                    "unknown skill provider '{provider}'; expected one of claudecode, codex, dsh"
                )
            );
        }
    }

    #[test]
    fn dsh_names_must_be_kebab_case() {
        let dsh = SkillInstaller::for_provider("dsh", SkillConflict::Skip).unwrap();
        for name in ["my-skill", "skill", "skill2", "a1-b2-c3", "2fa"] {
            dsh.validate_name(name).expect("kebab-case name");
        }
        for name in [
            "My_Skill",
            "My-Skill",
            "my--skill",
            "-leading",
            "trailing-",
            "under_score",
            "dotted.name",
            "",
            "..",
        ] {
            let error = dsh.validate_name(name).expect_err("non-kebab name");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert_eq!(
                error.message().as_ref(),
                format!("DSH skill names must be kebab-case: '{name}'")
            );
        }
    }

    #[test]
    fn claude_and_codex_keep_the_shared_portable_rule() {
        for provider in ["claudecode", "codex"] {
            let installer = SkillInstaller::for_provider(provider, SkillConflict::Skip).unwrap();
            for name in ["My.Skill", "my_skill", "skill-2", "A", "a.b_c-d"] {
                installer.validate_name(name).expect("portable name");
            }
            for name in ["bad name", "../escape", "", ".."] {
                let error = installer
                    .validate_name(name)
                    .expect_err("non-portable name");
                assert_eq!(
                    error.message().as_ref(),
                    format!("invalid portable skill name: '{name}'")
                );
            }
        }
    }

    #[test]
    fn directory_layouts_reject_a_flat_document_source() {
        let temp = tempdir().unwrap();
        let document = temp.path().join("flat-skill.md");
        std::fs::write(&document, "---\nname: flat-skill\n---\nbody\n").unwrap();
        let project = project_dir(&temp.path().join("project"));
        for provider in ["claudecode", "codex"] {
            let installer = SkillInstaller::for_provider(provider, SkillConflict::Skip).unwrap();
            let error = installer
                .install(&[SkillSource::new(&document)], Some(&project))
                .expect_err("flat source");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            assert_eq!(
                error.message().as_ref(),
                format!("the {provider} skill layout requires a SKILL.md directory bundle")
            );
        }
        assert!(!project.join(".claude").exists());
        assert!(!project.join(".agents").exists());
    }

    #[test]
    fn dsh_normalizes_a_flat_document_into_a_bundle_directory() {
        let temp = tempdir().unwrap();
        let document = temp.path().join("flat-skill.md");
        let text = "---\nname: flat-skill\ndescription: Flat\n---\nbody\n";
        std::fs::write(&document, text).unwrap();
        let project = project_dir(&temp.path().join("project"));
        let dsh = SkillInstaller::for_provider("dsh", SkillConflict::Skip).unwrap();
        let installed = dsh
            .install(&[SkillSource::new(&document)], Some(&project))
            .unwrap();
        let bundle = project.join(".dsh/skills/flat-skill");
        assert_eq!(installed, vec![bundle.clone()]);
        assert!(bundle.is_dir());
        assert_eq!(
            std::fs::read_to_string(bundle.join("SKILL.md")).unwrap(),
            text
        );
    }

    #[test]
    fn dsh_flat_document_without_front_matter_uses_the_file_stem() {
        let temp = tempdir().unwrap();
        let document = temp.path().join("plain-skill.md");
        std::fs::write(&document, "just instructions\n").unwrap();
        let project = project_dir(&temp.path().join("project"));
        let installed = install_for_provider(
            &[SkillSource::new(&document)],
            "dsh",
            Some(&project),
            SkillConflict::Skip,
        )
        .unwrap();
        assert_eq!(installed, vec![project.join(".dsh/skills/plain-skill")]);
        assert_eq!(
            std::fs::read_to_string(project.join(".dsh/skills/plain-skill/SKILL.md")).unwrap(),
            "just instructions\n"
        );
    }

    #[test]
    fn a_rejected_dsh_name_creates_no_discovery_directory() {
        let temp = tempdir().unwrap();
        let project = project_dir(&temp.path().join("project"));
        let dsh = SkillInstaller::for_provider("dsh", SkillConflict::Skip).unwrap();
        // A flat document named after a non-kebab name.
        let flat = temp.path().join("My_Skill.md");
        std::fs::write(&flat, "body\n").unwrap();
        let error = dsh
            .install(&[SkillSource::new(&flat)], Some(&project))
            .expect_err("non-kebab flat name");
        assert_eq!(
            error.message().as_ref(),
            "DSH skill names must be kebab-case: 'My_Skill'"
        );
        assert!(!project.join(".dsh").exists());
        // A bundle directory declaring a non-kebab front-matter name.
        let bundle = write_bundle(&temp.path().join("bundle"), "My_Skill");
        let error = dsh
            .install(&[SkillSource::new(&bundle)], Some(&project))
            .expect_err("non-kebab bundle name");
        assert_eq!(
            error.message().as_ref(),
            "DSH skill names must be kebab-case: 'My_Skill'"
        );
        assert!(!project.join(".dsh").exists());
    }

    #[test]
    fn install_with_no_sources_touches_nothing() {
        let temp = tempdir().unwrap();
        let project = project_dir(&temp.path().join("project"));
        for provider in ["claudecode", "codex", "dsh"] {
            let installer = SkillInstaller::for_provider(provider, SkillConflict::Skip).unwrap();
            assert_eq!(
                installer.install(&[], Some(&project)).unwrap(),
                Vec::<PathBuf>::new()
            );
        }
        assert_eq!(std::fs::read_dir(&project).unwrap().count(), 0);
    }

    #[test]
    fn install_for_provider_wraps_the_installer() {
        let temp = tempdir().unwrap();
        let library = temp.path().join("library");
        write_bundle(&library.join("one"), "one");
        write_bundle(&library.join("two"), "two");
        let project = project_dir(&temp.path().join("project"));
        let installed = install_for_provider(
            &[SkillSource::new(&library)],
            "codex",
            Some(&project),
            SkillConflict::Skip,
        )
        .unwrap();
        assert_eq!(
            installed,
            vec![
                project.join(".agents/skills/one"),
                project.join(".agents/skills/two"),
            ]
        );
        let error = install_for_provider(
            &[SkillSource::new(&library)],
            "native",
            Some(&project),
            SkillConflict::Skip,
        )
        .expect_err("unknown provider");
        assert_eq!(
            error.message().as_ref(),
            "unknown skill provider 'native'; expected one of claudecode, codex, dsh"
        );
        assert!(!project.join(".codex").exists());
    }
}
