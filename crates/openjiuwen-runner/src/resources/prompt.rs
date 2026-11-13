//! Prompt templates keyed on an id: [`PromptMgr`].
//!
//! Ports Python `openjiuwen/core/runner/resources_manager/prompt_manager.py`
//! (`PromptMgr.add_prompt` / `add_prompts` / `remove_prompt` / `get_prompt`).
//! Python stores `(template_id, PromptTemplate)` pairs in a thread-safe dict and
//! keeps no card object; [`Registry`] needs a [`ResourceCard`] to key, order,
//! and serialize entries, so [`PromptCard`] carries the bare id as its own name.
//!
//! Python's `None` guards have no Rust equivalent here: a template is a value
//! (`PromptTemplate`), not an `Option`, and the id is validated before it
//! reaches the registry, so Python's `ValueError('template_id is invalid, can
//! not be None')` becomes
//! [`openjiuwen_common::error::Error::Validation`] from the registry itself.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::PromptTemplate;
use serde::{Deserialize, Serialize};

use super::registry::{Provider, RegisterPolicy, Registry, ResourceCard};

/// The registry identity of one stored prompt template.
///
/// Python keys prompts on a bare id string (`PromptMgr.add_prompt(template_id,
/// template)`) in a `ThreadSafeDict`, with no card to hold. [`Registry`] needs
/// a [`ResourceCard`], so this card is the Rust identity carrier: the id is
/// Python's `template_id` and the name is the same string, because a prompt has
/// no separate display name (the template's own `name` field belongs to the
/// template, not to the registry entry).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCard {
    /// The registry key (Python's `template_id`).
    pub id: String,
}

impl PromptCard {
    /// Builds a card for `id`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl ResourceCard for PromptCard {
    fn id(&self) -> &str {
        &self.id
    }

    /// Returns the id: a prompt entry has no name of its own.
    fn name(&self) -> &str {
        &self.id
    }
}

/// The registry of prompt templates a host publishes.
///
/// Python parity: `PromptMgr`, including its overwrite semantics — Python's
/// `add_prompt` assigns into the dict, so re-adding an id silently replaces the
/// stored template. The Rust port keeps that contract by registering under
/// [`RegisterPolicy::Replace`]; there is no duplicate-id error on this path.
///
/// Errors name the resource `"prompt"`, for example
/// `validation: prompt id cannot be empty or whitespace only`.
///
/// ```
/// use openjiuwen_foundation::prompt::PromptTemplate;
/// use openjiuwen_runner::resources::prompt::PromptMgr;
///
/// let prompts = PromptMgr::new();
/// prompts
///     .add_prompt("greeting", PromptTemplate::new("hello {{name}}"))
///     .unwrap();
/// assert!(prompts.prompt("greeting").is_some());
/// assert!(prompts.prompt("missing").is_none());
/// ```
#[derive(Debug)]
pub struct PromptMgr {
    /// Stored templates, keyed on [`PromptCard::id`].
    prompts: Registry<PromptCard, PromptTemplate>,
}

impl PromptMgr {
    /// Builds an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prompts: Registry::with_kind("prompt"),
        }
    }

    /// Stores `template` under `id`, replacing any template already there.
    ///
    /// Python parity: `PromptMgr.add_prompt` overwrites silently, so a repeated
    /// id is `Ok` and the previous template is dropped, never an error.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only id; the existing entry is untouched in that case.
    pub fn add_prompt(&self, id: impl Into<String>, template: PromptTemplate) -> Result<()> {
        // The stored provider clones the template, so `get` hands back a copy of
        // what was registered and the registry keeps a single owner.
        let provider: Provider<PromptTemplate> = Arc::new(move || template.clone());
        self.prompts
            .register(PromptCard::new(id), provider, RegisterPolicy::Replace)
            .map(|_| ())
    }

    /// Stores every `(id, template)` pair in iteration order.
    ///
    /// Python parity: `PromptMgr.add_prompts` loops over `add_prompt`, so an
    /// invalid id aborts the batch and the entries added before it stay. Each
    /// pair overwrites an existing template with the same id.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for the first empty or
    /// whitespace-only id; earlier pairs are already stored.
    pub fn add_prompts(
        &self,
        prompts: impl IntoIterator<Item = (String, PromptTemplate)>,
    ) -> Result<()> {
        for (id, template) in prompts {
            self.add_prompt(id, template)?;
        }
        Ok(())
    }

    /// Returns the template stored under `id` (Python's `get_prompt`).
    ///
    /// A missing id returns `None`; Python returned `None` from the dict lookup
    /// while raising `ValueError` only for a `None` id, which cannot occur here
    /// as the id is a `&str`.
    #[must_use]
    pub fn prompt(&self, id: &str) -> Option<PromptTemplate> {
        self.prompts.get(id)
    }

    /// Removes `id` and returns its template; a missing id returns `None`.
    ///
    /// Python parity: `dict.pop(template_id, None)` returns the removed value.
    /// [`Registry::remove`] reports the removed card only, so the template is
    /// read first and then dropped from the registry; the two registry calls
    /// each take their own lock, so a concurrent removal can win the race and
    /// this call reports `None` rather than a stale template.
    #[must_use]
    pub fn remove_prompt(&self, id: &str) -> Option<PromptTemplate> {
        let template = self.prompts.get(id)?;
        self.prompts.remove(id).map(|_| template)
    }

    /// Returns the stored ids in insertion order.
    #[must_use]
    pub fn prompt_ids(&self) -> Vec<String> {
        self.prompts.ids()
    }

    /// Returns the number of stored templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Returns whether no template is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// Returns the mutation counter, incremented on every successful mutation.
    ///
    /// A host caching a projection of this registry compares the revision to
    /// decide whether to rebuild.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.prompts.revision()
    }

    /// Removes every template and returns how many were removed.
    ///
    /// A non-empty clear counts as one mutation; clearing an empty registry is a
    /// no-op and leaves the revision untouched.
    pub fn clear(&self) -> usize {
        self.prompts.clear()
    }

    /// Removes every template and resets the revision counter to zero.
    ///
    /// Python parity: [`ResourceMgr::release`](super::ResourceMgr::release) calls
    /// it to empty the prompt manager too.
    pub(crate) fn reset(&self) -> usize {
        self.prompts.reset()
    }
}

impl Default for PromptMgr {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    fn template(text: &str) -> PromptTemplate {
        PromptTemplate::new(text)
    }

    #[test]
    fn card_round_trips_through_json() {
        let card = PromptCard::new("greeting");
        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json, serde_json::json!({"id": "greeting"}));

        let restored = serde_json::from_str::<PromptCard>(r#"{"id": "greeting", "extra": 1}"#)
            .expect("unknown fields are ignored");
        assert_eq!(restored, card);
        assert_eq!(ResourceCard::name(&restored), "greeting");
    }

    #[test]
    fn added_prompt_resolves_by_id() {
        let prompts = PromptMgr::new();
        prompts.add_prompt("greeting", template("hello")).unwrap();

        assert_eq!(prompts.prompt("greeting"), Some(template("hello")));
        assert!(prompts.prompt("missing").is_none());
        assert_eq!(prompts.prompt_ids(), vec!["greeting".to_owned()]);
        assert_eq!(prompts.len(), 1);
        assert!(!prompts.is_empty());
    }

    // ports tests/unit_tests/core/runner/test_prompt_manager.py (add overwrites)
    #[test]
    fn add_prompt_overwrites_silently() {
        let prompts = PromptMgr::new();
        prompts.add_prompt("greeting", template("first")).unwrap();
        prompts.add_prompt("greeting", template("second")).unwrap();

        assert_eq!(prompts.prompt("greeting"), Some(template("second")));
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts.revision(), 2, "the overwrite counts as a mutation");
    }

    #[test]
    fn add_prompts_adds_in_order_and_stops_at_the_first_invalid_id() {
        let prompts = PromptMgr::new();
        prompts
            .add_prompts(vec![
                ("a".to_owned(), template("first")),
                ("b".to_owned(), template("second")),
            ])
            .unwrap();
        assert_eq!(prompts.prompt_ids(), vec!["a".to_owned(), "b".to_owned()]);

        let error = prompts
            .add_prompts(vec![
                ("c".to_owned(), template("third")),
                ("  ".to_owned(), template("ghost")),
                ("d".to_owned(), template("never")),
            ])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: prompt id cannot be empty or whitespace only"
        );
        assert_eq!(
            prompts.prompt_ids(),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            "pairs before the invalid id stay stored"
        );
        assert!(prompts.prompt("d").is_none());
    }

    #[test]
    fn removal_returns_the_template_and_drops_the_lookup() {
        let prompts = PromptMgr::new();
        prompts.add_prompt("a", template("first")).unwrap();
        prompts.add_prompt("b", template("second")).unwrap();

        assert_eq!(prompts.remove_prompt("a"), Some(template("first")));
        assert!(prompts.prompt("a").is_none());
        assert_eq!(prompts.prompt_ids(), vec!["b".to_owned()]);
        assert!(prompts.remove_prompt("a").is_none());
    }

    #[test]
    fn empty_and_whitespace_ids_are_rejected() {
        let prompts = PromptMgr::new();
        for id in ["", "   ", "\t\n"] {
            let error = prompts.add_prompt(id, template("ghost")).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: prompt id cannot be empty or whitespace only"
            );
        }
        assert!(prompts.is_empty());
        assert_eq!(prompts.revision(), 0);
        assert_eq!(
            PromptMgr::default().len(),
            0,
            "Default builds an empty manager"
        );
    }

    #[test]
    fn revision_counts_mutations_and_reset_zeroes_it() {
        let prompts = PromptMgr::new();
        assert_eq!(prompts.revision(), 0);

        prompts.add_prompt("a", template("first")).unwrap();
        assert_eq!(prompts.revision(), 1);

        prompts.add_prompt("b", template("second")).unwrap();
        assert_eq!(prompts.revision(), 2);

        assert_eq!(prompts.clear(), 2);
        assert_eq!(prompts.revision(), 3);
        assert_eq!(prompts.clear(), 0);
        assert_eq!(
            prompts.revision(),
            3,
            "clearing an empty registry is a no-op"
        );

        prompts.add_prompt("c", template("third")).unwrap();
        assert_eq!(prompts.reset(), 1);
        assert!(prompts.is_empty());
        assert_eq!(prompts.revision(), 0);
    }
}
