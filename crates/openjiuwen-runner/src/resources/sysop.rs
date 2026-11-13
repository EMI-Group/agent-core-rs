//! The system-operation registry: [`SysOperationMgr`].
//!
//! Ports Python `openjiuwen/core/runner/resources_manager/sys_operation_manager.py`
//! (v0.1.20): a manager keyed on the [`SysOperationCard`] id over the same generic
//! [`Registry`] the other resource kinds use, plus Python's
//! sandbox isolation-key guard (`_sandbox_key_owner_map`).
//!
//! Python registers a **live** `SysOperation` instance and hands that instance
//! back on every `get_sys_operation`; the Rust port keeps that contract (the
//! stored provider clones the registered `Arc`), unlike agents, workflows, and
//! teams, whose providers build a fresh instance per lookup.
//!
//! Two behaviours are worth naming:
//!
//! - **Isolation-key ownership.** A non-empty
//!   [`SysOperation::isolation_key_template`] (sandbox mode only) may belong to
//!   exactly one operation id. Registering a different id with the same
//!   template fails with [`openjiuwen_common::error::Error::Validation`], and
//!   removing an operation releases its template. The check and the
//!   registration happen under one manager-level lock, so two concurrent
//!   registrations cannot both claim the same template.
//! - **Tool specs are data, not tools.** [`SysOperationMgr::tool_specs`] returns
//!   the [`OperationToolSpec`] list an agent could be offered (ids
//!   `"<card>.<kind>.<method>"`). Projecting a spec into an
//!   `openjiuwen_foundation::tool::Tool` and publishing it in the tool registry
//!   is a caller concern: the specs are rendered from the operation parameter
//!   structs, and the dispatch back to the operation belongs to the host.
//!
//! ```no_run
//! use openjiuwen_runner::resources::sysop::SysOperationMgr;
//! use openjiuwen_sysop::{SysOperation, SysOperationCard};
//!
//! # fn example() -> openjiuwen_common::error::Result<()> {
//! let manager = SysOperationMgr::new();
//! manager.register(SysOperation::new(SysOperationCard::new("sys_op"))?)?;
//! let specs = manager.tool_specs("sys_op")?;
//! assert_eq!(specs.len(), 13);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::tool::OperationToolSpec;
use openjiuwen_sysop::{SysOperation, SysOperationCard};
use parking_lot::Mutex;

use super::registry::{Provider, RegisterPolicy, Registration, Registry};
use crate::error::{already_registered, invalid_id, not_found};

/// The registry noun used in error messages (`"sys_operation"`).
const KIND: &str = "sys_operation";

/// A factory for sys-operation services.
///
/// The built-in provider closes over the registered `Arc<SysOperation>` and
/// returns a clone of it, so every lookup yields the same service (Python
/// parity: `get_sys_operation` returns the instance that was added).
pub type SysOperationProvider = Provider<Arc<SysOperation>>;

/// Which operation id owns each non-empty sandbox isolation-key template.
///
/// Python parity: `SysOperationMgr._sandbox_key_owner_map`. The template is a
/// property of the built [`SysOperation`] (derived from its gateway
/// configuration), so the manager indexes it separately from the cards.
#[derive(Debug, Default)]
struct IsolationIndex {
    /// Isolation-key template -> owning operation id.
    owners: HashMap<String, String>,
}

impl IsolationIndex {
    /// Fails when `template` already belongs to a **different** operation.
    ///
    /// Re-registering the same id with its own template is legal (Python
    /// overwrites the entry with the same owner).
    fn ensure_free(&self, id: &str, template: &str) -> Result<()> {
        match self.owners.get(template) {
            Some(owner) if owner != id => Err(Error::Validation(format!(
                "sys_operation '{id}' cannot reuse isolation key template '{template}' \
                 already owned by '{owner}'"
            ))),
            _ => Ok(()),
        }
    }

    /// Records `id` as the owner of `template` (call [`Self::ensure_free`] first).
    fn claim(&mut self, id: &str, template: &str) {
        self.owners.insert(template.to_owned(), id.to_owned());
    }

    /// Drops every template owned by `id` (removal, and re-registration).
    fn release(&mut self, id: &str) {
        self.owners.retain(|_, owner| owner != id);
    }

    /// Drops every owner entry.
    fn clear(&mut self) {
        self.owners.clear();
    }
}

/// The runner's registry of [`SysOperation`] services, keyed on the card id.
///
/// Shared by the owning [`Runner`](crate::Runner) as part of
/// [`ResourceMgr`](super::ResourceMgr) wiring; every method takes `&self` and
/// mutates through the manager's locks.
///
/// Lock order is always *isolation index, then registry*: the index makes the
/// isolation-key check and the registration atomic, and nothing in the registry
/// ever calls back into the manager, so the order cannot invert.
#[derive(Debug)]
pub struct SysOperationMgr {
    /// The id-keyed entries (`kind` noun `"sys_operation"` in errors).
    registry: Registry<SysOperationCard, Arc<SysOperation>>,
    /// Isolation-key template ownership (Python `_sandbox_key_owner_map`).
    isolation: Mutex<IsolationIndex>,
}

impl SysOperationMgr {
    /// Builds an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Registry::with_kind(KIND),
            isolation: Mutex::new(IsolationIndex::default()),
        }
    }

    /// Registers `operation`, failing when its card id is already present or
    /// when its isolation-key template belongs to another operation.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id, a
    /// duplicate id (Python `SYS_OPERATION_MANAGER_PROCESS_ERROR`), or a
    /// template already owned by a different operation.
    pub fn register(&self, operation: SysOperation) -> Result<()> {
        self.register_with(operation, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `operation` under `policy`.
    ///
    /// [`RegisterPolicy::Keep`] keeps the existing entry (and its isolation-key
    /// ownership) and ignores the new instance entirely, so it never fails on a
    /// template conflict. [`RegisterPolicy::Replace`] swaps the instance, which
    /// may move the template: the previous template is released and the new one
    /// must be free.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid id, a
    /// duplicate id under [`RegisterPolicy::Fail`], or a template conflict.
    pub fn register_with(
        &self,
        operation: SysOperation,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        let card = operation.card().clone();
        let template = operation
            .isolation_key_template()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let operation = Arc::new(operation);
        self.register_entry(card, &operation, template.as_deref(), policy)
    }

    /// The shared registration path, with the isolation-key template injected.
    ///
    /// Sandbox mode is feature-gated, so the unit tests drive the guard through
    /// this seam with a fabricated template instead of building a sandbox
    /// service (see the `mod tests` comment).
    fn register_entry(
        &self,
        card: SysOperationCard,
        operation: &Arc<SysOperation>,
        template: Option<&str>,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        let id = card.id.clone();
        if id.trim().is_empty() {
            return Err(invalid_id(KIND));
        }

        let mut isolation = self.isolation.lock();
        if let Some(existing) = self.registry.card(&id) {
            match policy {
                RegisterPolicy::Fail => return Err(already_registered(KIND, &id)),
                RegisterPolicy::Keep => {
                    return Ok(Registration {
                        id,
                        name: existing.name,
                        replaced: false,
                    });
                }
                RegisterPolicy::Replace => {}
            }
        }

        // Validate before touching either store, so a rejected registration
        // leaves the existing ownership untouched.
        if let Some(template) = template {
            isolation.ensure_free(&id, template)?;
        }

        let stored = Arc::clone(operation);
        let provider: SysOperationProvider = Arc::new(move || Arc::clone(&stored));
        let registration = self.registry.register(card, provider, policy)?;

        isolation.release(&id);
        if let Some(template) = template {
            isolation.claim(&id, template);
        }
        Ok(registration)
    }

    /// Returns the service registered under `id` (the registered instance).
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Arc<SysOperation>> {
        self.registry.get(id)
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn card(&self, id: &str) -> Option<SysOperationCard> {
        self.registry.card(id)
    }

    /// Removes `id`, returning its card and releasing its isolation-key
    /// template.
    #[must_use]
    pub fn remove(&self, id: &str) -> Option<SysOperationCard> {
        let mut isolation = self.isolation.lock();
        let removed = self.registry.remove(id)?;
        isolation.release(id);
        Some(removed)
    }

    /// Returns the registered ids in registration order.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.registry.ids()
    }

    /// Returns clones of the registered cards in registration order.
    #[must_use]
    pub fn cards(&self) -> Vec<SysOperationCard> {
        self.registry.cards()
    }

    /// Returns the number of registered services.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Returns whether the manager holds no services.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    /// Returns the mutation counter (incremented on every successful mutation).
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.registry.revision()
    }

    /// Removes every service and releases every isolation-key template,
    /// returning how many were removed.
    pub fn clear(&self) -> usize {
        let mut isolation = self.isolation.lock();
        let removed = self.registry.clear();
        isolation.clear();
        removed
    }

    /// Removes every service and resets the revision counter to zero.
    ///
    /// Used by [`ResourceMgr::release`](super::ResourceMgr::release), which
    /// Python's `Runner::stop` calls to return the process to a pristine
    /// registry state.
    #[allow(
        dead_code,
        reason = "mirrors the other registries' reset seam; ResourceMgr::release is wired outside this file"
    )]
    pub(crate) fn reset(&self) -> usize {
        let mut isolation = self.isolation.lock();
        let removed = self.registry.reset();
        isolation.clear();
        removed
    }

    /// Returns the tool specs the service registered under `id` exposes.
    ///
    /// A local service contributes thirteen specs (ten filesystem, two shell,
    /// one code) whose ids are `"<card>.<kind>.<method>"`. Turning them into
    /// `Tool` implementations is the caller's job (see the module docs).
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::NotFound`] when `id` is not
    /// registered.
    pub fn tool_specs(&self, id: &str) -> Result<Vec<OperationToolSpec>> {
        let operation = self.registry.get(id).ok_or_else(|| not_found(KIND, id))?;
        Ok(operation.tool_specs())
    }
}

impl Default for SysOperationMgr {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// Builds a local-mode service; the local backends need no network.
    fn operation(id: &str) -> SysOperation {
        SysOperation::new(SysOperationCard::new(id)).expect("a local sys operation builds offline")
    }

    /// Registers `id` with a fabricated isolation-key template.
    ///
    /// Sandbox mode needs the `sandbox` feature plus a registered provider, so
    /// the guard is exercised through `register_entry` (the path the public
    /// API uses) with a template the caller supplies.
    fn register_with_template(
        manager: &SysOperationMgr,
        id: &str,
        template: Option<&str>,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        manager.register_entry(
            SysOperationCard::new(id),
            &Arc::new(operation(id)),
            template,
            policy,
        )
    }

    #[test]
    fn register_get_and_list_keep_one_instance() {
        let manager = SysOperationMgr::new();
        assert!(manager.is_empty());
        manager.register(operation("op-a")).unwrap();

        let service = manager.get("op-a").expect("registered");
        assert_eq!(service.card().id, "op-a");
        assert!(
            Arc::ptr_eq(&service, &manager.get("op-a").unwrap()),
            "the provider hands out the registered instance"
        );
        assert_eq!(manager.card("op-a").unwrap().id, "op-a");
        assert_eq!(manager.ids(), vec!["op-a"]);
        assert_eq!(manager.cards().len(), 1);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.revision(), 1);
        assert!(manager.get("missing").is_none());
        assert!(manager.card("missing").is_none());

        manager.register(operation("op-b")).unwrap();
        assert_eq!(manager.ids(), vec!["op-a", "op-b"]);
        assert_eq!(manager.revision(), 2);
    }

    // ports tests/unit_tests/core/runner/test_sys_operation_manager.py
    #[test]
    fn duplicate_registration_follows_the_policy() {
        let manager = SysOperationMgr::new();
        manager.register(operation("op-a")).unwrap();

        let error = manager.register(operation("op-a")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: sys_operation 'op-a' is already registered"
        );
        assert_eq!(manager.revision(), 1);

        let kept = manager
            .register_with(operation("op-a"), RegisterPolicy::Keep)
            .unwrap();
        assert!(!kept.replaced);
        assert_eq!(kept.id, "op-a");
        assert_eq!(manager.revision(), 1, "Keep is not a mutation");

        let replaced = manager
            .register_with(
                SysOperation::new(SysOperationCard::new("op-a").with_description("second"))
                    .unwrap(),
                RegisterPolicy::Replace,
            )
            .unwrap();
        assert!(replaced.replaced);
        assert_eq!(
            manager.card("op-a").unwrap().description.as_deref(),
            Some("second")
        );
        assert_eq!(manager.revision(), 2);
    }

    #[test]
    fn empty_and_whitespace_ids_are_rejected() {
        let manager = SysOperationMgr::new();
        // A `SysOperation` cannot itself be built from a blank card id, so the
        // manager-level guard is driven with a valid service behind a blank card.
        let service = Arc::new(operation("op-ghost"));
        for id in ["", "   ", "\t"] {
            let error = manager
                .register_entry(
                    SysOperationCard::new(id),
                    &service,
                    None,
                    RegisterPolicy::Fail,
                )
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: sys_operation id cannot be empty or whitespace only"
            );
        }
        assert!(manager.is_empty());
        assert_eq!(manager.revision(), 0);
    }

    #[test]
    fn remove_returns_the_card_and_releases_the_template() {
        let manager = SysOperationMgr::new();
        register_with_template(
            &manager,
            "op-a",
            Some("work_sandbox_aio_{session_id}"),
            RegisterPolicy::Fail,
        )
        .unwrap();
        assert_eq!(manager.revision(), 1);

        let removed = manager.remove("op-a").expect("registered");
        assert_eq!(removed.id, "op-a");
        assert!(manager.get("op-a").is_none());
        assert_eq!(manager.revision(), 2);
        assert!(manager.remove("op-a").is_none());
        assert_eq!(
            manager.revision(),
            2,
            "removing a missing id is no mutation"
        );

        register_with_template(
            &manager,
            "op-b",
            Some("work_sandbox_aio_{session_id}"),
            RegisterPolicy::Fail,
        )
        .expect("the removed operation released its isolation key template");
        assert_eq!(manager.ids(), vec!["op-b"]);
    }

    #[test]
    fn clear_and_reset_report_counts() {
        let manager = SysOperationMgr::new();
        manager.register(operation("op-a")).unwrap();
        manager.register(operation("op-b")).unwrap();

        assert_eq!(manager.clear(), 2);
        assert!(manager.is_empty());
        assert_eq!(manager.revision(), 3);
        assert_eq!(manager.clear(), 0);

        manager.register(operation("op-c")).unwrap();
        assert_eq!(manager.reset(), 1);
        assert!(manager.is_empty());
        assert_eq!(manager.revision(), 0);
    }

    #[test]
    fn tool_specs_cover_the_local_surface() {
        let manager = SysOperationMgr::new();
        manager.register(operation("op-a")).unwrap();

        let specs = manager.tool_specs("op-a").unwrap();
        assert_eq!(specs.len(), 13);
        assert_eq!(specs[0].tool_id, "op-a.fs.read_file");
        assert_eq!(specs[0].name, "read_file");
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.kind == openjiuwen_sysop::OperationKind::Fs)
                .count(),
            10
        );

        let error = manager.tool_specs("missing").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: sys_operation 'missing' is not registered"
        );
    }

    // ports the `_sandbox_key_owner_map` guard of sys_operation_manager.py
    #[test]
    fn two_operations_cannot_share_an_isolation_key_template() {
        let manager = SysOperationMgr::new();
        register_with_template(
            &manager,
            "op-a",
            Some("work_sandbox_aio_ep1"),
            RegisterPolicy::Fail,
        )
        .unwrap();

        let error = register_with_template(
            &manager,
            "op-b",
            Some("work_sandbox_aio_ep1"),
            RegisterPolicy::Fail,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: sys_operation 'op-b' cannot reuse isolation key template \
             'work_sandbox_aio_ep1' already owned by 'op-a'"
        );
        assert_eq!(manager.ids(), vec!["op-a"]);
        assert_eq!(manager.revision(), 1);

        register_with_template(
            &manager,
            "op-c",
            Some("work_sandbox_aio_ep2"),
            RegisterPolicy::Fail,
        )
        .expect("a different template is free");
    }

    #[test]
    fn the_same_operation_may_keep_or_replace_its_own_template() {
        let manager = SysOperationMgr::new();
        register_with_template(&manager, "op-a", Some("tpl-1"), RegisterPolicy::Fail).unwrap();

        let kept =
            register_with_template(&manager, "op-a", Some("tpl-1"), RegisterPolicy::Keep).unwrap();
        assert!(!kept.replaced, "Keep leaves the existing entry alone");
        assert_eq!(manager.revision(), 1);

        register_with_template(&manager, "op-a", Some("tpl-1"), RegisterPolicy::Replace)
            .expect("re-using the same template for the same id is legal");
        assert_eq!(manager.revision(), 2);

        // `Keep` on an already-registered id never claims the new instance's
        // template: the kept entry keeps its own. `Keep` on a *new* id still
        // registers it, so it does have to claim — and therefore conflicts.
        let error = register_with_template(&manager, "op-b", Some("tpl-1"), RegisterPolicy::Keep)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: sys_operation 'op-b' cannot reuse isolation key template 'tpl-1' \
             already owned by 'op-a'"
        );
        assert_eq!(manager.ids(), vec!["op-a"]);
        assert_eq!(manager.revision(), 2);
    }

    #[test]
    fn replace_moves_the_isolation_key_ownership() {
        let manager = SysOperationMgr::new();
        register_with_template(&manager, "op-a", Some("tpl-1"), RegisterPolicy::Fail).unwrap();

        register_with_template(&manager, "op-a", Some("tpl-2"), RegisterPolicy::Replace).unwrap();
        assert!(
            register_with_template(&manager, "op-b", Some("tpl-1"), RegisterPolicy::Fail).is_ok(),
            "the previous template was released"
        );

        let error = register_with_template(&manager, "op-c", Some("tpl-2"), RegisterPolicy::Fail)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("already owned by 'op-a'"),
            "{error}"
        );
    }

    #[test]
    fn replacing_onto_an_owned_template_leaves_the_owner_intact() {
        let manager = SysOperationMgr::new();
        register_with_template(&manager, "op-a", Some("tpl-1"), RegisterPolicy::Fail).unwrap();
        register_with_template(&manager, "op-b", Some("tpl-2"), RegisterPolicy::Fail).unwrap();

        let error =
            register_with_template(&manager, "op-b", Some("tpl-1"), RegisterPolicy::Replace)
                .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        // The rejected call neither registered nor released anything.
        assert_eq!(manager.revision(), 2);
        let error = register_with_template(&manager, "op-c", Some("tpl-2"), RegisterPolicy::Fail)
            .unwrap_err();
        assert!(
            error.to_string().contains("already owned by 'op-b'"),
            "{error}"
        );
    }

    #[test]
    fn default_builds_an_empty_manager() {
        let manager = SysOperationMgr::default();
        assert!(manager.is_empty());
        assert_eq!(manager.revision(), 0);
        assert!(format!("{manager:?}").contains("sys_operation"));
    }
}
