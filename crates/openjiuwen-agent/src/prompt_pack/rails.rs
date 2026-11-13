//! Declarative rail specs of a prompt pack.
//!
//! A pack names the kernel rails it wants in its `rails` list; [`RailSpec`] is
//! that list's item type. [`RailSpec::build`] constructs the rail a spec
//! describes from the host-supplied [`PromptPackContext`], and returns `None`
//! when the spec needs a host object the context does not carry, so
//! [`PromptPack::apply_to_with`](super::PromptPack::apply_to_with) can report
//! the spec as deferred instead of dropping it silently.
use super::pack::PromptPackContext;
use crate::rails::{AgentRail, AskUserRail, ConfirmInterruptRail, GuardrailRail, ModelBackupRail};
use crate::skills::SkillRail;
use openjiuwen_security::GuardrailPipeline;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One rail a pack asks the host to install.
///
/// The wire shape is internally tagged: `type` selects the variant and the
/// remaining keys configure it (`{ type: skill, budget: 2000 }`). Unknown keys
/// are tolerated. `guardrail`, `confirm_interrupt`, and `ask_user` are built
/// from the crate defaults; `skill`, `model_backup`, and `custom` may need host
/// objects and are deferred when they are missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RailSpec {
    /// Scan user input, model output, and tool traffic through a
    /// [`GuardrailPipeline`].
    ///
    /// A pack cannot describe guardrail rules, so this builds
    /// [`GuardrailRail::new`] over an empty pipeline; it is a placeholder a
    /// host replaces with a configured rail.
    Guardrail,
    /// Ask the user to approve every tool call before it runs.
    ConfirmInterrupt,
    /// Ask the user for input and use the answer as the tool result.
    AskUser,
    /// Advertise the skill catalog in the model window.
    Skill {
        /// Character budget of the rendered catalog; the rail default when
        /// omitted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        budget: Option<usize>,
    },
    /// Try the supplied backup models in order when the primary model fails.
    ///
    /// Declarative only: [`PromptPackContext::models`] supplies the models, so
    /// the spec is deferred when the context carries none.
    ModelBackup,
    /// A host-defined rail; always deferred, because only the host can build
    /// it. `name` is the name the host is expected to register.
    Custom {
        /// Name the host registers the rail under.
        name: String,
        /// Optional dispatch priority the host should apply.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        priority: Option<i32>,
        /// Free-form options passed through to the host.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        options: BTreeMap<String, Value>,
    },
}

impl RailSpec {
    /// The wire name of the variant (`"guardrail"`, `"confirm_interrupt"`, ...).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Guardrail => "guardrail",
            Self::ConfirmInterrupt => "confirm_interrupt",
            Self::AskUser => "ask_user",
            Self::Skill { .. } => "skill",
            Self::ModelBackup => "model_backup",
            Self::Custom { .. } => "custom",
        }
    }

    /// Builds the rail this spec describes from `ctx`.
    ///
    /// Returns the constructed rail, or `None` when the spec cannot be built
    /// from the context alone: `skill` without a
    /// [`PromptPackContext::skills`] manager, `model_backup` without a
    /// [`PromptPackContext::models`] backup model, and every `custom` spec.
    #[must_use]
    pub fn build(&self, ctx: &PromptPackContext) -> Option<Arc<dyn AgentRail>> {
        match self {
            Self::Guardrail => Some(Arc::new(GuardrailRail::new(Arc::new(
                GuardrailPipeline::default(),
            )))),
            Self::ConfirmInterrupt => Some(Arc::new(ConfirmInterruptRail::new())),
            Self::AskUser => Some(Arc::new(AskUserRail::new())),
            Self::Skill { budget } => {
                let manager = ctx.skills.clone()?;
                let rail = SkillRail::new(manager);
                Some(Arc::new(match budget {
                    Some(budget) => rail.with_budget(*budget),
                    None => rail,
                }))
            }
            Self::ModelBackup => {
                if ctx.models.is_empty() {
                    None
                } else {
                    Some(Arc::new(ModelBackupRail::new(ctx.models.clone())))
                }
            }
            Self::Custom { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{Skill, SkillManager};
    use openjiuwen_foundation::Model;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::json;

    fn specs(yaml: &str) -> Vec<RailSpec> {
        serde_yaml_ng::from_str(yaml).expect("the rail specs parse")
    }

    fn shared(model: ScriptedModel) -> Arc<dyn Model> {
        Arc::new(model)
    }

    #[test]
    fn every_variant_parses_from_its_tagged_wire_shape() {
        let specs = specs(
            r"
- { type: guardrail }
- { type: confirm_interrupt }
- { type: ask_user }
- { type: skill, budget: 2000 }
- { type: model_backup }
- { type: custom, name: mine, priority: 7, options: { strict: true } }
",
        );
        assert_eq!(specs.len(), 6);
        assert_eq!(specs[0].kind(), "guardrail");
        assert_eq!(specs[1].kind(), "confirm_interrupt");
        assert_eq!(specs[2].kind(), "ask_user");
        assert_eq!(specs[3], RailSpec::Skill { budget: Some(2000) });
        assert_eq!(specs[4], RailSpec::ModelBackup);
        assert_eq!(
            specs[5],
            RailSpec::Custom {
                name: "mine".to_owned(),
                priority: Some(7),
                options: BTreeMap::from([("strict".to_owned(), json!(true))]),
            }
        );
    }

    #[test]
    fn spec_round_trips_through_yaml() {
        let specs = specs(
            r"
- { type: skill }
- { type: custom, name: mine }
",
        );
        let yaml = serde_yaml_ng::to_string(&specs).expect("serialize");
        let back: Vec<RailSpec> = serde_yaml_ng::from_str(&yaml).expect("deserialize");
        assert_eq!(back, specs);
        assert_eq!(back[0], RailSpec::Skill { budget: None });
    }

    #[test]
    fn an_unknown_type_is_a_parse_error() {
        let error = serde_yaml_ng::from_str::<Vec<RailSpec>>("- { type: telepathy }").unwrap_err();
        assert!(error.to_string().contains("telepathy"), "{error}");
    }

    #[test]
    fn kernel_rails_are_built_from_the_defaults() {
        let ctx = PromptPackContext::new();
        assert_eq!(
            RailSpec::Guardrail
                .build(&ctx)
                .map(|rail| rail.name().to_owned()),
            Some("guardrail".to_owned())
        );
        assert_eq!(
            RailSpec::ConfirmInterrupt
                .build(&ctx)
                .map(|rail| rail.name().to_owned()),
            Some("confirm_interrupt".to_owned())
        );
        assert_eq!(
            RailSpec::AskUser
                .build(&ctx)
                .map(|rail| rail.name().to_owned()),
            Some("ask_user".to_owned())
        );
    }

    #[test]
    fn skill_needs_a_manager_and_honours_the_budget() {
        let spec = RailSpec::Skill { budget: Some(123) };
        assert!(spec.build(&PromptPackContext::new()).is_none());

        let ctx = PromptPackContext::new().with_skills(SkillManager::with_skills([Skill::new(
            "deploy",
            "deploy it",
        )]));
        let rail = spec.build(&ctx).expect("the manager is supplied");
        assert_eq!(rail.name(), "skill");
    }

    #[test]
    fn model_backup_needs_at_least_one_model() {
        let spec = RailSpec::ModelBackup;
        assert!(spec.build(&PromptPackContext::new()).is_none());

        let ctx = PromptPackContext::new().with_models([shared(ScriptedModel::text("backup"))]);
        let rail = spec.build(&ctx).expect("a backup model is supplied");
        assert_eq!(rail.name(), "model_backup");
    }

    #[test]
    fn custom_specs_are_always_deferred() {
        let spec = RailSpec::Custom {
            name: "mine".to_owned(),
            priority: None,
            options: BTreeMap::new(),
        };
        assert!(spec.build(&PromptPackContext::new()).is_none());
        let ctx = PromptPackContext::new()
            .with_skills(SkillManager::with_skills([Skill::new("a", "b")]))
            .with_models([shared(ScriptedModel::text("backup"))]);
        assert!(spec.build(&ctx).is_none());
    }
}
