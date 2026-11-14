#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for model-aware token accounting.
//!
//! These tests pin the offline contract of [`openjiuwen_context::tokenizers`]:
//! model-to-encoding mapping, `tiktoken` counting, registry resolution
//! (including the family boundary rule and ambiguity refusal), the selector
//! ladder, the native-tokenizer deferral, and the [`ContextConfig`] integration
//! that turns a model name into a working counter.
//!
//! Every test is offline and deterministic: the BPE vocabularies are embedded
//! in `tiktoken-rs`, and no test touches the network or a real tokenizer
//! artifact.

use openjiuwen_common::error::Error;
use openjiuwen_context::{
    load_native_tokenizer, ContextConfig, ModelEncoding, ModelTokenCounter, TokenCounterConfig,
    TokenCounterKind, TokenizerEngine, TokenizerMatchKind, TokenizerRegistry, TokenizerSelector,
    TokenizerSource, TokenizerSpec,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};

/// A spec that names one provider and model, with a derived tokenizer id.
fn spec(provider: &str, model: &str) -> TokenizerSpec {
    TokenizerSpec {
        provider: provider.to_owned(),
        model: model.to_owned(),
        tokenizer_id: Some(format!("{model}-tokenizer")),
        ..TokenizerSpec::default()
    }
}

/// A spec that names one provider and model family.
fn family_spec(provider: &str, family: &str) -> TokenizerSpec {
    TokenizerSpec {
        provider: provider.to_owned(),
        family: Some(family.to_owned()),
        tokenizer_id: Some(format!("{family}-tokenizer")),
        ..TokenizerSpec::default()
    }
}

#[test]
fn exact_model_names_map_to_their_encoding() {
    let cl100k = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-3.5",
        "gpt-35-turbo",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ];
    for model in cl100k {
        assert_eq!(
            ModelEncoding::from_model(model),
            Some(ModelEncoding::Cl100kBase),
            "{model} must use cl100k_base"
        );
    }

    let o200k = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.5",
        "gpt-5",
        "o1",
        "o3",
        "o4-mini",
    ];
    for model in o200k {
        assert_eq!(
            ModelEncoding::from_model(model),
            Some(ModelEncoding::O200kBase),
            "{model} must use o200k_base"
        );
    }
}

#[test]
fn deployment_prefixes_map_to_their_encoding() {
    let o200k = [
        "o1-preview",
        "o3-mini",
        "o4-mini-2025-04-16",
        "gpt-5-chat",
        "gpt-4.5-preview",
        "gpt-4.1-mini",
        "chatgpt-4o-latest",
        "gpt-4o-2024-08-06",
        "ft:gpt-4o-2024-08-06:my-org",
    ];
    for model in o200k {
        assert_eq!(
            ModelEncoding::from_model(model),
            Some(ModelEncoding::O200kBase),
            "{model} must use o200k_base"
        );
    }

    let cl100k = [
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-3.5-turbo-0301",
        "gpt-35-turbo-16k",
        "ft:gpt-4-0613:my-org",
        "ft:gpt-3.5-turbo-0613:my-org",
    ];
    for model in cl100k {
        assert_eq!(
            ModelEncoding::from_model(model),
            Some(ModelEncoding::Cl100kBase),
            "{model} must use cl100k_base"
        );
    }
}

#[test]
fn unknown_and_empty_models_have_no_encoding() {
    for model in [
        "",
        "   ",
        "claude-3-5-sonnet",
        "gemini-1.5",
        "unknown-model",
        "gpt-4.1x",
    ] {
        assert_eq!(ModelEncoding::from_model(model), None, "{model}");
    }
}

#[test]
fn model_matching_is_case_insensitive_and_trimmed() {
    assert_eq!(
        ModelEncoding::from_model("  GPT-4O  "),
        Some(ModelEncoding::O200kBase)
    );
    assert_eq!(
        ModelEncoding::from_model("GPT-4-0314"),
        Some(ModelEncoding::Cl100kBase)
    );
    assert_eq!(
        ModelEncoding::from_model("TeXt-EmBeDdInG-3-SmAlL"),
        Some(ModelEncoding::Cl100kBase)
    );
}

#[test]
fn encoding_names_and_serde_are_stable() {
    assert_eq!(ModelEncoding::Cl100kBase.name(), "cl100k_base");
    assert_eq!(ModelEncoding::O200kBase.name(), "o200k_base");
    assert_eq!(ModelEncoding::P50kBase.name(), "p50k_base");
    assert_eq!(ModelEncoding::R50kBase.name(), "r50k_base");

    assert_eq!(
        serde_json::to_string(&ModelEncoding::Cl100kBase).unwrap(),
        "\"cl100k_base\""
    );
    let restored: ModelEncoding = serde_json::from_str("\"o200k_base\"").unwrap();
    assert_eq!(restored, ModelEncoding::O200kBase);
}

#[test]
fn model_counter_counts_text_and_reports_metadata() {
    let counter = ModelTokenCounter::new("gpt-4o");
    assert_eq!(counter.model(), Some("gpt-4o"));
    assert_eq!(counter.encoding(), ModelEncoding::O200kBase);
    assert_eq!(counter.tokenizer(), "o200k_base");
    assert_eq!(counter.source(), TokenizerSource::Tiktoken);
    // Only a native tokenizer is not an estimate; a BPE vocabulary is a
    // close approximation of the provider's own tokenizer.
    assert!(counter.estimated());
    assert_eq!(counter.fallback_reason(), None);

    assert_eq!(counter.count_text(""), 0);
    assert!(counter.count_text("hello world") > 0);
    assert!(
        counter.count_text("a much longer sentence with more tokens") >= counter.count_text("a")
    );

    let measured = counter.measure("hello world");
    assert_eq!(measured.tokens, counter.count_text("hello world"));
    assert_eq!(measured.source, TokenizerSource::Tiktoken);
    assert!(measured.estimated);
    assert_eq!(measured.tokenizer.as_deref(), Some("o200k_base"));
    assert_eq!(measured.model.as_deref(), Some("gpt-4o"));
    assert_eq!(measured.fallback_reason, None);

    let messages = vec![
        BaseMessage::system("be brief"),
        BaseMessage::user("hello world"),
    ];
    let text_only = counter.count_text("be brief") + counter.count_text("hello world");
    assert!(
        counter.count_messages(&messages) > text_only,
        "message framing must be counted"
    );
    let measured_messages = counter.measure_messages(&messages);
    assert_eq!(measured_messages.tokens, counter.count_messages(&messages));
    assert_eq!(measured_messages.source, TokenizerSource::Tiktoken);
}

#[test]
fn model_counter_bpe_counts_differ_from_the_generic_estimate() {
    let text = "Tokenization isn't trivial: 1234567890 supercalifragilisticexpialidocious.";
    let approx = ApproxTokenCounter::with_chars_per_token(4);
    let gpt4 = ModelTokenCounter::new("gpt-4").count_text(text);
    let gpt4o = ModelTokenCounter::new("gpt-4o").count_text(text);

    assert_ne!(gpt4, approx.count_text(text));
    assert_ne!(gpt4o, approx.count_text(text));
    assert_ne!(gpt4, gpt4o);
}

#[test]
fn unknown_models_fall_back_to_cl100k_with_a_reason() {
    let counter = ModelTokenCounter::new("claude-3-5-sonnet");
    assert_eq!(counter.encoding(), ModelEncoding::Cl100kBase);
    assert_eq!(counter.source(), TokenizerSource::TiktokenFallback);
    assert_eq!(counter.fallback_reason(), Some("model_not_registered"));
    assert!(counter.estimated());
    assert!(counter.count_text("hello world") > 0);

    let default = ModelTokenCounter::default();
    assert_eq!(default.model(), None);
    assert_eq!(default.source(), TokenizerSource::TiktokenFallback);
    assert!(default.count_text("hello") > 0);
}

#[test]
fn registry_resolves_exact_matches_first() {
    let registry = TokenizerRegistry::new(vec![spec("openai", "gpt-4o"), spec("openai", "gpt-4")]);
    assert_eq!(registry.len(), 2);
    assert!(!registry.is_empty());

    let hit = registry.resolve("openai", "gpt-4o").unwrap();
    assert_eq!(hit.kind, TokenizerMatchKind::Exact);
    assert_eq!(hit.spec.model, "gpt-4o");
    assert_eq!(hit.spec.tokenizer_id.as_deref(), Some("gpt-4o-tokenizer"));

    // Case and surrounding whitespace are normalized on both sides.
    assert!(registry.resolve(" OpenAI ", " GPT-4O ").is_some());
    assert!(registry.resolve("openai", "gpt-4oX").is_none());
    // `gpt-4o-mini` is a family variant of `gpt-4o`, not an exact hit.
    let variant = registry.resolve("openai", "gpt-4o-mini").unwrap();
    assert_eq!(variant.kind, TokenizerMatchKind::Family);
}

#[test]
fn registry_family_match_obeys_the_boundary_rule() {
    let registry = TokenizerRegistry::new(vec![
        spec("openai", "gpt-4"),
        family_spec("ollama", "qwen2.5"),
    ]);

    let hit = registry.resolve("openai", "gpt-4-0314").unwrap();
    assert_eq!(hit.kind, TokenizerMatchKind::Family);
    assert_eq!(hit.spec.model, "gpt-4");

    // `gpt-4o` may not be treated as a `gpt-4` variant.
    assert!(registry.resolve("openai", "gpt-4o").is_none());

    let qwen = registry.resolve("ollama", "qwen2.5-72b").unwrap();
    assert_eq!(qwen.kind, TokenizerMatchKind::Family);
    assert_eq!(qwen.spec.family.as_deref(), Some("qwen2.5"));

    // A family matches itself, but not a word that merely starts with it.
    assert!(registry.resolve("ollama", "qwen2.5").is_some());
    assert!(registry.resolve("ollama", "qwen2.5pro").is_none());
}

#[test]
fn registry_requires_the_provider_to_match() {
    let registry = TokenizerRegistry::new(vec![
        spec("openai", "gpt-4"),
        family_spec("openai", "gpt-4o"),
    ]);
    assert!(registry.resolve("anthropic", "gpt-4-0314").is_none());
    assert!(registry.resolve("anthropic", "gpt-4o-2024").is_none());
    assert!(registry.resolve("openai", "gpt-4-0314").is_some());
}

#[test]
fn registry_refuses_to_guess_between_equal_longest_bases() {
    let registry = TokenizerRegistry::new(vec![
        spec("custom", "base"),
        TokenizerSpec {
            provider: "custom".to_owned(),
            model: "other".to_owned(),
            family: Some("base".to_owned()),
            ..TokenizerSpec::default()
        },
    ]);
    assert!(registry.resolve("custom", "base-v2").is_none());
}

#[test]
fn registry_duplicate_registration_is_last_wins() {
    let mut registry = TokenizerRegistry::new(Vec::new());
    registry.register(TokenizerSpec {
        tokenizer_id: Some("first".to_owned()),
        ..spec("openai", "gpt-4o")
    });
    registry.register(TokenizerSpec {
        tokenizer_id: Some("second".to_owned()),
        ..spec("openai", "gpt-4o")
    });

    let hit = registry.resolve("openai", "gpt-4o").unwrap();
    assert_eq!(hit.kind, TokenizerMatchKind::Exact);
    assert_eq!(hit.spec.tokenizer_id.as_deref(), Some("second"));

    // The same rule keeps a duplicate family registration unambiguous.
    let mut families = TokenizerRegistry::new(Vec::new());
    families.register(TokenizerSpec {
        tokenizer_id: Some("first".to_owned()),
        ..family_spec("openai", "gpt-4o")
    });
    families.register(TokenizerSpec {
        tokenizer_id: Some("second".to_owned()),
        ..family_spec("openai", "gpt-4o")
    });
    let hit = families.resolve("openai", "gpt-4o-2024").unwrap();
    assert_eq!(hit.kind, TokenizerMatchKind::Family);
    assert_eq!(hit.spec.tokenizer_id.as_deref(), Some("second"));
}

#[test]
fn specs_and_registry_round_trip_through_json() {
    let spec = TokenizerSpec {
        provider: "openai".to_owned(),
        model: "gpt-4o".to_owned(),
        family: Some("gpt-4o".to_owned()),
        engine: TokenizerEngine::Tiktoken,
        tokenizer_id: Some("o200k".to_owned()),
        source: Some("tiktoken".to_owned()),
        revision: Some("2024-08".to_owned()),
        artifact_path: Some("/tmp/tokenizer.json".to_owned()),
        sha256: Some("deadbeef".to_owned()),
        compatible_fallbacks: Vec::new(),
    };
    let registry = TokenizerRegistry::new(vec![spec.clone()]);
    let json = serde_json::to_string(&registry).unwrap();
    let restored: TokenizerRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(registry, restored);
    assert_eq!(restored.specs[0], spec);

    // Missing fields default, unknown fields are ignored.
    let partial: TokenizerSpec =
        serde_json::from_str(r#"{"provider": "p", "surprise": true}"#).unwrap();
    assert_eq!(partial.provider, "p");
    assert_eq!(partial.model, "");
    assert_eq!(partial.engine, TokenizerEngine::Auto);
    assert_eq!(partial.sha256, None);
}

#[test]
fn spec_validation_rejects_unidentifiable_entries() {
    assert!(matches!(
        TokenizerSpec::default().validate(),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        TokenizerSpec {
            provider: "   ".to_owned(),
            model: String::new(),
            ..TokenizerSpec::default()
        }
        .validate(),
        Err(Error::Config(_))
    ));
    assert!(spec("openai", "gpt-4o").validate().is_ok());
    assert!(family_spec("ollama", "qwen2.5").validate().is_ok());
}

#[test]
fn selector_prefers_an_explicit_spec_over_the_registry() {
    let explicit = TokenizerSpec {
        provider: "openai".to_owned(),
        model: "gpt-4o".to_owned(),
        tokenizer_id: Some("explicit".to_owned()),
        ..TokenizerSpec::default()
    };
    let registry = TokenizerRegistry::new(vec![TokenizerSpec {
        tokenizer_id: Some("registry".to_owned()),
        ..spec("openai", "gpt-4o")
    }]);

    let selected =
        TokenizerSelector::new("openai", "gpt-4o", Some(explicit), Some(registry)).select();
    assert_eq!(selected.tokenizer(), Some("explicit"));
    assert_eq!(selected.model(), Some("gpt-4o"));
    assert_eq!(selected.source(), TokenizerSource::Tiktoken);
    assert!(selected.count_text("hello world") > 0);
}

#[test]
fn selector_uses_an_exact_registry_entry() {
    let registry = TokenizerRegistry::new(vec![spec("openai", "gpt-4o")]);
    let selected = TokenizerSelector::new("openai", "gpt-4o", None, Some(registry)).select();
    assert_eq!(selected.tokenizer(), Some("gpt-4o-tokenizer"));
    assert_eq!(selected.source(), TokenizerSource::Tiktoken);
    assert!(selected.estimated());
}

#[test]
fn selector_reports_the_family_fallback_reason() {
    let registry = TokenizerRegistry::new(vec![TokenizerSpec {
        family: Some("gpt-4o".to_owned()),
        ..spec("openai", "gpt-4o")
    }]);
    let selected =
        TokenizerSelector::new("openai", "gpt-4o-2024-08-06", None, Some(registry)).select();

    assert_eq!(selected.source(), TokenizerSource::FamilyTokenizerFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some("model_variant_tokenizer_family")
    );
    assert!(selected.estimated());
    assert_eq!(selected.tokenizer(), Some("o200k_base"));
    assert!(selected.measure("hello world").tokens > 0);
}

#[test]
fn selector_estimates_unknown_models_with_a_reason() {
    let selected = TokenizerSelector::new("ollama", "qwen3-72b", None, None).select();
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
    assert_eq!(selected.fallback_reason(), Some("model_not_registered"));
    assert!(selected.estimated());
    assert_eq!(selected.tokenizer(), None);
    assert_eq!(selected.count_text("abcdef"), 2);

    let measured = selected.measure("abcdef");
    assert_eq!(measured.tokens, 2);
    assert_eq!(measured.source, TokenizerSource::StringLengthFallback);
    assert!(measured.estimated);
}

#[test]
fn selector_still_works_with_no_model_and_no_registry() {
    let selected = TokenizerSelector::new("", "", None, None).select();
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
    assert_eq!(selected.fallback_reason(), Some("tiktoken_disabled"));
    assert_eq!(selected.count_text("abc"), 1);
    assert_eq!(selected.count_text(""), 0);
}

#[test]
fn native_engine_reports_the_deferral_as_a_fallback() {
    let native = TokenizerSpec {
        engine: TokenizerEngine::Native,
        ..spec("openai", "gpt-4o")
    };
    let selected = TokenizerSelector::new("openai", "gpt-4o", Some(native), None).select();
    assert_eq!(selected.source(), TokenizerSource::TiktokenFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
    assert!(selected.estimated());
    assert!(selected.count_text("hello world") > 0);

    let unknown_native = TokenizerSpec {
        engine: TokenizerEngine::Native,
        ..spec("acme", "acme-1")
    };
    let selected = TokenizerSelector::new("acme", "acme-1", Some(unknown_native), None).select();
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
}

#[test]
fn loading_a_missing_native_tokenizer_is_a_typed_not_found() {
    // `tokenizer.json` artifacts are decoded in pure Rust; only a missing
    // artifact (or an unsupported model type) is refused, both typed.
    let error = load_native_tokenizer("/tmp/does-not-exist/tokenizer.json").unwrap_err();
    match error {
        Error::NotFound(message) => assert!(message.contains("tokenizer.json")),
        other => panic!("expected Error::NotFound, got {other:?}"),
    }
}

#[test]
fn config_builds_a_model_aware_counter() {
    let config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Model,
            chars_per_token: 4,
        },
        model_name: Some("gpt-4o".to_owned()),
        tokenizer_provider: Some("openai".to_owned()),
        tokenizer_registry: vec![spec("openai", "gpt-4o")],
        ..ContextConfig::default()
    };
    config.validate().unwrap();

    let counter = config.token_counter();
    assert_eq!(counter.count_text(""), 0);
    assert!(counter.count_text("hello world") > 0);

    let selected = TokenizerSelector::new("openai", "gpt-4o", None, None).select();
    assert_eq!(
        counter.count_text("hello world"),
        selected.count_text("hello world")
    );
}

#[test]
fn config_model_counter_works_without_a_model_or_registry() {
    let config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Model,
            chars_per_token: 4,
        },
        ..ContextConfig::default()
    };
    config.validate().unwrap();
    let counter = config.token_counter();
    // The selector's last rung is three characters per token.
    assert_eq!(counter.count_text("abcdef"), 2);
}

#[test]
fn config_rejects_an_invalid_registry_entry() {
    let config = ContextConfig {
        tokenizer_registry: vec![TokenizerSpec::default()],
        ..ContextConfig::default()
    };
    assert!(matches!(config.validate(), Err(Error::Config(_))));
}

#[test]
fn model_counter_kind_round_trips_and_existing_kinds_still_work() {
    assert_eq!(
        serde_json::to_string(&TokenCounterKind::Model).unwrap(),
        "\"model\""
    );
    let restored: TokenCounterKind = serde_json::from_str("\"model\"").unwrap();
    assert_eq!(restored, TokenCounterKind::Model);

    assert_eq!(
        ContextConfig::default().token_counter().count_text("abcd"),
        1
    );

    let characters = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 2,
        },
        ..ContextConfig::default()
    };
    assert_eq!(characters.token_counter().count_text("abcd"), 2);

    let tiktoken = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Tiktoken,
            chars_per_token: 4,
        },
        ..ContextConfig::default()
    }
    .token_counter();
    assert_eq!(tiktoken.count_text(""), 0);
    assert!(tiktoken.count_text("hello world") > 0);
}
