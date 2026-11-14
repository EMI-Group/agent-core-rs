//! Unit tests of strategy-name parsing and the deterministic ladder.

use super::*;

#[test]
fn strategy_names_parse_tolerantly() {
    assert_eq!(
        CompressionStrategy::parse(" Dialogue-Summary ").strategy,
        CompressionStrategy::DialogueSummary
    );
    assert_eq!(
        CompressionStrategy::parse("`current_round`").strategy,
        CompressionStrategy::CurrentRound
    );
    assert_eq!(
        CompressionStrategy::parse("NOOP").strategy,
        CompressionStrategy::Noop
    );
    let unknown = CompressionStrategy::parse("summarise everything");
    assert_eq!(unknown.strategy, CompressionStrategy::Noop);
    assert!(unknown
        .reason
        .as_deref()
        .is_some_and(|reason| reason.contains("unknown compression strategy")));
    assert!(CompressionStrategy::parse("").reason.is_some());
    assert_eq!(
        serde_json::to_string(&CompressionStrategy::RoundLevel).unwrap(),
        "\"round_level\""
    );
}

#[test]
fn the_ladder_is_total_and_measured() {
    let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig::default());
    let empty = strategy.deterministic(&AdaptiveContext::default());
    assert_eq!(empty.strategy, CompressionStrategy::Noop);
    assert_ne!(empty.reason, "");

    let dominant = AdaptiveContext {
        total_tokens: 1_000,
        messages: 4,
        completed_rounds: 2,
        completed_round_tokens: 900,
        ..AdaptiveContext::default()
    };
    assert_eq!(
        strategy.deterministic(&dominant).strategy,
        CompressionStrategy::DialogueSummary
    );

    let over_budget = AdaptiveContext {
        total_tokens: 1_000,
        messages: 2,
        budget_tokens: 900,
        ..AdaptiveContext::default()
    };
    assert_eq!(
        strategy.deterministic(&over_budget).strategy,
        CompressionStrategy::FullCompact
    );

    let nothing = AdaptiveContext {
        total_tokens: 1_000,
        messages: 3,
        completed_rounds: 1,
        completed_round_tokens: 10,
        budget_tokens: 5_000,
        ..AdaptiveContext::default()
    };
    assert_eq!(
        strategy.deterministic(&nothing).strategy,
        CompressionStrategy::Noop
    );
}
