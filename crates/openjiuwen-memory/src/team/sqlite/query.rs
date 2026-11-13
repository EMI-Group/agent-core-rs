//! Dynamic `WHERE` construction shared by the `SQLite` team memory queries.
//!
//! Every predicate goes through [`Where`], which numbers its placeholders (`?1`,
//! `?2`, ...) as parameters are pushed. A fixed clause such as `LIMIT ?12` can therefore
//! be appended to a dynamic predicate list without ever mixing a bare `?` with a
//! numbered placeholder in one statement. Values are always bound, never formatted into
//! the SQL text, so no caller input reaches a statement as SQL.
//!
//! # Degradation of the text search
//!
//! `text_contains` becomes
//! `lower(content) LIKE ? ESCAPE '\' OR lower(COALESCE(title, '')) LIKE ? ESCAPE '\'`
//! with the needle lowercased and `%`/`_`/`\` escaped. `SQLite`'s `lower`/`LIKE` fold
//! ASCII only, while the in-memory backend folds full Unicode with
//! [`str::to_lowercase`]; that is a documented, deliberate difference (the alternative
//! would be a per-row Rust scan, which defeats the point of the SQL filter).
use super::schema::render_ts;
use crate::team::model::{TeamMemoryFilter, TeamMemoryKind};
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
/// One bound value of a dynamic statement.
///
/// `Real` exists because `importance` is a `REAL` column: binding `min_importance`
/// through textual affinity conversion would make the comparison depend on SQLite's
/// column-affinity rules instead of comparing the numbers directly.
#[derive(Debug, Clone)]
pub(super) enum Param {
    /// A text value (`TEXT` column, JSON payload, or rendered timestamp).
    Text(String),
    /// An integer value (`LIMIT`/`OFFSET`).
    Int(i64),
    /// A floating-point value (`importance`).
    Real(f64),
}
/// Accumulates `AND`-joined predicates and their numbered bind values.
#[derive(Debug, Clone, Default)]
pub(super) struct Where {
    /// The predicates, joined with `AND` by [`Where::sql`].
    clauses: Vec<String>,
    /// The bind values, in placeholder order (`?1` is `params[0]`).
    params: Vec<Param>,
}
impl Where {
    /// Builds an empty predicate set; [`Where::sql`] renders it as `1 = 1`.
    pub(super) fn new() -> Self {
        Self::default()
    }
    /// Pushes one value and returns its 1-based placeholder index.
    pub(super) fn push(&mut self, param: Param) -> usize {
        self.params.push(param);
        self.params.len()
    }
    /// Appends a predicate that refers to the placeholders it names itself.
    pub(super) fn and(&mut self, clause: impl Into<String>) {
        self.clauses.push(clause.into());
    }
    /// Appends `prefix` followed by the next placeholder, binding `param`.
    pub(super) fn bind(&mut self, prefix: &str, param: Param) {
        let index = self.push(param);
        self.and(format!("{prefix}?{index}"));
    }
    /// Appends `prefix` followed by one placeholder per value (`kind IN (?1, ?2)`).
    pub(super) fn bind_list(&mut self, prefix: &str, params: Vec<Param>) {
        let mut indices = Vec::with_capacity(params.len());
        for param in params {
            indices.push(format!("?{}", self.push(param)));
        }
        self.and(format!("{prefix}({})", indices.join(", ")));
    }
    /// Renders the predicates joined with `AND`; empty means "every row".
    pub(super) fn sql(&self) -> String {
        if self.clauses.is_empty() {
            "1 = 1".to_owned()
        } else {
            self.clauses.join(" AND ")
        }
    }
    /// The bind values, in placeholder order.
    pub(super) fn params(&self) -> &[Param] {
        &self.params
    }
}
/// Converts a row count into the `SQLite` integer a `LIMIT`/`OFFSET` binds.
pub(super) fn int_param(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}
/// Escapes a `LIKE` needle so `%`, `_`, and `\` match literally (with `ESCAPE '\'`).
fn escape_like(needle: &str) -> String {
    let mut escaped = String::with_capacity(needle.len());
    for character in needle.chars() {
        if matches!(character, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}
/// Appends the `any of these kinds` predicate when the list is not empty.
pub(super) fn push_kinds(where_clause: &mut Where, kinds: &[TeamMemoryKind]) {
    if kinds.is_empty() {
        return;
    }
    let values = kinds
        .iter()
        .map(|kind| Param::Text(kind.as_str().to_owned()))
        .collect();
    where_clause.bind_list("kind IN ", values);
}
/// Appends `(expires_at IS NULL OR expires_at > now)`, the live-record predicate.
pub(super) fn push_live(where_clause: &mut Where, now: DateTime<Utc>) {
    let index = where_clause.push(Param::Text(render_ts(now)));
    where_clause.and(format!("(expires_at IS NULL OR expires_at > ?{index})"));
}
/// Appends the substring predicate over the content and the title.
fn push_text(where_clause: &mut Where, needle: &str) {
    let pattern = format!("%{}%", escape_like(&needle.to_lowercase()));
    let content = where_clause.push(Param::Text(pattern.clone()));
    let title = where_clause.push(Param::Text(pattern));
    where_clause.and(format!(
        "(lower(content) LIKE ?{content} ESCAPE '\\' \
         OR lower(COALESCE(title, '')) LIKE ?{title} ESCAPE '\\')"
    ));
}
/// Appends the "every tag present, case-insensitively" predicates.
///
/// One `EXISTS` subquery per required tag: `json_each` unrolls the stored JSON array and
/// the comparison folds both sides with `lower`.
fn push_tags(where_clause: &mut Where, tags: &[String]) {
    for tag in tags {
        let index = where_clause.push(Param::Text(tag.clone()));
        where_clause.and(format!(
            "EXISTS (SELECT 1 FROM json_each(tags_json) AS tag \
             WHERE lower(tag.value) = lower(?{index}))"
        ));
    }
}
/// Builds the predicate of `filter`, applied by `list`, `clear`, and `stats`.
///
/// Every clause the filter can express in SQL is pushed here: team, member, session,
/// kinds, exact visibility, required tags, the case-insensitive text search, the
/// inclusive minimum importance, the exclusive time bounds, and — unless
/// [`TeamMemoryFilter::include_expired`] is set — the live-record predicate evaluated at
/// `now`. `limit`/`offset` are deliberately *not* pushed: `clear` and `stats` ignore
/// them, and `list` appends them itself.
pub(super) fn filter_where(filter: &TeamMemoryFilter, now: DateTime<Utc>) -> Result<Where> {
    let mut where_clause = Where::new();
    where_clause.bind("team_id = ", Param::Text(filter.team_id.clone()));
    if let Some(member) = &filter.member_id {
        where_clause.bind("member_id = ", Param::Text(member.clone()));
    }
    if let Some(session) = &filter.session_id {
        where_clause.bind("session_id = ", Param::Text(session.clone()));
    }
    push_kinds(&mut where_clause, &filter.kinds);
    if let Some(visibility) = &filter.visibility {
        let json = serde_json::to_string(visibility).map_err(Error::external)?;
        where_clause.bind("visibility_json = ", Param::Text(json));
    }
    push_tags(&mut where_clause, &filter.tags);
    if let Some(needle) = &filter.text_contains {
        push_text(&mut where_clause, needle);
    }
    if let Some(minimum) = filter.min_importance {
        where_clause.bind("importance >= ", Param::Real(f64::from(minimum)));
    }
    if let Some(after) = filter.created_after {
        where_clause.bind("created_at > ", Param::Text(render_ts(after)));
    }
    if let Some(before) = filter.created_before {
        where_clause.bind("created_at < ", Param::Text(render_ts(before)));
    }
    if let Some(after) = filter.updated_after {
        where_clause.bind("updated_at > ", Param::Text(render_ts(after)));
    }
    if let Some(before) = filter.updated_before {
        where_clause.bind("updated_at < ", Param::Text(render_ts(before)));
    }
    if !filter.include_expired {
        push_live(&mut where_clause, now);
    }
    Ok(where_clause)
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }
    #[test]
    fn empty_where_matches_every_row() {
        let where_clause = Where::new();
        assert_eq!(where_clause.sql(), "1 = 1");
        assert!(where_clause.params().is_empty());
    }
    #[test]
    fn placeholders_are_numbered_in_bind_order() {
        let mut where_clause = Where::new();
        where_clause.bind("team_id = ", Param::Text("review-team".to_owned()));
        where_clause.bind_list(
            "kind IN ",
            vec![
                Param::Text("decision".to_owned()),
                Param::Text("task".to_owned()),
            ],
        );
        assert_eq!(where_clause.sql(), "team_id = ?1 AND kind IN (?2, ?3)");
        assert_eq!(where_clause.params().len(), 3);
        let mut mixed = Where::new();
        mixed.bind("team_id = ", Param::Text("review-team".to_owned()));
        let limit = mixed.push(Param::Int(10));
        assert_eq!(mixed.sql(), "team_id = ?1");
        assert_eq!(limit, 2);
    }
    #[test]
    fn text_search_escapes_like_wildcards() {
        let mut where_clause = Where::new();
        push_text(&mut where_clause, "100%_done\\");
        let sql = where_clause.sql();
        assert!(sql.contains("ESCAPE '\\'"), "{sql}");
        assert!(sql.contains("lower(COALESCE(title, ''))"), "{sql}");
        let Param::Text(pattern) = &where_clause.params()[0] else {
            panic!("expected a text parameter")
        };
        assert_eq!(pattern, "%100\\%\\_done\\\\%");
        assert_eq!(where_clause.params().len(), 2);
    }
    #[test]
    fn tags_become_one_exists_clause_each() {
        let mut where_clause = Where::new();
        push_tags(
            &mut where_clause,
            &["storage".to_owned(), "sqlite".to_owned()],
        );
        let sql = where_clause.sql();
        assert_eq!(sql.matches("json_each(tags_json)").count(), 2);
        assert!(sql.contains("lower(?1)"), "{sql}");
        assert!(sql.contains("lower(?2)"), "{sql}");
    }
    #[test]
    fn filter_where_covers_every_clause() {
        let mut filter = TeamMemoryFilter::for_team("review-team")
            .member("alice")
            .kind(TeamMemoryKind::Decision)
            .with_limit(5);
        filter.session_id = Some("s1".to_owned());
        filter.tags = vec!["storage".to_owned()];
        filter.visibility = Some(crate::team::model::TeamVisibility::Private);
        filter.text_contains = Some("sqlite".to_owned());
        filter.min_importance = Some(0.8);
        filter.created_after = Some(at(1_700_000_000));
        filter.created_before = Some(at(1_700_000_100));
        filter.updated_after = Some(at(1_700_000_050));
        filter.updated_before = Some(at(1_700_000_200));
        let where_clause = filter_where(&filter, at(1_800_000_000)).unwrap();
        let sql = where_clause.sql();
        for expected in [
            "team_id = ?1",
            "member_id = ?2",
            "session_id = ?3",
            "kind IN (?4)",
            "visibility_json = ?5",
            "json_each(tags_json)",
            "LIKE",
            "importance >= ?9",
            "created_at > ?10",
            "created_at < ?11",
            "updated_at > ?12",
            "updated_at < ?13",
            "expires_at IS NULL OR expires_at > ?14",
        ] {
            assert!(sql.contains(expected), "missing `{expected}` in `{sql}`");
        }
        assert!(!sql.contains("LIMIT"));
        assert_eq!(where_clause.params().len(), 14);
        assert!(matches!(
            &where_clause.params()[4],
            Param::Text(json) if json == "\"private\""
        ));
    }
    #[test]
    fn included_expiry_drops_the_live_predicate() {
        let mut filter = TeamMemoryFilter::for_team("review-team");
        filter.include_expired = true;
        let where_clause = filter_where(&filter, at(1_800_000_000)).unwrap();
        assert_eq!(where_clause.sql(), "team_id = ?1");
    }
    #[test]
    fn int_param_saturates() {
        assert_eq!(int_param(0), 0);
        assert_eq!(int_param(7), 7);
        assert_eq!(int_param(usize::MAX), i64::MAX);
    }
}
