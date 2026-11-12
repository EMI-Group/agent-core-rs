//! Deterministic name normalization and fuzzy name comparison.
//!
//! Ports the deterministic half of Python `openjiuwen.symphony.shared.naming`: names
//! are reduced to a stable `snake_case`-like key, then compared with a plain
//! Levenshtein distance. Both helpers are total and allocation-only — no locale,
//! clock, or hash-order dependence — so they can be used in ranking tie-breaks.

/// Normalizes `name` to a comparable key: lowercase alphanumerics joined by single
/// underscores, with leading and trailing underscores removed.
///
/// Every non-alphanumeric character is a separator. A camel-case boundary is *not*
/// split, so `readFile` becomes `readfile`; the key is a canonical spelling, not a
/// tokenizer.
///
/// ```
/// use openjiuwen_symphony::model::normalize_name_key;
///
/// assert_eq!(normalize_name_key("  Read_File (v2)!! "), "read_file_v2");
/// assert_eq!(normalize_name_key(""), "");
/// ```
#[must_use]
pub fn normalize_name_key(name: &str) -> String {
    let mut key = String::with_capacity(name.len());
    // Starts as `true` so leading separators are dropped instead of emitted.
    let mut pending_separator = false;
    for character in name.chars() {
        if character.is_alphanumeric() {
            if pending_separator && !key.is_empty() {
                key.push('_');
            }
            pending_separator = false;
            key.extend(character.to_lowercase());
        } else {
            pending_separator = true;
        }
    }
    key
}

/// The Levenshtein distance between the normalized keys of `left` and `right`.
///
/// The distance is computed over Unicode scalar values with the usual
/// insert/delete/substitute dynamic program, so it is symmetric and
/// `fuzzy_name_distance(value, value)` is always `0`.
///
/// ```
/// use openjiuwen_symphony::model::fuzzy_name_distance;
///
/// assert_eq!(fuzzy_name_distance("Read File", "read_file"), 0);
/// assert_eq!(fuzzy_name_distance("kitten", "sitting"), 3);
/// ```
#[must_use]
pub fn fuzzy_name_distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = normalize_name_key(left).chars().collect();
    let right: Vec<char> = normalize_name_key(right).chars().collect();
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0_usize; right.len() + 1];
    for (row, left_char) in left.iter().enumerate() {
        current[0] = row + 1;
        for (column, right_char) in right.iter().enumerate() {
            let substitution = previous[column] + usize::from(left_char != right_char);
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// The similarity of two names in `[0.0, 1.0]`, derived from their normalized keys.
///
/// It is `1 - distance / max(len)`; two empty keys are identical and score `1.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // names are far shorter than 2^53 characters
pub fn fuzzy_name_similarity(a: &str, b: &str) -> f64 {
    let left = normalize_name_key(a);
    let right = normalize_name_key(b);
    let longest = left.chars().count().max(right.chars().count());
    if longest == 0 {
        return 1.0;
    }
    let distance = fuzzy_name_distance(&left, &right);
    1.0 - (distance as f64) / (longest as f64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn keys_lowercase_collapse_and_trim_separators() {
        assert_eq!(normalize_name_key("Read__File"), "read_file");
        assert_eq!(normalize_name_key("__read_file__"), "read_file");
        assert_eq!(normalize_name_key("a--b..c"), "a_b_c");
        assert_eq!(normalize_name_key("  "), "");
        assert_eq!(normalize_name_key("!!!"), "");
        assert_eq!(normalize_name_key("Über-Ärger"), "über_ärger");
    }

    #[test]
    fn distance_is_symmetric_and_zero_for_equal_keys() {
        let pairs = [
            ("kitten", "sitting"),
            ("read file", "read_file"),
            ("", "abc"),
        ];
        for (left, right) in pairs {
            assert_eq!(
                fuzzy_name_distance(left, right),
                fuzzy_name_distance(right, left)
            );
            assert_eq!(fuzzy_name_distance(left, left), 0);
            assert_eq!(fuzzy_name_distance(right, right), 0);
        }
        assert_eq!(fuzzy_name_distance("", ""), 0);
        assert_eq!(fuzzy_name_distance("", "abc"), 3);
    }

    #[test]
    fn distance_counts_one_edit_per_operation() {
        assert_eq!(fuzzy_name_distance("abc", "abc"), 0);
        assert_eq!(fuzzy_name_distance("abc", "abd"), 1);
        assert_eq!(fuzzy_name_distance("abc", "ab"), 1);
        assert_eq!(fuzzy_name_distance("abc", "xabc"), 1);
    }

    #[test]
    fn similarity_is_bounded_and_handles_empty_keys() {
        assert!((fuzzy_name_similarity("same", "SAME") - 1.0).abs() < f64::EPSILON);
        assert!((fuzzy_name_similarity("", "   ") - 1.0).abs() < f64::EPSILON);
        let partial = fuzzy_name_similarity("kitten", "sitting");
        assert!(partial > 0.0 && partial < 1.0, "unexpected {partial}");
        assert_eq!(fuzzy_name_similarity("", "abc"), 0.0);
    }
}
