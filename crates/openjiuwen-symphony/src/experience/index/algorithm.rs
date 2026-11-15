//! Vector-algorithm names for the persisted experience index.
//!
//! Python passes a faiss index factory string (`"Flat"`, `"IVF4096,PQ16"`, ...) to
//! `faiss.index_factory`. An offline build has neither faiss nor numpy, so this module ports
//! the *names* and the coarse-quantiser geometry they imply, not faiss' binary index:
//!
//! | Requested name | Canonical form | Meaning here |
//! | --- | --- | --- |
//! | `flat`, `Flat`, ` FLAT ` | `flat` | exact exhaustive cosine search |
//! | `ivf`, `ivfflat` | `ivf` | IVF probe search, [`DEFAULT_IVF_NLIST`] coarse lists |
//! | `ivf4096`, `IVF 4096` | `ivf4096` | IVF probe search with 4096 requested lists |
//! | `IVF4096,PQ16`, ` ivf 4096 , pq 16 ` | `ivf4096,pq16` | the same, with the PQ suffix recorded |
//! | `ivfpq` | `ivf` | the PQ marker without a sub-quantiser count |
//!
//! Product quantisation is **parsed and recorded only**: no vector is compressed. The
//! `pq<m>` part of a name survives in the canonical form so a caller's configuration
//! round-trips and so the persisted manifest says which factory string built the index, but
//! every probed candidate is scored exactly against its stored vector — see
//! [`crate::experience::index`] for the search path.
//!
//! The canonical form is lowercase, whitespace-free, and keeps only meaningful digits, so
//! `" IVF 4096 , PQ 16 "` and `"ivf4096,pq16"` name the same index. A name this crate does
//! not implement (`hnsw`, `lsh`, ...) is an
//! [`Error::Validation`](crate::error::Error::Validation) naming `vector_algorithm`; so is a
//! blank name, an `ivf0`, and a `pq0`, because a zero-list or zero-sub-quantiser index cannot
//! hold a single vector.

use super::{DEFAULT_IVF_NLIST, FLAT_VECTOR_ALGORITHM};
use crate::error::{validation, Result};

/// The `flat` prefix of every IVF factory name.
const IVF_PREFIX: &str = "ivf";

/// The `pq` suffix of a product-quantisation factory name.
const PQ_MARKER: &str = "pq";

/// The separator between the coarse-quantiser and product-quantiser parts.
const SEPARATOR: char = ',';

/// A parsed vector-algorithm request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AlgorithmSpec {
    /// The canonical, persisted name ([`Self::canonical`]).
    canonical: String,
    /// The requested number of coarse lists, when the name carried one.
    nlist: Option<usize>,
}

impl AlgorithmSpec {
    /// The exact flat index.
    pub(crate) fn flat() -> Self {
        Self {
            canonical: FLAT_VECTOR_ALGORITHM.to_string(),
            nlist: None,
        }
    }

    /// Parses `raw` into a canonical algorithm request.
    ///
    /// The parse is case- and whitespace-insensitive; see the module documentation for the
    /// accepted forms.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming
    /// `vector_algorithm` for a blank name, an unimplemented name, or an `ivf`/`pq` part
    /// whose count is zero.
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        let canonical = normalize_algorithm(raw)?;
        Ok(Self {
            nlist: nlist_of(&canonical),
            canonical,
        })
    }

    /// The canonical, persisted name.
    pub(crate) fn canonical(&self) -> &str {
        &self.canonical
    }

    /// Reports whether this request selects the IVF probe index.
    pub(crate) fn is_ivf(&self) -> bool {
        !self.canonical.eq_ignore_ascii_case(FLAT_VECTOR_ALGORITHM)
    }

    /// The requested number of coarse lists, before it is clamped to the item count.
    pub(crate) fn nlist_request(&self) -> Option<usize> {
        self.nlist
    }
}

/// Parses a count that must be a non-zero decimal number.
fn parse_count(raw: &str, field: &str, digits: &str) -> Result<usize> {
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(unknown(raw));
    }
    match digits.parse::<usize>() {
        Ok(0) => Err(validation(
            "vector_algorithm",
            format!("{raw:?}: the {field} count must be greater than zero"),
        )),
        Ok(count) => Ok(count),
        Err(_) => Err(validation(
            "vector_algorithm",
            format!("{raw:?}: the {field} count does not fit in a usize"),
        )),
    }
}

/// The validation error for a name this crate does not implement.
fn unknown(raw: &str) -> crate::error::Error {
    validation(
        "vector_algorithm",
        format!(
            "{raw:?} is not supported: use {FLAT_VECTOR_ALGORITHM:?}, \"ivf\", \"ivfflat\", \
             \"ivf<nlist>\" or \"ivf<nlist>,pq<m>\""
        ),
    )
}

/// Normalizes a requested vector algorithm into its canonical, persisted name.
///
/// This is the crate's one algorithm grammar: [`AlgorithmSpec::parse`] calls it and then
/// reads the coarse-list count back out of the canonical name.
pub(crate) fn normalize_algorithm(raw: &str) -> Result<String> {
    let squeezed: String = raw
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if squeezed.is_empty() {
        return Err(validation("vector_algorithm", "must not be blank"));
    }
    let lowered = squeezed.to_ascii_lowercase();
    if lowered == FLAT_VECTOR_ALGORITHM {
        return Ok(FLAT_VECTOR_ALGORITHM.to_string());
    }
    let Some(rest) = lowered.strip_prefix(IVF_PREFIX) else {
        return Err(unknown(raw));
    };
    let (head, tail) = match rest.split_once(SEPARATOR) {
        Some((head, tail)) => (head, Some(tail)),
        None => (rest, None),
    };
    // `ivf`, `ivfflat` and `ivfpq` carry no coarse-quantiser count.
    if head.is_empty() || head == FLAT_VECTOR_ALGORITHM || head == PQ_MARKER {
        if tail.is_some() {
            return Err(unknown(raw));
        }
        return Ok(IVF_PREFIX.to_string());
    }
    let nlist = parse_count(raw, "nlist", head)?;
    let canonical = match tail {
        None => format!("{IVF_PREFIX}{nlist}"),
        Some(tail) => match tail.strip_prefix(PQ_MARKER) {
            // `ivf4096,pq` records the PQ marker without a sub-quantiser count.
            Some("") => format!("{IVF_PREFIX}{nlist}"),
            Some(digits) => {
                let pq = parse_count(raw, "pq", digits)?;
                format!("{IVF_PREFIX}{nlist},{PQ_MARKER}{pq}")
            }
            None => return Err(unknown(raw)),
        },
    };
    Ok(canonical)
}

/// The coarse-list count a canonical name requests (`None` when it names none).
fn nlist_of(canonical: &str) -> Option<usize> {
    let rest = canonical.strip_prefix(IVF_PREFIX)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// The effective number of coarse lists for an index holding `count` items.
///
/// The requested value is clamped to the item count — an index cannot have more lists than
/// vectors — and is at least one whenever the index is non-empty. An empty index reports
/// zero lists because there is nothing to quantise.
pub(crate) fn effective_nlist(request: Option<usize>, count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    request.unwrap_or(DEFAULT_IVF_NLIST).min(count).max(1)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn flat_is_case_and_whitespace_insensitive() {
        for raw in ["flat", "Flat", "FLAT", " flat ", "\tF L A T "] {
            let spec = AlgorithmSpec::parse(raw).expect("flat");
            assert_eq!(spec.canonical(), FLAT_VECTOR_ALGORITHM, "{raw}");
            assert!(!spec.is_ivf(), "{raw}");
            assert_eq!(spec.nlist_request(), None, "{raw}");
        }
    }

    #[test]
    fn ivf_names_normalize_to_the_canonical_form() {
        for (raw, canonical, nlist) in [
            ("ivf", "ivf", None),
            ("IVF", "ivf", None),
            ("ivfflat", "ivf", None),
            ("  IvfFlat  ", "ivf", None),
            ("ivfpq", "ivf", None),
            ("ivf4096", "ivf4096", Some(4096)),
            ("ivf 4096", "ivf4096", Some(4096)),
            ("IVF4096,PQ16", "ivf4096,pq16", Some(4096)),
            (" ivf 4096 , pq 16 ", "ivf4096,pq16", Some(4096)),
            ("ivf8,pq", "ivf8", Some(8)),
            ("ivf0008", "ivf8", Some(8)),
        ] {
            let spec = AlgorithmSpec::parse(raw).expect(raw);
            assert_eq!(spec.canonical(), canonical, "{raw}");
            assert!(spec.is_ivf(), "{raw}");
            assert_eq!(spec.nlist_request(), nlist, "{raw}");
        }
    }

    #[test]
    fn zero_counts_blank_names_and_unknown_names_are_validation_errors() {
        for raw in ["", "   ", "\n"] {
            let error = AlgorithmSpec::parse(raw).expect_err(raw);
            assert_eq!(error.kind(), ErrorKind::Validation, "{raw}");
            assert!(error.message().contains("vector_algorithm"), "{raw}");
        }
        for raw in ["ivf0", "IVF0", "ivf0,pq16", "ivf16,pq0", "ivf16,pq"] {
            if raw == "ivf16,pq" {
                // A bare `pq` marker is accepted and dropped from the canonical form.
                assert_eq!(
                    AlgorithmSpec::parse(raw).expect(raw).canonical(),
                    "ivf16",
                    "{raw}"
                );
                continue;
            }
            let error = AlgorithmSpec::parse(raw).expect_err(raw);
            assert_eq!(error.kind(), ErrorKind::Validation, "{raw}");
            assert!(error.message().contains("vector_algorithm"), "{raw}");
            assert!(error.message().contains("greater than zero"), "{raw}");
        }
        for raw in [
            "hnsw",
            "lsh",
            "ivf16,flat",
            "ivf16,sq8",
            "ivfp",
            "ivf,pq16",
            "flat,pq16",
        ] {
            let error = AlgorithmSpec::parse(raw).expect_err(raw);
            assert_eq!(error.kind(), ErrorKind::Validation, "{raw}");
            assert!(error.message().contains("vector_algorithm"), "{raw}");
            assert!(error.message().contains(raw.trim()), "{raw}");
        }
    }

    #[test]
    fn the_effective_list_count_is_clamped_to_the_item_count() {
        assert_eq!(effective_nlist(None, 0), 0);
        assert_eq!(effective_nlist(Some(4096), 0), 0);
        assert_eq!(effective_nlist(None, 1), 1);
        assert_eq!(effective_nlist(None, 3), 3);
        assert_eq!(effective_nlist(Some(2), 3), 2);
        assert_eq!(effective_nlist(None, 1000), DEFAULT_IVF_NLIST);
        assert_eq!(effective_nlist(Some(4096), 5000), 4096);
    }

    #[test]
    fn canonical_names_round_trip_through_the_parser() {
        for raw in ["flat", "ivf", "ivf4096", "ivf4096,pq16"] {
            let spec = AlgorithmSpec::parse(raw).expect(raw);
            let reparsed = AlgorithmSpec::parse(spec.canonical()).expect(raw);
            assert_eq!(reparsed, spec, "{raw}");
        }
    }
}
