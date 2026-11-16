//! `$ref` resolution for `OpenAPI` documents, including cross-document references.
//!
//! A `$ref` is split into a *document part* and a *fragment* (a JSON pointer):
//! `#/components/schemas/Pet` addresses the document the reference appears in,
//! while `common.yaml#/components/parameters/Session` and
//! `https://example.test/common.yaml#/x` address another one.
//!
//! # Resolution rules
//!
//! * `#/...` (and a bare `#`) address the document the reference appears in.
//! * A relative file path (`./common.yaml`, `../shared/common.yaml`) resolves
//!   against the directory of the document that declares it; an absolute path is
//!   used as written. Every document is read once and cached, and references are
//!   read and normalised lexically (no filesystem path resolution).
//! * A relative URL resolves against the URL of the document that declares it
//!   (an [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986) join through
//!   [`url::Url::join`]); an absolute `http(s)` URL is fetched with the client's
//!   [`HttpClient`].
//! * Only `http(s)` URLs and local filesystem paths are followed. Every other
//!   scheme (`file://`, `ftp://`, …) is a validation error, and a reference made
//!   by a document that itself came over `http(s)` stays a URL (a relative one is
//!   joined and fetched over HTTP), so a remote document never reaches the local
//!   filesystem.
//! * A document without a file or URL source (an inline
//!   [`OpenApiSpec::from_str`](super::spec::OpenApiSpec::from_str) document) can
//!   only resolve its own `#/...` pointers.
//! * At most [`MAX_REMOTE_DOCUMENTS`] external documents are loaded for one spec
//!   and a `$ref` chain may be at most [`MAX_REF_DEPTH`] hops long; a chain that
//!   returns to a document/pointer pair it already visited is rejected as a
//!   cycle.
//!
//! Every failure is an [`Error::Validation`] naming the reference as written,
//! the document it was resolved in, and the JSON pointer it targeted. Referenced
//! documents are loaded eagerly while the spec is assembled, so a broken
//! reference fails the load instead of the first call.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use serde_json::Value;
use url::Url;

use super::spec::parse_document;

/// Maximum number of hops followed through a chain of `$ref`s.
pub(crate) const MAX_REF_DEPTH: usize = 32;

/// Maximum number of external documents one spec may pull in.
///
/// The bound keeps a document graph from turning a spec load into an unbounded
/// crawl: the 33rd distinct referenced document is refused with a validation
/// error naming it.
pub(crate) const MAX_REMOTE_DOCUMENTS: usize = 32;

/// Deadline applied when a referenced document is fetched from a URL.
const DOCUMENT_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Label used for a document that has neither a file nor a URL source.
const INLINE_LABEL: &str = "<inline>";

/// Where one `OpenAPI` document was loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DocumentBase {
    /// An in-memory document with no source (`from_str`/`from_value`).
    Inline,
    /// A filesystem path (absolute after normalisation).
    File(PathBuf),
    /// An `http(s)` URL.
    Url(String),
}

impl DocumentBase {
    /// The cache key of this document, also used as its diagnostic label.
    fn key(&self) -> String {
        match self {
            Self::Inline => INLINE_LABEL.to_owned(),
            Self::File(path) => normalise(path).display().to_string(),
            Self::Url(url) => url.clone(),
        }
    }

    /// Resolves the document part of `reference` to the document it names.
    ///
    /// Returns `None` when the reference stays inside this document. `reference`
    /// is the whole `$ref` string as written and is only used in diagnostics.
    fn resolve_document(&self, document: &str, reference: &str) -> Result<Option<Self>> {
        let document = document.trim();
        if document.is_empty() {
            return Ok(None);
        }
        if document.starts_with("http://") || document.starts_with("https://") {
            let url = Url::parse(document).map_err(|error| {
                Error::validation(format!(
                    "invalid $ref `{reference}`: `{document}` is not a valid URL: {error}"
                ))
            })?;
            return Ok(Some(Self::Url(url.to_string())));
        }
        let scheme = document
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .or_else(|| document.strip_prefix("file:").map(|_| "file"));
        if let Some(scheme) = scheme {
            return Err(Error::validation(format!(
                "unsupported $ref `{reference}`: the `{scheme}:` scheme is not followed; \
                 use an `http(s)` URL or a local file path"
            )));
        }
        let path = Path::new(document);
        match self {
            Self::File(base) => {
                let target = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    base.parent().unwrap_or_else(|| Path::new(".")).join(path)
                };
                Ok(Some(Self::File(normalise(&target))))
            }
            Self::Url(base) => {
                // A relative reference of a URL document is a URL: it is joined
                // the RFC 3986 way and fetched over HTTP, never read from disk.
                let joined = Url::parse(base)
                    .and_then(|base| base.join(document))
                    .map_err(|error| {
                        Error::validation(format!(
                            "invalid $ref `{reference}`: cannot join `{document}` with the URL \
                             `{base}`: {error}"
                        ))
                    })?;
                Ok(Some(Self::Url(joined.to_string())))
            }
            Self::Inline => Err(Error::validation(format!(
                "$ref `{reference}` points outside its document, which is only resolvable when \
                 the spec is loaded from a file or URL"
            ))),
        }
    }
}

/// One cached document: its parsed body plus the base its own `$ref`s use.
#[derive(Debug, Clone)]
struct CachedDocument {
    document: Value,
    base: DocumentBase,
}

/// The `$ref` targets of one `OpenAPI` document and of the documents it reaches.
///
/// Built once per loaded spec; resolution afterwards performs no I/O.
#[derive(Debug, Clone)]
pub(crate) struct RefContext {
    root: CachedDocument,
    external: BTreeMap<String, CachedDocument>,
}

impl RefContext {
    /// A context for a document with no file or URL source.
    ///
    /// Only local `#/...` pointers resolve; every external reference is a
    /// validation error.
    pub(crate) fn inline(document: Value) -> Self {
        Self {
            root: CachedDocument {
                document,
                base: DocumentBase::Inline,
            },
            external: BTreeMap::new(),
        }
    }

    /// Builds the context for `document` loaded from `base`, eagerly loading
    /// every external document its `$ref`s reach with `http`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a reference uses an unsupported
    /// scheme, names a document that cannot be read, fetched, or parsed, or when
    /// [`MAX_REMOTE_DOCUMENTS`] is exceeded.
    pub(crate) async fn load(
        document: Value,
        base: DocumentBase,
        http: &HttpClient,
    ) -> Result<Self> {
        let root_key = base.key();
        let mut context = Self {
            root: CachedDocument { document, base },
            external: BTreeMap::new(),
        };
        let mut pending: VecDeque<String> = VecDeque::from([root_key]);
        while let Some(key) = pending.pop_front() {
            let (references, base) = match context.lookup(&key) {
                Some(entry) => {
                    let mut references = Vec::new();
                    collect_external_refs(&entry.document, &mut references);
                    (references, entry.base.clone())
                }
                None => continue,
            };
            for reference in references {
                let Some(target) =
                    base.resolve_document(split_reference(&reference).0, &reference)?
                else {
                    continue;
                };
                let target_key = target.key();
                if context.lookup(&target_key).is_some() {
                    continue;
                }
                if context.external.len() >= MAX_REMOTE_DOCUMENTS {
                    return Err(Error::validation(format!(
                        "$ref `{reference}` in document `{}` exceeds the limit of \
                         {MAX_REMOTE_DOCUMENTS} external documents per spec",
                        base.key()
                    )));
                }
                let document = fetch_document(http, &target, &reference, &base).await?;
                context.external.insert(
                    target_key.clone(),
                    CachedDocument {
                        document,
                        base: target,
                    },
                );
                pending.push_back(target_key);
            }
        }
        Ok(context)
    }

    /// The document the spec was loaded from.
    pub(crate) fn root_document(&self) -> &Value {
        &self.root.document
    }

    /// Resolves `reference` against the root document, following `$ref` chains.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed reference, a target that
    /// does not exist, a chain longer than [`MAX_REF_DEPTH`], or a cycle.
    pub(crate) fn resolve(&self, reference: &str) -> Result<Value> {
        self.resolve_from(&self.root.base.key(), reference)
    }

    /// Resolves `schema` when its top-level `$ref` must be inlined; sibling keys
    /// win over the referenced value.
    ///
    /// Nested references inside `properties` are intentionally left in place.
    ///
    /// # Errors
    ///
    /// Same as [`RefContext::resolve`].
    pub(crate) fn resolve_schema(&self, schema: &Value) -> Result<Value> {
        let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
            return Ok(schema.clone());
        };
        let resolved = self.resolve(reference)?;
        let Some(siblings) = schema.as_object() else {
            return Ok(resolved);
        };
        let Value::Object(mut merged) = resolved else {
            return Ok(resolved);
        };
        for (key, value) in siblings {
            if key != "$ref" {
                merged.insert(key.clone(), value.clone());
            }
        }
        Ok(Value::Object(merged))
    }

    /// Resolves `value` when it is a `$ref` object, otherwise clones it.
    ///
    /// # Errors
    ///
    /// Same as [`RefContext::resolve`].
    pub(crate) fn maybe_ref(&self, value: &Value) -> Result<Value> {
        match value.get("$ref").and_then(Value::as_str) {
            Some(reference) => self.resolve(reference),
            None => Ok(value.clone()),
        }
    }

    /// Looks one document up by cache key.
    fn lookup(&self, key: &str) -> Option<&CachedDocument> {
        if key == self.root.base.key() {
            return Some(&self.root);
        }
        self.external.get(key)
    }

    /// Resolves `reference` inside the document cached as `base_key`.
    fn resolve_from(&self, base_key: &str, reference: &str) -> Result<Value> {
        let mut visited: Vec<(String, String)> = Vec::new();
        let mut key = base_key.to_owned();
        let mut current = reference.trim().to_owned();
        for _ in 0..MAX_REF_DEPTH {
            let entry = self.lookup(&key).ok_or_else(|| {
                Error::validation(format!(
                    "unresolvable $ref `{reference}`: document `{key}` is not loaded"
                ))
            })?;
            let (document_part, pointer) = split_reference(&current);
            let target_key = match entry.base.resolve_document(document_part, &current)? {
                Some(target) => target.key(),
                None => key.clone(),
            };
            let target_entry = self.lookup(&target_key).ok_or_else(|| {
                Error::validation(format!(
                    "unresolvable $ref `{reference}`: document `{target_key}` is not loaded \
                     (JSON pointer `{pointer}`)"
                ))
            })?;
            let target = pointer_value(&target_entry.document, pointer, &current, &target_key)?;
            let marker = (target_key.clone(), pointer.to_owned());
            if visited.contains(&marker) {
                return Err(Error::validation(format!(
                    "cyclic $ref `{reference}` in the OpenAPI spec: `{current}` is reached twice \
                     (document `{target_key}`, JSON pointer `{pointer}`)"
                )));
            }
            visited.push(marker);
            match target.get("$ref").and_then(Value::as_str) {
                Some(next) => {
                    current = next.trim().to_owned();
                    key = target_key;
                }
                None => return Ok(target.clone()),
            }
        }
        Err(Error::validation(format!(
            "$ref `{reference}` in document `{base_key}` exceeds the maximum chain depth of \
             {MAX_REF_DEPTH} (JSON pointer `{}`)",
            split_reference(&current).1
        )))
    }
}

/// Splits a `$ref` into its document part and its fragment (with the `#`).
fn split_reference(reference: &str) -> (&str, &str) {
    match reference.split_once('#') {
        Some((document, _)) => (document, &reference[document.len()..]),
        None => (reference, ""),
    }
}

/// Walks one JSON pointer (`#/a/b`, `#`, or empty) inside `document`.
fn pointer_value<'a>(
    document: &'a Value,
    pointer: &str,
    reference: &str,
    document_key: &str,
) -> Result<&'a Value> {
    if pointer.is_empty() || pointer == "#" {
        return Ok(document);
    }
    let Some(rest) = pointer.strip_prefix('#') else {
        return Err(Error::validation(format!(
            "malformed $ref `{reference}` in document `{document_key}`: expected \
             `<document>#/<pointer>`"
        )));
    };
    let Some(rest) = rest.strip_prefix('/') else {
        return Err(Error::validation(format!(
            "malformed $ref `{reference}` in document `{document_key}`: `{pointer}` is not a \
             JSON pointer"
        )));
    };
    let mut current = document;
    for token in rest.split('/') {
        let token = token.replace("~1", "/").replace("~0", "~");
        current = match current {
            Value::Object(map) => map.get(&token),
            Value::Array(items) => token
                .parse::<usize>()
                .ok()
                .and_then(|index| items.get(index)),
            _ => None,
        }
        .ok_or_else(|| {
            Error::validation(format!(
                "$ref `{reference}` in document `{document_key}` does not resolve: `{token}` is \
                 missing (JSON pointer `{pointer}`)"
            ))
        })?;
    }
    Ok(current)
}

/// Collects every `$ref` string in `value` that points outside its document.
fn collect_external_refs(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
                let reference = reference.trim();
                if !split_reference(reference).0.is_empty() {
                    out.push(reference.to_owned());
                }
            }
            for nested in map.values() {
                collect_external_refs(nested, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_external_refs(item, out);
            }
        }
        _ => {}
    }
}

/// Reads and parses the document named by `target`.
async fn fetch_document(
    http: &HttpClient,
    target: &DocumentBase,
    reference: &str,
    containing: &DocumentBase,
) -> Result<Value> {
    let document = match target {
        DocumentBase::Url(url) => {
            let request = HttpRequest::get(url.clone()).timeout(DOCUMENT_FETCH_TIMEOUT);
            let response = http.execute(request).await?;
            if !response.is_success() {
                return Err(Error::validation(format!(
                    "failed to load the document `{url}` referenced by $ref `{reference}` in \
                     document `{}`: HTTP {}",
                    containing.key(),
                    response.status
                )));
            }
            parse_document(&response.text(), url)
        }
        DocumentBase::File(path) => {
            let metadata = std::fs::symlink_metadata(path).map_err(|error| {
                Error::validation(format!(
                    "failed to load the document `{}` referenced by $ref `{reference}` in \
                     document `{}`: {error}",
                    path.display(),
                    containing.key()
                ))
            })?;
            if metadata.file_type().is_symlink() {
                return Err(Error::validation(format!(
                    "the document `{}` referenced by $ref `{reference}` in document `{}` must \
                     not be a symbolic link",
                    path.display(),
                    containing.key()
                )));
            }
            if !metadata.is_file() {
                return Err(Error::validation(format!(
                    "the document `{}` referenced by $ref `{reference}` in document `{}` is not \
                     a file",
                    path.display(),
                    containing.key()
                )));
            }
            tokio::fs::read_to_string(path)
                .await
                .map_err(|error| {
                    Error::validation(format!(
                        "failed to load the document `{}` referenced by $ref `{reference}` in \
                     document `{}`: {error}",
                        path.display(),
                        containing.key()
                    ))
                })
                .and_then(|text| parse_document(&text, &path.display().to_string()))
        }
        DocumentBase::Inline => Err(Error::validation(format!(
            "$ref `{reference}` points outside its document, which is only resolvable when the \
             spec is loaded from a file or URL"
        ))),
    }?;
    Ok(document)
}

/// Makes `path` absolute so cache keys and diagnostics do not depend on the cwd.
fn normalise(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut normalised = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalised.pop() && !absolute.has_root() {
                    normalised.push("..");
                }
            }
            other => normalised.push(other.as_os_str()),
        }
    }
    normalised
}
