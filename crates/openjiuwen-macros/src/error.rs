//! Error plumbing for the macro.
//!
//! Every problem is a [`syn::Error`] carrying a span, so the entry point can
//! render it with [`syn::Error::to_compile_error`] at the exact token that
//! caused it. Nothing in this crate panics.

use std::fmt::Display;

use proc_macro2::Span;
use syn::Error;

/// Result alias used throughout the crate.
pub(crate) type Result<T> = std::result::Result<T, Error>;

/// Converts an [`Option`] or a foreign [`std::result::Result`] into a spanned
/// macro error.
pub(crate) trait Fail<T> {
    /// Fails with `message` at the call-site span.
    fn fail(self, message: impl Display) -> Result<T>;

    /// Fails with `message` at `span`.
    fn fail_at(self, span: Span, message: impl Display) -> Result<T>;
}

impl<T> Fail<T> for Option<T> {
    fn fail(self, message: impl Display) -> Result<T> {
        self.fail_at(Span::call_site(), message)
    }

    fn fail_at(self, span: Span, message: impl Display) -> Result<T> {
        self.ok_or_else(|| Error::new(span, message))
    }
}

impl<T, E: Display> Fail<T> for std::result::Result<T, E> {
    fn fail(self, message: impl Display) -> Result<T> {
        self.fail_at(Span::call_site(), message)
    }

    fn fail_at(self, span: Span, message: impl Display) -> Result<T> {
        self.map_err(|error| Error::new(span, format!("{message}: {error}")))
    }
}
