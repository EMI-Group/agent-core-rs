//! `#[tool]` attribute: parse, validate, expand.
//!
//! `expand` is the only entry point the crate root needs; it keeps the original
//! item on every error path so the compiler never reports cascading "item
//! missing" diagnostics after the real message.

mod attrs;
mod expand;
mod signature;

use proc_macro2::TokenStream;
use quote::quote;
use syn::ItemFn;

use crate::error::{Fail, Result};

/// Expands `#[tool]` for `item`, or emits the original item plus a spanned
/// `compile_error!`.
// The item token stream must survive until the error path re-emits it, so it is
// taken by value even though the success path only reads it.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    match try_expand(attr, &item) {
        Ok(expanded) => expanded,
        Err(error) => {
            let compile_error = error.to_compile_error();
            quote!(#item #compile_error)
        }
    }
}

fn try_expand(attr: TokenStream, item: &TokenStream) -> Result<TokenStream> {
    let function: ItemFn =
        syn::parse2(item.clone()).fail("`#[tool]` can only be applied to a free `async fn`")?;
    let attrs = attrs::parse(attr)?;
    expand::expand(&function, &attrs)
}
