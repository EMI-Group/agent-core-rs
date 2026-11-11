//! Parsing of the `#[tool(...)]` attribute arguments.
//!
//! The argument list is a comma-separated list of `key = value` pairs. Keys are
//! parsed with [`syn::ext::IdentExt::parse_any`] because `crate` is a keyword,
//! and every value kind is validated here so the expansion only sees typed data.

use std::collections::BTreeSet;

use proc_macro2::TokenStream;
use syn::ext::IdentExt;
use syn::parse::{ParseStream, Parser};
use syn::spanned::Spanned;
use syn::{Error, Expr, ExprLit, Lit, LitStr, Path, Token};

use crate::error::{Fail, Result};

/// `ToolCard.exposure` choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exposure {
    /// Always visible to the model.
    Direct,
    /// Hidden until tool search discovers it.
    Deferred,
}

/// `ToolCard.execution` choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Execution {
    /// The agent runtime executes the call.
    Server,
    /// The connected client executes the call.
    Client,
}

/// A parsed `#[tool(...)]` argument list; every key is optional.
pub(crate) struct ToolAttrs {
    pub(crate) name: Option<LitStr>,
    pub(crate) description: Option<LitStr>,
    pub(crate) input_params: Option<Expr>,
    pub(crate) stateless: Option<bool>,
    pub(crate) parallel_safe: Option<bool>,
    pub(crate) idempotent: Option<bool>,
    pub(crate) exposure: Option<Exposure>,
    pub(crate) execution: Option<Execution>,
    pub(crate) crate_path: Path,
    pub(crate) common_crate: Path,
}

impl Default for ToolAttrs {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            input_params: None,
            stateless: None,
            parallel_safe: None,
            idempotent: None,
            exposure: None,
            execution: None,
            crate_path: syn::parse_quote!(::openjiuwen_foundation),
            common_crate: syn::parse_quote!(::openjiuwen_common),
        }
    }
}

/// The canonical attribute keys, including the `crate` aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Name,
    Description,
    InputParams,
    Stateless,
    ParallelSafe,
    Idempotent,
    Exposure,
    Execution,
    Crate,
    CommonCrate,
}

impl Key {
    fn from_text(text: &str) -> Option<Self> {
        match text {
            "name" => Some(Self::Name),
            "description" => Some(Self::Description),
            "input_params" => Some(Self::InputParams),
            "stateless" => Some(Self::Stateless),
            "parallel_safe" => Some(Self::ParallelSafe),
            "idempotent" => Some(Self::Idempotent),
            "exposure" => Some(Self::Exposure),
            "execution" => Some(Self::Execution),
            "crate" | "crate_path" | "krate" => Some(Self::Crate),
            "common_crate" => Some(Self::CommonCrate),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Description => "description",
            Self::InputParams => "input_params",
            Self::Stateless => "stateless",
            Self::ParallelSafe => "parallel_safe",
            Self::Idempotent => "idempotent",
            Self::Exposure => "exposure",
            Self::Execution => "execution",
            Self::Crate => "crate",
            Self::CommonCrate => "common_crate",
        }
    }
}

/// Parses the attribute argument token stream.
pub(crate) fn parse(tokens: TokenStream) -> Result<ToolAttrs> {
    let mut attrs = ToolAttrs::default();
    if tokens.is_empty() {
        return Ok(attrs);
    }

    let mut seen: BTreeSet<Key> = BTreeSet::new();
    let parser = |input: ParseStream<'_>| -> Result<()> {
        while !input.is_empty() {
            let key_ident = input.call(syn::Ident::parse_any)?;
            let key = Key::from_text(&key_ident.to_string()).fail_at(
                key_ident.span(),
                format!(
                    "unknown `#[tool]` attribute `{key_ident}`; expected one of: name, \
                     description, input_params, stateless, parallel_safe, idempotent, \
                     exposure, execution, crate, common_crate"
                ),
            )?;
            if !seen.insert(key) {
                return Err(Error::new(
                    key_ident.span(),
                    format!("duplicate `#[tool]` attribute `{}`", key.as_str()),
                ));
            }
            if !input.peek(Token![=]) {
                return Err(Error::new(
                    key_ident.span(),
                    format!(
                        "`#[tool]` attribute `{key_ident}` must be written as `{} = <value>`",
                        key.as_str()
                    ),
                ));
            }
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;
            apply(&mut attrs, key, &value)?;
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(())
    };

    parser.parse2(tokens)?;
    Ok(attrs)
}

/// Stores one parsed `key = value` pair.
fn apply(attrs: &mut ToolAttrs, key: Key, value: &Expr) -> Result<()> {
    match key {
        Key::Name => {
            attrs.name = Some(string_value(value, "`name` must be a string literal")?);
        }
        Key::Description => {
            attrs.description = Some(string_value(
                value,
                "`description` must be a string literal",
            )?);
        }
        Key::InputParams => attrs.input_params = Some(value.clone()),
        Key::Stateless => attrs.stateless = Some(bool_value(value, key)?),
        Key::ParallelSafe => attrs.parallel_safe = Some(bool_value(value, key)?),
        Key::Idempotent => attrs.idempotent = Some(bool_value(value, key)?),
        Key::Exposure => attrs.exposure = Some(exposure_value(value)?),
        Key::Execution => attrs.execution = Some(execution_value(value)?),
        Key::Crate => {
            attrs.crate_path = path_value(
                value,
                "`crate` must be a path such as `::openjiuwen_foundation`",
            )?;
        }
        Key::CommonCrate => {
            attrs.common_crate = path_value(
                value,
                "`common_crate` must be a path such as `::openjiuwen_common`",
            )?;
        }
    }
    Ok(())
}

fn string_value(value: &Expr, message: &str) -> Result<LitStr> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(text),
        ..
    }) = value
    {
        return Ok(text.clone());
    }
    Err(Error::new(value.span(), message))
}

fn bool_value(value: &Expr, key: Key) -> Result<bool> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Bool(boolean),
        ..
    }) = value
    {
        return Ok(boolean.value);
    }
    Err(Error::new(
        value.span(),
        format!("`{}` must be `true` or `false`", key.as_str()),
    ))
}

fn exposure_value(value: &Expr) -> Result<Exposure> {
    const MESSAGE: &str = "`exposure` must be `\"direct\"` or `\"deferred\"`";
    let Expr::Lit(ExprLit {
        lit: Lit::Str(text),
        ..
    }) = value
    else {
        return Err(Error::new(value.span(), MESSAGE));
    };
    match text.value().as_str() {
        "direct" => Ok(Exposure::Direct),
        "deferred" => Ok(Exposure::Deferred),
        _ => Err(Error::new(text.span(), MESSAGE)),
    }
}

fn execution_value(value: &Expr) -> Result<Execution> {
    const MESSAGE: &str = "`execution` must be `\"server\"` or `\"client\"`";
    let Expr::Lit(ExprLit {
        lit: Lit::Str(text),
        ..
    }) = value
    else {
        return Err(Error::new(value.span(), MESSAGE));
    };
    match text.value().as_str() {
        "server" => Ok(Execution::Server),
        "client" => Ok(Execution::Client),
        _ => Err(Error::new(text.span(), MESSAGE)),
    }
}

/// Accepts a path either as a string literal (`"::openjiuwen_foundation"`) or
/// as a bare path expression (`::openjiuwen_foundation`).
fn path_value(value: &Expr, message: &str) -> Result<Path> {
    match value {
        Expr::Lit(ExprLit {
            lit: Lit::Str(text),
            ..
        }) => syn::parse_str::<Path>(&text.value()).fail_at(text.span(), message),
        Expr::Path(path) => Ok(path.path.clone()),
        other => Err(Error::new(other.span(), message)),
    }
}
