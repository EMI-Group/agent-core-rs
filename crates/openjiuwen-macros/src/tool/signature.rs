//! Validation of the annotated function signature.
//!
//! The macro accepts exactly one shape: a free, non-generic `async fn` taking a
//! by-value argument struct as its first parameter and an optional
//! `&ToolContext`/`ToolContext` as its second, returning either plain `T` or
//! `openjiuwen_common::error::Result<T>`.

use syn::spanned::Spanned;
use syn::{Error, FnArg, ItemFn, ReturnType, Type};

use crate::error::{Fail, Result};

/// How the optional second parameter receives the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContextArg {
    /// `&ToolContext`
    Borrowed,
    /// `ToolContext`
    Owned,
}

/// The validated shape of an annotated function.
pub(crate) struct ToolFn {
    /// Type of the argument struct (first parameter).
    pub(crate) args_type: Type,
    /// Optional context parameter.
    pub(crate) context: Option<ContextArg>,
    /// Whether the return type is `Result<..>` (errors are `?`-propagated).
    pub(crate) returns_result: bool,
}

const FREE_FN: &str = "`#[tool]` can only be applied to a free `async fn`";

/// Validates `function` and extracts what the expansion needs.
pub(crate) fn analyze(function: &ItemFn) -> Result<ToolFn> {
    let signature = &function.sig;
    if signature.asyncness.is_none()
        || signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
    {
        return Err(Error::new(signature.ident.span(), FREE_FN));
    }
    if let Some(receiver) = signature.receiver() {
        return Err(Error::new(receiver.span(), FREE_FN));
    }
    if !signature.generics.params.is_empty() {
        return Err(Error::new(
            signature.ident.span(),
            "`#[tool]` does not support generic functions",
        ));
    }

    let mut parameters = Vec::new();
    for argument in &signature.inputs {
        match argument {
            FnArg::Receiver(receiver) => {
                return Err(Error::new(receiver.span(), FREE_FN));
            }
            FnArg::Typed(parameter) => parameters.push(parameter),
        }
    }

    let first = parameters.first().fail_at(
        signature.ident.span(),
        "`#[tool]` requires an argument parameter: `async fn f(args: Args)` or `async fn \
         f(args: Args, ctx: &ToolContext)`",
    )?;
    if parameters.len() > 2 {
        return Err(Error::new(
            parameters[2].span(),
            "`#[tool]` supports at most two parameters: the argument struct and an optional \
             `&ToolContext`",
        ));
    }
    if is_tool_context(&first.ty) {
        return Err(Error::new(
            first.ty.span(),
            "the first parameter must be the argument struct, not `ToolContext`",
        ));
    }
    if let Type::Reference(reference) = &*first.ty {
        return Err(Error::new(
            reference.span(),
            "the argument parameter must be taken by value",
        ));
    }

    let context = match parameters.get(1) {
        Some(parameter) => Some(context_arg(&parameter.ty)?),
        None => None,
    };

    Ok(ToolFn {
        args_type: (*first.ty).clone(),
        context,
        returns_result: returns_result(&signature.output),
    })
}

/// Validates the optional context parameter type.
fn context_arg(ty: &Type) -> Result<ContextArg> {
    const MESSAGE: &str = "the second parameter must be `&ToolContext` or `ToolContext`";
    if let Type::Reference(reference) = ty {
        if reference.mutability.is_none() && is_tool_context(&reference.elem) {
            return Ok(ContextArg::Borrowed);
        }
        return Err(Error::new(ty.span(), MESSAGE));
    }
    if is_tool_context(ty) {
        Ok(ContextArg::Owned)
    } else {
        Err(Error::new(ty.span(), MESSAGE))
    }
}

/// Whether `ty` names `ToolContext`, optionally behind a shared reference.
fn is_tool_context(ty: &Type) -> bool {
    let ty = match ty {
        Type::Reference(reference) => &*reference.elem,
        other => other,
    };
    matches!(
        ty,
        Type::Path(path)
            if path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "ToolContext")
    )
}

/// Whether the return type's last path segment is `Result`.
fn returns_result(output: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = output else {
        return false;
    };
    matches!(
        &**ty,
        Type::Path(path)
            if path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Result")
    )
}
