//! Code generation for `#[tool]`.
//!
//! The original function is re-emitted unchanged; the wrapper struct, the
//! `Default`/`Tool` impls, and the `{fn}_tool` constructor are appended. Every
//! path in the generated code is fully qualified, and the wrapper simply
//! delegates to a [`FunctionTool`], so validation, default filling, and
//! cancellation live in exactly one place.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Ident, ItemFn, Path, Type, Visibility};

use crate::error::Result;
use crate::naming;

use super::attrs::{Execution, Exposure, ToolAttrs};
use super::signature::ToolFn;
use super::signature::{self, ContextArg};

/// Expands a validated `#[tool]` invocation.
pub(crate) fn expand(function: &ItemFn, attrs: &ToolAttrs) -> Result<TokenStream> {
    let contract = signature::analyze(function)?;
    let function_ident = &function.sig.ident;
    let bare_name = naming::bare_ident(function_ident);
    let tool_name = attrs
        .name
        .as_ref()
        .map_or_else(|| bare_name.clone(), syn::LitStr::value);
    let description = attrs.description.as_ref().map_or_else(
        || {
            crate::describe::description_from_docs(&function.attrs)
                .unwrap_or_else(|| naming::humanized_name(function_ident))
        },
        syn::LitStr::value,
    );

    let cfg_attrs: Vec<&Attribute> = function
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"))
        .collect();
    let input_params = input_params(attrs, &contract.args_type);
    let policy = policy_chain(attrs);
    let context_pattern = if contract.context.is_none() {
        quote!(_ctx)
    } else {
        quote!(ctx)
    };
    let invocation = invocation(function_ident, &contract);

    let generated = Generated {
        function,
        struct_ident: naming::struct_ident(function_ident),
        ctor_ident: naming::ctor_ident(function_ident),
        bare_name: &bare_name,
        tool_name: &tool_name,
        description: &description,
        input_params: &input_params,
        policy: &policy,
        invocation: &invocation,
        context_pattern: &context_pattern,
        args_type: &contract.args_type,
        cfg_attrs: &cfg_attrs,
        visibility: &function.vis,
        crate_path: &attrs.crate_path,
        common_crate: &attrs.common_crate,
    };
    Ok(generated.render())
}

/// Everything the two halves of the expansion need.
struct Generated<'a> {
    function: &'a ItemFn,
    struct_ident: Ident,
    ctor_ident: Ident,
    bare_name: &'a str,
    tool_name: &'a str,
    description: &'a str,
    input_params: &'a TokenStream,
    policy: &'a TokenStream,
    invocation: &'a TokenStream,
    context_pattern: &'a TokenStream,
    args_type: &'a Type,
    cfg_attrs: &'a [&'a Attribute],
    visibility: &'a Visibility,
    crate_path: &'a Path,
    common_crate: &'a Path,
}

impl Generated<'_> {
    /// Emits the original function plus every generated item.
    fn render(&self) -> TokenStream {
        let function = self.function;
        let cfg_attrs = self.cfg_attrs;
        let wrapper = self.wrapper();
        let tool = self.tool();

        quote! {
            #function

            #(#cfg_attrs)*
            #wrapper

            #(#cfg_attrs)*
            #tool
        }
    }

    /// The wrapper struct and its inherent impl (constants, constructor).
    fn wrapper(&self) -> TokenStream {
        let struct_ident = &self.struct_ident;
        let visibility = self.visibility;
        let crate_path = self.crate_path;
        let common_crate = self.common_crate;
        let tool_name = self.tool_name;
        let description = self.description;
        let input_params = self.input_params;
        let policy = self.policy;
        let invocation = self.invocation;
        let context_pattern = self.context_pattern;
        let args_type = self.args_type;
        let struct_doc = format!(
            "Tool generated from the `{}` function by `#[tool]`.",
            self.bare_name
        );
        let name_doc = format!("The model-facing name of the `{}` tool.", self.bare_name);
        let description_doc = format!(
            "The model-facing description of the `{}` tool.",
            self.bare_name
        );
        let new_doc = format!("Creates the `{}` tool.", self.bare_name);

        quote! {
            #[doc = #struct_doc]
            #[derive(Debug)]
            #visibility struct #struct_ident {
                inner: #crate_path::tool::FunctionTool,
            }

            impl #struct_ident {
                #[doc = #name_doc]
                #visibility const NAME: &'static str = #tool_name;

                #[doc = #description_doc]
                #visibility const DESCRIPTION: &'static str = #description;

                #[doc = #new_doc]
                #[must_use]
                #visibility fn new() -> Self {
                    let input_params: ::serde_json::Value = #input_params;
                    let card = #crate_path::tool::ToolCard::new(
                        Self::NAME,
                        Self::DESCRIPTION,
                        input_params,
                    )#policy;
                    let handler: #crate_path::tool::ToolHandler = ::std::sync::Arc::new(
                        |inputs: ::serde_json::Value, #context_pattern: #crate_path::tool::ToolContext| {
                            ::std::boxed::Box::pin(async move {
                                let args: #args_type = ::serde_json::from_value(inputs)
                                    .map_err(|error| {
                                        #common_crate::error::Error::validation(::std::format!(
                                            "invalid tool input: {error}"
                                        ))
                                    })?;
                                #invocation
                                ::serde_json::to_value(output).map_err(|error| {
                                    #common_crate::error::Error::tool(::std::format!(
                                        "failed to serialize tool output: {error}"
                                    ))
                                })
                            })
                        },
                    );
                    Self {
                        inner: #crate_path::tool::FunctionTool::new(card, handler),
                    }
                }
            }
        }
    }

    /// The `Default` impl, the `Tool` impl, and the boxed constructor.
    fn tool(&self) -> TokenStream {
        let struct_ident = &self.struct_ident;
        let ctor_ident = &self.ctor_ident;
        let visibility = self.visibility;
        let crate_path = self.crate_path;
        let common_crate = self.common_crate;
        let ctor_doc = format!(
            "Creates the `{}` tool as a shared `Tool` object.",
            self.bare_name
        );

        quote! {
            impl ::std::default::Default for #struct_ident {
                fn default() -> Self {
                    Self::new()
                }
            }

            #[::async_trait::async_trait]
            impl #crate_path::tool::Tool for #struct_ident {
                fn card(&self) -> &#crate_path::tool::ToolCard {
                    #crate_path::tool::Tool::card(&self.inner)
                }

                async fn invoke(
                    &self,
                    inputs: ::serde_json::Value,
                    ctx: &#crate_path::tool::ToolContext,
                ) -> #common_crate::error::Result<::serde_json::Value> {
                    #crate_path::tool::Tool::invoke(&self.inner, inputs, ctx).await
                }
            }

            #[doc = #ctor_doc]
            #[must_use]
            #visibility fn #ctor_ident() -> ::std::sync::Arc<dyn #crate_path::tool::Tool> {
                ::std::sync::Arc::new(#struct_ident::new())
            }
        }
    }
}

/// The argument schema: the user expression, or `schema_for::<Args>()`.
fn input_params(attrs: &ToolAttrs, args_type: &Type) -> TokenStream {
    let crate_path = &attrs.crate_path;
    attrs.input_params.as_ref().map_or_else(
        || quote!(#crate_path::tool::schema_for::<#args_type>()),
        |expression| quote!(#expression),
    )
}

/// The `.with_*` calls for the policy keys the user supplied.
fn policy_chain(attrs: &ToolAttrs) -> TokenStream {
    let crate_path = &attrs.crate_path;
    let mut policy = TokenStream::new();
    if let Some(stateless) = attrs.stateless {
        policy.extend(quote!(.with_stateless(#stateless)));
    }
    if let Some(parallel_safe) = attrs.parallel_safe {
        policy.extend(quote!(.with_parallel_safe(#parallel_safe)));
    }
    if let Some(idempotent) = attrs.idempotent {
        policy.extend(quote!(.with_idempotent(#idempotent)));
    }
    if let Some(exposure) = attrs.exposure {
        let variant = match exposure {
            Exposure::Direct => quote!(Direct),
            Exposure::Deferred => quote!(Deferred),
        };
        policy.extend(quote!(
            .with_exposure(#crate_path::tool::ToolExposure::#variant)
        ));
    }
    if let Some(execution) = attrs.execution {
        let variant = match execution {
            Execution::Server => quote!(Server),
            Execution::Client => quote!(Client),
        };
        policy.extend(quote!(
            .with_execution(#crate_path::tool::ToolExecution::#variant)
        ));
    }
    policy
}

/// The call to the annotated function plus output handling.
fn invocation(function_ident: &Ident, contract: &ToolFn) -> TokenStream {
    let call = match contract.context {
        None => quote!(#function_ident(args)),
        Some(ContextArg::Borrowed) => quote!(#function_ident(args, &ctx)),
        Some(ContextArg::Owned) => quote!(#function_ident(args, ctx)),
    };
    if contract.returns_result {
        quote!(let output = #call.await?;)
    } else {
        quote!(let output = #call.await;)
    }
}
