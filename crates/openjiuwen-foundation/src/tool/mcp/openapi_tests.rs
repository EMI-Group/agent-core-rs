//! Tests for the `OpenAPI`→tools MCP transport.
//!
//! Everything is offline: documents are inline JSON/YAML or temporary files, and
//! every HTTP call is answered by a loopback `wiremock` server. Calls are bounded
//! by the client's own request timeout, so no test needs a wall-clock wait.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::http::HttpClient;
use serde_json::{json, Map};
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::client::{connect_mcp_server, McpClient, McpClientType, McpServerConfig};
use super::openapi::call::build_url;
use super::openapi::{OpenApiClient, OpenApiSpec};

/// The document used by the naming/schema tests: it covers every naming rule,
/// both parameter sources, a JSON request body, and three `$ref` shapes.
const NAMING_SPEC: &str = r##"
{
  "openapi": "3.0.3",
  "servers": [{"url": "https://api.example.test/v1/"}],
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "description": "A pet payload",
        "properties": {"name": {"type": "string"}, "tag": {"type": "string"}},
        "required": ["name"]
      },
      "Tag": {"type": "string", "enum": ["cat", "dog"]},
      "CycleA": {"$ref": "#/components/schemas/CycleA"}
    },
    "parameters": {
      "TagParam": {
        "name": "tag",
        "in": "query",
        "required": false,
        "description": "Tag filter",
        "schema": {"$ref": "#/components/schemas/Tag"}
      }
    }
  },
  "paths": {
    "/pets/{petId}": {
      "parameters": [
        {
          "name": "petId",
          "in": "path",
          "required": true,
          "description": "Pet identifier",
          "schema": {"type": "string"}
        }
      ],
      "get": {
        "operationId": "getPets__v2",
        "summary": "Get one pet",
        "description": "Returns a single pet",
        "parameters": [
          {"name": "verbose", "in": "query", "schema": {"type": "boolean"}, "description": "Verbose output"},
          {"name": "X-Trace", "in": "header", "schema": {"type": "string"}},
          {"name": "session", "in": "cookie", "schema": {"type": "string"}}
        ],
        "responses": {"200": {"description": "ok"}}
      },
      "delete": {"responses": {"204": {"description": "gone"}}}
    },
    "/pets": {
      "post": {
        "operationId": "createPet",
        "requestBody": {
          "required": true,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}}
        },
        "responses": {"201": {"description": "created"}}
      },
      "get": {"summary": "List pets", "responses": {"200": {"description": "ok"}}}
    },
    "/owners": {
      "get": {"operationId": "getPets", "summary": "Owners", "responses": {"200": {"description": "ok"}}}
    },
    "/long": {
      "get": {
        "operationId": "averyveryverylongoperationidnamethatcertainlyexceedsthesixtyfourcharacterlimit",
        "responses": {"200": {"description": "ok"}}
      }
    },
    "/search": {
      "get": {
        "operationId": "searchPets",
        "parameters": [{"$ref": "#/components/parameters/TagParam"}],
        "responses": {"200": {"description": "ok"}}
      }
    }
  }
}
"##;

/// The tool names of `spec`, in document order.
fn names(spec: &OpenApiSpec) -> Vec<String> {
    spec.operations()
        .iter()
        .map(|operation| operation.name.clone())
        .collect()
}

/// The operation named `name` of `spec`.
fn operation<'a>(spec: &'a OpenApiSpec, name: &str) -> &'a super::openapi::OpenApiOperation {
    spec.operations()
        .iter()
        .find(|operation| operation.name == name)
        .unwrap()
}

/// A document whose single call target is `base`.
fn call_spec(base: &str) -> String {
    format!(
        r#"{{
          "openapi": "3.0.3",
          "servers": [{{"url": "{base}"}}],
          "paths": {{
            "/pets/{{petId}}": {{
              "get": {{
                "operationId": "getPet",
                "parameters": [
                  {{"name": "petId", "in": "path", "required": true, "schema": {{"type": "string"}}}},
                  {{"name": "verbose", "in": "query", "schema": {{"type": "boolean"}}}}
                ],
                "responses": {{"200": {{"description": "ok"}}}}
              }}
            }},
            "/pets": {{
              "post": {{
                "operationId": "createPet",
                "requestBody": {{
                  "required": true,
                  "content": {{
                    "application/json": {{
                      "schema": {{
                        "type": "object",
                        "properties": {{"name": {{"type": "string"}}}},
                        "required": ["name"]
                      }}
                    }}
                  }}
                }},
                "responses": {{"201": {{"description": "created"}}}}
              }}
            }}
          }}
        }}"#
    )
}

/// An `OpenAPI` client configuration for a spec stored at `spec_path`.
fn openapi_config(spec_path: &str) -> McpServerConfig {
    McpServerConfig {
        client_type: McpClientType::OpenApi,
        server_path: spec_path.to_owned(),
        ..McpServerConfig::default()
    }
    .with_server_id("openapi-id")
}

/// Writes `text` to `<dir>/<name>` and returns the path as a string.
fn write_spec(dir: &tempfile::TempDir, name: &str, text: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, text).unwrap();
    path.to_str().unwrap().to_owned()
}

#[test]
fn operation_names_follow_the_python_rules() {
    let spec = OpenApiSpec::from_str(NAMING_SPEC).unwrap();
    assert_eq!(spec.base_url(), "https://api.example.test/v1");
    let long_id = "averyveryverylongoperationidnamethatcertainlyexceedsthesixtyfourcharacterlimit";
    assert_eq!(
        names(&spec),
        vec![
            "getPets".to_owned(),
            "DELETE_/pets/{petId}".to_owned(),
            "createPet".to_owned(),
            "List pets".to_owned(),
            "getPets_2".to_owned(),
            long_id.chars().take(64).collect::<String>(),
            "searchPets".to_owned(),
        ]
    );

    // A path with no `operationId` and no `summary` falls back to `METHOD_path`.
    let fallback = OpenApiSpec::from_str(
        r#"{"servers":[{"url":"https://api.example.test"}],
            "paths":{"/pets/{petId}":{"get":{"responses":{}}}}}"#,
    )
    .unwrap();
    assert_eq!(names(&fallback), vec!["GET_/pets/{petId}".to_owned()]);
}

#[test]
fn descriptions_append_parameter_and_body_documentation() {
    let spec = OpenApiSpec::from_str(NAMING_SPEC).unwrap();
    assert_eq!(
        operation(&spec, "getPets").description,
        "Returns a single pet\n\npetId: Pet identifier\n\nverbose: Verbose output"
    );
    assert_eq!(
        operation(&spec, "DELETE_/pets/{petId}").description,
        "Executes DELETE /pets/{petId}\n\npetId: Pet identifier"
    );
    assert_eq!(
        operation(&spec, "createPet").description,
        "Executes POST /pets\n\nA pet payload"
    );
    assert_eq!(operation(&spec, "List pets").description, "List pets");
    assert_eq!(operation(&spec, "getPets_2").description, "Owners");
    assert_eq!(
        operation(&spec, "searchPets").description,
        "Executes GET /search\n\ntag: Tag filter"
    );
}

#[test]
fn input_schemas_flatten_parameters_and_json_bodies() {
    let spec = OpenApiSpec::from_str(NAMING_SPEC).unwrap();

    let get_pets = operation(&spec, "getPets");
    assert_eq!(
        get_pets.input_schema,
        json!({
            "type": "object",
            "properties": {
                "petId": {"type": "string", "description": "Pet identifier"},
                "verbose": {"type": "boolean", "description": "Verbose output"},
                "X-Trace": {"type": "string"},
                "session": {"type": "string"}
            },
            "required": ["petId"]
        })
    );
    assert_eq!(get_pets.method, "GET");
    assert_eq!(get_pets.path, "/pets/{petId}");
    // Cookie parameters are collected like any other input; they travel in the
    // `Cookie` header at call time instead of the query string or the body.
    let cookie = get_pets
        .parameters
        .iter()
        .find(|parameter| parameter.location == "cookie")
        .unwrap();
    assert_eq!(cookie.name, "session");
    assert!(!cookie.required);

    let create_pet = operation(&spec, "createPet");
    assert!(create_pet.body_required);
    assert_eq!(
        create_pet.input_schema,
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "tag": {"type": "string"}},
            "required": ["name"]
        })
    );
    assert_eq!(
        create_pet
            .body_schema
            .as_ref()
            .and_then(|schema| schema.get("description")),
        Some(&json!("A pet payload"))
    );

    // A `$ref` parameter schema resolves and keeps the parameter description.
    assert_eq!(
        operation(&spec, "searchPets").input_schema,
        json!({
            "type": "object",
            "properties": {
                "tag": {"type": "string", "enum": ["cat", "dog"], "description": "Tag filter"}
            },
            "required": []
        })
    );
}

#[test]
fn local_refs_resolve_and_cycles_are_rejected() {
    let spec = OpenApiSpec::from_str(NAMING_SPEC).unwrap();
    let pet = spec.resolve("#/components/schemas/Pet").unwrap();
    assert_eq!(pet.get("type"), Some(&json!("object")));

    let error = spec.resolve("#/components/schemas/CycleA").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("cyclic"));

    let missing = spec.resolve("#/components/schemas/Missing").unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::Validation);
    assert!(missing.to_string().contains("Missing"), "{missing}");

    // An inline document has no file or URL base, so an external reference is a
    // validation error instead of a silent skip.
    let external = spec.resolve("other.yaml#/Pet").unwrap_err();
    assert_eq!(external.kind(), ErrorKind::Validation);
    assert!(
        external.to_string().contains("other.yaml#/Pet"),
        "{external}"
    );

    // A document that references the cycle is rejected while being collected.
    let mut document: serde_json::Value = serde_json::from_str(NAMING_SPEC).unwrap();
    document["components"]["parameters"]["TagParam"]["schema"] =
        json!({"$ref": "#/components/schemas/CycleA"});
    let error = OpenApiSpec::from_value(document).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn yaml_files_match_json_specs_and_json_extension_is_strict() {
    let dir = tempfile::tempdir().unwrap();
    let json_spec = OpenApiSpec::from_str(NAMING_SPEC).unwrap();

    let yaml_path = write_spec(&dir, "pets.yaml", NAMING_SPEC);
    let yaml_spec = OpenApiSpec::load(&yaml_path).await.unwrap();
    assert_eq!(names(&json_spec), names(&yaml_spec));
    assert_eq!(json_spec.base_url(), yaml_spec.base_url());
    for (from_json, from_yaml) in json_spec.operations().iter().zip(yaml_spec.operations()) {
        assert_eq!(from_json.input_schema, from_yaml.input_schema);
        assert_eq!(from_json.description, from_yaml.description);
        assert_eq!(from_json.method, from_yaml.method);
        assert_eq!(from_json.path, from_yaml.path);
    }

    // `.json` is parsed strictly: YAML-only text fails with a validation error.
    let broken = write_spec(
        &dir,
        "pets.json",
        "paths: {}\nservers: [{url: 'http://x.test'}]\n",
    );
    let error = OpenApiSpec::load(&broken).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let missing = OpenApiSpec::load("/nonexistent/openapi.json")
        .await
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn get_call_maps_the_json_response_and_sends_query_parameters() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pets/42"))
        .and(query_param("verbose", "true"))
        .and(query_param("api_key", "s3cret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": 42, "name": "rex"})))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(&dir, "openapi.json", &call_spec(&server.uri()));
    let config = openapi_config(&spec_path).with_auth_query_param("api_key", "s3cret");
    let client = OpenApiClient::new(&config).unwrap();

    let result = client
        .call_tool("getPet", json!({"petId": "42", "verbose": true}), None)
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.data.get("name"), Some(&json!("rex")));
    assert_eq!(result.data.get("id"), Some(&json!(42)));
}

#[tokio::test]
async fn post_call_sends_the_declared_json_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/pets"))
        .and(body_json(json!({"name": "rex"})))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"id": 7})))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(&dir, "openapi.json", &call_spec(&server.uri()));
    let client = OpenApiClient::new(&openapi_config(&spec_path)).unwrap();

    let result = client
        .call_tool("createPet", json!({"name": "rex"}), None)
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.data.get("id"), Some(&json!(7)));
}

#[tokio::test]
async fn non_success_statuses_become_unsuccessful_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pets/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(&dir, "openapi.json", &call_spec(&server.uri()));
    let config = openapi_config(&spec_path);
    let http = HttpClient::with_defaults().unwrap();
    let client = OpenApiClient::with_http_client(&config, http).unwrap();

    let result = client
        .call_tool("getPet", json!({"petId": "missing"}), None)
        .await
        .unwrap();
    assert!(!result.success);
    assert_eq!(result.data.get("status"), Some(&json!(404)));
    assert_eq!(result.data.get("body"), Some(&json!("not found")));
    let error = result.error.unwrap();
    assert!(error.contains("404"), "{error}");
    assert!(error.contains("not found"), "{error}");
}

#[tokio::test]
async fn unknown_tools_are_mcp_errors() {
    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(&dir, "openapi.json", &call_spec("http://127.0.0.1:1"));
    let client = OpenApiClient::new(&openapi_config(&spec_path)).unwrap();
    let error = client.call_tool("nope", json!({}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Mcp);
    assert!(error.to_string().contains("unknown OpenAPI tool `nope`"));
    assert!(client.is_connected().await);
}

#[tokio::test]
async fn spec_loaded_from_url_through_connect_mcp_server() {
    let server = MockServer::start().await;
    let base = server.uri();
    Mock::given(method("GET"))
        .and(path("/openapi.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(call_spec(&base)))
        .mount(&server)
        .await;

    let config = openapi_config(&format!("{base}/openapi.json"));
    let client = connect_mcp_server(&config).unwrap();
    let tools = client.list_tools().await.unwrap();
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["getPet", "createPet"]
    );
    assert_eq!(tools[0].description, "Executes GET /pets/{petId}");
    assert_eq!(
        tools[0].input_schema["properties"]["petId"]["type"],
        "string"
    );
    assert!(client.is_connected().await);
    assert_eq!(client.server_id(), "openapi-id");
}

#[test]
fn playwright_transport_is_a_documented_deferral() {
    let config = McpServerConfig {
        client_type: McpClientType::Playwright,
        ..McpServerConfig::default()
    };
    let error = connect_mcp_server(&config).err().unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .to_string()
        .contains("playwright MCP transport requires a browser runtime"));
}

#[test]
fn client_type_parses_and_serializes_the_new_spellings() {
    assert_eq!(
        McpClientType::parse("openapi").unwrap(),
        McpClientType::OpenApi
    );
    assert_eq!(
        McpClientType::parse("Open_API").unwrap(),
        McpClientType::OpenApi
    );
    assert_eq!(
        McpClientType::parse("Playwright").unwrap(),
        McpClientType::Playwright
    );
    assert_eq!(McpClientType::OpenApi.as_str(), "openapi");
    assert_eq!(McpClientType::Playwright.as_str(), "playwright");
    assert_eq!(
        serde_json::to_value(McpClientType::OpenApi).unwrap(),
        json!("openapi")
    );
    assert_eq!(
        serde_json::to_value(McpClientType::Playwright).unwrap(),
        json!("playwright")
    );

    let config: McpServerConfig =
        serde_json::from_value(json!({"client_type": "open_api"})).unwrap();
    assert_eq!(config.client_type, McpClientType::OpenApi);

    let error = McpClientType::parse("carrier-pigeon").unwrap_err();
    for spelling in ["stdio", "sse", "streamable-http", "openapi", "playwright"] {
        assert!(error.to_string().contains(spelling), "{error}");
    }
}

/// An external document holding one reusable parameter and one reusable schema.
///
/// It is deliberately not a complete `OpenAPI` document (`components` only): a
/// referenced document is resolved as written, only the root needs `servers`.
const COMMON_YAML: &str = r"
components:
  parameters:
    Session:
      name: x-session
      in: cookie
      required: false
      description: Session cookie
      schema:
        type: string
  schemas:
    Pet:
      type: object
      properties:
        name:
          type: string
";

/// A document whose single operation carries the `$ref` `reference` verbatim.
fn ref_spec(base: &str, reference: &str) -> String {
    serde_json::to_string_pretty(&json!({
        "openapi": "3.0.3",
        "servers": [{"url": base}],
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "getPets",
                    "parameters": [{"$ref": reference}],
                    "responses": {"200": {"description": "ok"}}
                }
            }
        }
    }))
    .unwrap()
}

/// A document whose `GET /pets` operation declares exactly `parameters`.
fn parameter_spec(base: &str, parameters: serde_json::Value) -> String {
    serde_json::to_string_pretty(&json!({
        "openapi": "3.0.3",
        "servers": [{"url": base}],
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "session",
                    "parameters": parameters,
                    "responses": {"200": {"description": "ok"}}
                }
            }
        }
    }))
    .unwrap()
}

/// The two cookie parameters used by the header tests.
fn cookie_parameters() -> serde_json::Value {
    json!([
        {"name": "x-session", "in": "cookie", "schema": {"type": "string"}},
        {"name": "theme", "in": "cookie", "schema": {"type": "string"}}
    ])
}

#[test]
fn multipart_request_bodies_stay_ignored_and_documented() {
    let spec = OpenApiSpec::from_str(
        r#"{"servers":[{"url":"https://api.example.test"}],
            "paths":{"/upload":{"post":{"operationId":"upload","requestBody":{
              "required":true,
              "content":{"multipart/form-data":{"schema":{"type":"object","properties":{"file":{"type":"string"}}}}}
            },"responses":{"200":{"description":"ok"}}}}}}"#,
    )
    .unwrap();
    let upload = operation(&spec, "upload");
    // Only `application/json` bodies are mapped onto tool arguments; the other
    // media types are ignored with a `tracing::warn!` and no input properties.
    assert!(upload.body_schema.is_none());
    assert!(upload.body_required);
    assert_eq!(
        upload.input_schema,
        json!({"type": "object", "properties": {}, "required": []})
    );
}

#[tokio::test]
async fn remote_file_refs_resolve_against_the_declaring_document() {
    let dir = tempfile::tempdir().unwrap();
    write_spec(&dir, "common.yaml", COMMON_YAML);
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &ref_spec(
            "https://api.example.test",
            "./common.yaml#/components/parameters/Session",
        ),
    );

    let spec = OpenApiSpec::load(&spec_path).await.unwrap();
    let get = operation(&spec, "getPets");
    assert_eq!(get.parameters.len(), 1);
    assert_eq!(get.parameters[0].name, "x-session");
    assert_eq!(get.parameters[0].location, "cookie");
    assert_eq!(
        get.parameters[0].description.as_deref(),
        Some("Session cookie")
    );
    assert_eq!(
        get.input_schema["properties"]["x-session"]["type"],
        json!("string")
    );
    // The resolved document is reachable through the public accessor too.
    let session = spec
        .resolve("./common.yaml#/components/parameters/Session")
        .unwrap();
    assert_eq!(session["name"], json!("x-session"));

    // A relative path climbs out of the spec directory with `..`.
    let nested = dir.path().join("specs");
    std::fs::create_dir(&nested).unwrap();
    let nested_spec = write_spec(
        &dir,
        "specs/openapi.json",
        &ref_spec(
            "https://api.example.test",
            "../common.yaml#/components/parameters/Session",
        ),
    );
    let spec = OpenApiSpec::load(&nested_spec).await.unwrap();
    assert_eq!(operation(&spec, "getPets").parameters[0].name, "x-session");

    // An absolute file path and a whole-document ref both resolve.
    let absolute = write_spec(
        &dir,
        "absolute.json",
        &ref_spec(
            "https://api.example.test",
            &format!(
                "{}#/components/parameters/Session",
                dir.path().join("common.yaml").display()
            ),
        ),
    );
    let spec = OpenApiSpec::load(&absolute).await.unwrap();
    assert_eq!(operation(&spec, "getPets").parameters[0].name, "x-session");
}

#[tokio::test]
async fn http_refs_resolve_and_relative_urls_join_the_spec_url() {
    let server = MockServer::start().await;
    let base = server.uri();
    Mock::given(method("GET"))
        .and(path("/specs/openapi.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ref_spec(
            &base,
            "../common.yaml#/components/parameters/Session",
        )))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/common.yaml"))
        .respond_with(ResponseTemplate::new(200).set_body_string(COMMON_YAML))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/shared/common.yaml"))
        .respond_with(ResponseTemplate::new(200).set_body_string(COMMON_YAML))
        .mount(&server)
        .await;

    let spec = OpenApiSpec::load(&format!("{base}/specs/openapi.json"))
        .await
        .unwrap();
    assert_eq!(
        spec.source_url(),
        Some(format!("{base}/specs/openapi.json").as_str())
    );
    assert_eq!(spec.base_url(), base);
    assert_eq!(operation(&spec, "getPets").parameters[0].name, "x-session");
    // `../common.yaml` is joined against the spec URL, not the cwd.
    assert_eq!(
        spec.resolve("../common.yaml#/components/parameters/Session")
            .unwrap()["name"],
        json!("x-session")
    );

    // An absolute `http(s)` reference in a file-loaded spec is fetched too.
    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &ref_spec(
            &base,
            &format!("{base}/shared/common.yaml#/components/schemas/Pet"),
        ),
    );
    let spec = OpenApiSpec::load(&spec_path).await.unwrap();
    assert_eq!(
        spec.resolve(&format!(
            "{base}/shared/common.yaml#/components/schemas/Pet"
        ))
        .unwrap()["type"],
        json!("object")
    );
}

#[tokio::test]
async fn cyclic_and_unresolvable_remote_refs_are_validation_errors() {
    let dir = tempfile::tempdir().unwrap();
    write_spec(&dir, "a.yaml", "x:\n  $ref: \"b.yaml#/y\"\n");
    write_spec(&dir, "b.yaml", "y:\n  $ref: \"a.yaml#/x\"\n");
    let cyclic = write_spec(
        &dir,
        "cycle.json",
        &ref_spec("https://api.example.test", "./a.yaml#/x"),
    );
    let error = OpenApiSpec::load(&cyclic).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("cyclic"), "{message}");
    assert!(message.contains("./a.yaml#/x"), "{message}");
    assert!(message.contains("a.yaml"), "{message}");
    assert!(message.contains("#/x"), "{message}");

    let missing = write_spec(
        &dir,
        "missing.json",
        &ref_spec(
            "https://api.example.test",
            "./nope.yaml#/components/parameters/Session",
        ),
    );
    let error = OpenApiSpec::load(&missing).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("./nope.yaml#/components/parameters/Session"),
        "{message}"
    );
    assert!(message.contains("nope.yaml"), "{message}");
    assert!(message.contains("missing.json"), "{message}");

    // A pointer that does not exist inside an existing document is reported with
    // the document and the pointer.
    let dangling = write_spec(
        &dir,
        "dangling.json",
        &ref_spec("https://api.example.test", "./a.yaml#/nope"),
    );
    let error = OpenApiSpec::load(&dangling).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("./a.yaml#/nope"), "{message}");
    assert!(message.contains("a.yaml"), "{message}");
    assert!(message.contains("does not resolve"), "{message}");

    // A non-`http(s)` scheme is refused rather than followed.
    let scheme = write_spec(
        &dir,
        "scheme.json",
        &ref_spec("https://api.example.test", "file:///etc/passwd#/x"),
    );
    let error = OpenApiSpec::load(&scheme).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("file:"), "{error}");
}

#[tokio::test]
async fn cookie_parameters_are_sent_as_one_cookie_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pets"))
        .and(header("cookie", "x-session=abc; theme=dark"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/pets"))
        .and(header("cookie", "theme=dark"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": "optional"})))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &parameter_spec(&server.uri(), cookie_parameters()),
    );
    let client = OpenApiClient::new(&openapi_config(&spec_path)).unwrap();

    let result = client
        .call_tool(
            "session",
            json!({"x-session": "abc", "theme": "dark"}),
            None,
        )
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.data.get("ok"), Some(&json!(true)));

    // A missing optional cookie parameter is omitted instead of sent empty.
    let result = client
        .call_tool("session", json!({"theme": "dark"}), None)
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.data.get("ok"), Some(&json!("optional")));
}

#[tokio::test]
async fn cookie_values_are_percent_encoded() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pets"))
        .and(header("cookie", "x-session=a%20b%3Bc%0A; theme=dark"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &parameter_spec(&server.uri(), cookie_parameters()),
    );
    let client = OpenApiClient::new(&openapi_config(&spec_path)).unwrap();

    let result = client
        .call_tool(
            "session",
            json!({"x-session": "a b;c\n", "theme": "dark"}),
            None,
        )
        .await
        .unwrap();
    assert!(result.success);
}

#[tokio::test]
async fn missing_required_cookie_parameters_are_validation_errors() {
    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &parameter_spec(
            "http://127.0.0.1:1",
            json!([{"name": "x-session", "in": "cookie", "required": true, "schema": {"type": "string"}}]),
        ),
    );
    let client = OpenApiClient::new(&openapi_config(&spec_path)).unwrap();

    let error = client
        .call_tool("session", json!({}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("session"), "{message}");
    assert!(message.contains("x-session"), "{message}");
}

#[tokio::test]
async fn cookie_parameters_merge_into_a_caller_supplied_cookie_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pets"))
        .and(header("cookie", "base=1; x-session=abc; theme=dark"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spec_path = write_spec(
        &dir,
        "openapi.json",
        &parameter_spec(&server.uri(), cookie_parameters()),
    );
    let config = openapi_config(&spec_path).with_auth_header("Cookie", "base=1");
    let client = OpenApiClient::new(&config).unwrap();

    let result = client
        .call_tool(
            "session",
            json!({"x-session": "abc", "theme": "dark"}),
            None,
        )
        .await
        .unwrap();
    assert!(result.success);
}

#[test]
fn empty_spec_source_is_rejected() {
    let error = OpenApiClient::new(&McpServerConfig::default())
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("server_path"));
}

#[test]
fn build_url_encodes_path_parameters_and_reports_missing_ones() {
    let spec = OpenApiSpec::from_str(
        r#"{"servers":[{"url":"https://api.example.test"}],
            "paths":{"/pets/{petId}":{"get":{"operationId":"getPet","responses":{}}}}}"#,
    )
    .unwrap();
    let operation = &spec.operations()[0];

    let args = json!({"petId": "a b/c~"}).as_object().unwrap().clone();
    assert_eq!(
        build_url(spec.base_url(), &operation.path, &args, operation).unwrap(),
        "https://api.example.test/pets/a%20b%2Fc~"
    );

    let empty = Map::new();
    let error = build_url(spec.base_url(), &operation.path, &empty, operation).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("getPet"), "{error}");
    assert!(error.to_string().contains("petId"), "{error}");
}
