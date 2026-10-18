//! `GET /v1/models` and `GET /v1/models/{id}`: the gateway's model inventory.
//!
//! The inventory is the registry's own: the self-hosted workers' model cards
//! and, for an external worker, the cards its registration discovered with
//! the key the operator configured. No provider is asked at request time and
//! no credential is read here, so none can leave: a caller's own provider key
//! travels with a serving request to the one upstream that request names,
//! never with a listing, which could only show it to every configured
//! provider at once.
//!
//! An external worker's cards as they stand are not read: a serving request
//! for a model no worker serves refreshes them with that caller's own
//! credential (worker selection's refresh on a miss), so they may be one
//! caller's view of a provider; the listing reads the registration-time cards
//! alone and shows no caller's view to another.
//!
//! The list and the single model come in the shape of the API family the
//! caller speaks: a request carrying the `anthropic-version` header gets the
//! Anthropic objects (`type`, `display_name`, `created_at`, page markers),
//! every other request the OpenAI ones (`object`, `created`, `owned_by`).

use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{messages, model_card::ModelCard, models::ListModelsResponse};

use crate::{
    app_context::AppContext,
    middleware::into_messages_envelope,
    routers::error,
    worker::{RuntimeType, WorkerRegistry},
};

/// The API family a models request speaks, read from its headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelShape {
    OpenAi,
    Anthropic,
}

impl ModelShape {
    /// Anthropic clients send `anthropic-version` on every request; no
    /// OpenAI-compatible client does.
    fn from_headers(headers: &HeaderMap) -> Self {
        if headers.contains_key("anthropic-version") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }

    fn list(self, cards: Vec<ModelCard>) -> Response {
        match self {
            Self::OpenAi => Json(ListModelsResponse::from_model_cards(cards)).into_response(),
            Self::Anthropic => {
                Json(messages::ListModelsResponse::from_model_cards(cards)).into_response()
            }
        }
    }

    fn single(self, card: ModelCard) -> Response {
        match self {
            Self::OpenAi => Json(card.into_model_object()).into_response(),
            Self::Anthropic => Json(messages::ModelInfo::from_model_card(&card)).into_response(),
        }
    }

    /// The 404 of an unknown id: the same OpenAI-shaped envelope an unknown
    /// model gets on every other route, re-shaped into the Messages envelope
    /// (`not_found_error`, top-level `type`, `request_id`) for a Messages
    /// client, as `/v1/messages` answers it.
    async fn not_found(self, model_id: &str, request_id: String) -> Response {
        self.error(error::model_not_found(model_id), request_id)
            .await
    }

    /// The 503 of an inventory with no model in it (no worker registered, or
    /// every worker gone): the gateway's error envelope with a typed code, as
    /// every other refusal answers, in the caller's shape.
    async fn no_models(self, request_id: String) -> Response {
        self.error(
            error::service_unavailable("no_models_available", "No models available"),
            request_id,
        )
        .await
    }

    /// An OpenAI-shaped error response in the caller's shape: as it is for an
    /// OpenAI client, re-shaped into the Messages envelope for a Messages
    /// client.
    async fn error(self, response: Response, request_id: String) -> Response {
        match self {
            Self::OpenAi => response,
            Self::Anthropic => into_messages_envelope(response, request_id).await,
        }
    }
}

/// Answer `GET /v1/models` for the caller identified by `headers`
/// (`request_id` names the request in the Messages envelope of a refusal).
pub async fn list_models(
    context: &AppContext,
    headers: &HeaderMap,
    request_id: String,
) -> Response {
    list_models_with(&context.worker_registry, headers, request_id).await
}

/// [`list_models`] over an explicit registry. The headers choose the shape
/// of the answer and nothing else: a credential among them is not read, so
/// none can be sent on.
pub(crate) async fn list_models_with(
    registry: &WorkerRegistry,
    headers: &HeaderMap,
    request_id: String,
) -> Response {
    let shape = ModelShape::from_headers(headers);
    registry_models_response(registry, shape, request_id).await
}

/// Answer `GET /v1/models/{id}` from the inventory: the card of the model
/// `model_id` names (its id, or an alias the registry resolves), in the
/// caller's API shape, or the 404 an unknown model gets on every other route,
/// in the caller's error envelope (`request_id` names the request in the
/// Messages envelope).
pub async fn retrieve_model(
    context: &AppContext,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    retrieve_model_with(&context.worker_registry, headers, model_id, request_id).await
}

/// [`retrieve_model`] over an explicit registry. The lookup is by the
/// resolved canonical id alone: a canonical id that is also another card's
/// alias names its own card, as it does on every other route.
pub(crate) async fn retrieve_model_with(
    registry: &WorkerRegistry,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    let canonical = registry.resolve_model_alias(model_id);
    let wanted = canonical.as_deref().unwrap_or(model_id);
    let card = inventory_cards(registry)
        .into_iter()
        .find(|card| card.id == wanted);
    let shape = ModelShape::from_headers(headers);
    match card {
        Some(card) => shape.single(card),
        None => shape.not_found(model_id, request_id).await,
    }
}

/// The inventory: every self-hosted worker's model cards, and every external
/// worker's registration-time cards (`spec.models`, discovered with the
/// operator's key), never the cards a later refresh stored on it (the module
/// doc); a wildcard external worker has none.
fn inventory_cards(registry: &WorkerRegistry) -> Vec<ModelCard> {
    registry
        .get_all()
        .iter()
        .flat_map(|w| {
            if matches!(w.metadata().spec.runtime_type, RuntimeType::External) {
                w.metadata().spec.models.all().to_vec()
            } else {
                w.models()
            }
        })
        .collect()
}

/// The inventory as a list in the caller's shape. An empty list is the answer
/// when workers are registered and none has a card to list (wildcard
/// providers: the gateway serves, there is nothing to name); the 503 of an
/// empty registry (no worker registered, or every worker gone) comes in the
/// caller's error envelope.
async fn registry_models_response(
    registry: &WorkerRegistry,
    shape: ModelShape,
    request_id: String,
) -> Response {
    if registry.get_all().is_empty() {
        return shape.no_models(request_id).await;
    }
    shape.list(inventory_cards(registry))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::http::{header, HeaderValue, StatusCode};
    use openai_protocol::worker::HealthCheckConfig;
    use serde_json::Value;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, Worker, WorkerType};

    /// An inventory with no model in it answers the gateway's JSON error
    /// envelope with a typed code, as every other refusal does, in the
    /// caller's shape: the OpenAI envelope, or the Messages envelope with its
    /// `request_id` for a Messages client.
    #[tokio::test]
    async fn an_empty_inventory_answers_the_error_envelope_in_the_callers_shape() {
        let registry = WorkerRegistry::new();

        let response =
            registry_models_response(&registry, ModelShape::OpenAi, "req_1".to_string()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "no_models_available", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert!(body["error"]["type"].is_string(), "{body}");

        let response =
            registry_models_response(&registry, ModelShape::Anthropic, "req_2".to_string()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(response).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert_eq!(body["request_id"], "req_2", "{body}");
    }

    /// The listing is the registry's own inventory, an external worker's
    /// registration-time card next to the self-hosted one, and no provider
    /// is asked for it: neither a caller's bearer nor the provider's
    /// configured key is sent anywhere by the listing (a real stub provider
    /// counts hits), and the cards a refresh on a miss stored on the worker
    /// with some caller's credential are never shown.
    #[tokio::test]
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn the_listing_is_the_operators_inventory_and_no_credential_reaches_a_provider() {
        let hit_count = Arc::new(AtomicUsize::new(0));
        let hits = Arc::clone(&hit_count);
        let provider = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move || {
                let hits = Arc::clone(&hits);
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({
                        "object": "list",
                        "data": [{"id": "live-only", "object": "model"}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, provider).await;
        });

        let registry = WorkerRegistry::new();
        // The provider's worker as its registration leaves it: the configured
        // key and the card discovered with it.
        let external_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{addr}"))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .api_key("provider-key")
                .models(vec![ModelCard::new("provider-model")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(Arc::clone(&external_worker));
        let self_hosted: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("local-model")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(self_hosted);

        let mut with_bearer = HeaderMap::new();
        with_bearer.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer callers-own-key"),
        );
        for headers in [with_bearer, HeaderMap::new()] {
            let response = list_models_with(&registry, &headers, rid()).await;
            assert_eq!(response.status(), StatusCode::OK);
            let body = json_body(response).await;
            assert_eq!(
                sorted_ids(&body),
                ["local-model", "provider-model"],
                "{body}"
            );
        }
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            0,
            "the listing asked the provider"
        );

        // A refresh on a miss stored what some caller's credential could see;
        // the listing does not change.
        external_worker.set_models(vec![ModelCard::new("ft:some-callers-private-model")]);
        let body = json_body(list_models_with(&registry, &HeaderMap::new(), rid()).await).await;
        assert_eq!(
            sorted_ids(&body),
            ["local-model", "provider-model"],
            "{body}"
        );

        // The single-model route reads the same inventory.
        let response =
            retrieve_model_with(&registry, &HeaderMap::new(), "provider-model", rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await["id"], "provider-model");
    }

    /// Workers with no card to list (a wildcard provider, whose registration
    /// discovered nothing) answer an empty list in the caller's shape: the
    /// gateway serves; the 503 is for an empty registry.
    #[tokio::test]
    async fn workers_without_a_card_list_nothing_and_an_empty_registry_refuses() {
        let registry = WorkerRegistry::new();
        let wildcard: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://provider.invalid")
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(wildcard);

        let response = list_models_with(&registry, &HeaderMap::new(), rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["object"], "list", "{body}");
        assert_eq!(body["data"], serde_json::json!([]), "{body}");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let response = list_models_with(&registry, &headers, rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["data"], serde_json::json!([]), "{body}");
        assert_eq!(body["has_more"], false, "{body}");

        let response = list_models_with(&WorkerRegistry::new(), &HeaderMap::new(), rid()).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// The listed ids, sorted.
    fn sorted_ids(body: &Value) -> Vec<&str> {
        let mut ids: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        ids.sort_unstable();
        ids
    }

    fn registry_with(cards: Vec<ModelCard>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(cards)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker);
        registry
    }

    /// A request id for the tests that do not read it back.
    fn rid() -> String {
        "req_test".to_owned()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The same inventory answers both API families, each in its own shape:
    /// the OpenAI list by default, the Anthropic list (page markers, `type`,
    /// `display_name`, `created_at`) when the request carries the header
    /// every Anthropic client sends.
    #[tokio::test]
    async fn the_list_takes_the_shape_of_the_callers_api_family() {
        let mut card = ModelCard::new("m1");
        card.display_name = Some("Model One".to_owned());
        let registry = registry_with(vec![card]);

        let openai = json_body(list_models_with(&registry, &HeaderMap::new(), rid()).await).await;
        assert_eq!(openai["object"], "list");
        assert_eq!(openai["data"][0]["object"], "model");
        assert_eq!(openai["data"][0]["id"], "m1");
        assert!(openai.get("has_more").is_none());

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let anthropic = json_body(list_models_with(&registry, &headers, rid()).await).await;
        assert_eq!(anthropic["has_more"], false);
        assert_eq!(anthropic["first_id"], "m1");
        assert_eq!(anthropic["last_id"], "m1");
        assert_eq!(
            anthropic["data"][0],
            serde_json::json!({
                "type": "model",
                "id": "m1",
                "display_name": "Model One",
                "created_at": "1970-01-01T00:00:00Z"
            })
        );
    }

    /// A single model is found by its id or by an alias and answered in the
    /// caller's shape; an unknown id is the 404 every other route gives.
    #[tokio::test]
    async fn a_single_model_is_retrieved_by_id_or_alias() {
        let mut card = ModelCard::new("m1");
        card.aliases = vec!["alias-1".to_owned()];
        let registry = registry_with(vec![card]);

        let by_id = retrieve_model_with(&registry, &HeaderMap::new(), "m1", rid()).await;
        assert_eq!(by_id.status(), StatusCode::OK);
        let body = json_body(by_id).await;
        assert_eq!(body["object"], "model");
        assert_eq!(body["id"], "m1");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let by_alias = retrieve_model_with(&registry, &headers, "alias-1", rid()).await;
        assert_eq!(by_alias.status(), StatusCode::OK);
        let body = json_body(by_alias).await;
        assert_eq!(body["type"], "model");
        assert_eq!(body["id"], "m1");

        let unknown = retrieve_model_with(&registry, &headers, "nope", rid()).await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    /// An unknown id is refused in the envelope of the API the caller speaks:
    /// the OpenAI-shaped one by default, the Messages envelope (top-level
    /// `type`, `error.type` `not_found_error`, `request_id`) under the header
    /// every Anthropic client sends, as `/v1/messages` refuses the same model.
    #[tokio::test]
    async fn an_unknown_model_is_refused_in_the_callers_envelope() {
        let registry = registry_with(vec![ModelCard::new("m1")]);

        let openai = retrieve_model_with(&registry, &HeaderMap::new(), "nope", rid()).await;
        assert_eq!(openai.status(), StatusCode::NOT_FOUND);
        let body = json_body(openai).await;
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert!(body.get("type").is_none(), "{body}");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let messages =
            retrieve_model_with(&registry, &headers, "nope", "req_models_1".to_owned()).await;
        assert_eq!(messages.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            messages.headers()["content-type"],
            "application/json",
            "{:?}",
            messages.headers()
        );
        let body = json_body(messages).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["type"], "not_found_error", "{body}");
        assert_eq!(
            body["error"]["message"], "No worker available for model 'nope'",
            "{body}"
        );
        assert_eq!(body["request_id"], "req_models_1", "{body}");
        assert!(body["error"].get("code").is_none(), "{body}");
        assert!(body["error"].get("param").is_none(), "{body}");
    }

    /// A canonical id that is also another card's alias names its own card:
    /// the lookup goes by the resolved canonical id, never by a raw alias.
    #[tokio::test]
    async fn a_canonical_id_that_is_another_cards_alias_answers_its_own_card() {
        let mut aliased = ModelCard::new("org/first");
        aliased.aliases = vec!["org/second".to_owned()];
        let registry = registry_with(vec![aliased, ModelCard::new("org/second")]);

        let response = retrieve_model_with(&registry, &HeaderMap::new(), "org/second", rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["id"], "org/second");
    }
}
