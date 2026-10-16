//! Bearer JWTs on the serving routes (`data_plane_jwt`): a token the control
//! plane's validator accepts is the tenant its claim names, the keys keep
//! working next to it, and a refused token is told its category. Through the
//! middleware on an echo route (the identity it resolves) and through the
//! real app over a mock worker (the routes it guards).

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    middleware::from_fn_with_state,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use serde_json::{json, Value};
use smg::{
    config::{DataPlaneJwtConfig, TenantApiKeyEntry},
    middleware::{auth_middleware, AuthConfig, DataPlaneCaller},
};
use smg_auth::{ControlPlaneAuthConfig, ControlPlaneAuthState, JwtConfig};
use tower::ServiceExt;

use crate::common::{
    jwks::{key_set, KeySet, AUDIENCE, ISSUER},
    test_app::create_test_app_with_context_and_auth,
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

/// The control plane's auth state with the test key set as the JWT issuer.
#[expect(clippy::expect_used, reason = "test helper: failures panic")]
async fn control_plane() -> ControlPlaneAuthState {
    let config = ControlPlaneAuthConfig {
        jwt: Some(JwtConfig::new(ISSUER, AUDIENCE).with_jwks_uri(key_set().uri.clone())),
        api_keys: vec![],
        audit_enabled: false,
    };
    ControlPlaneAuthState::from_config(config)
        .await
        .expect("a validator over the test key set")
}

/// The serving check: the tenant key for `team-red`, the JWT check with
/// `claim` as the tenant claim.
async fn serving_auth(claim: &str) -> AuthConfig {
    AuthConfig::with_tenant_keys(
        None,
        &[TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "red-secret".to_string(),
        }],
    )
    .with_data_plane_jwt(
        Some(&DataPlaneJwtConfig {
            tenant_claim: claim.to_string(),
        }),
        Some(&control_plane().await),
    )
}

async fn echo_tenant(request: Request<Body>) -> impl IntoResponse {
    request
        .extensions()
        .get::<DataPlaneCaller>()
        .map(|caller| caller.tenant_key().to_string())
        .unwrap_or_else(|| "missing".to_string())
}

/// `GET /` through the middleware with `authorization`, on a route that
/// answers the tenant key it resolved.
#[expect(clippy::unwrap_used, reason = "test helper: failures panic")]
async fn resolve(auth: &AuthConfig, authorization: Option<String>) -> Response {
    let app = Router::new()
        .route("/", get(echo_tenant))
        .route_layer(from_fn_with_state(auth.clone(), auth_middleware));
    let mut request = Request::builder().uri("/");
    if let Some(authorization) = authorization {
        request = request.header(header::AUTHORIZATION, authorization);
    }
    app.oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[expect(clippy::unwrap_used, reason = "test helper: failures panic")]
async fn text(response: Response) -> String {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(body.to_vec()).unwrap()
}

/// The `Authorization` value for `token`; `None` is the caller's no-credential case.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the option is the shape `resolve` takes, so the no-credential case reads the same"
)]
fn bearer(token: &str) -> Option<String> {
    Some(format!("Bearer {token}"))
}

/// A refused token's response: 401, the `invalid_token` challenge, the
/// OpenAI-style body with the category in its message.
#[expect(clippy::unwrap_used, reason = "test helper: failures panic")]
async fn assert_refused(response: Response, told: &str) {
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer realm=\"data-plane\", error=\"invalid_token\"")
    );
    let body: Value = serde_json::from_str(&text(response).await).unwrap();
    assert_eq!(body["error"]["code"], "invalid_token", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(
        body["error"]["message"],
        format!("Invalid bearer token: {told}"),
        "{body}"
    );
}

/// The tenant claim names the tenant; the same id as a tenant key resolves
/// to the same identity; a token without the claim is its subject.
#[tokio::test]
async fn a_valid_jwt_is_the_tenant_its_claim_names() {
    let auth = serving_auth("tenant").await;
    let keys = key_set();

    let by_jwt = resolve(&auth, bearer(&keys.token(json!({ "tenant": "team-red" })))).await;
    assert_eq!(by_jwt.status(), StatusCode::OK);
    assert_eq!(text(by_jwt).await, "auth:team-red");

    let by_key = resolve(&auth, bearer("red-secret")).await;
    assert_eq!(by_key.status(), StatusCode::OK);
    assert_eq!(
        text(by_key).await,
        "auth:team-red",
        "the key names the same tenant"
    );

    let other = resolve(&auth, bearer(&keys.token(json!({ "tenant": "team-blue" })))).await;
    assert_eq!(text(other).await, "auth:team-blue");

    let by_subject = resolve(&auth, bearer(&keys.token(json!({})))).await;
    assert_eq!(by_subject.status(), StatusCode::OK);
    assert_eq!(
        text(by_subject).await,
        "auth:alice",
        "no tenant claim: the subject"
    );

    let padded = resolve(
        &auth,
        bearer(&keys.token(json!({ "tenant": "  team-red " }))),
    )
    .await;
    assert_eq!(text(padded).await, "auth:team-red", "the claim is trimmed");
}

/// Another claim can be the tenant's, a namespaced one included.
#[tokio::test]
async fn the_tenant_claim_is_configurable() {
    let auth = serving_auth("https://claims.example.com/org").await;
    let token = key_set().token(json!({
        "https://claims.example.com/org": "acme",
        "tenant": "not-this-one",
    }));

    let response = resolve(&auth, bearer(&token)).await;

    assert_eq!(text(response).await, "auth:acme");
}

/// Each refusal is told as a category, the full reason staying in the log;
/// a refused JWT is never tried as a key.
#[tokio::test]
async fn a_refused_jwt_is_told_its_category() {
    let auth = serving_auth("tenant").await;
    let keys = key_set();
    let now = KeySet::now();

    for (token, told) in [
        (keys.token(json!({ "exp": now - 3600 })), "token expired"),
        (
            keys.token(json!({ "nbf": now + 600 })),
            "token not yet valid",
        ),
        (
            keys.token(json!({ "aud": "another-api" })),
            "audience not accepted",
        ),
        (
            keys.token(json!({ "iss": "https://other.example.com" })),
            "issuer not accepted",
        ),
        (
            keys.token_under("rotated-away", json!({})),
            "signing key not known",
        ),
        (format!("{}x", keys.token(json!({}))), "token rejected"),
        ("not.a.jwt".to_string(), "token rejected"),
        (keys.token(json!({ "tenant": 42 })), "tenant claim unusable"),
        (keys.token(json!({ "tenant": "" })), "tenant claim unusable"),
        (
            keys.token(json!({ "tenant": "t".repeat(201) })),
            "tenant claim unusable",
        ),
        (keys.token(json!({ "sub": Value::Null })), "no tenant claim"),
    ] {
        assert_refused(resolve(&auth, bearer(&token)).await, told).await;
    }
}

/// A bearer that is neither a key nor shaped like a JWT, and no bearer at
/// all, are refused as before the JWT check existed, the message naming
/// both credentials.
#[tokio::test]
async fn other_bearers_are_refused_as_keys() {
    let auth = serving_auth("tenant").await;

    let unknown = resolve(&auth, bearer("not-a-configured-key")).await;
    assert_eq!(unknown.status(), StatusCode::UNAUTHORIZED);
    let body: Value = serde_json::from_str(&text(unknown).await).unwrap();
    assert_eq!(body["error"]["code"], "invalid_api_key");

    let none = resolve(&auth, None).await;
    assert_eq!(none.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        none.headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer realm=\"data-plane\"")
    );
    let body: Value = serde_json::from_str(&text(none).await).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("API key") && message.contains("JWT")),
        "{body}"
    );
}

/// Through the real app: the serving routes take a JWT, refuse its absence
/// and a tampered one, and keep taking the key; the probes stay open.
#[tokio::test]
async fn the_serving_routes_take_a_jwt_or_a_key() {
    let mut config = TestRouterConfig::round_robin(4360);
    config.api_key = Some("shared-secret".to_string());
    config.data_plane_jwt = Some(DataPlaneJwtConfig::default());
    let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20360)]).await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane().await),
    );
    let keys = key_set();

    let generate = |authorization: Option<String>| {
        let app = app.clone();
        async move {
            let mut request = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(header::CONTENT_TYPE, "application/json");
            if let Some(authorization) = authorization {
                request = request.header(header::AUTHORIZATION, authorization);
            }
            let body = json!({ "text": "hello", "stream": false }).to_string();
            app.oneshot(request.body(Body::from(body)).unwrap())
                .await
                .unwrap()
        }
    };

    let jwt = generate(bearer(&keys.token(json!({ "tenant": "team-red" })))).await;
    assert_eq!(jwt.status(), StatusCode::OK, "{}", text(jwt).await);

    let key = generate(bearer("shared-secret")).await;
    assert_eq!(key.status(), StatusCode::OK, "{}", text(key).await);

    let none = generate(None).await;
    assert_eq!(none.status(), StatusCode::UNAUTHORIZED);

    assert_refused(
        generate(bearer(&format!("{}x", keys.token(json!({}))))).await,
        "token rejected",
    )
    .await;

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        health.status(),
        StatusCode::OK,
        "the probes need no credential"
    );

    ctx.shutdown().await;
}
