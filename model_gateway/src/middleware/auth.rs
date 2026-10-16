//! Bearer-token auth for the serving routes: API keys, keyed by SHA-256
//! hash of the presented token rather than the raw value, and, when
//! configured, bearer JWTs validated with the control plane's JWT
//! configuration ([`DataPlaneJwt`]).
//!
//! Hashing first — not a constant-time comparison — is what makes the key
//! lookup safe: SHA-256's avalanche property means a guess that's "close" to
//! a valid token produces an unrelated hash, so there's no byte-at-a-time
//! timing signal to recover from a plain hash-keyed lookup, unlike comparing
//! raw secrets.
//!
//! The order is key first, then JWT: the hash lookup costs one SHA-256 and
//! cannot mistake a key for a token, so a configured key goes on working
//! whatever its shape. A bearer in the shape of a JWT (three dot-separated
//! segments) that is no configured key is then validated as one, when the
//! JWT check is configured; a JWT that fails does not fall back to the key
//! table, and the reason the client is told is a category (expired, not yet
//! valid, audience, issuer, unknown key, rejected), the full reason going to
//! the log and the audit trail with the request id.

use std::{collections::HashMap, sync::Arc};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use smg_auth::{bearer_token, AuditLogger, ControlPlaneAuthState, JwtValidator};
use tracing::{error, info, warn};

use super::request_id::RequestId;
use crate::{
    config::{DataPlaneJwtConfig, TenantApiKeyEntry},
    tenant::{
        authenticated_tenant_key_from_sha256, DataPlaneCaller, TenantIdentity, TenantKey,
        MAX_TENANT_ID_LEN,
    },
};

#[derive(Clone)]
pub struct AuthConfig {
    keys: HashMap<[u8; 32], TenantKey>,
    jwt: Option<Arc<DataPlaneJwt>>,
}

impl AuthConfig {
    /// Single shared key; every caller resolves to the same hash-derived
    /// identity. Use [`Self::with_tenant_keys`] for per-tenant separation.
    pub fn new(api_key: Option<String>) -> Self {
        Self::with_tenant_keys(api_key, &[])
    }

    /// `tenant_api_keys` adds per-tenant keys on top of `api_key`, each
    /// resolving to its own `auth:<tenant_id>` identity.
    pub fn with_tenant_keys(
        api_key: Option<String>,
        tenant_api_keys: &[TenantApiKeyEntry],
    ) -> Self {
        let mut keys = HashMap::with_capacity(tenant_api_keys.len() + 1);

        if let Some(key) = api_key {
            let key_hash = hash_key(&key);
            keys.insert(key_hash, authenticated_tenant_key_from_sha256(key_hash));
        }

        for entry in tenant_api_keys {
            keys.insert(
                hash_key(&entry.key),
                TenantIdentity::Authenticated(entry.tenant_id.as_str().into()).into_key(),
            );
        }

        Self { keys, jwt: None }
    }

    /// The configuration with the data-plane JWT check attached when `config`
    /// asks for it. The validator is the control plane's, so both planes
    /// share one issuer, audience and JWKS cache; without one (no
    /// control-plane auth state, or a validator that did not initialise)
    /// the check fails closed, as the control plane's does: every bearer JWT
    /// is refused and the keys keep working. Without `config`, nothing
    /// changes: the serving check stays with the keys.
    pub fn with_data_plane_jwt(
        mut self,
        config: Option<&DataPlaneJwtConfig>,
        control_plane: Option<&ControlPlaneAuthState>,
    ) -> Self {
        let Some(config) = config else {
            return self;
        };
        let validator = control_plane.and_then(|state| state.jwt_validator.clone());
        let audit = control_plane
            .map(|state| state.audit_logger.clone())
            .unwrap_or_else(|| AuditLogger::new(true));
        match &validator {
            Some(validator) => info!(
                issuer = %validator.config().issuer,
                tenant_claim = %config.tenant_claim,
                "Data-plane JWT authentication enabled: bearer JWTs on the serving routes are \
                 validated with the control plane's JWT configuration"
            ),
            None => error!(
                "Data-plane JWT authentication is configured but the JWT validator is not \
                 available: every bearer JWT on the serving routes is refused until the gateway \
                 restarts with a working JWT configuration (the API keys keep working)"
            ),
        }
        self.jwt = Some(Arc::new(DataPlaneJwt::new(
            validator,
            config.tenant_claim.clone(),
            audit,
        )));
        self
    }

    /// Whether the serving routes require a credential: a key or the JWT
    /// check is configured. Neither means [`auth_middleware`] passes every
    /// request through.
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty() || self.jwt.is_some()
    }

    /// Whether `token` is one of the gateway's own credentials: a configured
    /// key (shared or per-tenant) or, with the JWT check on, a bearer in the
    /// shape of a JWT (validated on the serving routes; here the question is
    /// only whose credential it is). For callers outside `auth_middleware`
    /// that must tell "one of ours" from "a foreign token" — e.g.
    /// `/v1/models`' BYOK short-circuit, which must not treat a tenant key
    /// or a gateway JWT as a foreign upstream-provider credential and
    /// forward it externally.
    pub fn contains_token(&self, token: &str) -> bool {
        self.keys.contains_key(&hash_key(token)) || (self.jwt.is_some() && looks_like_jwt(token))
    }
}

/// The data-plane JWT check (`--data-plane-jwt`): the control plane's
/// validator, the claim that names the tenant, and the audit logger refusals
/// are reported to. See [`DataPlaneJwtConfig`] for what a validated token
/// resolves to.
pub struct DataPlaneJwt {
    validator: Option<Arc<JwtValidator>>,
    tenant_claim: String,
    audit: AuditLogger,
}

/// Why a bearer JWT was not admitted: what the client is told (a category,
/// see [`smg_auth::JwtRefusal`]) and what the log keeps.
struct JwtRejection {
    told: &'static str,
    detail: String,
}

impl DataPlaneJwt {
    /// A check with `validator` (none: every JWT is refused), `tenant_claim`
    /// as the tenant's claim and `audit` for the refusals.
    pub fn new(
        validator: Option<Arc<JwtValidator>>,
        tenant_claim: impl Into<String>,
        audit: AuditLogger,
    ) -> Self {
        Self {
            validator,
            tenant_claim: tenant_claim.into(),
            audit,
        }
    }

    /// The tenant key a bearer JWT resolves to, or the refusal.
    async fn authenticate(&self, token: &str) -> Result<TenantKey, JwtRejection> {
        let Some(validator) = &self.validator else {
            return Err(JwtRejection {
                told: "token rejected",
                detail: "JWT validation is unavailable: the validator did not initialise at \
                         start-up"
                    .to_string(),
            });
        };
        let validated = validator
            .validate_identity(token)
            .await
            .map_err(|e| JwtRejection {
                told: e.refusal().as_str(),
                detail: e.to_string(),
            })?;
        let (claim, value) = match validated.claim(&self.tenant_claim) {
            Some(value) => (self.tenant_claim.as_str(), value),
            None => match validated.claim("sub") {
                Some(sub) => ("sub", sub),
                None => {
                    return Err(JwtRejection {
                        told: "no tenant claim",
                        detail: format!(
                            "the token carries neither the `{}` claim nor `sub`",
                            self.tenant_claim
                        ),
                    })
                }
            },
        };
        let tenant = value
            .as_str()
            .map(str::trim)
            .filter(|tenant| !tenant.is_empty() && tenant.len() <= MAX_TENANT_ID_LEN)
            .ok_or_else(|| JwtRejection {
                told: "tenant claim unusable",
                detail: format!(
                    "the `{claim}` claim is not a string of 1 to {MAX_TENANT_ID_LEN} bytes"
                ),
            })?;
        Ok(TenantIdentity::Authenticated(Arc::from(tenant)).into_key())
    }

    /// One WARN line and one audit event for a refusal, with the request id.
    fn report(&self, request: &Request<Body>, rejection: &JwtRejection) {
        let method = request.method().as_str();
        let path = request.uri().path();
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.0.as_str());
        warn!(
            method,
            path,
            request_id,
            reason = %rejection.detail,
            "a bearer JWT on a serving route was refused ({})",
            rejection.told
        );
        self.audit.log_auth_failure(
            method,
            path,
            &format!("Invalid JWT on a serving route: {}", rejection.detail),
            request_id,
        );
    }
}

fn hash_key(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// Whether a bearer has the shape of a JWT: three dot-separated segments.
/// A shape, not a verdict; the validator has the verdict.
fn looks_like_jwt(token: &str) -> bool {
    token.split('.').count() == 3
}

/// Middleware to validate a Bearer credential against the configured API
/// key(s) and, when configured, as a JWT. Only active when the router has
/// at least one key or the JWT check configured.
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    if !auth_config.is_enabled() {
        return next.run(request).await;
    }

    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token);
    let Some(token) = token else {
        return unauthorized(Refusal::NoCredential {
            jwt: auth_config.jwt.is_some(),
        });
    };

    if let Some(tenant_key) = auth_config.keys.get(&hash_key(token)).cloned() {
        request
            .extensions_mut()
            .insert(DataPlaneCaller::new(tenant_key));
        return next.run(request).await;
    }

    if let Some(jwt) = &auth_config.jwt {
        if looks_like_jwt(token) {
            return match jwt.authenticate(token).await {
                Ok(tenant_key) => {
                    request
                        .extensions_mut()
                        .insert(DataPlaneCaller::new(tenant_key));
                    next.run(request).await
                }
                Err(rejection) => {
                    jwt.report(&request, &rejection);
                    unauthorized(Refusal::InvalidToken(rejection.told))
                }
            };
        }
    }

    unauthorized(Refusal::InvalidKey)
}

/// Unconditionally rejects with 401 — for route groups that must never fall
/// back to open just because their auth config happens to be empty.
pub async fn deny_all_middleware(request: Request<Body>, _next: Next) -> Response {
    let token_presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token)
        .is_some();
    unauthorized(if token_presented {
        Refusal::InvalidKey
    } else {
        Refusal::NoCredential { jwt: false }
    })
}

/// What a 401 says: no credential at all (`jwt`: whether a bearer JWT would
/// also do), a bearer that is no configured key, or a JWT refused for the
/// category given.
enum Refusal {
    NoCredential { jwt: bool },
    InvalidKey,
    InvalidToken(&'static str),
}

/// The serving plane's 401: the `WWW-Authenticate` challenge RFC 6750
/// requires (`error="invalid_token"` when a credential was presented and
/// refused) and an OpenAI-style error body, so SDK clients surface a
/// message instead of a blank error.
fn unauthorized(refusal: Refusal) -> Response {
    let (challenge, code, message) = match refusal {
        Refusal::NoCredential { jwt: false } => (
            "Bearer realm=\"data-plane\"",
            "invalid_api_key",
            "Missing API key: send it as `Authorization: Bearer <key>`".to_string(),
        ),
        Refusal::NoCredential { jwt: true } => (
            "Bearer realm=\"data-plane\"",
            "invalid_api_key",
            "Missing credential: send an API key or a bearer JWT as `Authorization: Bearer <token>`"
                .to_string(),
        ),
        Refusal::InvalidKey => (
            "Bearer realm=\"data-plane\", error=\"invalid_token\"",
            "invalid_api_key",
            "Invalid API key".to_string(),
        ),
        Refusal::InvalidToken(told) => (
            "Bearer realm=\"data-plane\", error=\"invalid_token\"",
            "invalid_token",
            format!("Invalid bearer token: {told}"),
        ),
    };
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, challenge)],
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": code,
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
        middleware::from_fn_with_state,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    use super::*;

    async fn handler(request: Request<Body>) -> impl IntoResponse {
        request
            .extensions()
            .get::<DataPlaneCaller>()
            .map(|caller| caller.tenant_key().to_string())
            .unwrap_or_else(|| "missing".to_string())
    }

    fn app(auth_config: AuthConfig) -> Router {
        Router::new()
            .route("/", get(handler))
            .route_layer(from_fn_with_state(auth_config, auth_middleware))
    }

    async fn body_text(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        std::str::from_utf8(&body).unwrap().to_string()
    }

    #[tokio::test]
    async fn no_key_configured_allows_any_request() {
        let response = app(AuthConfig::new(None))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn shared_key_resolves_to_hash_derived_tenant() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));
        let expected = authenticated_tenant_key_from_sha256(hash_key("shared-secret"));

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer shared-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, expected.to_string());
    }

    #[tokio::test]
    async fn tenant_key_resolves_to_explicit_tenant_id() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer team-red-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "auth:team-red");
    }

    #[tokio::test]
    async fn distinct_tenant_keys_resolve_to_distinct_tenants() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[
                TenantApiKeyEntry {
                    tenant_id: "team-red".to_string(),
                    key: "red-secret".to_string(),
                },
                TenantApiKeyEntry {
                    tenant_id: "team-blue".to_string(),
                    key: "blue-secret".to_string(),
                },
            ],
        );

        for (key, expected_tenant) in [
            ("red-secret", "auth:team-red"),
            ("blue-secret", "auth:team-blue"),
        ] {
            let response = app(auth_config.clone())
                .oneshot(
                    Request::builder()
                        .uri("/")
                        .header(header::AUTHORIZATION, format!("Bearer {key}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(body_text(response).await, expected_tenant);
        }
    }

    #[tokio::test]
    async fn unknown_token_is_rejected() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer not-a-configured-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_header_is_rejected_when_keys_configured() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));

        let response = app(auth_config)
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn bearer_scheme_is_matched_case_insensitively() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));

        for scheme in ["bearer", "BEARER"] {
            let response = app(auth_config.clone())
                .oneshot(
                    Request::builder()
                        .uri("/")
                        .header(header::AUTHORIZATION, format!("{scheme} shared-secret"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK, "scheme {scheme}");
        }
    }

    fn challenge(response: &Response) -> String {
        response
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    }

    #[tokio::test]
    async fn missing_key_gets_a_bearer_challenge_and_a_message() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(challenge(&response), "Bearer realm=\"data-plane\"");
        let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
        assert_eq!(body["error"]["code"], "invalid_api_key");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| !message.is_empty()),
            "{body}"
        );
    }

    #[tokio::test]
    async fn unknown_key_gets_an_invalid_token_challenge() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer not-a-configured-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            challenge(&response),
            "Bearer realm=\"data-plane\", error=\"invalid_token\""
        );
    }

    #[tokio::test]
    async fn deny_all_carries_the_challenge() {
        let response = Router::new()
            .route("/", get(handler))
            .route_layer(axum::middleware::from_fn(deny_all_middleware))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(challenge(&response), "Bearer realm=\"data-plane\"");
    }

    #[test]
    fn contains_token_recognizes_shared_and_tenant_keys() {
        let auth_config = AuthConfig::with_tenant_keys(
            Some("shared-secret".to_string()),
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        assert!(auth_config.contains_token("shared-secret"));
        assert!(auth_config.contains_token("team-red-secret"));
        assert!(!auth_config.contains_token("not-a-configured-key"));
    }

    /// With the JWT check off, a bearer in the shape of a JWT is a key like
    /// any other: refused as an invalid key, never validated.
    #[tokio::test]
    async fn a_jwt_shaped_bearer_is_an_invalid_key_when_the_check_is_off() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer eyJhbGciOiJSUzI1NiJ9.e30.c2ln",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
        assert_eq!(body["error"]["code"], "invalid_api_key");
    }

    /// The JWT check configured without a validator fails closed: a JWT is
    /// refused, a key keeps working, a configured key with the shape of a JWT
    /// is found before the JWT check runs, and no credential at all is
    /// refused even without any key.
    #[tokio::test]
    async fn the_jwt_check_without_a_validator_refuses_jwts_and_keeps_the_keys() {
        use crate::config::DataPlaneJwtConfig;

        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "red.secret.key".to_string(),
            }],
        )
        .with_data_plane_jwt(Some(&DataPlaneJwtConfig::default()), None);
        assert!(auth_config.is_enabled());
        assert!(auth_config.contains_token("red.secret.key"));
        assert!(
            auth_config.contains_token("eyJhbGciOiJSUzI1NiJ9.e30.c2ln"),
            "a JWT-shaped bearer is one of ours when the check is on"
        );
        assert!(!auth_config.contains_token("sk-foreign-provider-key"));

        let call = |authorization: Option<&'static str>| {
            let app = app(auth_config.clone());
            async move {
                let mut request = Request::builder().uri("/");
                if let Some(authorization) = authorization {
                    request = request.header(header::AUTHORIZATION, authorization);
                }
                app.oneshot(request.body(Body::empty()).unwrap())
                    .await
                    .unwrap()
            }
        };

        let keyed = call(Some("Bearer red.secret.key")).await;
        assert_eq!(keyed.status(), StatusCode::OK);
        assert_eq!(body_text(keyed).await, "auth:team-red");

        let jwt = call(Some("Bearer eyJhbGciOiJSUzI1NiJ9.e30.c2ln")).await;
        assert_eq!(jwt.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            challenge(&jwt),
            "Bearer realm=\"data-plane\", error=\"invalid_token\""
        );
        let body: serde_json::Value = serde_json::from_str(&body_text(jwt).await).unwrap();
        assert_eq!(body["error"]["code"], "invalid_token");
        assert_eq!(
            body["error"]["message"],
            "Invalid bearer token: token rejected"
        );

        let none = call(None).await;
        assert_eq!(none.status(), StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = serde_json::from_str(&body_text(none).await).unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("bearer JWT")),
            "{body}"
        );

        let no_key =
            AuthConfig::new(None).with_data_plane_jwt(Some(&DataPlaneJwtConfig::default()), None);
        let response = app(no_key)
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "the JWT check alone requires a credential"
        );
    }
}
