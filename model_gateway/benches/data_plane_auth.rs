//! The serving routes' authentication, per request: no credential required,
//! an API key (one SHA-256 and a map lookup) and a bearer JWT (RS256 over a
//! cached key set), each as one request through `auth_middleware` on an
//! echo route, so the three cells differ by the credential check alone.
//!
//! Run: `cargo bench -p smg --bench data_plane_auth`
#![expect(
    clippy::expect_used,
    reason = "a benchmark's setup fails loudly or not at all"
)]

#[path = "../tests/common/jwks.rs"]
mod jwks;

use std::hint::black_box;

use axum::{
    body::Body, extract::Request, http::header, middleware::from_fn_with_state, routing::get,
    Router,
};
use criterion::{criterion_group, criterion_main, Criterion};
use jwks::{key_set, AUDIENCE, ISSUER};
use serde_json::json;
use smg::{
    config::{DataPlaneJwtConfig, TenantApiKeyEntry},
    middleware::{auth_middleware, AuthConfig},
};
use smg_auth::{ControlPlaneAuthConfig, ControlPlaneAuthState, JwtConfig};
use tokio::runtime::Runtime;
use tower::ServiceExt;

fn app(auth: AuthConfig) -> Router {
    Router::new()
        .route("/", get(|| async { "ok" }))
        .route_layer(from_fn_with_state(auth, auth_middleware))
}

fn request(authorization: Option<&str>) -> Request<Body> {
    let mut request = Request::builder().uri("/");
    if let Some(authorization) = authorization {
        request = request.header(header::AUTHORIZATION, authorization);
    }
    request.body(Body::empty()).expect("a request")
}

fn bench(c: &mut Criterion) {
    let rt = Runtime::new().expect("a runtime");
    let keys = key_set();
    let control_plane = rt.block_on(async {
        ControlPlaneAuthState::from_config(ControlPlaneAuthConfig {
            jwt: Some(JwtConfig::new(ISSUER, AUDIENCE).with_jwks_uri(keys.uri.clone())),
            api_keys: vec![],
            audit_enabled: false,
        })
        .await
        .expect("a validator over the key set")
    });
    let tenant_keys = [TenantApiKeyEntry {
        tenant_id: "team-red".to_string(),
        key: "red-secret".to_string(),
    }];
    let open = app(AuthConfig::new(None));
    let keyed = app(AuthConfig::with_tenant_keys(None, &tenant_keys));
    let jwt = app(AuthConfig::with_tenant_keys(None, &tenant_keys)
        .with_data_plane_jwt(Some(&DataPlaneJwtConfig::default()), Some(&control_plane)));
    let token = format!("Bearer {}", keys.token(json!({ "tenant": "team-red" })));
    // The first validation fetches the key set; the cells measure the cached path.
    let warm = rt.block_on(jwt.clone().oneshot(request(Some(&token))));
    assert_eq!(warm.expect("a response").status(), 200);

    let mut group = c.benchmark_group("data_plane_auth");
    group.bench_function("no_credential_required", |b| {
        b.iter(|| {
            let response = rt.block_on(open.clone().oneshot(request(None)));
            black_box(response.expect("a response").status())
        });
    });
    group.bench_function("api_key", |b| {
        b.iter(|| {
            let response = rt.block_on(keyed.clone().oneshot(request(Some("Bearer red-secret"))));
            black_box(response.expect("a response").status())
        });
    });
    group.bench_function("jwt_rs256_cached_jwks", |b| {
        b.iter(|| {
            let response = rt.block_on(jwt.clone().oneshot(request(Some(&token))));
            black_box(response.expect("a response").status())
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
