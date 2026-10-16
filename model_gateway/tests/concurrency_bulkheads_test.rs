//! The concurrency bulkheads through the whole HTTP stack: tenant
//! resolution, admission, the gateway's dispatch and a mock worker that
//! parks every request until released, so what is in flight is exact.
//!
//! | Refusal                | Status | Extra                                                          |
//! |------------------------|--------|----------------------------------------------------------------|
//! | tenant at its cap      | 429    | `X-SMG-Error-Code: tenant_concurrency_exceeded`, `Retry-After: 2` |
//! | model at its cap       | 503    | `X-SMG-Error-Code: model_concurrency_exceeded`, `Retry-After: 2`  |

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use portpicker::pick_unused_port;
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        set_scheduler_controls, HealthStatus, HoldGate, MockWorkerConfig, SchedulerControls,
        WorkerType,
    },
    AppTestContext,
};

/// The trusted tenant header, so each request names its tenant.
const TENANT_HEADER: &str = "x-smg-tenant-id";

/// A regular HTTP configuration with the given caps, trusting the tenant
/// header.
fn config(per_tenant: usize, per_model: usize) -> RouterConfig {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3002)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .max_concurrent_requests_per_tenant(per_tenant)
        .max_concurrent_requests_per_model(per_model)
        .queue_timeout_secs(60)
        .trust_tenant_header(true)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config
}

/// A mock worker whose `/generate` parks every request on `gate`.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn held_worker(gate: &Arc<HoldGate>) -> MockWorkerConfig {
    let port = pick_unused_port().expect("a free port for the mock worker");
    set_scheduler_controls(
        port,
        SchedulerControls {
            max_running_requests: None,
            hold_gate: Some(Arc::clone(gate)),
        },
    );
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

/// A non-streaming `/generate` by `tenant`. It names no model, so every
/// request counts under the same model key.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request(tenant: &str) -> Request<Body> {
    let payload = json!({ "text": "hello", "stream": false });
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .header(TENANT_HEADER, tenant)
        .body(Body::from(payload.to_string()))
        .expect("build request")
}

fn error_code(response: &axum::response::Response) -> Option<String> {
    response
        .headers()
        .get("x-smg-error-code")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn retry_after(response: &axum::response::Response) -> Option<String> {
    response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Drive a request to a response, bounded so a request that is admitted by
/// mistake (and parks on the gate) fails the test instead of hanging it.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn send(app: &axum::Router, request: Request<Body>) -> axum::response::Response {
    tokio::time::timeout(Duration::from_secs(8), app.clone().oneshot(request))
        .await
        .expect("the request did not complete in time (admitted instead of refused?)")
        .expect("the router service is infallible")
}

/// A request driven in the background: it parks at the worker until the
/// gate is released.
#[expect(
    clippy::disallowed_methods,
    reason = "test infra: holds a request open in a spawned task"
)]
fn send_in_background(
    app: &axum::Router,
    request: Request<Body>,
) -> tokio::task::JoinHandle<axum::response::Response> {
    let app = app.clone();
    tokio::spawn(async move {
        app.oneshot(request)
            .await
            .unwrap_or_else(|never| match never {})
    })
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn wait_arrivals(gate: &Arc<HoldGate>, n: u32) {
    tokio::time::timeout(Duration::from_secs(8), gate.wait_for_arrivals(n))
        .await
        .expect("the expected requests never reached the worker");
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn join(
    handle: tokio::task::JoinHandle<axum::response::Response>,
) -> axum::response::Response {
    tokio::time::timeout(Duration::from_secs(8), handle)
        .await
        .expect("the background request did not complete in time")
        .expect("the background request panicked")
}

/// One tenant at its cap is refused with 429 while another tenant's request
/// reaches the worker; once the held response ends, the tenant is served
/// again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_at_its_cap_is_held_while_another_tenant_is_served() {
    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config(1, 0), vec![held_worker(&gate)]).await;
    let app = ctx.create_app();

    let held = send_in_background(&app, generate_request("team-a"));
    wait_arrivals(&gate, 1).await;

    let refused = send(&app, generate_request("team-a")).await;
    assert_eq!(
        refused.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "the tenant's second in-flight request is refused"
    );
    assert_eq!(
        error_code(&refused).as_deref(),
        Some("tenant_concurrency_exceeded")
    );
    assert_eq!(retry_after(&refused).as_deref(), Some("2"));

    let other = send_in_background(&app, generate_request("team-b"));
    wait_arrivals(&gate, 1).await;

    gate.release();
    assert_eq!(join(held).await.status(), StatusCode::OK);
    assert_eq!(join(other).await.status(), StatusCode::OK);

    let again = send(&app, generate_request("team-a")).await;
    assert_eq!(
        again.status(),
        StatusCode::OK,
        "the slot came back with the held response"
    );

    ctx.shutdown().await;
}

/// The model at its cap refuses every tenant with 503 until a response for
/// it ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_at_its_cap_refuses_every_tenant_until_a_response_ends() {
    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config(0, 1), vec![held_worker(&gate)]).await;
    let app = ctx.create_app();

    let held = send_in_background(&app, generate_request("team-a"));
    wait_arrivals(&gate, 1).await;

    let refused = send(&app, generate_request("team-b")).await;
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error_code(&refused).as_deref(),
        Some("model_concurrency_exceeded")
    );
    assert_eq!(retry_after(&refused).as_deref(), Some("2"));

    gate.release();
    assert_eq!(join(held).await.status(), StatusCode::OK);

    let served = send(&app, generate_request("team-b")).await;
    assert_eq!(served.status(), StatusCode::OK);

    ctx.shutdown().await;
}

/// Without the caps, the default, the same traffic passes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_caps_the_same_traffic_passes() {
    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config(0, 0), vec![held_worker(&gate)]).await;
    let app = ctx.create_app();

    let first = send_in_background(&app, generate_request("team-a"));
    wait_arrivals(&gate, 1).await;
    let second = send_in_background(&app, generate_request("team-a"));
    wait_arrivals(&gate, 1).await;

    gate.release();
    assert_eq!(join(first).await.status(), StatusCode::OK);
    assert_eq!(join(second).await.status(), StatusCode::OK);

    ctx.shutdown().await;
}
