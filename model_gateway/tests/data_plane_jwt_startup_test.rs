//! `--data-plane-jwt` in the binary: a bearer JWT from the configured issuer
//! passes the serving routes' authentication, no credential and a tampered
//! token are refused with the data-plane challenge, and the flag without a
//! JWT issuer is refused before anything binds. The gateway runs as a
//! process on purpose: the validator is built at start-up from the control
//! plane's flags and handed to the serving check there.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;
#[path = "common/jwks.rs"]
mod jwks;

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, launch, serial, Until, EXIT_TIMEOUT, IP};
use jwks::{key_set, AUDIENCE, ISSUER};
use serde_json::json;

/// A running gateway, killed when dropped.
struct Gateway {
    child: Child,
    log: PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args`, its output in `log`, and waits until
    /// its health route answers on `addr`.
    fn start(args: &[String], addr: SocketAddr, log: &Path) -> Self {
        let file = File::create(log).expect("log file");
        let child = Command::new(env!("CARGO_BIN_EXE_smg"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(file.try_clone().expect("log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("spawn the gateway");
        let mut gateway = Self {
            child,
            log: log.to_path_buf(),
        };
        let deadline = Instant::now() + EXIT_TIMEOUT;
        loop {
            if let Some(status) = gateway.child.try_wait().expect("poll the gateway") {
                panic!(
                    "the gateway exited ({status}) before serving\n{}",
                    gateway.log()
                );
            }
            if let Some((status, _)) = http(addr, "GET", "/health", None, "") {
                if status == 200 {
                    return gateway;
                }
            }
            assert!(
                Instant::now() < deadline,
                "the gateway did not serve within {EXIT_TIMEOUT:?}\n{}",
                gateway.log()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One HTTP/1.1 request over a fresh connection: the status and the raw
/// response (head and body), `None` when the gateway does not answer.
fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    authorization: Option<&str>,
    body: &str,
) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        body.len()
    );
    if let Some(authorization) = authorization {
        head.push_str(&format!("Authorization: {authorization}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).ok()?;
    stream.write_all(body.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let status = response.strip_prefix("HTTP/1.1 ")?.get(..3)?.parse().ok()?;
    Some((status, response))
}

fn temp_log(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "smg-data-plane-jwt-{}-{name}.log",
        std::process::id()
    ))
}

/// The flags of a gateway whose serving routes take the test key set's
/// tokens: the control plane's JWT configuration pointed at the key set's
/// JWKS URI, and the data-plane switch with `tenant` as the tenant claim.
fn jwt_args(port: u16, metrics_port: u16) -> Vec<String> {
    let mut args = base_args(port, metrics_port);
    args.extend(
        [
            "--jwt-issuer",
            ISSUER,
            "--jwt-audience",
            AUDIENCE,
            "--jwt-jwks-uri",
            &key_set().uri,
            "--data-plane-jwt",
            "--jwt-tenant-claim",
            "tenant",
        ]
        .map(str::to_string),
    );
    args
}

#[test]
fn a_bearer_jwt_passes_the_serving_routes_of_the_binary() {
    let _serial = serial();
    let [port, metrics_port] = free_ports();
    let log = temp_log("serving");
    let gateway = Gateway::start(
        &jwt_args(port, metrics_port),
        SocketAddr::from((IP, port)),
        &log,
    );
    let addr = SocketAddr::from((IP, port));
    let body =
        json!({ "model": "any", "messages": [{ "role": "user", "content": "hi" }] }).to_string();
    let chat = |authorization: Option<&str>| {
        http(addr, "POST", "/v1/chat/completions", authorization, &body)
            .unwrap_or_else(|| panic!("no answer from the gateway\n{}", gateway.log()))
    };

    // A token the configured issuer signed: past authentication (the route
    // then finds no worker to serve the model, which is not 401).
    let token = key_set().token(json!({ "tenant": "team-red" }));
    let (status, response) = chat(Some(&format!("Bearer {token}")));
    assert_ne!(status, 401, "a valid JWT is authenticated\n{response}");

    let (status, response) = chat(None);
    assert_eq!(status, 401, "no credential\n{response}");
    assert!(
        response
            .to_ascii_lowercase()
            .contains("www-authenticate: bearer realm=\"data-plane\""),
        "the data-plane challenge\n{response}"
    );

    let (status, response) = chat(Some(&format!("Bearer {token}x")));
    assert_eq!(status, 401, "a tampered token\n{response}");
    assert!(
        response.contains("error=\"invalid_token\"")
            && response.contains("Invalid bearer token: token rejected"),
        "the refusal's category\n{response}"
    );

    let (status, _) = http(addr, "GET", "/health", None, "").expect("the probe answers");
    assert_eq!(status, 200, "the probes need no credential");

    let gateway_log = gateway.log();
    assert!(
        gateway_log.contains("Data-plane JWT authentication enabled"),
        "the start names the check\n{gateway_log}"
    );
}

/// The flag without the JWT configuration it validates with is refused
/// before any listener binds, naming the issuer flag.
#[test]
fn the_flag_without_a_jwt_issuer_is_refused_before_binding() {
    let launch = launch(&["--data-plane-jwt"], Until::Bound, &temp_log("refused"));

    assert!(
        launch.bound.is_none(),
        "a port answered although the flag cannot be honoured\n{}",
        launch.log
    );
    let status = launch.status.unwrap_or_else(|| {
        panic!(
            "the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{}",
            launch.log
        )
    });
    assert!(!status.success(), "{}", launch.log);
    assert!(
        launch.log.contains("--jwt-issuer"),
        "the error names the issuer flag\n{}",
        launch.log
    );
}
