//! A key set for the JWT tests: one RSA key pair generated once per test
//! binary, its public half served as a JWKS document by a thread on the
//! loopback (plain HTTP, which the validator admits for the loopback only),
//! and tokens signed with the private half. Included by path from the tests
//! that run the gateway as a process, and as `common::jwks` elsewhere.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::LazyLock,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use rsa::{
    pkcs8::{EncodePrivateKey, LineEnding},
    rand_core::OsRng,
    traits::PublicKeyParts,
    RsaPrivateKey,
};
use serde_json::{json, Value};

/// The issuer the test tokens name and the validator expects.
pub const ISSUER: &str = "https://issuer.example.com";
/// The audience the test tokens name and the validator expects.
pub const AUDIENCE: &str = "smg-serving";
/// The id of the published key.
pub const KID: &str = "serving-key-1";

/// The key pair, the JWKS document and the URI it is served at.
pub struct KeySet {
    private_key_pem: String,
    /// The JWKS URI (`http://127.0.0.1:<port>/jwks.json`).
    pub uri: String,
}

static KEY_SET: LazyLock<KeySet> = LazyLock::new(KeySet::start);

/// The binary's key set, generated and served on first use.
pub fn key_set() -> &'static KeySet {
    &KEY_SET
}

impl KeySet {
    fn start() -> Self {
        let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("an RSA key pair");
        let jwks = json!({
            "keys": [{
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": KID,
                "n": URL_SAFE_NO_PAD.encode(private_key.n().to_bytes_be()),
                "e": URL_SAFE_NO_PAD.encode(private_key.e().to_bytes_be()),
            }]
        })
        .to_string();
        let private_key_pem = private_key
            .to_pkcs8_pem(LineEnding::LF)
            .expect("the key as PEM")
            .to_string();
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let uri = format!(
            "http://127.0.0.1:{}/jwks.json",
            listener.local_addr().unwrap().port()
        );
        thread::spawn(move || serve(listener, jwks));
        Self {
            private_key_pem,
            uri,
        }
    }

    /// Seconds since the epoch, now.
    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// A token signed by the published key: the issuer, the audience, the
    /// subject `alice` and an hour of validity, with `claims` on top
    /// (overriding any of them).
    pub fn token(&self, claims: Value) -> String {
        self.token_under(KID, claims)
    }

    /// [`Self::token`] with `kid` in its header: a key id the set does not
    /// publish makes a token the validator finds no key for.
    pub fn token_under(&self, kid: &str, claims: Value) -> String {
        let now = Self::now();
        let mut all = json!({
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "alice",
            "iat": now,
            "exp": now + 3600,
        });
        for (name, value) in claims.as_object().expect("an object of claims") {
            all[name] = value.clone();
        }
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.to_string());
        let key = EncodingKey::from_rsa_pem(self.private_key_pem.as_bytes()).expect("the key");
        encode(&header, &all, &key).expect("a signed token")
    }
}

/// Answers every request on `listener` with the JWKS document, one
/// connection at a time; the document is small and the fetches are few.
fn serve(listener: TcpListener, jwks: String) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        answer(stream, &jwks);
    }
}

fn answer(mut stream: TcpStream, jwks: &str) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|window| window == b"\r\n\r\n") {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => head.extend_from_slice(&buf[..n]),
        }
    }
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        jwks.len(),
        jwks
    );
    let _ = stream.flush();
}
