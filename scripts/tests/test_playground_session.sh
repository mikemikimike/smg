#!/bin/bash
# Tests for scripts/playground/session.sh: joining the tailnet, getting the
# session ready, the timed hold, and teardown.
#
#   bash scripts/tests/test_playground_session.sh
#
# `tailscale`, `tailscaled` and `curl` are fakes that record their arguments,
# so nothing joins a real tailnet. Runs on Linux and macOS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/playground/session.sh"
SECRET="tskey-client-kTEST-0123456789abcdef"

fail() { echo "FAIL: $*"; exit 1; }
assert_eq() { [ "$1" = "$2" ] || fail "${3:-value}: expected '$1', got '$2'"; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "expected to contain '$2', got: $1"; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || fail "expected not to contain '$2'"; }

cleanup_sandbox() {
    [ -n "${T:-}" ] || return 0
    if [ -f "$T/state/tailscaled.pid" ]; then kill "$(cat "$T/state/tailscaled.pid")" 2> /dev/null || true; fi
    rm -rf "$T"
}
trap cleanup_sandbox EXIT

write_fakes() {
    local dir="$1"
    mkdir -p "$dir"
    # session.sh starts tailscaled with an empty environment, so this fake
    # gets its log paths baked in instead of reading FAKE_* variables.
    cat > "$dir/tailscaled" <<EOF
#!/bin/bash
echo "tailscaled \$*" >> "$T/log"
env > "$T/tailscaled.env"
for a in "\$@"; do case "\$a" in --socket=*) touch "\${a#--socket=}" ;; esac; done
exec sleep 300
EOF
    cat > "$dir/tailscale" <<'EOF'
#!/bin/bash
echo "tailscale $*" >> "$FAKE_LOG"
for a in "$@"; do
    case "$a" in
        --auth-key=file:*) cat "${a#--auth-key=file:}" > "$FAKE_AUTHKEY_SEEN" ;;
    esac
done
case " $* " in
    *" ip -4 "*) echo 100.64.0.7 ;;
    *" up "*) [ -z "${FAKE_UP_FAIL:-}" ] || exit 1 ;;
    *" status --json --peers=false "*)
        printf '{\n  "Self": {\n    "DNSName": "%s.tail1234.ts.net.",\n    "HostName": "x"\n  }\n}\n' "${FAKE_DNSNAME:-smg-pg-1-2}" ;;
esac
EOF
    chmod +x "$dir/tailscaled" "$dir/tailscale"
}

# setup: fresh sandbox with fake Tailscale binaries, a home dir, a workspace
# with a venv, and the environment a session job step would have.
setup() {
    cleanup_sandbox
    T="$(mktemp -d)"
    write_fakes "$T/tsbin"
    mkdir -p "$T/home" "$T/ws/.venv/bin" "$T/state"
    printf '# existing bashrc\ncase $- in *i*) ;; *) return;; esac\n' > "$T/home/.bashrc"
    BASE_ENV=(PATH="/usr/bin:/bin" HOME="$T/home" RUNNER_TEMP="$T"
        FAKE_LOG="$T/log" FAKE_AUTHKEY_SEEN="$T/authkey.seen"
        GITHUB_STEP_SUMMARY="$T/summary" GITHUB_TOKEN=ghs_jobtoken ACTIONS_ID_TOKEN_REQUEST_TOKEN=oidc
        HF_HOME=/models HF_TOKEN=hf_secret CUDA_HOME=/usr/local/cuda
        PG_STATE_DIR="$T/state" PG_HOME="$T/home" PG_WORKSPACE="$T/ws" PG_TS_BIN="$T/tsbin"
        PG_HOSTNAME=smg-pg-1-2 PG_HOURS=1 PG_GPUS=4 PG_SOURCE_LABEL="main @ 1111111"
        PG_SMG_STATUS=installed PG_ENGINE=sglang PG_ENGINE_STATUS=installed)
    : > "$T/log"
}

session() { env -i "${BASE_ENV[@]}" "$@"; }
# join with the secret, then ready without it, as the workflow does.
up() {
    session PG_TS_AUTHKEY="$SECRET" "$@" bash "$SCRIPT" join || fail "join failed"
    session "$@" bash "$SCRIPT" ready || fail "ready failed"
}

test_up_starts_userspace_tailscaled_and_joins() {
    setup
    up > "$T/up.out" 2>&1 || { cat "$T/up.out"; fail "up exited non-zero"; }
    local log; log="$(cat "$T/log")"
    assert_contains "$log" "--tun=userspace-networking"
    assert_contains "$log" "--port=0"
    assert_contains "$log" "--no-logs-no-support"
    assert_contains "$log" "--advertise-tags=tag:playground"
    assert_contains "$log" "--hostname=smg-pg-1-2"
    assert_contains "$log" "--ssh"
    assert_contains "$log" "--auth-key=file:"
    assert_eq "${SECRET}?ephemeral=false&preauthorized=true" "$(cat "$T/authkey.seen")" "auth key handed to tailscale"
}

test_secret_left_nowhere_on_disk() {
    setup
    up > /dev/null 2>&1
    rm -f "$T/authkey.seen" "$T/log"
    if grep -rqF "$SECRET" "$T"; then fail "secret left on disk: $(grep -rlF "$SECRET" "$T")"; fi
}

test_tailscaled_gets_scrubbed_environment() {
    setup
    up > /dev/null 2>&1
    local envdump; envdump="$(cat "$T/tailscaled.env")"
    assert_not_contains "$envdump" "ghs_jobtoken"
    assert_not_contains "$envdump" "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
    assert_not_contains "$envdump" "$SECRET"
}

test_summary_tells_how_to_connect() {
    setup
    up > /dev/null 2>&1
    local summary; summary="$(cat "$T/summary")"
    assert_contains "$summary" "ssh runner@smg-pg-1-2"
    assert_contains "$summary" "4× H100"
    assert_contains "$summary" "main @ 1111111"
    assert_contains "$summary" "sglang (installed)"
    assert_contains "$summary" "playground-done"
}

test_env_file_is_clean_and_usable() {
    setup
    up > /dev/null 2>&1
    local envfile="$T/home/.playground/env.sh"
    bash -n "$envfile" || fail "env.sh has a syntax error"
    local content; content="$(cat "$envfile")"
    assert_not_contains "$content" "ghs_jobtoken"
    assert_not_contains "$content" "ACTIONS_"
    assert_not_contains "$content" "GITHUB_TOKEN"
    assert_not_contains "$content" "$SECRET"
    local got
    got="$(env -i HOME="$T/home" PATH=/usr/bin:/bin bash -c ". '$envfile'; echo \"\$HF_HOME|\$HF_TOKEN|\$PATH\"")"
    assert_contains "$got" "/models|hf_secret|"
    assert_contains "$got" "$T/ws/.venv/bin"
    assert_contains "$got" "$T/state/bin"
}

test_shell_hooks_added_once() {
    setup
    up > /dev/null 2>&1
    up > /dev/null 2>&1
    assert_eq 1 "$(grep -c 'playground/env.sh' "$T/home/.bashrc")" ".bashrc hooks"
    assert_eq 1 "$(grep -c 'playground/env.sh' "$T/home/.profile")" ".profile hooks"
    # The hook must run before an existing early return for non-interactive shells.
    assert_contains "$(head -1 "$T/home/.bashrc")" "playground/env.sh"
}

test_missing_secret_fails_before_starting() {
    setup
    if session PG_TS_AUTHKEY= bash "$SCRIPT" join > "$T/up.out" 2>&1; then fail "join succeeded without a secret"; fi
    assert_contains "$(cat "$T/up.out")" "PLAYGROUND_TS_AUTHKEY_SECRET"
    assert_not_contains "$(cat "$T/log")" "tailscaled"
}

test_failed_join_fails_up() {
    setup
    if session PG_TS_AUTHKEY="$SECRET" FAKE_UP_FAIL=1 bash "$SCRIPT" join > /dev/null 2>&1; then fail "join succeeded after a failed tailscale up"; fi
    [ ! -e "$T/state/authkey" ] || fail "auth key file left after a failed join"
}

test_download_checks_sha256() {
    setup
    mkdir -p "$T/pkg/tailscale_9.9.9_amd64"
    write_fakes "$T/pkg/tailscale_9.9.9_amd64"
    tar -czf "$T/ts.tgz" -C "$T/pkg" tailscale_9.9.9_amd64
    mkdir -p "$T/curlbin"
    cat > "$T/curlbin/curl" <<EOF
#!/bin/bash
while [ \$# -gt 0 ]; do [ "\$1" = -o ] && { cp "$T/ts.tgz" "\$2"; exit 0; }; shift; done
exit 1
EOF
    chmod +x "$T/curlbin/curl"
    local good
    good="$( (sha256sum "$T/ts.tgz" 2> /dev/null || shasum -a 256 "$T/ts.tgz") | cut -d' ' -f1)"

    if session PATH="$T/curlbin:/usr/bin:/bin" PG_TS_BIN= PG_TS_VERSION=9.9.9 PG_TS_SHA256=deadbeef \
        PG_TS_AUTHKEY="$SECRET" bash "$SCRIPT" join > "$T/up.out" 2>&1; then
        fail "up accepted a tarball with the wrong checksum"
    fi
    assert_not_contains "$(cat "$T/log")" "tailscaled"

    session PATH="$T/curlbin:/usr/bin:/bin" PG_TS_BIN= PG_TS_VERSION=9.9.9 PG_TS_SHA256="$good" \
        PG_TS_AUTHKEY="$SECRET" bash "$SCRIPT" join > "$T/up.out" 2>&1 || { cat "$T/up.out"; fail "join rejected a good tarball"; }
    assert_contains "$(cat "$T/log")" "--tun=userspace-networking"
}

test_join_refuses_a_taken_machine_name() {
    setup
    if session PG_TS_AUTHKEY="$SECRET" FAKE_DNSNAME=smg-pg-1-2-1 bash "$SCRIPT" join > "$T/up.out" 2>&1; then
        fail "join accepted a renamed machine"
    fi
    assert_contains "$(cat "$T/up.out")" "smg-pg-1-2-1"
}

test_ready_runs_without_the_secret() {
    setup
    session PG_TS_AUTHKEY="$SECRET" bash "$SCRIPT" join > /dev/null 2>&1
    session bash "$SCRIPT" ready > "$T/ready.out" 2>&1 || { cat "$T/ready.out"; fail "ready needed the secret"; }
    assert_contains "$(cat "$T/summary")" "ssh runner@smg-pg-1-2"
}

test_hold_returns_at_deadline() {
    setup
    up PG_DURATION_SECONDS=2 > /dev/null 2>&1
    local start=$SECONDS
    session PG_POLL_SECONDS=1 bash "$SCRIPT" hold > "$T/hold.out" 2>&1
    [ $((SECONDS - start)) -le 5 ] || fail "hold ran $((SECONDS - start))s past a 2s deadline"
}

test_playground_done_ends_hold_early() {
    setup
    up PG_DURATION_SECONDS=600 > /dev/null 2>&1
    (sleep 1 && env -i PATH=/usr/bin:/bin "$T/state/bin/playground-done" > /dev/null) &
    local start=$SECONDS
    session PG_POLL_SECONDS=1 bash "$SCRIPT" hold > "$T/hold.out" 2>&1
    wait
    [ $((SECONDS - start)) -le 4 ] || fail "hold ignored playground-done for $((SECONDS - start))s"
}

test_hold_warns_open_terminals() {
    setup
    up PG_DURATION_SECONDS=2 > /dev/null 2>&1
    mkdir -p "$T/pts"; : > "$T/pts/0"
    session PG_POLL_SECONDS=1 PG_PTS_GLOB="$T/pts/*" bash "$SCRIPT" hold > /dev/null 2>&1
    assert_contains "$(cat "$T/pts/0")" "Playground ends in about 1 minute"
}

test_status_reports_time_left() {
    setup
    up PG_DURATION_SECONDS=5430 > /dev/null 2>&1
    local out; out="$(env -i PATH=/usr/bin:/bin "$T/state/bin/playground-status")"
    assert_contains "$out" "1h 30m left"
    assert_contains "$out" "UTC"
}

test_down_tolerates_stopped_tailscaled() {
    setup
    up > /dev/null 2>&1
    kill "$(cat "$T/state/tailscaled.pid")"
    session bash "$SCRIPT" down > /dev/null 2>&1 || fail "down failed with tailscaled already gone"
    assert_contains "$(cat "$T/log")" "logout"
}

test_down_without_up_succeeds() {
    setup
    session bash "$SCRIPT" down > /dev/null 2>&1 || fail "down failed when up never ran"
}

for t in $(declare -F | awk '{print $3}' | grep '^test_'); do
    "$t"
    echo "ok  $t"
done
echo "ALL PASS"
