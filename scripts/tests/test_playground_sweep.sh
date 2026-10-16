#!/bin/bash
# Tests for scripts/playground/sweep.sh: which tailnet machines get removed.
#
#   bash scripts/tests/test_playground_sweep.sh
#
# `curl` is a fake that serves a device list and records deletions, so no
# Tailscale API access is needed. Needs jq. Runs on Linux and macOS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/playground/sweep.sh"

fail() { echo "FAIL: $*"; exit 1; }
assert_eq() { [ "$1" = "$2" ] || fail "${3:-value}: expected '$1', got '$2'"; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"

iso_ago() {
    local e=$(($(date +%s) - $1))
    date -u -d "@$e" '+%Y-%m-%dT%H:%M:%SZ' 2> /dev/null || date -u -r "$e" '+%Y-%m-%dT%H:%M:%SZ'
}

cat > "$T/devices.json" <<EOF
{"devices": [
  {"id": "stale", "hostname": "smg-pg-1-1", "tags": ["tag:playground"], "connectedToControl": false, "lastSeen": "$(iso_ago 7200)"},
  {"id": "stale2", "hostname": "smg-pg-5-1", "tags": ["tag:playground"], "connectedToControl": false, "lastSeen": "$(iso_ago 10800)"},
  {"id": "recent", "hostname": "smg-pg-2-1", "tags": ["tag:playground"], "connectedToControl": false, "lastSeen": "$(iso_ago 60)"},
  {"id": "online", "hostname": "smg-pg-3-1", "tags": ["tag:playground"], "connectedToControl": true, "lastSeen": "$(iso_ago 90000)"},
  {"id": "untagged", "hostname": "laptop", "connectedToControl": false, "lastSeen": "$(iso_ago 90000)"},
  {"id": "unknown", "hostname": "smg-pg-4-1", "tags": ["tag:playground"], "lastSeen": "$(iso_ago 90000)"}
]}
EOF

cat > "$T/bin/curl" <<EOF
#!/bin/bash
args="\$*"
case "\$args" in
    *oauth/token*) [ -z "\${FAKE_TOKEN_FAIL:-}" ] || exit 22; echo '{"access_token":"tok","token_type":"Bearer"}' ;;
    *"-X DELETE"*)
        [[ "\${@: -1}" != */"\${FAKE_DELETE_FAIL:-none}" ]] || exit 22
        echo "\${@: -1}" >> "$T/deletes" ;;
    *tailnet/-/devices*) [[ "\$args" == *"Bearer tok"* ]] || exit 22; cat "$T/devices.json" ;;
    *) exit 3 ;;
esac
EOF
chmod +x "$T/bin/curl"

run_sweep() {
    : > "$T/deletes"
    env -i PATH="$T/bin:$(dirname "$(command -v jq)"):/usr/bin:/bin" \
        PG_TS_SWEEP_SECRET=tskey-client-sweep PG_TS_API=https://api.example "$@" \
        bash "$SCRIPT" > "$T/out" 2>&1
}

test_removes_only_stale_disconnected_playground_machines() {
    run_sweep || { cat "$T/out"; fail "sweep exited non-zero"; }
    assert_eq "https://api.example/api/v2/device/stale https://api.example/api/v2/device/stale2" \
        "$(sort "$T/deletes" | tr '\n' ' ' | sed 's/ $//')" "deleted devices"
}

test_one_failed_delete_does_not_stop_the_rest() {
    if run_sweep FAKE_DELETE_FAIL=stale; then fail "sweep reported success with a failed delete"; fi
    assert_eq "https://api.example/api/v2/device/stale2" "$(cat "$T/deletes")" "deleted devices"
    [[ "$(cat "$T/out")" == *"::warning::"*"smg-pg-1-1"* ]] || fail "no warning names the machine left behind"
}

test_token_failure_fails_the_run() {
    if run_sweep FAKE_TOKEN_FAIL=1; then fail "sweep succeeded without a token"; fi
    assert_eq "" "$(cat "$T/deletes")" "deleted devices"
}

test_missing_secret_fails_the_run() {
    if run_sweep PG_TS_SWEEP_SECRET=; then fail "sweep succeeded without a secret"; fi
}

for t in $(declare -F | awk '{print $3}' | grep '^test_'); do
    "$t"
    echo "ok  $t"
done
echo "ALL PASS"
