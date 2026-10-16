#!/bin/bash
# Tests for scripts/playground/gate.sh: who may launch, input validation, and
# the outputs the session job is built from.
#
#   bash scripts/tests/test_playground_gate.sh
#
# `gh` and `curl` are replaced by fakes on PATH, so no GitHub or PyPI access is
# needed. Runs on Linux and macOS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/playground/gate.sh"
MAIN_SHA="1111111111111111111111111111111111111111"
PR_SHA="2222222222222222222222222222222222222222"
TAG_SHA="3333333333333333333333333333333333333333"
REFUSAL="You don't have access to run Playground yet. Ask the repo owners to grant it."

fail() { echo "FAIL: $*"; exit 1; }
assert_eq() { [ "$1" = "$2" ] || fail "${3:-value}: expected '$1', got '$2'"; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "expected to contain '$2', got: $1"; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"

cat > "$T/bin/gh" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_LOG"
[ "$1" = api ] || exit 2
case "$2" in
    */collaborators/*/permission) echo "${FAKE_PERMISSION}" ;;
    */pulls/*) [ -n "$FAKE_PR_SHA" ] || { echo "HTTP 404: Not Found" >&2; exit 1; }; echo "$FAKE_PR_SHA ${FAKE_PR_REPO}" ;;
    */commits/*) [ -n "$FAKE_TAG_SHA" ] || { echo "HTTP 422" >&2; exit 1; }; echo "$FAKE_TAG_SHA" ;;
    *) exit 3 ;;
esac
EOF
cat > "$T/bin/curl" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_LOG"
[ "$FAKE_PYPI" = 200 ] || exit 22
EOF
chmod +x "$T/bin/gh" "$T/bin/curl"

# run_gate KEY=VALUE... : run the gate with defaults overridden; sets RC,
# SUMMARY and LOG for the assertions (outputs are read with `out`).
run_gate() {
    : > "$T/out"; : > "$T/summary"; : > "$T/log"
    set +e
    env -i PATH="$T/bin:/usr/bin:/bin" HOME="$T" \
        FAKE_LOG="$T/log" FAKE_PERMISSION=true FAKE_PR_SHA="$PR_SHA" FAKE_PR_REPO=smg-project/smg \
        FAKE_TAG_SHA="$TAG_SHA" FAKE_PYPI=200 \
        GITHUB_OUTPUT="$T/out" GITHUB_STEP_SUMMARY="$T/summary" GH_TOKEN=x \
        PG_ACTOR=alice PG_REF=refs/heads/main PG_SHA="$MAIN_SHA" PG_REPO=smg-project/smg \
        PG_GPUS=1 PG_HOURS=1 PG_SMG=main PG_SMG_REF= PG_ENGINE=skip \
        PG_RUNNER_1=1-gpu-h100 PG_RUNNER_2=2-gpu-h100 PG_RUNNER_4=4-gpu-h100 \
        "$@" bash "$SCRIPT" > "$T/stdout" 2>&1
    RC=$?
    set -e
    SUMMARY="$(cat "$T/summary")"; LOG="$(cat "$T/log")"
}
out() { sed -n "s/^$1=//p" "$T/out"; }
expect_refused() {
    [ "$RC" -ne 0 ] || fail "$1: expected failure, got exit 0"
    [ -z "$(out runner)" ] || fail "$1: runner output set on a refused run"
}

# The permission the gate checks, read from its jq filter so that no test
# names it either.
ROLE="$(sed -n 's/.*--jq \.user\.permissions\.\([a-z]*\).*/\1/p' "$SCRIPT")"

test_refused_caller() {
    run_gate FAKE_PERMISSION=false
    expect_refused "refused caller"
    assert_contains "$SUMMARY" "$REFUSAL"
    [ -n "$ROLE" ] || fail "could not read the checked permission from gate.sh"
    if grep -qi "$ROLE" "$T/summary" "$T/stdout"; then fail "refusal names the role"; fi
}

test_role_named_only_in_code_literal() {
    assert_eq 1 "$(grep -ci "$ROLE" "$SCRIPT")" "lines naming the role in gate.sh"
}

test_outputs_the_checked_launcher() {
    run_gate PG_ACTOR=carol
    assert_eq carol "$(out actor)" "actor"
}

test_requires_main() {
    run_gate PG_REF=refs/heads/feature
    expect_refused "non-main ref"
    assert_contains "$SUMMARY" "Start Playground from the main branch."
}

test_hours_validation() {
    local bad
    for bad in 0 25 1.5 abc ""; do
        run_gate PG_HOURS="$bad"
        expect_refused "hours='$bad'"
    done
    run_gate PG_HOURS=24
    assert_eq 0 "$RC" "exit for hours=24"
    assert_eq 1440 "$(out hold_minutes)" "hold_minutes"
    assert_eq 1590 "$(out job_timeout_minutes)" "job_timeout_minutes"
}

test_runner_labels() {
    run_gate PG_GPUS=8
    assert_eq 8-gpu-h100 "$(out runner)" "runner for 8 GPUs"
    run_gate PG_GPUS=4 PG_RUNNER_4=custom-4
    assert_eq custom-4 "$(out runner)" "runner for 4 GPUs"
    run_gate PG_GPUS=3
    expect_refused "gpus=3"
}

test_smg_main() {
    run_gate PG_SMG=main
    assert_eq 0 "$RC" "exit"
    assert_eq "$MAIN_SHA" "$(out source_sha)" "source_sha"
    assert_eq true "$(out build_wheel)" "build_wheel"
    assert_contains "$(out source_label)" "main @ 1111111"
}

test_smg_pr() {
    run_gate PG_SMG=pr PG_SMG_REF='#123'
    assert_eq 0 "$RC" "exit"
    assert_eq "$PR_SHA" "$(out source_sha)" "source_sha"
    assert_eq true "$(out build_wheel)" "build_wheel"
    assert_contains "$(out source_label)" "PR #123 @ 2222222"
    assert_contains "$LOG" "pulls/123"

    run_gate PG_SMG=pr PG_SMG_REF='1;rm -rf /'
    expect_refused "pr ref with shell syntax"
    [[ "$LOG" != *"rm -rf"* ]] || fail "bad pr ref reached gh"

    run_gate PG_SMG=pr PG_SMG_REF=123 FAKE_PR_SHA=
    expect_refused "missing PR"
    assert_contains "$SUMMARY" "#123"
}

test_smg_pr_from_fork_refused() {
    run_gate PG_SMG=pr PG_SMG_REF=123 FAKE_PR_REPO=someone/smg
    expect_refused "fork PR"
    assert_contains "$SUMMARY" "fork"
    run_gate PG_SMG=pr PG_SMG_REF=123 FAKE_PR_REPO=
    expect_refused "PR whose fork was deleted"
}

test_smg_release() {
    run_gate PG_SMG=release PG_SMG_REF=v1.13.0
    assert_eq 0 "$RC" "exit"
    assert_eq 1.13.0 "$(out smg_version)" "smg_version"
    assert_eq "$TAG_SHA" "$(out source_sha)" "source_sha"
    assert_eq false "$(out build_wheel)" "build_wheel"
    assert_contains "$LOG" "commits/v1.13.0"
    assert_contains "$LOG" "pypi.org/pypi/smg/1.13.0/json"

    run_gate PG_SMG=release PG_SMG_REF=1.13.0 FAKE_PYPI=404
    expect_refused "version missing on PyPI"

    run_gate PG_SMG=release PG_SMG_REF=1.13.0 FAKE_TAG_SHA=
    expect_refused "version without a tag"

    # shellcheck disable=SC2016  # the literal $(id) is the attack being refused
    run_gate PG_SMG=release PG_SMG_REF='1.13.0$(id)'
    expect_refused "version with shell syntax"
}

test_smg_skip() {
    run_gate PG_SMG=skip
    assert_eq 0 "$RC" "exit"
    assert_eq false "$(out build_wheel)" "build_wheel"
    assert_eq "$MAIN_SHA" "$(out source_sha)" "source_sha"
}

test_engine_validation() {
    run_gate PG_ENGINE=trtllm
    expect_refused "engine=trtllm"
    run_gate PG_ENGINE=sglang
    assert_eq 0 "$RC" "exit for engine=sglang"
}

for t in $(declare -F | awk '{print $3}' | grep '^test_'); do
    "$t"
    echo "ok  $t"
done
echo "ALL PASS"
