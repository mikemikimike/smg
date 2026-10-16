#!/bin/bash
# Gate for the Playground workflow: decide whether this launch may claim a GPU
# runner, validate the inputs, and resolve what the session job builds on.
#
# Runs on a GitHub-hosted runner before any GPU is requested, so a refused or
# malformed launch never holds a GPU. All inputs arrive through the
# environment (never interpolated into the workflow script):
#
#   PG_ACTOR PG_REF PG_SHA PG_REPO        who launched, from where
#   PG_GPUS PG_HOURS PG_SMG PG_SMG_REF PG_ENGINE   workflow_dispatch inputs
#   PG_RUNNER_1 PG_RUNNER_2 PG_RUNNER_4   runner labels (repo-variable overrides)
#   GH_TOKEN                              for `gh api`
#
# Writes to GITHUB_OUTPUT: actor (the launcher it checked), runner, source_sha,
# source_label, smg_version, build_wheel, hold_minutes, job_timeout_minutes.

set -euo pipefail

# Setup budget on top of the hold: engine installs (TokenSpeed nightly wheels,
# SGLang/vLLM venvs) plus the Tailscale join.
SETUP_MINUTES=150

fail() {
    echo "::error::$1"
    echo "### Playground did not start" >> "$GITHUB_STEP_SUMMARY"
    echo "$1" >> "$GITHUB_STEP_SUMMARY"
    exit 1
}

allowed="$(gh api "repos/${PG_REPO}/collaborators/${PG_ACTOR}/permission" --jq .user.permissions.admin 2>/dev/null || true)"
if [ "$allowed" != true ]; then
    fail "You don't have access to run Playground yet. Ask the repo owners to grant it."
fi

# The tailnet join secret is only released to runs from main.
if [ "$PG_REF" != refs/heads/main ]; then
    fail "Start Playground from the main branch."
fi

case "$PG_GPUS" in
    1) runner="$PG_RUNNER_1" ;;
    2) runner="$PG_RUNNER_2" ;;
    4) runner="$PG_RUNNER_4" ;;
    8) runner="8-gpu-h100" ;;
    *) fail "GPUs must be 1, 2, 4 or 8 (got '${PG_GPUS}')." ;;
esac

if ! [[ "$PG_HOURS" =~ ^[0-9]+$ ]] || [ "$PG_HOURS" -lt 1 ] || [ "$PG_HOURS" -gt 24 ]; then
    fail "Hours must be a whole number from 1 to 24 (got '${PG_HOURS}')."
fi

case "$PG_ENGINE" in
    skip | sglang | vllm | tokenspeed) ;;
    *) fail "Engine must be skip, sglang, vllm or tokenspeed (got '${PG_ENGINE}')." ;;
esac

smg_version=""
build_wheel=false
case "$PG_SMG" in
    main)
        source_sha="$PG_SHA"
        source_label="main @ ${PG_SHA:0:7}"
        build_wheel=true
        ;;
    pr)
        pr="${PG_SMG_REF#\#}"
        [[ "$pr" =~ ^[0-9]+$ ]] || fail "For SMG 'pr', set the PR number (got '${PG_SMG_REF}')."
        pr_head="$(gh api "repos/${PG_REPO}/pulls/${pr}" --jq '.head.sha + " " + (.head.repo.full_name // "")' 2>/dev/null)" \
            || fail "PR #${pr} was not found in ${PG_REPO}."
        read -r source_sha head_repo <<< "$pr_head"
        # The session runs the PR's setup code in a run started from main, so
        # with main's cache scope. Only branches pushed to this repo qualify.
        [ "${head_repo:-}" = "$PG_REPO" ] \
            || fail "PR #${pr} comes from a fork. Playground only runs PRs from branches in ${PG_REPO}; push the reviewed branch there and use that PR."
        source_label="PR #${pr} @ ${source_sha:0:7}"
        build_wheel=true
        ;;
    release)
        smg_version="${PG_SMG_REF#v}"
        [[ "$smg_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-]?[a-z0-9.]+)?$ ]] \
            || fail "For SMG 'release', set a version such as 1.13.0 (got '${PG_SMG_REF}')."
        source_sha="$(gh api "repos/${PG_REPO}/commits/v${smg_version}" --jq .sha 2>/dev/null)" \
            || fail "Tag v${smg_version} was not found in ${PG_REPO}."
        curl -fsS -o /dev/null "https://pypi.org/pypi/smg/${smg_version}/json" \
            || fail "smg ${smg_version} is not on PyPI."
        source_label="release ${smg_version} @ ${source_sha:0:7}"
        ;;
    skip)
        source_sha="$PG_SHA"
        source_label="main @ ${PG_SHA:0:7} (SMG not installed)"
        ;;
    *) fail "SMG must be main, pr, release or skip (got '${PG_SMG}')." ;;
esac

hold_minutes=$((PG_HOURS * 60))
{
    echo "actor=${PG_ACTOR}"
    echo "runner=${runner}"
    echo "source_sha=${source_sha}"
    echo "source_label=${source_label}"
    echo "smg_version=${smg_version}"
    echo "build_wheel=${build_wheel}"
    echo "hold_minutes=${hold_minutes}"
    echo "job_timeout_minutes=$((hold_minutes + SETUP_MINUTES))"
} >> "$GITHUB_OUTPUT"

echo "Playground: ${PG_GPUS} GPU(s) on ${runner} for ${PG_HOURS}h; source ${source_label}; engine ${PG_ENGINE}"
