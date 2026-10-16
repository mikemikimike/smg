#!/bin/bash
# Session plumbing for the Playground workflow, run on the GPU runner pod.
#
#   session.sh join    install Tailscale and join the tailnet; runs before any
#                      code from the selected source, so that code never sees
#                      the join secret before the machine name is taken
#   session.sh ready   prepare the shell environment, start the clock, write
#                      the run summary (after SMG/engine setup)
#   session.sh hold    block until the reservation ends or playground-done
#   session.sh down    leave the tailnet (always exits 0)
#   session.sh status  time left (also installed as `playground-status`)
#   session.sh done    end early (also installed as `playground-done`)
#
# The pod runs Tailscale without root: tailscaled uses userspace networking
# and Tailscale SSH logs people in as this same user. The runner pod shares the
# node's network namespace, so tailscaled picks a random UDP port and inbound
# tailnet traffic reaches the node's loopback (the tailnet policy limits ports).
#
# Environment (defaults in brackets):
#   PG_TS_AUTHKEY      join: Tailscale OAuth client secret (auth_keys, tag:playground)
#   PG_HOSTNAME        join: tailnet machine name
#   PG_HOURS           ready: reservation length; PG_DURATION_SECONDS overrides it
#   PG_GPUS PG_SOURCE_LABEL PG_SMG_STATUS PG_ENGINE PG_ENGINE_STATUS  ready: summary text
#   PG_STATE_DIR       [$RUNNER_TEMP/playground]
#   PG_HOME            [$HOME]
#   PG_WORKSPACE       [$GITHUB_WORKSPACE]
#   PG_TS_BIN          directory with tailscale/tailscaled; skips the download
#   PG_TS_VERSION PG_TS_SHA256   pinned release (overridable for tests)

set -euo pipefail

TS_VERSION="${PG_TS_VERSION:-1.102.5}"
TS_SHA256="${PG_TS_SHA256:-65e6d7f19ad7e1c87d20c2a21e92f38a96795cb897af54b04536590e1c148d12}"
TS_TAG="tag:playground"
TAILNET_PORTS="8000-8999 and 30000-30999"
WARN_AT_SECONDS=(1800 300)

STATE="${PG_STATE_DIR:-${RUNNER_TEMP:-/tmp}/playground}"
SOCK="$STATE/ts/tailscaled.sock"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

log() { echo "[playground] $*"; }
die() { echo "::error::$*"; exit 1; }

fmt_utc() {
    date -u -d "@$1" '+%Y-%m-%d %H:%M UTC' 2> /dev/null || date -u -r "$1" '+%Y-%m-%d %H:%M UTC'
}

sha256_of() {
    (sha256sum "$1" 2> /dev/null || shasum -a 256 "$1") | cut -d' ' -f1
}

ts() { "$TS_BIN/tailscale" --socket="$SOCK" "$@"; }

install_tailscale() {
    if [ -n "${PG_TS_BIN:-}" ]; then
        TS_BIN="$PG_TS_BIN"
        return
    fi
    local tgz="$STATE/tailscale.tgz"
    log "Downloading Tailscale ${TS_VERSION}"
    curl -fsSL --retry 3 -o "$tgz" "https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_amd64.tgz"
    [ "$(sha256_of "$tgz")" = "$TS_SHA256" ] || die "Tailscale download failed its checksum; not starting."
    tar -xzf "$tgz" -C "$STATE"
    TS_BIN="$STATE/tailscale_${TS_VERSION}_amd64"
}

start_tailscaled() {
    mkdir -p "$STATE/ts"
    # A scrubbed environment: Tailscale SSH sessions descend from tailscaled,
    # so the job's tokens must not be in its environment.
    local launcher=(env -i HOME="$HOME_DIR" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
    if command -v setsid > /dev/null; then launcher+=(setsid); fi
    nohup "${launcher[@]}" "$TS_BIN/tailscaled" \
        --tun=userspace-networking \
        --statedir="$STATE/ts" \
        --socket="$SOCK" \
        --port=0 \
        --no-logs-no-support \
        > "$STATE/tailscaled.log" 2>&1 < /dev/null &
    echo $! > "$STATE/tailscaled.pid"
    local _
    for _ in $(seq 1 60); do
        [ -e "$SOCK" ] && return 0
        kill -0 "$(cat "$STATE/tailscaled.pid")" 2> /dev/null || break
        sleep 0.5
    done
    cat "$STATE/tailscaled.log" >&2 || true
    die "tailscaled did not start."
}

join_tailnet() {
    local keyfile="$STATE/authkey"
    (umask 077 && printf '%s?ephemeral=false&preauthorized=true' "$PG_TS_AUTHKEY" > "$keyfile")
    # shellcheck disable=SC2064  # expand now: the path is fixed for this call
    trap "rm -f '$keyfile'" EXIT
    ts up \
        --auth-key="file:$keyfile" \
        --advertise-tags="$TS_TAG" \
        --hostname="$PG_HOSTNAME" \
        --ssh \
        --accept-dns=false \
        --accept-routes=false \
        --timeout=120s \
        || die "Joining the tailnet failed (see the step log)."
    rm -f "$keyfile"
    trap - EXIT
}

# Tailscale renames a machine whose name is taken (name-1). Someone holding a
# stolen join secret could take the name first and catch the ssh command from
# the run summary, so a renamed machine is a failure, not a fallback.
check_machine_name() {
    local actual
    actual="$(ts status --json --peers=false | sed -n 's/.*"DNSName": *"\([^.]*\)\..*/\1/p' | head -1)"
    [ "$actual" = "$PG_HOSTNAME" ] \
        || die "The tailnet named this machine '${actual}' instead of '${PG_HOSTNAME}': a machine with that name already exists. Remove it in the Tailscale console and re-run."
}

write_env_file() {
    local dir="$HOME_DIR/.playground" path="$PATH" var
    mkdir -p "$dir"
    [ -d "$WORKSPACE/.venv/bin" ] && path="$WORKSPACE/.venv/bin:$path"
    path="$STATE/bin:$path"
    {
        echo "# Generated by scripts/playground/session.sh for this playground session."
        printf 'export PATH=%q\n' "$path"
        printf 'export HF_HOME=%q\n' "${HF_HOME:-/models}"
        for var in HF_TOKEN CUDA_HOME LD_LIBRARY_PATH SMG_BAKED_VENV; do
            if [ -n "${!var:-}" ]; then printf 'export %s=%q\n' "$var" "${!var}"; fi
        done
        printf 'export PLAYGROUND_WORKSPACE=%q\n' "$WORKSPACE"
        printf 'export PLAYGROUND_STATE=%q\n' "$STATE"
        cat <<'EOF'
if [[ $- == *i* ]] && [ -z "${PLAYGROUND_GREETED:-}" ]; then
    export PLAYGROUND_GREETED=1
    if [ -f "$PLAYGROUND_WORKSPACE/.venv/bin/activate" ]; then
        . "$PLAYGROUND_WORKSPACE/.venv/bin/activate"
    fi
    cd "$PLAYGROUND_WORKSPACE" 2> /dev/null || true
    cat "$PLAYGROUND_STATE/banner.txt" 2> /dev/null
    playground-status
fi
EOF
    } > "$dir/env.sh"
}

hook_shell_startup() {
    local hook='[ -f ~/.playground/env.sh ] && . ~/.playground/env.sh  # playground'
    local rc="$HOME_DIR/.bashrc" f
    # .bashrc: first line, ahead of the usual early return for non-interactive
    # shells, so `ssh host cmd` gets the same PATH and variables.
    touch "$rc"
    if ! grep -qF 'playground/env.sh' "$rc"; then
        { echo "$hook"; cat "$rc"; } > "$rc.new" && mv "$rc.new" "$rc"
    fi
    for f in "$HOME_DIR/.profile" "$HOME_DIR/.bash_profile"; do
        [ "$f" = "$HOME_DIR/.profile" ] || [ -f "$f" ] || continue
        touch "$f"
        grep -qF 'playground/env.sh' "$f" || echo "$hook" >> "$f"
    done
}

install_commands() {
    mkdir -p "$STATE/bin"
    local cmd
    for cmd in status "done"; do
        printf '#!/bin/bash\nPG_STATE_DIR=%q exec bash %q %s "$@"\n' "$STATE" "$SELF" "$cmd" \
            > "$STATE/bin/playground-$cmd"
        chmod +x "$STATE/bin/playground-$cmd"
    done
}

write_banner_and_summary() {
    local deadline="$1" ip="$2" host="$3" ends
    ends="$(fmt_utc "$deadline")"
    cat > "$STATE/banner.txt" <<EOF

  SMG GPU playground: ${PG_GPUS}× H100, ${PG_SOURCE_LABEL}
  SMG ${PG_SMG_STATUS}; engine ${PG_ENGINE} (${PG_ENGINE_STATUS})
  Tailnet ports ${TAILNET_PORTS}. The node's network is shared with CI jobs:
  check a port is free (ss -ltn) before you bind it.
  /models is the node's shared model cache. Don't edit or delete files there.

EOF
    {
        echo "## Playground is ready"
        echo
        echo "Connect with Tailscale (see \`scripts/playground/README.md\`):"
        echo
        echo '```'
        echo "ssh runner@${host}"
        echo '```'
        echo
        echo "| | |"
        echo "|---|---|"
        echo "| GPUs | ${PG_GPUS}× H100 |"
        echo "| Source | ${PG_SOURCE_LABEL} |"
        echo "| SMG | ${PG_SMG_STATUS} |"
        echo "| Engine | ${PG_ENGINE} (${PG_ENGINE_STATUS}) |"
        echo "| Ends | ${ends}. End early with \`playground-done\` in the session or cancel this run |"
        echo "| Tailnet address | \`${host}\` / \`${ip}\`, ports ${TAILNET_PORTS} |"
    } >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
    echo "::notice title=Playground ready::ssh runner@${host} (until ${ends})"
}

cmd_join() {
    HOME_DIR="${PG_HOME:-$HOME}"
    [ -n "${PG_TS_AUTHKEY:-}" ] \
        || die "The 'playground' environment has no PLAYGROUND_TS_AUTHKEY_SECRET; see scripts/playground/README.md."
    mkdir -p "$STATE"
    chmod 700 "$STATE"
    install_tailscale
    start_tailscaled
    join_tailnet
    check_machine_name
    echo "$PG_HOSTNAME" > "$STATE/hostname"
    ts ip -4 | head -1 > "$STATE/ip"
    log "Joined the tailnet as ${PG_HOSTNAME} ($(cat "$STATE/ip"))"
}

cmd_ready() {
    HOME_DIR="${PG_HOME:-$HOME}"
    WORKSPACE="${PG_WORKSPACE:-${GITHUB_WORKSPACE:-$PWD}}"
    [ -f "$STATE/hostname" ] || die "Not on the tailnet; the join step did not complete."
    local host ip deadline
    host="$(cat "$STATE/hostname")"
    ip="$(cat "$STATE/ip")"
    deadline=$(($(date +%s) + ${PG_DURATION_SECONDS:-$((PG_HOURS * 3600))}))
    echo "$deadline" > "$STATE/deadline"
    rm -f "$STATE/done"
    install_commands
    write_env_file
    hook_shell_startup
    write_banner_and_summary "$deadline" "$ip" "$host"
    log "Ready: ssh runner@${host} until $(fmt_utc "$deadline")"
}

warn_terminals() {
    local msg="$1" tty
    log "$msg"
    # shellcheck disable=SC2086  # unquoted on purpose: the value is a glob
    for tty in ${PG_PTS_GLOB:-/dev/pts/[0-9]*}; do
        [ -w "$tty" ] && [ -O "$tty" ] || continue
        printf '\r\n*** %s ***\r\n' "$msg" >> "$tty" 2> /dev/null || true
    done
}

cmd_hold() {
    local deadline now left mins level sent=0 t
    deadline="$(cat "$STATE/deadline")"
    log "Holding until $(fmt_utc "$deadline")"
    while :; do
        if [ -e "$STATE/done" ]; then
            log "Ended early with playground-done."
            return 0
        fi
        now="$(date +%s)"
        left=$((deadline - now))
        if [ "$left" -le 0 ]; then
            log "Time is up."
            return 0
        fi
        # Each threshold warns once; when several are crossed between polls
        # (or at start), only the most urgent message goes out.
        level=0
        for t in "${WARN_AT_SECONDS[@]}"; do
            if [ "$left" -le "$t" ]; then level=$((level + 1)); fi
        done
        if [ "$level" -gt "$sent" ]; then
            sent="$level"
            mins=$(((left + 59) / 60))
            warn_terminals "Playground ends in about ${mins} minute$([ "$mins" -eq 1 ] || echo s) ($(fmt_utc "$deadline")). The pod and everything on it are deleted then."
        fi
        sleep "${PG_POLL_SECONDS:-15}"
    done
}

cmd_down() {
    if [ -n "${PG_TS_BIN:-}" ]; then
        TS_BIN="$PG_TS_BIN"
    else
        TS_BIN="$STATE/tailscale_${TS_VERSION}_amd64"
    fi
    if [ -x "$TS_BIN/tailscale" ]; then
        ts logout 2>&1 || log "logout failed; the daily sweep removes the machine"
    fi
    if [ -f "$STATE/tailscaled.pid" ]; then
        kill "$(cat "$STATE/tailscaled.pid")" 2> /dev/null || true
    fi
    return 0
}

cmd_status() {
    [ -f "$STATE/deadline" ] || { echo "No playground session here."; return 1; }
    if [ -e "$STATE/done" ]; then
        echo "Playground is ending now."
        return 0
    fi
    local deadline left
    deadline="$(cat "$STATE/deadline")"
    left=$((deadline - $(date +%s)))
    [ "$left" -gt 0 ] || left=0
    echo "Playground ends at $(fmt_utc "$deadline") ($((left / 3600))h $((left % 3600 / 60))m left). End early with playground-done."
}

cmd_done() {
    touch "$STATE/done"
    echo "Ending the playground. The pod is deleted within a minute."
}

case "${1:-}" in
    join) cmd_join ;;
    ready) cmd_ready ;;
    hold) cmd_hold ;;
    down) cmd_down ;;
    status) cmd_status ;;
    done) cmd_done ;;
    *) echo "usage: $0 join|ready|hold|down|status|done" >&2; exit 2 ;;
esac
