#!/bin/bash
# Remove playground machines that a crashed or evicted pod left on the tailnet.
#
# A normal session logs out at teardown (session.sh down); this catches the
# rest. Only machines tagged tag:playground that the control plane reports as
# disconnected, and that were last seen more than PG_STALE_MINUTES ago, are
# deleted. A machine whose connection state is not reported is left alone.
#
# Environment:
#   PG_TS_SWEEP_SECRET  Tailscale OAuth client secret with the devices:core scope
#   PG_TS_API           [https://api.tailscale.com]
#   PG_STALE_MINUTES    [15]

set -euo pipefail

API="${PG_TS_API:-https://api.tailscale.com}"
STALE_MINUTES="${PG_STALE_MINUTES:-15}"

[ -n "${PG_TS_SWEEP_SECRET:-}" ] || { echo "::error::PG_TS_SWEEP_SECRET is not set"; exit 1; }

# Tailscale identifies the OAuth client by its secret and ignores client_id
# (its own CLI sends a placeholder), so only the secret is configured.
token="$(curl -fsS -X POST "$API/api/v2/oauth/token" \
    -d client_id=playground-sweep \
    --data-urlencode "client_secret=${PG_TS_SWEEP_SECRET}" \
    -d grant_type=client_credentials | jq -r .access_token)"
[ -n "$token" ] && [ "$token" != null ] || { echo "::error::Tailscale token exchange failed"; exit 1; }

stale="$(curl -fsS -H "Authorization: Bearer ${token}" "$API/api/v2/tailnet/-/devices?fields=all" \
    | jq -r --argjson mins "$STALE_MINUTES" '
        .devices[]
        | select((.tags // []) | index("tag:playground"))
        | select(.connectedToControl == false)
        | select((.lastSeen | fromdateiso8601) < (now - $mins * 60))
        | "\(.id) \(.hostname)"')"

if [ -z "$stale" ]; then
    echo "No stale playground machines."
    exit 0
fi

while read -r id hostname; do
    echo "Removing ${hostname} (${id})"
    curl -fsS -X DELETE -H "Authorization: Bearer ${token}" "$API/api/v2/device/${id}"
done <<< "$stale"
