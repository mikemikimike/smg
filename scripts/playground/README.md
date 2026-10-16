# GPU Playground

Reserve a CI H100 runner for an experiment or a hands-on test and get a shell
in the pod. The pod is the same kind of runner CI uses, with `/models`, RDMA
and docker, and it is deleted when the session ends.

## Launch

Actions → **Playground** → **Run workflow** (from `main`):

| Input | Values | Default |
|---|---|---|
| GPUs | 1, 2, 4, 8 (H100) | 1 |
| Hours | 1-24 | 1 |
| SMG | `main`, `pr`, `release`, `skip` | `main` |
| SMG ref | PR number for `pr` (e.g. `2410`; branches in this repo only, not forks), version for `release` (e.g. `1.13.0`) | empty |
| Engine | `skip`, `sglang`, `vllm`, `tokenspeed` | `skip` |

- `main` and `pr` build the SMG wheel on the CPU pool first (no GPU is held
  while it compiles), then install it with the e2e test dependencies.
- `release` installs that version from PyPI and checks out its tag.
- The engine is installed by the same setup steps CI uses, at the pins of the
  checked-out source.
- The session job queues on the same runner labels as CI (`1-gpu-h100`,
  `2-gpu-h100`, `4-gpu-h100`, `8-gpu-h100`), so it waits when CI is busy.
  An 8-GPU session needs a whole free node.
- Setup time is not counted: the clock starts when the session is ready.

If the run stops at the first job with "You don't have access to run
Playground yet", ask the repo owners for access. Only the person who launched
a session can re-run it; anyone else starts a new run.

## Connect

One-time:
1. Install Tailscale (https://tailscale.com/download).
2. Sign in to the project's tailnet with your GitHub account. You need to be
   in its `group:playground`; ask the repo owners.

Each session: when the run summary says **Playground is ready**, it shows the
command, for example:

```bash
ssh runner@smg-pg-18234567890-1
```

The first connection opens a browser to confirm it's you, and asks again
every 12 hours. Other things that work over the tailnet:

```bash
curl http://smg-pg-18234567890-1:30000/v1/models           # a server you started in the pod
scp -r runner@smg-pg-18234567890-1:out ./out                # copy files out
```

VS Code: Remote-SSH to `runner@smg-pg-18234567890-1`.

## Inside the pod

- You land in the SMG checkout with its venv active.
- `playground-status` shows the time left; `playground-done` ends the session.
- **Ports**: the tailnet reaches ports 8000-8999 and 30000-30999. The pod
  shares the node's network with CI jobs, so check a port is free
  (`ss -ltn`) before binding it.
- **Models**: `HF_HOME=/models` is the node's model cache, shared with CI.
  Don't edit or delete files there. For a large one-off model use a pod-local
  cache: `export HF_HOME=$HOME/hf`. `HF_TOKEN` is set.
- **Tests**: with SMG from `main` or a PR, `pytest e2e_test/...` works as in CI.
- **git**: the checkout keeps no credentials; use your own to push.
- Everything in the pod is deleted when the session ends.

## Ending

The session ends at the chosen time (open terminals get a warning 30 and 5
minutes before), when you run `playground-done`, or when you cancel the run.
One session per person runs at a time; a second launch waits for the first.

## Security notes

- Only the GitHub permission check decides who can launch; only membership
  of `group:playground` decides who can connect. Nothing printed in the
  public run log lets anyone in.
- Traffic is WireGuard-encrypted end to end; no relay can read the session.
- The pod joins the tailnet with a Tailscale OAuth secret that can only add
  `tag:playground` machines. It lives in the `playground` GitHub environment,
  which only `main` can use, so a workflow edited on another branch can't
  join. The pod joins before any code from the selected source runs, and the
  session job has no GitHub OIDC (`id-token`) permission.
- With `pr`, the PR's own setup code runs in a run started from `main`, so
  with `main`'s cache scope. That's why PRs from forks are refused: review a
  fork's branch and push it to this repo first.
- Anyone in `group:playground` can reach any running session, and anything
  run in a session (yours, or a PR you are testing) can see what the pod
  holds, such as `HF_TOKEN`. Keep the group to people who can launch.

## One-time setup (repo owners)

1. **Tailnet.** Create a tailnet for the project at https://login.tailscale.com
   (sign in with GitHub). Use a dedicated tailnet: a shared one is only safe
   if its policy has no allow-all or broad member rules.
2. **Access policy.** In the Tailscale console, replace the default policy
   (it allows everything) with:

   ```json
   {
     "groups": {
       "group:playground": ["alice@github", "bob@github"]
     },
     "tagOwners": {
       "tag:playground": []
     },
     "grants": [
       {
         "src": ["group:playground"],
         "dst": ["tag:playground"],
         "ip": ["tcp:22", "tcp:8000-8999", "tcp:30000-30999"]
       }
     ],
     "ssh": [
       {
         "action": "check",
         "src": ["group:playground"],
         "dst": ["tag:playground"],
         "users": ["runner"]
       }
     ]
   }
   ```

   GitHub accounts appear as `<username>@github`. Nothing grants
   `tag:playground` as a source, so a playground pod cannot reach anything on
   the tailnet.
3. **OAuth clients** (Tailscale console → Settings → OAuth clients, called
   Trust credentials in newer consoles). Create two and copy each secret:
   - join: scope **Auth Keys: write**, tag `tag:playground`;
   - cleanup: scope **Devices Core: write**, tag `tag:playground`.
4. **GitHub environment.** Before the first run (a run against a missing
   environment creates it without restrictions): Settings → Environments →
   New environment `playground`; Deployment branches and tags → Selected →
   `main`. Add environment secrets:
   - `PLAYGROUND_TS_AUTHKEY_SECRET`: the join secret;
   - `PLAYGROUND_TS_SWEEP_SECRET`: the cleanup secret. Without it the cleanup
     job only warns, and machines left by crashed sessions (or by logouts
     that don't delete the machine) pile up until they hit the plan's
     tagged-machine limit.
5. **8-GPU runners.** Install the `8-gpu-h100` scale set
   (`scripts/k8s-runner-resources/README.md`).
6. **First run.** Launch a 1-GPU, 1-hour session and check: the run summary
   shows the ssh command; `ssh` works and asks for the browser check; `curl`
   reaches a server on port 30000; VS Code Remote-SSH connects;
   `playground-done` ends the run. Then check the tailnet's Machines page:
   the session's machine should be gone, or be removed by the cleanup job of
   the next run.

Free-plan limits to keep in mind: 6 users, and Tailscale SSH on 5 machines,
which caps concurrent sessions at 5.

## Maintenance

- Tailscale is pinned in `session.sh` (`TS_VERSION`, `TS_SHA256`). To bump,
  pick a release at least two weeks old and take the checksum from
  `https://pkgs.tailscale.com/stable/tailscale_<version>_amd64.tgz.sha256`.
- Tests (Linux or macOS, no network):

  ```bash
  bash scripts/tests/test_playground_gate.sh
  bash scripts/tests/test_playground_session.sh
  bash scripts/tests/test_playground_sweep.sh
  ```
