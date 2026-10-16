"""Structure checks for .github/workflows/playground.yml.

The workflow's safety rests on job conditions and step order that actionlint
cannot judge: cancelling must stop a session, the tailnet join must happen
before any code from the selected source runs, and a partial re-run must not
bypass the launch check.

    python3 -m pytest -q scripts/tests/test_playground_workflow.py
"""

import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/playground.yml"


def load_jobs():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def step_index(steps, predicate, what):
    for i, step in enumerate(steps):
        if predicate(step):
            return i
    raise AssertionError(f"no step {what}")


def runs(step, text):
    return text in step.get("run", "")


def runs_selected_source(step):
    """Steps that execute code from the checked-out source (possibly a PR)."""
    return step.get("uses", "").startswith("./") or any(
        runs(step, f"scripts/{name}")
        for name in ("ci_setup_python_venv.sh", "ci_install_e2e_deps.sh")
    )


class SessionJobTests(unittest.TestCase):
    def setUp(self):
        self.jobs = load_jobs()
        self.steps = self.jobs["session"]["steps"]

    def test_cancelling_the_run_stops_the_session(self):
        condition = self.jobs["session"]["if"]
        self.assertIn("!cancelled()", condition)
        self.assertNotIn("always()", condition)

    def test_launcher_is_rechecked_before_setup(self):
        check = step_index(
            self.steps,
            lambda s: "needs.gate.outputs.actor" in str(s.get("env", {}))
            and "github.triggering_actor" in str(s.get("env", {})),
            "comparing the gate's launcher with the re-run's",
        )
        first_source = step_index(self.steps, runs_selected_source, "running source code")
        self.assertLess(check, first_source)
        self.assertEqual(self.jobs["gate"]["outputs"]["actor"], "${{ steps.gate.outputs.actor }}")

    def test_join_happens_before_selected_source_code_runs(self):
        join = step_index(self.steps, lambda s: runs(s, "session.sh\" join"), "running join")
        first_source = step_index(self.steps, runs_selected_source, "running source code")
        self.assertLess(join, first_source)

    def test_join_secret_reaches_only_the_join_step(self):
        holders = [
            s.get("name")
            for s in self.steps
            if "PLAYGROUND_TS_AUTHKEY_SECRET" in str(s.get("env", {}))
        ]
        join = self.steps[step_index(self.steps, lambda s: runs(s, "session.sh\" join"), "join")]
        self.assertEqual(holders, [join.get("name")])

    def test_ready_runs_after_setup_and_before_hold(self):
        ready = step_index(self.steps, lambda s: runs(s, "session.sh\" ready"), "running ready")
        install = step_index(self.steps, lambda s: s.get("id") == "smg", "installing SMG")
        hold = step_index(self.steps, lambda s: runs(s, "session.sh\" hold"), "running hold")
        self.assertLess(install, ready)
        self.assertLess(ready, hold)


class BuildJobTests(unittest.TestCase):
    def setUp(self):
        self.job = load_jobs()["build"]

    def test_rust_cache_key_matches_main_ci(self):
        # rust-cache folds RUST* variables into its key; main CI builds with
        # RUSTC_WRAPPER=sccache, so the job must carry the same value.
        self.assertEqual(self.job.get("env", {}).get("RUSTC_WRAPPER"), "sccache")

    def test_cargo_steps_compile_without_sccache(self):
        cargo_steps = [
            s for s in self.job["steps"] if runs(s, "ci_build_wheel.sh") or runs(s, "cargo run")
        ]
        self.assertEqual(len(cargo_steps), 2)
        for step in cargo_steps:
            self.assertEqual(step.get("env", {}).get("RUSTC_WRAPPER"), "env", step.get("name"))

    def test_caches_are_never_saved(self):
        cache = next(s for s in self.job["steps"] if "rust-cache" in s.get("uses", ""))
        self.assertEqual(cache["with"]["save-if"], False)


if __name__ == "__main__":
    unittest.main()
