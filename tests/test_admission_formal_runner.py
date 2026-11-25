"""Exercise the formal CLI in a copied workspace with an inert Inspect executable."""

import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("evals/inspect/admission_formal_plan.json")
RUNNER = Path("scripts/run_openai_admission_formal.sh")
HELPER = RUNNER.with_suffix(".py")
MODEL = "openai/gpt-4o-mini"
CONDITIONS = ("vulnerable", "unsupported", "supported")
REVISION = "a" * 40
CANARY = "synthetic-value-excluded-from-provenance"


@pytest.fixture
def workspace(tmp_path):
    for path in (PLAN, RUNNER, HELPER):
        (tmp_path / path.parent).mkdir(exist_ok=True, parents=True)
        shutil.copy2(ROOT / path, tmp_path / path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "python": f'exec {shlex.quote(sys.executable)} "$@"',
        "dirname": 'printf "%s\\n" "${2%/*}"',
        "git": f'[ "$*" = "rev-parse --verify HEAD" ] || exit 1\nprintf "%s\\n" {REVISION}',
        "inspect": """
printf 'invoked\\n' >> calls
printf '%s\\n' "$@" > arguments
printf 'Log: unrelated.eval %s\\n' "$TEST_CANARY" >&2
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--log-dir" ]; then log_dir=$2; break; fi
    shift
done
[ -n "$log_dir" ] || exit 64
case "$STUB_MODE" in
    early_failure) exit 17 ;;
    no_log) exit 0 ;;
    wait) printf 'ready\\n'; read -r release || exit 1 ;;
    ambiguous) : > "$log_dir/second.eval" ;;
esac
: > "$log_dir/stub.eval"
[ "$STUB_MODE" != "late_failure" ] || exit 23
""",
    }
    for name, code in stubs.items():
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\n" + code + "\n", encoding="utf-8")
        executable.chmod(0o700)
    return tmp_path


def command(root, args, mode="normal"):
    # No inherited environment, real Inspect, provider SDK or project credentials.
    return dict(args=["/bin/bash", str(root / RUNNER), *args], cwd=root / "bin",
                env={"PATH": str(root / "bin"), "TEST_CANARY": CANARY, "STUB_MODE": mode},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def invoke(root, args, mode="normal"):
    return subprocess.run(**command(root, args, mode), timeout=10)


def provenance(root, run_id):
    return root / "logs" / "v3-formal" / run_id / "invocation.json"


def test_fixed_plan_and_ignored_outputs():
    plan = json.loads((ROOT / PLAN).read_text())
    assert plan["analysis_contract"] == "v3-admission-analysis-1" and plan["model"] == MODEL
    expected = {f"v3-formal-{condition}-{i:02d}" for condition in CONDITIONS for i in range(1, 11)}
    assert len(plan["run_ids"]) == len(set(plan["run_ids"])) == 30
    assert set(plan["run_ids"]) == expected
    assert os.access(ROOT / RUNNER, os.X_OK)
    directory = provenance(ROOT, "v3-formal-unsupported-09").parent
    for path in (directory / "invocation.json", directory / "any.eval"):
        assert subprocess.run(["git", "check-ignore", "-q", "--", path.relative_to(ROOT)], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("condition,number", zip(CONDITIONS, ("1", "05", "10")))
def test_exact_inspect_command_and_provenance(workspace, condition, number):
    result = invoke(workspace, [condition, number, MODEL])
    assert result.returncode == 0, result.stderr
    run_id = f"v3-formal-{condition}-{int(number):02d}"
    path = provenance(workspace, run_id)
    directory = path.parent.relative_to(workspace).as_posix()
    task = f"evals/inspect/admission_behavioral_eval.py@admission_behavior_{condition}_eval"
    expected = ["eval", task, "--model", MODEL, "--limit", "1", "--epochs", "1", "--no-epochs-reducer",
                "--max-retries", "0", "--retry-on-error", "0", "--no-log-realtime",
                "--log-dir", directory, "--log-format", "eval", "--display", "plain",
                "--metadata", f"formal_run_id={run_id}"]
    assert (workspace / "arguments").read_text().splitlines() == expected
    assert (workspace / "calls").read_text().splitlines() == ["invoked"]
    raw = path.read_text()
    assert CANARY in result.stderr and CANARY not in raw and "unrelated.eval" not in raw
    assert str(workspace) not in raw
    record = json.loads(raw)
    assert record["inspect_argv"] == ["inspect", *expected]
    assert (record["dataset"], record["analysis_contract"], record["schema_version"]) == (
        "formal", "v3-admission-analysis-1", "v3-admission-formal-invocation-1")
    assert (record["run_id"], record["task"], record["model"], record["condition"], record["git_revision"]) == (
        run_id, task, MODEL, condition, REVISION)
    assert record["configured"] == dict(limit=1, epochs=1, epochs_reducer=[], max_retries=0, retry_on_error=0,
                                        log_realtime=False, log_dir=directory, log_format="eval", display="plain")
    assert record["phase"] == "inspect_exited" and record["finished_at"]
    assert record["process_exit_status"] == record["runner_exit_status"] == 0
    assert record["log_status"] == "present" and record["eval_path"] == f"{directory}/stub.eval"
    assert record["eval_path"] in result.stdout


@pytest.mark.parametrize("args", [
    [], ["supported", "1"], ["other", "1", MODEL], ["supported", "0", MODEL],
    ["unsupported", "11", MODEL], ["vulnerable", "../1", MODEL], ["supported", "001", MODEL],
    ["unsupported", "1", "gpt-4o-mini"], ["unsupported", "1", "openai/other"],
    ["vulnerable", "10", MODEL, "--epochs"],
])
def test_invalid_request_never_claims_slot_or_calls_inspect(workspace, args):
    assert invoke(workspace, args).returncode == 64
    assert not (workspace / "logs").exists() and not (workspace / "calls").exists()


@pytest.mark.parametrize("mode,exit_code,inspect_status,log_status", [
    ("normal", 0, 0, "present"), ("early_failure", 17, 17, "absent"), ("late_failure", 23, 23, "present"),
    ("no_log", 1, 0, "absent"), ("ambiguous", 1, 0, "unresolved"),
])
def test_slot_is_never_retried_replaced_or_overwritten(workspace, mode, exit_code, inspect_status, log_status):
    args = ["unsupported", "3", MODEL]
    assert invoke(workspace, args, mode).returncode == exit_code
    path = provenance(workspace, "v3-formal-unsupported-03")
    original = path.read_bytes()
    record = json.loads(original)
    assert record["process_exit_status"] == inspect_status and record["runner_exit_status"] == exit_code
    assert record["log_status"] == log_status
    assert (record["eval_path"] is not None) == (log_status == "present")
    assert invoke(workspace, ["unsupported", "03", MODEL]).returncode != 0
    assert path.read_bytes() == original
    assert list(path.parent.parent.iterdir()) == [path.parent]
    assert (workspace / "calls").read_text().splitlines() == ["invoked"]


def test_missing_inspect_records_launch_failure(workspace):
    (workspace / "bin/inspect").rename(workspace / "unused-stub")
    args = ["supported", "2", MODEL]
    assert invoke(workspace, args).returncode == 127
    path = provenance(workspace, "v3-formal-supported-02")
    record = json.loads(path.read_text())
    assert record["phase"] == "launch_failed" and record["process_exit_status"] is None
    assert record["runner_exit_status"] == 127 and record["eval_path"] is None
    assert invoke(workspace, args).returncode != 0
    assert not (workspace / "calls").exists()


def test_unavailable_revision_is_recorded_without_launch(workspace):
    (workspace / "bin/git").write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_CANARY"\n')
    assert invoke(workspace, ["supported", "6", MODEL]).returncode != 0
    raw = provenance(workspace, "v3-formal-supported-06").read_text()
    record = json.loads(raw)
    assert record["git_revision"] is None and record["phase"] == "revision_unavailable"
    assert CANARY not in raw and not (workspace / "calls").exists()


def test_abandoned_slot_without_a_record_cannot_be_reused(workspace):
    path = provenance(workspace, "v3-formal-vulnerable-04")
    path.parent.mkdir(parents=True)
    assert invoke(workspace, ["vulnerable", "4", MODEL]).returncode != 0
    assert not path.exists() and not (workspace / "calls").exists()


def test_concurrent_duplicate_and_interruption_preserve_attempt(workspace):
    args = ["vulnerable", "7", MODEL]
    process = subprocess.Popen(**command(workspace, args, "wait"), stdin=subprocess.PIPE)
    try:
        assert select.select([process.stdout], [], [], 10)[0] and process.stdout.readline() == "ready\n"
        path = provenance(workspace, "v3-formal-vulnerable-07")
        before = path.read_bytes()
        reserved = json.loads(before)
        assert reserved["phase"] == "reserved" and reserved["process_exit_status"] is None
        assert reserved["inspect_argv"][1:] == (workspace / "arguments").read_text().splitlines()
        assert invoke(workspace, args).returncode != 0 and path.read_bytes() == before
        process.terminate()
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
        final = json.loads(path.read_text())
        assert final["process_exit_status"] == final["runner_exit_status"] == 128 + signal.SIGTERM
        assert final["runner_signal"] == final["termination_signal"] == signal.SIGTERM and final["eval_path"] is None
        assert invoke(workspace, args).returncode != 0
        assert (workspace / "calls").read_text().splitlines() == ["invoked"]
    finally:
        process.communicate(input="release\n", timeout=10)


def test_existing_symlink_directory_is_not_used(workspace):
    target = workspace / "other"
    target.mkdir()
    (workspace / "logs").symlink_to(target, target_is_directory=True)
    assert invoke(workspace, ["supported", "2", MODEL]).returncode != 0
    assert not list(target.iterdir()) and not (workspace / "calls").exists()
