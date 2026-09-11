"""Validate the shell command with inert executables, never a provider."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_openai_admission_smoke.sh"
sys.path.insert(0, str(ROOT))

from evals.inspect import admission_behavioral_eval


TASKS = {
    "vulnerable": "admission_behavior_vulnerable_eval",
    "unsupported": "admission_behavior_unsupported_eval",
    "supported": "admission_behavior_supported_eval",
}


def _environment(tmp_path):
    """A PATH containing only inert python/inspect plus dirname for cd."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scripts = {
        "python": '#!/bin/sh\nprintf "import-check\\n" >> "$RUNNER_CALLS"\nexit 0\n',
        "inspect": (
            '#!/bin/sh\nprintf "inspect\\n" >> "$RUNNER_CALLS"\n'
            'printf "%s\\n" "$@" > "$RUNNER_ARGUMENTS"\n'
            'printf "%s\\n" "$PWD" > "$RUNNER_CWD"\n'
            'printf "Log: logs/synthetic-admission.eval\\n"\n'
            'exit "${RUNNER_EXIT:-0}"\n'
        ),
        "dirname": '#!/bin/sh\nprintf "%s\\n" "${2%/*}"\n',
    }
    for name, content in scripts.items():
        path = fake_bin / name
        path.write_text(content)
        path.chmod(0o755)
    # Do not inherit credentials or other user environment configuration.
    return {
        "PATH": str(fake_bin),
        "RUNNER_CALLS": str(tmp_path / "calls"),
        "RUNNER_ARGUMENTS": str(tmp_path / "arguments"),
        "RUNNER_CWD": str(tmp_path / "cwd"),
    }


def _run(arguments, environment, cwd=ROOT):
    return subprocess.run(
        ["/bin/bash", str(RUNNER), *arguments], cwd=cwd, env=environment,
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("condition", TASKS)
def test_exact_command_single_invocation_and_plain_log_output(tmp_path, condition):
    environment = _environment(tmp_path)
    result = _run([condition, "openai/gpt-4o-mini"], environment, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text().splitlines() == ["import-check", "inspect"]
    assert (tmp_path / "arguments").read_text().splitlines() == [
        "eval", f"evals/inspect/admission_behavioral_eval.py@{TASKS[condition]}",
        "--model", "openai/gpt-4o-mini",
        "--limit", "1", "--epochs", "1", "--no-epochs-reducer",
        "--max-retries", "0", "--retry-on-error", "0",
        "--no-log-realtime", "--log-dir", "logs", "--log-format", "eval",
        "--display", "plain",
    ]
    assert (tmp_path / "cwd").read_text().strip() == str(ROOT)
    assert result.stdout.splitlines() == [
        f"Condition: {condition}", "Model: openai/gpt-4o-mini",
        "Log: logs/synthetic-admission.eval",
    ]


@pytest.mark.parametrize("arguments", [
    [], ["vulnerable"], ["", "openai/model"], ["unknown", "openai/model"],
    ["governed_supported", "openai/model"], ["arbitrary.py@task", "openai/model"],
    ["supported", "openai/model", "extra"],
    *[["unsupported", model] for model in (
        "", "model", "openai/", "/model", "other/model", "--model=x",
        "openai/a,openai/b", "openai/model extra", "openai/model\nextra",
        "openai/../../file", "openai/model;exit", "openai/$(id)", "openai/`id`",
    )],
])
def test_invalid_input_stops_before_any_dependency_or_inspect_execution(tmp_path, arguments):
    environment = _environment(tmp_path)
    result = _run(arguments, environment)
    assert result.returncode == 64
    assert "Usage:" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("dependency", ["inspect", "openai"])
def test_missing_dependency_never_invokes_inspect(tmp_path, dependency):
    environment = _environment(tmp_path)
    if dependency == "inspect":
        (tmp_path / "bin" / "inspect").rename(tmp_path / "unused-inspect")
    else:
        (tmp_path / "bin" / "python").write_text("#!/bin/sh\nexit 1\n")
    result = _run(["vulnerable", "openai/model"], environment)
    assert result.returncode == 1
    assert result.stdout == ""
    assert not (tmp_path / "arguments").exists()


def test_inspect_failure_is_returned_without_retry(tmp_path):
    environment = _environment(tmp_path)
    environment["RUNNER_EXIT"] = "17"
    result = _run(["supported", "openai/model"], environment)
    assert result.returncode == 17
    assert (tmp_path / "calls").read_text().splitlines() == ["import-check", "inspect"]


def test_runner_has_no_secret_handling_installation_or_extra_outputs():
    source = RUNNER.read_text()
    for forbidden in (".env", "OPENAI_API_KEY", "source ", "read ", "cat ", "grep ",
                      "printenv", "pip ", "install ", "tee ", "eval-retry"):
        assert forbidden not in source
    assert re.search(r"^\s*(for|while|until)\b", source, re.MULTILINE) is None
    assert source.count('inspect eval "$TASK"') == 1
    assert os.access(RUNNER, os.X_OK)
    assert subprocess.run(
        ["git", "check-ignore", "-q", "logs/synthetic-admission.eval"],
        cwd=ROOT, check=False,
    ).returncode == 0


@pytest.mark.parametrize("entry_point", TASKS.values())
def test_runner_targets_existing_single_sample_tasks_with_unchanged_bounds(entry_point):
    task = getattr(admission_behavioral_eval, entry_point)()
    assert len(task.dataset) == 1
    assert task.model is None
    assert task.turn_limit == 8 and task.message_limit == 24
    assert task.score_on_error is True
    assert task.config.parallel_tool_calls is False
    params = task.solver.__dict__["__registry_params__"]["behavior"]["params"]
    assert params["attempts"] == 1
    assert params["retry_refusals"] == 0
