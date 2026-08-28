"""Local, no-network checks for the OpenAI smoke-run security skeleton."""

import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
EPOCHS_HELPER = ROOT / "scripts" / "run_openai_epochs.sh"
SMOKE_HELPER = ROOT / "scripts" / "run_openai_smoke.sh"
PREFLIGHT_HELPER = ROOT / "scripts" / "security_preflight.sh"


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _stubbed_provider_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_inspect = fake_bin / "inspect"
    fake_inspect.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_inspect.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    return environment


def test_env_example_has_names_only_and_git_ignores_secret_env_files() -> None:
    assignments = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split("=", maxsplit=1)
        assignments[name] = value

    assert assignments == {
        "OPENAI_API_KEY": "",
        "OPENAI_PROJECT_ID": "",
    }
    assert _run(["git", "check-ignore", "-q", ".env"]).returncode == 0
    assert _run(["git", "check-ignore", "-q", ".env.local"]).returncode == 0
    assert _run(["git", "check-ignore", "-q", ".env.production"]).returncode == 0
    assert _run(["git", "check-ignore", "-q", ".env.example"]).returncode == 1
    assert _run(["git", "ls-files", "--error-unmatch", ".env"]).returncode != 0


def test_smoke_helper_has_no_embedded_key_and_requires_bounded_inputs() -> None:
    source = SMOKE_HELPER.read_text(encoding="utf-8")

    assert re.search(r"sk-[A-Za-z0-9_-]{16,}", source) is None
    assert "OPENAI_API_KEY" not in source
    assert ".env" not in source
    assert "source " not in source
    assert "read " not in source
    assert "command -v inspect" in source
    assert "python -c 'import openai'" in source
    assert 'case "$CONDITION"' in source
    assert "behavioral_eval.py@inventory_behavior_control_eval" in source
    assert "behavioral_eval.py@inventory_behavior_composition_eval" in source
    assert 'inspect eval "$TASK"' in source
    assert "--model \"$MODEL\"" in source
    assert "--display plain" in source

    result = _run([str(SMOKE_HELPER)])

    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert "inspect eval" not in result.stdout


def test_smoke_helper_rejects_non_openai_or_missing_model_without_inspect() -> None:
    wrong_provider = _run(
        [str(SMOKE_HELPER), "control", "other/synthetic-model"]
    )
    missing_name = _run([str(SMOKE_HELPER), "composition", "openai/"])
    arbitrary_task = _run(
        [
            str(SMOKE_HELPER),
            "evals/inspect/behavioral_eval.py@inventory_behavior_eval",
            "openai/synthetic-model",
        ]
    )
    too_many_arguments = _run(
        [
            str(SMOKE_HELPER),
            "control",
            "openai/synthetic-model",
            "extra",
        ]
    )

    assert wrong_provider.returncode != 0
    assert "model must begin with openai/" in wrong_provider.stderr
    assert missing_name.returncode != 0
    assert "include a model name" in missing_name.stderr
    assert arbitrary_task.returncode != 0
    assert "condition must be control or composition" in arbitrary_task.stderr
    assert too_many_arguments.returncode != 0
    assert "Usage:" in too_many_arguments.stderr


def test_smoke_helper_routes_allowlisted_conditions_without_network(
    tmp_path: Path,
) -> None:
    environment = _stubbed_provider_environment(tmp_path)

    expected_tasks = {
        "control": (
            "evals/inspect/behavioral_eval.py@"
            "inventory_behavior_control_eval"
        ),
        "composition": (
            "evals/inspect/behavioral_eval.py@"
            "inventory_behavior_composition_eval"
        ),
    }
    for condition, expected_task in expected_tasks.items():
        result = _run(
            [
                str(SMOKE_HELPER),
                condition,
                "openai/synthetic-model",
            ],
            env=environment,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "eval",
            expected_task,
            "--model",
            "openai/synthetic-model",
            "--display",
            "plain",
        ]


def test_epochs_helper_has_no_credential_handling_or_arbitrary_task_input() -> None:
    source = EPOCHS_HELPER.read_text(encoding="utf-8")

    assert re.search(r"sk-[A-Za-z0-9_-]{16,}", source) is None
    assert "OPENAI_API_KEY" not in source
    assert ".env" not in source
    assert "source " not in source
    assert "read " not in source
    assert 'case "$CONDITION"' in source
    assert "behavioral_eval.py@inventory_behavior_control_eval" in source
    assert "behavioral_eval.py@inventory_behavior_composition_eval" in source
    assert 'inspect eval "$TASK"' in source
    assert '--model "$MODEL"' in source
    assert '--epochs "$EPOCHS"' in source
    assert "--no-epochs-reducer" in source
    assert "--display plain" in source

    result = _run([str(EPOCHS_HELPER)])

    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert "inspect eval" not in result.stdout


def test_epochs_helper_rejects_invalid_condition_model_and_extra_arguments() -> None:
    invalid_condition = _run(
        [str(EPOCHS_HELPER), "unknown", "openai/synthetic-model", "10"]
    )
    arbitrary_task = _run(
        [
            str(EPOCHS_HELPER),
            "evals/inspect/behavioral_eval.py@inventory_behavior_eval",
            "openai/synthetic-model",
            "10",
        ]
    )
    invalid_provider = _run(
        [str(EPOCHS_HELPER), "control", "other/synthetic-model", "10"]
    )
    extra_argument = _run(
        [
            str(EPOCHS_HELPER),
            "control",
            "openai/synthetic-model",
            "10",
            "synthetic-credential",
        ]
    )

    assert invalid_condition.returncode != 0
    assert "condition must be control or composition" in invalid_condition.stderr
    assert arbitrary_task.returncode != 0
    assert "condition must be control or composition" in arbitrary_task.stderr
    assert invalid_provider.returncode != 0
    assert "model must begin with openai/" in invalid_provider.stderr
    assert extra_argument.returncode != 0
    assert "Usage:" in extra_argument.stderr


def test_epochs_helper_rejects_out_of_range_and_non_integer_epochs() -> None:
    for epochs in ("0", "-1", "1.5", "ten", "21"):
        result = _run(
            [
                str(EPOCHS_HELPER),
                "composition",
                "openai/synthetic-model",
                epochs,
            ]
        )

        assert result.returncode != 0
        assert "epochs must be an integer from 1 through 20" in result.stderr


def test_epochs_helper_constructs_exact_bounded_commands_without_network(
    tmp_path: Path,
) -> None:
    environment = _stubbed_provider_environment(tmp_path)
    expected_tasks = {
        "control": (
            "evals/inspect/behavioral_eval.py@"
            "inventory_behavior_control_eval"
        ),
        "composition": (
            "evals/inspect/behavioral_eval.py@"
            "inventory_behavior_composition_eval"
        ),
    }

    for condition, expected_task in expected_tasks.items():
        result = _run(
            [
                str(EPOCHS_HELPER),
                condition,
                "openai/synthetic-model",
                "10",
            ],
            env=environment,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "eval",
            expected_task,
            "--model",
            "openai/synthetic-model",
            "--epochs",
            "10",
            "--no-epochs-reducer",
            "--display",
            "plain",
        ]


def test_security_preflight_passes_for_repository_sources() -> None:
    result = _run([str(PREFLIGHT_HELPER)])

    assert result.returncode == 0, result.stderr
    assert "Security preflight passed." in result.stdout
    assert "not a complete secret scanner" in result.stdout


def test_security_preflight_reports_only_path_for_synthetic_marker() -> None:
    marker = "s" + "k-" + "SYNTHETIC_TEST_MARKER_000000000000"
    with tempfile.TemporaryDirectory(
        prefix="security-preflight-test-",
        dir=ROOT,
    ) as temporary_directory:
        suspicious_file = Path(temporary_directory) / "suspicious.txt"
        suspicious_file.write_text(marker, encoding="utf-8")

        result = _run([str(PREFLIGHT_HELPER)])

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert str(suspicious_file.relative_to(ROOT)) in output
        assert "Potential credential material detected" in output
        assert marker not in output
