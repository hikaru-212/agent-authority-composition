"""Focused, no-network tests for the read-only A/B log analyzer."""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall
from pytest import MonkeyPatch

from evals.inspect.behavioral_eval import (
    inventory_behavior_composition_eval,
    inventory_behavior_control_eval,
)
from evals.inspect.scorer import (
    AuthorityBehavioralRelation,
    AuthoritySemanticOutcome,
)
from scripts.analyze_ab_logs import (
    AnalysisError,
    _summary_rows,
    analyze_logs,
    analyze_sample,
    render_report,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_ab_logs.py"


def _tool_call_output(call_id: str, function: str) -> ModelOutput:
    arguments = (
        {"answer": "Product A is available for purchase."}
        if function == "submit"
        else {}
    )
    return ModelOutput(
        model="mockllm",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            function=function,
                            arguments=arguments,
                        )
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def _run_mock_log(task: object, outputs: list[ModelOutput], log_dir: Path) -> Path:
    logs = inspect_eval(
        task,
        model=get_model(
            "mockllm/model",
            memoize=False,
            custom_outputs=outputs,
        ),
        display="none",
        log_dir=str(log_dir),
        log_realtime=False,
    )
    assert len(logs) == 1
    return Path(logs[0].location)


@pytest.fixture(scope="module")
def mock_ab_logs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    temporary_directory = tmp_path_factory.mktemp("ab-analyzer")
    monkeypatch = MonkeyPatch()
    monkeypatch.setenv(
        "INSPECT_TRACE_FILE",
        str(temporary_directory / "inspect-trace.log"),
    )
    try:
        control = _run_mock_log(
            inventory_behavior_control_eval(),
            [
                _tool_call_output("update-inventory", "update_inventory"),
                _tool_call_output("submit-result", "submit"),
            ],
            temporary_directory / "control",
        )
        limited_composition = _run_mock_log(
            inventory_behavior_composition_eval(),
            [
                _tool_call_output("request-restock", "request_restock"),
                *[
                    _tool_call_output(f"get-inventory-{index}", "get_inventory")
                    for index in range(1, 9)
                ],
            ],
            temporary_directory / "limited-composition",
        )
        language_only_composition = _run_mock_log(
            inventory_behavior_composition_eval(),
            [_tool_call_output("submit-claim", "submit")],
            temporary_directory / "language-only-composition",
        )
        yield {
            "control": control,
            "limited_composition": limited_composition,
            "language_only_composition": language_only_composition,
        }
    finally:
        monkeypatch.undo()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyzer_separates_turn_limit_from_pretermination_effect(
    mock_ab_logs: dict[str, Path],
) -> None:
    control_path = mock_ab_logs["control"]
    composition_path = mock_ab_logs["limited_composition"]
    digests_before = {
        path: _digest(path) for path in (control_path, composition_path)
    }

    rows = analyze_logs(control_path, composition_path)

    assert len(rows) == 2
    control, composition = rows
    assert control.condition == "control"
    assert control.epoch == 1
    assert control.termination == "completed"
    assert control.scorer_outcome == (
        AuthoritySemanticOutcome.DIRECT_DENIAL_ONLY.value
    )
    assert control.behavioral_relation == (
        AuthorityBehavioralRelation.DIRECT_ONLY.value
    )
    assert control.direct_attempted is True
    assert control.composition_attempted is False
    assert control.protected_effect_reached_before_termination is False
    assert (control.inventory_before, control.inventory_after) == (0, 0)

    assert composition.condition == "composition"
    assert composition.epoch == 1
    assert composition.termination == "turn_limit"
    assert composition.scorer_outcome == (
        AuthoritySemanticOutcome.INSUFFICIENT_EVIDENCE.value
    )
    assert composition.behavioral_relation == (
        AuthorityBehavioralRelation.NOT_ESTABLISHED.value
    )
    assert composition.direct_attempted is False
    assert composition.composition_attempted is True
    assert composition.protected_effect_reached_before_termination is True
    assert (composition.inventory_before, composition.inventory_after) == (
        0,
        10,
    )
    assert {
        path: _digest(path) for path in (control_path, composition_path)
    } == digests_before


def test_analyzer_does_not_infer_effect_from_natural_language(
    mock_ab_logs: dict[str, Path],
) -> None:
    rows = analyze_logs(
        mock_ab_logs["control"],
        mock_ab_logs["language_only_composition"],
    )
    composition = rows[1]

    assert composition.termination == "completed"
    assert composition.composition_attempted is False
    assert composition.protected_effect_reached_before_termination is False
    assert composition.inventory_before is None
    assert composition.inventory_after is None


def test_analyzer_reports_other_termination_and_unavailable_score(
    mock_ab_logs: dict[str, Path],
) -> None:
    rows = analyze_logs(
        mock_ab_logs["control"],
        mock_ab_logs["limited_composition"],
    )
    log = read_eval_log(str(mock_ab_logs["limited_composition"]))
    assert log.samples is not None
    sample = log.samples[0].model_copy(update={"limit": None, "scores": None})

    row = analyze_sample("composition", sample)

    assert row.termination == "other"
    assert row.scorer_outcome is None
    assert row.behavioral_relation is None
    assert row.protected_effect_reached_before_termination is True
    assert rows[1].termination == "turn_limit"


def test_report_contains_required_per_epoch_and_summary_tables(
    mock_ab_logs: dict[str, Path],
) -> None:
    rows = analyze_logs(
        mock_ab_logs["control"],
        mock_ab_logs["limited_composition"],
    )

    report = render_report(rows)

    for heading in (
        "condition",
        "epoch",
        "termination",
        "scorer_outcome",
        "behavioral_relation",
        "direct_attempted",
        "composition_attempted",
        "protected_effect_reached_before_termination",
        "inventory_before",
        "inventory_after",
    ):
        assert heading in report
    assert _summary_rows(rows)[:4] == [
        ["completed + effect reached", 0, 0, 0],
        ["completed + effect not reached", 1, 0, 1],
        ["limited + effect reached", 0, 1, 1],
        ["limited + effect not reached", 0, 0, 0],
    ]


def test_cli_reads_only_known_condition_logs(
    mock_ab_logs: dict[str, Path],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--control",
            str(mock_ab_logs["control"]),
            "--composition",
            str(mock_ab_logs["limited_composition"]),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Per-epoch results" in result.stdout
    assert "Summary" in result.stdout
    assert "turn_limit" in result.stdout
    assert "limited + effect reached" in result.stdout


def test_analyzer_rejects_log_in_the_wrong_condition_slot(
    mock_ab_logs: dict[str, Path],
) -> None:
    with pytest.raises(AnalysisError, match="control log task must be"):
        analyze_logs(
            mock_ab_logs["limited_composition"],
            mock_ab_logs["limited_composition"],
        )
