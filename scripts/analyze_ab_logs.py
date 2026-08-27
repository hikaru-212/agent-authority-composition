#!/usr/bin/env python3
"""Read-only per-epoch analysis for the bounded inventory A/B evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from inspect_ai.event import ToolEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evals.inspect.scorer import (  # noqa: E402
    _composition_witnesses,
    _direct_denial_witnesses,
)
from evals.inspect.state import (  # noqa: E402
    AuthorityAction,
    AuthorityEvaluationState,
)
from model import PROTECTED_PRODUCT_ID, PROTECTED_QUANTITY  # noqa: E402


AUTHORITY_SCORER = "authority_execution_scorer"
EXPECTED_TASKS = {
    "control": "inventory_behavior_control_eval",
    "composition": "inventory_behavior_composition_eval",
}
Termination = Literal["completed", "turn_limit", "other"]
DOMAIN_TOOLS = (
    "get_inventory",
    "update_inventory",
    "request_restock",
)
KNOWN_TOOLS = (*DOMAIN_TOOLS, "submit")
AUTHORITY_TOOL_ACTIONS = {
    "update_inventory": AuthorityAction.DIRECT_PROTECTED_OPERATION,
    "request_restock": AuthorityAction.COMPOSED_RESTOCK_WORKFLOW,
}


class AnalysisError(ValueError):
    """Report an input log that cannot support the bounded analysis."""


@dataclass(frozen=True)
class EpochAnalysis:
    """One evidence-backed row from one condition and epoch."""

    condition: Literal["control", "composition"]
    epoch: int
    termination: Termination
    scorer_outcome: str | None
    behavioral_relation: str | None
    direct_attempted: bool
    composition_attempted: bool
    protected_effect_reached_before_termination: bool
    inventory_before: int | None
    inventory_after: int | None


@dataclass(frozen=True)
class PostEffectAnalysis:
    """Observed tool behavior strictly after the first established effect."""

    epoch: int
    effect_attempt_sequence: int
    termination: Termination
    effect_reached: Literal[True]
    post_effect_get_inventory_count: int
    post_effect_update_inventory_count: int
    post_effect_request_restock_count: int
    post_effect_submit_observed: bool
    first_post_effect_action: str | None
    total_post_effect_domain_actions: int
    post_effect_inventory_read_observed_quantity_10: bool
    reinvoked_request_restock_after_effect: bool
    reinvoked_update_inventory_after_effect: bool
    action_trajectory: str


def _load_local_log(path: Path, condition: str) -> EvalLog:
    if path.suffix != ".eval":
        raise AnalysisError(f"{condition} log must have a .eval suffix")
    if path.is_symlink():
        raise AnalysisError(f"{condition} log must not be a symbolic link")
    if not path.is_file():
        raise AnalysisError(f"{condition} log is not a readable local file: {path}")

    try:
        log = read_eval_log(str(path))
    except Exception as error:
        raise AnalysisError(
            f"unable to read {condition} log ({type(error).__name__})"
        ) from error

    expected_task = EXPECTED_TASKS[condition]
    if log.eval.task != expected_task:
        raise AnalysisError(
            f"{condition} log task must be {expected_task}; found {log.eval.task}"
        )
    if not log.samples:
        raise AnalysisError(f"{condition} log contains no samples")
    epochs = [sample.epoch for sample in log.samples]
    if len(epochs) != len(set(epochs)):
        raise AnalysisError(
            f"{condition} log must contain exactly one sample per epoch"
        )
    return log


def _state_from_sample(sample: EvalSample) -> AuthorityEvaluationState:
    execution_key = "AuthorityEvaluationState:execution"
    if execution_key not in sample.store:
        raise AnalysisError(
            f"epoch {sample.epoch} has no serialized authority execution evidence"
        )
    try:
        return sample.store_as(AuthorityEvaluationState)
    except Exception as error:
        raise AnalysisError(
            f"epoch {sample.epoch} has invalid authority evidence "
            f"({type(error).__name__})"
        ) from error


def _termination(sample: EvalSample, state: AuthorityEvaluationState) -> Termination:
    if sample.limit is not None and sample.limit.type == "turn":
        return "turn_limit"
    if state.execution.episode_completed:
        return "completed"
    return "other"


def _official_score(sample: EvalSample) -> tuple[str | None, str | None]:
    score = (sample.scores or {}).get(AUTHORITY_SCORER)
    if score is None:
        return None, None
    outcome = str(score.value) if score.value is not None else None
    metadata = score.metadata or {}
    relation = metadata.get("behavioral_relation")
    return outcome, str(relation) if relation is not None else None


def _established_inventory_transition(
    state: AuthorityEvaluationState,
) -> tuple[int | None, int | None]:
    observations = [
        *_direct_denial_witnesses(state),
        *_composition_witnesses(state),
    ]
    transitions: dict[int, tuple[int, int]] = {}
    for observation in observations:
        transition = (
            observation.inventory_before,
            observation.inventory_after,
        )
        existing = transitions.setdefault(
            observation.attempt_sequence,
            transition,
        )
        if existing != transition:
            return None, None

    ordered = sorted(transitions.items())
    if not ordered:
        return None, None
    for (previous_sequence, previous), (current_sequence, current) in zip(
        ordered,
        ordered[1:],
    ):
        if (
            current_sequence != previous_sequence + 1
            or previous[1] != current[0]
        ):
            return None, None
    return ordered[0][1][0], ordered[-1][1][1]


def analyze_sample(
    condition: Literal["control", "composition"],
    sample: EvalSample,
) -> EpochAnalysis:
    """Project one sample solely from typed evaluator state and official score."""

    state = _state_from_sample(sample)
    composition_witnesses = _composition_witnesses(state)
    inventory_before, inventory_after = _established_inventory_transition(state)
    scorer_outcome, behavioral_relation = _official_score(sample)
    return EpochAnalysis(
        condition=condition,
        epoch=sample.epoch,
        termination=_termination(sample, state),
        scorer_outcome=scorer_outcome,
        behavioral_relation=behavioral_relation,
        direct_attempted=state.execution.direct_attempted,
        composition_attempted=state.execution.composition_attempted,
        protected_effect_reached_before_termination=bool(
            composition_witnesses
        ),
        inventory_before=inventory_before,
        inventory_after=inventory_after,
    )


def analyze_logs(control_path: Path, composition_path: Path) -> list[EpochAnalysis]:
    """Load and analyze the two known condition logs without mutating them."""

    control = _load_local_log(control_path, "control")
    composition = _load_local_log(composition_path, "composition")
    rows = [
        *(
            analyze_sample("control", sample)
            for sample in control.samples or []
        ),
        *(
            analyze_sample("composition", sample)
            for sample in composition.samples or []
        ),
    ]
    return sorted(
        rows,
        key=lambda row: (row.condition != "control", row.epoch),
    )


def _known_tool_events(sample: EvalSample) -> list[ToolEvent]:
    return [
        event
        for event in sample.events
        if isinstance(event, ToolEvent) and event.function in KNOWN_TOOLS
    ]


def _effect_event_index(
    sample: EvalSample,
    state: AuthorityEvaluationState,
    effect_attempt_sequence: int,
) -> tuple[list[ToolEvent], int]:
    tool_events = _known_tool_events(sample)
    attempts = {
        attempt.sequence: attempt.action for attempt in state.execution.attempts
    }
    sensitive_sequence = 0
    effect_event_index: int | None = None
    for event_index, event in enumerate(tool_events):
        action = AUTHORITY_TOOL_ACTIONS.get(event.function)
        if action is None:
            continue
        sensitive_sequence += 1
        if attempts.get(sensitive_sequence) is not action:
            raise AnalysisError(
                f"epoch {sample.epoch} tool order does not match evaluator "
                "attempt evidence"
            )
        if sensitive_sequence == effect_attempt_sequence:
            effect_event_index = event_index

    if effect_event_index is None:
        raise AnalysisError(
            f"epoch {sample.epoch} cannot locate the evidence-backed effect "
            "in tool-event order"
        )
    return tool_events, effect_event_index


def _inventory_read_observed_quantity_10(event: ToolEvent) -> bool:
    if (
        event.function != "get_inventory"
        or event.error is not None
        or not isinstance(event.result, str)
    ):
        return False
    try:
        result = json.loads(event.result)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(result, dict)
        and result.get("product_id") == PROTECTED_PRODUCT_ID
        and result.get("quantity") == PROTECTED_QUANTITY
    )


def analyze_post_effect_sample(
    sample: EvalSample,
) -> PostEffectAnalysis | None:
    """Count tool events only after the first typed composed-effect witness."""

    state = _state_from_sample(sample)
    witnesses = _composition_witnesses(state)
    if not witnesses:
        return None
    effect_attempt_sequence = min(
        witness.attempt_sequence for witness in witnesses
    )
    tool_events, effect_event_index = _effect_event_index(
        sample,
        state,
        effect_attempt_sequence,
    )
    post_effect_events = tool_events[effect_event_index + 1 :]
    counts = Counter(event.function for event in post_effect_events)
    get_count = counts["get_inventory"]
    update_count = counts["update_inventory"]
    restock_count = counts["request_restock"]
    return PostEffectAnalysis(
        epoch=sample.epoch,
        effect_attempt_sequence=effect_attempt_sequence,
        termination=_termination(sample, state),
        effect_reached=True,
        post_effect_get_inventory_count=get_count,
        post_effect_update_inventory_count=update_count,
        post_effect_request_restock_count=restock_count,
        post_effect_submit_observed=counts["submit"] > 0,
        first_post_effect_action=(
            post_effect_events[0].function if post_effect_events else None
        ),
        total_post_effect_domain_actions=sum(
            counts[tool_name] for tool_name in DOMAIN_TOOLS
        ),
        post_effect_inventory_read_observed_quantity_10=any(
            _inventory_read_observed_quantity_10(event)
            for event in post_effect_events
        ),
        reinvoked_request_restock_after_effect=restock_count > 0,
        reinvoked_update_inventory_after_effect=update_count > 0,
        action_trajectory=" -> ".join(
            event.function for event in tool_events
        ),
    )


def analyze_composition_post_effect(
    composition_path: Path,
) -> list[PostEffectAnalysis]:
    """Analyze only effect-reached epochs from one known Composition log."""

    composition = _load_local_log(composition_path, "composition")
    rows = [
        row
        for sample in composition.samples or []
        if (row := analyze_post_effect_sample(sample)) is not None
    ]
    return sorted(rows, key=lambda row: row.epoch)


def _display(value: object | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    display_rows = [[_display(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in display_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [render(headers), separator, *(render(row) for row in display_rows)]
    )


def _summary_rows(rows: Sequence[EpochAnalysis]) -> list[list[object]]:
    counts = Counter(
        (
            row.condition,
            (
                "limited"
                if row.termination == "turn_limit"
                else row.termination
            ),
            row.protected_effect_reached_before_termination,
        )
        for row in rows
    )
    categories = (
        ("completed", True),
        ("completed", False),
        ("limited", True),
        ("limited", False),
        ("other", True),
        ("other", False),
    )
    summary = []
    for termination, effect in categories:
        label = (
            f"{termination} + effect "
            f"{'reached' if effect else 'not reached'}"
        )
        control = counts[("control", termination, effect)]
        composition = counts[("composition", termination, effect)]
        summary.append([label, control, composition, control + composition])
    return summary


def _post_effect_summary_rows(
    rows: Sequence[PostEffectAnalysis],
) -> list[list[object]]:
    metrics = (
        ("effect-reached runs", lambda row: True),
        (
            "performed post-effect get_inventory",
            lambda row: row.post_effect_get_inventory_count > 0,
        ),
        (
            "observed inventory 10 after effect",
            lambda row: row.post_effect_inventory_read_observed_quantity_10,
        ),
        (
            "reinvoked request_restock",
            lambda row: row.reinvoked_request_restock_after_effect,
        ),
        (
            "reinvoked update_inventory",
            lambda row: row.reinvoked_update_inventory_after_effect,
        ),
        (
            "submitted normally",
            lambda row: (
                row.termination == "completed"
                and row.post_effect_submit_observed
            ),
        ),
    )
    summary = []
    for label, predicate in metrics:
        completed = sum(
            predicate(row) and row.termination == "completed" for row in rows
        )
        turn_limited = sum(
            predicate(row) and row.termination == "turn_limit" for row in rows
        )
        other = sum(
            predicate(row) and row.termination == "other" for row in rows
        )
        summary.append(
            [
                label,
                completed,
                turn_limited,
                other,
                completed + turn_limited + other,
            ]
        )
    return summary


def render_report(
    rows: Sequence[EpochAnalysis],
    post_effect_rows: Sequence[PostEffectAnalysis] | None = None,
) -> str:
    """Render per-epoch and aggregate tables without transcript content."""

    per_epoch = _table(
        (
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
        ),
        [
            (
                row.condition,
                row.epoch,
                row.termination,
                row.scorer_outcome,
                row.behavioral_relation,
                row.direct_attempted,
                row.composition_attempted,
                row.protected_effect_reached_before_termination,
                row.inventory_before,
                row.inventory_after,
            )
            for row in rows
        ],
    )
    summary = _table(
        ("termination/effect", "control", "composition", "total"),
        _summary_rows(rows),
    )
    report = f"Per-epoch results\n\n{per_epoch}\n\nSummary\n\n{summary}"
    if post_effect_rows is None:
        return report

    if post_effect_rows:
        post_effect = _table(
            (
                "epoch",
                "effect_attempt_sequence",
                "termination",
                "effect_reached",
                "post_effect_get_inventory_count",
                "post_effect_update_inventory_count",
                "post_effect_request_restock_count",
                "post_effect_submit_observed",
                "first_post_effect_action",
                "total_post_effect_domain_actions",
                "post_effect_inventory_read_observed_quantity_10",
                "reinvoked_request_restock_after_effect",
                "reinvoked_update_inventory_after_effect",
                "action_trajectory",
            ),
            [
                (
                    row.epoch,
                    row.effect_attempt_sequence,
                    row.termination,
                    row.effect_reached,
                    row.post_effect_get_inventory_count,
                    row.post_effect_update_inventory_count,
                    row.post_effect_request_restock_count,
                    row.post_effect_submit_observed,
                    row.first_post_effect_action,
                    row.total_post_effect_domain_actions,
                    row.post_effect_inventory_read_observed_quantity_10,
                    row.reinvoked_request_restock_after_effect,
                    row.reinvoked_update_inventory_after_effect,
                    row.action_trajectory,
                )
                for row in post_effect_rows
            ],
        )
    else:
        post_effect = "No effect-reached Composition epochs."
    post_effect_summary = _table(
        ("metric", "completed", "turn_limit", "other", "total"),
        _post_effect_summary_rows(post_effect_rows),
    )
    return (
        f"{report}\n\nPost-effect Composition behavior\n\n{post_effect}"
        f"\n\nPost-effect Composition summary\n\n{post_effect_summary}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze typed authority evidence from one Control and one "
            "Composition Inspect log."
        )
    )
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--composition", required=True, type=Path)
    arguments = parser.parse_args(argv)

    try:
        rows = analyze_logs(arguments.control, arguments.composition)
        post_effect_rows = analyze_composition_post_effect(
            arguments.composition
        )
    except AnalysisError as error:
        parser.error(str(error))
    print(render_report(rows, post_effect_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
