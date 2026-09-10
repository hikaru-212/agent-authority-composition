#!/usr/bin/env python3
"""Read-only V2 reclassification and aggregation of local Inspect .eval logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence

from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.util import Store
from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.inspect.primitive_behavioral_scorer import classify_primitive_behavior
from evals.inspect.primitive_scorer import (
    PrimitiveAuthoritySnapshot,
    classify_primitive_execution,
)
from evals.inspect.state import AuthorityEvaluationState, RestockPrimitiveAttempt
from model import PROTECTED_QUANTITY


Condition = Literal["control", "treatment"]
EXPECTED_TASKS = {
    "control": "primitive_behavior_control_eval",
    "treatment": "primitive_behavior_treatment_eval",
}


class AnalysisError(ValueError):
    """Invalid input boundary or unreadable evidence; never include raw payloads."""


class SampleAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: Condition
    log_path: str
    model: str
    sample_id: int | str
    sample_uuid: str | None
    epoch: int
    semantic_outcome: str
    behavioral_outcome: str
    authoritative_effect_established: bool | None
    complete_effect_count: int
    effect_count_is_exact: bool
    model_directed_effect_count: int
    model_directed_effect_count_is_exact: bool
    final_evidence_backed_inventory: int | None
    first_effect_sequence: int | None
    post_effect_reinvocation: bool | None
    post_effect_primitive_attempts: tuple[RestockPrimitiveAttempt, ...] | None
    repeated_candidate_submission: bool | None
    repeated_candidate_submission_sequences: tuple[int, ...] | None
    model_primitive_call_count: int
    matched_primitive_attempt_count: int
    unexecuted_model_primitive_call_count: int
    unexecuted_model_primitive_call_ids: tuple[str, ...]
    normal_submit_observed: bool
    episode_completion: str
    normal_completion: bool
    turn_limit_reached: bool
    limit_type: str | None
    error_present: bool
    direct_denial_count: int
    incomplete_effect_attempt_sequences: tuple[int, ...]
    unattributed_accepted_fact_indices: tuple[int, ...] | None
    semantic_evidence_issues: tuple[str, ...]
    behavioral_evidence_issues: tuple[str, ...]
    analyzer_evidence_issues: tuple[str, ...]


class ConditionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: Condition
    sample_count: int
    models: tuple[str, ...]
    authoritative_effect_reached_count: int
    authoritative_effect_unknown_count: int
    semantic_outcome_distribution: dict[str, int]
    behavioral_outcome_distribution: dict[str, int]
    samples_with_model_directed_effect: int
    total_complete_effects: int
    complete_effect_total_is_exact: bool
    exact_complete_effects: int
    lower_bound_complete_effects: int
    samples_with_inexact_effect_count: int
    total_model_directed_effects: int
    model_directed_effect_total_is_exact: bool
    samples_with_inexact_model_directed_effect_count: int
    samples_with_repeated_authoritative_effects: int
    samples_with_inventory_above_protected_quantity: int
    final_inventory_distribution: dict[str, int]
    samples_with_post_effect_reinvocation: int
    samples_with_repeated_candidate_submission: int
    samples_with_unexecuted_model_primitive_calls: int
    total_model_primitive_calls: int
    total_matched_primitive_attempts: int
    total_unexecuted_model_primitive_calls: int
    normal_submit_observed_count: int
    episode_completed_count: int
    normal_completion_count: int
    turn_limit_count: int
    error_count: int
    insufficient_semantic_evidence_count: int
    insufficient_behavioral_evidence_count: int
    unknown_measurement_counts: dict[str, int]


class PrimaryComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: Condition
    candidate_preparation_available: bool
    sample_count: int
    authoritative_effect_established_samples: int
    model_directed_effect_established_samples: int


class AnalysisReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["v2-primitive-analysis-1"] = "v2-primitive-analysis-1"
    protected_quantity: int = PROTECTED_QUANTITY
    samples: tuple[SampleAnalysis, ...]
    condition_summaries: tuple[ConditionSummary, ...]
    primary_comparison: tuple[PrimaryComparison, ...]


def analyze_sample(condition: Condition, sample: EvalSample, *,
                   log_path: str = "", model: str = "") -> SampleAnalysis:
    """Reclassify a copy; never use the reconstructed PrivateAttr runtime."""

    if condition not in EXPECTED_TASKS:
        raise AnalysisError("unsupported condition")
    copied = sample.model_copy(deep=True)
    issues: list[str] = []
    snapshot = None
    score = (copied.scores or {}).get("primitive_execution_scorer")
    raw_snapshot = (score.metadata or {}).get("authority_snapshot") if score else None
    if raw_snapshot is not None:
        try:
            snapshot = PrimitiveAuthoritySnapshot.model_validate(raw_snapshot)
        except (ValidationError, TypeError):
            issues.append("invalid_persisted_authority_snapshot")
    # Missing serialized state is not a fresh, known-empty sample.
    if "AuthorityEvaluationState:artifact_namespace" not in copied.store:
        issues.append("missing_serialized_authority_state")
        snapshot = None
    try:
        state = copied.store_as(AuthorityEvaluationState)
        # Force validation of all fields used by the classifiers.
        state.model_dump()
    except (ValidationError, TypeError, ValueError):
        issues.append("malformed_serialized_authority_state")
        # Preserve the sample in the denominator, but do not bind an authority
        # snapshot to unusable state. The classifiers report unknown/insufficient.
        state = AuthorityEvaluationState(store=Store())
        snapshot = None
    semantic = classify_primitive_execution(state, authority_snapshot=snapshot)
    behavioral = classify_primitive_behavior(state, copied.events, authority_snapshot=snapshot)
    completed = semantic.episode_completion.value == "COMPLETED"
    return SampleAnalysis(
        condition=condition, log_path=log_path, model=model,
        sample_id=sample.id, sample_uuid=sample.uuid, epoch=sample.epoch,
        semantic_outcome=semantic.outcome.value,
        behavioral_outcome=behavioral.outcome.value,
        authoritative_effect_established=semantic.authoritative_effect_established,
        complete_effect_count=semantic.complete_effect_count,
        effect_count_is_exact=semantic.effect_count_is_exact,
        model_directed_effect_count=behavioral.model_directed_effect_count,
        model_directed_effect_count_is_exact=not behavioral.evidence_issues and not issues,
        final_evidence_backed_inventory=semantic.final_evidence_backed_inventory,
        first_effect_sequence=semantic.first_effect_sequence,
        post_effect_reinvocation=semantic.post_effect_reinvocation,
        post_effect_primitive_attempts=semantic.post_effect_primitive_attempts,
        repeated_candidate_submission=semantic.repeated_candidate_submission,
        repeated_candidate_submission_sequences=semantic.repeated_candidate_submission_sequences,
        model_primitive_call_count=behavioral.model_primitive_call_count,
        matched_primitive_attempt_count=behavioral.matched_primitive_attempt_count,
        unexecuted_model_primitive_call_count=behavioral.unexecuted_model_primitive_call_count,
        unexecuted_model_primitive_call_ids=behavioral.unexecuted_model_primitive_call_ids,
        normal_submit_observed=behavioral.normal_submit_observed,
        episode_completion=semantic.episode_completion.value,
        normal_completion=completed and behavioral.normal_submit_observed
        and sample.limit is None and sample.error is None,
        turn_limit_reached=sample.limit is not None and sample.limit.type == "turn",
        limit_type=sample.limit.type if sample.limit else None,
        error_present=sample.error is not None,
        direct_denial_count=semantic.direct_denial_count,
        incomplete_effect_attempt_sequences=semantic.incomplete_effect_attempt_sequences,
        unattributed_accepted_fact_indices=semantic.unattributed_accepted_fact_indices,
        semantic_evidence_issues=semantic.evidence_issues,
        behavioral_evidence_issues=behavioral.evidence_issues,
        analyzer_evidence_issues=tuple(issues),
    )


def summarize(condition: Condition, rows: Sequence[SampleAnalysis]) -> ConditionSummary:
    """Totals count witnesses; exactness is explicit and unknowns stay unknown."""

    rows = [row for row in rows if row.condition == condition]
    return ConditionSummary(
        condition=condition, sample_count=len(rows), models=tuple(sorted({r.model for r in rows})),
        authoritative_effect_reached_count=sum(r.authoritative_effect_established is True for r in rows),
        authoritative_effect_unknown_count=sum(r.authoritative_effect_established is None for r in rows),
        semantic_outcome_distribution=dict(sorted(Counter(r.semantic_outcome for r in rows).items())),
        behavioral_outcome_distribution=dict(sorted(Counter(r.behavioral_outcome for r in rows).items())),
        samples_with_model_directed_effect=sum(r.model_directed_effect_count > 0 for r in rows),
        total_complete_effects=sum(r.complete_effect_count for r in rows),
        complete_effect_total_is_exact=all(r.effect_count_is_exact for r in rows),
        exact_complete_effects=sum(r.complete_effect_count for r in rows if r.effect_count_is_exact),
        lower_bound_complete_effects=sum(r.complete_effect_count for r in rows if not r.effect_count_is_exact),
        samples_with_inexact_effect_count=sum(not r.effect_count_is_exact for r in rows),
        total_model_directed_effects=sum(r.model_directed_effect_count for r in rows),
        model_directed_effect_total_is_exact=all(r.model_directed_effect_count_is_exact for r in rows),
        samples_with_inexact_model_directed_effect_count=sum(not r.model_directed_effect_count_is_exact for r in rows),
        samples_with_repeated_authoritative_effects=sum(r.complete_effect_count > 1 for r in rows),
        samples_with_inventory_above_protected_quantity=sum(
            r.final_evidence_backed_inventory is not None
            and r.final_evidence_backed_inventory > PROTECTED_QUANTITY for r in rows
        ),
        final_inventory_distribution=dict(sorted(Counter(
            "unknown" if r.final_evidence_backed_inventory is None
            else str(r.final_evidence_backed_inventory) for r in rows
        ).items())),
        samples_with_post_effect_reinvocation=sum(r.post_effect_reinvocation is True for r in rows),
        samples_with_repeated_candidate_submission=sum(r.repeated_candidate_submission is True for r in rows),
        samples_with_unexecuted_model_primitive_calls=sum(r.unexecuted_model_primitive_call_count > 0 for r in rows),
        total_model_primitive_calls=sum(r.model_primitive_call_count for r in rows),
        total_matched_primitive_attempts=sum(r.matched_primitive_attempt_count for r in rows),
        total_unexecuted_model_primitive_calls=sum(r.unexecuted_model_primitive_call_count for r in rows),
        normal_submit_observed_count=sum(r.normal_submit_observed for r in rows),
        episode_completed_count=sum(r.episode_completion == "COMPLETED" for r in rows),
        normal_completion_count=sum(r.normal_completion for r in rows),
        turn_limit_count=sum(r.turn_limit_reached for r in rows),
        error_count=sum(r.error_present for r in rows),
        insufficient_semantic_evidence_count=sum(r.semantic_outcome == "INSUFFICIENT_EVIDENCE" for r in rows),
        insufficient_behavioral_evidence_count=sum(r.behavioral_outcome == "INSUFFICIENT_EVIDENCE" for r in rows),
        unknown_measurement_counts={field: sum(getattr(r, field) is None for r in rows) for field in (
            "authoritative_effect_established", "final_evidence_backed_inventory",
            "post_effect_reinvocation", "repeated_candidate_submission",
        )},
    )


def build_report(rows: Sequence[SampleAnalysis]) -> AnalysisReport:
    summaries = tuple(summarize(condition, rows) for condition in EXPECTED_TASKS
                      if any(row.condition == condition for row in rows))
    return AnalysisReport(
        samples=tuple(rows), condition_summaries=summaries,
        primary_comparison=tuple(PrimaryComparison(
            condition=s.condition, candidate_preparation_available=s.condition == "treatment",
            sample_count=s.sample_count,
            authoritative_effect_established_samples=s.authoritative_effect_reached_count,
            model_directed_effect_established_samples=s.samples_with_model_directed_effect,
        ) for s in summaries),
    )


def _local_path(path: Path) -> Path:
    """Reject escapes before filesystem access, then reject symlink components."""

    if "://" in str(path) or ".." in path.parts:
        raise AnalysisError("input must be a local worktree path without traversal")
    absolute = path if path.is_absolute() else ROOT / path
    try:
        parts = absolute.relative_to(ROOT).parts
    except ValueError:
        raise AnalysisError("input must remain inside this worktree") from None
    current = ROOT
    for part in parts:
        if part == ".env" or part.startswith(".env."):
            raise AnalysisError("environment files are not analysis inputs")
        current = current / part
        if current.is_symlink():
            raise AnalysisError("symbolic links are not analysis inputs")
    return absolute


def discover_logs(inputs: Sequence[Path]) -> list[Path]:
    found = []
    for supplied in inputs:
        path = _local_path(supplied)
        candidates = sorted(path.glob("*.eval")) if path.is_dir() else [path]
        if not candidates:
            raise AnalysisError("input directory contains no .eval files")
        for candidate in candidates:
            candidate = _local_path(candidate)
            if candidate.suffix != ".eval" or not candidate.is_file():
                raise AnalysisError("only regular .eval files are accepted")
            found.append(candidate)
    return found


def analyze_logs(*, control: Sequence[Path] = (), treatment: Sequence[Path] = ()) -> AnalysisReport:
    """Read explicit local logs once; reject duplicate inputs, never deduplicate effects."""

    rows = []
    seen_paths: set[Path] = set()
    seen_samples: set[tuple[str, str, int]] = set()
    for condition, inputs in (("control", control), ("treatment", treatment)):
        for path in discover_logs(inputs):
            if path in seen_paths:
                raise AnalysisError("duplicate input log")
            seen_paths.add(path)
            try:
                log = read_eval_log(path, resolve_attachments=False, format="eval")
            except Exception:
                raise AnalysisError("unable to read local .eval log") from None
            if log.eval.task != EXPECTED_TASKS[condition]:
                raise AnalysisError(f"{condition} input must contain the corresponding V2 task only")
            if not log.samples:
                raise AnalysisError("input log contains no samples")
            for sample in log.samples:
                identity = (log.eval.task_id, str(sample.id), sample.epoch)
                if identity in seen_samples:
                    raise AnalysisError("duplicate sample identity across input logs")
                seen_samples.add(identity)
                rows.append(analyze_sample(condition, sample,
                                           log_path=str(path.relative_to(ROOT)), model=log.eval.model))
    if not rows:
        raise AnalysisError("provide at least one control or treatment input")
    return build_report(rows)


def render_report(report: AnalysisReport) -> str:
    lines = [f"V2 primitive analysis ({report.schema_version})", "", "Primary comparison"]
    for comparison in report.primary_comparison:
        availability = "available" if comparison.candidate_preparation_available else "absent"
        lines.append(
            f"{comparison.condition} (preparation {availability}): "
            f"effect established {comparison.authoritative_effect_established_samples}/{comparison.sample_count}; "
            f"model-directed effect established {comparison.model_directed_effect_established_samples}/{comparison.sample_count}"
        )
    lines += ["", "Counts are witnessed lower bounds whenever the corresponding exactness flag is false.",
              "Unknown values are not false or zero. All samples remain in denominators."]
    for label, records in (("Per-sample evidence", report.samples),
                           ("Condition summaries", report.condition_summaries)):
        lines += ["", label]
        for record in records:
            lines.append("")
            lines.extend(f"  {key}: {json.dumps(value, ensure_ascii=True)}"
                         for key, value in record.model_dump(mode="json").items())
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--control", nargs="+", type=Path, default=[])
    parser.add_argument("--treatment", nargs="+", type=Path, default=[])
    parser.add_argument("--json", action="store_true", help="Print the versioned JSON report to stdout")
    args = parser.parse_args(argv)
    try:
        report = analyze_logs(control=args.control, treatment=args.treatment)
    except (AnalysisError, OSError) as error:
        parser.error(str(error) if isinstance(error, AnalysisError) else "unable to access local analysis inputs")
    print(report.model_dump_json(indent=2) if args.json else render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
