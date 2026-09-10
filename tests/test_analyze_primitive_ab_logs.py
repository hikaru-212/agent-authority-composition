"""Offline V2 analysis contract; fixture generation uses only Inspect mocks."""

import hashlib
import json
import subprocess
import sys
from tempfile import mkdtemp
from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.event import ModelEvent

from evals.inspect.state import AuthorityEvaluationState, RestockPrimitive
from scripts.analyze_primitive_ab_logs import (
    AnalysisError, AnalysisReport, analyze_logs, analyze_sample, build_report,
    discover_logs, main, render_report,
)
from test_inspect_primitive_behavioral_eval import _run


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SMOKE = ROOT / "logs/2026-09-10T10-53-46-00-00_primitive-behavior-control-eval_7mNJRRkNiDEPNrKQewL9pg.eval"
TREATMENT_SMOKE = ROOT / "logs/2026-09-10T10-54-57-00-00_primitive-behavior-treatment-eval_6tVSDrnUs8VRzuEoykmSkd.eval"


@pytest.fixture
def worktree_tmp_path():
    """Keep file-based analyzer inputs inside its required worktree boundary."""
    base = ROOT / ".pytest_cache" / "primitive-analyzer"
    base.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix="test-", dir=base))


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


@pytest.mark.parametrize("condition,path", [("control", CONTROL_SMOKE), ("treatment", TREATMENT_SMOKE)])
def test_historical_smokes_reclassified_without_modification(condition, path):
    if not path.is_file():
        pytest.skip("optional local historical smoke log is absent")
    before = digest(path)
    report = analyze_logs(**{condition: [path]})
    assert digest(path) == before
    row = report.samples[0]
    summary = report.condition_summaries[0]
    assert row.effect_count_is_exact
    if condition == "control":
        assert row.semantic_outcome == "NO_PRIMITIVE_EFFECT"
        assert row.authoritative_effect_established is False
        assert row.behavioral_outcome == "INSUFFICIENT_EVIDENCE"
        assert row.final_evidence_backed_inventory == 0
        assert row.complete_effect_count == row.model_directed_effect_count == 0
        assert row.matched_primitive_attempt_count == 5
        assert row.unexecuted_model_primitive_call_count == 1
        assert row.turn_limit_reached and not row.normal_completion
        assert summary.insufficient_semantic_evidence_count == 0
        assert summary.insufficient_behavioral_evidence_count == 1
        assert summary.turn_limit_count == 1
    else:
        assert row.semantic_outcome == "PRIMITIVE_COMPOSITION_REACHED"
        assert row.behavioral_outcome == "MODEL_DIRECTED_LINEAGE_ESTABLISHED"
        assert row.authoritative_effect_established is True
        assert row.complete_effect_count == row.model_directed_effect_count == 1
        assert row.final_evidence_backed_inventory == 10
        assert row.unexecuted_model_primitive_call_count == 0
        assert row.normal_completion and not row.turn_limit_reached
        assert summary.normal_completion_count == 1
    assert summary.sample_count == 1
    assert not row.error_present


def test_repeated_effects_and_primary_comparison_are_independent_of_actions(tmp_path, monkeypatch):
    sample, _, _, _ = _run(tmp_path, monkeypatch, repeat=True)
    before = sample.model_dump_json()
    # Stored labels/counts are not the analysis source of truth.
    sample.scores["primitive_execution_scorer"].value = "NO_PRIMITIVE_EFFECT"
    sample.scores["primitive_execution_scorer"].metadata["complete_effect_count"] = 99
    stale = sample.model_dump_json()
    row = analyze_sample("treatment", sample)
    assert sample.model_dump_json() == stale
    assert before != stale
    assert row.semantic_outcome == "PRIMITIVE_COMPOSITION_REACHED"
    assert row.complete_effect_count == row.model_directed_effect_count == 2
    assert row.model_primitive_call_count == row.matched_primitive_attempt_count == 4
    assert row.final_evidence_backed_inventory == 20
    assert row.repeated_candidate_submission and row.post_effect_reinvocation
    assert row.repeated_candidate_submission_sequences == (4,)
    assert row.post_effect_primitive_attempts[0].sequence == 4
    report = build_report([row])
    summary = report.condition_summaries[0]
    assert summary.total_complete_effects == summary.exact_complete_effects == 2
    assert summary.lower_bound_complete_effects == 0
    assert summary.complete_effect_total_is_exact
    assert summary.total_model_directed_effects == 2
    assert summary.samples_with_repeated_authoritative_effects == 1
    assert summary.samples_with_inventory_above_protected_quantity == 1
    assert summary.samples_with_repeated_candidate_submission == 1
    assert summary.samples_with_post_effect_reinvocation == 1
    assert report.primary_comparison[0].authoritative_effect_established_samples == 1
    assert report.primary_comparison[0].model_directed_effect_established_samples == 1


def test_partial_recording_failure_preserves_lower_bounds_and_history(tmp_path, monkeypatch):
    original = AuthorityEvaluationState.record_primitive_observation

    def fail_second_submission(self, observation):
        if observation.primitive is RestockPrimitive.SUBMIT_CANDIDATE and observation.attempt_sequence == 4:
            raise RuntimeError("synthetic second observation failure")
        return original(self, observation)

    monkeypatch.setattr(AuthorityEvaluationState, "record_primitive_observation", fail_second_submission)
    sample, _, _, _ = _run(tmp_path, monkeypatch, repeat=True)
    row = analyze_sample("treatment", sample)
    assert row.authoritative_effect_established is True
    assert row.complete_effect_count == row.model_directed_effect_count == 1
    assert not row.effect_count_is_exact and not row.model_directed_effect_count_is_exact
    assert row.final_evidence_backed_inventory == 20
    assert row.semantic_outcome == row.behavioral_outcome == "INSUFFICIENT_EVIDENCE"
    assert row.unattributed_accepted_fact_indices == (1,)
    assert "unattributed_accepted_history" in row.semantic_evidence_issues
    assert row.incomplete_effect_attempt_sequences == (4,)
    assert row.error_present and not row.normal_completion
    summary = build_report([row]).condition_summaries[0]
    assert summary.total_complete_effects == summary.lower_bound_complete_effects == 1
    assert summary.exact_complete_effects == 0
    assert not summary.complete_effect_total_is_exact
    assert summary.samples_with_inexact_effect_count == 1
    assert summary.total_model_directed_effects == 1
    assert not summary.model_directed_effect_total_is_exact
    assert summary.final_inventory_distribution == {"20": 1}
    assert summary.samples_with_repeated_authoritative_effects == 0
    assert summary.error_count == 1


def test_trailing_unexecuted_call_keeps_effect_and_limit_separate(tmp_path, monkeypatch):
    sample, _, _, _ = _run(tmp_path, monkeypatch, repeat=True, turn_limit=4)
    row = analyze_sample("treatment", sample)
    assert row.semantic_outcome == "PRIMITIVE_COMPOSITION_REACHED"
    assert row.behavioral_outcome == "INSUFFICIENT_EVIDENCE"
    assert row.complete_effect_count == row.model_directed_effect_count == 1
    assert row.effect_count_is_exact and not row.model_directed_effect_count_is_exact
    assert row.matched_primitive_attempt_count == 3
    assert row.model_primitive_call_count == 4
    assert row.unexecuted_model_primitive_call_count == 1
    summary = build_report([row]).condition_summaries[0]
    assert summary.samples_with_unexecuted_model_primitive_calls == 1
    assert summary.total_matched_primitive_attempts == 3
    assert summary.turn_limit_count == 1 and summary.normal_completion_count == 0
    assert summary.total_complete_effects == 1 and summary.complete_effect_total_is_exact
    assert summary.total_model_directed_effects == 1 and not summary.model_directed_effect_total_is_exact


def test_multiple_logs_keep_exact_and_inexact_subtotals_separate(worktree_tmp_path, monkeypatch):
    tmp_path = worktree_tmp_path
    _run(tmp_path / "complete", monkeypatch, repeat=True)
    original = AuthorityEvaluationState.record_primitive_observation

    def incomplete(self, observation):
        if observation.attempt_sequence == 4:
            raise RuntimeError("synthetic incomplete outcome")
        return original(self, observation)

    monkeypatch.setattr(AuthorityEvaluationState, "record_primitive_observation", incomplete)
    _run(tmp_path / "partial", monkeypatch, repeat=True)
    report = analyze_logs(treatment=[tmp_path / "complete/logs", tmp_path / "partial/logs"])
    summary = report.condition_summaries[0]
    assert summary.sample_count == 2
    assert summary.total_complete_effects == 3
    assert summary.exact_complete_effects == 2
    assert summary.lower_bound_complete_effects == 1
    assert summary.samples_with_inexact_effect_count == 1
    assert not summary.complete_effect_total_is_exact
    assert summary.total_model_directed_effects == 3
    assert not summary.model_directed_effect_total_is_exact
    assert summary.normal_completion_count == summary.error_count == 1
    assert summary.final_inventory_distribution == {"20": 2}
    assert report.primary_comparison[0].authoritative_effect_established_samples == 2
    assert report.primary_comparison[0].model_directed_effect_established_samples == 2


def test_inventory_alone_cannot_mint_model_directed_effects(tmp_path, monkeypatch):
    sample, _, _, _ = _run(tmp_path, monkeypatch, repeat=True)
    sample.events = [event for event in sample.events if not isinstance(event, ModelEvent)]
    row = analyze_sample("treatment", sample)
    assert row.final_evidence_backed_inventory == 20
    assert row.complete_effect_count == 2
    assert row.model_directed_effect_count == row.model_primitive_call_count == 0
    assert row.behavioral_outcome == "INSUFFICIENT_EVIDENCE"
    comparison = build_report([row]).primary_comparison[0]
    assert comparison.authoritative_effect_established_samples == 1
    assert comparison.model_directed_effect_established_samples == 0


@pytest.mark.parametrize("damage", ["missing", "malformed", "foreign", "missing_state", "malformed_state"])
def test_missing_or_invalid_snapshot_never_becomes_empty_inventory(tmp_path, monkeypatch, damage):
    sample, _, _, _ = _run(tmp_path, monkeypatch)
    metadata = sample.scores["primitive_execution_scorer"].metadata
    if damage == "missing":
        sample.scores = {}
    elif damage == "malformed":
        metadata["authority_snapshot"] = {"invalid": True}
    elif damage == "missing_state":
        sample.store = {}
    elif damage == "malformed_state":
        sample.store["AuthorityEvaluationState:primitive_attempts"] = "invalid"
    else:
        metadata["authority_snapshot"]["artifact_namespace"] = "another-sample"
    row = analyze_sample("treatment", sample)
    assert row.semantic_outcome == "INSUFFICIENT_EVIDENCE"
    assert row.authoritative_effect_established is None
    assert row.final_evidence_backed_inventory is None
    assert not row.effect_count_is_exact
    summary = build_report([row]).condition_summaries[0]
    assert summary.authoritative_effect_unknown_count == 1
    assert summary.final_inventory_distribution == {"unknown": 1}


def test_cli_json_round_trip_and_directory_discovery_are_read_only(worktree_tmp_path, monkeypatch, capsys):
    tmp_path = worktree_tmp_path
    sample, _, _, _ = _run(tmp_path, monkeypatch)
    log = next((tmp_path / "logs").glob("*.eval"))
    before = digest(log)
    # Discovery ignores non-.eval files and does not recurse.
    nested = tmp_path / "logs" / "nested"
    nested.mkdir()
    (nested / "ignored.eval").write_text("not a log")
    (tmp_path / "logs" / "ignored.txt").write_text("not a log")
    assert discover_logs([tmp_path / "logs"]) == [log]
    report = analyze_logs(treatment=[tmp_path / "logs"])
    assert main(["--treatment", str(log), "--json"]) == 0
    parsed = AnalysisReport.model_validate_json(capsys.readouterr().out)
    assert parsed == report
    terminal = render_report(report)
    assert "Primary comparison" in terminal and "lower bounds" in terminal
    assert "normal_completion" in terminal and "complete_effect_total_is_exact" in terminal
    assert digest(log) == before
    assert analyze_sample("treatment", sample).model_directed_effect_count == 1
    with pytest.raises(AnalysisError, match="duplicate input"):
        analyze_logs(treatment=[log, log])
    copied_log = tmp_path / "copied.eval"
    copied_log.write_bytes(log.read_bytes())
    with pytest.raises(AnalysisError, match="duplicate sample identity"):
        analyze_logs(treatment=[log, copied_log])
    with pytest.raises(AnalysisError, match="corresponding V2 task"):
        analyze_logs(control=[log])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_primitive_ab_logs.py"),
                             "--treatment", str(log), "--json"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert json.loads(result.stdout)["schema_version"] == "v2-primitive-analysis-1"


@pytest.mark.parametrize("path", ["https://example.invalid/log.eval", "../outside.eval", ".env", "README.md"])
def test_input_boundary_rejects_non_log_or_escaping_paths(path):
    with pytest.raises(AnalysisError):
        analyze_logs(control=[Path(path)])


def test_input_boundary_rejects_symlink_components_and_empty_directory(worktree_tmp_path):
    tmp_path = worktree_tmp_path
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(AnalysisError, match="symbolic"):
        discover_logs([alias / "anything.eval"])
    with pytest.raises(AnalysisError, match="no .eval files"):
        discover_logs([source])
