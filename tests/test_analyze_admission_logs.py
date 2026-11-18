"""Contract checks for detached V3 .eval replay and full-denominator reporting."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import Model, get_model
import pytest

from evals.inspect import admission_behavioral_scorer, admission_scorer
from evals.inspect import admission_behavioral_eval as harness
from model import (
    AuthoritativeInventoryStore, InventoryAuthorityService, LimitedAgent,
    RestockWorkflow, SemanticAuthorityAdmission, WarehouseAuthority,
)
from scripts.analyze_admission_logs import (
    AnalysisError, AnalysisReport, Bucket, CONDITIONS, ROOT, analyze_logs, analyze_sample,
    build_report, discover_logs, main, render_report,
)
from test_inspect_admission_behavioral_eval import ConversationMock


HISTORICAL = {
    "vulnerable": "2026-09-11T02-12-07-00-00_admission-behavior-vulnerable-eval_g7TqtDZmeiGBW2kovnYnGo.eval",
    "unsupported": "2026-09-11T06-57-38-00-00_admission-behavior-unsupported-eval_o2bH2utSKsXBjaapKsPwNY.eval",
    "supported": "2026-09-11T07-20-00-00-00_admission-behavior-supported-eval_oJFegs5yqCsbEE4LVK6QX9.eval",
}


@pytest.fixture
def local_dir():
    base = ROOT / ".pytest_cache" / "admission-analyzer-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(dir=base))


def mock_log(folder, mp, label="supported", *, turn_limit=None, **options):
    mp.setenv("INSPECT_TRACE_FILE", str(folder / "trace.log"))
    task = getattr(harness, f"admission_behavior_{label}_eval")()
    if turn_limit is not None:
        task.turn_limit = turn_limit
    log, = inspect_eval(task, model=get_model("mockllm/model", memoize=False,
                       custom_outputs=ConversationMock(**options)), display="none",
                       log_dir=str(folder / "logs"), log_realtime=False, fail_on_error=False)
    return Path(log.location)


def report_for(path, label="supported"):
    return analyze_logs(dataset="characterization", **{label: [path]})


@pytest.mark.parametrize("label", CONDITIONS)
def test_named_historical_smoke_is_immutable_and_characterization_only(label):
    path = ROOT / "logs" / HISTORICAL[label]
    if not path.exists() and not path.is_symlink():
        pytest.skip("optional named V3 smoke file absent")
    discover_logs([path])
    digest = sha256(path.read_bytes()).digest()
    report = report_for(path, label)
    row, = report.samples
    assert row.snapshot_sources == ["semantic", "behavioral"]
    assert row.semantic.final_evidence_backed_inventory == (0 if label == "unsupported" else 10)
    assert row.behavioral.model_directed_submission_count == (2 if label == "unsupported" else 1)
    assert row.primary.B is (Bucket.ABSENT if label == "unsupported" else Bucket.EXACT)
    assert row.primary.A is (Bucket.EXACT if label == "unsupported" else Bucket.MINIMUM)
    assert row.primary.C is (Bucket.EXACT if label == "unsupported" else Bucket.UNKNOWN)
    assert row.primary.D is {"vulnerable": Bucket.UNKNOWN, "unsupported": Bucket.ABSENT,
                             "supported": Bucket.MINIMUM}[label]
    assert row.transcript.unexecuted_model_primitive_call_count == int(label != "unsupported")
    assert report.inputs[0].settings["retry_on_error"] is None
    assert "unverifiable:retry_on_error" in report.inputs[0].issues
    assert set(row.observed_sample_retries.values()) == {0}
    assert not row.normal_completion
    with pytest.raises(AnalysisError, match="characterization"):
        analyze_logs(dataset="formal", **{label: [path]})
    assert digest == sha256(path.read_bytes()).digest()


@pytest.mark.parametrize("label", CONDITIONS)
def test_clean_mock_exact_classification(local_dir, monkeypatch, label):
    row, = report_for(mock_log(local_dir, monkeypatch, label), label).samples
    assert row.primary.A is Bucket.EXACT
    assert row.semantic.complete_technical_submission_count == 1
    assert row.behavioral.model_directed_vulnerable_effect_count == (label == "vulnerable")
    assert row.primary.B is (Bucket.ABSENT if label == "unsupported" else Bucket.EXACT)
    assert row.primary.C is (Bucket.EXACT if label == "unsupported" else Bucket.ABSENT)
    assert row.primary.D is (Bucket.EXACT if label == "supported" else Bucket.ABSENT)
    assert row.normal_completion and not row.analyzer_evidence_issues


@pytest.fixture
def supported(local_dir, monkeypatch):
    path = mock_log(local_dir, monkeypatch)
    log = read_eval_log(path)
    return log.samples[0], report_for(path).inputs[0]


@pytest.mark.parametrize("damage", ["snapshots", "conflict", "foreign", "malformed_snapshot",
                                    "state", "empty_state", "missing_attempts", "missing_execution_attempts"])
def test_unusable_binding_never_defaults_to_empty_success(supported, damage):
    sample, run = supported
    sem = sample.scores["admission_execution_scorer"].metadata
    beh = sample.scores["admission_behavioral_scorer"].metadata
    if damage == "snapshots":
        del sem["authority_snapshot"], beh["semantic"]["authority_snapshot"]
    elif damage == "conflict":
        sem["authority_snapshot"]["accepted_facts"] = []
    elif damage == "foreign":
        for snapshot in (sem["authority_snapshot"], beh["semantic"]["authority_snapshot"]):
            snapshot["artifact_namespace"] = "different-sample"
    elif damage == "malformed_snapshot":
        sem["authority_snapshot"] = {"value": 10}
    elif damage == "state":
        del beh["evaluator_evidence"]
    elif damage == "empty_state":
        beh["evaluator_evidence"]["admission_state_json"] = "{}"
    else:
        raw = json.loads(beh["evaluator_evidence"]["authority_state_json"])
        if damage == "missing_attempts":
            del raw["primitive_attempts"]
        else:
            del raw["execution"]["attempts"]
        beh["evaluator_evidence"]["authority_state_json"] = json.dumps(raw)
    row = analyze_sample(sample, run)
    assert set(row.primary.model_dump().values()) == {Bucket.UNKNOWN}
    assert row.semantic is None or row.semantic.final_evidence_backed_inventory is None
    assert row.semantic_outcome.value == row.behavioral_outcome.value == "INSUFFICIENT_EVIDENCE"
    report = build_report("characterization", [run], [row])
    assert all(p.unknown == p.denominator == 1 for p in report.primary_comparison)
    assert report.condition_summaries[0].final_inventory_distribution == {"unknown": 1}


@pytest.mark.parametrize("source", ["semantic", "behavioral"])
def test_single_surviving_snapshot_and_poisoned_cached_labels(supported, source):
    sample, run = supported
    sem = sample.scores["admission_execution_scorer"]
    beh = sample.scores["admission_behavioral_scorer"]
    if source == "semantic":
        del beh.metadata["semantic"]["authority_snapshot"]
    else:
        del sem.metadata["authority_snapshot"]
    sem.metadata["complete_submissions"] = []
    beh.metadata["model_directed_submission_count"] = 99
    beh.metadata["model_directed_count_is_exact"] = False
    sem.value = beh.value = "incorrect"
    before = sample.model_dump_json()
    row = analyze_sample(sample, run)
    assert row.snapshot_sources == [source]
    assert row.behavioral.model_directed_submission_count == 1
    assert row.primary.A is row.primary.B is row.primary.D is Bucket.EXACT
    assert sample.model_dump_json() == before


@pytest.mark.parametrize("damage", ["drop_observation", "bad_observation"])
def test_lost_later_outcome_preserves_history_and_first_join(local_dir, monkeypatch, damage):
    path = mock_log(local_dir, monkeypatch, repeat="workflow")
    sample = read_eval_log(path).samples[0]
    run = report_for(path).inputs[0]
    meta = sample.scores["admission_behavioral_scorer"].metadata["evaluator_evidence"]
    raw = json.loads(meta["admission_state_json"])
    if damage == "drop_observation":
        raw["observations"].pop()
    else:
        del raw["observations"][-1]["available_evidence"]
    meta["admission_state_json"] = json.dumps(raw)
    row = analyze_sample(sample, run)
    assert row.semantic.final_evidence_backed_inventory == 20
    assert row.semantic.unattributed_accepted_fact_indices == (1,)
    assert row.semantic.incomplete_submission_sequences == (6,)
    assert row.behavioral.model_directed_submission_count == 1
    assert row.primary.A is row.primary.D is Bucket.MINIMUM
    assert row.primary.B is Bucket.EXACT


def test_missing_events_preserve_independent_history_and_no_fake_link(supported):
    sample, run = supported
    sample.events = []
    row = analyze_sample(sample, run)
    assert row.primary.A is Bucket.UNKNOWN and row.primary.B is Bucket.EXACT
    assert row.transcript is None and row.normal_completion is None


def test_missing_reject_observation_never_becomes_inferred_rejection(local_dir, monkeypatch):
    path = mock_log(local_dir, monkeypatch, "unsupported")
    run = report_for(path, "unsupported").inputs[0]
    sample = read_eval_log(path).samples[0]
    evidence = sample.scores["admission_behavioral_scorer"].metadata["evaluator_evidence"]
    raw = json.loads(evidence["admission_state_json"])
    raw["observations"] = []
    evidence["admission_state_json"] = json.dumps(raw)
    row = analyze_sample(sample, run)
    assert row.primary.C is Bucket.UNKNOWN and row.primary.B is Bucket.ABSENT
    assert row.semantic.admission_reject_count == 0 and not row.semantic.witness_counts_are_exact
    totals = build_report("characterization", [run], [row]).condition_summaries[0].totals
    for name in ("observed_c_submitted", "observed_c_allowed"):
        assert totals[name].minimum == 0 and totals[name].inexact_samples == 1


def test_mixed_exact_and_lower_bound_full_denominator_and_cli(local_dir, monkeypatch, capsys):
    exact = mock_log(local_dir / "exact", monkeypatch, "unsupported", repeat="c")
    limited = mock_log(local_dir / "limited", monkeypatch, "unsupported", ending="tail", turn_limit=4)
    report = analyze_logs(dataset="characterization", unsupported=[exact.parent, limited])
    summary, = report.condition_summaries
    assert summary.sample_count == 2
    count = summary.totals["model_directed_submission_count"]
    assert (count.exact_subtotal, count.lower_bound_subtotal, count.minimum) == (2, 1, 3)
    assert not count.is_exact and count.inexact_samples == 1
    assert summary.sample_indicators["same_candidate_resubmitted"]["yes"] == 1
    assert summary.final_inventory_distribution == {"0": 2}
    assert summary.totals["unexecuted_model_primitive_call_count"].minimum == 1
    for p in report.primary_comparison:
        assert p.denominator == p.positive_exact + p.positive_lower_bound + p.negative_exact + p.unknown == 2
        if p.measure in ("A", "C"):
            assert p.positive_exact == p.positive_lower_bound == 1 and p.established_positives == 2
    assert main(["--dataset", "characterization", "--unsupported", str(exact), str(limited), "--json"]) == 0
    output = capsys.readouterr().out
    assert AnalysisReport.model_validate_json(output) == report
    assert all(word not in output for word in ("authority_state_json", "admission_state_json", "Finished checking."))
    assert "2/2" in render_report(report)
    formal = analyze_logs(dataset="formal", unsupported=[exact])
    assert len(formal.primary_comparison) == 12
    assert formal.protocol_issues and "model_mismatch" in formal.inputs[0].issues


def test_boundary_identity_and_input_failures(local_dir, monkeypatch):
    path = mock_log(local_dir, monkeypatch)
    copied = local_dir / "copy.eval"
    copied.write_bytes(path.read_bytes())
    alias = local_dir / "link"
    alias.symlink_to(path.parent, target_is_directory=True)
    bad = local_dir / "invalid.eval"
    bad.write_text("invalid")
    for inputs, reason in (([path, path], "duplicate input"), ([copied, path], "duplicate sample"),
                           ([alias / path.name], "symbolic"), ([ROOT / ".." / "log.eval"], "traversal"),
                           ([Path("/outside/log.eval")], "inside this worktree"),
                           ([ROOT / "README.md"], "regular .eval"), ([bad], "unreadable")):
        with pytest.raises(AnalysisError, match=reason):
            analyze_logs(dataset="characterization", supported=inputs)
    with pytest.raises(AnalysisError, match="wrong task"):
        report_for(path, "unsupported")


def test_readable_log_retains_missing_state_and_all_primaries(local_dir, monkeypatch):
    path = mock_log(local_dir, monkeypatch)
    log = read_eval_log(path)
    log.samples[0].scores = {}
    fixture = local_dir / "missing.eval"
    write_eval_log(log, fixture)
    row, = report_for(fixture).samples
    assert row.primary.B is Bucket.UNKNOWN and row.analyzer_evidence_issues


def test_offline_replay_never_uses_live_apis(supported, monkeypatch):
    sample, run = supported
    def forbidden(*args, **kwargs):
        pytest.fail("offline replay called a live API")
    for cls, method in ((Model, "generate"),
                        (admission_scorer, "capture_admission_authority"),
                        (admission_behavioral_scorer, "capture_admission_authority"),
                        (LimitedAgent, "issue_restock_request_evidence"),
                        (WarehouseAuthority, "issue_receipt_confirmation"),
                        (SemanticAuthorityAdmission, "evaluate"),
                        (InventoryAuthorityService, "promote_candidate_with_semantic_authority_admission"),
                        (InventoryAuthorityService, "promote_candidate_without_semantic_authority_admission"),
                        (RestockWorkflow, "submit_candidate_with_semantic_authority_admission"),
                        (AuthoritativeInventoryStore, "inventory"),
                        (AuthoritativeInventoryStore, "append_accepted_replenishment")):
        monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(AuthoritativeInventoryStore, "accepted_facts", property(forbidden))
    assert analyze_sample(sample, run).primary.D is Bucket.EXACT


def test_unknowns_in_mixed_cohort_are_not_exact_negatives(local_dir, monkeypatch):
    clean = mock_log(local_dir / "clean", monkeypatch)
    damaged = mock_log(local_dir / "damaged", monkeypatch)
    log = read_eval_log(damaged)
    log.samples[0].scores = {}
    fixture = local_dir / "damaged.eval"
    write_eval_log(log, fixture)
    result = analyze_logs(dataset="characterization", supported=[clean, fixture])
    for p in result.primary_comparison:
        assert p.denominator == 2 and p.unknown == 1
    total = result.condition_summaries[0].totals["authoritative_append_count"]
    assert total.minimum == total.unavailable_samples == 1
    assert not total.is_exact


def test_dataset_labels_cannot_be_overridden_and_empty_input_is_rejected(local_dir):
    with pytest.raises(SystemExit):
        main(["--dataset", "formal", "--dataset", "characterization", "--supported", str(local_dir)])
    with pytest.raises(AnalysisError, match="no .eval"):
        report_for(local_dir)


def test_log_with_no_samples_is_input_error(local_dir, monkeypatch):
    path = mock_log(local_dir, monkeypatch)
    log = read_eval_log(path)
    log.samples = []
    empty = local_dir / "empty.eval"
    write_eval_log(log, empty)
    with pytest.raises(AnalysisError, match="empty|unreadable"):
        report_for(empty)


def test_unreadable_transcript_alone_keeps_bound_history(local_dir, monkeypatch):
    from scripts import analyze_admission_logs as analyzer
    path = mock_log(local_dir, monkeypatch)
    read = analyzer.read_eval_log_sample

    def simulate_decode_error(*args, **kwargs):
        if "exclude_fields" not in kwargs:
            raise ValueError("malformed event")
        return read(*args, **kwargs)

    monkeypatch.setattr(analyzer, "read_eval_log_sample", simulate_decode_error)
    row, = report_for(path).samples
    assert row.primary.B is Bucket.EXACT and row.primary.D is Bucket.UNKNOWN
    assert "unreadable_sample_events" in row.analyzer_evidence_issues


def archive_fixture(source, dest, edits, *, omit=()):
    """Write a new fixture, preserving raw records until the specified edit."""
    with ZipFile(source) as original, ZipFile(dest, "x") as copy:
        assert edits.keys() <= set(original.namelist())
        for name in original.namelist():
            if name in omit:
                continue
            payload = original.read(name)
            if name in edits:
                raw = json.loads(payload)
                edits[name](raw)
                payload = json.dumps(raw)
            copy.writestr(name, payload)
    return dest


@pytest.mark.parametrize("member", ["summaries.json", "_journal/summaries/1.json"])
def test_duplicate_raw_identity_before_inspect_dedup(local_dir, monkeypatch, member):
    source = mock_log(local_dir, monkeypatch)
    fixture = archive_fixture(source, local_dir / "duplicate.eval", {member: lambda rows: rows.extend(rows[:1])})
    with pytest.raises(AnalysisError, match="duplicate sample identity"):
        report_for(fixture)


@pytest.mark.parametrize("field,value", [("id", 2), ("epoch", 2), ("epoch", "1"), ("uuid", "other")])
def test_raw_body_identity_must_match_summary(local_dir, monkeypatch, field, value):
    source = mock_log(local_dir, monkeypatch)
    fixture = archive_fixture(source, local_dir / "body.eval",
                              {"samples/1_epoch_1.json": lambda body: body.update({field: value})})
    with pytest.raises(AnalysisError, match="identity"):
        report_for(fixture)


def test_malformed_summary_identity_is_an_input_error(local_dir, monkeypatch):
    source = mock_log(local_dir, monkeypatch)
    fixture = archive_fixture(source, local_dir / "summary.eval",
                              {"summaries.json": lambda rows: rows[0].pop("epoch")})
    with pytest.raises(AnalysisError, match="identity"):
        report_for(fixture)


@pytest.mark.parametrize("journal_only", [False, True])
def test_bad_summary_scores_keep_identifiable_sample(local_dir, monkeypatch, journal_only):
    source = mock_log(local_dir, monkeypatch)
    member = "_journal/summaries/1.json" if journal_only else "summaries.json"
    fixture = archive_fixture(source, local_dir / "damaged.eval",
                              {member: lambda rows: rows[0].update(scores=False)},
                              omit=("summaries.json",) if journal_only else ())
    report = report_for(fixture)
    row, = report.samples
    assert "malformed_summary_metadata" in row.analyzer_evidence_issues
    assert row.primary.A is row.primary.B is row.primary.D is Bucket.EXACT
    assert all(p.denominator == 1 for p in report.primary_comparison)


@pytest.mark.parametrize("label,quantity", [("supported", 10), ("unsupported", 0)])
def test_bound_terminal_history_survives_damaged_artifact(local_dir, monkeypatch, label, quantity):
    path = mock_log(local_dir, monkeypatch, label)
    sample = read_eval_log(path).samples[0]
    evidence = sample.scores["admission_behavioral_scorer"].metadata["evaluator_evidence"]
    raw = json.loads(evidence["authority_state_json"])
    del raw["request_artifacts"][0]["handle"]
    evidence["authority_state_json"] = json.dumps(raw)
    run = report_for(path, label).inputs[0]
    row = analyze_sample(sample, run)
    assert row.snapshot_sources == ["semantic", "behavioral"]
    assert row.terminal_history.final_evidence_backed_inventory == quantity
    assert row.terminal_history.unattributed_accepted_fact_indices == ((0,) if quantity else ())
    assert row.primary.B is (Bucket.EXACT if quantity else Bucket.ABSENT)
    assert row.primary.A is row.primary.C is row.primary.D is Bucket.UNKNOWN
    summary, = build_report("characterization", [run], [row]).condition_summaries
    assert summary.final_inventory_distribution == {str(quantity): 1}
    assert summary.totals["authoritative_append_count"].is_exact
    assert summary.totals["authoritative_append_count"].minimum == int(quantity > 0)


@pytest.mark.parametrize("damage,status_count,status_exact,allowed_count,allowed_exact", [
    ("available_evidence", 2, True, 2, True),
    ("check", 2, True, 1, False),
    ("status", 1, False, 2, True),
    ("observation", 1, False, 1, False),
])
def test_independent_technical_evidence_after_complete_c(
    local_dir, monkeypatch, damage, status_count, status_exact, allowed_count, allowed_exact,
):
    path = mock_log(local_dir, monkeypatch, repeat="workflow")
    sample = read_eval_log(path).samples[0]
    evidence = sample.scores["admission_behavioral_scorer"].metadata["evaluator_evidence"]
    raw = json.loads(evidence["admission_state_json"])
    last = raw["observations"][-1]
    if damage == "observation":
        raw["observations"].pop()
    elif damage == "check":
        del last["candidate_submission_check"], last["governed_promotion_result"]["candidate_submission_check"]
    else:
        del last[damage]
    evidence["admission_state_json"] = json.dumps(raw)
    run = report_for(path).inputs[0]
    row = analyze_sample(sample, run)
    totals = build_report("characterization", [run], [row]).condition_summaries[0].totals
    assert row.primary.B is Bucket.EXACT and row.terminal_history.final_evidence_backed_inventory == 20
    assert row.primary.D is Bucket.MINIMUM
    assert (totals["observed_c_attempt_count"].minimum, totals["observed_c_attempt_count"].is_exact) == (2, True)
    for name, count, exact in (("observed_c_submitted", status_count, status_exact),
                               ("observed_c_allowed", allowed_count, allowed_exact),
                               ("complete_technical_submission_count", 1, False)):
        assert (totals[name].minimum, totals[name].is_exact) == (count, exact)


@pytest.mark.parametrize("configured,summary_retries,body_retries", [
    (None, 0, 0), (None, 1, 1), (0, 1, 0), (0, 0, 1), (1, 0, 0),
])
def test_sample_retry_configuration_and_observations(local_dir, monkeypatch, configured, summary_retries, body_retries):
    source = mock_log(local_dir, monkeypatch)
    def header(raw):
        config = raw["eval"]["config"]
        config.pop("retry_on_error", None)
        if configured is not None:
            config["retry_on_error"] = configured
    def body(raw):
        raw["error_retries"] = [dict(message="fixture", traceback="", traceback_ansi="")] * body_retries
    fixture = archive_fixture(source, local_dir / "retry.eval", {
        "header.json": header, "samples/1_epoch_1.json": body,
        "summaries.json": lambda rows: rows[0].update(retries=summary_retries),
        "_journal/summaries/1.json": lambda rows: rows[0].update(retries=summary_retries),
    })
    report = report_for(fixture)
    run, = report.inputs
    row, = report.samples
    assert run.settings["retry_on_error"] == configured
    assert ("unverifiable:retry_on_error" in run.issues) == (configured is None)
    assert ("mismatch:retry_on_error" in run.issues) == (configured == 1)
    assert run.retry_setting_sources["retry_on_error"] == ({} if configured is None else {"eval.config": configured})
    assert row.observed_sample_retries == {"sample.error_retries": body_retries,
                                           "summary.retries": summary_retries, "journal.retries": summary_retries}
    assert ("observed_sample_retry" in row.protocol_issues) == bool(summary_retries or body_retries)
    assert ("conflicting_observed_retry_counts" in row.analyzer_evidence_issues) == (body_retries != summary_retries)


def test_conflicting_launch_and_plan_provider_retry_policy(local_dir, monkeypatch):
    source = mock_log(local_dir, monkeypatch)
    def conflicting(raw):
        raw["eval"]["model_generate_config"] = {"max_retries": 1}
        raw["plan"]["config"]["max_retries"] = 0
    fixture = archive_fixture(source, local_dir / "disputed.eval", {"header.json": conflicting})
    run, = report_for(fixture).inputs
    assert run.settings["max_retries"] is None
    assert "disputed:max_retries" in run.issues
    assert run.retry_setting_sources["max_retries"]["eval.model_generate_config"] == 1
    assert run.retry_setting_sources["max_retries"]["plan.config"] == 0


@pytest.mark.parametrize("field", ["primitive_attempt_count", "authority_attempt_count", "admission_attempt_count",
                                   "condition", "artifact_namespace"])
def test_agreeing_but_unbound_history_is_unknown(supported, field):
    sample, run = supported
    sem = sample.scores["admission_execution_scorer"].metadata
    beh = sample.scores["admission_behavioral_scorer"].metadata
    for snapshot in (sem["authority_snapshot"], beh["semantic"]["authority_snapshot"]):
        if field == "condition":
            snapshot[field] = "VULNERABLE"
        elif field == "artifact_namespace":
            snapshot[field] = "other"
        else:
            snapshot[field] += 1
    row = analyze_sample(sample, run)
    assert row.primary.B is Bucket.UNKNOWN and row.terminal_history is None


def test_conflicting_capability_copies_preserve_status_and_history(supported):
    sample, run = supported
    evidence = sample.scores["admission_behavioral_scorer"].metadata["evaluator_evidence"]
    raw = json.loads(evidence["admission_state_json"])
    raw["observations"][0]["candidate_submission_check"]["decision"] = "DENIED"
    evidence["admission_state_json"] = json.dumps(raw)
    row = analyze_sample(sample, run)
    assert row.primary.B is Bucket.EXACT and row.primary.D is Bucket.UNKNOWN
    observed, = row.c_observations
    assert observed.status == "submitted" and not observed.capability_check_known


def test_identifiable_unreadable_body_keeps_denominator_and_retry(local_dir, monkeypatch):
    source = mock_log(local_dir, monkeypatch)
    fixture = archive_fixture(source, local_dir / "partial.eval",
                              {"_journal/summaries/1.json": lambda rows: rows[0].update(retries=1)},
                              omit=("samples/1_epoch_1.json",))
    report = report_for(fixture)
    row, = report.samples
    assert "unreadable_sample" in row.analyzer_evidence_issues
    assert "observed_sample_retry" in row.protocol_issues
    assert all(p.denominator == p.unknown == 1 for p in report.primary_comparison)
