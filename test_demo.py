"""Structured and terminal-facing checks for the final deterministic demo."""

from experiments.indirect_authority_escalation.demo import (
    render_demo,
    run_demo,
    run_direct_scenario,
    run_governed_scenario,
    run_positive_control,
    run_vulnerable_scenario,
)
from experiments.indirect_authority_escalation.model import (
    AuthorityPromotionDecision,
    PermissionDecision,
    Proposition,
    WarehouseReceiptObservation,
    evidence_kind_supports,
    evidence_matches_candidate,
    source_authorized_for,
)


def test_demo_case_1_reports_direct_denial_and_zero_inventory() -> None:
    result = run_direct_scenario()

    assert (
        result.direct_attempt.capability_check.decision
        is PermissionDecision.DENIED
    )
    assert result.final_state.authoritative_effect_reached is False
    assert result.final_state.accepted_facts == ()
    assert result.final_state.inventory == 0


def test_demo_case_2_reports_all_allowed_edges_and_inventory_ten() -> None:
    result = run_vulnerable_scenario()
    execution = result.workflow_execution
    assert execution.promotion_result is not None
    promotion = execution.promotion_result
    assert promotion.append_result is not None

    assert result.direct_attempt.capability_check.decision is PermissionDecision.DENIED
    assert execution.request_submission_check.decision is PermissionDecision.ALLOWED
    assert (
        promotion.candidate_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert (
        promotion.append_result.capability_check.decision
        is PermissionDecision.ALLOWED
    )
    assert execution.candidate is not None
    assert result.final_state.authoritative_effect_reached is True
    assert len(result.final_state.accepted_facts) == 1
    assert result.final_state.inventory == 10


def test_demo_case_3_reports_candidate_rejection_and_zero_inventory() -> None:
    result = run_governed_scenario()
    candidate = result.candidate_preparation.candidate
    assert candidate is not None
    assert result.promotion.admission_result is not None
    admission = result.promotion.admission_result

    assert result.direct_attempt.capability_check.decision is PermissionDecision.DENIED
    assert (
        result.candidate_preparation.request_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert (
        result.promotion.candidate_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert result.agent_evidence.issued_by_modeled_source is True
    assert evidence_matches_candidate(result.agent_evidence, candidate) is True
    assert evidence_kind_supports(
        result.agent_evidence.kind,
        Proposition.RESTOCK_REQUESTED,
    ) is True
    assert evidence_kind_supports(
        result.agent_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    ) is False
    assert admission.decision is AuthorityPromotionDecision.REJECT
    assert result.promotion.append_result is None
    assert result.final_state.authoritative_effect_reached is False
    assert result.final_state.inventory == 0


def test_demo_positive_control_uses_receipt_basis_and_accepts() -> None:
    run = run_demo()
    result = run.positive_control
    governed_request_capability = (
        run.governed.candidate_preparation.request_submission_check.capability
    )
    positive_request_capability = (
        result.candidate_preparation.request_submission_check.capability
    )
    governed_candidate_submission_capability = (
        run.governed.promotion.candidate_submission_check.capability
    )
    positive_candidate_submission_capability = (
        result.promotion.candidate_submission_check.capability
    )
    governed_candidate = run.governed.candidate_preparation.candidate
    candidate = result.candidate_preparation.candidate
    assert candidate is not None
    assert result.promotion.admission_result is not None
    admission = result.promotion.admission_result

    assert isinstance(result.warehouse_receipt, WarehouseReceiptObservation)
    assert positive_request_capability == governed_request_capability
    assert (
        positive_candidate_submission_capability
        == governed_candidate_submission_capability
    )
    assert candidate == governed_candidate
    assert result.warehouse_evidence.candidate_id == candidate.candidate_id
    assert (
        result.warehouse_evidence.receipt_id
        == result.warehouse_receipt.receipt_id
    )
    assert (
        result.warehouse_evidence.product_id
        == result.warehouse_receipt.product_id
    )
    assert result.warehouse_evidence.quantity == result.warehouse_receipt.quantity
    assert evidence_matches_candidate(result.warehouse_evidence, candidate) is True
    assert evidence_kind_supports(
        result.warehouse_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    ) is True
    assert source_authorized_for(
        result.warehouse_evidence.issuer,
        Proposition.STOCK_REPLENISHED,
    ) is True
    assert admission.decision is AuthorityPromotionDecision.ACCEPT
    assert result.promotion.append_result is not None
    assert result.final_state.authoritative_effect_reached is True
    assert len(result.final_state.accepted_facts) == 1
    assert result.final_state.inventory == 10


def test_rendered_demo_exposes_all_four_scenarios() -> None:
    rendered = render_demo(run_demo())

    assert "CASE 1 — DIRECT DENIAL" in rendered
    assert "CASE 2 — LOCALLY VALID AUTHORITY COMPOSITION" in rendered
    assert "CASE 3 — GOVERNED SEMANTIC PROMOTION" in rendered
    assert "POSITIVE CONTROL — MATCHING WAREHOUSE AUTHORITY BASIS" in rendered
    assert "semantic authority admission: NOT PRESENT" in rendered
    assert "promotion: REJECT" in rendered
    assert "promotion: ACCEPT" in rendered
