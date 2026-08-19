"""Executable PR3 semantic authority-admission cases and controls."""

import pytest

from experiments.indirect_authority_escalation.model import (
    AuthorityEvidence,
    AuthorityPromotionDecision,
    AuthoritativeInventoryStore,
    CandidatePreparationResult,
    CandidateStockReplenished,
    Component,
    EvidenceIssuer,
    EvidenceKind,
    InventoryAuthorityService,
    LimitedAgent,
    LocalCapability,
    LocalOperation,
    PROTECTED_CANDIDATE_ID,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    PROTECTED_REPLENISHMENT,
    PROTECTED_REQUEST_ID,
    PermissionDecision,
    Proposition,
    RestockRequest,
    RestockWorkflow,
    WarehouseAuthority,
    WarehouseReceiptObservation,
    evidence_kind_supports,
    evidence_matches_candidate,
    source_authorized_for,
)


def _build_model() -> tuple[
    LimitedAgent,
    RestockWorkflow,
    WarehouseAuthority,
    AuthoritativeInventoryStore,
]:
    store = AuthoritativeInventoryStore()
    authority_service = InventoryAuthorityService(store)
    workflow = RestockWorkflow(authority_service)
    return LimitedAgent(), workflow, WarehouseAuthority(), store


def _prepare_protected_candidate(
    *,
    agent: LimitedAgent,
    workflow: RestockWorkflow,
) -> CandidatePreparationResult:
    preparation = agent.prepare_restock_candidate(
        workflow=workflow,
        request=RestockRequest(
            product_id=PROTECTED_PRODUCT_ID,
            quantity=PROTECTED_QUANTITY,
            request_id=PROTECTED_REQUEST_ID,
        ),
    )
    assert preparation.candidate is not None
    return preparation


def _matching_warehouse_receipt() -> WarehouseReceiptObservation:
    return WarehouseReceiptObservation(
        receipt_id="warehouse-receipt-1",
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )


def test_case_3_keeps_workflow_edges_allowed_and_produces_candidate() -> None:
    agent, workflow, _, _ = _build_model()

    preparation = _prepare_protected_candidate(agent=agent, workflow=workflow)
    candidate = preparation.candidate
    assert candidate is not None
    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)
    governed_result = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(agent_evidence,),
    )

    assert preparation.request_submission_check.capability == LocalCapability(
        caller=Component.LIMITED_AGENT,
        operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
        target=Component.RESTOCK_WORKFLOW,
    )
    assert (
        preparation.request_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert candidate == CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
        candidate_id=PROTECTED_CANDIDATE_ID,
    )
    assert governed_result.candidate_submission_check.capability == LocalCapability(
        caller=Component.RESTOCK_WORKFLOW,
        operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
        target=Component.INVENTORY_AUTHORITY_SERVICE,
    )
    assert (
        governed_result.candidate_submission_check.decision
        is PermissionDecision.ALLOWED
    )


def test_exact_agent_evidence_is_correlated_but_semantically_insufficient() -> None:
    agent, workflow, _, store = _build_model()
    preparation = _prepare_protected_candidate(agent=agent, workflow=workflow)
    candidate = preparation.candidate
    assert candidate is not None

    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)
    governed_result = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(agent_evidence,),
    )

    assert agent_evidence.issued_by_modeled_source is True
    assert evidence_matches_candidate(agent_evidence, candidate) is True
    assert evidence_kind_supports(
        agent_evidence.kind,
        Proposition.RESTOCK_REQUESTED,
    ) is True
    assert evidence_kind_supports(
        agent_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    ) is False
    assert source_authorized_for(
        agent_evidence.issuer,
        Proposition.RESTOCK_REQUESTED,
    ) is True
    assert source_authorized_for(
        agent_evidence.issuer,
        Proposition.STOCK_REPLENISHED,
    ) is False

    assert governed_result.admission_result is not None
    admission_result = governed_result.admission_result
    assert admission_result.decision is AuthorityPromotionDecision.REJECT
    assert admission_result.qualifying_evidence is None
    assert len(admission_result.evidence_evaluations) == 1
    evaluation = admission_result.evidence_evaluations[0]
    assert evaluation.issued_by_modeled_source is True
    assert evaluation.matches_candidate is True
    assert evaluation.supports_stock_replenished is False
    assert evaluation.issuer_authorized_for_stock_replenished is False
    assert evaluation.qualifies_for_promotion is False
    assert governed_result.append_result is None
    assert store.accepted_facts == ()
    assert store.inventory(PROTECTED_PRODUCT_ID) == 0


def test_matching_warehouse_evidence_accepts_same_agent_origin_candidate() -> None:
    agent, workflow, warehouse, store = _build_model()
    preparation = _prepare_protected_candidate(agent=agent, workflow=workflow)
    candidate = preparation.candidate
    assert candidate is not None

    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)
    warehouse_receipt = _matching_warehouse_receipt()
    warehouse_evidence = warehouse.issue_receipt_confirmation(
        candidate=candidate,
        receipt=warehouse_receipt,
    )
    governed_result = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(agent_evidence, warehouse_evidence),
    )

    assert (
        preparation.request_submission_check.capability.caller
        is Component.LIMITED_AGENT
    )
    assert candidate == CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
        candidate_id=PROTECTED_CANDIDATE_ID,
    )
    assert warehouse_evidence.issued_by_modeled_source is True
    assert warehouse_evidence.candidate_id == candidate.candidate_id
    assert warehouse_evidence.receipt_id == warehouse_receipt.receipt_id
    assert warehouse_evidence.product_id == warehouse_receipt.product_id
    assert warehouse_evidence.quantity == warehouse_receipt.quantity
    assert evidence_matches_candidate(warehouse_evidence, candidate) is True
    assert evidence_kind_supports(
        warehouse_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    ) is True
    assert source_authorized_for(
        warehouse_evidence.issuer,
        Proposition.STOCK_REPLENISHED,
    ) is True

    assert governed_result.admission_result is not None
    admission_result = governed_result.admission_result
    assert admission_result.decision is AuthorityPromotionDecision.ACCEPT
    assert admission_result.qualifying_evidence == warehouse_evidence
    assert len(admission_result.evidence_evaluations) == 2
    warehouse_evaluation = admission_result.evidence_evaluations[1]
    assert warehouse_evaluation.issued_by_modeled_source is True
    assert warehouse_evaluation.matches_candidate is True
    assert warehouse_evaluation.supports_stock_replenished is True
    assert (
        warehouse_evaluation.issuer_authorized_for_stock_replenished
        is True
    )
    assert warehouse_evaluation.qualifies_for_promotion is True
    assert governed_result.append_result is not None
    assert governed_result.append_result.fact_appended is True
    assert store.accepted_facts == (PROTECTED_REPLENISHMENT,)
    assert store.inventory(PROTECTED_PRODUCT_ID) == PROTECTED_QUANTITY


@pytest.mark.parametrize(
    ("evidence_candidate", "warehouse_receipt"),
    (
        (
            CandidateStockReplenished(
                product_id=PROTECTED_PRODUCT_ID,
                quantity=PROTECTED_QUANTITY,
                candidate_id="candidate:other-request",
            ),
            WarehouseReceiptObservation(
                receipt_id="warehouse-receipt-candidate-id-mismatch",
                product_id=PROTECTED_PRODUCT_ID,
                quantity=PROTECTED_QUANTITY,
            ),
        ),
        (
            CandidateStockReplenished(
                product_id=PROTECTED_PRODUCT_ID,
                quantity=PROTECTED_QUANTITY,
                candidate_id=PROTECTED_CANDIDATE_ID,
            ),
            WarehouseReceiptObservation(
                receipt_id="warehouse-receipt-wrong-product",
                product_id="product-b",
                quantity=PROTECTED_QUANTITY,
            ),
        ),
        (
            CandidateStockReplenished(
                product_id=PROTECTED_PRODUCT_ID,
                quantity=PROTECTED_QUANTITY,
                candidate_id=PROTECTED_CANDIDATE_ID,
            ),
            WarehouseReceiptObservation(
                receipt_id="warehouse-receipt-wrong-quantity",
                product_id=PROTECTED_PRODUCT_ID,
                quantity=11,
            ),
        ),
    ),
    ids=("wrong-candidate-id", "wrong-product-id", "wrong-quantity"),
)
def test_warehouse_evidence_must_match_every_candidate_field(
    evidence_candidate: CandidateStockReplenished,
    warehouse_receipt: WarehouseReceiptObservation,
) -> None:
    agent, workflow, warehouse, store = _build_model()
    preparation = _prepare_protected_candidate(agent=agent, workflow=workflow)
    candidate = preparation.candidate
    assert candidate is not None

    mismatched_evidence = warehouse.issue_receipt_confirmation(
        candidate=evidence_candidate,
        receipt=warehouse_receipt,
    )
    governed_result = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(mismatched_evidence,),
    )

    assert mismatched_evidence.issued_by_modeled_source is True
    assert mismatched_evidence.candidate_id == evidence_candidate.candidate_id
    assert mismatched_evidence.receipt_id == warehouse_receipt.receipt_id
    assert mismatched_evidence.product_id == warehouse_receipt.product_id
    assert mismatched_evidence.quantity == warehouse_receipt.quantity
    assert evidence_kind_supports(
        mismatched_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    ) is True
    assert source_authorized_for(
        mismatched_evidence.issuer,
        Proposition.STOCK_REPLENISHED,
    ) is True
    assert evidence_matches_candidate(mismatched_evidence, candidate) is False
    assert governed_result.admission_result is not None
    evaluation = governed_result.admission_result.evidence_evaluations[0]
    assert evaluation.issued_by_modeled_source is True
    assert evaluation.matches_candidate is False
    assert evaluation.supports_stock_replenished is True
    assert evaluation.issuer_authorized_for_stock_replenished is True
    assert evaluation.qualifies_for_promotion is False
    assert (
        governed_result.admission_result.decision
        is AuthorityPromotionDecision.REJECT
    )
    assert governed_result.append_result is None
    assert store.accepted_facts == ()
    assert store.inventory(PROTECTED_PRODUCT_ID) == 0


def test_supported_issuance_api_does_not_let_workflow_impersonate_warehouse() -> None:
    agent, workflow, _, _ = _build_model()
    preparation = _prepare_protected_candidate(agent=agent, workflow=workflow)
    candidate = preparation.candidate
    assert candidate is not None

    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)

    assert agent_evidence.kind is EvidenceKind.AGENT_RESTOCK_REQUEST
    assert agent_evidence.issuer is EvidenceIssuer.LIMITED_AGENT
    assert not hasattr(workflow, "issue_receipt_confirmation")
    with pytest.raises(
        TypeError,
        match="evidence must be issued by a modeled source API",
    ):
        AuthorityEvidence(
            kind=EvidenceKind.WAREHOUSE_RECEIPT_CONFIRMATION,
            issuer=EvidenceIssuer.WAREHOUSE_AUTHORITY,
            candidate_id=candidate.candidate_id,
            receipt_id="warehouse-receipt-forged",
            product_id=candidate.product_id,
            quantity=candidate.quantity,
        )


def test_pr2_vulnerable_path_remains_executable_alongside_governed_path() -> None:
    agent, workflow, _, store = _build_model()

    vulnerable_result = agent.submit_restock_request(
        workflow=workflow,
        request=RestockRequest(
            product_id=PROTECTED_PRODUCT_ID,
            quantity=PROTECTED_QUANTITY,
            request_id=PROTECTED_REQUEST_ID,
        ),
    )

    assert vulnerable_result.candidate == CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
        candidate_id=PROTECTED_CANDIDATE_ID,
    )
    assert vulnerable_result.promotion_result is not None
    assert vulnerable_result.promotion_result.append_result is not None
    assert vulnerable_result.promotion_result.append_result.fact_appended is True
    assert store.accepted_facts == (PROTECTED_REPLENISHMENT,)
    assert store.inventory(PROTECTED_PRODUCT_ID) == PROTECTED_QUANTITY


def test_direct_agent_append_remains_denied_after_governed_path_is_added() -> None:
    agent, _, _, store = _build_model()

    direct_result = agent.attempt_direct_append(
        store=store,
        fact=PROTECTED_REPLENISHMENT,
    )

    assert direct_result.capability_check.decision is PermissionDecision.DENIED
    assert direct_result.fact_appended is False
    assert store.accepted_facts == ()
    assert store.inventory(PROTECTED_PRODUCT_ID) == 0
