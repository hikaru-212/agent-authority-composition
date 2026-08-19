"""Executable PR2 counterexamples for indirect authority composition."""

import pytest

from experiments.indirect_authority_escalation.model import (
    AcceptedStockReplenished,
    AuthoritativeInventoryStore,
    CandidateStockReplenished,
    Component,
    InventoryAuthorityService,
    LimitedAgent,
    LocalCapability,
    LocalOperation,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    PROTECTED_REPLENISHMENT,
    PermissionDecision,
    RestockRequest,
    RestockWorkflow,
)


def _build_vulnerable_model() -> tuple[
    LimitedAgent,
    RestockWorkflow,
    InventoryAuthorityService,
    AuthoritativeInventoryStore,
]:
    store = AuthoritativeInventoryStore()
    authority_service = InventoryAuthorityService(store)
    workflow = RestockWorkflow(authority_service)
    return LimitedAgent(), workflow, authority_service, store


def test_direct_agent_append_is_denied_without_mutating_authority() -> None:
    agent, _, _, store = _build_vulnerable_model()

    result = agent.attempt_direct_append(
        store=store,
        fact=PROTECTED_REPLENISHMENT,
    )

    assert result.capability_check.capability.caller is Component.LIMITED_AGENT
    assert (
        result.capability_check.capability.operation
        is LocalOperation.APPEND_ACCEPTED_REPLENISHMENT
    )
    assert result.capability_check.decision is PermissionDecision.DENIED
    assert result.attempted_fact == PROTECTED_REPLENISHMENT
    assert result.fact_appended is False
    assert store.accepted_facts == ()
    assert store.inventory(PROTECTED_PRODUCT_ID) == 0


def test_allowed_workflow_produces_and_submits_matching_candidate() -> None:
    agent, workflow, _, _ = _build_vulnerable_model()
    request = RestockRequest(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )

    result = agent.submit_restock_request(workflow=workflow, request=request)

    assert result.request_submission_check.decision is PermissionDecision.ALLOWED
    assert result.request_submission_check.capability == (
        LocalCapability(
            caller=Component.LIMITED_AGENT,
            operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
            target=Component.RESTOCK_WORKFLOW,
        )
    )
    assert result.candidate == CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )
    assert result.promotion_result is not None
    assert (
        result.promotion_result.candidate_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert result.promotion_result.candidate_submission_check.capability == (
        LocalCapability(
            caller=Component.RESTOCK_WORKFLOW,
            operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
            target=Component.INVENTORY_AUTHORITY_SERVICE,
        )
    )


def test_privileged_service_append_is_locally_allowed() -> None:
    _, _, _, store = _build_vulnerable_model()

    result = store.append_accepted_replenishment(
        caller=Component.INVENTORY_AUTHORITY_SERVICE,
        fact=PROTECTED_REPLENISHMENT,
    )

    assert result.capability_check.decision is PermissionDecision.ALLOWED
    assert result.capability_check.capability == LocalCapability(
        caller=Component.INVENTORY_AUTHORITY_SERVICE,
        operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
        target=Component.AUTHORITATIVE_INVENTORY_STORE,
    )
    assert result.fact_appended is True
    assert store.accepted_facts == (PROTECTED_REPLENISHMENT,)
    assert store.inventory(PROTECTED_PRODUCT_ID) == PROTECTED_QUANTITY


def test_locally_allowed_composed_path_reaches_directly_denied_effect() -> None:
    agent, workflow, _, store = _build_vulnerable_model()

    direct_result = agent.attempt_direct_append(
        store=store,
        fact=PROTECTED_REPLENISHMENT,
    )
    composed_result = agent.submit_restock_request(
        workflow=workflow,
        request=RestockRequest(
            product_id=PROTECTED_PRODUCT_ID,
            quantity=PROTECTED_QUANTITY,
        ),
    )

    assert direct_result.capability_check.decision is PermissionDecision.DENIED
    assert direct_result.capability_check.capability == LocalCapability(
        caller=Component.LIMITED_AGENT,
        operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
        target=Component.AUTHORITATIVE_INVENTORY_STORE,
    )
    assert direct_result.fact_appended is False

    assert (
        composed_result.request_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert composed_result.request_submission_check.capability == LocalCapability(
        caller=Component.LIMITED_AGENT,
        operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
        target=Component.RESTOCK_WORKFLOW,
    )
    assert composed_result.candidate is not None
    assert composed_result.promotion_result is not None
    promotion_result = composed_result.promotion_result
    assert (
        promotion_result.candidate_submission_check.decision
        is PermissionDecision.ALLOWED
    )
    assert promotion_result.candidate_submission_check.capability == LocalCapability(
        caller=Component.RESTOCK_WORKFLOW,
        operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
        target=Component.INVENTORY_AUTHORITY_SERVICE,
    )
    assert promotion_result.append_result is not None
    append_result = promotion_result.append_result
    assert append_result.capability_check.decision is PermissionDecision.ALLOWED
    assert append_result.capability_check.capability == LocalCapability(
        caller=Component.INVENTORY_AUTHORITY_SERVICE,
        operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
        target=Component.AUTHORITATIVE_INVENTORY_STORE,
    )

    assert composed_result.candidate == CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )
    assert append_result.appended_fact == PROTECTED_REPLENISHMENT
    assert store.accepted_facts == (PROTECTED_REPLENISHMENT,)
    assert store.inventory(PROTECTED_PRODUCT_ID) == PROTECTED_QUANTITY


def test_direct_and_composed_paths_target_the_same_authoritative_effect() -> None:
    direct_agent, _, _, direct_store = _build_vulnerable_model()
    composed_agent, composed_workflow, _, composed_store = (
        _build_vulnerable_model()
    )

    direct_result = direct_agent.attempt_direct_append(
        store=direct_store,
        fact=PROTECTED_REPLENISHMENT,
    )
    composed_result = composed_agent.submit_restock_request(
        workflow=composed_workflow,
        request=RestockRequest(
            product_id=PROTECTED_PRODUCT_ID,
            quantity=PROTECTED_QUANTITY,
        ),
    )

    assert direct_result.capability_check.decision is PermissionDecision.DENIED
    assert composed_result.promotion_result is not None
    assert composed_result.promotion_result.append_result is not None
    composed_effect = (
        composed_result.promotion_result.append_result.appended_fact
    )

    assert type(direct_result.attempted_fact) is AcceptedStockReplenished
    assert type(composed_effect) is AcceptedStockReplenished
    assert direct_result.attempted_fact == composed_effect
    assert direct_result.attempted_fact.product_id == PROTECTED_PRODUCT_ID
    assert direct_result.attempted_fact.quantity == PROTECTED_QUANTITY
    assert direct_store.accepted_facts == ()
    assert composed_store.accepted_facts == (PROTECTED_REPLENISHMENT,)


def test_candidate_cannot_be_appended_as_an_accepted_fact() -> None:
    _, _, _, store = _build_vulnerable_model()
    candidate = CandidateStockReplenished(
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )

    with pytest.raises(TypeError, match="fact must be AcceptedStockReplenished"):
        store.append_accepted_replenishment(
            caller=Component.INVENTORY_AUTHORITY_SERVICE,
            fact=candidate,  # type: ignore[arg-type]
        )

    assert store.accepted_facts == ()
    assert store.inventory(PROTECTED_PRODUCT_ID) == 0
