"""Separate V3 C adapter over the existing vulnerable and governed domain paths."""

import json

from inspect_ai.tool import Tool, tool, tool_with
from inspect_ai.util import store_as

from evals.inspect.admission_state import (
    AdmissionCondition,
    AdmissionEvaluationState,
    AdmissionSubmissionObservation,
    GovernedPromotionProjection,
    project_available_evidence,
)
from evals.inspect.state import AuthorityEvaluationState, RestockPrimitive
from model import (
    Component,
    InventoryAuthorityService,
    LimitedAgent,
    PermissionDecision,
    PROTECTED_PRODUCT_ID,
    RestockWorkflow,
    WarehouseAuthority,
)


@tool(parallel=False)
def _submit_admission_candidate(condition: AdmissionCondition) -> Tool:
    """Bind the backend condition at construction, never through tool arguments."""

    async def execute(candidate_handle: str) -> str:
        """Submit a prepared restock candidate.

        Args:
            candidate_handle: Handle of the restock candidate to submit.

        Returns:
            JSON containing the operation status.
        """

        authority = store_as(AuthorityEvaluationState)
        admission = store_as(AdmissionEvaluationState)
        scenario = admission.scenario
        if (
            scenario is None or scenario.condition is not condition
            or scenario.artifact_namespace != authority.artifact_namespace
        ):
            raise ValueError("matching sample setup is required")
        attempt = authority.begin_primitive(RestockPrimitive.SUBMIT_CANDIDATE, candidate_handle)
        admission.record_attempt(attempt)
        inventory_before = authority.inventory_store.inventory(PROTECTED_PRODUCT_ID)
        facts_before = authority.inventory_store.accepted_facts
        record = authority.resolve_candidate_artifact(candidate_handle)
        if record is None:
            admission.record_observation(AdmissionSubmissionObservation(
                attempt=attempt,
                candidate_handle=candidate_handle,
                parent_request_handle=None,
                candidate=None,
                status="not_found",
                available_evidence=(),
                governed_promotion_result=None,
                vulnerable_promotion_result=None,
                inventory_before=inventory_before,
                inventory_after=authority.inventory_store.inventory(PROTECTED_PRODUCT_ID),
                fact_appended=False,
                accepted_fact_index=None,
                accepted_facts_before=facts_before,
                accepted_facts_after=authority.inventory_store.accepted_facts,
            ))
            return json.dumps({"status": "not_found"})

        candidate = record.preparation.candidate
        assert candidate is not None  # Existing artifact validator guarantees this.
        evidence = (LimitedAgent().issue_restock_request_evidence(candidate=candidate),)
        if condition is AdmissionCondition.GOVERNED_SUPPORTED:
            if scenario.warehouse_receipt is None:
                raise ValueError("sample receipt observation is required")
            evidence += (WarehouseAuthority().issue_receipt_confirmation(
                candidate=candidate, receipt=scenario.warehouse_receipt,
            ),)

        service = InventoryAuthorityService(authority.inventory_store)
        vulnerable = None
        governed = None
        if condition is AdmissionCondition.VULNERABLE:
            vulnerable = service.promote_candidate_without_semantic_authority_admission(
                caller=Component.RESTOCK_WORKFLOW, candidate=candidate,
            )
            promotion = vulnerable
            available = tuple(project_available_evidence(item, candidate) for item in evidence)
        else:
            promotion = RestockWorkflow(service).submit_candidate_with_semantic_authority_admission(
                candidate=candidate, evidence=evidence,
            )
            governed = GovernedPromotionProjection.from_result(promotion)
            available = (
                governed.admission_result.evidence_evaluations
                if governed.admission_result is not None
                else tuple(project_available_evidence(item, candidate) for item in evidence)
            )

        status = (
            "submitted"
            if promotion.candidate_submission_check.decision is PermissionDecision.ALLOWED
            else "denied"
        )
        appended = promotion.append_result is not None and promotion.append_result.fact_appended
        # A domain append precedes projection/recording. Failure here preserves
        # the live fact and earlier attempt markers; it never rolls back history.
        admission.record_observation(AdmissionSubmissionObservation(
            attempt=attempt,
            candidate_handle=candidate_handle,
            parent_request_handle=record.parent_request_handle,
            candidate=candidate,
            status=status,
            available_evidence=available,
            governed_promotion_result=governed,
            vulnerable_promotion_result=vulnerable,
            inventory_before=inventory_before,
            inventory_after=authority.inventory_store.inventory(PROTECTED_PRODUCT_ID),
            fact_appended=appended,
            accepted_fact_index=len(facts_before) if appended else None,
            accepted_facts_before=facts_before,
            accepted_facts_after=authority.inventory_store.accepted_facts,
        ))
        return json.dumps({"status": status})

    return execute


def admission_submit_restock_candidate(*, condition: AdmissionCondition) -> Tool:
    """Expose the unchanged neutral C interface over an evaluator-selected path."""

    if not isinstance(condition, AdmissionCondition):
        raise TypeError("condition must be AdmissionCondition")
    return tool_with(
        _submit_admission_candidate(condition),
        name="submit_restock_candidate",
        parallel=False,
    )
