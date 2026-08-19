"""Reviewer-facing deterministic runner for the authority-composition demo."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    AcceptedStockReplenished,
    AppendResult,
    AuthorityEvidence,
    AuthoritativeInventoryStore,
    CandidatePreparationResult,
    GovernedPromotionResult,
    InventoryAuthorityService,
    LimitedAgent,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    PROTECTED_REPLENISHMENT,
    PROTECTED_REQUEST_ID,
    Proposition,
    RestockRequest,
    RestockWorkflow,
    WarehouseAuthority,
    WarehouseReceiptObservation,
    WorkflowExecutionResult,
    evidence_kind_supports,
    evidence_matches_candidate,
    source_authorized_for,
)


@dataclass(frozen=True)
class FinalAuthorityState:
    """Capture accepted history and its derived inventory after one scenario."""

    accepted_facts: tuple[AcceptedStockReplenished, ...]
    inventory: int

    @property
    def authoritative_effect_reached(self) -> bool:
        """Return whether the protected fact entered authoritative history."""

        return PROTECTED_REPLENISHMENT in self.accepted_facts


@dataclass(frozen=True)
class DirectScenarioResult:
    """Structured observations for Case 1 direct denial."""

    direct_attempt: AppendResult
    final_state: FinalAuthorityState


@dataclass(frozen=True)
class VulnerableScenarioResult:
    """Structured observations for Case 2 unchecked promotion."""

    direct_attempt: AppendResult
    workflow_execution: WorkflowExecutionResult
    final_state: FinalAuthorityState


@dataclass(frozen=True)
class GovernedScenarioResult:
    """Structured observations for Case 3 agent-evidence rejection."""

    direct_attempt: AppendResult
    candidate_preparation: CandidatePreparationResult
    agent_evidence: AuthorityEvidence
    promotion: GovernedPromotionResult
    final_state: FinalAuthorityState


@dataclass(frozen=True)
class PositiveControlResult:
    """Structured observations for warehouse-grounded acceptance."""

    direct_attempt: AppendResult
    candidate_preparation: CandidatePreparationResult
    agent_evidence: AuthorityEvidence
    warehouse_receipt: WarehouseReceiptObservation
    warehouse_evidence: AuthorityEvidence
    promotion: GovernedPromotionResult
    final_state: FinalAuthorityState


@dataclass(frozen=True)
class DemoRun:
    """Contain all four scenarios produced from independent fresh stores."""

    direct: DirectScenarioResult
    vulnerable: VulnerableScenarioResult
    governed: GovernedScenarioResult
    positive_control: PositiveControlResult


def _build_components() -> tuple[
    LimitedAgent,
    RestockWorkflow,
    WarehouseAuthority,
    AuthoritativeInventoryStore,
]:
    store = AuthoritativeInventoryStore()
    authority_service = InventoryAuthorityService(store)
    workflow = RestockWorkflow(authority_service)
    return LimitedAgent(), workflow, WarehouseAuthority(), store


def _protected_request() -> RestockRequest:
    return RestockRequest(
        request_id=PROTECTED_REQUEST_ID,
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )


def _direct_attempt(
    agent: LimitedAgent,
    store: AuthoritativeInventoryStore,
) -> AppendResult:
    return agent.attempt_direct_append(
        store=store,
        fact=PROTECTED_REPLENISHMENT,
    )


def _final_state(store: AuthoritativeInventoryStore) -> FinalAuthorityState:
    return FinalAuthorityState(
        accepted_facts=store.accepted_facts,
        inventory=store.inventory(PROTECTED_PRODUCT_ID),
    )


def run_direct_scenario() -> DirectScenarioResult:
    """Run Case 1 against a fresh authoritative store."""

    agent, _, _, store = _build_components()
    return DirectScenarioResult(
        direct_attempt=_direct_attempt(agent, store),
        final_state=_final_state(store),
    )


def run_vulnerable_scenario() -> VulnerableScenarioResult:
    """Run Case 2 through the deliberately vulnerable PR2 method."""

    agent, workflow, _, store = _build_components()
    direct_attempt = _direct_attempt(agent, store)
    workflow_execution = agent.submit_restock_request(
        workflow=workflow,
        request=_protected_request(),
    )
    if workflow_execution.candidate is None:
        raise RuntimeError("vulnerable scenario did not produce a candidate")
    if workflow_execution.promotion_result is None:
        raise RuntimeError("vulnerable scenario did not reach promotion")
    if workflow_execution.promotion_result.append_result is None:
        raise RuntimeError("vulnerable scenario did not attempt privileged append")
    return VulnerableScenarioResult(
        direct_attempt=direct_attempt,
        workflow_execution=workflow_execution,
        final_state=_final_state(store),
    )


def run_governed_scenario() -> GovernedScenarioResult:
    """Run Case 3 with exactly correlated agent-request evidence only."""

    agent, workflow, _, store = _build_components()
    direct_attempt = _direct_attempt(agent, store)
    preparation = agent.prepare_restock_candidate(
        workflow=workflow,
        request=_protected_request(),
    )
    candidate = preparation.candidate
    if candidate is None:
        raise RuntimeError("governed scenario did not produce a candidate")
    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)
    promotion = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(agent_evidence,),
    )
    if promotion.admission_result is None:
        raise RuntimeError("governed scenario did not reach semantic admission")
    return GovernedScenarioResult(
        direct_attempt=direct_attempt,
        candidate_preparation=preparation,
        agent_evidence=agent_evidence,
        promotion=promotion,
        final_state=_final_state(store),
    )


def run_positive_control() -> PositiveControlResult:
    """Run the governed path with matching warehouse-owned receipt facts."""

    agent, workflow, warehouse, store = _build_components()
    direct_attempt = _direct_attempt(agent, store)
    preparation = agent.prepare_restock_candidate(
        workflow=workflow,
        request=_protected_request(),
    )
    candidate = preparation.candidate
    if candidate is None:
        raise RuntimeError("positive control did not produce a candidate")
    agent_evidence = agent.issue_restock_request_evidence(candidate=candidate)
    warehouse_receipt = WarehouseReceiptObservation(
        receipt_id="warehouse-receipt-1",
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
    )
    warehouse_evidence = warehouse.issue_receipt_confirmation(
        candidate=candidate,
        receipt=warehouse_receipt,
    )
    promotion = workflow.submit_candidate_with_semantic_authority_admission(
        candidate=candidate,
        evidence=(agent_evidence, warehouse_evidence),
    )
    if promotion.admission_result is None:
        raise RuntimeError("positive control did not reach semantic admission")
    return PositiveControlResult(
        direct_attempt=direct_attempt,
        candidate_preparation=preparation,
        agent_evidence=agent_evidence,
        warehouse_receipt=warehouse_receipt,
        warehouse_evidence=warehouse_evidence,
        promotion=promotion,
        final_state=_final_state(store),
    )


def run_demo() -> DemoRun:
    """Run the complete deterministic progression from four fresh stores."""

    return DemoRun(
        direct=run_direct_scenario(),
        vulnerable=run_vulnerable_scenario(),
        governed=run_governed_scenario(),
        positive_control=run_positive_control(),
    )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def render_demo(run: DemoRun) -> str:
    """Render all structured scenario observations for a terminal reviewer."""

    vulnerable_execution = run.vulnerable.workflow_execution
    vulnerable_promotion = vulnerable_execution.promotion_result
    if vulnerable_promotion is None or vulnerable_promotion.append_result is None:
        raise RuntimeError("vulnerable result is incomplete")

    governed_candidate = run.governed.candidate_preparation.candidate
    governed_admission = run.governed.promotion.admission_result
    if governed_candidate is None or governed_admission is None:
        raise RuntimeError("governed result is incomplete")

    positive_candidate = run.positive_control.candidate_preparation.candidate
    positive_admission = run.positive_control.promotion.admission_result
    if positive_candidate is None or positive_admission is None:
        raise RuntimeError("positive-control result is incomplete")

    warehouse_evidence = run.positive_control.warehouse_evidence
    receipt = run.positive_control.warehouse_receipt
    governed_request_check = (
        run.governed.candidate_preparation.request_submission_check
    )
    positive_request_check = (
        run.positive_control.candidate_preparation.request_submission_check
    )
    governed_candidate_submission_check = (
        run.governed.promotion.candidate_submission_check
    )
    positive_candidate_submission_check = (
        run.positive_control.promotion.candidate_submission_check
    )
    agent_evidence = run.governed.agent_evidence
    agent_exact_match = evidence_matches_candidate(
        agent_evidence,
        governed_candidate,
    )
    agent_supports_request = evidence_kind_supports(
        agent_evidence.kind,
        Proposition.RESTOCK_REQUESTED,
    )
    agent_supports_replenishment = evidence_kind_supports(
        agent_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    )
    same_limited_origin = (
        positive_request_check.capability.caller
        == governed_request_check.capability.caller
    )
    same_workflow_path = (
        positive_request_check.capability
        == governed_request_check.capability
        and positive_candidate_submission_check.capability
        == governed_candidate_submission_check.capability
    )
    warehouse_exact_match = evidence_matches_candidate(
        warehouse_evidence,
        positive_candidate,
    )
    warehouse_supports_replenishment = evidence_kind_supports(
        warehouse_evidence.kind,
        Proposition.STOCK_REPLENISHED,
    )
    warehouse_issuer_authorized = source_authorized_for(
        warehouse_evidence.issuer,
        Proposition.STOCK_REPLENISHED,
    )

    lines = [
        "INDIRECT AUTHORITY ESCALATION — DETERMINISTIC LOCAL MODEL",
        "",
        "CASE 1 — DIRECT DENIAL",
        (
            "direct append permission: "
            f"{run.direct.direct_attempt.capability_check.decision.value}"
        ),
        (
            "authoritative effect reached: "
            f"{_yes_no(run.direct.final_state.authoritative_effect_reached)}"
        ),
        f"accepted facts: {len(run.direct.final_state.accepted_facts)}",
        f"inventory: {run.direct.final_state.inventory}",
        "RESULT: LOCAL PERMISSION BOUNDARY PRESERVED",
        "",
        "CASE 2 — LOCALLY VALID AUTHORITY COMPOSITION",
        (
            "direct append permission: "
            f"{run.vulnerable.direct_attempt.capability_check.decision.value}"
        ),
        (
            "LIMITED_AGENT -> RESTOCK_WORKFLOW: "
            f"{vulnerable_execution.request_submission_check.decision.value}"
        ),
        (
            "RESTOCK_WORKFLOW -> INVENTORY_AUTHORITY_SERVICE: "
            f"{vulnerable_promotion.candidate_submission_check.decision.value}"
        ),
        (
            "INVENTORY_AUTHORITY_SERVICE -> STORE: "
            f"{vulnerable_promotion.append_result.capability_check.decision.value}"
        ),
        f"candidate produced: {_yes_no(vulnerable_execution.candidate is not None)}",
        "semantic authority admission: NOT PRESENT",
        (
            "authoritative effect reached: "
            f"{_yes_no(run.vulnerable.final_state.authoritative_effect_reached)}"
        ),
        f"accepted facts: {len(run.vulnerable.final_state.accepted_facts)}",
        f"inventory: {run.vulnerable.final_state.inventory}",
        "RESULT: PROMOTION INVARIANT VIOLATED",
        "",
        "CASE 3 — GOVERNED SEMANTIC PROMOTION",
        (
            "direct append permission: "
            f"{run.governed.direct_attempt.capability_check.decision.value}"
        ),
        (
            "workflow request: "
            f"{governed_request_check.decision.value}"
        ),
        (
            "candidate submission: "
            f"{run.governed.promotion.candidate_submission_check.decision.value}"
        ),
        f"candidate produced: {_yes_no(governed_candidate is not None)}",
        (
            "agent evidence issued correctly: "
            f"{_yes_no(run.governed.agent_evidence.issued_by_modeled_source)}"
        ),
        (
            "agent evidence exact match: "
            f"{_yes_no(agent_exact_match)}"
        ),
        (
            "agent evidence supports RESTOCK_REQUESTED: "
            f"{_yes_no(agent_supports_request)}"
        ),
        (
            "agent evidence supports STOCK_REPLENISHED: "
            f"{_yes_no(agent_supports_replenishment)}"
        ),
        f"promotion: {governed_admission.decision.value}",
        (
            "authoritative effect reached: "
            f"{_yes_no(run.governed.final_state.authoritative_effect_reached)}"
        ),
        f"accepted facts: {len(run.governed.final_state.accepted_facts)}",
        f"inventory: {run.governed.final_state.inventory}",
        "RESULT: PROMOTION INVARIANT PRESERVED",
        "",
        "POSITIVE CONTROL — MATCHING WAREHOUSE AUTHORITY BASIS",
        f"same limited-agent origin: {_yes_no(same_limited_origin)}",
        f"same workflow path: {_yes_no(same_workflow_path)}",
        f"same candidate: {_yes_no(positive_candidate == governed_candidate)}",
        (
            "warehouse receipt observation: "
            f"{receipt.receipt_id} / {receipt.product_id} / {receipt.quantity}"
        ),
        (
            "warehouse evidence exact match: "
            f"{_yes_no(warehouse_exact_match)}"
        ),
        (
            "warehouse evidence supports STOCK_REPLENISHED: "
            f"{_yes_no(warehouse_supports_replenishment)}"
        ),
        (
            "warehouse issuer authorized: "
            f"{_yes_no(warehouse_issuer_authorized)}"
        ),
        f"promotion: {positive_admission.decision.value}",
        (
            "authoritative effect reached: "
            f"{_yes_no(run.positive_control.final_state.authoritative_effect_reached)}"
        ),
        f"accepted facts: {len(run.positive_control.final_state.accepted_facts)}",
        f"inventory: {run.positive_control.final_state.inventory}",
        "RESULT: EVIDENCE-BASED GOVERNANCE CONFIRMED",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run and print the deterministic reviewer-facing comparison."""

    print(render_demo(run_demo()), end="")


if __name__ == "__main__":
    main()
