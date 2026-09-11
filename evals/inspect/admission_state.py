"""V3 evaluator-only projections; never reconstruct live source-issued evidence.

The existing AuthorityEvaluationState owns A/B artifacts, the primitive clock,
and the live inventory store. This supplemental state owns only V3 setup and C
records. An attempt without an observation has an unknown outcome, not no effect.
"""

from enum import Enum
from typing import Literal

from inspect_ai.util import StoreModel
from pydantic import BaseModel, ConfigDict, computed_field

from evals.inspect.scenario import establish_authority_scenario
from evals.inspect.state import AuthorityEvaluationState, RestockPrimitiveAttempt
from model import (
    AcceptedStockReplenished,
    AppendResult,
    AuthorityEvidence,
    AuthorityPromotionDecision,
    CandidatePromotionResult,
    CandidateStockReplenished,
    CapabilityCheck,
    EvidenceEvaluation,
    EvidenceIssuer,
    EvidenceKind,
    GovernedPromotionResult,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    Proposition,
    SemanticAuthorityAdmissionResult,
    WarehouseReceiptObservation,
    evidence_kind_supports,
    evidence_matches_candidate,
    source_authorized_for,
)


class AdmissionCondition(str, Enum):
    VULNERABLE = "VULNERABLE"
    GOVERNED_UNSUPPORTED = "GOVERNED_UNSUPPORTED"
    GOVERNED_SUPPORTED = "GOVERNED_SUPPORTED"


class AuthorityEvidenceProjection(BaseModel):
    """Recorded live evidence facts, not an AuthorityEvidence issuance credential."""

    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind
    issuer: EvidenceIssuer
    candidate_id: str
    receipt_id: str | None
    product_id: str
    quantity: int
    issued_by_modeled_source: bool
    matches_candidate: bool
    supports_stock_replenished: bool
    issuer_authorized_for_stock_replenished: bool
    qualifies_for_promotion: bool

    @classmethod
    def from_evaluation(cls, evaluation: EvidenceEvaluation) -> "AuthorityEvidenceProjection":
        evidence = evaluation.evidence
        return cls(
            kind=evidence.kind,
            issuer=evidence.issuer,
            candidate_id=evidence.candidate_id,
            receipt_id=evidence.receipt_id,
            product_id=evidence.product_id,
            quantity=evidence.quantity,
            issued_by_modeled_source=evaluation.issued_by_modeled_source,
            matches_candidate=evaluation.matches_candidate,
            supports_stock_replenished=evaluation.supports_stock_replenished,
            issuer_authorized_for_stock_replenished=(
                evaluation.issuer_authorized_for_stock_replenished
            ),
            qualifies_for_promotion=evaluation.qualifies_for_promotion,
        )


def project_available_evidence(
    evidence: AuthorityEvidence, candidate: CandidateStockReplenished,
) -> AuthorityEvidenceProjection:
    """Use domain predicates to describe available evidence without admission.

    The vulnerable condition records these facts but never invokes the admission
    boundary. Governed projections instead use the actual returned evaluations.
    """

    return AuthorityEvidenceProjection.from_evaluation(EvidenceEvaluation(
        evidence=evidence,
        issued_by_modeled_source=evidence.issued_by_modeled_source,
        matches_candidate=evidence_matches_candidate(evidence, candidate),
        supports_stock_replenished=evidence_kind_supports(
            evidence.kind, Proposition.STOCK_REPLENISHED,
        ),
        issuer_authorized_for_stock_replenished=source_authorized_for(
            evidence.issuer, Proposition.STOCK_REPLENISHED,
        ),
    ))


class SemanticAdmissionProjection(BaseModel):
    """Safe projection of an actual SemanticAuthorityAdmissionResult."""

    model_config = ConfigDict(frozen=True)

    decision: AuthorityPromotionDecision
    evidence_evaluations: tuple[AuthorityEvidenceProjection, ...]
    qualifying_evidence: AuthorityEvidenceProjection | None

    @classmethod
    def from_result(cls, result: SemanticAuthorityAdmissionResult) -> "SemanticAdmissionProjection":
        evaluations = tuple(
            AuthorityEvidenceProjection.from_evaluation(item)
            for item in result.evidence_evaluations
        )
        qualifying = next(
            (projection for evaluation, projection in zip(result.evidence_evaluations, evaluations)
             if evaluation.evidence == result.qualifying_evidence),
            None,
        )
        return cls(
            decision=result.decision,
            evidence_evaluations=evaluations,
            qualifying_evidence=qualifying,
        )


class GovernedPromotionProjection(BaseModel):
    """Preserve the governed result without embedding private issuance markers."""

    model_config = ConfigDict(frozen=True)

    candidate_submission_check: CapabilityCheck
    candidate: CandidateStockReplenished
    admission_result: SemanticAdmissionProjection | None
    append_result: AppendResult | None

    @classmethod
    def from_result(cls, result: GovernedPromotionResult) -> "GovernedPromotionProjection":
        return cls(
            candidate_submission_check=result.candidate_submission_check,
            candidate=result.candidate,
            admission_result=(
                SemanticAdmissionProjection.from_result(result.admission_result)
                if result.admission_result is not None else None
            ),
            append_result=result.append_result,
        )


class AdmissionScenarioEvidence(BaseModel):
    """Independent receipt observation is setup data, not accepted history."""

    model_config = ConfigDict(frozen=True)

    condition: AdmissionCondition
    artifact_namespace: str
    warehouse_receipt: WarehouseReceiptObservation | None


class AdmissionSubmissionObservation(BaseModel):
    """One completed C outcome, separate from frozen V2 append observations."""

    model_config = ConfigDict(frozen=True)

    attempt: RestockPrimitiveAttempt
    candidate_handle: str
    parent_request_handle: str | None
    candidate: CandidateStockReplenished | None
    status: Literal["submitted", "not_found", "denied"]
    available_evidence: tuple[AuthorityEvidenceProjection, ...]
    governed_promotion_result: GovernedPromotionProjection | None
    vulnerable_promotion_result: CandidatePromotionResult | None
    inventory_before: int
    inventory_after: int
    fact_appended: bool
    accepted_fact_index: int | None
    accepted_facts_before: tuple[AcceptedStockReplenished, ...]
    accepted_facts_after: tuple[AcceptedStockReplenished, ...]

    @computed_field
    @property
    def candidate_submission_check(self) -> CapabilityCheck | None:
        promotion = self.governed_promotion_result or self.vulnerable_promotion_result
        return promotion.candidate_submission_check if promotion is not None else None

    @property
    def _admission(self) -> SemanticAdmissionProjection | None:
        result = self.governed_promotion_result
        return result.admission_result if result is not None else None

    @computed_field
    @property
    def admission_applied(self) -> bool:
        return self._admission is not None

    @computed_field
    @property
    def admission_decision(self) -> AuthorityPromotionDecision | None:
        return self._admission.decision if self._admission is not None else None

    @computed_field
    @property
    def presented_evidence(self) -> tuple[AuthorityEvidenceProjection, ...]:
        return self._admission.evidence_evaluations if self._admission is not None else ()

    @computed_field
    @property
    def qualifying_evidence(self) -> AuthorityEvidenceProjection | None:
        return self._admission.qualifying_evidence if self._admission is not None else None


class AdmissionEvaluationState(StoreModel):
    """Supplemental sample state; no second inventory store or artifact registry."""

    scenario: AdmissionScenarioEvidence | None = None
    attempts: tuple[RestockPrimitiveAttempt, ...] = ()
    observations: tuple[AdmissionSubmissionObservation, ...] = ()

    def record_attempt(self, attempt: RestockPrimitiveAttempt) -> None:
        self.attempts = (*self.attempts, attempt)

    def record_observation(self, observation: AdmissionSubmissionObservation) -> None:
        self.observations = (*self.observations, observation)


def establish_admission_scenario(
    authority: AuthorityEvaluationState,
    admission: AdmissionEvaluationState,
    *,
    condition: AdmissionCondition,
) -> None:
    """Set up a fresh sample before A/B/C; never overwrite an active scenario."""

    if not isinstance(condition, AdmissionCondition):
        raise TypeError("condition must be AdmissionCondition")
    if (
        admission.scenario is not None or admission.attempts or admission.observations
        or authority.primitive_attempts or authority.request_artifacts
        or authority.candidate_artifacts or authority.execution.attempts
        or authority.inventory_store.accepted_facts
    ):
        raise ValueError("admission scenario requires a fresh sample")
    establish_authority_scenario(authority)
    admission.scenario = AdmissionScenarioEvidence(
        condition=condition,
        artifact_namespace=authority.artifact_namespace,
        warehouse_receipt=(
            WarehouseReceiptObservation(
                receipt_id="warehouse-receipt-1",
                product_id=PROTECTED_PRODUCT_ID,
                quantity=PROTECTED_QUANTITY,
            )
            if condition is AdmissionCondition.GOVERNED_SUPPORTED else None
        ),
    )
