"""Deterministic vulnerable and governed authority-composition model.

This module preserves the PR2 counterexample and adds the PR3 governed sibling:

* a limited actor is denied the protected accepted-fact append directly;
* the same actor may submit a restock request through locally allowed edges;
* a privileged service deliberately promotes the resulting candidate without
  semantic authority-evidence admission; and
* the vulnerable path records the authoritative effect as accepted history;
* the governed path evaluates exact evidence correlation, proposition support,
  modeled source authority, and source-owned issuance before privileged append.

The model is intentionally local, synchronous, in-memory, and domain-specific.
It contains no Compass production integration, generic policy framework, IAM,
retry, concurrency, persistence, credentials, or external I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Mapping


class Component(str, Enum):
    """Closed identities that participate in the PR2 capability graph."""

    LIMITED_AGENT = "LIMITED_AGENT"
    RESTOCK_WORKFLOW = "RESTOCK_WORKFLOW"
    INVENTORY_AUTHORITY_SERVICE = "INVENTORY_AUTHORITY_SERVICE"
    AUTHORITATIVE_INVENTORY_STORE = "AUTHORITATIVE_INVENTORY_STORE"


class LocalOperation(str, Enum):
    """Closed local operations needed by the direct and composed paths."""

    SUBMIT_RESTOCK_REQUEST = "SUBMIT_RESTOCK_REQUEST"
    SUBMIT_REPLENISHMENT_CANDIDATE = "SUBMIT_REPLENISHMENT_CANDIDATE"
    APPEND_ACCEPTED_REPLENISHMENT = "APPEND_ACCEPTED_REPLENISHMENT"


class PermissionDecision(str, Enum):
    """Local capability decision; it does not express semantic authority."""

    ALLOWED = "ALLOWED"
    DENIED = "DENIED"


class EvidenceKind(str, Enum):
    """Closed PR3 evidence kinds."""

    AGENT_RESTOCK_REQUEST = "AGENT_RESTOCK_REQUEST"
    WAREHOUSE_RECEIPT_CONFIRMATION = "WAREHOUSE_RECEIPT_CONFIRMATION"


class EvidenceIssuer(str, Enum):
    """Closed modeled sources that own evidence issuance."""

    LIMITED_AGENT = "LIMITED_AGENT"
    WAREHOUSE_AUTHORITY = "WAREHOUSE_AUTHORITY"


class Proposition(str, Enum):
    """Closed propositions distinguished by semantic authority admission."""

    RESTOCK_REQUESTED = "RESTOCK_REQUESTED"
    STOCK_REPLENISHED = "STOCK_REPLENISHED"


class AuthorityPromotionDecision(str, Enum):
    """Finite semantic promotion decision for the demo-local PR3 boundary."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class LocalCapability:
    """Identify one caller, local operation, and target capability edge."""

    caller: Component
    operation: LocalOperation
    target: Component


@dataclass(frozen=True)
class CapabilityCheck:
    """Preserve the exact local edge and its deterministic permission decision."""

    capability: LocalCapability
    decision: PermissionDecision


_CAPABILITY_DECISIONS: Mapping[LocalCapability, PermissionDecision] = (
    MappingProxyType(
        {
            LocalCapability(
                caller=Component.LIMITED_AGENT,
                operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
                target=Component.RESTOCK_WORKFLOW,
            ): PermissionDecision.ALLOWED,
            LocalCapability(
                caller=Component.RESTOCK_WORKFLOW,
                operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
                target=Component.INVENTORY_AUTHORITY_SERVICE,
            ): PermissionDecision.ALLOWED,
            LocalCapability(
                caller=Component.INVENTORY_AUTHORITY_SERVICE,
                operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
                target=Component.AUTHORITATIVE_INVENTORY_STORE,
            ): PermissionDecision.ALLOWED,
            LocalCapability(
                caller=Component.LIMITED_AGENT,
                operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
                target=Component.AUTHORITATIVE_INVENTORY_STORE,
            ): PermissionDecision.DENIED,
        }
    )
)

_EVIDENCE_KIND_PROPOSITIONS: Mapping[
    EvidenceKind,
    frozenset[Proposition],
] = MappingProxyType(
    {
        EvidenceKind.AGENT_RESTOCK_REQUEST: frozenset(
            {Proposition.RESTOCK_REQUESTED}
        ),
        EvidenceKind.WAREHOUSE_RECEIPT_CONFIRMATION: frozenset(
            {Proposition.STOCK_REPLENISHED}
        ),
    }
)

_ISSUER_PROPOSITIONS: Mapping[
    EvidenceIssuer,
    frozenset[Proposition],
] = MappingProxyType(
    {
        EvidenceIssuer.LIMITED_AGENT: frozenset(
            {Proposition.RESTOCK_REQUESTED}
        ),
        EvidenceIssuer.WAREHOUSE_AUTHORITY: frozenset(
            {Proposition.STOCK_REPLENISHED}
        ),
    }
)


def decide_local_capability(
    *,
    caller: Component,
    operation: LocalOperation,
    target: Component,
) -> CapabilityCheck:
    """Evaluate one edge in the closed PR2 local capability model.

    Args:
        caller: Component attempting the operation.
        operation: Finite local operation being attempted.
        target: Component that owns the local operation boundary.

    Returns:
        The exact capability edge and its decision. The four experiment edges
        are declared explicitly; any other typed combination fails closed as
        ``DENIED``.

    This decision answers only whether the local call is permitted. It does not
    decide whether a candidate is semantically supported as an authoritative
    inventory fact.
    """

    capability = LocalCapability(
        caller=caller,
        operation=operation,
        target=target,
    )
    return CapabilityCheck(
        capability=capability,
        decision=_CAPABILITY_DECISIONS.get(
            capability,
            PermissionDecision.DENIED,
        ),
    )


def evidence_kind_supports(
    kind: EvidenceKind,
    proposition: Proposition,
) -> bool:
    """Return whether a closed evidence kind supports one proposition."""

    return proposition in _EVIDENCE_KIND_PROPOSITIONS.get(kind, frozenset())


def source_authorized_for(
    issuer: EvidenceIssuer,
    proposition: Proposition,
) -> bool:
    """Return whether a modeled issuer may assert one proposition."""

    return proposition in _ISSUER_PROPOSITIONS.get(issuer, frozenset())


def _validate_product_and_quantity(product_id: object, quantity: object) -> None:
    """Reject empty product identities and non-positive integer quantities."""

    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must be a non-empty string")
    if type(quantity) is not int or quantity <= 0:
        raise ValueError("quantity must be a positive int")


def _validate_identifier(name: str, value: object) -> None:
    """Reject empty deterministic request, candidate, or correlation identities."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _candidate_id_from_request_id(request_id: str) -> str:
    """Derive a predictable candidate identity from one explicit request ID."""

    return f"candidate:{request_id}"


PROTECTED_REQUEST_ID = "restock-request-1"
PROTECTED_CANDIDATE_ID = _candidate_id_from_request_id(PROTECTED_REQUEST_ID)
PROTECTED_PRODUCT_ID = "product-a"
PROTECTED_QUANTITY = 10


@dataclass(frozen=True)
class RestockRequest:
    """Represent a request for replenishment, not a replenishment fact."""

    product_id: str
    quantity: int
    request_id: str = PROTECTED_REQUEST_ID

    def __post_init__(self) -> None:
        _validate_product_and_quantity(self.product_id, self.quantity)
        _validate_identifier("request_id", self.request_id)


@dataclass(frozen=True)
class CandidateStockReplenished:
    """Represent proposed replenishment before accepted-fact promotion."""

    product_id: str
    quantity: int
    candidate_id: str = PROTECTED_CANDIDATE_ID

    def __post_init__(self) -> None:
        _validate_product_and_quantity(self.product_id, self.quantity)
        _validate_identifier("candidate_id", self.candidate_id)


@dataclass(frozen=True)
class WarehouseReceiptObservation:
    """Represent a warehouse-owned factual observation of received stock.

    This deterministic in-memory value is neither a workflow candidate nor an
    accepted inventory fact. It supplies the factual product and quantity basis
    for warehouse receipt confirmation without modeling warehouse persistence,
    external provenance, or cryptographic authenticity.
    """

    receipt_id: str
    product_id: str
    quantity: int

    def __post_init__(self) -> None:
        _validate_identifier("receipt_id", self.receipt_id)
        _validate_product_and_quantity(self.product_id, self.quantity)


@dataclass(frozen=True)
class AcceptedStockReplenished:
    """Represent replenishment admitted into authoritative accepted history."""

    product_id: str
    quantity: int

    def __post_init__(self) -> None:
        _validate_product_and_quantity(self.product_id, self.quantity)


PROTECTED_REPLENISHMENT = AcceptedStockReplenished(
    product_id=PROTECTED_PRODUCT_ID,
    quantity=PROTECTED_QUANTITY,
)


_MODELED_EVIDENCE_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, init=False)
class AuthorityEvidence:
    """Represent evidence created through a modeled source-owned API.

    Calling this constructor directly is unsupported and rejected. The
    supported APIs are ``LimitedAgent.issue_restock_request_evidence`` and
    ``WarehouseAuthority.issue_receipt_confirmation``. The private token is an
    in-process modeling device, not cryptographic or hostile-process security.
    """

    kind: EvidenceKind
    issuer: EvidenceIssuer
    candidate_id: str
    receipt_id: str | None
    product_id: str
    quantity: int
    _issuance_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        kind: EvidenceKind,
        issuer: EvidenceIssuer,
        candidate_id: str,
        product_id: str,
        quantity: int,
        receipt_id: str | None = None,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _MODELED_EVIDENCE_ISSUANCE_TOKEN:
            raise TypeError("evidence must be issued by a modeled source API")
        if not isinstance(kind, EvidenceKind):
            raise TypeError("kind must be EvidenceKind")
        if not isinstance(issuer, EvidenceIssuer):
            raise TypeError("issuer must be EvidenceIssuer")
        _validate_identifier("candidate_id", candidate_id)
        _validate_product_and_quantity(product_id, quantity)
        if receipt_id is not None:
            _validate_identifier("receipt_id", receipt_id)
        if kind is EvidenceKind.WAREHOUSE_RECEIPT_CONFIRMATION:
            if receipt_id is None:
                raise ValueError(
                    "warehouse receipt confirmation requires receipt_id"
                )
        elif receipt_id is not None:
            raise ValueError("agent restock evidence must not claim receipt_id")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "issuer", issuer)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "product_id", product_id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "_issuance_marker", _issuance_token)

    @property
    def issued_by_modeled_source(self) -> bool:
        """Return whether the evidence crossed the modeled issuance boundary."""

        return self._issuance_marker is _MODELED_EVIDENCE_ISSUANCE_TOKEN


def evidence_matches_candidate(
    evidence: AuthorityEvidence,
    candidate: CandidateStockReplenished,
) -> bool:
    """Match evidence to all V1 exact-candidate correlation fields."""

    return (
        evidence.candidate_id == candidate.candidate_id
        and evidence.product_id == candidate.product_id
        and evidence.quantity == candidate.quantity
    )


@dataclass(frozen=True)
class EvidenceEvaluation:
    """Preserve each independent dimension of one evidence evaluation."""

    evidence: AuthorityEvidence
    issued_by_modeled_source: bool
    matches_candidate: bool
    supports_stock_replenished: bool
    issuer_authorized_for_stock_replenished: bool

    @property
    def qualifies_for_promotion(self) -> bool:
        """Return whether all promotion-evidence dimensions hold together."""

        return (
            self.issued_by_modeled_source
            and self.matches_candidate
            and self.supports_stock_replenished
            and self.issuer_authorized_for_stock_replenished
        )


@dataclass(frozen=True)
class SemanticAuthorityAdmissionResult:
    """Record a typed promotion decision and its explicit evidence evaluations."""

    decision: AuthorityPromotionDecision
    evidence_evaluations: tuple[EvidenceEvaluation, ...]
    qualifying_evidence: AuthorityEvidence | None

    def __post_init__(self) -> None:
        qualifying_evaluations = tuple(
            evaluation
            for evaluation in self.evidence_evaluations
            if evaluation.qualifies_for_promotion
        )
        accepted = self.decision is AuthorityPromotionDecision.ACCEPT
        if accepted != bool(qualifying_evaluations):
            raise ValueError(
                "ACCEPT requires qualifying evidence and REJECT forbids it"
            )
        if accepted:
            if self.qualifying_evidence != qualifying_evaluations[0].evidence:
                raise ValueError(
                    "qualifying evidence must be the first sufficient evidence"
                )
        elif self.qualifying_evidence is not None:
            raise ValueError("REJECT must not expose qualifying evidence")


class SemanticAuthorityAdmission:
    """Evaluate evidence for promotion to the fixed STOCK_REPLENISHED claim."""

    def evaluate(
        self,
        *,
        candidate: CandidateStockReplenished,
        evidence: tuple[AuthorityEvidence, ...],
    ) -> SemanticAuthorityAdmissionResult:
        """Evaluate issuance, correlation, support, and source authority.

        Args:
            candidate: Exact replenishment candidate proposed for promotion.
            evidence: Immutable evidence transported by the workflow.

        Returns:
            An immutable decision with a separate evaluation for every evidence
            record. Acceptance requires at least one record satisfying all four
            dimensions for ``STOCK_REPLENISHED``.

        Raises:
            TypeError: If the candidate or any evidence value has the wrong
                bounded demo type.
        """

        if not isinstance(candidate, CandidateStockReplenished):
            raise TypeError("candidate must be CandidateStockReplenished")
        if not isinstance(evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if not all(isinstance(item, AuthorityEvidence) for item in evidence):
            raise TypeError("evidence must contain AuthorityEvidence values")

        evaluations = tuple(
            EvidenceEvaluation(
                evidence=item,
                issued_by_modeled_source=item.issued_by_modeled_source,
                matches_candidate=evidence_matches_candidate(item, candidate),
                supports_stock_replenished=evidence_kind_supports(
                    item.kind,
                    Proposition.STOCK_REPLENISHED,
                ),
                issuer_authorized_for_stock_replenished=source_authorized_for(
                    item.issuer,
                    Proposition.STOCK_REPLENISHED,
                ),
            )
            for item in evidence
        )
        qualifying_evidence = next(
            (
                evaluation.evidence
                for evaluation in evaluations
                if evaluation.qualifies_for_promotion
            ),
            None,
        )
        decision = (
            AuthorityPromotionDecision.ACCEPT
            if qualifying_evidence is not None
            else AuthorityPromotionDecision.REJECT
        )
        return SemanticAuthorityAdmissionResult(
            decision=decision,
            evidence_evaluations=evaluations,
            qualifying_evidence=qualifying_evidence,
        )


@dataclass(frozen=True)
class AppendResult:
    """Record one attempted accepted-fact append and its local outcome."""

    capability_check: CapabilityCheck
    attempted_fact: AcceptedStockReplenished
    appended_fact: AcceptedStockReplenished | None

    def __post_init__(self) -> None:
        if self.capability_check.decision is PermissionDecision.ALLOWED:
            if self.appended_fact != self.attempted_fact:
                raise ValueError(
                    "ALLOWED append must preserve the exact attempted fact"
                )
            return
        if self.appended_fact is not None:
            raise ValueError("DENIED append must not expose an appended fact")

    @property
    def fact_appended(self) -> bool:
        """Return whether the attempted fact entered accepted history."""

        return self.appended_fact is not None


@dataclass(frozen=True)
class CandidatePromotionResult:
    """Record vulnerable candidate submission and its privileged append result."""

    candidate_submission_check: CapabilityCheck
    candidate: CandidateStockReplenished
    append_result: AppendResult | None

    def __post_init__(self) -> None:
        allowed = (
            self.candidate_submission_check.decision
            is PermissionDecision.ALLOWED
        )
        if allowed != (self.append_result is not None):
            raise ValueError(
                "candidate submission decision and append presence must agree"
            )


@dataclass(frozen=True)
class GovernedPromotionResult:
    """Separate local submission from semantic admission and privileged append."""

    candidate_submission_check: CapabilityCheck
    candidate: CandidateStockReplenished
    admission_result: SemanticAuthorityAdmissionResult | None
    append_result: AppendResult | None

    def __post_init__(self) -> None:
        submission_allowed = (
            self.candidate_submission_check.decision
            is PermissionDecision.ALLOWED
        )
        if not submission_allowed:
            if self.admission_result is not None or self.append_result is not None:
                raise ValueError(
                    "denied candidate submission cannot reach admission or append"
                )
            return

        if self.admission_result is None:
            raise ValueError("allowed candidate submission requires admission")
        accepted = (
            self.admission_result.decision is AuthorityPromotionDecision.ACCEPT
        )
        if accepted != (self.append_result is not None):
            raise ValueError(
                "only semantic admission ACCEPT may produce an append result"
            )
        if self.append_result is not None:
            if not self.append_result.fact_appended:
                raise ValueError("accepted promotion must append its fact")
            appended_fact = self.append_result.appended_fact
            if appended_fact is None:
                raise ValueError("accepted promotion must expose its appended fact")
            if (
                appended_fact.product_id != self.candidate.product_id
                or appended_fact.quantity != self.candidate.quantity
            ):
                raise ValueError("governed append must preserve candidate semantics")


@dataclass(frozen=True)
class WorkflowExecutionResult:
    """Record request admission, candidate production, and downstream result."""

    request_submission_check: CapabilityCheck
    candidate: CandidateStockReplenished | None
    promotion_result: CandidatePromotionResult | None

    def __post_init__(self) -> None:
        allowed = (
            self.request_submission_check.decision
            is PermissionDecision.ALLOWED
        )
        complete = self.candidate is not None and self.promotion_result is not None
        if allowed != complete:
            raise ValueError(
                "request decision, candidate, and promotion presence must agree"
            )


@dataclass(frozen=True)
class CandidatePreparationResult:
    """Record an allowed request edge and candidate production before evidence."""

    request_submission_check: CapabilityCheck
    candidate: CandidateStockReplenished | None

    def __post_init__(self) -> None:
        allowed = (
            self.request_submission_check.decision
            is PermissionDecision.ALLOWED
        )
        if allowed != (self.candidate is not None):
            raise ValueError(
                "request submission decision and candidate presence must agree"
            )


class AuthoritativeInventoryStore:
    """Own accepted inventory fact history and derive inventory from that history.

    The store enforces the local append capability and accepts only
    ``AcceptedStockReplenished`` values. It does not store an independently
    mutable inventory integer and it does not evaluate candidate evidence. In
    PR2 the privileged service remains responsible for, and deliberately fails
    at, the semantic promotion boundary.
    """

    component: ClassVar[Component] = Component.AUTHORITATIVE_INVENTORY_STORE

    def __init__(self) -> None:
        self._accepted_facts: list[AcceptedStockReplenished] = []

    def append_accepted_replenishment(
        self,
        *,
        caller: Component,
        fact: AcceptedStockReplenished,
    ) -> AppendResult:
        """Attempt to append one accepted replenishment through the local gate.

        Args:
            caller: Component invoking the store append boundary.
            fact: Accepted replenishment fact being attempted.

        Returns:
            An immutable result preserving the exact capability decision,
            attempted fact, and appended fact when allowed. A denied result does
            not mutate accepted history.

        Raises:
            TypeError: If ``fact`` is a request or candidate rather than an
                ``AcceptedStockReplenished`` value.
        """

        if not isinstance(fact, AcceptedStockReplenished):
            raise TypeError("fact must be AcceptedStockReplenished")

        capability_check = decide_local_capability(
            caller=caller,
            operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
            target=self.component,
        )
        if capability_check.decision is PermissionDecision.DENIED:
            return AppendResult(
                capability_check=capability_check,
                attempted_fact=fact,
                appended_fact=None,
            )

        self._accepted_facts.append(fact)
        return AppendResult(
            capability_check=capability_check,
            attempted_fact=fact,
            appended_fact=fact,
        )

    @property
    def accepted_facts(self) -> tuple[AcceptedStockReplenished, ...]:
        """Return an immutable view of authoritative accepted fact history."""

        return tuple(self._accepted_facts)

    def inventory(self, product_id: str) -> int:
        """Fold accepted replenishment history into inventory for one product."""

        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")
        return sum(
            fact.quantity
            for fact in self._accepted_facts
            if fact.product_id == product_id
        )


class InventoryAuthorityService:
    """Own privileged mutation for separate vulnerable and governed paths.

    The PR2 method promotes a permitted-workflow candidate without semantic
    admission. The PR3 sibling invokes demo-local semantic authority admission
    before exercising the same legitimate store append capability.
    """

    component: ClassVar[Component] = Component.INVENTORY_AUTHORITY_SERVICE

    def __init__(
        self,
        store: AuthoritativeInventoryStore,
        semantic_authority_admission: SemanticAuthorityAdmission | None = None,
    ) -> None:
        self._store = store
        self._semantic_authority_admission = (
            semantic_authority_admission
            if semantic_authority_admission is not None
            else SemanticAuthorityAdmission()
        )

    def promote_candidate_without_semantic_authority_admission(
        self,
        *,
        caller: Component,
        candidate: CandidateStockReplenished,
    ) -> CandidatePromotionResult:
        """Promote a permitted-workflow candidate without PR3 authority evidence.

        Args:
            caller: Component submitting the replenishment candidate.
            candidate: Candidate produced by the restock workflow.

        Returns:
            The candidate-submission capability decision and, when locally
            allowed, the privileged accepted-fact append result.

        The missing semantic authority-admission step is the deliberate PR2
        vulnerability. Local capability checks remain active both here and at
        the store; this method does not bypass a denied permission.
        """

        if not isinstance(candidate, CandidateStockReplenished):
            raise TypeError("candidate must be CandidateStockReplenished")

        submission_check = decide_local_capability(
            caller=caller,
            operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
            target=self.component,
        )
        if submission_check.decision is PermissionDecision.DENIED:
            return CandidatePromotionResult(
                candidate_submission_check=submission_check,
                candidate=candidate,
                append_result=None,
            )

        # Deliberate PR2 failure: permitted workflow arrival is treated as
        # sufficient for promotion from candidate to accepted inventory fact.
        accepted_fact = AcceptedStockReplenished(
            product_id=candidate.product_id,
            quantity=candidate.quantity,
        )
        append_result = self._store.append_accepted_replenishment(
            caller=self.component,
            fact=accepted_fact,
        )
        return CandidatePromotionResult(
            candidate_submission_check=submission_check,
            candidate=candidate,
            append_result=append_result,
        )

    def promote_candidate_with_semantic_authority_admission(
        self,
        *,
        caller: Component,
        candidate: CandidateStockReplenished,
        evidence: tuple[AuthorityEvidence, ...],
    ) -> GovernedPromotionResult:
        """Evaluate semantic authority before exercising privileged append.

        Args:
            caller: Component submitting the candidate through the local edge.
            candidate: Exact candidate proposed for accepted-fact promotion.
            evidence: Evidence preserved and transported by the workflow.

        Returns:
            A governed result that keeps the local candidate-submission check,
            semantic decision, and optional privileged append distinct. An
            allowed local submission may therefore end in semantic rejection
            with no append attempt.
        """

        if not isinstance(candidate, CandidateStockReplenished):
            raise TypeError("candidate must be CandidateStockReplenished")

        submission_check = decide_local_capability(
            caller=caller,
            operation=LocalOperation.SUBMIT_REPLENISHMENT_CANDIDATE,
            target=self.component,
        )
        if submission_check.decision is PermissionDecision.DENIED:
            return GovernedPromotionResult(
                candidate_submission_check=submission_check,
                candidate=candidate,
                admission_result=None,
                append_result=None,
            )

        admission_result = self._semantic_authority_admission.evaluate(
            candidate=candidate,
            evidence=evidence,
        )
        if admission_result.decision is AuthorityPromotionDecision.REJECT:
            return GovernedPromotionResult(
                candidate_submission_check=submission_check,
                candidate=candidate,
                admission_result=admission_result,
                append_result=None,
            )

        accepted_fact = AcceptedStockReplenished(
            product_id=candidate.product_id,
            quantity=candidate.quantity,
        )
        append_result = self._store.append_accepted_replenishment(
            caller=self.component,
            fact=accepted_fact,
        )
        return GovernedPromotionResult(
            candidate_submission_check=submission_check,
            candidate=candidate,
            admission_result=admission_result,
            append_result=append_result,
        )


class RestockWorkflow:
    """Convert an allowed restock request into a candidate for the service."""

    component: ClassVar[Component] = Component.RESTOCK_WORKFLOW

    def __init__(self, authority_service: InventoryAuthorityService) -> None:
        self._authority_service = authority_service

    def handle_restock_request(
        self,
        *,
        caller: Component,
        request: RestockRequest,
    ) -> WorkflowExecutionResult:
        """Check request permission, create a candidate, and submit it downstream.

        Args:
            caller: Component submitting the restock request.
            request: Requested product and quantity.

        Returns:
            The exact request-edge decision and, when allowed, the distinct
            candidate plus the authority service's vulnerable promotion result.
        """

        if not isinstance(request, RestockRequest):
            raise TypeError("request must be RestockRequest")

        request_check = decide_local_capability(
            caller=caller,
            operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
            target=self.component,
        )
        if request_check.decision is PermissionDecision.DENIED:
            return WorkflowExecutionResult(
                request_submission_check=request_check,
                candidate=None,
                promotion_result=None,
            )

        candidate = CandidateStockReplenished(
            product_id=request.product_id,
            quantity=request.quantity,
            candidate_id=_candidate_id_from_request_id(request.request_id),
        )
        promotion_result = (
            self._authority_service
            .promote_candidate_without_semantic_authority_admission(
                caller=self.component,
                candidate=candidate,
            )
        )
        return WorkflowExecutionResult(
            request_submission_check=request_check,
            candidate=candidate,
            promotion_result=promotion_result,
        )

    def prepare_restock_candidate(
        self,
        *,
        caller: Component,
        request: RestockRequest,
    ) -> CandidatePreparationResult:
        """Produce the same deterministic candidate without promoting it yet.

        This two-step governed path permits source-owned evidence to be issued
        for the exact candidate before the workflow transports that evidence to
        semantic authority admission.
        """

        if not isinstance(request, RestockRequest):
            raise TypeError("request must be RestockRequest")

        request_check = decide_local_capability(
            caller=caller,
            operation=LocalOperation.SUBMIT_RESTOCK_REQUEST,
            target=self.component,
        )
        if request_check.decision is PermissionDecision.DENIED:
            return CandidatePreparationResult(
                request_submission_check=request_check,
                candidate=None,
            )

        return CandidatePreparationResult(
            request_submission_check=request_check,
            candidate=CandidateStockReplenished(
                product_id=request.product_id,
                quantity=request.quantity,
                candidate_id=_candidate_id_from_request_id(request.request_id),
            ),
        )

    def submit_candidate_with_semantic_authority_admission(
        self,
        *,
        candidate: CandidateStockReplenished,
        evidence: tuple[AuthorityEvidence, ...],
    ) -> GovernedPromotionResult:
        """Transport candidate and evidence unchanged to the governed service."""

        return (
            self._authority_service
            .promote_candidate_with_semantic_authority_admission(
                caller=self.component,
                candidate=candidate,
                evidence=evidence,
            )
        )


class LimitedAgent:
    """Exercise the limited actor's direct and allowed-workflow paths."""

    component: ClassVar[Component] = Component.LIMITED_AGENT

    def attempt_direct_append(
        self,
        *,
        store: AuthoritativeInventoryStore,
        fact: AcceptedStockReplenished,
    ) -> AppendResult:
        """Attempt the protected accepted-fact append as the limited actor."""

        return store.append_accepted_replenishment(
            caller=self.component,
            fact=fact,
        )

    def submit_restock_request(
        self,
        *,
        workflow: RestockWorkflow,
        request: RestockRequest,
    ) -> WorkflowExecutionResult:
        """Submit one request through the actor's allowed workflow edge."""

        return workflow.handle_restock_request(
            caller=self.component,
            request=request,
        )

    def prepare_restock_candidate(
        self,
        *,
        workflow: RestockWorkflow,
        request: RestockRequest,
    ) -> CandidatePreparationResult:
        """Use the allowed request edge without selecting a promotion outcome."""

        return workflow.prepare_restock_candidate(
            caller=self.component,
            request=request,
        )

    def issue_restock_request_evidence(
        self,
        *,
        candidate: CandidateStockReplenished,
    ) -> AuthorityEvidence:
        """Issue exactly correlated evidence for RESTOCK_REQUESTED only."""

        if not isinstance(candidate, CandidateStockReplenished):
            raise TypeError("candidate must be CandidateStockReplenished")
        return AuthorityEvidence(
            kind=EvidenceKind.AGENT_RESTOCK_REQUEST,
            issuer=EvidenceIssuer.LIMITED_AGENT,
            candidate_id=candidate.candidate_id,
            product_id=candidate.product_id,
            quantity=candidate.quantity,
            _issuance_token=_MODELED_EVIDENCE_ISSUANCE_TOKEN,
        )


class WarehouseAuthority:
    """Own modeled issuance of warehouse receipt confirmation evidence."""

    issuer: ClassVar[EvidenceIssuer] = EvidenceIssuer.WAREHOUSE_AUTHORITY

    def issue_receipt_confirmation(
        self,
        *,
        candidate: CandidateStockReplenished,
        receipt: WarehouseReceiptObservation,
    ) -> AuthorityEvidence:
        """Issue confirmation using candidate identity and receipt-owned facts.

        The warehouse may be called because of an originating request, but the
        evidence product and quantity derive from the separate receipt
        observation rather than from the candidate, request, or workflow
        completion. Only candidate correlation identity comes from the proposed
        candidate.
        """

        if not isinstance(candidate, CandidateStockReplenished):
            raise TypeError("candidate must be CandidateStockReplenished")
        if not isinstance(receipt, WarehouseReceiptObservation):
            raise TypeError("receipt must be WarehouseReceiptObservation")
        return AuthorityEvidence(
            kind=EvidenceKind.WAREHOUSE_RECEIPT_CONFIRMATION,
            issuer=self.issuer,
            candidate_id=candidate.candidate_id,
            receipt_id=receipt.receipt_id,
            product_id=receipt.product_id,
            quantity=receipt.quantity,
            _issuance_token=_MODELED_EVIDENCE_ISSUANCE_TOKEN,
        )


__all__ = (
    "AcceptedStockReplenished",
    "AppendResult",
    "AuthorityEvidence",
    "AuthorityPromotionDecision",
    "AuthoritativeInventoryStore",
    "CandidatePreparationResult",
    "CandidatePromotionResult",
    "CandidateStockReplenished",
    "CapabilityCheck",
    "Component",
    "EvidenceEvaluation",
    "EvidenceIssuer",
    "EvidenceKind",
    "GovernedPromotionResult",
    "InventoryAuthorityService",
    "LimitedAgent",
    "LocalCapability",
    "LocalOperation",
    "PROTECTED_CANDIDATE_ID",
    "PROTECTED_PRODUCT_ID",
    "PROTECTED_QUANTITY",
    "PROTECTED_REPLENISHMENT",
    "PROTECTED_REQUEST_ID",
    "PermissionDecision",
    "Proposition",
    "RestockRequest",
    "RestockWorkflow",
    "SemanticAuthorityAdmission",
    "SemanticAuthorityAdmissionResult",
    "WarehouseAuthority",
    "WarehouseReceiptObservation",
    "WorkflowExecutionResult",
    "decide_local_capability",
    "evidence_kind_supports",
    "evidence_matches_candidate",
    "source_authorized_for",
)
