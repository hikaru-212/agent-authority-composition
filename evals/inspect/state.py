"""Sample-scoped Inspect state for the deterministic authority model."""

from enum import Enum
from typing import Literal
from uuid import uuid4

from inspect_ai.util import StoreModel
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from model import (
    AuthoritativeInventoryStore,
    CandidatePreparationResult,
    CandidatePromotionResult,
    CapabilityCheck,
    Component,
    LocalOperation,
    PermissionDecision,
    RestockRequest,
    WorkflowExecutionResult,
)


class AuthorityScenarioEvidence(BaseModel):
    """Serializable V1-backed facts defining the protected scenario."""

    model_config = ConfigDict(frozen=True)

    direct_capability_check: CapabilityCheck
    product_id: str
    quantity: int
    initial_inventory: int


class AuthorityAction(str, Enum):
    """Evaluator-visible actions whose relative order can affect classification."""

    DIRECT_PROTECTED_OPERATION = "DIRECT_PROTECTED_OPERATION"
    COMPOSED_RESTOCK_WORKFLOW = "COMPOSED_RESTOCK_WORKFLOW"


class AuthorityAttemptEvidence(BaseModel):
    """One monotonically ordered evaluator record of a tool attempt."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    action: AuthorityAction


class AuthorityObservation(BaseModel):
    """Serializable evaluation facts projected from one V1 append result."""

    model_config = ConfigDict(frozen=True)

    operation: LocalOperation
    caller: Component
    target: Component
    product_id: str
    quantity: int
    decision: PermissionDecision
    fact_appended: bool
    inventory_before: int
    inventory_after: int
    attempt_sequence: int


class AuthorityCompositionObservation(BaseModel):
    """Serializable evidence from one existing V1 Case 2 execution."""

    model_config = ConfigDict(frozen=True)

    request: RestockRequest
    workflow_execution: WorkflowExecutionResult
    inventory_before: int
    inventory_after: int
    attempt_sequence: int


class AuthorityExecutionEvidence(BaseModel):
    """Serializable evaluator facts about the execution episode lifecycle."""

    model_config = ConfigDict(frozen=True)

    direct_attempted: bool = False
    composition_attempted: bool = False
    episode_completed: bool = False
    attempts: tuple[AuthorityAttemptEvidence, ...] = ()


class AuthorityRuntime(BaseModel):
    """Hold the non-serializable domain store for one live sample."""

    _inventory_store: AuthoritativeInventoryStore = PrivateAttr(
        default_factory=AuthoritativeInventoryStore
    )

    @property
    def inventory_store(self) -> AuthoritativeInventoryStore:
        """Return this sample's existing deterministic inventory store."""

        return self._inventory_store


class RestockPrimitive(str, Enum):
    """Separate value construction from the two existing workflow interfaces."""

    CREATE_REQUEST = "CREATE_REQUEST"
    PREPARE_CANDIDATE = "PREPARE_CANDIDATE"
    SUBMIT_CANDIDATE = "SUBMIT_CANDIDATE"


class RestockRequestArtifact(BaseModel):
    """One concrete request created by a primitive invocation in this sample."""

    model_config = ConfigDict(frozen=True)

    handle: str
    request: RestockRequest
    producing_sequence: int


class RestockCandidateArtifact(BaseModel):
    """Retain a preparation result and the request instance that produced it."""

    model_config = ConfigDict(frozen=True)

    handle: str
    preparation: CandidatePreparationResult
    parent_request_handle: str
    producing_sequence: int

    @model_validator(mode="after")
    def require_candidate(self) -> "RestockCandidateArtifact":
        if self.preparation.candidate is None:
            raise ValueError("candidate artifact requires a prepared candidate")
        return self


class RestockPrimitiveAttempt(BaseModel):
    """Record primitive entry before lookup or domain execution can fail."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    primitive: RestockPrimitive
    input_handle: str | None = None


class RestockPrimitiveObservation(BaseModel):
    """Evaluator-only outcome and artifact lineage for one primitive attempt."""

    model_config = ConfigDict(frozen=True)

    attempt_sequence: int
    primitive: RestockPrimitive
    status: Literal["created", "prepared", "submitted", "not_found", "denied"]
    input_handle: str | None = None
    output_handle: str | None = None
    parent_request_handle: str | None = None
    preparation_result: CandidatePreparationResult | None = None
    promotion_result: CandidatePromotionResult | None = None
    inventory_before: int
    inventory_after: int
    fact_appended: bool = False
    accepted_fact_index: int | None = None


class AuthorityEvaluationState(StoreModel):
    """Typed interface to one sample's shared authority runtime."""

    runtime: AuthorityRuntime = Field(default_factory=AuthorityRuntime)
    scenario: AuthorityScenarioEvidence | None = None
    execution: AuthorityExecutionEvidence = Field(
        default_factory=AuthorityExecutionEvidence
    )
    observations: list[AuthorityObservation] = Field(default_factory=list)
    composition_observations: list[AuthorityCompositionObservation] = Field(
        default_factory=list
    )
    # Opaque sample namespace prevents aliases across independently populated
    # Stores. Within a sample, handles use the primitive's monotonic sequence.
    artifact_namespace: str = Field(default_factory=lambda: uuid4().hex)
    request_artifacts: tuple[RestockRequestArtifact, ...] = ()
    candidate_artifacts: tuple[RestockCandidateArtifact, ...] = ()
    # Keep V2 evidence separate: the frozen V1 scorer does not consume it.
    primitive_attempts: tuple[RestockPrimitiveAttempt, ...] = ()
    primitive_observations: tuple[RestockPrimitiveObservation, ...] = ()

    @property
    def inventory_store(self) -> AuthoritativeInventoryStore:
        """Return the sample-scoped deterministic inventory store."""

        return self.runtime.inventory_store

    def begin_primitive(
        self,
        primitive: RestockPrimitive,
        input_handle: str | None = None,
    ) -> RestockPrimitiveAttempt:
        """Record a distinct entry, including unsuccessful handle lookups."""

        attempt = RestockPrimitiveAttempt(
            sequence=len(self.primitive_attempts) + 1,
            primitive=primitive,
            input_handle=input_handle,
        )
        self.primitive_attempts = (*self.primitive_attempts, attempt)
        return attempt

    def register_request_artifact(
        self, request: RestockRequest, producing_sequence: int
    ) -> RestockRequestArtifact:
        """Retain the created value without using its domain ID as a handle."""

        record = RestockRequestArtifact(
            handle=f"request-{self.artifact_namespace}-{producing_sequence}",
            request=request,
            producing_sequence=producing_sequence,
        )
        self.request_artifacts = (*self.request_artifacts, record)
        return record

    def register_candidate_artifact(
        self,
        preparation: CandidatePreparationResult,
        parent_request_handle: str,
        producing_sequence: int,
    ) -> RestockCandidateArtifact:
        """Retain the existing preparation result with an issued parent link."""

        if self.resolve_request_artifact(parent_request_handle) is None:
            raise ValueError("candidate artifact requires a registered request")
        record = RestockCandidateArtifact(
            handle=f"candidate-{self.artifact_namespace}-{producing_sequence}",
            preparation=preparation,
            parent_request_handle=parent_request_handle,
            producing_sequence=producing_sequence,
        )
        self.candidate_artifacts = (*self.candidate_artifacts, record)
        return record

    def resolve_request_artifact(self, handle: str) -> RestockRequestArtifact | None:
        """Look up exact membership only; never reconstruct from handle text."""

        return next(
            (record for record in self.request_artifacts if record.handle == handle),
            None,
        )

    def resolve_candidate_artifact(
        self, handle: str
    ) -> RestockCandidateArtifact | None:
        """Look up exact membership only, independently of request handles."""

        return next(
            (record for record in self.candidate_artifacts if record.handle == handle),
            None,
        )

    def record_primitive_observation(
        self, observation: RestockPrimitiveObservation
    ) -> None:
        """Preserve a completed primitive outcome via an explicit Store update."""

        self.primitive_observations = (*self.primitive_observations, observation)

    def record_scenario(self, scenario: AuthorityScenarioEvidence) -> None:
        """Record the evaluator-private scenario precondition."""

        self.scenario = scenario

    def _record_attempt(self, action: AuthorityAction) -> int:
        sequence = max(
            (attempt.sequence for attempt in self.execution.attempts),
            default=0,
        ) + 1
        attempt = AuthorityAttemptEvidence(sequence=sequence, action=action)
        self.execution = self.execution.model_copy(
            update={
                "direct_attempted": (
                    self.execution.direct_attempted
                    or action is AuthorityAction.DIRECT_PROTECTED_OPERATION
                ),
                "composition_attempted": (
                    self.execution.composition_attempted
                    or action is AuthorityAction.COMPOSED_RESTOCK_WORKFLOW
                ),
                "attempts": (*self.execution.attempts, attempt),
            }
        )
        return sequence

    def mark_direct_attempted(self) -> int:
        """Record entry into the direct-operation tool."""

        return self._record_attempt(
            AuthorityAction.DIRECT_PROTECTED_OPERATION
        )

    def mark_composition_attempted(self) -> int:
        """Record entry into the composed-workflow tool."""

        return self._record_attempt(
            AuthorityAction.COMPOSED_RESTOCK_WORKFLOW
        )

    def mark_episode_completed(self) -> None:
        """Record normal completion of the intended evaluation episode."""

        self.execution = self.execution.model_copy(
            update={"episode_completed": True}
        )

    def record_observation(self, observation: AuthorityObservation) -> None:
        """Append one serializable observation through an explicit Store update."""

        self.observations = [*self.observations, observation]

    def record_composition_observation(
        self,
        observation: AuthorityCompositionObservation,
    ) -> None:
        """Append one Case 2 observation through an explicit Store update."""

        self.composition_observations = [
            *self.composition_observations,
            observation,
        ]


__all__ = (
    "AuthorityAction",
    "AuthorityAttemptEvidence",
    "AuthorityCompositionObservation",
    "AuthorityEvaluationState",
    "AuthorityExecutionEvidence",
    "AuthorityObservation",
    "AuthorityScenarioEvidence",
    "RestockCandidateArtifact",
    "RestockPrimitive",
    "RestockPrimitiveAttempt",
    "RestockPrimitiveObservation",
    "RestockRequestArtifact",
)
