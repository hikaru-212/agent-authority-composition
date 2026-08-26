"""Sample-scoped Inspect state for the deterministic authority model."""

from enum import Enum

from inspect_ai.util import StoreModel
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from model import (
    AuthoritativeInventoryStore,
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

    @property
    def inventory_store(self) -> AuthoritativeInventoryStore:
        """Return the sample-scoped deterministic inventory store."""

        return self.runtime.inventory_store

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
)
