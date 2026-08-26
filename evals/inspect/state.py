"""Sample-scoped Inspect state for the deterministic authority model."""

from inspect_ai.util import StoreModel
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from model import (
    AuthoritativeInventoryStore,
    Component,
    LocalOperation,
    PermissionDecision,
)


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
    observations: list[AuthorityObservation] = Field(default_factory=list)

    @property
    def inventory_store(self) -> AuthoritativeInventoryStore:
        """Return the sample-scoped deterministic inventory store."""

        return self.runtime.inventory_store

    def record_observation(self, observation: AuthorityObservation) -> None:
        """Append one serializable observation through an explicit Store update."""

        self.observations = [*self.observations, observation]


__all__ = ("AuthorityEvaluationState", "AuthorityObservation")
