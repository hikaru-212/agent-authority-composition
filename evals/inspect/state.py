"""Sample-scoped Inspect state for the deterministic authority model."""

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field, PrivateAttr

from model import AuthoritativeInventoryStore


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

    @property
    def inventory_store(self) -> AuthoritativeInventoryStore:
        """Return the sample-scoped deterministic inventory store."""

        return self.runtime.inventory_store


__all__ = ("AuthorityEvaluationState",)
