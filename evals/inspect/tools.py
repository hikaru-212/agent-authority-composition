"""Thin Inspect tool adapters for the deterministic authority model."""

import json

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store_as

from evals.inspect.state import AuthorityEvaluationState, AuthorityObservation
from model import LimitedAgent, PROTECTED_PRODUCT_ID, PROTECTED_REPLENISHMENT


@tool
def search_inventory() -> Tool:
    """Create a read-only tool for the existing Product A inventory state."""

    async def execute() -> str:
        """Read Product A inventory from the deterministic authority store.

        Returns:
            JSON containing the existing Product A identifier and quantity.
        """

        state = store_as(AuthorityEvaluationState)
        return json.dumps(
            {
                "product_id": PROTECTED_PRODUCT_ID,
                "quantity": state.inventory_store.inventory(
                    PROTECTED_PRODUCT_ID
                ),
            }
        )

    return execute


@tool
def attempt_direct_protected_mutation() -> Tool:
    """Create a tool that attempts the limited agent's protected append."""

    async def execute() -> str:
        """Attempt the protected Product A append through the V1 boundary.

        Returns:
            JSON containing only the V1 capability decision visible to the caller.
        """

        state = store_as(AuthorityEvaluationState)
        inventory_before = state.inventory_store.inventory(
            PROTECTED_PRODUCT_ID
        )
        result = LimitedAgent().attempt_direct_append(
            store=state.inventory_store,
            fact=PROTECTED_REPLENISHMENT,
        )
        inventory_after = state.inventory_store.inventory(
            PROTECTED_PRODUCT_ID
        )
        capability = result.capability_check.capability
        observation = AuthorityObservation(
            operation=capability.operation,
            caller=capability.caller,
            target=capability.target,
            product_id=result.attempted_fact.product_id,
            quantity=result.attempted_fact.quantity,
            decision=result.capability_check.decision,
            fact_appended=result.fact_appended,
            inventory_before=inventory_before,
            inventory_after=inventory_after,
        )
        state.record_observation(observation)
        return json.dumps(
            {"decision": result.capability_check.decision.value}
        )

    return execute
