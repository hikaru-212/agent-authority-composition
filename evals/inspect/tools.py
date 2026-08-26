"""Thin Inspect tool adapters for the deterministic authority model."""

import json

from inspect_ai.tool import Tool, tool

from model import AuthoritativeInventoryStore, PROTECTED_PRODUCT_ID


@tool
def search_inventory() -> Tool:
    """Create a read-only tool for the existing Product A inventory state."""

    store = AuthoritativeInventoryStore()

    async def execute() -> str:
        """Read Product A inventory from the deterministic authority store.

        Returns:
            JSON containing the existing Product A identifier and quantity.
        """

        return json.dumps(
            {
                "product_id": PROTECTED_PRODUCT_ID,
                "quantity": store.inventory(PROTECTED_PRODUCT_ID),
            }
        )

    return execute
