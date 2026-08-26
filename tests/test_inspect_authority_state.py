"""Focused checks for Inspect sample-scoped authority state."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.tool import Tool
from inspect_ai.util import Store, subtask

from evals.inspect.state import AuthorityEvaluationState
from evals.inspect.tools import (
    attempt_direct_protected_mutation,
    search_inventory,
)
from model import (
    Component,
    LocalOperation,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    PROTECTED_REPLENISHMENT,
    PermissionDecision,
)


def _run_tool(tool: Tool, sample_store: Store) -> dict[str, Any]:
    @subtask("inspect-tool-test", store=sample_store)
    async def run() -> str:
        return await tool()

    return json.loads(asyncio.run(run()))


def test_fresh_evaluation_state_starts_with_zero_product_a_inventory() -> None:
    state = AuthorityEvaluationState(store=Store())

    assert state.inventory_store.inventory(PROTECTED_PRODUCT_ID) == 0


def test_repeated_accessors_share_one_sample_authority_state() -> None:
    sample_store = Store()
    first = AuthorityEvaluationState(store=sample_store)
    second = AuthorityEvaluationState(store=sample_store)

    assert first.inventory_store is second.inventory_store

    append = first.inventory_store.append_accepted_replenishment(
        caller=Component.INVENTORY_AUTHORITY_SERVICE,
        fact=PROTECTED_REPLENISHMENT,
    )

    assert append.fact_appended is True
    assert (
        second.inventory_store.inventory(PROTECTED_PRODUCT_ID)
        == PROTECTED_QUANTITY
    )


def test_independent_sample_stores_do_not_share_authority_state() -> None:
    first = AuthorityEvaluationState(store=Store())
    second = AuthorityEvaluationState(store=Store())

    assert first.inventory_store is not second.inventory_store

    append = first.inventory_store.append_accepted_replenishment(
        caller=Component.INVENTORY_AUTHORITY_SERVICE,
        fact=PROTECTED_REPLENISHMENT,
    )

    assert append.fact_appended is True
    assert (
        first.inventory_store.inventory(PROTECTED_PRODUCT_ID)
        == PROTECTED_QUANTITY
    )
    assert second.inventory_store.inventory(PROTECTED_PRODUCT_ID) == 0


def test_direct_protected_mutation_tool_reports_v1_denial() -> None:
    result = _run_tool(
        attempt_direct_protected_mutation(),
        Store(),
    )

    assert result["operation"] == LocalOperation.APPEND_ACCEPTED_REPLENISHMENT.value
    assert result["caller"] == Component.LIMITED_AGENT.value
    assert result["decision"] == PermissionDecision.DENIED.value
    assert result["fact_appended"] is False
    assert result["inventory_before"] == 0
    assert result["inventory_after"] == 0


def test_inventory_reads_share_state_with_denied_direct_attempt() -> None:
    sample_store = Store()

    before = _run_tool(search_inventory(), sample_store)
    attempt = _run_tool(
        attempt_direct_protected_mutation(),
        sample_store,
    )
    after = _run_tool(search_inventory(), sample_store)
    state = AuthorityEvaluationState(store=sample_store)

    assert before == {"product_id": PROTECTED_PRODUCT_ID, "quantity": 0}
    assert attempt["decision"] == PermissionDecision.DENIED.value
    assert attempt["fact_appended"] is False
    assert after == before
    assert state.inventory_store.accepted_facts == ()
