"""Focused checks for Inspect sample-scoped authority state."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.tool import Tool
from inspect_ai.util import Store, subtask
from pytest import MonkeyPatch

from evals.inspect.state import AuthorityEvaluationState, AuthorityObservation
from evals.inspect.tools import (
    attempt_direct_protected_mutation,
    search_inventory,
)
from model import (
    Component,
    LimitedAgent,
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
    assert state.observations == []


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


def test_direct_protected_mutation_separates_response_from_evidence(
    monkeypatch: MonkeyPatch,
) -> None:
    v1_results: list[Any] = []
    attempt_direct_append = LimitedAgent.attempt_direct_append

    def record_v1_result(*args: Any, **kwargs: Any) -> Any:
        result = attempt_direct_append(*args, **kwargs)
        v1_results.append(result)
        return result

    monkeypatch.setattr(
        LimitedAgent,
        "attempt_direct_append",
        record_v1_result,
    )
    sample_store = Store()
    result = _run_tool(
        attempt_direct_protected_mutation(),
        sample_store,
    )
    state = AuthorityEvaluationState(store=sample_store)

    assert len(v1_results) == 1
    v1_result = v1_results[0]
    assert v1_result.capability_check.decision is PermissionDecision.DENIED
    assert result == {"decision": v1_result.capability_check.decision.value}
    for evaluator_only_field in (
        "inventory_before",
        "inventory_after",
        "fact_appended",
    ):
        assert evaluator_only_field not in result
    assert len(state.observations) == 1

    observation = state.observations[0]
    assert isinstance(observation, AuthorityObservation)
    expected_observation = {
        "operation": LocalOperation.APPEND_ACCEPTED_REPLENISHMENT.value,
        "caller": Component.LIMITED_AGENT.value,
        "target": Component.AUTHORITATIVE_INVENTORY_STORE.value,
        "product_id": PROTECTED_PRODUCT_ID,
        "quantity": PROTECTED_QUANTITY,
        "decision": PermissionDecision.DENIED.value,
        "fact_appended": False,
        "inventory_before": 0,
        "inventory_after": 0,
    }
    assert observation.model_dump(mode="json") == expected_observation
    assert json.loads(observation.model_dump_json()) == expected_observation
    assert state.inventory_store.inventory(PROTECTED_PRODUCT_ID) == 0


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
    assert attempt == {"decision": PermissionDecision.DENIED.value}
    assert after == before
    assert state.inventory_store.accepted_facts == ()
    assert len(state.observations) == 1
    assert state.observations[0].fact_appended is False


def test_independent_sample_stores_do_not_share_observations() -> None:
    first_store = Store()
    second_store = Store()

    _run_tool(attempt_direct_protected_mutation(), first_store)

    first = AuthorityEvaluationState(store=first_store)
    second = AuthorityEvaluationState(store=second_store)

    assert len(first.observations) == 1
    assert second.observations == []
    assert second.inventory_store.inventory(PROTECTED_PRODUCT_ID) == 0
