"""Evaluator-only setup for the deterministic protected authority scenario."""

from inspect_ai.solver import Generate, Solver, TaskState, solver

from evals.inspect.state import (
    AuthorityEvaluationState,
    AuthorityScenarioEvidence,
)
from model import (
    Component,
    LocalOperation,
    PROTECTED_PRODUCT_ID,
    PROTECTED_QUANTITY,
    decide_local_capability,
)


def create_authority_scenario_evidence(
    state: AuthorityEvaluationState,
) -> AuthorityScenarioEvidence:
    """Project the existing V1 direct permission check into typed evidence."""

    direct_capability_check = decide_local_capability(
        caller=Component.LIMITED_AGENT,
        operation=LocalOperation.APPEND_ACCEPTED_REPLENISHMENT,
        target=Component.AUTHORITATIVE_INVENTORY_STORE,
    )
    return AuthorityScenarioEvidence(
        direct_capability_check=direct_capability_check,
        product_id=PROTECTED_PRODUCT_ID,
        quantity=PROTECTED_QUANTITY,
        initial_inventory=state.inventory_store.inventory(
            PROTECTED_PRODUCT_ID
        ),
    )


def establish_authority_scenario(state: AuthorityEvaluationState) -> None:
    """Record scenario facts before evaluated behavior begins."""

    state.record_scenario(create_authority_scenario_evidence(state))


@solver
def setup_authority_scenario() -> Solver:
    """Create the evaluator-only, non-mutating scenario setup step."""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        establish_authority_scenario(state.store_as(AuthorityEvaluationState))
        return state

    return solve


__all__ = (
    "create_authority_scenario_evidence",
    "establish_authority_scenario",
    "setup_authority_scenario",
)
