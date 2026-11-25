"""Three bounded V3 tasks with the same neutral model-facing interface."""

import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentSubmit, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.inspect.admission_behavioral_scorer import admission_behavioral_scorer
from evals.inspect.admission_scorer import admission_execution_scorer
from evals.inspect.admission_state import (
    AdmissionCondition, AdmissionEvaluationState, establish_admission_scenario,
)
from evals.inspect.admission_tools import admission_submit_restock_candidate
from evals.inspect.behavioral_eval import INVENTORY_AGENT_PROMPT, INVENTORY_BEHAVIOR_PROMPT
from evals.inspect.behavioral_tools import inventory_behavior_tools
from evals.inspect.lifecycle import with_authority_episode_completion
from evals.inspect.primitive_tools import create_restock_request, prepare_restock_candidate
from evals.inspect.state import AuthorityEvaluationState


@solver
def setup_admission_scenario(condition: AdmissionCondition) -> Solver:
    """Seed only evaluator state before the common model conversation."""
    if not isinstance(condition, AdmissionCondition):
        raise TypeError("condition must be AdmissionCondition")

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        establish_admission_scenario(
            state.store_as(AuthorityEvaluationState), state.store_as(AdmissionEvaluationState),
            condition=condition,
        )
        return state

    return solve


def admission_behavior_tools(*, condition: AdmissionCondition) -> tuple[Tool, ...]:
    return (
        *inventory_behavior_tools(composition_access=False),
        create_restock_request(),
        prepare_restock_candidate(),
        admission_submit_restock_candidate(condition=condition),
    )


def admission_behavior_agent(*, condition: AdmissionCondition) -> Agent:
    return react(
        prompt=INVENTORY_AGENT_PROMPT,
        tools=admission_behavior_tools(condition=condition),
        attempts=1,
        submit=AgentSubmit(description="Submit the result."),
        retry_refusals=0,
        compaction=None,
        truncation="disabled",
    )


def admission_behavior_task(*, condition: AdmissionCondition) -> Task:
    return Task(
        dataset=[Sample(input=INVENTORY_BEHAVIOR_PROMPT)],
        setup=setup_admission_scenario(condition),
        solver=with_authority_episode_completion(as_solver(admission_behavior_agent(condition=condition))),
        scorer=[admission_execution_scorer(condition), admission_behavioral_scorer(condition)],
        config=GenerateConfig(parallel_tool_calls=False),
        message_limit=24,
        turn_limit=8,
        score_on_error=True,
    )


@task
def admission_behavior_vulnerable_eval() -> Task:
    return admission_behavior_task(condition=AdmissionCondition.VULNERABLE)


@task
def admission_behavior_unsupported_eval() -> Task:
    return admission_behavior_task(condition=AdmissionCondition.GOVERNED_UNSUPPORTED)


@task
def admission_behavior_supported_eval() -> Task:
    return admission_behavior_task(condition=AdmissionCondition.GOVERNED_SUPPORTED)
