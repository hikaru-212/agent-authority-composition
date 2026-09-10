"""Provider-independent, bounded primitive inventory ReAct conditions."""

import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentSubmit, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import Tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.inspect.behavioral_eval import (
    INVENTORY_AGENT_PROMPT,
    INVENTORY_BEHAVIOR_PROMPT,
)
from evals.inspect.behavioral_tools import inventory_behavior_tools
from evals.inspect.lifecycle import with_authority_episode_completion
from evals.inspect.primitive_behavioral_scorer import primitive_behavioral_scorer
from evals.inspect.primitive_scorer import primitive_execution_scorer
from evals.inspect.primitive_tools import (
    create_restock_request,
    prepare_restock_candidate,
    submit_restock_candidate,
)
from evals.inspect.scenario import setup_authority_scenario


def primitive_behavior_tools(*, preparation_access: bool = True) -> tuple[Tool, ...]:
    """Vary only preparation availability; never expose the workflow shortcut."""

    return (
        *inventory_behavior_tools(composition_access=False),
        create_restock_request(),
        *((prepare_restock_candidate(),) if preparation_access else ()),
        submit_restock_candidate(),
    )


def primitive_behavior_agent(*, preparation_access: bool = True) -> Agent:
    return react(
        prompt=INVENTORY_AGENT_PROMPT,
        tools=primitive_behavior_tools(preparation_access=preparation_access),
        attempts=1,
        submit=AgentSubmit(description="Submit the result."),
        retry_refusals=0,
        compaction=None,
        truncation="disabled",
    )


def _primitive_behavior_task(*, preparation_access: bool) -> Task:
    return Task(
        dataset=[Sample(input=INVENTORY_BEHAVIOR_PROMPT)],
        setup=setup_authority_scenario(),
        solver=with_authority_episode_completion(
            as_solver(primitive_behavior_agent(preparation_access=preparation_access))
        ),
        scorer=[primitive_execution_scorer(), primitive_behavioral_scorer()],
        config=GenerateConfig(parallel_tool_calls=False),
        message_limit=24,
        turn_limit=8,
        # Preserve a live authority snapshot on ordinary episode exceptions.
        # This does not turn an error into normal completion or retry a sample.
        score_on_error=True,
    )


@task
def primitive_behavior_control_eval() -> Task:
    """Inventory behavior without candidate preparation."""
    return _primitive_behavior_task(preparation_access=False)


@task
def primitive_behavior_treatment_eval() -> Task:
    """Inventory behavior with all three separate primitives."""
    return _primitive_behavior_task(preparation_access=True)
