"""Minimal Inspect task proving inventory-tool plumbing into the V1 model."""

import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolCall

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.inspect.tools import search_inventory


@task
def authority_eval() -> Task:
    """Force one Product A inventory read so Inspect records the tool result."""

    return Task(
        dataset=[
            Sample(
                input=(
                    "Call search_inventory once to read the current Product A "
                    "inventory."
                )
            )
        ],
        solver=[
            use_tools(search_inventory()),
            generate(tool_calls="single"),
        ],
        model=get_model(
            "mockllm/model",
            memoize=False,
            custom_outputs=[
                ModelOutput(
                    model="mockllm",
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(
                                content="",
                                tool_calls=[
                                    ToolCall(
                                        id="search-inventory-1",
                                        function="search_inventory",
                                        arguments={},
                                    )
                                ],
                            ),
                            stop_reason="tool_calls",
                        )
                    ],
                )
            ],
        ),
    )
