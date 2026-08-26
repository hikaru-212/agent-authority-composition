from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def hello_eval():
    return Task(
        dataset=[
            Sample(
                input="What is the capital of France?",
                target="Paris",
            )
        ],
        solver=generate(),
        scorer=match(),
    )