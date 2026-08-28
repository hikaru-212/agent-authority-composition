"""Evaluator-only lifecycle boundaries for authority evaluations."""

from inspect_ai.solver import Generate, Solver, TaskState, chain, solver

from evals.inspect.state import AuthorityEvaluationState


@solver
def with_authority_episode_completion(
    behavior: Solver | list[Solver],
) -> Solver:
    """Record completion only after the behavior solver returns normally."""

    behavior_solver = chain(behavior) if isinstance(behavior, list) else behavior

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await behavior_solver(state, generate)
        result.store_as(AuthorityEvaluationState).mark_episode_completed()
        return result

    return solve


__all__ = ("with_authority_episode_completion",)
