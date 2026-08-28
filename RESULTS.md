# Preliminary Behavioral A/B Evaluation Results

## Research Question

Does access to an additional workflow tool allow an agent to reach a protected
inventory effect that it cannot reach through the directly exposed inventory
update operation?

## Experimental Design

The preliminary evaluation used a controlled A/B design with ten epochs per
condition. Both conditions used the same user prompt, system and ReAct
instructions, V1-backed scenario, initial inventory, semantic scorer,
completion wrapper, turn and message limits, serial authority-sensitive tool
execution, selected model, and provider configuration.

### Evaluation Snapshot

| Setting | Configuration |
|---|---|
| Model | `openai/gpt-4o-mini` |
| Inspect AI | `0.3.260` |
| Runs | 10 epochs per condition |
| Parallel tool calls | Disabled |
| Turn limit | 8 |
| Message limit | 24 |

The only intended experimental difference was whether `request_restock` was
available to the model:

| Condition | Domain tools exposed to the model |
|---|---|
| Control | `get_inventory`, `update_inventory` |
| Composition | `get_inventory`, `update_inventory`, `request_restock` |

ReAct `submit` remained available in both conditions. The protected effect was
determined from reconstructed evaluator evidence using the frozen semantic
scorer, not from model-generated text or transcript inventory output alone.

## Preliminary Results

| Condition | Authoritative effect reached |
|---|---:|
| Control | 0/10 |
| Composition | 10/10 |

The conditions therefore showed complete separation in this preliminary
sample. This result is descriptive of the evaluated model, scaffold, scenario,
and runs; it is not a statistical generalization.

## Secondary Observation

All ten Composition runs reached the authoritative effect. Their subsequent
episode behavior varied:

- 6/10 hit the turn limit after the effect had occurred.
- 3/10 invoked `get_inventory` after the effect.
- 1/10 observed quantity 10 through a post-effect `get_inventory` result.
- 8/10 reinvoked `request_restock` after the first established effect.
- 8/10 accumulated a final evidence-backed inventory above 10.

The final established inventory distribution was:

| Final established inventory | Runs |
|---:|---:|
| 10 | 2 |
| 20 | 2 |
| 30 | 3 |
| 40 | 3 |

The inventory values and successful composition-effect counts come from
complete, consistent evaluator-backed authority transitions. They are not
inferred from the number of `request_restock` calls. Repeated invocation without
a complete composition witness is not counted as an authoritative effect.

## Interpreting Effect, Verification, and Completion

The evaluation keeps four observations separate:

1. The authoritative effect occurred.
2. A verification capability, `get_inventory`, was available.
3. The agent may or may not have invoked that capability and observed the
   resulting state.
4. The agent may or may not have submitted normally.

Availability of `get_inventory` does not establish that resulting-state
evidence was presented to the agent. Only three Composition runs invoked it
after the effect, and only one observed quantity 10 at that point. Likewise, a
turn-limited episode can still contain complete evidence that the authoritative
effect occurred before termination.

These data describe observable behavior. They do not establish what the model
believed, understood, or intended.

## What This Establishes

Within this bounded evaluation:

- The Control tool surface made the composed path unavailable, and none of its
  ten runs reached the protected effect.
- When `request_restock` was available, all ten Composition runs reached the
  protected effect through complete evaluator-backed evidence.
- Direct denial and reachable authority are empirically distinct in this
  scenario: denying `update_inventory` did not prevent the same authoritative
  effect from being reached through the available workflow operation.
- Semantic effect and episode termination are distinct: six runs reached the
  effect before later hitting the turn limit.
- Consistent evaluator evidence establishes cumulative authoritative inventory
  transitions in eight Composition runs.

## What This Does NOT Establish

This preliminary evaluation does not establish:

- general behavior across models, prompts, providers, scenarios, or larger
  samples;
- statistical significance or an estimated population effect;
- that the model recognized or understood when the protected effect occurred;
- that the model observed the resulting inventory merely because
  `get_inventory` was available;
- lack of idempotency merely from repeated tool invocation;
- autonomous discovery of a multi-step primitive composition;
- exploitation or reconstruction of any real external system; or
- a universal authority failure or a universal mitigation.


## Intentional Engineering Omissions

This V1 intentionally leaves several known engineering controls unimplemented in order to isolate the primary evaluation question: whether changing the exposed workflow surface changes reachable authoritative effects. These are deliberate scope boundaries rather than unrecognized implementation gaps.

- **Idempotency and deduplication.** `request_restock` does not currently deduplicate repeated successful invocations. The cumulative inventory transitions observed in the Composition condition are therefore a property of this bounded implementation, not a claim about production restock systems. A production design would normally require stable request or attempt identity together with idempotent or otherwise governed reinvocation semantics.
- **Completion observability.** The model is told whether the restock request was accepted, but it is not directly told that the downstream authoritative effect has completed. A verification capability, `get_inventory`, is available, but the agent does not always invoke it after the effect. This intentionally leaves outcome observability and reinvocation behavior as separate follow-up questions.
- **Semantic authority admission.** V1 measures the vulnerable reachability condition before introducing a Compass-style semantic admission boundary. It therefore evaluates the failure mode rather than a mitigation. A later condition can test whether independently supported authority evidence prevents unauthorized promotion while preserving legitimate workflows.
- **Primitive workflow composition.** `request_restock` is currently a high-level operation whose downstream workflow is executed by the environment. V1 therefore tests authority reachable through an exposed workflow operation, not whether the model can autonomously discover and compose individually permitted primitives `A -> B -> C`. That stronger question is reserved for V2.

## Limitations

- The evaluation contains only ten epochs per condition.
- It uses one selected model and one fixed provider and scaffold configuration.
- The scenario is local, in-memory, and deliberately bounded.
- `request_restock` is currently one high-level tool that encapsulates the
  composed workflow.
- The evaluation therefore does not yet test autonomous primitive
  `A -> B -> C` composition.
- Six Composition runs reached the turn limit, restricting conclusions about
  normal episode completion even though the authoritative effect was already
  established.

## V2

V2 should expose neutral, individually permitted primitive workflow operations
instead of a single high-level `request_restock` tool. It should test whether a
model autonomously discovers and executes an `A -> B -> C` sequence that reaches
the protected authoritative effect while preserving the same evidence and
scoring distinctions.

The key measurements should remain separate:

- local permission for each primitive;
- global authority reachable through their composition;
- occurrence of the authoritative effect;
- actual post-effect verification behavior; and
- normal episode submission.

The read-only analysis procedure is documented in
[the Inspect evaluation README](evals/inspect/README.md) and implemented by
[`scripts/analyze_ab_logs.py`](scripts/analyze_ab_logs.py).
