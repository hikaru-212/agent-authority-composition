# Preliminary V1 and V2 Behavioral Evaluation Results

## V2 research question

Can a tool-using model select and execute separately exposed workflow primitives,
consuming the actual outputs of earlier calls, to reach an authoritative inventory
effect that remains unavailable through its direct inventory-update interface?

## V2 experimental design

The formal preliminary experiment used `openai/gpt-4o-mini` for 10 Control and
10 Treatment runs. The analysis contract, `v2-primitive-analysis-1`, was frozen
before execution. The pre-experiment tag was `v2-primitive-pre-experiment`; the
experiment ran from commit `fa173c9`. These results refer to the formal 20-run
experiment, not the earlier provider smoke samples.

Both conditions used the same model, prompt, scenario, initial inventory of 0,
ReAct scaffold, semantic scorer, behavioral scorer, turn/message limits, serial
tool execution, provider configuration, and retry configuration. The only
intended independent variable was candidate-preparation availability.

| Condition | Model-facing tools |
|---|---|
| Control | `get_inventory`, `update_inventory`, `create_restock_request`, `submit_restock_candidate`, `submit` |
| Treatment | The same tools plus `prepare_restock_candidate` |

Neither condition exposed the old high-level `request_restock` shortcut.
Control lacked the intermediate preparation operation required to produce a
valid candidate artifact. Removing it is an experimental ablation for causal
isolation, not a proposed production mitigation.

The analysis uses persisted evaluator state, accepted-history snapshots, and
model/tool events under the [frozen analysis contract](evals/inspect/PRIMITIVE_ANALYSIS.md).
The [read-only V2 analyzer](scripts/analyze_primitive_ab_logs.py) reclassifies
that evidence with the semantic and behavioral classifiers. Final inventory or
model-generated text alone does not establish a model-directed effect.

## V2 primary results

| Measurement | Control | Treatment |
|---|---:|---:|
| Runs | 10 | 10 |
| Authoritative effect established | 0/10 | 10/10 |
| Samples with at least one established model-directed primitive effect | 0/10 | 10/10 |
| Semantic outcome distribution | 10/10 `NO_PRIMITIVE_EFFECT` | 10/10 `PRIMITIVE_COMPOSITION_REACHED` |
| Total complete authoritative effects | 0 | 12 |
| Total established model-directed effects (witnessed lower bound) | 0 | 12 |
| Final evidence-backed inventory | 0 in all 10 runs | 10 in 8 runs; 20 in 2 runs |

When candidate preparation was absent, none of the ten runs reached the
authoritative effect. When it was available, all ten runs produced at least one
complete, transcript-linked and artifact-linked A → B → C trajectory reaching
that effect. Direct inventory update remained denied.

These observations distinguish primitive composition from direct permission in
this bounded environment. They describe this model, prompt, scenario, and set
of runs; they are not estimates of population-level behavior.

The 12 model-directed effects are complete established witnesses. Because some
runs ended with unresolved model-selected calls, this count should not be read
as an exhaustive count of every possible model-directed effect.

## V2 secondary observations

| Treatment measurement | Runs |
|---|---:|
| Post-effect primitive reinvocation | 9/10 |
| Repeated authoritative effects | 2/10 |
| Final inventory above the protected quantity of 10 | 2/10 |
| Repeated submission of the same candidate | 0/10 |
| Turn limit | 9/10 |
| Normal completion | 1/10 |
| Errors | 0/10 |

In each of the two runs ending with inventory 20, the model completed a new
workflow chain that produced a second authoritative effect. These were
workflow-level re-initiations, not same-candidate resubmissions. The evidence
does not require describing them as retries. Additional effects are counted
from complete witnesses, not inferred from inventory or tool-call counts alone.

Effect occurrence remained distinct from normal episode completion: all ten
Treatment runs established an effect, although nine later reached the turn
limit. Post-effect workflow activity could also produce additional authoritative
effects.

## V2 evidence interpretation and measurement choices

```text
semantic outcome != behavioral evidence completeness != episode completion
```

Behavioral `INSUFFICIENT_EVIDENCE` does not mean that no model-directed effect
was established. Several samples completed a model-directed chain before a
later model-selected primitive call remained unexecuted at the turn limit.
That unresolved selection does not erase the earlier witness or create a new
effect. The primary lineage measurement is samples with at least one
established model-directed effect: 10/10 Treatment and 0/10 Control.

An effect may also occur before evaluator outcome recording completes. Missing
outcome evidence is therefore not proof that no effect occurred. The scorers
preserve lower-bound witnesses and classify incomplete evidence conservatively.

V2 intentionally remains non-idempotent during measurement: repeated workflow
execution must remain observable as repeated authoritative effects rather than
being hidden by deduplication. This is a measurement choice, not a recommendation
for non-idempotent production behavior or evidence of an effective retry
mitigation.

## V2 limitations

This preliminary experiment does not establish:

- general behavior across models, prompts, or providers;
- statistical significance or population-level rates;
- model belief, understanding, or intent;
- discovery of a hidden or opaque exploit—the tool semantics provide operational
  cues;
- a production-system vulnerability or universal authority escalation;
- effectiveness of Compass semantic admission;
- effectiveness of idempotency or retry mitigation; or
- safety of arbitrary agent tooling.

## V1 vs V2

In V1, the model invoked one high-level `request_restock` tool whose
implementation internally executed the composed workflow. V1 established that
workflow access expanded reachable authority in the evaluated scenario.

In V2, the model had to select separate primitives and consume actual returned
handles:

```text
A: create_restock_request()                    -> request_handle
B: prepare_restock_candidate(request_handle)   -> candidate_handle
C: submit_restock_candidate(candidate_handle) -> authoritative effect
```

A constructs a request value; it is not a capability-checked edge. B and C
preserve the existing allowed workflow interfaces, after which the service
performs its existing authoritative append.

V2 strengthens the bounded claim: the model can itself select and execute the
exposed primitive workflow sequence that reaches the authoritative effect.
This is not a claim that it discovered a hidden vulnerability or understood the
authority semantics.

<details>
<summary>Frozen V1 results and interpretation</summary>

### V1 research question

Does access to an additional workflow tool allow an agent to reach a protected
inventory effect that it cannot reach through the directly exposed inventory
update operation?

### Experimental Design

The preliminary evaluation used a controlled A/B design with ten epochs per
condition. Both conditions used the same user prompt, system and ReAct
instructions, V1-backed scenario, initial inventory, semantic scorer,
completion wrapper, turn and message limits, serial authority-sensitive tool
execution, selected model, and provider configuration.

#### Evaluation Snapshot

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

### Preliminary Results

| Condition | Authoritative effect reached |
|---|---:|
| Control | 0/10 |
| Composition | 10/10 |

The conditions therefore showed complete separation in this preliminary
sample. This result is descriptive of the evaluated model, scaffold, scenario,
and runs; it is not a statistical generalization.

### Secondary Observation

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

### Interpreting Effect, Verification, and Completion

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

### What This Establishes

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

### What This Does NOT Establish

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


### Intentional Engineering Omissions

This V1 intentionally leaves several known engineering controls unimplemented in order to isolate the primary evaluation question: whether changing the exposed workflow surface changes reachable authoritative effects. These are deliberate scope boundaries rather than unrecognized implementation gaps.

- **Idempotency and deduplication.** `request_restock` does not currently deduplicate repeated successful invocations. The cumulative inventory transitions observed in the Composition condition are therefore a property of this bounded implementation, not a claim about production restock systems. A production design would normally require stable request or attempt identity together with idempotent or otherwise governed reinvocation semantics.
- **Completion observability.** The model is told whether the restock request was accepted, but it is not directly told that the downstream authoritative effect has completed. A verification capability, `get_inventory`, is available, but the agent does not always invoke it after the effect. This intentionally leaves outcome observability and reinvocation behavior as separate follow-up questions.
- **Semantic authority admission.** V1 measures the vulnerable reachability condition before introducing a Compass-style semantic admission boundary. It therefore evaluates the failure mode rather than a mitigation. A later condition can test whether independently supported authority evidence prevents unauthorized promotion while preserving legitimate workflows.
- **Primitive workflow composition.** In V1, `request_restock` is a high-level operation whose downstream workflow is executed by the environment. V1 therefore tests authority reachable through an exposed workflow operation, not whether the model can select and execute separate workflow primitives `A -> B -> C`. The completed V2 evaluation above addresses that stronger question.

### Limitations

- The evaluation contains only ten epochs per condition.
- It uses one selected model and one fixed provider and scaffold configuration.
- The scenario is local, in-memory, and deliberately bounded.
- `request_restock` is currently one high-level tool that encapsulates the
  composed workflow.
- V1 does not test model-directed primitive
  `A -> B -> C` composition.
- Six Composition runs reached the turn limit, restricting conclusions about
  normal episode completion even though the authoritative effect was already
  established.

The V1 read-only analysis procedure is documented in
[the Inspect evaluation README](evals/inspect/README.md) and implemented by
[`scripts/analyze_ab_logs.py`](scripts/analyze_ab_logs.py).

</details>

## V3: retain workflow capability, govern acceptance

V3 should keep the useful A+B+C workflow available while introducing
Compass-style semantic admission before a proposed transition becomes durable
authoritative state:

```text
Agent -> A -> B -> C -> proposed candidate / transition
                    -> semantic admission -> ALLOW or REJECT
                    -> accepted history only on ALLOW
```

The boundary to test is:

```text
proposal != authorized transition
candidate != accepted history
workflow access != state-transition authority
```

Control's removal of B is an experimental ablation. V3 asks the stronger,
production-oriented question: can the system retain A+B+C capability while
preventing unsupported composed actions from becoming authoritative business
truth? The V2 results do not yet answer that mitigation question.
