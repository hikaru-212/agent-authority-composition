# Reachable Authority Through Allowed Workflows

Can an actor that is denied a protected operation still reach the same
authoritative effect through a sequence of individually allowed operations?

This repository gives a deterministic counterexample showing that the answer
can be yes.

## Minimal counterexample

The limited actor cannot append the protected inventory fact directly:

```text
LIMITED_AGENT
→ AuthoritativeInventoryStore
→ APPEND_ACCEPTED_REPLENISHMENT
= DENIED
```

In the deliberately vulnerable model, the same actor can reach the same effect
through locally permitted workflow edges:

```text
LIMITED_AGENT
→ RestockWorkflow                     ALLOWED
→ InventoryAuthorityService           ALLOWED
→ AuthoritativeInventoryStore         ALLOWED
→ AcceptedStockReplenished(product-a, 10)
```

Every invoked edge is allowed. The failure occurs because workflow arrival is
treated as sufficient reason to promote a candidate into an accepted fact:

```text
∀ edge ∈ path:
    locally_authorized(edge)

does not imply

globally_authorized_promotion(path)
```

The central distinction is:

```text
direct permission != reachable authority
```

## Why it matters

Local permission checks bound individual operations; they do not necessarily
bound the authoritative effects reachable through compositions of those
operations. A privileged service can exercise its legitimate capability on
behalf of a weaker workflow input while losing the semantic distinction
between what was requested and what was established.

The inventory example makes that distinction concrete:

```text
restock requested  != stock replenished
user demand        != warehouse evidence
candidate produced != fact established
```

The problem generalizes to multi-agent and service workflows wherever weaker
propositions can be promoted into stronger authoritative claims.

## Proposed semantic boundary

The governed sibling path separates proposal from acceptance:

```text
candidate
    ↓
semantic authority admission
    ↓
accepted fact
```

```text
Agent writes candidate.
Authority service writes accepted fact.
```

Admission checks whether evidence was issued through a modeled source,
correlates to the exact candidate, supports the proposition being promoted,
and comes from a source authorized for that proposition. Correlation alone is
not authority.

This boundary leaves the workflow available. Unsupported agent-request
evidence is rejected, while the same candidate can be accepted when it is
supported by a separate modeled warehouse observation.

## Current result

The executable model demonstrates four cases against fresh in-memory stores:

| Case | What it establishes | Result |
|---|---|---|
| Direct denial | The closed capability rule denies the exercised append | `DENIED`, inventory `0` |
| Locally valid indirect failure | Allowed edges can compose into the protected effect | accepted, inventory `10` |
| Semantically governed rejection | Correlated request evidence is insufficient for replenishment | `REJECT`, inventory `0` |
| Authorized positive control | Modeled warehouse-grounded evidence permits the same workflow | `ACCEPT`, inventory `10` |

```text
DIRECT DENIAL
    ↓
LOCALLY VALID INDIRECT FAILURE
    ↓
SEMANTICALLY GOVERNED REJECTION
    ↓
AUTHORIZED POSITIVE CONTROL
```

The deterministic core model and demo are single-process, synchronous,
deterministic, in-memory, and standard-library-only. They intentionally exclude network services,
databases, credentials, concurrency, retries, LLMs, and stochastic behavior so
the authority-composition question remains isolated. The test suite uses
`pytest` as its runner.

The Inspect-based LLM evaluations are layered separately on top of this
deterministic core.

This is a bounded research model, not a reconstruction or exploit of an
external system and not a claim of a universal solution. The problem was
identified while developing Compass, but this repository has no Compass
runtime dependency and stands on its own.

## Evaluation results

The completed preliminary V1 and V2 evaluations are summarized in
[Preliminary V1 and V2 Behavioral Evaluation Results](RESULTS.md). In V2,
Control established an authoritative effect in 0/10 runs; Treatment established
at least one model-directed primitive workflow effect in 10/10. The report
separates effects, evidence completeness, and episode completion, and states
the bounded interpretation and limitations.

## Read and run

Follow the material in progressively greater depth:

1. [Research brief](RESEARCH_BRIEF.md) — the technical argument, current
   evidence, limits, and open questions.
2. [Long-form concept note](shared_workflow_is_not_shared_authority.md) — the
   full conceptual derivation and commerce context.
3. [Executable demo documentation](docs/demo.md) — exact flows, invariants,
   implementation map, test coverage, and experimental boundaries.
4. [Demo source](demo.py) and [model source](model.py) — the deterministic
   implementation.
5. Tests: [orchestration](tests/test_demo.py),
   [counterexample](tests/test_model.py), and
   [semantic admission](tests/test_semantic_authority_admission.py).

## Reproducible setup

From the repository root, create and activate a virtual environment, then
install the test dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the deterministic executable demonstration:

```bash
python demo.py
```

Run the verification suite:

```bash
pytest -q
```

The demo runtime uses only the Python standard library. `pytest` is required
only for the test suite.
