# When Local Permission Does Not Bound Reachable Authority

This brief describes a finite, executable authority-composition
counterexample. It separates the result demonstrated by the repository from
the broader questions the result motivates.

## 1. Problem

Local authorization usually answers a question about one operation:

```text
May this caller invoke this operation on this target?
```

Reachable authority asks a different question:

```text
Which authoritative effects can this actor cause through every workflow path
available to it?
```

Those questions are not equivalent. Denying an actor a direct operation does
not establish that the operation's authoritative effect is unreachable through
composition.

The repository models one protected effect precisely: membership of
`AcceptedStockReplenished(product_id="product-a", quantity=10)` in
`AuthoritativeInventoryStore.accepted_facts`. Inventory is derived only by
folding that accepted history. Constructing a request, candidate, evidence
record, or even an `AcceptedStockReplenished` value does not by itself create
authoritative state.

The limited actor's direct append is denied. A deliberately vulnerable
workflow nevertheless lets the actor cause the exact same accepted fact to be
appended. The actor does not acquire the privileged store capability and no
denied edge is bypassed. Instead, `InventoryAuthorityService` exercises its
own valid capability after making an unsupported semantic promotion.

Formally, the counterexample establishes:

```text
∀ edge ∈ path:
    locally_authorized(edge)

does not imply

globally_authorized_promotion(path)
```

The problem depends on distinctions that local operation checks alone do not
capture:

```text
direct permission          != reachable authority
edge-valid                 != path-authorized
local capability authority != semantic evidence authority
candidate produced         != fact established
evidence exists            != evidence sufficient
evidence correlation       != evidence authority
```

## 2. Why it matters

Multi-agent and service systems routinely transform propositions as work moves
through a workflow. A useful request or observation can become unsafe when
workflow completion, service identity, or data correlation is mistaken for
evidence supporting a stronger claim.

The repository uses a commerce example. A user request may justify the
proposition:

```text
restock requested
```

It does not establish:

```text
stock replenished
```

An unsafe flow can erase that distinction:

```text
user demand
    ↓
restock request
    ↓
workflow candidate
    ↓
privileged service promotion
    ↓
accepted replenishment fact
```

The inventory domain is only a concrete instance of a broader authority
problem. Similar proposition-strengthening can occur wherever workflows turn
recommendations into decisions, reports into attestations, requests into
approvals, or proposed actions into durable facts.

The relevant semantic separations are:

```text
user demand           != warehouse evidence
restock request       != stock replenishment
agent-generated event != authority fact
shared workflow       != shared authority
```

The long-form concept note also discusses search-time versus commit-time truth.
That timing issue follows from the broader conceptual boundary but is not an
executed case in this repository's replenishment demo.

## 3. Minimal counterexample

The direct path attempts the protected append at the store boundary:

```text
LIMITED_AGENT
→ AuthoritativeInventoryStore
→ APPEND_ACCEPTED_REPLENISHMENT
= DENIED
```

The store remains unchanged: accepted-fact count is `0`, and derived inventory
is `0`.

The vulnerable path starts with the same limited actor and targets the exact
same replenishment fact:

```text
LIMITED_AGENT
→ RestockWorkflow                     ALLOWED
→ InventoryAuthorityService           ALLOWED
→ AuthoritativeInventoryStore         ALLOWED
→ AcceptedStockReplenished(product-a, 10)
```

The path performs three local checks:

1. `LIMITED_AGENT` may submit a restock request to `RestockWorkflow`.
2. `RestockWorkflow` may submit a replenishment candidate to
   `InventoryAuthorityService`.
3. `InventoryAuthorityService` may append an accepted replenishment to the
   authoritative store.

All three return `ALLOWED`. `RestockWorkflow` produces a
`CandidateStockReplenished`, and the deliberately vulnerable service method
promotes it without semantic authority admission. The store then contains one
accepted fact and derived inventory is `10`.

This is not a demonstration that ACLs are irrelevant. The direct ACL works as
specified. It is a demonstration that valid local edges do not, by themselves,
justify the semantic transition performed across the full path.

## 4. Proposed boundary

The governed sibling path inserts an explicit boundary between a proposed
claim and an accepted fact:

```text
CandidateStockReplenished
    ↓
SemanticAuthorityAdmission
    ↓
AuthorityPromotionDecision
    ↓
AcceptedStockReplenished, only after ACCEPT
```

The responsibility split is:

```text
Agent writes candidate.
Authority service writes accepted fact.
```

In this model, a candidate can be promoted only if at least one individual
evidence record satisfies every part of the promotion invariant:

```text
promoted(candidate)
⇒
∃ evidence:
    issued_by_modeled_source(evidence)
    ∧ matches(evidence, candidate)
    ∧ supports(evidence.kind, STOCK_REPLENISHED)
    ∧ source_authorized_for(
          evidence.issuer,
          STOCK_REPLENISHED
      )
```

The implementation keeps these dimensions separate and does not assemble
partial authority from different evidence records. Exact V1 correlation
compares `candidate_id`, `product_id`, and `quantity`.

This separation matters because an evidence record can be validly issued and
exactly correlated while remaining semantically insufficient. The limited
actor's `AGENT_RESTOCK_REQUEST` evidence matches the candidate exactly, but its
kind supports only `RESTOCK_REQUESTED`, and its issuer is authorized only for
that proposition. It therefore cannot establish `STOCK_REPLENISHED`.

`SemanticAuthorityAdmission` is a domain-specific boundary in this demo. It is
not a generic path-authorization algorithm, IAM system, RBAC layer, policy
language, production persistence gate, or proof of truth.

## 5. Positive control

The positive control tests whether semantic admission merely blocks every
agent-triggered workflow. It does not.

The case begins with the same limited-agent origin, the same allowed workflow
edges, and the same candidate as the governed rejection case. It then adds a
separately represented `WarehouseReceiptObservation` and asks the modeled
`WarehouseAuthority` to issue receipt-confirmation evidence.

The evidence basis remains explicit:

| Evidence field | Basis |
|---|---|
| `candidate_id` | Candidate correlation identity |
| `receipt_id` | Warehouse receipt observation |
| `product_id` | Warehouse receipt observation |
| `quantity` | Warehouse receipt observation |
| `kind` | Fixed warehouse-confirmation kind |
| `issuer` | Fixed warehouse-authority issuer |

That evidence matches the exact candidate, supports `STOCK_REPLENISHED`, and
comes from a modeled source authorized for that proposition. Admission returns
`ACCEPT`; the service appends the fact; inventory becomes `10`.

The distinction is:

```text
causally triggered by the request
!=
authority derived from the request
```

A request may cause an authority-owned observation to be consulted. What
matters in this model is that the factual basis for product and quantity comes
from the separate authority-owned observation, not solely from the request,
candidate, or fact that the workflow completed.

This is a modeled positive control, not proof of real warehouse truth. The
observation is an in-memory value, and the source-owned issuance mechanism is a
modeling device rather than cryptographic provenance or hostile-process
isolation.

## 6. Current research progress

V1 contains a complete deterministic comparison:

```text
✓ direct denial
✓ vulnerable indirect composition
✓ governed rejection
✓ independently grounded modeled positive control
✓ executable tests
```

Each scenario uses a fresh store and targets the same protected effect:

| Case | Local workflow | Semantic evidence | Promotion | Accepted facts | Inventory |
|---|---|---|---|---:|---:|
| Direct denial | Not reached | N/A | Not reached | 0 | 0 |
| Vulnerable composition | All invoked edges allowed | Not checked | Occurs | 1 | 10 |
| Governed rejection | Same edges allowed | Correlated but insufficient | `REJECT` | 0 | 0 |
| Positive control | Same edges allowed | Modeled warehouse-grounded | `ACCEPT` | 1 | 10 |

The tests also cover the exact same direct and composed target fact, candidate
versus accepted-fact type separation, exact-correlation mismatches, modeled
issuance ownership, and coexistence of vulnerable and governed paths.

The model and demo intentionally are:

```text
single-process
synchronous
deterministic
in-memory
standard-library-only
```

They intentionally do not depend on:

```text
LLMs
network services
databases
credentials
concurrency
retry behavior
stochastic model behavior
```

Removing those variables isolates the authority-composition structure: the
unsafe result does not depend on model sampling, network timing, retries, or
distributed-state races. The existing test suite uses `pytest` as its runner;
that runner is not part of the model or demo runtime.

The problem was identified while developing Compass, but this standalone
repository has no Compass runtime dependency and should be evaluated on the
finite model and claims presented here.

## 7. What the demo establishes

For the exercised finite paths, the implementation supports these bounded
claims:

- A directly denied authoritative effect becomes reachable through
  individually permitted workflow edges in the deliberately vulnerable model.
- The effect is reached through a privileged service's legitimate local
  capability after unsupported semantic promotion, not because the limited
  actor receives that capability.
- Semantic authority admission rejects unsupported promotion without disabling
  request submission or candidate production.
- The same workflow is accepted when evidence correlated to the exact candidate
  is grounded in a separate modeled authority-owned observation.
- Evidence correlation and semantic authority are independently observable
  properties in the model.

These are existence and structural claims, not frequency estimates or a proof
about arbitrary systems.

## 8. What it does not establish

This repository does not claim or prove that it:

- reproduces OpenAI internal architecture;
- reproduces Hugging Face internal architecture;
- explains the mechanism of a particular external incident;
- exploits a real external system;
- proves real warehouse truth;
- provides cryptographic provenance, authentication, or hostile-process
  isolation;
- implements complete causal provenance, IAM, PKI, or a general authority
  policy;
- implements static or runtime path-level reachable-authority analysis;
- represents production Compass policy or integration;
- proves that Compass universally prevents authority escalation;
- exhaustively covers authority-composition failures; or
- quantifies real-world AI failure probability.

The positive control establishes sufficiency only under the model's fixed
propositions, closed issuer mappings, in-memory observation, and exact
correlation rule. Semantic admission here rejects and accepts the specific
cases exercised; it does not prove arbitrary facts true or composed workflows
universally safe.

## 9. Open research questions

The counterexample motivates questions that V1 does not answer:

- How should authority be represented across composed agent and service
  workflows?
- Can path-level authority be derived compositionally from local permissions,
  or must it be analyzed as a separate property?
- What evidence is sufficient for a particular semantic promotion, and how
  should proposition strength be represented?
- How should evidence and authority provenance survive delegation across agents
  or services?
- Can reachable authoritative effects be analyzed statically or before
  runtime?
- Where should semantic admission live in a practical architecture, and which
  component owns its policy?
- How should independent authority be authenticated across real trust
  boundaries?
- How does the model change under asynchronous execution, concurrency, retries,
  distributed state, partial failure, or adversarial components?
- How should time-sensitive evidence be revalidated at commit boundaries?

For the broader conceptual derivation, continue to
[Shared Workflow Is Not Shared Authority](shared_workflow_is_not_shared_authority.md).
For implementation details and exact experimental boundaries, then see the
[executable demo documentation](docs/demo.md).
