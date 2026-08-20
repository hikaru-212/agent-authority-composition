# Executable Authority-Composition Demonstration

[← Research landing page](../README.md)

## Purpose and scope

This document describes the repository's deterministic executable comparison
from direct denial through vulnerable composition, governed rejection, and a
warehouse-grounded positive control.

The model is a bounded, Compass-inspired authority-promotion experiment. It is
not production Compass policy or integration, and it does not model a real
external system.

## Research question

Can an actor reach the same authoritative effect through individually allowed
workflow operations when its direct attempt is denied?

```text
direct permission denied
!=
authoritative effect unreachable
```

The direct local capability is denied:

```text
LIMITED_AGENT
→ AuthoritativeInventoryStore
→ APPEND_ACCEPTED_REPLENISHMENT
= DENIED
```

The deliberately vulnerable composition reaches the same protected effect:

```text
LIMITED_AGENT
→ RestockWorkflow                     ALLOWED
→ InventoryAuthorityService           ALLOWED
→ AuthoritativeInventoryStore         ALLOWED
→ AcceptedStockReplenished(product-a, 10)
```

## Why the direct permission check is not enough

Case 1 establishes that the store's closed capability rule denies the
exercised direct append as specified. A direct append by `LIMITED_AGENT` is
denied and accepted history remains empty.

Case 2 then uses only allowed operations. `RestockWorkflow` produces a
`CandidateStockReplenished`; the deliberately vulnerable authority-service
method treats arrival through that workflow as sufficient for promotion and
uses its legitimate store capability. Every invoked local edge is valid, yet
the same protected effect denied in Case 1 enters accepted history.

The failure is invalid semantic promotion across an allowed path, not a
bypassed denied edge or a missing store ACL:

```text
∀ edge ∈ path:
    locally_authorized(edge)

does not imply

globally_authorized_promotion(path)
```

The model keeps these distinctions observable:

```text
direct permission
!=
reachable authority

edge-valid
!=
path-authorized

local capability authority
!=
semantic evidence authority

candidate produced
!=
fact established

evidence exists
!=
evidence sufficient

evidence correlation
!=
evidence authority
```

## Vulnerable and governed flows

Both paths remain executable in the model.

### Vulnerable path

```text
RestockRequest
→ CandidateStockReplenished
→ InventoryAuthorityService
→ promote_candidate_without_semantic_authority_admission(...)
→ AcceptedStockReplenished
→ AuthoritativeInventoryStore
```

The local request, candidate-submission, and privileged-append checks all
return `ALLOWED`. Semantic authority admission is absent, so the candidate is
promoted and inventory becomes `10`.

### Governed path

```text
RestockRequest
→ CandidateStockReplenished
→ SemanticAuthorityAdmission
→ AuthorityPromotionDecision
→ AcceptedStockReplenished, only after ACCEPT
→ AuthoritativeInventoryStore
```

The workflow permissions do not change. With exactly correlated
`AGENT_RESTOCK_REQUEST` evidence, the request and candidate-submission edges
remain allowed, but promotion is rejected because that evidence supports only
`RESTOCK_REQUESTED`, not `STOCK_REPLENISHED`.

Authoritative state is membership in
`AuthoritativeInventoryStore.accepted_facts`. Merely constructing a request,
candidate, evidence value, or `AcceptedStockReplenished` Python value does not
establish authoritative history. Inventory is a deterministic fold over the
accepted fact sequence.

## Semantic promotion invariant

For every candidate promoted to an accepted replenishment, at least one
evidence record must satisfy all of these conditions:

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

For V1, exact correlation compares:

```text
candidate_id
product_id
quantity
```

Evidence issuance, correlation, proposition support, and issuer authority are
separate checks. `SemanticAuthorityAdmission` is a demo-local semantic
boundary, not production persistence admission, generic RBAC, IAM, or a policy
language.

## Warehouse authority basis

The positive control does not relabel candidate data as warehouse evidence. It
uses two independently represented values:

```text
CandidateStockReplenished
= what the workflow proposes

WarehouseReceiptObservation
= what the modeled warehouse authority observed
```

`WarehouseAuthority.issue_receipt_confirmation(...)` receives both objects.
The resulting evidence takes its fields from these sources:

| Evidence field | Source |
|---|---|
| `candidate_id` | `CandidateStockReplenished` correlation identity |
| `receipt_id` | `WarehouseReceiptObservation` |
| `product_id` | `WarehouseReceiptObservation` |
| `quantity` | `WarehouseReceiptObservation` |
| `kind` | Fixed as `WAREHOUSE_RECEIPT_CONFIRMATION` |
| `issuer` | Fixed as `WAREHOUSE_AUTHORITY` |

The original request may causally trigger a warehouse check. Authority does
not require the warehouse to be absent from the causal path:

```text
causally triggered by the request
!=
authority derived from the request
```

The warehouse confirmation is authority-bearing in this model because its
factual product and quantity basis comes from the warehouse-owned receipt
observation rather than solely from the request, candidate, or workflow
completion. Exact correlation is still required; a receipt with the wrong
product, quantity, or candidate correlation identity is rejected.

## Four-case comparison

Each scenario begins with a fresh store and targets the same protected effect:

```text
AcceptedStockReplenished(product_id="product-a", quantity=10)
```

| Observation | Direct | Vulnerable indirect | Governed | Positive control |
|---|---:|---:|---:|---:|
| Limited-agent direct append | `DENIED` | `DENIED` | `DENIED` | `DENIED` |
| Existing workflow edges | N/A | `ALLOWED` | `ALLOWED` | `ALLOWED` |
| Candidate produced | `NO` | `YES` | `YES` | `YES` |
| Semantic authority admission | N/A | `NOT PRESENT` | `PRESENT` | `PRESENT` |
| Sufficient authority evidence | N/A | `UNCHECKED` | `NO` | `YES` |
| Promotion | Not reached | Occurs unchecked | `REJECT` | `ACCEPT` |
| Accepted facts | `0` | `1` | `0` | `1` |
| Inventory | `0` | `10` | `0` | `10` |
| Promotion invariant | `PRESERVED` | `VIOLATED` | `PRESERVED` | `PRESERVED` |

The complete progression is:

```text
DIRECT DENIAL
    ↓
LOCALLY VALID INDIRECT FAILURE
    ↓
SEMANTICALLY GOVERNED REJECTION
    ↓
AUTHORIZED POSITIVE CONTROL
```

## Run the demo

From the repository root, using Python 3.10 or later:

```bash
python demo.py
```

The runner uses four fresh in-memory stores, calls the structured model, and
prints the comparison. It contains no duplicated authority or admission
policy.

## Run the tests

The test suite uses `pytest`; run it in an environment where `pytest` is
already available:

```bash
pytest -q
```

The suite covers direct denial, the vulnerable composed path, governed
rejection, warehouse-grounded acceptance, issuance ownership,
exact-correlation mismatches, and reviewer-facing orchestration.

## Implementation map

```text
demo.py
= scenario orchestration and reviewer-facing terminal output

model.py
= local capability graph, candidate/fact types, semantic admission,
  modeled evidence issuance, and authoritative store

tests/
├── test_demo.py
│   = four-case orchestration and rendered-output checks
├── test_model.py
│   = direct-denial and vulnerable-composition counterexamples
└── test_semantic_authority_admission.py
    = governed rejection, positive control, issuance, and mismatch checks
```

The model and demo are deliberately single-process, synchronous,
deterministic, in-memory, and standard-library-only. They have no LLM,
network, external service, database, credentials, concurrency, retry,
stochastic simulation, or sandbox behavior. The test suite's only runner
requirement is `pytest`.

## Bounded claims

The executable cases support these claims:

- A deterministic local model exhibits an authority-composition failure in
  which a directly denied authoritative effect becomes reachable through
  individually permitted workflow edges.
- A semantic authority-promotion boundary rejects unsupported promotion while
  leaving the legitimate workflow available.
- The same agent-originated workflow is accepted when evidence correlated to
  the exact candidate is grounded in a modeled authority-owned warehouse
  observation.

These are structural claims about the finite paths exercised by this local
model. V1 demonstrates that the unsafe path exists; it does not estimate how
frequently such a path would occur.

## Non-claims and experimental limits

The demo does not claim or prove that it:

- reproduces OpenAI or Hugging Face internal architecture;
- shows that a particular external incident used this mechanism;
- exploits a real external system;
- proves real warehouse truth;
- provides cryptographic security or real external provenance;
- represents production Compass policy;
- proves that Compass universally prevents authority escalation; or
- quantifies real-world AI failure probability.

It also does not provide production hostile-process isolation, component
identity, complete causal provenance, IAM, PKI, cryptographic authentication,
warehouse infrastructure, or exhaustive authority-escalation coverage.
