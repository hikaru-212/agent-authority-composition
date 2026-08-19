# Indirect Authority Escalation Demo

## Status

```text
branch
= experiment/indirect-authority-escalation-demo

V1 implementation
= complete

execution
= deterministic, local, in memory
```

The current source contains the complete comparison from direct denial through
the vulnerable composition failure, governed rejection, and warehouse-grounded
positive control. A reviewer does not need to check out earlier experiment
commits.

This is a bounded Compass-inspired semantic authority-promotion experiment. It
does not add production Compass policy or integration.

## Research Question

Can an actor reach the same authoritative effect through individually allowed
workflow operations when its direct attempt is denied?

```text
direct permission denied
!=
authoritative effect unreachable
```

The deterministic counterexample is:

```text
LIMITED_AGENT
→ RestockWorkflow                     ALLOWED
→ InventoryAuthorityService           ALLOWED
→ AuthoritativeInventoryStore         ALLOWED
→ AcceptedStockReplenished(product-a, 10)
```

while the direct local capability remains:

```text
LIMITED_AGENT
→ AuthoritativeInventoryStore
→ APPEND_ACCEPTED_REPLENISHMENT
= DENIED
```

## Why the Direct Permission Check Is Not Enough

Case 1 proves that the store's local permission boundary exists and works. A
direct append by `LIMITED_AGENT` is denied and accepted history remains empty.

Case 2 then uses only allowed operations. `RestockWorkflow` produces a
`CandidateStockReplenished`; the deliberately vulnerable authority-service
method treats arrival through that workflow as sufficient for promotion and
uses its legitimate store capability. Every invoked local edge is valid, yet
the same protected effect denied in Case 1 enters accepted history.

The failure is therefore not a missing store ACL or a bypassed denied edge. It
is invalid semantic promotion across an allowed path:

```text
∀ edge in path: locally authorized(edge)

does not imply

globally authorized promotion(path)
```

The demo keeps these distinctions visible:

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

## Vulnerable Versus Governed Flow

Both paths remain executable in the current model.

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

The existing workflow permissions do not change. With exactly correlated
`AGENT_RESTOCK_REQUEST` evidence, the request and candidate-submission edges
remain allowed, but promotion is rejected because that evidence supports only
`RESTOCK_REQUESTED`, not `STOCK_REPLENISHED`.

Authoritative state is membership in
`AuthoritativeInventoryStore.accepted_facts`. Merely constructing a request,
candidate, evidence value, or `AcceptedStockReplenished` Python value does not
establish authoritative history. Inventory is a deterministic fold over the
accepted fact sequence.

## Semantic Promotion Invariant

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
preserved as separate observable checks. `SemanticAuthorityAdmission` is a
demo-local semantic boundary, not production persistence admission, generic
RBAC, IAM, or a policy DSL.

## Warehouse Authority Basis

The positive control does not relabel candidate data as warehouse evidence. It
uses two independently represented values:

```text
CandidateStockReplenished
= what the workflow proposes

WarehouseReceiptObservation
= what the modeled warehouse authority observed
```

`WarehouseAuthority.issue_receipt_confirmation(...)` receives both objects.
The resulting evidence takes:

| Evidence field | Source |
|---|---|
| `candidate_id` | `CandidateStockReplenished` correlation identity |
| `receipt_id` | `WarehouseReceiptObservation` |
| `product_id` | `WarehouseReceiptObservation` |
| `quantity` | `WarehouseReceiptObservation` |
| `kind` | Fixed as `WAREHOUSE_RECEIPT_CONFIRMATION` |
| `issuer` | Fixed as `WAREHOUSE_AUTHORITY` |

The original request may causally trigger a warehouse check. Authority does not
require the warehouse to be absent from the causal path:

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

## Four-Case Comparison

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

This progression is reproduced from the current source:

```text
DIRECT DENIAL
    ↓
LOCALLY VALID INDIRECT FAILURE
    ↓
SEMANTICALLY GOVERNED REJECTION
    ↓
AUTHORIZED POSITIVE CONTROL
```

## How to Run

From the repository root, with the project Python environment activated:

```bash
python -m experiments.indirect_authority_escalation.demo
```

The runner uses four fresh in-memory stores, calls the existing structured
model, and prints the complete comparison. It contains no duplicated authority
or admission policy.

## How to Run Focused Tests

```bash
python -m pytest tests/experiments/indirect_authority_escalation/ -q
```

The focused suite covers direct denial, the vulnerable composed path, governed
rejection, warehouse-grounded acceptance, issuance ownership, exact-correlation
mismatches, and the reviewer-facing orchestration.

## Implementation Map

```text
experiments/indirect_authority_escalation/
├── README.md
├── demo.py
└── model.py

tests/experiments/indirect_authority_escalation/
├── test_demo.py
├── test_model.py
└── test_semantic_authority_admission.py
```

The implementation is deliberately single-process, synchronous, deterministic,
in-memory, and standard-library-only. It has no LLM, network, external service,
database, credentials, concurrency, retry, stochastic simulation, or sandbox
behavior.

## What This Demo Proves

The executable tests support these bounded claims:

```text
A deterministic local model demonstrates an authority-composition failure
where a directly denied authoritative effect becomes reachable through
individually permitted workflow edges.
```

```text
A semantic authority-promotion boundary rejects unsupported promotion while
leaving the legitimate workflow available.
```

```text
The same agent-originated workflow can be accepted when exact-candidate
evidence is grounded in a modeled authority-owned warehouse observation.
```

These are structural claims about the finite paths exercised by this local
model. V1 demonstrates that the unsafe authority path exists; it does not
estimate how frequently such a path would occur.

## What This Demo Does Not Prove

The demo does not claim or prove:

```text
this reproduces OpenAI internal architecture
this reproduces Hugging Face internal architecture
a specific external incident used this exact mechanism
this exploits a real external system
WarehouseReceiptObservation proves real-world warehouse truth
the modeled issuance token is cryptographic security
modeled evidence has real external provenance
production Compass already implements this inventory policy
Compass universally prevents authority escalation
the demo quantifies real AI failure probability
```

It also does not provide production hostile-process isolation, component
identity, complete causal provenance, IAM, PKI, cryptographic authentication,
warehouse infrastructure, or exhaustive authority-escalation coverage.
