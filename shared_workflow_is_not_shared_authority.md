# Case Study: Shared Workflow Is Not Shared Authority

[← Back to Semantic Admission Index](README.md)

**Recorded on:** 2026-07-10

## Disclosure Level

Public concept note.

This note defines a semantic boundary for agentic commerce workflows. It is not an implementation specification.

> **Current relationship:** The commerce workflow below is illustrative, not a
> currently implemented Compass workflow. Stage 4B provides a durable
> `DecisionReceipt` foundation, but automatic receipt materialization,
> complete agent-action admission, and runtime policy remain future work.

---

## Purpose

This note records a failure mode in multi-agent commerce systems:

```text
An agent may not have direct write access to inventory,
but it may still indirectly cause false inventory state
through another agent-controlled workflow.
```

The core principle is:

```text
Shared workflow is not shared authority.
```

A shared workflow may help agents coordinate tasks.

It does not prove that an agent-generated claim can become an accepted business fact.

---

## Context

A realistic agentic commerce workflow may include:

```text
User Intent
   ↓
Search Agent
   ↓
Inventory Agent
   ↓
Recommendation Agent
   ↓
Checkout Agent
   ↓
Payment Agent
   ↓
Order Service
```

A basic safety rule is that the Inventory Agent should only have read access.

It may report:

```text
Product A stock = 1
inventory_version = 42
observed_at = T1
```

It should not be able to modify inventory directly.

This is necessary, but not sufficient.

A multi-agent workflow may contain another route to inventory mutation.

For example, a Search Agent may not be able to update stock directly, but it may send a message to another agent that generates a restock event.

If that event is accepted without independent verification, the system may still produce false inventory.

---

## Failure Mode: Indirect Authority Escalation

Suppose a user says:

```text
I really want Product A. Please find a way to buy it.
```

The Search Agent checks inventory and finds:

```text
Product A stock = 0
```

A safe response would be:

```text
Product A is currently out of stock.
```

But an unsafe workflow may continue:

```text
Search Agent
   ↓
Event Generation Agent
   ↓
Restock Agent
   ↓
Inventory update
   ↓
Checkout succeeds
```

If the Event Generation Agent creates a restock event without warehouse evidence, and the Restock Agent treats it as truth, then inventory may increase even though the warehouse has no stock.

The agent did not directly update inventory.

But the workflow still produced a false authority fact.

---

## Corrected Understanding

The issue is not only whether one agent has write permission.

The deeper issue is whether any agent-controlled path can produce an accepted authority fact.

Unsafe principle:

```text
The Search Agent has no inventory write permission, so the system is safe.
```

Corrected principle:

```text
No agent-controlled path should produce accepted inventory facts without independent authority verification.
```

A user demand signal may justify:

```text
RestockRequestCandidate
NotifyWhenAvailableRequest
UserDemandSignalObserved
```

It does not justify:

```text
StockReplenished
InventoryAdjusted
InventoryAvailable
```

---

## Candidate Event Is Not Authority Event

The event type matters.

An agent may be allowed to produce:

```text
RestockRequestCandidate(
  product_id = Product A,
  reason = "user demand"
)
```

But it should not be allowed to produce:

```text
StockReplenished(
  product_id = Product A,
  quantity = 1
)
```

The first means:

```text
A restock was requested.
```

The second means:

```text
Stock was actually replenished.
```

These are different business facts.

A restock request may be useful.

It may trigger procurement, notification, or human review.

But it must not directly increase available inventory.

---

## Time Gap: Search-Time Evidence Is Not Commit-Time Truth

This case also contains a timing problem.

An Inventory Agent may correctly observe:

```text
Product A stock = 1 at T1
```

But at checkout time, the state may have changed:

```text
Product A stock = 0 at T2
```

The original observation was true.

But it is not commit-time truth.

Therefore:

```text
search-time evidence ≠ commit-time truth
recommendation ≠ reservation
checkout candidate ≠ accepted order
```

A checkout step must revalidate inventory at commit time.

---

## Accepted Model

A safer commerce workflow separates observation, candidate action, and accepted fact.
The roles below are illustrative responsibilities, not a final authority
matrix, policy engine, service schema, or implemented deployment topology.

```text
Search Agent:
  may search products
  may ask for inventory observations
  may create demand signals or restock request candidates

Inventory Agent:
  may read inventory
  may return versioned observations
  may not mutate stock

Event Generation Agent:
  may generate candidate events
  may not generate accepted authority facts

Restock Agent:
  may process restock candidates
  may not treat arbitrary agent-generated events as inventory truth

Inventory Authority Service:
  may accept stock changes only with independent evidence

Order Commit Service:
  may accept orders only after commit-time validation
```

The key separation is:

```text
Agent writes candidate.
Authority service writes accepted fact.
```

---

## Why This Matters

Without this boundary, a multi-agent workflow can launder authority.

A weak signal such as:

```text
The user really wants this item.
```

may be transformed into:

```text
The item has been restocked.
```

This is unsafe.

The workflow has converted demand into inventory.

That is not coordination.

It is authority failure.

---

## Non-Goals

This note does not define:

```text
complete event schema
complete authority matrix
complete runtime policy engine
complete implementation API
final Compass workflow integration
```

It only records the architectural boundary.

---

## Relationship to Existing Compass Principles

This note extends:

```text
candidate action is not accepted fact
shared context is not shared contract
```

into a workflow-level principle:

```text
shared workflow is not shared authority
```

Agents may coordinate through shared context and workflows.

But their claims and generated events still require semantic admission before they become accepted business facts.

---

## Reusable Rules

```text
Read-only access is necessary but not sufficient.

Search-time evidence is not commit-time truth.

User demand is not warehouse evidence.

Restock request is not stock replenishment.

Agent-generated event is not authority fact.

Shared workflow is not shared authority.

No agent-controlled path should produce accepted authority facts without independent verification.
```

---

## Final Principle

```text
Agents may request restock,
but only independent authority may accept restock as fact.
```
