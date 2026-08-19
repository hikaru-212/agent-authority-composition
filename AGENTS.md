# AGENTS.md

## Repository Purpose

This repository is a standalone research package extracted from a larger Compass project.

Its purpose is to communicate and demonstrate the following research problem:

> An actor may be denied a protected operation directly, yet still reach the same authoritative effect through a composition of individually allowed workflow operations.

The repository is intended for research discussion and external review.

## Working Rules

Before making changes:

1. Read the relevant existing files completely.
2. Understand the current research claims before reorganizing or rewriting them.
3. Preserve the distinction between:
   - direct permission and reachable authority;
   - locally authorized edges and globally authorized paths;
   - candidate actions and accepted facts;
   - evidence correlation and evidence authority.

## Scope

You may reorganize documentation and improve reviewer-facing presentation when explicitly requested.

Do not silently change the underlying research claim, demo semantics, or test expectations.

The existing long-form concept note should be preserved as a deep-dive document. Do not remove it merely because shorter entry documents are introduced.

## Code and Tests

Unless explicitly requested:

- do not change demo behavior;
- do not change test semantics;
- do not add dependencies;
- do not introduce frameworks;
- do not add network, LLM, database, or external-service integration.

The current deterministic, local, in-memory model is intentional.

## Repository Safety

Operate only inside this repository.

Do not modify parent directories, sibling repositories, global configuration, credentials, or external resources.

Do not stage, commit, push, create branches, or modify remotes unless explicitly requested.

Do not delete original material unless explicitly requested.

## Research-Claim Discipline

Do not claim that this repository:

- reproduces the internal architecture of a real external AI system;
- proves that a specific external incident used this mechanism;
- demonstrates exploitation of a real external system;
- proves that Compass universally prevents authority escalation.

Keep demonstrated results separate from hypotheses, motivation, and future research questions.

## Preferred Reviewer Flow

When reorganizing the repository, optimize for this reading path:

```text
README
    ↓
Research brief
    ↓
Long-form concept note
    ↓
Executable demo and tests
```

The top-level README should be concise and help a new researcher understand the problem, significance, current result, and where to go next.