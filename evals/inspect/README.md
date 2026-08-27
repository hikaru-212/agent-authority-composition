# Inspect behavioral harness

`behavioral_eval.py` contains two provider-independent ReAct tasks for a
controlled tool-access comparison. Both use the same prompt, setup, agent
instructions, scorer, completion wrapper, limits, generation configuration,
and one-sample dataset:

- `inventory_behavior_control_eval` exposes `get_inventory` and
  `update_inventory`.
- `inventory_behavior_composition_eval` additionally exposes
  `request_restock`.

ReAct `submit` is available in both. The older `inventory_behavior_eval` name
remains a compatibility alias for `inventory_behavior_composition_eval`.
Neither task selects or configures a model provider itself.

## Security-first OpenAI smoke run

Create a project-scoped API key in the provider dashboard. Prefer a dedicated
development/evaluation project and key, and set a reasonable project budget or
spend limit. Never paste the key into source code, Git, prompts, tool arguments,
or logs.

Use `.env.example` as a names-only reference and create a local `.env`:

```bash
cp .env.example .env
```

Put `OPENAI_API_KEY` and, optionally, `OPENAI_PROJECT_ID` in that local file.
Never commit `.env`; Git ignores it. Inspect loads `.env` natively, so the smoke
helper does not read, source, copy, print, or validate credentials.

Run the repository hygiene preflight:

```bash
./scripts/security_preflight.sh
```

Then run exactly one behavioral evaluation per condition, supplying the
non-secret OpenAI model name at invocation:

```bash
./scripts/run_openai_smoke.sh control "openai/<model>"
./scripts/run_openai_smoke.sh composition "openai/<model>"
```

The condition argument is allowlisted and maps only to the two task names
above; it cannot select an arbitrary file or task. Each command runs the
single sample defined by its selected task.

## Preliminary bounded epochs

For preliminary per-epoch inspection, the bounded epochs helper accepts only
the two known conditions and an integer from 1 through 20. It leaves epoch
reduction disabled so each outcome remains directly inspectable:

```bash
./scripts/run_openai_epochs.sh control openai/gpt-4o-mini 10
./scripts/run_openai_epochs.sh composition openai/gpt-4o-mini 10
```

The helper delegates credential loading to Inspect and does not accept, read,
or print credentials. Apart from the selected known task and requested epoch
count, it does not override scaffold or model settings.

Analyze one completed Control log and one Composition log without contacting a
provider or reading transcript prose:

```bash
python scripts/analyze_ab_logs.py \
  --control logs/<control>.eval \
  --composition logs/<composition>.eval
```

The report keeps agent termination separate from the protected system effect.
In particular, complete typed composition evidence can establish an append and
inventory transition before a turn limit even when the official lifecycle-aware
score remains `INSUFFICIENT_EVIDENCE`. For effect-reached Composition epochs,
it also reports only the later tool-event trajectory and counts, including
post-effect inventory reads, repeated restock or update actions, and normal
submission. These are behavioral observations; the report does not infer model
belief or intent. It also reports the final consistent evidence-backed
inventory and the number of distinct complete composition effects per epoch,
including a Composition distribution and cumulative-side-effect counts. Tool
call counts and transcript inventory text are not used for that state summary.

Review the result afterward:

```bash
inspect view
```

`.env.example` documents variable names only. OpenAI is the first documented
provider, but the behavioral task itself does not select or configure a
provider.

The mocked tests use scripted outputs only to verify plumbing. They do not claim
that a mock model autonomously discovered an inventory workflow.
