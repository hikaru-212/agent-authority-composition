# Inspect behavioral harness

`behavioral_eval.py` is a provider-independent ReAct task. It gives the selected
model three bounded inventory tools and records the result with the deterministic
authority evaluator. It does not select or configure a model provider itself.

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

Then run exactly one behavioral evaluation, supplying the non-secret OpenAI
model name as the argument:

```bash
./scripts/run_openai_smoke.sh "openai/<model>"
```

Review the result afterward:

```bash
inspect view
```

`.env.example` documents variable names only. OpenAI is the first documented
provider, but the behavioral task itself does not select or configure a
provider.

The mocked tests use scripted outputs only to verify plumbing. They do not claim
that a mock model autonomously discovered an inventory workflow.
