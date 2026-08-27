# Inspect behavioral harness

`behavioral_eval.py` is a provider-independent ReAct task. It gives the selected
model three bounded inventory tools and records the result with the deterministic
authority evaluator. It does not select or configure a model provider itself.

## Future real-model run

1. Use an environment with Inspect AI and any dependency required by the chosen
   provider installed.
2. Export that provider's supported API-key variable in the invoking shell.
3. Select the provider and model with Inspect's `--model` option.
4. Review the resulting evaluator-private log with `inspect view`.

Generic command shape:

```bash
export <PROVIDER_API_KEY>="<set-in-shell>"
inspect eval evals/inspect/behavioral_eval.py --model <provider>/<model>
inspect view
```

OpenAI is one possible provider, not a project default or dependency:

```bash
export OPENAI_API_KEY="<set-in-shell>"
inspect eval evals/inspect/behavioral_eval.py --model openai/<model>
inspect view
```

Do not place credentials in prompts, tool arguments, metadata, source files, or
logs. The repository does not create or require a `.env` file.

The mocked tests use scripted outputs only to verify plumbing. They do not claim
that a mock model autonomously discovered an inventory workflow.
