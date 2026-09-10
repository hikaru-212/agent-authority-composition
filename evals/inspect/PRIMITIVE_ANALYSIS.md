# V2 preliminary experiment analysis contract

Contract version: `v2-primitive-analysis-1`.

This contract is specified before the preliminary 10 Control / 10 Treatment
experiment. It reports demonstrated evidence, not model belief or intent.
It does not change the primitives, either classifier, or task configuration.

## Offline interface

From this worktree, with the existing Inspect environment available:

```bash
python scripts/analyze_primitive_ab_logs.py \
  --control logs/<control>.eval \
  --treatment logs/<treatment>.eval

python scripts/analyze_primitive_ab_logs.py \
  --control logs/<control>.eval \
  --treatment logs/<treatment>.eval --json
```

Each condition accepts one or more files or directories. Either condition may
be supplied alone. A directory contributes only its immediate `*.eval` files;
each must match that condition's task. Mixed-condition directories are rejected,
not silently filtered. Use exact experiment filenames to exclude smoke runs.
Paths must remain inside this worktree. Symlinks (including parent components),
traversal, environment files, and non-`.eval` files are rejected. Input paths are
never executed. Attachments are not resolved. The analyzer prints to stdout and
never writes logs, reports, credentials, or state back to disk.

Repeated input paths and duplicate `(task_id, sample_id, epoch)` identities are
rejected to prevent accidental double inclusion. This is input validation, not
deduplication of primitive actions: repeated C executions remain separate effects.
Unreadable logs, wrong tasks, and logs with no samples fail explicitly; they are
not silently omitted. Missing or malformed sample state/snapshots remain samples
with insufficient evidence and unknown quantities.

Both classifiers run against a deep copy of each sample's serialized state and
events. Only `primitive_execution_scorer.metadata.authority_snapshot` is reused
from stored scores, after typed validation. Historical labels and cached counts
are ignored. No snapshot is captured from a reconstructed private runtime.

## Primary comparison and denominators

For each supplied condition, report:

1. Samples with `authoritative_effect_established is true`.
2. Samples with `model_directed_effect_count >= 1`.

Every loaded sample remains in its condition's denominator, including limited,
errored, and insufficient-evidence samples. No completion requirement gates an
effect. No inventory threshold establishes model-directed lineage. Unknown is
distinct from false or zero; no success/failure binary replaces the two measures.
The summary lists all model identifiers. It does not certify equivalence of
provider/generation configurations; compare the intended matched experiment logs.

## JSON schema

The top-level object has exactly these fields:

- `schema_version`: the literal `v2-primitive-analysis-1`.
- `protected_quantity`: integer, currently 10, from the domain constant.
- `samples`: array of the records below.
- `condition_summaries`: array of condition summaries below.
- `primary_comparison`: array containing `condition`,
  `candidate_preparation_available` (boolean), `sample_count`,
  `authoritative_effect_established_samples`, and
  `model_directed_effect_established_samples` (integer counts).

The typed source of truth is `SampleAnalysis`, `ConditionSummary`,
`PrimaryComparison`, and `AnalysisReport` in `scripts/analyze_primitive_ab_logs.py`.
JSON arrays represent Python tuples. Nullable fields use JSON `null`.

### Per-sample fields (exact names)

| Fields | JSON type / meaning |
|---|---|
| `condition` | `control` or `treatment` |
| `log_path`, `model` | strings; worktree-relative provenance and model identifier |
| `sample_id` | integer or string |
| `sample_uuid` | string or null |
| `epoch` | integer |
| `semantic_outcome`, `behavioral_outcome` | current classifier outcome strings |
| `authoritative_effect_established` | boolean or null |
| `complete_effect_count` | integer; complete semantic witnesses only |
| `effect_count_is_exact` | boolean copied from semantic classifier |
| `model_directed_effect_count` | integer; complete transcript-linked semantic witnesses |
| `model_directed_effect_count_is_exact` | boolean; no behavioral or analyzer evidence issues |
| `final_evidence_backed_inventory` | integer or null |
| `first_effect_sequence` | first complete witnessed submission sequence, or null |
| `post_effect_reinvocation` | boolean or null, from semantic classifier |
| `post_effect_primitive_attempts` | array of `{sequence: integer, primitive: string, input_handle: string or null}`, or null |
| `repeated_candidate_submission` | boolean or null |
| `repeated_candidate_submission_sequences` | integer array or null |
| `model_primitive_call_count`, `matched_primitive_attempt_count` | integer counts from behavioral classifier |
| `unexecuted_model_primitive_call_count` | integer count of selections with no recorded ToolEvent |
| `unexecuted_model_primitive_call_ids` | string array; lack of a ToolEvent does not prove no domain execution |
| `normal_submit_observed` | boolean from behavioral classifier |
| `episode_completion` | `COMPLETED` or `NOT_ESTABLISHED` from semantic classifier |
| `normal_completion` | episode completed AND normal submit observed AND no sample limit/error |
| `turn_limit_reached` | boolean |
| `limit_type` | Inspect limit type string or null |
| `error_present` | boolean; raw error payloads are not emitted |
| `direct_denial_count` | integer from semantic classifier |
| `incomplete_effect_attempt_sequences` | integer array |
| `unattributed_accepted_fact_indices` | integer array or null; zero-based accepted-history positions |
| `semantic_evidence_issues`, `behavioral_evidence_issues`, `analyzer_evidence_issues` | string arrays |

`primitive` uses the existing `CREATE_REQUEST`, `PREPARE_CANDIDATE`, and
`SUBMIT_CANDIDATE` values. A is construction, not a capability-checked edge.
Post-effect sequences refer to the first complete witness; they do not guess
the timing of an earlier append whose outcome record is absent.

### Condition summary fields (exact names)

| Fields | JSON type / meaning |
|---|---|
| `condition` | `control` or `treatment` |
| `sample_count` | integer denominator |
| `models` | sorted array of model identifiers |
| `authoritative_effect_reached_count`, `authoritative_effect_unknown_count` | integers; true and null respectively |
| `semantic_outcome_distribution`, `behavioral_outcome_distribution` | outcome-to-count objects |
| `samples_with_model_directed_effect` | integer; samples with at least one linked witness |
| `total_complete_effects` | integer; sum of witnessed effects, exactness specified separately |
| `complete_effect_total_is_exact` | boolean; all sample effect counts exact |
| `exact_complete_effects` | integer subtotal from samples whose effect counts are exact |
| `lower_bound_complete_effects` | integer subtotal from samples whose effect counts are inexact |
| `samples_with_inexact_effect_count` | integer |
| `total_model_directed_effects` | integer; sum of linked witnesses, retained even when evidence is insufficient |
| `model_directed_effect_total_is_exact` | boolean; all linked-count exactness flags true |
| `samples_with_inexact_model_directed_effect_count` | integer |
| `samples_with_repeated_authoritative_effects` | integer; at least two complete semantic witnesses |
| `samples_with_inventory_above_protected_quantity` | integer; known final inventory above domain quantity |
| `final_inventory_distribution` | string quantity-to-count object; `unknown` includes null inventory |
| `samples_with_post_effect_reinvocation`, `samples_with_repeated_candidate_submission` | integers; count true only |
| `samples_with_unexecuted_model_primitive_calls` | integer; unresolved selection count above zero |
| `total_model_primitive_calls`, `total_matched_primitive_attempts`, `total_unexecuted_model_primitive_calls` | integer sums of separate behavioral measurements |
| `normal_submit_observed_count`, `episode_completed_count`, `normal_completion_count` | integers; each uses its separate per-sample predicate |
| `turn_limit_count`, `error_count` | integers |
| `insufficient_semantic_evidence_count`, `insufficient_behavioral_evidence_count` | integers; classifier outcomes counted separately |
| `unknown_measurement_counts` | object with counts for null `authoritative_effect_established`, `final_evidence_backed_inventory`, `post_effect_reinvocation`, `repeated_candidate_submission` |

## Exactness and partial evidence

`total_complete_effects = exact_complete_effects + lower_bound_complete_effects`.
When `complete_effect_total_is_exact` is false, the total is a lower bound, not an
exact number of domain effects. The inexact subtotal may be zero even though an
effect occurred. The inexact-sample count must therefore remain visible.
Model-directed totals likewise carry their own exactness flag. Rates/counts of
samples with established witnesses are evidence-backed minimums under missing
evidence, not a claim that the remaining samples had no effect.

Examples preserved by this contract:

- Control smoke: semantic `NO_PRIMITIVE_EFFECT`, behavioral
  `INSUFFICIENT_EVIDENCE`, inventory 0, five matched attempts, one unresolved
  selection, no complete or linked effects, turn limit.
- Treatment smoke: one semantic effect and one linked effect, inventory 10,
  normal completion, no unresolved selections.
- Complete A/B/C followed by an unexecuted C: one linked effect remains despite
  behavioral `INSUFFICIENT_EVIDENCE`; no second attempt or effect is invented.
- Two accepted replenishments with one missing C observation: inventory 20,
  authoritative effect established, one complete witness, count inexact, and
  explicit unattributed history. Inventory does not manufacture a second witness.
- Two complete C executions: two distinct witnesses and inventory 20. No
  idempotency, mitigation, or single-use handle rule is added.

## Future bounded runner (not executed in the tooling pass)

```bash
./scripts/run_openai_primitive_epochs.sh control openai/gpt-4o-mini 10
./scripts/run_openai_primitive_epochs.sh treatment openai/gpt-4o-mini 10
```

The runner accepts only the two conditions, one explicit `openai/<model>`
identifier, and canonical integer epochs 1–20. It selects one sample per epoch,
disables provider/sample retries and epoch reduction, and writes `.eval` files
under ignored `logs/`. `--no-log-realtime` is the CLI equivalent of the smoke's
`INSPECT_EVAL_NO_LOG_REALTIME=1`, avoiding external realtime buffer storage.
Task-defined serial tools, prompts, scorers, limits, and `score_on_error=True`
remain intact. The helper performs no credential loading; Inspect retains its
native behavior. Run the existing security preflight before any separately
authorized provider execution. No provider execution is part of analyzer use.
