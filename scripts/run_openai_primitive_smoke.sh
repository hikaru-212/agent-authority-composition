#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {control|treatment} openai/<model>\n' "${0##*/}" >&2
}

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
  usage
  exit 64
fi

CONDITION="$1"
MODEL="$2"

case "$CONDITION" in
  control)
    TASK="evals/inspect/primitive_behavioral_eval.py@primitive_behavior_control_eval"
    ;;
  treatment)
    TASK="evals/inspect/primitive_behavioral_eval.py@primitive_behavior_treatment_eval"
    ;;
  *)
    printf 'Error: condition must be control or treatment.\n' >&2
    usage
    exit 64
    ;;
esac

# Reject lists, whitespace, and arbitrary provider/task specifications.
if [[ ! "$MODEL" =~ ^openai/[A-Za-z0-9][A-Za-z0-9._:-]*$ ]]; then
  printf 'Error: model must be one explicit openai/<model> identifier.\n' >&2
  usage
  exit 64
fi

# Resolve tasks and ignored logs relative to this repository.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v inspect >/dev/null 2>&1; then
  printf 'Error: inspect is not available on PATH.\n' >&2
  exit 1
fi

if ! python -c 'import openai' >/dev/null 2>&1; then
  printf 'Error: Python cannot import the openai package.\n' >&2
  exit 1
fi

inspect eval "$TASK" \
  --model "$MODEL" \
  --limit 1 \
  --epochs 1 \
  --no-epochs-reducer \
  --max-retries 0 \
  --retry-on-error 0 \
  --log-dir logs \
  --log-format eval \
  --display plain
