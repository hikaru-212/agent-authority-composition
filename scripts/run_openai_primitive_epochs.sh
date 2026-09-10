#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {control|treatment} openai/<model> <epochs:1-20>\n' "${0##*/}" >&2
}

if [[ $# -ne 3 || -z "$1" || -z "$2" || -z "$3" ]]; then
  usage
  exit 64
fi

CONDITION="$1"
MODEL="$2"
EPOCHS="$3"

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

if [[ ! "$MODEL" =~ ^openai/[A-Za-z0-9][A-Za-z0-9._:-]*$ ]]; then
  printf 'Error: model must be one explicit openai/<model> identifier.\n' >&2
  usage
  exit 64
fi

if [[ ! "$EPOCHS" =~ ^([1-9]|1[0-9]|20)$ ]]; then
  printf 'Error: epochs must be an integer from 1 through 20.\n' >&2
  usage
  exit 64
fi

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
  --epochs "$EPOCHS" \
  --no-epochs-reducer \
  --max-retries 0 \
  --retry-on-error 0 \
  --no-log-realtime \
  --log-dir logs \
  --log-format eval \
  --display plain
