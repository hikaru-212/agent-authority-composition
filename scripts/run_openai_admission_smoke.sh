#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {vulnerable|unsupported|supported} openai/<model>\n' "${0##*/}" >&2
}

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
  usage
  exit 64
fi

CONDITION="$1"
MODEL="$2"

case "$CONDITION" in
  vulnerable)
    TASK="evals/inspect/admission_behavioral_eval.py@admission_behavior_vulnerable_eval"
    ;;
  unsupported)
    TASK="evals/inspect/admission_behavioral_eval.py@admission_behavior_unsupported_eval"
    ;;
  supported)
    TASK="evals/inspect/admission_behavioral_eval.py@admission_behavior_supported_eval"
    ;;
  *)
    printf 'Error: condition must be vulnerable, unsupported, or supported.\n' >&2
    usage
    exit 64
    ;;
esac

# Require one explicit OpenAI identifier; reject lists and arbitrary task paths.
if [[ ! "$MODEL" =~ ^openai/[A-Za-z0-9][A-Za-z0-9._:-]*$ ]]; then
  printf 'Error: model must be one explicit openai/<model> identifier.\n' >&2
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

printf 'Condition: %s\nModel: %s\n' "$CONDITION" "$MODEL"
# Inspect's plain display reports the resulting .eval path in its Log: line.
# Task-owned ReAct settings remain unchanged; this is one smoke sample only.
inspect eval "$TASK" \
  --model "$MODEL" \
  --limit 1 \
  --epochs 1 \
  --no-epochs-reducer \
  --max-retries 0 \
  --retry-on-error 0 \
  --no-log-realtime \
  --log-dir logs \
  --log-format eval \
  --display plain
