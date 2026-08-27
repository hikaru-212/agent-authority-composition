#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {control|composition} openai/<model> <epochs:1-20>\n' "${0##*/}" >&2
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
    TASK="evals/inspect/behavioral_eval.py@inventory_behavior_control_eval"
    ;;
  composition)
    TASK="evals/inspect/behavioral_eval.py@inventory_behavior_composition_eval"
    ;;
  *)
    printf 'Error: condition must be control or composition.\n' >&2
    usage
    exit 64
    ;;
esac

if [[ "$MODEL" != openai/* || "$MODEL" == "openai/" ]]; then
  printf 'Error: model must begin with openai/ and include a model name.\n' >&2
  usage
  exit 64
fi

if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] ||
   [[ ${#EPOCHS} -gt 2 ]] ||
   (( 10#$EPOCHS < 1 || 10#$EPOCHS > 20 )); then
  printf 'Error: epochs must be an integer from 1 through 20.\n' >&2
  usage
  exit 64
fi

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
  --epochs "$EPOCHS" \
  --no-epochs-reducer \
  --display plain
