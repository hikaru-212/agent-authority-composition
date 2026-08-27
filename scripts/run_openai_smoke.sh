#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {control|composition} openai/<model>\n' "${0##*/}" >&2
}

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
  usage
  exit 64
fi

CONDITION="$1"
MODEL="$2"

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
  --display plain
