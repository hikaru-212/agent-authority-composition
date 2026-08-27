#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s openai/<model>\n' "${0##*/}" >&2
}

if [[ $# -ne 1 || -z "$1" ]]; then
  usage
  exit 64
fi

MODEL="$1"

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

inspect eval evals/inspect/behavioral_eval.py \
  --model "$MODEL" \
  --display plain
