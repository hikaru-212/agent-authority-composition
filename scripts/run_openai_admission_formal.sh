#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec python -I -S -B scripts/run_openai_admission_formal.py "$@"
