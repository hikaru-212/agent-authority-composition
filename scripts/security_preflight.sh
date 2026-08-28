#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf 'Error: security preflight must run inside a Git repository.\n' >&2
  exit 1
}
cd "$ROOT"

failed=0

warn_file() {
  printf 'Potential credential material detected in: %s\n' "$1" >&2
  printf 'Warning: review and remove the potential credential before continuing.\n' >&2
  failed=1
}

if ! git check-ignore -q -- .env; then
  printf 'Warning: .env is not ignored by Git.\n' >&2
  failed=1
fi

if git ls-files --error-unmatch -- .env >/dev/null 2>&1; then
  printf 'Warning: .env is tracked by Git.\n' >&2
  failed=1
fi

while IFS= read -r tracked_zip; do
  [[ -z "$tracked_zip" ]] && continue
  printf 'Warning: review archive is tracked: %s\n' "$tracked_zip" >&2
  failed=1
done < <(git ls-files 'codex-review-*.zip')

has_non_placeholder_openai_assignment() {
  awk '
    BEGIN { found = 0 }
    {
      value = $0
      original = value
      sub(/^[[:space:]]*(export[[:space:]]+)?OPENAI_API_KEY[[:space:]]*=[[:space:]]*/, "", value)
      if (value != original) {
        sub(/[[:space:]]*(#[^\r\n]*)?$/, "", value)
        if (value != "" && value != "\"\"" && value != "\047\047" &&
            value !~ /^[\"\047]?<[^>]+>[\"\047]?$/) {
          found = 1
          exit
        }
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$1"
}

while IFS= read -r -d '' path; do
  if [[ -L "$path" || ! -r "$path" ]]; then
    printf 'Warning: security preflight cannot safely scan: %s\n' "$path" >&2
    failed=1
    continue
  fi

  suspicious=0
  if LC_ALL=C grep -aEq 'sk-[A-Za-z0-9_-]{16,}' -- "$path"; then
    suspicious=1
  elif LC_ALL=C grep -aEiq 'authorization[[:space:]]*:[[:space:]]*bearer[[:space:]]+[A-Za-z0-9._~+/-]{12,}' -- "$path"; then
    suspicious=1
  elif has_non_placeholder_openai_assignment "$path"; then
    suspicious=1
  fi

  if [[ "$suspicious" -eq 1 ]]; then
    warn_file "$path"
  fi
done < <(git ls-files -co --exclude-standard -z)

if [[ "$failed" -ne 0 ]]; then
  printf 'Security preflight failed. No matching values were printed.\n' >&2
  exit 1
fi

printf 'Security preflight passed.\n'
printf 'Note: this is a repository hygiene check, not a complete secret scanner.\n'
