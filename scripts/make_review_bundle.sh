#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
TMP="/tmp/codex-review-$STAMP"
OUT="$ROOT/codex-review-$STAMP.zip"

mkdir -p "$TMP/files"

git status --short > "$TMP/git-status.txt"
git diff --stat HEAD > "$TMP/diff-stat.txt"
git diff --binary HEAD > "$TMP/changes.patch"
git log --oneline --decorate -10 > "$TMP/git-log.txt"

{
  git diff --name-only HEAD
  git ls-files --others --exclude-standard
} | sort -u > "$TMP/file-list.txt"

if [[ -s "$TMP/file-list.txt" ]]; then
  rsync -R --files-from="$TMP/file-list.txt" ./ "$TMP/files/"
fi

(
  cd /tmp
  zip -qr "$OUT" "codex-review-$STAMP"
)

rm -rf "$TMP"

echo "Created:"
echo "$OUT"
