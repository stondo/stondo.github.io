#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
content_dir="$repo_root/content/cairnkeep"
production_dir="$repo_root/production/cairnkeep-series"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

mapfile -t articles < <(
  find "$content_dir" -maxdepth 1 -type f \
    -name '[0-7][0-7]-*.md' -print | sort
)

[[ ${#articles[@]} -eq 8 ]] ||
  fail "expected 8 numbered articles, found ${#articles[@]}"

mapfile -t videos < <(
  find "$production_dir" -maxdepth 1 -type f \
    -name '[0-7][0-7]-*.md' -print | sort
)

[[ ${#videos[@]} -eq 8 ]] ||
  fail "expected 8 numbered video scripts, found ${#videos[@]}"

for article in "${articles[@]}"; do
  grep -q '^draft: true$' "$article" ||
    fail "article must remain a draft during review: ${article#$repo_root/}"
  grep -q '^description:' "$article" ||
    fail "missing description: ${article#$repo_root/}"
  grep -q '^summary:' "$article" ||
    fail "missing summary: ${article#$repo_root/}"
  grep -q '^tags:' "$article" ||
    fail "missing tags: ${article#$repo_root/}"
done

scan_paths=("$content_dir" "$production_dir")

if grep -R -n $'\u2014' "${scan_paths[@]}"; then
  fail "em dash found in Cairnkeep publication material"
fi

if grep -R -E -i -n \
  '(siemens|netcup|192\.168\.|10\.[0-9]+\.[0-9]+\.[0-9]+|npm_[A-Za-z0-9]+)' \
  "${scan_paths[@]}"; then
  fail "private organization, host, address, or token marker found"
fi

for checkpoint in \
  course-00-app \
  course-01-bootstrap \
  course-02-memory \
  course-03-quality \
  course-04-operation \
  course-05-evidence \
  course-06-governance \
  course-07-evaluation; do
  grep -R -q "$checkpoint" "${scan_paths[@]}" ||
    fail "missing course checkpoint reference: $checkpoint"
done

printf 'Cairnkeep publication drafts validated: 8 articles, 8 video scripts.\n'
