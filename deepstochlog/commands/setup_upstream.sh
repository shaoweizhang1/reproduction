#!/usr/bin/env bash
# Prepare the pinned official checkout and the disclosed compatibility fixes.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
checkout="$project_root/upstream"
source_repo="${1:-https://github.com/ML-KULeuven/deepstochlog.git}"
commit=aed95319411b8b5190b5b418b65b523b1436bcfd
if [[ -e "$checkout" || -L "$checkout" ]]; then
  echo "Refusing to overwrite existing checkout: $checkout" >&2
  exit 1
fi
mkdir -p "$(dirname "$checkout")"
git clone "$source_repo" "$checkout"
git -C "$checkout" checkout --detach "$commit"
git -C "$checkout" apply --check "$project_root/patches/upstream.patch"
git -C "$checkout" apply "$project_root/patches/upstream.patch"
echo "Prepared upstream $commit at $checkout"
