#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "" || ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage:
  tools/setup_fork_remote.sh <your-fork-url>

Example:
  tools/setup_fork_remote.sh git@github.com:yourname/parameter-golf.git
  tools/setup_fork_remote.sh https://github.com/yourname/parameter-golf.git

This keeps the OpenAI repo as `upstream` and points `origin` at your fork.
EOF
  exit 0
fi

fork_url="$1"
upstream_url="https://github.com/openai/parameter-golf.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$fork_url"
else
  git remote add origin "$fork_url"
fi

if git remote get-url upstream >/dev/null 2>&1; then
  git remote set-url upstream "$upstream_url"
else
  git remote add upstream "$upstream_url"
fi

echo "Configured remotes:"
git remote -v
