#!/usr/bin/env bash
# Deploy to Hugging Face Spaces with a single command.
# Bash equivalent of scripts/deploy-hf.ps1 — see that file for the full
# explanation. Same behaviour, same prereqs.

set -euo pipefail

repo=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "Not inside a git repo. cd into the MixID folder first."
  exit 1
}
cd "$repo"

if ! git remote | grep -qx "hf"; then
  echo "ERROR: no 'hf' remote configured."
  echo ""
  echo "Run this once first (replace <your-username>):"
  echo "  git remote add hf https://huggingface.co/spaces/<your-username>/mixid"
  exit 1
fi

git rev-parse --verify main >/dev/null 2>&1 || {
  echo "No 'main' branch found locally."
  exit 1
}

if ! git show-ref --verify --quiet refs/heads/hf-deploy; then
  echo "Creating local hf-deploy branch (one-time)..."
  git branch hf-deploy main
fi

worktree="$repo/.hf-deploy-worktree"
if [ -d "$worktree" ]; then
  git worktree remove "$worktree" --force >/dev/null 2>&1 || true
  rm -rf "$worktree"
fi

trap 'git worktree remove "$worktree" --force >/dev/null 2>&1 || true; rm -rf "$worktree"' EXIT

echo "Setting up deploy worktree..."
git worktree add "$worktree" hf-deploy >/dev/null

(
  cd "$worktree"
  git reset --hard main >/dev/null

  # Strip non-runtime files that have tripped HF's abuse handler in the past
  # (specifically docs discussing Cloudflare Tunnel got flagged).
  echo "Pruning non-runtime files from the HF deploy..."
  for d in docs articles notebooks tests eval huggingface_space scripts .env.example .gitattributes; do
    if [ -e "$d" ]; then
      git rm -rf "$d" >/dev/null 2>&1 || true
    fi
  done

  echo "Promoting Hugging Face files to root..."
  cp -f "$repo/huggingface_space/README.md" README.md
  cp -f "$repo/huggingface_space/Dockerfile" Dockerfile
  git add README.md Dockerfile
  git commit -m "deploy: HF Spaces (auto-generated from main)" --allow-empty >/dev/null
  echo "Pushing to Hugging Face..."
  git push hf hf-deploy:main --force
)

echo
echo "Deployed to Hugging Face Spaces."
echo "Watch the build at: https://huggingface.co/spaces/<your-username>/mixid"
echo "(check the 'Logs' tab; first build takes 5-10 min, redeploys ~1 min)"
