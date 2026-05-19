# Deploy to Hugging Face Spaces with a single command.
#
# What it does:
#   1. Creates a temporary git worktree on a deploy-only branch.
#   2. Resets that branch to match `main`.
#   3. Promotes the HF-flavored README + Dockerfile to repo root.
#   4. Pushes to the `hf` remote (force, since the branch is regenerated each time).
#   5. Cleans up the worktree.
#
# Your main checkout never moves — you can run this with uncommitted work
# on main and it won't touch your working tree.
#
# Prereqs (one-time):
#   1. Add the HF remote:
#        git remote add hf https://huggingface.co/spaces/<your-username>/mixid
#   2. Generate an HF write token at https://huggingface.co/settings/tokens
#      The first push will prompt for username + token (cached afterwards).

$ErrorActionPreference = "Stop"

# 1. Locate repo root
$repo = (git rev-parse --show-toplevel 2>$null)
if (-not $repo) {
    Write-Error "Not inside a git repo. cd into the MixID folder first."
    exit 1
}
Set-Location $repo

# 2. Verify the hf remote is configured
$remotes = git remote
if (-not ($remotes -match "^hf$")) {
    Write-Host "ERROR: no 'hf' remote configured." -ForegroundColor Red
    Write-Host ""
    Write-Host "Run this once first (replace <your-username>):"
    Write-Host "  git remote add hf https://huggingface.co/spaces/<your-username>/mixid"
    Write-Host ""
    exit 1
}

# 3. Verify there's a main branch with at least one commit
git rev-parse --verify main 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "No 'main' branch found locally. Are you on a feature branch?"
    exit 1
}

# 4. Ensure the local hf-deploy branch exists
git show-ref --verify --quiet refs/heads/hf-deploy 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating local hf-deploy branch (one-time)..."
    git branch hf-deploy main | Out-Null
}

# 5. Cleanup any lingering worktree from a previous failed run
$worktree = Join-Path $repo ".hf-deploy-worktree"
if (Test-Path $worktree) {
    git worktree remove $worktree --force 2>$null | Out-Null
    Remove-Item $worktree -Recurse -Force -ErrorAction SilentlyContinue
}

# 6. Create the worktree on hf-deploy
Write-Host "Setting up deploy worktree..."
git worktree add $worktree hf-deploy 2>&1 | Out-Null
Push-Location $worktree
try {
    # 7. Reset the worktree's hf-deploy branch to mirror main
    git reset --hard main 2>&1 | Out-Null

    # 8. Strip files that aren't needed at runtime AND have historically
    # tripped HF's abuse handler (e.g., docs that discuss Cloudflare Tunnel
    # got flagged as 'Blocked by rule: Cloudflare' on a prior deploy).
    # The HF container only needs: mixid/, config.py, pyproject.toml,
    # Dockerfile, README.md.
    Write-Host "Pruning non-runtime files from the HF deploy..."
    $prune = @(
        "docs",
        "articles",
        "notebooks",
        "tests",
        "eval",
        "huggingface_space",
        "scripts",
        ".env.example",
        ".gitattributes"
    )
    foreach ($p in $prune) {
        if (Test-Path $p) {
            git rm -rf $p 2>&1 | Out-Null
        }
    }

    # 9. Promote the HF-flavored files to root
    Write-Host "Promoting Hugging Face files to root..."
    Copy-Item -Force (Join-Path $repo "huggingface_space\README.md") "README.md"
    Copy-Item -Force (Join-Path $repo "huggingface_space\Dockerfile") "Dockerfile"

    # 10. Commit the swap (allow-empty for re-deploys without changes)
    git add README.md Dockerfile
    $msg = "deploy: HF Spaces (auto-generated from main)"
    git commit -m $msg --allow-empty 2>&1 | Out-Null

    # 10. Push to HF (force, since we regenerate the branch every deploy)
    Write-Host "Pushing to Hugging Face..."
    git push hf hf-deploy:main --force
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Push failed. Common causes:" -ForegroundColor Red
        Write-Host "  - Wrong remote URL. Check 'git remote get-url hf'."
        Write-Host "  - HF token expired or missing 'Write' scope."
        Write-Host "    Make a new one at https://huggingface.co/settings/tokens"
        throw "push failed"
    }
}
finally {
    Pop-Location
    # 11. Clean up the worktree
    git worktree remove $worktree --force 2>&1 | Out-Null
    Remove-Item $worktree -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Deployed to Hugging Face Spaces." -ForegroundColor Green
Write-Host "Watch the build at: https://huggingface.co/spaces/<your-username>/mixid"
Write-Host "(check the 'Logs' tab; first build takes 5-10 min, redeploys ~1 min)"
