#!/usr/bin/env bash
#
# Pre-publish sanitization. Run before pushing to a public repo.
#
# Checks:
#   1. No 12-digit AWS account IDs in source
#   2. No literal email addresses (use placeholders)
#   3. No API keys, access keys, secrets in source
#   4. .gitignore includes the right exclusions
#   5. cdk.out/, cdk.context.json, .env are not staged
#   6. gitleaks (if installed) clean run
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

red(){ printf "\033[31m%s\033[0m\n" "$*"; }
green(){ printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

err=0

# Files to exclude from scans (they contain example/placeholder patterns intentionally).
SKIP_FILES_RE='^(scripts/pre-publish-check\.sh|\.gitleaks\.toml)$'

scan_files() {
  git ls-files | grep -vE "$SKIP_FILES_RE" | grep -v '^package-lock'
}

echo "==> Scanning for 12-digit numbers in tracked files (potential AWS account IDs)..."
hits="$(scan_files | xargs grep -lE '(^|[^0-9])[0-9]{12}([^0-9]|$)' 2>/dev/null || true)"
if [ -n "$hits" ]; then
  yellow "  Files containing 12-digit numbers (review each):"
  for f in $hits; do
    matches="$(grep -nE '(^|[^0-9])[0-9]{12}([^0-9]|$)' "$f" || true)"
    echo "    $f"
    echo "$matches" | sed 's/^/      /'
  done
  yellow "  Verify none are real AWS account IDs (placeholders like 123456789012 are fine)."
fi

echo "==> Scanning for real AWS access key patterns..."
# Look for AKIA followed by 16 chars in something not commented out
if scan_files | xargs grep -lE '\bAKIA[0-9A-Z]{16}\b' 2>/dev/null | grep -v '^scripts/'; then
  red "  FAIL: AWS access key pattern in tracked files"
  err=1
else
  green "  ✓ No AWS access keys"
fi

echo "==> Scanning for private keys / GitHub PATs..."
if scan_files | xargs grep -lE -- '-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}' 2>/dev/null; then
  red "  FAIL: private key or GitHub token in tracked files"
  err=1
else
  green "  ✓ No private keys / GitHub tokens"
fi

echo "==> Scanning for personal email addresses (use placeholders in source)..."
if scan_files | xargs grep -lE '\b[A-Za-z0-9._%+-]+@(gmail|outlook|yahoo|hotmail|icloud)\.com\b' 2>/dev/null; then
  yellow "  ⚠  personal email found in tracked files (review):"
  scan_files | xargs grep -nE '\b[A-Za-z0-9._%+-]+@(gmail|outlook|yahoo|hotmail|icloud)\.com\b' 2>/dev/null | head -10
  yellow "  Consider replacing with placeholder if it's the deployer's address."
fi

echo "==> Scanning for the project's known-sensitive identifiers..."
# The user's GitHub handle leaked once in ci.ts (now removed). Watchdog for re-leak.
if scan_files | xargs grep -lE '\bAndrewNguyenn?\b' 2>/dev/null; then
  yellow "  ⚠  GitHub handle 'AndrewNguyenn' found in tracked files:"
  scan_files | xargs grep -nE '\bAndrewNguyenn?\b' 2>/dev/null | head -10
  yellow "  Use CDK context (githubOwner) or env var instead."
fi

echo "==> Verifying .gitignore covers cdk.out, cdk.context.json, .env..."
for pat in "cdk.out/" "cdk.context.json" ".env"; do
  if grep -qF "$pat" .gitignore; then
    green "  ✓ .gitignore: $pat"
  else
    red "  FAIL: .gitignore missing pattern: $pat"
    err=1
  fi
done

echo "==> Checking that secret-likely files aren't staged..."
for f in cdk.context.json .env config.local.js frontend/dist/config.js infra/cdk.out; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    red "  FAIL: $f is tracked. Should be in .gitignore."
    err=1
  fi
done

if command -v gitleaks >/dev/null 2>&1; then
  echo "==> Running gitleaks..."
  if ! gitleaks detect --no-git -v; then
    red "  FAIL: gitleaks found leaks"
    err=1
  else
    green "  ✓ gitleaks clean"
  fi
else
  yellow "⚠  gitleaks not installed. Recommended: brew install gitleaks"
fi

if [ "$err" -ne 0 ]; then
  red "Pre-publish check FAILED. Fix above before pushing."
  exit 1
fi
green "✓ Pre-publish check passed."
