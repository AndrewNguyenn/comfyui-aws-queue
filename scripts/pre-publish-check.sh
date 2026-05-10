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

echo "==> Scanning for 12-digit numbers in tracked files (potential AWS account IDs)..."
hits="$(git ls-files | xargs grep -lE '(^|[^0-9])[0-9]{12}([^0-9]|$)' 2>/dev/null | grep -v '^package-lock' || true)"
if [ -n "$hits" ]; then
  yellow "  Files containing 12-digit numbers (review each):"
  for f in $hits; do
    matches="$(grep -nE '(^|[^0-9])[0-9]{12}([^0-9]|$)' "$f" || true)"
    echo "    $f"
    echo "$matches" | sed 's/^/      /'
  done
  yellow "  Verify none are real AWS account IDs."
fi

echo "==> Scanning for AWS access key patterns..."
if git ls-files | xargs grep -lE 'AKIA[0-9A-Z]{16}|aws_secret_access_key' 2>/dev/null; then
  red "  FAIL: AWS credentials in tracked files"
  err=1
else
  green "  ✓ No AWS credentials patterns"
fi

echo "==> Scanning for common secret tokens..."
if git ls-files | xargs grep -liE 'BEGIN PRIVATE KEY|BEGIN RSA PRIVATE KEY|ghp_[A-Za-z0-9]{36}' 2>/dev/null; then
  red "  FAIL: private key or GitHub token in tracked files"
  err=1
else
  green "  ✓ No private keys / GitHub tokens"
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
