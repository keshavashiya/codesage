#!/bin/sh
# CodeSage Pre-Commit Hook
# Installed: {installed_at}
# Mode: {mode} | Severity threshold: {severity}
#
# Blocks commits with issues at or above the severity threshold.
# To bypass (use sparingly): git commit --no-verify

# Check if codesage is available
if ! command -v codesage >/dev/null 2>&1; then
    echo "[codesage] Not installed or not in PATH — skipping review."
    echo "           Install: pipx install pycodesage"
    exit 0
fi

echo "[codesage] Reviewing staged changes ({mode} mode, blocking on {severity}+)..."

# Run review on staged changes only
if codesage review --staged --mode {mode} --severity {severity}; then
    echo "[codesage] Review passed."
    exit 0
else
    exit_code=$?
    if [ "$exit_code" = "1" ]; then
        echo ""
        echo "[codesage] Commit blocked: fix the issues above."
        echo "           To bypass: git commit --no-verify"
        exit 1
    else
        # Non-1 exit = unexpected error; warn but don't block
        echo "[codesage] Review error (exit $exit_code) — allowing commit."
        exit 0
    fi
fi
