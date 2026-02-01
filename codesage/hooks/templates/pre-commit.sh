#!/bin/sh
# CodeSage Pre-Commit Hook
# Installed: {installed_at}
#
# This hook runs security scanning on staged files before commit.
# To bypass: git commit --no-verify

set -e

# Check if codesage is available
if ! command -v codesage &> /dev/null; then
    echo "CodeSage is not installed or not in PATH"
    echo "Install with: pipx install pycodesage"
    exit 1
fi

echo "Running CodeSage security scan..."

# Run security scan on staged files
if codesage security scan --staged --exit-on-findings {severity_flag}; then
    echo "Security scan passed"
    exit 0
else
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
        echo ""
        echo "Security issues found. Commit blocked."
        echo "Fix the issues above or use 'git commit --no-verify' to bypass."
        exit 1
    else
        echo "Security scan failed with error code: $exit_code"
        exit $exit_code
    fi
fi
