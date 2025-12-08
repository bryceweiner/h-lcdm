#!/usr/bin/env bash
# Script to remove xAI API key from Git history
# Based on: https://docs.github.com/en/code-security/secret-scanning/working-with-secret-scanning-and-push-protection/working-with-push-protection-from-the-command-line#resolving-a-blocked-push

set -euo pipefail

echo "⚠️  WARNING: This script will rewrite Git history!"
echo "⚠️  Make sure you have a backup and are on a branch that hasn't been pushed yet."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# The commit that introduced the secret
SECRET_COMMIT="cf56a91"

echo ""
echo "Step 1: Staging current changes (removing hardcoded API key)..."
git add pipeline/common/grok_analysis.py

echo ""
echo "Step 2: Starting interactive rebase to edit commit $SECRET_COMMIT..."
echo "In the editor that opens:"
echo "  1. Change 'pick' to 'edit' for commit $SECRET_COMMIT"
echo "  2. Save and close"
echo ""
read -p "Press Enter to continue..."

# Start interactive rebase
git rebase -i "${SECRET_COMMIT}~1"

echo ""
echo "Step 3: The rebase has paused. Now we'll remove the secret from the commit..."
echo "The file has already been staged with the fix."
echo ""
read -p "Press Enter to amend the commit..."

# Amend the commit to remove the secret
git commit --amend --no-edit

echo ""
echo "Step 4: Continuing rebase..."
git rebase --continue

echo ""
echo "✅ Done! The secret has been removed from Git history."
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo "1. If you've already pushed this branch, you'll need to force push:"
echo "   git push --force-with-lease origin $(git branch --show-current)"
echo ""
echo "2. Set up your environment variable:"
echo "   export XAI_API_KEY='your-xai-api-key-here'"
echo ""
echo "3. Or create a .env file (already gitignored):"
echo "   echo 'XAI_API_KEY=your-xai-api-key-here' > .env"

