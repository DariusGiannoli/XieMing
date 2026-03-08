#!/bin/bash
# Push current state to Hugging Face Spaces
set -e

# Auto-commit any uncommitted changes before pushing
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "→ Uncommitted changes detected — committing to main first..."
  git add .
  git commit -m "chore: auto-save before deploy $(date '+%Y-%m-%d %H:%M')"
fi

echo "→ Pushing to HF Spaces..."
git push hf main --force
echo "✓ Done! Space is updating at https://huggingface.co/spaces/DariusGiannoli/PerceptionBenchmark"
