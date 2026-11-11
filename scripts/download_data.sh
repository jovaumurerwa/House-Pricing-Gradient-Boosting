#!/usr/bin/env bash
set -euo pipefail

KEY="$HOME/.kaggle/kaggle.json"
COMPETITION="house-prices-advanced-regression-techniques"

if [ ! -f "$KEY" ]; then
  echo "Kaggle API key not found at $KEY"
  echo "Get one at kaggle.com > Settings > API > Create New Token, then move it there."
  exit 1
fi
chmod 600 "$KEY"

mkdir -p data
kaggle competitions download -c "$COMPETITION" -p data
unzip -o -q "data/$COMPETITION.zip" -d data
rm "data/$COMPETITION.zip"
echo "Data ready in ./data"
