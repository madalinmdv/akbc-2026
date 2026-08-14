#!/bin/bash
# Generate predictions with every strategy, then ensemble them.
# Usage: ./runmodel.sh [train|val|test]
set -euo pipefail

SPLIT="${1:-val}"

for script in question masked fewshot least_to_most awards; do
    python3 "scripts/${script}.py" --dataset "$SPLIT"
done

python3 scripts/ensemble.py --dataset "$SPLIT"
