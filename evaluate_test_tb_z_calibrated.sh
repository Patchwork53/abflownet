#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT="${ROOT:-$REPO_ROOT/results/codesign_single_tb_z_calibrated_200k}"
WORKERS="${WORKERS:-32}"
TIMEOUT_S="${TIMEOUT_S:-900}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
PYTHON="${PYTHON:-python}"
REWARD_WEIGHTED_REPO="${REWARD_WEIGHTED_REPO:-$REPO_ROOT/../reward_weighted}"
RELAX_SCRIPT="$REWARD_WEIGHTED_REPO/scripts/launch_relax_designs.py"
EVAL_REPO="$REWARD_WEIGHTED_REPO"
AGGREGATE_SCRIPT="$REWARD_WEIGHTED_REPO/abflownet/tools/eval/energy_eval.py"
RELAX_LOG="$REPO_ROOT/results/relax_tb_z_calibrated.log"
EVAL_LOG="$REPO_ROOT/results/eval_tb_z_calibrated.log"
EXPECTED_DESIGNS=34800
EXPECTED_REFS=348

if [[ ! -d "$ROOT" ]]; then
    echo "Results root not found: $ROOT" >&2
    exit 1
fi

if [[ ! -f "$RELAX_SCRIPT" || ! -f "$AGGREGATE_SCRIPT" ]]; then
    echo "Reward-weighted repository not found: $REWARD_WEIGHTED_REPO" >&2
    echo "Set REWARD_WEIGHTED_REPO to its checkout path." >&2
    exit 1
fi

count_relaxed() {
    local pattern="$1"
    if [[ "$pattern" == "design" ]]; then
        find "$ROOT" -type f -name '????_relaxed.pdb' ! -name 'REF1_relaxed.pdb' | wc -l
    else
        find "$ROOT" -type f -name 'REF1_relaxed.pdb' | wc -l
    fi
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "Relaxation attempt $attempt/$MAX_ATTEMPTS"
    set +e
    "$PYTHON" -u "$RELAX_SCRIPT" \
        --root "$ROOT" \
        --workers "$WORKERS" \
        --timeout_s "$TIMEOUT_S" \
        >> "$RELAX_LOG" 2>&1
    relax_rc=$?
    set -e

    designs=$(count_relaxed design)
    refs=$(count_relaxed ref)
    echo "Relaxation attempt $attempt finished: rc=$relax_rc designs=$designs refs=$refs"

    if (( designs == EXPECTED_DESIGNS && refs == EXPECTED_REFS )); then
        break
    fi
done

designs=$(count_relaxed design)
refs=$(count_relaxed ref)
if (( designs != EXPECTED_DESIGNS || refs != EXPECTED_REFS )); then
    echo "Preparation incomplete: designs=$designs/$EXPECTED_DESIGNS refs=$refs/$EXPECTED_REFS" >&2
    exit 1
fi

echo "Preparation complete; starting scoring"
ROOT_ABS=$(realpath "$ROOT")
(
    cd "$EVAL_REPO"
    "$PYTHON" -u -m abflownet.tools.eval.run \
        --root "$ROOT_ABS" \
        --pfx relaxed \
        --once
) > "$EVAL_LOG" 2>&1

"$PYTHON" "$AGGREGATE_SCRIPT" --csv_path "$ROOT/summary.csv"
echo "Evaluation complete: $ROOT/summary.csv"
