#!/bin/bash
set -euo pipefail

TEST_SCRIPT="design_testset.py"
CONFIG="configs/test/codesign_single_tb_z_calibrated.yml"
CHECKPOINT="logs/codesign_single_tb_z_calibrated/checkpoints/200000.pt"
OUT_ROOT="./results"
TAG="200k"
RESULT_ROOT="${OUT_ROOT}/codesign_single_tb_z_calibrated_${TAG}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-57}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ -e "$RESULT_ROOT" && "$ALLOW_EXISTING" != "1" ]]; then
    echo "Result directory already exists: $RESULT_ROOT" >&2
    echo "Refusing to mix or overwrite generations." >&2
    echo "To resume intentionally, set ALLOW_EXISTING=1 and START_INDEX=<next index>." >&2
    exit 1
fi

if ! [[ "$START_INDEX" =~ ^[0-9]+$ && "$END_INDEX" =~ ^[0-9]+$ ]] ||
   (( START_INDEX < 0 || END_INDEX > 57 || START_INDEX > END_INDEX )); then
    echo "Expected 0 <= START_INDEX <= END_INDEX <= 57." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$DEVICE"

echo "Checkpoint: $CHECKPOINT"
echo "Results:    $RESULT_ROOT"
echo "Indices:    $START_INDEX-$END_INDEX"
echo "GPU:        $DEVICE"

for ((index = START_INDEX; index <= END_INDEX; index++)); do
    echo "Running calibrated-Z test index: $index"
    "$PYTHON" "$TEST_SCRIPT" "$index" \
        --config "$CONFIG" \
        --out_root "$OUT_ROOT" \
        --tag "$TAG"

    echo "Calibrated-Z test index $index completed successfully."
    echo "----------------------------------"
done

echo "All calibrated-Z test generations completed."
echo "Results: $RESULT_ROOT"
echo "Next: python ../reward_weighted/scripts/launch_relax_designs.py --root $RESULT_ROOT --workers 32"
