#!/bin/bash
set -euo pipefail

TEST_SCRIPT="design_testset.py"
CONFIG="configs/test/codesign_single.yml"
OUT_ROOT="./results"
TAG="rw_baseline"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="$DEVICE"

for index in {0..57}; do
    echo "Running test for index: $index (GPU=$DEVICE)"
    python "$TEST_SCRIPT" "$index" \
        --config "$CONFIG" \
        --out_root "$OUT_ROOT" \
        --tag "$TAG"

    echo "Test for index $index completed successfully."
    echo "----------------------------------"
done

echo "All tests completed."
echo "Next: python scripts/launch_relax_designs.py --root ${OUT_ROOT}/codesign_single_${TAG} --workers 32"
