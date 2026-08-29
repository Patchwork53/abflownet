#!/bin/bash
set -euo pipefail

TEST_SCRIPT="design_testset.py"
CONFIG="configs/test/codesign_single_no_tb.yml"
OUT_ROOT="./results"
TAG="ablate_195k_200k"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="$DEVICE"

for index in {0..57}; do
    echo "Running ablation test index: $index (GPU=$DEVICE)"
    "$PYTHON" "$TEST_SCRIPT" "$index" \
        --config "$CONFIG" \
        --out_root "$OUT_ROOT" \
        --tag "$TAG"

    echo "Ablation test index $index completed successfully."
    echo "----------------------------------"
done

RESULT_ROOT="${OUT_ROOT}/codesign_single_no_tb_${TAG}"
echo "All ablation tests completed."
echo "Results: ${RESULT_ROOT}"
echo "Next: python ../reward_weighted/scripts/launch_relax_designs.py --root ${RESULT_ROOT} --workers 32"
