#!/bin/sh
set -eu

if [ -z "${DS4_TEST_MODEL:-}" ]; then
    echo "cuda-prefill-regression: set DS4_TEST_MODEL to the GGUF path" >&2
    exit 2
fi

bench=${DS4_BENCH:-./ds4-bench}
prompt=${DS4_TEST_LONG_PROMPT:-tests/long_context_story_prompt.txt}
if [ ! -x "$bench" ]; then
    echo "cuda-prefill-regression: missing executable $bench" >&2
    exit 2
fi
if [ ! -f "$prompt" ]; then
    echo "cuda-prefill-regression: missing prompt $prompt" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/ds4-prefill-shapes.XXXXXX")
trap 'rm -rf "$work"' EXIT INT TERM
mkdir -p "$work/shared" "$work/materialized"

run_bench() {
    output_dir=$1
    shift
    env \
        -u DS4_CUDA_NO_HC_SHARED_INTERMEDIATE \
        -u DS4_CUDA_HC_SHARED_INTERMEDIATE_FALSE_TIMING \
        -u DS4_CUDA_NO_HC_SPLIT_NORM_F16 \
        -u DS4_CUDA_NO_PREFILL_NORM_F16_REUSE \
        -u DS4_CUDA_NO_PREFILL_HYBRID_8192 \
        -u DS4_METAL_DISABLE_HC_FUSION \
        DS4_CUDA_Q8_F16_CACHE_MB=0 \
        DS4_CUDA_INDEXER_MXFP4_NATIVE=1 \
        DS4_CUDA_ATTN_COMPACT_KV_EXACT=1 \
        DS4_CUDA_INDEXER_QAT_WARP_EXACT=1 \
        DS4_CUDA_ATTN_OUTPUT_UNPACK_VEC4_EXACT=1 \
        "$@" \
        "$bench" \
        -m "$DS4_TEST_MODEL" \
        --cuda \
        --prompt-file "$prompt" \
        --ctx-alloc 32768 \
        --ctx-start 10319 \
        --ctx-max 25693 \
        --step-incr 15374 \
        --prefill-chunk 4096 \
        --gen-tokens 0 \
        --dump-frontier-logits-dir "$output_dir"
}

# This two-frontier sequence exercises all shapes that the capacity-only
# optimization checks used to miss:
#   cold:   hybrid 8192 -> partial 2127
#   resume: unaligned 1969 -> hybrid 8192 -> canonical 4096 -> tail 1117
run_bench "$work/shared"
run_bench "$work/materialized" DS4_CUDA_NO_HC_SHARED_INTERMEDIATE=1

for frontier in 010319 025693; do
    shared="$work/shared/frontier_${frontier}.logits.json"
    materialized="$work/materialized/frontier_${frontier}.logits.json"
    if ! cmp -s "$shared" "$materialized"; then
        echo "cuda-prefill-regression: logit mismatch at frontier $frontier" >&2
        exit 1
    fi
done

echo "cuda prefill shape regression: OK (10319 -> 25693, byte-identical logits)"
