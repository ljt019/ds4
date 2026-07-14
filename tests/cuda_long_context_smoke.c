#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_mxfp4_guard_fallback(void) {
    enum {
        N_TOKENS = 16,
        N_HEAD = 64,
        HEAD_DIM = 128,
        N_COMP = 128,
    };
    const uint64_t q_count =
        (uint64_t)N_TOKENS * N_HEAD * HEAD_DIM;
    const uint64_t weight_count = (uint64_t)N_TOKENS * N_HEAD;
    const uint64_t comp_count = (uint64_t)N_COMP * HEAD_DIM;
    const uint64_t score_count = (uint64_t)N_TOKENS * N_COMP;
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *weights_host =
        (float *)malloc((size_t)weight_count * sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *baseline_host =
        (float *)malloc((size_t)score_count * sizeof(float));
    float *guarded_host =
        (float *)malloc((size_t)score_count * sizeof(float));
    if (!q_host || !weights_host || !comp_host ||
        !baseline_host || !guarded_host) {
        free(guarded_host);
        free(baseline_host);
        free(comp_host);
        free(weights_host);
        free(q_host);
        return 1;
    }
    for (uint64_t i = 0; i < weight_count; i++) weights_host[i] = 1.0f;

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *weights =
        ds4_gpu_tensor_alloc(weight_count * sizeof(float));
    ds4_gpu_tensor *comp =
        ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *baseline =
        ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *guarded =
        ds4_gpu_tensor_alloc(score_count * sizeof(float));
    int rc = 1;
    if (!q || !weights || !comp || !baseline || !guarded ||
        !ds4_gpu_tensor_write(
            weights, 0, weights_host, weight_count * sizeof(float))) {
        goto cleanup;
    }

    /* The test owns these opt-in controls for its short-lived process. */
    unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE_FALSE_TIMING");
    unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
    unsetenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE");
    for (uint32_t test = 0; test < 2u; test++) {
        memset(q_host, 0, (size_t)q_count * sizeof(float));
        memset(comp_host, 0, (size_t)comp_count * sizeof(float));
        if (test == 0u) {
            /* Not exactly representable as E2M1: exercises pack fallback. */
            q_host[0] = 0.1f;
            for (uint32_t c = 0; c < N_COMP; c++) {
                comp_host[(uint64_t)c * HEAD_DIM] = 1.0f;
            }
        } else {
            /* Every value is exact E2M1/F16, but the two active K32 blocks
             * differ by ten product-exponent steps.  The conservative
             * certificate must reject range 10 (>9). */
            for (uint32_t t = 0; t < N_TOKENS; t++) {
                for (uint32_t h = 0; h < N_HEAD; h++) {
                    float *row = q_host +
                        ((uint64_t)t * N_HEAD + h) * HEAD_DIM;
                    for (uint32_t d = 0; d < 32u; d++) row[d] = 1.0f;
                    for (uint32_t d = 32u; d < 64u; d++) row[d] = 1024.0f;
                }
            }
            for (uint32_t c = 0; c < N_COMP; c++) {
                float *row = comp_host + (uint64_t)c * HEAD_DIM;
                for (uint32_t d = 0; d < 64u; d++) row[d] = 1.0f;
            }
        }
        if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
            !ds4_gpu_tensor_write(
                comp, 0, comp_host, comp_count * sizeof(float)) ||
            !ds4_gpu_indexer_scores_prefill_tensor(
                baseline, q, weights, comp,
                N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                baseline, 0, baseline_host, score_count * sizeof(float))) {
            goto cleanup;
        }

        if (setenv("DS4_CUDA_INDEXER_MXFP4_NATIVE", "1", 1) != 0 ||
            !ds4_gpu_indexer_scores_prefill_tensor(
                guarded, q, weights, comp,
                N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                guarded, 0, guarded_host, score_count * sizeof(float))) {
            goto cleanup;
        }
        unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
        if (memcmp(baseline_host, guarded_host,
                   (size_t)score_count * sizeof(float)) != 0) {
            fprintf(stderr,
                    "SM120A MXFP4 guard fallback mismatch case=%u\n",
                    test);
            goto cleanup;
        }
    }

    /* Exercise the cheaper QAT-provenance specialization.  Q is transformed
     * as one prefix; K is transformed through two consecutive cache views,
     * matching the engine's append path.  The REQUIRE control makes a lost
     * view/root marker observable instead of silently taking the full guard. */
    memset(q_host, 0, (size_t)q_count * sizeof(float));
    memset(comp_host, 0, (size_t)comp_count * sizeof(float));
    for (uint64_t row = 0; row < (uint64_t)N_TOKENS * N_HEAD; row++) {
        q_host[row * HEAD_DIM] = 1.0f;
    }
    for (uint32_t row = 0; row < N_COMP; row++) {
        comp_host[(uint64_t)row * HEAD_DIM] = 1.0f;
    }
    if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) ||
        !ds4_gpu_dsv4_indexer_qat_tensor(
            q, N_TOKENS * N_HEAD, HEAD_DIM)) {
        goto cleanup;
    }
    for (uint32_t half = 0; half < 2u; half++) {
        const uint64_t rows = N_COMP / 2u;
        const uint64_t bytes = rows * HEAD_DIM * sizeof(float);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(comp, half * bytes, bytes);
        if (!view ||
            !ds4_gpu_dsv4_indexer_qat_tensor(view, (uint32_t)rows, HEAD_DIM)) {
            ds4_gpu_tensor_free(view);
            goto cleanup;
        }
        ds4_gpu_tensor_free(view);
    }
    if (!ds4_gpu_indexer_scores_prefill_tensor(
            baseline, q, weights, comp,
            N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            baseline, 0, baseline_host, score_count * sizeof(float)) ||
        setenv("DS4_CUDA_INDEXER_MXFP4_NATIVE", "1", 1) != 0 ||
        setenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE", "1", 1) != 0 ||
        !ds4_gpu_indexer_scores_prefill_tensor(
            guarded, q, weights, comp,
            N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            guarded, 0, guarded_host, score_count * sizeof(float))) {
        goto cleanup;
    }
    unsetenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE");
    unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
    if (memcmp(baseline_host, guarded_host,
               (size_t)score_count * sizeof(float)) != 0) {
        fprintf(stderr, "SM120A MXFP4 QAT-provenance mismatch\n");
        goto cleanup;
    }

    /* A generic tensor write must revoke the trusted prefix.  The native
     * call then returns to the full per-value guard and falls back on 0.1f. */
    {
        const float nonrepresentable = 0.1f;
        if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
            !ds4_gpu_dsv4_indexer_qat_tensor(
                q, N_TOKENS * N_HEAD, HEAD_DIM) ||
            !ds4_gpu_tensor_write(q, 0, &nonrepresentable,
                                  sizeof(nonrepresentable))) {
            goto cleanup;
        }
        if (setenv("DS4_CUDA_INDEXER_MXFP4_NATIVE", "1", 1) != 0 ||
            setenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE",
                   "1", 1) != 0) {
            goto cleanup;
        }
        if (ds4_gpu_indexer_scores_prefill_tensor(
                guarded, q, weights, comp,
                N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) != 0) {
            fprintf(stderr, "SM120A MXFP4 QAT invalidation was not observed\n");
            goto cleanup;
        }
        unsetenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE");
        unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
        if (!ds4_gpu_indexer_scores_prefill_tensor(
                baseline, q, weights, comp,
                N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                baseline, 0, baseline_host, score_count * sizeof(float)) ||
            setenv("DS4_CUDA_INDEXER_MXFP4_NATIVE", "1", 1) != 0 ||
            !ds4_gpu_indexer_scores_prefill_tensor(
                guarded, q, weights, comp,
                N_COMP, N_TOKENS, N_HEAD, HEAD_DIM, 4u, 1.0f) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                guarded, 0, guarded_host, score_count * sizeof(float))) {
            goto cleanup;
        }
        unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
        if (memcmp(baseline_host, guarded_host,
                   (size_t)score_count * sizeof(float)) != 0) {
            fprintf(stderr, "SM120A MXFP4 QAT invalidation mismatch\n");
            goto cleanup;
        }
    }
    rc = 0;

cleanup:
    unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE");
    unsetenv("DS4_CUDA_INDEXER_MXFP4_NATIVE_FALSE_TIMING");
    unsetenv("DS4_CUDA_INDEXER_MXFP4_REQUIRE_QAT_PROVENANCE");
    ds4_gpu_tensor_free(guarded);
    ds4_gpu_tensor_free(baseline);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(q);
    free(guarded_host);
    free(baseline_host);
    free(comp_host);
    free(weights_host);
    free(q_host);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = 0;
    const int mxfp4 = ds4_gpu_sm120a_mxfp4_lane_map_test();
    if (mxfp4 == 0) {
        rc = 1;
    } else if (mxfp4 > 0) {
        fprintf(stderr, "cuda-regression: SM120A MXFP4 conformance: OK\n");
    } else {
        fprintf(stderr, "cuda-regression: SM120A MXFP4 conformance: skipped\n");
    }
    if (mxfp4 > 0) {
        if (check_mxfp4_guard_fallback() != 0) {
            rc = 1;
        } else {
            fprintf(stderr, "cuda-regression: SM120A MXFP4 guard fallback: OK\n");
        }
    }
    if (!ds4_gpu_indexer_mxfp4_pack_roundtrip_test()) {
        rc = 1;
    } else {
        fprintf(stderr, "cuda-regression: indexer MXFP4 pack round trip: OK\n");
    }
    if (check_large_topk() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
