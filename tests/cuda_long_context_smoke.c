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

static int check_hc_shared_active_rows_case(
        const float *model,
        uint64_t     model_bytes,
        uint32_t     active_rows,
        uint32_t     capacity_rows) {
    enum {
        N_EMBD = 4096,
        N_HC = 4,
        MIX_HC = 24,
    };
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = 3u * sizeof(float);
    const uint64_t norm_offset = (3u + MIX_HC) * sizeof(float);
    const uint64_t active_norm = (uint64_t)active_rows * N_EMBD;
    const uint64_t capacity_norm = (uint64_t)capacity_rows * N_EMBD;
    const uint64_t active_mix = (uint64_t)active_rows * MIX_HC;
    const uint64_t capacity_mix = (uint64_t)capacity_rows * MIX_HC;
    const uint64_t active_residual =
        (uint64_t)active_rows * N_HC * N_EMBD;
    const uint64_t capacity_residual =
        (uint64_t)capacity_rows * N_HC * N_EMBD;
    const float float_canary = 12345.25f;
    const uint16_t half_canary = 0x5a5au;

    float *mix_host = (float *)malloc((size_t)capacity_mix * sizeof(float));
    float *residual_host =
        (float *)malloc((size_t)capacity_residual * sizeof(float));
    float *ref_norm_host =
        (float *)malloc((size_t)active_norm * sizeof(float));
    uint16_t *ref_half_host =
        (uint16_t *)malloc((size_t)active_norm * sizeof(uint16_t));
    float *ref_split_host =
        (float *)malloc((size_t)active_mix * sizeof(float));
    float *shared_norm_host =
        (float *)malloc((size_t)capacity_norm * sizeof(float));
    uint16_t *shared_half_host =
        (uint16_t *)malloc((size_t)capacity_norm * sizeof(uint16_t));
    float *shared_split_host =
        (float *)malloc((size_t)capacity_mix * sizeof(float));
    if (!mix_host || !residual_host || !ref_norm_host || !ref_half_host ||
        !ref_split_host || !shared_norm_host || !shared_half_host ||
        !shared_split_host) {
        free(shared_split_host);
        free(shared_half_host);
        free(shared_norm_host);
        free(ref_split_host);
        free(ref_half_host);
        free(ref_norm_host);
        free(residual_host);
        free(mix_host);
        return 1;
    }

    for (uint64_t i = 0; i < capacity_mix; i++) {
        mix_host[i] = i < active_mix
            ? (float)((int)((i * 13u) % 19u) - 9) * 0.0625f
            : float_canary;
        shared_split_host[i] = float_canary;
    }
    for (uint64_t i = 0; i < capacity_residual; i++) {
        residual_host[i] = i < active_residual
            ? (float)((int)((i * 5u) % 23u) - 11) * 0.03125f
            : float_canary;
    }
    for (uint64_t i = 0; i < capacity_norm; i++) {
        shared_norm_host[i] = float_canary;
        shared_half_host[i] = half_canary;
    }

    ds4_gpu_tensor *ref_out =
        ds4_gpu_tensor_alloc(active_norm * sizeof(float));
    ds4_gpu_tensor *ref_norm =
        ds4_gpu_tensor_alloc(active_norm * sizeof(float));
    ds4_gpu_tensor *ref_half =
        ds4_gpu_tensor_alloc(active_norm * sizeof(uint16_t));
    ds4_gpu_tensor *ref_split =
        ds4_gpu_tensor_alloc(active_mix * sizeof(float));
    ds4_gpu_tensor *ref_mix =
        ds4_gpu_tensor_alloc(active_mix * sizeof(float));
    ds4_gpu_tensor *ref_residual =
        ds4_gpu_tensor_alloc(active_residual * sizeof(float));

    ds4_gpu_tensor *shared_norm =
        ds4_gpu_tensor_alloc(capacity_norm * sizeof(float));
    ds4_gpu_tensor *shared_half_root =
        ds4_gpu_tensor_alloc(capacity_norm * sizeof(uint16_t));
    ds4_gpu_tensor *shared_split_root =
        ds4_gpu_tensor_alloc(capacity_mix * sizeof(float));
    ds4_gpu_tensor *shared_mix_root =
        ds4_gpu_tensor_alloc(capacity_mix * sizeof(float));
    ds4_gpu_tensor *shared_residual =
        ds4_gpu_tensor_alloc(capacity_residual * sizeof(float));
    ds4_gpu_tensor *shared_half = shared_half_root ?
        ds4_gpu_tensor_view(
            shared_half_root, 0, active_norm * sizeof(uint16_t)) : NULL;
    ds4_gpu_tensor *shared_split = shared_split_root ?
        ds4_gpu_tensor_view(
            shared_split_root, 0, active_mix * sizeof(float)) : NULL;
    ds4_gpu_tensor *shared_mix = shared_mix_root ?
        ds4_gpu_tensor_view(
            shared_mix_root, 0, active_mix * sizeof(float)) : NULL;

    int rc = 1;
    if (!ref_out || !ref_norm || !ref_half || !ref_split || !ref_mix ||
        !ref_residual || !shared_norm || !shared_half_root ||
        !shared_split_root || !shared_mix_root || !shared_residual ||
        !shared_half || !shared_split || !shared_mix) {
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(
            ref_mix, 0, mix_host, active_mix * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            ref_residual, 0, residual_host,
            active_residual * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            shared_norm, 0, shared_norm_host,
            capacity_norm * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            shared_half_root, 0, shared_half_host,
            capacity_norm * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(
            shared_split_root, 0, shared_split_host,
            capacity_mix * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            shared_mix_root, 0, mix_host,
            capacity_mix * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            shared_residual, 0, residual_host,
            capacity_residual * sizeof(float))) {
        goto cleanup;
    }

    unsetenv("DS4_CUDA_HC_SHARED_INTERMEDIATE_FALSE_TIMING");
    if (!ds4_gpu_hc_split_weighted_sum_norm_f16_tensor(
            ref_out, ref_norm, ref_half, ref_split, ref_mix, ref_residual,
            model, model_bytes, scale_offset, base_offset, norm_offset,
            N_EMBD, N_HC, 3u, 1.0e-6f, 1.0e-6f)) {
        fprintf(stderr,
                "HC materialized reference rejected active=%u capacity=%u\n",
                active_rows, capacity_rows);
        goto cleanup;
    }
    if (!ds4_gpu_hc_split_weighted_sum_norm_f16_tensor(
            NULL, shared_norm, shared_half, shared_split, shared_mix,
            shared_residual, model, model_bytes, scale_offset, base_offset,
            norm_offset, N_EMBD, N_HC, 3u, 1.0e-6f, 1.0e-6f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            ref_norm, 0, ref_norm_host, active_norm * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            ref_half, 0, ref_half_host,
            active_norm * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_read(
            ref_split, 0, ref_split_host, active_mix * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            shared_norm, 0, shared_norm_host,
            capacity_norm * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            shared_half_root, 0, shared_half_host,
            capacity_norm * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_read(
            shared_split_root, 0, shared_split_host,
            capacity_mix * sizeof(float))) {
        fprintf(stderr,
                "HC shared active-row call rejected active=%u capacity=%u\n",
                active_rows, capacity_rows);
        goto cleanup;
    }

    if (memcmp(ref_norm_host, shared_norm_host,
               (size_t)active_norm * sizeof(float)) != 0 ||
        memcmp(ref_half_host, shared_half_host,
               (size_t)active_norm * sizeof(uint16_t)) != 0 ||
        memcmp(ref_split_host, shared_split_host,
               (size_t)active_mix * sizeof(float)) != 0) {
        fprintf(stderr,
                "HC shared active-row mismatch active=%u capacity=%u\n",
                active_rows, capacity_rows);
        goto cleanup;
    }
    for (uint64_t i = active_norm; i < capacity_norm; i++) {
        if (shared_norm_host[i] != float_canary ||
            shared_half_host[i] != half_canary) {
            fprintf(stderr,
                    "HC shared active-row tail overwrite active=%u capacity=%u index=%llu\n",
                    active_rows, capacity_rows, (unsigned long long)i);
            goto cleanup;
        }
    }
    for (uint64_t i = active_mix; i < capacity_mix; i++) {
        if (shared_split_host[i] != float_canary) {
            fprintf(stderr,
                    "HC shared split tail overwrite active=%u capacity=%u index=%llu\n",
                    active_rows, capacity_rows, (unsigned long long)i);
            goto cleanup;
        }
    }
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(shared_mix);
    ds4_gpu_tensor_free(shared_split);
    ds4_gpu_tensor_free(shared_half);
    ds4_gpu_tensor_free(shared_residual);
    ds4_gpu_tensor_free(shared_mix_root);
    ds4_gpu_tensor_free(shared_split_root);
    ds4_gpu_tensor_free(shared_half_root);
    ds4_gpu_tensor_free(shared_norm);
    ds4_gpu_tensor_free(ref_residual);
    ds4_gpu_tensor_free(ref_mix);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(ref_half);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(ref_out);
    free(shared_split_host);
    free(shared_half_host);
    free(shared_norm_host);
    free(ref_split_host);
    free(ref_half_host);
    free(ref_norm_host);
    free(residual_host);
    free(mix_host);
    return rc;
}

static int check_hc_shared_active_rows(void) {
    enum {
        MIX_HC = 24,
        N_EMBD = 4096,
        MODEL_FLOATS = 3 + MIX_HC + N_EMBD,
    };
    static float model[MODEL_FLOATS] __attribute__((aligned(4096)));
    memset(model, 0, sizeof(model));
    model[0] = model[1] = model[2] = 1.0f;
    for (uint32_t i = 0; i < MIX_HC; i++) {
        model[3u + i] =
            (float)((int)((i * 7u) % 11u) - 5) * 0.03125f;
    }
    for (uint32_t i = 0; i < N_EMBD; i++) {
        model[3u + MIX_HC + i] = 1.0f + (float)(i % 4u) * 0.125f;
    }

    const uint64_t model_bytes = sizeof(model);
    if (setenv("DS4_CUDA_COPY_MODEL", "1", 1) != 0) return 1;
    const int map_ok = ds4_gpu_set_model_map(model, model_bytes);
    unsetenv("DS4_CUDA_COPY_MODEL");
    if (!map_ok) return 1;
    /* Mirrors the exact server failures: an 8192-row FFN allocation running
     * a canonical 4096-row chunk, and a 4096-row attention allocation running
     * a non-capacity tail. */
    if (check_hc_shared_active_rows_case(model, model_bytes, 4u, 8u) != 0 ||
        check_hc_shared_active_rows_case(model, model_bytes, 3u, 4u) != 0) {
        return 1;
    }
    return 0;
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
    /* Installs a synthetic model mapping, so keep this last before cleanup. */
    if (check_hc_shared_active_rows() != 0) {
        rc = 1;
    } else {
        fprintf(stderr, "cuda-regression: HC shared active-row views: OK\n");
    }
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
