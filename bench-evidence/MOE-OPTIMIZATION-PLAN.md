# MoE prefill optimization — diagnosis & plan (2026-07-12)

Follow-up to the transient-f16-GEMM patch (branch `cuda-transient-f16-gemm`,
prefill 403→1504 t/s). After that patch, the MoE expert kernels are the tall
pole. This note captures the validated diagnosis for the next optimization.

## Model shape (DeepSeek V4 Flash IQ2XXS)
- 256 experts, 6 used/token, 1 shared expert, 43 layers
- embedding 4096, expert FFN 2048
- gate/up experts: iq2_xxs [256][2048][4096]; down: q2_K [256][4096][2048]

## Post-patch profile (nsys, 4096-tok prefill chunk, cache=0)
- `moe_gate_up_mid_expert_tile8_rowspan_kernel<1024>` — **36%**
- `moe_down_expert_tile16_row2048_kernel` — **26%**
- attention (indexed+static) — ~18%
- our transient dense GEMMs — ~5% (was 76% pre-patch)

## Root cause: redundant weight read + dequant (NOT dp4a-ALU-bound)
Both MoE kernels group tokens into small tiles (gate/up=8, down=16) and stage
the tiles' **activations** in shared memory, reusing each dequantized weight
block only across those 8/16 tokens. With ~96 tokens routed to each expert,
each expert's weight is read from VRAM and iq2/q2K-dequantized **~6–12× per
layer**. Bandwidth arithmetic puts the kernel at ~2.6× the pure weight-read
floor → a mix of redundant reads + redundant dequant + dp4a, not ALU-bound.
Tile size is capped at 8 for gate/up by the **48 KB static-shared limit**
(staging 8 tokens × 16 q8_K blocks ≈ 37 KB).

## Empirical confirmation (tile-size A/B, 8192-tok prefill)
| config | prefill | vs tile8 |
|---|---|---|
| tile8 (default) | 1648 t/s | — |
| tile4 (2× more re-reads) | 456 t/s | 3.6× slower |
| no expert tiling (max re-reads) | 326 t/s | 5× slower |

Super-linear sensitivity to tokens-per-weight-read ⇒ redundancy is THE
dominant cost, and we're on a steep part of the curve → pushing
tokens-per-weight-read **above** 8 should keep paying.

## llama.cpp cross-check
Its MoE (mul_mat_id) sorts tokens by expert (`mm_ids_helper`) and streams each
expert's **entire** token block through MMQ tiles via `expert_bounds`/`ids_dst`
— never re-reading the weight per small tile. That's the proven structure ds4
is missing.

## Plan (incremental, verify logits at each step)
1. ~~**tile16 gate/up** via dynamic shared memory~~ — **TRIED, REFUTED
   2026-07-12.** Implemented `moe_gate_up_mid_expert_tile16_rowspan_kernel`
   (dynamic shared, even-tile8-start folding, block8 LUT called twice), gated
   behind `DS4_CUDA_MOE_GATE_TILE16` (off by default; branch
   `cuda-moe-tensorcore`, uncommitted experiment). Result: **1646→1105 t/s
   (SLOWER)**, cosine 0.9959 top-1 match. Cause: 16 tokens need ~74 KB dynamic
   shared vs 37 KB static for tile8 → halves blocks/SM → the latency-hiding
   occupancy loss outweighs the reduced weight re-reads. **Lesson: the
   activation-staging design can't grow past tile8; tile8 is its sweet spot.**
   The redundancy is still real (tile4 A/B proved it) but is not attackable by
   bigger activation-staged tiles.
   INTERMEDIATE TRIED TOO (Experiment A, 2026-07-12): tile16 WITHOUT activation
   staging (L2-backed). Result: **1644→938 t/s (WORSE than staged tile16)**,
   cosine 0.9980 top-1 match. Cause: without shared staging each token's
   activation is re-read from L2 once PER ROW (×1024 for ROW_SPAN=1024) →
   L2-bandwidth wall. So the two tile16 variants fail for opposite reasons
   (staged=occupancy, unstaged=L2 amplification), which brackets tile8 as a
   genuine local optimum: activations MUST be shared-staged (row reuse), and
   that caps the tile at 8. Tile size cannot beat tile8 in either direction.

2. **Full tiled quantized GEMM** (the ONLY remaining lever, but a real
   multi-day project): tile M(rows)×N(tokens)×K(in_dim/32), stage a small
   K-slice of BOTH dequantized weight and activations in shared per step
   (BM*32 + BN*32 int8 ≈ 2 KB for 32×32 — occupancy-safe), dp4a/mma accumulate
   BM×BN, reuse each loaded element across the whole tile. This is the only
   structure that cuts weight-read+dequant redundancy WITHOUT the occupancy or
   L2 penalties. Complication: iq2_xxs has TWO-level scales (per-256-block `d`
   + per-32 sub-scale `ls`) that must be folded into the K-accumulation
   correctly — this is why llama.cpp's iq2 MMQ is intricate. dp4a first
   (verify logits), then int8 tensor-core (mma m16n8k32).

## VERDICT (2026-07-12) — tile8 PROVEN near-optimal; tiled GEMM built & measured
The tiled quantized GEMM was actually built and made numerically correct
(cosine 0.9989, top-1 match) — the two-level iq2 scales fold into the
K-accumulation fine. Three variants measured, all < tile8 (1647 bench):
  - v1 small-shared (3 KB) / 256 syncs: 1592
  - v2 big-shared (26 KB) / serial stage: 1099
  - v3 big-shared / parallel stage:      1214
Occupancy dominates (small shared wins). Kernel is occupancy/latency-bound.

ROOFLINE PROBE (halve the dp4a work, DS4_CUDA_GEMM_HALFK, timing-only, wrong
results): GEMM 1200 -> 1549 t/s = only 1.29x for HALF the compute. Workload is
~29% compute-scalable, ~71% memory/latency/coordination. THE KILLER RESULT:
even the half-compute GEMM (1549, cheating) still loses to honest tile8 (1647).

CONCLUSION FOR IQ2 GATE/UP: no staged tiled-GEMM structure -- even with compute
magically free -- beats tile8, because tile8 has near-zero coordination overhead
(no weight staging, no per-block syncs, inline fused dequant+dp4a, max occupancy)
and this workload is occupancy-bound. **tile8 remains the practical limit for
the IQ2 gate/up kernel tested here.** This was initially mistaken for a whole-
engine ~1647 bench / ~1500 served limit; the fresh q2_K down follow-up below
breaks that ceiling. The shipped 4x (transient GEMM, 403->1500) still stands.
The GEMM kernel stays on branch
cuda-moe-tensorcore as a correct, reusable scaffold (gated
DS4_CUDA_MOE_GATE_GEMM) for a future GPU/quant where the compute/memory balance
shifts enough to make tiling pay.

## Verification protocol (established)
- Build on pod (sm_120): rsync ds4_cuda.cu → `make cuda CUDA_ARCH=sm_120`.
- Logit parity: `ds4-bench --dump-frontier-logits-dir`, compare cosine (want
  >0.999) + top-1 vs a kill-switched baseline run of the SAME binary.
- Speed: `ds4-bench --gen-tokens 0 --ctx-start 8192 --ctx-max 8192`.
- Home GPU (RTX 3060, 12 GB) can't hold the 87 GB model — pod only.

## Fresh follow-up: q2_K down optimization breaks the 1647 ceiling (2026-07-12)

The earlier verdict remains valid for the IQ2 gate/up tiled-GEMM experiments,
but it did not cover the q2_K down inner loop. That loop had a separate,
profitable defect: `dev_dot_q2_K_q8_K_block8` was token-major and reloaded,
shifted, and masked the same packed 2-bit weights once for each of eight
tokens. The adjacent q4_K implementation was already weight-major.

Implemented on `cuda-moe-tensorcore`:

- `dev_dot_q2_K_q8_K_block8_full`: a full-eight weight-major helper. Packed
  q2 weights/scales are loaded and unpacked once, while the 512 required dp4a
  operations and their integer accumulation order remain unchanged. Partial
  expert tiles retain the generic helper.
- `moe_down_expert_tile16_row2048_interleaved_kernel`: a same-binary gated
  kernel variant using the full-eight helper.
- Deterministic tile16 down summation is now the default for this fast path.
  Writing per-route rows plus the existing fixed-order `moe_sum_kernel` was
  consistently faster than six contending atomics and also makes logits
  reproducible. `DS4_CUDA_MOE_ATOMIC_DOWN=1` still forces atomics.
- `dev_unpack_iq2_signs` no longer recomputes parity. Every byte in
  `cuda_ksigns_iq2xs` already contains the parity bit and has even parity, so
  the old popcount/xor was exactly a no-op.
- The optimized path is default when the down weights are q2_K, expert tiling
  is enabled with tile8, `n_tokens >= 128`, and `midq_blocks == 8` (2048-wide
  expert middle). The full-eight helper is used by the default row2048 launch.
  `DS4_CUDA_MOE_NO_DOWN_Q2_INTERLEAVED=1` is the baseline kill switch.

Measured with the established random 8192-token prompt, cache disabled:

| configuration | repeated prefill t/s | mean | vs baseline |
|---|---:|---:|---:|
| kill-switched baseline q2_K down | 1654.26, 1656.91 | 1655.59 | — |
| interleaved + atomic down (ablation) | 1732.90, 1734.08 | 1733.49 | +4.7% |
| interleaved + deterministic down, before IQ2 parity edit (ablation) | 1747.38, 1747.90 | 1747.64 | +5.6% |
| optimized plain default, including IQ2 parity edit | 1753.85, 1755.33 | 1754.59 | +6.0% |

Correctness: the normal atomic baseline is nondeterministic across processes
(baseline-vs-baseline cosine 0.997086, maxdiff 1.136947), so it cannot satisfy
the nominal >0.999 comparison threshold itself. With both variants routed
through deterministic tile16 output + `moe_sum_kernel`, the full 129,280-logit
comparison is bit-exact: cosine 1.000000000, maxdiff 0, top-1 match. The IQ2
parity change was also bit-exact against the pre-change deterministic dump.
The deterministic baseline flags were
`DS4_CUDA_MOE_NO_DOWN_Q2_INTERLEAVED=1`,
`DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`, and
`DS4_CUDA_MOE_DOWN_TILE16_NO_ATOMIC=1`; optimized deterministic is the plain
default. For the atomic ablation, use `DS4_CUDA_MOE_ATOMIC_DOWN=1`.

Boundary checks at 127, 128, and 129 prompt tokens also compared all 129,280
logits exactly (cosine 1.0, maxdiff 0, top-1 match), covering fallback, fast-path
activation, and the first uneven batch.

Compiler resources stayed effectively flat: the optimized down kernel uses
168 registers/thread, 320-byte stack, and 38,400 bytes shared versus 166 / 320 /
38,400 for baseline. The gain therefore comes from removed q2 unpack/load work,
not an occupancy change.

Served-speed validation used fresh nonces plus the same random-number prompt
through `/v1/completions`: 12,879 prompt tokens, `cached_tokens=0`, one generated
token. The same-request kill-switch A/B was 1549.45 t/s (8.312 s) baseline versus
1631.26 t/s (7.895 s) optimized, a 5.28% served gain. Optimized full 4096-token
chunks were 1678.86--1790.44 t/s; end-to-end client time was 7.93 s.

The repository suite passed with `DS4_CUDA_Q8_F16_CACHE_MB=0`, including the
30,474-token long-context test. Its five CUDA tensor-vector runs were internally
repeatable with zero logit delta; old-vs-new kernel equivalence is established
by the kill-switched full-vocabulary comparisons above, not by that repeatability
test.

Rejected follow-ups on top of the win:

- down row span 2048 (default) beat 1024 and 512: 1646.59 vs 1639.08 vs 1613.45
  before this patch.
- gate row2048 averaged 1736.26 versus 1747.66 for the default row1024; gate
  row512 measured 1744.18 once. Keep row1024.

The register-fed m16n8k32 IQ2 design remains feasible but unimplemented. It
requires a wholesale warp remap (one warp = 16 rows × 8 tokens), so it is no
longer justified as the first follow-up now that the simpler q2_K defect has
produced a verified end-to-end win.

## 2000 tok/s follow-up: fuse gate/up activation loads (2026-07-12)

The 2000 tok/s target was reached and verified. The decisive observation was
that gate and up consume the same staged q8 activation blocks. The old tile8
kernel called the IQ2 block8 helper twice, so for each token in a 256-weight
block it issued the same 64 shared q8 word loads once for gate and again for up.
The new paired helper decodes the
gate and up IQ2 rows independently but loads each q8 word once and feeds it to
both dp4a streams. Integer accumulation and the final floating-point expression
for every output remain in their original order.

Two smaller changes stack on top:

- q2_K down processes its two eight-route groups sequentially. This removes the
  hot `acc[16]` local frame and generic partial-helper frame: compiler resources
  fell from 128 registers / 192-byte stack to 56 registers / zero stack, with
  shared memory unchanged at 38,464 bytes.
- Indexed attention uses three 24-head groups instead of four 16-head groups.
  This cuts duplicated KV/top-k staging while remaining launchable at 72
  registers × 768 threads. A 32-head/1024-thread variant was rejected because
  72 registers/thread exceeds the block register limit; the launch failed with
  `too many resources requested`. Static heads16 was also rejected (1776.35 vs
  1782.16), while indexed rows16 was only a small win (1800.40 vs 1783.62).

The IQ2 pair kernel uses 96 registers, a 64-byte stack, and 40,576 bytes shared,
so it retains the same shared-memory occupancy limit as the old 80-register
kernel. A scalar full-eight IQ2 specialization was exact but slower
(1783.19 -> 1752.60); lower register count did not compensate for its expanded
front-end/code footprint.

Final defaults and same-binary kill switches:

- paired IQ2 gate/up: default; `DS4_CUDA_MOE_NO_GATE_IQ2_PAIR=1`
- sequential-eight shared q2_K down: default;
  `DS4_CUDA_MOE_NO_DOWN_Q2_SHARED_SEQ8=1`
- indexed attention heads24: default;
  `DS4_CUDA_NO_ATTN_INDEXED_HEADS24=1`

Final random 8192-token, cache-disabled ABBA benchmark:

| configuration | runs (tok/s) | mean | delta |
|---|---:|---:|---:|
| previous shared-q2 baseline | 1781.65, 1785.25 | 1783.45 | — |
| new plain default | 2063.86, 2066.74 | 2065.30 | +15.80% |

The no-flags full-logit run measured 2064.32 tok/s. All 129,280 logits were
bit-exact against the kill-switched same-binary baseline: cosine 1.000000000,
max absolute difference 0, and identical top-1. Full-vocabulary checks at 127,
128, and 129 prompt tokens were also bit-exact, covering fallback, fast-path
activation, and partial expert tiles.

Production-server probes used fresh nonces and reported `cached_tokens=0` for
12,879 prompt tokens. Whole-prompt prefill averaged 1903.89 and 1920.97 tok/s;
through 12,288 tokens (three full chunks) it averaged 2031.97 and 2051.38 tok/s.
The final underfilled 591-token tail ran at ~825 tok/s and pulled the whole
request below 2000. Full chunks ranged from 1968.66 to 2169.27 tok/s.

Repository validation passed after enabling the defaults: `make
cuda-regression CUDA_ARCH=sm_120`, the full `make test CUDA_ARCH=sm_120` suite,
30,474-token long-context recall, official logprob vectors, tool-call tests,
and all five tensor-equivalence cases. Tensor equivalence reported zero RMS and
zero maximum absolute logit difference in every case.

Final pod state: tmux session `ds4` was restored with the production 200,000-
context command, cache disabled, and no experimental environment flags. The
`/v1/models` health check succeeded after the final optimized restart.

## Decode optimization: batch-1 occupancy + GEMV coalescing (2026-07-12)

Profiled the DECODE path (nsys, 512 prefill + 128 gen). Key facts: ds4 uses
ZERO CUDA graphs; ~1586 kernel launches/token; decode 41.4 t/s vs a 21.9 ms/tok
(45.6 t/s) GPU-busy ceiling → ~9.4% wall is GPU-idle launch-starvation.

CUDA graphs would recover most of that idle (~+7-10%, ~44-45 t/s) but it's a big
execution-model refactor (default-stream launches, per-token device syncs,
KV-position in kernel args). Deferred.

SIMPLER WINS FOUND FIRST (contained, attack the GPU-busy floor, not the idle):
- **Decode gate/up under-fills the GPU.** moe_gate_up_mid_decode_lut_qwarp32
  launched only ~102 blocks on 188 SMs (128 rows/block). Templated on RPB;
  fine-32 (384 blocks) fills the machine. **+2.1% decode, BIT-EXACT** (cosine
  1.0, maxdiff 0). Now the default; DS4_CUDA_MOE_DECODE_NOFINE / _FINE64 override.
- **matmul_f16_ordered_chunks GEMV was uncoalesced + serial-reduce.** Each of 32
  threads walked a contiguous chunk (neighbors hit far addresses) and thread 0
  serially summed 32 partials. New matmul_f16_coalesced_gemv_kernel: interleaved
  coalesced reads + warp-shuffle reduction (still deterministic). **+2% more**
  (stacks to +4.2% total, 41.34→43.07), cosine 0.999999988, top-1 match. Behind
  DS4_CUDA_F16_COALESCED_GEMV=1 (opt-in: changes numerics ~0.0036 max vs the
  serial kernel, so left flag-gated pending a determinism call).

Remaining decode targets (same techniques): attention_decode_mixed (12.9%) and
the q8_0 projection matmuls (~22%) likely under-fill too. CUDA graphs (+7-10%)
still available on top. All on branch cuda-moe-tensorcore.

## Decode update: paired GEMV coalesced too (2026-07-12)
Added matmul_f16_coalesced_pair_gemv_kernel (same coalesce+shuffle fix for the
paired f16 GEMV, 6.5% of decode), under the same DS4_CUDA_F16_COALESCED_GEMV flag.

Final decode stack (512 prefill + 128 gen, cache off):
  original ................... 41.34 t/s
  + gate/up fine-32 tiling ... 42.20 t/s  (+0.86, BIT-EXACT, live default)
  + coalesced GEMV single .... 43.07 t/s  (+0.87, flag)
  + coalesced GEMV pair ...... 43.60 t/s  (+0.53, flag)
  TOTAL +2.26 t/s (+5.5%), cosine 0.999999988 top-1 match.

Harder tier remaining (deferred, same "fill the GPU" idea but bigger rewrites):
- attention_decode_mixed (12.9%) launches grid(1, n_head)=64 blocks on 188 SMs
  (34% full) -- needs a flash-attention-style KV split + partial-softmax combine,
  not a naive re-tile.
- CUDA graphs (~+7-10% of the launch-idle) -- execution-model refactor.
- q8_0 projection matmuls already well-structured (512 blocks, warp-shuffle) -- no win.

## 2515+ tok/s exact prefill stack: q2 MMA, stream-K attention, and IQ2 scalarization (2026-07-12)

Further large-prefill work raised the exact no-flags path from the preceding
2065.30 tok/s baseline to 2515.33 tok/s, a cumulative 21.79% gain. Every
promoted change has a same-binary kill switch, and the final 129,280-logit
comparison against the two newest IQ2 kill switches is bit-exact: cosine
1.000000000, maximum absolute difference 0, and identical top-1.

The promoted stack is:

- q2_K down uses a register-fed `m16n8k32` INT8 MMA kernel from 128-token
  expert-tiled batches onward. Per-16 q2 scales are folded into positive int8
  A fragments (`q * scale <= 45`), while q8 block sums implement the min term.
  The native K256 floating expression and fixed eight-block tree are preserved.
  Large prefills use a 128-thread geometry. Kill switches:
  `DS4_CUDA_MOE_NO_DOWN_Q2_MMA=1` and
  `DS4_CUDA_MOE_NO_DOWN_Q2_MMA_BLOCK128=1`.
- Indexed 64-head attention uses the exact heads32 stream-K online kernel with
  20 staged KV rows for `n_tokens >= 2048`. It preserves row, score, reduction,
  and online-softmax order. Kill switch:
  `DS4_CUDA_NO_ATTN_INDEXED_HEADS32_STREAMK20=1`.
- Large-prefill IQ2 gate/up uses 384 threads, 48 row lanes, and a 1056-row span.
  Shared route metadata removes the per-thread route arrays and the entire
  compiler stack frame (`REG80, STACK160 -> REG80, STACK0`). Kill switches:
  `DS4_CUDA_MOE_NO_GATE_IQ2_BASEPTR_BLOCK384=1` and
  `DS4_CUDA_MOE_NO_GATE_IQ2_BASEPTR_SHARED_META=1`.
- Full eight-route tiles use a paired gate/up IQ2 specialization with scalar
  integer accumulators. It retains the shared q8 load fusion and preserves each
  route's ib32 accumulation order. Partial tiles keep the generic helper. Kill
  switch: `DS4_CUDA_MOE_NO_GATE_IQ2_PAIR_FULL8=1`.
- The q2 MMA inner loop loads adjacent scale bytes together and packs their
  high nibbles for signed `dp2a` against paired q8 block sums. This halves scale
  loads/shuffles and replaces scalar min-correction multiply-adds without
  changing the integer value. Kill switch:
  `DS4_CUDA_MOE_NO_DOWN_Q2_MMA_DP2A=1`.
- Adjacent q2 scale segments share one aligned 32-bit load and one cross-row
  shuffle, halving that scale-fetch/shuffle work while retaining the original
  segment, MMA, dp2a, and floating reduction order. Kill switch:
  `DS4_CUDA_MOE_NO_DOWN_Q2_MMA_SCALEPAIR32=1`.

Representative cache-disabled 8192-token measurements:

| exact configuration | repeated/final prefill t/s |
|---|---:|
| preceding exact default | 2063.42, 2064.07 |
| + q2 MMA + attention stage20 | 2345.83, 2348.12 |
| + q2 block128 + IQ2 block384 geometry | 2407.11, 2409.52 |
| + shared IQ2 route metadata | 2425.65, 2424.56 |
| + paired IQ2 full-eight specialization | 2462.02, 2459.60 |
| + packed q2 scale/min `dp2a` | 2485.71, 2492.76 |
| + paired 32-bit q2 scale loads | 2514.22, 2508.13 |
| final promoted no-flags full-logit run | 2515.33 |

Full-vocabulary promotion checks were exact for q2 MMA plus attention, geometry,
shared route metadata, and paired full-eight IQ2. The final cumulative comparison
measured 2403.83 tok/s with the two newest IQ2 defaults killed versus 2461.52
tok/s no-flags. The promoted q2 `dp2a` comparison measured 2453.54 killed versus
2481.11 no-flags. The final scale-pair comparison measured 2482.76 killed
versus 2515.33 no-flags. All comparisons had all 129,280 logits identical.

Rejected exact occupancy/locality follow-ups:

- q2 MMA `__launch_bounds__(128,6)` reached 80 registers but introduced an
  8-byte stack frame and regressed twice to 2305.51 and 2302.46 tok/s versus
  2408.95 and 2411.98 defaults. The verified default remains the literal
  unbounded kernel at 95 registers and zero stack.
- Interleaving the two heads32 attention blocks for each token in one-dimensional
  launch order was resource-identical and exact by construction, but flat:
  2421.76 and 2422.44 versus 2420.98 and 2423.45 defaults. It was removed.
- Compact IQ2 activation staging reduced shared memory by 4096 bytes and
  registers from 80 to 78, but its SoA/cooperative-copy access path collapsed
  throughput to 1500.62 and 1508.69 tok/s. It was removed.
- Feeding the q2 MMA C fragment directly back into the next MMA preserved the
  integer result and resources but lengthened the tensor dependency chain;
  throughput regressed to 2359.50 and 2362.37 tok/s. It was removed.
- Retaining one attention KV `float4` across score and value update was
  resource-identical but flat (2485.31/2484.02 versus 2481.56/2484.19). It was
  removed.
- A distinct exact stage20 `cp.async` kernel double-buffered 82,960 bytes of
  dynamic shared KV, overlapped the next scattered gather, and halved CTA
  barriers. It remained spill-free but slightly regressed to 2505.72/2506.22
  versus 2508.68/2514.50, so it was removed.

Final repository validation passed after the last promotions: CUDA regression,
the full `make test CUDA_ARCH=sm_120` suite, 30,474-token long-context recall,
tool-call quality/recovery, official logprob vectors, local golden vectors,
server tests, and all five tensor-equivalence cases. Tensor equivalence reported
zero RMS and zero maximum absolute logit error in every case.

The final production-server probe used a fresh nanosecond nonce and reported
`cached_tokens=0` for 12,879 prompt tokens. Server prefill completed in 5.689 s,
or 2264.03 tok/s for the whole prompt. Through 12,288 tokens (three full chunks)
the average was 2468.61 tok/s; chunk rates were 2564.24, 2468.30, and 2380.75
tok/s. The final underfilled 591-token tail ran at 831.41 tok/s. Client elapsed
including the one generated token was 5.725 s.

Post-final nsys profile (8192 tokens, 2498.97 tok/s under profiling) attributes
33.3% to the promoted IQ2 gate/up kernel, 17.6% to indexed stage20 attention,
and 13.1% to q2 MMA down. Their per-layer averages are 12.434 ms, 13.432 ms,
and 4.879 ms respectively. The top three exact kernels therefore remain 64.0%
of GPU kernel time; further work should prioritize IQ2 instruction/decode cost,
then attention, rather than dense GEMMs.
