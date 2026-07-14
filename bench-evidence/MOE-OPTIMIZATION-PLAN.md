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

## 3885 tok/s exact fixed-sm120a stack: paired-head TMA and HC/RMS fusion (2026-07-13)

This round deliberately treats the deployment as a fixed appliance: one RTX
PRO 6000 Blackwell Server Edition with 96 GB, compiled for `sm_120a`. The TMA
paths are not a portability abstraction and are never selected by default in a
non-`sm_120a` build. All promotion comparisons below used the cache-disabled
random 8192-token prompt and a same-binary control. Every promoted path matched
all 129,280 frontier logits bit-for-bit (cosine 1.000000000, maximum absolute
difference 0, and top-1 token 223).

The attention work first converted the exact heads32 kernel's scattered KV
gather into a 20-row TMA ping-pong pipeline. Deterministic buffer parity removed
the phase state that had caused an early spill; the final specialization is
`REG63, STACK0, LOCAL0`. It measured 3719.353 tok/s versus 3701.200 for scalar
heads32, or +0.490%. Artifact: `~/attn-tma-pipeline-abba-1783950731`.

The larger win was deduplicating KV work across pairs of heads. A fixed
512-thread scalar kernel assigns two heads to each of 16 warps and stages 40 KV
rows per iteration. It remained spill-free at `REG120` and measured 3744.327
tok/s versus 3711.920 for heads32 TMA, +0.873%. Its TMA successor retains the
two-head reuse while using the exact 20-row ping-pong gather; it compiles at
`REG114, STACK0, LOCAL0` and measured 3792.333 versus 3745.463 for the scalar
pair kernel, +1.251%. Artifacts:
`~/attn-pair-vs-tma-abba-1783951940` and
`~/attn-pair-tma-gate-interleave-1783952875`. The promoted large-prefill
selection is pair-head TMA on `sm_120a`; the layered kill switches are
`DS4_CUDA_NO_ATTN_INDEXED_PAIR_HEADS_TMA=1` (fall back to scalar pair-head) and
`DS4_CUDA_NO_ATTN_INDEXED_PAIR_HEADS=1` (fall back to heads32).

The live N16 gate/up kernel was already persistent over output-row blocks. Its
next exact change double-buffered the gate fragment in shared memory so the
following iteration's required gate-to-up synchronization also protects reuse,
removing 15 otherwise redundant end-of-row-block barriers. Static shared memory
rose from 11,520 to 19,712 bytes, while the kernel stayed spill-free at the
hard `REG128` residency cliff. On top of pair-head TMA, the interleaved means
were 3802.640 tok/s double-buffered versus 3792.333 single-buffered, +0.272%.
The default can be disabled with
`DS4_CUDA_MOE_NO_GATE_IQ2_MMA_N16_DOUBLE_GATE_FRAG=1`. A four-way full-logit
check is retained at `~/attn-combined-gen32-1783952601`.

The final material change fuses fixed-shape HC expansion with the immediately
following RMS normalization and F16 conversion. One 256-thread CTA owns a
complete `[4,4096]` HC row, keeps its 64 values per thread live, preserves the
old destination-major RMS reduction order, writes the required FP32 HC result,
and also produces an F16 scratch tensor consumed directly by the next cuBLAS
projection. Explicit graph provenance prevents stale scratch reuse across the
attention-post/FFN-pre and FFN-post/next-attention boundaries. Both add and
non-add specializations compile `REG96, STACK0, LOCAL0`.

The first fusion build differed by one ULP because fast math reassociated the
initial split multiply with the following additions. Using an explicit
round-to-nearest initial multiply and dependent round-to-nearest FMAs restored
the original operation order. The repaired path matched all 129,280 logits
exactly. An eight-run ABBA measured:

| configuration | runs (tok/s) | mean | delta |
|---|---:|---:|---:|
| pair-head TMA + gate double-fragment | 3799.70, 3793.13, 3806.38, 3804.09 | 3800.825 | — |
| + fused HC expand/RMS/F16 | 3891.73, 3883.38, 3876.31, 3891.64 | 3885.765 | +2.235% |

The fusion is now the large-prefill default and can be disabled with
`DS4_CUDA_NO_HC_EXPAND_RMS_F16=1`; `DS4_CUDA_HC_EXPAND_RMS_F16=1` also permits
the specialization below the default 2048-row threshold. A final promoted
same-binary full-logit run measured 3782.67 tok/s killed versus 3885.38 default,
again with maximum absolute logit difference 0.

A fresh no-flags nsys capture ran at 3871.84 tok/s and replaced all stale
pre-N16 budget estimates:

| kernel/stage | total GPU time | share |
|---|---:|---:|
| pair-head indexed TMA attention | 427.722 ms | 19.8% |
| N16 IQ2 gate/up | 399.884 ms | 18.5% |
| q2_K MMA down | 228.060 ms | 10.6% |
| attention decode | 91.505 ms | 4.2% |
| static attention | 73.932 ms | 3.4% |
| indexer scores | 70.750 ms | 3.3% |
| head RMS | 61.528 ms | 2.9% |
| f32-to-f16 conversion | 53.884 ms | 2.5% |
| attention packing | 48.534 ms | 2.3% |
| fused HC kernels (46.037 + 41.750) | 87.787 ms | ~4.0% |
| indexed weighted sum | 39.236 ms | 1.8% |

The capture also contains 108.445 ms of one-time IQ2 repacking during startup;
it is not part of timed steady-state inference and must not be ranked as a
prefill optimization target. The current top three steady-state kernels are now
48.9% of GPU kernel time, with indexed pair-head attention again the largest
single exact lever.

## 4068 tok/s exact stack: IQ2 AUX64 and direct attention layouts (2026-07-13)

The next fixed-`sm_120a` round removed decode and format-conversion work without
changing any model arithmetic. All promoted comparisons used the cache-disabled
random 8192-token prompt. Full-logit checks covered all 129,280 vocabulary
entries and required cosine 1.0, maximum absolute difference 0, and identical
top-1 token 223.

The N16 IQ2 expert-plane representation now groups each row's two 32-bit aux
words into one aligned 64-bit load. The two words retain their original lane
ownership and bit-reversed K visitation; only the load transaction changes. An
eight-run ABBA measured 3927.852 versus 3879.960 tok/s, +47.892 or +1.234%.
Post-change profiling reduced the gate kernel to 374.096 ms. Artifact:
`~/iq2-aux64-abba-1783955383`. Kill switch:
`DS4_CUDA_MOE_NO_GATE_IQ2_N16_PLANE_AUX64=1`.

A signed-half IQ2 decode LUT was also exact, but collapsed throughput to roughly
1147 tok/s because the added table dependency serialized the hot path. It was
removed completely; do not retry it.

The indexed pair-head TMA producer now performs inverse RoPE and writes output-
A's fixed group-major F16 activation layout directly from its attention
epilogue. A dual FP32 safety store preserves fallback behavior. The packed
specialization is spill-free at `REG116, STACK0, LOCAL0`. A three-way balanced
comparison measured:

| producer | mean tok/s | delta versus old producer |
|---|---:|---:|
| old FP32 producer + separate inverse-RoPE/pack | 3929.842 | — |
| direct F16 + FP32 safety store | 3977.717 | +1.218% |
| direct F16 without FP32 store | 3965.023 | +0.895% |

The packed-only pair variant was therefore rejected. Artifact:
`~/attn-direct-threeway-1783958018.csv`. The direct dual-store path is the
large-prefill `sm_120a` default; kill switch:
`DS4_CUDA_NO_ATTN_INDEXED_PAIR_TMA_PACKED_F16=1`.

Output-A's grouped FP32 lows are now transposed directly to token-major F16 and
fed to output-B's unchanged FP32-output cuBLAS GEMM. This removes the separate
FP32 unpack plus FP32-to-F16 conversion and applies to both prepacked and normal
attention producers. The generalized all-layer ABBA measured 4018.015 versus
3985.295 tok/s, +32.720 or +0.821%. Artifacts:
`~/attn-low-all-1783959537` and `~/attn-low-all-abba-1783959642.csv`. Kill
switch: `DS4_CUDA_NO_ATTN_OUTPUT_LOW_F16_DIRECT=1`.

A direct-pack nsys capture proved that the pair producer absorbed inverse RoPE
and F16 packing at effectively zero added kernel time: pair attention remained
427.758 ms versus 427.722 ms before the change, while standalone attention
packing fell from 48.534 to 24.833 ms. Artifact:
`~/ds4-direct-pack-profile-1783958433.nsys-rep`.

## Stage40 direct packing and promoted 4K profile (2026-07-13)

The remaining raw/static and ratio-128 attention layers use the exact 24-warp
stage40 kernels. Their epilogues now share 32 inverse-RoPE coefficient pairs per
CTA, reproduce the old rounded normalization and RoPE operations, and emit the
same group-major F16 layout directly. Both also retain a post-RoPE FP32 safety
copy. Resources remain well below the 768-thread register limit:

| specialization | registers | stack/local |
|---|---:|---:|
| static stage40 packed dual-store | 80 | 0 / 0 |
| decode stage40 packed dual-store | 72 | 0 / 0 |

The first full check measured 4067.51 versus 4010.24 tok/s and was bit-exact over
all 129,280 logits. The balanced ABBA was:

| configuration | runs (tok/s) | mean | stddev |
|---|---:|---:|---:|
| prior direct-pair + low-F16 stack | 4009.31, 4012.91, 4004.88, 4009.86 | 4009.240 | 3.31 |
| + stage40 direct pack | 4065.13, 4071.35, 4065.09, 4071.59 | 4068.290 | 3.67 |

That is +59.05 tok/s or +1.473%. Artifacts:
`~/stage40-exact-1783960735` and `~/stage40-abba-1783960735.csv`. The no-FP32
variant was also exact and averaged 4072.063 versus 4064.248 for dual-store,
only +0.192%; it remains opt-in because that margin does not justify removing
the safety fallback. Artifact: `~/stage40-store-abba-1783960735.csv`. The
dual-store path is the large-prefill default; kill switch:
`DS4_CUDA_NO_ATTN_STAGE40_PACKED_F16=1`.

The no-FP32 path was rechecked after the exact stack reached ~4303 tok/s and Q
RMS/RoPE moved into the producer.  An initial two-run sample was inconclusive,
but the mirrored eight-run audit reproduced the old result: dual-store runs of
4306.29, 4296.70, 4309.02, and 4302.41 averaged 4303.605 tok/s, while no-FP32
runs of 4318.05, 4305.73, 4302.17, and 4318.74 averaged 4311.172 tok/s.  That is
+7.568 tok/s or +0.1758%.  All 129,280 frontier logits were byte-identical,
including top-1 token 223.  Artifact:
`~/stage40-nof32-exact-audit-1783969438`.  No successful prepacked-output path
consumes the FP32 heads; they exist only for a projection fallback, and the
no-FP32 path already fails closed if that projection cannot launch.  The
fixed-appliance default now uses no-FP32.  A single explicit `store_f32` value
is passed from the host into CUDA and reused for post-success fallback
bookkeeping, so template selection cannot disagree with whether `batch_heads`
is valid.  Kill switch `DS4_CUDA_NO_ATTN_STAGE40_PACKED_NO_F32=1` restores the
dual store.  The final same-binary check measured 4297.57 tok/s dual-store
versus 4304.08 default and produced identical SHA-256 hashes for the complete
129,280-logit JSON files.  Artifact:
`~/stage40-nof32-promoted-exact-1783971800`.  The CUDA long-context regression
also passes.

The final same-binary promotion check killed direct pair packing, low-F16
output, and stage40 direct packing independently. It measured 3928.98 tok/s
killed versus 4063.46 no-flags, with all 129,280 logits bit-identical. Artifact:
`~/attn-promoted-exact-1783961585`.

A fresh no-flags nsys capture ran at 4043.31 tok/s under profiling. Artifact:
`~/ds4-prefill-promoted-4060-1783961764.nsys-rep`.

| kernel/stage | total GPU time | share |
|---|---:|---:|
| indexed pair-head TMA attention | 428.027 ms | 20.8% |
| N16 IQ2 gate/up | 374.001 ms | 18.1% |
| q2_K MMA down | 228.144 ms | 11.1% |
| packed stage40 decode attention | 92.991 ms | 4.5% |
| packed stage40 static attention | 74.948 ms | 3.6% |
| indexer scores | 70.660 ms | 3.4% |
| standalone Q head RMS | 61.535 ms | 3.0% |
| F32-to-F16 conversions | 41.720 ms | 2.0% |
| output low-F16 transpose | 12.023 ms | 0.6% |
| remaining RoPE kernels | 8.142 ms | 0.4% |

The next measured exact target is the 61.5 ms standalone Q head-RMS pass. Every
production attention consumer already holds all 512 Q values for one head in
registers, so reproducing the old 256-thread RMS tree and forward RoPE inside
those consumers can remove an 86 GiB full-model read/write round trip. The
experiment must remain opt-in until intermediate Q values and all frontier
logits are bit-identical; pair TMA must stay below the `REG128` cliff.

## 4210 tok/s exact stack: fused Q RMS/forward-RoPE (2026-07-13)

The fixed-shape packed attention producers now accept raw Q-B output and perform
the Q head RMS normalization plus forward RoPE in registers.  The warp-local RMS
tree reproduces the old 256-thread shared-memory tree operand-for-operand; the
forward rotation reuses the already staged inverse-RoPE coefficient with an
exact sine-sign flip.  Any ineligible or failed packed launch prepares Q through
the original standalone RMS and RoPE path before falling back.  `Qraw`, `Qnorm`,
`Qcur`, attention diagnostics, steering, small batches, and unsupported shapes
retain the original path.

Final `sm_120a` resources clear every occupancy and spill gate:

| fused specialization | registers | stack/local |
|---|---:|---:|
| pair TMA packed dual-store | 118 | 0 / 0 |
| static stage40 packed dual-store | 79 | 0 / 0 |
| decode stage40 packed dual-store | 78 | 0 / 0 |

The first full comparison measured 4057.92 tok/s with fusion killed and 4210.07
tok/s with fusion enabled.  All 129,280 frontier logits were bit-identical
(cosine 1.0, maximum difference 0, zero differing entries, top-1 token 223).
Artifact: `~/q-fusion-exact-1783963495`.

The eight-run mirrored ABBA was:

| configuration | runs (tok/s) | mean |
|---|---:|---:|
| standalone Q RMS/RoPE | 4062.95, 4066.92, 4069.54, 4066.73 | 4066.535 |
| fused Q RMS/RoPE | 4206.02, 4219.19, 4208.05, 4203.55 | 4209.203 |

That is +142.668 tok/s or +3.509%.  Artifact:
`~/q-fusion-abba-1783963651.csv`.  The path is now the no-flags large-prefill
default; kill switch: `DS4_CUDA_NO_ATTN_Q_RMS_ROPE_FUSED=1`.  The final promoted
same-binary comparison measured 4054.98 tok/s killed versus 4218.39 default and
again produced all 129,280 logits with maximum difference 0.  Artifact:
`~/q-fusion-promoted-exact-1783964030`.

A fresh no-flags nsys capture ran at 4191.63 tok/s.  Artifact:
`~/ds4-prefill-qfusion-1783964115.nsys-rep`.  The standalone Q head-RMS kernel is
absent, while pair attention stayed effectively flat (427.366 ms after fusion
versus 428.027 ms before it).  The full report accounts for 1994.267 ms of GPU
kernel execution; 108.435 ms is one-time startup IQ2 repacking, leaving 1885.831
ms of steady-state kernels against a 1954.371 ms measured prefill.  Therefore the
earlier Fable estimate of a 622 ms / 31% "unprofiled bucket" was a truncation
artifact, not a real accounting gap.

| current steady-state kernel/stage | total GPU time |
|---|---:|
| indexed pair-head TMA attention | 427.366 ms |
| N16 IQ2 gate/up | 373.739 ms |
| q2_K MMA down | 227.968 ms |
| packed stage40 decode attention | 92.916 ms |
| fused HC expand/RMS/F16, both variants | 87.526 ms |
| packed stage40 static attention | 75.942 ms |
| shared/dense projection GEMMs (individual kernels) | 5--77 ms each |
| indexer scores | 70.707 ms |
| F32-to-F16 conversions | 41.748 ms |
| HC split weighted sum | 39.250 ms |
| q8-to-F16 dequantization | 35.508 ms |
| routed-expert sum | 27.871 ms |

The active optimization objective remains exact prefill throughput toward 6000;
the separate memory-first priority asserted by the external consult is not
adopted unless the owner explicitly changes the objective.  Useful conclusions
from that consult still stand: shared-memory AUX `cp.async` is unlikely to beat
the current register/L2 path, full-layer INT8 expansion needs an explicit VRAM
ledger, and route-tile widening plus conversion/write-read fusion deserve
measurement against this current profile.

## 4281 tok/s exact stack: weighted-norm F16 reuse (2026-07-13)

The attention and FFN weighted RMS kernels now optionally emit both the
unchanged FP32 norm and the exact elementwise FP16 conversion used by the dense
prefill GEMMs.  The half output reuses `batch_flat_hc` only after the preceding
HC-normalized activation has been consumed and its provenance invalidated.  A
local validity flag—not the HC provenance state—keeps it live through all
full-chunk consumers until the subsequent HC expansion overwrites the scratch.
The FP32 result remains available for routed-expert quantization, ratio-4 tail
refresh, imatrix/debug output, and every fallback.

The reused half activation feeds attention Q-A and KV, attention-compressor
KV/gate, indexer-compressor KV/gate and projection, the FFN router, and shared
expert gate/up.  Each call fails closed to its original FP32 API if the cached
or transient half GEMM is unavailable.  Non-CUDA builds do not reference the
new hooks.  Diagnostic serial-F16 mode also rejects the shortcut, preserving
its reference arithmetic.  The dual RMS kernel keeps the original two
`FMUL.FTZ` dependency chain and converts only the rounded second result:
`REG20, STACK0, LOCAL0`, versus `REG18, STACK0, LOCAL0` for the old FP32-only
kernel.

The first same-binary exact comparison measured 4195.97 tok/s with reuse killed
and 4285.93 tok/s with it enabled.  All 129,280 frontier logits were identical:
cosine 1, maximum difference 0, zero differing values, and matching top-1.
Artifact: `~/norm-f16-exact-1783965788`.

The balanced ABBA/BAAB series was:

| configuration | runs (tok/s) | mean |
|---|---:|---:|
| original per-consumer conversion | 4210.93, 4205.79, 4204.52, 4204.64 | 4206.470 |
| weighted-norm F16 reuse | 4281.44, 4283.11, 4279.14, 4278.47 | 4280.540 |

That is +74.070 tok/s or +1.761%.  Artifact:
`~/norm-f16-abba-1783965937.csv`.  The path is now the no-flags large-prefill
default, with independent kill switch
`DS4_CUDA_NO_PREFILL_NORM_F16_REUSE=1`; the old positive opt-in remains
harmless.  The final promoted same-binary comparison measured 4209.02 tok/s
killed versus 4292.55 default and again produced all 129,280 logits with
maximum difference 0.  Artifact:
`~/norm-f16-promoted-exact-1783966239`.

A fresh no-flags nsys capture ran at 4275.60 tok/s.  Artifact:
`~/ds4-prefill-normreuse-1783966352.nsys-rep`.  `f32_to_f16_kernel` fell from
41.748 ms / 1,103 launches to 4.408 ms / 383 launches.  In particular, all 720
grid-65,536 conversions of the 4096-token by 4096-wide attention/FFN norms are
gone.  The 172 dual-output RMS launches cost 16.376 ms total; their extra half
store therefore did not consume the eliminated conversion time.  Total GPU
kernel time is 1961.776 ms including 108.528 ms of one-time IQ2 repacking, or
1853.248 ms steady-state against a 1915.987 ms measured prefill wall time.

The repository CUDA long-context regression also passes on the promoted
`sm_120a` binary.  The next low-risk exact traffic target is the 27.872 ms
six-slot routed-expert sum: retaining the slot-ordered accumulation in the FFN
HC expansion can remove 86 launches and the routed-output write/read.  Larger
post-4281 experiments remain the four-head-per-warp attention discriminator and
the two-CTA cluster-N32 gate decode-sharing design; neither is allowed to bypass
the same full-logit and balanced-timing gates.

## 4303 tok/s exact stack: deferred MoE sum fused into FFN HC/RMS (2026-07-13)

The deterministic routed-MoE down path now has a CUDA-only mode that leaves its
six `[token, route, 4096]` outputs materialized instead of launching
`moe_sum_kernel`.  The following fixed FFN HC/RMS kernel performs the identical
left fold from positive zero over routes 0 through 5, adds the shared expert,
then retains the existing HC multiply/four-FMA chain, RMS traversal, FP32 HC
store, and F16 norm store.  This removes the intermediate 4096-wide routed sum
write/read and one launch per FFN call.

The deferred wrapper rejects every non-deterministic or incompatible path
before quantization: atomic down, direct-sum, transient down, Q4, fewer than
2048 rows, a route count other than six, or an output width other than 4096.
Host selection also excludes `ffn_moe_out` diagnostics, materialized/debugged
`ffn_out`, directional steering, ineligible HC fusion, and shared-down F16.
Other gate/up/mid/down diagnostics remain valid.  The post hook owns
`batch_routed_out` as scratch and can reconstruct the original sum plus old HC
kernel if its fused launch is unavailable, so no fallback consumes an
unwritten sum.

The new kernel is `REG96, STACK0, LOCAL0, SHARED2048`, identical to the old
FFN HC/RMS specialization and safely below its 128-register residency limit.
Compiled SASS has exactly six dependent `FADD.FTZ` instructions starting from
`RZ`, a seventh dependent add for the shared output, then the unchanged
`FMUL.FTZ` and four dependent `FFMA.FTZ` operations.

The first exact A/B measured 4286.50 tok/s with the old sum and 4291.17 tok/s
with fusion.  All 129,280 logits were identical (cosine 1, maximum difference
0, zero differing values, matching top-1).  Artifact:
`~/moe-sum-hc-exact-1783967579`.

The balanced ABBA/BAAB discriminator was:

| configuration | runs (tok/s) | mean |
|---|---:|---:|
| materialized routed sum | 4278.16, 4291.18, 4282.59, 4276.06 | 4281.998 |
| deferred sum in HC/RMS | 4304.48, 4295.68, 4311.01, 4300.44 | 4302.903 |

That is +20.905 tok/s or +0.488%.  Artifact:
`~/moe-sum-hc-abba-1783967674.csv`.  The no-flags large-prefill path now uses
the fusion; kill switch: `DS4_CUDA_NO_MOE_SUM_HC_FUSED=1`.  The old positive
opt-in remains harmless.  The final promoted comparison measured 4274.42 tok/s
killed versus 4297.33 default and again matched all 129,280 logits exactly.
Artifact: `~/moe-sum-hc-promoted-exact-1783968052`.

The nsys capture `~/ds4-prefill-moe-sum-hc-1783967930.nsys-rep` confirms the
traffic model.  The old 27.872 ms / 86-launch `moe_sum_kernel` disappears; the
FFN HC/RMS stage grows from 46.027 ms to 65.048 ms, a net 8.851 ms reduction.
Steady-state GPU kernel time falls from 1853.248 ms to 1844.340 ms.  The
promoted binary also passes the CUDA long-context regression.

## Rejected: four heads per attention warp (2026-07-13)

The indexed TMA attention kernel was specialized from 16 warps owning two heads
each to eight warps owning four heads each.  The mapping
`blockIdx.y * 32 + warp + {0, 8, 16, 24}` covered the same 32 heads per CTA,
retained the same two CTAs per token, and preserved each head's sorted row
order, warp reduction, online-softmax recurrence, sink epilogue, inverse RoPE,
and packed/F32 store order.  Cooperative row-list initialization and the TMA
barrier/ping-pong pipeline are block-size independent, so the 256-thread launch
was source-audited as exact-feasible.

The production `<ONE_EXP, PACKED_F16, STORE_F32, FUSE_Q_RMS_ROPE> =
<true, true, true, true>` specialization compiled at `REG226, STACK0, LOCAL0,
SHARED4352`; all instantiated variants were spill-free at REG216--230.  Thus
register spilling was not the failure.  Dynamic shared memory still limited the
kernel to one CTA per SM, so the change cut resident warps from 16/48 to 8/48.
It halved shared-memory K reads per head group but did not reduce the two global
TMA gathers or total per-head arithmetic.

The short mirrored timing discriminator measured 4285.14 and 4288.82 tok/s for
the four-head variant (mean 4286.98), against pair-head controls of 4301.34,
4316.00, and 4299.74 tok/s (mean 4305.69).  That is -18.71 tok/s or -0.435%,
well below the roughly +1% end-to-end go threshold.  The variant was therefore
stopped before the expensive full-logit comparison and removed from the source.
This also falsifies the idea that more per-warp head ILP alone can recover the
attention time: on this shape, retaining 16 resident warps is worth more than
halving the shared K loads.  Fable's 40--90 ms same-token gather-dedup estimate
does not apply to this experiment because `grid.y=2` and both global gathers
remain.

## Rejected: bounded four-expert signed-INT8 expansion (2026-07-13)

Fable's full-layer signed-INT8 expansion idea was reduced from a 4 GiB resident
buffer to a 66.5 MiB rolling group of four experts.  The timing discriminator
materialized the exact register-fed MMA fragments plus their unfused IQ2
metadata:

| component | layout | bytes |
|---|---|---:|
| signed A fragments | 4 experts x 2 projections x 128 row tiles x 16 K256 x 8 K32 x 32 lanes x `int4` | 64 MiB |
| odd post-MMA scales | row-tile-native `uint8_t` | 2 MiB |
| raw weight scales | row-tile-native FP16 bits | 0.5 MiB |

The expander reproduced the live LUT/xor-shuffle `a0..a3` mapping exactly and
ran as 512 128-thread CTAs.  A spill-free 188-CTA persistent N16 consumer used
device-count prefixes and three 6/5/5 row slabs, so an average group completed
in one SM wave without a host count read or queue.  The consumer retained the
same Q8 LDSM tile, post-MMA `ls`, FP scaling, clamp, SwiGLU, and route scatter.
A separate same-binary oracle instantiated the native plane/LDSM/AUX64 kernel
with the same sequential nonexact K reduction, avoiding credit for changing the
FP tree.

Compiler resources all cleared:

| kernel | registers | stack/local | static shared |
|---|---:|---:|---:|
| native false oracle | 119 | 0 / 0 | 19,712 B |
| expand consumer | 127 | 0 / 0 | 17,664 B |
| expander | 40 | 0 / 0 | 3,328 B |

The architecture nevertheless lost decisively on a true 8192-token macrochunk.
The native false oracle measured 4483.28 tok/s, while expansion measured
3556.37 tok/s (-20.7%).  CUDA-event stage timing isolated the failure:
native-false gate/up totaled 349.056 ms over 43 layers, versus 809.171 ms for
expanded-false, an extra 460.115 ms or 2.32x gate time.  The 64 expand/consume
pairs per layer create 5,504 kernel pairs over the model and the N16 consumer
rereads roughly terabytes of four-byte-expanded fragments from L2.  Those costs
overwhelm the removed IQ2 LUT/shuffle decode even though the rolling buffer
fits cache.  The path missed the 12% timing gate by hundreds of milliseconds,
so the expensive full-logit oracle comparison was not run and all experiment
code was removed.  Pre-expansion is not a viable N16 lever; it would need true
wider-N reuse without DSM-scale synchronization overhead to change this result.

## Rejected as-is: true 8192-token prefill chunk (2026-07-13)

Forcing `--prefill-chunk 8192` exposed a real scheduling upper bound: 4503.19
and 4515.21 tok/s versus nearby 4096-chunk controls around 4306 tok/s, roughly
+4.6%, with context buffers growing only from ~540 MiB to ~929 MiB.  It is not
an exact optimization as implemented.  Artifact
`~/chunk8192-exact-1783970500` compared the same binary at chunk4096 and
chunk8192: all 129,280 logits differed, maximum difference 1.48881338, cosine
0.9990413175, while top-1 remained token 223.  The ratio-4 compressor explicitly
refreshes its four-row frontier at every chunk boundary through the small-batch
path, so omitting the 4096 boundary changes later FP8 rounding and index replay.
The result is therefore retained as evidence for an exact hybrid schedule—4096
attention/compressor microchunks with an 8192 dense/FFN macro-batch—not as a
promotable flag change.

## 4424 tok/s exact hybrid: 4096 attention, 8192 FFN (2026-07-13)

The chunk-size upper bound was recovered without changing the model's
load-bearing 4096-token attention boundary.  The CUDA-only fixed-appliance
discriminator reserves an 8192-row physical batch arena while keeping the
semantic chunk size at 4096 and the raw-SWA ring at 4352 rows.  For each layer
it executes attention at positions 0 and 4096 in the original order, then one
FFN over all 8192 rows, one HC buffer swap, and one command synchronization.
Only the complete 8192-token macro-boundary is exposed to checkpoint progress.

The first macro-FFN attempt measured 4424.30 tok/s but was not exact: all
129,280 logits differed, maximum difference 0.97723294, cosine
0.9987420696, with top-1 still 223.  Layer-0 dumps localized the issue rather
than discarding the schedule:

- both 4096 attention `hc_attn_post` files were byte-identical to canonical;
- the first difference was `hc_ffn_pre`, caused by the F16 HC projection
  changing cuBLAS shape from two M=4096 calls to one M=8192 call;
- splitting only that projection restored the concatenated intermediate SHA
  exactly and retained 4419.28 tok/s;
- the following FFN norm remained byte-identical, while router logits and
  softmax weights differed even though top-k expert IDs were identical;
- splitting the F16 router projection into two M=4096 calls restored the full
  reference.  Shared-expert Q8 GEMMs, routed MoE, SwiGLU, HC post, and the
  deterministic six-route sum all remain macro-batched at M=8192.

The resulting same-binary full-logit check measured 4433.41 tok/s and produced
the identical complete JSON SHA-256 as canonical: zero differing entries,
maximum difference 0, and top-1 token 223.  Artifact:
`~/hybrid-default-refactor-exact-1783971980`.  A separate saved-executable
check first proved that the attention-alias refactor with the hybrid disabled
was itself byte-identical to the pre-refactor binary.

The mirrored ABBA/BAAB discriminator was:

| configuration | runs (tok/s) | mean |
|---|---:|---:|
| canonical 2 x 4096 | 4309.59, 4303.16, 4303.49, 4308.47 | 4306.178 |
| exact hybrid | 4427.86, 4424.21, 4424.12, 4421.13 | 4424.330 |

That is +118.153 tok/s or +2.744%.  Artifact:
`~/hybrid-exact-abba-1783973019.csv`.  Persistent-state checks also matched
complete frontier-logit hashes after resuming the hybrid cache to 8224 tokens
and through a full additional 4096-token chunk at 12288.  Artifacts:
`~/hybrid-exact-resume32-1783973304` and
`~/hybrid-exact-resume4096-1783973387`.  The CUDA long-context regression
passes.

The fresh profile `~/ds4-prefill-hybrid-exact-1783973615.nsys-rep` ran at
4407.43 tok/s under nsys.  IQ2 repacking remains a one-time startup operation
and is excluded from the benchmark's prefill timer.  Relative to the prior
exact stack, steady kernel time fell by about 54 ms:

| kernel/stage | exact hybrid GPU time |
|---|---:|
| indexed pair-head TMA attention | 426.958 ms |
| N16 IQ2 gate/up | 343.591 ms |
| q2_K MMA down | 210.080 ms |
| packed stage40 decode attention | 92.674 ms |
| packed stage40 static attention | 74.824 ms |
| indexer scores | 70.681 ms |

The gain is therefore real routed-MoE amortization, not the nonexact attention
shortcut: combining both halves reduces per-expert N16 padding and cuts
gate/up by about 30 ms and down by about 18 ms while all attention kernels stay
unchanged.  The current implementation is still explicitly gated by
`DS4_CUDA_PREFILL_HYBRID_8192=1`,
`DS4_CUDA_PREFILL_HYBRID_SPLIT_HC_PRE=1`, and
`DS4_CUDA_PREFILL_HYBRID_SPLIT_ROUTER=1` pending default-policy cleanup and
generalization beyond one cold 8192-token macro.

## Promoted exact hybrid: repeated macros and 200k-safe scratch (2026-07-13)

The exact hybrid is now a production-capable fixed-appliance default rather
than a one-frontier discriminator. Two separate memory changes were required
before promotion. First, the graph records an explicit 4096-row attention
capacity independently of its 8192-row physical FFN capacity. Attention-only
scratch, including the two `comp_cap x rows` indexer score/mask matrices, Q,
heads, indexer-Q, and compressor/output intermediates, remains sized for 4096.
This preserved the 4430.09 tok/s path and the canonical full-logit SHA, but a
200k-context server still ran out of memory after model preparation.

The final allocator uses the strict attention-then-FFN lifetime boundary to
overlay four large scratch groups on one ordered CUDA stream:

| attention lifetime | FFN lifetime |
|---|---|
| `batch_q` | `batch_routed_mid` |
| `batch_heads` | `batch_routed_gate` |
| `indexer_scores` | `batch_routed_down` |
| `comp_mask` | packed FFN norm/shared/router/routed-output views |

Every logical tensor is a non-owning, exactly bounded view; the four owners are
freed only after all views. `batch_q_half` deliberately remains independent
because its shared-down F16 result survives routed gate/up/mid execution. The
phase-alias build measured 4415.10 tok/s and reproduced all 129,280 canonical
logits byte-for-byte. Artifact: `~/hybrid-phase-alias-exact-1783976000`.

This recovered production memory without reducing context. The canonical
200k server used 95,575 MiB with 1,676 MiB free. The promoted hybrid server
used 95,665 MiB with 1,586 MiB free—only 90 MiB additional VRAM—and returned a
healthy `/v1/models` response with `context_length: 200000`. Startup artifact:
`~/hybrid-mem3-server.log` (the subsequent no-flags promotion was rechecked in
`~/hybrid-default-mem-server.log`).

The scheduler now accepts an absolute macro start, validates every compressed
and indexer frontier against `start / ratio`, and consumes repeated aligned
8192-token macros inside ordinary chunked prefill. Each macro still executes
attention at absolute `start` and `start + 4096`, runs the two exact M=4096 HC
and router projections, and macro-batches only the proven row-independent FFN
work. Initial unaligned prefixes and final <=4096 tails use the canonical path.
Progress/checkpoint callbacks observe only the completed macro end; a failed
macro is never replayed after mutating cache state.

The generalized full-vocabulary checks were all byte-identical to canonical:

| case | canonical | exact hybrid | artifact |
|---|---:|---:|---|
| cold 12288 (8192 + 4096 tail) | 4184.86 | 4259.16 | `~/hybrid-generalize-1783977000/{canonical12,hybrid12}` |
| cold 16384 (two macros) | 4097.62 | 4202.63 | `~/hybrid-generalize-1783977000/{canonical16,hybrid16}` |
| resume 4096 -> 12288, suffix rate | 4079.99 | 4185.87 | `~/hybrid-generalize-1783977000/{canonical-resume4-12,hybrid-resume4-12}` |

Both the 4096 checkpoint and resumed 12288 frontier matched in the resume
test. At 16384 the repeated schedule gains 105.01 tok/s or 2.563%; the benefit
therefore survives longer attention and nonzero cache frontiers.

For the fixed CUDA Flash shape, exact hybrid allocation/scheduling and both
required split projections are now the no-flags default. The single rollback
switch is `DS4_CUDA_NO_PREFILL_HYBRID_8192=1`; setting it measured 4299.11
tok/s and the same canonical SHA, while the promoted default measured 4417.79
tok/s and the same SHA. Artifact: `~/hybrid-promoted-default-1783978000`.
Separate split kill switches exist for diagnostics, but eligibility fails
closed to canonical if either exact split is disabled. MTP and SSD-streaming
sessions do not reserve the unused 8192 arena.

## Long-file profile: second exact-hybrid macro (2026-07-13)

The capture `~/ds4-prefill-hybrid-16k-20260713.nsys-rep` measured two
consecutive 8192-token macro-prefills in one session. Under profiling, the first
macro ran at 4407.04 tok/s and the resumed `8192 -> 16384` macro at 4002.17
tok/s. Embedding-kernel timestamps delimit the two ranges exactly; their GPU
budgets are:

| kernel/stage | first macro | second macro |
|---|---:|---:|
| indexed pair-head attention | 426.926 ms | 473.683 ms |
| IQ2 gate/up N16 | 343.317 ms | 342.583 ms |
| q2_K down N16 | 209.869 ms | 209.350 ms |
| stage40 decode attention | 92.667 ms | 229.071 ms |
| stage40 static attention | 74.848 ms | 0 ms |
| indexer scores | 70.697 ms | 187.520 ms |
| all CUDA work | 1792.176 ms | 2040.874 ms |

MoE is effectively flat across macros. The long-context slowdown is attention:
indexed attention grows by 46.8 ms, while decode attention and the indexer grow
by 136.4 ms and 116.8 ms. This supersedes the stale 4068-tok/s budget used by
the second Fable consult: the fresh cold-macro wall/kernel gap is only about
66 ms, not an unattributed 622 ms/31% bucket, and the consult's proposed fused
Q-RMS/RoPE and no-FP32 stage40 paths are already promoted.

The existing smaller indexer WMMA tiles were also timed without rebuilding.
For the resumed second macro, WMMA128 delivered 4007.87 tok/s, WMMA64 3909.26
(-2.46%), and WMMA32 3763.59 (-6.09%). Their extra CTAs/repeated B-tile loads
outweigh any occupancy benefit, so the kill-switch ladder is not a route to the
long-context win.

## Rejected: lane-0 online-softmax scalar state (2026-07-13)

The pair-head indexed attention kernel was specialized so only lane 0 updated
each head's online-softmax max/sum/`expf` state, then broadcast the exact old
and row scales to the other lanes. QK reduction, per-row order, accumulator
FMAs, sink arithmetic, and final normalization were unchanged. The compiled
production specialization improved from `REG118` to `REG107`, with
`STACK0/LOCAL0`, and all 129,280 logits remained byte-identical.

It nevertheless fell from the promoted 4417.79-class result to 4165.30 tok/s,
about -5.7%. Predicating the warp-uniform scalar work does not recover enough
SFU throughput to pay for two additional scale broadcasts per row; shared
memory already fixes occupancy at one CTA/SM, so the register reduction has no
residency value. Artifact: `~/lane0-softmax-1783979000`. The specialization
and its flag were removed completely, and the pod was restored from
`~/pre-lane0-1783978500`.

## Rejected: branch-specialized ONE_EXP online softmax (2026-07-13)

The next discriminator kept the exact baseline rounding points while splitting
the common `score <= max` and rare growing-max arms. The new pair-head
specialization compiled substantially smaller (`REG106`, `STACK0/LOCAL0`)
than the promoted `REG118` kernel, and its full 129,280-logit dump was
byte-identical (SHA-256 `37b51e2894498f647d836851e8ddef811ea18fb0d602eaa8c5072a1ec9787c21`).

Despite the resource improvement, same-binary timing fell from 4432.09 to
4247.95 tok/s (-4.15%). The explicit branch and altered dependency chains cost
more than the unused multiply-by-one operations; the lower register count still
cannot create a second resident CTA because shared memory fixes occupancy at one
CTA/SM. Artifact: `~/attn-grow-20260713`. The specialization and flag were
removed, restoring `ds4_cuda.cu` SHA-256
`23e075ec33c39a8695419fdcc1a1351e575c32f58ebd474dc30327c93777f7d5`.

## Rejected: compact N8 expert-route tails (2026-07-13)

The gate/up kernel received an exact 256-thread N8 specialization for residual
expert counts of 1--8 routes. Compact N16 metadata retained every full group
and 9--15-route tail on the promoted kernel, while one count-derived N8
candidate CTA per expert handled only the disjoint small tail. The N8 kernel
compiled at `REG108, STACK0, LOCAL0`, with 7,424 B static and 33,280 B dynamic
shared memory, allowing two resident CTAs. Its MMA lane mapping, IQ2 decode,
post-MMA scales, and exact reduction tree were unchanged; all 129,280 logits
were byte-identical.

The extra compact-metadata work and 256 candidate CTAs per layer overwhelmed
the small amount of skipped tail work: same-binary timing fell from 4416.23 to
4342.46 tok/s (-1.67%). Artifact: `~/gate-n8-tail-20260713`. The N8
specialization, metadata builders, and flag were removed.

## Rejected: stage40 one-exp online softmax (2026-07-13)

The packed no-FP32/fused-Q static and decode stage40 kernels received a
fixed-shape specialization that replaced the two online-softmax exponentials
with the exact ONE_EXP selection already used by indexed attention. It compiled
at `REG80` (static) and `REG79` (decode), `STACK0/LOCAL0`, one register above
their baselines and below the 85-register residency ceiling. SASS removed one
dynamic `MUFU.EX2`, and full logits matched byte-for-byte at both 8192 and
16384 frontiers.

It did not improve throughput: baseline versus ONE_EXP was 4421.59 versus
4415.81 tok/s at the first macro and 4007.60 versus 4002.77 on the resumed
second macro. Direct profiles were worse as well: aggregate decode stage40 grew
from 321.738 to 325.573 ms and static stage40 from 74.848 to 75.888 ms. The
branch/dependency cost outweighs the saved SFU instruction on this kernel.
Artifacts: `~/stage40-one-exp-20260713` and
`~/ds4-stage40-one-exp-16k-20260713.nsys-rep`. The specialization was removed.

## Exact scratch-free N32 gate feasibility boundary (2026-07-13)

The promoted persistent N16 gate kernel is already `REG128` with 19,712 B
static and 66,560 B dynamic shared memory. A full-K Q8 activation panel costs
4,160 B per route; N32 therefore requires 133,120 B dynamic and 152,832 B total,
far above the 101,376-B per-block limit. N24 also fails at 99,840 B dynamic plus
unavoidable static state. Exact N32 doubles the reduction/accumulator state and
projects to roughly REG168 before incidental address/decode pressure.

Changing loop order does not recover the requested mechanism under the
memory-neutral constraint. K-outer traversal needs roughly 512 KiB of global
partial-output scratch for 2,048 rows, while row-tile-outer traversal restages
the N32 activation panel sixteen times. The previously removed honest M64/N32
streaming proxy compiled at REG147 and improved its much older stack by only
0.272%; the old BM32/BN32 GEMM variants likewise lost to tile8 and were not
maxdiff-0. With persistent scratch and cluster/DSM explicitly disallowed, there
is no promotable exact N32 implementation on this block resource envelope.

## Rejected at compile gate: canonical q2_K non-trans LDSM (2026-07-13)

A default-off exact down specialization cooperatively gathered each warp's
canonical 16-row by 32-byte q2 chunk into an 8-KiB shared page, then attempted
to load the four native MMA-A registers with non-trans
`ldmatrix.m16n16.x2.b8`. This would have removed the duplicated row-strided q
loads and cross-row shuffles without changing any packed q bytes, scale/min
correction, MMA, K order, or FP tree.

The appliance toolchain rejects that instruction form on sm120a:
`ptxas: Modifier '.trans' require for instruction ldmatrix with shape
'.m16n16'`. The compile failure was the predeclared falsifier. A transposed
shared layout would require a materially more expensive byte-transpose staging
mechanism, so no transpose/stmatrix detour was pursued. The helper,
specialization, and flag were removed before benchmarking.

## Promoted: register-only IQ2 sign decode (2026-07-13)

The fixed exact N16 gate/up kernel no longer stages or reads the 128-byte IQ2
sign table.  Each seven-bit sign index is expanded entirely in registers with
two multiply/spread operations, parity popcount, and PTX `prmt.b32` sign-byte
replication.  Conditional packed-byte negation is expressed as xor plus a
per-byte one; IQ2's nonzero magnitudes make cross-byte carry impossible.  An
exhaustive host proof covered all 128 sign indices, all 256 grid entries, and
both four-byte halves.

The new specialization remained `REG128, STACK0, LOCAL0` with the same 19,712
bytes of static shared memory.  The old/default-false SASS matched the stable
binary instruction-for-instruction, while the register-sign specialization
fell from 1,768 to 1,576 static instructions, emitted 32 sign-replicating
PRMTs, and contained no sign-table `LDS.U8`.  Full 129,280-vocabulary logits
were byte-identical (SHA-256
`37b51e2894498f647d836851e8ddef811ea18fb0d602eaa8c5072a1ec9787c21`).

Balanced same-binary cold-8192 timing was:

| path | runs (tok/s) | mean |
|---|---|---:|
| staged sign table | 4418.37, 4423.49, 4413.16, 4437.59 | 4423.15 |
| register signs | 4486.27, 4510.03, 4515.75, 4511.94 | 4506.00 |

The gain is 82.85 tok/s or 1.87%.  Paired nsys captures attribute it directly:
the 43 gate launches fell from 343.351 ms to 308.281 ms, saving 35.07 ms
(-10.21%).  The register path is now the sm_120a fixed-shape default for
prefill; `DS4_CUDA_MOE_NO_GATE_IQ2_N16_REG_SIGN_DECODE=1` restores the exact
staged-table specialization.  Artifacts are in `~/iq2-regsign-20260713`.

## Rejected: warp-block Q8-to-F16 indexing (2026-07-13)

The scalar Q8 expansion kernel's per-element runtime quotient/remainder was
replaced by an sm_120a-only two-dimensional launch: each 256-thread CTA kept
the same eight consecutive Q8 blocks, one block per warp, while `blockIdx.y`
supplied the output row.  The literal `I2F.S16 -> F2FP.F16 -> HMUL2`
arithmetic and output mapping were unchanged.  The specialization compiled at
`REG14, STACK0, LOCAL0` versus `REG18` for scalar, and removed the reciprocal
and division-helper call from SASS.  Full logits were byte-identical.

The removed indexing was not a large enough part of the kernel.  Paired nsys
captures measured 559 launches at 32.041 ms scalar versus 29.105 ms warp-block,
only 2.94 ms or 9.2% saved.  The first cold wall-clock pair was correspondingly
flat at 4505.65 versus 4507.69 tok/s.  This missed the predeclared 15% direct
kernel and 0.25% end-to-end gates, so the kernel, launch helper, and flag were
removed.  Artifacts are in `~/q8-warpblocks-20260713`.

## Validated: hybrid Q-B transient-weight reuse (2026-07-13)

The exact hybrid's two 4096-token attention microchunks previously expanded
the same 1024-by-32768 Q-B Q8 weight twice per layer.  A default-off path now
uses the first 64 MiB of the existing 384-MiB `batch_routed_up` allocation as
an ephemeral F16 weight view: microchunk zero performs the identical scalar
Q8-to-F16 expansion, and microchunk one reuses those bytes.  Both execute the
same FP32-to-F16 activation conversion and the same
`cublasGemmEx(T,N,32768,4096,1024,F16/F16->F32)` arguments.  The ordinary path
is replayed on any eligibility or launch failure.  The buffer remains idle
until FFN after both attention halves, and hybrid eligibility already excludes
MTP and SSD, so there is no allocation, lifetime, or arena-skip change.

Full logits were byte-identical.  The first cold pair improved from 4506.98 to
4517.98 tok/s.  Paired nsys captures proved the intended mechanism: scalar
Q8-to-F16 launches fell from 559 to 516 and aggregate time from 32.056 to
27.611 ms, saving exactly 4.444 ms.  The discriminator is
`DS4_CUDA_PREFILL_HYBRID_REUSE_QB_F16=1`; promotion is deferred while the same
proven overlay mechanism is extended to the other four repeated attention
weights.  Artifacts are in `~/qb-reuse-20260713`.

## Promoted: raw-word q2_K shuffle hoist (2026-07-13)

The fixed fused N16 down kernel formerly extracted each of four 2-bit q planes
and then repeated the same two cross-row warp shuffles inside every segment.
The promoted specialization shuffles/selects the packed `ql`/`qh` source words
once per chunk, then performs the four shifts and masks on the already-selected
words.  Copy/select commutes exactly with `(word >> shift) & 0x03030303`, so
all four MMA A words, scale/min arithmetic, integer accumulation, and FP
reduction order are unchanged.

The default-false SASS matched the previous binary instruction-for-instruction.
The new path stayed at `REG128, STACK16, LOCAL0`, while static instructions fell
from 1,712 to 1,672, `SHFL.BFLY` from 27 to 15, and `SEL` from 136 to 100;
`WARPSYNC` remained unchanged at 19.  All 129,280 logits were byte-identical.
Direct profile time for the 43 down launches fell from 210.267 to 202.271 ms,
saving 8.00 ms or 3.80%.

Balanced cold-8192 timing was:

| path | runs (tok/s) | mean |
|---|---|---:|
| post-extraction shuffles | 4506.28, 4516.94, 4521.18, 4510.25 | 4513.66 |
| raw-word shuffle hoist | 4527.76, 4534.16, 4532.00, 4529.82 | 4530.94 |

That is +17.27 tok/s or +0.383%.  Raw-word shuffling is now the fixed N16
default; `DS4_CUDA_MOE_NO_DOWN_Q2_N16_RAW_Q_SHUFFLE=1` restores the exact old
specialization.  Artifacts are in `~/q2-raw-shuffle-20260713`.

## Promoted: exact-hybrid attention Q8 weight lifetime reuse (2026-07-13)

The two 4096-token attention microchunks in each exact 8192-token macro used
to expand the same five Q8 weights independently: Q-A, KV, Q-B, grouped
output-A, and output-B.  The promoted path lends attention five nonowning F16
views over the first 204 MiB of the already allocated 384-MiB
`batch_routed_up` arena.  Microchunk zero performs the identical scalar Q8
expansion and microchunk one reuses it.  The views are disjoint, validity is
tracked independently per weight and reset per layer, and the ordered CUDA
stream consumes both microchunks before FFN reclaims the arena.  Hybrid
eligibility already excludes MTP and SSD sessions, so this adds no allocation,
no synchronization, and no persistent VRAM.

Q-A, KV, and output-B retain the exact F16-input `cublasGemmEx` geometry.
Output-A retains the grouped `cublasGemmStridedBatchedEx` geometry and its
special transient eligibility rules.  Cached weights still take priority;
each failed or ineligible external path replays the ordinary implementation,
including an in-place output-B recovery after output-A has overwritten the
packed-head scratch.

All 129,280 logits were byte-identical at both 8192 and 16384 frontiers:
SHA-256 `e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`
at 8192 and
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`
at 16384.  Paired nsys captures measured the scalar expansion kernel falling
from 559 launches / 32.056 ms to 344 launches / 17.777 ms, a 14.28-ms saving;
the major attention and MoE kernels were unchanged.  Balanced cold-8192 timing
was:

| path | runs (tok/s) | mean |
|---|---|---:|
| repeated expansion | 4524.51, 4530.21, 4538.42, 4539.35 | 4533.12 |
| five-weight reuse | 4558.83, 4575.53, 4579.15, 4565.32 | 4569.71 |

That is +36.59 tok/s or +0.807%.  At 16384, the same exact path improved
4296.27 to 4331.36 tok/s.  It is now the exact-hybrid default;
`DS4_CUDA_NO_PREFILL_HYBRID_REUSE_ATTN_Q8_F16=1` restores repeated expansion,
while `DS4_CUDA_PREFILL_HYBRID_REUSE_QB_F16=1` remains a Q-B-only diagnostic
when the full path is killed.  Artifacts are
`~/all5-{base,var}-20260713-a`,
`~/all5-{base,var}16-20260713-a`, and
`~/all5-{base,var}-profile-20260713.nsys-rep`.

## Promoted: compile-time IQ2 plane strides (2026-07-13)

The live exact gate/up specialization is shape-locked to 2,048 expert rows,
but its planar IQ2 address expressions still carried the row count at runtime.
Consequently every K step rebuilt eight AUX addresses with a chain of seven
`IMAD.WIDE.U32` instructions using the runtime 16-KiB plane stride.  The
promoted specialization makes the proven layout constants visible to ptxas:
the K-plane stride is `2048 * 66 = 0x21000`, the d-plane is `0x1000`, and each
of the eight AUX planes is `0x4000` apart.

The specialization remains `REG128, STACK0, LOCAL0` with 19,712 bytes static
and 66,560 bytes dynamic shared memory.  Its eight hot AUX reads compile to a
single base plus immediate offsets `+0x1000, +0x5000, ... +0x1d000`; full
static `IMAD` count drops from 225 to 215 and the seven runtime-stride IMADs
disappear.  MMA, IQ2 decode, signs, scales, reductions, and all storage are
unchanged.  Full 129,280-vocabulary logits were byte-identical with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.

The 43 gate launches fell from 308.272 to 301.516 ms, saving 6.76 ms or 2.19%.
Balanced cold-8192 timing was:

| path | runs (tok/s) | mean |
|---|---|---:|
| runtime plane strides | 4559.08, 4576.60, 4564.54, 4562.77 | 4565.75 |
| fixed plane strides | 4595.78, 4579.14, 4595.64, 4591.67 | 4590.56 |

That is +24.81 tok/s or +0.543%.  The fixed specialization is now the
sm_120a prefill default;
`DS4_CUDA_MOE_NO_GATE_IQ2_N16_FIXED_PLANE_2048=1` restores runtime strides.
Artifacts are `~/fixedplane-{base,var}-20260713-a`,
`~/fixedplane-var-profile-20260713.nsys-rep`, and
`~/fixedplane-20260713-all.sass`.

## Rejected: register-resident indexer WMMA epilogue (2026-07-13)

The live `indexer_scores_wmma128_kernel` stored each warp's 16-by-16 FP32
accumulator fragment to an 8-KiB shared C tile, synchronized the CTA, and
loaded the same values back for the ReLU and 64-head weighted accumulation.
An sm_120a-only discriminator used the fixed accumulator lane map to keep all
eight values per lane in registers.  It preserved the identical eight K16
WMMA calls per head, the same head-order FP32 accumulation, and the same final
score addresses, while deleting the C-tile transport and one CTA barrier per
head.

The compiler evidence was clean: the variant used `REG51, STACK0, LOCAL0` and
38,144 bytes of shared memory versus `REG48, STACK0, LOCAL0` and 46,080 bytes
for the baseline, retaining two resident CTAs per SM.  Static SASS retained
16 `HMMA.16816` instructions and reduced `BAR.SYNC` sites from four to three,
`STS` from 12 to nine, and `LDS` from 24 to 18.  Full frontier logits were
byte-identical with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.

The removed transport was already hidden by the existing MMA/staging
pipeline.  Same-binary nsys captures measured 42 score launches at 70.628 ms
for the shared-C baseline and 70.366 ms for the register epilogue: only
0.262 ms or 0.37% saved.  This missed the predeclared 18-ms direct and 1%
end-to-end gates, so the kernel and flag were removed.  Artifacts are
`~/indexer-reg-{base,var}-20260713-a` and
`~/indexer-reg-{base,var}-profile-20260713-a.nsys-rep`.

## Rejected: transient F16 indexer-Q cache (2026-07-13)

The live WMMA128 indexer score kernel converts the same Q values from F32 to
F16 once per compressed-component CTA.  A memory-neutral discriminator used
the CUDA-dead 64-MiB `batch_group_tmp` lifetime to cache the exact F16 bits
once per 4096-token microchunk.  The QAT producer retained its canonical F32
output and additionally wrote `__float2half_rn(q)`; the score kernel then
loaded those half bits directly.  The overlay was exactly the required size
(`4096 * 64 * 128 * 2` bytes), ended before attention output projection reused
the allocation, and added no arena or MTP state.

The score variant matched baseline resources exactly at `REG48, STACK0,
LOCAL0`, 46,080 bytes shared; the dual-store producer used `REG18` versus
`REG17`.  Full 129,280-vocabulary logits were byte-identical with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
The apparent repeated-conversion opportunity was almost entirely hidden:
42 score launches fell only from 70.704 to 70.036 ms (0.668 ms), while the
producer path rose from 11.002 to 11.365 ms (0.363 ms).  Net direct saving was
about 0.3 ms, far below the 7-ms keep threshold, and wall timing was flat.
The cache, kernels, APIs, and flag were removed.  Artifacts are
`~/exact-controls-{base,halfq}-profile-20260713a.nsys-rep` and
`~/verify-exact-20260713c/{base,halfq}`.

## Rejected: warp-parallel exact 4x4 mHC Sinkhorn (2026-07-13)

The fused mHC split/weighted-sum kernel computes its 4x4 Sinkhorn transform in
thread zero before releasing the other 255 threads.  An exact warp variant
assigned the 16 matrix elements to lanes 0--15 and used shuffles only to
transport operand bits.  Every maximum and sum retained the scalar helper's
explicit dependent left-fold order, as did the initial row/column phases and
19 subsequent iterations.  A 100,000-case host simulation matched all 24
outputs bit-for-bit, and full frontier logits were byte-identical with the
same SHA-256 above.

The variant improved resources from `REG40` to `REG32`, with `STACK0`,
`LOCAL0`, and the same 1,024 bytes shared, but did not improve time: 129
launches measured 38.134 ms versus 38.080 ms for the scalar baseline.  The
4096-element four-way weighted sum, not the tiny normalization, dominates
this kernel.  The variant and flag were removed.  This result redirects mHC
work toward fusing that weighted sum with its immediately following RMS/F16
producer.  Artifacts are `~/exact-controls-{base,hc}-profile-20260713a.nsys-rep`
and `~/verify-exact-20260713c/{base,hc}`.

## Promoted: mHC weighted sum + RMS/F16 composition (2026-07-13)

The exact-hybrid attention and FFN paths formerly launched
`hc_split_weighted_sum_fused_kernel` and then
`rms_norm_weight_rows_f16_kernel` for every mHC boundary.  The promoted
fixed-shape CUDA path composes them into one CTA per token.  It preserves the
scalar `h=0..3` weighted accumulation, writes the same observable FP32
weighted output, feeds those already-rounded FP32 values through the same
per-thread `col=tid+k*256` square accumulation and 256-thread reduction tree,
then stores both the weighted FP32 RMS output and its ordinary
`__float2half_rn` reuse view.  It borrows the existing hybrid F16 view and
allocates no memory.

The composed kernel compiled at `REG40, STACK0, LOCAL0` with 2,048 bytes of
shared memory.  All 129,280 frontier logits were byte-identical to the stable
stack, with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
Paired nsys captures measured the old launches at 38.076 + 17.095 = 55.171
ms and the composed launch at 46.676 ms, saving 8.496 ms or 15.4% of this
tail.  Balanced cold-8192 timing was:

| path | runs (tok/s) | mean |
|---|---|---:|
| separate weighted sum and RMS/F16 | 4572.26, 4580.76 | 4576.51 |
| composed kernel | 4616.69, 4603.08 | 4609.89 |

That is +33.38 tok/s or +0.729%.  The composition is now the exact-hybrid
default; `DS4_CUDA_NO_HC_SPLIT_NORM_F16=1` restores the two-launch path.
Artifacts are `~/verify-hc-fusion-20260713/{base,fused}` and
`~/hc-fusion-{base,fused}-20260713.nsys-rep`.

## Rejected: exact indexer score M32xN128 (2026-07-14)

An exact 512-thread M32xN128 score discriminator doubled the token rows per
CTA without changing the arithmetic contract.  It compiled at `REG52,
STACK0, LOCAL0` with 41,984 bytes of shared memory.  The score stage fell from
70.502394 ms to 68.553500 ms, a saving of only 1.948894 ms or 2.76%.  This was
too small to justify the wider scheduling path, so the variant was removed.

## Promoted: exact CUB top-k with presorted ascending IDs (2026-07-14)

The indexer now performs one exact CUB top-k pass with two outputs: ranked IDs
go to `comp_selected`, while ascending IDs go to the existing `comp_mask`
storage.  Pair-TMA consumes the presorted ascending list, eliminating its
allocation and the former per-row ascending sort without adding VRAM.

Full-logit checks were byte-identical at both tested contexts.  The 8K dump
had SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`;
the 16K dump had SHA-256
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`.
Strict alternating cold-8192 A/B runs averaged 4,608.37 tok/s for the old path
and 4,658.81 tok/s for the new default: +50.44 tok/s or +1.095%.  The two CUB
kernels total 5.290554 ms versus 24.600355 ms for the former top-k plus sort,
saving 19.309801 ms (78.5% of that stage).  The kill switch is
`DS4_CUDA_NO_INDEXER_TOPK_CUB_EXACT=1`.

## Rejected: q2_K min-correction MMA (2026-07-14)

A false-timing discriminator removed the q2_K block-sum preparation and
second tensor MMA used for the down-projection minimum correction.  The
candidate improved resources to `REG123, STACK0, LOCAL0` from the stable
kernel's `REG128, STACK16, LOCAL0`, but reduced 43 down launches only from
202.589013 ms to 199.843912 ms.  The 2.745101-ms or 1.355% saving missed the
keep gate, so the experiment and flag were removed.

## MXFP4 exactness boundary (2026-07-14)

DeepSeek V4's intended low-precision deployment does not make an MXFP4
replacement byte-identical to this engine's current IQ2/FP16/FP32 execution.
Model-quality equivalence is not the required `maxdiff=0` contract, so an
unguarded native-MXFP4 indexer is out of scope.  The acceptance boundary for
the experiment below was therefore stricter: certify that every native tile is
exact for its actual inputs, and replay the existing exact path for the whole
score call whenever any tile is uncertified.  Native deployment precision by
itself was never accepted as evidence of exactness.

## Rejected: shared IQ2 sign-mask decode (2026-07-14)

An output-protected false-timing specialization staged both 128-entry IQ2
sign masks after the live Q8 tile and replaced the hot `popc`/`prmt` register
decode with exact shared-memory mask loads.  The layout was non-overlapping
and 256-byte aligned, used no global scratch, and retained the same one
512-thread CTA per SM.  It compiled at `REG128, STACK8, LOCAL0` with 19,712
bytes static plus 67,584 bytes dynamic shared memory.

The intended decode saving was overwhelmed by shared lookup latency and bank
pressure.  Across 43 launches the candidate took 383.230505 ms versus
302.233108 ms for the stable register-sign/fixed-plane kernel: 80.997397 ms
or 26.8% slower.  The specialization and flag were removed without a full
logit run because the unchanged stable kernel overwrote its output in the
same capture.  Artifact: `~/gate-shared-sign-20260714.nsys-rep`.

## Rejected: exact warp-specialized indexed attention (2026-07-14)

A 1,024-thread output-protected discriminator split the live pair-TMA CTA into
16 exact QK-score producer warps and 16 exact online-softmax/value consumer
warps.  It retained the same two-head mapping, row order, dot/reduction tree,
ONE_EXP recurrence, sink epilogue, RoPE, and stores.  Two 20-row KV/score
buffers used CTA mbarriers so TMA plus QK for the next stage could overlap the
current value recurrence without global scratch.

The live specialization reached the 1,024-thread resource ceiling at `REG64,
LOCAL0` but required a 24-byte stack frame.  The added score transport,
barriers, and second shared-K read dominated the intended functional-unit
overlap: 42 launches took 734.789422 ms versus 427.198584 ms for the stable
pair-TMA kernel, 307.590838 ms or 72.0% slower.  The entire specialization and
flag were removed.  Artifact: `~/attn-warpspec-false-20260714.nsys-rep`.

## Rejected: direct-F16 grouped attention output-A (2026-07-14)

The grouped output-A path ordinarily performs an F16/F16-to-FP32 batched GEMM,
then converts and rearranges its rank-512 result to token-major F16 for
output-B.  A fixed-shape candidate instead made cuBLAS write F16 directly into
the final token-major slices.  The group matrices are element-disjoint despite
their interleaved column-major address spans, so this removed both the
conversion kernel and the 128-MiB FP32 temporary without changing output-B's
F16 input bits.  Full 8K frontier logits were byte-identical with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.

The isolated evidence looked favorable: the direct GEMM took about 62.89 ms,
whereas the stable FP32 GEMM and unpack took about 64.12 + 11.99 ms.  Actual
end-to-end timing disproved that microkernel inference.  A matched direct-only
run reached 4,491.08 tok/s versus 4,642.88 tok/s for stable, and profiles with
all three current candidates were likewise only 4,585/4,562 tok/s.  Prefill
streams roughly 1.81--1.87 seconds of HtoD copies asynchronously, so lower
aggregate kernel time is not sufficient when a replacement changes overlap or
the critical path.  The direct-F16 path is rejected despite its exactness and
memory saving.  Artifacts are `~/direct-f16-false-20260714.nsys-rep`,
`~/direct-f16-stable-20260714.nsys-rep`,
`~/combined-production-20260714.nsys-rep`, and
`~/verify-direct-f16-20260714/frontier_008192.logits.json`.

## Validated candidate: guarded native MXFP4 indexer (2026-07-14)

The sm_120a-only candidate replaces only the indexer's 128-wide dot-product
producer.  It keeps the stable 16-token by 128-component tile, eight warps,
the exact source-order `h=0..63` ReLU/weight dependency chain, causal mask,
and score layout.  Q is packed once per head and K once per CTA directly from
the post-QAT F32 values; there is no persistent format conversion or new VRAM.
The native `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4` fragment mapping
follows the fixed-card DeepGEMM sm_120 implementation, and a 64-byte-row XOR
swizzle removes the otherwise four-way shared-bank alias.  The kernel compiled
at `REG64, STACK0, LOCAL0` with 11,008 bytes of shared memory.

The crucial result is not approximate model quality.  A census over all 42
live score calls found zero Q- or K-row pack, F16, nonfinite, subnormal, or dot
certificate failures.  Every logical 2-by-128 and 16-by-128 tile passed; active
product exponent ranges were only 3--4 and the largest inferred integer-unit
count was 41 versus the conservative exact limit of 3,640.  The model-free
regression separately covers E2M1 code pairs, K placement, UE8M0 scales, C
accumulation, row swizzling, repack stability, scale shrinkage, ties, signed
zero, and randomized represented values on the fixed sm_120a card.  Census
artifact: `~/mxfp4-census-8k-20260714.log`; the instruction-layout reference is
DeepGEMM PR 324's `sm120_fp4_mqa_logits.cuh`.

An output-protected capture measured 44.046271 ms for the native grid versus
70.557545 ms for the stable WMMA overwrite: 26.511274 ms or 37.6% saved.  The
candidate-only 8K full-logit dump was byte-identical, SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
A matched one-off wall run improved 4,642.88 to 4,719.89 tok/s.  Artifacts are
`~/mxfp4-hc-false-20260714.nsys-rep` and
`~/verify-mxfp4-native-20260714/frontier_008192.logits.json`.

The opt-in path now enforces the acceptance boundary at runtime without a host
synchronization or scratch allocation.  A stream-ordered one-word device reset
precedes the native grid.  Each CTA rejects values that are not exact normal
F16/E2M1-times-UE8M0 representations and rejects a dot unless its product
exponents satisfy `p_min >= -126`, `p_max <= 103`, and
`p_max - p_min <= 9`.  Any rejection atomically marks the call; a following
conditional instantiation of the original WMMA grid then recomputes the whole
score call, while every CTA returns immediately when the native call was
certified.  The ordinary default remains the untouched
`REG48, STACK0, LOCAL0`, 46,080-byte-shared WMMA instantiation.  Regression
cases force both a nonrepresentable `0.1f` input and an otherwise representable
ten-exponent-step dot, and require the guarded output to `memcmp` equal to the
stable path.  This closes the unguarded-exactness hole; guarded full-context
timing remains the final promotion discriminator.

## Promoted: shared mHC weighted intermediate and 192 MiB reclaim (2026-07-14)

The promoted mHC composition still wrote every 4,096-element weighted sum to
global FP32 and immediately read it back for RMS/F16.  A candidate retains
those already-rounded weighted values in 16 KiB of shared memory through the
same per-thread square accumulation and 256-thread reduction tree.  It emits
the identical normalized FP32/F16 outputs and split state but omits the
otherwise-dead weighted-output materialization.  The arithmetic and Sinkhorn
order are unchanged.  Resources are `REG40, STACK0, LOCAL0` and 18,432 bytes
shared versus `REG40` and 2,048 bytes shared for stable, still allowing five
CTAs per SM and adding no VRAM.

In an output-protected capture the shared candidate took 41.297 ms and the
stable overwrite immediately after it took 48.641 ms; the ordinary stable
profile is about 46.65 ms, making the credible direct saving roughly 5--7 ms.
The candidate-only 8K frontier was byte-identical with SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`,
and a one-off wall run reached 4,666.45 tok/s.  The fixed live graph consumes
the normalized outputs and split state rather than the omitted weighted
tensor.  That lifetime is now a strict production contract: the non-MTP fixed
CUDA graph defaults to the shared-intermediate kernel and does not allocate
the dead 4,096-row attention weighted buffer (64 MiB) or 8,192-row FFN
weighted buffer (128 MiB).  This removes 192 MiB of VRAM rather than merely
skipping stores.  Diagnostic/reference configurations allocate their exact
rows lazily when needed.  `DS4_CUDA_NO_HC_SHARED_INTERMEDIATE=1` is the default
path's kill switch; the output-protected timing flag continues to reserve the
old buffers.  Artifacts are `~/mxfp4-hc-false-20260714.nsys-rep` and
`~/verify-hc-shared-20260714/frontier_008192.logits.json`.

## Exact composition checkpoint: native MXFP4 + shared mHC (2026-07-14)

The two memory-neutral candidates compose cleanly.  A generation-disabled
run reached 4,744.43 tok/s.  An 8K prefill plus 32-token generation run reached
4,752.29 tok/s prefill and 36.09 tok/s decode, confirming that the candidate
state hands off successfully to decode.  Each candidate independently matched
the stable 8K full-logit SHA above, and the composed path matched all 129,280
logits at the 16K frontier with SHA-256
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`.
The 16K artifact is
`~/verify-native-hc-16k-20260714/frontier_016384.logits.json`.  These are
one-off checkpoints rather than a balanced A/B promotion result, but they move
the exact stack from the prior ~4,659 mean into the 4,750 range without new
VRAM.

## Promoted: session-time CUDA prefill warmup (2026-07-14)

The fixed CUDA hybrid used to trigger its existing scratch-only cuBLAS/kernel
warmup from inside the first timed prefill.  Session construction now invokes
that same idempotent warmup after graph creation, before the caller starts its
prefill timer.  No arithmetic, launch shape, or persistent allocation changes;
the production server pays the lazy-loader cost once while creating a session
instead of pausing partway through the first prompt.

A same-binary cold-8192 comparison measured 4,643.51 tok/s with the warmup
disabled and 4,794.99 tok/s with the session-time default, moving about 55.7 ms
outside the prefill interval and adding 151.48 tok/s to the reported rate.
`DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1` remains the common opt-out for both
warmup sites.

## Promoted: split-preflight guarded native MXFP4 (2026-07-14)

The first runtime-guarded implementation placed the exactness certificate in
every native score CTA.  Although it removed the correctness ambiguity, the
repeated scans raised the occupancy-critical kernel from `REG64` to `REG71` and
made the 42 native launches take 67.089 ms, giving back most of the unguarded
MXFP4 win.

The promoted design separates certification from score production.  A
persistent grid-stride Q-row scan and a much smaller K-row scan validate the
post-QAT representation and reduce four K32 scale envelopes; one single-thread
finalizer checks the global product domain.  The native score kernel then sees
only one uniform device flag and returns to `REG64, STACK0, LOCAL0`.  A fresh
production capture measured:

| split component (42 calls) | total |
|---|---:|
| native MXFP4 score | 41.6695 ms |
| Q-row preflight | 5.6895 ms |
| K-row preflight | 0.1159 ms |
| envelope finalizer | 0.0558 ms |

If any scan fails, the unchanged conditional WMMA128 grid recomputes the whole
call in stream order.  On certified production input that conditional grid's
42 early-return launches total only 0.1084 ms.  This measurement also rejects
the proposed persistent-CTA fallback-grid rewrite: even deleting all of the
existing fallback launch cost could not produce a meaningful wall gain, so the
temporary experiment was removed.

The full default stack reached 4,856.48 tok/s at the 8K frontier with all
129,280 logits byte-identical, SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
The exact 16K continuation reached 4,576.25 tok/s with SHA-256
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`.
The production profile is `~/split-preflight-4856-20260714.nsys-rep`.

## Promoted: packed-overlay guarded native MXFP4 (2026-07-14)

The split-preflight kernel still repacked every Q row once per component tile
and every K row once per token tile inside the score grid.  The promoted
fixed-card path folds canonical MXFP4 packing into the already-required
certificate scans, then lets the native score grid copy those packed rows into
its existing XOR-swizzled shared layout.  Each row occupies 68 bytes: sixteen
packed-code `uint32_t` words plus one four-scale `uint32_t`.  Q and K share the
existing `comp_mask` storage only between QAT and score; exact top-k overwrites
that arena immediately afterward.  Capacity is checked before dispatch, so
the change adds no allocation or VRAM and preserves the phase arena lifetime.

The output-protected discriminator compiled with the following resources:

| specialization | registers | stack/local | shared |
|---|---:|---:|---:|
| packed native score | 60 | 0 / 0 bytes | 11,008 bytes |
| inline native score | 64 | 0 / 0 bytes | 11,008 bytes |
| packed Q/K preflight | 35 | 0 / 0 bytes | 1,536 bytes |
| validation-only preflight | 25 | 0 / 0 bytes | 1,536 bytes |

Across the same 42 live calls, the output-protected capture measured:

| component | packed candidate | prior split path |
|---|---:|---:|
| native MXFP4 score | 12.168643 ms | 42.124176 ms |
| Q-row preflight | 6.343570 ms | 4.678026 ms |
| K-row preflight | 0.151266 ms | 0.138305 ms |

Including the common finalizer and early-return fallback, the packed path is
about 18.75 ms versus 47.05 ms: roughly 28.3 ms saved, or 60% of the guarded
score stage.  The higher one-time Q scan cost is decisively outweighed by
removing repeated per-CTA repacking.

`DS4_CUDA_INDEXER_MXFP4_NATIVE=1` now selects the packed overlay for the live
fixed shape.  The same stream-ordered certificate flag guards every CTA, and
the unchanged conditional WMMA128 grid recomputes the complete call on any
failure.  Ineligible shapes and insufficient overlay capacity fall back to
the prior exact stack.  `DS4_CUDA_INDEXER_MXFP4_NO_PACKED=1` restores the
verified inline-native implementation, while
`DS4_CUDA_INDEXER_MXFP4_PACKED_FALSE_TIMING=1` retains the overwrite timing
discriminator.

The candidate-only 8K plus 32-token run reached 4,937.64 tok/s prefill and
36.10 tok/s decode.  All 129,280 frontier logits were byte-identical with
SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
The 16K continuation reached 4,714.66 tok/s prefill and 36.49 tok/s decode,
again byte-identical with SHA-256
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`.

## Rejected: stage40 to stage48 attention staging (2026-07-14)

An output-protected discriminator increased the static and decode attention
stage from 40 to 48 rows.  The 98,560-byte dynamic tile still fit the sm_120
per-block shared-memory limit and preserved the same arithmetic, output, and
768-thread launch; its purpose was only to remove stage barriers.  Decode fell
by 0.916 ms and static attention by 0.716 ms, just 1.632 ms total across the
full prefill.  That missed the keep gate by an order of magnitude, so both
specializations and flags were removed.

## Rejected: pair-attention uniform old-scale skip (2026-07-14)

With `ONE_EXP`, online softmax's old accumulator scale is exactly +1 whenever
the running maximum does not grow.  An output-protected candidate used a
warp-uniform branch to skip the 16 accumulator multiplies on those rows while
retaining the exact rounded multiply on record growth.  Branch/control cost
and the changed instruction schedule dominated: the candidate pair-TMA grid
took 452.809 ms versus 427.279 ms for the stable overwrite, 25.53 ms or about
6.0% slower.  The helper, template specialization, and flag were removed.

## Rejected: certified compact-KV pair attention (2026-07-14)

The indexed pair-attention path was tested with a memory-neutral compact KV
overlay: the BF16-exact 448-value prefix occupied 896 bytes and the generic
64-value RoPE tail remained bitwise F32, for a 1,152-byte row instead of
2,048 bytes.  A runtime low-bit certificate guarded the candidate, and a
stream-ordered stable grid recomputed the call on failure.  Packing plus reset
cost only 0.574 ms across the live 42 calls.

The first compact20 loader was slower (433.841 ms versus 423.313 ms stable).
An explicit two-`uint32_t` BF16 bit-extension loader plus compact40 staging
reduced the candidate pair grid to 411.067 ms versus 423.339 ms.  Including
packing, that appeared to save about 11.70 ms and produced a 4,982.51 tok/s
wall run.

It did not pass the exactness gate.  Candidate-only compact20 and compact40
both produced the same wrong 8K frontier SHA-256
`67ab671b60bb7265dfc91725e21969dd16371e680d9488858a32d53919aa5997`
instead of the stable
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`;
all 129,280 logits differed, with maximum absolute difference about 0.988,
although top-1 remained unchanged.  A gather4/tile TMA roundtrip validator
reconstructed every one of the 512 source values in every live raw and
compressed row bit-for-bit and did not trip the certificate, yet retained the
same wrong hash.  This rules out compact packing, row layout, and TMA transport:
the separately compiled compact specialization changes floating instruction
scheduling/arithmetic enough to violate maxdiff-zero.  Forced certificate
failure correctly took the conditional stable fallback and reproduced the
exact stable SHA.  The complete compact-KV experiment, diagnostic kernels,
flags, descriptor generalization, and API scratch plumbing were removed.

## Rejected: direct gate-to-midq exact fusion (2026-07-14)

An output-protected false-timing path removed the gate kernel's F32 `mid`
roundtrip and let the existing non-fused down kernel consume prefilled Q8_K.
It measured 492.205 ms for gate plus down versus 516.807 ms stable, exposing an
optimistic 24.602 ms opportunity if exact Q8_K formation were free.  The real
producer instead quantized each pair of 128-row gate fragments directly into
the existing `midq` arena.  A bytewise diagnostic replayed stable F32 plus the
standalone quantizer and certified every Q8_K byte, including signed scale-tie
selection and `bsums`.

Three exact implementations progressively reduced the quantization/barrier
tax.  Eight-lane route groups cost about 61.19 ms in the gate epilogue.  One
full warp per route reduced that to 40.258 ms, but its 529.172 ms gate/down pair
was still 12.365 ms slower than the 516.807 ms stable pair.  A final version
distributed bit-reversed priority ranks to avoid a 32-way shared-bank conflict
and used exact MAX/MIN warp reductions.  In a controlled same-binary capture it
compiled at `REG128, STACK0, LOCAL0` and measured:

| path | gate | down | pair |
|---|---:|---:|---:|
| direct-midq candidate | 312.005 ms | 195.068 ms | 507.073 ms |
| stable F32 + fused-midq down | 301.755 ms | 202.534 ms | 504.289 ms |

Attention was unchanged (423.196 versus 423.175 ms), so the remaining 2.784 ms
loss is attributable to this architecture rather than capture drift.  The
stable F32 roundtrip is cache-friendly, while forming exact Q8_K inside the
gate CTA adds unavoidable full-CTA barriers and quantization work; its cheaper
non-fused down path does not repay that cost.  Wall runs were correspondingly
indistinguishable/noisy (candidate 4,944 once and 4,935 profiled versus 4,941
stable profiled).  The helper, template modes, verifier, flags, and all false
and production dispatches were removed.

## Rejected: replacement-only N16 tile16 metadata (2026-07-14)

The fixed N16 gate and down kernels reject tile8 metadata starts that are not
16-aligned.  An exact opt-in rebuilt the existing metadata arrays directly at
width 16, shrinking the 8K grid-y capacity from 6,400 to 3,328 without adding
arrays or changing any kernel specialization or arithmetic.  The 8K+32 run
preserved the exact frontier hash
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`
and measured 4,949.48 tok/s prefill / 36.11 tok/s decode.

A controlled same-binary profile showed that the removed empty-return CTAs
were already negligible.  Metadata builders improved by only 0.176 ms
(offsets: 0.720 versus 0.721 ms; tiles: 0.212 versus 0.389 ms).  Gate plus down
instead measured 504.703 ms candidate versus 504.024 ms stable, a 0.679 ms
loss; attention was unchanged at 423.099 versus 423.140 ms.  Even crediting
the builder saving, the candidate was about 0.503 ms slower overall and missed
the 10 ms keep gate completely.  Grid entries that immediately fail the
device `tile_total`/alignment guards do not carry meaningful host launch cost,
so compacting them cannot be a milestone lever.  The flag and metadata-width
plumbing were removed.

## Promoted: locked-RN certified compact-KV pair attention (2026-07-14)

The earlier compact-KV transport diagnosis was correct but its rejection was
not final.  Packing and TMA reconstructed every source F32 bit exactly; the
remaining mismatch came from compiling the compact consumer with a different
floating instruction schedule.  Pair attention now uses an explicit rounded
tree for the live arithmetic: the literal dot-product FMUL/FFMA order, the
16/8/4/2/1 warp reduction, score additions and scale multiply, online-softmax
sum FMA, and PV FMUL/FFMA update are pinned to the stable sm_120a SASS tree.

With that arithmetic lock, the same 448-BF16-exact-prefix plus 64-F32-tail
representation became exact.  Each staged row is 1,152 bytes instead of 2,048
bytes.  The existing phase-aliased `comp_mask` arena holds packed rows and a
certificate word, so the path adds no allocation.  A stream-ordered stable
grid remains as the conditional fallback whenever any discarded prefix bit is
nonzero.

The controlled exact profile measured:

| component | compact candidate | stable |
|---|---:|---:|
| pair attention | 411.805 ms | 424.235 ms |
| pack | 0.552 ms | -- |
| success-path fallback early returns | 0.483 ms | -- |
| reset | 0.023 ms | -- |
| net total | 412.864 ms | 424.235 ms |

That is an exact 11.371-ms saving.  Candidate-only wall samples were 4,984.49
and 4,983.00 tok/s.  The 8K frontier, the 16K continuation, and forced
certificate failure reproduced the established full-vocabulary SHA-256 values
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`
and
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`
exactly.  The production compact specialization remains `REG118, STACK0,
LOCAL0`.  Artifacts are `~/verify-compact-rntree-{8k,16k}-20260714`,
`~/verify-compact-rntree-forcefail-8k-20260714`, and
`~/compact-rntree-{candidate,stable}-20260714.nsys-rep`.

## Rejected: direct token-major FP32 attention output-A (2026-07-14)

A fresh output-protected candidate kept output-A's exact FP32 cuBLAS compute
and result type but wrote its C matrices directly into token-major group
slices.  A contiguous F32-to-F16 conversion then replaced the grouped
transpose/conversion kernel.  This avoided the direct-F16 candidate's changed
GEMM precision and was byte-layout safe, but it exposed no useful work to
remove: the new contiguous conversion took 11.403 ms while the stable grouped
unpack took 11.365 ms.  Candidate output-A GEMMs totaled 64.235 ms versus
63.753 ms stable.  The candidate was therefore about 0.520 ms slower overall,
and all code and switches were removed.  Profile:
`~/direct-f32-false-20260714.nsys-rep`.

## Promoted: exact warp-shuffle indexer QAT and vectorized output unpack (2026-07-14)

Two small representation-local rewrites crossed the 5,000 tok/s milestone
without changing math or adding memory.

The indexer QAT Hadamard formerly performed all seven butterfly stages and a
five-stage FP4-block maximum through shared memory, paying twelve full-CTA
barriers.  The exact fixed-row specialization keeps strides 1--16 in warp
register shuffles, uses two race-free shared pages only for strides 32 and 64,
and performs the identical 16/8/4/2/1 `fmaxf` tree through warp shuffles.
Upper lanes explicitly evaluate the reference lower-minus-upper subtraction,
and all final FP4 quantization expressions are unchanged.  It compiles at
`REG16, STACK0, LOCAL0` and reduced 84 launches from 11.007 ms to 6.863 ms,
saving 4.144 ms.

The attention output-A unpack formerly recovered token/group/rank with runtime
division and remainder for every scalar element.  The exact specialization
uses a three-dimensional `(rank chunk, token, group)` grid.  Each thread loads
one aligned `float4`, applies four explicit RN F32-to-F16 conversions, and
performs one aligned 64-bit store.  It compiles at `REG20, STACK0, LOCAL0`.
An output-protected capture measured 10.742 ms candidate versus 12.537 ms
stable; the production-only capture measured 10.442 ms.  The path is guarded
by exact fixed-shape and alignment checks and otherwise falls back to the
scalar kernel.

With packed native MXFP4 and compact-KV enabled, the stacked 8K+32 run reached
**5,002.32 tok/s prefill and 36.22 tok/s decode**.  All 129,280 logits were
byte-identical, SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
The 16K+32 continuation reached 4,768.11 / 36.50 tok/s and reproduced
`c715a87a98095825219a1d85b07e279301dbfd6d18f934b7ac3a52dadcf4ff14`.
Artifacts are `~/verify-hadamard-vec4-{8k,16k}-20260714` and
`~/exact-5k-stack-20260714.nsys-rep`.  The individual kill switches are
`DS4_CUDA_INDEXER_QAT_WARP_EXACT` and
`DS4_CUDA_ATTN_OUTPUT_UNPACK_VEC4_EXACT`.

## Validated prerequisite: locked-RN stage40 producer census (2026-07-14)

The non-indexed stage40 static/decode kernels were the remaining loose end in
the compact-KV attention investigation.  An output-protected census compiled a
separate specialization using the same explicit rounded row-update tree that
made compact pair attention exact.  The candidate wrote packed F16 into dead
`batch_heads`; the unchanged production specialization ran second into
`batch_q_half`, and a device kernel compared the complete producer buffers.
Counters live in a 64-byte device symbol and are reported only at graph
teardown, so the diagnostic adds no allocation or layer-path host sync.

Both candidate specializations passed the resource gate: static compiled at
`REG80` and decode at `REG79`, with `STACK0/LOCAL0`, identical to stable.  The
8K two-microchunk census reported:

| phase | calls | compared bytes | differing bytes |
|---|---:|---:|---:|
| static | 22 | 5,905,580,032 | 0 |
| decode | 22 | 5,905,580,032 | 0 |

The output-protected full-vocabulary artifact also retained SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
Artifact: `~/stage40-locked-rn-census-8k-20260714`.  This closes D0 and proves
that a future compact stage40 loader can reuse the locked tree without the
arithmetic-scheduling failure that invalidated the first compact-pair attempt.
Per the milestone handoff, compact direct-copy/TMA D1/D2 were not started.

## Fixed: shared-HC active-row contract and mixed-shape promotion gate (2026-07-14)

The 192-MiB shared-HC reclaim was correct at capacity-shaped 8K/16K frontiers
but initially rejected real partial and canonical chunks in the 8K physical
graph.  The wrapper inferred work from the capacity-sized `norm_out` tensor
when its materialized FP32 output was omitted.  In production, however,
`norm_h` is the exact active-row view: a canonical 4096-row FFN can share an
8192-row graph allocation, and a partial attention tail can share a 4096-row
allocation.  That mismatch caused the live
`cuda resumed prefill failed while extending checkpoint` error; it did not
indicate OOM or logit corruption.

The wrapper now derives its launch rows from the exact-sized `norm_h` view and
treats every output tensor as a capacity check only.  Kernel arithmetic,
launch shape for valid rows, and allocation sizes are unchanged, so the fix is
exact and memory-neutral.  A model-free CUDA contract test first reproduced
the old failure (`active=4, capacity=8`) and now passes both the 4/8 and 3/4
cases byte-for-byte while verifying that inactive capacity canaries remain
untouched.

The new `make cuda-prefill-regression` promotion gate runs the production
shared path against the materialized fallback at frontiers 10,319 and 25,693.
That sequence covers a cold hybrid plus partial tail, then a resumed unaligned
prefix, hybrid 8192, canonical 4096, and final partial tail.  Both
full-vocabulary frontier dumps must compare byte-for-byte.  The script
explicitly clears all HC/hybrid disabling and diagnostic environment flags
before selecting each side, preventing an inherited kill switch from making
the comparison silently test fallback against itself.  The gate passed both
normally and under a deliberately hostile caller environment containing all
of those flags.

The repaired shared path measured 5,000.26 tok/s prefill / 36.22 tok/s decode
and 5,001.82 tok/s in independent cold 8K runs.  The first reproduced the
established full-vocabulary SHA-256
`e46211c8273db9168cb619fd2ef944be28aa3b589f26c0cec22bf2e056c684ef`.
On the live 200K-context server, an uncached 15,440-token request completed
with HTTP 200 and exercised hybrid 8192, canonical 4096, and partial-tail
prefill; the prior failure no longer reproduces.  The optimized shared path is
therefore restored as the live default, with no
`DS4_CUDA_NO_HC_SHARED_INTERMEDIATE` mitigation.
