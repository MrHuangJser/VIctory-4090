# Findings & Measurements on NVIDIA GeForce RTX 4090 D (Ada Lovelace)

Hardware tested:
- **GPU**: NVIDIA GeForce RTX 4090 D (Ada Lovelace, SM 8.9, 24,148 MiB VRAM, Driver 621.24, Resizable BAR enabled)
- **CPU**: AMD Ryzen 9 7900X3D (12 cores / 24 threads)
- **RAM**: 31.6 GiB (32 GB physical RAM)
- **OS**: Windows 11 x64

This fork starts from [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) (tuned for AMD RDNA 3 / RX 7900 XTX with 64 GB RAM). Below is the complete audit of what was AMD-specific, what failed or was disabled on NVIDIA, what we empirically measured on RTX 4090 D, and what was optimized for NVIDIA Ada + 32 GB RAM.

---

## 1. Hardware Capability Audit (RTX 4090 D vs RX 7900 XTX)

Measured via `indirect::tests::nvidia_ada_hardware_probe` and `emit::valu_tests::probe_host_fast_float_bits`:

| Capability | RX 7900 XTX (RDNA 3) | RTX 4090 D (Ada SM 8.9) | Impact in `Victory-4090` |
|---|---|---|---|
| Wave / Warp size (`WaveLaneCountMin..Max`) | 32..64 (native Wave64) | 32..32 (Wave32 only) | Wave64 compute lowers to Wave32; divergent `s_barrier` can TDR NVIDIA (`VICTORY_GROUP_SYNC=1`) |
| `ResourceBindingTier` | Tier 3 | Tier 3 | Full bindless SM 6.6 |
| Max `SHADER_VISIBLE` CBV/SRV/UAV heap | > 4,194,304 (`raw_twins` on) | **Strict 1,000,000** (`1,048,576` fails) | `DynamicBits` u64-bitset allocator + `VICTORY_DYNAMIC_SLOTS` unlocked for capped heaps |
| `ExecuteIndirect` `Dispatch(X > 65535)` | Supported (`1 << 20`) | **Supported (`1 << 20`, 0 errors!)** | Unlocked for NVIDIA (`0x10de`); fixes 1440p/4K bottom-tile truncation |
| `D3D12_HEAP_TYPE_GPU_UPLOAD` (ReBAR) | Supported (~10 GB/s CPU write) | **Supported (76.3 GB/s CPU write, 453.4 GB/s GPU read)** | Auto-offloads 2 GiB read-only windows to VRAM: **4.32x faster GPU reads** & **-2 GiB SysRAM** |
| Raw read past typed/structured view | No bounds check (page fault risk) | Hardware bounds check returns `0x0` | No need to hoard 1 GiB scratch / 5 min dead buffers on 32 GB RAM |
| Fast Float (`MAX_MIN`, `LEGACY`, `MED3`) | Exact (`rdna::ALL = 15`) | **Bit-exact (`rdna::NVIDIA = 7`)** | Unlocked on NVIDIA; saves 4–12 ALU ops per op |
| Fast Float (`FRACT`) | Exact | **Mismatch** (`-1e-20 -> 1.0` instead of `0x3f7fffff`) | Safely kept OFF on NVIDIA |
| 128-bit `WIDE` raw loads (`LDG.E.128`) | 0 ms gain (AMD fuses scalar loads) | Auto-enabled on NVIDIA | Emits 128-bit vector raw loads natively |

---

## 2. Empirical Microbenchmarks on RTX 4090 D

### 2.1 Wide Indirect Compute Dispatches (`> 65,535` groups in X)

Upstream `VIctory-7900XTX` gated `widest()` behind `VendorId == 0x1002` (AMD only), truncating NVIDIA indirect compute dispatches to `65,535` groups (losing bottom 8x4 screen tiles in 1440p/4K tile lists of `67,440` and `115,200` groups).

Direct hardware test on RTX 4090 D (`indirect::tests::card_runs_wide_indirect_dispatches`):

| Asked Groups (`X, 1, 1`) | Executed Groups | Max `SV_GroupID.x` | Missed/Duplicate Groups |
|---:|---:|---:|---|
| 65,535 | 65,535 | 65,534 | `None` |
| 67,440 | 67,440 | 67,439 | `None` |
| 115,200 | 115,200 | 115,199 | `None` |
| 262,144 | 262,144 | 262,143 | `None` |
| 1,048,576 (`1 << 20`) | 1,048,576 | 1,048,575 | `None` |

Both single-record clamp and batched clamp (`VICTORY_INDIRECT_BATCH=2`) now pass `card_runs_wide_indirect_dispatches` on NVIDIA.

### 2.2 Indirect Draw Barriers & Batch Clamping

`indirect::bench::bench_indirect_draw_barriers` (3,200 draws/frame, median of 9 frames) and `indirect::tests::batch_costs` (2,355 indirect dispatches) on RTX 4090 D:

- **Windowed indirect draw clamping + card budget**: `34.80 ms -> 0.82 ms` (**-97.6% GPU time**, `0.26 us/draw`)
- **Batched indirect dispatch clamping** (`batch every 48` vs `each on its own`): `8.84 ms -> 0.61 ms` (**14.5x faster**)

### 2.3 Fast Float Lowering Verification (`valu_tests::probe_host_fast_float_bits`)

144 float edge cases per instruction (`0.0, -0.0, 1.0, -1.0, 0.3, -2.5e-7, -1e-20, -1e-38, 3e38, +inf, -inf, NaN`) run directly on RTX 4090 D:

- `MAX_MIN` (`v_max_f32`, `v_min_f32`, `v_max3_f32`, `v_min3_f32`): **0 mismatches, 0 signed-zero diffs**
- `MED3` (`v_med3_f32` between ordered constants `0..1`, `1..0`, `1..-1`): **0 mismatches, 0 signed-zero diffs**
- `LEGACY` (`v_mul_legacy_f32`, `v_mad_legacy_f32`, all 6 variants): **0 mismatches** (same zero-sign relaxation as RDNA)
- `FRACT` (`v_fract_f32`): **12 mismatches** (`-1e-20` returns `1.0` instead of `0x3f7fffff`) — **kept disabled on NVIDIA**

`victory-shader` now exposes `set_host_nvidia(true)` which enables `rdna::NVIDIA` (`MAX_MIN | LEGACY | MED3 = 7`) plus 128-bit `WIDE` raw loads (`LDG.E.128`), verified permanently by `emit::valu_tests::nvidia_forms_give_what_the_spelled_out_ones_give`.

### 2.4 Real Game Shader Pack (`shaders.rpf`) Translation Check

Full translation of all 26,503 containers (26,084 programs, 40.6M IR instructions) in Wave32 mode (`VICTORY_HOST_MAX_WAVE=32 VICTORY_HOST_NVIDIA=1`):
- Vertex + Pixel: **14,247 / 14,247** (100%)
- Mesh: **8,765 / 8,765** (100%)
- Compute: **2,677 / 2,682** (99.81%; same 5 dynamic-indexed BVH shadow-mask shaders as upstream)
- Zero regressions with `rdna::NVIDIA` + `WIDE` enabled.

### 2.5 GPU Memory Hash Pass (`CARD_COMPARE`) Reduction Tuning

A/B measurement of `indirect::tests::card_compare_costs` on RTX 4090 D:

| Workload | Original (63-step serial) | 6-step Tree (6 barriers) | **2-Stage 8x8 (2 barriers)** |
|---|---:|---:|---:|
| 80x5 passes x 1 piece (25 MB) | 0.50 ms (49 GB/s) | 0.49 ms (49 GB/s) | **0.47 ms (52 GB/s, -6.0%)** |
| 80x5 passes x 5 pieces (125 MB) | 0.52 ms (237 GB/s) | 0.54 ms (225 GB/s) | **0.48 ms (252 GB/s, -7.7%)** |
| 80x5 passes x 64 pieces (1600 MB, L2) | 0.72 ms (2174 GB/s) | 0.70 ms (2219 GB/s) | **0.69 ms (2269 GB/s, -4.2%)** |
| 80x5 passes x 500 pieces (12500 MB) | 13.80 ms (885 GB/s) | 16.07 ms (760 GB/s) | **13.78 ms (886 GB/s)** |

The 6-barrier binary tree hurt occupancy on 2-warp NVIDIA threadgroups at 500 pieces; switching to a **2-barrier 8x8 parallel reduction** (with matching CPU `torn::finish`) beat the original across every workload with bit-for-bit CPU/GPU hash equality.

### 2.6 Resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) Bandwidth

`indirect::tests::nvidia_ada_hardware_probe` (64 MB CPU write, 512 MB GPU compute read):

| Heap Type | CPU Write (64 MB) | GPU Read (512 MB) | System RAM Used |
|---|---:|---:|---:|
| `D3D12_HEAP_TYPE_UPLOAD` (System RAM) | 0.64 ms (98.2 GB/s) | 4.77 ms (104.9 GB/s) | 64 MB |
| `D3D12_HEAP_TYPE_GPU_UPLOAD` (ReBAR VRAM) | 0.82 ms (76.3 GB/s) | **1.10 ms (453.4 GB/s, 4.32x)** | **0 MB** |

---

## 3. 32 GB Physical RAM Optimizations (`tight_ram()`, auto-enabled on <= 36 GiB RAM)

Upstream `VIctory-7900XTX` used **40–46 GB private RAM** on a 64 GB PC (9 GB idle command allocators, 7 GB loose windows, 4 GB uploads, plus peak shader compiler allocations). On a 32 GB RAM + RTX 4090 (24 GB VRAM) machine, `Victory-4090` automatically activates a tighter memory profile (`VICTORY_LOW_RAM=0/1` overrides):

1. **Idle Command List Allocators (`list.rs`)**: Default idle pool reduced `512 -> 128` (`VICTORY_LIST_POOL_MAX`), and `list::trim_idle` evicts oldest/largest allocators from the front down to `32` (`< 6 GiB` free RAM) or `16` (`< 3 GiB` free RAM) every second (~6–7 GB saved).
2. **ReBAR Window Offload + 1 MiB Grid (`shadow.rs`)**: Automatically offloads up to `2048 MiB` of read-only windows into RTX 4090 VRAM via ReBAR (`VICTORY_WINDOWS_BAR_MB`, **-2 GiB SysRAM** & **4.32x faster GPU reads**) and tightens remaining system-RAM windows from 4 MiB to 1 MiB grid (`VICTORY_WINDOW_KB=1024`, ~1.5–2 GB saved).
3. **Scratch, Send & Snapshot Pools (`host.rs`, `host/send.rs`, `shadow.rs`)**: Caps spare scratch chunks to `128 MiB` (was `1 GiB` under AMD's `keep_freed`), send pool to `192 MiB` (was `768 MiB`), snapshot pool to `192 MiB` (was `512 MiB`), tightens `ram_idle_ms` thresholds (`16 GiB` keep bar, `8/4 GiB` pressure tiers), and trims spare pools under `< 6 GiB` free RAM (~2.5–3.5 GB saved).
4. **NVIDIA Pipeline Compiler Concurrency (`pipeline/make.rs`)**: Caps default pipeline worker threads to `8` (was `16`) on NVIDIA + `<= 36 GiB` RAM to prevent 16 concurrent NVVM compiles from spiking 12–16 GB RAM on cold cache (~6 GB peak saved).
5. **Bitset Dynamic Descriptor Allocator (`descriptor/host.rs`)**: Replaced byte-by-byte `Vec<bool>` scan with 64-bit word `DynamicBits` (TZCNT/LZCNT) and honored `VICTORY_DYNAMIC_SLOTS` on NVIDIA's 1,000,000-view capped heap.
