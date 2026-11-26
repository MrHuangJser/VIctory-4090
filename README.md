# Victory-4090 (VIctory for NVIDIA RTX 4090 / Ada Lovelace)

A fork of [VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) (originally forked from [Marvisak/VIctory](https://github.com/Marvisak/VIctory) and [fxceless/victory-fork](https://github.com/fxceless/victory-fork)), engineered to port, verify, and surpass its AMD RDNA 3 optimizations on **NVIDIA Ada Lovelace (RTX 4090 / 4090 D, SM 8.9, 24 GB VRAM, Resizable BAR)** — including dedicated memory optimizations for **32 GB system RAM** hosts.

**This repository contains no game files, code or assets.** You need your own copy of the game. It is not affiliated with or endorsed by the game's developer or publisher.

## Key NVIDIA RTX 4090 & 32 GB RAM Enhancements

Full hardware probes, A/B microbenchmarks, and architectural rationale are in [docs/findings-4090.md](docs/findings-4090.md) (upstream RDNA 3 notes kept in [docs/findings.md](docs/findings.md)).

### GPU & Shader Optimizations
- **Hardware-verified NVIDIA Fast Float Lowering (`rdna::NVIDIA = MAX_MIN | LEGACY | MED3`)**: Empirically verified bit-for-bit across 144 float edge cases on RTX 4090 D; unlocks fast hardware `FMNMX` / `med3` / `legacy_mul` while keeping `FRACT` safely spelled out (NVIDIA `Frc(-1e-20)` returns `1.0`).
- **Vectorized 128-bit Raw Buffer Loads (`WIDE` / `LDG.E.128`)**: Automatically enabled on NVIDIA hosts (unlike AMD's scalar load unit, NVIDIA benefits from wider 128-bit vector loads and smaller DXIL IR).
- **Wide Indirect Compute Dispatches (`> 65,535` groups, up to `1 << 20`)**: Unlocked for NVIDIA (`0x10de`) after hardware verification on RTX 4090 D (`67,440`..`1,048,576` groups, 0 missed groups) — fixes 1440p/4K bottom-screen tile truncation.
- **2-Stage 8x8 Parallel Reduction in `CARD_COMPARE` (+ matching CPU `torn::finish`)**: Replaces the 63-step serial tail loop in thread 0 with a 2-barrier 8x8 parallel reduction (+4.2%–7.7% faster small/mid batches, **2,269 GB/s** in 4090 L2, **886 GB/s** in GDDR6X).
- **Resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) Window Offloading**: Measured **76.3 GB/s CPU write** and **453.4 GB/s GPU read (4.32x faster than system RAM `UPLOAD`)** on RTX 4090 D; auto-offloads up to 2 GiB of read-only windows directly into 4090 VRAM on 32 GB RAM hosts.
- **64-bit Word Bitset Dynamic Descriptor Allocator (`DynamicBits`)**: 64x denser hardware-accelerated (`trailing_zeros`/`trailing_ones`) slot search under lock, plus `VICTORY_DYNAMIC_SLOTS` support on NVIDIA's strict 1,000,000-view shader-visible heap.
- **NVIDIA Warp32 Barrier Safety (`VICTORY_GROUP_SYNC=1`)**: Prevents NVIDIA TDR hangs caused by divergent `s_barrier` calls while preserving memory fences and uniform barriers.

### 32 GB System RAM Profile (Auto-detected on `<= 36 GiB` RAM, `VICTORY_LOW_RAM=0/1`)
- Caps idle D3D12 command list allocators to `128` (down from `512` / ~9 GB) and actively trims oldest allocators down to `32`/`16` when free RAM drops below `6`/`3 GiB`.
- Moves 2 GiB of read-only windows into 4090 VRAM via ReBAR and tightens remaining system-RAM windows to a 1 MiB grid (`VICTORY_WINDOW_KB=1024`).
- Tightens scratch (`128 MiB`), send (`192 MiB`), and snapshot (`192 MiB`) pools with low-RAM sweep trimming.
- Caps NVIDIA pipeline compiler workers to `8` (down from `16`) to avoid multi-GB NVVM compile spikes during world loading.

## How to Build & Run

Requirements: Windows 10/11 x64, AVX2 CPU, NVIDIA RTX GPU (SM 6.6 + Mesh Shaders + Resizable BAR recommended), Rust (pinned toolchain), MSVC build tools, and your own copy of the title.

```powershell
copy victory.example.toml victory.toml      # set paths OUTSIDE this folder
cargo build --locked --profile diag -p victory --features diagnostics
scripts\play.ps1 -Exe <path of your copy's executable>
```

Useful `scripts\play.ps1` flags:
- `-Safe`: disables newest aggressive async/ReBAR paths for troubleshooting.
- `-RayTracing`: enables the title's ray-traced indirect lighting (`VICTORY_BVH_BUILDER=1`).
- `-LowRam`: forces the 32 GB RAM profile explicitly (already auto-detected on `<= 36 GiB` RAM hosts).
- `-Awake`: enables the async-compute clock keep-awake thread (off by default on NVIDIA).

Pre-translating shaders and warming pipelines ahead of a run:

```powershell
target\diag\victory.exe shaders <path-to-shaders.rpf>
target\diag\victory.exe warm --used
```

Running the RTX 4090 hardware verification & benchmark suite:

```powershell
cargo test --locked -p victory-shader --lib -- nvidia_forms_give_what_the_spelled_out_ones_give emit::wave64_tests::
cargo test --locked --release -p victory-gpu --lib -- card_runs_wide_indirect_dispatches dynamic_bits_tracks_runs_across_word_boundaries ram_idle_ms_scales_with_pressure
cargo test --locked --release -p victory-gpu --lib -- nvidia_ada_hardware_probe card_compare_costs indirect::bench --ignored --nocapture
```

## Credits

This fork builds directly on the community's work, keeping full git history and commit attribution:
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)**: the upstream fork this NVIDIA Ada / RTX 4090 fork builds upon.
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**: the original loader, runtime services, D3D12 backend, and shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**, **[DexNom/VIctory](https://github.com/DexNom/VIctory)**, **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**, **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**, **[myokex/victory-amd](https://github.com/myokex/victory-amd)**, **Geckomacabre**, **cxy0072019-sketch**, and **[XCL](https://github.com/ivall/xcl)** (reference only).

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE) and [docs/asset-policy.md](docs/asset-policy.md).
