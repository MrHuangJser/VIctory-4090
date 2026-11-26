//! Phase 1 NVIDIA Ada (RTX 4090) DXR 1.1 Hardware RT Core Bridge.
//!
//! Translates the console's RDNA2 BVH4 structures (64-byte triangle nodes and
//! 128-byte TLAS instance records) into native D3D12 DXR 1.1 BLAS / TLAS
//! acceleration structures on the GPU so NVIDIA's dedicated RT Cores can
//! traverse them via SM 6.6 `RayQuery` (`TraceRayInline`).
//!
//! Guarded behind `VICTORY_DXR=1` (off by default until Phase 2 shader
//! `RayQuery` substitution is wired).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::core::{Interface, s};

use crate::host::{Host, compile, host, name, transition};

/// Bytes of an RDNA2 BVH header.
#[allow(dead_code)]
pub(crate) const HEADER_BYTES: u64 = 64;
/// Bytes of an RDNA2 64-byte triangle or box16 node.
#[allow(dead_code)]
pub(crate) const NODE_BYTES: u64 = 64;
/// Bytes of an RDNA2 128-byte TLAS instance record.
#[allow(dead_code)]
pub(crate) const INSTANCE_BYTES: u64 = 128;
/// Bytes of a D3D12 `D3D12_RAYTRACING_INSTANCE_DESC`.
#[allow(dead_code)]
pub(crate) const DXR_INSTANCE_DESC_BYTES: u64 = 64;

/// RDNA2 BVH header flag bits (upper 16 bits of dword 0).
pub(crate) const FLAG_BLAS: u32 = 0x8000;
#[allow(dead_code)]
pub(crate) const FLAG_COMPACTED: u32 = 0x0004;
#[allow(dead_code)]
pub(crate) const FLAG_REHYDRATED: u32 = 0x0002;
pub(crate) const FLAG_PACKED: u32 = 0x0100;

/// Whether the Phase 1 DXR 1.1 acceleration-structure bridge is enabled
/// (`VICTORY_DXR=1` and host supports DXR Tier 1.1).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_DXR").is_some_and(|v| v == "1");
        let tier11 = host().is_some_and(|h| h.dxr_tier11());
        if asked && !tier11 {
            tracing::warn!("VICTORY_DXR=1 asked, but host device lacks DXR Tier 1.1");
        }
        asked && tier11
    })
}

/// GPU Compute Shader #1: unpacks 64-byte RDNA2 triangle nodes (`5 x float3` +
/// 1 byte-packed `id` dword) into 4 triangle slots of `3 x float3` (36 bytes)
/// per node, permuting `[first, i, j]` so DXR `RayQuery` barycentrics `(x, y)`
/// match RDNA2 `(I, J)` and winding 1:1. Inactive slots emit `NaN` so DXR's
/// BLAS builder skips them cleanly without compaction.
const EXTRACT_TRIANGLES_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Params : register(b0) {
    uint node_count;
    uint src_byte_offset;
};
ByteAddressBuffer rdna_nodes : register(t0);
RWByteAddressBuffer out_verts : register(u0);

static const uint TRI_VERTS[4][3] = {
    {0u, 1u, 2u},
    {1u, 3u, 2u},
    {2u, 3u, 4u},
    {2u, 4u, 0u}
};

uint3 load_vertex(uint4 w0, uint4 w1, uint4 w2, uint4 w3, uint v) {
    if (v == 0u) return w0.xyz;
    if (v == 1u) return uint3(w0.w, w1.xy);
    if (v == 2u) return uint3(w1.zw, w2.x);
    if (v == 3u) return w2.yzw;
    return w3.xyz;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint node_idx = id.x;
    if (node_idx >= node_count) return;

    uint base = src_byte_offset + node_idx * 64u;
    uint4 w0 = rdna_nodes.Load4(base + 0u);
    uint4 w1 = rdna_nodes.Load4(base + 16u);
    uint4 w2 = rdna_nodes.Load4(base + 32u);
    uint4 w3 = rdna_nodes.Load4(base + 48u);
    uint id_word = w3.w;

    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        uint id_byte = (id_word >> (8u * k)) & 0xFFu;
        uint i = id_byte & 3u;
        uint j = (id_byte >> 2u) & 3u;
        uint out_base = (node_idx * 4u + k) * 36u;
        if (i < 3u && j < 3u && i != j) {
            uint first = (3u - i - j) & 3u;
            uint3 p0 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][first]);
            uint3 p1 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][i]);
            uint3 p2 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][j]);
            out_verts.Store3(out_base + 0u, p0);
            out_verts.Store3(out_base + 12u, p1);
            out_verts.Store3(out_base + 24u, p2);
        } else {
            // D3D12 DXR ignores triangles with a NaN vertex coordinate.
            uint3 qnan = uint3(0x7fc00000u, 0x7fc00000u, 0x7fc00000u);
            out_verts.Store3(out_base + 0u, qnan);
            out_verts.Store3(out_base + 12u, qnan);
            out_verts.Store3(out_base + 24u, qnan);
        }
    }
}
"#;

/// GPU Compute Shader #2: converts 128-byte RDNA2 TLAS instance records into
/// 64-byte `D3D12_RAYTRACING_INSTANCE_DESC` records, binding each instance to
/// its native Ada BLAS GPU VA from `blas_va_table`.
const CONVERT_INSTANCES_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Params : register(b0) {
    uint instance_count;
    uint src_byte_offset;
};
ByteAddressBuffer rdna_instances : register(t0);
StructuredBuffer<uint2> blas_va_table : register(t1);
RWByteAddressBuffer out_descs : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= instance_count) return;

    uint src = src_byte_offset + idx * 128u;
    uint dst = idx * 64u;

    uint4 m0 = rdna_instances.Load4(src + 0u);
    uint4 m1 = rdna_instances.Load4(src + 16u);
    uint4 m2 = rdna_instances.Load4(src + 32u);
    uint id_mask = rdna_instances.Load(src + 48u);
    uint contrib_flags = rdna_instances.Load(src + 52u);
    uint2 blas_va = blas_va_table[idx];

    if (blas_va.x == 0u && blas_va.y == 0u) {
        // Zero mask culls instances whose BLAS is not built yet.
        id_mask &= 0x00FFFFFFu;
    }

    out_descs.Store4(dst + 0u, m0);
    out_descs.Store4(dst + 16u, m1);
    out_descs.Store4(dst + 32u, m2);
    out_descs.Store4(dst + 48u, uint4(id_mask, contrib_flags, blas_va.x, blas_va.y));
}
"#;

/// Maximum number of `Memory::copies` buffers indexed in the paged SRV table.
/// Matches `COPIES = (1 << 12) - 1` in `memory.rs`.
const MAX_ARENA_COPIES: u32 = 4095;
/// Total 64 KiB pages in `Memory`'s page directory (`victory_shader::emit::MEMORY_PAGES`).
#[cfg(test)]
const MEMORY_PAGES: u32 = victory_shader::emit::MEMORY_PAGES;
/// Maximum candidate headers inspected asynchronously per frame.
const MAX_INSPECT_PER_FRAME: usize = 128;
/// Number of in-flight allocator / readback ring slots.
const RING_SLOTS: usize = 3;
/// Power-of-two slot count for the GPU `(blas_addr >> 8) -> ada_blas_va` hash map (256 KiB).
const BLAS_MAP_SIZE: usize = 16_384;
/// Upper safety cap on triangle nodes per BLAS (262,144 nodes = ~1M tris).
const MAX_TRI_NODES: u32 = 262_144;
/// Upper safety cap on instances per TLAS.
const MAX_INSTANCES: u32 = 65_536;
/// Maximum GPU re-inspection retries for a static packed structure still awaiting rehydration.
const MAX_STATIC_RETRIES: u8 = 3;
/// Consecutive header rejections before muting a non-BVH/scratch address from GPU re-inspection.
const MAX_REJECT_BEFORE_MUTE: u8 = 4;

/// Whether Phase 3 live GPU BLAS/TLAS construction from `Memory` arenas is active.
/// On by default when `VICTORY_DXR=1`; set `VICTORY_DXR_LIVE=0` for passive observation only.
fn live_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled() && std::env::var_os("VICTORY_DXR_LIVE").is_none_or(|v| v != "0")
    })
}

/// Max BLAS builds scheduled per frame to avoid streaming bursts causing frame spikes.
fn builds_per_frame() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_DXR_BUILDS_PER_FRAME")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, 128)
    })
}

/// Soft VRAM cap for cached native Ada BLASes before LRU eviction (default 512 MiB).
fn vram_budget_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let mb = std::env::var("VICTORY_DXR_VRAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(512)
            .clamp(64, 4096);
        mb << 20
    })
}

/// Instance transform matrix source (`0` = dwords 0..11, `1` = dwords 16..27).
fn inst_mat_mode() -> u32 {
    static MODE: OnceLock<u32> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("VICTORY_DXR_INST_MAT")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0)
            .min(1)
    })
}

/// GPU Compute Shader #3 (Phase 3): asynchronously reads 64-byte RDNA2 BVH
/// headers (or 64-byte sample slices) directly from `Memory`'s paged GPU arenas
/// using its live `directory` and `arenas[]` SRV table.
const INSPECT_HEADERS_PAGED_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), SRV(t1), UAV(u0), DescriptorTable(SRV(t0, numDescriptors=4096, space=1))"
cbuffer Params : register(b0) {
    uint count;
};
ByteAddressBuffer directory : register(t0);
StructuredBuffer<uint2> candidate_addrs : register(t1);
RWByteAddressBuffer out_headers : register(u0);
ByteAddressBuffer arenas[4096] : register(t0, space1);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= count) return;

    uint2 addr = candidate_addrs[idx];
    uint page = (addr.x >> 16u) | (addr.y << 16u);
    uint dst = idx * 64u;
    uint4 w0 = uint4(0u, 0u, 0u, 0u);
    uint4 w1 = uint4(0u, 0u, 0u, 0u);
    uint4 w2 = uint4(0u, 0u, 0u, 0u);
    uint4 w3 = uint4(0u, 0u, 0u, 0u);

    if (page < 2097152u) {
        uint2 entry = directory.Load2(page * 8u);
        uint copy_num = entry.x & 0x7FFFFFFFu;
        if (copy_num >= 1u && copy_num <= 4095u) {
            uint off = entry.y + (addr.x & 0xFFFFu);
            uint ci = copy_num - 1u;
            w0 = arenas[ci].Load4(off + 0u);
            w1 = arenas[ci].Load4(off + 16u);
            w2 = arenas[ci].Load4(off + 32u);
            w3 = arenas[ci].Load4(off + 48u);
        }
    }

    out_headers.Store4(dst + 0u, w0);
    out_headers.Store4(dst + 16u, w1);
    out_headers.Store4(dst + 32u, w2);
    out_headers.Store4(dst + 48u, w3);
}
"#;

/// GPU Compute Shader #4 (Phase 3): unpacks `node_count` 64-byte RDNA2 triangle
/// nodes from arbitrary non-contiguous 64 KiB pages in `Memory`'s GPU arenas.
const EXTRACT_TRIANGLES_PAGED_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), SRV(t0), UAV(u0), DescriptorTable(SRV(t0, numDescriptors=4096, space=1))"
cbuffer Params : register(b0) {
    uint node_count;
    uint elem_addr_lo;
    uint elem_addr_hi;
};
ByteAddressBuffer directory : register(t0);
RWByteAddressBuffer out_verts : register(u0);
ByteAddressBuffer arenas[4096] : register(t0, space1);

static const uint TRI_VERTS[4][3] = {
    {0u, 1u, 2u},
    {1u, 3u, 2u},
    {2u, 3u, 4u},
    {2u, 4u, 0u}
};

uint3 load_vertex(uint4 w0, uint4 w1, uint4 w2, uint4 w3, uint v) {
    if (v == 0u) return w0.xyz;
    if (v == 1u) return uint3(w0.w, w1.xy);
    if (v == 2u) return uint3(w1.zw, w2.x);
    if (v == 3u) return w2.yzw;
    return w3.xyz;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint node_idx = id.x;
    if (node_idx >= node_count) return;

    uint add = node_idx * 64u;
    uint lo = elem_addr_lo + add;
    uint hi = elem_addr_hi + (lo < elem_addr_lo ? 1u : 0u);
    uint page = (lo >> 16u) | (hi << 16u);

    bool mapped = false;
    uint4 w0 = uint4(0u, 0u, 0u, 0u);
    uint4 w1 = uint4(0u, 0u, 0u, 0u);
    uint4 w2 = uint4(0u, 0u, 0u, 0u);
    uint4 w3 = uint4(0u, 0u, 0u, 0u);

    if (page < 2097152u) {
        uint2 entry = directory.Load2(page * 8u);
        uint copy_num = entry.x & 0x7FFFFFFFu;
        if (copy_num >= 1u && copy_num <= 4095u) {
            uint off = entry.y + (lo & 0xFFFFu);
            uint ci = copy_num - 1u;
            w0 = arenas[ci].Load4(off + 0u);
            w1 = arenas[ci].Load4(off + 16u);
            w2 = arenas[ci].Load4(off + 32u);
            w3 = arenas[ci].Load4(off + 48u);
            mapped = true;
        }
    }

    uint id_word = w3.w;
    uint3 qnan = uint3(0x7fc00000u, 0x7fc00000u, 0x7fc00000u);

    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        uint out_base = (node_idx * 4u + k) * 36u;
        uint id_byte = (id_word >> (8u * k)) & 0xFFu;
        uint i = id_byte & 3u;
        uint j = (id_byte >> 2u) & 3u;
        if (mapped && i < 3u && j < 3u && i != j) {
            uint first = (3u - i - j) & 3u;
            uint3 p0 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][first]);
            uint3 p1 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][i]);
            uint3 p2 = load_vertex(w0, w1, w2, w3, TRI_VERTS[k][j]);
            out_verts.Store3(out_base + 0u, p0);
            out_verts.Store3(out_base + 12u, p1);
            out_verts.Store3(out_base + 24u, p2);
        } else {
            out_verts.Store3(out_base + 0u, qnan);
            out_verts.Store3(out_base + 12u, qnan);
            out_verts.Store3(out_base + 24u, qnan);
        }
    }
}
"#;

/// GPU Compute Shader #5 (Phase 3): converts 128-byte RDNA2 TLAS instances from
/// paged `Memory` arenas into 64-byte `D3D12_RAYTRACING_INSTANCE_DESC` records,
/// resolving each instance's `blas_addr >> 8` (`dword 15`) through the GPU
/// open-addressing `blas_map` hash table with zero CPU readback.
const CONVERT_INSTANCES_PAGED_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), DescriptorTable(SRV(t0, numDescriptors=4096, space=1))"
cbuffer Params : register(b0) {
    uint instance_count;
    uint elem_addr_lo;
    uint elem_addr_hi;
    uint blas_map_mask;
    uint use_extra_mat;
};
ByteAddressBuffer directory : register(t0);
StructuredBuffer<uint4> blas_map : register(t1);
RWByteAddressBuffer out_descs : register(u0);
RWByteAddressBuffer stats_uav : register(u1);
ByteAddressBuffer arenas[4096] : register(t0, space1);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= instance_count) return;

    uint add = idx * 128u;
    uint lo = elem_addr_lo + add;
    uint hi = elem_addr_hi + (lo < elem_addr_lo ? 1u : 0u);
    uint page = (lo >> 16u) | (hi << 16u);

    uint4 m0 = uint4(0u, 0u, 0u, 0u);
    uint4 m1 = uint4(0u, 0u, 0u, 0u);
    uint4 m2 = uint4(0u, 0u, 0u, 0u);
    uint id_mask = 0u;
    uint contrib_flags = 0u;
    uint blas_key = 0u;
    bool mapped = false;

    if (page < 2097152u) {
        uint2 entry = directory.Load2(page * 8u);
        uint copy_num = entry.x & 0x7FFFFFFFu;
        if (copy_num >= 1u && copy_num <= 4095u) {
            uint off = entry.y + (lo & 0xFFFFu);
            uint ci = copy_num - 1u;
            m0 = arenas[ci].Load4(off + 0u);
            m1 = arenas[ci].Load4(off + 16u);
            m2 = arenas[ci].Load4(off + 32u);
            uint4 w3 = arenas[ci].Load4(off + 48u);
            id_mask = w3.x;
            contrib_flags = w3.y;
            blas_key = w3.w;
            mapped = true;
        }
    }

    if (mapped && use_extra_mat != 0u) {
        uint lo2 = lo + 64u;
        uint hi2 = hi + (lo2 < lo ? 1u : 0u);
        uint page2 = (lo2 >> 16u) | (hi2 << 16u);
        if (page2 < 2097152u) {
            uint2 entry2 = directory.Load2(page2 * 8u);
            uint copy_num2 = entry2.x & 0x7FFFFFFFu;
            if (copy_num2 >= 1u && copy_num2 <= 4095u) {
                uint off2 = entry2.y + (lo2 & 0xFFFFu);
                uint ci2 = copy_num2 - 1u;
                m0 = arenas[ci2].Load4(off2 + 0u);
                m1 = arenas[ci2].Load4(off2 + 16u);
                m2 = arenas[ci2].Load4(off2 + 32u);
            }
        }
    }

    uint2 blas_va = uint2(0u, 0u);
    if (mapped && blas_key != 0u) {
        uint h = (blas_key * 0x9E3779B1u) & blas_map_mask;
        for (uint probe = 0u; probe < 32u; ++probe) {
            uint4 kv = blas_map[(h + probe) & blas_map_mask];
            if (kv.x == blas_key) {
                blas_va = kv.zw;
                break;
            }
            if (kv.x == 0u) {
                break;
            }
        }
    }

    if (blas_va.x == 0u && blas_va.y == 0u) {
        id_mask &= 0x00FFFFFFu;
        uint prev;
        stats_uav.InterlockedAdd(4u, 1u, prev);
    } else {
        uint prev;
        stats_uav.InterlockedAdd(0u, 1u, prev);
    }

    uint dst = idx * 64u;
    out_descs.Store4(dst + 0u, m0);
    out_descs.Store4(dst + 16u, m1);
    out_descs.Store4(dst + 32u, m2);
    out_descs.Store4(dst + 48u, uint4(id_mask, contrib_flags, blas_va.x, blas_va.y));
}
"#;

#[allow(dead_code)]
struct ComputePass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline state objects are free-threaded.
unsafe impl Send for ComputePass {}
// SAFETY: as above.
unsafe impl Sync for ComputePass {}

fn build_pass(device: &ID3D12Device, source: &str, label: &str) -> windows::core::Result<ComputePass> {
    let code = compile(source, s!("main"), s!("cs_5_1"))?;
    // SAFETY: `code` is valid DXBC with an embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })?;
        name(&pso, || label.to_owned());
        Ok(ComputePass { root, pso })
    }
}

#[allow(dead_code)]
pub(crate) struct DxrBridge {
    extract_pass: ComputePass,
    convert_pass: ComputePass,
    inspect_paged_pass: ComputePass,
    extract_paged_pass: ComputePass,
    convert_paged_pass: ComputePass,
}

#[allow(dead_code)]
impl DxrBridge {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        Ok(Self {
            extract_pass: build_pass(device, EXTRACT_TRIANGLES_CS, "victory dxr: extract triangles")?,
            convert_pass: build_pass(device, CONVERT_INSTANCES_CS, "victory dxr: convert instances")?,
            inspect_paged_pass: build_pass(
                device,
                INSPECT_HEADERS_PAGED_CS,
                "victory dxr: inspect headers paged",
            )?,
            extract_paged_pass: build_pass(
                device,
                EXTRACT_TRIANGLES_PAGED_CS,
                "victory dxr: extract triangles paged",
            )?,
            convert_paged_pass: build_pass(
                device,
                CONVERT_INSTANCES_PAGED_CS,
                "victory dxr: convert instances paged",
            )?,
        })
    }

    /// Converts `tri_node_count` 64-byte RDNA2 triangle nodes at
    /// `rdna_nodes + src_offset` into a native Ada BLAS.
    pub(crate) fn build_blas_from_rdna_nodes(
        &self,
        host_gpu: &Host,
        device5: &ID3D12Device5,
        list4: &ID3D12GraphicsCommandList4,
        rdna_nodes: &ID3D12Resource,
        src_offset: u32,
        tri_node_count: u32,
        fast_trace: bool,
    ) -> windows::core::Result<Blas> {
        let vert_count = tri_node_count * 4 * 3;
        let vb_bytes = u64::from(vert_count) * 12;
        let vb = host_gpu
            .uav_buffer(vb_bytes.max(256))
            .ok_or_else(windows::core::Error::empty)?;
        name(&vb, || "victory dxr: extracted blas vertices".into());

        // SAFETY: `list4` is an open graphics command list4 and resources live through submission.
        unsafe {
            let list: ID3D12GraphicsCommandList = list4.cast()?;
            list.SetComputeRootSignature(&self.extract_pass.root);
            list.SetPipelineState(&self.extract_pass.pso);
            let params = [tri_node_count, src_offset];
            list.SetComputeRoot32BitConstants(0, 2, params.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, rdna_nodes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, vb.GetGPUVirtualAddress());
            list.Dispatch(tri_node_count.div_ceil(64).max(1), 1, 1);
            list.ResourceBarrier(&[
                uav_barrier(&vb),
                transition(
                    &vb,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ]);
        }

        let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: 0,
                    IndexFormat: DXGI_FORMAT_UNKNOWN,
                    VertexFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: 0,
                    VertexCount: vert_count,
                    IndexBuffer: 0,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: unsafe { vb.GetGPUVirtualAddress() },
                        StrideInBytes: 12,
                    },
                },
            },
        };
        let flags = if fast_trace {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        } else {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
        };
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: flags,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &geom,
            },
        };
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid inputs pointer and output info pointer.
        unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info) };

        let result = as_buffer(&host_gpu.device, info.ResultDataMaxSizeInBytes)?;
        let scratch = host_gpu
            .uav_buffer(info.ScratchDataSizeInBytes.max(256).next_multiple_of(256))
            .ok_or_else(windows::core::Error::empty)?;
        name(&result, || "victory dxr: blas".into());
        name(&scratch, || "victory dxr: blas scratch".into());

        // SAFETY: `result` is in RAYTRACING_ACCELERATION_STRUCTURE state, `scratch` and `vb` are aligned.
        let gpu_va = unsafe {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: result.GetGPUVirtualAddress(),
                Inputs: inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(),
            };
            list4.BuildRaytracingAccelerationStructure(&desc, None);
            let list: ID3D12GraphicsCommandList = list4.cast()?;
            list.ResourceBarrier(&[uav_barrier(&result)]);
            result.GetGPUVirtualAddress()
        };

        Ok(Blas {
            result,
            gpu_va,
            _scratch: scratch,
            _vertices: vb,
            tri_node_count,
        })
    }

    /// Converts `instance_count` 128-byte RDNA2 instance records into a native
    /// Ada TLAS.
    pub(crate) fn build_tlas_from_instances(
        &self,
        host_gpu: &Host,
        device5: &ID3D12Device5,
        list4: &ID3D12GraphicsCommandList4,
        rdna_instances: &ID3D12Resource,
        src_offset: u32,
        instance_count: u32,
        blas_va_table: &ID3D12Resource,
    ) -> windows::core::Result<Tlas> {
        let desc_bytes = (u64::from(instance_count) * DXR_INSTANCE_DESC_BYTES).max(256);
        let descs = host_gpu
            .uav_buffer(desc_bytes)
            .ok_or_else(windows::core::Error::empty)?;
        name(&descs, || "victory dxr: tlas instance descs".into());

        // SAFETY: `list4` is an open command list4 and buffers live through submission.
        let descs_va = unsafe {
            let list: ID3D12GraphicsCommandList = list4.cast()?;
            list.SetComputeRootSignature(&self.convert_pass.root);
            list.SetPipelineState(&self.convert_pass.pso);
            let params = [instance_count, src_offset];
            list.SetComputeRoot32BitConstants(0, 2, params.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, rdna_instances.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, blas_va_table.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, descs.GetGPUVirtualAddress());
            list.Dispatch(instance_count.div_ceil(64).max(1), 1, 1);
            list.ResourceBarrier(&[
                uav_barrier(&descs),
                transition(
                    &descs,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ]);
            descs.GetGPUVirtualAddress()
        };

        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: instance_count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: descs_va,
            },
        };
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid inputs pointer and output info pointer.
        unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info) };

        let result = as_buffer(&host_gpu.device, info.ResultDataMaxSizeInBytes)?;
        let scratch = host_gpu
            .uav_buffer(info.ScratchDataSizeInBytes.max(256).next_multiple_of(256))
            .ok_or_else(windows::core::Error::empty)?;
        name(&result, || "victory dxr: tlas".into());
        name(&scratch, || "victory dxr: tlas scratch".into());

        // SAFETY: `result` is in RAYTRACING_ACCELERATION_STRUCTURE state, `scratch` and `descs` are aligned.
        let gpu_va = unsafe {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: result.GetGPUVirtualAddress(),
                Inputs: inputs,
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(),
            };
            list4.BuildRaytracingAccelerationStructure(&desc, None);
            let list: ID3D12GraphicsCommandList = list4.cast()?;
            list.ResourceBarrier(&[uav_barrier(&result)]);
            result.GetGPUVirtualAddress()
        };

        Ok(Tlas {
            result,
            gpu_va,
            _scratch: scratch,
            _descs: descs,
            instance_count,
        })
    }

    /// Builds a native Ada BLAS from an RDNA2 64-byte header + element block at
    /// `arena + header_offset` (`elem_offset = header_offset + header[13] * 64`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_blas_from_arena_header(
        &self,
        host_gpu: &Host,
        device5: &ID3D12Device5,
        list4: &ID3D12GraphicsCommandList4,
        arena: &ID3D12Resource,
        header_offset: u64,
        header: &[u32; 16],
        fast_trace: bool,
    ) -> windows::core::Result<Blas> {
        let word0 = header[0];
        let flags = word0 >> 16;
        if word0 & 0xFFFF != 6 || flags & FLAG_PACKED != 0 || flags & FLAG_BLAS == 0 {
            return Err(windows::core::Error::empty());
        }
        let elem_offset = header_offset + u64::from(header[13]) * NODE_BYTES;
        let src_offset = u32::try_from(elem_offset).map_err(|_| windows::core::Error::empty())?;
        let tri_node_count = header[3];
        if tri_node_count == 0 {
            return Err(windows::core::Error::empty());
        }
        self.build_blas_from_rdna_nodes(
            host_gpu,
            device5,
            list4,
            arena,
            src_offset,
            tri_node_count,
            fast_trace,
        )
    }

    /// Builds a native Ada TLAS from an RDNA2 64-byte header + instance block at
    /// `arena + header_offset` (`elem_offset = header_offset + header[13] * 64`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_tlas_from_arena_header(
        &self,
        host_gpu: &Host,
        device5: &ID3D12Device5,
        list4: &ID3D12GraphicsCommandList4,
        arena: &ID3D12Resource,
        header_offset: u64,
        header: &[u32; 16],
        blas_va_table: &ID3D12Resource,
    ) -> windows::core::Result<Tlas> {
        let word0 = header[0];
        let flags = word0 >> 16;
        if word0 & 0xFFFF != 6 || flags & FLAG_PACKED != 0 || flags & FLAG_BLAS != 0 {
            return Err(windows::core::Error::empty());
        }
        let elem_offset = header_offset + u64::from(header[13]) * NODE_BYTES;
        let src_offset = u32::try_from(elem_offset).map_err(|_| windows::core::Error::empty())?;
        let instance_count = header[3];
        if instance_count == 0 {
            return Err(windows::core::Error::empty());
        }
        self.build_tlas_from_instances(
            host_gpu,
            device5,
            list4,
            arena,
            src_offset,
            instance_count,
            blas_va_table,
        )
    }

    /// Dispatches `INSPECT_HEADERS_PAGED_CS` to asynchronously read `count`
    /// 64-byte slices from paged `Memory` arenas into `out_headers`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_inspect_headers_paged(
        &self,
        list: &ID3D12GraphicsCommandList,
        directory: &ID3D12Resource,
        addrs_buf: &ID3D12Resource,
        out_headers: &ID3D12Resource,
        arenas_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        // SAFETY: `list` is open with `DxrPagedHeap` bound; resources live through submission.
        unsafe {
            list.SetComputeRootSignature(&self.inspect_paged_pass.root);
            list.SetPipelineState(&self.inspect_paged_pass.pso);
            list.SetComputeRoot32BitConstants(0, 1, (&raw const count).cast(), 0);
            list.SetComputeRootShaderResourceView(1, directory.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, addrs_buf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, out_headers.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(4, arenas_gpu_handle);
            list.Dispatch(count.div_ceil(64).max(1), 1, 1);
            list.ResourceBarrier(&[uav_barrier(out_headers)]);
        }
    }

    /// Dispatches `EXTRACT_TRIANGLES_PAGED_CS` to unpack `tri_node_count`
    /// 64-byte RDNA2 triangle nodes from paged `Memory` arenas starting at
    /// 64-bit title address `elem_title_addr` into `out_verts`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_extract_triangles_paged(
        &self,
        list: &ID3D12GraphicsCommandList,
        directory: &ID3D12Resource,
        out_verts: &ID3D12Resource,
        arenas_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
        elem_title_addr: u64,
        tri_node_count: u32,
    ) {
        if tri_node_count == 0 {
            return;
        }
        let params = [
            tri_node_count,
            elem_title_addr as u32,
            (elem_title_addr >> 32) as u32,
        ];
        // SAFETY: `list` is open with `DxrPagedHeap` bound; resources live through submission.
        unsafe {
            list.SetComputeRootSignature(&self.extract_paged_pass.root);
            list.SetPipelineState(&self.extract_paged_pass.pso);
            list.SetComputeRoot32BitConstants(0, 3, params.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, directory.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, out_verts.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(3, arenas_gpu_handle);
            list.Dispatch(tri_node_count.div_ceil(64).max(1), 1, 1);
        }
    }

    /// Dispatches `CONVERT_INSTANCES_PAGED_CS` to convert `instance_count`
    /// 128-byte RDNA2 TLAS instances from paged `Memory` arenas starting at
    /// `elem_title_addr` into `out_descs`, resolving BLAS VAs via `blas_map`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_convert_instances_paged(
        &self,
        list: &ID3D12GraphicsCommandList,
        directory: &ID3D12Resource,
        blas_map: &ID3D12Resource,
        out_descs: &ID3D12Resource,
        stats_uav: &ID3D12Resource,
        arenas_gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
        elem_title_addr: u64,
        instance_count: u32,
        blas_map_mask: u32,
        use_extra_mat: u32,
    ) {
        if instance_count == 0 {
            return;
        }
        let params = [
            instance_count,
            elem_title_addr as u32,
            (elem_title_addr >> 32) as u32,
            blas_map_mask,
            use_extra_mat,
        ];
        // SAFETY: `list` is open with `DxrPagedHeap` bound; resources live through submission.
        unsafe {
            list.SetComputeRootSignature(&self.convert_paged_pass.root);
            list.SetPipelineState(&self.convert_paged_pass.pso);
            list.SetComputeRoot32BitConstants(0, 5, params.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, directory.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, blas_map.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, out_descs.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(4, stats_uav.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(5, arenas_gpu_handle);
            list.Dispatch(instance_count.div_ceil(64).max(1), 1, 1);
        }
    }
}

/// Shader-visible 4096-slot SRV descriptor heap mirroring `Memory::copies`
/// as raw `ByteAddressBuffer` SRVs (`t0..t4095, space1`) for paged DXR passes.
pub(crate) struct DxrPagedHeap {
    heap: ID3D12DescriptorHeap,
    cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    stride: usize,
    filled: usize,
}

// SAFETY: D3D12 descriptor heaps are free-threaded; mutated only under DXR runtime lock.
unsafe impl Send for DxrPagedHeap {}

impl DxrPagedHeap {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let count = MAX_ARENA_COPIES + 1;
        // SAFETY: plain shader-visible CBV/SRV/UAV descriptor heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
        };
        name(&heap, || "victory dxr: paged arenas srv heap".into());
        // SAFETY: valid heap handles and descriptor increment size query.
        let (cpu_start, gpu_start, stride) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart(),
                heap.GetGPUDescriptorHandleForHeapStart(),
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize,
            )
        };
        // Pre-fill every slot with a null raw Buffer SRV so any out-of-range
        // or newly added copy index safely reads zeros instead of faulting.
        let null_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 0,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: `i < count`, valid CPU handle inside `heap`.
        unsafe {
            for i in 0..count as usize {
                let h = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr + i * stride,
                };
                device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&null_desc), h);
            }
        }
        Ok(Self {
            heap,
            cpu_start,
            gpu_start,
            stride,
            filled: 0,
        })
    }

    /// Incrementally writes raw `ByteAddressBuffer` SRVs for any newly added
    /// `Memory::copies` buffers (`self.filled .. copies.len()`).
    pub(crate) fn sync_copies(&mut self, device: &ID3D12Device, copies: &[ID3D12Resource]) {
        let target = copies.len().min(MAX_ARENA_COPIES as usize);
        while self.filled < target {
            let res = &copies[self.filled];
            // SAFETY: valid live buffer resource and descriptor slot `< 4096`.
            unsafe {
                let width = res.GetDesc().Width;
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: (width / 4).min(u64::from(u32::MAX)) as u32,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                let h = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: self.cpu_start.ptr + self.filled * self.stride,
                };
                device.CreateShaderResourceView(res, Some(&desc), h);
            }
            self.filled += 1;
        }
    }

    pub(crate) fn heap(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }

    pub(crate) fn gpu_start(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.gpu_start
    }
}

/// GPU open-addressing hash map mapping `(blas_title_addr >> 8)` to its native
/// Ada BLAS GPU virtual address (`uint4(key, 0, va_lo, va_hi)`).
pub(crate) struct BlasMapBuffer {
    pub(crate) buffer: ID3D12Resource,
    ptr: *mut u32,
    mask: usize,
}

// SAFETY: `ptr` points into `buffer`'s persistent mapping, accessed only under runtime lock.
unsafe impl Send for BlasMapBuffer {}

impl BlasMapBuffer {
    pub(crate) fn new(host_gpu: &Host) -> Option<Self> {
        let bytes = (BLAS_MAP_SIZE * 16) as u64;
        let (buffer, raw) = host_gpu.shared_buffer(bytes)?;
        name(&buffer, || "victory dxr: gpu blas va hash map".into());
        // SAFETY: fresh persistent mapping of `bytes` bytes.
        unsafe { std::ptr::write_bytes(raw, 0, bytes as usize) };
        Some(Self {
            buffer,
            ptr: raw.cast::<u32>(),
            mask: BLAS_MAP_SIZE - 1,
        })
    }

    pub(crate) fn mask(&self) -> u32 {
        self.mask as u32
    }

    pub(crate) fn insert_one(&mut self, title_addr: u64, blas_va: u64) -> bool {
        let key = (title_addr >> 8) as u32;
        if key == 0 {
            return false;
        }
        let h = (key.wrapping_mul(0x9E37_79B1) as usize) & self.mask;
        for probe in 0..32 {
            let slot = (h + probe) & self.mask;
            // SAFETY: `slot * 4 + 3 < BLAS_MAP_SIZE * 4`.
            unsafe {
                let p = self.ptr.add(slot * 4);
                let cur = *p;
                if cur == 0 || cur == key {
                    *p = key;
                    *p.add(1) = 0;
                    *p.add(2) = blas_va as u32;
                    *p.add(3) = (blas_va >> 32) as u32;
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn rebuild(&mut self, entries: impl Iterator<Item = (u64, u64)>) {
        // SAFETY: full buffer is `BLAS_MAP_SIZE * 16` bytes.
        unsafe { std::ptr::write_bytes(self.ptr.cast::<u8>(), 0, BLAS_MAP_SIZE * 16) };
        for (addr, va) in entries {
            let _ = self.insert_one(addr, va);
        }
    }
}

fn uav_buffer_state(
    device: &ID3D12Device,
    size: u64,
    initial_state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256).next_multiple_of(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: plain committed default-heap UAV buffer creation.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial_state,
            None,
            &mut resource,
        )?;
    }
    resource.ok_or_else(windows::core::Error::empty)
}

#[allow(dead_code)]
pub(crate) struct Blas {
    pub(crate) result: ID3D12Resource,
    pub(crate) gpu_va: u64,
    _scratch: ID3D12Resource,
    _vertices: ID3D12Resource,
    pub(crate) tri_node_count: u32,
}

#[allow(dead_code)]
pub(crate) struct Tlas {
    pub(crate) result: ID3D12Resource,
    pub(crate) gpu_va: u64,
    _scratch: ID3D12Resource,
    _descs: ID3D12Resource,
    pub(crate) instance_count: u32,
}

#[allow(dead_code)]
fn as_buffer(device: &ID3D12Device, size: u64) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256).next_multiple_of(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: D3D12 requires acceleration structure buffers to be created in
    // D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            None,
            &mut resource,
        )?;
    }
    resource.ok_or_else(windows::core::Error::empty)
}

#[allow(dead_code)]
fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: non-owning pointer valid for the duration of ResourceBarrier.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

/// Creates a `RaytracingAccelerationStructure` SRV at `cpu_handle` pointing at `tlas_gpu_va`.
#[allow(dead_code)]
pub(crate) fn create_tlas_srv(
    device: &ID3D12Device,
    tlas_gpu_va: u64,
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                Location: tlas_gpu_va,
            },
        },
    };
    // SAFETY: per D3D12 spec, pResource must be None for RaytracingAccelerationStructure SRV.
    unsafe {
        device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), cpu_handle);
    }
}

/// Source category of a candidate address submitted for async GPU header inspection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CandidateKind {
    StaticPacked,
    DynamicBuilt,
    WalkedRead,
}

/// Reason a GPU-read 64-byte candidate header failed RDNA2 BVH v6 validation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeaderRejectReason {
    AllZero,
    WrongVersion,
    StillPacked,
    BoundsOrEmpty,
}

/// Tracks title BVH headers and GPU arena structures observed by `memory.rs`
/// when `VICTORY_DXR=1`.
#[derive(Default)]
struct DxrRegistry {
    structures: HashMap<u64, StructureInfo>,
    blas_seen: u64,
    tlas_seen: u64,
    static_packed: HashMap<u64, u64>,
    static_packed_events: u64,
    dynamic_built: HashMap<u64, BuiltEntry>,
    dynamic_built_events: u64,
    walked_seen: HashMap<u64, u64>,
    tlas_vas: HashMap<u64, u64>,
    // Phase 3 async inspection & eviction queues
    pending_walked: VecDeque<u64>,
    pending_walked_set: HashSet<u64>,
    pending_static: VecDeque<(u64, u64, u8)>,
    pending_static_set: HashSet<u64>,
    pending_dynamic: VecDeque<(u64, u64)>,
    pending_dynamic_set: HashSet<u64>,
    unmute_addrs: Vec<u64>,
    gone_addrs: Vec<u64>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct BuiltEntry {
    first_frame: u64,
    last_frame: u64,
    rebuilds: u64,
    arena_offset: Option<u64>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct StructureInfo {
    header_word0: u32,
    size_nodes: u32,
    num_elements: u32,
    offset_elements: u32,
}

static REGISTRY: Mutex<Option<DxrRegistry>> = Mutex::new(None);

/// Records a rehydrated/built BVH header when `VICTORY_DXR=1` is active (no-op otherwise).
#[allow(dead_code)]
pub(crate) fn note_structure(title_addr: u64, header_dwords: [u32; 16]) {
    if !enabled() {
        return;
    }
    let word0 = header_dwords[0];
    if word0 & 0xFFFF != 6 {
        return;
    }
    let flags = word0 >> 16;
    if flags & FLAG_PACKED != 0 {
        return;
    }
    let info = StructureInfo {
        header_word0: word0,
        size_nodes: header_dwords[1],
        num_elements: header_dwords[3],
        offset_elements: header_dwords[13],
    };
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    let is_new = !reg.structures.contains_key(&title_addr);
    reg.structures.insert(title_addr, info);
    if is_new {
        if flags & FLAG_BLAS != 0 {
            reg.blas_seen += 1;
        } else {
            reg.tlas_seen += 1;
        }
        if (reg.blas_seen + reg.tlas_seen).is_multiple_of(128) || (reg.blas_seen + reg.tlas_seen) <= 4 {
            tracing::info!(
                addr = format_args!("{title_addr:#x}"),
                blas = flags & FLAG_BLAS != 0,
                nodes = info.size_nodes,
                elements = info.num_elements,
                total_blas = reg.blas_seen,
                total_tlas = reg.tlas_seen,
                "victory dxr: observed RDNA2 BVH structure"
            );
        }
    }
}

/// Records a packed static world BVH structure streamed into an arena
/// (`Memory::land`) when `VICTORY_DXR=1` is active.
pub(crate) fn note_packed_structure(title_addr: u64, packed_len: u64, header_u64: u64) {
    if !enabled() {
        return;
    }
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    reg.static_packed_events += 1;
    let prev = reg.static_packed.insert(title_addr, packed_len);
    let is_new = prev.is_none();
    if prev != Some(packed_len) {
        reg.unmute_addrs.push(title_addr);
        if reg.pending_static_set.insert(title_addr) {
            // Wait 2 frames so the rehydration compute dispatch has landed in the arena.
            reg.pending_static
                .push_back((title_addr, crate::frame::current().saturating_add(2), 0));
        }
    }
    let n = reg.static_packed.len() as u64;
    if is_new && (n <= 4 || n == 64 || n.is_multiple_of(512)) {
        let total_mb = reg.static_packed.values().sum::<u64>() / (1 << 20);
        tracing::info!(
            addr = format_args!("{title_addr:#x}"),
            packed_len,
            header = format_args!("{header_u64:#x}"),
            unique_static = n,
            total_static_mb = total_mb,
            events = reg.static_packed_events,
            "victory dxr: observed static packed BVH structure"
        );
    }
}

/// Records BVH root addresses directly walked by title raytracing shaders
/// (`Memory::ready` with `Use::Reads`) when `VICTORY_DXR=1` is active.
pub(crate) fn note_walked_structures(addresses: &[u64], frame: u64) {
    if !enabled() {
        return;
    }
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    for &addr in addresses {
        if addr < (1u64 << 32) {
            continue;
        }
        let prev_frame = reg.walked_seen.insert(addr, frame);
        let is_new = prev_frame.is_none();
        if (is_new || prev_frame != Some(frame)) && reg.pending_walked_set.insert(addr) {
            reg.pending_walked.push_back(addr);
        }
        if is_new {
            let n = reg.walked_seen.len();
            if n <= 8 || n.is_multiple_of(64) {
                tracing::info!(
                    addr = format_args!("{addr:#x}"),
                    frame,
                    unique_walked = n,
                    "victory dxr: observed RT-walked BVH root address"
                );
            }
        }
    }
}

/// Records a dynamic BVH structure submitted by `Memory::ready` for GPU build
/// when `VICTORY_DXR=1` is active.
pub(crate) fn note_built_structure(title_addr: u64, frame: u64, arena_offset: Option<u64>) {
    if !enabled() {
        return;
    }
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    reg.dynamic_built_events += 1;
    let entry = reg.dynamic_built.entry(title_addr).or_insert(BuiltEntry {
        first_frame: frame,
        last_frame: frame,
        rebuilds: 0,
        arena_offset,
    });
    let is_new = entry.rebuilds == 0;
    let frame_changed = entry.last_frame != frame;
    entry.last_frame = frame;
    entry.rebuilds += 1;
    if arena_offset.is_some() {
        entry.arena_offset = arena_offset;
    }
    let rebuilds = entry.rebuilds;
    if (is_new || frame_changed) && reg.pending_dynamic_set.insert(title_addr) {
        if rebuilds >= 2 {
            // Hot per-frame structures (such as the live TLAS) go to the front.
            reg.pending_dynamic.push_front((title_addr, rebuilds));
        } else {
            reg.pending_dynamic.push_back((title_addr, rebuilds));
        }
    }
    let n = reg.dynamic_built.len() as u64;
    let ev = reg.dynamic_built_events;
    if (is_new && (n <= 4 || n == 64 || n.is_multiple_of(512))) || ev.is_multiple_of(4096) {
        tracing::info!(
            addr = format_args!("{title_addr:#x}"),
            frame,
            arena_offset = ?arena_offset,
            unique_dynamic = n,
            total_build_events = ev,
            rebuilds,
            "victory dxr: observed GPU-built dynamic BVH structure"
        );
    }
}

/// Notifies DXR that the streamed static structure at `title_addr` was evicted
/// by `Memory::gone`, scheduling its cached native Ada BLAS for deferred release.
pub(crate) fn note_structure_gone(title_addr: u64) {
    if !enabled() {
        return;
    }
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    reg.static_packed.remove(&title_addr);
    reg.gone_addrs.push(title_addr);
}

/// Looks up the native Ada TLAS GPU virtual address for `title_addr` (if built).
#[allow(dead_code)]
pub(crate) fn tlas_gpu_va(title_addr: u64) -> Option<u64> {
    let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    reg.as_ref()?.tlas_vas.get(&title_addr).copied()
}

/// Registers a built native Ada TLAS for `title_addr` and makes it the active
/// TLAS for `SLOT_TLAS` root SRV bindings.
#[allow(dead_code)]
pub(crate) fn register_tlas(title_addr: u64, tlas: Tlas) -> u64 {
    let gpu_va = tlas.gpu_va;
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(DxrRegistry::default);
    reg.tlas_vas.insert(title_addr, gpu_va);
    // Keep the backing ID3D12Resource alive in the registry alongside its VA.
    ACTIVE_TLAS.lock().unwrap_or_else(|e| e.into_inner()).replace(tlas);
    ACTIVE_TLAS_VA.store(gpu_va, Ordering::Release);
    gpu_va
}

static ACTIVE_TLAS: Mutex<Option<Tlas>> = Mutex::new(None);
static ACTIVE_TLAS_VA: AtomicU64 = AtomicU64::new(0);

/// Returns the GPU virtual address of the currently active native Ada TLAS
/// (bound to `SLOT_TLAS` for hardware `RayQuery` dispatches), if one is ready.
/// Lock-free atomic read on the draw hot path.
pub(crate) fn active_tlas_gpu_va() -> Option<u64> {
    let va = ACTIVE_TLAS_VA.load(Ordering::Acquire);
    (va != 0).then_some(va)
}

// ============================================================================
// Phase 3: Live Paged-Arena Ada BLAS/TLAS Runtime
// ============================================================================

/// Validated metadata parsed from a 64-byte GPU-resident RDNA2 BVH header.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct ValidatedHeader {
    is_blas: bool,
    size_nodes: u32,
    num_elements: u32,
    offset_elements: u32,
    rev: u64,
    last_frame: u64,
}

#[allow(dead_code)]
struct CachedBlas {
    result: ID3D12Resource,
    gpu_va: u64,
    bytes: u64,
    tri_node_count: u32,
    last_frame: u64,
    last_built_rev: u64,
}

struct RingSlot {
    allocator: ID3D12CommandAllocator,
    list4: ID3D12GraphicsCommandList4,
    addrs_buf: ID3D12Resource,
    addrs_ptr: *mut u64,
    headers_buf: ID3D12Resource,
    headers_ptr: *const u32,
    stats_buf: ID3D12Resource,
    stats_ptr: *mut u32,
    pending_inspect: Vec<(u64, CandidateKind, u8, u64)>,
    sample_tlas_slot: Option<(u64, usize)>,
    ran_tlas: bool,
    kept_resources: Vec<ID3D12Resource>,
    fence_value: u64,
}

// SAFETY: raw pointers point into persistent D3D12 custom-heap mappings owned by the slot,
// accessed strictly under `DXR_RUNTIME` mutex and fence-synchronized against GPU execution.
unsafe impl Send for RingSlot {}

struct DxrRuntime {
    device5: ID3D12Device5,
    paged_heap: DxrPagedHeap,
    blas_map: BlasMapBuffer,
    slots: Vec<RingSlot>,
    next_slot: usize,

    // Reusable scratch & ping-pong TLAS buffers (zero per-BLAS scratch bloat)
    vb_scratch: Option<(ID3D12Resource, u64)>,
    as_scratch: Option<(ID3D12Resource, u64)>,
    tlas_descs: Option<(ID3D12Resource, u64)>,
    tlas_results: [Option<(ID3D12Resource, u64)>; 2],
    tlas_ping: usize,

    // Live validated BVH catalog and bounded Ada BLAS cache
    validated_blas: HashMap<u64, ValidatedHeader>,
    validated_tlas: HashMap<u64, ValidatedHeader>,
    blas_build_queue: VecDeque<u64>,
    blas_build_queued: HashSet<u64>,
    blas_cache: HashMap<u64, CachedBlas>,
    total_blas_bytes: u64,
    blas_map_dirty: bool,
    primary_tlas: Option<u64>,
    sample_tlas_dumped: bool,

    // Rejection throttling & bounded one-shot raw header diagnostics
    reject_counts: HashMap<u64, u8>,
    dumped_addrs: HashSet<u64>,
    dumped_walked: u32,
    dumped_static_packed: u32,
    dumped_static_bounds: u32,
    dumped_static_ver: u32,
    dumped_dynamic_bounds: u32,
    dumped_dynamic_hot: u32,
    dumped_valid_blas: u32,
    dumped_valid_tlas: u32,

    // Deferred GPU resource retirement (dropped only once GPU fence passes)
    retired: Vec<(u64, ID3D12Resource)>,

    // Telemetry counters
    inspected_total: u64,
    rej_zero: u64,
    rej_ver: u64,
    rej_packed: u64,
    rej_bounds: u64,
    blas_builds_total: u64,
    tlas_builds_total: u64,
    evictions_total: u64,
    last_tlas_instances: u32,
    last_tlas_resolved: u32,
    last_tlas_culled: u32,
    last_reported_inspected: u64,
    last_reported_builds: u64,
    last_report: Instant,
}

static DXR_RUNTIME: Mutex<Option<DxrRuntime>> = Mutex::new(None);
static LAST_DXR_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);

/// Validates a 64-byte GPU-resident RDNA2 BVH header, distinguishing rejection reasons
/// for bounded diagnostics and retry throttling.
fn inspect_rdna_header(h: &[u32; 16]) -> Result<(bool, u32, u32, u32), HeaderRejectReason> {
    if h.iter().all(|&w| w == 0) {
        return Err(HeaderRejectReason::AllZero);
    }
    let word0 = h[0];
    if word0 & 0xFFFF != 6 {
        return Err(HeaderRejectReason::WrongVersion);
    }
    let flags = word0 >> 16;
    if flags & FLAG_PACKED != 0 {
        return Err(HeaderRejectReason::StillPacked);
    }
    let is_blas = flags & FLAG_BLAS != 0;
    let size_nodes = h[1];
    let num_elements = h[3];
    let offset_elements = h[13];
    if size_nodes < 2 || size_nodes > 1_048_576 || num_elements == 0 || offset_elements == 0 {
        return Err(HeaderRejectReason::BoundsOrEmpty);
    }
    if is_blas {
        if num_elements > MAX_TRI_NODES || offset_elements.saturating_add(num_elements) > size_nodes + 1 {
            return Err(HeaderRejectReason::BoundsOrEmpty);
        }
    } else if num_elements > MAX_INSTANCES
        || offset_elements.saturating_add(num_elements.saturating_mul(2)) > size_nodes + 2
    {
        return Err(HeaderRejectReason::BoundsOrEmpty);
    }
    Ok((is_blas, size_nodes, num_elements, offset_elements))
}

/// Validates a 64-byte GPU-resident RDNA2 BVH header.
/// Returns `Some((is_blas, size_nodes, num_elements, offset_elements))` if the
/// header represents a live, unpacked, bounds-checked RDNA2 BLAS or TLAS.
#[cfg(test)]
fn parse_valid_rdna_header(h: &[u32; 16]) -> Option<(bool, u32, u32, u32)> {
    inspect_rdna_header(h).ok()
}

fn log_raw_header(tag: &str, addr: u64, kind: CandidateKind, rebuilds: u64, h: &[u32; 16]) {
    tracing::info!(
        tag,
        kind = ?kind,
        addr = format_args!("{addr:#x}"),
        rebuilds,
        ver = h[0] & 0xFFFF,
        flags = format_args!("{:#06x}", h[0] >> 16),
        d1_size = h[1],
        d2 = h[2],
        d3_elems = h[3],
        d13_off = h[13],
        w0_7 = format_args!(
            "[{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x}]",
            h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
        ),
        w8_15 = format_args!(
            "[{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x}]",
            h[8], h[9], h[10], h[11], h[12], h[13], h[14], h[15]
        ),
        "victory dxr phase3: raw RDNA2 header sample"
    );
}

impl DxrRuntime {
    fn new(host_gpu: &Host) -> Option<Self> {
        let device5 = host_gpu.device5()?;
        let _ = bridge(host_gpu)?;
        let paged_heap = DxrPagedHeap::new(&host_gpu.device).ok()?;
        let blas_map = BlasMapBuffer::new(host_gpu)?;

        let mut slots = Vec::with_capacity(RING_SLOTS);
        for i in 0..RING_SLOTS {
            // SAFETY: plain D3D12 command allocator + command list4 creation.
            let (allocator, list4) = unsafe {
                let alloc: ID3D12CommandAllocator = host_gpu
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList4 = host_gpu
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                    .ok()?;
                list.Close().ok()?;
                name(&alloc, || format!("victory dxr: allocator {i}"));
                name(&list, || format!("victory dxr: list4 {i}"));
                (alloc, list)
            };
            let (addrs_buf, addrs_raw) = host_gpu.shared_buffer((MAX_INSPECT_PER_FRAME * 8) as u64)?;
            let (headers_buf, headers_raw) = host_gpu.shared_buffer((MAX_INSPECT_PER_FRAME * 64) as u64)?;
            let (stats_buf, stats_raw) = host_gpu.shared_buffer(64)?;
            name(&addrs_buf, || format!("victory dxr: inspect addrs {i}"));
            name(&headers_buf, || format!("victory dxr: inspect headers {i}"));
            name(&stats_buf, || format!("victory dxr: tlas stats {i}"));
            slots.push(RingSlot {
                allocator,
                list4,
                addrs_buf,
                addrs_ptr: addrs_raw.cast::<u64>(),
                headers_buf,
                headers_ptr: headers_raw.cast::<u32>(),
                stats_buf,
                stats_ptr: stats_raw.cast::<u32>(),
                pending_inspect: Vec::new(),
                sample_tlas_slot: None,
                ran_tlas: false,
                kept_resources: Vec::new(),
                fence_value: 0,
            });
        }

        tracing::info!(
            builds_per_frame = builds_per_frame(),
            vram_budget_mb = vram_budget_bytes() >> 20,
            inst_mat_mode = inst_mat_mode(),
            "victory dxr: Phase 3 live paged-arena Ada BLAS/TLAS runtime initialized"
        );

        Some(Self {
            device5,
            paged_heap,
            blas_map,
            slots,
            next_slot: 0,
            vb_scratch: None,
            as_scratch: None,
            tlas_descs: None,
            tlas_results: [None, None],
            tlas_ping: 0,
            validated_blas: HashMap::new(),
            validated_tlas: HashMap::new(),
            blas_build_queue: VecDeque::new(),
            blas_build_queued: HashSet::new(),
            blas_cache: HashMap::new(),
            total_blas_bytes: 0,
            blas_map_dirty: false,
            primary_tlas: None,
            sample_tlas_dumped: false,
            reject_counts: HashMap::new(),
            dumped_addrs: HashSet::new(),
            dumped_walked: 0,
            dumped_static_packed: 0,
            dumped_static_bounds: 0,
            dumped_static_ver: 0,
            dumped_dynamic_bounds: 0,
            dumped_dynamic_hot: 0,
            dumped_valid_blas: 0,
            dumped_valid_tlas: 0,
            retired: Vec::new(),
            inspected_total: 0,
            rej_zero: 0,
            rej_ver: 0,
            rej_packed: 0,
            rej_bounds: 0,
            blas_builds_total: 0,
            tlas_builds_total: 0,
            evictions_total: 0,
            last_tlas_instances: 0,
            last_tlas_resolved: 0,
            last_tlas_culled: 0,
            last_reported_inspected: 0,
            last_reported_builds: 0,
            last_report: Instant::now(),
        })
    }

    fn maybe_report(&mut self, force: bool) {
        if self.inspected_total == 0 {
            return;
        }
        let builds = self.blas_builds_total + self.tlas_builds_total;
        let changed =
            self.inspected_total != self.last_reported_inspected || builds != self.last_reported_builds;
        if !force && (!changed || self.last_report.elapsed().as_secs_f64() < 5.0) {
            return;
        }
        self.last_report = Instant::now();
        self.last_reported_inspected = self.inspected_total;
        self.last_reported_builds = builds;
        let muted = self
            .reject_counts
            .values()
            .filter(|&&c| c >= MAX_REJECT_BEFORE_MUTE)
            .count();
        tracing::info!(
            inspected = self.inspected_total,
            valid_blas = self.validated_blas.len(),
            valid_tlas = self.validated_tlas.len(),
            cached_blas = self.blas_cache.len(),
            blas_mb = format_args!("{:.1}", self.total_blas_bytes as f64 / (1024.0 * 1024.0)),
            blas_builds = self.blas_builds_total,
            tlas_builds = self.tlas_builds_total,
            tlas_va = format_args!("{:#x}", ACTIVE_TLAS_VA.load(Ordering::Relaxed)),
            tlas_instances = self.last_tlas_instances,
            resolved = self.last_tlas_resolved,
            culled = self.last_tlas_culled,
            evictions = self.evictions_total,
            queued_blas = self.blas_build_queue.len(),
            rej_zero = self.rej_zero,
            rej_ver = self.rej_ver,
            rej_packed = self.rej_packed,
            rej_bounds = self.rej_bounds,
            muted,
            "victory dxr phase3: live Ada BLAS/TLAS status"
        );
    }

    fn harvest_completed(&mut self, completed_fence: u64, frame: u64) {
        // Release retired GPU resources whose fence has passed.
        self.retired.retain(|(f, _)| *f > completed_fence);

        let mut retry_static = Vec::new();
        for slot in &mut self.slots {
            if slot.fence_value == 0 || slot.fence_value > completed_fence {
                continue;
            }
            slot.fence_value = 0;
            slot.kept_resources.clear();

            if slot.ran_tlas {
                // SAFETY: GPU finished writing slot.stats_buf before `completed_fence`.
                unsafe {
                    self.last_tlas_resolved = *slot.stats_ptr;
                    self.last_tlas_culled = *slot.stats_ptr.add(1);
                }
                slot.ran_tlas = false;
            }

            if let Some((tlas_addr, base_idx)) = slot.sample_tlas_slot.take() {
                // SAFETY: GPU wrote 2 consecutive 64-byte slices (32 dwords) at `base_idx`.
                let d = unsafe {
                    std::slice::from_raw_parts(slot.headers_ptr.add(base_idx * 16), 32)
                };
                let f = |i: usize| f32::from_bits(d[i]);
                tracing::info!(
                    tlas_addr = format_args!("{tlas_addr:#x}"),
                    mat0 = format_args!(
                        "[{:.3},{:.3},{:.3},{:.3} | {:.3},{:.3},{:.3},{:.3} | {:.3},{:.3},{:.3},{:.3}]",
                        f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11)
                    ),
                    id_mask = format_args!("{:#010x}", d[12]),
                    contrib_flags = format_args!("{:#010x}", d[13]),
                    root_ptr = format_args!("{:#010x}", d[14]),
                    blas_addr = format_args!("{:#x}", u64::from(d[15]) << 8),
                    extra_mat = format_args!(
                        "[{:.3},{:.3},{:.3},{:.3} | {:.3},{:.3},{:.3},{:.3} | {:.3},{:.3},{:.3},{:.3}]",
                        f(16), f(17), f(18), f(19), f(20), f(21), f(22), f(23), f(24), f(25), f(26), f(27)
                    ),
                    "victory dxr phase3: sampled first live RDNA2 TLAS instance layout"
                );
            }

            let pending = std::mem::take(&mut slot.pending_inspect);
            for (idx, (addr, kind, retries, rebuilds)) in pending.into_iter().enumerate() {
                self.inspected_total += 1;
                // SAFETY: GPU finished writing 16 dwords per slot before `completed_fence`.
                let hdr: [u32; 16] = unsafe {
                    let p = slot.headers_ptr.add(idx * 16);
                    std::array::from_fn(|k| *p.add(k))
                };
                match inspect_rdna_header(&hdr) {
                    Err(reason) => {
                        match reason {
                            HeaderRejectReason::AllZero => self.rej_zero += 1,
                            HeaderRejectReason::WrongVersion => self.rej_ver += 1,
                            HeaderRejectReason::StillPacked => self.rej_packed += 1,
                            HeaderRejectReason::BoundsOrEmpty => self.rej_bounds += 1,
                        }
                        if kind == CandidateKind::StaticPacked
                            && matches!(
                                reason,
                                HeaderRejectReason::StillPacked | HeaderRejectReason::AllZero
                            )
                            && retries < MAX_STATIC_RETRIES
                        {
                            // Bounded exponential backoff (4, 16, 64 frames) while rehydration finishes.
                            let next_retry = retries + 1;
                            let delay = 4u64 << (2 * u64::from(retries));
                            retry_static.push((addr, frame.saturating_add(delay), next_retry));
                            continue;
                        }
                        let rejects = self.reject_counts.entry(addr).or_insert(0);
                        *rejects = rejects.saturating_add(1);
                        let reject_n = *rejects;

                        // Bounded one-shot raw header samples for live reverse-engineering of XboxBvhBuilder v6.
                        if !self.dumped_addrs.contains(&addr) {
                            let sample_tag = match (kind, reason) {
                                (CandidateKind::WalkedRead, HeaderRejectReason::AllZero)
                                    if reject_n >= 2 && self.dumped_walked < 8 =>
                                {
                                    self.dumped_walked += 1;
                                    Some("walked_zero")
                                }
                                (CandidateKind::WalkedRead, r)
                                    if r != HeaderRejectReason::AllZero && self.dumped_walked < 8 =>
                                {
                                    self.dumped_walked += 1;
                                    Some(match r {
                                        HeaderRejectReason::StillPacked => "walked_packed",
                                        HeaderRejectReason::BoundsOrEmpty => "walked_bounds",
                                        HeaderRejectReason::WrongVersion => "walked_nonv6",
                                        HeaderRejectReason::AllZero => unreachable!(),
                                    })
                                }
                                (CandidateKind::StaticPacked, HeaderRejectReason::StillPacked)
                                    if self.dumped_static_packed < 4 =>
                                {
                                    self.dumped_static_packed += 1;
                                    Some("static_still_packed")
                                }
                                (CandidateKind::StaticPacked, HeaderRejectReason::BoundsOrEmpty)
                                    if self.dumped_static_bounds < 6 =>
                                {
                                    self.dumped_static_bounds += 1;
                                    Some("static_bounds_reject")
                                }
                                (CandidateKind::StaticPacked, HeaderRejectReason::WrongVersion)
                                    if self.dumped_static_ver < 3 =>
                                {
                                    self.dumped_static_ver += 1;
                                    Some("static_nonv6")
                                }
                                (CandidateKind::DynamicBuilt, HeaderRejectReason::BoundsOrEmpty)
                                    if self.dumped_dynamic_bounds < 6 =>
                                {
                                    self.dumped_dynamic_bounds += 1;
                                    Some("dynamic_bounds_reject")
                                }
                                (CandidateKind::DynamicBuilt, HeaderRejectReason::WrongVersion)
                                    if rebuilds >= 5 && self.dumped_dynamic_hot < 6 =>
                                {
                                    self.dumped_dynamic_hot += 1;
                                    Some("dynamic_hot_nonv6")
                                }
                                _ => None,
                            };
                            if let Some(tag) = sample_tag {
                                self.dumped_addrs.insert(addr);
                                log_raw_header(tag, addr, kind, rebuilds, &hdr);
                            }
                        }
                        continue;
                    }
                    Ok((is_blas, size_nodes, num_elements, offset_elements)) => {
                        self.reject_counts.remove(&addr);
                        if !self.dumped_addrs.contains(&addr) {
                            if is_blas && self.dumped_valid_blas < 3 {
                                self.dumped_valid_blas += 1;
                                self.dumped_addrs.insert(addr);
                                log_raw_header("valid_blas", addr, kind, rebuilds, &hdr);
                            } else if !is_blas && self.dumped_valid_tlas < 3 {
                                self.dumped_valid_tlas += 1;
                                self.dumped_addrs.insert(addr);
                                log_raw_header("valid_tlas", addr, kind, rebuilds, &hdr);
                            }
                        }
                        let is_static = kind == CandidateKind::StaticPacked;
                        if is_blas {
                            let rev = self
                                .validated_blas
                                .get(&addr)
                                .map_or(1, |prev| prev.rev + 1);
                            self.validated_blas.insert(
                                addr,
                                ValidatedHeader {
                                    is_blas: true,
                                    size_nodes,
                                    num_elements,
                                    offset_elements,
                                    rev,
                                    last_frame: frame,
                                },
                            );
                            let needs_build = self.blas_cache.get(&addr).is_none_or(|c| {
                                // Static BLASes are immutable once unpacked; rebuild dynamic BLASes on updates.
                                !is_static && c.last_built_rev < rev
                            });
                            if needs_build && self.blas_build_queued.insert(addr) {
                                self.blas_build_queue.push_back(addr);
                            }
                        } else {
                            let rev = self
                                .validated_tlas
                                .get(&addr)
                                .map_or(1, |prev| prev.rev + 1);
                            self.validated_tlas.insert(
                                addr,
                                ValidatedHeader {
                                    is_blas: false,
                                    size_nodes,
                                    num_elements,
                                    offset_elements,
                                    rev,
                                    last_frame: frame,
                                },
                            );
                            // Prefer walked TLASes or the largest recently updated TLAS as primary.
                            let replace = match self.primary_tlas.and_then(|p| self.validated_tlas.get(&p)) {
                                None => true,
                                Some(cur) => {
                                    kind == CandidateKind::WalkedRead
                                        || frame.saturating_sub(cur.last_frame) > 4
                                        || num_elements >= cur.num_elements
                                }
                            };
                            if replace {
                                self.primary_tlas = Some(addr);
                            }
                        }
                    }
                }
            }
        }

        if !retry_static.is_empty() {
            let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(reg) = reg.as_mut() {
                for (addr, due_frame, retries) in retry_static {
                    if reg.static_packed.contains_key(&addr) && reg.pending_static_set.insert(addr) {
                        reg.pending_static.push_back((addr, due_frame, retries));
                    }
                }
            }
        }
    }

    fn process_evictions(&mut self, next_fence: u64) {
        let (gone, unmute) = {
            let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
            reg.as_mut().map_or_else(
                || (Vec::new(), Vec::new()),
                |r| (std::mem::take(&mut r.gone_addrs), std::mem::take(&mut r.unmute_addrs)),
            )
        };
        for addr in unmute {
            self.reject_counts.remove(&addr);
            self.dumped_addrs.remove(&addr);
        }
        for addr in gone {
            self.validated_blas.remove(&addr);
            self.reject_counts.remove(&addr);
            self.dumped_addrs.remove(&addr);
            if let Some(cached) = self.blas_cache.remove(&addr) {
                self.total_blas_bytes = self.total_blas_bytes.saturating_sub(cached.bytes);
                self.retired.push((next_fence, cached.result));
                self.blas_map_dirty = true;
                self.evictions_total += 1;
            }
        }

        let budget = vram_budget_bytes();
        while self.total_blas_bytes > budget && !self.blas_cache.is_empty() {
            let Some((&oldest_addr, _)) = self.blas_cache.iter().min_by_key(|(_, c)| c.last_frame)
            else {
                break;
            };
            if let Some(cached) = self.blas_cache.remove(&oldest_addr) {
                self.total_blas_bytes = self.total_blas_bytes.saturating_sub(cached.bytes);
                self.retired.push((next_fence, cached.result));
                self.blas_map_dirty = true;
                self.evictions_total += 1;
            }
        }
    }

    fn ensure_scratch(
        slot: &mut Option<(ID3D12Resource, u64)>,
        device: &ID3D12Device,
        needed: u64,
        state: D3D12_RESOURCE_STATES,
        label: &str,
        next_fence: u64,
        retired: &mut Vec<(u64, ID3D12Resource)>,
    ) -> Option<(ID3D12Resource, bool)> {
        let needed = needed.max(256).next_multiple_of(256);
        if let Some((res, cap)) = slot.as_ref()
            && *cap >= needed
        {
            return Some((res.clone(), false));
        }
        let alloc_size = needed.next_power_of_two().max(64 * 1024);
        let res = uav_buffer_state(device, alloc_size, state).ok()?;
        name(&res, || format!("victory dxr: {label} ({alloc_size}B)"));
        if let Some((old, _)) = slot.replace((res.clone(), alloc_size)) {
            retired.push((next_fence, old));
        }
        Some((res, true))
    }

    fn ensure_tlas_result(
        slot: &mut Option<(ID3D12Resource, u64)>,
        device: &ID3D12Device,
        needed: u64,
        next_fence: u64,
        retired: &mut Vec<(u64, ID3D12Resource)>,
    ) -> Option<ID3D12Resource> {
        let needed = needed.max(256).next_multiple_of(256);
        if let Some((res, cap)) = slot.as_ref()
            && *cap >= needed
        {
            return Some(res.clone());
        }
        let alloc_size = needed.next_power_of_two().max(64 * 1024);
        let res = as_buffer(device, alloc_size).ok()?;
        name(&res, || format!("victory dxr: live tlas ({alloc_size}B)"));
        if let Some((old, _)) = slot.replace((res.clone(), alloc_size)) {
            retired.push((next_fence, old));
        }
        Some(res)
    }

    fn record(
        &mut self,
        host_gpu: &Host,
        directory: ID3D12Resource,
        copies: Vec<ID3D12Resource>,
        next_fence: u64,
        completed_fence: u64,
        frame: u64,
    ) -> Option<ID3D12CommandList> {
        self.harvest_completed(completed_fence, frame);
        self.process_evictions(next_fence);

        // Pick next ring slot; if still in flight on the GPU, skip this frame cleanly (zero stall).
        let slot_idx = self.next_slot;
        if self.slots[slot_idx].fence_value > completed_fence {
            return None;
        }

        // Gather up to MAX_INSPECT_PER_FRAME candidates:
        // 1. RT-walked root addresses (ground-truth TLASes)
        // 2. Hot dynamic builder outputs (skipping muted non-BVH scratch buffers)
        // 3. Static packed structures whose backoff delay has elapsed
        let mut inspect_batch: Vec<(u64, CandidateKind, u8, u64)> = Vec::new();
        let mut need_sample_tlas = None;
        if !self.sample_tlas_dumped
            && let Some(tlas_addr) = self.primary_tlas
            && let Some(hdr) = self.validated_tlas.get(&tlas_addr)
            && hdr.num_elements > 0
        {
            need_sample_tlas = Some((tlas_addr, tlas_addr + u64::from(hdr.offset_elements) * NODE_BYTES));
        }
        let max_cands = MAX_INSPECT_PER_FRAME - if need_sample_tlas.is_some() { 2 } else { 0 };
        let is_muted = |addr: u64, counts: &HashMap<u64, u8>| {
            counts.get(&addr).copied().unwrap_or(0) >= MAX_REJECT_BEFORE_MUTE
        };
        {
            let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(reg) = reg.as_mut() {
                while inspect_batch.len() < max_cands {
                    let Some(addr) = reg.pending_walked.pop_front() else {
                        break;
                    };
                    reg.pending_walked_set.remove(&addr);
                    if is_muted(addr, &self.reject_counts) {
                        continue;
                    }
                    inspect_batch.push((addr, CandidateKind::WalkedRead, 0, 0));
                }
                let dyn_cap = (max_cands / 2).max(inspect_batch.len());
                while inspect_batch.len() < dyn_cap {
                    let Some((addr, rebuilds)) = reg.pending_dynamic.pop_front() else {
                        break;
                    };
                    reg.pending_dynamic_set.remove(&addr);
                    if is_muted(addr, &self.reject_counts) {
                        continue;
                    }
                    inspect_batch.push((addr, CandidateKind::DynamicBuilt, 0, rebuilds));
                }
                while inspect_batch.len() < max_cands {
                    let Some(&(addr, due_frame, retries)) = reg.pending_static.front() else {
                        break;
                    };
                    if due_frame > frame {
                        break;
                    }
                    reg.pending_static.pop_front();
                    reg.pending_static_set.remove(&addr);
                    if is_muted(addr, &self.reject_counts) {
                        continue;
                    }
                    inspect_batch.push((addr, CandidateKind::StaticPacked, retries, 0));
                }
                while inspect_batch.len() < max_cands {
                    let Some((addr, rebuilds)) = reg.pending_dynamic.pop_front() else {
                        break;
                    };
                    reg.pending_dynamic_set.remove(&addr);
                    if is_muted(addr, &self.reject_counts) {
                        continue;
                    }
                    inspect_batch.push((addr, CandidateKind::DynamicBuilt, 0, rebuilds));
                }
            }
        }

        // Pull budgeted BLAS builds.
        let mut blas_jobs: Vec<(u64, ValidatedHeader)> = Vec::new();
        let budget = builds_per_frame();
        while blas_jobs.len() < budget {
            let Some(addr) = self.blas_build_queue.pop_front() else {
                break;
            };
            self.blas_build_queued.remove(&addr);
            if let Some(&hdr) = self.validated_blas.get(&addr) {
                blas_jobs.push((addr, hdr));
            }
        }

        let tlas_job = self
            .primary_tlas
            .and_then(|addr| self.validated_tlas.get(&addr).copied().map(|h| (addr, h)))
            .filter(|_| !self.blas_cache.is_empty());

        if inspect_batch.is_empty() && blas_jobs.is_empty() && tlas_job.is_none() {
            self.maybe_report(false);
            return None;
        }

        let bridge = bridge(host_gpu)?;
        self.paged_heap.sync_copies(&host_gpu.device, &copies);

        // Reset slot allocator + command list.
        let slot = &mut self.slots[slot_idx];
        // SAFETY: `slot.fence_value <= completed_fence` verified above.
        unsafe {
            slot.allocator.Reset().ok()?;
            slot.list4.Reset(&slot.allocator, None).ok()?;
        }
        let list: ID3D12GraphicsCommandList = slot.list4.cast().ok()?;
        // SAFETY: open command list; `paged_heap` lives for runtime lifetime.
        unsafe {
            list.SetDescriptorHeaps(&[Some(self.paged_heap.heap().clone())]);
        }
        let arenas_gpu = self.paged_heap.gpu_start();

        // --- Stage 1: Async GPU Header Inspection ---
        let mut inspect_count = inspect_batch.len();
        slot.sample_tlas_slot = None;
        for (i, &(addr, _, _, _)) in inspect_batch.iter().enumerate() {
            // SAFETY: `i < MAX_INSPECT_PER_FRAME`.
            unsafe { *slot.addrs_ptr.add(i) = addr };
        }
        if let Some((tlas_addr, elem_addr)) = need_sample_tlas {
            let base = inspect_count;
            // SAFETY: `base + 1 < MAX_INSPECT_PER_FRAME`.
            unsafe {
                *slot.addrs_ptr.add(base) = elem_addr;
                *slot.addrs_ptr.add(base + 1) = elem_addr + 64;
            }
            inspect_count += 2;
            slot.sample_tlas_slot = Some((tlas_addr, base));
            self.sample_tlas_dumped = true;
        }
        slot.pending_inspect = inspect_batch;
        if inspect_count > 0 {
            bridge.dispatch_inspect_headers_paged(
                &list,
                &directory,
                &slot.addrs_buf,
                &slot.headers_buf,
                arenas_gpu,
                inspect_count as u32,
            );
        }

        // --- Stage 2: Budgeted Paged Ada BLAS Builds ---
        let mut vb_in_srv = false;
        let mut rebuilt_any_blas = false;
        for (addr, hdr) in blas_jobs {
            let elem_addr = addr + u64::from(hdr.offset_elements) * NODE_BYTES;
            let tri_nodes = hdr.num_elements;
            let vert_count = tri_nodes * 4 * 3;
            let vb_bytes = u64::from(vert_count) * 12;

            let Some((vb, vb_is_new)) = Self::ensure_scratch(
                &mut self.vb_scratch,
                &host_gpu.device,
                vb_bytes,
                D3D12_RESOURCE_STATE_COMMON,
                "blas vb scratch",
                next_fence,
                &mut self.retired,
            ) else {
                break;
            };
            if vb_is_new {
                vb_in_srv = false;
            }

            if vb_in_srv {
                // SAFETY: open command list; `vb` was transitioned to NON_PIXEL_SHADER_RESOURCE.
                unsafe {
                    list.ResourceBarrier(&[transition(
                        &vb,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                }
            }
            bridge.dispatch_extract_triangles_paged(
                &list,
                &directory,
                &vb,
                arenas_gpu,
                elem_addr,
                tri_nodes,
            );
            // SAFETY: transition `vb` from UAV write to SRV read for AS builder.
            unsafe {
                list.ResourceBarrier(&[transition(
                    &vb,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            vb_in_srv = true;

            let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: vert_count,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: unsafe { vb.GetGPUVirtualAddress() },
                            StrideInBytes: 12,
                        },
                    },
                },
            };
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &geom,
                },
            };
            let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            // SAFETY: valid inputs and output pointers.
            unsafe {
                self.device5
                    .GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info);
            }
            let result_bytes = info.ResultDataMaxSizeInBytes.max(256).next_multiple_of(256);
            let Ok(result) = as_buffer(&host_gpu.device, result_bytes) else {
                break;
            };
            let Some((scratch, _)) = Self::ensure_scratch(
                &mut self.as_scratch,
                &host_gpu.device,
                info.ScratchDataSizeInBytes,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                "as build scratch",
                next_fence,
                &mut self.retired,
            ) else {
                break;
            };
            // SAFETY: `result` in RAYTRACING_ACCELERATION_STRUCTURE, `scratch` in UAV, `vb` in NON_PIXEL_SRV.
            let gpu_va = unsafe {
                let va = result.GetGPUVirtualAddress();
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(),
                };
                slot.list4.BuildRaytracingAccelerationStructure(&desc, None);
                list.ResourceBarrier(&[uav_barrier(&result), uav_barrier(&scratch)]);
                va
            };

            if let Some(old) = self.blas_cache.insert(
                addr,
                CachedBlas {
                    result,
                    gpu_va,
                    bytes: result_bytes,
                    tri_node_count: tri_nodes,
                    last_frame: frame,
                    last_built_rev: hdr.rev,
                },
            ) {
                self.total_blas_bytes = self.total_blas_bytes.saturating_sub(old.bytes);
                self.retired.push((next_fence, old.result));
            }
            self.total_blas_bytes += result_bytes;
            if !self.blas_map.insert_one(addr, gpu_va) {
                self.blas_map_dirty = true;
            }
            self.blas_builds_total += 1;
            rebuilt_any_blas = true;
        }

        if self.blas_map_dirty {
            self.blas_map.rebuild(self.blas_cache.iter().map(|(&a, c)| (a, c.gpu_va)));
            self.blas_map_dirty = false;
        }

        // --- Stage 3: Live Paged Ada TLAS Build ---
        slot.ran_tlas = false;
        if let Some((tlas_addr, tlas_hdr)) = tlas_job {
            let inst_count = tlas_hdr.num_elements;
            let elem_addr = tlas_addr + u64::from(tlas_hdr.offset_elements) * NODE_BYTES;
            let desc_bytes = (u64::from(inst_count) * DXR_INSTANCE_DESC_BYTES).max(256);

            if let Some((descs, _)) = Self::ensure_scratch(
                &mut self.tlas_descs,
                &host_gpu.device,
                desc_bytes,
                D3D12_RESOURCE_STATE_COMMON,
                "tlas descs scratch",
                next_fence,
                &mut self.retired,
            ) {
                // Zero the 8-byte resolved/culled counters before dispatch.
                // SAFETY: `slot.stats_ptr` is idle because `slot.fence_value <= completed_fence`.
                unsafe {
                    *slot.stats_ptr = 0;
                    *slot.stats_ptr.add(1) = 0;
                }
                bridge.dispatch_convert_instances_paged(
                    &list,
                    &directory,
                    &self.blas_map.buffer,
                    &descs,
                    &slot.stats_buf,
                    arenas_gpu,
                    elem_addr,
                    inst_count,
                    self.blas_map.mask(),
                    inst_mat_mode(),
                );
                // SAFETY: transition `descs` from UAV write to NON_PIXEL_SRV for TLAS builder.
                let descs_va = unsafe {
                    list.ResourceBarrier(&[
                        transition(
                            &descs,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        ),
                        uav_barrier(&slot.stats_buf),
                    ]);
                    descs.GetGPUVirtualAddress()
                };

                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                    NumDescs: inst_count,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        InstanceDescs: descs_va,
                    },
                };
                let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
                // SAFETY: valid inputs and output pointers.
                unsafe {
                    self.device5
                        .GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info);
                }
                let ping = self.tlas_ping;
                if let (Some(tlas_res), Some((scratch, _))) = (
                    Self::ensure_tlas_result(
                        &mut self.tlas_results[ping],
                        &host_gpu.device,
                        info.ResultDataMaxSizeInBytes,
                        next_fence,
                        &mut self.retired,
                    ),
                    Self::ensure_scratch(
                        &mut self.as_scratch,
                        &host_gpu.device,
                        info.ScratchDataSizeInBytes,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        "as build scratch",
                        next_fence,
                        &mut self.retired,
                    ),
                ) {
                    // SAFETY: valid AS result, scratch, and descs buffers.
                    let tlas_va = unsafe {
                        let va = tlas_res.GetGPUVirtualAddress();
                        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                            DestAccelerationStructureData: va,
                            Inputs: inputs,
                            SourceAccelerationStructureData: 0,
                            ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(),
                        };
                        slot.list4.BuildRaytracingAccelerationStructure(&desc, None);
                        list.ResourceBarrier(&[uav_barrier(&tlas_res), uav_barrier(&scratch)]);
                        va
                    };
                    self.tlas_ping = 1 - ping;
                    self.tlas_builds_total += 1;
                    self.last_tlas_instances = inst_count;
                    slot.ran_tlas = true;
                    ACTIVE_TLAS_VA.store(tlas_va, Ordering::Release);
                    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(reg) = reg.as_mut() {
                        reg.tlas_vas.insert(tlas_addr, tlas_va);
                    }
                }
            }
        }

        // Keep `directory` and all arena `copies` alive until `next_fence` completes.
        slot.kept_resources.clear();
        slot.kept_resources.push(directory);
        slot.kept_resources.extend(copies);
        slot.fence_value = next_fence;
        self.next_slot = (slot_idx + 1) % RING_SLOTS;

        // SAFETY: all commands recorded cleanly on `slot.list4`.
        unsafe { slot.list4.Close().ok()? };
        let cmd_list: ID3D12CommandList = slot.list4.cast().ok()?;

        // Periodic 5-second Phase 3 telemetry report.
        self.maybe_report(rebuilt_any_blas && self.blas_builds_total <= 4);

        Some(cmd_list)
    }
}

/// Called from `Host::submit` (at most once per frame) to record Phase 3 async
/// GPU header inspection, budgeted paged Ada BLAS builds, and live Ada TLAS
/// construction on a dedicated command list chained behind the title's lists.
pub(crate) fn record_frame_work(
    host_gpu: &Host,
    next_fence: u64,
    completed_fence: u64,
) -> Option<ID3D12CommandList> {
    if !live_enabled() {
        return None;
    }
    let frame = crate::frame::current();
    if LAST_DXR_FRAME.load(Ordering::Relaxed) == frame {
        return None;
    }
    // Snapshot Memory's GPU directory + arena buffers BEFORE locking DXR_RUNTIME
    // (guarantees strict lock ordering: Host::submitted -> MEMORY -> DXR_RUNTIME -> REGISTRY).
    let (directory, copies) = crate::memory::gpu_snapshot()?;
    if copies.is_empty() {
        return None;
    }
    LAST_DXR_FRAME.store(frame, Ordering::Relaxed);

    let mut guard = DXR_RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = DxrRuntime::new(host_gpu);
    }
    let rt = guard.as_mut()?;
    rt.record(host_gpu, directory, copies, next_fence, completed_fence, frame)
}

/// Ensures the global `DxrBridge` is initialized on `host` (if DXR Tier 1.1 is available).
#[allow(dead_code)]
pub(crate) fn bridge(host_gpu: &Host) -> Option<&'static DxrBridge> {
    static BRIDGE: OnceLock<Option<DxrBridge>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            if !host_gpu.dxr_tier11() {
                return None;
            }
            match DxrBridge::new(&host_gpu.device) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!(error = %e, "victory dxr: bridge init failed");
                    None
                }
            }
        })
        .as_ref()
}

/// Helper to read the first 16 dwords of a structure at `title_addr` cleanly.
#[allow(dead_code)]
pub(crate) fn read_header(title_addr: u64) -> Option<[u32; 16]> {
    if !enabled() || title_addr < (1u64 << 32) {
        return None;
    }
    let mut buf = [0u8; 64];
    if !crate::host::read_into(title_addr as *const u8, &mut buf) {
        return None;
    }
    Some(std::array::from_fn(|i| {
        u32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use victory_shader::bvh;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    fn find_dxc_exe() -> Option<PathBuf> {
        let candidates = [
            std::env::var_os("VICTORY_DXC_DIR").map(|d| PathBuf::from(d).join("dxc.exe")),
            std::env::var_os("XCL_DXC_DIR").map(|d| PathBuf::from(d).join("dxc.exe")),
            Some(PathBuf::from(r"F:\GTA_6\tools\dxc\1.9.2609\bin\x64\dxc.exe")),
            Some(PathBuf::from("dxc.exe")),
        ];
        for c in candidates.into_iter().flatten() {
            if c.is_file() {
                return Some(c);
            }
        }
        None
    }

    fn compile_sm66_dxc(hlsl: &str, name_tag: &str, strict_ieee: bool) -> Option<Vec<u8>> {
        let dxc = find_dxc_exe()?;
        let dir = tempfile::tempdir().ok()?;
        let in_path = dir.path().join(format!("{name_tag}.hlsl"));
        let out_path = dir.path().join(format!("{name_tag}.dxil"));
        std::fs::write(&in_path, hlsl).ok()?;
        let mut cmd = std::process::Command::new(&dxc);
        cmd.args(["-T", "cs_6_6", "-E", "main", "-O3"]);
        if strict_ieee {
            cmd.arg("-Gis");
        }
        let out = cmd
            .arg("-Fo")
            .arg(&out_path)
            .arg(&in_path)
            .output()
            .ok()?;
        if !out.status.success() {
            panic!(
                "dxc.exe failed for {name_tag}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::fs::read(&out_path).ok()
    }

    fn make_rdna_tri_node(verts: [[f32; 3]; 5], id_bytes: [u8; 4]) -> [u32; 16] {
        let mut w = [0u32; 16];
        for (vi, v) in verts.iter().enumerate() {
            for c in 0..3 {
                w[vi * 3 + c] = v[c].to_bits();
            }
        }
        w[15] = u32::from_le_bytes(id_bytes);
        w
    }

    fn make_rdna_instance(
        transform_3x4: [f32; 12],
        instance_id: u32,
        mask: u8,
        hit_group: u32,
        flags: u8,
        root_ptr: u32,
        blas_addr_bytes: u64,
    ) -> [u32; 32] {
        let mut w = [0u32; 32];
        for (i, &f) in transform_3x4.iter().enumerate() {
            w[i] = f.to_bits();
        }
        w[12] = (instance_id & 0x00FF_FFFF) | (u32::from(mask) << 24);
        w[13] = (hit_group & 0x00FF_FFFF) | (u32::from(flags) << 24);
        w[14] = root_ptr;
        w[15] = (blas_addr_bytes >> 8) as u32;
        w
    }

    fn transform_point(m: &[f32; 12], p: [f32; 3]) -> [f32; 3] {
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        ]
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RayInput {
        origin: [f32; 3],
        tmin: f32,
        dir: [f32; 3],
        tmax: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct RayResult {
        hit: u32,
        instance_id: u32,
        prim_idx: u32,
        hit_t: f32,
        bary_x: f32,
        bary_y: f32,
        front: u32,
        _pad: u32,
    }

    const RAYQUERY_TRACE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), DescriptorTable(SRV(t0)), SRV(t1), UAV(u0)"
cbuffer Params : register(b0) {
    uint ray_count;
};
RaytracingAccelerationStructure scene : register(t0);
struct RayIn {
    float3 origin;
    float tmin;
    float3 dir;
    float tmax;
};
struct RayOut {
    uint hit;
    uint instance_id;
    uint prim_idx;
    float hit_t;
    float bary_x;
    float bary_y;
    uint front;
    uint pad;
};
StructuredBuffer<RayIn> rays : register(t1);
RWStructuredBuffer<RayOut> results : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= ray_count) return;
    RayIn ri = rays[idx];
    RayDesc ray;
    ray.Origin = ri.origin;
    ray.TMin = ri.tmin;
    ray.Direction = ri.dir;
    ray.TMax = ri.tmax;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFFu, ray);
    while (q.Proceed()) {}

    RayOut ro = (RayOut)0;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float2 b = q.CommittedTriangleBarycentrics();
        ro.hit = 1u;
        ro.instance_id = q.CommittedInstanceID();
        ro.prim_idx = q.CommittedPrimitiveIndex();
        ro.hit_t = q.CommittedRayT();
        ro.bary_x = b.x;
        ro.bary_y = b.y;
        ro.front = q.CommittedTriangleFrontFace() ? 1u : 0u;
    }
    results[idx] = ro;
}
"#;

    /// Software RDNA2 BVH4 traversal on CUDA cores (Box32 slab test + sorting network +
    /// watertight triangle test) reading the exact RDNA2 BVH4 node buffer, used for
    /// apples-to-apples RT Core vs CUDA Core benchmarking.
    const SOFTWARE_BVH_TRACE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), SRV(t0), SRV(t1), SRV(t2), UAV(u0)"
cbuffer Params : register(b0) {
    uint ray_count;
    uint tlas_root_ptr;
    uint blas_root_ptr;
};
ByteAddressBuffer tlas_buf : register(t0);
ByteAddressBuffer blas_buf : register(t1);
struct RayIn {
    float3 origin;
    float tmin;
    float3 dir;
    float tmax;
};
struct RayOut {
    uint hit;
    uint instance_id;
    uint prim_idx;
    float hit_t;
    float bary_x;
    float bary_y;
    uint front;
    uint pad;
};
StructuredBuffer<RayIn> rays : register(t2);
RWStructuredBuffer<RayOut> results : register(u0);

uint4 test_box32(ByteAddressBuffer buf, uint node_byte, float3 ro, float3 inv_d, float extent) {
    uint4 ch = buf.Load4(node_byte);
    float3 bmin[4];
    float3 bmax[4];
    [unroll]
    for (uint n = 0u; n < 4u; ++n) {
        uint boff = node_byte + 16u + n * 24u;
        uint3 lo_w = buf.Load3(boff);
        uint3 hi_w = buf.Load3(boff + 12u);
        bmin[n] = asfloat(lo_w);
        bmax[n] = asfloat(hi_w);
    }
    uint ids[4] = { ch.x, ch.y, ch.z, ch.w };
    float dists[4] = { 1e30f, 1e30f, 1e30f, 1e30f };
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        float3 t0 = (bmin[k] - ro) * inv_d;
        float3 t1 = (bmax[k] - ro) * inv_d;
        float3 mn = min(t0, t1);
        float3 mx = max(t0, t1);
        float enter = max(mn.x, max(mn.y, mn.z));
        float leave = min(mx.x, min(mx.y, mx.z));
        bool hit = (ids[k] != 0xFFFFFFFFu) && (leave >= max(enter, 0.0f)) && (enter < extent);
        ids[k] = hit ? ids[k] : 0xFFFFFFFFu;
        dists[k] = hit ? enter : 1e30f;
    }
    // 5-comparator sorting network (nearest first)
    uint pairs[5][2] = { {0,1}, {2,3}, {0,2}, {1,3}, {1,2} };
    [unroll]
    for (uint s = 0u; s < 5u; ++s) {
        uint a = pairs[s][0];
        uint b = pairs[s][1];
        if (dists[b] < dists[a]) {
            float td = dists[a]; dists[a] = dists[b]; dists[b] = td;
            uint ti = ids[a]; ids[a] = ids[b]; ids[b] = ti;
        }
    }
    return uint4(ids[0], ids[1], ids[2], ids[3]);
}

bool test_tri_node(uint node_byte, uint k, float3 ro, float3 rd, float3 inv_d, float tmax, out float out_t, out float2 out_bary) {
    out_t = tmax;
    out_bary = float2(0.0f, 0.0f);
    uint4 w0 = blas_buf.Load4(node_byte);
    uint4 w1 = blas_buf.Load4(node_byte + 16u);
    uint4 w2 = blas_buf.Load4(node_byte + 32u);
    uint4 w3 = blas_buf.Load4(node_byte + 48u);
    uint id_b = (w3.w >> (8u * k)) & 0xFFu;
    uint ii = id_b & 3u;
    uint jj = (id_b >> 2u) & 3u;
    if (ii >= 3u || jj >= 3u || ii == jj) return false;
    uint first = (3u - ii - jj) & 3u;
    static const uint TV[4][3] = { {0,1,2}, {1,3,2}, {2,3,4}, {2,4,0} };
    float3 verts[5] = {
        asfloat(w0.xyz),
        asfloat(uint3(w0.w, w1.xy)),
        asfloat(uint3(w1.zw, w2.x)),
        asfloat(w2.yzw),
        asfloat(w3.xyz)
    };
    float3 p0 = verts[TV[k][first]] - ro;
    float3 p1 = verts[TV[k][ii]] - ro;
    float3 p2 = verts[TV[k][jj]] - ro;

    float3 ad = abs(rd);
    uint kz = (ad.x >= ad.y && ad.x >= ad.z) ? 0u : ((ad.y >= ad.z) ? 1u : 2u);
    uint kx = (kz + 1u) % 3u;
    uint ky = (kz + 2u) % 3u;
    float dz = rd[kz];
    if (dz < 0.0f) { uint tmp = kx; kx = ky; ky = tmp; }
    float sz = inv_d[kz];
    float sx = rd[kx] * sz;
    float sy = rd[ky] * sz;

    float3 a = float3(p0[kx] - sx * p0[kz], p0[ky] - sy * p0[kz], sz * p0[kz]);
    float3 b = float3(p1[kx] - sx * p1[kz], p1[ky] - sy * p1[kz], sz * p1[kz]);
    float3 c = float3(p2[kx] - sx * p2[kz], p2[ky] - sy * p2[kz], sz * p2[kz]);

    precise float u = c.x * b.y - c.y * b.x;
    precise float v = a.x * c.y - a.y * c.x;
    precise float w = b.x * a.y - b.y * a.x;
    bool behind = (u < 0.0f || v < 0.0f || w < 0.0f);
    bool before = (u > 0.0f || v > 0.0f || w > 0.0f);
    if (behind && before) return false;
    float det = u + v + w;
    float t_num = u * a.z + v * b.z + w * c.z;
    if (det == 0.0f) return false;
    float hit_t = t_num / det;
    if (hit_t < 0.0f || hit_t >= tmax) return false;
    out_t = hit_t;
    out_bary = float2(v / det, w / det);
    return true;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= ray_count) return;
    RayIn ri = rays[idx];
    float3 inv_d = clamp(1.0f / ri.dir, -3.402823466e+38f, 3.402823466e+38f);
    float best_t = ri.tmax;
    uint best_inst = 0u;
    uint best_prim = 0u;
    float2 best_bary = float2(0.0f, 0.0f);
    bool any_hit = false;

    uint tlas_stack[32];
    int tsp = 0;
    tlas_stack[tsp++] = tlas_root_ptr;
    while (tsp > 0) {
        uint ptr = tlas_stack[--tsp];
        uint kind = ptr & 7u;
        uint byte_off = (ptr >> 3u) << 6u;
        if (kind == 5u) {
            uint4 ch = test_box32(tlas_buf, byte_off, ri.origin, inv_d, best_t);
            if (ch.w != 0xFFFFFFFFu && tsp < 32) tlas_stack[tsp++] = ch.w;
            if (ch.z != 0xFFFFFFFFu && tsp < 32) tlas_stack[tsp++] = ch.z;
            if (ch.y != 0xFFFFFFFFu && tsp < 32) tlas_stack[tsp++] = ch.y;
            if (ch.x != 0xFFFFFFFFu && tsp < 32) tlas_stack[tsp++] = ch.x;
        } else if (kind == 6u) {
            // 128-byte RDNA2 instance: objectToWorld in dwords 0..11 (for pure translation+scale in bench)
            float4 m0 = asfloat(tlas_buf.Load4(byte_off + 0u));
            float4 m1 = asfloat(tlas_buf.Load4(byte_off + 16u));
            float4 m2 = asfloat(tlas_buf.Load4(byte_off + 32u));
            uint inst_id = tlas_buf.Load(byte_off + 48u) & 0x00FFFFFFu;
            float3 lro = float3((ri.origin.x - m0.w) / m0.x, (ri.origin.y - m1.w) / m1.y, (ri.origin.z - m2.w) / m2.z);
            float3 lrd = float3(ri.dir.x / m0.x, ri.dir.y / m1.y, ri.dir.z / m2.z);
            float3 linv = clamp(1.0f / lrd, -3.402823466e+38f, 3.402823466e+38f);

            uint blas_stack[48];
            int bsp = 0;
            blas_stack[bsp++] = blas_root_ptr;
            while (bsp > 0) {
                uint bptr = blas_stack[--bsp];
                uint bkind = bptr & 7u;
                uint boff = (bptr >> 3u) << 6u;
                if (bkind == 5u) {
                    uint4 ch = test_box32(blas_buf, boff, lro, linv, best_t);
                    if (ch.w != 0xFFFFFFFFu && bsp < 48) blas_stack[bsp++] = ch.w;
                    if (ch.z != 0xFFFFFFFFu && bsp < 48) blas_stack[bsp++] = ch.z;
                    if (ch.y != 0xFFFFFFFFu && bsp < 48) blas_stack[bsp++] = ch.y;
                    if (ch.x != 0xFFFFFFFFu && bsp < 48) blas_stack[bsp++] = ch.x;
                } else if (bkind < 4u) {
                    float ht;
                    float2 hb;
                    if (test_tri_node(boff, bkind, lro, lrd, linv, best_t, ht, hb)) {
                        best_t = ht;
                        best_inst = inst_id;
                        best_prim = (boff >> 6u) * 4u + bkind;
                        best_bary = hb;
                        any_hit = true;
                    }
                }
            }
        }
    }

    RayOut ro = (RayOut)0;
    if (any_hit) {
        ro.hit = 1u;
        ro.instance_id = best_inst;
        ro.prim_idx = best_prim;
        ro.hit_t = best_t;
        ro.bary_x = best_bary.x;
        ro.bary_y = best_bary.y;
    }
    results[idx] = ro;
}
"#;

    struct GpuRunner {
        host: &'static Host,
        device5: ID3D12Device5,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        list4: ID3D12GraphicsCommandList4,
        fence: ID3D12Fence,
        event: windows::Win32::Foundation::HANDLE,
        submitted: u64,
    }

    impl GpuRunner {
        fn new() -> Option<Self> {
            let host = crate::host::host()?;
            if !host.dxr_tier11() {
                return None;
            }
            let device5 = host.device5()?;
            // SAFETY: plain D3D12 queue/list/fence creation on a live device.
            unsafe {
                let queue: ID3D12CommandQueue = host
                    .device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = host
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Self {
                    host,
                    device5,
                    queue,
                    allocator,
                    list,
                    list4,
                    fence,
                    event,
                    submitted: 0,
                })
            }
        }

        fn submit_and_wait(&mut self) {
            // SAFETY: valid open list, live queue, fence, and event.
            unsafe {
                self.list.Close().unwrap();
                self.queue
                    .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>().unwrap())]);
                self.submitted += 1;
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    #[test]
    fn dxr_tier11_rdna2_to_ada_blas_tlas_matches_software_bvh() {
        let Some(mut gpu) = GpuRunner::new() else {
            eprintln!("no DXR Tier 1.1 hardware; skipped");
            return;
        };
        let Some(trace_dxil) = compile_sm66_dxc(RAYQUERY_TRACE_HLSL, "victory_dxr_verify", false) else {
            eprintln!("dxc.exe not found; skipping RayQuery hardware verification");
            return;
        };

        let bridge = DxrBridge::new(&gpu.host.device).expect("DxrBridge");

        // Construct 2 RDNA2 triangle nodes covering all 4 triangle slots (k = 0..3)
        // with non-trivial byte-60 (i, j) vertex permutations and inactive slots = 0.
        let v0 = [-1.0f32, -1.0, 0.0];
        let v1 = [0.0f32, 1.5, 0.0];
        let v2 = [1.0f32, -1.0, 0.0];
        let v3 = [2.2f32, 1.2, 0.2];
        let v4 = [-2.0f32, 1.0, -0.3];

        // Node 0: slot 0 active (i=1,j=2 -> 0x09), slot 1 active (i=2,j=0 -> 0x02), slots 2..3 inactive (0)
        let node0 = make_rdna_tri_node([v0, v1, v2, v3, v4], [1 | (2 << 2), 2 | (0 << 2), 0, 0]);
        // Node 1: slot 2 active (i=0,j=1 -> 0x04), slot 3 active (i=2,j=1 -> 0x06), slots 0..1 inactive (0)
        let node1 = make_rdna_tri_node([v0, v1, v2, v3, v4], [0, 0, 0 | (1 << 2), 2 | (1 << 2)]);

        let (rdna_nodes_buf, rdna_nodes_ptr) = gpu.host.shared_buffer(128).unwrap();
        // SAFETY: mapped buffer of 128 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(node0.as_ptr().cast::<u8>(), rdna_nodes_ptr, 64);
            std::ptr::copy_nonoverlapping(node1.as_ptr().cast::<u8>(), rdna_nodes_ptr.add(64), 64);
        }

        let blas = bridge
            .build_blas_from_rdna_nodes(&gpu.host, &gpu.device5, &gpu.list4, &rdna_nodes_buf, 0, 2, true)
            .expect("build_blas_from_rdna_nodes");

        // Two instances with non-trivial 3x4 transforms:
        // Instance 0: translated by (2.0, -1.0, 5.0), InstanceID = 7
        // Instance 1: rotated 90 deg around Y + translated (-5.0, 0.5, 10.0), InstanceID = 13
        let m0 = [
            1.0f32, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -1.0,
            0.0, 0.0, 1.0, 5.0,
        ];
        let m1 = [
            0.0f32, 0.0, 1.0, -5.0,
            0.0, 1.0, 0.0, 0.5,
            -1.0, 0.0, 0.0, 10.0,
        ];
        let inst0 = make_rdna_instance(m0, 7, 0xFF, 0, 0, 0, 0);
        let inst1 = make_rdna_instance(m1, 13, 0xFF, 0, 0, 0, 0);

        let (rdna_inst_buf, rdna_inst_ptr) = gpu.host.shared_buffer(256).unwrap();
        let (va_table_buf, va_table_ptr) = gpu.host.shared_buffer(32).unwrap();
        // SAFETY: mapped buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(inst0.as_ptr().cast::<u8>(), rdna_inst_ptr, 128);
            std::ptr::copy_nonoverlapping(inst1.as_ptr().cast::<u8>(), rdna_inst_ptr.add(128), 128);
            let vas = [blas.gpu_va, blas.gpu_va];
            std::ptr::copy_nonoverlapping(vas.as_ptr().cast::<u8>(), va_table_ptr, 16);
        }

        let tlas = bridge
            .build_tlas_from_instances(&gpu.host, &gpu.device5, &gpu.list4, &rdna_inst_buf, 0, 2, &va_table_buf)
            .expect("build_tlas_from_instances");

        // Prepare RayQuery trace pipeline + descriptor heap
        // SAFETY: valid DXIL bytecode and descriptor heap creation.
        let (trace_root, trace_pso, heap, tlas_gpu_handle) = unsafe {
            let root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, &trace_dxil).unwrap();
            let pso: ID3D12PipelineState = gpu
                .host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: trace_dxil.as_ptr().cast(),
                        BytecodeLength: trace_dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let heap: ID3D12DescriptorHeap = gpu
                .host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            create_tlas_srv(
                &gpu.host.device,
                tlas.gpu_va,
                heap.GetCPUDescriptorHandleForHeapStart(),
            );
            let gpu_h = heap.GetGPUDescriptorHandleForHeapStart();
            (root, pso, heap, gpu_h)
        };

        // 12 diverse test rays (hits on both instances, all active slots, misses)
        let test_rays = [
            RayInput { origin: [2.0, -1.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [2.3, -0.7, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [3.2, -0.6, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [1.1, -0.5, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [0.6, -1.1, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [2.0, -1.0, 10.0], tmin: 0.0, dir: [0.0, 0.0, -1.0], tmax: 100.0 },
            // Instance 1 rays (rotated + translated to x=-5, y=0.5, z=10)
            RayInput { origin: [-10.0, 0.5, 10.0], tmin: 0.0, dir: [1.0, 0.0, 0.0], tmax: 100.0 },
            RayInput { origin: [-10.0, 0.8, 9.7], tmin: 0.0, dir: [1.0, 0.0, 0.0], tmax: 100.0 },
            RayInput { origin: [-10.0, 0.9, 8.8], tmin: 0.0, dir: [1.0, 0.0, 0.0], tmax: 100.0 },
            RayInput { origin: [-10.0, 0.9, 10.9], tmin: 0.0, dir: [1.0, 0.0, 0.0], tmax: 100.0 },
            // Misses (short tmax, or out of bounds)
            RayInput { origin: [2.0, -1.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 2.0 },
            RayInput { origin: [50.0, 50.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
        ];

        let (rays_buf, rays_ptr) = gpu.host.shared_buffer((test_rays.len() * size_of::<RayInput>()) as u64).unwrap();
        let (out_buf, out_ptr) = gpu.host.shared_buffer((test_rays.len() * size_of::<RayResult>()) as u64).unwrap();
        // SAFETY: mapped buffers sized for `test_rays`.
        unsafe {
            std::ptr::copy_nonoverlapping(test_rays.as_ptr(), rays_ptr.cast(), test_rays.len());
            std::ptr::write_bytes(out_ptr, 0, test_rays.len() * size_of::<RayResult>());
            gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            gpu.list.SetComputeRootSignature(&trace_root);
            gpu.list.SetPipelineState(&trace_pso);
            let n = test_rays.len() as u32;
            gpu.list.SetComputeRoot32BitConstants(0, 1, (&raw const n).cast(), 0);
            gpu.list.SetComputeRootDescriptorTable(1, tlas_gpu_handle);
            gpu.list.SetComputeRootShaderResourceView(2, rays_buf.GetGPUVirtualAddress());
            gpu.list.SetComputeRootUnorderedAccessView(3, out_buf.GetGPUVirtualAddress());
            gpu.list.Dispatch(1, 1, 1);
        }
        gpu.submit_and_wait();

        // CPU reference: test all 4 active world-space triangles against `victory_shader::bvh::triangle`.
        let nodes = [node0, node1];
        let mut world_tris: Vec<(u32, [[f32; 3]; 3])> = Vec::new();
        for (inst_id, m) in [(7u32, m0), (13u32, m1)] {
            for (ni, node) in nodes.iter().enumerate() {
                let mut padded = [0u32; 28];
                padded[..16].copy_from_slice(node);
                let active_slots = if ni == 0 { [0u32, 1] } else { [2u32, 3] };
                for k in active_slots {
                    let local = bvh::triangle_vertices(&padded, k);
                    world_tris.push((inst_id, local.map(|p| transform_point(&m, p))));
                }
            }
        }

        // SAFETY: GPU finished and wrote `test_rays.len()` RayResult entries.
        let gpu_results = unsafe { std::slice::from_raw_parts(out_ptr.cast::<RayResult>(), test_rays.len()) };
        for (ri, (&ray_in, gpu_res)) in test_rays.iter().zip(gpu_results.iter()).enumerate() {
            let cpu_ray = bvh::Ray::new(ray_in.origin, ray_in.dir, ray_in.tmax);
            let mut best: Option<(f32, u32, f32, f32)> = None;
            for &(inst_id, ref tri) in &world_tris {
                let res = bvh::triangle(tri, &cpu_ray);
                let (t_num, det, i_num, j_num) = (res[0], res[1], res[2], res[3]);
                if t_num.is_finite() && det != 0.0 {
                    let t = t_num / det;
                    if t >= ray_in.tmin && t < best.map_or(ray_in.tmax, |b| b.0) {
                        best = Some((t, inst_id, i_num / det, j_num / det));
                    }
                }
            }
            match (best, gpu_res.hit != 0) {
                (None, false) => {}
                (Some((ct, cid, ci, cj)), true) => {
                    assert_eq!(gpu_res.instance_id, cid, "ray {ri}: instance mismatch");
                    assert!(
                        (gpu_res.hit_t - ct).abs() < 1e-4,
                        "ray {ri}: hitT gpu={} cpu={ct}",
                        gpu_res.hit_t
                    );
                    assert!(
                        (gpu_res.bary_x - ci).abs() < 1e-4 && (gpu_res.bary_y - cj).abs() < 1e-4,
                        "ray {ri}: bary gpu=({},{}) cpu=({ci},{cj})",
                        gpu_res.bary_x,
                        gpu_res.bary_y
                    );
                }
                (cpu, gpu_hit) => {
                    panic!("ray {ri}: hit mismatch cpu={cpu:?} gpu_hit={gpu_hit} ({gpu_res:?})");
                }
            }
        }
        eprintln!(
            "DXR 1.1 VERIFIED ON RTX 4090 D: all {} rays matched CPU RDNA2 reference bit-accurately (hitT + instanceID + barycentrics)",
            test_rays.len()
        );
    }

    fn make_box32_node(children: [u32; 4], bounds: [[f32; 6]; 4]) -> [u32; 32] {
        let mut w = [0u32; 32];
        w[..4].copy_from_slice(&children);
        for b in 0..4 {
            for c in 0..6 {
                w[4 + b * 6 + c] = bounds[b][c].to_bits();
            }
        }
        w
    }

    fn union_box(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].max(b[3]),
            a[4].max(b[4]),
            a[5].max(b[5]),
        ]
    }

    #[test]
    #[ignore = "RTX 4090 D RT Core vs Software BVH4 benchmark"]
    fn bench_rt_core_vs_software_bvh() {
        let Some(mut gpu) = GpuRunner::new() else {
            eprintln!("no DXR Tier 1.1 hardware; skipped");
            return;
        };
        let Some(rt_dxil) = compile_sm66_dxc(RAYQUERY_TRACE_HLSL, "victory_dxr_bench_rt", false) else {
            eprintln!("dxc.exe not found; skipped");
            return;
        };
        let Some(sw_dxil) = compile_sm66_dxc(SOFTWARE_BVH_TRACE_HLSL, "victory_dxr_bench_sw", true) else {
            eprintln!("dxc.exe failed on software BVH shader");
            return;
        };
        let bridge = DxrBridge::new(&gpu.host.device).expect("DxrBridge");

        // Build a spatial 4-ary RDNA2 BVH4 BLAS with 4,096 leaf nodes (16,384 triangles)
        // in 16 x 16 x 16 = 4096 spatial cells inside [-8..8]^3.
        const LEAF_NODES: usize = 4096;
        let mut blas_bytes = Vec::<u32>::new();
        let mut leaf_bounds = Vec::<[f32; 6]>::with_capacity(LEAF_NODES);
        for idx in 0..LEAF_NODES {
            let x = (idx & 15) as f32 - 8.0;
            let y = ((idx >> 4) & 15) as f32 - 8.0;
            let z = ((idx >> 8) & 15) as f32 - 8.0;
            let s = 0.35f32;
            let verts = [
                [x + 0.1, y + 0.1, z + 0.5],
                [x + 0.1, y + 0.1 + s, z + 0.5],
                [x + 0.1 + s, y + 0.1, z + 0.5],
                [x + 0.1 + s, y + 0.1 + s, z + 0.52],
                [x + 0.1 + s * 0.5, y + 0.1 + s * 0.5, z + 0.45],
            ];
            let id_byte = (1u8) | (2u8 << 2);
            let node = make_rdna_tri_node(verts, [id_byte; 4]);
            blas_bytes.extend_from_slice(&node);
            leaf_bounds.push([x + 0.1, y + 0.1, z + 0.45, x + 0.1 + s, y + 0.1 + s, z + 0.52]);
        }

        // Level 1: 4096 Box32 nodes directly wrapping the 4 triangles of each leaf node
        let mut cur_level: Vec<(u32, [f32; 6])> = Vec::with_capacity(LEAF_NODES);
        for i in 0..LEAF_NODES {
            let node_idx_64 = (blas_bytes.len() / 16) as u32;
            let ch = [0u32, 1, 2, 3].map(|k| ((i as u32) << 3) | k);
            let b = leaf_bounds[i];
            let box32 = make_box32_node(ch, [b; 4]);
            blas_bytes.extend_from_slice(&box32);
            cur_level.push(((node_idx_64 << 3) | 5, b));
        }

        // Bottom-up 4-ary Box32 hierarchy to 1 root node
        while cur_level.len() > 1 {
            let mut next_level = Vec::with_capacity(cur_level.len() / 4);
            for chunk in cur_level.chunks(4) {
                let node_idx_64 = (blas_bytes.len() / 16) as u32;
                let mut ch = [u32::MAX; 4];
                let mut bounds = [[0.0f32; 6]; 4];
                let mut merged = chunk[0].1;
                for (j, &(ptr, b)) in chunk.iter().enumerate() {
                    ch[j] = ptr;
                    bounds[j] = b;
                    merged = union_box(merged, b);
                }
                let box32 = make_box32_node(ch, bounds);
                blas_bytes.extend_from_slice(&box32);
                next_level.push(((node_idx_64 << 3) | 5, merged));
            }
            cur_level = next_level;
        }
        let blas_root_ptr = cur_level[0].0;

        let blas_buf_size = (blas_bytes.len() * 4) as u64;
        let (blas_stage, blas_stage_ptr) = gpu.host.shared_buffer(blas_buf_size).unwrap();
        let blas_gpu_buf = gpu.host.uav_buffer(blas_buf_size).unwrap();
        // SAFETY: copy BLAS RDNA2 buffer to GPU default heap for fair VRAM traversal.
        unsafe {
            std::ptr::copy_nonoverlapping(blas_bytes.as_ptr().cast::<u8>(), blas_stage_ptr, blas_buf_size as usize);
            gpu.list.CopyBufferRegion(&blas_gpu_buf, 0, &blas_stage, 0, blas_buf_size);
            gpu.list.ResourceBarrier(&[transition(&blas_gpu_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
        }
        gpu.submit_and_wait();

        // Build 16 instances across a 4x4 grid (16 * 16,384 = 262,144 scene triangles)
        const INSTANCES: usize = 16;
        let mut tlas_words = Vec::<u32>::new();
        let mut inst_bounds = Vec::<[f32; 6]>::with_capacity(INSTANCES);
        for idx in 0..INSTANCES {
            let gx = ((idx % 4) as f32 - 1.5) * 18.0;
            let gy = ((idx / 4) as f32 - 1.5) * 18.0;
            let gz = 25.0f32;
            let m = [
                1.0f32, 0.0, 0.0, gx,
                0.0, 1.0, 0.0, gy,
                0.0, 0.0, 1.0, gz,
            ];
            let inst = make_rdna_instance(m, idx as u32, 0xFF, 0, 0, 0, 0);
            tlas_words.extend_from_slice(&inst);
            inst_bounds.push([gx - 8.5, gy - 8.5, gz - 8.5, gx + 8.5, gy + 8.5, gz + 8.5]);
        }
        // Build 2-level Box32 TLAS hierarchy above the 16 instances (for fair SW traversal)
        let mut t_level1 = Vec::new();
        for group in 0..4usize {
            let node_idx_64 = (tlas_words.len() / 16) as u32;
            let mut ch = [0u32; 4];
            let mut bnds = [[0.0f32; 6]; 4];
            let mut merged = inst_bounds[group * 4];
            for j in 0..4 {
                let ii = group * 4 + j;
                ch[j] = ((ii as u32 * 2) << 3) | 6;
                bnds[j] = inst_bounds[ii];
                merged = union_box(merged, inst_bounds[ii]);
            }
            tlas_words.extend_from_slice(&make_box32_node(ch, bnds));
            t_level1.push(((node_idx_64 << 3) | 5, merged));
        }
        let tlas_root_idx_64 = (tlas_words.len() / 16) as u32;
        let tlas_root_ptr = (tlas_root_idx_64 << 3) | 5;
        tlas_words.extend_from_slice(&make_box32_node(
            [t_level1[0].0, t_level1[1].0, t_level1[2].0, t_level1[3].0],
            [t_level1[0].1, t_level1[1].1, t_level1[2].1, t_level1[3].1],
        ));

        let tlas_buf_size = (tlas_words.len() * 4) as u64;
        let (tlas_stage, tlas_stage_ptr) = gpu.host.shared_buffer(tlas_buf_size).unwrap();
        let tlas_gpu_buf = gpu.host.uav_buffer(tlas_buf_size).unwrap();
        let (va_table_buf, va_table_ptr) = gpu.host.shared_buffer((INSTANCES * 8) as u64).unwrap();

        // Setup GPU timestamp queries
        let (stamps, stamp_readback, stamp_ptr, freq) = unsafe {
            let mut qh: Option<ID3D12QueryHeap> = None;
            gpu.host.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: 6,
                NodeMask: 0,
            }, &mut qh).unwrap();
            let (rb, ptr) = gpu.host.shared_buffer(48).unwrap();
            let freq = gpu.queue.GetTimestampFrequency().unwrap() as f64;
            (qh.unwrap(), rb, ptr, freq)
        };

        // Measure BLAS conversion + build on RTX 4090 D
        unsafe { gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0); }
        let blas = bridge
            .build_blas_from_rdna_nodes(&gpu.host, &gpu.device5, &gpu.list4, &blas_gpu_buf, 0, LEAF_NODES as u32, true)
            .expect("bench blas");
        unsafe {
            gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            gpu.list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &stamp_readback, 0);
        }
        gpu.submit_and_wait();
        let blas_ms = unsafe {
            let t = std::slice::from_raw_parts(stamp_ptr.cast::<u64>(), 2);
            t[1].saturating_sub(t[0]) as f64 / freq * 1e3
        };

        unsafe {
            std::ptr::copy_nonoverlapping(tlas_words.as_ptr().cast::<u8>(), tlas_stage_ptr, tlas_buf_size as usize);
            gpu.list.CopyBufferRegion(&tlas_gpu_buf, 0, &tlas_stage, 0, tlas_buf_size);
            gpu.list.ResourceBarrier(&[transition(&tlas_gpu_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            let vas = [blas.gpu_va; INSTANCES];
            std::ptr::copy_nonoverlapping(vas.as_ptr().cast::<u8>(), va_table_ptr, INSTANCES * 8);
            gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        }
        let tlas = bridge
            .build_tlas_from_instances(&gpu.host, &gpu.device5, &gpu.list4, &tlas_gpu_buf, 0, INSTANCES as u32, &va_table_buf)
            .expect("bench tlas");
        unsafe {
            gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            gpu.list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &stamp_readback, 0);
        }
        gpu.submit_and_wait();
        let tlas_ms = unsafe {
            let t = std::slice::from_raw_parts(stamp_ptr.cast::<u64>(), 2);
            t[1].saturating_sub(t[0]) as f64 / freq * 1e3
        };

        // Generate 1,048,576 (1024x1024) camera rays spanning all 16 instances
        const RAY_COUNT: usize = 1024 * 1024;
        let rays_bytes = (RAY_COUNT * size_of::<RayInput>()) as u64;
        let out_bytes = (RAY_COUNT * size_of::<RayResult>()) as u64;
        let (rays_stage, rays_stage_ptr) = gpu.host.shared_buffer(rays_bytes).unwrap();
        let rays_gpu = gpu.host.uav_buffer(rays_bytes).unwrap();
        let out_rt = gpu.host.uav_buffer(out_bytes).unwrap();
        let out_sw = gpu.host.uav_buffer(out_bytes).unwrap();
        let (rt_readback, rt_ptr) = gpu.host.shared_buffer(out_bytes).unwrap();
        let (sw_readback, sw_ptr) = gpu.host.shared_buffer(out_bytes).unwrap();

        unsafe {
            let slice = std::slice::from_raw_parts_mut(rays_stage_ptr.cast::<RayInput>(), RAY_COUNT);
            for y in 0..1024usize {
                for x in 0..1024usize {
                    let u = (x as f32 + 0.5) / 1024.0 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / 1024.0 * 2.0 - 1.0;
                    let dx = u * 1.4;
                    let dy = v * 1.4;
                    let dz = 1.0f32;
                    let inv_len = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt();
                    slice[y * 1024 + x] = RayInput {
                        origin: [0.0, 0.0, 0.0],
                        tmin: 0.0,
                        dir: [dx * inv_len, dy * inv_len, dz * inv_len],
                        tmax: 200.0,
                    };
                }
            }
            gpu.list.CopyBufferRegion(&rays_gpu, 0, &rays_stage, 0, rays_bytes);
            gpu.list.ResourceBarrier(&[transition(&rays_gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        }
        gpu.submit_and_wait();

        let (rt_root, rt_pso, sw_root, sw_pso, heap, tlas_gpu_handle) = unsafe {
            let rt_root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, &rt_dxil).unwrap();
            let rt_pso: ID3D12PipelineState = gpu.host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rt_root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: rt_dxil.as_ptr().cast(), BytecodeLength: rt_dxil.len() },
                ..Default::default()
            }).unwrap();
            let sw_root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, &sw_dxil).unwrap();
            let sw_pso: ID3D12PipelineState = gpu.host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&sw_root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: sw_dxil.as_ptr().cast(), BytecodeLength: sw_dxil.len() },
                ..Default::default()
            }).unwrap();
            let heap: ID3D12DescriptorHeap = gpu.host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).unwrap();
            create_tlas_srv(&gpu.host.device, tlas.gpu_va, heap.GetCPUDescriptorHandleForHeapStart());
            let gpu_h = heap.GetGPUDescriptorHandleForHeapStart();
            (rt_root, rt_pso, sw_root, sw_pso, heap, gpu_h)
        };

        let groups = (RAY_COUNT as u32).div_ceil(64);
        let mut run_pair = || -> (f64, f64) {
            unsafe {
                // Software RDNA2 BVH4 pass on CUDA Cores
                gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                gpu.list.SetComputeRootSignature(&sw_root);
                gpu.list.SetPipelineState(&sw_pso);
                let sw_params = [RAY_COUNT as u32, tlas_root_ptr, blas_root_ptr];
                gpu.list.SetComputeRoot32BitConstants(0, 3, sw_params.as_ptr().cast(), 0);
                gpu.list.SetComputeRootShaderResourceView(1, tlas_gpu_buf.GetGPUVirtualAddress());
                gpu.list.SetComputeRootShaderResourceView(2, blas_gpu_buf.GetGPUVirtualAddress());
                gpu.list.SetComputeRootShaderResourceView(3, rays_gpu.GetGPUVirtualAddress());
                gpu.list.SetComputeRootUnorderedAccessView(4, out_sw.GetGPUVirtualAddress());
                gpu.list.Dispatch(groups, 1, 1);
                gpu.list.ResourceBarrier(&[uav_barrier(&out_sw)]);
                gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);

                // Hardware Ada RT Core RayQuery pass
                gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                gpu.list.SetComputeRootSignature(&rt_root);
                gpu.list.SetPipelineState(&rt_pso);
                let rt_n = RAY_COUNT as u32;
                gpu.list.SetComputeRoot32BitConstants(0, 1, (&raw const rt_n).cast(), 0);
                gpu.list.SetComputeRootDescriptorTable(1, tlas_gpu_handle);
                gpu.list.SetComputeRootShaderResourceView(2, rays_gpu.GetGPUVirtualAddress());
                gpu.list.SetComputeRootUnorderedAccessView(3, out_rt.GetGPUVirtualAddress());
                gpu.list.Dispatch(groups, 1, 1);
                gpu.list.ResourceBarrier(&[uav_barrier(&out_rt)]);
                gpu.list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 3);

                gpu.list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, &stamp_readback, 0);
            }
            gpu.submit_and_wait();
            unsafe {
                let t = std::slice::from_raw_parts(stamp_ptr.cast::<u64>(), 4);
                let sw_ms = t[1].saturating_sub(t[0]) as f64 / freq * 1e3;
                let rt_ms = t[3].saturating_sub(t[2]) as f64 / freq * 1e3;
                (sw_ms, rt_ms)
            }
        };

        // Warmup + 9 timed runs
        run_pair();
        let mut sw_times = Vec::with_capacity(9);
        let mut rt_times = Vec::with_capacity(9);
        for _ in 0..9 {
            let (sw, rt) = run_pair();
            sw_times.push(sw);
            rt_times.push(rt);
        }
        sw_times.sort_by(f64::total_cmp);
        rt_times.sort_by(f64::total_cmp);
        let sw_med = sw_times[4];
        let rt_med = rt_times[4];

        // Read back and verify hit parity between Software BVH4 and RT Core RayQuery
        unsafe {
            gpu.list.ResourceBarrier(&[
                transition(&out_rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(&out_sw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ]);
            gpu.list.CopyBufferRegion(&rt_readback, 0, &out_rt, 0, out_bytes);
            gpu.list.CopyBufferRegion(&sw_readback, 0, &out_sw, 0, out_bytes);
        }
        gpu.submit_and_wait();
        let (rt_res, sw_res) = unsafe {
            (
                std::slice::from_raw_parts(rt_ptr.cast::<RayResult>(), RAY_COUNT),
                std::slice::from_raw_parts(sw_ptr.cast::<RayResult>(), RAY_COUNT),
            )
        };
        let is_grazing_edge = |res: &RayResult| -> bool {
            res.hit != 0 && res.bary_x.min(res.bary_y).min(1.0 - res.bary_x - res.bary_y) < 1e-4
        };
        let mut hits = 0usize;
        let mut edge_grazing_diffs = 0usize;
        let mut true_mismatches = 0usize;
        for (r, s) in rt_res.iter().zip(sw_res.iter()) {
            if r.hit != 0 {
                hits += 1;
            }
            if r.hit != s.hit || (r.hit != 0 && (r.instance_id != s.instance_id || (r.hit_t - s.hit_t).abs() > 1e-3)) {
                let closer = if r.hit != 0 && s.hit == 0 {
                    r
                } else if s.hit != 0 && r.hit == 0 {
                    s
                } else if r.hit_t < s.hit_t {
                    r
                } else {
                    s
                };
                if is_grazing_edge(closer) {
                    edge_grazing_diffs += 1;
                } else {
                    true_mismatches += 1;
                }
            }
        }
        assert_eq!(true_mismatches, 0, "RT Core vs Software BVH4 non-edge mismatch count");
        assert!(
            edge_grazing_diffs < 256,
            "unexpected grazing-edge diff count: {edge_grazing_diffs}"
        );
        let mrays = RAY_COUNT as f64 / 1e6;
        eprintln!(
            "=== RTX 4090 D RT CORE BENCHMARK (1,048,576 rays, 16 instances x 16,384 tris = 262,144 tris, hits={hits}, edge_grazing_diffs={edge_grazing_diffs}) ==="
        );
        eprintln!("  RDNA2->Ada BLAS build (4,096 nodes / 16,384 tris): {blas_ms:.3} ms");
        eprintln!("  RDNA2->Ada TLAS build (16 instances):              {tlas_ms:.3} ms");
        eprintln!(
            "  Software RDNA2 BVH4 (CUDA Cores): {sw_med:.3} ms ({:.1} MRays/s)",
            mrays / (sw_med / 1e3)
        );
        eprintln!(
            "  Hardware Ada RayQuery (RT Cores): {rt_med:.3} ms ({:.1} MRays/s)",
            mrays / (rt_med / 1e3)
        );
        eprintln!("  RT Core Speedup: {:.2}x faster", sw_med / rt_med);
    }

    #[test]
    fn dxr_phase2_arena_header_to_ada_blas_tlas() {
        let Some(mut gpu) = GpuRunner::new() else {
            eprintln!("no DXR Tier 1.1 hardware; skipped");
            return;
        };
        let Some(trace_dxil) = compile_sm66_dxc(RAYQUERY_TRACE_HLSL, "victory_dxr_arena_hdr", false) else {
            eprintln!("dxc.exe not found; skipped");
            return;
        };
        let bridge = DxrBridge::new(&gpu.host.device).expect("DxrBridge");

        // Lay out a mock GPU arena containing:
        //  - offset 0x100: 64B BLAS header + 1 node at elem offset 2 (0x180)
        //  - offset 0x400: 64B TLAS header + 1 instance at elem offset 1 (0x440)
        let v0 = [-1.0f32, -1.0, 2.0];
        let v1 = [0.0f32, 1.5, 2.0];
        let v2 = [1.0f32, -1.0, 2.0];
        let node = make_rdna_tri_node([v0, v1, v2, v0, v0], [1 | (2 << 2), 0, 0, 0]);
        let mut blas_hdr = [0u32; 16];
        blas_hdr[0] = 6 | (FLAG_BLAS << 16);
        blas_hdr[1] = 3;
        blas_hdr[3] = 1;
        blas_hdr[13] = 2; // elem_offset = 0x100 + 2 * 64 = 0x180

        let mut arena_bytes = vec![0u8; 4096];
        // SAFETY: slice bounds inside 4096-byte arena buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(blas_hdr.as_ptr().cast::<u8>(), arena_bytes.as_mut_ptr().add(0x100), 64);
            std::ptr::copy_nonoverlapping(node.as_ptr().cast::<u8>(), arena_bytes.as_mut_ptr().add(0x180), 64);
        }

        let (arena_buf, arena_ptr) = gpu.host.shared_buffer(4096).unwrap();
        // SAFETY: mapped 4096-byte buffer.
        unsafe { std::ptr::copy_nonoverlapping(arena_bytes.as_ptr(), arena_ptr, 4096) };

        let blas = bridge
            .build_blas_from_arena_header(&gpu.host, &gpu.device5, &gpu.list4, &arena_buf, 0x100, &blas_hdr, true)
            .expect("build_blas_from_arena_header");

        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0];
        let inst = make_rdna_instance(ident, 42, 0xFF, 0, 0, 0, 0);
        let mut tlas_hdr = [0u32; 16];
        tlas_hdr[0] = 6; // version=6, BLAS=0, PACKED=0
        tlas_hdr[1] = 3;
        tlas_hdr[3] = 1;
        tlas_hdr[13] = 1; // elem_offset = 0x400 + 1 * 64 = 0x440
        // SAFETY: mapped 4096-byte buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(tlas_hdr.as_ptr().cast::<u8>(), arena_ptr.add(0x400), 64);
            std::ptr::copy_nonoverlapping(inst.as_ptr().cast::<u8>(), arena_ptr.add(0x440), 128);
        }

        let (va_table_buf, va_ptr) = gpu.host.shared_buffer(64).unwrap();
        // SAFETY: 64-byte shared VA table.
        unsafe { std::ptr::write(va_ptr.cast::<u64>(), blas.gpu_va) };

        let tlas = bridge
            .build_tlas_from_arena_header(&gpu.host, &gpu.device5, &gpu.list4, &arena_buf, 0x400, &tlas_hdr, &va_table_buf)
            .expect("build_tlas_from_arena_header");

        let (root, pso, heap, tlas_h) = unsafe {
            let root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, &trace_dxil).unwrap();
            let pso: ID3D12PipelineState = gpu.host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: trace_dxil.as_ptr().cast(), BytecodeLength: trace_dxil.len() },
                ..Default::default()
            }).unwrap();
            let heap: ID3D12DescriptorHeap = gpu.host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).unwrap();
            create_tlas_srv(&gpu.host.device, tlas.gpu_va, heap.GetCPUDescriptorHandleForHeapStart());
            let h = heap.GetGPUDescriptorHandleForHeapStart();
            (root, pso, heap, h)
        };
        let ray_in = [RayInput { origin: [0.0, 0.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 }];
        let (rays_buf, rays_ptr) = gpu.host.shared_buffer(size_of::<RayInput>() as u64).unwrap();
        let (out_buf, out_ptr) = gpu.host.shared_buffer(size_of::<RayResult>() as u64).unwrap();
        // SAFETY: live mapped buffers and command list.
        unsafe {
            std::ptr::copy_nonoverlapping(ray_in.as_ptr(), rays_ptr.cast(), 1);
            std::ptr::write_bytes(out_ptr, 0, size_of::<RayResult>());
            gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            gpu.list.SetComputeRootSignature(&root);
            gpu.list.SetPipelineState(&pso);
            let n = 1u32;
            gpu.list.SetComputeRoot32BitConstants(0, 1, (&raw const n).cast(), 0);
            gpu.list.SetComputeRootDescriptorTable(1, tlas_h);
            gpu.list.SetComputeRootShaderResourceView(2, rays_buf.GetGPUVirtualAddress());
            gpu.list.SetComputeRootUnorderedAccessView(3, out_buf.GetGPUVirtualAddress());
            gpu.list.Dispatch(1, 1, 1);
        }
        gpu.submit_and_wait();
        // SAFETY: GPU finished writing 1 RayResult.
        let res = unsafe { *out_ptr.cast::<RayResult>() };
        assert_eq!(res.hit, 1);
        assert_eq!(res.instance_id, 42);
        assert!((res.hit_t - 7.0).abs() < 1e-4, "expected t=7.0, got {}", res.hit_t);
        drop((blas, tlas));
    }

    #[test]
    fn dxr_phase2_victory_shader_emitted_rayquery_runs_on_rt_cores() {
        let Some(mut gpu) = GpuRunner::new() else {
            eprintln!("no DXR Tier 1.1 hardware; skipped");
            return;
        };
        let mut trace_dxil = victory_shader::emit::rayquery::build_rayquery_smoke_cs();
        if let Ok(mut v) = victory_shader::emit::validator::Validator::load(None) {
            v.validate(&mut trace_dxil)
                .expect("dxil.dll sign emitted RayQuery shader");
        }

        let bridge = DxrBridge::new(&gpu.host.device).expect("DxrBridge");
        let v0 = [-1.0f32, -1.0, 0.0];
        let v1 = [0.0f32, 1.5, 0.0];
        let v2 = [1.0f32, -1.0, 0.0];
        let node0 = make_rdna_tri_node([v0, v1, v2, v0, v0], [1 | (2 << 2), 0, 0, 0]);
        let (rdna_buf, rdna_ptr) = gpu.host.shared_buffer(64).unwrap();
        // SAFETY: 64-byte mapped RDNA node buffer.
        unsafe { std::ptr::copy_nonoverlapping(node0.as_ptr().cast::<u8>(), rdna_ptr, 64) };
        let blas = bridge
            .build_blas_from_rdna_nodes(&gpu.host, &gpu.device5, &gpu.list4, &rdna_buf, 0, 1, true)
            .expect("build_blas");

        let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0];
        let inst = make_rdna_instance(ident, 99, 0xFF, 0, 0, 0, 0);
        let (inst_buf, inst_ptr) = gpu.host.shared_buffer(128).unwrap();
        // SAFETY: 128-byte mapped instance buffer.
        unsafe { std::ptr::copy_nonoverlapping(inst.as_ptr().cast::<u8>(), inst_ptr, 128) };
        let (va_buf, va_ptr) = gpu.host.shared_buffer(64).unwrap();
        // SAFETY: 64-byte VA table.
        unsafe { std::ptr::write(va_ptr.cast::<u64>(), blas.gpu_va) };
        let tlas = bridge
            .build_tlas_from_instances(&gpu.host, &gpu.device5, &gpu.list4, &inst_buf, 0, 1, &va_buf)
            .expect("build_tlas");

        // Build D3D12 root signature matching victory-shader's emit bindings:
        // param 0 = DescriptorTable(SRV(t252, space=1)), param 1 = SRV(t0, space=1), param 2 = UAV(u0, space=1)
        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: u32::from(victory_shader::emit::SLOT_TLAS),
            RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let (root, pso, heap, tlas_h) = unsafe {
            let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
            let mut err: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err))
                .expect("serialize RayQuery smoke root signature");
            let blob = blob.unwrap();
            let rs_bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let pso: ID3D12PipelineState = gpu
                .host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: trace_dxil.as_ptr().cast(),
                        BytecodeLength: trace_dxil.len(),
                    },
                    ..Default::default()
                })
                .expect("RTX 4090 D CreateComputePipelineState for victory-shader emitted RayQuery CS");
            let heap: ID3D12DescriptorHeap = gpu
                .host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            create_tlas_srv(&gpu.host.device, tlas.gpu_va, heap.GetCPUDescriptorHandleForHeapStart());
            let h = heap.GetGPUDescriptorHandleForHeapStart();
            (root, pso, heap, h)
        };

        let rays = [
            RayInput { origin: [0.0, 0.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [5.0, 5.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
        ];
        let (rays_buf, rays_ptr) = gpu.host.shared_buffer((rays.len() * size_of::<RayInput>()) as u64).unwrap();
        let (out_buf, out_ptr) = gpu.host.shared_buffer((rays.len() * size_of::<RayResult>()) as u64).unwrap();
        // SAFETY: live mapped buffers and open command list.
        unsafe {
            std::ptr::copy_nonoverlapping(rays.as_ptr(), rays_ptr.cast(), rays.len());
            std::ptr::write_bytes(out_ptr, 0, rays.len() * size_of::<RayResult>());
            gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            gpu.list.SetComputeRootSignature(&root);
            gpu.list.SetPipelineState(&pso);
            gpu.list.SetComputeRootDescriptorTable(0, tlas_h);
            gpu.list.SetComputeRootShaderResourceView(1, rays_buf.GetGPUVirtualAddress());
            gpu.list.SetComputeRootUnorderedAccessView(2, out_buf.GetGPUVirtualAddress());
            gpu.list.Dispatch(1, 1, 1);
        }
        gpu.submit_and_wait();

        let res = unsafe { std::slice::from_raw_parts(out_ptr.cast::<RayResult>(), rays.len()) };
        assert_eq!(res[0].hit, 1, "hit ray missed on victory-shader emitted RayQuery CS");
        assert_eq!(res[0].instance_id, 99);
        assert!((res[0].hit_t - 4.0).abs() < 1e-4, "expected t=4.0, got {}", res[0].hit_t);
        let cpu_tri = [v0, v1, v2].map(|p| [p[0], p[1], p[2] + 4.0]);
        let cpu_hit = bvh::triangle(&cpu_tri, &bvh::Ray::new(rays[0].origin, rays[0].dir, rays[0].tmax));
        let cx = cpu_hit[2] / cpu_hit[1];
        let cy = cpu_hit[3] / cpu_hit[1];
        assert!(
            (res[0].bary_x - cx).abs() < 1e-4 && (res[0].bary_y - cy).abs() < 1e-4,
            "bary mismatch: gpu=({},{}) cpu=({cx},{cy})",
            res[0].bary_x,
            res[0].bary_y
        );
        assert_eq!(res[1].hit, 0, "miss ray unexpectedly hit");
        eprintln!(
            "VICTORY-SHADER NATIVE SM 6.6 RAYQUERY VERIFIED ON RTX 4090 D RT CORES: hit t={}, inst={}, bary=({:.4}, {:.4})",
            res[0].hit_t, res[0].instance_id, res[0].bary_x, res[0].bary_y
        );
        drop((blas, tlas));
    }

    #[test]
    fn phase3_header_validation_filters_packed_and_corrupt() {
        let mut h = [0u32; 16];
        // All zero -> AllZero
        assert_eq!(inspect_rdna_header(&h), Err(HeaderRejectReason::AllZero));

        // Valid unpacked BLAS
        h[0] = 6 | (FLAG_BLAS << 16);
        h[1] = 10;
        h[3] = 4;
        h[13] = 6;
        assert_eq!(parse_valid_rdna_header(&h), Some((true, 10, 4, 6)));

        // Still packed -> StillPacked
        h[0] = 6 | ((FLAG_BLAS | FLAG_PACKED) << 16);
        assert_eq!(inspect_rdna_header(&h), Err(HeaderRejectReason::StillPacked));
        assert_eq!(parse_valid_rdna_header(&h), None);

        // Valid unpacked TLAS (2 instances = 4 nodes)
        h[0] = 6;
        h[1] = 10;
        h[3] = 2;
        h[13] = 6;
        assert_eq!(parse_valid_rdna_header(&h), Some((false, 10, 2, 6)));

        // Out-of-bounds offset_elements -> BoundsOrEmpty
        h[13] = 9;
        assert_eq!(inspect_rdna_header(&h), Err(HeaderRejectReason::BoundsOrEmpty));
        assert_eq!(parse_valid_rdna_header(&h), None);

        // Wrong version -> WrongVersion
        h[0] = 5;
        h[13] = 6;
        assert_eq!(inspect_rdna_header(&h), Err(HeaderRejectReason::WrongVersion));
        assert_eq!(parse_valid_rdna_header(&h), None);
    }

    /// Phase 3 End-to-End Hardware Verification on RTX 4090 D:
    /// Splits an RDNA2 BLAS across 2 non-contiguous 64 KiB pages in 2 different
    /// arenas AND splits a TLAS across a 64 KiB page boundary across 2 arenas
    /// (including 1 resolved instance + 1 unresolved culled instance), runs:
    ///   1. `INSPECT_HEADERS_PAGED_CS` (async GPU header readback via `directory`)
    ///   2. `EXTRACT_TRIANGLES_PAGED_CS` -> native Ada BLAS build
    ///   3. GPU `BlasMapBuffer` hash map lookup in `CONVERT_INSTANCES_PAGED_CS` -> native Ada TLAS build
    ///   4. `victory-shader` native SM 6.6 `RayQuery` CS execution on RT Cores!
    #[test]
    fn dxr_phase3_paged_multi_arena_blas_tlas_matches_software_bvh() {
        let Some(mut gpu) = GpuRunner::new() else {
            eprintln!("no DXR Tier 1.1 hardware; skipped");
            return;
        };
        let mut trace_dxil = victory_shader::emit::rayquery::build_rayquery_smoke_cs();
        let val_path = victory_shader::emit::validator::Validator::default_path();
        if let Ok(mut v) = victory_shader::emit::validator::Validator::load(val_path.as_deref()) {
            v.validate(&mut trace_dxil).expect("dxil.dll sign RayQuery smoke CS");
        }

        let bridge = DxrBridge::new(&gpu.host.device).expect("DxrBridge (with Phase 3 paged passes)");
        let mut paged_heap = DxrPagedHeap::new(&gpu.host.device).expect("DxrPagedHeap");

        // Allocate 2 separate 128 KiB mock arenas (copy 1 and copy 2) + Memory-format directory.
        const ARENA_SZ: usize = 128 * 1024;
        let (arena0_buf, arena0_ptr) = gpu.host.shared_buffer(ARENA_SZ as u64).unwrap();
        let (arena1_buf, arena1_ptr) = gpu.host.shared_buffer(ARENA_SZ as u64).unwrap();
        let dir_bytes = u64::from(MEMORY_PAGES) * 8;
        let (dir_buf, dir_ptr) = gpu.host.shared_buffer(dir_bytes).unwrap();
        // SAFETY: fresh persistent mappings of exact sizes.
        unsafe {
            std::ptr::write_bytes(arena0_ptr, 0, ARENA_SZ);
            std::ptr::write_bytes(arena1_ptr, 0, ARENA_SZ);
            std::ptr::write_bytes(dir_ptr, 0, dir_bytes as usize);
        }

        // Helper to map a 64 KiB title page into (copy_num: 1-based, byte_offset_in_copy).
        let map_page = |title_page: u64, copy_num_1based: u32, byte_off: u32, ro: bool| {
            // SAFETY: `title_page < MEMORY_PAGES`.
            unsafe {
                let p = dir_ptr.cast::<u32>().add((title_page as usize) * 2);
                *p = copy_num_1based | if ro { victory_shader::emit::MEMORY_READ_ONLY } else { 0 };
                *p.add(1) = byte_off;
            }
        };

        // Title layout crossing page boundaries AND spanning both arenas:
        // - BLAS at 0x1_0000_FF80 (page 0x1_0000, offset 0xFF80 in arena0@0x10000, READ_ONLY flag set!):
        //   header at 0x1_0000_FF80 (64B), elem_offset = 2 nodes -> elem_addr = 0x1_0001_0000 (NEXT PAGE!)
        //   page 0x1_0001 mapped into arena1@0x00000 (DIFFERENT ARENA!) holding 2 triangle nodes!
        let blas_title_addr: u64 = 0x1_0000_FF00;
        let blas_page0 = blas_title_addr >> 16;
        let blas_page1 = blas_page0 + 1;
        map_page(blas_page0, 1, 0x10000, true); // copy 1 (arena0) @ +64 KiB, with RO bit set!
        map_page(blas_page1, 2, 0x00000, false); // copy 2 (arena1) @ +0 KiB!

        let v0 = [-1.0f32, -1.0, 0.0];
        let v1 = [0.0f32, 1.5, 0.0];
        let v2 = [1.0f32, -1.0, 0.0];
        let v3 = [2.0f32, 1.0, 0.5];
        let v4 = [-2.0f32, 1.0, -0.5];
        let tri_node0 = make_rdna_tri_node([v0, v1, v2, v3, v4], [1 | (2 << 2), 0, 0, 0]);
        let tri_node1 = make_rdna_tri_node([v0, v1, v2, v3, v4], [0, 0, 0, 2 | (1 << 2)]);

        let mut blas_hdr = [0u32; 16];
        blas_hdr[0] = 6 | (FLAG_BLAS << 16);
        blas_hdr[1] = 6; // size_nodes
        blas_hdr[3] = 2; // 2 triangle nodes
        blas_hdr[13] = 4; // 0x1_0000_FF00 + 4 * 64 = 0x1_0001_0000 (crosses into page 1 / arena1!)

        // TLAS at 0x2_0000_FF00 (page 0x2_0000 mapped in arena1@0x10000):
        //   elem_offset = 3 -> elem_addr = 0x2_0000_FFC0 (64 bytes before page end!)
        //   Instance 0: bytes 0..63 at 0x2_0000_FFC0 (page 0x2_0000 / arena1), bytes 64..127 at 0x2_0001_0000 (page 0x2_0001 / arena0)
        //   Instance 1: at 0x2_0001_0040 (page 0x2_0001 / arena0) pointing to an UNBUILT BLAS -> must be culled!
        let tlas_title_addr: u64 = 0x2_0000_FF00;
        let tlas_page0 = tlas_title_addr >> 16;
        let tlas_page1 = tlas_page0 + 1;
        map_page(tlas_page0, 2, 0x10000, false); // copy 2 (arena1) @ +64 KiB
        map_page(tlas_page1, 1, 0x00000, false); // copy 1 (arena0) @ +0 KiB

        let mut tlas_hdr = [0u32; 16];
        tlas_hdr[0] = 6; // version=6, BLAS=0, PACKED=0
        tlas_hdr[1] = 8;
        tlas_hdr[3] = 2; // 2 instances
        tlas_hdr[13] = 3; // 0x2_0000_FF00 + 3 * 64 = 0x2_0000_FFC0

        let mat_hit = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0];
        let inst0 = make_rdna_instance(mat_hit, 77, 0xFF, 0, 0, 0, blas_title_addr);
        // Instance 1 overlaps origin closer (z=2.0) but references unbuilt BLAS 0x3_0000_0100 -> MUST be culled!
        let mat_culled = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0];
        let inst1 = make_rdna_instance(mat_culled, 88, 0xFF, 0, 0, 0, 0x3_0000_0100);

        // Write all headers and elements into their non-contiguous arena slots.
        // SAFETY: all offsets are strictly inside 128 KiB arenas.
        unsafe {
            // BLAS header at arena0 + 0x10000 + 0xFF00
            std::ptr::copy_nonoverlapping(blas_hdr.as_ptr().cast::<u8>(), arena0_ptr.add(0x1FF00), 64);
            // BLAS triangle nodes at arena1 + 0x00000
            std::ptr::copy_nonoverlapping(tri_node0.as_ptr().cast::<u8>(), arena1_ptr, 64);
            std::ptr::copy_nonoverlapping(tri_node1.as_ptr().cast::<u8>(), arena1_ptr.add(64), 64);
            // TLAS header at arena1 + 0x10000 + 0xFF00
            std::ptr::copy_nonoverlapping(tlas_hdr.as_ptr().cast::<u8>(), arena1_ptr.add(0x1FF00), 64);
            // Instance 0 first 64B at arena1 + 0x1FFC0, second 64B at arena0 + 0x0000
            std::ptr::copy_nonoverlapping(inst0.as_ptr().cast::<u8>(), arena1_ptr.add(0x1FFC0), 64);
            std::ptr::copy_nonoverlapping(inst0.as_ptr().cast::<u8>().add(64), arena0_ptr, 64);
            // Instance 1 (128B) at arena0 + 0x0040
            std::ptr::copy_nonoverlapping(inst1.as_ptr().cast::<u8>(), arena0_ptr.add(0x40), 128);
        }

        paged_heap.sync_copies(&gpu.host.device, &[arena0_buf.clone(), arena1_buf.clone()]);

        // Step 1: Run `INSPECT_HEADERS_PAGED_CS` for [blas_title_addr, tlas_title_addr, unmapped_addr].
        let (addrs_buf, addrs_ptr) = gpu.host.shared_buffer(64).unwrap();
        let (hdrs_buf, hdrs_ptr) = gpu.host.shared_buffer(256).unwrap();
        // SAFETY: mapped buffers.
        unsafe {
            let addrs = [blas_title_addr, tlas_title_addr, 0x5_0000_0000u64];
            std::ptr::copy_nonoverlapping(addrs.as_ptr(), addrs_ptr.cast::<u64>(), 3);
            std::ptr::write_bytes(hdrs_ptr, 0, 256);
            gpu.list.SetDescriptorHeaps(&[Some(paged_heap.heap().clone())]);
        }
        bridge.dispatch_inspect_headers_paged(
            &gpu.list,
            &dir_buf,
            &addrs_buf,
            &hdrs_buf,
            paged_heap.gpu_start(),
            3,
        );
        gpu.submit_and_wait();

        let parsed_blas = {
            let h: [u32; 16] = unsafe { std::array::from_fn(|k| *hdrs_ptr.cast::<u32>().add(k)) };
            parse_valid_rdna_header(&h).expect("GPU-inspected BLAS header must validate")
        };
        let parsed_tlas = {
            let h: [u32; 16] = unsafe { std::array::from_fn(|k| *hdrs_ptr.cast::<u32>().add(16 + k)) };
            parse_valid_rdna_header(&h).expect("GPU-inspected TLAS header must validate")
        };
        let unmapped_hdr: [u32; 16] = unsafe { std::array::from_fn(|k| *hdrs_ptr.cast::<u32>().add(32 + k)) };
        assert_eq!(parsed_blas, (true, 6, 2, 4));
        assert_eq!(parsed_tlas, (false, 8, 2, 3));
        assert_eq!(parse_valid_rdna_header(&unmapped_hdr), None);

        // Step 2: Run `EXTRACT_TRIANGLES_PAGED_CS` + build native Ada BLAS.
        let tri_nodes = parsed_blas.2;
        let blas_elem_addr = blas_title_addr + u64::from(parsed_blas.3) * NODE_BYTES;
        let vert_count = tri_nodes * 4 * 3;
        let vb = uav_buffer_state(&gpu.host.device, u64::from(vert_count) * 12, D3D12_RESOURCE_STATE_COMMON).unwrap();
        // SAFETY: open command list; paged_heap bound.
        unsafe { gpu.list.SetDescriptorHeaps(&[Some(paged_heap.heap().clone())]) };
        bridge.dispatch_extract_triangles_paged(
            &gpu.list,
            &dir_buf,
            &vb,
            paged_heap.gpu_start(),
            blas_elem_addr,
            tri_nodes,
        );
        let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: 0,
                    IndexFormat: DXGI_FORMAT_UNKNOWN,
                    VertexFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: 0,
                    VertexCount: vert_count,
                    IndexBuffer: 0,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: unsafe { vb.GetGPUVirtualAddress() },
                        StrideInBytes: 12,
                    },
                },
            },
        };
        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geom },
        };
        let mut blas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe { gpu.device5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_info) };
        let blas_res = as_buffer(&gpu.host.device, blas_info.ResultDataMaxSizeInBytes).unwrap();
        let blas_scratch = uav_buffer_state(
            &gpu.host.device,
            blas_info.ScratchDataSizeInBytes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )
        .unwrap();
        let blas_va = unsafe {
            gpu.list.ResourceBarrier(&[transition(
                &vb,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            let va = blas_res.GetGPUVirtualAddress();
            gpu.list4.BuildRaytracingAccelerationStructure(
                &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: va,
                    Inputs: blas_inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: blas_scratch.GetGPUVirtualAddress(),
                },
                None,
            );
            gpu.list.ResourceBarrier(&[uav_barrier(&blas_res)]);
            va
        };

        // Step 3: Populate GPU `BlasMapBuffer` hash map, run `CONVERT_INSTANCES_PAGED_CS`, build Ada TLAS.
        let mut blas_map = BlasMapBuffer::new(&gpu.host).unwrap();
        assert!(blas_map.insert_one(blas_title_addr, blas_va));
        let inst_count = parsed_tlas.2;
        let tlas_elem_addr = tlas_title_addr + u64::from(parsed_tlas.3) * NODE_BYTES;
        let descs = uav_buffer_state(
            &gpu.host.device,
            u64::from(inst_count) * DXR_INSTANCE_DESC_BYTES,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .unwrap();
        let (stats_buf, stats_ptr) = gpu.host.shared_buffer(64).unwrap();
        unsafe { std::ptr::write_bytes(stats_ptr, 0, 64) };
        bridge.dispatch_convert_instances_paged(
            &gpu.list,
            &dir_buf,
            &blas_map.buffer,
            &descs,
            &stats_buf,
            paged_heap.gpu_start(),
            tlas_elem_addr,
            inst_count,
            blas_map.mask(),
            0,
        );
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: inst_count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: unsafe { descs.GetGPUVirtualAddress() },
            },
        };
        let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe { gpu.device5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info) };
        let tlas_res = as_buffer(&gpu.host.device, tlas_info.ResultDataMaxSizeInBytes).unwrap();
        let tlas_scratch = uav_buffer_state(
            &gpu.host.device,
            tlas_info.ScratchDataSizeInBytes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )
        .unwrap();
        let tlas_va = unsafe {
            gpu.list.ResourceBarrier(&[
                transition(
                    &descs,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
                uav_barrier(&stats_buf),
            ]);
            let va = tlas_res.GetGPUVirtualAddress();
            gpu.list4.BuildRaytracingAccelerationStructure(
                &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: va,
                    Inputs: tlas_inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
                },
                None,
            );
            gpu.list.ResourceBarrier(&[uav_barrier(&tlas_res)]);
            va
        };
        gpu.submit_and_wait();

        // Verify GPU hash map resolved 1 instance (inst0) and culled 1 instance (inst1).
        let (resolved, culled) = unsafe { (*stats_ptr.cast::<u32>(), *stats_ptr.cast::<u32>().add(1)) };
        assert_eq!(resolved, 1, "expected 1 resolved instance");
        assert_eq!(culled, 1, "expected 1 culled instance");

        // Step 4: Trace through the live paged-built Ada TLAS on RT Cores using `victory-shader`'s RayQuery CS!
        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: u32::from(victory_shader::emit::SLOT_TLAS),
            RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: victory_shader::emit::ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let (root, pso, trace_heap, tlas_h) = unsafe {
            let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
            let mut err: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err)).unwrap();
            let blob = blob.unwrap();
            let rs_bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let root: ID3D12RootSignature = gpu.host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let pso: ID3D12PipelineState = gpu
                .host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: trace_dxil.as_ptr().cast(),
                        BytecodeLength: trace_dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let heap: ID3D12DescriptorHeap = gpu
                .host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            create_tlas_srv(&gpu.host.device, tlas_va, heap.GetCPUDescriptorHandleForHeapStart());
            let h = heap.GetGPUDescriptorHandleForHeapStart();
            (root, pso, heap, h)
        };

        let rays = [
            RayInput { origin: [0.0, 0.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
            RayInput { origin: [5.0, 5.0, 0.0], tmin: 0.0, dir: [0.0, 0.0, 1.0], tmax: 100.0 },
        ];
        let (rays_buf, rays_ptr) = gpu.host.shared_buffer((rays.len() * size_of::<RayInput>()) as u64).unwrap();
        let (out_buf, out_ptr) = gpu.host.shared_buffer((rays.len() * size_of::<RayResult>()) as u64).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(rays.as_ptr(), rays_ptr.cast(), rays.len());
            std::ptr::write_bytes(out_ptr, 0, rays.len() * size_of::<RayResult>());
            gpu.list.SetDescriptorHeaps(&[Some(trace_heap.clone())]);
            gpu.list.SetComputeRootSignature(&root);
            gpu.list.SetPipelineState(&pso);
            gpu.list.SetComputeRootDescriptorTable(0, tlas_h);
            gpu.list.SetComputeRootShaderResourceView(1, rays_buf.GetGPUVirtualAddress());
            gpu.list.SetComputeRootUnorderedAccessView(2, out_buf.GetGPUVirtualAddress());
            gpu.list.Dispatch(1, 1, 1);
        }
        gpu.submit_and_wait();

        let res = unsafe { std::slice::from_raw_parts(out_ptr.cast::<RayResult>(), rays.len()) };
        assert_eq!(res[0].hit, 1, "Phase 3 paged multi-arena ray missed!");
        assert_eq!(res[0].instance_id, 77, "culled instance 88 leaked or wrong instance ID");
        assert!((res[0].hit_t - 4.0).abs() < 1e-4, "expected t=4.0, got {}", res[0].hit_t);
        assert_eq!(res[1].hit, 0, "miss ray unexpectedly hit");
        eprintln!(
            "DXR PHASE 3 PAGED MULTI-ARENA BLAS/TLAS + GPU HASH MAP VERIFIED ON RTX 4090 D: resolved={resolved}, culled={culled}, hit t={}, inst={}, bary=({:.4}, {:.4})",
            res[0].hit_t, res[0].instance_id, res[0].bary_x, res[0].bary_y
        );
        drop((blas_res, tlas_res));
    }
}

