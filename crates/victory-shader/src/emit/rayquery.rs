//! DXIL SM 6.6 `RayQuery` (`TraceRayInline`) emission helpers for hardware
//! RT Core traversal on DXR Tier 1.1 hosts (Phase 2B).

use super::bitcode::{Attr, Bin, Cast, Term, V};
use super::shader::{SLOT_TLAS, Shader, flags};

impl Shader {
    /// Allocates an inline ray query (`dx.op.allocateRayQuery`, opcode 178).
    pub fn allocate_ray_query(&mut self, block: usize, const_ray_flags: u32) -> V {
        self.flags |= flags::RAYTRACING_TIER_1_1;
        let f = self.m.i32(const_ray_flags);
        let i32t = self.t.i32;
        self.op(block, "allocateRayQuery", Attr::NoUnwind, i32t, 178, &[f])
    }

    /// Initializes inline traversal against a `RaytracingAccelerationStructure`
    /// (`dx.op.rayQuery_TraceRayInline`, opcode 179).
    #[allow(clippy::too_many_arguments)]
    pub fn ray_query_trace_inline(
        &mut self,
        block: usize,
        rq: V,
        tlas: V,
        ray_flags: V,
        instance_mask: V,
        origin: [V; 3],
        tmin: V,
        dir: [V; 3],
        tmax: V,
    ) {
        self.flags |= flags::RAYTRACING_TIER_1_1;
        let void = self.t.void;
        self.op(
            block,
            "rayQuery_TraceRayInline",
            Attr::NoUnwind,
            void,
            179,
            &[
                rq,
                tlas,
                ray_flags,
                instance_mask,
                origin[0],
                origin[1],
                origin[2],
                tmin,
                dir[0],
                dir[1],
                dir[2],
                tmax,
            ],
        );
    }

    /// Advances hardware BVH traversal (`dx.op.rayQuery_Proceed.i1`, opcode 180).
    pub fn ray_query_proceed(&mut self, block: usize, rq: V) -> V {
        let i1t = self.t.i1;
        self.op(block, "rayQuery_Proceed.i1", Attr::NoUnwind, i1t, 180, &[rq])
    }

    /// Reads `CommittedStatus()` (`0` = none, `1` = triangle, `2` = procedural).
    pub fn ray_query_committed_status(&mut self, block: usize, rq: V) -> V {
        let i32t = self.t.i32;
        self.op(block, "rayQuery_StateScalar.i32", Attr::ReadOnly, i32t, 184, &[rq])
    }

    /// Reads `CommittedRayT()`.
    pub fn ray_query_committed_ray_t(&mut self, block: usize, rq: V) -> V {
        let f32t = self.t.f32;
        self.op(block, "rayQuery_StateScalar.f32", Attr::ReadOnly, f32t, 200, &[rq])
    }

    /// Reads `CommittedInstanceID()`.
    pub fn ray_query_committed_instance_id(&mut self, block: usize, rq: V) -> V {
        let i32t = self.t.i32;
        self.op(block, "rayQuery_StateScalar.i32", Attr::ReadOnly, i32t, 208, &[rq])
    }

    /// Reads `CommittedPrimitiveIndex()`.
    pub fn ray_query_committed_primitive_index(&mut self, block: usize, rq: V) -> V {
        let i32t = self.t.i32;
        self.op(block, "rayQuery_StateScalar.i32", Attr::ReadOnly, i32t, 210, &[rq])
    }

    /// Reads `CommittedTriangleBarycentrics()` (`(u, v)`).
    pub fn ray_query_committed_barycentrics(&mut self, block: usize, rq: V) -> (V, V) {
        let f32t = self.t.f32;
        let c0 = self.m.i8(0);
        let c1 = self.m.i8(1);
        let u = self.op(block, "rayQuery_StateVector.f32", Attr::ReadOnly, f32t, 194, &[rq, c0]);
        let v = self.op(block, "rayQuery_StateVector.f32", Attr::ReadOnly, f32t, 194, &[rq, c1]);
        (u, v)
    }
}

/// Builds a standalone SM 6.6 DXIL compute shader using `victory-shader`'s own
/// bitcode emitter (no `dxc.exe`) to verify end-to-end RT Core execution on the
/// host GPU:
/// - `t0` (`space1`): `ByteAddressBuffer` of 32-byte `RayIn { origin.xyz, tmin, dir.xyz, tmax }`
/// - `t252` (`space1`, [`SLOT_TLAS`]): `RaytracingAccelerationStructure`
/// - `u0` (`space1`): `RWByteAddressBuffer` of 32-byte `RayOut { hit, inst, prim, t, bx, by, 0, 0 }`
pub fn build_rayquery_smoke_cs() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([64, 1, 1]);

    let entry = s.entry;
    let rays_srv = s.root_srv_handle(0);
    let tlas = s.tlas_handle(SLOT_TLAS);
    let out_uav = s.root_uav_handle(0);

    let tid_c = s.m.i32(0);
    let tid = s.op(entry, "threadId.i32", Attr::ReadNone, s.t.i32, 93, &[tid_c]);
    let five = s.m.i32(5);
    let ray_byte = s.m.bin(entry, Bin::Shl, tid, five);
    let sixteen = s.m.i32(16);
    let ray_byte2 = s.m.bin(entry, Bin::Add, ray_byte, sixteen);

    let undef_i32 = s.m.undef(s.t.i32);
    let mask_xyzw = s.m.i8(15);
    let align4 = s.m.i32(4);
    let res_ret = s.t.res_ret;
    let r0 = s.op(
        entry,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        res_ret,
        139,
        &[rays_srv, ray_byte, undef_i32, mask_xyzw, align4],
    );
    let r1 = s.op(
        entry,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        res_ret,
        139,
        &[rays_srv, ray_byte2, undef_i32, mask_xyzw, align4],
    );
    let ox = s.m.extract(entry, r0, 0);
    let oy = s.m.extract(entry, r0, 1);
    let oz = s.m.extract(entry, r0, 2);
    let tmin = s.m.extract(entry, r0, 3);
    let dx = s.m.extract(entry, r1, 0);
    let dy = s.m.extract(entry, r1, 1);
    let dz = s.m.extract(entry, r1, 2);
    let tmax = s.m.extract(entry, r1, 3);

    let rq = s.allocate_ray_query(entry, 0);
    let flags_z = s.m.i32(0);
    let inst_mask = s.m.i32(0xFF);
    s.ray_query_trace_inline(entry, rq, tlas, flags_z, inst_mask, [ox, oy, oz], tmin, [dx, dy, dz], tmax);

    let loop_blk = s.m.new_block();
    let done_blk = s.m.new_block();
    s.m.terminate(entry, Term::Br(loop_blk));

    let more = s.ray_query_proceed(loop_blk, rq);
    s.m.terminate(loop_blk, Term::CondBr(more, loop_blk, done_blk));

    let hit = s.ray_query_committed_status(done_blk, rq);
    let inst = s.ray_query_committed_instance_id(done_blk, rq);
    let prim = s.ray_query_committed_primitive_index(done_blk, rq);
    let t_f32 = s.ray_query_committed_ray_t(done_blk, rq);
    let (bx_f32, by_f32) = s.ray_query_committed_barycentrics(done_blk, rq);

    let i32t = s.t.i32;
    let t_i32 = s.m.cast(done_blk, Cast::Bitcast, t_f32, i32t);
    let bx_i32 = s.m.cast(done_blk, Cast::Bitcast, bx_f32, i32t);
    let by_i32 = s.m.cast(done_blk, Cast::Bitcast, by_f32, i32t);
    let zero = s.m.i32(0);
    let void_ty = s.t.void;

    s.op(
        done_blk,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        void_ty,
        140,
        &[out_uav, ray_byte, undef_i32, hit, inst, prim, t_i32, mask_xyzw, align4],
    );
    s.op(
        done_blk,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        void_ty,
        140,
        &[out_uav, ray_byte2, undef_i32, bx_i32, by_i32, zero, zero, mask_xyzw, align4],
    );
    s.m.terminate(done_blk, Term::Ret);

    s.finish()
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::emit::validator::Validator;

    #[test]
    fn emitted_rayquery_smoke_cs_validates_with_dxil_dll() {
        let mut dxil = build_rayquery_smoke_cs();
        let path = Validator::default_path();
        let Ok(mut v) = Validator::load(path.as_deref()) else {
            eprintln!("dxil.dll not found; skipped RayQuery DXIL validation");
            return;
        };
        v.validate(&mut dxil)
            .expect("dxil.dll validation of emitted RayQuery shader");
    }
}
