// Auto-generated shader embedding
#include <string>

const char* wgsl_argmax = R"wgsl(@group(0) @binding(0)
#ifdef VEC4
var<storage, read_write> src: array<vec4<f32>>;
#define VEC_SIZE 4
#else
var<storage, read_write> src: array<f32>;
#define VEC_SIZE 1
#endif

@group(0) @binding(1)
var<storage, read_write> dst: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    ne0: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;

struct Pair {
    value: f32,
    index: i32
};

var<workgroup> shared_max: array<Pair, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row_idx = params.offset_src + wid.x * params.ne0;
    var local_pair = Pair(FLOAT_MIN, -1);
#ifdef VEC4
    for (var col = lid.x; col < params.ne0/VEC_SIZE; col += WG_SIZE) {
        let vec_val = src[row_idx / VEC_SIZE + col];
        for (var v = 0u; v < VEC_SIZE; v++) {
            let val = vec_val[v];
            if (val >= local_pair.value) {
                local_pair = Pair(val, i32(col * VEC_SIZE + v));
            }
        }
    }
#else
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        if (src[row_idx + col] >= local_pair.value) {
            local_pair = Pair(src[row_idx + col], i32(col));
        }
    }
#endif
    shared_max[lid.x] = local_pair;
    workgroupBarrier();
    var offset: u32 = WG_SIZE >> 1;
    while (offset > 0) {
        if (lid.x < offset) {
            let a = shared_max[lid.x];
            let b = shared_max[lid.x + offset];
            if (b.value > a.value) {
                shared_max[lid.x] = b;
            } else if (b.value == a.value && b.index > a.index) {
                shared_max[lid.x] = b;
            }
        }
        workgroupBarrier();
        offset >>= 1;
    }
    if (lid.x == 0u) {
        dst[params.offset_dst + wid.x] = shared_max[0].index;
    }
}
)wgsl";

const char* wgsl_argsort = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // src/dst dimensions
    src_ne0: u32,
    ne1: u32,
    ne2: u32,

    ne0: u32,
    top_k: u32,

    npr: u32,   // tiles per row
    nrows: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shmem_idx: array<u32, WG_SIZE>;

#if ORDER == 0
#define EXTREME_VALUE 1e30
#define SWAP_COMPARE_UP >
#define SWAP_COMPARE_DOWN <
#else
#define EXTREME_VALUE -1e30
#define SWAP_COMPARE_UP <
#define SWAP_COMPARE_DOWN >
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let linear = wid.x + wid.y * num_wg.x;
    // guard against overprovisioned workgroups
    if (linear >= params.npr * params.nrows) {
        return;
    }
    let tile = linear % params.npr;
    var row = linear / params.npr;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_base = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let tile_base = tile * WG_SIZE;
    let idx = tile_base + lid.x;
    shmem_idx[lid.x] = select(params.src_ne0, idx, idx < params.src_ne0);
    workgroupBarrier();

    var k = 2u;
    while (k <= WG_SIZE) {
        var j = k >> 1;
        while (j > 0) {
            let ixj = lid.x ^ j;
            if (ixj > lid.x) {
                let dir_up = (lid.x & k) == 0;
                let a_idx = shmem_idx[lid.x];
                let b_idx = shmem_idx[ixj];
                let a_val = select(EXTREME_VALUE, src[row_base + a_idx], a_idx < params.src_ne0);
                let b_val = select(EXTREME_VALUE, src[row_base + b_idx], b_idx < params.src_ne0);
                let should_swap = select(
                    (a_val SWAP_COMPARE_DOWN b_val),
                    (a_val SWAP_COMPARE_UP b_val),
                    dir_up);
                if (should_swap) {
                    shmem_idx[lid.x] = b_idx;
                    shmem_idx[ixj] = a_idx;
                }
            }
            workgroupBarrier();
            j >>= 1;
        }
        k <<= 1;
    }

    let out_idx = tile * params.top_k + lid.x;
    if (out_idx < params.ne0 && lid.x < params.top_k) {
        let row_dst = params.offset_dst +
            i1 * params.stride_dst1 +
            i2 * params.stride_dst2 +
            i3 * params.stride_dst3;
        dst[row_dst + out_idx] = i32(shmem_idx[lid.x]);
    }
}
)wgsl";

const char* wgsl_argsort_merge = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> idx_in: array<i32>;

@group(0) @binding(2)
var<storage, read_write> idx_out: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_in: u32,  // in elements
    offset_out: u32, // in elements

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx1: u32,
    stride_idx2: u32,
    stride_idx3: u32,

    stride_out1: u32,
    stride_out2: u32,
    stride_out3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,

    top_k: u32,

    len: u32,
    nm: u32,
    nrows: u32
};

@group(0) @binding(3)
var<uniform> params: Params;

fn take_left(a_idx: i32, b_idx: i32, row_base: u32) -> bool {
    let a_val = src[row_base + u32(a_idx)];
    let b_val = src[row_base + u32(b_idx)];
#if ORDER == 0
    return a_val <= b_val;
#else
    return a_val >= b_val;
#endif
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let linear = wid.x + wid.y * num_wg.x;
    // guard against overprovisioned workgroups
    if (linear >= params.nm * params.nrows) {
        return;
    }

    let start = (linear % params.nm) * params.len * 2;
    let len0 = min(params.len, params.ne0 - start);
    let rem1 = select(0, params.ne0 - (start + params.len), params.ne0 > (start + params.len));
    let len1 = min(params.len, rem1);
    let total = len0 + len1;
    let chunk = (total + WG_SIZE - 1u) / WG_SIZE;
    let k0 = lid.x * chunk;
    let k1 = min(min(k0 + chunk, total), params.top_k);
    // guard against overprovisioned threads
    if (k0 >= params.top_k || k0 >= total) {
        return;
    }

    var row = linear / params.nm;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_src = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let row_in = params.offset_in +
        i1 * params.stride_idx1 +
        i2 * params.stride_idx2 +
        i3 * params.stride_idx3;

    let row_out = params.offset_out +
        i1 * params.stride_out1 +
        i2 * params.stride_out2 +
        i3 * params.stride_out3;


    var low: u32 = select(0, k0 - len1, k0 > len1);
    var high: u32 = min(k0, len0);

    while (low < high) {
        let mid = (low + high) >> 1;
        let idx0 = idx_in[row_in + start + mid];
        let idx1 = idx_in[row_in + start + params.len + (k0 - mid - 1)];
        if (take_left(idx0, idx1, row_src)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    var i = low;
    var j = k0 - i;
    var k = k0;
    while (k < k1) {
        var take_l = false;
        if (i >= len0) {
            take_l = false;
        } else if (j >= len1) {
            take_l = true;
        } else {
            let idx0 = idx_in[row_in + start + i];
            let idx1 = idx_in[row_in + start + params.len + j];
            take_l = take_left(idx0, idx1, row_src);
        }

        let out_idx = select(
            idx_in[row_in + start + params.len + j],
            idx_in[row_in + start + i],
            take_l);
        idx_out[row_out + start + k] = out_idx;
        i = select(i, i + 1, take_l);
        j = select(j + 1, j, take_l);
        k += 1;
    }
}
)wgsl";

const char* wgsl_binary = R"wgsl(enable f16;

struct Params {
    ne: u32,

    // offsets in elements
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    offset_merged_src0: u32,
    offset_merged_src1: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,

    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
};

fn src0_index(_i: u32) -> u32 {
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;

    return a_i0 * params.stride_src0_0 +
           a_i1 * params.stride_src0_1 +
           a_i2 * params.stride_src0_2 +
           a_i3 * params.stride_src0_3;
}

fn src1_index(_i: u32) -> u32 {
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;

    // handle repetition of b
    // index loops back to the beginning and repeats after elements are exhausted = modulo
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;

    // compute index for position in b's flat array
    return b_i0 * params.stride_src1_0 +
           b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 +
           b_i3 * params.stride_src1_3;
}

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_F16
#define DataType f16
#endif

#ifdef SRC_OVERLAP
@group(0) @binding(0)
var<storage, read_write> merged_src: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(2)
var<uniform> params: Params;
#else
@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> src1 : array<DataType>;
#if defined(INPLACE) || defined(OVERLAP)
@group(0) @binding(2)
var<uniform> params: Params;

#else
@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
var<uniform> params: Params;
#endif
#endif

fn op(a: DataType, b: DataType) -> DataType {
#ifdef OP_ADD
    return a + b;
#elif defined(OP_SUB)
    return a - b;
#elif defined(OP_MUL)
    return a * b;
#elif defined(OP_DIV)
    return a / b;
#endif
}

fn update(dst_i: u32, src0_i: u32, src1_i: u32) {
#ifdef SRC_OVERLAP
    let result = op(merged_src[src0_i], merged_src[src1_i]);
#else
    let result = op(src0[src0_i], src1[src1_i]);
#endif

#ifdef INPLACE
    src0[src0_i] = result;
#elif defined(OVERLAP)
    src1[src1_i] = result;
#else
    dst[dst_i] = result;
#endif
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        let src0_i = params.offset_src0 + params.offset_merged_src0 + src0_index(gid.x);
        let src1_i = params.offset_src1 + params.offset_merged_src1 + src1_index(gid.x);
        update(params.offset_dst + gid.x, src0_i, src1_i);
    }
}
)wgsl";

const char* wgsl_concat = R"wgsl(struct Params {
    ne: u32,

    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    dim: u32,
    src0_nedim: u32
};

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_I32
#define DataType i32
#endif

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> src1 : array<DataType>;

@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    if (gid.x < params.ne) {
        var i = gid.x;
        let i3 = i / (params.ne2 * params.ne1 * params.ne0);
        i = i % (params.ne2 * params.ne1 * params.ne0);
        let i2 = i / (params.ne1 * params.ne0);
        i = i % (params.ne1 * params.ne0);
        let i1 = i / params.ne0;
        let i0 = i % params.ne0;

        var ni = array<u32, 4>(i0, i1, i2, i3);

        if (ni[params.dim] < params.src0_nedim) {
            let src_i = ni[0] * params.stride_src0_0 +
                             ni[1] * params.stride_src0_1 +
                             ni[2] * params.stride_src0_2 +
                             ni[3] * params.stride_src0_3;
            dst[params.offset_dst + gid.x] = src0[params.offset_src0 + src_i];
        } else {
            ni[params.dim] -= params.src0_nedim;
            let src_i = ni[0] * params.stride_src1_0 +
                             ni[1] * params.stride_src1_1 +
                             ni[2] * params.stride_src1_2 +
                             ni[3] * params.stride_src1_3;
            dst[params.offset_dst + gid.x] = src1[params.offset_src1 + src_i];
        }
    }
}
)wgsl";

const char* wgsl_cpy = R"wgsl(enable f16;

#ifdef SRC_F32
#define SRC_TYPE f32
#elif defined(SRC_F16)
#define SRC_TYPE f16
#endif

#ifdef DST_F32
#define DST_TYPE f32
#elif defined(DST_F16)
#define DST_TYPE f16
#elif defined(DST_I32)
#define DST_TYPE i32
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> dst: array<DST_TYPE>;

struct Params{
    ne: u32,
    offset_src: u32,
    offset_dst: u32,

    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,


    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let i3 = i / (params.src_ne2 * params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne2 * params.src_ne1 * params.src_ne0);
    let i2 = i / (params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne1 * params.src_ne0);
    let i1 = i / params.src_ne0;
    let i0 = i % params.src_ne0;

    var j = gid.x;
    let j3 = j / (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    let j2 = j / (params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne1 * params.dst_ne0);
    let j1 = j / params.dst_ne0;
    let j0 = j % params.dst_ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

    let dst_idx = j0 * params.stride_dst0 + j1 * params.stride_dst1 +
                  j2 * params.stride_dst2 + j3 * params.stride_dst3;

    dst[params.offset_dst + dst_idx] = DST_TYPE((src[params.offset_src + src_idx]));
}

)wgsl";

const char* wgsl_cumsum = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    ne0: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row_idx = params.offset_src + wid.x * params.ne0;
    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;
    var local_sum: f32 = 0.0;
    for (var col = lid.x * elems; col < (lid.x + 1) * elems && col < params.ne0; col ++) {
        local_sum += src[row_idx + col];
    }
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    // upsweep
    var offset = 1u;
    while (offset < WG_SIZE) {
        let idx = (lid.x + 1) * offset * 2 - 1;
        if (idx < WG_SIZE) {
            shared_sum[idx] = shared_sum[idx] + shared_sum[idx - offset];
        }
        workgroupBarrier();
        offset <<= 1;
    }

    // set last to 0 for exclusive sum
    if (lid.x == 0) {
        shared_sum[WG_SIZE - 1] = 0.0;
    }
    workgroupBarrier();

    // downsweep
    offset = WG_SIZE >> 1;
    while (offset > 0) {
        let idx = (lid.x + 1) * offset * 2 - 1;
        if (idx < WG_SIZE) {
            let t = shared_sum[idx - offset];
            shared_sum[idx - offset] = shared_sum[idx];
            shared_sum[idx] = shared_sum[idx] + t;
        }
        workgroupBarrier();
        offset = offset >> 1;
    }

    // shared_sum[lid] is exclusive prefix sum up to this thread.
    var running_sum = shared_sum[lid.x];
    for (var col = lid.x * elems; col < (lid.x + 1) * elems && col < params.ne0; col ++) {
        running_sum += src[row_idx + col];
        dst[params.offset_dst + wid.x * params.ne0 + col] = running_sum;
    }
}
)wgsl";

const char* wgsl_flash_attn = R"wgsl(diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

#ifdef KV_F32
#define KV_TYPE f32
#elif defined(KV_Q4_0) || defined(KV_Q8_0)
#define KV_TYPE u32
#else
#define KV_TYPE f16
#endif

// Default values
#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64

// The number of rows/columns/k in a subgroup matrix. MxK * KxN = MxN
// Note that the "K" here does not correspond to the K in attention's Q/K/V, it's just the common dimension.
#define SG_MAT_M 8
#define SG_MAT_N 8
#define SG_MAT_K 8

// Each workgroup processes one subgroup matrix of Q rows
#define Q_TILE SG_MAT_M
#define KV_TILE 16
#define WG_SIZE 64

// Number of subgroup-matrix-width blocks that span the KV tile. SG_MAT_N must divide KV_TILE.
#define KV_BLOCKS (KV_TILE / SG_MAT_N)

// Quantization constants/helpers
#define BLOCK_SIZE 32
#define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_V ((HEAD_DIM_V + BLOCK_SIZE - 1) / BLOCK_SIZE)
// number of quantized elements processed per thread
#if defined(KV_Q4_0)
#define NQ 16
// Q4_0 has 32 elements, 1 f16 for scale, 8 f16 for 4-bit weights
#define F16_PER_BLOCK 9
#define BLOCK_SIZE_BYTES 18u
#define WEIGHTS_PER_F16 4
#elif defined(KV_Q8_0)
#define NQ 8
// Q8_0 has 32 elements, 1 f16 for scale, 16 f16 for 8-bit weights
#define F16_PER_BLOCK 17
#define BLOCK_SIZE_BYTES 34u
#define WEIGHTS_PER_F16 2
#endif
#define F16_PER_THREAD (NQ / WEIGHTS_PER_F16)

// Ok not to put these in a define block, compiler will remove if unused
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

#if defined(KV_Q4_0) || defined(KV_Q8_0)
fn load_k_u16_at(byte_offset: u32) -> u32 {
    let word = K[byte_offset / 4u];
    let shift = (byte_offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_k_u32_at(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = K[word_idx];
    if (shift == 0u) {
        return lo;
    }
    let hi = K[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
}

fn load_v_u16_at(byte_offset: u32) -> u32 {
    let word = V[byte_offset / 4u];
    let shift = (byte_offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_v_u32_at(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = V[word_idx];
    if (shift == 0u) {
        return lo;
    }
    let hi = V[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
}

fn f16_from_u16(bits: u32) -> f16 {
    let packed = unpack2x16float(bits);
    return f16(packed[0]);
}
#endif

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // shapes of Q/K/V
    n_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,

    // strides (in elements)
    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_v1: u32,
    stride_v2: u32,
    stride_v3: u32,
    stride_mask3: u32,

    // repeat factors for K/V, e.g., MHA vs. MQA vs. GQA
    q_per_kv: u32,

    // softmax params
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
@group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;

#if defined(MASK) && defined(SINKS)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 5
#define PARAMS_BINDING 6
#elif defined(MASK)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#elif defined(SINKS)
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#else
#define DST_BINDING 3
#define PARAMS_BINDING 4
#endif

@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

// The number of Q rows processed per workgroup
var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;

#ifndef KV_DIRECT
const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
// we can reuse the same shmem for K and V since we only need one at a time
var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
#endif

var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>; // output shmem

#ifdef MASK
// storage for mask values
var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
#endif

// storage for output of Q*K^T scores for online softmax (S matrix from paper)
// also storage for diagonal matrix during online softmax (P matrix from paper)
// note that we reuse the same storage for both since we only need one at a time
var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;

// Storage for row max and exp sum during online softmax
var<workgroup> row_max_shmem: array<f32, Q_TILE>;
var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;

fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32) -> f32 {
    var v = select(FLOAT_MIN,
                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
                   kv_idx < KV_TILE);
#ifdef LOGIT_SOFTCAP
    v = params.logit_softcap * tanh(v);
#endif
#ifdef MASK
    let mask_val = select(0.0, f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
    let mask_term = slope * mask_val;
    v += mask_term;
#endif
    return v;
}

fn load_f32x4(buf: ptr<storage, array<vec4<f32>>, read_write>, scalar_index: u32) -> vec4<f32> {
    return (*buf)[scalar_index >> 2u];
}

fn load_kvx4(buf: ptr<storage, array<vec4<KV_TYPE>>, read_write>, scalar_index: u32) -> vec4<KV_TYPE> {
    return (*buf)[scalar_index >> 2u];
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32) {

    // initialize row max for online softmax
    for (var i = local_id.x; i < Q_TILE; i += WG_SIZE) {
        row_max_shmem[i] = FLOAT_MIN;
        exp_sum_shmem[i] = 0.0;
    }

    for (var i = local_id.x; i < Q_TILE * HEAD_DIM_V; i += WG_SIZE) {
        o_shmem[i] = 0.0;
    }

    // workgroups per head/batch
    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;

    // batch index
    let batch_idx = wg_id.x / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let dst_batch_offset = params.offset_dst + batch_idx * dst3_stride;
    let wg_in_batch = wg_id.x % wg_per_batch;

    // head index
    let head_idx = wg_in_batch / wg_per_head;
    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
    let k_head_idx = head_idx / params.q_per_kv;
    let v_head_idx = k_head_idx;
    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
    let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;

    // starting Q row for this workgroup
    let wg_in_head = wg_in_batch % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;

#ifdef MASK
    // mask offset
    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
#endif

    // note that the output is permuted, the layout is [head_dim_v, n_heads, seq_len_q, batch_size]
    let dst_global_offset = dst_batch_offset + q_row_start * dst2_stride + head_idx * HEAD_DIM_V;

    let head = f32(head_idx);
    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), params.max_bias > 0);

    // load q tile into shared memory
    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_row;
        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
        q_shmem[elem_idx] = f16(select(
            0.0,
            Q[global_q_row_offset + q_col],
            head_q_row < params.seq_len_q && q_col < HEAD_DIM_QK));
    }

    for (var kv_tile = 0u; kv_tile < params.seq_len_kv; kv_tile += KV_TILE) {
      // clear inter_shmem to ensure zero-initialized accumulators
        for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
            inter_shmem[elem_idx] = 0.0;
        }

      // load k tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let block_byte_base = global_block_idx * BLOCK_SIZE_BYTES;
              let d = f16_from_u16(load_k_u16_at(block_byte_base));
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                  let q_packed = load_k_u32_at(q_byte_offset);
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let block_byte_base = global_block_idx * BLOCK_SIZE_BYTES;
              let d = f16_from_u16(load_k_u16_at(block_byte_base));
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                  let q_packed = load_k_u32_at(q_byte_offset);
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
          let k_row = elem_idx / HEAD_DIM_QK;
          let k_col = elem_idx % HEAD_DIM_QK;
          let global_k_row = kv_tile + k_row;
          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
          kv_shmem[elem_idx] = f16(select(
              0.0,
              K[global_k_row_offset + k_col],
              global_k_row < params.seq_len_kv && k_col < HEAD_DIM_QK));
      }
#endif

      workgroupBarrier();

      // accumulate q block * k block into registers across the entire KV tile
      // TODO: this loop seems to be the current largest bottleneck
      // this bracket exists to scope the lifetime of variables, reducing register pressure
      {
#ifdef KV_DIRECT
          let k_block_row = kv_tile + subgroup_id * SG_MAT_N;
          var k_global_offset = k_head_offset + k_block_row * params.stride_k1;
#else
          var k_block_offset = subgroup_id * SG_MAT_N * HEAD_DIM_QK;
#endif
          for (var kv_block = subgroup_id; kv_block < KV_BLOCKS; kv_block += num_subgroups) {
              let inter_offset = kv_block * SG_MAT_N;
              var acc: subgroup_matrix_result<f16, SG_MAT_N, SG_MAT_M> = subgroupMatrixLoad<subgroup_matrix_result<f16, SG_MAT_N, SG_MAT_M>>(&inter_shmem, inter_offset, false, KV_TILE);

              var q_cur = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M>>(&q_shmem, 0u, false, HEAD_DIM_QK);

#ifdef KV_DIRECT
              var k_cur = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&K, k_global_offset + 0u, true, params.stride_k1);
#else
              var k_cur = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&kv_shmem, k_block_offset + 0u, true, HEAD_DIM_QK);
#endif

              var t: u32 = 1u;
              for (; t + 1u < HEAD_DIM_QK / SG_MAT_K; t += 2u) {
                  let h0 = t * SG_MAT_K;
                  var q0 = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M>>(&q_shmem, h0, false, HEAD_DIM_QK);
#ifdef KV_DIRECT
                  var k0 = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&K, k_global_offset + h0, true, params.stride_k1);
#else
                  var k0 = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&kv_shmem, k_block_offset + h0, true, HEAD_DIM_QK);
#endif
                  acc = subgroupMatrixMultiplyAccumulate(q_cur, k_cur, acc);
                  q_cur = q0;
                  k_cur = k0;

                  let h1 = (t + 1u) * SG_MAT_K;
                  var q1g = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M>>(&q_shmem, h1, false, HEAD_DIM_QK);
#ifdef KV_DIRECT
                  var k1g = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&K, k_global_offset + h1, true, params.stride_k1);
#else
                  var k1g = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&kv_shmem, k_block_offset + h1, true, HEAD_DIM_QK);
#endif
                  acc = subgroupMatrixMultiplyAccumulate(q_cur, k_cur, acc);
                  q_cur = q1g;
                  k_cur = k1g;
              }

              // handle odd tail
              if (t < HEAD_DIM_QK / SG_MAT_K) {
                  let h = t * SG_MAT_K;
                  var qn = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M>>(&q_shmem, h, false, HEAD_DIM_QK);
#ifdef KV_DIRECT
                  var kn = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&K, k_global_offset + h, true, params.stride_k1);
#else
                  var kn = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(&kv_shmem, k_block_offset + h, true, HEAD_DIM_QK);
#endif
                  acc = subgroupMatrixMultiplyAccumulate(q_cur, k_cur, acc);
                  q_cur = qn;
                  k_cur = kn;
              }

              acc = subgroupMatrixMultiplyAccumulate(q_cur, k_cur, acc);

#ifdef KV_DIRECT
              k_global_offset += num_subgroups * SG_MAT_N * params.stride_k1;
#else
              k_block_offset += num_subgroups * SG_MAT_N * HEAD_DIM_QK;
#endif
              subgroupMatrixStore(&inter_shmem, inter_offset, acc, false, KV_TILE);
          }
      }


#ifdef MASK
      // load mask tile into shared memory for this KV block
      // TODO: optimize and skip if mask is -INF for the entire tile
      for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
          let mask_row = elem_idx / KV_TILE;
          let mask_col = elem_idx % KV_TILE;
          let global_q_row = q_row_start + mask_row;
          let global_k_col = kv_tile + mask_col;
          let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
          let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
          mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
      }
#endif

      workgroupBarrier();

      // online softmax
      for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
          let global_q_row = q_row_start + q_tile_row;
          if (global_q_row >= params.seq_len_q) {
              break;
          }

          // initialize running max for this row
          var prev_max = row_max_shmem[q_tile_row];
          var final_max = prev_max;
          // pass 1: compute final max across the full KV tile in chunks
          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
              let kv_idx = kv_offset + sg_inv_id;
              let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope);
              final_max = subgroupMax(max(final_max, softmax_term));
          }

          var total_exp_term: f32 = 0.0;
          // pass 2: compute exp sum and write P using final_max
          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
              let kv_idx = kv_offset + sg_inv_id;
              let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope);
              let cur_p = select(0.0,
                                 exp(softmax_term - final_max),
                                 kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
              total_exp_term += subgroupAdd(cur_p);
              if (kv_idx < KV_TILE) {
                  inter_shmem[kv_idx + q_tile_row * KV_TILE] = f16(cur_p);
              }
          }

          let cur_exp = exp(prev_max - final_max);

          if (sg_inv_id == 0) {
              row_max_shmem[q_tile_row] = final_max;
              exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * cur_exp + total_exp_term;
          }

          for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
              let idx = q_tile_row * HEAD_DIM_V + elem_idx;
              o_shmem[idx] = f16(f32(o_shmem[idx]) * cur_exp);
          }
      }

      // load v tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let block_byte_base = global_block_idx * BLOCK_SIZE_BYTES;
              let d = f16_from_u16(load_v_u16_at(block_byte_base));
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                  let q_packed = load_v_u32_at(q_byte_offset);
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let block_byte_base = global_block_idx * BLOCK_SIZE_BYTES;
              let d = f16_from_u16(load_v_u16_at(block_byte_base));
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                  let q_packed = load_v_u32_at(q_byte_offset);
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE) {
          let v_row = elem_idx / HEAD_DIM_V;
          let v_col = elem_idx % HEAD_DIM_V;
          let global_v_row = kv_tile + v_row;
          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
          kv_shmem[elem_idx] = f16(select(
              0.0,
              V[global_v_row_offset + v_col],
              global_v_row < params.seq_len_kv && v_col < HEAD_DIM_V));
      }
#endif

      workgroupBarrier();

      // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
      // we want to compute O += P * V across the full KV tile
      for (var head_dim_block = subgroup_id * SG_MAT_N;
           head_dim_block < HEAD_DIM_V;
           head_dim_block += num_subgroups * SG_MAT_N) {
              // load O submatrix from shared memory
              var o_sg_mat: subgroup_matrix_result<f16, SG_MAT_N, SG_MAT_M> = subgroupMatrixLoad<subgroup_matrix_result<f16, SG_MAT_N, SG_MAT_M>>(
                  &o_shmem,
                  head_dim_block,
                  false,
                  HEAD_DIM_V
              );
              for (var kv_block = 0u; kv_block < KV_BLOCKS; kv_block++) {
                  let p_offset = kv_block * SG_MAT_N;
                  var p_sg_mat: subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M> = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_K, SG_MAT_M>>(
                      &inter_shmem,
                      p_offset,
                      false,
                      KV_TILE
                  );

                  // load V submatrix from global or shared memory
#ifdef KV_DIRECT
                  let v_block_row = kv_tile + kv_block * SG_MAT_N;
                  let v_global_offset = v_head_offset + v_block_row * params.stride_v1 + head_dim_block;
                  var v_sg_mat: subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(
                      &V,
                      v_global_offset,
                      false,
                      params.stride_v1
                  );
#else
                  let v_block_offset = kv_block * SG_MAT_N * HEAD_DIM_V;
                  var v_sg_mat: subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_N, SG_MAT_K>>(
                      &kv_shmem,
                      v_block_offset + head_dim_block,
                      false,
                      HEAD_DIM_V
                  );
#endif
                  // O += P * V
                  o_sg_mat = subgroupMatrixMultiplyAccumulate(p_sg_mat, v_sg_mat, o_sg_mat);
              }
              // store O back to shared memory
              subgroupMatrixStore(&o_shmem, head_dim_block, o_sg_mat, false, HEAD_DIM_V);
      }
      workgroupBarrier();
    }

#ifdef SINKS
    // add sinks (applied once after processing all KV tiles)
    for (var q_tile_row = subgroup_id;
         q_tile_row < Q_TILE;
         q_tile_row += num_subgroups) {
            // no need to process rows beyond seq_len_q
            let global_q_row = q_row_start + q_tile_row;
            if (global_q_row >= params.seq_len_q) {
                break;
            }

            var prev_max = row_max_shmem[q_tile_row];

            // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
            let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
            let new_max = subgroupMax(max(prev_max, sink_val));
            let max_exp = exp(prev_max - new_max);
            let sink_exp = exp(sink_val - new_max);

            let sink_exp_sum = subgroupAdd(sink_exp);

            if (sg_inv_id == 0) {
                exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
            }

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
                let val = f32(o_shmem[idx]) * max_exp;
                o_shmem[idx] = f16(val);
            }
    }
    workgroupBarrier();
#endif
    for (var q_tile_row = subgroup_id;
        q_tile_row < Q_TILE;
        q_tile_row += num_subgroups) {

        let global_q_row = q_row_start + q_tile_row;
        if (global_q_row >= params.seq_len_q) { break; }

        let exp_sum = exp_sum_shmem[q_tile_row];
        let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);

        let row_base: u32 = dst_global_offset + q_tile_row * dst2_stride;

        for (var elem_base = sg_inv_id * 4u;
            elem_base < HEAD_DIM_V;
            elem_base += subgroup_size * 4u) {

            let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
            let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
            let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
            let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);

            let v = vec4<f32>(
                f32(o_shmem[i0]) * scale,
                f32(o_shmem[i1]) * scale,
                f32(o_shmem[i2]) * scale,
                f32(o_shmem[i3]) * scale
            );

            let dst_vec_index: u32 = (row_base + elem_base) >> 2u;
            dst[dst_vec_index] = v;
        }
    }
}
)wgsl";

const char* wgsl_flash_attn_vec_blk = R"wgsl(diagnostic(off, subgroup_uniformity);
enable f16;

#define KV_TILE 32
#define WG_SIZE 32

struct Params {
    offset_mask: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    stride_mask3: u32,
    // Number of KV blocks and Q blocks per batch.
    // nblk0 = ceil(seq_len_kv / KV_TILE), nblk1 = seq_len_q.
    nblk0: u32,
    nblk1: u32,
};

@group(0) @binding(0) var<storage, read> mask: array<f16>;
@group(0) @binding(1) var<storage, read_write> blk: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const MASK_MIN: f32 = -65504.0;
const MASK_MAX: f32 = 65504.0;
var<workgroup> wg_min: array<f32, WG_SIZE>;
var<workgroup> wg_max: array<f32, WG_SIZE>;
var<workgroup> wg_any: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Dispatch mapping:
    //  - x indexes KV blocks
    //  - y flattens (batch_idx, q_blk) as y = batch_idx * nblk1 + q_blk
    let kv_blk = wg_id.x;
    let y = wg_id.y;
    let q_blk = y % params.nblk1;
    let batch_idx = y / params.nblk1;
    if (kv_blk >= params.nblk0) {
        return;
    }

    let q_start = q_blk;
    let k_start = kv_blk * KV_TILE;

    let mask_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
    let mask_batch_base = params.offset_mask + mask_batch * params.stride_mask3;

    // We keep min/max to classify:
    //  - fully masked (max <= MASK_MIN)
    //  - all-zero mask (min == 0 && max == 0)
    //  - mixed/general mask
    var local_min = MASK_MAX;
    var local_max = -MASK_MAX;
    var local_any = 0u;

    let q_row = q_start;
    if (q_row < params.seq_len_q) {
        let row_base = mask_batch_base + q_row * params.seq_len_kv;
        for (var k_rel = local_id.x; k_rel < KV_TILE; k_rel += WG_SIZE) {
            let k_col = k_start + k_rel;
            if (k_col >= params.seq_len_kv) {
                continue;
            }
            let mv = f32(mask[row_base + k_col]);
            local_min = min(local_min, mv);
            local_max = max(local_max, mv);
            local_any = 1u;
        }
    }

    wg_min[local_id.x] = local_min;
    wg_max[local_id.x] = local_max;
    wg_any[local_id.x] = local_any;
    workgroupBarrier();

    // Thread 0 writes one state per block.
    if (local_id.x == 0u) {
        var mmin = wg_min[0];
        var mmax = wg_max[0];
        var many = wg_any[0];
        for (var i = 1u; i < WG_SIZE; i += 1u) {
            mmin = min(mmin, wg_min[i]);
            mmax = max(mmax, wg_max[i]);
            many = max(many, wg_any[i]);
        }

        var state = 0u;
        if (many != 0u) {
            if (mmax <= MASK_MIN) {
                state = 0u;
            } else if (mmin == 0.0 && mmax == 0.0) {
                state = 2u;
            } else {
                state = 1u;
            }
        }

        let blk_idx = (batch_idx * params.nblk1 + q_blk) * params.nblk0 + kv_blk;
        blk[blk_idx] = state;
    }
}
)wgsl";

const char* wgsl_flash_attn_vec_reduce = R"wgsl(diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;

// Default values
#define HEAD_DIM_V 64
#define WG_SIZE 128

struct Params {
    nrows: u32,
    seq_len_q: u32,
    n_heads: u32,
    offset_dst: u32,
    nwg: u32,
    tmp_data_base: u32,
    tmp_stats_base: u32,
};

@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    let rid = wg_id.x;
    if (rid >= params.nrows) {
        return;
    }

    let rows_per_batch = params.n_heads * params.seq_len_q;
    let batch_idx = rid / rows_per_batch;
    let rem = rid % rows_per_batch;
    let head_idx = rem / params.seq_len_q;
    let q_row = rem % params.seq_len_q;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;
    let row_base = params.offset_dst + batch_idx * dst3_stride + q_row * dst2_stride + head_idx * HEAD_DIM_V;

    let thread = sg_inv_id;
    if (params.nwg > subgroup_size) {
        return;
    }

    let stats_base = params.tmp_stats_base + rid * (2u * params.nwg);
    let active_thread = thread < params.nwg;
    let si = select(0.0, tmp[stats_base + 2u * thread + 0u], active_thread);
    let mi = select(FLOAT_MIN, tmp[stats_base + 2u * thread + 1u], active_thread);
    let m = subgroupMax(mi);
    let ms = select(0.0, exp(mi - m), active_thread);
    let s = subgroupAdd(si * ms);
    let inv_s = select(0.0, 1.0 / s, s != 0.0);

    let row_tmp_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg);
    for (var elem_base = subgroup_id * 4u; elem_base < HEAD_DIM_V; elem_base += num_subgroups * 4u) {
        var weighted = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (active_thread) {
            let src = row_tmp_base + thread * HEAD_DIM_V + elem_base;
            weighted = vec4<f32>(tmp[src + 0u], tmp[src + 1u], tmp[src + 2u], tmp[src + 3u]) * ms;
        }

        let sum_x = subgroupAdd(weighted.x);
        let sum_y = subgroupAdd(weighted.y);
        let sum_z = subgroupAdd(weighted.z);
        let sum_w = subgroupAdd(weighted.w);

        if (thread == 0u) {
            let dst_vec_index = (row_base + elem_base) >> 2u;
            dst[dst_vec_index] = vec4<f32>(sum_x, sum_y, sum_z, sum_w) * inv_s;
        }
    }
}
)wgsl";

const char* wgsl_flash_attn_vec_split = R"wgsl(diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

#ifdef KV_F32
#define KV_TYPE f32
#else
#define KV_TYPE f16
#endif

#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64


#define SG_MAT_M 8
#define SG_MAT_N 8
#define SG_MAT_K 8

#define Q_TILE SG_MAT_M
#define KV_TILE 16
#define WG_SIZE 64
#ifndef VEC_NE
#define VEC_NE 4u
#endif

#define KV_BLOCKS (KV_TILE / SG_MAT_N)

#define BLOCK_SIZE 32
#define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_V ((HEAD_DIM_V + BLOCK_SIZE - 1) / BLOCK_SIZE)
#if defined(KV_Q4_0)
#define NQ 16
#define F16_PER_BLOCK 9
#define WEIGHTS_PER_F16 4
#elif defined(KV_Q8_0)
#define NQ 8
#define F16_PER_BLOCK 17
#define WEIGHTS_PER_F16 2
#endif
#define F16_PER_THREAD (NQ / WEIGHTS_PER_F16)

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // shapes of Q/K/V
    n_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,

    // strides (in elements)
    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_v1: u32,
    stride_v2: u32,
    stride_v3: u32,
    stride_mask3: u32,

    // repeat factors for K/V, e.g., MHA vs. MQA vs. GQA
    q_per_kv: u32,

    // softmax params
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,

#ifdef BLK
    blk_base: u32,
    blk_nblk0: u32,
    blk_nblk1: u32,
#endif

    tmp_data_base: u32,
    tmp_stats_base: u32,
    nwg: u32,
};

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
#if defined(KV_Q4_0) || defined(KV_Q8_0)
@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
#else
@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
#endif
#if defined(KV_Q4_0) || defined(KV_Q8_0)
@group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;
#else
@group(0) @binding(2) var<storage, read_write> V: array<vec4<KV_TYPE>>;
#endif
#if defined(MASK) && defined(SINKS)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
#ifdef BLK
#define BLK_BINDING 5
#define TMP_BINDING 6
#define DST_BINDING 7
#define PARAMS_BINDING 8
#else
#define TMP_BINDING 5
#define DST_BINDING 6
#define PARAMS_BINDING 7
#endif
#elif defined(MASK)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#ifdef BLK
#define BLK_BINDING 4
#define TMP_BINDING 5
#define DST_BINDING 6
#define PARAMS_BINDING 7
#else
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#endif
#elif defined(SINKS)
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#else
#define TMP_BINDING 3
#define DST_BINDING 4
#define PARAMS_BINDING 5
#endif

#ifdef BLK
@group(0) @binding(BLK_BINDING) var<storage, read_write> blk: array<u32>;
#endif
@group(0) @binding(TMP_BINDING) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;

#ifndef KV_DIRECT
const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
// we can reuse the same shmem for K and V since we only need one at a time
var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
#endif

var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>;

#ifdef MASK
// storage for mask values
var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
#endif

// note that we reuse the same storage for both since we only need one at a time
var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;

// Storage for row max and exp sum during online softmax
var<workgroup> row_max_shmem: array<f32, Q_TILE>;
var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;
var<workgroup> blk_state_wg: u32;

fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32, has_bias: bool, apply_mask: bool) -> f32 {
    var v = select(FLOAT_MIN,
                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
                   kv_idx < KV_TILE);
#ifdef LOGIT_SOFTCAP
    v = params.logit_softcap * tanh(v);
#endif
#ifdef MASK
    if (apply_mask) {
        var mask_val = select(0.0,f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
        v += select(mask_val, slope * mask_val, has_bias);
    }
#endif
    return v;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_invocation_id) sg_inv_id: u32) {

    // initialize row max for online softmax
    for (var i = local_id.x; i < Q_TILE; i += WG_SIZE) {
        row_max_shmem[i] = FLOAT_MIN;
        exp_sum_shmem[i] = 0.0;
    }

    for (var i = local_id.x; i < Q_TILE * HEAD_DIM_V; i += WG_SIZE) {
        o_shmem[i] = 0.0;
    }

    // workgroups per head/batch
    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;

    let iwg = wg_id.x % params.nwg;
    let base_wg_id = wg_id.x / params.nwg;

    // batch index
    let batch_idx = base_wg_id / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let wg_in_batch = base_wg_id % wg_per_batch;

    // head index
    let head_idx = wg_in_batch / wg_per_head;
    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
    let k_head_idx = head_idx / params.q_per_kv;
    let v_head_idx = k_head_idx;
    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
    let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;

    // starting Q row for this workgroup
    let wg_in_head = wg_in_batch % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;

#ifdef MASK
    // mask offset
    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
#endif

    let head = f32(head_idx);
    let has_bias = params.max_bias > 0.0;
    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), has_bias);

    // load q tile into shared memory
    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_row;
        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
        q_shmem[elem_idx] = f16(select(
            0.0,
            Q[global_q_row_offset + q_col],
            head_q_row < params.seq_len_q && q_col < HEAD_DIM_QK));
    }

    for (var kv_tile = iwg * KV_TILE; kv_tile < params.seq_len_kv; kv_tile += KV_TILE * params.nwg) {
#ifdef BLK
        let q_blk = q_row_start / Q_TILE;
        let kv_blk = kv_tile / KV_TILE;
        let blk_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
        let blk_idx = params.blk_base + (blk_batch * params.blk_nblk1 + q_blk) * params.blk_nblk0 + kv_blk;
        let blk_state_local = blk[blk_idx];
#else
        let blk_state_local = 1u;
#endif
        if (local_id.x == 0u) {
            blk_state_wg = blk_state_local;
        }
        workgroupBarrier();
        let blk_state = blk_state_wg;
        let skip_tile = blk_state == 0u;
        for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
            inter_shmem[elem_idx] = f16(0.0);
        }

      // load k tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = K[base_idx];
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = K[base_idx + 1u + block_offset + j];
                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = K[base_idx];
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = K[base_idx + 1u + block_offset + j];
                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * 4u) {
          let k_row = elem_idx / HEAD_DIM_QK;
          let k_col = elem_idx % HEAD_DIM_QK;
          let global_k_row = kv_tile + k_row;
          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
          let in_bounds = global_k_row < params.seq_len_kv && (k_col + 3u) < HEAD_DIM_QK;
          let vec_idx = (global_k_row_offset + k_col) >> 2u;
          let k4 = select(vec4<KV_TYPE>(0.0), K[vec_idx], in_bounds);
          kv_shmem[elem_idx + 0u] = f16(k4.x);
          kv_shmem[elem_idx + 1u] = f16(k4.y);
          kv_shmem[elem_idx + 2u] = f16(k4.z);
          kv_shmem[elem_idx + 3u] = f16(k4.w);
      }
#endif

      workgroupBarrier();

      // accumulate q block * k block into registers across the entire KV tile
      if (!skip_tile) {
        let num_of_threads = subgroup_size / VEC_NE;
        let tx = sg_inv_id % num_of_threads;
        let ty = sg_inv_id / num_of_threads;
          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
              let global_q_row = q_row_start + q_tile_row;
              if (global_q_row >= params.seq_len_q) {
                  continue;
              }
              let local_q_row_offset = q_tile_row * HEAD_DIM_QK;

              for (var kv_base : u32 = 0u; kv_base < KV_TILE; kv_base += VEC_NE) {
                  let kv_idx = kv_base + ty;
                  var partial_sum: f32 = 0.0;
                  let kv_valid = kv_idx < KV_TILE && (kv_tile + kv_idx) < params.seq_len_kv;
                  if (kv_valid) {
                    for (var i = tx; i < (HEAD_DIM_QK / 4u); i += num_of_threads) {
                        let q_off = local_q_row_offset + i * 4u;

                        let qv = vec4<f32>(
                            f32(q_shmem[q_off + 0u]),
                            f32(q_shmem[q_off + 1u]),
                            f32(q_shmem[q_off + 2u]),
                            f32(q_shmem[q_off + 3u]));
#ifdef KV_DIRECT
                        let idx = k_head_offset + (kv_tile + kv_idx) * params.stride_k1 + (i * 4u);
                        let kv = vec4<f32>(K[idx >> 2u]);
#else
                        let idx = kv_idx * HEAD_DIM_QK + (i * 4u);
                        let kv = vec4<f32>(
                            f32(kv_shmem[idx + 0u]),
                            f32(kv_shmem[idx + 1u]),
                            f32(kv_shmem[idx + 2u]),
                            f32(kv_shmem[idx + 3u]));
#endif
                        partial_sum += dot(qv, kv);
                    }
                  }
                  var sum = partial_sum;
                  // Reduce over tx threads (NL) for this ty stripe.
                  var tx_delta = num_of_threads >> 1u;
                  loop {
                      if (tx_delta == 0u) {
                          break;
                      }
                      let sh = subgroupShuffleDown(sum, tx_delta);
                      if (tx < tx_delta) {
                          sum += sh;
                      }
                      tx_delta >>= 1u;
                  }

                  let sum_bcast = subgroupShuffle(sum, num_of_threads * ty);
                  if (tx == 0u && kv_valid) {
                      let dst_idx = q_tile_row * KV_TILE + kv_idx;
                      inter_shmem[dst_idx] = f16(sum_bcast);
                  }
              }
          }
      }


#ifdef MASK
      let apply_mask = !skip_tile && (blk_state != 2u);
      if (apply_mask) {
          // load mask tile into shared memory for this KV block
          for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
              let mask_row = elem_idx / KV_TILE;
              let mask_col = elem_idx % KV_TILE;
              let global_q_row = q_row_start + mask_row;
              let global_k_col = kv_tile + mask_col;
              let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
              let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
              mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
          }
      }
#else
      let apply_mask = false;
#endif

      workgroupBarrier();

      // online softmax
      if (!skip_tile) {
          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
              let global_q_row = q_row_start + q_tile_row;
              if (global_q_row >= params.seq_len_q) {
                  break;
              }

              var prev_max = row_max_shmem[q_tile_row];
              var final_max = prev_max;
              // pass 1: compute final max across the full KV tile in chunks
              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                  let kv_idx = kv_offset + sg_inv_id;
                  let kv_valid = kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE;
                  let softmax_term = select(FLOAT_MIN,
                                            calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask),
                                            kv_valid);
                  final_max = subgroupMax(max(final_max, softmax_term));
              }

              var total_exp_term: f32 = 0.0;
              // pass 2: compute exp sum and write P using final_max
              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                  let kv_idx = kv_offset + sg_inv_id;
                  let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask);
                  let cur_p = select(0.0,
                                     exp(softmax_term - final_max),
                                     kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
                  total_exp_term += subgroupAdd(cur_p);
                  if (kv_idx < KV_TILE) {
                      inter_shmem[kv_idx + q_tile_row * KV_TILE] = f16(cur_p);
                  }
              }

              let cur_exp = exp(prev_max - final_max);

              if (sg_inv_id == 0) {
                  row_max_shmem[q_tile_row] = final_max;
                  exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * cur_exp + total_exp_term;
              }

              for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                  let idx = q_tile_row * HEAD_DIM_V + elem_idx;
                  o_shmem[idx] = f16(f32(o_shmem[idx]) * cur_exp);
              }
          }
      }

      // load v tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = V[base_idx];
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = V[base_idx + 1u + block_offset + j];
                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = V[base_idx];
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = V[base_idx + 1u + block_offset + j];
                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * 4u) {
          let v_row = elem_idx / HEAD_DIM_V;
          let v_col = elem_idx % HEAD_DIM_V;
          let global_v_row = kv_tile + v_row;
          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
          let in_bounds = global_v_row < params.seq_len_kv && (v_col + 3u) < HEAD_DIM_V;
          let vec_idx = (global_v_row_offset + v_col) >> 2u;
          let v4 = select(vec4<KV_TYPE>(0.0), V[vec_idx], in_bounds);
          kv_shmem[elem_idx + 0u] = f16(v4.x);
          kv_shmem[elem_idx + 1u] = f16(v4.y);
          kv_shmem[elem_idx + 2u] = f16(v4.z);
          kv_shmem[elem_idx + 3u] = f16(v4.w);
      }
#endif

      workgroupBarrier();

      if (!skip_tile) {
          // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
          // we want to compute O += P * V across the full KV tile
          let ne_threads : u32 = VEC_NE;
          let nl_threads = max(1u, subgroup_size / ne_threads);
          let tx_pv = sg_inv_id % nl_threads;
          let ty_pv = sg_inv_id / nl_threads;
          for (var q_tile_row = subgroup_id;
               q_tile_row < Q_TILE;
               q_tile_row += num_subgroups) {
              for (var vec_col = tx_pv; vec_col < (HEAD_DIM_V / 4u); vec_col += nl_threads) {
                  var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                  for (var cc = 0u; cc < KV_TILE / ne_threads; cc += 1u) {
                      let kv_idx = cc * ne_threads + ty_pv;
                      let v_row = kv_tile + kv_idx;
                      if (v_row >= params.seq_len_kv) {
                          continue;
                      }

                      let p = f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]);
#ifdef KV_DIRECT
                      let v_idx = v_head_offset + v_row * params.stride_v1 + vec_col * 4u;
                      let v4 = vec4<f32>(V[v_idx >> 2u]);
#else
                      let v_idx = kv_idx * HEAD_DIM_V + vec_col * 4u;
                      let v4 = vec4<f32>(
                          f32(kv_shmem[v_idx + 0u]),
                          f32(kv_shmem[v_idx + 1u]),
                          f32(kv_shmem[v_idx + 2u]),
                          f32(kv_shmem[v_idx + 3u]));
#endif
                      lo += p * v4;
                  }

                  var lo_x = lo.x;
                  var lo_y = lo.y;
                  var lo_z = lo.z;
                  var lo_w = lo.w;
                  // Reduce over ty threads (NE) for this tx thread.
                  var ty_delta = ne_threads >> 1u;
                  loop {
                      if (ty_delta == 0u) {
                          break;
                      }
                      let thread_delta = ty_delta * nl_threads;
                      let shx = subgroupShuffleDown(lo_x, thread_delta);
                      let shy = subgroupShuffleDown(lo_y, thread_delta);
                      let shz = subgroupShuffleDown(lo_z, thread_delta);
                      let shw = subgroupShuffleDown(lo_w, thread_delta);
                      if (ty_pv < ty_delta) {
                          lo_x += shx;
                          lo_y += shy;
                          lo_z += shz;
                          lo_w += shw;
                      }
                      ty_delta >>= 1u;
                  }

                  if (ty_pv == 0u) {
                      let elem_base = vec_col * 4u;
                      let o_base_idx = q_tile_row * HEAD_DIM_V + elem_base;
                      o_shmem[o_base_idx + 0u] = f16(f32(o_shmem[o_base_idx + 0u]) + lo_x);
                      o_shmem[o_base_idx + 1u] = f16(f32(o_shmem[o_base_idx + 1u]) + lo_y);
                      o_shmem[o_base_idx + 2u] = f16(f32(o_shmem[o_base_idx + 2u]) + lo_z);
                      o_shmem[o_base_idx + 3u] = f16(f32(o_shmem[o_base_idx + 3u]) + lo_w);
                  }
              }
          }
      }

        workgroupBarrier();
    }


#ifdef SINKS
    // Sinks are global terms and must be applied exactly once across split workgroups.
    if (iwg == 0u) {
        for (var q_tile_row = subgroup_id;
             q_tile_row < Q_TILE;
             q_tile_row += num_subgroups) {
                let global_q_row = q_row_start + q_tile_row;
                if (global_q_row >= params.seq_len_q) {
                    break;
                }

                var prev_max = row_max_shmem[q_tile_row];

                // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
                let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
                let new_max = subgroupMax(max(prev_max, sink_val));
                let max_exp = exp(prev_max - new_max);
                let sink_exp = exp(sink_val - new_max);

                let sink_exp_sum = subgroupAdd(sink_exp);

                if (sg_inv_id == 0) {
                    row_max_shmem[q_tile_row] = new_max;
                    exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
                }

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
                o_shmem[idx] = f16(f32(o_shmem[idx]) * max_exp);
            }
        }
        workgroupBarrier();
    }
#endif
    let rows_per_batch = params.n_heads * params.seq_len_q;
    for (var q_tile_row = subgroup_id;
         q_tile_row < Q_TILE;
         q_tile_row += num_subgroups) {

        let global_q_row = q_row_start + q_tile_row;
        if (global_q_row >= params.seq_len_q) { break; }

        if (params.nwg == 1u) {
            let exp_sum = exp_sum_shmem[q_tile_row];
            let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
            let row_base: u32 =
                params.offset_dst + batch_idx * dst3_stride + global_q_row * dst2_stride + head_idx * HEAD_DIM_V;

            for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);

                let v = vec4<f32>(
                    f32(o_shmem[i0]) * scale,
                    f32(o_shmem[i1]) * scale,
                    f32(o_shmem[i2]) * scale,
                    f32(o_shmem[i3]) * scale
                );

                let dst_vec_index: u32 = (row_base + elem_base) >> 2u;
                dst[dst_vec_index] = v;
            }
        } else {
            let rid = batch_idx * rows_per_batch + head_idx * params.seq_len_q + global_q_row;
            let tmp_row_data_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
            let tmp_row_stats_base = params.tmp_stats_base + rid * (2u * params.nwg) + 2u * iwg;

            for (var elem_base = sg_inv_id * 4u;
                elem_base < HEAD_DIM_V;
                elem_base += subgroup_size * 4u) {

                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);

                let tbase = tmp_row_data_base + elem_base;
                tmp[tbase + 0u] = f32(o_shmem[i0]);
                tmp[tbase + 1u] = f32(o_shmem[i1]);
                tmp[tbase + 2u] = f32(o_shmem[i2]);
                tmp[tbase + 3u] = f32(o_shmem[i3]);
            }

            if (sg_inv_id == 0u) {
                tmp[tmp_row_stats_base + 0u] = exp_sum_shmem[q_tile_row];
                tmp[tmp_row_stats_base + 1u] = row_max_shmem[q_tile_row];
            }
        }
    }
}
)wgsl";

const char* wgsl_gated_delta_net = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src_q: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src_k: array<f32>;

@group(0) @binding(2)
var<storage, read_write> src_v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> src_g: array<f32>;

@group(0) @binding(4)
var<storage, read_write> src_beta: array<f32>;

@group(0) @binding(5)
var<storage, read_write> src_state: array<f32>;

@group(0) @binding(6)
var<storage, read_write> dst: array<f32>;

struct Params {
    h: u32,
    n_tokens: u32,
    n_seqs: u32,
    s_off: u32,

    sq1: u32,
    sq2: u32,
    sq3: u32,

    sv1: u32,
    sv2: u32,
    sv3: u32,

    sb1: u32,
    sb2: u32,
    sb3: u32,

    neq1: u32,
    rq3: u32,
    scale: f32,
};

@group(0) @binding(7)
var<uniform> params: Params;

var<workgroup> sh_k: array<f32, S_V>;
var<workgroup> sh_q: array<f32, S_V>;
var<workgroup> sh_g: array<f32, S_V>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let head_id = workgroup_id.x;
    let seq_id = workgroup_id.y;
    let col = local_id.x;

    let iq1 = head_id % params.neq1;
    let iq3 = seq_id / params.rq3;

    let state_size = S_V * S_V;
    let state_base = (seq_id * params.h + head_id) * state_size;

    var state: array<f32, S_V>;
    for (var i = 0u; i < S_V; i++) {
        state[i] = src_state[state_base + col * S_V + i];
    }

    var attn_off = (seq_id * params.n_tokens * params.h + head_id) * S_V;

    for (var t = 0u; t < params.n_tokens; t++) {
        let q_off = iq3 * params.sq3 + t * params.sq2 + iq1 * params.sq1;
        let k_off = q_off;
        let v_off = seq_id * params.sv3 + t * params.sv2 + head_id * params.sv1;
        let gb_off = seq_id * params.sb3 + t * params.sb2 + head_id * params.sb1;

        sh_q[col] = src_q[q_off + col];
        sh_k[col] = src_k[k_off + col];

#ifdef KDA
        let g_base = gb_off * S_V;
        sh_g[col] = exp(src_g[g_base + col]);
#endif

        workgroupBarrier();

        let v_val = src_v[v_off + col];
        let beta_val = src_beta[gb_off];

        var kv_col = 0.0;
        var delta_col = 0.0;
        var attn_col = 0.0;

#ifdef KDA
        for (var i = 0u; i < S_V; i++) {
            kv_col += (sh_g[i] * state[i]) * sh_k[i];
        }

        delta_col = (v_val - kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = sh_g[i] * state[i] + sh_k[i] * delta_col;
            attn_col += state[i] * sh_q[i];
        }
#else
        let g_val = exp(src_g[gb_off]);

        for (var i = 0u; i < S_V; i++) {
            kv_col += state[i] * sh_k[i];
        }

        delta_col = (v_val - g_val * kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = g_val * state[i] + sh_k[i] * delta_col;
            attn_col += state[i] * sh_q[i];
        }
#endif

        dst[attn_off + col] = attn_col * params.scale;
        attn_off += S_V * params.h;

        workgroupBarrier();
    }

    for (var i = 0u; i < S_V; i++) {
        dst[params.s_off + state_base + col * S_V + i] = state[i];
    }
}
)wgsl";

static const char wgsl_get_rows_part0[] = R"wgsl(enable f16;
#define DECLARE_BYTE_LOADERS_SRC
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f305, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf0c)wgsl";

static const char wgsl_get_rows_part1[] = R"wgsl(7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif

#ifdef F32_VEC
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    dst[(dst_base / 4) + offset] = src[(src_base / 4) + offset];
}
#endif

#ifdef F32
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    dst[dst_base + offset] = src[src_base + offset];
}
#endif

#ifdef F16
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    dst[dst_base + offset] = f32(src[src_base + offset]);
}
#endif

#ifdef I32
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    dst[dst_base + offset] = src[src_base + offset];
}
#endif

#ifdef Q4_0
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 18; // Block stride: 18 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    for (var j: u32 = 0u; j < 4; j++) {
        let q_byte_offset = block_byte_base + 2 + j * 4;
        let q_packed = load_u32_at_src(q_byte_offset);
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0) * d;
            let q_lo = (f32(q_byte & 0xFu) - 8.0) * d;
            let dst_offset = dst_base + offset * 32 + j * 4 + k;
            dst[dst_offset] = q_lo;
            dst[dst_offset + 16u] = q_hi;
        }
    }
}
#endif

#ifdef Q4_1
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_q4_1 = src[src_base + offset];
    let d = f32(block_q4_1.d);
    let m = f32(block_q4_1.m);
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q4_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let q_hi = f32((q_byte >> 4) & 0xF) * d + m;
            let q_lo = f32(q_byte & 0xF) * d + m;
            let dst_offset = dst_base + offset * 32 + j * 4 + k;
            dst[dst_offset] = q_lo;
            dst[dst_offset + 16] = q_hi;
        }
    }
}
#endif

#ifdef Q5_0
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 22; // Block stride: 22 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    let qh_packed = load_u32_at_src(block_byte_base + 2);
    for (var j: u32 = 0; j < 4; j++) {
        let q_byte_offset = block_byte_base + 6 + j * 4;
        let q_packed = load_u32_at_src(q_byte_offset);

        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);

            let qh_hi = (qh_packed >> (j * 4 + k + 12)) & 0x10;
            let q_hi = (f32(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;

            let qh_lo = ((qh_packed >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = (f32((q_byte & 0xF) | qh_lo) - 16.0) * d;

            let dst_offset = dst_base + offset * 32 + j * 4 + k;
            dst[dst_offset] = q_lo;
            dst[dst_offset + 16] = q_hi;
        }
    }
}
#endif

#ifdef Q5_1
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_q5_1 = src[src_base + offset];
    let d = f32(block_q5_1.d);
    let m = f32(block_q5_1.m);
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q5_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let qh_hi = (block_q5_1.qh >> (j * 4 + k + 12)) & 0x10;
            let q_hi = f32(((q_byte >> 4) & 0xF) | qh_hi) * d + m;
            let qh_lo = ((block_q5_1.qh >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = f32((q_byte & 0xF) | qh_lo) * d + m;
            let dst_offset = dst_base + offset * 32 + j * 4 + k;
            dst[dst_offset] = q_lo;
            dst[dst_offset + 16] = q_hi;
        }
    }
}
#endif

#ifdef Q8_0
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 34; // Block stride: 34 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    for (var j: u32 = 0u; j < 8u; j++) {
        let q_byte_offset = block_byte_base + 2u + j * 4u;
        let q_packed = load_u32_at_src(q_byte_offset);
        for (var k: u32 = 0u; k < 4u; k++) {
            let q_byte = get_byte_i32(q_packed, k);
            let q_val = f32(q_byte) * d;
            let dst_offset = dst_base + offset * 32u + j * 4u + k;
            dst[dst_offset] = q_val;
        }
    }
}
#endif

#ifdef Q2_K
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block = src[src_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var dst_i = dst_base + offset * 256;
    var is: u32 = 0;
    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(block.scales[is / 4], is % 4);
                is++;
                let dl = d * f32(sc & 0xF);
                let ml = m * f32(sc >> 4);
                for (var l: u32 = 0u; l < 16; l++) {
                    let q_idx = q_b_idx + k + l;
                    let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                    let qs_val = (q_byte >> shift) & 3;
                    dst[dst_i] = (f32(qs_val) * dl - ml);
                    dst_i++;
                }
            }
        }
    }
}
#endif

#ifdef Q3_K
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 110; // Block stride: 110 bytes

    // Bytes 108-109: f16 scale 'd'
    let d = load_f16_as_f32_at_src(block_byte_base + 108);

    // Bytes 96-107: 12 bytes of scales (3 u32s)
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;

    var scale_vals: array<u32, 4>;
    scale_vals[0] = load_u32_at_src(block_byte_base + 96);
    scale_vals[1] = load_u32_at_src(block_byte_base + 100);
    scale_vals[2] = load_u32_at_src(block_byte_base + 104);

    var tmp: u32 = scale_vals[2];
    scale_vals[2] = ((scale_vals[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    scale_vals[3] = ((scale_vals[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4);
    scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    // Bytes 0-31: 32 bytes of hmask (8 u32s)
    var hmask_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        hmask_vals[i] = load_u32_at_src(block_byte_base + i * 4);
    }

    // Bytes 32-95: 64 bytes of qs (16 u32s)
    var qs_vals: array<u32, 16>;
    for (var i: u32 = 0u; i < 16; i++) {
        qs_vals[i] = load_u32_at_src(block_byte_base + 32 + i * 4);
    }

    var dst_i = dst_base + offset * 256;
    var is: u32 = 0;
    var m: u32 = 1;

    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(scale_vals[is / 4], is % 4);
                is++;
                let dl = d * (f32(sc) - 32.0);

                for (var l: u32 = 0; l < 16; l++) {
                    let q_idx = q_b_idx + k + l;
                    let hm_idx = k + l;
                    let q_byte = get_byte(qs_vals[q_idx / 4], q_idx % 4);
                    let hmask_byte = get_byte(hmask_vals[hm_idx / 4], hm_idx % 4);

                    let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
                    let qs_val = (q_byte >> shift) & 3;
                    dst[dst_i] = (f32(qs_val) - hm) * dl;
                    dst_i++;
                }
            }
            m <<= 1;
        }
    }
}
#endif

#ifdef Q4_K
// 8 blocks of 32 elements each
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block = src[src_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var dst_i = dst_base + offset * 256;
    var is: u32 = 0;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                dst[dst_i] = (f32(qs_val) * dl - ml);
                dst_i++;
            }
        }
    }
}
#endif

#ifdef Q5_K
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block = src[src_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var dst_i = dst_base + offset * 256;
    var is: u32 = 0;
    var u: u32 = 1;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qh_byte = get_byte(block.qh[l / 4], l % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);
                dst[dst_i] = (f32(qs_val) + qh_val) * dl - ml;
                dst_i++;
            }
            u <<= 1;
        }
    }
}
#endif

#ifdef Q6_K
// 16 blocks of 16 elements each
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 210; // Block stride: 210 bytes

    // Bytes 208-209: f16 scale 'd'
    let d = load_f16_as_f32_at_src(block_byte_base + 208);

    // Bytes 0-127: 128 bytes of ql (32 u32s)
    var ql_vals: array<u32, 32>;
    for (var i: u32 = 0; i < 32; i++) {
        ql_vals[i] = load_u32_at_src(block_byte_base + i * 4);
    }

    // Bytes 128-191: 64 bytes of qh (16 u32s)
    var qh_vals: array<u32, 16>;
    for (var i: u32 = 0; i < 16u; i++) {
        qh_vals[i] = load_u32_at_src(block_byte_base + 128 + i * 4u);
    }

    // Bytes 192-207: 16 bytes of scales (4 u32s)
    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        scale_vals[i] = load_u32_at_src(block_byte_base + 192 + i * 4);
    }

    var dst_i = dst_base + offset * 256;
    var qh_b_idx: u32 = 0;
    var sc_b_idx: u32 = 0;
    for (var ql_b_idx: u32 = 0; ql_b_idx < 128; ql_b_idx += 64) {
        for (var l: u32 = 0; l < 32; l++) {
            let ql13_b = get_byte(ql_vals[(ql_b_idx + l) / 4], (ql_b_idx + l) % 4);
            let ql24_b = get_byte(ql_vals[(ql_b_idx + l + 32) / 4], (ql_b_idx + l + 32) % 4);
            let qh_b = get_byte(qh_vals[(qh_b_idx + l) / 4], (qh_b_idx + l) % 4);

            let q1 = f32((ql13_b & 0xF) | ((qh_b & 3) << 4)) - 32.0;
            let q2 = f32((ql24_b & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32.0;
            let q3 = f32((ql13_b >> 4) | (((qh_b >> 4) & 3) << 4)) - 32.0;
            let q4 = f32((ql24_b >> 4) | (((qh_b >> 6) & 3) << 4)) - 32.0;

            let is = l/16;
            let is1 = sc_b_idx + is;
            let sc1 = get_byte_i32(scale_vals[is1 / 4], is1 % 4);
            let is2 = sc_b_idx + is + 2;
            let sc2 = get_byte_i32(scale_vals[is2 / 4], is2 % 4);
            let is3 = sc_b_idx + is + 4;
            let sc3 = get_byte_i32(scale_vals[is3 / 4], is3 % 4);
            let is4 = sc_b_idx + is + 6;
            let sc4 = get_byte_i32(scale_vals[is4 / 4], is4 % 4);

            dst[dst_i + l] = (q1 * f32(sc1)) * d;
            dst[dst_i + l + 32] = (q2 * f32(sc2)) * d;
            dst[dst_i + l + 64] = (q3 * f32(sc3)) * d;
            dst[dst_i + l + 96] = (q4 * f32(sc4)) * d;
        }
        dst_i += 128;
        qh_b_idx += 32;
        sc_b_idx += 8;
    }
}
#endif

#ifdef IQ2_XXS
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 66; // Block stride: 66 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let aux0_offset = block_byte_base + 2 + ib * 2;
        let aux1_offset = block_byte_base + 2 + (ib + 2) * 2;
        let aux0 = load_u32_at_src(aux0_offset);
        let aux1 = load_u32_at_src(aux1_offset);
        let db = d * (0.5 + f32(aux1 >> 28)) * 0.25;
        for (var l: u32 = 0; l < 4; l++) {
            let ig = get_byte(aux0, l) * 8;
            let is = (aux1 >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xxs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                dst[dst_i] = db * f32(g) * m;
                dst_i++;
            }
        }
    }
}
#endif



#ifdef IQ2_XS
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 74; // Block stride: 74 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;

    var scale_vals = array<u32, 2>(
        load_u32_at_src(block_byte_base + 66),
        load_u32_at_src(block_byte_base + 70)
    );

    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let s = get_byte(scale_vals[ib / 16], (ib % 16) / 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        for (var l: u32 = 0; l < 4; l++) {
            let qs_offset = block_byte_base + 2 + (ib + l) * 2;
            let qs_val = load_u32_at_src(qs_offset) & 0xFFFF;
            let ig = (qs_val & 511) * 8;
            let is = qs_val >> 9;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                dst[dst_i] = dl * f32(g) * m;
                dst_i++;
            }
        }
    }
}
#endif

#ifdef IQ2_S
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 82; // Block stride: 82 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;

    var qs_vals : array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qs_vals[i] = load_u32_at_src(block_byte_base + 2 + i * 4);
    }

    var qh_vals: array<u32, 2>;
    qh_vals[0] = load_u32_at_src(block_byte_base + 66);
    qh_vals[1] = load_u32_at_src(block_byte_base + 70);

    var scale_vals: array<u32, 2>;
    scale_vals[0] = load_u32_at_src(block_byte_base + 74);
    scale_vals[1] = load_u32_at_src(block_byte_base + 78);

    for (var ib: u32 = 0; ib < 8; ib ++) {
        let s = get_byte(scale_vals[ib / 4], ib % 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        let qs_w = qs_vals[ib];
        for (var l: u32 = 0; l < 4; l++) {
            let qh_b = (get_byte(qh_vals[ib / 4], ib % 4) << (8 - 2 * l)) & 0x300;
            let ig = (get_byte(qs_w, l) | qh_b) * 8;
            let signs = get_byte(qs_vals[ib + 8], l);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2s_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                dst[dst_i] = dl * f32(g) * m;
                dst_i++;
            }
        }
    }
}
#endif

#ifdef IQ3_XXS
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 98; // Block stride: 98 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;
    for (var ib: u32 = 0; ib < 16; ib += 2) {
        let sc_sign_offset = block_byte_base + 2 + (ib + 32) * 2;
        let sc_sign = load_u32_at_src(sc_sign_offset);
        let db = d * (0.5 + f32(sc_sign >> 28)) * 0.5;
        for (var l: u32 = 0; l < 4; l++) {
            let is = (sc_sign >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let ig_val = load_u32_at_src(block_byte_base + 2 + (ib * 2 + l) * 2) & 0xFFFF;
            let ig1 = get_byte(ig_val, 0);
            let ig2 = get_byte(ig_val, 1);
            for (var j: u32 = 0; j < 4; j++) {
                let g1 = get_byte(iq3xxs_grid[ig1], j);
                let g2 = get_byte(iq3xxs_grid[ig2], j);
                let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                dst[dst_i] = db * f32(g1) * m1;
                dst[dst_i + 4] = db * f32(g2) * m2;
                dst_i++;
            }
            dst_i += 4;
        }
    }
}
#endif

#ifdef IQ3_S
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 110; // Block stride: 110 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;

    var qh_vals = array<u32, 2>(
        load_u32_at_src(block_byte_base + 66),
        load_u32_at_src(block_byte_base + 70)
    );

    var sign_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        sign_vals[i] = load_u32_at_src(block_byte_base + 74 + i * 4);
    }

    var scale_vals = load_u32_at_src(block_byte_base + 106);

    for (var ib: u32 = 0; ib < 4; ib++) {
        let s = get_byte(scale_vals, ib);
        let db = array<f32, 2>(
            d * (1.0 + 2.0 * f32(s & 0xF)),
            d * (1.0 + 2.0 * f32(s >> 4))
        );
        for (var k: u32 = 0; k < 2; k++) {
            let dl = db[k];
            let qh_byte = get_byte(qh_vals[ib / 2], (ib % 2) * 2 + k);
            let sign_w = sign_vals[ib * 2 + k];
            for (var l: u32 = 0; l < 4; l++) {
                let signs = get_byte(sign_w, l);
                let ig_val = load_u32_at_src(block_byte_base + 2 + (ib * 8 + k * 4 + l) * 2) & 0xFFFF;
                let ig1 = get_byte(ig_val, 0) | ((qh_byte << ((8 - (2 * l)))) & 256);
                let ig2 = get_byte(ig_val, 1) | ((qh_byte << ((7 - (2 * l)))) & 256);
                for (var j: u32 = 0; j < 4; j++) {
                    let g1 = get_byte(iq3s_grid[ig1], j);
                    let g2 = get_byte(iq3s_grid[ig2], j);
                    let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                    let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                    dst[dst_i] = dl * f32(g1) * m1;
                    dst[dst_i + 4] = dl * f32(g2) * m2;
                    dst_i++;
                }
                dst_i += 4;
            }
        }
    }
}
#endif

#ifdef IQ1_S
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 50; // Block stride: 50 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 256;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let qh = load_u32_at_src(block_byte_base + 34 + ib * 2) & 0xFFFF;
        let dl = d * (2.0 * f32((qh >> 12) & 7) + 1.0);
        let delta = select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x8000) != 0);
        let qs_w = load_u32_at_src(block_byte_base + 2 + ib * 4);
        for (var l: u32 = 0; l < 4; l++) {
            let ig = (get_byte(qs_w, l) | (((qh >> (3 * l)) & 7) << 8)) * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                dst[dst_i] = dl * (f32(gs) + delta);
                dst_i++;
            }
        }
    }
}
#endif

#ifdef IQ1_M
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block = src[src_base + offset];

    let scale = ((block.scales[0] >> 12) & 0xF) | ((block.scales[0] >> 24) & 0x00F0) | ((block.scales[1] >> 4) & 0x0F00) | ((block.scales[1] >> 16) & 0xF000);
    let d = f32(bitcast<vec2<f16>>(scale).x);
    var dst_i = dst_base + offset * 256;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let sw = (block.scales[ib / 4] >> (16 * ((ib / 2) % 2))) & 0xFFFF;
        let s1 : u32 = (sw >> (6 * (ib % 2))) & 0x7;
        let s2 : u32 = (sw >> (6 * (ib % 2) + 3)) & 0x7;
        var dl = array<f32, 2>(
            d * f32(2 * s1 + 1),
            d * f32(2 * s2 + 1)
        );

        let qh = block.qh[ib / 2] >> (16 * (ib % 2));
        var idx = array<u32, 4>(
            get_byte(block.qs[ib], 0) | ((qh << 8) & 0x700),
            get_byte(block.qs[ib], 1) | ((qh << 4) & 0x700),
            get_byte(block.qs[ib], 2) | ((qh) & 0x700),
            get_byte(block.qs[ib], 3) | ((qh >> 4) & 0x700)
        );
        var delta = array<f32, 4>(
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x80) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x80) != 0)
        );
        for (var l: u32 = 0; l < 4; l++) {
            let ig = idx[l] * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                dst[dst_i] = dl[l/2] * (f32(gs) + delta[l]);
                dst_i++;
            }
        }
    }
}
#endif

#ifdef IQ4_NL
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block_byte_base = (src_base + offset) * 18; // Block stride: 18 bytes
    let d = load_f16_as_f32_at_src(block_byte_base);
    var dst_i = dst_base + offset * 32;
    var qs: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        qs[i] = load_u32_at_src(block_byte_base + 2 + i * 4);
    }
    for (var j: u32 = 0; j < 16; j++) {
        let qsb = get_byte(qs[j / 4], j % 4);
        dst[dst_i] = d * f32(kvalues_iq4nl[qsb & 0xF]);
        dst[dst_i + 16] = d * f32(kvalues_iq4nl[qsb >> 4]);
        dst_i++;
    }
}
#endif

#ifdef IQ4_XS
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let block = src[src_base + offset];
    let d = unpack2x16float(block.d_scales_h)[0];
    let scales_h = block.d_scales_h >> 16;
    var dst_i = dst_base + offset * 256;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let ls = ((get_byte(block.scales_l, ib / 2) >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        let dl = d * (f32(ls) - 32.0);
        for (var j: u32 = 0; j < 16; j++) {
            let iqs = ib * 16 + j;
            let qsb = get_byte(block.qs[iqs / 4], iqs % 4);
            dst[dst_i] = dl * f32(kvalues_iq4nl[qsb & 0xF]);
            dst[dst_i + 16] = dl * f32(kvalues_iq4nl[qsb >> 4]);
            dst_i++;
        }
        dst_i += 16;
    }
}
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> idx: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<DST_TYPE>;

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of dst
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
#ifdef FLOAT_PARALLEL
    let blocks_per_row = params.ne0 / BLOCK_SIZE;
    let row_count = params.n_rows * params.ne2 * params.ne3;

    if (gid.x >= blocks_per_row * row_count) {
        return;
    }

    let block_idx = gid.x % blocks_per_row;
    var row_idx = gid.x / blocks_per_row;
    let i_dst3 = row_idx / (params.ne2 * params.n_rows);

    row_idx = row_idx % (params.ne2 * params.n_rows);
    let i_dst2 = row_idx / params.n_rows;
    let i_dst1 = row_idx % params.n_rows;

    let i_idx2 = i_dst3 % params.idx2;
    let i_idx1 = i_dst2 % params.idx1;
    let i_idx0 = i_dst1;

    let i_idx = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;

    let idx_val = u32(idx[i_idx]);

    let i_src_row = params.offset_src + idx_val * params.stride_src1 + i_dst2 * params.stride_src2 + i_dst3 * params.stride_src3;
    let i_dst_row = params.offset_dst + i_dst1 * params.stride_dst1 + i_dst2 * params.stride_dst2 + i_dst3 * params.stride_dst3;

    copy_elements(i_src_row, i_dst_row, block_idx);
#else
    if (gid.x >= params.n_rows * params.ne2 * params.ne3) {
        return;
    }
    var i = gid.x;
    let i_dst3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_dst2 = i / params.n_rows;
    let i_dst1 = i % params.n_rows;

    let i_idx2 = i_dst3 % params.idx2;
    let i_idx1 = i_dst2 % params.idx1;
    let i_idx0 = i_dst1;

    let i_idx = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;

    let idx_val = u32(idx[i_idx]);

    let i_src_row = params.offset_src + idx_val * params.stride_src1 + i_dst2 * params.stride_src2 + i_dst3 * params.stride_src3;
    let i_dst_row = params.offset_dst + i_dst1 * params.stride_dst1 + i_dst2 * params.stride_dst2 + i_dst3 * params.stride_dst3;

    for (var i: u32 = 0; i < params.ne0/BLOCK_SIZE; i++) {
      copy_elements(i_src_row, i_dst_row, i);
    }
#endif
}
)wgsl";

static const std::string& wgsl_get_rows_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(98747);
        tmp.append(wgsl_get_rows_part0);
        tmp.append(wgsl_get_rows_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_get_rows = wgsl_get_rows_str().c_str();

const char* wgsl_glu = R"wgsl(enable f16;

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_F16
#define DataType f16
#endif

#ifdef OP_REGLU
fn op(a: DataType, b: DataType) -> DataType {
    return max(a, 0) * b;
}
#endif

#ifdef OP_GEGLU
const SQRT_2_OVER_PI: DataType =  0.79788456080286535587989211986876;
const GELU_COEF_A: DataType = 0.044715;

fn op(a: DataType, b: DataType) -> DataType {
    let val = SQRT_2_OVER_PI * a * (1.0 + GELU_COEF_A * a * a);
    return 0.5 * a * (2.0 - 2.0/ (exp(2* val) + 1)) * b;
}
#endif

#ifdef OP_SWIGLU
fn op(a: DataType, b: DataType) -> DataType {
    return a / (1.0 + exp(-a)) * b;
}
#endif
#ifdef OP_SWIGLU_OAI
fn op(a: f32, b: f32) -> f32 {
    let xi = min(a, params.limit);
    let gi = max(min(b, params.limit), -params.limit);
    var out_glu = xi / (1.0 + exp(-xi * params.alpha));
    out_glu = out_glu * (1.0 + gi);
    return out_glu;
}
#endif
#ifdef OP_GEGLU_ERF
const p_erf: DataType = 0.3275911;
const a1_erf: DataType = 0.254829592;
const a2_erf: DataType = -0.284496736;
const a3_erf: DataType = 1.421413741;
const a4_erf: DataType = -1.453152027;
const a5_erf: DataType = 1.061405429;
const SQRT_2_INV: DataType = 0.7071067811865476;

fn op(a: DataType, b: DataType) -> DataType {
    let a_div_sqr2 = a * SQRT_2_INV;
    let sign_x = sign(a_div_sqr2);
    let x = abs(a_div_sqr2);
    let t = 1.0 / (1.0 + p_erf * x);
    let y = 1.0 - (((((a5_erf * t + a4_erf) * t + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x));
    let erf_approx = sign_x * y;
    return 0.5 * a * (1.0 + erf_approx) * b;
}
#endif
#ifdef OP_GEGLU_QUICK
const GELU_QUICK_COEF: DataType = -1.702;

fn op(a: DataType, b: DataType) -> DataType {
    return a * (1.0 / (1.0 + exp(GELU_QUICK_COEF * a))) * b;
}
#endif

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // shape of dst
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    swapped: u32,
    alpha: f32,
    limit: f32,
}

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

#ifdef NO_SPLIT
@group(0) @binding(1)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(2)
var<uniform> params: Params;

fn a_value(base: u32) -> DataType {
    let offset: u32 = select(0, params.ne0, params.swapped != 0);
    return src0[base + offset];
}

fn b_value(base: u32) -> DataType {
    let offset: u32 = select(params.ne0, 0, params.swapped != 0);
    return src0[base + offset];
}

#else
@group(0) @binding(1)
var<storage, read_write> src1: array<DataType>;

@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
var<uniform> params: Params;

fn a_value(base: u32) -> DataType {
    return src0[base];
}

fn b_value(base: u32) -> DataType {
    return src1[base];
}

#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let i_a = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01 + i0;
    let i_b = params.offset_src1 + i3 * params.stride_src13 + i2 * params.stride_src12 + i1 * params.stride_src11 + i0;
    let i_dst = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1 + i0;

    dst[i_dst] = op(a_value(i_a), b_value(i_b));
}
)wgsl";

const char* wgsl_memset = R"wgsl(@group(0) @binding(0)
var<storage, read_write> output_buffer: array<u32>;

struct Params {
    offset: u32, // in bytes
    size: u32,   // in bytes
    value: u32,  // 4 8-bit values, which are either repeating (memset_tensor) or may be separate (cleaning up unaligned set_tensor operations)
};

@group(0) @binding(1)
var<uniform> params: Params;

override wg_size: u32;
override bytes_per_thread: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * bytes_per_thread;
    let start = params.offset;
    let end = params.offset + params.size;

    for (var j: u32 = 0u; j < bytes_per_thread; j += 4) {
        let byte_index = start + i + j;
        if (byte_index + 4 <= end) {
            output_buffer[byte_index >> 2] = params.value;
        } else {
            // Handle tail (unaligned)
            for (var k: u32 = 0; k < 4; k++) {
                let idx = byte_index + k;
                if (idx < end) {
                    let word_idx = idx >> 2;
                    let bit_offset = (idx & 3) * 8u;
                    let mask = ~(0xffu << bit_offset);
                    let existing = output_buffer[word_idx];
                    output_buffer[word_idx] = (existing & mask) | (params.value & (0xffu << bit_offset));
                }
            }
        }
    }
}
)wgsl";

static const char wgsl_mul_mat_part0[] = R"wgsl(enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f305, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf)wgsl";

static const char wgsl_mul_mat_part1[] = R"wgsl(0c7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif

#ifdef FLOAT
const BLOCK_SIZE = 1u;

#elif defined(Q4_0) || defined(Q4_1) || defined(Q5_0) || defined(Q5_1) || defined(Q8_0) || defined(Q8_1) || defined(IQ4_NL)
const BLOCK_SIZE = 32u;

#elif defined(Q2_K) || defined(Q3_K) || defined(Q4_K) || defined(Q5_K) || defined(Q6_K) || defined(IQ2_XXS) || defined(IQ2_XS) || defined(IQ2_S) || defined(IQ3_XXS) || defined(IQ3_S) || defined(IQ1_S) || defined(IQ1_M) || defined(IQ4_XS)
const BLOCK_SIZE = 256u;
#endif

#ifdef FLOAT
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    return f32(src0[src0_idx_base + offset]) * f32(src1[src1_idx_base + offset]);
}
#endif

#ifdef Q4_0
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 18; // Block stride: 18 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_byte_offset = block_byte_base + 2 + j * 4;
        let q_packed = load_u32_at_src0(q_byte_offset);
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
            let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#endif

#ifdef Q4_1
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q4_1 = src0[src0_idx_base + offset];
    let d = f32(block_q4_1.d);
    let m = f32(block_q4_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q4_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let q_hi = f32((q_byte >> 4) & 0xF) * d + m;
            let q_lo = f32(q_byte & 0xF) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#endif

#ifdef Q5_0
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 22; // Block stride: 22 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var sum: f32 = 0.0;
    let qh_packed = load_u32_at_src0(block_byte_base + 2);
    for (var j: u32 = 0; j < 4; j++) {
        let q_byte_offset = block_byte_base + 6 + j * 4;
        let q_packed = load_u32_at_src0(q_byte_offset);
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let qh_hi = (qh_packed >> (j * 4 + k + 12)) & 0x10;
            let q_hi = (f32(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
            let qh_lo = ((qh_packed >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = (f32((q_byte & 0xF) | qh_lo) - 16.0) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#endif

#ifdef Q5_1
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q5_1 = src0[src0_idx_base + offset];
    let d = f32(block_q5_1.d);
    let m = f32(block_q5_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q5_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let qh_hi = (block_q5_1.qh >> (j * 4 + k + 12)) & 0x10;
            let q_hi = f32(((q_byte >> 4) & 0xF) | qh_hi) * d + m;
            let qh_lo = ((block_q5_1.qh >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = f32((q_byte & 0xF) | qh_lo) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#endif

#ifdef Q8_0
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 34; // Block stride: 34 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 8; j++) {
        let q_byte_offset = block_byte_base + 2 + j * 4;
        let q_packed = load_u32_at_src0(q_byte_offset);
        for (var k: u32 = 0u; k < 4u; k++) {
            let q_byte = get_byte_i32(q_packed, k);
            let q_val = f32(q_byte) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_val * f32(src1[src1_offset]);
        }
    }
    return sum;
}
#endif

#ifdef Q8_1
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q8_1 = src0[src0_idx_base + offset];
    let d = f32(block_q8_1.d);
    let m = f32(block_q8_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 8; j++) {
        let q_packed = block_q8_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte_i32(q_packed, k);
            let q_val = f32(q_byte) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_val * f32(src1[src1_offset]);
        }
    }
    return sum;
}
#endif

#ifdef Q2_K
// 16 blocks of 16 elements each
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(block.scales[is / 4], is % 4);
                is++;
                let dl = d * f32(sc & 0xF);
                let ml = m * f32(sc >> 4);
                for (var l: u32 = 0u; l < 16; l++) {
                    let q_idx = q_b_idx + k + l;
                    let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                    let qs_val = (q_byte >> shift) & 3;
                    sum += (f32(qs_val) * dl - ml) * src1[src1_i];
                    src1_i++;
                }
            }
        }
    }
    return sum;
}
#endif

#ifdef Q3_K
// 16 blocks of 16 elements each
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 110; // Block stride: 110 bytes

    // Bytes 108-109: f16 scale 'd'
    let d = load_f16_as_f32_at_src0(block_byte_base + 108);

    // extract 6-bit scales, which consist of 4-bits from first 8 bytes of scale,
    // and 2-bits from the last 4 bytes
    // Bytes 96-107: 12 bytes of scales (3 u32s)
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    var scale_vals: array<u32, 4>;
    scale_vals[0] = load_u32_at_src0(block_byte_base + 96);
    scale_vals[1] = load_u32_at_src0(block_byte_base + 100);
    scale_vals[2] = load_u32_at_src0(block_byte_base + 104);

    var tmp: u32 = scale_vals[2];
    scale_vals[2] = ((scale_vals[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    scale_vals[3] = ((scale_vals[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4);
    scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    // Bytes 0-31: 32 bytes of hmask (8 u32s)
    var hmask_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        hmask_vals[i] = load_u32_at_src0(block_byte_base + i * 4);
    }

    // Bytes 32-95: 64 bytes of qs (16 u32s)
    var qs_vals: array<u32, 16>;
    for (var i: u32 = 0u; i < 16; i++) {
        qs_vals[i] = load_u32_at_src0(block_byte_base + 32 + i * 4);
    }

    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    var m: u32 = 1;
    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(scale_vals[is / 4], is % 4);
                is++;
                let dl = d * (f32(sc) - 32.0);
                for (var l: u32 = 0u; l < 16u; l++) {
                    let q_idx = q_b_idx + k + l;
                    let hm_idx = k + l;
                    let q_byte = get_byte(qs_vals[q_idx / 4], q_idx % 4);
                    let hmask_byte = get_byte(hmask_vals[hm_idx / 4], hm_idx % 4);
                    let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
                    let qs_val = (q_byte >> shift) & 3;
                    sum += ((f32(qs_val) - hm) * dl) * src1[src1_i];
                    src1_i++;
                }
            }
            m <<= 1;
        }
    }
    return sum;
}
#endif

#ifdef Q4_K
// 8 blocks of 32 elements each
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                sum += (f32(qs_val) * dl - ml) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif

#ifdef Q5_K
// 8 blocks of 32 elements each
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    var u: u32 = 1;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qh_byte = get_byte(block.qh[l / 4], l % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);
                sum += ((f32(qs_val) + qh_val) * dl - ml) * src1[src1_i];
               src1_i++;
            }
            u <<= 1;
        }
    }
    return sum;
}
#endif

#ifdef Q6_K
// 16 blocks of 16 elements each
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 210; // Block stride: 210 bytes

    // Bytes 208-209: f16 scale 'd'
    let d = load_f16_as_f32_at_src0(block_byte_base + 208);

    // Bytes 0-127: 128 bytes of ql (32 u32s)
    var ql_vals: array<u32, 32>;
    for (var i: u32 = 0; i < 32; i++) {
        ql_vals[i] = load_u32_at_src0(block_byte_base + i * 4);
    }

    // Bytes 128-191: 64 bytes of qh (16 u32s)
    var qh_vals: array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qh_vals[i] = load_u32_at_src0(block_byte_base + 128 + i * 4);
    }

    // Bytes 192-207: 16 bytes of scales (4 u32s)
    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        scale_vals[i] = load_u32_at_src0(block_byte_base + 192 + i * 4);
    }

    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var qh_b_idx: u32 = 0;
    var sc_b_idx: u32 = 0;
    for (var ql_b_idx: u32 = 0; ql_b_idx < 128; ql_b_idx += 64) {
        for (var l: u32 = 0; l < 32; l++) {
            let ql13_b = get_byte(ql_vals[(ql_b_idx + l) / 4], (ql_b_idx + l) % 4);
            let ql24_b = get_byte(ql_vals[(ql_b_idx + l + 32) / 4], (ql_b_idx + l + 32) % 4);
            let qh_b = get_byte(qh_vals[(qh_b_idx + l) / 4], (qh_b_idx + l) % 4);

            let q1 = f32((ql13_b & 0xF) | ((qh_b & 3) << 4)) - 32.0;
            let q2 = f32((ql24_b & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32.0;
            let q3 = f32((ql13_b >> 4) | (((qh_b >> 4) & 3) << 4)) - 32.0;
            let q4 = f32((ql24_b >> 4) | (((qh_b >> 6) & 3) << 4)) - 32.0;

            let is = l/16;
            let is1 = sc_b_idx + is;
            let sc1 = get_byte_i32(scale_vals[is1 / 4], is1 % 4);
            let is2 = sc_b_idx + is + 2;
            let sc2 = get_byte_i32(scale_vals[is2 / 4], is2 % 4);
            let is3 = sc_b_idx + is + 4;
            let sc3 = get_byte_i32(scale_vals[is3 / 4], is3 % 4);
            let is4 = sc_b_idx + is + 6;
            let sc4 = get_byte_i32(scale_vals[is4 / 4], is4 % 4);

            sum += d * f32(sc1) * q1 * src1[src1_i + l];
            sum += d * f32(sc2) * q2 * src1[src1_i + l + 32];
            sum += d * f32(sc3) * q3 * src1[src1_i + l + 64];
            sum += d * f32(sc4) * q4 * src1[src1_i + l + 96];
        }
        src1_i += 128;
        qh_b_idx += 32;
        sc_b_idx += 8;
    }
    return sum;
}
#endif

#ifdef IQ2_XXS
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 66; // Block stride: 66 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let aux0_offset = block_byte_base + 2 + ib * 2;
        let aux1_offset = block_byte_base + 2 + (ib + 2) * 2;
        let aux0 = load_u32_at_src0(aux0_offset);
        let aux1 = load_u32_at_src0(aux1_offset);
        let db = d * (0.5 + f32(aux1 >> 28)) * 0.25;
        for (var l: u32 = 0; l < 4; l++) {
            let ig = get_byte(aux0, l) * 8;
            let is = (aux1 >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xxs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += db * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif

#ifdef IQ2_XS
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 74; // Block stride: 74 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;

    var scale_vals = array<u32, 2>(
        load_u32_at_src0(block_byte_base + 66),
        load_u32_at_src0(block_byte_base + 70)
    );

    var sum = 0.0;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let s = get_byte(scale_vals[ib / 16], (ib % 16) / 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        for (var l: u32 = 0; l < 4; l++) {
            let qs_offset = block_byte_base + 2 + (ib + l) * 2;
            let qs_val = load_u32_at_src0(qs_offset) & 0xFFFF;
            let ig = (qs_val & 511) * 8;
            let is = qs_val >> 9;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += dl * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif

#ifdef IQ2_S
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 82; // Block stride: 82 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;

    var qs_vals : array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qs_vals[i] = load_u32_at_src0(block_byte_base + 2 + i * 4);
    }

    var qh_vals: array<u32, 2>;
    qh_vals[0] = load_u32_at_src0(block_byte_base + 66);
    qh_vals[1] = load_u32_at_src0(block_byte_base + 70);

    var scale_vals: array<u32, 2>;
    scale_vals[0] = load_u32_at_src0(block_byte_base + 74);
    scale_vals[1] = load_u32_at_src0(block_byte_base + 78);

    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib ++) {
        let s = get_byte(scale_vals[ib / 4], ib % 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        let qs_w = qs_vals[ib];
        for (var l: u32 = 0; l < 4; l++) {
            let qh_b = (get_byte(qh_vals[ib / 4], ib % 4) << (8 - 2 * l)) & 0x300;
            let ig = (get_byte(qs_w, l) | qh_b) * 8;
            let signs = get_byte(qs_vals[ib + 8], l);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2s_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += dl * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif

#ifdef IQ3_XXS
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 98; // Block stride: 98 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 16; ib += 2) {
        let sc_sign_offset = block_byte_base + 2 + (ib + 32) * 2;
        let sc_sign = load_u32_at_src0(sc_sign_offset);
        let db = d * (0.5 + f32(sc_sign >> 28)) * 0.5;
        for (var l: u32 = 0; l < 4; l++) {
            let is = (sc_sign >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let ig_val = load_u32_at_src0(block_byte_base + 2 + (ib * 2 + l) * 2) & 0xFFFF;
            let ig1 = get_byte(ig_val, 0);
            let ig2 = get_byte(ig_val, 1);
            for (var j: u32 = 0; j < 4; j++) {
                let g1 = get_byte(iq3xxs_grid[ig1], j);
                let g2 = get_byte(iq3xxs_grid[ig2], j);
                let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                sum += db * f32(g1) * m1 * src1[src1_i];
                sum += db * f32(g2) * m2 * src1[src1_i + 4];
                src1_i++;
            }
            src1_i += 4;
        }
    }
    return sum;
}
#endif

#ifdef IQ3_S
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 110; // Block stride: 110 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;

    var qh_vals = array<u32, 2>(
        load_u32_at_src0(block_byte_base + 66),
        load_u32_at_src0(block_byte_base + 70)
    );

    var sign_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        sign_vals[i] = load_u32_at_src0(block_byte_base + 74 + i * 4);
    }

    var scale_vals = load_u32_at_src0(block_byte_base + 106);

    var sum = 0.0;
    for (var ib: u32 = 0; ib < 4; ib++) {
        let s = get_byte(scale_vals, ib);
        let db = array<f32, 2>(
            d * (1.0 + 2.0 * f32(s & 0xF)),
            d * (1.0 + 2.0 * f32(s >> 4))
        );
        for (var k: u32 = 0; k < 2; k++) {
            let dl = db[k];
            let qh_byte = get_byte(qh_vals[ib / 2], (ib % 2) * 2 + k);
            let sign_w = sign_vals[ib * 2 + k];
            for (var l: u32 = 0; l < 4; l++) {
                let signs = get_byte(sign_w, l);
                let ig_val = load_u32_at_src0(block_byte_base + 2 + (ib * 8 + k * 4 + l) * 2) & 0xFFFF;
                let ig1 = get_byte(ig_val, 0) | ((qh_byte << ((8 - (2 * l)))) & 256);
                let ig2 = get_byte(ig_val, 1) | ((qh_byte << ((7 - (2 * l)))) & 256);
                for (var j: u32 = 0; j < 4; j++) {
                    let g1 = get_byte(iq3s_grid[ig1], j);
                    let g2 = get_byte(iq3s_grid[ig2], j);
                    let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                    let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                    sum += dl * f32(g1) * m1 * src1[src1_i];
                    sum += dl * f32(g2) * m2 * src1[src1_i + 4];
                    src1_i++;
                }
                src1_i += 4;
            }
        }
    }
    return sum;
}
#endif

#ifdef IQ1_S
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 50; // Block stride: 50 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let qh = load_u32_at_src0(block_byte_base + 34 + ib * 2) & 0xFFFF;
        let dl = d * (2.0 * f32((qh >> 12) & 7) + 1.0);
        let delta = select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x8000) != 0);
        let qs_w = load_u32_at_src0(block_byte_base + 2 + ib * 4);
        for (var l: u32 = 0; l < 4; l++) {
            let ig = (get_byte(qs_w, l) | (((qh >> (3 * l)) & 7) << 8)) * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                sum += dl * (f32(gs) + delta) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif


#ifdef IQ1_M
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];

    let scale = ((block.scales[0] >> 12) & 0xF) | ((block.scales[0] >> 24) & 0x00F0) | ((block.scales[1] >> 4) & 0x0F00) | ((block.scales[1] >> 16) & 0xF000);
    let d = f32(bitcast<vec2<f16>>(scale).x);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let sw = (block.scales[ib / 4] >> (16 * ((ib / 2) % 2))) & 0xFFFF;
        let s1 : u32 = (sw >> (6 * (ib % 2))) & 0x7;
        let s2 : u32 = (sw >> (6 * (ib % 2) + 3)) & 0x7;
        var dl = array<f32, 2>(
            d * f32(2 * s1 + 1),
            d * f32(2 * s2 + 1)
        );

        let qh = block.qh[ib / 2] >> (16 * (ib % 2));
        var idx = array<u32, 4>(
            get_byte(block.qs[ib], 0) | ((qh << 8) & 0x700),
            get_byte(block.qs[ib], 1) | ((qh << 4) & 0x700),
            get_byte(block.qs[ib], 2) | ((qh) & 0x700),
            get_byte(block.qs[ib], 3) | ((qh >> 4) & 0x700)
        );
        var delta = array<f32, 4>(
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x80) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x80) != 0)
        );
        for (var l: u32 = 0; l < 4; l++) {
            let ig = idx[l] * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                sum += dl[l/2] * (f32(gs) + delta[l]) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}
#endif

#ifdef IQ4_NL
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_byte_base = (src0_idx_base + offset) * 18; // Block stride: 18 bytes
    let d = load_f16_as_f32_at_src0(block_byte_base);
    var src1_i = src1_idx_base + offset * 32;
    var sum = 0.0;
    var qs: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        qs[i] = load_u32_at_src0(block_byte_base + 2 + i * 4);
    }
    for (var j: u32 = 0; j < 16; j++) {
        let qsb = get_byte(qs[j / 4], j % 4);
        sum += d * f32(kvalues_iq4nl[qsb & 0xF]) * src1[src1_i];
        sum += d * f32(kvalues_iq4nl[qsb >> 4]) * src1[src1_i + 16];
        src1_i++;
    }
    return sum;
}
#endif

#ifdef IQ4_XS
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = unpack2x16float(block.d_scales_h)[0];
    let scales_h = block.d_scales_h >> 16;
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let ls = ((get_byte(block.scales_l, ib / 2) >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        let dl = d * (f32(ls) - 32.0);
        for (var j: u32 = 0; j < 16; j++) {
            let iqs = ib * 16 + j;
            let qsb = get_byte(block.qs[iqs / 4], iqs % 4);
            sum += dl * f32(kvalues_iq4nl[qsb & 0xF]) * src1[src1_i];
            sum += dl * f32(kvalues_iq4nl[qsb >> 4]) * src1[src1_i + 16];
            src1_i++;
        }
        src1_i += 16;
    }
    return sum;
}
#endif

struct MulMatParams {
    offset_src0: u32, // in elements/blocks
    offset_src1: u32, // in elements/blocks
    offset_dst: u32, // in elements/blocks
    m: u32,
    n: u32,
    k: u32,
    // all strides are in elements/blocks
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,

    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let global_idx = wg_linear * 256u + local_id.x;

    let total = params.m * params.n * params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (global_idx >= total) {
        return;
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = global_idx / dst3_stride;
    let src03_idx = dst3_idx / params.broadcast3; // src0 may be broadcast along the third dimension
    let src13_idx = dst3_idx; // src1 is not broadcast
    let dst3_rem = global_idx % dst3_stride;

    let dst2_idx = dst3_rem / dst2_stride;
    let src02_idx = dst2_idx / params.broadcast2; // src0 may also be broadcast along the second dimension
    let src12_idx = dst2_idx; // src1 is not broadcast

    let dst2_rem = dst3_rem % dst2_stride;

    let row = dst2_rem / params.m; // output row
    let col = dst2_rem % params.m; // output column

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11;

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k/BLOCK_SIZE; i = i + 1u) {
        sum += multiply_add(src0_idx_base, src1_idx_base, i);
    }
    dst[params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.m + col] = sum;
}
)wgsl";

static const std::string& wgsl_mul_mat_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(100856);
        tmp.append(wgsl_mul_mat_part0);
        tmp.append(wgsl_mul_mat_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_mul_mat = wgsl_mul_mat_str().c_str();

static const char wgsl_mul_mat_id_part0[] = R"wgsl(enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f305, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf)wgsl";

static const char wgsl_mul_mat_id_part1[] = R"wgsl(0c7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif
#ifdef VEC
#define VEC_SIZE 4
#define SHMEM_TYPE vec4<f16>
#define DST_TYPE vec4<f32>
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn store_shmem(val: vec4<f16>, idx: u32) {
    shmem[idx] = val.x;
    shmem[idx + 1] = val.y;
    shmem[idx + 2] = val.z;
    shmem[idx + 3] = val.w;
}
#endif // VEC

#ifdef SCALAR
#define VEC_SIZE 1
#define SHMEM_TYPE f16
#define DST_TYPE f32
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn store_shmem(val: f16, idx: u32) {
    shmem[idx] = val;
}
#endif // SCALAR

#ifdef INIT_SRC0_SHMEM_FLOAT
fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;
        let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
        let src0_val = select( // taking a slight performance hit to avoid oob
            SRC0_TYPE(0.0),
            src0[src0_idx/VEC_SIZE],
            global_m < params.m && global_k < params.k);
        store_shmem(SHMEM_TYPE(src0_val), elem_idx);
    }
}
#endif // INIT_SRC0_SHMEM_FLOAT

#ifndef MUL_MAT_ID
#ifdef INIT_SRC1_SHMEM_FLOAT
fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_n = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_n = offset_n + tile_n;
        let global_k = k_outer + tile_k;
        let src1_idx = batch_offset + global_n * params.stride_11 + global_k;
        let src1_val = select(
            SRC1_TYPE(0.0),
            src1[src1_idx/VEC_SIZE],
            global_n < params.n && global_k < params.k);
        store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
    }
}
#endif // INIT_SRC1_SHMEM_FLOAT
#endif

#ifdef INIT_SRC0_SHMEM_Q4_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 18u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                    let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_0

#ifdef INIT_SRC0_SHMEM_Q4_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 20u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_lo = f16(q_byte & 0xF) * d + m;
                    let q_hi = f16((q_byte >> 4) & 0xF) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_1

#ifdef INIT_SRC0_SHMEM_Q5_0
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 22u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let qh_packed = load_u32_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < 2; j++) {
                let q_byte_offset = block_byte_base + 6u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo) - 16.0) * d;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_0

#ifdef INIT_SRC0_SHMEM_Q5_1
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 24u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);
            let qh_packed = load_u32_at_src0(block_byte_base + 4u);

            for (var j = 0u; j < 2; j++) {

                let q_byte_offset = block_byte_base + 8u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi)) * d + m;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo)) * d + m;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_1

#ifdef INIT_SRC0_SHMEM_Q8_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 34u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_0

#ifdef INIT_SRC0_SHMEM_Q8_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 36u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread, 2 threads per block

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_1

#ifdef INIT_SRC0_SHMEM_Q2_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 84u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    // Use standard thread layout instead of lane/row_group
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 80u);
        let dmin = load_f16_at_src0(block_byte_base + 82u);

        // Decode the element at position k_in_block
        let block_of_32 = k_in_block / 32u;
        let pos_in_32 = k_in_block % 32u;

        let q_b_idx = (block_of_32 / 4u) * 32u;
        let shift = (block_of_32 % 4u) * 2u;
        let k = (pos_in_32 / 16u) * 16u;
        let l = pos_in_32 % 16u;

        let is = k_in_block / 16u;

        let sc_packed = load_u32_at_src0(block_byte_base + 4u * (is / 4u));
        let sc = get_byte(sc_packed, is % 4u);

        let dl = d * f16(sc & 0xFu);
        let ml = dmin * f16(sc >> 4u);

        let q_idx = q_b_idx + k + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));
        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q2_K

#ifdef INIT_SRC0_SHMEM_Q3_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 110u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 108u);

        // Load and unpack scales
        let kmask1: u32 = 0x03030303u;
        let kmask2: u32 = 0x0f0f0f0fu;

        var scale_vals: array<u32, 4>;
        for (var i: u32 = 0u; i < 4u; i++) {
            scale_vals[i] = load_u32_at_src0(block_byte_base + 96u + 4u * i);
        }

        var tmp: u32 = scale_vals[2];
        scale_vals[2] = ((scale_vals[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        scale_vals[3] = ((scale_vals[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4u);
        scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

        // Load hmask and qs arrays
        var hmask_vals: array<u32, 8>;
        for (var i: u32 = 0u; i < 8u; i++) {
            hmask_vals[i] = load_u32_at_src0(block_byte_base + 4u * i);
        }

        var qs_vals: array<u32, 16>;
        for (var i: u32 = 0u; i < 16u; i++) {
            qs_vals[i] = load_u32_at_src0(block_byte_base + 32u + 4u * i);
        }

        let half = k_in_block / 128u;           // 0 or 1
        let pos_in_half = k_in_block % 128u;    // 0-127
        let shift_group = pos_in_half / 32u;    // 0-3
        let pos_in_32 = pos_in_half % 32u;      // 0-31
        let k_group = pos_in_32 / 16u;          // 0 or 1
        let l = pos_in_32 % 16u;                // 0-15

        let q_b_idx = half * 32u;               // 0 or 32
        let shift = shift_group * 2u;           // 0, 2, 4, 6
        let k = k_group * 16u;                  // 0 or 16
        let is = k_in_block / 16u;              // 0-15

        // m increments every 32 elements across entire 256 element block
        let m_shift = k_in_block / 32u;         // 0-7
        let m: u32 = 1u << m_shift;             // 1,2,4,8,16,32,64,128

        let sc = get_byte(scale_vals[is / 4u], is % 4u);
        let dl = d * (f16(sc) - 32.0);

        let q_idx = q_b_idx + k + l;
        let hm_idx = k + l;

        let q_byte = get_byte(qs_vals[q_idx / 4u], q_idx % 4u);
        let hmask_byte = get_byte(hmask_vals[hm_idx / 4u], hm_idx % 4u);

        let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = (f16(qs_val) - f16(hm)) * dl;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q3_K

#ifdef INIT_SRC0_SHMEM_Q4_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 144u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);

        // Map k_in_block to loop structure:
        // Outer loop over 64-element groups (alternating q_b_idx)
        // Inner loop over 2 shifts per group
        let group_of_64 = k_in_block / 64u;  // 0-3 (maps to q_b_idx)
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q4_K

#ifdef INIT_SRC0_SHMEM_Q5_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 176u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);


        // The original loop processes elements in groups of 64
        // Each group of 64: q_b_idx cycles through [0,32,64,96], shift cycles [0,4]
        // But u increments EVERY 32 elements (after each l loop)
        let group_of_64 = k_in_block / 64u;  // 0-3
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        // u increments every 32 elements (0->1, 1->2, 2->4, 3->8, 4->16, 5->32, 6->64, 7->128)
        let u_shift = k_in_block / 32u;      // 0-7
        let u: u32 = 1u << u_shift;

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 48u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);

        let qh_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (l / 4u));

        let qh_byte = get_byte(qh_packed, l % 4u);

        let qs_val = (q_byte >> shift) & 0xFu;
        let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);

        let q_val = (f16(qs_val) + f16(qh_val)) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q5_K

#ifdef INIT_SRC0_SHMEM_Q6_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 210u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let half = k_in_block / 128u;
        let pos_in_half = k_in_block % 128u;
        let quarter = pos_in_half / 32u;
        let l = pos_in_half % 32u;

        let ql_b_idx = half * 64u;
        let qh_b_idx = half * 32u;
        let sc_b_idx = half * 8u;

        // Load only ql13 word needed
        let ql13_flat = ql_b_idx + l;
        let ql13 = load_u32_at_src0(block_byte_base + ql13_flat);
        let ql13_b = get_byte(ql13, 0u);

        // Load only ql24 word needed
        let ql24_flat = ql_b_idx + l + 32u;
        let ql24 = load_u32_at_src0(block_byte_base + ql24_flat);
        let ql24_b = get_byte(ql24, 0u);

        // Load only qh word needed
        let qh_flat = qh_b_idx + l;
        let qh = load_u32_at_src0(block_byte_base + 128u + qh_flat);
        let qh_b = get_byte(qh, 0u);

        let q1 = f16((ql13_b & 0xFu) | ((qh_b & 3u) << 4u)) - f16(32.0);
        let q2 = f16((ql24_b & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - f16(32.0);
        let q3 = f16((ql13_b >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - f16(32.0);
        let q4 = f16((ql24_b >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - f16(32.0);

        // Load only the scale word needed
        let is = l / 16u;
        let sc_idx = sc_b_idx + is + quarter * 2u;
        let sc = load_u32_at_src0(block_byte_base + 192u + sc_idx);
        let sc_val = get_byte_i32(sc, 0u);

        let d = load_f16_at_src0(block_byte_base + 208u);

        var q_val: f16;
        if (quarter == 0u) {
            q_val = q1;
        } else if (quarter == 1u) {
            q_val = q2;
        } else if (quarter == 2u) {
            q_val = q3;
        } else {
            q_val = q4;
        }

        shmem[elem_idx] = d * f16(sc_val) * q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q6_K

#ifdef VEC
fn store_val(acc: array<array<f16, TILE_M>, TILE_N>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(f32(acc[tn][tm]), f32(acc[tn][tm + 1]), f32(acc[tn][tm + 2]), f32(acc[tn][tm + 3]));
}
#endif

#ifdef SCALAR
fn store_val(acc: array<array<f16, TILE_M>, TILE_N>, tn: u32, tm: u32) -> f32 {
    return f32(acc[tn][tm]);
}
#endif

struct MulMatIdParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    k: u32,
    m: u32,
    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,
    b_ne1: u32,

    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // [cols, rows, n_expert]
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // [cols, b_ne1, n_tokens]
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;   // [rows, n_expert_used, n_tokens]
@group(0) @binding(3) var<storage, read_write> global_gathered_expert_used: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(4) var<storage, read_write> global_gathered_tokens: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(5) var<storage, read_write> gathered_count_ids: array<u32>; // [n_expert]

@group(0) @binding(6) var<uniform> params: MulMatIdParams;

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
const TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> shmem: array<f16, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;
var<workgroup> gathered_expert_used: array<u32, TILE_N * WORKGROUP_SIZE_N>;
var<workgroup> gathered_tokens: array<u32, TILE_N * WORKGROUP_SIZE_N>;

#ifdef INIT_SRC1_SHMEM_FLOAT
fn init_shmem_id_src1(thread_id: u32, offset_src1: u32, rest_token_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_n = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        if (tile_n < rest_token_n) {
            let global_src10 = k_outer + tile_k;
            let expert_used_idx = gathered_expert_used[tile_n] % params.b_ne1;
            let token_idx = gathered_tokens[tile_n];
            let src1_idx = offset_src1 + token_idx * params.stride_12 + expert_used_idx * params.stride_11 + global_src10;
            let src1_val = select(
                SRC1_TYPE(0.0),
                src1[src1_idx/VEC_SIZE],
                global_src10 < params.k);
            store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
        } else {
            store_shmem(SHMEM_TYPE(0.0), TILE_SRC0_SHMEM + elem_idx);
        }
    }
}
#endif // INIT_SRC1_SHMEM_FLOAT

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    var expert_idx:u32 = 0xFFFFFFFFu;
    var wg_in_batch:u32 = 0;
    var wg_sum:u32 = 0;
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    for (var i = 0u;i < params.n_expert;i += 1) {
        let wg_n_count = (gathered_count_ids[i] + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
        let wg_per_matrix = wg_m_count * wg_n_count;
        if (wg_sum <= wg_linear && wg_linear < wg_sum + wg_per_matrix) {
            expert_idx = i;
            wg_in_batch = wg_linear - wg_sum;
            break;
        }
        wg_sum += wg_per_matrix;
    }

    let is_valid = expert_idx != 0xFFFFFFFFu;

    var wg_m: u32 = 0;
    var wg_n: u32 = 0;
    var offset_wg_m: u32 = 0;
    var offset_wg_n: u32 = 0;
    var rest_token_n: u32 = 0;
    var src0_batch_offset: u32 = 0;

    wg_m = wg_in_batch % wg_m_count;
    wg_n = wg_in_batch / wg_m_count;

    offset_wg_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    offset_wg_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    if (is_valid) {
        rest_token_n = gathered_count_ids[expert_idx] - offset_wg_n;
        let global_gathered_base = expert_idx * params.n_tokens + offset_wg_n;
        for (var i = thread_id; i < TILE_N * WORKGROUP_SIZE_N && offset_wg_n + i < gathered_count_ids[expert_idx]; i += TOTAL_WORKGROUP_SIZE) {
            gathered_expert_used[i] = global_gathered_expert_used[global_gathered_base + i];
            gathered_tokens[i] = global_gathered_tokens[global_gathered_base + i];
        }
        src0_batch_offset = params.offset_src0 + expert_idx * params.stride_02;
    }

    workgroupBarrier();

    let output_row_base = offset_wg_m + local_m * TILE_M;
    let output_col_base = offset_wg_n + local_n * TILE_N;

    let dst2_stride = params.m * params.n_expert_used;
    let dst1_stride = params.m;

    var acc: array<array<f16, TILE_M>, TILE_N>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        if (is_valid) {
            init_shmem_src0(thread_id, src0_batch_offset, offset_wg_m, k_outer);
            init_shmem_id_src1(thread_id, params.offset_src1, rest_token_n, k_outer);
        }

        workgroupBarrier();

        if (is_valid) {
            let k_end = min(TILE_K, params.k - k_outer);

            for (var k_inner = 0u; k_inner < k_end; k_inner++) {
                var src0_tile: array<f16, TILE_M>;
                for (var tm = 0u; tm < TILE_M; tm++) {
                    let src0_m = local_m * TILE_M + tm;
                    let src0_idx = k_inner + src0_m * TILE_K;
                    src0_tile[tm] = shmem[src0_idx];
                }
                for (var tn = 0u; tn < TILE_N; tn++) {
                    let src1_n = local_n * TILE_N + tn;
                    let src1_idx = src1_n * TILE_K + k_inner;
                    let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
                    for (var tm = 0u; tm < TILE_M; tm++) {
                        acc[tn][tm] += src0_tile[tm] * src1_val;
                    }
                }
            }
        }

        workgroupBarrier();
    }

    if (is_valid) {
        for (var tn = 0u; tn < TILE_N; tn++) {
            let n_idx = output_col_base + tn;
            if (n_idx < gathered_count_ids[expert_idx]) {
                let dst1_idx = gathered_expert_used[n_idx - offset_wg_n];
                let dst2_idx = gathered_tokens[n_idx - offset_wg_n];
                let dst12_offset = params.offset_dst + dst2_idx * dst2_stride + dst1_idx * dst1_stride;
                for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                    let global_row = output_row_base + tm;
                    if (global_row < params.m) {
                        let dst_idx = dst12_offset + global_row;
                        dst[dst_idx/VEC_SIZE] = store_val(acc, tn, tm);
                    }
                }
            }
        }
    }
}
)wgsl";

static const std::string& wgsl_mul_mat_id_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(107208);
        tmp.append(wgsl_mul_mat_id_part0);
        tmp.append(wgsl_mul_mat_id_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_mul_mat_id = wgsl_mul_mat_id_str().c_str();

const char* wgsl_mul_mat_id_gather = R"wgsl(enable f16;

struct MulMatIdGatherParams {
    offset_ids: u32,

    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,

    stride_ids_1: u32,
};

@group(0) @binding(0) var<storage, read_write> ids: array<i32>;        // [n_expert_used, n_tokens]
@group(0) @binding(1) var<storage, read_write> global_gathered_expert_used: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(2) var<storage, read_write> global_gathered_tokens: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(3) var<storage, read_write> gathered_count_ids: array<u32>; // [n_expert]

@group(0) @binding(4) var<uniform> params: MulMatIdGatherParams;

var<workgroup> count:atomic<u32>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let own_expert = wg_id.y * num_wg.x + wg_id.x; // the expert assigned to this workgroup

    if (own_expert < params.n_expert) {
        if (thread_id == 0u) {
            atomicStore(&count, 0);
        }

        workgroupBarrier();

        for (var i = thread_id;i < params.n_expert_used * params.n_tokens;i += WG_SIZE) {
            let row = i / params.n_expert_used;
            let col = i % params.n_expert_used;
            let expert = u32(ids[params.offset_ids + row * params.stride_ids_1 + col]);
            if (own_expert == expert) {
                let pos = atomicAdd(&count, 1u);
                let gathered_id = own_expert * params.n_tokens + pos;
                global_gathered_expert_used[gathered_id] = col;
                global_gathered_tokens[gathered_id] = row;
            }
        }

        workgroupBarrier();

        if (thread_id == 0u) {
            gathered_count_ids[own_expert] = atomicLoad(&count);
        }
    }
}
)wgsl";

static const char wgsl_mul_mat_reg_tile_part0[] = R"wgsl(enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f305, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf)wgsl";

static const char wgsl_mul_mat_reg_tile_part1[] = R"wgsl(0c7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif
#ifdef VEC
#define VEC_SIZE 4
#define SHMEM_TYPE vec4<f16>
#define DST_TYPE vec4<f32>
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn store_shmem(val: vec4<f16>, idx: u32) {
    shmem[idx] = val.x;
    shmem[idx + 1] = val.y;
    shmem[idx + 2] = val.z;
    shmem[idx + 3] = val.w;
}
#endif // VEC

#ifdef SCALAR
#define VEC_SIZE 1
#define SHMEM_TYPE f16
#define DST_TYPE f32
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn store_shmem(val: f16, idx: u32) {
    shmem[idx] = val;
}
#endif // SCALAR

#ifdef INIT_SRC0_SHMEM_FLOAT
fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;
        let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
        let src0_val = select( // taking a slight performance hit to avoid oob
            SRC0_TYPE(0.0),
            src0[src0_idx/VEC_SIZE],
            global_m < params.m && global_k < params.k);
        store_shmem(SHMEM_TYPE(src0_val), elem_idx);
    }
}
#endif // INIT_SRC0_SHMEM_FLOAT

#ifndef MUL_MAT_ID
#ifdef INIT_SRC1_SHMEM_FLOAT
fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_n = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_n = offset_n + tile_n;
        let global_k = k_outer + tile_k;
        let src1_idx = batch_offset + global_n * params.stride_11 + global_k;
        let src1_val = select(
            SRC1_TYPE(0.0),
            src1[src1_idx/VEC_SIZE],
            global_n < params.n && global_k < params.k);
        store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
    }
}
#endif // INIT_SRC1_SHMEM_FLOAT
#endif

#ifdef INIT_SRC0_SHMEM_Q4_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 18u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                    let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_0

#ifdef INIT_SRC0_SHMEM_Q4_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 20u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_lo = f16(q_byte & 0xF) * d + m;
                    let q_hi = f16((q_byte >> 4) & 0xF) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_1

#ifdef INIT_SRC0_SHMEM_Q5_0
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 22u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let qh_packed = load_u32_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < 2; j++) {
                let q_byte_offset = block_byte_base + 6u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo) - 16.0) * d;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_0

#ifdef INIT_SRC0_SHMEM_Q5_1
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 24u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);
            let qh_packed = load_u32_at_src0(block_byte_base + 4u);

            for (var j = 0u; j < 2; j++) {

                let q_byte_offset = block_byte_base + 8u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi)) * d + m;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo)) * d + m;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_1

#ifdef INIT_SRC0_SHMEM_Q8_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 34u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_0

#ifdef INIT_SRC0_SHMEM_Q8_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 36u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread, 2 threads per block

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_1

#ifdef INIT_SRC0_SHMEM_Q2_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 84u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    // Use standard thread layout instead of lane/row_group
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 80u);
        let dmin = load_f16_at_src0(block_byte_base + 82u);

        // Decode the element at position k_in_block
        let block_of_32 = k_in_block / 32u;
        let pos_in_32 = k_in_block % 32u;

        let q_b_idx = (block_of_32 / 4u) * 32u;
        let shift = (block_of_32 % 4u) * 2u;
        let k = (pos_in_32 / 16u) * 16u;
        let l = pos_in_32 % 16u;

        let is = k_in_block / 16u;

        let sc_packed = load_u32_at_src0(block_byte_base + 4u * (is / 4u));
        let sc = get_byte(sc_packed, is % 4u);

        let dl = d * f16(sc & 0xFu);
        let ml = dmin * f16(sc >> 4u);

        let q_idx = q_b_idx + k + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));
        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q2_K

#ifdef INIT_SRC0_SHMEM_Q3_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 110u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 108u);

        // Load and unpack scales
        let kmask1: u32 = 0x03030303u;
        let kmask2: u32 = 0x0f0f0f0fu;

        var scale_vals: array<u32, 4>;
        for (var i: u32 = 0u; i < 4u; i++) {
            scale_vals[i] = load_u32_at_src0(block_byte_base + 96u + 4u * i);
        }

        var tmp: u32 = scale_vals[2];
        scale_vals[2] = ((scale_vals[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        scale_vals[3] = ((scale_vals[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4u);
        scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

        // Load hmask and qs arrays
        var hmask_vals: array<u32, 8>;
        for (var i: u32 = 0u; i < 8u; i++) {
            hmask_vals[i] = load_u32_at_src0(block_byte_base + 4u * i);
        }

        var qs_vals: array<u32, 16>;
        for (var i: u32 = 0u; i < 16u; i++) {
            qs_vals[i] = load_u32_at_src0(block_byte_base + 32u + 4u * i);
        }

        let half = k_in_block / 128u;           // 0 or 1
        let pos_in_half = k_in_block % 128u;    // 0-127
        let shift_group = pos_in_half / 32u;    // 0-3
        let pos_in_32 = pos_in_half % 32u;      // 0-31
        let k_group = pos_in_32 / 16u;          // 0 or 1
        let l = pos_in_32 % 16u;                // 0-15

        let q_b_idx = half * 32u;               // 0 or 32
        let shift = shift_group * 2u;           // 0, 2, 4, 6
        let k = k_group * 16u;                  // 0 or 16
        let is = k_in_block / 16u;              // 0-15

        // m increments every 32 elements across entire 256 element block
        let m_shift = k_in_block / 32u;         // 0-7
        let m: u32 = 1u << m_shift;             // 1,2,4,8,16,32,64,128

        let sc = get_byte(scale_vals[is / 4u], is % 4u);
        let dl = d * (f16(sc) - 32.0);

        let q_idx = q_b_idx + k + l;
        let hm_idx = k + l;

        let q_byte = get_byte(qs_vals[q_idx / 4u], q_idx % 4u);
        let hmask_byte = get_byte(hmask_vals[hm_idx / 4u], hm_idx % 4u);

        let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = (f16(qs_val) - f16(hm)) * dl;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q3_K

#ifdef INIT_SRC0_SHMEM_Q4_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 144u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);

        // Map k_in_block to loop structure:
        // Outer loop over 64-element groups (alternating q_b_idx)
        // Inner loop over 2 shifts per group
        let group_of_64 = k_in_block / 64u;  // 0-3 (maps to q_b_idx)
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q4_K

#ifdef INIT_SRC0_SHMEM_Q5_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 176u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);


        // The original loop processes elements in groups of 64
        // Each group of 64: q_b_idx cycles through [0,32,64,96], shift cycles [0,4]
        // But u increments EVERY 32 elements (after each l loop)
        let group_of_64 = k_in_block / 64u;  // 0-3
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        // u increments every 32 elements (0->1, 1->2, 2->4, 3->8, 4->16, 5->32, 6->64, 7->128)
        let u_shift = k_in_block / 32u;      // 0-7
        let u: u32 = 1u << u_shift;

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 48u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);

        let qh_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (l / 4u));

        let qh_byte = get_byte(qh_packed, l % 4u);

        let qs_val = (q_byte >> shift) & 0xFu;
        let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);

        let q_val = (f16(qs_val) + f16(qh_val)) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q5_K

#ifdef INIT_SRC0_SHMEM_Q6_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 210u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let half = k_in_block / 128u;
        let pos_in_half = k_in_block % 128u;
        let quarter = pos_in_half / 32u;
        let l = pos_in_half % 32u;

        let ql_b_idx = half * 64u;
        let qh_b_idx = half * 32u;
        let sc_b_idx = half * 8u;

        // Load only ql13 word needed
        let ql13_flat = ql_b_idx + l;
        let ql13 = load_u32_at_src0(block_byte_base + ql13_flat);
        let ql13_b = get_byte(ql13, 0u);

        // Load only ql24 word needed
        let ql24_flat = ql_b_idx + l + 32u;
        let ql24 = load_u32_at_src0(block_byte_base + ql24_flat);
        let ql24_b = get_byte(ql24, 0u);

        // Load only qh word needed
        let qh_flat = qh_b_idx + l;
        let qh = load_u32_at_src0(block_byte_base + 128u + qh_flat);
        let qh_b = get_byte(qh, 0u);

        let q1 = f16((ql13_b & 0xFu) | ((qh_b & 3u) << 4u)) - f16(32.0);
        let q2 = f16((ql24_b & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - f16(32.0);
        let q3 = f16((ql13_b >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - f16(32.0);
        let q4 = f16((ql24_b >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - f16(32.0);

        // Load only the scale word needed
        let is = l / 16u;
        let sc_idx = sc_b_idx + is + quarter * 2u;
        let sc = load_u32_at_src0(block_byte_base + 192u + sc_idx);
        let sc_val = get_byte_i32(sc, 0u);

        let d = load_f16_at_src0(block_byte_base + 208u);

        var q_val: f16;
        if (quarter == 0u) {
            q_val = q1;
        } else if (quarter == 1u) {
            q_val = q2;
        } else if (quarter == 2u) {
            q_val = q3;
        } else {
            q_val = q4;
        }

        shmem[elem_idx] = d * f16(sc_val) * q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q6_K

#ifdef VEC
fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(acc[tm][tn], acc[tm + 1][tn], acc[tm + 2][tn], acc[tm + 3][tn]);
}
#endif

#ifdef SCALAR
fn store_val(acc: array<array<f32, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return acc[tm][tn];
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
const TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;

var<workgroup> shmem: array<f16, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    let wg_n_count = (params.n + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_per_matrix = wg_m_count * wg_n_count;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    let batch_idx = wg_linear / wg_per_matrix;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
    let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    var acc: array<array<f32, TILE_N>, TILE_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // see mul_mat_decls.tmpl
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f16, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
                src0_tile[tm] = shmem[src0_idx];
            }
            for (var tn = 0u; tn < TILE_N; tn++) {
                let src1_n = local_n * TILE_N + tn;
                let src1_idx = src1_n * TILE_K + k_inner;
                let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
                for (var tm = 0u; tm < TILE_M; tm++) {
                      acc[tm][tn] += f32(src0_tile[tm]) * f32(src1_val);
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_col = output_col_base + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                let global_row = output_row_base + tm;
                if (global_row < params.m) {
                    let dst_idx = dst_batch_offset + global_col * params.m + global_row;
                    dst[dst_idx/VEC_SIZE] = store_val(acc, tn, tm);
                }
            }
        }
    }
}
)wgsl";

static const std::string& wgsl_mul_mat_reg_tile_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(105052);
        tmp.append(wgsl_mul_mat_reg_tile_part0);
        tmp.append(wgsl_mul_mat_reg_tile_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_mul_mat_reg_tile = wgsl_mul_mat_reg_tile_str().c_str();

static const char wgsl_mul_mat_subgroup_matrix_part0[] = R"wgsl(diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

#define DECLARE_BYTE_LOADERS_SRC0
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5)wgsl";

static const char wgsl_mul_mat_subgroup_matrix_part1[] = R"wgsl(d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f305, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf0c7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif
#ifdef VEC
#define VEC_SIZE 4
#define SHMEM_TYPE vec4<f16>
#define DST_TYPE vec4<f32>
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn store_shmem(val: vec4<f16>, idx: u32) {
    shmem[idx] = val.x;
    shmem[idx + 1] = val.y;
    shmem[idx + 2] = val.z;
    shmem[idx + 3] = val.w;
}
#endif // VEC

#ifdef SCALAR
#define VEC_SIZE 1
#define SHMEM_TYPE f16
#define DST_TYPE f32
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn store_shmem(val: f16, idx: u32) {
    shmem[idx] = val;
}
#endif // SCALAR

#ifdef INIT_SRC0_SHMEM_FLOAT
fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;
        let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
        let src0_val = select( // taking a slight performance hit to avoid oob
            SRC0_TYPE(0.0),
            src0[src0_idx/VEC_SIZE],
            global_m < params.m && global_k < params.k);
        store_shmem(SHMEM_TYPE(src0_val), elem_idx);
    }
}
#endif // INIT_SRC0_SHMEM_FLOAT

#ifndef MUL_MAT_ID
#ifdef INIT_SRC1_SHMEM_FLOAT
fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_n = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        let global_n = offset_n + tile_n;
        let global_k = k_outer + tile_k;
        let src1_idx = batch_offset + global_n * params.stride_11 + global_k;
        let src1_val = select(
            SRC1_TYPE(0.0),
            src1[src1_idx/VEC_SIZE],
            global_n < params.n && global_k < params.k);
        store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
    }
}
#endif // INIT_SRC1_SHMEM_FLOAT
#endif

#ifdef INIT_SRC0_SHMEM_Q4_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 18u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                    let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_0

#ifdef INIT_SRC0_SHMEM_Q4_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 20u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);
                    let q_lo = f16(q_byte & 0xF) * d + m;
                    let q_hi = f16((q_byte >> 4) & 0xF) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_lo;
                    shmem[shmem_idx + j * 2 + k + 16u] = q_hi;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q4_1

#ifdef INIT_SRC0_SHMEM_Q5_0
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 22u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let qh_packed = load_u32_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < 2; j++) {
                let q_byte_offset = block_byte_base + 6u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo) - 16.0) * d;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_0

#ifdef INIT_SRC0_SHMEM_Q5_1
// 32 weights per block, each at 4 bits each = 32 * 4 = 128 bits / 16 = 8 f16s per block
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 24u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
// tile_k is defined as 32u, so blocks_k ends up being 1 always
override BLOCKS_K = TILE_K / BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 16 / 4 = 4 f16s per thread, each thread should handle 4 f16s * 4 weights per = 16 weights

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {

    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx    = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx   = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m   = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k  = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx  = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

            let d  = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);
            let qh_packed = load_u32_at_src0(block_byte_base + 4u);

            for (var j = 0u; j < 2; j++) {

                let q_byte_offset = block_byte_base + 8u + 2u * (block_offset + j * 2u);
                let q_packed = load_u32_at_src0(q_byte_offset);

                let j_adjusted = j + (block_offset / 2u);


                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte(q_packed, k);

                    let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                    let q_hi = (f16(((q_byte >> 4) & 0xF) | qh_hi)) * d + m;
                    let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                    let q_lo = (f16((q_byte & 0xF) | qh_lo)) * d + m;

                    shmem[shmem_idx + j * 4u + k]        = q_lo; // store first weight
                    shmem[shmem_idx + j * 4u + k + 16u]  = q_hi; // store second weight
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q5_1

#ifdef INIT_SRC0_SHMEM_Q8_0
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 34u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 2u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_0

#ifdef INIT_SRC0_SHMEM_Q8_1
const BLOCK_SIZE = 32u;
const BLOCK_SIZE_BYTES = 36u;
// the number of blocks per k-tile. Note that this currently only works if TILE_K is a multiple of BLOCK_SIZE, which may need to be rethought for larger quantized types.
override BLOCKS_K = TILE_K/BLOCK_SIZE;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u; // 2 8-bit weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16; // 8 f16s per thread, 2 threads per block

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;

        let tile_m = blck_idx / BLOCKS_K;
        let global_m = offset_m + tile_m;
        let block_k = blck_idx % BLOCKS_K;
        let global_k = k_outer / BLOCK_SIZE + block_k;

        if (global_m < params.m && global_k < params.k / BLOCK_SIZE) {
            let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
            let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
            let d = load_f16_at_src0(block_byte_base);
            let m = load_f16_at_src0(block_byte_base + 2u);

            for (var j = 0u; j < F16_PER_THREAD; j+=2) {
                let q_byte_offset = block_byte_base + 4u + 2u * (block_offset + j);
                let q_packed = load_u32_at_src0(q_byte_offset);
                for (var k = 0u; k < 4u; k++) {
                    let q_byte = get_byte_i32(q_packed, k);

                    let q_val = f16(q_byte) * d + m;
                    shmem[shmem_idx + j * 2 + k] = q_val;
                }
            }
        }
    }
}
#endif // INIT_SRC0_SHMEM_Q8_1

#ifdef INIT_SRC0_SHMEM_Q2_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 84u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    // Use standard thread layout instead of lane/row_group
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 80u);
        let dmin = load_f16_at_src0(block_byte_base + 82u);

        // Decode the element at position k_in_block
        let block_of_32 = k_in_block / 32u;
        let pos_in_32 = k_in_block % 32u;

        let q_b_idx = (block_of_32 / 4u) * 32u;
        let shift = (block_of_32 % 4u) * 2u;
        let k = (pos_in_32 / 16u) * 16u;
        let l = pos_in_32 % 16u;

        let is = k_in_block / 16u;

        let sc_packed = load_u32_at_src0(block_byte_base + 4u * (is / 4u));
        let sc = get_byte(sc_packed, is % 4u);

        let dl = d * f16(sc & 0xFu);
        let ml = dmin * f16(sc >> 4u);

        let q_idx = q_b_idx + k + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));
        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q2_K

#ifdef INIT_SRC0_SHMEM_Q3_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 110u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base + 108u);

        // Load and unpack scales
        let kmask1: u32 = 0x03030303u;
        let kmask2: u32 = 0x0f0f0f0fu;

        var scale_vals: array<u32, 4>;
        for (var i: u32 = 0u; i < 4u; i++) {
            scale_vals[i] = load_u32_at_src0(block_byte_base + 96u + 4u * i);
        }

        var tmp: u32 = scale_vals[2];
        scale_vals[2] = ((scale_vals[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        scale_vals[3] = ((scale_vals[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4u);
        scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

        // Load hmask and qs arrays
        var hmask_vals: array<u32, 8>;
        for (var i: u32 = 0u; i < 8u; i++) {
            hmask_vals[i] = load_u32_at_src0(block_byte_base + 4u * i);
        }

        var qs_vals: array<u32, 16>;
        for (var i: u32 = 0u; i < 16u; i++) {
            qs_vals[i] = load_u32_at_src0(block_byte_base + 32u + 4u * i);
        }

        let half = k_in_block / 128u;           // 0 or 1
        let pos_in_half = k_in_block % 128u;    // 0-127
        let shift_group = pos_in_half / 32u;    // 0-3
        let pos_in_32 = pos_in_half % 32u;      // 0-31
        let k_group = pos_in_32 / 16u;          // 0 or 1
        let l = pos_in_32 % 16u;                // 0-15

        let q_b_idx = half * 32u;               // 0 or 32
        let shift = shift_group * 2u;           // 0, 2, 4, 6
        let k = k_group * 16u;                  // 0 or 16
        let is = k_in_block / 16u;              // 0-15

        // m increments every 32 elements across entire 256 element block
        let m_shift = k_in_block / 32u;         // 0-7
        let m: u32 = 1u << m_shift;             // 1,2,4,8,16,32,64,128

        let sc = get_byte(scale_vals[is / 4u], is % 4u);
        let dl = d * (f16(sc) - 32.0);

        let q_idx = q_b_idx + k + l;
        let hm_idx = k + l;

        let q_byte = get_byte(qs_vals[q_idx / 4u], q_idx % 4u);
        let hmask_byte = get_byte(hmask_vals[hm_idx / 4u], hm_idx % 4u);

        let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
        let qs_val = (q_byte >> shift) & 3u;

        let q_val = (f16(qs_val) - f16(hm)) * dl;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q3_K

#ifdef INIT_SRC0_SHMEM_Q4_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 144u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);

        // Map k_in_block to loop structure:
        // Outer loop over 64-element groups (alternating q_b_idx)
        // Inner loop over 2 shifts per group
        let group_of_64 = k_in_block / 64u;  // 0-3 (maps to q_b_idx)
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;

        let q_val = f16(qs_val) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q4_K

#ifdef INIT_SRC0_SHMEM_Q5_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 176u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let d = load_f16_at_src0(block_byte_base);
        let dmin = load_f16_at_src0(block_byte_base + 2u);


        // The original loop processes elements in groups of 64
        // Each group of 64: q_b_idx cycles through [0,32,64,96], shift cycles [0,4]
        // But u increments EVERY 32 elements (after each l loop)
        let group_of_64 = k_in_block / 64u;  // 0-3
        let pos_in_64 = k_in_block % 64u;    // 0-63
        let shift_group = pos_in_64 / 32u;   // 0 or 1
        let l = pos_in_64 % 32u;             // 0-31

        let q_b_idx = group_of_64 * 32u;     // 0, 32, 64, 96
        let shift = shift_group * 4u;        // 0 or 4
        let is = k_in_block / 32u;           // 0-7

        // u increments every 32 elements (0->1, 1->2, 2->4, 3->8, 4->16, 5->32, 6->64, 7->128)
        let u_shift = k_in_block / 32u;      // 0-7
        let u: u32 = 1u << u_shift;

        var sc: u32;
        var mn: u32;

        let scale_base = block_byte_base + 4u;

        if (is < 4u) {
            let sc_byte = get_byte(load_u32_at_src0(scale_base), is % 4u);
            let min_byte = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);
            sc = sc_byte & 63u;
            mn = min_byte & 63u;
        } else {
            let sc_min_lo = get_byte(load_u32_at_src0(scale_base + 8), (is + 4u) % 4u);
            let sc_hi = get_byte(load_u32_at_src0(scale_base), (is - 4u) % 4u);
            let min_hi = get_byte(load_u32_at_src0(scale_base + 4), is % 4u);

            sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
            mn = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
        }

        let dl = d * f16(sc);
        let ml = dmin * f16(mn);

        let q_idx = q_b_idx + l;
        let q_packed = load_u32_at_src0(block_byte_base + 48u + 4u * (q_idx / 4u));

        let q_byte = get_byte(q_packed, q_idx % 4u);

        let qh_packed = load_u32_at_src0(block_byte_base + 16u + 4u * (l / 4u));

        let qh_byte = get_byte(qh_packed, l % 4u);

        let qs_val = (q_byte >> shift) & 0xFu;
        let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);

        let q_val = (f16(qs_val) + f16(qh_val)) * dl - ml;
        shmem[elem_idx] = q_val;
    }
}

#endif // INIT_SRC0_SHMEM_Q5_K

#ifdef INIT_SRC0_SHMEM_Q6_K
const BLOCK_SIZE = 256u;
const BLOCK_SIZE_BYTES = 210u;

fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
    for (var elem_idx = thread_id; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE) {
        let tile_m = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;

        let global_m = offset_m + tile_m;
        let global_k = k_outer + tile_k;

        if (global_m >= params.m || global_k >= params.k) {
            shmem[elem_idx] = f16(0.0);
            continue;
        }

        let block_k = global_k / BLOCK_SIZE;
        let k_in_block = global_k % BLOCK_SIZE;

        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;

        let half = k_in_block / 128u;
        let pos_in_half = k_in_block % 128u;
        let quarter = pos_in_half / 32u;
        let l = pos_in_half % 32u;

        let ql_b_idx = half * 64u;
        let qh_b_idx = half * 32u;
        let sc_b_idx = half * 8u;

        // Load only ql13 word needed
        let ql13_flat = ql_b_idx + l;
        let ql13 = load_u32_at_src0(block_byte_base + ql13_flat);
        let ql13_b = get_byte(ql13, 0u);

        // Load only ql24 word needed
        let ql24_flat = ql_b_idx + l + 32u;
        let ql24 = load_u32_at_src0(block_byte_base + ql24_flat);
        let ql24_b = get_byte(ql24, 0u);

        // Load only qh word needed
        let qh_flat = qh_b_idx + l;
        let qh = load_u32_at_src0(block_byte_base + 128u + qh_flat);
        let qh_b = get_byte(qh, 0u);

        let q1 = f16((ql13_b & 0xFu) | ((qh_b & 3u) << 4u)) - f16(32.0);
        let q2 = f16((ql24_b & 0xFu) | (((qh_b >> 2u) & 3u) << 4u)) - f16(32.0);
        let q3 = f16((ql13_b >> 4u) | (((qh_b >> 4u) & 3u) << 4u)) - f16(32.0);
        let q4 = f16((ql24_b >> 4u) | (((qh_b >> 6u) & 3u) << 4u)) - f16(32.0);

        // Load only the scale word needed
        let is = l / 16u;
        let sc_idx = sc_b_idx + is + quarter * 2u;
        let sc = load_u32_at_src0(block_byte_base + 192u + sc_idx);
        let sc_val = get_byte_i32(sc, 0u);

        let d = load_f16_at_src0(block_byte_base + 208u);

        var q_val: f16;
        if (quarter == 0u) {
            q_val = q1;
        } else if (quarter == 1u) {
            q_val = q2;
        } else if (quarter == 2u) {
            q_val = q3;
        } else {
            q_val = q4;
        }

        shmem[elem_idx] = d * f16(sc_val) * q_val;
    }
}
#endif // INIT_SRC0_SHMEM_Q6_K

// TODO: this shader path does not work with some models like qwen2.5 on Metal devices, f16 accumulation causes NaNs.
// See https://github.com/ggml-org/llama.cpp/issues/21602

#ifdef VEC
fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = vec4<f32>(
        f32(shmem[shmem_idx]),
        f32(shmem[shmem_idx + 1]),
        f32(shmem[shmem_idx + 2]),
        f32(shmem[shmem_idx + 3])
    );
}
#endif

#ifdef SCALAR
fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = f32(shmem[shmem_idx]);
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

// SRC0_TYPE and SRC1_TYPE are defined in mul_mat_decls, which is included
@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

const WG_M_SG_TILE_SIZE = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
const WG_N_SG_TILE_SIZE = SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

// For portability we assume the max subgroup size, meaning some subgroups will be masked out if the
// runtime subgroup size is smaller.
const EXPECTED_SUBGROUPS = SUBGROUP_M * SUBGROUP_N;
const TOTAL_WORKGROUP_SIZE = SUBGROUP_M * SUBGROUP_N * MAX_SUBGROUP_SIZE;
const TILE_SRC0_SHMEM = TILE_K * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
const TILE_SRC1_SHMEM = TILE_K * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

const SG_MAT_ACCUM_SHMEM = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_M_SIZE * SUBGROUP_MATRIX_N_SIZE;

// We reuse shmem for accumulation matrices
const SHMEM_SIZE = max(TILE_SRC0_SHMEM + TILE_SRC1_SHMEM, SG_MAT_ACCUM_SHMEM);

var<workgroup> shmem: array<f16, SHMEM_SIZE>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let subgroup_m = subgroup_id % SUBGROUP_M;
    let subgroup_n = subgroup_id / SUBGROUP_M;

    let wg_m_count = (params.m + WG_M_SG_TILE_SIZE - 1) / WG_M_SG_TILE_SIZE;
    let wg_n_count = (params.n + WG_N_SG_TILE_SIZE - 1) / WG_N_SG_TILE_SIZE;
    let wg_per_matrix = wg_m_count * wg_n_count;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    let batch_idx = wg_linear / wg_per_matrix;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    let offset_m = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
    let offset_n = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

    var acc_sg_mat : array<array<subgroup_matrix_result<f16, SUBGROUP_MATRIX_N_SIZE, SUBGROUP_MATRIX_M_SIZE>, SUBGROUP_MATRIX_N>, SUBGROUP_MATRIX_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // see mul_mat_decls.tmpl
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        if (subgroup_id < EXPECTED_SUBGROUPS) {

            for (var k_inner = 0u; k_inner < TILE_K; k_inner += SUBGROUP_MATRIX_K_SIZE) {

                let src0_shmem_idx_base = subgroup_m * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE * TILE_K + k_inner;
                var src0_sg_mats: array<subgroup_matrix_left<f16, SUBGROUP_MATRIX_K_SIZE, SUBGROUP_MATRIX_M_SIZE>, SUBGROUP_MATRIX_M>;
                for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                    src0_sg_mats[m] = subgroupMatrixLoad<subgroup_matrix_left<f16, SUBGROUP_MATRIX_K_SIZE, SUBGROUP_MATRIX_M_SIZE>>(
                        &shmem,
                        src0_shmem_idx_base + m * SUBGROUP_MATRIX_M_SIZE * TILE_K,
                        false,
                        TILE_K
                    );
                }

                let src1_shmem_idx_base = TILE_SRC0_SHMEM + subgroup_n * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE * TILE_K + k_inner;
                for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
                    let src1_sg_mat = subgroupMatrixLoad<subgroup_matrix_right<f16, SUBGROUP_MATRIX_N_SIZE, SUBGROUP_MATRIX_K_SIZE>>(
                        &shmem,
                        src1_shmem_idx_base + n * SUBGROUP_MATRIX_N_SIZE * TILE_K,
                        true,
                        TILE_K
                    );
                    for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                        acc_sg_mat[m][n] = subgroupMatrixMultiplyAccumulate(src0_sg_mats[m], src1_sg_mat, acc_sg_mat[m][n]);
                    }
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    // Stage the subgroup matrix tiles into shared memory
    // This uses WG_M_SG_TILE_SIZE as the stride (number of columns in the workgroup tile).
    let WG_TILE_STRIDE = WG_M_SG_TILE_SIZE;
    let tile_row_base_local = subgroup_n * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;
    let tile_col_base_local = subgroup_m * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;

    if (subgroup_id < EXPECTED_SUBGROUPS) { // 2-5% performance hit :(
        for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
            for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                let local_row = tile_row_base_local + n * SUBGROUP_MATRIX_N_SIZE;
                let local_col = tile_col_base_local + m * SUBGROUP_MATRIX_M_SIZE;
                let out_base = local_row * WG_TILE_STRIDE + local_col;
                subgroupMatrixStore(&shmem, out_base, acc_sg_mat[m][n], true, WG_TILE_STRIDE);
            }
        }
    }

    workgroupBarrier();

    // Cooperative write: iterate over the entire workgroup tile
    let tile_rows = WG_N_SG_TILE_SIZE;
    let tile_cols = WG_M_SG_TILE_SIZE;
    let total_tile_elems = tile_rows * tile_cols;
    let tile_dst_row_base = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
    let tile_dst_col_base = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

    for (var idx = thread_id * VEC_SIZE; idx < total_tile_elems; idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let local_row = idx % WG_TILE_STRIDE;
        let local_col = idx / WG_TILE_STRIDE;

        let global_row = tile_dst_row_base + local_row;
        let global_col = tile_dst_col_base + local_col;

        if (global_col < params.n && global_row < params.m) {
            let dst_idx = dst_batch_offset + global_col * params.m + global_row;
            store_dst(idx, dst_idx/VEC_SIZE);
        }
    }
}
)wgsl";

static const std::string& wgsl_mul_mat_subgroup_matrix_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(108131);
        tmp.append(wgsl_mul_mat_subgroup_matrix_part0);
        tmp.append(wgsl_mul_mat_subgroup_matrix_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_mul_mat_subgroup_matrix = wgsl_mul_mat_subgroup_matrix_str().c_str();

static const char wgsl_mul_mat_vec_part0[] = R"wgsl(#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif
enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#ifdef BYTE_HELPERS
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}
#endif

#ifdef U32_DEQUANT_HELPERS
#ifdef DECLARE_BYTE_LOADERS_SRC
fn load_u16_at_src(byte_offset: u32) -> u32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

fn load_u32_at_src(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src[word_idx];
    let hi = src[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
    let word = src[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif

#ifdef DECLARE_BYTE_LOADERS_SRC0
fn load_u16_at_src0(byte_offset: u32) -> u32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

// Always reads the 4-byte-aligned word containing byte_offset.
// Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
// this is used in k-quants for better performance
fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
    return src0[(byte_offset & ~3u) / 4u];
}

fn load_u32_at_src0(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 0x3u) * 8u;
    let lo = src0[word_idx];
    let hi = src0[word_idx + 1u];
    let shifted = (lo >> shift) | (hi << (32u - shift));
    return select(shifted, lo, shift == 0u);
}

fn load_f16_at_src0(byte_offset: u32) -> f16 {
    let packed = unpack2x16float(load_u16_at_src0(byte_offset));
    return f16(packed[0]);
}

fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
    let word = src0[byte_offset / 4u];
    let shift = (byte_offset & 0x2u) * 8u;
    let d_bits = (word >> shift) & 0xFFFFu;
    return unpack2x16float(d_bits)[0];
}
#endif
#endif



#ifdef Q4_1_T
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};
#endif


#ifdef Q5_1_T
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};
#endif


#ifdef Q8_1_T
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};
#endif

#ifdef Q2_K_T
struct q2_K {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};
#endif


#if defined(Q4_K_SCALE_MIN) || defined(Q5_K_SCALE_MIN)
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}
#endif
#ifdef Q4_K_T
struct q4_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};
#endif

#ifdef Q5_K_T
struct q5_K {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};
#endif

#ifdef IQ1_M_T
struct iq1_m {
    qs: array<u32, 8>,
    qh: array<u32, 4>,
    scales: array<u32, 2>
};
#endif

#ifdef IQ4_XS_T
struct iq4_xs {
    d_scales_h: u32,
    scales_l: u32,
    qs: array<u32, 32>
};
#endif

#if defined(IQ2_XXS_TABLES) || defined(IQ2_XS_TABLES) || defined(IQ2_S_TABLES) || defined(IQ3_XXS_TABLES) || defined(IQ3_S_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);
#endif

#ifdef IQ2_XXS_GRID
const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);
#endif

#ifdef IQ2_XS_GRID
const iq2xs_grid = array<u32, 1024>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x2b080808, 0x08080808,
    0x2b08082b, 0x08080808, 0x2b081919, 0x08080808, 0x2b082b08, 0x08080808, 0x2b190819, 0x08080808,
    0x2b191908, 0x08080808, 0x2b192b19, 0x08080808, 0x2b2b0808, 0x08080808, 0x08080819, 0x08080819,
    0x08081908, 0x08080819, 0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819,
    0x0819082b, 0x08080819, 0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x08192b2b, 0x08080819,
    0x082b0819, 0x08080819, 0x082b1908, 0x08080819, 0x19080808, 0x08080819, 0x1908082b, 0x08080819,
    0x19081919, 0x08080819, 0x19082b08, 0x08080819, 0x19190819, 0x08080819, 0x19191908, 0x08080819,
    0x192b0808, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819, 0x2b081908, 0x08080819,
    0x2b190808, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x08081919, 0x0808082b,
    0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b, 0x082b0808, 0x0808082b,
    0x19080819, 0x0808082b, 0x19081908, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908,
    0x0808192b, 0x08081908, 0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908,
    0x08191919, 0x08081908, 0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908,
    0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908, 0x19082b08, 0x08081908,
    0x19190819, 0x08081908, 0x19191908, 0x08081908, 0x1919192b, 0x08081908, 0x192b0808, 0x08081908,
    0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908, 0x08080808, 0x08081919,
    0x0808082b, 0x08081919, 0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x082b0808, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x19190808, 0x08081919, 0x192b0819, 0x08081919, 0x2b080808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x08190808, 0x0808192b, 0x082b192b, 0x0808192b, 0x19080808, 0x0808192b,
    0x1908082b, 0x0808192b, 0x2b081908, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08082b2b, 0x08082b08, 0x08190819, 0x08082b08,
    0x08191908, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x19192b08, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b2b0808, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19, 0x08081908, 0x08082b19,
    0x08190808, 0x08082b19, 0x19080808, 0x08082b19, 0x2b080819, 0x08082b19, 0x2b082b19, 0x08082b19,
    0x08080808, 0x08082b2b, 0x082b0808, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x2b19192b, 0x08082b2b,
    0x2b2b0808, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x0808192b, 0x08190808,
    0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808, 0x08191919, 0x08190808,
    0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808, 0x19080808, 0x08190808,
    0x1908082b, 0x08190808, 0x19081919, 0x08190808, 0x19082b08, 0x08190808, 0x19190819, 0x08190808,
    0x19191908, 0x08190808, 0x192b0808, 0x08190808, 0x192b2b2b, 0x08190808, 0x2b080819, 0x08190808,
    0x2b081908, 0x08190808, 0x2b190808, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819,
    0x08081919, 0x08190819, 0x08082b08, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x082b0808, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819, 0x19190808, 0x08190819,
    0x2b080808, 0x08190819, 0x2b191908, 0x08190819, 0x2b19192b, 0x08190819, 0x08080819, 0x0819082b,
    0x08081908, 0x0819082b, 0x0808192b, 0x0819082b, 0x08190808, 0x0819082b, 0x19080808, 0x0819082b,
    0x192b0808, 0x0819082b, 0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908,
    0x08082b08, 0x08191908, 0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x082b0808, 0x08191908,
    0x19080819, 0x08191908, 0x19081908, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x08080819, 0x08191919, 0x08081908, 0x08191919,
    0x08190808, 0x08191919, 0x19080808, 0x08191919, 0x08080808, 0x0819192b, 0x08191908, 0x0819192b,
    0x19082b19, 0x0819192b, 0x08080819, 0x08192b08, 0x08081908, 0x08192b08, 0x08190808, 0x08192b08,
    0x0819082b, 0x08192b08, 0x19080808, 0x08192b08, 0x19191908, 0x08192b08, 0x2b08192b, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x192b192b, 0x08192b19, 0x19190819, 0x08192b2b,
    0x2b2b2b19, 0x08192b2b, 0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808,
    0x08082b08, 0x082b0808, 0x08082b2b, 0x082b0808, 0x08190819, 0x082b0808, 0x08191908, 0x082b0808,
    0x082b0808, 0x082b0808, 0x19080819, 0x082b0808, 0x19081908, 0x082b0808, 0x19190808, 0x082b0808,
    0x2b080808, 0x082b0808, 0x2b2b0808, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819,
    0x08190808, 0x082b0819, 0x19080808, 0x082b0819, 0x19082b08, 0x082b0819, 0x192b1919, 0x082b0819,
    0x08080808, 0x082b082b, 0x082b082b, 0x082b082b, 0x2b080808, 0x082b082b, 0x2b2b2b08, 0x082b082b,
    0x08080819, 0x082b1908, 0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x082b2b19, 0x082b1908,
    0x19080808, 0x082b1908, 0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x1919082b, 0x082b1919,
    0x2b192b19, 0x082b1919, 0x08080819, 0x082b192b, 0x08192b2b, 0x082b192b, 0x2b2b192b, 0x082b192b,
    0x08080808, 0x082b2b08, 0x08082b08, 0x082b2b08, 0x08082b2b, 0x082b2b08, 0x082b0808, 0x082b2b08,
    0x19191919, 0x082b2b08, 0x2b082b08, 0x082b2b08, 0x2b2b082b, 0x082b2b08, 0x192b2b08, 0x082b2b19,
    0x2b190808, 0x082b2b19, 0x08082b08, 0x082b2b2b, 0x082b0808, 0x082b2b2b, 0x2b08082b, 0x082b2b2b,
    0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808, 0x08081908, 0x19080808,
    0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808, 0x0819082b, 0x19080808,
    0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x082b0819, 0x19080808, 0x082b1908, 0x19080808,
    0x19080808, 0x19080808, 0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808,
    0x19082b2b, 0x19080808, 0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x192b0808, 0x19080808,
    0x192b1919, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808, 0x2b190808, 0x19080808,
    0x08080808, 0x19080819, 0x0808082b, 0x19080819, 0x08081919, 0x19080819, 0x08082b08, 0x19080819,
    0x08190819, 0x19080819, 0x08191908, 0x19080819, 0x082b0808, 0x19080819, 0x19080819, 0x19080819,
    0x19081908, 0x19080819, 0x19190808, 0x19080819, 0x2b080808, 0x19080819, 0x2b081919, 0x19080819,
    0x2b2b082b, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b, 0x08190808, 0x1908082b,
    0x0819082b, 0x1908082b, 0x082b2b19, 0x1908082b, 0x19080808, 0x1908082b, 0x08080808, 0x19081908,
    0x0808082b, 0x19081908, 0x08081919, 0x19081908, 0x08082b08, 0x19081908, 0x08190819, 0x19081908,
    0x08191908, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x19080819, 0x19081908,
    0x19081908, 0x19081908, 0x19190808, 0x19081908, 0x2b080808, 0x19081908, 0x2b191908, 0x19081908,
    0x08080819, 0x19081919, 0x08081908, 0x19081919, 0x08190808, 0x19081919, 0x082b1908, 0x19081919,
    0x19080808, 0x19081919, 0x2b192b2b, 0x19081919, 0x08080808, 0x1908192b, 0x08082b2b, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x08080819, 0x19082b08, 0x08081908, 0x19082b08,
    0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08, 0x19191908, 0x19082b08,
    0x192b082b, 0x19082b08, 0x08080808, 0x19082b19, 0x08190819, 0x19082b19, 0x19081908, 0x19082b19,
    0x19190808, 0x19082b19, 0x192b2b19, 0x19082b19, 0x08081908, 0x19082b2b, 0x08080808, 0x19190808,
    0x0808082b, 0x19190808, 0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808,
    0x08191908, 0x19190808, 0x082b0808, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808,
    0x19081908, 0x19190808, 0x19190808, 0x19190808, 0x2b080808, 0x19190808, 0x08080819, 0x19190819,
    0x08081908, 0x19190819, 0x08190808, 0x19190819, 0x08191919, 0x19190819, 0x19080808, 0x19190819,
    0x1908082b, 0x19190819, 0x08080808, 0x1919082b, 0x19081908, 0x1919082b, 0x2b2b2b2b, 0x1919082b,
    0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x08190808, 0x19191908, 0x082b0819, 0x19191908,
    0x19080808, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b2b0819, 0x19191908,
    0x08080808, 0x19191919, 0x08082b08, 0x19191919, 0x2b080808, 0x19191919, 0x2b082b08, 0x19191919,
    0x082b0819, 0x1919192b, 0x192b2b08, 0x1919192b, 0x2b2b0819, 0x1919192b, 0x08080808, 0x19192b08,
    0x08191908, 0x19192b08, 0x19080819, 0x19192b08, 0x19190808, 0x19192b08, 0x2b192b19, 0x19192b08,
    0x08192b2b, 0x19192b19, 0x19080808, 0x19192b19, 0x1908082b, 0x19192b19, 0x2b081919, 0x19192b2b,
    0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808,
    0x19191908, 0x192b0808, 0x192b082b, 0x192b0808, 0x2b08192b, 0x192b0808, 0x2b2b2b19, 0x192b0808,
    0x08080808, 0x192b0819, 0x082b1908, 0x192b082b, 0x19082b2b, 0x192b082b, 0x2b19082b, 0x192b082b,
    0x08080808, 0x192b1908, 0x0819192b, 0x192b1908, 0x08190808, 0x192b1919, 0x19080808, 0x192b1919,
    0x19081919, 0x192b1919, 0x2b2b1908, 0x192b1919, 0x08080819, 0x192b2b08, 0x192b2b2b, 0x192b2b08,
    0x082b1919, 0x192b2b19, 0x0808192b, 0x192b2b2b, 0x19191908, 0x192b2b2b, 0x192b082b, 0x192b2b2b,
    0x08080808, 0x2b080808, 0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808,
    0x08190819, 0x2b080808, 0x08191908, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b2b2b, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b08082b, 0x2b080808, 0x2b2b2b08, 0x2b080808, 0x2b2b2b2b, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x0808192b, 0x2b080819, 0x08190808, 0x2b080819, 0x19080808, 0x2b080819,
    0x19190819, 0x2b080819, 0x19192b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x082b0808, 0x2b08082b,
    0x2b080808, 0x2b08082b, 0x2b08082b, 0x2b08082b, 0x2b2b0808, 0x2b08082b, 0x2b2b2b08, 0x2b08082b,
    0x08080819, 0x2b081908, 0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x19080808, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b082b19, 0x2b081908,
    0x08080808, 0x2b081919, 0x19081908, 0x2b081919, 0x2b2b1919, 0x2b081919, 0x08192b08, 0x2b08192b,
    0x192b2b2b, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08082b08, 0x2b082b08, 0x082b1919, 0x2b082b08,
    0x19192b2b, 0x2b082b08, 0x2b080808, 0x2b082b08, 0x2b08082b, 0x2b082b08, 0x2b2b2b08, 0x2b082b08,
    0x0808192b, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x2b080808, 0x2b082b2b, 0x2b082b08, 0x2b082b2b,
    0x2b19192b, 0x2b082b2b, 0x2b2b2b08, 0x2b082b2b, 0x08080819, 0x2b190808, 0x08081908, 0x2b190808,
    0x08190808, 0x2b190808, 0x19080808, 0x2b190808, 0x1919192b, 0x2b190808, 0x2b081908, 0x2b190808,
    0x08080808, 0x2b190819, 0x082b082b, 0x2b190819, 0x192b1908, 0x2b190819, 0x1919192b, 0x2b19082b,
    0x2b082b19, 0x2b19082b, 0x08080808, 0x2b191908, 0x08081919, 0x2b191908, 0x19081908, 0x2b191908,
    0x19190808, 0x2b191908, 0x19192b08, 0x2b191908, 0x082b2b19, 0x2b191919, 0x2b190808, 0x2b191919,
    0x2b19082b, 0x2b191919, 0x19080819, 0x2b19192b, 0x19190819, 0x2b192b08, 0x2b2b192b, 0x2b192b08,
    0x19082b19, 0x2b192b19, 0x08191919, 0x2b192b2b, 0x192b0808, 0x2b192b2b, 0x08080808, 0x2b2b0808,
    0x0808082b, 0x2b2b0808, 0x08082b08, 0x2b2b0808, 0x08082b2b, 0x2b2b0808, 0x082b0808, 0x2b2b0808,
    0x082b2b2b, 0x2b2b0808, 0x2b2b0808, 0x2b2b0808, 0x19190819, 0x2b2b0819, 0x19192b19, 0x2b2b0819,
    0x2b2b192b, 0x2b2b0819, 0x08080808, 0x2b2b082b, 0x0808082b, 0x2b2b082b, 0x08082b08, 0x2b2b082b,
    0x082b2b2b, 0x2b2b082b, 0x2b080808, 0x2b2b082b, 0x2b2b0808, 0x2b2b082b, 0x19080808, 0x2b2b1908,
    0x2b191919, 0x2b2b1908, 0x192b1919, 0x2b2b192b, 0x2b192b08, 0x2b2b192b, 0x08082b2b, 0x2b2b2b08,
    0x082b0808, 0x2b2b2b08, 0x082b082b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b0808, 0x2b2b2b08,
    0x2b2b2b08, 0x2b2b2b08, 0x08081908, 0x2b2b2b19, 0x2b081908, 0x2b2b2b19, 0x2b08192b, 0x2b2b2b19,
    0x082b2b08, 0x2b2b2b2b, 0x082b2b2b, 0x2b2b2b2b, 0x2b190819, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ2_S_GRID
const iq2s_grid = array<u32, 2048>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x0819192b, 0x08080808,
    0x08192b19, 0x08080808, 0x082b0808, 0x08080808, 0x082b082b, 0x08080808, 0x082b1919, 0x08080808,
    0x082b2b08, 0x08080808, 0x19080819, 0x08080808, 0x19081908, 0x08080808, 0x1908192b, 0x08080808,
    0x19082b19, 0x08080808, 0x19190808, 0x08080808, 0x1919082b, 0x08080808, 0x19191919, 0x08080808,
    0x19192b08, 0x08080808, 0x192b0819, 0x08080808, 0x192b1908, 0x08080808, 0x192b192b, 0x08080808,
    0x192b2b19, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b081919, 0x08080808,
    0x2b082b08, 0x08080808, 0x2b190819, 0x08080808, 0x2b191908, 0x08080808, 0x2b2b0808, 0x08080808,
    0x2b2b1919, 0x08080808, 0x2b2b2b2b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819,
    0x0808192b, 0x08080819, 0x08082b19, 0x08080819, 0x08190808, 0x08080819, 0x0819082b, 0x08080819,
    0x08191919, 0x08080819, 0x08192b08, 0x08080819, 0x082b0819, 0x08080819, 0x082b1908, 0x08080819,
    0x19080808, 0x08080819, 0x1908082b, 0x08080819, 0x19081919, 0x08080819, 0x19082b08, 0x08080819,
    0x19190819, 0x08080819, 0x19191908, 0x08080819, 0x1919192b, 0x08080819, 0x19192b19, 0x08080819,
    0x192b0808, 0x08080819, 0x192b1919, 0x08080819, 0x192b2b08, 0x08080819, 0x2b080819, 0x08080819,
    0x2b081908, 0x08080819, 0x2b190808, 0x08080819, 0x2b19082b, 0x08080819, 0x2b191919, 0x08080819,
    0x2b2b0819, 0x08080819, 0x2b2b1908, 0x08080819, 0x08080808, 0x0808082b, 0x0808082b, 0x0808082b,
    0x08081919, 0x0808082b, 0x08082b08, 0x0808082b, 0x08190819, 0x0808082b, 0x08191908, 0x0808082b,
    0x082b0808, 0x0808082b, 0x082b2b2b, 0x0808082b, 0x19080819, 0x0808082b, 0x19081908, 0x0808082b,
    0x1908192b, 0x0808082b, 0x19082b19, 0x0808082b, 0x19190808, 0x0808082b, 0x19191919, 0x0808082b,
    0x2b080808, 0x0808082b, 0x2b081919, 0x0808082b, 0x2b082b2b, 0x0808082b, 0x2b191908, 0x0808082b,
    0x2b2b082b, 0x0808082b, 0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x0808192b, 0x08081908,
    0x08082b19, 0x08081908, 0x08190808, 0x08081908, 0x0819082b, 0x08081908, 0x08191919, 0x08081908,
    0x08192b08, 0x08081908, 0x082b0819, 0x08081908, 0x082b1908, 0x08081908, 0x082b192b, 0x08081908,
    0x082b2b19, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19081919, 0x08081908,
    0x19082b08, 0x08081908, 0x19082b2b, 0x08081908, 0x19190819, 0x08081908, 0x19191908, 0x08081908,
    0x1919192b, 0x08081908, 0x19192b19, 0x08081908, 0x192b0808, 0x08081908, 0x192b082b, 0x08081908,
    0x192b1919, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b08192b, 0x08081908,
    0x2b082b19, 0x08081908, 0x2b190808, 0x08081908, 0x2b191919, 0x08081908, 0x2b192b08, 0x08081908,
    0x2b2b0819, 0x08081908, 0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919,
    0x08081919, 0x08081919, 0x08082b08, 0x08081919, 0x08082b2b, 0x08081919, 0x08190819, 0x08081919,
    0x08191908, 0x08081919, 0x0819192b, 0x08081919, 0x08192b19, 0x08081919, 0x082b0808, 0x08081919,
    0x082b1919, 0x08081919, 0x082b2b08, 0x08081919, 0x19080819, 0x08081919, 0x19081908, 0x08081919,
    0x1908192b, 0x08081919, 0x19082b19, 0x08081919, 0x19190808, 0x08081919, 0x1919082b, 0x08081919,
    0x19191919, 0x08081919, 0x19192b08, 0x08081919, 0x192b0819, 0x08081919, 0x192b1908, 0x08081919,
    0x2b080808, 0x08081919, 0x2b08082b, 0x08081919, 0x2b081919, 0x08081919, 0x2b082b08, 0x08081919,
    0x2b190819, 0x08081919, 0x2b191908, 0x08081919, 0x2b2b0808, 0x08081919, 0x08080819, 0x0808192b,
    0x08081908, 0x0808192b, 0x0808192b, 0x0808192b, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b,
    0x08191919, 0x0808192b, 0x19080808, 0x0808192b, 0x19081919, 0x0808192b, 0x19082b08, 0x0808192b,
    0x19190819, 0x0808192b, 0x19191908, 0x0808192b, 0x192b0808, 0x0808192b, 0x2b080819, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b190808, 0x0808192b, 0x08080808, 0x08082b08, 0x0808082b, 0x08082b08,
    0x08081919, 0x08082b08, 0x08082b08, 0x08082b08, 0x08190819, 0x08082b08, 0x08191908, 0x08082b08,
    0x0819192b, 0x08082b08, 0x08192b19, 0x08082b08, 0x082b0808, 0x08082b08, 0x082b1919, 0x08082b08,
    0x082b2b2b, 0x08082b08, 0x19080819, 0x08082b08, 0x19081908, 0x08082b08, 0x1908192b, 0x08082b08,
    0x19082b19, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x19191919, 0x08082b08,
    0x19192b08, 0x08082b08, 0x192b0819, 0x08082b08, 0x192b1908, 0x08082b08, 0x2b080808, 0x08082b08,
    0x2b081919, 0x08082b08, 0x2b191908, 0x08082b08, 0x2b2b2b2b, 0x08082b08, 0x08080819, 0x08082b19,
    0x08081908, 0x08082b19, 0x08190808, 0x08082b19, 0x0819082b, 0x08082b19, 0x08191919, 0x08082b19,
    0x08192b08, 0x08082b19, 0x082b0819, 0x08082b19, 0x19080808, 0x08082b19, 0x19081919, 0x08082b19,
    0x19082b08, 0x08082b19, 0x19190819, 0x08082b19, 0x19191908, 0x08082b19, 0x192b0808, 0x08082b19,
    0x2b080819, 0x08082b19, 0x2b190808, 0x08082b19, 0x08080808, 0x08082b2b, 0x08190819, 0x08082b2b,
    0x08191908, 0x08082b2b, 0x082b082b, 0x08082b2b, 0x082b2b08, 0x08082b2b, 0x082b2b2b, 0x08082b2b,
    0x19190808, 0x08082b2b, 0x2b192b19, 0x08082b2b, 0x08080819, 0x08190808, 0x08081908, 0x08190808,
    0x0808192b, 0x08190808, 0x08082b19, 0x08190808, 0x08190808, 0x08190808, 0x0819082b, 0x08190808,
    0x08191919, 0x08190808, 0x08192b08, 0x08190808, 0x082b0819, 0x08190808, 0x082b1908, 0x08190808,
    0x082b192b, 0x08190808, 0x19080808, 0x08190808, 0x1908082b, 0x08190808, 0x19081919, 0x08190808,
    0x19082b08, 0x08190808, 0x19190819, 0x08190808, 0x19191908, 0x08190808, 0x1919192b, 0x08190808,
    0x19192b19, 0x08190808, 0x192b0808, 0x08190808, 0x192b082b, 0x08190808, 0x192b1919, 0x08190808,
    0x192b2b08, 0x08190808, 0x2b080819, 0x08190808, 0x2b081908, 0x08190808, 0x2b08192b, 0x08190808,
    0x2b190808, 0x08190808, 0x2b191919, 0x08190808, 0x2b192b08, 0x08190808, 0x2b2b0819, 0x08190808,
    0x2b2b1908, 0x08190808, 0x08080808, 0x08190819, 0x0808082b, 0x08190819, 0x08081919, 0x08190819,
    0x08082b08, 0x08190819, 0x08082b2b, 0x08190819, 0x08190819, 0x08190819, 0x08191908, 0x08190819,
    0x0819192b, 0x08190819, 0x08192b19, 0x08190819, 0x082b0808, 0x08190819, 0x082b082b, 0x08190819,
    0x082b1919, 0x08190819, 0x082b2b08, 0x08190819, 0x19080819, 0x08190819, 0x19081908, 0x08190819,
    0x1908192b, 0x08190819, 0x19082b19, 0x08190819, 0x19190808, 0x08190819, 0x1919082b, 0x08190819,
    0x19191919, 0x08190819, 0x19192b08, 0x08190819, 0x192b0819, 0x08190819, 0x192b1908, 0x08190819,
    0x2b080808, 0x08190819, 0x2b08082b, 0x08190819, 0x2b081919, 0x08190819, 0x2b082b08, 0x08190819,
    0x2b190819, 0x08190819, 0x2b191908, 0x08190819, 0x08080819, 0x0819082b, 0x08081908, 0x0819082b,
    0x08082b19, 0x0819082b, 0x08190808, 0x0819082b, 0x08191919, 0x0819082b, 0x082b0819, 0x0819082b,
    0x082b1908, 0x0819082b, 0x19080808, 0x0819082b, 0x19081919, 0x0819082b, 0x19190819, 0x0819082b,
    0x19191908, 0x0819082b, 0x2b080819, 0x0819082b, 0x2b081908, 0x0819082b, 0x2b190808, 0x0819082b,
    0x08080808, 0x08191908, 0x0808082b, 0x08191908, 0x08081919, 0x08191908, 0x08082b08, 0x08191908,
    0x08190819, 0x08191908, 0x08191908, 0x08191908, 0x0819192b, 0x08191908, 0x08192b19, 0x08191908,
    0x082b0808, 0x08191908, 0x082b1919, 0x08191908, 0x082b2b08, 0x08191908, 0x19080819, 0x08191908,
    0x19081908, 0x08191908, 0x1908192b, 0x08191908, 0x19082b19, 0x08191908, 0x19190808, 0x08191908,
    0x1919082b, 0x08191908, 0x19191919, 0x08191908, 0x19192b08, 0x08191908, 0x192b0819, 0x08191908,
    0x192b1908, 0x08191908, 0x2b080808, 0x08191908, 0x2b08082b, 0x08191908, 0x2b081919, 0x08191908,
    0x2b082b08, 0x08191908, 0x2b190819, 0x08191908, 0x2b191908, 0x08191908, 0x2b2b0808, 0x08191908,
    0x08080819, 0x08191919, 0x08081908, 0x08191919, 0x0808192b, 0x08191919, 0x08082b19, 0x08191919,
    0x08190808, 0x08191919, 0x0819082b, 0x08191919, 0x08191919, 0x08191919, 0x08192b08, 0x08191919,
    0x082b0819, 0x08191919, 0x082b1908, 0x08191919, 0x19080808, 0x08191919, 0x1908082b, 0x08191919,
    0x19081919, 0x08191919, 0x19082b08, 0x08191919, 0x19190819, 0x08191919, 0x19191908, 0x08191919,
    0x192b0808, 0x08191919, 0x2b080819, 0x08191919, 0x2b081908, 0x08191919, 0x2b190808, 0x08191919,
    0x08080808, 0x0819192b, 0x08081919, 0x0819192b, 0x08082b08, 0x0819192b, 0x08190819, 0x0819192b,
    0x08191908, 0x0819192b, 0x082b0808, 0x0819192b, 0x19080819, 0x0819192b, 0x19081908, 0x0819192b,
    0x19190808, 0x0819192b, 0x2b080808, 0x0819192b, 0x2b2b2b2b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x0808192b, 0x08192b08, 0x08082b19, 0x08192b08, 0x08190808, 0x08192b08,
    0x08191919, 0x08192b08, 0x08192b08, 0x08192b08, 0x082b0819, 0x08192b08, 0x19080808, 0x08192b08,
    0x1908082b, 0x08192b08, 0x19081919, 0x08192b08, 0x19082b08, 0x08192b08, 0x19190819, 0x08192b08,
    0x19191908, 0x08192b08, 0x192b0808, 0x08192b08, 0x2b080819, 0x08192b08, 0x2b081908, 0x08192b08,
    0x08080808, 0x08192b19, 0x0808082b, 0x08192b19, 0x08081919, 0x08192b19, 0x08082b08, 0x08192b19,
    0x08190819, 0x08192b19, 0x08191908, 0x08192b19, 0x082b0808, 0x08192b19, 0x19080819, 0x08192b19,
    0x19081908, 0x08192b19, 0x19190808, 0x08192b19, 0x192b2b19, 0x08192b19, 0x2b2b082b, 0x08192b19,
    0x08081908, 0x08192b2b, 0x08190808, 0x08192b2b, 0x19080808, 0x08192b2b, 0x1919192b, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08081919, 0x082b0808, 0x08082b08, 0x082b0808,
    0x08190819, 0x082b0808, 0x08191908, 0x082b0808, 0x0819192b, 0x082b0808, 0x08192b19, 0x082b0808,
    0x082b0808, 0x082b0808, 0x082b1919, 0x082b0808, 0x082b2b2b, 0x082b0808, 0x19080819, 0x082b0808,
    0x19081908, 0x082b0808, 0x19190808, 0x082b0808, 0x1919082b, 0x082b0808, 0x19191919, 0x082b0808,
    0x192b1908, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b082b2b, 0x082b0808, 0x2b191908, 0x082b0808,
    0x2b2b2b2b, 0x082b0808, 0x08080819, 0x082b0819, 0x08081908, 0x082b0819, 0x08190808, 0x082b0819,
    0x0819082b, 0x082b0819, 0x08191919, 0x082b0819, 0x082b0819, 0x082b0819, 0x19080808, 0x082b0819,
    0x1908082b, 0x082b0819, 0x19081919, 0x082b0819, 0x19190819, 0x082b0819, 0x19191908, 0x082b0819,
    0x192b0808, 0x082b0819, 0x2b080819, 0x082b0819, 0x2b081908, 0x082b0819, 0x2b190808, 0x082b0819,
    0x08080808, 0x082b082b, 0x08082b2b, 0x082b082b, 0x082b082b, 0x082b082b, 0x082b2b08, 0x082b082b,
    0x082b2b2b, 0x082b082b, 0x19081908, 0x082b082b, 0x19190808, 0x082b082b, 0x2b082b08, 0x082b082b,
    0x2b082b2b, 0x082b082b, 0x2b2b2b08, 0x082b082b, 0x08080819, 0x082b1908, 0x08081908, 0x082b1908,
    0x0808192b, 0x082b1908, 0x08082b19, 0x082b1908, 0x08190808, 0x082b1908, 0x08191919, 0x082b1908,
    0x08192b08, 0x082b1908, 0x082b0819, 0x082b1908, 0x082b1908, 0x082b1908, 0x19080808, 0x082b1908,
    0x1908082b, 0x082b1908, 0x19081919, 0x082b1908, 0x19082b08, 0x082b1908, 0x19190819, 0x082b1908,
    0x19191908, 0x082b1908, 0x192b0808, 0x082b1908, 0x2b080819, 0x082b1908, 0x2b081908, 0x082b1908,
    0x2b190808, 0x082b1908, 0x08080808, 0x082b1919, 0x08081919, 0x082b1919, 0x08082b08, 0x082b1919,
    0x08190819, 0x082b1919, 0x08191908, 0x082b1919, 0x082b0808, 0x082b1919, 0x19080819, 0x082b1919,
    0x19081908, 0x082b1919, 0x19190808, 0x082b1919, 0x192b192b, 0x082b1919, 0x2b080808, 0x082b1919,
    0x08080819, 0x082b192b, 0x08081908, 0x082b192b, 0x08190808, 0x082b192b, 0x19080808, 0x082b192b,
    0x19192b19, 0x082b192b, 0x08080808, 0x082b2b08, 0x08081919, 0x082b2b08, 0x08190819, 0x082b2b08,
    0x08191908, 0x082b2b08, 0x19080819, 0x082b2b08, 0x19081908, 0x082b2b08, 0x19190808, 0x082b2b08,
    0x2b082b2b, 0x082b2b08, 0x2b2b2b2b, 0x082b2b08, 0x08080819, 0x082b2b19, 0x08081908, 0x082b2b19,
    0x08190808, 0x082b2b19, 0x2b191919, 0x082b2b19, 0x08082b2b, 0x082b2b2b, 0x082b082b, 0x082b2b2b,
    0x192b1908, 0x082b2b2b, 0x2b082b08, 0x082b2b2b, 0x2b082b2b, 0x082b2b2b, 0x08080819, 0x19080808,
    0x08081908, 0x19080808, 0x0808192b, 0x19080808, 0x08082b19, 0x19080808, 0x08190808, 0x19080808,
    0x0819082b, 0x19080808, 0x08191919, 0x19080808, 0x08192b08, 0x19080808, 0x08192b2b, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x082b192b, 0x19080808, 0x19080808, 0x19080808,
    0x1908082b, 0x19080808, 0x19081919, 0x19080808, 0x19082b08, 0x19080808, 0x19082b2b, 0x19080808,
    0x19190819, 0x19080808, 0x19191908, 0x19080808, 0x1919192b, 0x19080808, 0x19192b19, 0x19080808,
    0x192b0808, 0x19080808, 0x192b082b, 0x19080808, 0x192b1919, 0x19080808, 0x2b080819, 0x19080808,
    0x2b081908, 0x19080808, 0x2b190808, 0x19080808, 0x2b191919, 0x19080808, 0x2b192b08, 0x19080808,
    0x2b2b0819, 0x19080808, 0x2b2b1908, 0x19080808, 0x08080808, 0x19080819, 0x0808082b, 0x19080819,
    0x08081919, 0x19080819, 0x08082b08, 0x19080819, 0x08190819, 0x19080819, 0x08191908, 0x19080819,
    0x0819192b, 0x19080819, 0x08192b19, 0x19080819, 0x082b0808, 0x19080819, 0x082b082b, 0x19080819,
    0x082b1919, 0x19080819, 0x19080819, 0x19080819, 0x19081908, 0x19080819, 0x1908192b, 0x19080819,
    0x19082b19, 0x19080819, 0x19190808, 0x19080819, 0x1919082b, 0x19080819, 0x19191919, 0x19080819,
    0x19192b08, 0x19080819, 0x192b0819, 0x19080819, 0x192b1908, 0x19080819, 0x2b080808, 0x19080819,
    0x2b08082b, 0x19080819, 0x2b081919, 0x19080819, 0x2b082b08, 0x19080819, 0x2b190819, 0x19080819,
    0x2b191908, 0x19080819, 0x2b2b0808, 0x19080819, 0x08080819, 0x1908082b, 0x08081908, 0x1908082b,
    0x08190808, 0x1908082b, 0x0819082b, 0x1908082b, 0x08191919, 0x1908082b, 0x08192b08, 0x1908082b,
    0x082b1908, 0x1908082b, 0x19080808, 0x1908082b, 0x19081919, 0x1908082b, 0x19082b08, 0x1908082b,
    0x19190819, 0x1908082b, 0x19191908, 0x1908082b, 0x192b0808, 0x1908082b, 0x2b080819, 0x1908082b,
    0x2b081908, 0x1908082b, 0x08080808, 0x19081908, 0x0808082b, 0x19081908, 0x08081919, 0x19081908,
    0x08082b08, 0x19081908, 0x08082b2b, 0x19081908, 0x08190819, 0x19081908, 0x08191908, 0x19081908,
    0x0819192b, 0x19081908, 0x08192b19, 0x19081908, 0x082b0808, 0x19081908, 0x082b082b, 0x19081908,
    0x082b1919, 0x19081908, 0x082b2b08, 0x19081908, 0x19080819, 0x19081908, 0x19081908, 0x19081908,
    0x1908192b, 0x19081908, 0x19082b19, 0x19081908, 0x19190808, 0x19081908, 0x1919082b, 0x19081908,
    0x19191919, 0x19081908, 0x19192b08, 0x19081908, 0x192b0819, 0x19081908, 0x192b1908, 0x19081908,
    0x2b080808, 0x19081908, 0x2b08082b, 0x19081908, 0x2b081919, 0x19081908, 0x2b082b08, 0x19081908,
    0x2b190819, 0x19081908, 0x2b191908, 0x19081908, 0x2b2b0808, 0x19081908, 0x08080819, 0x19081919,
    0x08081908, 0x19081919, 0x0808192b, 0x19081919, 0x08082b19, 0x19081919, 0x08190808, 0x19081919,
    0x0819082b, 0x19081919, 0x08191919, 0x19081919, 0x08192b08, 0x19081919, 0x082b0819, 0x19081919,
    0x082b1908, 0x19081919, 0x19080808, 0x19081919, 0x1908082b, 0x19081919, 0x19081919, 0x19081919,
    0x19082b08, 0x19081919, 0x19190819, 0x19081919, 0x19191908, 0x19081919, 0x192b0808, 0x19081919,
    0x192b2b2b, 0x19081919, 0x2b080819, 0x19081919, 0x2b081908, 0x19081919, 0x2b190808, 0x19081919,
    0x08080808, 0x1908192b, 0x0808082b, 0x1908192b, 0x08081919, 0x1908192b, 0x08082b08, 0x1908192b,
    0x08190819, 0x1908192b, 0x08191908, 0x1908192b, 0x082b0808, 0x1908192b, 0x19080819, 0x1908192b,
    0x19081908, 0x1908192b, 0x19190808, 0x1908192b, 0x2b080808, 0x1908192b, 0x2b2b1919, 0x1908192b,
    0x08080819, 0x19082b08, 0x08081908, 0x19082b08, 0x08082b19, 0x19082b08, 0x08190808, 0x19082b08,
    0x0819082b, 0x19082b08, 0x08191919, 0x19082b08, 0x08192b08, 0x19082b08, 0x082b0819, 0x19082b08,
    0x082b1908, 0x19082b08, 0x19080808, 0x19082b08, 0x1908082b, 0x19082b08, 0x19081919, 0x19082b08,
    0x19082b08, 0x19082b08, 0x19190819, 0x19082b08, 0x19191908, 0x19082b08, 0x192b0808, 0x19082b08,
    0x2b081908, 0x19082b08, 0x2b190808, 0x19082b08, 0x08080808, 0x19082b19, 0x0808082b, 0x19082b19,
    0x08081919, 0x19082b19, 0x08082b08, 0x19082b19, 0x08190819, 0x19082b19, 0x08191908, 0x19082b19,
    0x082b0808, 0x19082b19, 0x19080819, 0x19082b19, 0x19081908, 0x19082b19, 0x19190808, 0x19082b19,
    0x2b080808, 0x19082b19, 0x2b19192b, 0x19082b19, 0x08080819, 0x19082b2b, 0x08081908, 0x19082b2b,
    0x08190808, 0x19082b2b, 0x19080808, 0x19082b2b, 0x08080808, 0x19190808, 0x0808082b, 0x19190808,
    0x08081919, 0x19190808, 0x08082b08, 0x19190808, 0x08190819, 0x19190808, 0x08191908, 0x19190808,
    0x0819192b, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x082b082b, 0x19190808,
    0x082b1919, 0x19190808, 0x082b2b08, 0x19190808, 0x19080819, 0x19190808, 0x19081908, 0x19190808,
    0x1908192b, 0x19190808, 0x19082b19, 0x19190808, 0x19190808, 0x19190808, 0x1919082b, 0x19190808,
    0x19191919, 0x19190808, 0x19192b08, 0x19190808, 0x192b0819, 0x19190808, 0x192b1908, 0x19190808,
    0x2b080808, 0x19190808, 0x2b08082b, 0x19190808, 0x2b081919, 0x19190808, 0x2b082b08, 0x19190808,
    0x2b190819, 0x19190808, 0x2b191908, 0x19190808, 0x08080819, 0x19190819, 0x08081908, 0x19190819,
    0x0808192b, 0x19190819, 0x08082b19, 0x19190819, 0x08190808, 0x19190819, 0x0819082b, 0x19190819,
    0x08191919, 0x19190819, 0x08192b08, 0x19190819, 0x082b0819, 0x19190819, 0x082b1908, 0x19190819,
    0x19080808, 0x19190819, 0x1908082b, 0x19190819, 0x19081919, 0x19190819, 0x19082b08, 0x19190819,
    0x19190819, 0x19190819, 0x19191908, 0x19190819, 0x192b0808, 0x19190819, 0x2b080819, 0x19190819,
    0x2b081908, 0x19190819, 0x2b190808, 0x19190819, 0x08080808, 0x1919082b, 0x08081919, 0x1919082b,
    0x08082b08, 0x1919082b, 0x08190819, 0x1919082b, 0x08191908, 0x1919082b, 0x082b0808, 0x1919082b,
    0x19080819, 0x1919082b, 0x19081908, 0x1919082b, 0x19190808, 0x1919082b, 0x192b2b19, 0x1919082b,
    0x2b080808, 0x1919082b, 0x08080819, 0x19191908, 0x08081908, 0x19191908, 0x0808192b, 0x19191908,
    0x08082b19, 0x19191908, 0x08190808, 0x19191908, 0x0819082b, 0x19191908, 0x08191919, 0x19191908,
    0x08192b08, 0x19191908, 0x082b0819, 0x19191908, 0x082b1908, 0x19191908, 0x19080808, 0x19191908,
    0x1908082b, 0x19191908, 0x19081919, 0x19191908, 0x19082b08, 0x19191908, 0x19190819, 0x19191908,
    0x19191908, 0x19191908, 0x192b0808, 0x19191908, 0x2b080819, 0x19191908, 0x2b081908, 0x19191908,
    0x2b190808, 0x19191908, 0x08080808, 0x19191919, 0x0808082b, 0x19191919, 0x08081919, 0x19191919,
    0x08082b08, 0x19191919, 0x08190819, 0x19191919, 0x08191908, 0x19191919, 0x082b0808, 0x19191919,
    0x19080819, 0x19191919, 0x19081908, 0x19191919, 0x19190808, 0x19191919, 0x2b080808, 0x19191919,
    0x08080819, 0x1919192b, 0x08081908, 0x1919192b, 0x08190808, 0x1919192b, 0x082b192b, 0x1919192b,
    0x19080808, 0x1919192b, 0x08080808, 0x19192b08, 0x0808082b, 0x19192b08, 0x08081919, 0x19192b08,
    0x08082b08, 0x19192b08, 0x08190819, 0x19192b08, 0x08191908, 0x19192b08, 0x082b0808, 0x19192b08,
    0x19080819, 0x19192b08, 0x19081908, 0x19192b08, 0x19190808, 0x19192b08, 0x19192b2b, 0x19192b08,
    0x2b080808, 0x19192b08, 0x08080819, 0x19192b19, 0x08081908, 0x19192b19, 0x08190808, 0x19192b19,
    0x19080808, 0x19192b19, 0x08080808, 0x19192b2b, 0x08192b19, 0x19192b2b, 0x2b081919, 0x19192b2b,
    0x2b2b2b08, 0x19192b2b, 0x08080819, 0x192b0808, 0x08081908, 0x192b0808, 0x0808192b, 0x192b0808,
    0x08190808, 0x192b0808, 0x0819082b, 0x192b0808, 0x08191919, 0x192b0808, 0x08192b08, 0x192b0808,
    0x082b0819, 0x192b0808, 0x082b1908, 0x192b0808, 0x19080808, 0x192b0808, 0x19081919, 0x192b0808,
    0x19082b08, 0x192b0808, 0x19190819, 0x192b0808, 0x19191908, 0x192b0808, 0x192b0808, 0x192b0808,
    0x2b081908, 0x192b0808, 0x2b190808, 0x192b0808, 0x08080808, 0x192b0819, 0x0808082b, 0x192b0819,
    0x08081919, 0x192b0819, 0x08082b08, 0x192b0819, 0x08190819, 0x192b0819, 0x08191908, 0x192b0819,
    0x082b0808, 0x192b0819, 0x19080819, 0x192b0819, 0x19081908, 0x192b0819, 0x19190808, 0x192b0819,
    0x2b080808, 0x192b0819, 0x2b192b19, 0x192b0819, 0x08081908, 0x192b082b, 0x08190808, 0x192b082b,
    0x19080808, 0x192b082b, 0x1919192b, 0x192b082b, 0x2b2b0819, 0x192b082b, 0x08080808, 0x192b1908,
    0x08081919, 0x192b1908, 0x08082b08, 0x192b1908, 0x08190819, 0x192b1908, 0x08191908, 0x192b1908,
    0x082b0808, 0x192b1908, 0x19080819, 0x192b1908, 0x19081908, 0x192b1908, 0x19190808, 0x192b1908,
    0x2b080808, 0x192b1908, 0x08080819, 0x192b1919, 0x08081908, 0x192b1919, 0x08190808, 0x192b1919,
    0x19080808, 0x192b1919, 0x19082b2b, 0x192b1919, 0x192b2b08, 0x192b1919, 0x2b19082b, 0x192b1919,
    0x08080808, 0x192b192b, 0x2b191908, 0x192b192b, 0x08080819, 0x192b2b08, 0x08081908, 0x192b2b08,
    0x08190808, 0x192b2b08, 0x192b1919, 0x192b2b08, 0x2b192b08, 0x192b2b08, 0x08080808, 0x192b2b19,
    0x082b2b2b, 0x192b2b19, 0x1908082b, 0x192b2b2b, 0x2b2b0819, 0x192b2b2b, 0x08080808, 0x2b080808,
    0x0808082b, 0x2b080808, 0x08081919, 0x2b080808, 0x08082b08, 0x2b080808, 0x08190819, 0x2b080808,
    0x08191908, 0x2b080808, 0x08192b19, 0x2b080808, 0x082b0808, 0x2b080808, 0x082b1919, 0x2b080808,
    0x19080819, 0x2b080808, 0x19081908, 0x2b080808, 0x19190808, 0x2b080808, 0x1919082b, 0x2b080808,
    0x19191919, 0x2b080808, 0x19192b08, 0x2b080808, 0x192b0819, 0x2b080808, 0x2b080808, 0x2b080808,
    0x2b081919, 0x2b080808, 0x2b190819, 0x2b080808, 0x2b191908, 0x2b080808, 0x08080819, 0x2b080819,
    0x08081908, 0x2b080819, 0x08082b19, 0x2b080819, 0x08190808, 0x2b080819, 0x0819082b, 0x2b080819,
    0x08191919, 0x2b080819, 0x08192b08, 0x2b080819, 0x082b0819, 0x2b080819, 0x082b1908, 0x2b080819,
    0x19080808, 0x2b080819, 0x1908082b, 0x2b080819, 0x19081919, 0x2b080819, 0x19082b08, 0x2b080819,
    0x19190819, 0x2b080819, 0x19191908, 0x2b080819, 0x2b080819, 0x2b080819, 0x2b081908, 0x2b080819,
    0x2b190808, 0x2b080819, 0x2b2b2b19, 0x2b080819, 0x08080808, 0x2b08082b, 0x08081919, 0x2b08082b,
    0x08082b2b, 0x2b08082b, 0x08190819, 0x2b08082b, 0x08191908, 0x2b08082b, 0x19080819, 0x2b08082b,
    0x19081908, 0x2b08082b, 0x19190808, 0x2b08082b, 0x08080819, 0x2b081908, 0x08081908, 0x2b081908,
    0x0808192b, 0x2b081908, 0x08082b19, 0x2b081908, 0x08190808, 0x2b081908, 0x0819082b, 0x2b081908,
    0x08191919, 0x2b081908, 0x08192b08, 0x2b081908, 0x082b0819, 0x2b081908, 0x19080808, 0x2b081908,
    0x1908082b, 0x2b081908, 0x19081919, 0x2b081908, 0x19082b08, 0x2b081908, 0x19190819, 0x2b081908,
    0x19191908, 0x2b081908, 0x192b0808, 0x2b081908, 0x2b080819, 0x2b081908, 0x2b081908, 0x2b081908,
    0x2b190808, 0x2b081908, 0x08080808, 0x2b081919, 0x0808082b, 0x2b081919, 0x08081919, 0x2b081919,
    0x08082b08, 0x2b081919, 0x08190819, 0x2b081919, 0x08191908, 0x2b081919, 0x082b0808, 0x2b081919,
    0x19080819, 0x2b081919, 0x19081908, 0x2b081919, 0x19190808, 0x2b081919, 0x2b080808, 0x2b081919,
    0x2b082b2b, 0x2b081919, 0x08080819, 0x2b08192b, 0x08081908, 0x2b08192b, 0x08190808, 0x2b08192b,
    0x082b2b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x08080808, 0x2b082b08, 0x08081919, 0x2b082b08,
    0x08190819, 0x2b082b08, 0x08191908, 0x2b082b08, 0x19080819, 0x2b082b08, 0x19081908, 0x2b082b08,
    0x19190808, 0x2b082b08, 0x2b2b082b, 0x2b082b08, 0x08080819, 0x2b082b19, 0x08081908, 0x2b082b19,
    0x19080808, 0x2b082b19, 0x192b1919, 0x2b082b19, 0x082b082b, 0x2b082b2b, 0x19192b08, 0x2b082b2b,
    0x19192b2b, 0x2b082b2b, 0x2b08082b, 0x2b082b2b, 0x2b2b082b, 0x2b082b2b, 0x08080819, 0x2b190808,
    0x08081908, 0x2b190808, 0x08082b19, 0x2b190808, 0x08190808, 0x2b190808, 0x0819082b, 0x2b190808,
    0x08191919, 0x2b190808, 0x08192b08, 0x2b190808, 0x082b1908, 0x2b190808, 0x19080808, 0x2b190808,
    0x1908082b, 0x2b190808, 0x19081919, 0x2b190808, 0x19082b08, 0x2b190808, 0x19190819, 0x2b190808,
    0x19191908, 0x2b190808, 0x192b0808, 0x2b190808, 0x2b080819, 0x2b190808, 0x2b081908, 0x2b190808,
    0x2b190808, 0x2b190808, 0x08080808, 0x2b190819, 0x08081919, 0x2b190819, 0x08190819, 0x2b190819,
    0x08191908, 0x2b190819, 0x19080819, 0x2b190819, 0x19081908, 0x2b190819, 0x19190808, 0x2b190819,
    0x19192b2b, 0x2b190819, 0x08080819, 0x2b19082b, 0x08081908, 0x2b19082b, 0x08190808, 0x2b19082b,
    0x19080808, 0x2b19082b, 0x2b2b192b, 0x2b19082b, 0x08080808, 0x2b191908, 0x0808082b, 0x2b191908,
    0x08081919, 0x2b191908, 0x08082b08, 0x2b191908, 0x08190819, 0x2b191908, 0x08191908, 0x2b191908,
    0x082b0808, 0x2b191908, 0x19080819, 0x2b191908, 0x19081908, 0x2b191908, 0x19190808, 0x2b191908,
    0x2b080808, 0x2b191908, 0x2b19192b, 0x2b191908, 0x08080819, 0x2b191919, 0x08081908, 0x2b191919,
    0x08190808, 0x2b191919, 0x19080808, 0x2b191919, 0x2b192b08, 0x2b191919, 0x2b2b0819, 0x2b191919,
    0x08080808, 0x2b19192b, 0x1908192b, 0x2b19192b, 0x192b1908, 0x2b19192b, 0x08080819, 0x2b192b08,
    0x08081908, 0x2b192b08, 0x08190808, 0x2b192b08, 0x082b192b, 0x2b192b08, 0x19080808, 0x2b192b08,
    0x2b2b2b19, 0x2b192b08, 0x08080808, 0x2b192b19, 0x19082b19, 0x2b192b19, 0x1919082b, 0x2b192b19,
    0x2b190808, 0x2b192b2b, 0x08080808, 0x2b2b0808, 0x08081919, 0x2b2b0808, 0x08082b2b, 0x2b2b0808,
    0x08191908, 0x2b2b0808, 0x082b082b, 0x2b2b0808, 0x082b2b2b, 0x2b2b0808, 0x19080819, 0x2b2b0808,
    0x19081908, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b2b082b, 0x2b2b0808, 0x2b2b2b2b, 0x2b2b0808,
    0x19080808, 0x2b2b0819, 0x192b1919, 0x2b2b0819, 0x0808082b, 0x2b2b082b, 0x08082b2b, 0x2b2b082b,
    0x082b082b, 0x2b2b082b, 0x082b2b08, 0x2b2b082b, 0x082b2b2b, 0x2b2b082b, 0x2b08082b, 0x2b2b082b,
    0x2b082b08, 0x2b2b082b, 0x2b082b2b, 0x2b2b082b, 0x2b2b2b08, 0x2b2b082b, 0x08080819, 0x2b2b1908,
    0x08081908, 0x2b2b1908, 0x08190808, 0x2b2b1908, 0x19080808, 0x2b2b1908, 0x2b082b19, 0x2b2b1908,
    0x2b2b1908, 0x2b2b1908, 0x08080808, 0x2b2b1919, 0x08192b19, 0x2b2b1919, 0x19190819, 0x2b2b192b,
    0x08082b2b, 0x2b2b2b08, 0x082b2b08, 0x2b2b2b08, 0x2b2b082b, 0x2b2b2b08, 0x19191908, 0x2b2b2b19,
    0x2b08192b, 0x2b2b2b19, 0x08082b08, 0x2b2b2b2b, 0x08082b2b, 0x2b2b2b2b, 0x082b0808, 0x2b2b2b2b,
    0x082b082b, 0x2b2b2b2b, 0x082b2b08, 0x2b2b2b2b, 0x2b082b08, 0x2b2b2b2b, 0x2b2b2b2b, 0x2b2b2b2b
);
#endif

#ifdef IQ3_XXS_GRID
const iq3xxs_grid = array<u32, 256>(
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04
);
#endif

#ifdef IQ3_S_GRID
const iq3s_grid = array<u32, 512>(
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
);
#endif

#if defined(IQ1_S_GRID) || defined(IQ1_M_GRID)

const IQ1_DELTA: f32 = 0.125;

const iq1_grid = array<u32, 1024>(
    0xfffdffff, 0xfff7fff0, 0xffccfff5, 0xffdfffc0, 0xffd7ffdd, 0xff30ffd5, 0xff03ff0c, 0xff10ff01,
    0xff7dff7f, 0xff75ff77, 0xff5fff40, 0xff57ff5d, 0xfcf3ff55, 0xfcccfcf0, 0xfcc1fcc3, 0xfcc5fcc4,
    0xfc3cfcd0, 0xfc34fc31, 0xfc00fc0d, 0xfc1cfc05, 0xfc11fc13, 0xfc70fc17, 0xfc43fc4c, 0xfc50fc41,
    0xfdfdfdff, 0xfdf5fdf7, 0xfddffdc0, 0xfdd7fddd, 0xfd30fdd5, 0xfd04fd0c, 0xfd14fd13, 0xfd7dfd7f,
    0xfd75fd77, 0xfd40fd4c, 0xfd5ffd44, 0xfd57fd5d, 0xf3ccfd55, 0xf3c1f3c3, 0xf33cf3d0, 0xf300f334,
    0xf313f3)wgsl";

static const char wgsl_mul_mat_vec_part1[] = R"wgsl(05, 0xf34cf310, 0xf350f344, 0xf0f3f0fc, 0xf0f1f0f0, 0xf0c7f0c0, 0xf0d4f0c5, 0xf030f03f,
    0xf00ff035, 0xf003f00c, 0xf001f000, 0xf01ff004, 0xf010f01d, 0xf015f017, 0xf04cf07c, 0xf047f040,
    0xf05cf045, 0xf050f053, 0xf054f051, 0xf1c4f1c3, 0xf133f13c, 0xf10df10f, 0xf107f100, 0xf11cf11f,
    0xf114f111, 0xf14cf170, 0xf144f143, 0xf7fdf7ff, 0xf7f5f7f7, 0xf7dff7c0, 0xf7d7f7dd, 0xf730f7d5,
    0xf701f70c, 0xf77ff710, 0xf777f77d, 0xf740f775, 0xf75df75f, 0xf755f757, 0xf4ccf4f0, 0xf4c4f4c3,
    0xf4d0f4d3, 0xf40ff43c, 0xf400f40c, 0xf413f41c, 0xf44cf414, 0xf441f443, 0xf450f444, 0xf5fdf5ff,
    0xf5f5f5f7, 0xf5dff5c0, 0xf5d7f5dd, 0xf530f5d5, 0xf504f50c, 0xf510f51c, 0xf57df57f, 0xf577f570,
    0xf540f575, 0xf55df55f, 0xf555f557, 0xcfcccfcf, 0xcfc4cfc3, 0xcfd0cfd3, 0xcf33cf3c, 0xcf00cf0f,
    0xcf1ccf07, 0xcf10cf13, 0xcf4ccf14, 0xcf41cf43, 0xcf50cf5c, 0xccf3ccfc, 0xccf4ccf1, 0xcccdcccf,
    0xccc7ccc0, 0xccd3ccdc, 0xcc30ccd4, 0xcc0fcc35, 0xcc0dcc0c, 0xcc00cc03, 0xcc04cc01, 0xcc10cc1f,
    0xcc4dcc73, 0xcc5ccc40, 0xcdcccc53, 0xcdc1cdc3, 0xcd3fcdd0, 0xcd34cd31, 0xcd00cd0d, 0xcd05cd07,
    0xcd11cd13, 0xcd4ccd70, 0xcd41cd43, 0xc3fccd50, 0xc3f4c3f1, 0xc3c0c3c3, 0xc3c4c3c7, 0xc3d1c3dc,
    0xc330c33c, 0xc337c331, 0xc30cc335, 0xc300c303, 0xc304c301, 0xc310c31d, 0xc373c317, 0xc34fc374,
    0xc340c343, 0xc344c347, 0xc35cc345, 0xc350c353, 0xc0fdc354, 0xc0f5c0f0, 0xc0c3c0cc, 0xc0c1c0c0,
    0xc0dfc0c4, 0xc0d0c0dd, 0xc0d5c0d7, 0xc033c03c, 0xc031c030, 0xc00dc00c, 0xc000c003, 0xc004c001,
    0xc01cc005, 0xc010c013, 0xc014c011, 0xc07dc07f, 0xc070c073, 0xc075c077, 0xc04cc04f, 0xc040c043,
    0xc044c041, 0xc05fc045, 0xc050c05d, 0xc1f3c1fc, 0xc1f1c1f0, 0xc1c1c1c0, 0xc1c5c1c7, 0xc1d1c1dc,
    0xc13dc13f, 0xc130c133, 0xc135c137, 0xc100c10c, 0xc107c101, 0xc11cc104, 0xc110c113, 0xc114c117,
    0xc171c115, 0xc14dc175, 0xc153c140, 0xc7ccc154, 0xc7d0c7c1, 0xc733c73c, 0xc734c731, 0xc700c70f,
    0xc705c707, 0xc71cc71f, 0xc711c713, 0xc770c714, 0xc743c74c, 0xc4cfc750, 0xc4c0c4cd, 0xc4dcc4c5,
    0xc43dc4d0, 0xc430c433, 0xc40cc437, 0xc400c403, 0xc404c401, 0xc41fc405, 0xc415c410, 0xc44cc474,
    0xc440c44d, 0xc45cc447, 0xc454c451, 0xc5c1c5f4, 0xc5d1c5d3, 0xc531c533, 0xc50fc534, 0xc500c50d,
    0xc51cc507, 0xc514c511, 0xc54cc570, 0xc545c541, 0xdffddfff, 0xdff5dff7, 0xdfdfdfc0, 0xdfd0dfdd,
    0xdfd5dfd7, 0xdf0cdf30, 0xdf1cdf04, 0xdf7fdf10, 0xdf77df7d, 0xdf40df75, 0xdf5ddf5f, 0xdf57df50,
    0xdcf0df55, 0xdcc3dccc, 0xdcd0dcc4, 0xdc33dc3d, 0xdc00dc34, 0xdc05dc07, 0xdc13dc1c, 0xdc11dc10,
    0xdc4fdc70, 0xdc44dc41, 0xddfcdc50, 0xddf5ddf7, 0xddc0ddcc, 0xdddddddf, 0xddd5ddd7, 0xdd0cdd30,
    0xdd04dd01, 0xdd7cdd10, 0xdd75dd77, 0xdd40dd4c, 0xdd5ddd5f, 0xdd55dd57, 0xd3c3d3f0, 0xd3c4d3c1,
    0xd333d3d0, 0xd331d330, 0xd30dd334, 0xd307d300, 0xd311d305, 0xd34cd370, 0xd344d343, 0xd350d35c,
    0xd0c0d0f4, 0xd0d4d0dc, 0xd030d03f, 0xd00cd037, 0xd000d003, 0xd01dd004, 0xd017d010, 0xd04fd074,
    0xd040d043, 0xd045d047, 0xd053d05c, 0xd054d051, 0xd1cfd1f0, 0xd1c4d1cd, 0xd13cd1d0, 0xd100d134,
    0xd11cd11f, 0xd173d114, 0xd14fd171, 0xd7ffd145, 0xd7f7d7fd, 0xd7c0d7f5, 0xd7ddd7df, 0xd7d5d7d7,
    0xd70cd730, 0xd710d703, 0xd77dd77f, 0xd775d777, 0xd75dd75f, 0xd755d757, 0xd4ccd4f4, 0xd4c4d4c3,
    0xd431d4d0, 0xd40dd434, 0xd41cd400, 0xd411d413, 0xd470d414, 0xd441d44f, 0xd453d444, 0xd5ffd450,
    0xd5f7d5fd, 0xd5dfd5f5, 0xd5d7d5dd, 0xd530d5d5, 0xd501d50c, 0xd510d504, 0xd57dd57f, 0xd575d577,
    0xd55fd540, 0xd557d55d, 0x3ff0d555, 0x3fc13fcc, 0x3f343fd0, 0x3f003f0d, 0x3f053f07, 0x3f133f1c,
    0x3f433f11, 0x3f5c3f44, 0x3cff3f51, 0x3cf33cfc, 0x3cf43cf1, 0x3cc03ccd, 0x3cc73cc1, 0x3cdc3cc5,
    0x3cd43cd1, 0x3c373c30, 0x3c0c3c35, 0x3c003c03, 0x3c043c01, 0x3c103c05, 0x3c153c17, 0x3c733c7c,
    0x3c4f3c71, 0x3c403c4d, 0x3c5c3c5f, 0x3df03c5d, 0x3dc33dcc, 0x3dd03dc1, 0x3d0d3d3c, 0x3d053d00,
    0x3d143d13, 0x3d433d74, 0x33fc3d50, 0x33c433c0, 0x333033d4, 0x33353337, 0x3303330c, 0x33013300,
    0x331d331c, 0x33173310, 0x337c3315, 0x33743371, 0x334d334f, 0x335f3340, 0x3354335c, 0x30fd30fc,
    0x30f530f0, 0x30c330cc, 0x30c130c0, 0x30df30c4, 0x30d530d0, 0x3033303c, 0x30313030, 0x300f3034,
    0x3003300c, 0x30013000, 0x30043007, 0x3013301c, 0x30113010, 0x307d3014, 0x30703073, 0x304c3077,
    0x30403043, 0x30443041, 0x30503045, 0x30553057, 0x31f031fc, 0x31c331f4, 0x31c731c0, 0x31dc31c5,
    0x31d431d3, 0x313d313f, 0x31373130, 0x310c310f, 0x3100310d, 0x31043101, 0x3110311d, 0x317c3117,
    0x31753170, 0x31403143, 0x3153315c, 0x37f03151, 0x37c037cc, 0x37d037c5, 0x3734373d, 0x3700370f,
    0x371c3707, 0x37113713, 0x37703714, 0x3743374c, 0x37443741, 0x34fc3750, 0x34f134f0, 0x34cf34f5,
    0x34c034c3, 0x34dc34c7, 0x34d134d3, 0x3430343f, 0x340c3435, 0x3403340d, 0x34013400, 0x341f3404,
    0x3410341d, 0x34153411, 0x34743471, 0x3440344d, 0x34473441, 0x3453345c, 0x34543451, 0x353335c1,
    0x35343531, 0x35073500, 0x35133505, 0x35433514, 0x0ffc3550, 0x0ff00ff3, 0x0ff40ff1, 0x0fc00fcd,
    0x0fdc0fc5, 0x0fd40fd3, 0x0f300f3f, 0x0f0c0f37, 0x0f000f03, 0x0f040f01, 0x0f170f10, 0x0f740f71,
    0x0f470f40, 0x0f5c0f5f, 0x0f540f51, 0x0cf70cf0, 0x0cf50cf4, 0x0cc30ccc, 0x0cc10cc0, 0x0cc40cc7,
    0x0cd00cdf, 0x0cd70cd1, 0x0c3c0cd5, 0x0c300c33, 0x0c340c31, 0x0c0c0c0f, 0x0c030c0d, 0x0c010c00,
    0x0c040c07, 0x0c1c0c05, 0x0c100c13, 0x0c140c11, 0x0c700c7d, 0x0c430c4c, 0x0c410c40, 0x0c5f0c44,
    0x0c550c50, 0x0df10dfc, 0x0dc00dcd, 0x0ddc0dc5, 0x0d3d0dd3, 0x0d350d30, 0x0d030d0c, 0x0d010d00,
    0x0d1d0d04, 0x0d700d10, 0x0d4d0d4f, 0x0d440d40, 0x0d530d45, 0x03f003f3, 0x03c303cc, 0x03c103c0,
    0x03c403c7, 0x03d003dc, 0x03d503d7, 0x0333033c, 0x03310330, 0x03350334, 0x030c030f, 0x03000303,
    0x03070301, 0x03050304, 0x031d031c, 0x03100313, 0x03140311, 0x0377037f, 0x034c0375, 0x03400343,
    0x03440341, 0x0353035c, 0x03550350, 0x00fd00fc, 0x00f000f3, 0x00f400f1, 0x00cc00cf, 0x00c300cd,
    0x00c100c0, 0x00c500c4, 0x00d300dc, 0x00d100d0, 0x003f00d4, 0x003d003c, 0x00300033, 0x00370031,
    0x000f0034, 0x000d000c, 0x00000003, 0x00070001, 0x00050004, 0x001c001f, 0x00100013, 0x00170011,
    0x00150014, 0x0073007c, 0x00740070, 0x004f0075, 0x0043004c, 0x00410040, 0x00440047, 0x0053005c,
    0x00510050, 0x01ff0054, 0x01fd01fc, 0x01f101f3, 0x01f401f7, 0x01c301cc, 0x01c701c0, 0x01df01c4,
    0x01dd01dc, 0x01d001d3, 0x01d701d1, 0x013c01d4, 0x01310130, 0x01340137, 0x010f0135, 0x010d010c,
    0x01000103, 0x01070101, 0x01050104, 0x0113011c, 0x01140110, 0x0170017d, 0x01770171, 0x01750174,
    0x0140014c, 0x015d0145, 0x01510150, 0x01540157, 0x07f007f3, 0x07f407f1, 0x07c007cf, 0x07dc07c7,
    0x073007d5, 0x07350737, 0x0703070c, 0x07010700, 0x07040707, 0x071d071f, 0x07100713, 0x0774077d,
    0x074d074f, 0x07470740, 0x0754075c, 0x04fd04fc, 0x04f504f0, 0x04c304cc, 0x04c104c0, 0x04d004c4,
    0x0433043c, 0x04310430, 0x040f0434, 0x040d040c, 0x04000403, 0x04070401, 0x04050404, 0x0413041c,
    0x04110410, 0x047c0414, 0x04740470, 0x0443044c, 0x04410440, 0x04440447, 0x05f30450, 0x05c005f7,
    0x05df05c5, 0x05d105d0, 0x053005d4, 0x05340537, 0x0500050c, 0x05070501, 0x051d0504, 0x05170510,
    0x057c0515, 0x054d0575, 0x05410540, 0x05450547, 0x1ff0055c, 0x1fc11fc3, 0x1fd01fc4, 0x1f0f1f33,
    0x1f011f00, 0x1f051f07, 0x1f131f1c, 0x1f141f11, 0x1f411f7c, 0x1cfc1f50, 0x1cf11cf3, 0x1ccd1cf4,
    0x1cdc1cc0, 0x1cd11cdd, 0x1c301cd4, 0x1c0c1c34, 0x1c011c00, 0x1c101c04, 0x1c151c11, 0x1c751c73,
    0x1c401c4d, 0x1c511c5c, 0x1dcc1c54, 0x1dc41dc1, 0x1d3c1d3f, 0x1d001d31, 0x1d071d01, 0x1d701d1f,
    0x1d411d4c, 0x13cc1d50, 0x13c013cd, 0x13c513c1, 0x13d113dc, 0x133f13d4, 0x1330133d, 0x13351337,
    0x1303130c, 0x13011300, 0x13051304, 0x131d131f, 0x13731310, 0x13741370, 0x134d134f, 0x13401343,
    0x13471341, 0x135c1345, 0x13541353, 0x10f710f0, 0x10cc10f5, 0x10c110c0, 0x103310c4, 0x10311030,
    0x100f1034, 0x1003100c, 0x10011000, 0x101c1004, 0x10101013, 0x10141011, 0x10741071, 0x104c1075,
    0x10411040, 0x10451044, 0x1050105d, 0x10571051, 0x11f411fd, 0x11df11c0, 0x11d711d1, 0x113f11d4,
    0x11371130, 0x110c1135, 0x11001103, 0x11071101, 0x111f1105, 0x11171110, 0x117d117f, 0x11751170,
    0x11411143, 0x11441147, 0x1153115f, 0x11551151, 0x17c417c1, 0x173c17d0, 0x1700170d, 0x171c1705,
    0x17701714, 0x1747174c, 0x14fc1751, 0x14cf14f3, 0x14dc14c0, 0x14d114d3, 0x143f14d4, 0x1430143c,
    0x14371431, 0x1403140c, 0x14011400, 0x141f1404, 0x14151410, 0x1473147d, 0x14401475, 0x1453145c,
    0x14541450, 0x15c115cc, 0x153c15c7, 0x15341533, 0x1500150f, 0x15051507, 0x15101513, 0x15711514,
    0x15471543, 0x15511545, 0x7ffd7fff, 0x7ff57ff7, 0x7fdd7fdf, 0x7fd57fd7, 0x7f0f7f30, 0x7f037f0c,
    0x7f047f01, 0x7f7f7f10, 0x7f777f7d, 0x7f407f75, 0x7f5d7f5f, 0x7f557f57, 0x7ccc7cf0, 0x7cc17cc3,
    0x7cd07cc4, 0x7c337c3c, 0x7c0f7c34, 0x7c007c0d, 0x7c077c01, 0x7c137c04, 0x7c147c11, 0x7c747c70,
    0x7c417c43, 0x7c507c44, 0x7dfd7dff, 0x7df57df7, 0x7ddf7dc0, 0x7dd77ddd, 0x7d0c7dd5, 0x7d047d03,
    0x7d7f7d10, 0x7d777d7d, 0x7d407d75, 0x7d5d7d5f, 0x7d557d57, 0x73c473c3, 0x7333733c, 0x7300730c,
    0x731c7305, 0x73147313, 0x73447343, 0x70f470fc, 0x70c070cd, 0x70d170c5, 0x703f70d4, 0x7030703c,
    0x700c7037, 0x70007003, 0x70047001, 0x70107005, 0x70177011, 0x707c7015, 0x70717073, 0x704f7074,
    0x7040704d, 0x70517047, 0x71c171cc, 0x71d071c4, 0x7133713c, 0x71357134, 0x7100710f, 0x71057104,
    0x7111711c, 0x71707115, 0x7145714c, 0x77ff7153, 0x77f777fd, 0x77c077f5, 0x77dd77df, 0x77d577d7,
    0x7730773c, 0x7703770c, 0x77107704, 0x777f7714, 0x7777777d, 0x77407775, 0x775d775f, 0x77557757,
    0x74f174f0, 0x74c374cc, 0x74d074c1, 0x7433743c, 0x74347431, 0x740d740f, 0x74057400, 0x7413741c,
    0x74417470, 0x74507444, 0x75fd75ff, 0x75f575f7, 0x75df75c0, 0x75d775dd, 0x753075d5, 0x7503750c,
    0x757f7501, 0x7577757d, 0x75407575, 0x755d755f, 0x75557557, 0x4fcc4ff0, 0x4fc74fc1, 0x4fd04fc4,
    0x4f314f3c, 0x4f004f34, 0x4f054f07, 0x4f154f14, 0x4f4c4f70, 0x4f414f43, 0x4f504f44, 0x4cf34cfc,
    0x4cf44cf1, 0x4cc04ccf, 0x4cc54cc7, 0x4cd34cdc, 0x4cd44cd1, 0x4c304c3f, 0x4c0c4c0f, 0x4c004c03,
    0x4c044c01, 0x4c104c1d, 0x4c714c73, 0x4c404c4d, 0x4c5c4c47, 0x4c514c53, 0x4df04c54, 0x4dc34dcc,
    0x4dd04dc4, 0x4d314d33, 0x4d0f4d34, 0x4d004d0d, 0x4d114d07, 0x4d704d14, 0x4d414d43, 0x43fc4d54,
    0x43f143f3, 0x43c043cf, 0x43d143c7, 0x4335433f, 0x4303430c, 0x43014300, 0x43044307, 0x431c431f,
    0x4310431d, 0x43714373, 0x4343434d, 0x43474340, 0x4354435c, 0x40f040ff, 0x40f540f7, 0x40cc40cf,
    0x40c040c3, 0x40c440c1, 0x40d040dc, 0x40d540d4, 0x4033403c, 0x40314030, 0x400f4034, 0x400d400c,
    0x40004003, 0x40074001, 0x40054004, 0x4013401c, 0x40114010, 0x407c4014, 0x40774070, 0x404d404c,
    0x40404043, 0x40444041, 0x405f4045, 0x4050405d, 0x40554057, 0x41f341fc, 0x41c041cf, 0x41df41c4,
    0x41d441d1, 0x41374130, 0x410c4134, 0x4100410d, 0x41044101, 0x41174110, 0x4173417d, 0x41754174,
    0x4143414d, 0x41534140, 0x41544151, 0x47c147f0, 0x47d047c4, 0x4731473c, 0x470d470f, 0x47014700,
    0x47134705, 0x47704710, 0x4741474c, 0x47504744, 0x44f144f3, 0x44cf44f4, 0x44c044cd, 0x44c544c7,
    0x44dc44df, 0x44d144d3, 0x443d443f, 0x44374430, 0x440c4435, 0x44004403, 0x44044401, 0x4410441d,
    0x44154411, 0x4473447c, 0x444d444f, 0x44454440, 0x4451445c, 0x45c045f0, 0x453345d0, 0x45344531,
    0x4500450f, 0x451c4507, 0x454c4570, 0x45404543, 0x5fff4541, 0x5ff75ffd, 0x5fc05ff5, 0x5fdd5fdf,
    0x5fd55fd7, 0x5f0c5f30, 0x5f015f03, 0x5f7f5f04, 0x5f775f7d, 0x5f405f75, 0x5f5d5f5f, 0x5f555f57,
    0x5cf45cf0, 0x5cc35ccc, 0x5cc45cc1, 0x5c315cc5, 0x5c0c5c34, 0x5c075c00, 0x5c1c5c05, 0x5c705c13,
    0x5c4d5c4f, 0x5c445c41, 0x5df75dfd, 0x5dcf5df5, 0x5ddd5dc4, 0x5dd55dd7, 0x5d0c5d30, 0x5d045d01,
    0x5d7f5d10, 0x5d775d7d, 0x5d405d75, 0x5d5d5d5f, 0x5d555d57, 0x53d053c4, 0x5333533c, 0x5303530f,
    0x53075300, 0x531c5305, 0x53115310, 0x53145317, 0x50f15370, 0x50cf50f4, 0x50c050cd, 0x50d150c7,
    0x503d50d4, 0x500c5030, 0x50005003, 0x50045001, 0x50155010, 0x5073507c, 0x50715070, 0x504d5074,
    0x50475040, 0x51cc51f0, 0x51c551c1, 0x51d051dc, 0x51315133, 0x510d5135, 0x51015100, 0x511f5107,
    0x5171511d, 0x5140514f, 0x51445141, 0x5153515c, 0x57ff5151, 0x57f757fd, 0x57df57f5, 0x57d757dd,
    0x570c57d5, 0x57015703, 0x577f5704, 0x5777577d, 0x57405775, 0x575d575f, 0x57555757, 0x54c354f0,
    0x54dc54c4, 0x543c54d0, 0x5400540f, 0x541c5405, 0x54145411, 0x5441544f, 0x55fd55ff, 0x55f555f7,
    0x55dd55df, 0x55d555d7, 0x5503550c, 0x557f5501, 0x5577557d, 0x55405575, 0x555d555f, 0x55555557
);

#endif

#if defined(IQ4_NL_GRID) || defined(IQ4_XS_GRID)

const kvalues_iq4nl = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

#endif

#ifdef U32_DEQUANT_HELPERS
#define SRC0_TYPE u32

fn byte_of(v: u32, b: u32) -> u32 {
    return (v >> (b * 8u)) & 0xFFu;
}

fn sbyte_of(v: u32, b: u32) -> i32 {
    let raw = i32((v >> (b * 8u)) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}
#endif

#ifdef VEC
#define VEC_SIZE 4u
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(dot(SRC1_TYPE(src0_val), src1_val));
}
#endif

#ifdef SCALAR
#define VEC_SIZE 1u
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>;
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@group(0) @binding(3) var<uniform> params: MulMatParams;

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
#ifdef USE_SUBGROUP_REDUCTION
  , @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let thread_id = local_id.x;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    let row_base = (wg_linear % output_groups) * OUTPUTS_PER_WG;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    let dst_idx_base = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row_base;

    var acc: array<f32, OUTPUTS_PER_WG>;

#ifdef MUL_ACC_FLOAT
    let k_vec = params.k / VEC_SIZE;
    let src1_idx_base_vec = src1_idx_base / VEC_SIZE;

    // Each thread walks K, loads from the vector, and updates
    // a small block of output rows held in registers.
    for (var k = thread_id; k < k_vec; k += WG_SIZE) {
        let x = src1[src1_idx_base_vec + k];
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let src0_idx = (src0_batch_offset + output_row * params.stride_01) / VEC_SIZE + k;
                acc[row] += inner_dot(src0[src0_idx], x);
            }
        }
    }
#endif

#ifdef MUL_ACC_Q4_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 18
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % 4;
    for (var block = thread_id/THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE/THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                var row_sum = 0.0;

                let q_packed = load_u32_at_src0(block_byte_base + 2u + 4u * thread_within_block);
                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let q_lo = (f32(q_byte & 0xFu) - 8.0) * d;
                    let q_hi = (f32((q_byte >> 4u) & 0xFu) - 8.0) * d;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q4_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 20
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                let m = f32(load_f16_at_src0(block_byte_base + 2u));
                var row_sum = 0.0;

                let q_packed = load_u32_at_src0(block_byte_base + 4u + 4u * thread_within_block);
                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let q_lo = f32(q_byte & 0xFu) * d + m;
                    let q_hi = f32((q_byte >> 4u) & 0xFu) * d + m;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q5_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 22
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                let qh_packed = load_u32_at_src0(block_byte_base + 2u);
                let q_packed = load_u32_at_src0(block_byte_base + 6u + 4u * thread_within_block);
                let qh_shift = thread_within_block * 4u;
                var row_sum = 0.0;

                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let qh_lo = ((qh_packed >> (qh_shift + byte_idx)) << 4u) & 0x10u;
                    let qh_hi = (qh_packed >> (qh_shift + byte_idx + 12u)) & 0x10u;
                    let q_lo = (f32((q_byte & 0xFu) | qh_lo) - 16.0) * d;
                    let q_hi = (f32(((q_byte >> 4u) & 0xFu) | qh_hi) - 16.0) * d;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q5_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 24
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                let m = f32(load_f16_at_src0(block_byte_base + 2u));
                let qh_packed = load_u32_at_src0(block_byte_base + 4u);
                let q_packed = load_u32_at_src0(block_byte_base + 8u + 4u * thread_within_block);
                let qh_shift = thread_within_block * 4u;
                var row_sum = 0.0;

                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let qh_lo = ((qh_packed >> (qh_shift + byte_idx)) << 4u) & 0x10u;
                    let qh_hi = (qh_packed >> (qh_shift + byte_idx + 12u)) & 0x10u;
                    let q_lo = f32((q_byte & 0xFu) | qh_lo) * d + m;
                    let q_hi = f32(((q_byte >> 4u) & 0xFu) | qh_hi) * d + m;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q8_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 34
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = f32(src1[x_base + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                var row_sum = 0.0;

                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    let q_packed = load_u32_at_src0(block_byte_base + 2u + 4u * (thread_within_block * 2u + packed_idx));
                    for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                        let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d;
                        row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                    }
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q8_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 36
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
            x_block[i] = f32(src1[x_base + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_f16_at_src0(block_byte_base));
                let m = f32(load_f16_at_src0(block_byte_base + 2u));
                var row_sum = 0.0;

                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    let q_packed = load_u32_at_src0(block_byte_base + 4u + 4u * (thread_within_block * 2u + packed_idx));
                    for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                        let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d + m;
                        row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                    }
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q2_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 84
#define THREADS_PER_BLOCK 16

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let lane = tid / 2u;
    let phase = tid % 2u;
    let iq = lane / 4u;
    let ir = lane % 4u;
    let is = ir / 2u;

    let y_offset = 128u * iq + 8u * ir + 4u * phase;
    let sc0_byte = 8u * iq + is;
    let sc2_byte = 8u * iq + is + 2u;
    let sc4_byte = 8u * iq + is + 4u;
    let sc6_byte = 8u * iq + is + 6u;
    let qs_byte = 16u + (16u * iq + 4u * ir) * 2u + 4u * phase;

    let num_blocks = params.k / BLOCK_SIZE;

    for (var block = block_group; block < num_blocks; block += num_block_groups) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
        var x_block: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            x_block[i]       = f32(src1[x_base + i]);
            x_block[i + 4u]  = f32(src1[x_base + 32u + i]);
            x_block[i + 8u]  = f32(src1[x_base + 64u + i]);
            x_block[i + 12u] = f32(src1[x_base + 96u + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                let dall = f32(load_f16_at_src0(block_byte_base + 80u));
                let dmin = f32(load_f16_at_src0(block_byte_base + 82u)) * (1.0 / 16.0);

                let sc0 = byte_of(load_u32_at_src0_aligned(block_byte_base + sc0_byte), sc0_byte & 3u);
                let sc2 = byte_of(load_u32_at_src0_aligned(block_byte_base + sc2_byte), sc2_byte & 3u);
                let sc4 = byte_of(load_u32_at_src0_aligned(block_byte_base + sc4_byte), sc4_byte & 3u);
                let sc6 = byte_of(load_u32_at_src0_aligned(block_byte_base + sc6_byte), sc6_byte & 3u);

                let q_u32 = load_u32_at_src0_aligned(block_byte_base + qs_byte);
                let qs0 = q_u32 & 0xFFFFu;
                let qs1 = q_u32 >> 16u;

                var sumy = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

                sumy[0] = x_block[0] + x_block[1] + x_block[2] + x_block[3];
                sumy[1] = x_block[4] + x_block[5] + x_block[6] + x_block[7];
                sumy[2] = x_block[8] + x_block[9] + x_block[10] + x_block[11];
                sumy[3] = x_block[12] + x_block[13] + x_block[14] + x_block[15];

                acc1[0] = x_block[0] * f32(qs0 & 0x0003u) + x_block[2] * f32(qs1 & 0x0003u);
                acc2[0] = x_block[1] * f32(qs0 & 0x0300u) + x_block[3] * f32(qs1 & 0x0300u);
                acc1[1] = x_block[4] * f32(qs0 & 0x000Cu) + x_block[6] * f32(qs1 & 0x000Cu);
                acc2[1] = x_block[5] * f32(qs0 & 0x0C00u) + x_block[7] * f32(qs1 & 0x0C00u);
                acc1[2] = x_block[8] * f32(qs0 & 0x0030u) + x_block[10] * f32(qs1 & 0x0030u);
                acc2[2] = x_block[9] * f32(qs0 & 0x3000u) + x_block[11] * f32(qs1 & 0x3000u);
                acc1[3] = x_block[12] * f32(qs0 & 0x00C0u) + x_block[14] * f32(qs1 & 0x00C0u);
                acc2[3] = x_block[13] * f32(qs0 & 0xC000u) + x_block[15] * f32(qs1 & 0xC000u);

                acc[row] += dall * ((acc1[0] + (1.0/256.0) * acc2[0]) * f32(sc0 & 0xFu) +
                                    (acc1[1] + (1.0/256.0) * acc2[1]) * f32(sc2 & 0xFu) / 4.0 +
                                    (acc1[2] + (1.0/256.0) * acc2[2]) * f32(sc4 & 0xFu) / 16.0 +
                                    (acc1[3] + (1.0/256.0) * acc2[3]) * f32(sc6 & 0xFu) / 64.0)
                          - dmin * (sumy[0] * f32(sc0 & 0xF0u) + sumy[1] * f32(sc2 & 0xF0u) +
                                    sumy[2] * f32(sc4 & 0xF0u) + sumy[3] * f32(sc6 & 0xF0u));
            }
        }
    }
#endif


#ifdef MUL_ACC_Q3_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 110
#define THREADS_PER_BLOCK 16

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let lane = tid / 2u;
    let phase = tid % 2u;
    let ip = lane / 4u;
    let il = 2u * ((lane % 4u) / 2u);
    let ir = lane % 2u;
    let l0 = 8u * ir;

    let q_byte = 32u + 32u * ip + l0 + 16u * phase;
    let h_byte = l0 + 16u * phase;
    let y_offset = 128u * ip + 32u * il + l0 + 16u * phase;

    let s_shift1 = 4u * ip;
    let s_shift2 = s_shift1 + il;

    let v1 = select(64.0, 4.0, il == 0u);
    let v2 = 4.0 * v1;
    let shift = 2u * il;

    var qm0: u32; var qm1: u32; var qm2: u32; var qm3: u32;
    if (il == 0u) {
        qm0 = 0x0003u; qm1 = 0x0300u; qm2 = 0x000Cu; qm3 = 0x0C00u;
    } else {
        qm0 = 0x0030u; qm1 = 0x3000u; qm2 = 0x00C0u; qm3 = 0xC000u;
    }

    let mm_idx = 2u * ip + il / 2u;
    var hm0: u32; var hm1: u32; var hm2: u32; var hm3: u32;
    switch (mm_idx) {
        case 0u: { hm0=0x0001u; hm1=0x0100u; hm2=0x0002u; hm3=0x0200u; }
        case 1u: { hm0=0x0004u; hm1=0x0400u; hm2=0x0008u; hm3=0x0800u; }
        case 2u: { hm0=0x0010u; hm1=0x1000u; hm2=0x0020u; hm3=0x2000u; }
        default: { hm0=0x0040u; hm1=0x4000u; hm2=0x0080u; hm3=0x8000u; }
    }

    let num_blocks = params.k / BLOCK_SIZE;

    for (var block = block_group; block < num_blocks; block += num_block_groups) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
        var x_block: array<f32, 16>;
        for (var i = 0u; i < 8u; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 8u] = f32(src1[x_base + 32u + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                let d = f32(load_f16_at_src0(block_byte_base + 108u));
                let a_base = 96u;
                let a_il0 = load_u16_at_src0(block_byte_base + a_base + il * 2u);
                let a_il1 = load_u16_at_src0(block_byte_base + a_base + (il + 1u) * 2u);
                let a_4 = load_u16_at_src0(block_byte_base + a_base + 8u);
                let a_5 = load_u16_at_src0(block_byte_base + a_base + 10u);

                var scales32 = a_4 | (a_5 << 16u);
                let aux32 = ((scales32 >> s_shift2) << 4u) & 0x30303030u;
                scales32 = a_il0 | (a_il1 << 16u);
                scales32 = ((scales32 >> s_shift1) & 0x0F0F0F0Fu) | aux32;

                let scale0 = f32(i32(byte_of(scales32, phase + 0u)) - 32);
                let scale1 = f32(i32(byte_of(scales32, phase + 2u)) - 32);

                let q_u32_0 = load_u32_at_src0(block_byte_base + q_byte + 0u);
                let q_u32_1 = load_u32_at_src0(block_byte_base + q_byte + 4u);
                let h_u32_0 = load_u32_at_src0(block_byte_base + h_byte + 0u);
                let h_u32_1 = load_u32_at_src0(block_byte_base + h_byte + 4u);

                var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
                var s4 = 0.0; var s5 = 0.0; var s6 = 0.0;

                for (var l = 0u; l < 8u; l += 2u) {
                    let q_u32 = select(q_u32_0, q_u32_1, l >= 4u);
                    let qs = select(q_u32 & 0xFFFFu, q_u32 >> 16u, (l & 2u) != 0u);
                    let h_u32 = select(h_u32_0, h_u32_1, l >= 4u);
                    let hv = select(h_u32 & 0xFFFFu, h_u32 >> 16u, (l & 2u) != 0u);

                    s1 += x_block[l + 0u] * f32(qs & qm0);
                    s2 += x_block[l + 1u] * f32(qs & qm1);
                    s3 += select(0.0, x_block[l + 0u], (hv & hm0) == 0u) +
                          select(0.0, x_block[l + 1u], (hv & hm1) == 0u);
                    s4 += x_block[l + 8u] * f32(qs & qm2);
                    s5 += x_block[l + 9u] * f32(qs & qm3);
                    s6 += select(0.0, x_block[l + 8u], (hv & hm2) == 0u) +
                          select(0.0, x_block[l + 9u], (hv & hm3) == 0u);
                }

                let d1 = d * (s1 + (1.0/256.0) * s2 - s3 * v1);
                let d2 = d * (s4 + (1.0/256.0) * s5 - s6 * v2);
                acc[row] += (d1 * scale0 + 0.25 * d2 * scale1) / f32(1u << shift);
            }
        }
    }
#endif

#ifdef MUL_ACC_Q4_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 144
#define THREADS_PER_BLOCK 16

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let il = tid / 4u;
    let ir = tid % 4u;
    let im = il / 2u;
    let in = il % 2u;
    let l0 = 4u * (2u * ir + in);

    let y_offset = 64u * im + l0;
    let q_offset = 32u * im + l0;
    let sc0_byte = 4u + im * 2u;
    let sc2_byte = 4u + (im + 2u) * 2u;
    let sc4_byte = 4u + (im + 4u) * 2u;

    let num_blocks = params.k / BLOCK_SIZE;

    for (var block = block_group; block < num_blocks; block += num_block_groups) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
        var x_block: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            x_block[i]       = f32(src1[x_base + i]);
            x_block[i + 4u]  = f32(src1[x_base + 32u + i]);
            x_block[i + 8u]  = f32(src1[x_base + 128u + i]);
            x_block[i + 12u] = f32(src1[x_base + 160u + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                let d = f32(load_f16_at_src0(block_byte_base + 0u));
                let dmin = f32(load_f16_at_src0(block_byte_base + 2u));

                let sc0_u32 = load_u32_at_src0_aligned(block_byte_base + sc0_byte);
                let sc0 = select(sc0_u32 & 0xFFFFu, sc0_u32 >> 16u, (sc0_byte & 2u) != 0u);
                let sc2_u32 = load_u32_at_src0_aligned(block_byte_base + sc2_byte);
                let sc2 = select(sc2_u32 & 0xFFFFu, sc2_u32 >> 16u, (sc2_byte & 2u) != 0u);
                let sc4_u32 = load_u32_at_src0_aligned(block_byte_base + sc4_byte);
                let sc4 = select(sc4_u32 & 0xFFFFu, sc4_u32 >> 16u, (sc4_byte & 2u) != 0u);

                let sc16_0 = sc0 & 0x3F3Fu;
                let sc16_1 = sc2 & 0x3F3Fu;
                let sc16_2 = (sc4 & 0x0F0Fu) | ((sc0 & 0xC0C0u) >> 2u);
                let sc16_3 = ((sc4 >> 4u) & 0x0F0Fu) | ((sc2 & 0xC0C0u) >> 2u);

                let scale0 = f32(sc16_0 & 0xFFu);
                let scale1 = f32((sc16_0 >> 8u) & 0xFFu);
                let min0 = f32(sc16_1 & 0xFFu);
                let min1 = f32((sc16_1 >> 8u) & 0xFFu);
                let scale2 = f32(sc16_2 & 0xFFu);
                let scale3 = f32((sc16_2 >> 8u) & 0xFFu);
                let min2 = f32(sc16_3 & 0xFFu);
                let min3 = f32((sc16_3 >> 8u) & 0xFFu);

                let q1_u32 = load_u32_at_src0_aligned(block_byte_base + 16u + q_offset);
                let q2_u32 = load_u32_at_src0_aligned(block_byte_base + 80u + q_offset);

                var dot = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var sumx = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                for (var i = 0u; i < 4u; i++) {
                    let q1b = byte_of(q1_u32, i);
                    let q2b = byte_of(q2_u32, i);
                    dot[0] += x_block[i] * f32(q1b & 0x0Fu);
                    dot[1] += x_block[i + 4u] * f32(q1b >> 4u);
                    dot[2] += x_block[i + 8u] * f32(q2b & 0x0Fu);
                    dot[3] += x_block[i + 12u] * f32(q2b >> 4u);
                    sumx[0] += x_block[i];
                    sumx[1] += x_block[i + 4u];
                    sumx[2] += x_block[i + 8u];
                    sumx[3] += x_block[i + 12u];
                }

                acc[row] += d * (dot[0] * scale0 + dot[1] * scale1 + dot[2] * scale2 + dot[3] * scale3)
                          - dmin * (sumx[0] * min0 + sumx[1] * min1 + sumx[2] * min2 + sumx[3] * min3);
            }
        }
    }
#endif

#ifdef MUL_ACC_Q5_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 176
#define THREADS_PER_BLOCK 16

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let il = tid / 4u;
    let ir = tid % 4u;
    let im = il / 2u;
    let in = il % 2u;
    let l0 = 4u * (2u * ir + in);

    let y_offset = 64u * im + l0;
    let q_offset = 48u + 32u * im + l0;
    let qh_offset = 16u + 8u * ir + 4u * in;
    let sc0_byte = 4u + im * 2u;
    let sc2_byte = 4u + (im + 2u) * 2u;
    let sc4_byte = 4u + (im + 4u) * 2u;

    let hm1 = 1u << (2u * im);
    let hm2 = hm1 << 1u;
    let hm3 = hm1 << 4u;
    let hm4 = hm2 << 4u;

    let num_blocks = params.k / BLOCK_SIZE;

    for (var block = block_group; block < num_blocks; block += num_block_groups) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
        var x_block: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            x_block[i]       = f32(src1[x_base + i]);
            x_block[i + 4u]  = f32(src1[x_base + 32u + i]);
            x_block[i + 8u]  = f32(src1[x_base + 128u + i]);
            x_block[i + 12u] = f32(src1[x_base + 160u + i]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                let d = f32(load_f16_at_src0(block_byte_base + 0u));
                let dmin = f32(load_f16_at_src0(block_byte_base + 2u));

                let sc0_u32 = load_u32_at_src0_aligned(block_byte_base + sc0_byte);
                let sc0 = select(sc0_u32 & 0xFFFFu, sc0_u32 >> 16u, (sc0_byte & 2u) != 0u);
                let sc2_u32 = load_u32_at_src0_aligned(block_byte_base + sc2_byte);
                let sc2 = select(sc2_u32 & 0xFFFFu, sc2_u32 >> 16u, (sc2_byte & 2u) != 0u);
                let sc4_u32 = load_u32_at_src0_aligned(block_byte_base + sc4_byte);
                let sc4 = select(sc4_u32 & 0xFFFFu, sc4_u32 >> 16u, (sc4_byte & 2u) != 0u);

                let sc16_0 = sc0 & 0x3F3Fu;
                let sc16_1 = sc2 & 0x3F3Fu;
                let sc16_2 = (sc4 & 0x0F0Fu) | ((sc0 & 0xC0C0u) >> 2u);
                let sc16_3 = ((sc4 >> 4u) & 0x0F0Fu) | ((sc2 & 0xC0C0u) >> 2u);

                let f0 = f32(sc16_0 & 0xFFu);
                let f1 = f32((sc16_0 >> 8u) & 0xFFu);
                let m0 = f32(sc16_1 & 0xFFu);
                let m1 = f32((sc16_1 >> 8u) & 0xFFu);
                let f4 = f32(sc16_2 & 0xFFu);
                let f5 = f32((sc16_2 >> 8u) & 0xFFu);
                let m4 = f32(sc16_3 & 0xFFu);
                let m5 = f32((sc16_3 >> 8u) & 0xFFu);

                let q1_u32 = load_u32_at_src0_aligned(block_byte_base + q_offset);
                let q2_u32 = load_u32_at_src0_aligned(block_byte_base + q_offset + 64u);
                let qh_u32 = load_u32_at_src0_aligned(block_byte_base + qh_offset);

                var vals = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var sumy = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                for (var i = 0u; i < 4u; i++) {
                    let q1b = byte_of(q1_u32, i);
                    let q2b = byte_of(q2_u32, i);
                    let qhb = byte_of(qh_u32, i);

                    let yl0 = x_block[i];
                    let yl8 = x_block[i + 4u];
                    let yh0 = x_block[i + 8u];
                    let yh8 = x_block[i + 12u];

                    sumy[0] += yl0;
                    sumy[1] += yl8;
                    sumy[2] += yh0;
                    sumy[3] += yh8;

                    let q0 = f32((q1b & 0x0Fu) | select(0u, 0x10u, (qhb & hm1) != 0u));
                    let q1 = f32((q1b >> 4u) | select(0u, 0x10u, (qhb & hm2) != 0u));
                    let q2 = f32((q2b & 0x0Fu) | select(0u, 0x10u, (qhb & hm3) != 0u));
                    let q3 = f32((q2b >> 4u) | select(0u, 0x10u, (qhb & hm4) != 0u));

                    vals[0] += yl0 * q0;
                    vals[1] += yl8 * q1;
                    vals[2] += yh0 * q2;
                    vals[3] += yh8 * q3;
                }

                acc[row] += d * (f0 * vals[0] + f1 * vals[1] + f4 * vals[2] + f5 * vals[3])
                          - dmin * (sumy[0] * m0 + sumy[1] * m1 +
                                    sumy[2] * m4 + sumy[3] * m5);
            }
        }
    }
#endif

#ifdef MUL_ACC_Q6_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 210
#define THREADS_PER_BLOCK 16

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let ip = tid / 8u;
    let il = tid % 8u;
    let l0 = 4u * il;
    let is = 8u * ip + l0 / 16u;

    let y_offset = 128u * ip + l0;
    let q_offset_l = 64u * ip + l0;
    let q_offset_h = 32u * ip + l0;

    let num_blocks = params.k / BLOCK_SIZE;
    let sc_base_byte = 192u + (is & ~3u);
    let sc_byte_pos = is & 3u;

    for (var block = block_group; block < num_blocks; block += num_block_groups) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
        var x_block: array<f32, 16>;
        for (var l = 0u; l < 4u; l++) {
            x_block[l]       = f32(src1[x_base + l]);
            x_block[l + 4u]  = f32(src1[x_base + 32u + l]);
            x_block[l + 8u]  = f32(src1[x_base + 64u + l]);
            x_block[l + 12u] = f32(src1[x_base + 96u + l]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                let d = f32(load_f16_at_src0(block_byte_base + 208u));
                let ql1_u32 = load_u32_at_src0(block_byte_base + q_offset_l);
                let ql2_u32 = load_u32_at_src0(block_byte_base + q_offset_l + 32u);
                let qh_u32 = load_u32_at_src0(block_byte_base + 128u + q_offset_h);
                let sc_u32_0 = load_u32_at_src0(block_byte_base + sc_base_byte);
                let sc_u32_1 = load_u32_at_src0(block_byte_base + sc_base_byte + 4u);

                let sc0 = sbyte_of(sc_u32_0, sc_byte_pos);
                let sc2 = sbyte_of(sc_u32_0, sc_byte_pos + 2u);
                let sc4 = sbyte_of(sc_u32_1, sc_byte_pos);
                let sc6 = sbyte_of(sc_u32_1, sc_byte_pos + 2u);

                var sums = vec4<f32>(0.0, 0.0, 0.0, 0.0);

                for (var l = 0u; l < 4u; l++) {
                    let q1b = byte_of(ql1_u32, l);
                    let q2b = byte_of(ql2_u32, l);
                    let qhb = byte_of(qh_u32, l);

                    let dq0 = f32(i32((q1b & 0x0Fu) | ((qhb & 0x03u) << 4u)) - 32);
                    let dq1 = f32(i32((q2b & 0x0Fu) | ((qhb & 0x0Cu) << 2u)) - 32);
                    let dq2 = f32(i32((q1b >> 4u) | (qhb & 0x30u)) - 32);
                    let dq3 = f32(i32((q2b >> 4u) | ((qhb & 0xC0u) >> 2u)) - 32);

                    sums[0] += x_block[l] * dq0;
                    sums[1] += x_block[l + 4u] * dq1;
                    sums[2] += x_block[l + 8u] * dq2;
                    sums[3] += x_block[l + 12u] * dq3;
                }

                acc[row] += d * (sums[0] * f32(sc0) + sums[1] * f32(sc2) +
                                 sums[2] * f32(sc4) + sums[3] * f32(sc6));
            }
        }
    }
#endif

#ifdef USE_SUBGROUP_REDUCTION
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        let subgroup_total = subgroupAdd(acc[row]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(row, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var row = subgroup_id; (row < OUTPUTS_PER_WG) && (row_base + row < params.m); row += num_subgroups) {
        let output_row = row_base + row;
        var row_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            row_acc += partial_sums[partial_index(row, k)];
        }
        let row_total = subgroupAdd(row_acc);
        if (subgroup_invocation_id == 0) {
            dst[dst_idx_base + row] = row_total;
        }
    }
#endif

#ifdef USE_WORKGROUP_REDUCTION
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        partial_sums[partial_index(row, thread_id)] = acc[row];
    }

    workgroupBarrier();

    var stride = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                partial_sums[partial_index(row, thread_id)] += partial_sums[partial_index(row, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < OUTPUTS_PER_WG) {
        let output_row = row_base + thread_id;
        if (output_row < params.m) {
            dst[dst_idx_base + thread_id] = partial_sums[partial_index(thread_id, 0)];
        }
    }
#endif
}
)wgsl";

static const std::string& wgsl_mul_mat_vec_str() {
    static const std::string s = []{
        std::string tmp;
        tmp.reserve(108180);
        tmp.append(wgsl_mul_mat_vec_part0);
        tmp.append(wgsl_mul_mat_vec_part1);
        return tmp;
    }();
    return s;
}
const char* wgsl_mul_mat_vec = wgsl_mul_mat_vec_str().c_str();

const char* wgsl_pad = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
    src_ne3: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    dst_ne3: u32,

    // Pad sizes (in elements)
    lp0: u32,
    rp0: u32,
    lp1: u32,
    rp1: u32,
    lp2: u32,
    rp2: u32,
    lp3: u32,
    rp3: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

fn wrap_around(idx: i32, n: u32) -> u32 {
    return u32(idx + i32(n)) % n;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let dst_plane = params.dst_ne2 * params.dst_ne1 * params.dst_ne0;
    let i3 = i / dst_plane;
    i = i % dst_plane;
    let i2 = i / (params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne1 * params.dst_ne0);
    let i1 = i / params.dst_ne0;
    let i0 = i % params.dst_ne0;

    var value: f32 = 0.0;

#ifdef CIRCULAR
    let ci0 = wrap_around(i32(i0) - i32(params.lp0), params.src_ne0);
    let ci1 = wrap_around(i32(i1) - i32(params.lp1), params.src_ne1);
    let ci2 = wrap_around(i32(i2) - i32(params.lp2), params.src_ne2);
    let ci3 = wrap_around(i32(i3) - i32(params.lp3), params.src_ne3);
    let circular_src_idx = ci0 * params.stride_src0 + ci1 * params.stride_src1 +
                           ci2 * params.stride_src2 + ci3 * params.stride_src3;
    value = src[params.offset_src + circular_src_idx];
#else
    let is_src =
        (i0 >= params.lp0 && i0 < params.dst_ne0 - params.rp0) &&
        (i1 >= params.lp1 && i1 < params.dst_ne1 - params.rp1) &&
        (i2 >= params.lp2 && i2 < params.dst_ne2 - params.rp2) &&
        (i3 >= params.lp3 && i3 < params.dst_ne3 - params.rp3);
    if (is_src) {
        let src_idx = (i0 - params.lp0) * params.stride_src0 + (i1 - params.lp1) * params.stride_src1 +
                      (i2 - params.lp2) * params.stride_src2 + (i3 - params.lp3) * params.stride_src3;
        value = src[params.offset_src + src_idx];
    }
#endif

    dst[params.offset_dst + gid.x] = value;
}
)wgsl";

const char* wgsl_repeat = R"wgsl(enable f16;

struct Params {
    ne: u32,

    offset_src0: u32,
    offset_dst: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    a_ne3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
};

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_I32
#define DataType i32
#endif
#ifdef TYPE_I16
// same size (16-bit) is sufficient for repeat
#define DataType f16
#endif

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        var i = gid.x;
        let i3 = i / (params.ne2 * params.ne1 * params.ne0);
        i = i % (params.ne2 * params.ne1 * params.ne0);
        let i2 = i / (params.ne1 * params.ne0);
        i = i % (params.ne1 * params.ne0);
        let i1 = i / params.ne0;
        let i0 = i % params.ne0;

        let a_i0 = i0 % params.a_ne0;
        let a_i1 = i1 % params.a_ne1;
        let a_i2 = i2 % params.a_ne2;
        let a_i3 = i3 % params.a_ne3;

        let a_index = a_i0 * params.stride_src0_0 +
                           a_i1 * params.stride_src0_1 +
                           a_i2 * params.stride_src0_2 +
                           a_i3 * params.stride_src0_3;

        dst[params.offset_dst + gid.x] = src0[params.offset_src0 + a_index];
    }
}
)wgsl";

const char* wgsl_rope = R"wgsl(enable f16;

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_F16
#define DataType f16
#endif

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_src2: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    n_threads: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    n_dims: u32,
    mode: u32,
    theta_scale: f32,
    attn_factor: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_dim0: f32,
    corr_dim1: f32,
    sections0: u32,
    sections1: u32,
    sections2: u32,
    sections3: u32
};

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;
@group(0) @binding(1)
var<storage, read_write> src1: array<i32>;

#ifdef INPLACE

#ifdef FF_FUNC

@group(0) @binding(2)
var<storage, read_write> src2: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

#else

@group(0) @binding(2)
var<uniform> params: Params;

#endif

#else

#ifdef FF_FUNC
@group(0) @binding(2)
var<storage, read_write> src2: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(4)
var<uniform> params: Params;

#else
@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
var<uniform> params: Params;
#endif
#endif

#ifdef FF_FUNC
fn freq_factor(i: u32) -> f32 {
    return src2[params.offset_src2 + i/2];
}

#else
fn freq_factor(i: u32) -> f32 {
    return 1.0f;
}
#endif
#ifdef INPLACE
fn rotate(i_dst0: u32, i_dst1: u32, out0: f32, out1: f32) {
    src0[i_dst0] = DataType(out0);
    src0[i_dst1] = DataType(out1);
}
#else
fn rotate(i_dst0: u32, i_dst1: u32, out0: f32, out1: f32) {
    dst[i_dst0] = DataType(out0);
    dst[i_dst1] = DataType(out1);
}
#endif

fn rope_yarn_ramp(low: f32, high: f32, i: u32) -> f32 {
    let y = (f32(i / 2) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// returns vector of (cos_theta, sin_theta)
// TODO: check performance of instantiating once on the CPU and passed as buffer, since it's repeated per-row
fn rope_yarn(theta_extrap: f32, i: u32) -> vec2<f32> {
    var mscale = params.attn_factor;
    var theta  = params.freq_scale * theta_extrap;
    if (params.ext_factor != 0.0f) {
        let ramp_mix = rope_yarn_ramp(params.corr_dim0, params.corr_dim1, i) * params.ext_factor;
        theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
    }
    return vec2<f32>(cos(theta) * mscale, sin(theta) * mscale);
}

fn pair_base(i0: u32, div_2: bool) -> u32 {
    if (div_2) {
        return i0 / 2;
    } else {
        return i0;
    }
}

fn pair_offset(is_neox: bool, is_mrope: bool, is_vision: bool) -> u32 {
    if (is_vision) {
        return params.n_dims;
    } else if (is_neox || is_mrope) {
        return params.n_dims / 2;
    } else {
        return 1;
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // two elements per n_threads
    if (gid.x >= params.n_threads) {
        return;
    }

    let is_neox = bool(params.mode & 2);
    let is_mrope = bool(params.mode & 8);
    let is_imrope = params.mode == 40;
    let is_vision = params.mode == 24;

    var i = gid.x * 2; // start index for this thread
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let i_src_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    if (i0 >= params.n_dims && !is_vision) {
        let i_src = i_src_row + i0;
        let i_dst = i_dst_row + i0;
        rotate(i_dst, i_dst + 1, f32(src0[i_src]), f32(src0[i_src + 1]));
        return;
    }

    var theta_base_mult: u32 = 0;
    var theta_scale_pwr: u32 = i0 / 2;
    if (is_mrope) {
        let sect_dims = params.sections0 + params.sections1 + params.sections2 + params.sections3;
        let sec_w = params.sections1 + params.sections0;
        let sec_e = params.sections2 + sec_w;
        let sector = (i0 / 2) % sect_dims;
        if (is_imrope) {
          if (sector % 3 == 1 && sector < 3 * params.sections1) {
              theta_base_mult = 1;
          } else if (sector % 3 == 2 && sector < 3 * params.sections2) {
              theta_base_mult = 2;
          } else if (sector % 3 == 0 && sector < 3 * params.sections0) {
              theta_base_mult = 0;
          } else {
              theta_base_mult = 3;
          }
        } else {
          if (sector >= params.sections0 && sector < sec_w) {
              theta_base_mult = 1;
              if (is_vision) {
                  theta_scale_pwr = sector - params.sections0;
              }
          } else if (sector >= sec_w && sector < sec_e) {
              theta_base_mult = 2;
              if (is_vision) {
                  theta_scale_pwr = sector - sec_w;
              }
          } else if (sector >= sec_e) {
              if (is_vision) {
                  theta_scale_pwr = sector - sec_e;
                  theta_scale_pwr = (i0 / 2) % sec_e;
              }
              theta_base_mult = 3;
          } else if (is_vision) {
              theta_scale_pwr = sector;
          }
        }
    }
    let theta_base = f32(src1[params.offset_src1 + i2 + params.ne2 * theta_base_mult]) * pow(params.theta_scale, f32(theta_scale_pwr));
    let thetas = rope_yarn(theta_base/freq_factor(i0), i0);

    let i_src = i_src_row + pair_base(i0, is_neox || is_mrope || is_vision);
    let i_dst = i_dst_row + pair_base(i0, is_neox || is_mrope || is_vision);

    let x0 = f32(src0[i_src]);
    let x1 = f32(src0[i_src + pair_offset(is_neox, is_mrope, is_vision)]);
    rotate(i_dst, i_dst + pair_offset(is_neox, is_mrope, is_vision), x0 * thetas.x - x1 * thetas.y, x0 * thetas.y + x1 * thetas.x);

}
)wgsl";

const char* wgsl_row_norm = R"wgsl(#ifdef INPLACE
fn update(src_offset: u32, dst_offset: u32, scale: f32) {
    src[dst_offset] = scale * src[src_offset];
}

@group(0) @binding(1)
var<uniform> params: Params;
#else
fn update(src_offset: u32, dst_offset: u32, scale: f32) {
    dst[dst_offset] = scale * src[src_offset];
}

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;
#endif

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src/dst
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    // one thread per row
    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    var sum = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        sum += pow(src[i_src_row + col], 2.0);
        col += WG_SIZE;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    sum = scratch[0];

#ifdef RMS_NORM
    let scale = 1.0/sqrt(sum/f32(params.ne0) + params.eps);
#elif defined(L2_NORM)
    let scale = 1.0/max(sqrt(sum), params.eps);
#endif

    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        update(i_src_row + col, i_dst_row + col, scale);
        col += WG_SIZE;
    }
}
)wgsl";

const char* wgsl_scale = R"wgsl(#ifdef INPLACE
@group(0) @binding(1)
var<uniform> params: Params;

fn store_scale(val: f32, offset: u32) {
    src[offset] = val;
}
#else
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn store_scale(val: f32, offset: u32) {
    dst[offset] = val;
}
#endif

struct Params {
    offset_src: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    scale: f32,
    bias: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let i_src = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1 + i0;
    let i_dst = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1 + i0;

    store_scale(src[i_src] * params.scale + params.bias, i_dst);
}
)wgsl";

const char* wgsl_set = R"wgsl(#ifdef TYPE_I32
#define TYPE i32
#else
#define TYPE f32
#endif

#ifndef INPLACE
@group(0) @binding(0)
var<storage, read_write> src0: array<TYPE>;
#define SRC1_BINDING 1
#else
#define SRC1_BINDING 0
#endif

#define DST_BINDING SRC1_BINDING + 1
#define PARAMS_BINDING SRC1_BINDING + 2

@group(0) @binding(SRC1_BINDING)
var<storage, read_write> src1: array<TYPE>;

@group(0) @binding(DST_BINDING)
var<storage, read_write> dst: array<TYPE>;

struct Params {
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_view: u32,

    stride_src10: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst10: u32,
    stride_dst11: u32,
    stride_dst12: u32,
    stride_dst13: u32,

    src1_ne0: u32,
    src1_ne1: u32,
    src1_ne2: u32,
    src1_ne3: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

fn decode_src1_coords(idx: u32) -> vec4<u32> {
    var i = idx;
    let plane = params.src1_ne2 * params.src1_ne1 * params.src1_ne0;
    let i3 = i / plane;
    i = i % plane;
    let row = params.src1_ne1 * params.src1_ne0;
    let i2 = i / row;
    i = i % row;
    let i1 = i / params.src1_ne0;
    let i0 = i % params.src1_ne0;
    return vec4<u32>(i0, i1, i2, i3);
}

fn decode_view_coords(rel: u32) -> vec4<u32> {
    let i3 = rel / params.stride_dst13;
    let rem3 = rel % params.stride_dst13;
    let i2 = rem3 / params.stride_dst12;
    let rem2 = rem3 % params.stride_dst12;
    let i1 = rem2 / params.stride_dst11;
    let i0 = rem2 % params.stride_dst11;
    return vec4<u32>(i0, i1, i2, i3);
}

fn view_rel_from_coords(coords: vec4<u32>) -> u32 {
    return coords.x * params.stride_dst10 + coords.y * params.stride_dst11 +
           coords.z * params.stride_dst12 + coords.w * params.stride_dst13;
}

fn src1_idx_from_coords(coords: vec4<u32>) -> u32 {
    return coords.x * params.stride_src10 + coords.y * params.stride_src11 +
           coords.z * params.stride_src12 + coords.w * params.stride_src13;
}

fn in_set_view(rel: u32, coords: vec4<u32>) -> bool {
    return view_rel_from_coords(coords) == rel;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

#ifdef INPLACE
    let coords = decode_src1_coords(gid.x);

    let src1_idx = params.offset_src1 + src1_idx_from_coords(coords);
    let dst_idx = params.offset_view + view_rel_from_coords(coords);

    dst[dst_idx] = src1[src1_idx];
#else
    let rel = select(params.ne, gid.x - params.offset_view, gid.x >= params.offset_view);
    let coords = decode_view_coords(rel);

    if (rel < params.stride_dst13 * params.src1_ne3 && in_set_view(rel, coords)) {
        dst[gid.x] = src1[params.offset_src1 + src1_idx_from_coords(coords)];
    } else {
        dst[gid.x] = src0[params.offset_src0 + gid.x];
    }
#endif
}
)wgsl";

const char* wgsl_set_rows = R"wgsl(enable f16;

#ifdef DST_F32
#define DST_INNER_TYPE f32
#else
#define DST_INNER_TYPE f16
#endif

#ifdef VEC4
#define SRC_TYPE vec4<f32>
#define DST_TYPE vec4<DST_INNER_TYPE>
#define VEC_SIZE 4
#else
#define SRC_TYPE f32
#define DST_TYPE DST_INNER_TYPE
#define VEC_SIZE 1
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<DST_TYPE>;

#ifdef I64_IDX
@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;
#define PARAMS_BINDING 4
#else
#define PARAMS_BINDING 3
#endif

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= (params.ne3 * params.ne2 * params.n_rows * params.ne0) / VEC_SIZE) {
        return;
    }

    // getting the row from gid
    let elems_per_row = params.ne0 / VEC_SIZE;
    var i = gid.x / elems_per_row;

    let i_src3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_src2 = i / params.n_rows;
    let i_src1 = i % params.n_rows;

    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

#ifdef I64_IDX
    let idx_high = (params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2) * 2;

    let idx_val = idx[idx_high];
    let idx_low_val = idx[idx_high + 1];

    if (idx_low_val != 0) {
        // Upper bits of index are not zero, output will be incorrect
        atomicStore(&error, 1);
        return;
    }
#else
    let idx_i = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;
    let idx_val = idx[idx_i];
#endif

    let i_dst_row = params.offset_dst + idx_val * params.stride_dst1 + i_src2 * params.stride_dst2 + i_src3 * params.stride_dst3;
    let i_src_row = params.offset_src + i_src1 * params.stride_src1 + i_src2 * params.stride_src2 + i_src3 * params.stride_src3;

    let col_idx = (gid.x % elems_per_row);
    dst[i_dst_row/VEC_SIZE + col_idx] = DST_TYPE(src[i_src_row/VEC_SIZE + col_idx]);
}
)wgsl";

const char* wgsl_soft_max = R"wgsl(enable f16;

#ifdef MASK_F32
#define MaskType f32
#endif
#ifdef MASK_F16
#define MaskType f16
#endif

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // shape of src0/dst
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    // shape of src1
    ne12: u32,
    ne13: u32,

    scale: f32,
    max_bias: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

#ifdef HAS_MASK
#ifdef HAS_SINK
@group(0) @binding(1)
var<storage, read_write> mask: array<MaskType>;
@group(0) @binding(2)
var<storage, read_write> sinks: array<f32>;

#ifdef INPLACE
@group(0) @binding(3)
var<uniform> params: Params;

#else
@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(4)
var<uniform> params: Params;
#endif

#else
@group(0) @binding(1)
var<storage, read_write> mask: array<MaskType>;

#ifdef INPLACE
@group(0) @binding(2)
var<uniform> params: Params;

#else
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
#endif
#endif

#else
#ifdef HAS_SINK
@group(0) @binding(1)
var<storage, read_write> sinks: array<f32>;

#ifdef INPLACE
@group(0) @binding(2)
var<uniform> params: Params;

#else
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
#endif

#else
#ifdef INPLACE
@group(0) @binding(1)
var<uniform> params: Params;
#else
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;
#endif
#endif
#endif

#ifdef INPLACE
fn inter_value(i: u32) -> f32 {
    return src[i];
}
fn update(i: u32, val: f32) {
    src[i] = val;
}

#else
fn inter_value(i: u32) -> f32 {
    return dst[i];
}
fn update(i: u32, val: f32) {
    dst[i] = val;
}
#endif

#ifdef HAS_MASK
fn mask_val(i: u32) -> f32 {
    return f32(mask[i]);
}

#else
fn mask_val(i: u32) -> f32 {
    return 0.0;
}
#endif

#ifdef HAS_SINK
fn lower_max_bound(i2: u32) -> f32 {
    return sinks[params.offset_sinks + i2];
}
fn add_sinks(val: f32, i2: u32, max_val: f32) -> f32 {
    return val + exp(sinks[params.offset_sinks + i2] - max_val);
}
#else
fn lower_max_bound(i2: u32) -> f32 {
    return -1e30;
}
fn add_sinks(val: f32, i2: u32, max_val: f32) -> f32 {
    return val;
}
#endif

const CACHE_SIZE: u32 = 16;
var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src0_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_src1_row = params.offset_src1 + (i3 % params.ne13) * params.stride_src13 + (i2 % params.ne12) * params.stride_src12 + i1 * params.stride_src11;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    let head = f32(i2);
    let slope = select(1, select(pow(params.m1, 2 * (head - params.n_head_log2) + 1), pow(params.m0, head + 1), head < params.n_head_log2), params.max_bias > 0);

    var cache: array<f32, CACHE_SIZE>;

    var max_val = lower_max_bound(i2);
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let val = src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col);
        max_val = max(max_val, val);
        if (col < CACHE_SIZE) {
            cache[col] = val;
        }
        col += WG_SIZE;
    }

    scratch[lid.x] = max_val;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + offset]);
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier();

    var sum = 0.0f;
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let val = select(src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col),
                         cache[col], col < CACHE_SIZE);
        let ex = exp(val - row_max);
        sum += ex;
        if (col < CACHE_SIZE) {
            cache[col] = ex;
        } else {
            update(i_dst_row + col, ex);
        }
        col += WG_SIZE;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();
    offset = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let row_sum = add_sinks(scratch[0], i2, row_max);

    let sum_recip = 1.0 / row_sum;
    col = lid.x;
    for  (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        update(i_dst_row + col, select(inter_value(i_dst_row + col), cache[col], col < CACHE_SIZE) * sum_recip);
        col += WG_SIZE;
    }
}

)wgsl";

const char* wgsl_solve_tri = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src00: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src10: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    k: u32,
    ne2: u32,
    ne3: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> shA: array<f32, BATCH_N * N>;
var<workgroup> shB: array<f32, BATCH_N * K_TILE>;

fn src0_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_src0 +
           col * params.stride_src00 +
           row * params.stride_src01 +
           i2 * params.stride_src02 +
           i3 * params.stride_src03;
}

fn src1_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_src1 +
           col * params.stride_src10 +
           row * params.stride_src11 +
           i2 * params.stride_src12 +
           i3 * params.stride_src13;
}

fn dst_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_dst +
           col * params.stride_dst0 +
           row * params.stride_dst1 +
           i2 * params.stride_dst2 +
           i3 * params.stride_dst3;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let batch = workgroup_id.y;
    let col = workgroup_id.x * WG_SIZE + local_id.x;
    let i3 = batch / params.ne2;
    let i2 = batch % params.ne2;
    let active_lane = local_id.x < K_TILE;
    let active_col = active_lane && col < params.k;

    var X: array<f32, N>;

    for (var row_base = 0u; row_base < N; row_base += BATCH_N) {
        let cur_n = min(BATCH_N, N - row_base);

        for (var i = local_id.x; i < cur_n * N; i += WG_SIZE) {
            let tile_row = i / N;
            let tile_col = i % N;
            shA[i] = src0[src0_idx(row_base + tile_row, tile_col, i2, i3)];
        }

        for (var i = local_id.x; i < cur_n * K_TILE; i += WG_SIZE) {
            let tile_row = i / K_TILE;
            let tile_col = i % K_TILE;
            let global_col = workgroup_id.x * WG_SIZE + tile_col;
            let sh_idx = tile_row * K_TILE + tile_col;

            if (global_col < params.k) {
                shB[sh_idx] = src1[src1_idx(row_base + tile_row, global_col, i2, i3)];
            } else {
                shB[sh_idx] = 0.0;
            }
        }

        workgroupBarrier();

        if (active_col) {
            for (var row_offset = 0u; row_offset < cur_n; row_offset++) {
                let r = row_base + row_offset;
                var b = shB[row_offset * K_TILE + local_id.x];
                let a_row = row_offset * N;

                for (var t = 0u; t < r; t++) {
                    b -= shA[a_row + t] * X[t];
                }

                let x = b / shA[a_row + r];
                X[r] = x;
                dst[dst_idx(r, col, i2, i3)] = x;
            }
        }

        workgroupBarrier();
    }
}
)wgsl";

const char* wgsl_ssm_conv = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src01: u32,
    stride_src02: u32,
    stride_src11: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,

    nc: u32,
    nr: u32,
    n_t: u32,
    n_s: u32,
    token_tiles: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(BLOCK_SIZE, TOKENS_PER_WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i1 = gid.x;
    let tile_y = gid.y / TOKENS_PER_WG;
    let local_token = gid.y % TOKENS_PER_WG;
    let i3 = tile_y / params.token_tiles;
    let token_tile = tile_y % params.token_tiles;
    let i2 = token_tile * TOKENS_PER_WG + local_token;

    if (i1 >= params.nr || i2 >= params.n_t || i3 >= params.n_s) {
        return;
    }

    let src0_base = params.offset_src0 + i3 * params.stride_src02 + i2 + i1 * params.stride_src01;
    let src1_base = params.offset_src1 + i1 * params.stride_src11;

    var sum = 0.0;

#ifdef VECTORIZED
    sum =
        src0[src0_base + 0u] * src1[src1_base + 0u] +
        src0[src0_base + 1u] * src1[src1_base + 1u] +
        src0[src0_base + 2u] * src1[src1_base + 2u] +
        src0[src0_base + 3u] * src1[src1_base + 3u];
#else
    for (var i0 = 0u; i0 < params.nc; i0++) {
        sum += src0[src0_base + i0] * src1[src1_base + i0];
    }
#endif

    let dst_idx = params.offset_dst + i3 * params.stride_dst2 + i2 * params.stride_dst1 + i1 * params.stride_dst0;
    dst[dst_idx] = sum;
}
)wgsl";

const char* wgsl_sum_rows = R"wgsl(@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    var local_sum: f32 = 0.0;
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        local_sum += src[i_src_row + col];
    }
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();
    // reduce within workgroup
    var offset: u32 = WG_SIZE >> 1;
    while (offset > 0) {
        if (lid.x < offset) {
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + offset];
        }
        workgroupBarrier();
        offset >>= 1;
    }

    if (lid.x == 0) {
        dst[params.offset_dst + wid.x] = shared_sum[0];
    }
}
)wgsl";

const char* wgsl_unary = R"wgsl(#ifdef TYPE_F16
enable f16;
#define TYPE f16
#else
#define TYPE f32
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<TYPE>;

#ifndef INPLACE
@group(0) @binding(1)
var<storage, read_write> dst: array<TYPE>;
#define PARAMS_BINDING 2
#else
#define PARAMS_BINDING 1
#endif

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    ne0: u32,
    ne1: u32,
    ne2: u32,
#ifdef CLAMP
    clamp_min: f32,
    clamp_max: f32,
#endif
#ifdef FILL
    fill_val: f32,
#endif
#ifdef XIELU
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,
#endif

};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
      return;
    }
    var i = gid.x;
    let ne2 = params.ne2;
#ifdef DIAG
    let ne1 = params.ne0;
#else
    let ne1 = params.ne1;
#endif
    let ne0 = params.ne0;

    let i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    let i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    let i1 = i / ne0;
    let i0 = i % ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

#ifdef ABS
    let res = abs(src[params.offset_src + src_idx]);
#endif
#ifdef SGN
    let res = select(TYPE(select(0.0, -1.0, src[params.offset_src + src_idx] < 0.0)), TYPE(1.0),
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef NEG
    let res = -src[params.offset_src + src_idx];
#endif
#ifdef STEP
    let res = TYPE(select(0.0, 1.0, src[params.offset_src + src_idx] > 0.0));
#endif
#ifdef TANH
    let res = tanh(clamp(src[params.offset_src + src_idx], -9.010913, 9.010913));
#endif
#ifdef RELU
    let res = select(0.0, src[params.offset_src + src_idx], src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef ELU
    let res = select(exp(src[params.offset_src + src_idx]) - 1.0, src[params.offset_src + src_idx],
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef HARDSIGMOID
    let res = min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef SIGMOID
    let res = 1.0 / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef SILU
    let res = src[params.offset_src + src_idx] / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef EXP
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32));
#endif
#ifdef LOG
    let res = TYPE(log(f32(src[params.offset_src + src_idx])));
#endif
#ifdef CLAMP
    let res = clamp(src[params.offset_src + src_idx], TYPE(params.clamp_min), TYPE(params.clamp_max));
#endif
#ifdef FILL
    let res = TYPE(params.fill_val);
#endif
#ifdef HARDSWISH
    let res = src[params.offset_src + src_idx] *
              min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef GELU
    let res = 0.5 * src[params.offset_src + src_idx] *
              (1.0 + tanh(clamp(sqrt(2.0 / 3.14159265) *
                               (src[params.offset_src + src_idx] +
                                0.044715 * pow(src[params.offset_src + src_idx], 3.0)),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_QUICK
    let res = src[params.offset_src + src_idx] * 0.5 *
              (1.0 + tanh(clamp(0.79788456 *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] *
                                    src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_ERF
    let res = 0.5 * src[params.offset_src + src_idx] *
              (1.0 + tanh(clamp(0.79788456 *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] *
                                    src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef XIELU
    let val = f32(src[params.offset_src + src_idx]);
    let res =
        TYPE(select(
            ((exp(min(val, params.eps)) - 1.0) - val) * params.alpha_n + params.beta * val,
            params.alpha_p * val * val + params.beta * val,
            val > 0.0));
#endif
#ifdef SOFTPLUS
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(select(log(1.0 + exp(src_f32)), src_f32, src_f32 > 20.0));
#endif
#ifdef EXPM1
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(exp(src_f32) - 1.0);
#endif
#ifdef FLOOR
    let res = floor(src[params.offset_src + src_idx]);
#endif
#ifdef CEIL
    let res = ceil(src[params.offset_src + src_idx]);
#endif
#ifdef ROUND
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let result = select(ceil(src_f32 - 0.5), floor(src_f32 + 0.5), src_f32 >= 0.0);
    let res = TYPE(result);
#endif
#ifdef TRUNC
    let res = trunc(src[params.offset_src + src_idx]);
#endif
#ifdef SQR
    let res = src[params.offset_src + src_idx] * src[params.offset_src + src_idx];
#endif
#ifdef SQRT
    let res = TYPE(sqrt(f32(src[params.offset_src + src_idx])));
#endif
#ifdef SIN
    let res_f32 = sin(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif
#ifdef COS
    let res_f32 = cos(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif
#ifdef DIAG
    let res = select(0.0, src[params.offset_src + i0 + i2 * params.stride_src2 + i3 * params.stride_src3], i0 == i1);
#endif
#ifdef TRI
#ifdef TRI_TYPE_LOWER
    let res = select(0.0, src[params.offset_src + src_idx], i0 < i1);
#elif TRI_TYPE_LOWER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 <= i1);
#elif TRI_TYPE_UPPER
    let res = select(0.0, src[params.offset_src + src_idx], i0 > i1);
#elif TRI_TYPE_UPPER_DIAG
    let res = select(0.0, src[params.offset_src + src_idx], i0 >= i1);
#endif
#endif

#ifdef INPLACE
    src[params.offset_src + src_idx] = res;
#else
    dst[params.offset_dst + gid.x] = res;
#endif
}
)wgsl";

