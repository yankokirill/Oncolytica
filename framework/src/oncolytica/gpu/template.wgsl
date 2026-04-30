// ===========================================================================
// Boilerplate for GPU spatial grid implementation
// ===========================================================================

// ── Constants ─────────────────────────────────────────────────────────────────

const WORKGROUP_SIZE: u32 = 256u;

// {{ USER_CONSTANTS }}

const MooreOffsets = array<vec3<i32>, 27>(
    vec3<i32>(-1, -1, -1), vec3<i32>(0, -1, -1), vec3<i32>(1, -1, -1),
    vec3<i32>(-1, 0, -1),  vec3<i32>(0, 0, -1),  vec3<i32>(1, 0, -1),
    vec3<i32>(-1, 1, -1),  vec3<i32>(0, 1, -1),  vec3<i32>(1, 1, -1),
    vec3<i32>(-1, -1, 0),  vec3<i32>(0, -1, 0),  vec3<i32>(1, -1, 0),
    vec3<i32>(-1, 0, 0),   vec3<i32>(0, 0, 0),   vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0),   vec3<i32>(0, 1, 0),   vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1),  vec3<i32>(0, -1, 1),  vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1),   vec3<i32>(0, 0, 1),   vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1),   vec3<i32>(0, 1, 1),   vec3<i32>(1, 1, 1)
);

// ── Data Structures ──────────────────────────────────────────────────────────

struct GlobalUniforms {
    NumVoxelTableEntries: u32,
    TissueGridDimX:       u32,
    TissueGridDimY:       u32,
    TissueGridDimZ:       u32,
    TissueVoxelSize:      f32,
    BS_L1_offset:         u32,
    BS_L2_offset:         u32,

    // {{ USER_CONFIG }}
};

struct SimState {
    TotalAgents: u32,
    NewTotalAgents: atomic<u32>,
};

struct VoxelInfo {
    count: atomic<u32>,
    startIndex: u32,
};

struct AgentSortData {
    voxel_key: u32,    // Z-order code
    local_offset: u32,
};

// struct Cell { ... }
// struct Tissue { ... }
// struct Chemistry { ... }

// {{ USER_STRUCTS }}

// ── Buffer Bindings ──────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> U: GlobalUniforms;
@group(0) @binding(1) var<storage, read_write> State: SimState;
@group(0) @binding(2) var<storage, read_write> MetricsBuffer: Metrics;

@group(1) @binding(0) var<storage, read> Cells_In: array<Cell>;
@group(1) @binding(1) var<storage, read_write> Cells_Out: array<Cell>;
@group(1) @binding(2) var<storage, read_write> SortData: array<AgentSortData>;

@group(2) @binding(0) var<storage, read_write> VoxelTable: array<VoxelInfo>;
@group(2) @binding(1) var<storage, read_write> BlockSums: array<u32>;

@group(3) @binding(0) var<storage, read>       Tissue_In: array<Tissue>;
@group(3) @binding(1) var<storage, read_write> Tissue_Out: array<Tissue>;

@group(3) @binding(2) var<storage, read>       Chemistry_In: array<Chemistry>;
@group(3) @binding(3) var<storage, read_write> Chemistry_Out: array<Chemistry>;

// ── Helper Functions ─────────────────────────────────────────────────────────

fn _next_rand(state: ptr<function, u32>) -> f32 {
    var s = *state;
    s = s ^ (s << 13u);
    s = s ^ (s >> 17u);
    s = s ^ (s << 5u);
    *state = s;
    return f32(s) * 2.3283064365386963e-10;
}

fn _rand_dir(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = _next_rand(state);
    let r2 = _next_rand(state);

    let z = r1 * 2.0 - 1.0;
    let phi = r2 * 6.283185307179586;

    let radius_xy = sqrt(max(0.0, 1.0 - z * z));

    return vec3<f32>(
        radius_xy * cos(phi),
        radius_xy * sin(phi),
        z
    );
}

fn interleave_9bit(input: u32) -> u32 {
    var x = input & 0x1ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8))  & 0x0300f00fu;
    x = (x | (x << 4))  & 0x030c30c3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}

fn get_chemical_voxel_key(large_coord: vec3<i32>) -> u32 {
    let x = interleave_9bit(u32(large_coord.x));
    let y = interleave_9bit(u32(large_coord.y));
    let z = interleave_9bit(u32(large_coord.z));
    return x | (y << 1) | (z << 2);
}

fn get_sub_voxel_index(small_coord: vec3<i32>) -> u32 {
    let x = u32(small_coord.x & 1);
    let y = u32(small_coord.y & 1);
    let z = u32(small_coord.z & 1);
    return x | (y << 1) | (z << 2);
}

fn get_tissue_voxel_key(small_coord: vec3<i32>) -> u32 {
    let large_coord = small_coord / 2;
    let j = get_chemical_voxel_key(large_coord);
    let i = get_sub_voxel_index(small_coord);
    return (j << 3u) | i;
}

fn z_order_hash(coord: vec3<i32>) -> u32 {
    return get_tissue_voxel_key(coord);
}

fn tissue_coord_in_bounds(x: i32, y: i32, z: i32) -> bool {
    return u32(x) < U.TissueGridDimX &&
           u32(y) < U.TissueGridDimY &&
           u32(z) < U.TissueGridDimZ;
}

// {{ USER_METHODS }}

@compute @workgroup_size(1)
fn Kernel_UpdateState(@builtin(global_invocation_id) id: vec3<u32>) {
    let dead_count = atomicLoad(&VoxelTable[U.NumVoxelTableEntries - 1u].count);

    let new_total = atomicLoad(&State.NewTotalAgents);
    State.TotalAgents = new_total - dead_count;
    atomicStore(&State.NewTotalAgents, State.TotalAgents);
}

// ===========================================================================
//                 STAGE 1: SPATIAL GRID CONSTRUCTION
// ===========================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_ClearVoxels(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= U.NumVoxelTableEntries) { return; }

    atomicStore(&VoxelTable[idx].count, 0u);
    VoxelTable[idx].startIndex = 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_CountAndOffset(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= atomicLoad(&State.NewTotalAgents)) { return; }

    let cell = Cells_In[idx];
    let z_key = z_order_hash(vec3<i32>(cell.pos / U.TissueVoxelSize));

    let local_offset = atomicAdd(&VoxelTable[z_key].count, 1u);

    SortData[idx].voxel_key = z_key;
    SortData[idx].local_offset = local_offset;
}

// ===========================================================================
//                       STAGE 2: PREFIX SUM
// ===========================================================================

var<workgroup> scan_buf: array<u32, 256>;

fn blelloch_exclusive_scan(lid: u32) -> u32 {
    var stride = 1u;
    while stride < WORKGROUP_SIZE {
        workgroupBarrier();
        if (lid & ((stride << 1u) - 1u)) == ((stride << 1u) - 1u) {
            scan_buf[lid] += scan_buf[lid - stride];
        }
        stride <<= 1u;
    }

    workgroupBarrier();
    let total = scan_buf[WORKGROUP_SIZE - 1u];
    workgroupBarrier();

    if lid == WORKGROUP_SIZE - 1u { scan_buf[lid] = 0u; }

    stride = WORKGROUP_SIZE >> 1u;
    while stride > 0u {
        workgroupBarrier();
        if (lid & ((stride << 1u) - 1u)) == ((stride << 1u) - 1u) {
            let tmp                = scan_buf[lid - stride];
            scan_buf[lid - stride] = scan_buf[lid];
            scan_buf[lid]         += tmp;
        }
        stride >>= 1u;
    }
    workgroupBarrier();
    return total;
}

// ── L0: VoxelTable[gidx].count → VoxelTable[gidx].startIndex ─────────────
// Dispatch: ceil(NumTissueVoxels / 256) workgroups
// 128³ → 8192 wg; 256³ → 65536 wg; 512³ → 524288 wg

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_Scan_L0(
    @builtin(global_invocation_id) gid:   vec3<u32>,
    @builtin(local_invocation_id)  lid_v: vec3<u32>,
    @builtin(workgroup_id)         wgid:  vec3<u32>,
) {
    let gidx = gid.x;
    let lid  = lid_v.x;

    scan_buf[lid] = 0u;
    if (gidx < U.NumVoxelTableEntries) {
        scan_buf[lid] = atomicLoad(&VoxelTable[gidx].count);
    }

    let block_sum = blelloch_exclusive_scan(lid);

    if gidx < U.NumVoxelTableEntries {
        VoxelTable[gidx].startIndex = scan_buf[lid];
    }
    if lid == 0u {
        BlockSums[wgid.x] = block_sum;
    }
}

// ── L1: BlockSums[0..N1-1] → BlockSums[0..N1-1]  (prefix sums of L0 sums)
// Dispatch: ceil(BS_L1_offset / 256) workgroups
// 128³ → 32 wg; 256³ → 256 wg; 512³ → 2048 wg

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_Scan_L1(
    @builtin(global_invocation_id) gid:   vec3<u32>,
    @builtin(local_invocation_id)  lid_v: vec3<u32>,
    @builtin(workgroup_id)         wgid:  vec3<u32>,
) {
    let gidx = gid.x;
    let lid  = lid_v.x;

    scan_buf[lid] = 0u;
    if (gidx < U.BS_L1_offset) {
        scan_buf[lid] = BlockSums[gidx];
    }

    let block_sum = blelloch_exclusive_scan(lid);

    if gidx < U.BS_L1_offset {
        BlockSums[gidx] = scan_buf[lid];
    }
    if lid == 0u {
        BlockSums[U.BS_L1_offset + wgid.x] = block_sum;
    }
}

// ── L2: BlockSums[N1..N1+N2-1] (один workgroup; N2 ≤ 256 для 128³/256³)
// Dispatch: 1 workgroup
// Для 512³ N2=2048 > 256 → нужен ещё один уровень L2/L3 (см. примечание)

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_Scan_L2(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let lid     = lid_v.x;
    let src_idx = U.BS_L1_offset + lid;
    let n2      = U.BS_L2_offset - U.BS_L1_offset;

    scan_buf[lid] = 0u;
    if (lid < n2) {
        scan_buf[lid] = BlockSums[src_idx];
    }

    blelloch_exclusive_scan(lid);

    if lid < n2 { BlockSums[src_idx] = scan_buf[lid]; }
}

// ── AddBack L1: add BlockSums[N1+wg] to all BlockSums[wg*256..] ──
// Dispatch: ceil(BS_L1_offset / 256) workgroups (same 32/256/2048)

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_AddBack_L1(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let gidx   = gid.x;
    let offset = BlockSums[U.BS_L1_offset + wgid.x];
    if gidx < U.BS_L1_offset {
        BlockSums[gidx] += offset;
    }
}

// ── AddBack L0: add BlockSums[wg] to VoxelTable[wg*256..].startIndex
// Dispatch: ceil(NumTissueVoxels / 256) workgroups

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_AddBack_L0(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let gidx = gid.x;
    if gidx < U.NumVoxelTableEntries {
        VoxelTable[gidx].startIndex += BlockSums[wgid.x];
    }
}

// ===========================================================================
//         STAGE 3: SCATTER — place cells into the sorted buffer
// ===========================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn Kernel_Scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= atomicLoad(&State.NewTotalAgents) { return; }

    let s    = SortData[idx];
    let dest = VoxelTable[s.voxel_key].startIndex + s.local_offset;
    Cells_Out[dest] = Cells_In[idx];
}

// ===========================================================================
//                        STAGE 4: MAIN LOGIC
// ===========================================================================

// {{ USER_LOGIC }}
