// ===========================================================================
// Boilerplate for GPU spatial grid implementation
// ===========================================================================

// ── Constants ─────────────────────────────────────────────────────────────────

const WORKGROUP_SIZE: u32 = 256u;

const ACTIVATION_THR: f32 = 0.0005;
const AM_FACTOR: f32 = 0.1;
const AP_THRESHOLD: f32 = 0.0166;
const BETA: f32 = 0.5;
const C_PA: f32 = 5.0;
const K_HILL: f32 = 100.0;
const PDGF_CONSUME: f32 = 0.042;
const PDGF_D: f32 = 0.0417;
const PDGF_DECAY: f32 = 0.005;
const PDGF_SECRETE: f32 = 0.417;
const TAU_MOVE: f32 = 0.71;
const TAU_STOP: f32 = 1.17;
const TYPE_INACTIVE: i32 = 0;
const TYPE_INFECTED: i32 = 1;
const TYPE_RECRUITED: i32 = 2;

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

    treatment_am: u32,
    treatment_ap: u32,
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

struct Cell {
    pos: vec3<f32>,
    _rng_state: u32,
    cell_type: i32,
    p_pot: f32,
    m_pot: f32,
    div_clock: f32,
    is_moving: i32,
    persistence_timer: f32,
    move_dir_x: f32,
    move_dir_y: f32,
    prev_x: f32,
    prev_y: f32,
};

struct Tissue {
    _coord: vec3<i32>,
    _rng_state: u32,
    is_white_matter: i32,
    tract_dir_x: f32,
    tract_dir_y: f32,
    carrying_capacity: i32,
};

struct Chemistry {
    _coord: vec3<i32>,
    _rng_state: u32,
    pdgf: f32,
};

struct Metrics {
    infected_count: atomic<i32>,
    recruited_count: atomic<i32>,
    inactive_count: atomic<i32>,
    total_cells: atomic<i32>,
    total_pdgf: f32,
};

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

fn is_tissue_coord_in_bounds(coord: vec3<i32>) -> bool {
    return u32(coord.x) < U.TissueGridDimX &&
           u32(coord.y) < U.TissueGridDimY &&
           u32(coord.z) < U.TissueGridDimZ;
}

fn is_chem_coord_in_bounds(coord: vec3<i32>) -> bool {
    return u32(coord.x) < (U.TissueGridDimX / 2u) &&
           u32(coord.y) < (U.TissueGridDimY / 2u) &&
           u32(coord.z) < (U.TissueGridDimZ / 2u);
}

fn _gamma_infected(c_pp: f32) -> f32 {
    var num: f32;
    var den: f32;
    num = (C_PA + c_pp);
    den = ((C_PA + c_pp) + K_HILL);
    return (num / den);
}

fn _gamma_recruited(c_pp: f32) -> f32 {
    var beta_k: f32;
    var den: f32;
    beta_k = (BETA * K_HILL);
    den = (c_pp + beta_k);
    return (c_pp / den);
}

// ===========================================================================
//                 STAGE 0: TOTAL AGENTS UPDATE
// ===========================================================================

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

@compute @workgroup_size(256, 1, 1)
fn Kernel_CellRule_0(@builtin(global_invocation_id) _id: vec3<u32>) {
    let cell_index = _id.x;
    if (cell_index >= State.TotalAgents) { return; }
    var cell = Cells_In[cell_index];
    var _rng = cell._rng_state;
    var chem: Chemistry;
    var tissue: Tissue;
    var c_pp: f32;
    var gamma: f32;
    var p: f32;
    var m: f32;
    var n_local: i32;
    var nb: Cell;
    var is_quiescent: i32;
    var div_a: f32;
    var nx: f32;
    var ny: f32;
    var d_rng: f32;
    var d_timer: f32;
    var daughter: Cell;
    var go_u: f32;
    var angle: f32;
    var bm_u1: f32;
    var bm_u2: f32;
    var bm_z: f32;
    var base: f32;
    var stop_u: f32;
    chem = Chemistry_In[get_chemical_voxel_key(clamp(vec3<i32>(cell.pos / (U.TissueVoxelSize * 2.0)), vec3<i32>(0, 0, 0), vec3<i32>(i32(U.TissueGridDimX / 2u) - 1, i32(U.TissueGridDimY / 2u) - 1, i32(U.TissueGridDimZ / 2u) - 1)))];
    tissue = Tissue_In[z_order_hash(clamp(vec3<i32>(cell.pos / U.TissueVoxelSize), vec3<i32>(0, 0, 0), vec3<i32>(i32(U.TissueGridDimX) - 1, i32(U.TissueGridDimY) - 1, i32(U.TissueGridDimZ) - 1)))];
    c_pp = clamp(chem.pdgf, 0.0, 1000000.0);
    if ((U.treatment_ap != 0)) {
        if ((cell.cell_type != 0)) {
            if ((cell.p_pot >= AP_THRESHOLD)) {
                cell.pos = vec3<f32>(f32(U.TissueGridDimX), f32(U.TissueGridDimY), f32(U.TissueGridDimZ)) * U.TissueVoxelSize + vec3<f32>(U.TissueVoxelSize) / 2;
                return;
            }
        }
    }
    if ((cell.cell_type == 0)) {
        if ((c_pp > ACTIVATION_THR)) {
            cell.cell_type = 2;
            cell.persistence_timer = TAU_STOP;
        }
        return;
    }
    gamma = 0.0;
    if ((cell.cell_type == 1)) {
        gamma = _gamma_infected(c_pp);
    } else {
        gamma = _gamma_recruited(c_pp);
    }
    p = (cell.p_pot * gamma);
    m = (cell.m_pot * gamma);
    if ((U.treatment_am != 0)) {
        m = (m * AM_FACTOR);
    }
    n_local = 0;
    {
        let _my_voxel = vec3<i32>(cell.pos / U.TissueVoxelSize);
        for (var _k0 = 0u; _k0 < 27u; _k0 = _k0 + 1u) {
            let _nv1 = _my_voxel + MooreOffsets[_k0];
            if (!is_tissue_coord_in_bounds(_nv1)) { continue; }
            let _nkey2 = z_order_hash(_nv1);
            let _ns3 = VoxelTable[_nkey2].startIndex;
            let _ne4 = _ns3 + atomicLoad(&VoxelTable[_nkey2].count);
            for (var _j5 = _ns3; _j5 < _ne4; _j5 = _j5 + 1u) {
                if (_j5 == cell_index) { continue; }
                var nb = Cells_In[_j5];
                if ((nb.cell_type != 0)) {
                    n_local += 1;
                }
            }
        }
    }
    is_quiescent = 0;
    if ((n_local >= tissue.carrying_capacity)) {
        is_quiescent = 1;
    }
    if ((is_quiescent == 0)) {
        if ((p > 0.0)) {
            cell.div_clock += p;
            if ((cell.div_clock >= 1.0)) {
                cell.div_clock -= 1.0;
                let _rand_0 = _next_rand(&_rng);
                div_a = (_rand_0 * 6.2831853);
                nx = clamp((cell.pos.x + (0.25 * cos(div_a))), 0.0, 145.99);
                ny = clamp((cell.pos.y + (0.25 * sin(div_a))), 0.0, 99.99);
                let _rand_1 = _next_rand(&_rng);
                d_rng = clamp(_rand_1, 0.0001, 1.0);
                d_timer = clamp((-log(d_rng) * TAU_STOP), 0.1, 100.0);
                let _rand_2 = _next_rand(&_rng);
                daughter = Cell();
                daughter.pos = vec3<f32>(nx, ny, 0.5);
                daughter.cell_type = cell.cell_type;
                daughter.p_pot = cell.p_pot;
                daughter.m_pot = cell.m_pot;
                daughter.div_clock = (_rand_2 * 0.3);
                daughter.is_moving = 0;
                daughter.persistence_timer = d_timer;
                daughter.prev_x = nx;
                daughter.prev_y = ny;
                let _spawn_idx6 = atomicAdd(&State.NewTotalAgents, 1u);
                if (_spawn_idx6 < arrayLength(&Cells_Out)) {
                    Cells_Out[_spawn_idx6] = daughter;
                }
            }
        }
    }
    cell.persistence_timer -= 1.0;
    if ((cell.persistence_timer <= 0.0)) {
        cell.is_moving = i32(!((cell.is_moving != 0)));
        if ((cell.is_moving != 0)) {
            let _rand_3 = _next_rand(&_rng);
            go_u = clamp(_rand_3, 0.0001, 1.0);
            cell.persistence_timer = clamp((-log(go_u) * TAU_MOVE), 0.1, 100.0);
            angle = 0.0;
            if ((tissue.is_white_matter != 0)) {
                let _rand_4 = _next_rand(&_rng);
                bm_u1 = clamp(_rand_4, 0.0001, 1.0);
                let _rand_5 = _next_rand(&_rng);
                bm_u2 = _rand_5;
                bm_z = (sqrt((-2.0 * log(bm_u1))) * cos((6.2831853 * bm_u2)));
                base = atan2(tissue.tract_dir_y, tissue.tract_dir_x);
                angle = (base + (bm_z * 0.52359877));
            } else {
                let _rand_6 = _next_rand(&_rng);
                angle = (_rand_6 * 6.2831853);
            }
            cell.move_dir_x = cos(angle);
            cell.move_dir_y = sin(angle);
        } else {
            let _rand_7 = _next_rand(&_rng);
            stop_u = clamp(_rand_7, 0.0001, 1.0);
            cell.persistence_timer = clamp((-log(stop_u) * TAU_STOP), 0.1, 100.0);
        }
    }
    cell.prev_x = cell.pos.x;
    cell.prev_y = cell.pos.y;
    if (((cell.is_moving != 0)) && ((is_quiescent == 0)) && ((m > 0.0))) {
        cell.pos.x = clamp((cell.pos.x + (cell.move_dir_x * m)), 0.0, 145.99);
        cell.pos.y = clamp((cell.pos.y + (cell.move_dir_y * m)), 0.0, 99.99);
    }
    cell._rng_state = _rng;
    Cells_Out[cell_index] = cell;
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_ChemRule_0(@builtin(global_invocation_id) _id: vec3<u32>) {
    let chem_index = _id.x;
    if (chem_index >= U.NumVoxelTableEntries << 3) { return; }
    var chem = Chemistry_In[chem_index];
    var _rng = chem._rng_state;
    var secretion: f32;
    var consumption: f32;
    var c: Cell;
    var lap: f32;
    var nb: Chemistry;
    var exch: f32;
    var lap_term: f32;
    var decay_term: f32;
    var new_val: f32;
    secretion = 0.0;
    consumption = 0.0;
    {
        let _tstart1 = chem_index * 8u;
        for (var _i0 = 0u; _i0 < 8u; _i0 = _i0 + 1u) {
            let _tkey3 = _tstart1 + _i0;
            let _ts4 = VoxelTable[_tkey3].startIndex;
            let _te5 = _ts4 + atomicLoad(&VoxelTable[_tkey3].count);
            for (var _j2 = _ts4; _j2 < _te5; _j2 = _j2 + 1u) {
                var c = Cells_In[_j2];
                if ((c.cell_type == 1)) {
                    secretion += PDGF_SECRETE;
                }
                if ((c.cell_type != 0)) {
                    consumption += PDGF_CONSUME;
                }
            }
        }
    }
    lap = 0.0;
    {
        for (var _k6 = 0u; _k6 < 27u; _k6 = _k6 + 1u + u32(_k6 == 13)) {
            let _nc7 = chem._coord + MooreOffsets[_k6];
            if (!is_chem_coord_in_bounds(_nc7)) { continue; }
            let _nkey8 = get_chemical_voxel_key(_nc7);
            var nb = Chemistry_In[_nkey8];
            lap += (nb.pdgf - chem.pdgf);
        }
    }
    exch = (chem.pdgf + secretion);
    exch = (exch - consumption);
    lap_term = ((PDGF_D * 0.5) * lap);
    exch = (exch + lap_term);
    decay_term = (1.0 - PDGF_DECAY);
    new_val = (exch * decay_term);
    chem.pdgf = clamp(new_val, 0.0, 1000000.0);
    chem._rng_state = _rng;
    Chemistry_Out[chem_index] = chem;
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_MetricRule_0(@builtin(global_invocation_id) _id: vec3<u32>) {
    let cell_index = _id.x;
    if (cell_index >= State.TotalAgents) { return; }
    let cell = Cells_In[cell_index];
    atomicAdd(&MetricsBuffer.total_cells, 1);
    if ((cell.cell_type == 1)) {
        atomicAdd(&MetricsBuffer.infected_count, 1);
    }
    if ((cell.cell_type == 2)) {
        atomicAdd(&MetricsBuffer.recruited_count, 1);
    }
    if ((cell.cell_type == 0)) {
        atomicAdd(&MetricsBuffer.inactive_count, 1);
    }
}
