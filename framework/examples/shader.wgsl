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
const TWO_PI: f32 = 6.2831853;
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
    total_alive: atomic<i32>,
    infected_count: atomic<i32>,
    recruited_count: atomic<i32>,
    inactive_count: atomic<i32>,
    total_cells: atomic<i32>,
    total_pdgf: f32,
};

struct Tuple_f32_f32 {
    get_0: f32,
    get_1: f32,
}

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

fn _interleave_9bit(input: u32) -> u32 {
    var x = input & 0x1ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8))  & 0x0300f00fu;
    x = (x | (x << 4))  & 0x030c30c3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}

fn _get_chemical_voxel_key(large_coord: vec3<i32>) -> u32 {
    let x = _interleave_9bit(u32(large_coord.x));
    let y = _interleave_9bit(u32(large_coord.y));
    let z = _interleave_9bit(u32(large_coord.z));
    return x | (y << 1) | (z << 2);
}

fn _get_sub_voxel_index(small_coord: vec3<i32>) -> u32 {
    let x = u32(small_coord.x & 1);
    let y = u32(small_coord.y & 1);
    let z = u32(small_coord.z & 1);
    return x | (y << 1) | (z << 2);
}

fn _get_tissue_voxel_key(small_coord: vec3<i32>) -> u32 {
    let large_coord = small_coord / 2;
    let j = _get_chemical_voxel_key(large_coord);
    let i = _get_sub_voxel_index(small_coord);
    return (j << 3u) | i;
}

fn _z_order_hash(coord: vec3<i32>) -> u32 {
    return _get_tissue_voxel_key(coord);
}

fn _is_tissue_coord_in_bounds(coord: vec3<i32>) -> bool {
    return u32(coord.x) < U.TissueGridDimX &&
           u32(coord.y) < U.TissueGridDimY &&
           u32(coord.z) < U.TissueGridDimZ;
}

fn _is_chem_coord_in_bounds(coord: vec3<i32>) -> bool {
    return u32(coord.x) < (U.TissueGridDimX / 2u) &&
           u32(coord.y) < (U.TissueGridDimY / 2u) &&
           u32(coord.z) < (U.TissueGridDimZ / 2u);
}

fn cell_attempt_division(_self: ptr<function, Cell>, p: f32, _rng_state: ptr<function, u32>) {
    if ((p <= 0.0)) {
        return;
    }
    (*_self).div_clock += p;
    if (((*_self).div_clock < 1.0)) {
        return;
    }
    (*_self).div_clock -= 1.0;
    cell_spawn_daughter((*_self), _rng_state);
}

fn cell_enter_moving_state(_self: ptr<function, Cell>, tissue: Tissue, _rng_state: ptr<function, u32>) {
    var rng_u: f32;
    var angle: f32;
    rng_u = clamp(_next_rand(_rng_state), 0.0001, 1.0);
    (*_self).persistence_timer = clamp((-log(rng_u) * TAU_MOVE), 0.1, 100.0);
    angle = cell_sample_direction((*_self), tissue, _rng_state);
    (*_self).move_dir_x = cos(angle);
    (*_self).move_dir_y = sin(angle);
}

fn cell_enter_stopped_state(_self: ptr<function, Cell>, _rng_state: ptr<function, u32>) {
    var rng_u: f32;
    rng_u = clamp(_next_rand(_rng_state), 0.0001, 1.0);
    (*_self).persistence_timer = clamp((-log(rng_u) * TAU_STOP), 0.1, 100.0);
}

fn cell_move(_self: ptr<function, Cell>, m: f32, _rng_state: ptr<function, u32>) {
    (*_self).prev_x = (*_self).pos.x;
    (*_self).prev_y = (*_self).pos.y;
    (*_self).pos.x = clamp(((*_self).pos.x + ((*_self).move_dir_x * m)), 0.0, 599.9);
    (*_self).pos.y = clamp(((*_self).pos.y + ((*_self).move_dir_y * m)), 0.0, 599.9);
}

fn cell_phenotype_gamma_infected(_self: Cell, c_pp: f32, _rng_state: ptr<function, u32>) -> f32 {
    var num: f32;
    var den: f32;
    num = (C_PA + c_pp);
    den = ((C_PA + c_pp) + K_HILL);
    return (num / den);
}

fn cell_phenotype_gamma_recruited(_self: Cell, c_pp: f32, _rng_state: ptr<function, u32>) -> f32 {
    return (c_pp / (c_pp + (BETA * K_HILL)));
}

fn cell_sample_direction(_self: Cell, tissue: Tissue, _rng_state: ptr<function, u32>) -> f32 {
    var u1: f32;
    var u2: f32;
    var z: f32;
    var base: f32;
    if ((tissue.is_white_matter != 0)) {
        u1 = clamp(_next_rand(_rng_state), 0.0001, 1.0);
        u2 = _next_rand(_rng_state);
        z = (sqrt((-2.0 * log(u1))) * cos((TWO_PI * u2)));
        base = atan2(tissue.tract_dir_y, tissue.tract_dir_x);
        return (base + (z * 0.52359877));
    }
    return (_next_rand(_rng_state) * TWO_PI);
}

fn cell_save_position(_self: ptr<function, Cell>, _rng_state: ptr<function, u32>) {
    (*_self).prev_x = (*_self).pos.x;
    (*_self).prev_y = (*_self).pos.y;
}

fn cell_spawn_daughter(_self: Cell, _rng_state: ptr<function, u32>) {
    var angle: f32;
    var nx: f32;
    var ny: f32;
    var rng_u: f32;
    var d_timer: f32;
    var daughter: Cell;
    angle = (_next_rand(_rng_state) * TWO_PI);
    nx = clamp((_self.pos.x + (5.0 * cos(angle))), 0.0, 599.9);
    ny = clamp((_self.pos.y + (5.0 * sin(angle))), 0.0, 599.9);
    rng_u = clamp(_next_rand(_rng_state), 0.0001, 1.0);
    d_timer = clamp((-log(rng_u) * TAU_STOP), 0.1, 100.0);
    daughter = Cell();
    let _spawn_idx0 = atomicAdd(&State.NewTotalAgents, 1u);
    if (_spawn_idx0 < arrayLength(&Cells_Out)) {
        Cells_Out[_spawn_idx0] = daughter;
    }
}

fn cell_tick_persistence(_self: ptr<function, Cell>, tissue: Tissue, _rng_state: ptr<function, u32>) {
    (*_self).persistence_timer -= 1.0;
    if (((*_self).persistence_timer > 0.0)) {
        return;
    }
    (*_self).is_moving = i32(!(((*_self).is_moving != 0)));
    if (((*_self).is_moving != 0)) {
        cell_enter_moving_state(_self, tissue, _rng_state);
    } else {
        cell_enter_stopped_state(_self, _rng_state);
    }
}

fn cell_try_activate(_self: ptr<function, Cell>, c_pp: f32, _rng_state: ptr<function, u32>) -> i32 {
    if ((c_pp > ACTIVATION_THR)) {
        (*_self).cell_type = TYPE_RECRUITED;
        (*_self).persistence_timer = TAU_STOP;
        return true;
    }
    return false;
}

fn tissue_is_overcrowded(_self: Tissue, active_neighbor_count: i32, _rng_state: ptr<function, u32>) -> i32 {
    return (active_neighbor_count >= _self.carrying_capacity);
}

fn chemistry_compute_laplacian(_self: Chemistry, _rng_state: ptr<function, u32>) -> f32 {
    var lap: f32;
    var nb: Chemistry;
    lap = 0.0;
    return lap;
}

fn chemistry_step_pdgf(_self: ptr<function, Chemistry>, _rng_state: ptr<function, u32>) {
    var lap: f32;
    var new_val: f32;
    var _tuple_0: i32;
    _tuple_0 = chemistry_compute_secretion_consumption((*_self), _rng_state);
    lap = chemistry_compute_laplacian((*_self), _rng_state);
    new_val = (((((*_self).pdgf + _tuple_0.get_0) - _tuple_0.get_1) + ((PDGF_D * 0.5) * lap)) * (1.0 - PDGF_DECAY));
    (*_self).pdgf = clamp(new_val, 0.0, 1000000.0);
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
    let z_key = _z_order_hash(vec3<i32>(cell.pos / U.TissueVoxelSize));

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
    var active_neighbors: i32;
    var nb: Cell;
    var is_quiescent: i32;
    var _tuple_0: i32;
    chem = Chemistry_In[_get_chemical_voxel_key(clamp(vec3<i32>(cell.pos / (U.TissueVoxelSize * 2.0)), vec3<i32>(0, 0, 0), vec3<i32>(i32(U.TissueGridDimX / 2u) - 1, i32(U.TissueGridDimY / 2u) - 1, i32(U.TissueGridDimZ / 2u) - 1)))];
    tissue = Tissue_In[_z_order_hash(clamp(vec3<i32>(cell.pos / U.TissueVoxelSize), vec3<i32>(0, 0, 0), vec3<i32>(i32(U.TissueGridDimX) - 1, i32(U.TissueGridDimY) - 1, i32(U.TissueGridDimZ) - 1)))];
    c_pp = clamp(chem.pdgf, 0.0, 1000000.0);
    if (((U.treatment_ap != 0)) && ((cell.cell_type != TYPE_INACTIVE))) {
        if ((cell.p_pot >= AP_THRESHOLD)) {
            cell.pos = vec3<f32>(f32(U.TissueGridDimX), f32(U.TissueGridDimY), f32(U.TissueGridDimZ)) * U.TissueVoxelSize + vec3<f32>(U.TissueVoxelSize) / 2;
            cell._rng_state = _rng;
            Cells_Out[cell_index] = cell;
            return;
        }
    }
    if ((cell.cell_type == TYPE_INACTIVE)) {
        cell_try_activate(&cell, c_pp, &_rng);
        cell._rng_state = _rng;
        Cells_Out[cell_index] = cell;
        return;
    }
    p = 0.0;
    m = 0.0;
    _tuple_0 = cell_compute_phenotype(cell, c_pp, &_rng);
    active_neighbors = 0;
    {
        let _my_voxel = vec3<i32>(cell.pos / U.TissueVoxelSize);
        for (var _k1 = 0u; _k1 < 27u; _k1 = _k1 + 1u) {
            let _nv2 = _my_voxel + MooreOffsets[_k1];
            if (!_is_tissue_coord_in_bounds(_nv2)) { continue; }
            let _nkey3 = _z_order_hash(_nv2);
            let _ns4 = VoxelTable[_nkey3].startIndex;
            let _ne5 = _ns4 + atomicLoad(&VoxelTable[_nkey3].count);
            for (var _j6 = _ns4; _j6 < _ne5; _j6 = _j6 + 1u) {
                if (_j6 == cell_index) { continue; }
                var nb = Cells_In[_j6];
                if ((nb.cell_type != TYPE_INACTIVE)) {
                    active_neighbors += 1;
                }
            }
        }
    }
    is_quiescent = tissue_is_overcrowded(tissue, active_neighbors, &_rng);
    if (!((is_quiescent != 0))) {
        cell_attempt_division(&cell, _tuple_0.get_0, &_rng);
    }
    cell_tick_persistence(&cell, tissue, &_rng);
    cell_save_position(&cell, &_rng);
    if (((cell.is_moving != 0)) && (!((is_quiescent != 0))) && ((_tuple_0.get_1 > 0.0))) {
        cell_move(&cell, _tuple_0.get_1, &_rng);
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
    chemistry_step_pdgf(&chem, &_rng);
    chem._rng_state = _rng;
    Chemistry_Out[chem_index] = chem;
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_MetricRule_0(@builtin(global_invocation_id) _id: vec3<u32>) {
    let cell_index = _id.x;
    if (cell_index >= State.TotalAgents) { return; }
    let cell = Cells_In[cell_index];
    atomicAdd(&MetricsBuffer.total_cells, 1);
    if ((cell.cell_type == TYPE_INFECTED)) {
        atomicAdd(&MetricsBuffer.infected_count, 1);
    }
    if ((cell.cell_type == TYPE_RECRUITED)) {
        atomicAdd(&MetricsBuffer.recruited_count, 1);
    }
    if ((cell.cell_type == TYPE_INACTIVE)) {
        atomicAdd(&MetricsBuffer.inactive_count, 1);
    }
}
