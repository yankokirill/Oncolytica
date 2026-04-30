// ===========================================================================
// КОНСТАНТЫ ИЗ PYTHON
// ===========================================================================
const ENV_EMPTY: i32 = 0;
const ENV_GRAY: i32  = 1;
const ENV_WHITE: i32 = 2;

const TYPE_INFECTED: i32  = 1;
const TYPE_RECRUITED: i32 = 2;

const STATE_DELETED: i32   = 0;
const STATE_ACTIVE: i32    = 1;
const STATE_QUIESCENT: i32 = 2;

const KAPPA_GRAY: f32  = 16.0;
const KAPPA_WHITE: f32 = 10.66; 

const PDGF_DECAY_RATE: f32       = 0.1;
const PDGF_SECRETION_RATE: f32   = 50.0;
const PDGF_CONSUMPTION_RATE: f32 = 10.0;
const RECRUITMENT_THRESHOLD: f32 = 0.0005;

const C_PA: f32   = 10.0;
const K_HALF: f32 = 50.0;
const BETA: f32   = 0.5;

const AP_DRUG_ACTIVE: i32 = 0; // Транслировано как i32 (0 - False, 1 - True)
const AP_DRUG_THRESHOLD: f32 = 0.0166666; // 1.0 / 60.0

const MAX_AGENTS: i32         = 100000;
const MAX_CELL_PER_VOXEL: i32 = 16;

// ===========================================================================
// СТРУКТУРЫ ДАННЫХ
// ===========================================================================
// Сгенерировано из `GliomaCell`. Соблюдено выравнивание WGSL (по 16 байт)
struct CellData {
    position: vec3<f32>,
    type_id: i32,       // Занимает 4-й слот (16 байт)
    
    dir: vec3<f32>,
    state: i32,         // (16 байт)
    
    p_pot: f32,
    m_pot: f32,
    cycle_timer: f32,
    gamma_response: f32,// (16 байт)
    
    rngState: u32,
    _pad: vec3<i32>     // (16 байт)
}

struct GlobalUniforms {
    DeltaTime: f32, TimeTime: f32, DomainMin: vec3<f32>, DomainMax: vec3<f32>,
    SpatialGridDim: i32, VoxelGridDim: i32, SpatialVoxelSize: f32, _pad: i32,
}

// ===========================================================================
// БИНДИНГИ
// ===========================================================================
@group(0) @binding(0) var<uniform> U: GlobalUniforms;
@group(0) @binding(1) var EnvGrid_R: texture_3d<i32>;
@group(0) @binding(2) var EnvGrid_W: texture_storage_3d<r32sint, read_write>;
@group(0) @binding(3) var FieldGrid_R: texture_3d<f32>;
@group(0) @binding(4) var FieldGrid_W: texture_storage_3d<rgba32float, read_write>;

@group(1) @binding(0) var<storage, read> Cells_Read: array<CellData>;
@group(1) @binding(1) var<storage, read_write> Cells_Write: array<CellData>;
@group(1) @binding(2) var<storage, read_write> NextAgentIndex: atomic<i32>; 

@group(2) @binding(0) var<storage, read_write> SpatialCellCount: array<atomic<i32>>;
@group(2) @binding(1) var<storage, read_write> SpatialCellSlots: array<i32>;

// ===========================================================================
// СИСТЕМНЫЕ ФУНКЦИИ И ХЕШИРОВАНИЕ
// ===========================================================================
struct RandState { result: f32, rng: u32 }
fn NextRand(x: u32) -> RandState {
    var val = x; val = val ^ (val << 13u); val = val ^ (val >> 17u); val = val ^ (val << 5u);
    return RandState(f32(val) * 2.3283064365386963e-10, val);
}

// Транслятор добавил функцию нормального распределения (Box-Muller)
fn NextRandNormal(rng: u32, mean: f32, std_dev: f32) -> RandState {
    let r1 = NextRand(rng);
    let r2 = NextRand(r1.rng);
    let u1 = max(r1.result, 1e-7); 
    let z0 = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * r2.result);
    return RandState(mean + z0 * std_dev, r2.rng);
}

fn WorldToSpatialCell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(clamp((pos - U.DomainMin) / U.SpatialVoxelSize, vec3<f32>(0.0), vec3<f32>(f32(U.SpatialGridDim - 1))));
}
fn SpatialCellIndex(cell: vec3<i32>) -> i32 {
    return cell.x + cell.y * U.SpatialGridDim + cell.z * U.SpatialGridDim * U.SpatialGridDim;
}

// ===========================================================================
// ФАЗА 1: СЛОЙ СРЕДЫ (Environment)
// ===========================================================================
@compute @workgroup_size(8, 8, 8)
fn Kernel_EnvironmentRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = u32(U.VoxelGridDim);
    if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
    let voxel_coord = vec3<i32>(id);
    let env_type_id = textureLoad(EnvGrid_R, voxel_coord, 0).r;
    
    // Плотность автоматически рассчитывается и хранится в SpatialCellCount 
    // Записываем тип среды обратно (ткань статична в этой модели)
    textureStore(EnvGrid_W, voxel_coord, vec4<i32>(env_type_id, 0, 0, 0));
}

// ===========================================================================
// ФАЗА 2: СЛОЙ ПОЛЕЙ (Fields)
// ===========================================================================
@compute @workgroup_size(8, 8, 8)
fn Kernel_ReactionRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = u32(U.VoxelGridDim);
    if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
    
    let voxel_coord = vec3<i32>(id);
    var fields = textureLoad(FieldGrid_R, voxel_coord, 0);
    var pdgf = fields.r;

    // === INJECT: @reaction_rule process_pdgf ===
    pdgf -= pdgf * PDGF_DECAY_RATE * U.DeltaTime;
    
    let sidx = SpatialCellIndex(voxel_coord); 
    let cell_cnt = min(atomicLoad(&SpatialCellCount[sidx]), MAX_CELL_PER_VOXEL);
    
    for (var t = 0; t < cell_cnt; t++) {
        let c = Cells_Read[SpatialCellSlots[sidx * MAX_CELL_PER_VOXEL + t]];
        if (c.state != STATE_DELETED) {
            if (c.type_id == TYPE_INFECTED) {
                pdgf += PDGF_SECRETION_RATE * U.DeltaTime;
            }
            pdgf -= PDGF_CONSUMPTION_RATE * U.DeltaTime;
        }
    }
    fields.r = max(pdgf, 0.0);
    // ===========================================

    textureStore(FieldGrid_W, voxel_coord, fields);
}

@compute @workgroup_size(8, 8, 8)
fn Kernel_DiffusionRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = u32(U.VoxelGridDim);
    if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
    let v = vec3<i32>(id);
    let c = textureLoad(FieldGrid_R, v, 0);
    
    // === INJECT: @diffusion_rule spread_pdgf ===
    let diff_coeff = 0.15;
    
    // Чтение соседей (с проверкой границ)
    let xp = textureLoad(FieldGrid_R, clamp(v + vec3<i32>(1,0,0), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);
    let xm = textureLoad(FieldGrid_R, clamp(v - vec3<i32>(1,0,0), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);
    let yp = textureLoad(FieldGrid_R, clamp(v + vec3<i32>(0,1,0), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);
    let ym = textureLoad(FieldGrid_R, clamp(v - vec3<i32>(0,1,0), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);
    let zp = textureLoad(FieldGrid_R, clamp(v + vec3<i32>(0,0,1), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);
    let zm = textureLoad(FieldGrid_R, clamp(v - vec3<i32>(0,0,1), vec3<i32>(0), vec3<i32>(i32(dim)-1)), 0);

    let laplacian = xp.r + xm.r + yp.r + ym.r + zp.r + zm.r - 6.0 * c.r;
    let new_pdgf = c.r + diff_coeff * laplacian * U.DeltaTime;
    // ===========================================

    textureStore(FieldGrid_W, v, vec4<f32>(max(new_pdgf, 0.0), c.g, c.b, c.a));
}

// ===========================================================================
// ФАЗА 3: СЛОЙ КЛЕТОК (Update Rules)
// ===========================================================================

@compute @workgroup_size(256, 1, 1)
fn Kernel_UpdateRule_Pass1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _c = Cells_Read[i];
    if (_c.state == STATE_DELETED) { Cells_Write[i] = _c; return; }

    let my_voxel = WorldToSpatialCell(_c.position);
    let my_env = textureLoad(EnvGrid_R, my_voxel, 0).r;
    let my_fields = textureLoad(FieldGrid_R, my_voxel, 0);

    // === INJECT: @update_rule pass_1_calculate_phenotype ===
    let capacity = select(KAPPA_GRAY, KAPPA_WHITE, my_env == ENV_WHITE);
    let local_density = f32(atomicLoad(&SpatialCellCount[SpatialCellIndex(my_voxel)]));

    if (local_density >= capacity) {
        _c.state = STATE_QUIESCENT;
    } else {
        let Cp = my_fields.r; // PDGF field
        
        if (_c.type_id == TYPE_INFECTED) {
            _c.gamma_response = (C_PA + Cp) / (C_PA + Cp + K_HALF);
            _c.state = STATE_ACTIVE;
        } else if (_c.type_id == TYPE_RECRUITED) {
            if (Cp < RECRUITMENT_THRESHOLD) {
                _c.gamma_response = 0.0;
                _c.state = STATE_QUIESCENT;
            } else {
                _c.gamma_response = Cp / (Cp + BETA * K_HALF);
                _c.state = STATE_ACTIVE;
            }
        }
    }
    // =======================================================
    Cells_Write[i] = _c;
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_UpdateRule_Pass2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _c = Cells_Read[i];
    if (_c.state != STATE_ACTIVE) { Cells_Write[i] = _c; return; }

    let my_voxel = WorldToSpatialCell(_c.position);
    let my_env = textureLoad(EnvGrid_R, my_voxel, 0).r;

    // === INJECT: @update_rule pass_2_migration ===
    let actual_speed = _c.m_pot * _c.gamma_response;
    var dir = _c.dir;

    if (my_env == ENV_WHITE) {
        let r1 = NextRandNormal(_c.rngState, 0.0, 0.2);
        let r2 = NextRandNormal(r1.rng, 0.0, 0.2);
        let r3 = NextRandNormal(r2.rng, 0.0, 0.2);
        _c.rngState = r3.rng;
        dir += vec3<f32>(r1.result, r2.result, r3.result);
    } else {
        let r1 = NextRand(_c.rngState);
        let r2 = NextRand(r1.rng);
        let r3 = NextRand(r2.rng);
        _c.rngState = r3.rng;
        dir += vec3<f32>(r1.result * 2.0 - 1.0, r2.result * 2.0 - 1.0, r3.result * 2.0 - 1.0);
    }
    
    dir = normalize(dir + vec3<f32>(1e-8));
    _c.dir = dir;
    _c.position += dir * actual_speed * U.DeltaTime;
    _c.position = clamp(_c.position, U.DomainMin, U.DomainMax);
    // =============================================
    Cells_Write[i] = _c;
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_UpdateRule_Pass3(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _c = Cells_Read[i];
    if (_c.state != STATE_ACTIVE) { Cells_Write[i] = _c; return; }

    // === INJECT: @update_rule pass_3_treatment_and_cycle ===
    let actual_prolif_rate = _c.p_pot * _c.gamma_response;

    if (AP_DRUG_ACTIVE == 1) {
        if (actual_prolif_rate > AP_DRUG_THRESHOLD) {
            _c.state = STATE_DELETED;
            Cells_Write[i] = _c;
            return;
        }
    }
    
    _c.cycle_timer -= actual_prolif_rate * U.DeltaTime;
    // =======================================================
    Cells_Write[i] = _c;
}

// ===========================================================================
// ФАЗА 4: ЖИЗНЕННЫЙ ЦИКЛ (Division Rule)
// ===========================================================================
@compute @workgroup_size(256, 1, 1)
fn Kernel_DivisionRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _c = Cells_Read[i];
    if (_c.state == STATE_DELETED) { Cells_Write[i] = _c; return; }

    var create_child: bool = false;
    var child: CellData;

    // === INJECT: @division_rule lifecycle ===
    if (_c.state == STATE_ACTIVE && _c.cycle_timer <= 0.0) {
        _c.cycle_timer = 1.0; // Сброс родительского таймера
        
        create_child = true;
        child = _c; // Копирование базы (в т.ч. p_pot и m_pot)
        
        child.dir = -_c.dir; // Отскок
        child.cycle_timer = 1.0;
    }
    // ========================================

    if (create_child) {
        let child_idx = u32(atomicAdd(&NextAgentIndex, 1));
        if (child_idx < u32(MAX_AGENTS)) {
            child.rngState = _c.rngState ^ 0x12345678u;
            Cells_Write[child_idx] = child;
        }
    }
    Cells_Write[i] = _c;
}
