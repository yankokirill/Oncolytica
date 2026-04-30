// ===========================================================================
// КОНСТАНТЫ
// ===========================================================================
const STATE_DELETED: i32       = 0;
const STATE_PROLIFERATING: i32 = 1;
const STATE_QUIESCENT: i32     = 2;
const STATE_NECROTIC: i32      = 3;

const TISSUE_ECM: i32    = 0;
const TISSUE_VESSEL: i32 = 1;

const MAX_AGENTS: i32         = 100000;
const GRID_DIM: i32           = 64;
const CELL_RADIUS: f32        = 0.5;
const CELL_DIAMETER: f32      = 1.0;
const MAX_CELL_PER_VOXEL: i32 = 16;
const CONSUMPTION_SCALE: i32  = 10000;
const MAX_SPECIES: i32        = 8;
const ENERGY_DIVIDE_THR: f32  = 5.0;
const ENERGY_MOVE_COST: f32   = 0.05;
const ENERGY_SCALE: f32       = 1000.0;
const MAINTENANCE_COST: f32   = 0.0;

const VELOCITY_DAMPING: f32 = 0.85;
const VELOCITY_IMPULSE: f32 = 0.4;
const CHEMO_FORCE: f32      = 2.0;

const HYPOXIA_METRIC_THR: f32 = 0.1;
const CROWD_THRESHOLD: i32    = 12;

// Индексы метрик
const MA_N_PROLIF: i32    = 0;   
const MA_N_QUIESC: i32    = 1;   
const MA_N_NECR: i32      = 2;   
const MA_N_CSC: i32       = 3;   
const MA_SUM_DX: i32      = 4;
const MA_SUM_DY: i32      = 5;
const MA_SUM_DZ: i32      = 6;
const MA_SUM_R2_LIVE: i32 = 7;   
const MA_SUM_R2_NECR: i32 = 8;   
const MA_MAX_R_LIVE: i32  = 9;   
const MA_MAX_R_NECR: i32  = 10;   
const MA_SUM_ENERGY: i32  = 11;   
const MA_SUM_O2: i32      = 12;   
const MA_SUM_GLU: i32     = 13;   
const MA_SUM_SPEED: i32   = 14;   
const MA_N_CROWDED: i32   = 15;   
const MA_N_HYPOXIC: i32   = 16;   
const MA_COUNT: i32       = 17;
const DIVISION_COUNT: i32 = 18;   

const MO_TOTAL_LIVE: i32     = 0;   
const MO_N_PROLIF: i32       = 1;
const MO_N_QUIESC: i32       = 2;
const MO_N_NECR: i32         = 3;
const MO_FRAC_PROLIF: i32    = 4;   
const MO_FRAC_QUIESC: i32    = 5;   
const MO_FRAC_NECR: i32      = 6;   
const MO_FRAC_HYPOXIC: i32   = 7;   
const MO_CSC_COUNT: i32      = 8;
const MO_CSC_FRAC: i32       = 9;   
const MO_CENTROID_X: i32     = 10;   
const MO_CENTROID_Y: i32     = 11;
const MO_CENTROID_Z: i32     = 12;
const MO_R_GYRATION: i32     = 13;   
const MO_R_INVASIVE: i32     = 14;   
const MO_R_NECROTIC: i32     = 15;   
const MO_RIM_THICKNESS: i32  = 16;   
const MO_MEAN_ENERGY: i32    = 17;
const MO_MEAN_O2: i32        = 18;
const MO_MEAN_GLU: i32       = 19;
const MO_MEAN_SPEED: i32     = 20;   
const MO_CROWD_FRAC: i32     = 21;   
const MO_STEP: i32           = 22;  
const MO_COUNT: i32          = 23; 

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

// ===========================================================================
// СТРУКТУРЫ
// ===========================================================================
struct SpeciesParam {
    consumeO2: f32, consumeGlu: f32, hypoxiaDeathThr: f32, baseMoveProb: f32,
    chemoTaxisO2: f32, chemoTaxisGlu: f32, baseDivideTime: f32, probSymmetricDiv: f32,
    maxProliferations: f32, spontaneousDeath: f32, pad1: f32, pad2: f32,
    pad3: vec4<f32>
}

// ВАЖНО: В WGSL vec3 имеет выравнивание 16 байт. 
struct CellData {
    position: vec3<f32>,
    speciesID: i32,
    customParams: vec3<f32>,
    state: i32,
    aliveAge: f32,
    deadAge: f32,
    energy: f32,
    prolifCapacity: i32,
    rngState:  u32,
    _reserved: vec3<i32>
}

struct DivReq {
    parentIdx: i32, childSpeciesID: i32, childProlifCap: i32
}

struct GlobalUniforms {
    DeltaTime: f32,
    StepCount: i32,
    TotalAgents: i32,
    InitialAgents: i32,

    NecroticDecay: f32,
    DiffusionRateO2: f32,
    DiffusionRateGlu: f32,
    DiffusionRateChemo: f32,

    OxygenBoundaryValue: f32,
    GluBoundaryValue: f32,
    ChemoSourceLevel: f32,
    RepulsionStiffness: f32,

    AdhesionStiffness: f32,
    AdhesionRange: f32,
    DampingCoeff: f32,
    VesselRadius: f32,

    VesselCenterWorld: vec3<f32>,
    RequestCount: u32,

    LiveAgentCountBase: u32,
    DomainCenter: vec3<f32>,

    DomainMin: vec3<f32>,
    OxygenGridSize: f32,

    DomainMax: vec3<f32>,
    SpatialGridDim: i32,

    SpatialVoxelSize: f32,
    SpatialVoxelCount: i32,
    NumSpecies: i32,
    pad: i32
}

// ===========================================================================
// БИНДИНГИ (BINDINGS)
// ===========================================================================
// Group 0: Управление и поля
@group(0) @binding(0) var<uniform> U: GlobalUniforms;

@group(0) @binding(1) var samplerFieldGrid_R: sampler;
@group(0) @binding(2) var FieldGrid_R: texture_3d<f32>;
// В WebGPU нужна поддержка read_write_storage_texture или просто write
@group(0) @binding(3) var FieldGrid_W: texture_storage_3d<rgba32float, read_write>; 

@group(0) @binding(4) var EnvironmentMask: texture_3d<i32>;
@group(0) @binding(5) var EnvironmentMaskWrite: texture_storage_3d<r32sint, write>;

// Group 1: Агенты
@group(1) @binding(0) var<storage, read> Cells_Read: array<CellData>;
@group(1) @binding(1) var<storage, read_write> Cells_Write: array<CellData>;
@group(1) @binding(2) var<storage, read> SpeciesTable: array<SpeciesParam>;
@group(1) @binding(3) var<storage, read> InteractionMatrix: array<f32>;

// Group 2: Пространственная сетка и Метрики
@group(2) @binding(0) var<storage, read_write> SpatialCellCount: array<atomic<i32>>;
@group(2) @binding(1) var<storage, read_write> SpatialCellSlots: array<i32>;
@group(2) @binding(2) var<storage, read_write> MetricsAccum: array<atomic<i32>>;
@group(2) @binding(3) var<storage, read_write> MetricsOut: array<f32>;


// ===========================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Из отсутствующих #include)
// ===========================================================================

struct RandState {
    result: f32,
    rng:    u32,
}

// Xorshift (стандарт для GPU)
// FIX: Параметр 'x' неизменяемый. Создаем локальную переменную 'val'.
fn NextRand(x: u32) -> RandState {
    var val = x;
    val = val ^ (val << 13u);
    val = val ^ (val >> 17u);
    val = val ^ (val << 5u);
    return RandState(f32(val) * 2.3283064365386963e-10, val);
}

struct RandVecState {
    result: vec3<f32>,
    rng:    u32,
}

// Векторная версия
fn RandUnitVec(x: u32) -> RandVecState {
    let r1 = NextRand(x);
    let u = r1.result * 2.0 - 1.0;
    
    let r2 = NextRand(r1.rng);
    let theta = r2.result * 6.28318530718;
    
    let r = sqrt(1.0 - u * u);
    return RandVecState(vec3<f32>(r * cos(theta), r * sin(theta), u), r2.rng);
}

fn WorldToSpatialCell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(clamp((pos - U.DomainMin) / U.SpatialVoxelSize, vec3<f32>(0.0), vec3<f32>(f32(U.SpatialGridDim - 1))));
}

fn SpatialCellIndex(cell: vec3<i32>) -> i32 {
    return cell.x + cell.y * U.SpatialGridDim + cell.z * U.SpatialGridDim * U.SpatialGridDim;
}

fn WorldToVoxel(pos: vec3<f32>) -> vec3<i32> {
    let dim = U.DomainMax - U.DomainMin;
    let uvw = (pos - U.DomainMin) / dim;
    return vec3<i32>(clamp(uvw * U.OxygenGridSize, vec3<f32>(0.0), vec3<f32>(U.OxygenGridSize - 1.0)));
}

fn VoxelIndex(x: i32, y: i32, z: i32) -> i32 {
    let gs = i32(U.OxygenGridSize);
    return x + y * gs + z * gs * gs;
}

fn SampleField(pos: vec3<f32>) -> vec4<f32> {
    let dim = U.DomainMax - U.DomainMin;
    let uvw = (pos - U.DomainMin) / dim;
    // textureSampleLevel нужен в compute шейдерах
    return textureSampleLevel(FieldGrid_R, samplerFieldGrid_R, uvw, 0.0);
}

// ===========================================================================
// ЯДРА: МЕТРИКИ
// ===========================================================================
@compute @workgroup_size(17, 1, 1)
fn Kernel_ClearMetrics(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 17u) {
        atomicStore(&MetricsAccum[id.x], 0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_CollectMetrics(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    let c = Cells_Read[i];
    let delta = c.position - U.DomainCenter;
    let r = length(delta);
    
    if (c.state == STATE_DELETED) { return; }

    if (c.state == STATE_NECROTIC) {
        atomicAdd(&MetricsAccum[MA_N_NECR], 1);
        atomicAdd(&MetricsAccum[MA_SUM_R2_NECR], i32(r * r));
        atomicMax(&MetricsAccum[MA_MAX_R_NECR], i32(r * 100.0));
        return;
    }

    let sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];

    if (c.state == STATE_PROLIFERATING) {
        atomicAdd(&MetricsAccum[MA_N_PROLIF], 1);
    } else {
        atomicAdd(&MetricsAccum[MA_N_QUIESC], 1);
    }

    if (sp.probSymmetricDiv < 1.0) {
        atomicAdd(&MetricsAccum[MA_N_CSC], 1);
    }

    atomicAdd(&MetricsAccum[MA_SUM_DX], i32(delta.x * 10.0));
    atomicAdd(&MetricsAccum[MA_SUM_DY], i32(delta.y * 10.0));
    atomicAdd(&MetricsAccum[MA_SUM_DZ], i32(delta.z * 10.0));
    atomicAdd(&MetricsAccum[MA_SUM_R2_LIVE], i32(r * r));
    atomicMax(&MetricsAccum[MA_MAX_R_LIVE], i32(r * 100.0));

    let field = SampleField(c.position);

    atomicAdd(&MetricsAccum[MA_SUM_ENERGY], i32(c.energy * 10.0));
    atomicAdd(&MetricsAccum[MA_SUM_O2], i32(field.r * 1000.0));
    atomicAdd(&MetricsAccum[MA_SUM_GLU], i32(field.g * 1000.0));
    atomicAdd(&MetricsAccum[MA_SUM_SPEED], i32(length(c.customParams) * 1000.0));

    if (field.r < HYPOXIA_METRIC_THR) {
        atomicAdd(&MetricsAccum[MA_N_HYPOXIC], 1);
    }

    let sidx = SpatialCellIndex(WorldToSpatialCell(c.position));
    if (atomicLoad(&SpatialCellCount[sidx]) > CROWD_THRESHOLD) {
        atomicAdd(&MetricsAccum[MA_N_CROWDED], 1);
    }
}

@compute @workgroup_size(1, 1, 1)
fn Kernel_FinalizeMetrics(@builtin(global_invocation_id) id: vec3<u32>) {
    let nProlif = atomicLoad(&MetricsAccum[MA_N_PROLIF]);
    let nQuiesc = atomicLoad(&MetricsAccum[MA_N_QUIESC]);
    let nNecr   = atomicLoad(&MetricsAccum[MA_N_NECR]);
    let nCSC    = atomicLoad(&MetricsAccum[MA_N_CSC]);
    let nHypoxic= atomicLoad(&MetricsAccum[MA_N_HYPOXIC]);
    let nCrowd  = atomicLoad(&MetricsAccum[MA_N_CROWDED]);

    let nLive = nProlif + nQuiesc;
    let nTotal = nLive + nNecr;

    let fLive = max(f32(nLive), 1.0);
    let fTotal = max(f32(nTotal), 1.0);

    MetricsOut[MO_TOTAL_LIVE] = f32(nLive);
    MetricsOut[MO_N_PROLIF]   = f32(nProlif);
    MetricsOut[MO_N_QUIESC]   = f32(nQuiesc);
    MetricsOut[MO_N_NECR]     = f32(nNecr);

    MetricsOut[MO_FRAC_PROLIF]  = f32(nProlif) / fLive;
    MetricsOut[MO_FRAC_QUIESC]  = f32(nQuiesc) / fLive;
    MetricsOut[MO_FRAC_NECR]    = f32(nNecr) / fTotal;
    MetricsOut[MO_FRAC_HYPOXIC] = f32(nHypoxic) / fLive;

    MetricsOut[MO_CSC_COUNT] = f32(nCSC);
    MetricsOut[MO_CSC_FRAC]  = f32(nCSC) / fLive;

    MetricsOut[MO_CENTROID_X] = U.DomainCenter.x + f32(atomicLoad(&MetricsAccum[MA_SUM_DX])) / (fLive * 10.0);
    MetricsOut[MO_CENTROID_Y] = U.DomainCenter.y + f32(atomicLoad(&MetricsAccum[MA_SUM_DY])) / (fLive * 10.0);
    MetricsOut[MO_CENTROID_Z] = U.DomainCenter.z + f32(atomicLoad(&MetricsAccum[MA_SUM_DZ])) / (fLive * 10.0);

    let meanR2 = f32(atomicLoad(&MetricsAccum[MA_SUM_R2_LIVE])) / fLive;
    MetricsOut[MO_R_GYRATION] = sqrt(meanR2);
    
    let rInvasive = f32(atomicLoad(&MetricsAccum[MA_MAX_R_LIVE])) / 100.0;
    
    var rNecrotic = 0.0;
    if (nNecr > 0) {
        rNecrotic = 0.5 * pow(f32(nNecr) / 0.65, 0.333333);
    }

    MetricsOut[MO_R_INVASIVE]    = rInvasive;
    MetricsOut[MO_R_NECROTIC]    = rNecrotic;
    MetricsOut[MO_RIM_THICKNESS] = max(rInvasive - rNecrotic, 0.0);

    MetricsOut[MO_MEAN_ENERGY] = f32(atomicLoad(&MetricsAccum[MA_SUM_ENERGY])) / (fLive * 10.0);
    MetricsOut[MO_MEAN_O2]     = f32(atomicLoad(&MetricsAccum[MA_SUM_O2])) / (fLive * 1000.0);
    MetricsOut[MO_MEAN_GLU]    = f32(atomicLoad(&MetricsAccum[MA_SUM_GLU])) / (fLive * 1000.0);
    MetricsOut[MO_MEAN_SPEED]  = f32(atomicLoad(&MetricsAccum[MA_SUM_SPEED])) / (fLive * 1000.0);
    MetricsOut[MO_CROWD_FRAC]  = f32(nCrowd) / fLive;

    MetricsOut[MO_STEP] = f32(U.StepCount);
}


// ===========================================================================
// ЯДРА: ПОЛЯ (Инициализация и Диффузия)
// ===========================================================================
@compute @workgroup_size(8, 8, 8)
fn Kernel_InitField(@builtin(global_invocation_id) id: vec3<u32>) {
    let gs = u32(U.OxygenGridSize);
    if (id.x >= gs || id.y >= gs || id.z >= gs) { return; }
    textureStore(FieldGrid_W, vec3<i32>(id), vec4<f32>(U.OxygenBoundaryValue, U.GluBoundaryValue, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 8)
fn Kernel_InitEnvironment(@builtin(global_invocation_id) id: vec3<u32>) {
    let gs = i32(U.OxygenGridSize);
    if (i32(id.x) >= gs || i32(id.y) >= gs || i32(id.z) >= gs) { return; }
    
    let uvw = (vec3<f32>(vec3<i32>(id)) + 0.5) / f32(gs);
    let wp = U.DomainMin + uvw * (U.DomainMax - U.DomainMin);
    let dist2D = length(vec2<f32>(wp.x - U.VesselCenterWorld.x, wp.z - U.VesselCenterWorld.z));
    
    let maskVal = select(TISSUE_ECM, TISSUE_VESSEL, dist2D <= U.VesselRadius);
    textureStore(EnvironmentMaskWrite, vec3<i32>(id), vec4<i32>(maskVal, 0, 0, 0));
}

@compute @workgroup_size(8, 8, 8)
fn Kernel_ApplyReaction(@builtin(global_invocation_id) id: vec3<u32>) {
    let gs = i32(U.OxygenGridSize);
    if (i32(id.x) >= gs || i32(id.y) >= gs || i32(id.z) >= gs) { return; }
    
    let vIdx = VoxelIndex(i32(id.x), i32(id.y), i32(id.z));
    let cellCount = atomicLoad(&SpatialCellCount[vIdx]);

    var totalK_O2: f32 = 0.0;
    var totalK_Glu: f32 = 0.0;

    for (var t = 0; t < cellCount; t++) {   
        let agentIdx = SpatialCellSlots[vIdx * MAX_CELL_PER_VOXEL + t];
        let c = Cells_Read[agentIdx];
        if (c.state != STATE_NECROTIC) {
            let sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
            totalK_O2 += sp.consumeO2;
            totalK_Glu += sp.consumeGlu;
        }
    }

    var field = textureLoad(FieldGrid_R, vec3<i32>(id), 0);
    field.r = field.r / (1.0 + totalK_O2 * U.DeltaTime);
    field.g = field.g / (1.0 + totalK_Glu * U.DeltaTime);
    textureStore(FieldGrid_W, vec3<i32>(id), field);
}

@compute @workgroup_size(8, 8, 8)
fn Kernel_DiffusionSolver(@builtin(global_invocation_id) id: vec3<u32>) {
    let gs = u32(U.OxygenGridSize);
    if (id.x >= gs || id.y >= gs || id.z >= gs) { return; }
    let v = vec3<i32>(id);

    if (v.x == 0 || v.x == i32(gs) - 1 || v.y == 0 || v.y == i32(gs) - 1 || v.z == 0 || v.z == i32(gs) - 1) {
        textureStore(FieldGrid_W, v, vec4<f32>(U.OxygenBoundaryValue, U.GluBoundaryValue, 0.0, 0.0));
        return;
    }

    if (textureLoad(EnvironmentMask, v, 0).r == TISSUE_VESSEL) {
        textureStore(FieldGrid_W, v, vec4<f32>(1.0, 1.0, U.ChemoSourceLevel, 0.0));
        return;
    }

    let c = textureLoad(FieldGrid_R, v, 0);
    let xp = textureLoad(FieldGrid_R, v + vec3<i32>(1, 0, 0), 0);
    let xm = textureLoad(FieldGrid_R, v - vec3<i32>(1, 0, 0), 0);
    let yp = textureLoad(FieldGrid_R, v + vec3<i32>(0, 1, 0), 0);
    let ym = textureLoad(FieldGrid_R, v - vec3<i32>(0, 1, 0), 0);
    let zp = textureLoad(FieldGrid_R, v + vec3<i32>(0, 0, 1), 0);
    let zm = textureLoad(FieldGrid_R, v - vec3<i32>(0, 0, 1), 0);

    let lapO2 = xp.r + xm.r + yp.r + ym.r + zp.r + zm.r - 6.0 * c.r;
    let lapGlu = xp.g + xm.g + yp.g + ym.g + zp.g + zm.g - 6.0 * c.g;

    textureStore(FieldGrid_W, v, vec4<f32>(
        clamp(c.r + U.DiffusionRateO2 * lapO2, 0.0, 1.0),
        clamp(c.g + U.DiffusionRateGlu * lapGlu, 0.0, 1.0),
        0.0, 0.0
    ));
}


// ===========================================================================
// ЯДРА: КЛЕТКИ (Метаболизм, Взаимодействие, Деление)
// ===========================================================================
@compute @workgroup_size(256, 1, 1)
fn Kernel_CellUpdate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var c = Cells_Read[i];

    if (c.state == STATE_DELETED) {
        Cells_Write[i] = c;
        return;
    }
    
    if (c.state == STATE_NECROTIC) {
        c.deadAge = min(c.deadAge + U.DeltaTime, U.NecroticDecay);
        Cells_Write[i] = c;
        return;
    }

    let sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
    let field = SampleField(c.position);
    let o2 = field.r;
    let glu = field.g;

    let energyGain = (sp.consumeO2 * clamp(o2, 0.0, 1.0) + sp.consumeGlu * clamp(glu, 0.0, 1.0)) * ENERGY_SCALE;
    c.energy = clamp(c.energy + (energyGain - MAINTENANCE_COST) * U.DeltaTime, 0.0, 100.0);

    let metabolicRate = clamp(o2, 0.0, 1.0) * clamp(glu, 0.0, 1.0);
    c.aliveAge += U.DeltaTime * mix(0.2, 1.0, metabolicRate);

    if (o2 < sp.hypoxiaDeathThr) {
        let deathSeverity = clamp((sp.hypoxiaDeathThr - o2) / sp.hypoxiaDeathThr, 0.0, 1.0);
        let rand = NextRand(c.rngState);
        c.rngState = rand.rng;
        c.state = select(STATE_QUIESCENT, STATE_NECROTIC, rand.result < deathSeverity * U.DeltaTime * 5.0);
    } else if (c.energy < 5.0) {
        c.state = STATE_QUIESCENT;
    } else {
        c.state = STATE_PROLIFERATING;
    }

    Cells_Write[i] = c;
}

fn GetCellRadius(c: CellData) -> f32 {
    let baseRadius = 0.5;
    if (c.state == STATE_NECROTIC) {
        // Лизис: клетка сжимается до 70% радиуса
        let shrinkProgress = clamp(c.deadAge / U.NecroticDecay, 0.0, 1.0);
        return mix(baseRadius, baseRadius * 0.7, shrinkProgress);
    }
    return baseRadius;
}

@compute @workgroup_size(128, 1, 1)
fn Kernel_Interaction(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var c = Cells_Read[i];
    if (c.state == STATE_DELETED) {
        // Записываем обратно, чтобы не появился "мусор" с прошлого кадра
        Cells_Write[i] = c;
        return;
    }

    let isNecrotic = (c.state == STATE_NECROTIC);
    let baseCell = WorldToSpatialCell(c.position);
    let myRadius = GetCellRadius(c);
    
    var F_mech = vec3<f32>(0.0);
    // var survivalProb = 1.0; // Отключено до заполнения матрицы в Rust

    let fallbackRand = RandUnitVec(c.rngState);
    c.rngState = fallbackRand.rng;

    for (var k = 0; k < 27; k++) {
        let nc = clamp(baseCell + MooreOffsets[k], vec3<i32>(0), vec3<i32>(U.SpatialGridDim - 1));
        let ni = SpatialCellIndex(nc);
        // Используем clamp, чтобы избежать выхода за пределы массива, если cnt > MAX_CELL_PER_VOXEL
        let cnt = clamp(atomicLoad(&SpatialCellCount[ni]), 0, MAX_CELL_PER_VOXEL);

        for (var t = 0; t < cnt; t++) {
            let j = SpatialCellSlots[ni * MAX_CELL_PER_VOXEL + t];
            if (j == i32(i)) { continue; }

            let other = Cells_Read[j];
            if (other.state == STATE_DELETED) { continue; }
            
            // InteractionMatrix временно отключена
            // if (!isNecrotic) { ... }

            // МЕХАНИКА
            let delta = c.position - other.position;
            let dist = length(delta);

            if (dist < 1e-5) {
                F_mech += fallbackRand.result * U.RepulsionStiffness * 0.1;
                continue;
            }

            let otherRadius = GetCellRadius(other);
            let interactionDist = myRadius + otherRadius;
            let n_ij = delta / dist;

            if (dist < interactionDist) { // РЕПУЛЬСИЯ
                let overlap = interactionDist - dist;
                let currentRepulsion = select(U.RepulsionStiffness, U.RepulsionStiffness * 0.3, isNecrotic || other.state == STATE_NECROTIC);
                let F_rep = currentRepulsion * overlap * sqrt(overlap);
                F_mech += n_ij * F_rep;
            } else if (!isNecrotic && other.state != STATE_NECROTIC) { // АДГЕЗИЯ
                let adgDistFar = interactionDist + U.AdhesionRange;
                if (dist < adgDistFar) {
                    let gap = dist - interactionDist;
                    let falloff = 0.5 * (1.0 + cos(3.14159 * gap / U.AdhesionRange));
                    let F_adh = U.AdhesionStiffness * falloff; // В референсе нет * gap
                    F_mech -= n_ij * F_adh;
                }
            }
        }
    }

    // Проверка выживания (временно отключена)
    // let death_check = NextRand(c.rngState);
    // c.rngState = death_check.rng;
    // if (death_check.result > survivalProb) { c.state = STATE_NECROTIC; }

    var F_chemo = vec3<f32>(0.0);
    var F_rand = vec3<f32>(0.0);

    if (c.state != STATE_NECROTIC) {
        let vox = WorldToVoxel(c.position);
        let sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
        
        // Градиент полей
        let xp_o2 = textureLoad(FieldGrid_R, vox + vec3<i32>(1,0,0), 0).r; let xp_glu = textureLoad(FieldGrid_R, vox + vec3<i32>(1,0,0), 0).g;
        let xm_o2 = textureLoad(FieldGrid_R, vox - vec3<i32>(1,0,0), 0).r; let xm_glu = textureLoad(FieldGrid_R, vox - vec3<i32>(1,0,0), 0).g;
        let yp_o2 = textureLoad(FieldGrid_R, vox + vec3<i32>(0,1,0), 0).r; let yp_glu = textureLoad(FieldGrid_R, vox + vec3<i32>(0,1,0), 0).g;
        let ym_o2 = textureLoad(FieldGrid_R, vox - vec3<i32>(0,1,0), 0).r; let ym_glu = textureLoad(FieldGrid_R, vox - vec3<i32>(0,1,0), 0).g;
        let zp_o2 = textureLoad(FieldGrid_R, vox + vec3<i32>(0,0,1), 0).r; let zp_glu = textureLoad(FieldGrid_R, vox + vec3<i32>(0,0,1), 0).g;
        let zm_o2 = textureLoad(FieldGrid_R, vox - vec3<i32>(0,0,1), 0).r; let zm_glu = textureLoad(FieldGrid_R, vox - vec3<i32>(0,0,1), 0).g;
        
        let gradO2 = 0.5 * vec3<f32>(xp_o2 - xm_o2, yp_o2 - ym_o2, zp_o2 - zm_o2);
        let gradGlu = 0.5 * vec3<f32>(xp_glu - xm_glu, yp_glu - ym_glu, zp_glu - zm_glu);
        
        // Хемотаксис (ПРОПОРЦИОНАЛЬНЫЙ)
        let chemoDir = gradO2 * sp.chemoTaxisO2 + gradGlu * sp.chemoTaxisGlu;
        F_chemo = chemoDir * CHEMO_FORCE;

        // Случайное движение
        let r_vec = RandUnitVec(c.rngState);
        c.rngState = r_vec.rng;
        F_rand = r_vec.result * sp.baseMoveProb * VELOCITY_IMPULSE;
    }

    // ИНТЕГРИРОВАНИЕ
    var vel = c.customParams;
    let damping = select(U.DampingCoeff, U.DampingCoeff * 2.0, isNecrotic);
    let F_total = F_mech + F_chemo + F_rand - vel * damping;
    vel += F_total * U.DeltaTime;

    let speed = length(vel);
    let maxSpeed = U.SpatialVoxelSize * 0.8;
    if (speed > maxSpeed) {
        vel = (vel / speed) * maxSpeed;
    }

    c.position += vel * U.DeltaTime;
    c.position = clamp(c.position, U.DomainMin + 0.1, U.DomainMax - 0.1);
    c.customParams = vel;

    if (!isNecrotic) {
        c.energy -= ENERGY_MOVE_COST * speed * U.DeltaTime;
        c.energy = max(0.0, c.energy); 
    }

    Cells_Write[i] = c;
}

struct RandomOffsetState {
    result: vec3<f32>,
    rng: u32,
}

fn RandomOffset(state: u32) -> RandomOffsetState {
    let unitDirState = RandUnitVec(state);
    return RandomOffsetState(unitDirState.result * (CELL_DIAMETER * 0.75), unitDirState.rng);
}


@compute @workgroup_size(128, 1, 1)
fn Kernel_Division(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= u32(MAX_AGENTS)) { return; }

    var c = Cells_Read[idx];

    let sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
    if (c.state == STATE_PROLIFERATING &&
        c.aliveAge >= sp.baseDivideTime &&
        c.energy >= ENERGY_DIVIDE_THR) {
        
        var rand = NextRand(c.rngState);
        c.rngState = rand.rng;
        let isStem = (sp.probSymmetricDiv < 1.0);

        if (!isStem && (c.prolifCapacity <= 0 || 
           (sp.spontaneousDeath > 0.0 && rand.result < sp.spontaneousDeath))) {
            c.state = STATE_NECROTIC;
        } else {
            // 1. Обновляем родителя
            c.aliveAge = 0.0;
            c.energy *= 0.5;
            
            var childSpecies = c.speciesID;
            var childCap = c.prolifCapacity;

            if (isStem) {
                let rand_div = NextRand(c.rngState);
                c.rngState = rand_div.rng;
                let symmetric = rand_div.result < sp.probSymmetricDiv;
                if (!symmetric) {
                    childSpecies = 0;
                    childCap = i32(SpeciesTable[0].maxProliferations);
                }
            } else {
                c.prolifCapacity -= 1;
                childCap = c.prolifCapacity;
            }

            // 2. Формируем ребенка
            let offset = RandomOffset(c.rngState);
            c.rngState = offset.rng;

            var child: CellData;
            child.position = c.position + offset.result;
            child.speciesID = childSpecies;
            child.state = STATE_PROLIFERATING;
            child.aliveAge = 0.0;
            child.deadAge = 0.0;
            child.energy = c.energy;
            child.prolifCapacity = childCap;
            child.customParams = vec3<f32>(0.0);
            child.rngState = c.rngState ^ 0x12345678u;
            child._reserved = vec3<i32>(0);

            // 3. Выделяем место под ребенка в конце массива и записываем
            let child_idx = u32(atomicAdd(&MetricsAccum[DIVISION_COUNT], 1));
            if (child_idx < u32(MAX_AGENTS)) {
                Cells_Write[child_idx] = child;
            }
        }
    }

    // Всегда записываем живую/мертвую клетку обратно в ЕЁ ЖЕ слот
    Cells_Write[idx] = c;
}

// ===========================================================================
// ЯДРА: ПРОСТРАНСТВЕННАЯ СЕТКА
// ===========================================================================
@compute @workgroup_size(256, 1, 1)
fn SpatialGrid_Clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u32(U.SpatialVoxelCount)) {
        atomicStore(&SpatialCellCount[id.x], 0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn SpatialGrid_Insert_Interaction(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    let c = Cells_Read[i];
    if (c.state == STATE_DELETED) { return; }

    let voxel = SpatialCellIndex(WorldToSpatialCell(c.position));
    
    let slot = atomicAdd(&SpatialCellCount[voxel], 1);
    if (slot < MAX_CELL_PER_VOXEL) {
        SpatialCellSlots[voxel * MAX_CELL_PER_VOXEL + slot] = i32(i);
    }
}

@compute @workgroup_size(256, 1, 1)
fn SpatialGrid_Insert_Field(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    let c = Cells_Read[i];
    if (c.state == STATE_DELETED) { return; }
    
    let vox = WorldToVoxel(c.position);
    let vIdx = VoxelIndex(vox.x, vox.y, vox.z);

    let slot = atomicAdd(&SpatialCellCount[vIdx], 1);
    if (slot < MAX_CELL_PER_VOXEL) {
        SpatialCellSlots[vIdx * MAX_CELL_PER_VOXEL + slot] = i32(i);
    }
}

// ===========================================================================
// КОПИРОВАНИЕ В МАТЕРИАЛ (Обход ограничений WGPU COPY_DST)
// ===========================================================================
// Буфер рендер-материала для записи результата копирования.
// Вынесен на @group(3) чтобы не конфликтовать с @group(1) (Cells_Read/Write).
@group(3) @binding(0) var<storage, read_write> RenderBuffer: array<CellData>;

@compute @workgroup_size(256, 1, 1)
fn Kernel_CopyToMat(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u32(MAX_AGENTS)) {
        RenderBuffer[id.x] = Cells_Read[id.x];
    }
}
