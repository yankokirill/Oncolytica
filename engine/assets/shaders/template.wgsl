// ===========================================================================
// КОНСТАНТЫ (Транслятор может дополнять этот блок)
// ===========================================================================
const STATE_DELETED: i32 = 0;
const STATE_ALIVE: i32   = 1;
const STATE_DEAD: i32    = 2;

// Системные константы
const MAX_AGENTS: i32         = 100000;
const MAX_CELL_PER_VOXEL: i32 = 16;
const WORKGROUP_SIZE: i32     = 256;

// === INJECT: USER CONSTANTS ===
// (Сюда транслятор будет вставлять константы из Python)

// ===========================================================================
// СТРУКТУРЫ ДАННЫХ
// ===========================================================================

// Среда (Environment) хранится в 3D текстуре формата r32sint
// Поля (Fields) хранятся в 3D текстуре формата rgba32float 

// ВАЖНО: Выравнивание памяти! Скармливаем Python-у vec3 как 4 флоата.
struct CellData {
    position: vec3<f32>,
    type_id: i32,       // Занимает 4-й слот после vec3
    
    state: i32,
    id: i32,
    energy: f32,
    custom_val: f32,
    
    // Системные поля
    rngState: u32,
    _pad: vec4<i32>     // Добиваем до 16 байт
    
    // === INJECT: USER CELL FIELDS ===
}

struct GlobalUniforms {
    DeltaTime: f32,
    TimeTime: f32,
    DomainMin: vec3<f32>,
    DomainMax: vec3<f32>,
    
    SpatialGridDim: i32,  // Размер сетки по одной оси (напр. 64)
    VoxelGridDim: i32,    // Размер сетки для вокселей среды (напр. 64)
    SpatialVoxelSize: f32,
    _pad: i32,
}

// ===========================================================================
// БИНДИНГИ
// ===========================================================================

// Group 0: Глобальные данные и Ландшафт
@group(0) @binding(0) var<uniform> U: GlobalUniforms;
@group(0) @binding(1) var EnvGrid_R: texture_3d<i32>;
@group(0) @binding(2) var EnvGrid_W: texture_storage_3d<r32sint, read_write>;
@group(0) @binding(3) var FieldGrid_R: texture_3d<f32>;
@group(0) @binding(4) var FieldGrid_W: texture_storage_3d<rgba32float, read_write>;

// Group 1: Агенты (Клетки)
@group(1) @binding(0) var<storage, read> Cells_Read: array<CellData>;
@group(1) @binding(1) var<storage, read_write> Cells_Write: array<CellData>;
// Буфер для счетчика добавленных агентов при делении
@group(1) @binding(2) var<storage, read_write> NextAgentIndex: atomic<i32>; 

// Group 2: Пространственная сетка (Spatial Grid)
@group(2) @binding(0) var<storage, read_write> SpatialCellCount: array<atomic<i32>>;
@group(2) @binding(1) var<storage, read_write> SpatialCellSlots: array<i32>;

// ===========================================================================
// СИСТЕМНЫЕ ФУНКЦИИ И ХЕШИРОВАНИЕ
// ===========================================================================

// RNG
struct RandState { result: f32, rng: u32 }
fn NextRand(x: u32) -> RandState {
    var val = x;
    val = val ^ (val << 13u);
    val = val ^ (val >> 17u);
    val = val ^ (val << 5u);
    return RandState(f32(val) * 2.3283064365386963e-10, val);
}

// Конвертация мировых координат в индексы сетки клеток
fn WorldToSpatialCell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(clamp(
        (pos - U.DomainMin) / U.SpatialVoxelSize, 
        vec3<f32>(0.0), 
        vec3<f32>(f32(U.SpatialGridDim - 1))
    ));
}

fn SpatialCellIndex(cell: vec3<i32>) -> i32 {
    return cell.x + cell.y * U.SpatialGridDim + cell.z * U.SpatialGridDim * U.SpatialGridDim;
}

// ===========================================================================
// ЯДРА: СИСТЕМНАЯ СЕТКА (SPATIAL GRID)
// ===========================================================================
// Запускается каждую итерацию перед физикой/клетками

@compute @workgroup_size(256, 1, 1)
fn Kernel_SpatialGrid_Clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let max_idx = u32(U.SpatialGridDim * U.SpatialGridDim * U.SpatialGridDim);
    if (idx < max_idx) {
        atomicStore(&SpatialCellCount[idx], 0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn Kernel_SpatialGrid_Insert(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    let c = Cells_Read[i];
    if (c.state == STATE_DELETED) { return; }

    let voxel = SpatialCellIndex(WorldToSpatialCell(c.position));
    
    // Атомарно занимаем слот в вокселе
    let slot = atomicAdd(&SpatialCellCount[voxel], 1);
    if (slot < MAX_CELL_PER_VOXEL) {
        SpatialCellSlots[voxel * MAX_CELL_PER_VOXEL + slot] = i32(i);
    }
}

// ===========================================================================
// ФАЗА 1: СЛОЙ СРЕДЫ (Environment)
// ===========================================================================
@compute @workgroup_size(8, 8, 8)
fn Kernel_EnvironmentRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = u32(U.VoxelGridDim);
    if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
    
    let voxel_coord = vec3<i32>(id);
    var env_type_id = textureLoad(EnvGrid_R, voxel_coord, 0).r;

    // Системный доступ к клеткам внутри вокселя для чтения
    let sidx = SpatialCellIndex(voxel_coord); // Предполагаем сетки совпадают 1:1
    let cell_cnt = min(atomicLoad(&SpatialCellCount[sidx]), MAX_CELL_PER_VOXEL);

    // === INJECT: @environment_rule LOGIC ===
    // Пример вставки:
    // var cancer_count = 0;
    // for (var t = 0; t < cell_cnt; t++) {
    //     let c = Cells_Read[SpatialCellSlots[sidx * MAX_CELL_PER_VOXEL + t]];
    //     if (c.type_id == CANCER_TYPE) { cancer_count++; }
    // }
    // if (cancer_count > 5) { env_type_id = 0; }
    // =======================================

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

    // === INJECT: @reaction_rule LOGIC ===
    // fields.r *= 0.99;
    // ====================================

    textureStore(FieldGrid_W, voxel_coord, fields);
}

@compute @workgroup_size(8, 8, 8)
fn Kernel_DiffusionRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = u32(U.VoxelGridDim);
    if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
    
    let v = vec3<i32>(id);
    let c = textureLoad(FieldGrid_R, v, 0);
    
    // Считывание соседей для лапласиана (можно сделать красивее, но для примера сойдет)
    let xp = textureLoad(FieldGrid_R, v + vec3<i32>(1,0,0), 0);
    let xm = textureLoad(FieldGrid_R, v - vec3<i32>(1,0,0), 0);
    // ... yp, ym, zp, zm

    // === INJECT: @diffusion_rule LOGIC ===
    // =====================================

    textureStore(FieldGrid_W, v, c);
}

// ===========================================================================
// ФАЗА 3: СЛОЙ КЛЕТОК (Update Rules)
// ===========================================================================
// Транслятор сгенерирует столько Kernel_UpdateRule_*, сколько декораторов @update_rule

@compute @workgroup_size(256, 1, 1)
fn Kernel_UpdateRule_Pass1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _с = Cells_Read[i];
    if (_с.state == STATE_DELETED) { 
        Cells_Write[i] = _с; 
        return; 
    }

    // Системная подготовка данных (доступ к полям, среде и соседям)
    let my_voxel = WorldToSpatialCell(_с.position);
    let my_env = textureLoad(EnvGrid_R, my_voxel, 0).r;
    let my_fields = textureLoad(FieldGrid_R, my_voxel, 0);

    // === INJECT: @update_rule LOGIC (Pass 1) ===
    // Пример трансляции Python кода:
    // if (_с.state == STATE_DEAD) {
    //     // Поиск соседа...
    //     _с.custom_val = predator_id;
    // }
    // ===========================================

    Cells_Write[i] = _с;
}

// ===========================================================================
// ФАЗА 4: ЖИЗНЕННЫЙ ЦИКЛ (Division & Death)
// ===========================================================================
@compute @workgroup_size(256, 1, 1)
fn Kernel_DivisionRule(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(MAX_AGENTS)) { return; }

    var _с = Cells_Read[i];
    if (_с.state == STATE_DELETED) { 
        Cells_Write[i] = _с; 
        return; 
    }

    var create_child: bool = false;
    var child: CellData;

    // === INJECT: @division_rule LOGIC ===
    // Пример из Python: 
    // if (_с.energy > 100.0) {
    //     _с.energy /= 2.0;
    //     create_child = true;
    //     child = _с; // Копируем базу
    // } else if (_с.energy <= 0.0) {
    //     _с.state = STATE_DELETED;
    // }
    // ====================================

    // Системный код создания потомка
    if (create_child) {
        let child_idx = u32(atomicAdd(&NextAgentIndex, 1));
        if (child_idx < u32(MAX_AGENTS)) {
            // Обязательно меняем RNG state для потомка
            child.rngState = _с.rngState ^ 0x12345678u;
            Cells_Write[child_idx] = child;
        }
    }

    // Сохраняем себя
    Cells_Write[i] = _с;
}
