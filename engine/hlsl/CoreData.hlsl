// ---------------------------------------------------------------------------
// ДВИЖЕНИЕ
// ---------------------------------------------------------------------------
#define VELOCITY_DAMPING    0.85f   // затухание скорости за кадр
#define VELOCITY_IMPULSE    0.4f    // сила случайного импульса
#define CHEMO_FORCE         2.0f    // усиление хемотаксиса

// ---------------------------------------------------------------------------
// КОНСТАНТЫ
// ---------------------------------------------------------------------------
#define STATE_PROLIFERATING  0
#define STATE_QUIESCENT      1
#define STATE_NECROTIC       2
#define STATE_DELETED        3

#define TISSUE_ECM     0
#define TISSUE_VESSEL  1

#define MAX_AGENTS          100000
#define GRID_DIM            64
#define CELL_RADIUS         0.5f
#define CELL_DIAMETER       1.0f
#define MAX_CELL_PER_VOXEL  16
#define CONSUMPTION_SCALE   10000
#define MAX_SPECIES         8
#define ENERGY_DIVIDE_THR   10.0f
#define ENERGY_MOVE_COST    0.05f
#define ENERGY_SCALE        1000.0f
#define MAINTENANCE_COST    0.0f

// ---------------------------------------------------------------------------
// MetricsAccum Indices (int, atomic)
// ---------------------------------------------------------------------------
#define MA_N_PROLIF       0   
#define MA_N_QUIESC       1   
#define MA_N_NECR         2   
#define MA_N_CSC          3   
#define MA_SUM_DX         4
#define MA_SUM_DY         5
#define MA_SUM_DZ         6
#define MA_SUM_R2_LIVE    7   
#define MA_SUM_R2_NECR    8   
#define MA_MAX_R_LIVE     9   
#define MA_MAX_R_NECR    10   
#define MA_SUM_ENERGY    11   
#define MA_SUM_O2        12   
#define MA_SUM_GLU       13   
#define MA_SUM_SPEED     14   
#define MA_N_CROWDED     15   
#define MA_N_HYPOXIC     16   
#define MA_COUNT         17   

// ---------------------------------------------------------------------------
// MetricsOut Indices (float, for C#)
// ---------------------------------------------------------------------------
#define MO_TOTAL_LIVE     0   
#define MO_N_PROLIF       1
#define MO_N_QUIESC       2
#define MO_N_NECR         3
#define MO_FRAC_PROLIF    4   
#define MO_FRAC_QUIESC    5   
#define MO_FRAC_NECR      6   
#define MO_FRAC_HYPOXIC   7   
#define MO_CSC_COUNT      8
#define MO_CSC_FRAC       9   
#define MO_CENTROID_X    10   
#define MO_CENTROID_Y    11
#define MO_CENTROID_Z    12
#define MO_R_GYRATION    13   
#define MO_R_INVASIVE    14   
#define MO_R_NECROTIC    15   
#define MO_RIM_THICKNESS 16   
#define MO_MEAN_ENERGY   17
#define MO_MEAN_O2       18
#define MO_MEAN_GLU      19
#define MO_MEAN_SPEED    20   
#define MO_CROWD_FRAC    21   
#define MO_COUNT         22   

#define HYPOXIA_METRIC_THR  0.1f
#define CROWD_THRESHOLD     12

// ---------------------------------------------------------------------------
// СТРУКТУРЫ (64 байта каждая, выровнены с C#)
// ---------------------------------------------------------------------------
struct SpeciesParam
{
    float consumeO2, consumeGlu, hypoxiaDeathThr, baseMoveProb;
    float chemoTaxisO2, chemoTaxisGlu, baseDivideTime, probSymmetricDiv;
    float maxProliferations, spontaneousDeath;
    float pad1, pad2;
    float4 pad3;
};

struct CellData
{
    float3 position;
    int speciesID;
    int state;
    float aliveAge;
    float deadAge;
    float energy;
    int prolifCapacity;
    float3 customParams;
    float4 reserved;
    // TOTAL = 64 байт
};

struct DivReq
{
    int parentIdx, childSpeciesID, childProlifCap;
};

static const int3 MooreOffsets[27] =
{
    int3(-1, -1, -1), int3(0, -1, -1), int3(1, -1, -1),
    int3(-1, 0, -1), int3(0, 0, -1), int3(1, 0, -1),
    int3(-1, 1, -1), int3(0, 1, -1), int3(1, 1, -1),

    int3(-1, -1, 0), int3(0, -1, 0), int3(1, -1, 0),
    int3(-1, 0, 0), int3(0, 0, 0), int3(1, 0, 0),
    int3(-1, 1, 0), int3(0, 1, 0), int3(1, 1, 0),

    int3(-1, -1, 1), int3(0, -1, 1), int3(1, -1, 1),
    int3(-1, 0, 1), int3(0, 0, 1), int3(1, 0, 1),
    int3(-1, 1, 1), int3(0, 1, 1), int3(1, 1, 1)
};

// ---------------------------------------------------------------------------
// ГЛОБАЛЬНЫЕ ПЕРМЕННЫЕ
// ---------------------------------------------------------------------------
float DeltaTime;
int FrameCount;
int TotalAgents;
int InitialAgents;
float NecroticDecay;

float DiffusionRateO2, DiffusionRateGlu, DiffusionRateChemo;
float OxygenBoundaryValue, GluBoundaryValue, ChemoSourceLevel;

float RepulsionStiffness;
float AdhesionStiffness;
float AdhesionRange;
float DampingCoeff;

float3 VesselCenterWorld;
float VesselRadius;

uint RequestCount, LiveAgentCountBase;

float3 DomainCenter;

float3 DomainMin, DomainMax;
float OxygenGridSize;

int SpatialGridDim;
float SpatialVoxelSize;
int SpatialVoxelCount;

int NumSpecies;


// ===========================================================================
// БУФФЕРЫ
// ===========================================================================
SamplerState samplerFieldGrid_R;
Texture3D<float4> FieldGrid_R;
RWTexture3D<float4> FieldGrid_W;

StructuredBuffer<CellData> Cells_Read;
RWStructuredBuffer<CellData> Cells_Write;

StructuredBuffer<SpeciesParam> SpeciesTable;
StructuredBuffer<float> InteractionMatrix;

RWStructuredBuffer<int> SpatialCellCount;
RWStructuredBuffer<int> SpatialCellSlots;


RWStructuredBuffer<int> ConsumptionMapO2;
RWStructuredBuffer<int> ConsumptionMapGlu;

Texture3D<int> EnvironmentMask;
RWTexture3D<int> EnvironmentMaskWrite;

RWStructuredBuffer<int> MetricsAccum;
RWStructuredBuffer<float> MetricsOut;
