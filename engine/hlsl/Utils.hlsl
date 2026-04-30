// ===========================================================================
// Рандом
// ===========================================================================

uint PCGHash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand01(uint cellIdx, uint salt)
{
    uint seed = PCGHash(cellIdx * 2654435769u ^ (uint) FrameCount * 1013904223u ^ salt);
    return (float) PCGHash(seed) / 4294967295.0f;
}

float3 RandUnitVec(uint cellIdx, uint salt)
{
    float theta = Rand01(cellIdx, salt) * 6.28318f;
    float phi = Rand01(cellIdx, salt + 1u) * 3.14159f;
    float sp = sin(phi);
    return float3(sp * cos(theta), sp * sin(theta), cos(phi));
}

// ===========================================================================
// УТИЛИТЫ
// ===========================================================================

int3 WorldToVoxel(float3 wp)
{
    float3 uvw = saturate((wp - DomainMin) / (DomainMax - DomainMin));
    int gs = (int) OxygenGridSize;
    return int3((int) (uvw.x * gs), (int) (uvw.y * gs), (int) (uvw.z * gs));
}

int VoxelIndex(int x, int y, int z)
{
    int gs = (int) OxygenGridSize;
    return clamp(x, 0, gs - 1) + clamp(y, 0, gs - 1) * gs + clamp(z, 0, gs - 1) * gs * gs;
}

int3 WorldToSpatialCell(float3 wp)
{
    float3 p = (wp - DomainMin) / SpatialVoxelSize;
    return int3(
        clamp((int) floor(p.x), 0, SpatialGridDim - 1),
        clamp((int) floor(p.y), 0, SpatialGridDim - 1),
        clamp((int) floor(p.z), 0, SpatialGridDim - 1)
    );
}

int SpatialCellIndex(int3 c)
{
    return c.x + c.y * SpatialGridDim + c.z * SpatialGridDim * SpatialGridDim;
}

float4 SampleField(float3 wp)
{
    float3 uvw = saturate((wp - DomainMin) / (DomainMax - DomainMin));
    return FieldGrid_R.SampleLevel(samplerFieldGrid_R, uvw, 0);
}

void FieldGradient(int3 v, out float3 gradO2, out float3 gradGlu)
{
    int gs = (int) OxygenGridSize;
    v = clamp(v, int3(1, 1, 1), int3(gs - 2, gs - 2, gs - 2));
    float4 xp = FieldGrid_R[v + int3(1, 0, 0)];
    float4 xm = FieldGrid_R[v - int3(1, 0, 0)];
    float4 yp = FieldGrid_R[v + int3(0, 1, 0)];
    float4 ym = FieldGrid_R[v - int3(0, 1, 0)];
    float4 zp = FieldGrid_R[v + int3(0, 0, 1)];
    float4 zm = FieldGrid_R[v - int3(0, 0, 1)];
    gradO2 = float3(xp.r - xm.r, yp.r - ym.r, zp.r - zm.r) * 0.5f;
    gradGlu = float3(xp.g - xm.g, yp.g - ym.g, zp.g - zm.g) * 0.5f;
}
