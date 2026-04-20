// ===========================================================================
// ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ
// ===========================================================================
[numthreads(8, 8, 8)]
void Kernel_InitField(uint3 id : SV_DispatchThreadID)
{
    uint gs = (uint) OxygenGridSize;
    if (id.x >= gs || id.y >= gs || id.z >= gs)
        return;
    // Инициализируем граничными значениями — сосуд перезапишет позже
    FieldGrid_W[id] = float4(OxygenBoundaryValue, GluBoundaryValue, 0.0f, 0.0f);
}

[numthreads(8, 8, 8)]
void Kernel_InitEnvironment(uint3 id : SV_DispatchThreadID)
{
    int gs = (int) OxygenGridSize;
    if ((int) id.x >= gs || (int) id.y >= gs || (int) id.z >= gs)
        return;
    float3 uvw = (float3(id) + 0.5f) / (float) gs;
    float3 wp = DomainMin + uvw * (DomainMax - DomainMin);
    float dist2D = length(float2(wp.x - VesselCenterWorld.x, wp.z - VesselCenterWorld.z));
    EnvironmentMaskWrite[id] = (dist2D <= VesselRadius) ? TISSUE_VESSEL : TISSUE_ECM;
}

// ===========================================================================
// ПОЛЯ: ПОТРЕБЛЕНИЕ ? РЕАКЦИЯ ? ДИФФУЗИЯ
// ===========================================================================

[numthreads(8, 8, 8)]
void Kernel_ApplyReaction(uint3 id : SV_DispatchThreadID)
{
    int gs = (int) OxygenGridSize;
    if ((int) id.x >= gs || (int) id.y >= gs || (int) id.z >= gs)
        return;
    
    int vIdx = VoxelIndex(id.x, id.y, id.z);
    int cellCount = SpatialCellCount[vIdx];

    float totalK_O2 = 0.0f;
    float totalK_Glu = 0.0f;

    // Читаем клетки, которые попали в этот воксель поля
    for (int t = 0; t < cellCount; t++)
    {   
        int agentIdx = SpatialCellSlots[vIdx * MAX_CELL_PER_VOXEL + t];
        CellData c = Cells_Read[agentIdx];
        if (c.state != STATE_NECROTIC)
        {
            SpeciesParam sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
            totalK_O2 += sp.consumeO2;
            totalK_Glu += sp.consumeGlu;
        }
    }

    // Применяем реакцию
    float4 field = FieldGrid_R[id];
    field.r = field.r / (1.0f + totalK_O2 * DeltaTime);
    field.g = field.g / (1.0f + totalK_Glu * DeltaTime);
    FieldGrid_W[id] = field;
}

[numthreads(8, 8, 8)]
void Kernel_DiffusionSolver(uint3 id : SV_DispatchThreadID)
{
    uint gs = (uint) OxygenGridSize;
    if (id.x >= gs || id.y >= gs || id.z >= gs)
        return;
    int3 v = int3(id);

    // Граничные условия: Дирихле (фиксированное значение)
    if (v.x == 0 || v.x == gs - 1 || v.y == 0 || v.y == gs - 1 || v.z == 0 || v.z == gs - 1)
    {
        FieldGrid_W[v] = float4(OxygenBoundaryValue, GluBoundaryValue, 0, 0);
        return;
    }

    // Сосуд — источник питательных веществ (Дирихле = 1.0)
    if (EnvironmentMask[v] == TISSUE_VESSEL)
    {
        FieldGrid_W[v] = float4(1.0f, 1.0f, ChemoSourceLevel, 0.0f);
        return;
    }

    float4 c = FieldGrid_R[v];
    float4 xp = FieldGrid_R[v + int3(1, 0, 0)];
    float4 xm = FieldGrid_R[v - int3(1, 0, 0)];
    float4 yp = FieldGrid_R[v + int3(0, 1, 0)];
    float4 ym = FieldGrid_R[v - int3(0, 1, 0)];
    float4 zp = FieldGrid_R[v + int3(0, 0, 1)];
    float4 zm = FieldGrid_R[v - int3(0, 0, 1)];

    float lapO2 = xp.r + xm.r + yp.r + ym.r + zp.r + zm.r - 6.0f * c.r;
    float lapGlu = xp.g + xm.g + yp.g + ym.g + zp.g + zm.g - 6.0f * c.g;

    FieldGrid_W[v] = float4(
        saturate(c.r + DiffusionRateO2 * lapO2),
        saturate(c.g + DiffusionRateGlu * lapGlu),
        0, 0
    );
}
