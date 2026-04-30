// ===========================================================================
// Kernel_ClearMetrics
// ===========================================================================
[numthreads(17, 1, 1)] // Should match MA_COUNT
void Kernel_ClearMetrics(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 17)
        MetricsAccum[id.x] = 0;
}

// ===========================================================================
// Kernel_CollectMetrics
// ===========================================================================
[numthreads(256, 1, 1)]
void Kernel_CollectMetrics(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint) TotalAgents)
        return;

    CellData c = Cells_Read[i];
    
    float3 delta = c.position - DomainCenter;
    float r = length(delta);
    
    if (c.state == STATE_DELETED)
        return;

    if (c.state == STATE_NECROTIC)
    {
        InterlockedAdd(MetricsAccum[MA_N_NECR], 1);
        InterlockedAdd(MetricsAccum[MA_SUM_R2_NECR], (int) (r * r));
        InterlockedMax(MetricsAccum[MA_MAX_R_NECR], (int) (r * 100.0f));
        return;
    }

    SpeciesParam sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];

    if (c.state == STATE_PROLIFERATING)
        InterlockedAdd(MetricsAccum[MA_N_PROLIF], 1);
    else
        InterlockedAdd(MetricsAccum[MA_N_QUIESC], 1);

    if (sp.probSymmetricDiv < 1.0f)
        InterlockedAdd(MetricsAccum[MA_N_CSC], 1);

    InterlockedAdd(MetricsAccum[MA_SUM_DX], (int) (delta.x * 10.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_DY], (int) (delta.y * 10.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_DZ], (int) (delta.z * 10.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_R2_LIVE], (int) (r * r));
    InterlockedMax(MetricsAccum[MA_MAX_R_LIVE], (int) (r * 100.0f));

    float4 field = SampleField(c.position);

    InterlockedAdd(MetricsAccum[MA_SUM_ENERGY], (int) (c.energy * 10.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_O2], (int) (field.r * 1000.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_GLU], (int) (field.g * 1000.0f));
    InterlockedAdd(MetricsAccum[MA_SUM_SPEED], (int) (length(c.customParams) * 1000.0f));

    if (field.r < HYPOXIA_METRIC_THR)
        InterlockedAdd(MetricsAccum[MA_N_HYPOXIC], 1);

    int sidx = SpatialCellIndex(c.position);
    if (SpatialCellCount[sidx] > CROWD_THRESHOLD)
        InterlockedAdd(MetricsAccum[MA_N_CROWDED], 1);
}

// ===========================================================================
// Kernel_FinalizeMetrics
// ===========================================================================
[numthreads(1, 1, 1)]
void Kernel_FinalizeMetrics(uint3 id : SV_DispatchThreadID)
{
    int nProlif = MetricsAccum[MA_N_PROLIF];
    int nQuiesc = MetricsAccum[MA_N_QUIESC];
    int nNecr = MetricsAccum[MA_N_NECR];
    int nCSC = MetricsAccum[MA_N_CSC];
    int nHypoxic = MetricsAccum[MA_N_HYPOXIC];
    int nCrowd = MetricsAccum[MA_N_CROWDED];

    int nLive = nProlif + nQuiesc;
    int nTotal = nLive + nNecr;

    float fLive = max((float) nLive, 1.0f);
    float fTotal = max((float) nTotal, 1.0f);

    MetricsOut[MO_TOTAL_LIVE] = (float) nLive;
    MetricsOut[MO_N_PROLIF] = (float) nProlif;
    MetricsOut[MO_N_QUIESC] = (float) nQuiesc;
    MetricsOut[MO_N_NECR] = (float) nNecr;

    MetricsOut[MO_FRAC_PROLIF] = (float) nProlif / fLive;
    MetricsOut[MO_FRAC_QUIESC] = (float) nQuiesc / fLive;
    MetricsOut[MO_FRAC_NECR] = (float) nNecr / fTotal;
    MetricsOut[MO_FRAC_HYPOXIC] = (float) nHypoxic / fLive;

    MetricsOut[MO_CSC_COUNT] = (float) nCSC;
    MetricsOut[MO_CSC_FRAC] = (float) nCSC / fLive;

    MetricsOut[MO_CENTROID_X] = DomainCenter.x + (float) MetricsAccum[MA_SUM_DX] / (fLive * 10.0f);
    MetricsOut[MO_CENTROID_Y] = DomainCenter.y + (float) MetricsAccum[MA_SUM_DY] / (fLive * 10.0f);
    MetricsOut[MO_CENTROID_Z] = DomainCenter.z + (float) MetricsAccum[MA_SUM_DZ] / (fLive * 10.0f);

    float meanR2 = (float) MetricsAccum[MA_SUM_R2_LIVE] / fLive;
    MetricsOut[MO_R_GYRATION] = sqrt(meanR2);
    
    float rInvasive = (float) MetricsAccum[MA_MAX_R_LIVE] / 100.0f;
    
    float rNecrotic = 0.0f;
    if (nNecr > 0)
    {
        rNecrotic = 0.5f * pow((float) nNecr / 0.65f, 0.333333f);
    }

    MetricsOut[MO_R_INVASIVE] = rInvasive;
    MetricsOut[MO_R_NECROTIC] = rNecrotic;
    MetricsOut[MO_RIM_THICKNESS] = max(rInvasive - rNecrotic, 0.0f);

    // =======================================================================

    MetricsOut[MO_MEAN_ENERGY] = (float) MetricsAccum[MA_SUM_ENERGY] / (fLive * 10.0f);
    MetricsOut[MO_MEAN_O2] = (float) MetricsAccum[MA_SUM_O2] / (fLive * 1000.0f);
    MetricsOut[MO_MEAN_GLU] = (float) MetricsAccum[MA_SUM_GLU] / (fLive * 1000.0f);
    MetricsOut[MO_MEAN_SPEED] = (float) MetricsAccum[MA_SUM_SPEED] / (fLive * 1000.0f);

    MetricsOut[MO_CROWD_FRAC] = (float) nCrowd / fLive;
}
    