[numthreads(256, 1, 1)]
void SpatialGrid_Clear(uint3 id : SV_DispatchThreadID)
{
    if (id.x < SpatialVoxelCount)
        SpatialCellCount[id.x] = 0;
}

// Вставка для взаимодействий (Использует SpatialVoxelSize)
[numthreads(256, 1, 1)]
void SpatialGrid_Insert_Interaction(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint) TotalAgents)
        return;

    CellData c = Cells_Read[i];
    
    if (c.state == STATE_DELETED)
        return;

    int voxel = SpatialCellIndex(WorldToSpatialCell(c.position));
    int slot;
    InterlockedAdd(SpatialCellCount[voxel], 1, slot);
    if (slot < MAX_CELL_PER_VOXEL)
        SpatialCellSlots[voxel * MAX_CELL_PER_VOXEL + slot] = (int) i;
}

// Вставка для полей (Использует OxygenGridSize)
[numthreads(256, 1, 1)]
void SpatialGrid_Insert_Field(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint) TotalAgents)
        return;

    CellData c = Cells_Read[i];

    if (c.state == STATE_DELETED)
        return;
    
    // Используем функции для сетки полей!
    int3 voxel = WorldToVoxel(c.position);
    int vIdx = VoxelIndex(voxel.x, voxel.y, voxel.z);

    int slot;
    InterlockedAdd(SpatialCellCount[vIdx], 1, slot);
    if (slot < MAX_CELL_PER_VOXEL)
        SpatialCellSlots[vIdx * MAX_CELL_PER_VOXEL + slot] = (int) i;
}