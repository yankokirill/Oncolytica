// ===========================================================================
// МЕТАБОЛИЗМ: обновление энергии, возраста и состояния клетки
// ===========================================================================

[numthreads(256, 1, 1)]
void Kernel_CellUpdate(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint) TotalAgents)
        return;

    CellData c = Cells_Read[i];

    if (c.state == STATE_DELETED)
    {
        Cells_Write[i] = c;
        return;
    }
    
    // 1. Мертвые клетки гниют и исчезают
    if (c.state == STATE_NECROTIC)
    {
        c.deadAge = min(c.deadAge + DeltaTime, NecroticDecay);
        Cells_Write[i] = c;
        return;
    }

    // 2. Локальные ресурсы
    SpeciesParam sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];
    float4 field = SampleField(c.position);
    float o2 = field.r;
    float glu = field.g;

    // 3. РАСЧЕТ ЭНЕРГИИ (Нормализованный)
    float energyGain = (sp.consumeO2 * saturate(o2) + sp.consumeGlu * saturate(glu)) * ENERGY_SCALE;
    
    c.energy = clamp(c.energy + (energyGain - MAINTENANCE_COST) * DeltaTime, 0.0f, 100.0f);

    // 4. Старение (зависит от метаболизма)
    float metabolicRate = saturate(o2) * saturate(glu);
    c.aliveAge += DeltaTime * lerp(0.2f, 1.0f, metabolicRate);

    // 5. Определение состояния
    if (o2 < sp.hypoxiaDeathThr)
    {
        // Вероятность смерти растет по мере падения O2 к нулю
        float deathSeverity = saturate((sp.hypoxiaDeathThr - o2) / sp.hypoxiaDeathThr);
        c.state = (Rand01(i, 42u) < deathSeverity * DeltaTime * 5.0f) ? STATE_NECROTIC : STATE_QUIESCENT;
    }
    else if (c.energy < 5.0f)
    {
        // Энергии мало -> впадаем в спячку
        c.state = STATE_QUIESCENT;
    }
    else
    {
        // Ресурсов полно -> пролиферация
        c.state = STATE_PROLIFERATING;
    }

    Cells_Write[i] = c;
}

// ===========================================================================
// ВЗАИМОДЕЙСТВИЕ
// ===========================================================================

float GetCellRadius(CellData c)
{
    float baseRadius = CELL_DIAMETER * 0.5f;
    if (c.state == STATE_NECROTIC)
    {
        // Лизис: клетка сжимается до 40% от исходного радиуса (~10% от объема)
        float shrinkProgress = saturate(c.deadAge / NecroticDecay);
        return lerp(baseRadius, baseRadius * 0.7f, shrinkProgress);
    }
    return baseRadius;
}

[numthreads(128, 1, 1)]
void Kernel_Interaction(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint) TotalAgents)
        return;

    CellData c = Cells_Read[i];
    
    if (c.state == STATE_DELETED)
    {
        Cells_Write[i] = c;
        return;
    }
    
    bool isNecrotic = (c.state == STATE_NECROTIC);
    SpeciesParam sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];

    int3 baseCell = WorldToSpatialCell(c.position);
    uint randSalt = 0u;
    bool willDie = false;

    float3 F_mech = float3(0.0f, 0.0f, 0.0f);
    
    // Считаем собственный эффективный радиус
    float myRadius = GetCellRadius(c);

    for (int k = 0; k < 27; k++)
    {
        int3 nc = clamp(baseCell + MooreOffsets[k], 0, SpatialGridDim - 1);
        int ni = SpatialCellIndex(nc);
        int cnt = SpatialCellCount[ni];

        for (int t = 0; t < cnt; t++)
        {
            int j = SpatialCellSlots[ni * MAX_CELL_PER_VOXEL + t];
            if (j == (int) i)
                continue;

            CellData other = Cells_Read[j];

            // 1. ВЗАИМОДЕЙСТВИЕ (killProb) - только для живых
            if (!isNecrotic)
            {
                float killProb = InteractionMatrix[other.speciesID * NumSpecies + c.speciesID];
                if (killProb > 0.0f && Rand01(i, randSalt++) < killProb * DeltaTime)
                    willDie = true;
            }

            // 2. МЕХАНИЧЕСКИЕ СИЛЫ
            float3 delta = c.position - other.position;
            float dist = length(delta);

            if (dist < 1e-5f)
            {
                F_mech += RandUnitVec(i, (uint) j) * RepulsionStiffness * 0.1f;
                continue;
            }

            // Вычисляем дистанцию взаимодействия на основе индивидуальных радиусов
            float otherRadius = GetCellRadius(other);
            float interactionDist = myRadius + otherRadius;
            float3 n_ij = delta / dist;

            // --- РЕПУЛЬСИЯ (перекрытие) ---
            if (dist < interactionDist)
            {
                float overlap = interactionDist - dist;
                // Некротические клетки "мягче", поэтому жесткость отталкивания ниже
                float currentRepulsion = isNecrotic || (other.state == STATE_NECROTIC)
                                         ? RepulsionStiffness * 0.3f
                                         : RepulsionStiffness;
                float F_rep = currentRepulsion * overlap * sqrt(overlap);
                F_mech += n_ij * F_rep;
            }
            // --- АДГЕЗИЯ ---
            // Адгезия работает только если ОБЕ клетки живые. Мертвый дебрис не цепляется.
            else if (!isNecrotic && other.state != STATE_NECROTIC)
            {
                float adgDistFar = interactionDist + AdhesionRange;
                if (dist < adgDistFar)
                {
                    float gap = dist - interactionDist;
                    float falloff = 0.5f * (1.0f + cos(3.14159f * gap / AdhesionRange));
                    float F_adh = AdhesionStiffness * gap * falloff;
                    F_mech -= n_ij * F_adh;
                }
            }
        }
    }

    c.state = willDie ? STATE_NECROTIC : c.state;

    // --- 3 и 4. ХЕМОТАКСИС И МОТИЛЬНОСТЬ (Только для живых) ---
    float3 F_chemo = float3(0.0f, 0.0f, 0.0f);
    float3 F_rand = float3(0.0f, 0.0f, 0.0f);

    if (c.state != STATE_NECROTIC)
    {
        int3 vox = WorldToVoxel(c.position);
        float3 gradO2, gradGlu;
        FieldGradient(vox, gradO2, gradGlu);

        float3 chemoDir = gradO2 * sp.chemoTaxisO2 + gradGlu * sp.chemoTaxisGlu;
        float chemoMag = length(chemoDir);
        if (chemoMag > 1e-6f)
            F_chemo = (chemoDir / chemoMag) * chemoMag * CHEMO_FORCE;

        if (sp.baseMoveProb > 0.0f)
            F_rand = RandUnitVec(i, 200u + randSalt) * sp.baseMoveProb * VELOCITY_IMPULSE;
    }

    // --- 5. ИНТЕГРИРОВАНИЕ ---
    float3 vel = c.customParams;
    // Некротические клетки испытывают большее сопротивление среды (вязкий дебрис)
    float currentDamping = isNecrotic ? DampingCoeff * 2.0f : DampingCoeff;
    float3 F_total = F_mech + F_chemo + F_rand - vel * currentDamping;

    vel += F_total * DeltaTime;

    float speed = length(vel);
    float maxSpeed = SpatialVoxelSize * 0.8f;
    if (speed > maxSpeed)
        vel = (vel / speed) * maxSpeed;

    // --- 6. ОБНОВЛЕНИЕ ПОЗИЦИИ ---
    float3 newPos = c.position + vel * DeltaTime;
    newPos = clamp(newPos, DomainMin + 0.1f, DomainMax - 0.1f);

    c.position = newPos;
    c.customParams = vel;
    
    if (!isNecrotic)
        c.energy -= ENERGY_MOVE_COST * length(vel) * DeltaTime;

    Cells_Write[i] = c;
}

// ===========================================================================
// ДЕЛЕНИЕ
//
// Логика CSC (isStem = probSymmetricDiv < 1.0):
//   - Rand < probSymmetricDiv → симметричное деление (ребёнок = CSC)
//   - иначе                   → асимметричное (ребёнок = CC с лимитом пролиферации)
//
// Логика CC (isStem = false, probSymmetricDiv == 1.0):
//   - prolifCapacity <= 0 → некроз (лимит Хейфлика)
//   - spontaneousDeath * DeltaTime → случайная смерть при делении
// ===========================================================================

float3 RandomOffset(uint idx, uint frame)
{
    float3 dir = float3(
        Rand01(idx, frame * 7u) * 2.0f - 1.0f,
        Rand01(idx, frame * 13u) * 2.0f - 1.0f,
        Rand01(idx, frame * 23u) * 2.0f - 1.0f
    );
    float l = length(dir);
    return (l > 0.001f) ? dir / l * (CELL_DIAMETER * 0.75f) : float3(0.75f, 0, 0);
}


[numthreads(128, 1, 1)]
void Kernel_Division(uint3 id : SV_DispatchThreadID)
{
    // Сначала выполняем все проверки (валидность агента, состояние, энергия, возраст)
    if (2 * id.x + 1 + InitialAgents < (uint) TotalAgents)
    {
        CellData c = Cells_Read[id.x];
        
        // ВАЖНО: Если поток не должен делить клетку, он просто не участвует в логике ниже
        if (c.state == STATE_PROLIFERATING)
        {
            SpeciesParam sp = SpeciesTable[clamp(c.speciesID, 0, MAX_SPECIES - 1)];

            if (c.energy >= ENERGY_DIVIDE_THR && c.aliveAge >= sp.baseDivideTime)
            {
                uint randSalt = 0u;
                bool isStem = (sp.probSymmetricDiv < 1.0f);

                // Проверка на спонтанную смерть
                if (!isStem && sp.spontaneousDeath > 0.0f && Rand01(id.x, randSalt++) < sp.spontaneousDeath)
                {
                    c.state = STATE_NECROTIC;
                    Cells_Write[id.x] = c;
                }
                // Проверка на лимит Хейфлика
                else if (!isStem && c.prolifCapacity <= 0)
                {
                    c.state = STATE_NECROTIC;
                    Cells_Write[id.x] = c;
                }
                else
                {
                    // Сброс родителя
                    c.aliveAge = 0.0f;
                    c.energy *= 0.5f;
                    
                    DivReq req;
                    if (isStem)
                    {
                        bool symmetric = (Rand01(id.x, randSalt++) < sp.probSymmetricDiv);
                        req.childSpeciesID = symmetric ? c.speciesID : 0;
                        req.childProlifCap = symmetric ? 9999 : (int) SpeciesTable[0].maxProliferations;
                    }
                    else
                    {
                        c.prolifCapacity -= 1;
                        req.childSpeciesID = c.speciesID;
                        req.childProlifCap = c.prolifCapacity;
                    }
                    
                    CellData child;
                    child.position = c.position + RandomOffset(id.x, (uint) FrameCount);
                    child.state = STATE_PROLIFERATING;
                    child.aliveAge = 0.0f;
                    child.deadAge = 0.0f;
                    child.speciesID = req.childSpeciesID;
                    child.energy = 5.0f;
                    child.prolifCapacity = req.childProlifCap;
                    child.customParams = float3(0, 0, 0); // velocity = 0 при рождении
                    child.reserved = float4(0, 0, 0, 0);
                    
                    // Обновляем родителя в глобальной памяти
                    Cells_Write[id.x].state = STATE_DELETED;
                    Cells_Write[2 * id.x + InitialAgents] = c;
                    Cells_Write[2 * id.x + InitialAgents + 1] = child;
                }
            }
        }
    }
}
