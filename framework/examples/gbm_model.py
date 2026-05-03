from __future__ import annotations
import math
import random
import time
import matplotlib.pyplot as plt
import oncolytica as ol

# =============================================================================
# 1. ОПРЕДЕЛЕНИЕ ТИПОВ И БАЗОВЫХ КЛАССОВ
# =============================================================================

_S = "MultiscaleGBM"
_T = "BrainTissue"
_C = "BrainChem"
_A = "GBMCell"
_M = "GBMMetrics"
_P = "Params"

TissueBase    = ol.Tissue[_S, _T, _C, _A, _M, _P]
ChemistryBase = ol.Chemistry[_S, _T, _C, _A, _M, _P]
CellBase      = ol.Cell[_S, _T, _C, _A, _M, _P]
MetricsBase   = ol.Metrics[_S, _T, _C, _A, _M, _P]

# Глобальные константы модели
TYPE_INACTIVE:  ol.i32 = 0
TYPE_INFECTED:  ol.i32 = 1
TYPE_RECRUITED: ol.i32 = 2

PDGF_D:        ol.f32 = 0.0417
PDGF_DECAY:    ol.f32 = 0.005
PDGF_SECRETE:  ol.f32 = 0.417
PDGF_CONSUME:  ol.f32 = 0.042
K_HILL:        ol.f32 = 100.0
C_PA:          ol.f32 = 5.0
BETA:          ol.f32 = 0.5
ACTIVATION_THR: ol.f32 = 0.0005
TAU_MOVE:      ol.f32 = 0.71
TAU_STOP:      ol.f32 = 1.17
AP_THRESHOLD:  ol.f32 = 0.0166
AM_FACTOR:     ol.f32 = 0.10

TWO_PI:        ol.f32 = 6.2831853


# =============================================================================
# 2. СТРУКТУРЫ ДАННЫХ
# =============================================================================

class BrainTissue(TissueBase):
    is_white_matter:   ol.bool = False
    tract_dir_x:       ol.f32  = 0.0
    tract_dir_y:       ol.f32  = 0.0
    carrying_capacity: ol.i32  = 10

    def init_white_matter(self, angle_noise_std: float = math.pi / 6.0):
        """Настроить воксель как белое вещество с случайной ориентацией тракта."""
        self.is_white_matter   = True
        self.carrying_capacity = 7
        ang = random.gauss(0.0, angle_noise_std)
        self.tract_dir_x = math.cos(ang)
        self.tract_dir_y = math.sin(ang)

    def is_overcrowded(self, active_neighbor_count: ol.i32) -> ol.bool:
        """Проверить, превышена ли ёмкость вокселя."""
        return active_neighbor_count >= self.carrying_capacity


class BrainChem(ChemistryBase):
    pdgf: ol.f32 = 0.0

    def compute_secretion_consumption(self) -> tuple[ol.f32, ol.f32]:
        """Суммировать вклад клеток в секрецию и потребление PDGF."""
        secretion:   ol.f32 = 0.0
        consumption: ol.f32 = 0.0
        for c in self.cells:
            if c.cell_type == TYPE_INFECTED:
                secretion += PDGF_SECRETE
            if c.cell_type != TYPE_INACTIVE:
                consumption += PDGF_CONSUME
        return secretion, consumption

    def compute_laplacian(self) -> ol.f32:
        """Вычислить дискретный лапласиан поля PDGF."""
        lap: ol.f32 = 0.0
        for nb in self.neighbors:
            lap += nb.pdgf - self.pdgf
        return lap

    def step_pdgf(self):
        """Обновить поле PDGF: диффузия + секреция - потребление - распад."""
        secretion, consumption = self.compute_secretion_consumption()
        lap       = self.compute_laplacian()
        new_val   = (self.pdgf + secretion - consumption + PDGF_D * 0.5 * lap) * (1.0 - PDGF_DECAY)
        self.pdgf = ol.math.clamp(new_val, 0.0, 1_000_000.0)


class GBMCell(CellBase):
    pos:              ol.vec3
    cell_type:        ol.i32  = 0
    p_pot:            ol.f32  = 0.0
    m_pot:            ol.f32  = 0.0
    div_clock:        ol.f32  = 0.0
    is_moving:        ol.bool = False
    persistence_timer: ol.f32 = 0.0
    move_dir_x:       ol.f32  = 0.0
    move_dir_y:       ol.f32  = 0.0
    prev_x:           ol.f32  = 0.0
    prev_y:           ol.f32  = 0.0

    # ------------------------------------------------------------------ #
    #  Фенотип                                                             #
    # ------------------------------------------------------------------ #

    def phenotype_gamma_infected(self, c_pp: ol.f32) -> ol.f32:
        """Функция гаммы Hill для инфицированных клеток."""
        num: ol.f32 = C_PA + c_pp
        den: ol.f32 = C_PA + c_pp + K_HILL
        return num / den

    def phenotype_gamma_recruited(self, c_pp: ol.f32) -> ol.f32:
        """Функция гаммы Hill для рекрутированных клеток."""
        return c_pp / (c_pp + BETA * K_HILL)

    def compute_phenotype(self, c_pp: ol.f32, apply_am: ol.bool) -> tuple[ol.f32, ol.f32]:
        """Вернуть эффективный пролиферативный (p) и миграционный (m) потенциал."""
        if self.cell_type == TYPE_INFECTED:
            gamma = self.phenotype_gamma_infected(c_pp)
        else:
            gamma = self.phenotype_gamma_recruited(c_pp)

        p: ol.f32 = self.p_pot * gamma
        m: ol.f32 = self.m_pot * gamma * (AM_FACTOR if apply_am else 1.0)
        return p, m

    # ------------------------------------------------------------------ #
    #  Активация предшественника                                           #
    # ------------------------------------------------------------------ #

    def try_activate(self, c_pp: ol.f32) -> ol.bool:
        """Активировать неактивный предшественник при достаточной концентрации PDGF."""
        if c_pp > ACTIVATION_THR:
            self.cell_type        = TYPE_RECRUITED
            self.persistence_timer = TAU_STOP
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Деление                                                             #
    # ------------------------------------------------------------------ #

    def attempt_division(self, p: ol.f32):
        """Накапливать делительные часы и при достижении порога порождать дочернюю клетку."""
        if p <= 0.0:
            return

        self.div_clock += p
        if self.div_clock < 1.0:
            return

        self.div_clock -= 1.0
        self._spawn_daughter()

    def _spawn_daughter(self):
        """Разместить дочернюю клетку на расстоянии 5 мкм в случайном направлении."""
        angle: ol.f32 = ol.random() * TWO_PI
        nx: ol.f32 = ol.math.clamp(self.pos.x + 5.0 * ol.math.cos(angle), 0.0, 599.9)
        ny: ol.f32 = ol.math.clamp(self.pos.y + 5.0 * ol.math.sin(angle), 0.0, 599.9)

        rng:     ol.f32 = ol.math.clamp(ol.random(), 0.0001, 1.0)
        d_timer: ol.f32 = ol.math.clamp(-ol.math.log(rng) * TAU_STOP, 0.1, 100.0)

        daughter = GBMCell(
            pos=ol.vec3(nx, ny, 7.5),
            cell_type=self.cell_type,
            p_pot=self.p_pot,
            m_pot=self.m_pot,
            div_clock=ol.random() * 0.3,
            is_moving=False,
            persistence_timer=d_timer,
            prev_x=nx,
            prev_y=ny,
        )
        self.divide(daughter)

    # ------------------------------------------------------------------ #
    #  Миграция                                                            #
    # ------------------------------------------------------------------ #

    def tick_persistence(self, tissue: BrainTissue):
        """Уменьшить таймер персистентности и при истечении переключить режим движения."""
        self.persistence_timer -= 1.0
        if self.persistence_timer > 0.0:
            return

        self.is_moving = not self.is_moving
        if self.is_moving:
            self._enter_moving_state(tissue)
        else:
            self._enter_stopped_state()

    def _enter_moving_state(self, tissue: BrainTissue):
        """Выбрать новое направление и установить таймер движения."""
        rng: ol.f32 = ol.math.clamp(ol.random(), 0.0001, 1.0)
        self.persistence_timer = ol.math.clamp(-ol.math.log(rng) * TAU_MOVE, 0.1, 100.0)
        angle = self._sample_direction(tissue)
        self.move_dir_x = ol.math.cos(angle)
        self.move_dir_y = ol.math.sin(angle)

    def _enter_stopped_state(self):
        """Установить таймер остановки."""
        rng: ol.f32 = ol.math.clamp(ol.random(), 0.0001, 1.0)
        self.persistence_timer = ol.math.clamp(-ol.math.log(rng) * TAU_STOP, 0.1, 100.0)

    def _sample_direction(self, tissue: BrainTissue) -> ol.f32:
        """Вернуть угол направления: анизотропный (белое вещество) или равномерный."""
        if tissue.is_white_matter:
            u1: ol.f32 = ol.math.clamp(ol.random(), 0.0001, 1.0)
            u2: ol.f32 = ol.random()
            z:  ol.f32 = ol.math.sqrt(-2.0 * ol.math.log(u1)) * ol.math.cos(TWO_PI * u2)
            base = ol.math.atan2(tissue.tract_dir_y, tissue.tract_dir_x)
            return base + z * 0.52359877   # ± 30°
        return ol.random() * TWO_PI

    def move(self, m: ol.f32):
        """Сдвинуть клетку вдоль текущего направления на расстояние m."""
        self.prev_x = self.pos.x
        self.prev_y = self.pos.y
        self.pos.x  = ol.math.clamp(self.pos.x + self.move_dir_x * m, 0.0, 599.9)
        self.pos.y  = ol.math.clamp(self.pos.y + self.move_dir_y * m, 0.0, 599.9)

    def save_position(self):
        """Зафиксировать текущую позицию как «предыдущую» (без смещения)."""
        self.prev_x = self.pos.x
        self.prev_y = self.pos.y


class GBMMetrics(MetricsBase):
    infected_count:  ol.i32 = 0
    recruited_count: ol.i32 = 0
    inactive_count:  ol.i32 = 0
    total_cells:     ol.i32 = 0
    total_pdgf:      ol.f32 = 0.0

    def record(self, cell: GBMCell):
        """Обновить счётчики по одной клетке."""
        self.total_cells += 1
        if cell.cell_type == TYPE_INFECTED:  self.infected_count  += 1
        if cell.cell_type == TYPE_RECRUITED: self.recruited_count += 1
        if cell.cell_type == TYPE_INACTIVE:  self.inactive_count  += 1


class Params(ol.Params):
    treatment_ap: ol.bool = False
    treatment_am: ol.bool = False


# =============================================================================
# 3. ЛОГИКА СИМУЛЯЦИИ (WGSL Кернелы)
# =============================================================================

class MultiscaleGBM(ol.Simulation[BrainTissue, BrainChem, GBMCell, GBMMetrics, Params]):

    @ol.chemistry_rule(iterations=1)
    def update_pdgf_field(self, chem: BrainChem):
        chem.step_pdgf()

    @ol.cell_rule
    def cell_behavior(self, cell: GBMCell):
        chem    = self.chemistry_at(cell.pos)
        tissue  = self.tissue_at(cell.pos)
        c_pp    = ol.math.clamp(chem.pdgf, 0.0, 1_000_000.0)

        # Терапия AP: уничтожить активную клетку при достаточном потенциале
        if self.params.treatment_ap and cell.cell_type != TYPE_INACTIVE:
            if cell.p_pot >= AP_THRESHOLD:
                cell.die()
                return

        # Активация предшественника; дальнейшее поведение — в следующем шаге
        if cell.cell_type == TYPE_INACTIVE:
            cell.try_activate(c_pp)
            return

        p, m = cell.compute_phenotype(c_pp, apply_am=self.params.treatment_am)

        # Проверка перенаселённости
        active_neighbors: ol.i32 = 0
        for nb in cell.neighbors:
            active_neighbors += nb.cell_type != TYPE_INACTIVE
        is_quiescent = tissue.is_overcrowded(active_neighbors)

        # Деление
        if not is_quiescent:
            cell.attempt_division(p)

        # Переключение режима движения
        cell.tick_persistence(tissue)

        # Физическое смещение
        cell.save_position()
        if cell.is_moving and not is_quiescent and m > 0.0:
            cell.move(m)

    @ol.metric_rule
    def collect_metrics(self, cell: GBMCell, metrics: GBMMetrics):
        metrics.record(cell)


# =============================================================================
# 4. ИНИЦИАЛИЗАЦИЯ И РЕГИСТРАЦИЯ
# =============================================================================

def _init_tissue(engine, white_matter_j_range=(25, 31)):
    """Заполнить воксели ткани: обычное и белое вещество."""
    j_lo, j_hi = white_matter_j_range
    for i in range(40):
        for j in range(40):
            for k in range(2):
                vox = engine.tissue.sample(i, j, k)
                if j_lo <= j <= j_hi:
                    vox.init_white_matter()
                # Обычные вокселя используют значения по умолчанию из класса


def _init_chemistry(engine):
    """Задать гауссово начальное распределение PDGF в центре домена."""
    for i in range(20):
        for j in range(20):
            chem = engine.chemistry.sample(i, j, 0)
            cx, cy = i * 30.0 + 15.0, j * 30.0 + 15.0
            dist2  = (cx - 300.0) ** 2 + (cy - 300.0) ** 2
            chem.pdgf = 300.0 * math.exp(-dist2 / (2 * 50.0 ** 2))


def _init_cells(engine):
    """Разместить начальную популяцию клеток трёх типов."""
    # 10 инфицированных клеток в центре
    for _ in range(10):
        x = max(0.0, min(599.9, 300.0 + random.gauss(0, 5.0)))
        y = max(0.0, min(599.9, 300.0 + random.gauss(0, 5.0)))
        engine.cells.add(GBMCell(pos=ol.vec3(x, y, 7.5), cell_type=TYPE_INFECTED,  p_pot=0.025, m_pot=0.217))

    # 10 рекрутированных клеток рядом с центром
    for _ in range(10):
        x = max(0.0, min(599.9, 300.0 + random.gauss(0, 10.0)))
        y = max(0.0, min(599.9, 300.0 + random.gauss(0, 10.0)))
        engine.cells.add(GBMCell(pos=ol.vec3(x, y, 7.5), cell_type=TYPE_RECRUITED, p_pot=0.028, m_pot=0.25))

    # 5% от общей сетки — неактивные предшественники, равномерно распределённые
    for _ in range(int(40 * 40 * 0.05)):
        x = random.uniform(0.0, 599.9)
        y = random.uniform(0.0, 599.9)
        engine.cells.add(GBMCell(pos=ol.vec3(x, y, 7.5), cell_type=TYPE_INACTIVE,  p_pot=0.028, m_pot=0.25))


def build_simulation(seed: int = 42, backend: str = "gpu"):
    random.seed(seed)
    engine = ol.Engine(backend=backend)

    # Сетка 40×40×2; воксель = 15 мкм → физический домен 600×600 мкм
    engine.setup_geometry(tissue_shape=(40, 40, 2), tissue_voxel_size=15.0, cell_diameter=10.0)
    engine.setup_tissue(BrainTissue)
    engine.setup_chemistry(BrainChem)

    _init_tissue(engine)
    _init_chemistry(engine)
    _init_cells(engine)

    model = MultiscaleGBM()
    engine.load_model(model)
    return engine, model


def measure_tumor_diameter_mm(engine) -> float:
    """Оценить диаметр опухоли (в мм) по размаху активных клеток."""
    xs = [c.pos.x for c in engine.cells if getattr(c, "_alive", True) and c.cell_type != TYPE_INACTIVE]
    ys = [c.pos.y for c in engine.cells if getattr(c, "_alive", True) and c.cell_type != TYPE_INACTIVE]

    if len(xs) < 2:
        return 0.0

    diam_um = 0.5 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return diam_um / 1000.0
