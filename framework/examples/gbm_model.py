"""
gbm_model.py — Multiscale GBM simulation based on Gallaher et al. (2020).

Reference:
    Gallaher J.A. et al. (2020). From cells to tissue: How cell scale
    heterogeneity impacts glioblastoma growth and treatment response.
    PLoS Comput Biol 16(2): e1007672.

Units
-----
Spatial : 1 voxel = 100 μm  (VOXEL_UM constant)
Temporal : 1 step  = 1 hour

Domain
------
145 × 100 × 1 voxels  ≈  14.5 mm × 10.0 mm  (2-D coronal brain slice)

Cell types
----------
0  TYPE_INACTIVE  — dormant recruitable progenitor (oligodendrocyte precursor)
1  TYPE_INFECTED  — PDGF-overexpressing retrovirus-infected cell (GFP, green)
2  TYPE_RECRUITED — activated progenitor driven by paracrine PDGF (dsRed, red)
"""

from __future__ import annotations

import math
import random
from typing import Tuple, Optional

import oncolytica as ol

# ── Cell-type integer constants ───────────────────────────────────────────────
# These are extracted as WGSL constants by the translator (§13.2).
TYPE_INACTIVE  = 0
TYPE_INFECTED  = 1
TYPE_RECRUITED = 2

# Math constants — extracted as WGSL const by the translator.
# Use these instead of math.pi inside @ol.*_rule methods.
PI     = 3.141592653589793
TWO_PI = 6.283185307179586

# ── Domain ────────────────────────────────────────────────────────────────────
GRID_X   = 145      # voxels (x)
GRID_Y   = 100      # voxels (y)
VOXEL_UM = 100.0    # μm per voxel
CENTER_X = GRID_X / 2.0
CENTER_Y = GRID_Y / 2.0

# Spatial-hash bucket size (engine cell_diameter).  Must be ≥ max neighbor radius.
HASH_CELL = 1.0     # voxels — enables correct ol.neighbors() up to radius=1.0


# =====================================================================
# 1.  MEMORY LAYOUTS
# =====================================================================

class BrainTissue(ol.TissueData):
    """One 100 μm tissue voxel: white/gray matter + local carrying capacity."""
    is_white_matter:    ol.bool
    tract_dir_x:        ol.f32   # unit-vector of white-matter tract (2-D)
    tract_dir_y:        ol.f32
    carrying_capacity:  ol.i32   # max active cells per voxel (κ or 2κ/3)


class BrainChem(ol.ChemistryData):
    """PDGF concentration in one voxel (ng/mL)."""
    pdgf: ol.f32


class GBMCell(ol.CellData):
    """Single cell agent.  pos.z is unused (2-D simulation, always 0)."""
    cell_type:          ol.i32   # TYPE_* constant above
    p_pot:              ol.f32   # max proliferation rate (1/h)  — Eq. 1 ppot
    m_pot:              ol.f32   # max migration speed  (vox/h)  — Eq. 1 mpot
    div_clock:          ol.f32   # [0,1) progress to next division
    is_moving:          ol.bool
    persistence_timer:  ol.f32   # hours left in current stop / go phase
    move_dir_x:         ol.f32   # unit movement direction (2-D)
    move_dir_y:         ol.f32
    prev_x:             ol.f32   # position at start of step (speed measurement)
    prev_y:             ol.f32


class GBMMetrics(ol.MetricsData):
    """Per-day aggregate statistics."""
    infected_count:  ol.i32
    recruited_count: ol.i32
    inactive_count:  ol.i32
    total_cells:     ol.i32
    total_pdgf:      ol.f32


# =====================================================================
# 2.  SIMULATION CLASS
# =====================================================================

class MultiscaleGBM(ol.Simulation[BrainTissue, BrainChem, GBMCell, GBMMetrics]):
    """
    Hybrid off-lattice agent-based model.

    Key biology modelled
    --------------------
    • PDGF field: secretion (infected), consumption (all active),
      first-order decay, and 2-D diffusion via chem.neighbors macro.
    • Per-cell Hill-function response  γ(c_pp)  (Eq. 2):
        infected  : (c_pa + c_pp) / (c_pa + c_pp + k)
        recruited : c_pp  / (c_pp + β·k)
    • Proliferation clock (Eq. 1):  p = p_pot · γ
    • Stop-and-go migration (Eq. 1): m = m_pot · γ
    • Quiescence when local density ≥ carrying capacity.
    • White-matter persistence (Gaussian angle bias; Box-Muller on GPU).
    • Inactive progenitors wake when c_pp > threshold.

    Parameters
    ----------
    Defaults calibrated to paper ranges (Table 1).  Override any of them
    when constructing to run sensitivity experiments.
    """

    def __init__(
        self,
        # ── PDGF dynamics ─────────────────────────────────────────────
        pdgf_d         = 0.0417,  # diffusion coeff  [vox²/h]
                                   # NOTE: the chem.neighbors macro sums 8 in-plane
                                   # neighbors (Moore 2D).  The chemistry rule applies
                                   # a 0.5 factor so the effective stencil matches the
                                   # original 4-point FD scheme; recalibrate if needed.
        pdgf_decay     = 0.005,   # fractional decay per step
        pdgf_secrete   = 0.417,   # infected secretion  [ng/mL·cell⁻¹·h⁻¹]
        pdgf_consume   = 0.042,   # all-cell consumption [ng/mL·cell⁻¹·h⁻¹]
        # ── Hill-function (Eq. 2) ─────────────────────────────────────
        k              = 100.0,   # half-max PDGF concentration [ng/mL]
        c_pa           = 5.0,     # autocrine boost for infected  [ng/mL]
        beta           = 0.5,     # recruited activation-barrier modifier β
        activation_thr = 5e-4,   # c_pp threshold to wake progenitors [ng/mL]
        # ── Migration (from ex-vivo data) ─────────────────────────────
        tau_move       = 0.71,    # mean go-persistence  [h]  (42.6 min / 60)
        tau_stop       = 1.17,    # mean stop-persistence [h] (70.1 min / 60)
        # ── Carrying capacity ─────────────────────────────────────────
        kappa          = 10,      # max active cells per voxel (gray matter)
        # ── Domain ────────────────────────────────────────────────────
        grid_x         = GRID_X,
        grid_y         = GRID_Y,
    ):
        super().__init__()
        self.pdgf_d          = pdgf_d
        self.pdgf_decay      = pdgf_decay
        self.pdgf_secrete    = pdgf_secrete
        self.pdgf_consume    = pdgf_consume
        self.k               = k
        self.c_pa            = c_pa
        self.beta            = beta
        self.activation_thr  = activation_thr
        self.tau_move        = tau_move
        self.tau_stop        = tau_stop
        self.kappa           = kappa
        self.grid_x          = grid_x
        self.grid_y          = grid_y
        self.cx              = grid_x / 2.0
        self.cy              = grid_y / 2.0

        # Treatment flags — toggled by experiment code
        self.treatment_ap    = False   # anti-proliferative
        self.ap_threshold    = 0.015   # kill cells with p_pot ≥ this [1/h]
        self.treatment_am    = False   # anti-migratory
        self.am_factor       = 0.10    # residual speed fraction under AM

    # ── Hill-function helpers (Eq. 2) ────────────────────────────────────────
    # Leading underscores removed: self._foo() → __foo() in WGSL (double
    # underscore) which is unexpected.  Without the underscore the translator
    # emits _gamma_infected() / _gamma_recruited() as intended.

    def gamma_infected(self, c_pp: float) -> float:
        """Autocrine + paracrine response for infected cells."""
        return (self.c_pa + c_pp) / (self.c_pa + c_pp + self.k)

    def gamma_recruited(self, c_pp: float) -> float:
        """Paracrine-only response with lowered barrier (β modifier).

        denom = c_pp + β·k ≥ β·k = 50 ng/mL at defaults → never zero.
        The original Python guard `if denom > 0 else 0.0` used ast.IfExp,
        which is not supported by the translator (§14); omitted safely.
        """
        denom = c_pp + self.beta * self.k
        return c_pp / denom

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Chemistry: diffusion + first-order decay
    # ─────────────────────────────────────────────────────────────────────────

    @ol.chemistry_rule()
    def update_pdgf_field(self, chem: BrainChem):
        """
        2-D explicit-scheme diffusion via the chem.neighbors macro (§8.4),
        followed by first-order decay — all in a single GPU kernel pass.

        In a z=1 domain the Moore macro yields 8 in-plane neighbors (z±1
        clamp to self → contribute 0).  The 0.5 factor corrects the sum
        back to an equivalent 4-point FD Laplacian for the same D value.

        Double-buffering guarantees all reads are from Chemistry_In while
        writes go to Chemistry_Out, so no explicit synchronisation is needed.
        """
        lap = 0.0
        for nb in chem.neighbors:
            lap += nb.pdgf - chem.pdgf
        # 0.5 factor: 8-neighbor sum ≈ 2 × 4-point Laplacian in 2D
        new_val = (chem.pdgf + self.pdgf_d * 0.5 * lap) * (1.0 - self.pdgf_decay)
        chem.pdgf = ol.math.clamp(new_val, 0.0, 1e15)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Cell loop (Fig. 2A flowchart)
    # ─────────────────────────────────────────────────────────────────────────

    @ol.cell_rule
    def cell_behavior(self, cell: GBMCell):
        """
        For each active cell implements the paper's cell loop:

        1.  Activation     — inactive → recruited if c_pp > threshold
        2.  γ(c_pp)        — Hill-function PDGF response (Eq. 1–2)
        3.  Quiescence     — suspend division + movement if density ≥ κ
        4.  Proliferation  — progress div_clock; divide when ≥ 1
        5.  PDGF exchange  — infected secrete; all cells consume
        6.  Stop-and-Go    — update persistence timer; sample new direction
        7.  Displacement   — move if is_moving and not quiescent

        All random draws use cell.random() (Xorshift32, §10).
        All math uses ol.math.* so the translator can emit WGSL built-ins.
        """
        chem   = self._engine.chemistry.sample(cell.pos)
        tissue = self._engine.tissue.sample(cell.pos)
        c_pp   = ol.math.clamp(chem.pdgf, 0.0, 1e15)

        # Pre-compute domain bounds as f32 to avoid i32/f32 promotion issues
        max_x = self.grid_x - 0.01
        max_y = self.grid_y - 0.01

        # ── Anti-proliferative treatment (Exp. 4 / 5) ─────────────────
        if self.treatment_ap and cell.cell_type != TYPE_INACTIVE:
            if cell.p_pot >= self.ap_threshold:
                cell.die()
                return

        # ── 1. Inactive progenitors: activation by PDGF ────────────────
        if cell.cell_type == TYPE_INACTIVE:
            if c_pp > self.activation_thr:
                cell.cell_type         = TYPE_RECRUITED
                cell.persistence_timer = self.tau_stop
            return   # inactive cells do nothing else this step

        # ── 2. γ(c_pp) — Hill-function response (Eq. 2) ──────────────
        # self.gamma_*(c_pp) calls are tracked by the call graph (§5) and
        # emitted as _gamma_infected() / _gamma_recruited() in WGSL.
        if cell.cell_type == TYPE_INFECTED:
            gamma = self.gamma_infected(c_pp)
        else:
            gamma = self.gamma_recruited(c_pp)

        p = cell.p_pot * gamma                # realized proliferation rate [1/h]
        m = cell.m_pot * gamma                # realized migration speed [vox/h]
        if self.treatment_am:
            m = m * self.am_factor            # *= not used: safe for local vars

        # ── 3. Quiescence check (Moore neighbourhood, 100 μm radius) ───
        # sum() generator comprehensions are not a supported macro form (§14).
        # Rewritten as explicit for-loop so the translator emits the Z-order
        # neighbor search correctly (§8.1).
        n_local = 0
        for nb in ol.neighbors(cell, radius=1.0):
            if nb.cell_type != TYPE_INACTIVE:
                n_local += 1
        is_quiescent = n_local >= tissue.carrying_capacity

        # ── 4. Proliferation ────────────────────────────────────────────
        if not is_quiescent and p > 0.0:
            cell.div_clock += p               # dt = 1 h
            if cell.div_clock >= 1.0:
                cell.div_clock -= 1.0

                # Daughter placed 25 μm away in a random direction
                div_a  = cell.random() * TWO_PI      # must be assigned first (§14.1)
                off    = 0.25                         # 25 μm / 100 μm·vox⁻¹
                nx     = ol.math.clamp(
                    cell.pos.x + off * ol.math.cos(div_a), 0.0, max_x
                )
                ny     = ol.math.clamp(
                    cell.pos.y + off * ol.math.sin(div_a), 0.0, max_y
                )

                # Daughter phenotype — random initial div_clock and stop timer
                d_clock = cell.random() * 0.3
                d_rng   = ol.math.clamp(cell.random(), 0.0001, 1.0)
                d_timer = ol.math.clamp(
                    -ol.math.log(d_rng) * self.tau_stop, 0.1, 100.0
                )

                daughter = GBMCell(
                    pos               = ol.vec3(nx, ny, 0.0),
                    cell_type         = cell.cell_type,
                    p_pot             = cell.p_pot,
                    m_pot             = cell.m_pot,
                    div_clock         = d_clock,
                    is_moving         = False,
                    persistence_timer = d_timer,
                    move_dir_x        = 0.0,
                    move_dir_y        = 0.0,
                    prev_x            = nx,
                    prev_y            = ny,
                )
                cell.divide(daughter)

        # ── 5. PDGF exchange ────────────────────────────────────────────
        if cell.cell_type == TYPE_INFECTED:
            chem.pdgf += self.pdgf_secrete
        chem.pdgf = ol.math.clamp(chem.pdgf - self.pdgf_consume, 0.0, 1e15)

        # ── 6. Stop-and-Go: update persistence timer ───────────────────
        cell.persistence_timer -= 1.0         # dt = 1 h
        if cell.persistence_timer <= 0.0:
            cell.is_moving = not cell.is_moving

            if cell.is_moving:
                # Exponential go-duration: -ln(U) * τ_move
                # (Inverse CDF of Exp; clamp U away from 0 to avoid log(0))
                go_u    = ol.math.clamp(cell.random(), 0.0001, 1.0)
                go_time = ol.math.clamp(-ol.math.log(go_u) * self.tau_move, 0.1, 100.0)
                cell.persistence_timer = go_time

                # Choose new direction
                if tissue.is_white_matter:
                    # Persistent random walk: Gaussian (σ=30°) around tract axis.
                    # Box-Muller transform — two uniform draws required.
                    # Each draw must be assigned before use (EnforceAssignmentPass).
                    bm_u1  = ol.math.clamp(cell.random(), 0.0001, 1.0)
                    bm_u2  = cell.random()
                    bm_z   = ol.math.sqrt(-2.0 * ol.math.log(bm_u1)) \
                             * ol.math.cos(TWO_PI * bm_u2)
                    base   = ol.math.atan2(tissue.tract_dir_y, tissue.tract_dir_x)
                    angle  = base + bm_z * (PI / 6.0)
                else:
                    # Uniform random walk in gray matter
                    angle = cell.random() * TWO_PI

                cell.move_dir_x = ol.math.cos(angle)
                cell.move_dir_y = ol.math.sin(angle)

            else:
                # Exponential stop-duration
                stop_u    = ol.math.clamp(cell.random(), 0.0001, 1.0)
                stop_time = ol.math.clamp(
                    -ol.math.log(stop_u) * self.tau_stop, 0.1, 100.0
                )
                cell.persistence_timer = stop_time

        # ── 7. Displacement ─────────────────────────────────────────────
        cell.prev_x = cell.pos.x
        cell.prev_y = cell.pos.y

        if cell.is_moving and not is_quiescent and m > 0.0:
            cell.pos.x = ol.math.clamp(
                cell.pos.x + cell.move_dir_x * m, 0.0, max_x
            )
            cell.pos.y = ol.math.clamp(
                cell.pos.y + cell.move_dir_y * m, 0.0, max_y
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Metrics (every 24 steps = daily)
    # ─────────────────────────────────────────────────────────────────────────

    @ol.metric_rule(interval=24)
    def collect_cell_stats(self, cell: GBMCell, metrics: GBMMetrics):
        metrics.total_cells += 1
        if cell.cell_type == TYPE_INFECTED:
            metrics.infected_count += 1
        elif cell.cell_type == TYPE_RECRUITED:
            metrics.recruited_count += 1
        else:
            metrics.inactive_count += 1

    @ol.metric_rule(interval=24)
    def collect_chem_stats(self, chem: BrainChem, metrics: GBMMetrics):
        metrics.total_pdgf += chem.pdgf


# =====================================================================
# 3.  SETUP HELPER  (CPU-only; uses Python stdlib random and math)
# =====================================================================

def build_simulation(
    n_infected           = 100,
    n_recruited          = 100,
    progenitor_density   = 0.02,    # fraction of voxels with an inactive cell
    initial_pdgf         = 300.0,   # ng/mL at center bolus
    p_pot_infected_mean  = 0.025,   # 1/h  (intermitotic time ~40 h)
    p_pot_recruited_mean = 0.028,
    m_pot_infected_mean  = 0.217,   # vox/h  (21.7 μm/h / 100 μm·vox⁻¹)
    m_pot_recruited_mean = 0.250,   # vox/h  (25.0 μm/h)
    sigma_p              = 0.005,   # std-dev of p_pot (intrinsic heterogeneity)
    sigma_m              = 0.030,   # std-dev of m_pot
    kappa_gray           = 10,
    kappa_white          = 7,       # ≈ 2κ/3 from paper
    seed: Optional[int]  = None,
    backend: str         = "cpu",
    **sim_kwargs,
) -> Tuple:
    """
    Build and initialise a full simulation following paper's initial conditions.

    Returns
    -------
    engine : ol.Engine
    sim    : MultiscaleGBM
    """
    if seed is not None:
        random.seed(seed)

    sim    = MultiscaleGBM(kappa=kappa_gray, **sim_kwargs)
    engine = ol.Engine(tissue_voxel_size=VOXEL_UM, cell_diameter=HASH_CELL, backend=backend)

    # ── Tissue field ───────────────────────────────────────────────────
    t_field = engine.setup_geometry(tissue_shape=(GRID_X, GRID_Y, 1))
    t_field = engine.setup_tissue(BrainTissue)
    c_field = engine.setup_chemistry(BrainChem)

    for i in range(GRID_X):
        for j in range(GRID_Y):
            vox = t_field[i, j, 0]
            # White matter: two bands mirroring corpus callosum region
            # (upper strip y=62–78, matching rat brain atlas geometry)
            wm = (62 <= j <= 78)
            vox.is_white_matter   = wm
            vox.carrying_capacity = kappa_white if wm else kappa_gray
            if wm:
                # Near-horizontal tract direction with small variability
                ang = random.gauss(0.0, math.pi / 6.0)
                vox.tract_dir_x = math.cos(ang)
                vox.tract_dir_y = math.sin(ang)

    # ── Chemistry field: initial PDGF bolus ───────────────────────────
    # Gaussian blob centred on injection site (paper: initial injury response)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            dist2 = (i - CENTER_X)**2 + (j - CENTER_Y)**2
            c_field[i, j, 0].pdgf = initial_pdgf * math.exp(-dist2 / (2 * 4.0))

    # ── Cell initialisation ────────────────────────────────────────────

    def _make_cell(x, y, ctype, p_mean, m_mean):
        return GBMCell(
            pos               = ol.vec3(x, y, 0.0),
            cell_type         = ctype,
            p_pot             = max(0.001, random.gauss(p_mean, sigma_p)),
            m_pot             = max(0.001, random.gauss(m_mean, sigma_m)),
            div_clock         = random.random(),
            is_moving         = False,
            persistence_timer = random.expovariate(1.0 / sim.tau_stop),
            move_dir_x        = 0.0,
            move_dir_y        = 0.0,
            prev_x            = x,
            prev_y            = y,
        )

    # 100 infected cells: tight cluster within 1 voxel (100 μm) of centre
    for _ in range(n_infected):
        r = random.random() * 1.0
        a = random.uniform(0.0, 2.0 * math.pi)
        x = ol.math.clamp(CENTER_X + r * math.cos(a), 0.0, GRID_X - 0.01)
        y = ol.math.clamp(CENTER_Y + r * math.sin(a), 0.0, GRID_Y - 0.01)
        engine.cells.add(_make_cell(x, y, TYPE_INFECTED,
                                    p_pot_infected_mean, m_pot_infected_mean))

    # 100 recruited cells: dispersed within 3 voxels (300 μm) of centre
    for _ in range(n_recruited):
        r = random.uniform(0.5, 3.0)
        a = random.uniform(0.0, 2.0 * math.pi)
        x = ol.math.clamp(CENTER_X + r * math.cos(a), 0.0, GRID_X - 0.01)
        y = ol.math.clamp(CENTER_Y + r * math.sin(a), 0.0, GRID_Y - 0.01)
        engine.cells.add(_make_cell(x, y, TYPE_RECRUITED,
                                    p_pot_recruited_mean, m_pot_recruited_mean))

    # Inactive progenitors: randomly distributed throughout domain
    n_prog = int(GRID_X * GRID_Y * progenitor_density)
    for _ in range(n_prog):
        x = random.uniform(0.0, GRID_X - 0.01)
        y = random.uniform(0.0, GRID_Y - 0.01)
        cell = GBMCell(
            pos               = ol.vec3(x, y, 0.0),
            cell_type         = TYPE_INACTIVE,
            p_pot             = max(0.001, random.gauss(p_pot_recruited_mean, sigma_p)),
            m_pot             = max(0.001, random.gauss(m_pot_recruited_mean, sigma_m)),
            div_clock         = 0.0,
            is_moving         = False,
            persistence_timer = 0.0,
            move_dir_x        = 0.0,
            move_dir_y        = 0.0,
            prev_x            = x,
            prev_y            = y,
        )
        engine.cells.add(cell)

    engine.load_model(sim)
    return engine, sim


# =====================================================================
# 4.  DIAMETER MEASUREMENT UTILITY  (CPU-only)
# =====================================================================

def measure_tumor_diameter_mm(engine: ol.Engine,
                               density_threshold: float = 0.1,
                               kappa: int = 10) -> float:
    """
    Estimate average maximum tumor diameter (mm).

    Method mirrors the paper: count cells in each voxel; find the maximum
    distance between voxels whose density exceeds `density_threshold × κ`.
    Returns diameter in mm.
    """
    density: dict = {}
    for cell in engine.cells:
        if cell.cell_type == TYPE_INACTIVE:
            continue
        ix = int(cell.pos.x)
        iy = int(cell.pos.y)
        density[(ix, iy)] = density.get((ix, iy), 0) + 1

    thr_count = density_threshold * kappa
    active_voxels = [xy for xy, cnt in density.items() if cnt >= thr_count]

    if len(active_voxels) < 2:
        max_r = 0.0
        for cell in engine.cells:
            if cell.cell_type == TYPE_INACTIVE:
                continue
            r = math.sqrt((cell.pos.x - CENTER_X)**2 + (cell.pos.y - CENTER_Y)**2)
            if r > max_r:
                max_r = r
        return 2.0 * max_r * VOXEL_UM / 1000.0   # mm

    xs = [v[0] for v in active_voxels]
    ys = [v[1] for v in active_voxels]
    diam_vox = 0.5 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return diam_vox * VOXEL_UM / 1000.0           # mm


if __name__ == '__main__':
    x = 240
    d = []
    while x > 0:
        d.append(str(x % 2))
        x //= 2
    print(''.join(d[::-1]))