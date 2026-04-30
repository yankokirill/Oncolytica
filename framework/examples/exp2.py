"""
master_experiments.py — Unified Validation & Benchmark Suite for MultiscaleGBM.

Включает в себя:
1. Тест производительности (CPU vs GPU) на 14-дневной симуляции.
2. Воспроизведение 5 ключевых результатов из статьи Gallaher et al. (2020)
   с использованием разных фенотипов (Nodular, Intermediate, Diffuse).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import List, Dict

# Импорт модели
from gbm_model import (
    build_simulation, measure_tumor_diameter_mm,
    TYPE_INACTIVE, TYPE_INFECTED, TYPE_RECRUITED,
    VOXEL_UM
)

# ── ANSI Цвета ────────────────────────────────────────────────────────────────
GRN = "\033[92m";
RED = "\033[91m";
YLW = "\033[93m"
CYN = "\033[96m";
RST = "\033[0m";
BLD = "\033[1m"

DAYS_TO_HOURS = 24

# ── Фенотипы из статьи (для разных экспериментов) ─────────────────────────────
PROFILES = {
    "Nodular": {
        "p_pot_infected_mean": 0.04, "p_pot_recruited_mean": 0.045,
        "m_pot_infected_mean": 0.05, "m_pot_recruited_mean": 0.06,
        "sigma_p": 0.008, "sigma_m": 0.01
    },
    "Diffuse": {
        "p_pot_infected_mean": 0.015, "p_pot_recruited_mean": 0.018,
        "m_pot_infected_mean": 0.45, "m_pot_recruited_mean": 0.50,
        "sigma_p": 0.005, "sigma_m": 0.08
    },
    "Intermediate": {
        "p_pot_infected_mean": 0.025, "p_pot_recruited_mean": 0.028,
        "m_pot_infected_mean": 0.217, "m_pot_recruited_mean": 0.250,
        "sigma_p": 0.005, "sigma_m": 0.03
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ОБЩИЙ РАННЕР
# ─────────────────────────────────────────────────────────────────────────────

def _run(engine, sim, n_hours: int, *, quiet=False,
         ap_start: int = None, am_start: int = None) -> List[Dict]:
    snapshots = []
    for h in range(1, n_hours + 1):
        if ap_start is not None and h == ap_start:
            sim.treatment_ap = True
            if not quiet: print(f"  [h={h}] AP treatment started.")
        if am_start is not None and h == am_start:
            sim.treatment_am = True
            if not quiet: print(f"  [h={h}] AM treatment started.")

        engine.run_step()

        # Очистка мертвых клеток (опционально, зависит от реализации engine)
        if hasattr(engine.cells, "_data"):
            engine.cells._data = [c for c in engine.cells._data if getattr(c, "_alive", True)]

        if h % DAYS_TO_HOURS == 0:
            # Сбор метрик
            m = engine.metrics if hasattr(engine, 'metrics') else engine.get_metrics()
            day = h // DAYS_TO_HOURS
            diam = measure_tumor_diameter_mm(engine)
            n_inf = m.infected_count
            n_rec = m.recruited_count
            total = n_inf + n_rec
            ir = n_inf / n_rec if n_rec > 0 else float("inf")

            snap = dict(day=day, hour=h, diameter_mm=diam, infected=n_inf,
                        recruited=n_rec, inactive=m.inactive_count, total_active=total,
                        ir_ratio=ir, n_cells=len([c for c in engine.cells if getattr(c, "_alive", True)]))
            snapshots.append(snap)
            if not quiet:
                print(
                    f"  Day {day:2d}: D={diam:.2f}mm | inf={n_inf:4d} rec={n_rec:4d} | I/R={ir:.2f} | cells={snap['n_cells']}")
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# 0. БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark():
    print(f"\n{BLD}{CYN}═══ BENCHMARK: CPU vs GPU (10 дней симуляции) ═══{RST}")
    days = 10

    # 1. CPU Run
    print("Запуск на CPU...")
    engine_cpu, sim_cpu = build_simulation(seed=42, backend="cpu", **PROFILES["Intermediate"])
    t0_cpu = time.perf_counter()
    _run(engine_cpu, sim_cpu, days * DAYS_TO_HOURS, quiet=False)
    t_cpu = time.perf_counter() - t0_cpu

    # 2. GPU Run
    print("Запуск на GPU...")
    engine_gpu, sim_gpu = build_simulation(seed=42, backend="gpu", **PROFILES["Intermediate"])
    t0_gpu = time.perf_counter()
    _run(engine_gpu, sim_gpu, days * DAYS_TO_HOURS, quiet=False)
    t_gpu = time.perf_counter() - t0_gpu

    print(f"\n  Время CPU : {t_cpu:.2f} сек")
    print(f"  Время GPU : {t_gpu:.2f} сек")
    if t_gpu > 0:
        print(f"  {GRN}{BLD}Ускорение : {t_cpu / t_gpu:.1f}x{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТЫ (Используют GPU для скорости)
# ─────────────────────────────────────────────────────────────────────────────

def exp1_growth_dynamics(n_runs: int = 2):
    print(f"\n{BLD}{CYN}═══ Exp 1: Growth Dynamics (Fig. 4A) ═══{RST}")
    checkpoints = {5: [], 10: [], 17: []}

    for run in range(1, n_runs + 1):
        engine, sim = build_simulation(seed=run * 42, backend="gpu", **PROFILES["Intermediate"])
        snaps = _run(engine, sim, 17 * DAYS_TO_HOURS, quiet=False)
        for s in snaps:
            if s["day"] in checkpoints: checkpoints[s["day"]].append(s["diameter_mm"])

    print(f"\n{'Day':>4}  {'Mean diam (mm)':>14}  {'Target':>7}  Result")
    targets = {5: 1.7, 10: 2.4, 17: 3.2}
    for day in [5, 10, 17]:
        mean_d = sum(checkpoints[day]) / len(checkpoints[day])
        tgt = targets[day]
        ok = abs(mean_d - tgt) / tgt <= 0.5
        print(
            f"  {day:2d}d  {mean_d:>8.2f} mm      {tgt:>5.1f} mm  {GRN if ok else RED}{'PASS' if ok else 'FAIL'}{RST}")


def exp2_cell_composition():
    print(f"\n{BLD}{CYN}═══ Exp 2: Cell Composition (Fig. 4B) ═══{RST}")
    engine, sim = build_simulation(seed=17, backend="gpu", **PROFILES["Intermediate"])
    snaps = _run(engine, sim, 17 * DAYS_TO_HOURS, quiet=False)
    last = snaps[-1]
    rec_fraction = last["recruited"] / (last["infected"] + last["recruited"])
    print(f"  Day 17 Recruited Fraction: {rec_fraction:.1%} (Target > 50%)")


def exp4_antiproliferative():
    print(f"\n{BLD}{CYN}═══ Exp 4: Anti-proliferative Treatment (Fig. 5A) ═══{RST}")
    TOTAL_DAYS = 30
    results = {}
    for label, use_ap in [("Control", False), ("AP-treated", True)]:
        engine, sim = build_simulation(seed=1234, backend="gpu", **PROFILES["Intermediate"])
        sim.ap_threshold = 1.0 / 60.0  # Корректировка по статье (60h threshold)
        snaps = _run(engine, sim, TOTAL_DAYS * DAYS_TO_HOURS, quiet=False, ap_start=14 * 24 if use_ap else None)

        p_pots = [c.p_pot for c in engine.cells if getattr(c, "_alive", True) and c.cell_type != TYPE_INACTIVE]
        mean_ppot = sum(p_pots) / len(p_pots) if p_pots else 0.0
        results[label] = {"d": snaps[-1]["diameter_mm"], "p": mean_ppot}

    print(f"  Control    : D={results['Control']['d']:.2f}mm | Mean p_pot={results['Control']['p']:.4f}/h")
    print(f"  AP-treated : D={results['AP-treated']['d']:.2f}mm | Mean p_pot={results['AP-treated']['p']:.4f}/h")
    if results['AP-treated']['p'] < results['Control']['p']:
        print(f"  {GRN}Успех: Терапия AP произвела отбор медленно делящихся (резистентных) клеток.{RST}")


def exp5_treatment_comparison():
    print(f"\n{BLD}{CYN}═══ Exp 5: Diffuse Tumor Treatment (Fig. 9B) ═══{RST}")
    for label, do_ap, do_am in [("No Tx", False, False), ("AP only", True, False), ("AM only", False, True),
                                ("AP+AM", True, True)]:
        engine, sim = build_simulation(seed=777, backend="gpu", **PROFILES["Diffuse"])  # Используем Диффузный фенотип
        snaps = _run(engine, sim, 28 * DAYS_TO_HOURS, quiet=False,
                     ap_start=14 * 24 if do_ap else None, am_start=14 * 24 if do_am else None)
        d14 = snaps[13]["diameter_mm"]
        d28 = snaps[-1]["diameter_mm"]
        print(f"  {label:<8s} | D(14)={d14:.2f} -> D(28)={d28:.2f} | Δ = {d28 - d14:+.2f}mm")


if __name__ == "__main__":
    print(f"{BLD}Oncolytica · Master Suite (CPU/GPU Benchmark + Biol. Validation){RST}")
    print(f"{'─' * 60}")

    run_benchmark()
    exp1_growth_dynamics(n_runs=1)
    exp2_cell_composition()
    exp4_antiproliferative()
    exp5_treatment_comparison()