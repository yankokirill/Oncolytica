import time
import math
import numpy as np
import matplotlib.pyplot as plt
from gbm_model import build_simulation, measure_tumor_diameter_mm

# =============================================================================
# 5. ЭКСПЕРИМЕНТЫ И ГРАФИКИ
# =============================================================================

DAYS_TO_HOURS = 24


def run_simulation(engine, model, days, ap_day=None, am_day=None):
    history = {'day': [], 'diameter': [], 'infected': [], 'recruited': []}

    for h in range(1, days * DAYS_TO_HOURS + 1):
        if ap_day and h == ap_day * DAYS_TO_HOURS:
            model.treatment_ap = True
        if am_day and h == am_day * DAYS_TO_HOURS:
            model.treatment_am = True

        collect = (h % DAYS_TO_HOURS == 0)
        engine.run_step(collect)

        if collect:
            day = h // DAYS_TO_HOURS
            metrics = engine.get_metrics()
            diam = measure_tumor_diameter_mm(engine)

            history['day'].append(day)
            history['diameter'].append(diam)
            history['infected'].append(metrics.infected_count)
            history['recruited'].append(metrics.recruited_count)

    return history


def run_all_experiments():
    print("\n--- Oncolytica GBM Model Validation ---")

    # 1. Benchmark CPU vs GPU
    print("\n1. Running Benchmark (5 Days)...")
    t0 = time.perf_counter()
    eng_cpu, mod_cpu = build_simulation(backend="cpu")
    run_simulation(eng_cpu, mod_cpu, days=5)
    t_cpu = time.perf_counter() - t0
    print(f"   CPU Time: {t_cpu:.2f} s")

    t0 = time.perf_counter()
    eng_gpu, mod_gpu = build_simulation(backend="gpu")
    run_simulation(eng_gpu, mod_gpu, days=5)
    t_gpu = time.perf_counter() - t0
    print(f"   GPU Time: {t_gpu:.2f} s")
    print(f"   Speedup: {t_cpu / t_gpu:.1f}x")

    # 2. Growth Dynamics
    print("\n2. Validating Growth Dynamics (17 Days)...")
    eng_dyn, mod_dyn = build_simulation(backend="gpu", seed=100)
    hist_dyn = run_simulation(eng_dyn, mod_dyn, days=17)

    print(f"   Day 5  Diam: {hist_dyn['diameter'][4]:.2f} mm")
    print(f"   Day 10 Diam: {hist_dyn['diameter'][9]:.2f} mm")
    print(f"   Day 17 Diam: {hist_dyn['diameter'][16]:.2f} mm")

    # 3. Treatment Comparisons
    print("\n3. Testing Therapies (28 Days, Treatment at Day 14)...")
    treatments = [
        ("Control", None, None),
        ("AP Only", 14, None),
        ("AM Only", None, 14),
        ("AP+AM", 14, 14)
    ]

    tx_results = {}
    hist_ctrl, hist_ap = None, None
    for name, ap_d, am_d in treatments:
        eng_tx, mod_tx = build_simulation(backend="gpu", seed=42)
        hist_tx = run_simulation(eng_tx, mod_tx, days=28, ap_day=ap_d, am_day=am_d)

        d14 = hist_tx['diameter'][13]
        d28 = hist_tx['diameter'][27]
        tx_results[name] = d28 - d14
        print(f"   [{name}] ΔD (Day 14->28) = {d28 - d14:+.2f} mm")

        if name == "Control": hist_ctrl = hist_tx
        if name == "AP Only": hist_ap = hist_tx

    # 4. Generating Plots
    print("\n4. Generating plots (validation_results.png)...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot A
    axs[0, 0].bar(['CPU', 'GPU'], [t_cpu, t_gpu], color=['#ff9999', '#66b3ff'])
    axs[0, 0].set_title('Performance Benchmark (5 days)')
    axs[0, 0].set_ylabel('Execution Time (s)')

    # Plot B
    axs[0, 1].plot(hist_dyn['day'], hist_dyn['diameter'], 'k-', lw=2, label='Simulation')
    axs[0, 1].set_title('Tumor Growth Dynamics (Fig 4A)')
    axs[0, 1].set_xlabel('Days')
    axs[0, 1].set_ylabel('Diameter (mm)')
    axs[0, 1].legend()

    # Plot C
    axs[1, 0].plot(hist_ctrl['day'], hist_ctrl['diameter'], 'k--', label='Control')
    axs[1, 0].plot(hist_ap['day'], hist_ap['diameter'], 'b-', lw=2, label='Anti-Proliferative (AP)')
    axs[1, 0].axvline(14, color='red', linestyle=':', label='Therapy Start')
    axs[1, 0].set_title('Anti-Proliferative Treatment Response (Fig 5A)')
    axs[1, 0].set_xlabel('Days')
    axs[1, 0].set_ylabel('Diameter (mm)')
    axs[1, 0].legend()

    # Plot D
    names = list(tx_results.keys())
    deltas = list(tx_results.values())
    colors = ['gray', 'blue', 'orange', 'purple']
    axs[1, 1].bar(names, deltas, color=colors)
    axs[1, 1].axhline(0, color='black', linewidth=1)
    axs[1, 1].set_title('Change in Tumor Diameter (Day 14 -> 28) (Fig 9)')
    axs[1, 1].set_ylabel('Δ Diameter (mm)')

    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300)
    print("   Saved 'validation_results.png'. Done!")


if __name__ == "__main__":
    run_all_experiments()
