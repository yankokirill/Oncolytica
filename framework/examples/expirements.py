import time
import matplotlib.pyplot as plt
from gbm_model import build_simulation  # Импортируем вашу функцию сборки модели

DAYS_TO_HOURS = 24


def run_simulation(engine, model, days, ap_day=None, am_day=None):
    """
    Запускает симуляцию на заданное количество дней.
    Возвращает словарь с историей метрик.
    """
    history = {
        'day': [],
        'tumor_burden': [],
        'infected': [],
        'recruited': []
    }

    for h in range(1, days * DAYS_TO_HOURS + 1):
        # Включение терапии
        if ap_day and h == ap_day * DAYS_TO_HOURS:
            model.params.treatment_ap = True
            print(f"   [Day {ap_day}] Anti-Proliferative treatment STARTED")
        if am_day and h == am_day * DAYS_TO_HOURS:
            model.params.treatment_am = True
            print(f"   [Day {am_day}] Anti-Migratory treatment STARTED")

        # Собираем метрики раз в сутки
        collect = (h % DAYS_TO_HOURS == 0)
        engine.run_step(collect_metrics=collect)

        if collect:
            day = h // DAYS_TO_HOURS
            metrics = engine.get_metrics()

            # Простая метрика массы опухоли вместо геометрического диаметра
            tumor_burden = metrics.infected_count + metrics.recruited_count

            history['day'].append(day)
            history['tumor_burden'].append(tumor_burden)
            history['infected'].append(metrics.infected_count)
            history['recruited'].append(metrics.recruited_count)

    return history


def run_all_experiments():
    print("\n=== Oncolytica GBM: Replicating Gallaher et al. ===")

    # ---------------------------------------------------------
    # 1. Benchmark CPU vs GPU
    # ---------------------------------------------------------
    print("\n1. Running Benchmark (5 Days)...")

    t0 = time.perf_counter()
    eng_cpu, mod_cpu = build_simulation(backend="cpu", seed=42)
    run_simulation(eng_cpu, mod_cpu, days=5)
    t_cpu = time.perf_counter() - t0
    print(f"   CPU Time: {t_cpu:.2f} s")

    t0 = time.perf_counter()
    eng_gpu, mod_gpu = build_simulation(backend="gpu", seed=42)
    run_simulation(eng_gpu, mod_gpu, days=5)
    t_gpu = time.perf_counter() - t0
    print(f"   GPU Time: {t_gpu:.2f} s")

    if t_gpu > 0:
        print(f"   🚀 GPU Speedup: {t_cpu / t_gpu:.1f}x")

    # ---------------------------------------------------------
    # 2. Growth Dynamics (Fig 3 & 4 replication)
    # ---------------------------------------------------------
    print("\n2. Validating Baseline Growth Dynamics (17 Days)...")
    eng_dyn, mod_dyn = build_simulation(backend="gpu", seed=100)
    hist_dyn = run_simulation(eng_dyn, mod_dyn, days=17)

    print(f"   Day  5 Cells: {hist_dyn['tumor_burden'][4]}")
    print(f"   Day 10 Cells: {hist_dyn['tumor_burden'][9]}")
    print(
        f"   Day 17 Cells: {hist_dyn['tumor_burden'][16]} (Infected: {hist_dyn['infected'][16]}, Recruited: {hist_dyn['recruited'][16]})")

    # ---------------------------------------------------------
    # 3. Treatment Comparisons (Fig 5 & 9 replication)
    # ---------------------------------------------------------
    print("\n3. Testing Therapies (28 Days total, Treatment starts at Day 14)...")
    treatments = [
        ("Control", None, None),
        ("AP Only", 14, None),
        ("AM Only", None, 14),
        ("AP+AM", 14, 14)
    ]

    histories = {}
    for name, ap_d, am_d in treatments:
        print(f"\n   Running scenario: {name}")
        eng_tx, mod_tx = build_simulation(backend="gpu", seed=42)
        hist_tx = run_simulation(eng_tx, mod_tx, days=28, ap_day=ap_d, am_day=am_d)

        histories[name] = hist_tx

        d14_burden = hist_tx['tumor_burden'][13]
        d28_burden = hist_tx['tumor_burden'][27]
        print(f"   [{name}] Tumor cells: {d14_burden} (Day 14) -> {d28_burden} (Day 28)")

    # ---------------------------------------------------------
    # 4. Generating Plots
    # ---------------------------------------------------------
    print("\n4. Generating plots (validation_results.png)...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # A: Performance
    axs[0, 0].bar(['CPU', 'GPU'], [t_cpu, t_gpu], color=['#ff9999', '#66b3ff'])
    axs[0, 0].set_title('A. Performance Benchmark (5 days)')
    axs[0, 0].set_ylabel('Execution Time (seconds)')
    for i, v in enumerate([t_cpu, t_gpu]):
        axs[0, 0].text(i, v + (max(t_cpu, t_gpu) * 0.02), f"{v:.1f}s", ha='center')

    # B: Growth Dynamics (Recruited vs Infected)
    axs[0, 1].plot(hist_dyn['day'], hist_dyn['tumor_burden'], 'k-', lw=2, label='Total Tumor Cells')
    axs[0, 1].plot(hist_dyn['day'], hist_dyn['recruited'], 'r--', lw=2, label='Recruited Cells')
    axs[0, 1].plot(hist_dyn['day'], hist_dyn['infected'], 'g--', lw=2, label='Infected Cells')
    axs[0, 1].set_title('B. Tumor Growth Dynamics (Fig 3 & 4)')
    axs[0, 1].set_xlabel('Days')
    axs[0, 1].set_ylabel('Cell Count')
    axs[0, 1].legend()

    # C: Therapy Dynamics
    colors = {'Control': 'gray', 'AP Only': 'blue', 'AM Only': 'orange', 'AP+AM': 'purple'}
    for name, hist in histories.items():
        axs[1, 0].plot(hist['day'], hist['tumor_burden'], label=name, color=colors[name], lw=2)

    axs[1, 0].axvline(14, color='red', linestyle=':', label='Therapy Start')
    axs[1, 0].set_title('C. Treatment Responses (Fig 5 & 9)')
    axs[1, 0].set_xlabel('Days')
    axs[1, 0].set_ylabel('Tumor Cell Count')
    axs[1, 0].legend()

    # D: Therapy Delta (Bar chart of change from day 14 to 28)
    names = list(histories.keys())
    deltas = [histories[n]['tumor_burden'][27] - histories[n]['tumor_burden'][13] for n in names]

    bars = axs[1, 1].bar(names, deltas, color=[colors[n] for n in names])
    axs[1, 1].axhline(0, color='black', linewidth=1)
    axs[1, 1].set_title('D. Change in Tumor Burden (Day 14 -> 28)')
    axs[1, 1].set_ylabel('Δ Cell Count')

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        offset = 100 if yval >= 0 else -100
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2, yval + offset, int(yval), ha='center',
                       va='bottom' if yval >= 0 else 'top')

    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300)
    print("   ✅ Saved 'validation_results.png'. Done!")


if __name__ == "__main__":
    run_all_experiments()
