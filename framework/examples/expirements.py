import matplotlib.pyplot as plt
from gbm_model import build_simulation

DAYS_TO_HOURS = 24


def run_simulation(engine, days, ap_day=None, am_day=None):
    history = {'day': [], 'tumor_burden': [], 'infected': [], 'recruited': []}

    for h in range(1, days * DAYS_TO_HOURS + 1):
        if ap_day and h == ap_day * DAYS_TO_HOURS:
            engine.update_params(treatment_ap=True)
        if am_day and h == am_day * DAYS_TO_HOURS:
            engine.update_params(treatment_am=True)

        collect = (h % DAYS_TO_HOURS == 0)
        engine.run_step(collect_metrics=collect)

        if collect:
            day = h // DAYS_TO_HOURS
            m = engine.get_metrics()
            history['day'].append(day)
            history['tumor_burden'].append(m.infected_count + m.recruited_count)
            history['infected'].append(m.infected_count)
            history['recruited'].append(m.recruited_count)

    cells_x_inf, cells_y_inf = [], []
    cells_x_rec, cells_y_rec = [], []
    engine.sync_to_host()
    for c in engine.cells:
        if c.cell_type == 1:
            cells_x_inf.append(c.pos.x)
            cells_y_inf.append(c.pos.y)
        elif c.cell_type == 2:
            cells_x_rec.append(c.pos.x)
            cells_y_rec.append(c.pos.y)

    spatial_data = (cells_x_inf, cells_y_inf, cells_x_rec, cells_y_rec)
    return history, spatial_data


def plot_spatial(ax, spatial_data, title):
    x_inf, y_inf, x_rec, y_rec = spatial_data
    ax.scatter(x_rec, y_rec, c='red', s=2, alpha=0.5, label='Recruited')
    ax.scatter(x_inf, y_inf, c='green', s=2, alpha=0.5, label='Infected')
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower right', markerscale=3, fontsize=7)


def run_all_experiments():
    print("=== Oncolytica GBM: Replicating Gallaher et al. ===")

    # 1. Базовый рост (17 дней)
    print("\n1. Running Baseline Growth (17 Days)...")
    eng_dyn = build_simulation(seed=100)
    hist_dyn, spatial_dyn = run_simulation(eng_dyn, days=17)
    print(f"   Day 17 | Infected: {hist_dyn['infected'][-1]}  "
          f"Recruited: {hist_dyn['recruited'][-1]}  "
          f"Total: {hist_dyn['tumor_burden'][-1]}")

    # 2. Четыре режима терапии (28 дней, старт с дня 14)
    print("\n2. Testing Therapies (start at Day 14)...")
    treatments = [
        ("Control",        None, None),
        ("AP Only",        14,   None),
        ("AM Only",        None, 14),
        ("AP+AM (Combo)",  14,   14),
    ]

    histories = {}
    spatials = {}
    colors = {
        "Control":       'gray',
        "AP Only":       '#4a9eda',
        "AM Only":       '#e06c75',
        "AP+AM (Combo)": '#c678dd',
    }

    for name, ap_d, am_d in treatments:
        print(f"  -> {name}")
        eng_tx = build_simulation(seed=42)
        hist, spat = run_simulation(eng_tx, days=28, ap_day=ap_d, am_day=am_d)
        histories[name] = hist
        spatials[name] = spat
        print(f"     Day 28 burden: {hist['tumor_burden'][-1]}")

    # 3. Рисуем
    print("\n3. Generating plots...")
    import numpy as np
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("GBM Agent-Based Model · Gallaher et al. (2020) Replication",
                 fontsize=13, fontweight='bold', y=0.98)

    # A: базовая динамика
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(hist_dyn['day'], hist_dyn['tumor_burden'], 'k-',  lw=2, label='Total active')
    ax1.plot(hist_dyn['day'], hist_dyn['recruited'],    'r--', lw=1.5, label='Recruited')
    ax1.plot(hist_dyn['day'], hist_dyn['infected'],     'g--', lw=1.5, label='Infected')
    ax1.set_yscale('log')
    ax1.set_title('A  Tumor growth dynamics\n(Fig 3 & 4)', fontsize=9)
    ax1.set_xlabel('Days'); ax1.set_ylabel('Cell count (log)')
    ax1.legend(fontsize=7)

    # B: I/R ratio
    ax2 = fig.add_subplot(2, 4, 2)
    ir = [i / max(r, 1) for i, r in zip(hist_dyn['infected'], hist_dyn['recruited'])]
    ax2.plot(hist_dyn['day'], ir, color='#f0a500', lw=2)
    ax2.axhline(1.0, color='gray', lw=0.8, linestyle=':')
    ax2.set_title('B  Infected / Recruited ratio\n(Fig 4B)', fontsize=9)
    ax2.set_xlabel('Days'); ax2.set_ylabel('I / R')
    ax2.set_ylim(bottom=0)

    # C: сравнение терапий
    ax3 = fig.add_subplot(2, 4, 3)
    for name, hist in histories.items():
        ax3.plot(hist['day'], hist['tumor_burden'],
                 label=name, color=colors[name], lw=2)
    ax3.axvline(14, color='red', linestyle=':', lw=1.2, label='Therapy start')
    ax3.set_title('C  Treatment responses\n(Fig 5 & 9)', fontsize=9)
    ax3.set_xlabel('Days'); ax3.set_ylabel('Active cell count')
    ax3.legend(fontsize=7)

    # D: Ki67-прокси
    ax4 = fig.add_subplot(2, 4, 4)
    tx_idx = 13  # день 14 → индекс 13
    names_ki = ["Control", "AP Only", "AP+AM (Combo)"]
    pre  = [histories[n]['infected'][tx_idx] / max(histories[n]['tumor_burden'][tx_idx], 1) * 100
            for n in names_ki]
    post = [histories[n]['infected'][-1]     / max(histories[n]['tumor_burden'][-1], 1)     * 100
            for n in names_ki]
    x = np.arange(len(names_ki))
    w = 0.35
    b1 = ax4.bar(x - w/2, pre,  w, label='Day 14 (pre-tx)',  color='#4a9eda', alpha=0.85)
    b2 = ax4.bar(x + w/2, post, w, label='Day 28 (post-tx)', color='#e06c75', alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.replace(' ', '\n') for n in names_ki], fontsize=8)
    ax4.set_title('D  Infected fraction (Ki67 proxy)\n(Fig 8)', fontsize=9)
    ax4.set_ylabel('% of active cells')
    ax4.legend(fontsize=7)
    for bar in list(b1) + list(b2):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=7)

    # E–H: пространственные карты
    spatial_order = ["Control", "AP Only", "AM Only", "AP+AM (Combo)"]
    subplot_letters = 'EFGH'
    for idx, name in enumerate(spatial_order):
        ax = fig.add_subplot(2, 4, 5 + idx)
        ax.set_facecolor('#080808')
        plot_spatial(ax, spatials[name], f'{subplot_letters[idx]}  {name}\n(Day 28)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'gbm_results.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Done! Saved: {out}")


if __name__ == "__main__":
    run_all_experiments()
