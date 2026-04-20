# example.py — тестовый входной файл для Py2WGSL

# ── Enum-класс → WGSL const ───────────────────────────────────────────────────

class State:
    Alive    = 0
    Necrotic = 1
    Deleted  = 2


# ── Struct → WGSL struct ──────────────────────────────────────────────────────

class CellData:
    idx:     int
    type_id: int
    state:   int
    energy:  float


# ── Вспомогательная функция (будет тополого-отсортирована) ────────────────────

def clamp_energy(e: float) -> float:
    if e < 0.0:
        return 0.0
    if e > 100.0:
        return 100.0
    return e


def apply_decay(energy: float, rate: float) -> float:
    result: float = energy - rate
    return clamp_energy(result)


# ── Класс-агент → compute kernel ─────────────────────────────────────────────

class Cell(CellData):

    @update_rule
    def update(self, neighbors, env, fields):
        """Правило обновления клетки."""

        # Применяем затухание
        self.energy = apply_decay(self.energy, 0.5)

        # Проверяем жизнеспособность
        if self.energy < 5.0:
            self.state = State.Necrotic
        elif self.energy > 80.0:
            self.state = State.Alive
