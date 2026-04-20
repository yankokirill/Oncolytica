"""
tests.py — тесты для Py2WGSL MVP 1
Запуск: python tests.py
"""

import sys
import textwrap
sys.path.insert(0, ".")
from transpiler import transpile, TranspileError


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  ✓  {name}")
        PASS += 1
    else:
        print(f"  ✗  {name}" + (f"\n       → {detail}" if detail else ""))
        FAIL += 1

def expect_ok(name: str, source: str, *fragments: str) -> str:
    """Транслирует source; проверяет наличие всех fragment в выводе."""
    try:
        result = transpile(textwrap.dedent(source))
        for frag in fragments:
            check(f"{name}: содержит «{frag}»", frag in result,
                  f"Вывод:\n{result}")
        return result
    except TranspileError as e:
        check(name, False, str(e))
        return ""

def expect_err(name: str, source: str, *fragments: str) -> None:
    """Транслирует source; ожидает TranspileError, содержащую fragments."""
    try:
        result = transpile(textwrap.dedent(source))
        check(name, False,
              f"Ожидалась ошибка, но код был успешно транслирован:\n{result}")
    except TranspileError as e:
        for frag in fragments:
            check(f"{name}: ошибка содержит «{frag}»", frag in str(e),
                  f"Полная ошибка: {e}")

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Тесты
# ─────────────────────────────────────────────────────────────────────────────

section("1. Struct")

expect_ok("Простой struct", """
    class Particle:
        x: float
        y: float
        mass: int
""",
    "struct Particle {",
    "x: f32,",
    "y: f32,",
    "mass: i32,",
)

expect_ok("Struct с векторным полем", """
    class Node:
        pos: vec3f
        vel: vec3f
        active: bool
""",
    "pos: vec3<f32>,",
    "vel: vec3<f32>,",
    "active: bool,",
)


section("2. Enum → const")

expect_ok("Enum-класс", """
    class State:
        Alive    = 0
        Necrotic = 1
        Deleted  = 2
""",
    "const STATE_ALIVE: i32 = 0;",
    "const STATE_NECROTIC: i32 = 1;",
    "const STATE_DELETED: i32 = 2;",
)


section("3. Свободные функции")

expect_ok("Простая функция", """
    def add(a: float, b: float) -> float:
        return a + b
""",
    "fn add(a: f32, b: f32) -> f32 {",
    "return (a + b);",
)

expect_ok("Функция с void-возвратом", """
    def noop(x: int) -> None:
        pass
""",
    "fn noop(x: i32) {",
)

expect_ok("if/elif/else", """
    def classify(x: float) -> int:
        if x < 0.0:
            return -1
        elif x > 1.0:
            return 1
        else:
            return 0
""",
    "if ((x < 0.0))",
    "} else if ((x > 1.0))",
    "} else {",
)

expect_ok("while loop", """
    def count(n: int) -> int:
        i: int = 0
        while i < n:
            i += 1
        return i
""",
    "while ((i < n)) {",
    "i += 1;",
    "var i: i32;",   # hoisted
)

expect_ok("for range(N)", """
    def sum_n(n: u32) -> u32:
        s: u32 = 0
        for i in range(n):
            s += i
        return s
""",
    "for (var i: u32 = 0u; i < n; i += 1u)",
    "s += i;",
)

expect_ok("for range(start, stop)", """
    def count_from(start: int, stop: int) -> int:
        s: int = 0
        for i in range(start, stop):
            s += i
        return s
""",
    "for (var i: i32 = start; i < stop; i += 1i)",
)


section("4. Hoisting переменных")

result = expect_ok("Hoisting в начало функции", """
    def f(x: float) -> float:
        y: float = x * 2.0
        z: float = y + 1.0
        return z
""",
    "var y: f32;",
    "var z: f32;",
)
# Объявления должны идти раньше присваиваний
if result:
    decl_y   = result.find("var y: f32;")
    assign_y = result.find("y = (x * 2.0);")
    check("Hoisting: var y раньше y = ...",
          decl_y < assign_y,
          f"pos decl={decl_y}, assign={assign_y}")


section("5. Топологическая сортировка")

result = expect_ok("Вызываемая функция объявлена раньше", """
    def caller(x: float) -> float:
        return helper(x)

    def helper(x: float) -> float:
        return x * 2.0
""",
    "fn helper",
    "fn caller",
)
if result:
    check("helper раньше caller",
          result.find("fn helper") < result.find("fn caller"))

expect_err("Цикл между функциями", """
    def a(x: float) -> float:
        return b(x)
    def b(x: float) -> float:
        return a(x)
""",
    "Обнаружена циклическая зависимость",
    "a, b",
)


section("6. @update_rule → compute kernel")

expect_ok("Базовый @update_rule", """
    class State:
        Alive    = 0
        Necrotic = 1

    class CellData:
        state:  int
        energy: float

    class Cell(CellData):
        @update_rule
        def update(self, neighbors, env, fields):
            if self.energy < 5.0:
                self.state = State.Necrotic
""",
    "@compute @workgroup_size(256, 1, 1)",
    "fn kernel_cell_update",
    "@builtin(global_invocation_id)",
    "var<storage, read> cells_read",
    "var<storage, read_write> cells_write",
    "var<uniform> total_agents: u32;",
    "var c: CellData = cells_read[i];",
    "cells_write[i] = c;",
    "if ((c.energy < 5.0))",
    "c.state = STATE_NECROTIC;",
)

expect_ok("Bindings group/binding корректны", """
    class PData:
        x: float

    class P(PData):
        @update_rule
        def update(self, n, e, f):
            pass
""",
    "@group(0) @binding(0)",
    "@group(0) @binding(1)",
    "@group(0) @binding(2)",
)

expect_ok("Enum в теле @update_rule", """
    class Phase:
        Liquid = 0
        Gas    = 1

    class ParticleData:
        phase: int

    class Particle(ParticleData):
        @update_rule
        def update(self, n, e, f):
            self.phase = Phase.Gas
""",
    "p.phase = PHASE_GAS;",
)


section("7. Обработка ошибок")

# Неизвестные имена типов проходят как struct-ref (намеренно).
# Проверяем, что dict действительно проходит без ошибки:
expect_ok("Неизвестный тип → struct ref (валидно)", """
    class Foo:
        x: MyStruct
""",
    "x: MyStruct,",
)
# Реальная ошибка: generic-тип, не list[T]
expect_err("Неподдерживаемый generic-тип", """
    class Foo:
        x: dict[str, int]
""",
    "Неподдерживаемый generic-тип",
)
# Проверяем list comprehension:
expect_err("List comprehension", """
    def f(n: int) -> int:
        return [x for x in range(n)]
""",
    "List comprehensions не поддерживаются",
)

expect_err("try/except", """
    def f(x: float) -> float:
        try:
            return x
        except:
            return 0.0
""",
    "try/except блоки не поддерживаются",
)

expect_err("Агент без определённого struct", """
    class Cell(UnknownData):
        @update_rule
        def update(self, n, e, f):
            pass
""",
    "UnknownData",
    "ещё не определён",
)

expect_err("Цикл из одной функции (самовызов)", """
    def f(x: float) -> float:
        return f(x)
""",
    "вызывает саму себя",
    "Рекурсия не поддерживается",
)


section("8. Полный пример (из ТЗ)")

FULL_EXAMPLE = """
class State:
    Alive    = 0
    Necrotic = 1
    Deleted  = 2

class CellData:
    idx:     int
    type_id: int
    state:   int
    energy:  float

def clamp_energy(e: float) -> float:
    if e < 0.0:
        return 0.0
    if e > 100.0:
        return 100.0
    return e

def apply_decay(energy: float, rate: float) -> float:
    result: float = energy - rate
    return clamp_energy(result)

class Cell(CellData):
    @update_rule
    def update(self, neighbors, env, fields):
        self.energy = apply_decay(self.energy, 0.5)
        if self.energy < 5.0:
            self.state = State.Necrotic
        elif self.energy > 80.0:
            self.state = State.Alive
"""

result = expect_ok("Полный пример транслируется без ошибок",
    FULL_EXAMPLE,
    "struct CellData",
    "fn clamp_energy",
    "fn apply_decay",
    "fn kernel_cell_update",
    "clamp_energy(result)",   # вызов внутри apply_decay
    "apply_decay(c.energy",
    "c.state = STATE_NECROTIC",
    "c.state = STATE_ALIVE",
)

# clamp_energy должна быть перед apply_decay
if result:
    check("clamp_energy перед apply_decay (topo sort)",
          result.find("fn clamp_energy") < result.find("fn apply_decay"))

if result:
    print("\n──── Сгенерированный WGSL ────────────────────────────────\n")
    print(result)


# ─────────────────────────────────────────────────────────────────────────────
# Итог
# ─────────────────────────────────────────────────────────────────────────────

section("Итог")
total = PASS + FAIL
print(f"\n  Прошло: {PASS}/{total}  |  Упало: {FAIL}/{total}\n")
sys.exit(0 if FAIL == 0 else 1)
