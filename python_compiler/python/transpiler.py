"""
Py2WGSL Transpiler — MVP 1
==========================

Поддерживаемые конструкции:
  - Structs   : класс только с аннотированными полями  → WGSL struct
  - Enums     : класс только с целочисленными присваиваниями → WGSL const
  - Agents    : класс с базовым struct + @update_rule  → compute kernel
  - Скалярные типы : int → i32, float → f32, bool → bool
  - Векторные типы : vec2f, vec3f, vec4f, vec3u, …
  - Арифметика, сравнения, булевы операции
  - if / elif / else, while, for i in range(N)
  - Объявления переменных поднимаются в начало функции (hoisting)
  - Топологическая сортировка свободных функций, ошибка при цикле
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Ошибки
# ─────────────────────────────────────────────────────────────────────────────

class TranspileError(Exception):
    def __init__(self, lineno: Optional[int], message: str):
        self.lineno = lineno
        self.message = message
        loc = f"line {lineno}" if lineno is not None else "unknown location"
        super().__init__(f"Error at {loc}: {message}")


# ─────────────────────────────────────────────────────────────────────────────
# Таблицы маппинга
# ─────────────────────────────────────────────────────────────────────────────

TYPE_MAP: dict[str, str] = {
    "int":     "i32",  "float":   "f32",  "bool":    "bool",
    "u32":     "u32",  "i32":     "i32",  "f32":     "f32",
    "vec2f":   "vec2<f32>", "vec3f":   "vec3<f32>", "vec4f":   "vec4<f32>",
    "vec2i":   "vec2<i32>", "vec3i":   "vec3<i32>", "vec4i":   "vec4<i32>",
    "vec2u":   "vec2<u32>", "vec3u":   "vec3<u32>", "vec4u":   "vec4<u32>",
    "mat4x4f": "mat4x4<f32>",
}

BUILTIN_FUNC_MAP: dict[str, str] = {
    k: k for k in (
        "min", "max", "abs", "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "dot", "normalize", "pow",
        "sqrt", "floor", "ceil", "round", "clamp",
        "mix", "step", "smoothstep", "cross",
        "reflect", "sign", "fract",
        "log", "log2", "exp", "exp2", "distance",
    )
}

BIN_OP_MAP: dict[type, str] = {
    ast.Add: "+",  ast.Sub: "-",  ast.Mult: "*",   ast.Div: "/",
    ast.Mod: "%",  ast.FloorDiv: "/",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
}
CMP_OP_MAP: dict[type, str] = {
    ast.Eq: "==", ast.NotEq: "!=",
    ast.Lt: "<",  ast.LtE: "<=",
    ast.Gt: ">",  ast.GtE: ">=",
}
BOOL_OP_MAP: dict[type, str] = {ast.And: "&&", ast.Or: "||"}
UNARY_OP_MAP: dict[type, str] = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!"}

# Конструкции Python, которые точно не поддерживаются
UNSUPPORTED_STMT_NAMES: dict[type, str] = {
    ast.Try:          "try/except блоки",
    ast.With:         "оператор 'with'",
    ast.Delete:       "оператор 'del'",
    ast.Raise:        "оператор 'raise'",
    ast.Assert:       "оператор 'assert'",
    ast.Global:       "оператор 'global'",
    ast.Nonlocal:     "оператор 'nonlocal'",
    ast.AsyncFor:     "async for",
    ast.AsyncWith:    "async with",
    ast.AsyncFunctionDef: "async функции",
}


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _map_type(node: ast.expr, lineno: int) -> str:
    """Аннотация Python → строка WGSL-типа."""
    if isinstance(node, ast.Name):
        return TYPE_MAP.get(node.id, node.id)   # неизвестное имя → struct ref
    if isinstance(node, ast.Attribute):
        raise TranspileError(lineno,
            f"Квалифицированный тип '{ast.unparse(node)}' не поддерживается. "
            "Используйте простое имя типа.")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "list":
            inner = _map_type(node.slice, lineno)
            return f"array<{inner}>"
        raise TranspileError(lineno,
            f"Неподдерживаемый generic-тип '{ast.unparse(node)}'. "
            "Из обобщённых типов поддерживается только 'list[T]'.")
    raise TranspileError(lineno,
        f"Невозможно транслировать аннотацию типа '{ast.unparse(node)}'.")


def _camel_to_snake(name: str) -> str:
    """'CellData' → 'cell_data'"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _strip_data_suffix(name: str) -> str:
    """'CellData' → 'Cell', 'Particle' → 'Particle'"""
    return name[:-4] if name.endswith("Data") else name


def _plural_snake(name: str) -> str:
    """'Cell' → 'cells', 'CellData' → 'cells'"""
    base = _strip_data_suffix(name)
    return _camel_to_snake(base) + "s"


def _has_decorator(fdef: ast.FunctionDef, name: str) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == name)
        for d in fdef.decorator_list
    )


# ─────────────────────────────────────────────────────────────────────────────
# Транспилятор выражений
# ─────────────────────────────────────────────────────────────────────────────

class ExprTranspiler(ast.NodeVisitor):
    """
    Обходит AST-узел выражения и возвращает строку WGSL-выражения.
    self_alias: имя, которым заменяется 'self' (например, 'c').
    enum_names: имена классов-перечислений (State.Alive → STATE_ALIVE).
    """

    def __init__(self, enum_names: set[str], self_alias: Optional[str] = None):
        self._enums = enum_names
        self._self = self_alias

    def tx(self, node: ast.expr) -> str:
        return self.visit(node)

    # ── Литералы ──────────────────────────────────────────────────────────────

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            s = repr(v)
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        raise TranspileError(node.lineno,
            f"Неподдерживаемый тип литерала: {type(v).__name__}.")

    # ── Имена ─────────────────────────────────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> str:
        if self._self and node.id == "self":
            return self._self
        return node.id

    def visit_Attribute(self, node: ast.Attribute) -> str:
        # Доступ к enum: State.Necrotic → STATE_NECROTIC
        if isinstance(node.value, ast.Name) and node.value.id in self._enums:
            return f"{node.value.id.upper()}_{node.attr.upper()}"
        obj = self.visit(node.value)
        return f"{obj}.{node.attr}"

    # ── Операции ──────────────────────────────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> str:
        op = BIN_OP_MAP.get(type(node.op))
        if op is None:
            raise TranspileError(node.lineno,
                f"Неподдерживаемый бинарный оператор '{type(node.op).__name__}'.")
        return f"({self.tx(node.left)} {op} {self.tx(node.right)})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        op = UNARY_OP_MAP.get(type(node.op))
        if op is None:
            raise TranspileError(node.lineno,
                f"Неподдерживаемый унарный оператор '{type(node.op).__name__}'.")
        return f"({op}{self.tx(node.operand)})"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = BOOL_OP_MAP[type(node.op)]
        parts = f" {op} ".join(f"({self.tx(v)})" for v in node.values)
        return f"({parts})"

    def visit_Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1:
            raise TranspileError(node.lineno,
                "Цепочки сравнений (a < b < c) не поддерживаются. "
                "Используйте явный 'and' для объединения сравнений.")
        op = CMP_OP_MAP.get(type(node.ops[0]))
        if op is None:
            raise TranspileError(node.lineno,
                f"Неподдерживаемый оператор сравнения '{type(node.ops[0]).__name__}'.")
        return f"({self.tx(node.left)} {op} {self.tx(node.comparators[0])})"

    # ── Вызовы ────────────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> str:
        if node.keywords:
            raise TranspileError(node.lineno,
                "Именованные аргументы (kwargs) не поддерживаются.")

        args_str = ", ".join(self.tx(a) for a in node.args)

        if isinstance(node.func, ast.Name):
            fname = node.func.id
            # Приведение типов: int(x) → i32(x)
            if fname in TYPE_MAP:
                return f"{TYPE_MAP[fname]}({args_str})"
            # Встроенные функции WGSL
            if fname in BUILTIN_FUNC_MAP:
                return f"{BUILTIN_FUNC_MAP[fname]}({args_str})"
            # Пользовательская функция или конструктор вектора
            return f"{fname}({args_str})"

        if isinstance(node.func, ast.Attribute):
            obj = self.tx(node.func.value)
            return f"{obj}.{node.func.attr}({args_str})"

        raise TranspileError(node.lineno,
            "Неподдерживаемое выражение вызова.")

    # ── Индексирование ────────────────────────────────────────────────────────

    def visit_Subscript(self, node: ast.Subscript) -> str:
        return f"{self.tx(node.value)}[{self.tx(node.slice)}]"

    # ── Тернарный оператор ────────────────────────────────────────────────────

    def visit_IfExp(self, node: ast.IfExp) -> str:
        # a if cond else b  →  select(b, a, cond)
        return (f"select({self.tx(node.orelse)}, "
                f"{self.tx(node.body)}, {self.tx(node.test)})")

    # ── Запрещённые выражения ─────────────────────────────────────────────────

    def visit_ListComp(self, node: ast.ListComp) -> str:
        raise TranspileError(node.lineno,
            "List comprehensions не поддерживаются. Используйте цикл 'for'.")

    def visit_Lambda(self, node: ast.Lambda) -> str:
        raise TranspileError(node.lineno,
            "Lambda-выражения не поддерживаются.")

    def generic_visit(self, node: ast.AST) -> str:
        raise TranspileError(
            getattr(node, "lineno", None),
            f"Неподдерживаемый тип выражения '{type(node).__name__}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Сбор объявлений переменных для hoisting
# ─────────────────────────────────────────────────────────────────────────────

def _collect_vars(stmts: list[ast.stmt]) -> dict[str, str]:
    """
    Рекурсивно собирает все аннотированные объявления (AnnAssign) в теле функции.
    Возвращает {имя_переменной: wgsl_тип}.
    Бросает исключение при конфликте типов для одной переменной.
    """
    result: dict[str, str] = {}

    def _walk(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                wtype = _map_type(node.annotation, node.lineno)
                if name in result and result[name] != wtype:
                    raise TranspileError(node.lineno,
                        f"Переменная '{name}' объявлена с конфликтующими типами: "
                        f"'{result[name]}' и '{wtype}'.")
                result[name] = wtype
            # Рекурсия в составные операторы
            if isinstance(node, ast.If):
                _walk(node.body)
                _walk(node.orelse)
            elif isinstance(node, (ast.While, ast.For)):
                _walk(node.body)

    _walk(stmts)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Транспилятор операторов
# ─────────────────────────────────────────────────────────────────────────────

class StmtTranspiler:
    """
    Транслирует список Python-операторов в список строк WGSL.
    self_alias : 'self' заменяется на это имя (None → без замены).
    enum_names : множество имён enum-классов.
    indent_size: размер отступа в пробелах.
    """

    def __init__(
        self,
        enum_names: set[str],
        self_alias: Optional[str] = None,
        indent_size: int = 4,
    ):
        self._indent = indent_size
        self._expr = ExprTranspiler(enum_names, self_alias)

    def tx_body(self, stmts: list[ast.stmt], depth: int = 1) -> list[str]:
        lines: list[str] = []
        for stmt in stmts:
            lines.extend(self._tx_stmt(stmt, depth))
        return lines

    def _pad(self, depth: int) -> str:
        return " " * (self._indent * depth)

    def _tx_stmt(self, node: ast.stmt, depth: int) -> list[str]:
        pad = self._pad(depth)

        # ── Аннотированное присваивание: x: int = expr ──────────────────────
        if isinstance(node, ast.AnnAssign):
            # Объявление поднято наверх (hoisting); здесь только присваивание
            if node.value is not None:
                target = self._expr.tx(node.target)
                value = self._expr.tx(node.value)
                return [f"{pad}{target} = {value};"]
            return []  # голая аннотация без значения

        # ── Обычное присваивание: x = expr ──────────────────────────────────
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise TranspileError(node.lineno,
                    "Множественные цели присваивания (a = b = expr) не поддерживаются.")
            target = self._expr.tx(node.targets[0])
            value = self._expr.tx(node.value)
            return [f"{pad}{target} = {value};"]

        # ── Составное присваивание: x += expr ───────────────────────────────
        if isinstance(node, ast.AugAssign):
            op = BIN_OP_MAP.get(type(node.op))
            if op is None:
                raise TranspileError(node.lineno,
                    f"Неподдерживаемый оператор составного присваивания "
                    f"'{type(node.op).__name__}'.")
            target = self._expr.tx(node.target)
            value = self._expr.tx(node.value)
            return [f"{pad}{target} {op}= {value};"]

        # ── return ───────────────────────────────────────────────────────────
        if isinstance(node, ast.Return):
            if node.value is None:
                return [f"{pad}return;"]
            return [f"{pad}return {self._expr.tx(node.value)};"]

        # ── Вызов как оператор ───────────────────────────────────────────────
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                return [f"{pad}{self._expr.tx(node.value)};"]
            if isinstance(node.value, ast.Constant):
                return []  # docstring внутри тела функции — игнорируем
            raise TranspileError(node.lineno,
                f"Выражение-оператор '{ast.unparse(node.value)}' не поддерживается "
                "(только вызовы функций допустимы как самостоятельный оператор).")

        # ── if / elif / else ─────────────────────────────────────────────────
        if isinstance(node, ast.If):
            return self._tx_if(node, depth)

        # ── while ────────────────────────────────────────────────────────────
        if isinstance(node, ast.While):
            if node.orelse:
                raise TranspileError(node.lineno,
                    "while/else не поддерживается.")
            cond = self._expr.tx(node.test)
            lines = [f"{pad}while ({cond}) {{"]
            lines.extend(self.tx_body(node.body, depth + 1))
            lines.append(f"{pad}}}")
            return lines

        # ── for i in range(...) ──────────────────────────────────────────────
        if isinstance(node, ast.For):
            return self._tx_for(node, depth)

        # ── pass / break / continue ──────────────────────────────────────────
        if isinstance(node, ast.Pass):
            return []
        if isinstance(node, ast.Break):
            return [f"{pad}break;"]
        if isinstance(node, ast.Continue):
            return [f"{pad}continue;"]

        # ── Явно неподдерживаемые конструкции ────────────────────────────────
        for node_type, desc in UNSUPPORTED_STMT_NAMES.items():
            if isinstance(node, node_type):
                raise TranspileError(
                    getattr(node, "lineno", None),
                    f"{desc} не поддерживаются. "
                    "Только ограниченное подмножество Python транслируется в WGSL.")

        raise TranspileError(
            getattr(node, "lineno", None),
            f"Неподдерживаемый оператор '{type(node).__name__}'.")

    def _tx_if(self, node: ast.If, depth: int) -> list[str]:
        pad = self._pad(depth)
        cond = self._expr.tx(node.test)
        lines = [f"{pad}if ({cond}) {{"]
        lines.extend(self.tx_body(node.body, depth + 1))

        orelse = node.orelse
        if not orelse:
            lines.append(f"{pad}}}")
        elif len(orelse) == 1 and isinstance(orelse[0], ast.If):
            # elif: рекурсивно генерируем внутренний if и присоединяем через '} else'
            inner = self._tx_if(orelse[0], depth)
            lines.append(f"{pad}}} else {inner[0].lstrip()}")
            lines.extend(inner[1:])
        else:
            lines.append(f"{pad}}} else {{")
            lines.extend(self.tx_body(orelse, depth + 1))
            lines.append(f"{pad}}}")

        return lines

    def _tx_for(self, node: ast.For, depth: int) -> list[str]:
        pad = self._pad(depth)

        if node.orelse:
            raise TranspileError(node.lineno, "for/else не поддерживается.")

        if not (
            isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name) and
            node.iter.func.id == "range"
        ):
            raise TranspileError(node.lineno,
                "Только 'for var in range(...)' поддерживается. "
                "Для произвольной итерации используйте 'while'.")

        if not isinstance(node.target, ast.Name):
            raise TranspileError(node.lineno,
                "Переменная цикла должна быть простым именем.")

        var = node.target.id
        args = node.iter.args

        if len(args) == 1:
            start, stop, step = "0u", self._expr.tx(args[0]), "1u"
            vtype = "u32"
        elif len(args) == 2:
            start, stop, step = self._expr.tx(args[0]), self._expr.tx(args[1]), "1i"
            vtype = "i32"
        elif len(args) == 3:
            start = self._expr.tx(args[0])
            stop  = self._expr.tx(args[1])
            step  = self._expr.tx(args[2])
            vtype = "i32"
        else:
            raise TranspileError(node.lineno, "range() принимает 1–3 аргумента.")

        init = f"var {var}: {vtype} = {start}"
        cond = f"{var} < {stop}"
        incr = f"{var} += {step}"

        lines = [f"{pad}for ({init}; {cond}; {incr}) {{"]
        lines.extend(self.tx_body(node.body, depth + 1))
        lines.append(f"{pad}}}")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Внутренние структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructDef:
    name: str
    fields: list[tuple[str, str]]   # (имя_поля, wgsl_тип)

@dataclass
class EnumDef:
    name: str
    members: list[tuple[str, int]]  # (имя_члена, целое_значение)

@dataclass
class KernelDef:
    fn_name: str          # имя функции в WGSL (kernel_cell_update)
    data_struct: str      # имя struct (CellData)
    agent_name: str       # имя агента без суффикса Data (Cell)
    body_stmts: list[ast.stmt]
    binding_group: int
    binding_offset: int


# ─────────────────────────────────────────────────────────────────────────────
# Топологическая сортировка функций
# ─────────────────────────────────────────────────────────────────────────────

def _topo_sort(funcs: dict[str, ast.FunctionDef]) -> list[str]:
    """
    Возвращает имена функций в порядке «вызываемые перед вызывающими».
    Бросает TranspileError при обнаружении цикла.
    """
    # Сначала проверяем прямую рекурсию (самовызов)
    for name, fdef in funcs.items():
        for node in ast.walk(fdef):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == name):
                raise TranspileError(
                    getattr(node, "lineno", None),
                    f"Функция '{name}' вызывает саму себя. "
                    "Рекурсия не поддерживается в WGSL.")

    # Граф зависимостей: name → множество имён функций, которые name вызывает
    calls: dict[str, set[str]] = {name: set() for name in funcs}
    for name, fdef in funcs.items():
        for node in ast.walk(fdef):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id in funcs and
                    node.func.id != name):
                calls[name].add(node.func.id)

    # Кан: in_degree[A] = количество функций, которые A вызывает (зависимости A)
    in_degree: dict[str, int] = {name: len(deps) for name, deps in calls.items()}

    # Обратный граф: rdeps[B] = {A, …} — кто зависит от B
    rdeps: dict[str, set[str]] = defaultdict(set)
    for name, deps in calls.items():
        for dep in deps:
            rdeps[dep].add(name)

    queue = sorted(name for name in funcs if in_degree[name] == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(rdeps[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(funcs):
        cycle = sorted(n for n in funcs if n not in result)
        raise TranspileError(None,
            f"Обнаружена циклическая зависимость между функциями: "
            f"{', '.join(cycle)}. "
            "Рекурсия не поддерживается в WGSL.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Главный транспайлер
# ─────────────────────────────────────────────────────────────────────────────

class Py2WGSLTranspiler:
    WORKGROUP_SIZE = 256

    def __init__(self):
        self._structs: dict[str, StructDef] = {}
        self._enums:   dict[str, EnumDef]   = {}
        self._kernels: list[KernelDef]       = []
        self._functions: dict[str, ast.FunctionDef] = {}
        self._binding_counter = 0

    # ── Публичный API ─────────────────────────────────────────────────────────

    def transpile(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise TranspileError(e.lineno, f"Синтаксическая ошибка Python: {e.msg}") from e

        self._collect(tree)
        return self._generate()

    # ── Первый проход: сбор информации ───────────────────────────────────────

    def _collect(self, tree: ast.Module) -> None:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._collect_class(node)
            elif isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                pass  # импорты тихо игнорируются
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                pass  # docstring модуля
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                pass  # константы уровня модуля (пока не поддерживаются)
            else:
                raise TranspileError(
                    getattr(node, "lineno", None),
                    f"Неподдерживаемый оператор верхнего уровня '{type(node).__name__}'.")

    def _collect_class(self, node: ast.ClassDef) -> None:
        """Определяет тип класса и регистрирует его."""
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]

        # Тело без docstring и pass
        body = [
            s for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            and not isinstance(s, ast.Pass)
        ]

        ann_fields  = [s for s in body if isinstance(s, ast.AnnAssign)]
        int_assigns = [s for s in body if isinstance(s, ast.Assign)]
        methods     = [s for s in body if isinstance(s, ast.FunctionDef)]
        other       = [s for s in body
                       if not isinstance(s, (ast.AnnAssign, ast.Assign, ast.FunctionDef))]

        if other:
            raise TranspileError(
                getattr(other[0], "lineno", None),
                f"Неподдерживаемый оператор внутри класса '{node.name}'.")

        # ── Агентный класс: есть базовый struct и @update_rule ────────────────
        update_methods = [m for m in methods if _has_decorator(m, "update_rule")]
        if bases and update_methods:
            data_struct_name = bases[0]
            if data_struct_name not in self._structs:
                raise TranspileError(node.lineno,
                    f"Класс-агент '{node.name}' наследует от '{data_struct_name}', "
                    f"но struct '{data_struct_name}' ещё не определён. "
                    "Убедитесь, что структура данных объявлена до класса-агента.")

            for method in update_methods:
                g = 0
                b = self._binding_counter
                self._binding_counter += 3  # read_buf, write_buf, uniform

                agent_base = _strip_data_suffix(data_struct_name)
                self._kernels.append(KernelDef(
                    fn_name=f"kernel_{_camel_to_snake(agent_base)}_{method.name}",
                    data_struct=data_struct_name,
                    agent_name=agent_base,
                    body_stmts=method.body,
                    binding_group=g,
                    binding_offset=b,
                ))
            return

        # ── Struct: только аннотированные поля, без базового класса ──────────
        if not bases and ann_fields and not methods:
            bad = [s for s in int_assigns]
            if bad:
                raise TranspileError(bad[0].lineno,
                    f"Класс-структура '{node.name}' не должен содержать "
                    "присваивания без аннотаций. "
                    "Для перечислений используйте отдельный класс без аннотаций.")
            fields = []
            for stmt in ann_fields:
                if not isinstance(stmt.target, ast.Name):
                    raise TranspileError(stmt.lineno,
                        "Поля структуры должны быть простыми именами.")
                fname = stmt.target.id
                ftype = _map_type(stmt.annotation, stmt.lineno)
                fields.append((fname, ftype))
            self._structs[node.name] = StructDef(node.name, fields)
            return

        # ── Enum: только целочисленные присваивания, без базового класса ─────
        if not bases and int_assigns and not ann_fields and not methods:
            members: list[tuple[str, int]] = []
            for stmt in int_assigns:
                if not (len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.Constant) and
                        isinstance(stmt.value.value, int)):
                    raise TranspileError(stmt.lineno,
                        f"В enum-классе '{node.name}': допустимы только "
                        "простые целочисленные присваивания (например, Alive = 0).")
                members.append((stmt.targets[0].id, stmt.value.value))
            self._enums[node.name] = EnumDef(node.name, members)
            return

        # ── Пустой класс ─────────────────────────────────────────────────────
        if not body:
            raise TranspileError(node.lineno,
                f"Класс '{node.name}' пуст. Добавьте поля или методы.")

        # ── Не удалось определить тип ─────────────────────────────────────────
        raise TranspileError(node.lineno,
            f"Класс '{node.name}' не соответствует ни одному поддерживаемому шаблону.\n"
            "  Поддерживаются:\n"
            "    1) Struct  — только аннотированные поля, без базового класса\n"
            "    2) Enum    — только целочисленные присваивания, без базового класса\n"
            "    3) Agent   — базовый struct + метод с @update_rule")

    # ── Второй проход: генерация WGSL ────────────────────────────────────────

    def _generate(self) -> str:
        enum_names = set(self._enums.keys())
        out: list[str] = []

        # 1. Константы (enum-классы)
        if self._enums:
            out.append("// ── Константы " + "─" * 55)
            for edef in self._enums.values():
                for member, value in edef.members:
                    const_name = f"{edef.name.upper()}_{member.upper()}"
                    out.append(f"const {const_name}: i32 = {value};")
            out.append("")

        # 2. Структуры
        if self._structs:
            out.append("// ── Структуры " + "─" * 55)
            for sdef in self._structs.values():
                out.append(f"struct {sdef.name} {{")
                for fname, ftype in sdef.fields:
                    out.append(f"    {fname}: {ftype},")
                out.append("}")
                out.append("")

        # 3. Storage buffers для агентов
        if self._kernels:
            out.append("// ── Storage Buffers " + "─" * 49)
            for k in self._kernels:
                plural = _plural_snake(k.data_struct)
                g, b = k.binding_group, k.binding_offset
                out += [
                    f"@group({g}) @binding({b})",
                    f"var<storage, read> {plural}_read: array<{k.data_struct}>;",
                    "",
                    f"@group({g}) @binding({b + 1})",
                    f"var<storage, read_write> {plural}_write: array<{k.data_struct}>;",
                    "",
                    f"@group({g}) @binding({b + 2})",
                    f"var<uniform> total_agents: u32;",
                    "",
                ]

        # 4. Свободные функции (топологически отсортированные)
        if self._functions:
            sorted_names = _topo_sort(self._functions)
            out.append("// ── Функции " + "─" * 57)
            for fname in sorted_names:
                out.extend(self._generate_function(self._functions[fname], enum_names))
                out.append("")

        # 5. Compute kernels
        if self._kernels:
            out.append("// ── Compute Kernels " + "─" * 49)
            for kernel in self._kernels:
                out.extend(self._generate_kernel(kernel, enum_names))
                out.append("")

        return "\n".join(out)

    # ── Генерация свободной функции ───────────────────────────────────────────

    def _generate_function(
        self,
        fdef: ast.FunctionDef,
        enum_names: set[str],
    ) -> list[str]:
        lines: list[str] = []

        # Параметры
        params: list[str] = []
        for arg in fdef.args.args:
            if arg.annotation is None:
                raise TranspileError(arg.lineno,
                    f"Параметр '{arg.arg}' функции '{fdef.name}' "
                    "должен иметь аннотацию типа.")
            wtype = _map_type(arg.annotation, arg.lineno)
            params.append(f"{arg.arg}: {wtype}")

        # Возвращаемый тип
        ret_type = self._return_type(fdef)

        params_str = ", ".join(params)
        if ret_type == "void":
            lines.append(f"fn {fdef.name}({params_str}) {{")
        else:
            lines.append(f"fn {fdef.name}({params_str}) -> {ret_type} {{")

        body_stmts = _filter_docstring(fdef.body)
        self._emit_body(lines, body_stmts, enum_names, self_alias=None, depth=1)

        lines.append("}")
        return lines

    # ── Генерация compute kernel из @update_rule ─────────────────────────────

    def _generate_kernel(
        self,
        kernel: KernelDef,
        enum_names: set[str],
    ) -> list[str]:
        lines: list[str] = []
        plural   = _plural_snake(kernel.data_struct)
        cell_var = kernel.agent_name[0].lower()   # Cell → c, Particle → p

        lines.append(f"@compute @workgroup_size({self.WORKGROUP_SIZE}, 1, 1)")
        lines.append(
            f"fn {kernel.fn_name}("
            f"@builtin(global_invocation_id) id: vec3<u32>) {{"
        )

        # Преамбула
        lines += [
            f"    var i: u32 = id.x;",
            f"    if (i >= total_agents) {{",
            f"        return;",
            f"    }}",
            f"    var {cell_var}: {kernel.data_struct} = {plural}_read[i];",
            "",
        ]

        body_stmts = _filter_docstring(kernel.body_stmts)
        self._emit_body(
            lines, body_stmts, enum_names,
            self_alias=cell_var, depth=1,
        )

        # Запись обратно в буфер
        lines += [
            "",
            f"    {plural}_write[i] = {cell_var};",
            "}",
        ]
        return lines

    # ── Общая логика: hoisting + тело ─────────────────────────────────────────

    def _emit_body(
        self,
        out: list[str],
        stmts: list[ast.stmt],
        enum_names: set[str],
        self_alias: Optional[str],
        depth: int,
    ) -> None:
        pad = " " * (4 * depth)

        # Hoisting: собрать все объявления и поднять наверх
        hoisted = _collect_vars(stmts)
        if hoisted:
            out.append(f"{pad}// объявления переменных")
            for vname, vtype in hoisted.items():
                out.append(f"{pad}var {vname}: {vtype};")
            out.append("")

        # Тело
        st = StmtTranspiler(enum_names, self_alias=self_alias)
        out.extend(st.tx_body(stmts, depth=depth))

    @staticmethod
    def _return_type(fdef: ast.FunctionDef) -> str:
        if fdef.returns is None:
            return "void"
        ret = fdef.returns
        if isinstance(ret, ast.Constant) and ret.value is None:
            return "void"
        if isinstance(ret, ast.Name) and ret.id == "None":
            return "void"
        return _map_type(ret, fdef.lineno)


# ─────────────────────────────────────────────────────────────────────────────
# Утилита
# ─────────────────────────────────────────────────────────────────────────────

def _filter_docstring(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Убирает docstring (первый Constant-Expr в теле)."""
    if (stmts and
            isinstance(stmts[0], ast.Expr) and
            isinstance(stmts[0].value, ast.Constant)):
        return stmts[1:]
    return stmts


# ─────────────────────────────────────────────────────────────────────────────
# Публичный API
# ─────────────────────────────────────────────────────────────────────────────

def transpile(source: str) -> str:
    """
    Транслирует Python-код в WGSL.
    При ошибке бросает TranspileError с номером строки и описанием.
    """
    return Py2WGSLTranspiler().transpile(source)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python transpiler.py <входной_файл.py>", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sys.argv[1], encoding="utf-8") as f:
            src = f.read()
        print(transpile(src))
    except TranspileError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Файл не найден: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
