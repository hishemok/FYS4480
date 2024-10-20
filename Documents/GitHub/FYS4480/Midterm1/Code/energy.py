import numpy as np
import sympy as sp

Z = sp.symbols("Z")

def to_eV(energy: float) -> float:
    return energy * 27.2114

def convert_sqrt(expression: str) -> str:
    """Convert from Mathematica's Sqrt[...] to Sympy's sqrt(...)."""
    return expression.replace("Sqrt[", "sqrt(").replace("]", ")")

def read_matrix_elements(n_max: int,Z_val: int) -> np.ndarray:
    """Matrix elements from file. 
    On form: <11|V|11> = (5*Z)/8
    <11|V|12> = (4096*Sqrt[2]*Z)/64827
    <11|V|13> = (1269*Sqrt[3]*Z)/50000"""

    elements = np.zeros((n_max, n_max, n_max, n_max))

    with open("Midterm1/Code/matrix_elements.txt", "r") as f:
        for line in f:
            l,r = line.split(" = ")
            (a, b), (c, d) = l[1:-1].split("|V|")
            idx = tuple(int(x) - 1 for x in (a, b, c, d))
            val = convert_sqrt(r.strip())
            val = sp.parse_expr(val)

            value = val.subs("Z", Z_val)

            elements[idx] = value

    return elements

class Electron:
    def __init__(self, spin, level):
        self.spin = spin
        self.level = level

def kroenecker_delta(a: int, b: int) -> int:
    """Kroenecker delta."""
    return 1 if a == b else 0

def groundstate(F: int) -> np.ndarray:
    """Ground state for F particles."""
    state = np.zeros((2, 3))
    state[:, :F] = 1
    return state

def get_hole_particle(state: np.ndarray, F: int) -> tuple[Electron, Electron]:
    hole = np.argwhere(state[:, :F] == 0)[0]
    particle = np.argwhere(state[:, F:] == 1)[0]
    particle = particle + np.array([0, F])

    return Electron(*hole), Electron(*particle)

def antisymmetrized(i: Electron, j: Electron, a: Electron, b: Electron, Z_val: int) -> sp.Expr:
    """Antisymmetrized matrix element."""
    return (
        read_matrix_elements(3, Z_val)[i.level, j.level, a.level, b.level]
        - read_matrix_elements(3, Z_val)[i.level, j.level, b.level, a.level]
    ) * kroenecker_delta(i.spin, a.spin) * kroenecker_delta(j.spin, b.spin)

def h0(i: Electron, a: Electron, Z_val: int) -> sp.Expr:
    """One-body operator."""
    n = i.level + 1
    m = a.level + 1
    return - Z_val**2 /(2* n * m) * kroenecker_delta(n, m)

def f(a: Electron, b: Electron, Z_val: int) -> sp.Expr:
    """Two-body operator."""
    return read_matrix_elements(3, Z_val)[0, 0, a.level, b.level]

def delta(i: Electron, j: Electron) -> int:
    """Kroenecker delta."""
    return kroenecker_delta(i.spin, j.spin) * kroenecker_delta(i.level, j.level)

def reference_energy(F: int, Z_val: int) -> sp.Expr:
    onebody = 0
    two_body = 0
    ground = groundstate(F)
    hole_arrs = np.argwhere(ground[:, :F] == 1)
    holes = [Electron(*hole) for hole in hole_arrs]

    for hole in holes:
        onebody += h0(hole, hole, Z_val)

    for hole in holes:
        for other_hole in holes:
            two_body += antisymmetrized(hole, other_hole, hole, other_hole, Z_val) / 2

    return onebody + two_body

def energy_from_state(bra_state: np.ndarray, ket_state: np.ndarray, F: int, Z_val: int) -> sp.Expr:

    if np.all(bra_state == groundstate(F)) and np.all(
        ket_state == groundstate(F)
    ):
        return reference_energy(F,Z_val)

    if np.all(bra_state == groundstate(F)) or np.all(
        ket_state == groundstate(F)
    ):
        acting = bra_state if np.all(ket_state == groundstate(F)) else ket_state
        i, a = get_hole_particle(acting, F)

        return f(i, a)

    i, a = get_hole_particle(bra_state, F)
    j, b = get_hole_particle(ket_state, F)

    energy = reference_energy(F) * delta(i, j) * delta(a, b)
    energy += f(a, b) * delta(i, j)
    energy -= f(i, j) * delta(a, b)
    energy += antisymmetrized(a, j, i, b)

    return energy

if __name__ == "__main__":
    F = 2
    Z = 1
    bra = groundstate(F)
    ket = groundstate(F)
    print(energy_from_state(bra, ket, F, Z))
