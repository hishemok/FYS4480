import numpy as np
import sympy as sp
import re
from typing import NamedTuple

Z = sp.symbols("Z")

def convert_sqrt(expression: str) -> str:
    """Convert from Mathematica's Sqrt[...] to Sympy's sqrt(...)."""
    return re.sub(r"Sqrt\[(.*?)\]", r"sqrt(\1)", expression)

def read_elements(Z_val: int) -> np.ndarray:
    """Read matrix elements from the file and substitute Z with Z_val.

    Args:
        Z_val (int): The value to substitute for Z.

    Returns:
        np.ndarray: The matrix elements.
    """
    path = "Midterm1/Src/Matrix_elements.txt"
    elements = np.zeros((3, 3, 3, 3), dtype=object)

    with open(path, "r") as infile:
        for line in infile:
            left, right = line.split(" = ")
            (a, b), (c, d) = left[1:-1].split("|V|")
            indices = tuple(int(x) - 1 for x in (a, b, c, d))

            val = sp.parse_expr(convert_sqrt(right.strip()))
            value = val.subs(Z, Z_val)

            elements[indices] = value
    return elements

def to_eV(value: float) -> float:
    return value * 27.211386

class Electron(NamedTuple):
    n: int  # Principal quantum number
    spin: int  # Spin state

    # Calculate the one-body energy contribution
    def one_body(self,Z_val) -> float:
        return -(Z_val**2) / (2 * (self.n+1)**2)
    

class ElectronEnergy:
    def __init__(self, F: int, Z_val: int) -> None:
        self.F = F
        self.Z = Z_val
        self.groundstate = np.zeros((2, 3))
        self.groundstate[:, :F] = 1
        self.values = read_elements(self.Z)
        self.ref_energy = self.reference_energy()

    def reference_energy(self) -> sp.Expr:
        holes = [Electron(*hole) for hole in np.argwhere(self.groundstate[:, :self.F] == 1)]
        onebody = sum(self.h0(hole, hole) for hole in holes)
        two_body = sum(self.antisymmetrized(h1, h2, h1, h2) / 2 for h1 in holes for h2 in holes)
        return onebody + two_body

    def get_hole_particle(self, state: np.ndarray) -> tuple[Electron, Electron]:
        hole = Electron(*np.argwhere(state[:, :self.F] == 0)[0])
        particle = Electron(*np.argwhere(state[:, self.F:] == 1)[0] + [0, self.F])
        return hole, particle

    def energy_from_state(self, bra: np.ndarray, ket: np.ndarray) -> sp.Expr:
        if np.array_equal(bra, ket) and np.array_equal(bra, self.groundstate):
            return self.ref_energy
        if np.array_equal(bra, self.groundstate) or np.array_equal(ket, self.groundstate):
            i, a = self.get_hole_particle(bra if np.array_equal(ket, self.groundstate) else ket)
            return self.f(i, a)
        i, a = self.get_hole_particle(bra)
        j, b = self.get_hole_particle(ket)
        return (self.ref_energy * self.delta(i, j) * self.delta(a, b)
                + self.f(a, b) * self.delta(i, j) - self.f(i, j) * self.delta(a, b)
                + self.antisymmetrized(a, j, i, b))

    def delta(self, alpha: Electron, beta: Electron) -> int:
        return 1 if alpha == beta else 0

    def h0(self, p: Electron, q: Electron) -> sp.Expr:
        return p.one_body(self.Z) if self.delta(p, q) else 0

    def v(self, p: Electron, q: Electron, r: Electron, s: Electron) -> sp.Expr:
        return self.values[p.n, q.n, r.n, s.n] if self.spin_ok(p, q, r, s) and p != q and r != s else 0

    def antisymmetrized(self, p: Electron, q: Electron, r: Electron, s: Electron) -> sp.Expr:
        return self.v(p, q, r, s) - self.v(p, q, s, r)

    def f(self, p: Electron, q: Electron) -> sp.Expr:
        energy = self.h0(p, q)
        for spin in range(2):
            for k in range(self.F):
                k_s = Electron(spin, k)
                energy += self.antisymmetrized(p, k_s, q, k_s)
        return energy

    def spin_ok(self, a: Electron, b: Electron, c: Electron, d: Electron) -> bool:
        return a.spin == c.spin and b.spin == d.spin

    def annihilate_and_create(self, sigma: int, i: int, a: int) -> np.ndarray:
        new_state = np.copy(self.groundstate)
        new_state[sigma, i] -= 1
        new_state[sigma, a] += 1
        return new_state

    def get_total_states(self) -> list[np.ndarray]:
        return [self.groundstate] + [self.annihilate_and_create(s, i, a) 
                                     for i in range(self.F) for a in range(self.F, 3) for s in range(2)]

    def get_hamiltonian(self) -> sp.Matrix:
        total_states = self.get_total_states()
        Hamiltonian = sp.zeros(len(total_states))
        for i, bra in enumerate(total_states):
            for j, ket in enumerate(total_states):
                Hamiltonian[i, j] = self.energy_from_state(bra, ket)
        return Hamiltonian

    def get_energy(self, z_val: int) -> float:
        eigenvals = self.get_hamiltonian().subs(Z, z_val).eigenvals()
        return min(sp.re(eigenval.evalf()) for eigenval in eigenvals)

def calc(F: int, z_val: int) -> None:
    setup = ElectronEnergy(F, z_val)
    print(f"Ground energy: {setup.ref_energy}")
    print(f"Eigenvalue energy: {setup.get_energy(z_val)}")
    print(f"Ground energy (eV): {to_eV(setup.ref_energy)}")
    print(f"Eigenvalue energy (eV): {to_eV(setup.get_energy(z_val))}")

if __name__ == "__main__":
    print("Helium:")
    calc(F=1, z_val=2)  # Helium
    print("\nBeryllium:")
    calc(F=2, z_val=4)  # Beryllium
