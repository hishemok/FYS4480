import numpy as np
import sympy as sp
from get_energy import to_eV, read_elements


class HartreeFock:
    def __init__(self, F: int, Z_val: int) -> None:
        self.F = F
        self.Z = Z_val
        self.num_orbitals = 6
        self.levels = np.arange(1, 4).repeat(2)
        self.spins = np.array([0, 1] * 3)
        self.electrons = np.vstack([self.levels, self.spins]).T
        self.values = self.prepare_values()
        self.coulomb_integrals = self.setup_coulomb()

    def prepare_values(self) -> np.ndarray:
        Z = sp.symbols("Z")
        values = read_elements(self.Z)
        return np.vectorize(lambda x: x.subs(Z, self.Z).evalf())(values).astype(np.float64)

    def setup_coulomb(self) -> np.ndarray:
        coulomb_integrals = np.zeros((self.num_orbitals,) * 4)
        for i, alpha in enumerate(self.electrons):
            for j, beta in enumerate(self.electrons):
                for k, gamma in enumerate(self.electrons):
                    for l, delta in enumerate(self.electrons):
                        coulomb_integrals[i, j, k, l] = self.antisymmetrized(alpha, beta, gamma, delta)
        return coulomb_integrals

    def antisymmetrized(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> float:
        return self.V(alpha, beta, gamma, delta) - self.V(alpha, beta, delta, gamma)

    def V(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> float:
        if (alpha[1] == gamma[1]) and (beta[1] == delta[1]) and (np.any(alpha != beta)) and (np.any(gamma != delta)):
            indices = np.array([alpha[0], beta[0], gamma[0], delta[0]]) - 1
            return self.values[tuple(indices)]
        return 0

    def groundstate_energy(self, coeffs: np.ndarray) -> float:
        C = coeffs[: self.F, :]

        h0 = np.einsum("ab, ia, ib ->", self.h, C, C)
        h1 = np.einsum("abcd, ia, jb, ic, jd ->", self.coulomb_integrals, C, C, C, C)
        return h0 + 0.5 * h1

    def setup_density_matrix(self, coefficients: np.ndarray) -> np.ndarray:
        return np.einsum("ia, ib -> ab", coefficients[:self.F, :], coefficients[:self.F, :])

    def iteration(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        density_matrix = self.setup_density_matrix(coefficients)
        fock_matrix = self.h + self.coulomb(density_matrix)
        energy, coefficients = self.diagonalize(fock_matrix)
        return energy, coefficients

    def diagonalize(self, fock_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A,B = np.linalg.eigh(fock_matrix)
        return A,B.T

    def coulomb(self, density_matrix: np.ndarray) -> np.ndarray:
        return np.einsum("abcd, bd -> ac", self.coulomb_integrals, density_matrix)

    def run(self, max_iter=100, tol=1e-14):
        coeffs = np.eye(self.num_orbitals)
        old_energies = np.zeros(self.num_orbitals)

        for iteration in range(max_iter):
            density_matrix = self.setup_density_matrix(coeffs)
            fock_matrix = self.h + self.coulomb(density_matrix)
            energies, coeffs = self.diagonalize(fock_matrix)

            if np.linalg.norm(energies - old_energies, ord=1) / self.num_orbitals < tol:
                print(f"Converged in {iteration} iterations")
                break

            old_energies = energies
        print(f"Converged in {iteration} iterations")
        return energies, coeffs

    def groundstate_loop(self, coeffs: np.ndarray) -> float:
        # Compute h0 using einsum correctly
        h0 = np.einsum("ab, ia, ib ->", self.h, coeffs[:self.F, :], coeffs[:self.F, :])


        # Calculate h1 using a more efficient method
        h1 = np.sum([
            self.coulomb_integrals[alpha, beta, gamma, delta] *
            coeffs[i, alpha] * coeffs[j, beta] * coeffs[i, gamma] * coeffs[j, delta]
            for i in range(self.F) for j in range(self.F)
            for alpha in range(self.num_orbitals)
            for beta in range(self.num_orbitals)
            for gamma in range(self.num_orbitals)
            for delta in range(self.num_orbitals)
        ])

        return h0 + 0.5 * h1



def first_iteration() -> None:
    print("Results after one iteration:")
    print("Helium:")
    hf = HartreeFock(2, 2)
    coefficients = np.eye(6)
    energies, coefficients = hf.iteration(coefficients)
    groundstate_energy = hf.groundstate_energy(coefficients)

    assert np.isclose(groundstate_energy, hf.groundstate_loop(coefficients))

    print(f"New single-particle energies: {energies}")
    print(f"New ground state energy: {groundstate_energy}")
    print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")

    print("Beryllium:")
    hf = HartreeFock(4, 4)
    coefficients = np.eye(6)
    energies, coefficients = hf.iteration(coefficients)
    groundstate_energy = hf.groundstate_energy(coefficients)

    assert np.isclose(groundstate_energy, hf.groundstate_loop(coefficients))

    print(f"New single-particle energies: {energies}")
    print(f"New ground state energy: {groundstate_energy}")
    print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")


def iter_until_convergence(tol: float = 1e-12) -> None:
    print(f"Results after full convergence (tolerance: {tol}):")
    print("Helium:")
    hf = HartreeFock(2, 2)
    energies, coefficients = hf.run(tol=tol)
    groundstate_energy = hf.groundstate_energy(coefficients)

    print(f"Final single-particle energies: {energies}")
    print(f"Final ground state energy: {groundstate_energy}")
    print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")

    print("Beryllium:")
    hf = HartreeFock(4, 4)
    print("berillium run")
    energies, coefficients = hf.run(tol=tol)
    groundstate_energy = hf.groundstate_energy(coefficients)

    print(f"Final single-particle energies: {energies}")
    print(f"Final ground state energy: {groundstate_energy}")
    print(f"In electron volts: {to_eV(groundstate_energy)} eV\n")


if __name__ == "__main__":
    print("first_iteration")
    first_iteration()
    print("iter_until_convergence")
    iter_until_convergence()
