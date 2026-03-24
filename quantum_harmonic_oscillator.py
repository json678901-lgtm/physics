"""
1D Quantum Harmonic Oscillator — Full Simulation
=================================================
Dimensionless units:  ħ = m = ω = 1
Analytical eigenvalues:  E_n = n + 1/2

Approach
--------
* Finite-difference discretisation of kinetic energy on a uniform grid
  with Dirichlet boundary conditions (ψ vanishes at ±L).
* The resulting tridiagonal Hamiltonian is solved with
  scipy.linalg.eigh_tridiagonal for speed and numerical stability.
* Time evolution uses the Crank-Nicolson (CN) scheme, which is
  unconditionally stable and unitary (norm-preserving).

Author: Antigravity (AI pair-programmer)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ──────────────────────────────────────────────
#  1. Spatial grid
# ──────────────────────────────────────────────

def make_grid(L: float = 8.0, N: int = 512):
    """
    Uniform grid on [-L, L] with N interior points.
    Dirichlet BCs:  ψ(-L) = ψ(L) = 0.

    Parameters
    ----------
    L : half-width of the domain (should be large enough that the
        wave-function of interest has decayed to ~0 at the boundary).
    N : number of interior grid points.

    Returns
    -------
    x  : 1-D array of grid positions  (length N)
    dx : grid spacing
    """
    dx = 2 * L / (N + 1)
    x = np.linspace(-L + dx, L - dx, N)
    return x, dx


# ──────────────────────────────────────────────
#  2. Hamiltonian (tridiagonal)
# ──────────────────────────────────────────────

def build_hamiltonian(x, dx):
    """
    Build the tridiagonal Hamiltonian for the 1-D QHO.

    H = T + V
    with  T_{ii}   =  1/dx²           (kinetic diagonal)
          T_{i,i±1}= -1/(2 dx²)       (kinetic off-diagonal)
          V_{ii}   =  ½ x_i²          (harmonic potential)

    In dimensionless units (ħ = m = 1) the kinetic-energy operator is
        T = -½ d²/dx²
    discretised via the standard 3-point central-difference stencil.

    Returns
    -------
    diag     : main diagonal  (length N)
    off_diag : sub/super diagonal  (length N-1)
    """
    kinetic_diag = 1.0 / (dx ** 2)          # coefficient of ψ_i
    kinetic_off  = -0.5 / (dx ** 2)         # coefficient of ψ_{i±1}
    potential    = 0.5 * x ** 2             # V(x) = ½ω²x² with ω = 1

    diag     = kinetic_diag + potential      # T_ii + V_ii
    off_diag = np.full(len(x) - 1, kinetic_off)
    return diag, off_diag


# ──────────────────────────────────────────────
#  3. Eigenvalue solver + normalisation
# ──────────────────────────────────────────────

def solve_eigenstates(x, dx, n_states: int = 10):
    """
    Solve for the lowest `n_states` eigenvalues / eigenfunctions.

    Uses scipy.linalg.eigh_tridiagonal with select='i' to return only
    the desired range, which is O(N·n_states) — much faster than a full
    dense solve.

    All eigenvectors are normalised so that  ∫|ψ|² dx = 1  using the
    trapezoidal rule (≈ dx · Σ |ψ|²  for interior points with Dirichlet BCs).
    """
    diag, off_diag = build_hamiltonian(x, dx)

    energies, psi = eigh_tridiagonal(
        diag, off_diag,
        select='i', select_range=(0, n_states - 1)
    )

    # Normalise each eigenfunction
    for k in range(n_states):
        norm = np.sqrt(np.trapezoid(psi[:, k] ** 2, dx=dx))
        psi[:, k] /= norm

    return energies, psi


# ──────────────────────────────────────────────
#  4. Analytical comparison
# ──────────────────────────────────────────────

def compare_energies(energies):
    """Print numerical vs analytical eigenvalues and their absolute errors."""
    n_states = len(energies)
    analytical = np.arange(n_states) + 0.5

    print("\n" + "=" * 55)
    print(f"  {'n':>3}  {'E_num':>12}  {'E_exact':>12}  {'|Error|':>12}")
    print("=" * 55)
    for n in range(n_states):
        err = abs(energies[n] - analytical[n])
        print(f"  {n:3d}  {energies[n]:12.8f}  {analytical[n]:12.8f}  {err:12.2e}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
#  5. Static plots
# ──────────────────────────────────────────────

def plot_eigenfunctions(x, energies, psi, n_show: int = 5):
    """
    Plot the first `n_show` eigenfunctions offset by their energy.
    This makes it easy to see the node count and parity.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    scale = 0.6  # visual scaling for ψ amplitude

    # potential curve
    V = 0.5 * x ** 2
    ax.plot(x, V, 'k--', lw=1.0, label='V(x) = ½x²')

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, n_show))
    for n in range(n_show):
        offset = energies[n]
        ax.axhline(offset, color=colours[n], lw=0.6, ls=':')
        ax.plot(x, offset + scale * psi[:, n], color=colours[n],
                lw=1.4, label=f'ψ_{n}  (E={energies[n]:.4f})')

    ax.set_xlabel('x')
    ax.set_ylabel('Energy / ψ_n (offset)')
    ax.set_title('Eigenfunctions of the 1-D Quantum Harmonic Oscillator')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, energies[n_show - 1] + 1.5)
    fig.tight_layout()
    return fig


def plot_probability_densities(x, energies, psi, n_show: int = 5):
    """Plot |ψ_n(x)|² for the first `n_show` states."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, n_show))

    for n in range(n_show):
        ax.plot(x, psi[:, n] ** 2, color=colours[n], lw=1.5,
                label=f'|ψ_{n}|²')

    ax.set_xlabel('x')
    ax.set_ylabel('|ψ_n(x)|²')
    ax.set_title('Probability Densities')
    ax.legend(fontsize=9)
    ax.set_xlim(-6, 6)
    fig.tight_layout()
    return fig


def plot_energy_levels(x, energies, n_show: int = 5):
    """Draw the potential well with horizontal energy-level lines."""
    fig, ax = plt.subplots(figsize=(7, 6))
    V = 0.5 * x ** 2
    ax.plot(x, V, 'k-', lw=1.5, label='V(x) = ½x²')

    colours = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_show))
    for n in range(n_show):
        ax.axhline(energies[n], color=colours[n], lw=2,
                   label=f'E_{n} = {energies[n]:.4f}')

    ax.set_xlabel('x')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Levels in the Harmonic Potential')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, energies[n_show - 1] + 1.5)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  6. Time evolution — Crank-Nicolson
# ──────────────────────────────────────────────

def _cn_banded_matrices(diag, off_diag, dt):
    """
    Build the banded matrices for the Crank-Nicolson step:

        (I + i·dt/2 · H) ψ^{n+1}  =  (I - i·dt/2 · H) ψ^n

    Both sides are tridiagonal. We store them in scipy's banded format
    (3 × N) for use with solve_banded.

    Returns
    -------
    A_band : left-hand side  (implicit, to invert)
    B_band : right-hand side (explicit multiply)
    """
    N = len(diag)
    alpha = 0.5j * dt

    # --- left-hand side  A = I + iα H ---
    A_main  =  1.0 + alpha * diag
    A_upper =        alpha * off_diag
    A_lower =        alpha * off_diag

    A_band = np.zeros((3, N), dtype=complex)
    A_band[0, 1:] = A_upper          # super-diagonal
    A_band[1, :]  = A_main           # main diagonal
    A_band[2, :-1] = A_lower         # sub-diagonal

    # --- right-hand side  B = I - iα H ---
    B_main  =  1.0 - alpha * diag
    B_upper =       -alpha * off_diag
    B_lower =       -alpha * off_diag

    B_band = np.zeros((3, N), dtype=complex)
    B_band[0, 1:] = B_upper
    B_band[1, :]  = B_main
    B_band[2, :-1] = B_lower

    return A_band, B_band


def _apply_banded_rhs(B_band, psi):
    """Multiply the RHS banded matrix B by the vector ψ (tridiagonal matvec)."""
    N = len(psi)
    result = B_band[1, :] * psi                       # main diagonal
    result[1:]  += B_band[0, 1:] * psi[:-1]           # super-diag contribution
    result[:-1] += B_band[2, :-1] * psi[1:]           # sub-diag contribution
    return result


def gaussian_wavepacket(x, x0: float = -2.0, sigma: float = 0.5, k0: float = 2.0):
    """
    Normalised Gaussian wave packet:
        ψ(x) ∝ exp(-(x-x0)²/(2σ²)) · exp(i k0 x)

    Parameters
    ----------
    x0    : centre position
    sigma : width
    k0    : initial momentum (in units of ħ=1)
    """
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
    dx = x[1] - x[0]
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, dx=dx))
    return psi / norm


def evolve_crank_nicolson(x, dx, psi0, dt: float = 0.005, n_steps: int = 2000,
                          save_every: int = 10):
    """
    Time-evolve ψ using the Crank-Nicolson scheme.

    Parameters
    ----------
    psi0       : initial wave-function (complex array, length N)
    dt         : time step
    n_steps    : total number of time steps
    save_every : save a snapshot every this many steps

    Returns
    -------
    snapshots : list of (t, |ψ|²) tuples
    """
    diag, off_diag = build_hamiltonian(x, dx)
    A_band, B_band = _cn_banded_matrices(diag, off_diag, dt)

    psi = psi0.copy()
    snapshots = [(0.0, np.abs(psi) ** 2)]

    for step in range(1, n_steps + 1):
        rhs = _apply_banded_rhs(B_band, psi)
        psi = solve_banded((1, 1), A_band, rhs)

        if step % save_every == 0:
            snapshots.append((step * dt, np.abs(psi) ** 2))

    return snapshots


# ──────────────────────────────────────────────
#  7. Animation
# ──────────────────────────────────────────────

def animate_evolution(x, snapshots, interval: int = 30):
    """
    Animate |ψ(x,t)|² using matplotlib.animation.FuncAnimation.

    Parameters
    ----------
    snapshots : list of (t, prob_density) as returned by evolve_crank_nicolson
    interval  : delay between frames in ms
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    V = 0.5 * x ** 2

    ax.plot(x, V * 0.3, 'k--', lw=0.8, alpha=0.4, label='V(x) (scaled)')
    line, = ax.plot(x, snapshots[0][1], lw=1.6, color='#0077cc')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top')
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, np.max([s[1].max() for s in snapshots]) * 1.15)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ(x, t)|²')
    ax.set_title('Time Evolution of a Gaussian Wave Packet in a Harmonic Potential')
    ax.legend(loc='upper right', fontsize=9)

    def _update(frame):
        t, prob = snapshots[frame]
        line.set_ydata(prob)
        time_text.set_text(f't = {t:.3f}')
        return line, time_text

    anim = animation.FuncAnimation(
        fig, _update, frames=len(snapshots),
        interval=interval, blit=True
    )
    return fig, anim


# ──────────────────────────────────────────────
#  8. Main entry point
# ──────────────────────────────────────────────

def main():
    # --- Grid & eigenstates ---
    N_GRID   = 512       # interior grid points
    L        = 8.0       # domain half-width
    N_STATES = 10        # number of eigenstates to compute
    N_SHOW   = 5         # number to display in plots

    x, dx = make_grid(L=L, N=N_GRID)
    energies, psi = solve_eigenstates(x, dx, n_states=N_STATES)

    # --- Analytical comparison ---
    compare_energies(energies)

    # --- Static plots ---
    fig1 = plot_eigenfunctions(x, energies, psi, n_show=N_SHOW)
    fig2 = plot_probability_densities(x, energies, psi, n_show=N_SHOW)
    fig3 = plot_energy_levels(x, energies, n_show=N_SHOW)

    # --- Time evolution ---
    print("Running Crank-Nicolson time evolution …")
    psi0 = gaussian_wavepacket(x, x0=-2.0, sigma=0.5, k0=2.0)
    snapshots = evolve_crank_nicolson(x, dx, psi0,
                                      dt=0.005, n_steps=2000, save_every=10)
    print(f"  {len(snapshots)} frames captured.\n")

    fig4, anim = animate_evolution(x, snapshots)

    plt.show()


if __name__ == "__main__":
    main()
