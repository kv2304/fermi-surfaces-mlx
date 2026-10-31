"""
The Fermi surface of an anisotropic tight-binding band as n grows.

[E] = t_x, for all energies
[k] = 1/a
[v] = t_x a/ħ
[g] = 1/t_x, per unit cell per spin
[n] = electrons per unit cell
"""

import itertools
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# transfer integrals
TRANSFER_X = 1.0
TRANSFER_Y = 1.0
TRANSFER_Z = 0.4

# energy grid for g(E)
BAND_MINIMUM = -2 * (TRANSFER_X + TRANSFER_Y + TRANSFER_Z)
N_ENERGIES = 4000
ENERGY_STEP = -2 * BAND_MINIMUM / N_ENERGIES
ENERGIES = BAND_MINIMUM + ENERGY_STEP * (np.arange(N_ENERGIES) + 0.5)


def dispersion_1d(n_cells, transfer):
    """Return one cosine term of eqn (1) on an evenly spaced k grid."""
    k = 2 * np.pi * (np.arange(n_cells) + 0.5) / n_cells - np.pi
    return mx.array(-2 * transfer * np.cos(k), dtype=mx.float32)


@mx.compile
def count_states(E_x_planes, E_y, E_z):
    """Count the states at each energy on a stack of constant-kx planes."""
    E = E_x_planes[:, None, None] + E_y[None, :, None] + E_z[None, None, :]
    index = mx.clip(mx.floor((E - BAND_MINIMUM) / ENERGY_STEP), 0, N_ENERGIES - 1).astype(mx.int32)
    return mx.zeros(N_ENERGIES).at[index.reshape(-1)].add(1.0)


def density_of_states(n_cells, n_planes=16):
    """Return the per-spin density of states g(E) for n_cells³ unit cells."""
    E_x = dispersion_1d(n_cells, TRANSFER_X)
    E_y = dispersion_1d(n_cells, TRANSFER_Y)
    E_z = dispersion_1d(n_cells, TRANSFER_Z)
    states = mx.zeros(N_ENERGIES)
    for i in range(0, n_cells, n_planes):
        states = states + count_states(E_x[i:i + n_planes], E_y, E_z)
        mx.eval(states)
    return np.array(states) / (n_cells**3 * ENERGY_STEP)


def chemical_potential(g, n, kT=0.01):
    """Solve 2∫g f dE = n for every electron density at once by bisection."""
    g_dE, E, n = mx.array(g * ENERGY_STEP), mx.array(ENERGIES), mx.array(n)
    mu_low, mu_high = mx.full(n.shape, BAND_MINIMUM - 0.5), mx.full(n.shape, -BAND_MINIMUM + 0.5)
    for _ in range(50):
        mu = (mu_low + mu_high) / 2
        electrons = 2 * (g_dE * mx.sigmoid((mu[:, None] - E) / kT)).sum(1)
        too_few = electrons < n
        mu_low, mu_high = mx.where(too_few, mu, mu_low), mx.where(too_few, mu_high, mu)
    return np.array((mu_low + mu_high) / 2)


def g_effective_mass(E):
    """Return the per-spin density of states of eqn (3)."""
    transfer_product = TRANSFER_X * TRANSFER_Y * TRANSFER_Z
    return np.sqrt(np.clip(E - BAND_MINIMUM, 0, None)) / (4 * np.pi**2 * np.sqrt(transfer_product))


def check_density_of_states(mesh_sizes=(64, 128, 256, 512), window=0.1):
    """Compare low-energy g(E) with eqn (3) as the mesh grows."""
    near_minimum = ENERGIES < BAND_MINIMUM + window
    for n_cells in mesh_sizes:
        g = density_of_states(n_cells)
        ratio = g[near_minimum].sum() / g_effective_mass(ENERGIES[near_minimum]).sum()
        print(f"{n_cells}³: g vs (3) error {abs(ratio - 1):.1%}")


def time_devices(n_cells=512):
    """Time the density of states on the GPU and the CPU."""
    for device in (mx.gpu, mx.cpu):
        mx.set_default_device(device)
        density_of_states(64)
        start = time.perf_counter()
        density_of_states(n_cells)
        print(f"{device} {n_cells}³: {time.perf_counter() - start:.2f} s")
    mx.set_default_device(mx.gpu)


def run_checks(g, n, mu):
    """Print the verification table against the textbook formulas."""
    print(f"Σ g dE = {g.sum() * ENERGY_STEP:.4f}")
    print(f"max |g(E) - g(-E)| / max g = {np.abs(g - g[::-1]).max() / g.max():.2e}")
    print(f"max |μ(n) + μ(2-n)| = {np.abs(mu + np.interp(2 - n, n, mu))[1:-1].max():.2e}")
    n_low = np.array([0.002, 0.005, 0.01])
    transfer_product = TRANSFER_X * TRANSFER_Y * TRANSFER_Z
    for n_i, mu_i in zip(n_low, chemical_potential(g, n_low)):
        E_F = BAND_MINIMUM + (3 * np.pi**2 * n_i * np.sqrt(transfer_product)) ** (2 / 3)
        error = abs((mu_i - BAND_MINIMUM) / (E_F - BAND_MINIMUM) - 1)
        print(f"n = {n_i:.3f}: μ = {mu_i:.4f}, E_F from (3) = {E_F:.4f}, error {error:.1%}")
    n_below = 2 * np.cumsum(g) * ENERGY_STEP
    signs = itertools.product((1, -1), repeat=3)
    critical = {-2 * (s_x * TRANSFER_X + s_y * TRANSFER_Y + s_z * TRANSFER_Z) for s_x, s_y, s_z in signs}
    topology_changes = sorted(critical)[1:-1]
    for E_c in topology_changes:
        print(f"Fermi surface changes topology at μ/t_x = {E_c:+.2f}, n = {np.interp(E_c, ENERGIES + ENERGY_STEP / 2, n_below):.3f}")


def animate(n, mu, frames=100, grid=44, path="fermi_surface.gif"):
    """Animate the Fermi surface coloured by |v| as the band fills."""
    transfers = np.array([TRANSFER_X, TRANSFER_Y, TRANSFER_Z])
    k = np.linspace(-np.pi, np.pi, grid)
    k_grid = np.stack(np.meshgrid(k, k, k, indexing="ij"), axis=-1)
    E = -2 * (transfers * np.cos(k_grid)).sum(-1)
    n_frames = np.linspace(0.02, 1.98, frames)
    mu_frames = np.interp(n_frames, n, mu)
    v_max = 2 * np.linalg.norm(transfers)

    plt.style.use(["dark_background", {"grid.color": (1, 1, 1, 0.3), "axes3d.xaxis.panecolor": "black",
                                       "axes3d.yaxis.panecolor": "black", "axes3d.zaxis.panecolor": "black"}])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    fig.colorbar(ScalarMappable(Normalize(0, v_max), "plasma"), ax=ax, shrink=0.35, pad=0.08,
                 label=r"$|\mathbf{v}| \,/\, (t_x a/\hbar)$")
    zone = np.pi * np.array([-1, -0.5, 0, 0.5, 1])
    zone_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), zlim=(-np.pi, np.pi), box_aspect=(1, 1, 1),
           xticks=zone, yticks=zone, zticks=zone, xticklabels=zone_labels, yticklabels=zone_labels,
           zticklabels=zone_labels, xlabel="$k_x a$", ylabel="$k_y a$", zlabel="$k_z a$")
    ax.zaxis.set_rotate_label(False)

    def draw(i):
        """Draw one frame of the animation."""
        for old in list(ax.collections):
            old.remove()
        verts, faces, _, _ = marching_cubes(E, mu_frames[i], spacing=(k[1] - k[0],) * 3)
        verts -= np.pi
        v = 2 * np.linalg.norm(transfers * np.sin(verts), axis=1)
        colours = plt.cm.plasma(v[faces].mean(1) / v_max)
        ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=colours, edgecolors=colours, linewidth=0.3))
        ax.view_init(elev=20, azim=30 + 0.5 * i)
        ax.set_title(f"n = {n_frames[i]:.2f}, $\\mu/t_x$ = {mu_frames[i]:+.2f}")

    anim = FuncAnimation(fig, draw, frames=frames, interval=100)
    anim.save(path, writer=PillowWriter(fps=10), dpi=120)
    return anim


if __name__ == "__main__":
    check_density_of_states()
    time_devices()
    g = density_of_states(512)
    n = np.linspace(0, 2, 201)
    mu = chemical_potential(g, n)
    run_checks(g, n, mu)
    anim = animate(n, mu)
    plt.show()
