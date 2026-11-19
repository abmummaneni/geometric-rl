"""
Spectral solver for controlled dye mixing on the flat torus.

This module implements a finite-dimensional approximation ``S_h`` of the solution
operator ``S_tau`` of the advection-diffusion equation

    d_t rho + u . grad(rho) = kappa * Laplace(rho),        (x, y) in T^2 = [0, 1)^2,

with periodic boundary conditions. On each control interval of length ``tau`` the
velocity ``u`` is one element of a finite, user-defined set of shear flows

    u(x, y) = (a g(y), 0)    or    u(x, y) = (0, a g(x)),

where ``g`` is a 1-periodic profile. Shear flows are automatically divergence-free.

Method
------
The state is the array of concentration values at the ``n x n`` cell centres,
``rho[i, j] = rho(x_i, y_j)`` with ``x_i = (i + 1/2) / n``, so ``S_h`` is a linear map
on ``R^(n x n)``. It is applied with FFTs rather than formed explicitly (see
``SpectralSolver.matrix`` for the assembled matrix). Each control interval is split
into ``substeps`` Strang steps

    D(dt/2) o T(dt) o D(dt/2),        dt = tau / substeps,

and both factors are diagonal in a suitable Fourier basis:

* ``D(dt)`` (diffusion) multiplies the 2-D Fourier coefficients by
  ``exp(-4 pi^2 kappa |m|^2 dt)``. It is exact on every resolved mode.
* ``T(dt)`` (shear) translates each row ``y = y_j`` of an x-shear rigidly by
  ``s_j = a dt g(y_j)``; after a 1-D FFT along ``x`` this is multiplication by
  ``exp(-2 pi i k s_j)``. It is exact for band-limited rows. Columns of a y-shear are
  treated in the same way.

The remaining errors are the Strang splitting error, which is O(dt^2), and
frequencies ``|k| >= n/2`` that the grid cannot represent (aliasing). The latter is
monitored by ``spectral_tail``.

Quick start
-----------
>>> import numpy as np
>>> from torus_spectral import SpectralSolver, default_controls, shear_x, mixing_cost
>>> controls = default_controls(amplitude=1.0)          # 0, +b1, -b1, +b2, -b2
>>> controls.append(shear_x(1.0, lambda s: np.cos(2 * np.pi * s), "+cos-x"))
>>> solver = SpectralSolver(64, controls, tau=0.25, kappa=1e-3, substeps=5)
>>> rho0 = solver.sample(lambda x, y: 1 + 0.5 * np.cos(2 * np.pi * x))
>>> trajectory = solver.rollout(rho0, [1, 3, 5] * 6 + [0, 0])   # shape (21, 64, 64)
>>> phi = mixing_cost(trajectory)                                # shape (21,)

Running the module as a script prints a validation report and exits with status 1 if
any check fails::

    python torus_spectral.py            # default settings
    python torus_spectral.py --help     # list of options
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Shear",
    "Control",
    "shear_x",
    "shear_y",
    "sine_profile",
    "default_controls",
    "cell_centres",
    "grid",
    "wavenumbers",
    "SpectralSolver",
    "mixing_cost",
    "total_mass",
    "variance",
    "spectral_tail",
    "transport_solution",
    "heat_mode_solution",
    "cosine_initial_condition",
    "DEFAULT_ACTIONS",
    "ValidationConfig",
    "run_validation",
]

Array = NDArray[np.float64]
Profile = Callable[[Array], Array]
InitialCondition = Callable[[Array, Array], Array]

_ARRAY_AXES = (-2, -1)  # every field operation acts on the last two axes, so stacks are supported


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Shear:
    """A shear flow on the torus.

    Parameters
    ----------
    direction
        ``"x"`` for ``u = (a g(y), 0)``: every row ``y = const`` translates along ``x``.
        ``"y"`` for ``u = (0, a g(x))``: every column ``x = const`` translates along ``y``.
    amplitude
        The constant ``a``.
    profile
        The 1-periodic function ``g``. It must act elementwise on NumPy arrays.
    name
        Label used in reports and plots.
    """

    direction: Literal["x", "y"]
    amplitude: float
    profile: Profile
    name: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("x", "y"):
            raise ValueError(f"direction must be 'x' or 'y', got {self.direction!r}")
        try:
            ends = np.broadcast_to(np.asarray(self.profile(np.array([0.0, 1.0])), dtype=float), (2,))
        except (TypeError, ValueError) as error:
            raise ValueError("profile must act elementwise on NumPy arrays") from error
        if not np.isclose(ends[0], ends[1], rtol=0.0, atol=1e-9):
            raise ValueError(
                f"profile must be 1-periodic to respect the periodic boundary, "
                f"but g(0) = {ends[0]:.6g} and g(1) = {ends[1]:.6g}"
            )
        if not self.name:
            object.__setattr__(self, "name", f"{self.direction}-shear(a={self.amplitude:g})")

    def displacement(self, coords: Array, duration: float) -> Array:
        """Rigid displacement ``a * duration * g(c)`` of the line at coordinate ``c``."""
        coords = np.asarray(coords, dtype=float)
        values = np.broadcast_to(np.asarray(self.profile(coords), dtype=float), coords.shape)
        return self.amplitude * duration * values

    def __str__(self) -> str:
        return self.name


Control = Optional[Shear]
"""A control is a shear flow, or ``None`` for "no stirring" (pure diffusion)."""


def shear_x(amplitude: float, profile: Profile, name: str = "") -> Shear:
    """Shear flow ``u = (amplitude * profile(y), 0)``."""
    return Shear("x", amplitude, profile, name)


def shear_y(amplitude: float, profile: Profile, name: str = "") -> Shear:
    """Shear flow ``u = (0, amplitude * profile(x))``."""
    return Shear("y", amplitude, profile, name)


def sine_profile(s: Array) -> Array:
    """The profile ``g(s) = sin(2 pi s)``."""
    return np.sin(2 * np.pi * s)


def default_controls(amplitude: float = 1.0) -> list[Control]:
    """The five project controls: no stirring, ``+-a (sin 2 pi y, 0)``, ``+-a (0, sin 2 pi x)``."""
    return [
        None,
        shear_x(+amplitude, sine_profile, "+b1"),
        shear_x(-amplitude, sine_profile, "-b1"),
        shear_y(+amplitude, sine_profile, "+b2"),
        shear_y(-amplitude, sine_profile, "-b2"),
    ]


def control_label(control: Control) -> str:
    return "no stirring" if control is None else control.name


# ---------------------------------------------------------------------------
# Grid and Fourier utilities
# ---------------------------------------------------------------------------
def cell_centres(n: int) -> Array:
    """Cell-centre coordinates ``(i + 1/2) / n`` for ``i = 0, ..., n - 1``."""
    return (np.arange(n) + 0.5) / n


def grid(n: int) -> tuple[Array, Array]:
    """Coordinate arrays ``X, Y`` with ``X[i, j] = x_i`` and ``Y[i, j] = y_j``."""
    centres = cell_centres(n)
    return np.meshgrid(centres, centres, indexing="ij")


def wavenumbers(n: int) -> Array:
    """Integer wavenumbers in FFT order: ``0, 1, ..., n/2 - 1, -n/2, ..., -1``."""
    return np.fft.fftfreq(n, d=1.0 / n)


@lru_cache(maxsize=None)
def _squared_wavenumbers(n: int) -> Array:
    k = wavenumbers(n)
    m2 = k[:, None] ** 2 + k[None, :] ** 2
    m2.setflags(write=False)
    return m2


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class SpectralSolver:
    """Finite-dimensional approximation ``S_h`` of ``S_tau`` on an ``n x n`` periodic grid.

    Parameters
    ----------
    n
        Grid size; must be even.
    controls
        The finite control set. The action ``a`` selects ``controls[a]``.
    tau
        Length of one control interval.
    kappa
        Diffusivity (``kappa = 0`` gives pure transport).
    substeps
        Number of Strang steps per control interval.

    Notes
    -----
    All multipliers are precomputed, so a step costs ``substeps`` 1-D FFT pairs and
    ``substeps + 1`` 2-D FFT pairs. States may be single ``n x n`` arrays or stacks of
    shape ``(..., n, n)``.
    """

    def __init__(
        self,
        n: int,
        controls: Sequence[Control],
        *,
        tau: float = 0.25,
        kappa: float = 1e-3,
        substeps: int = 5,
    ) -> None:
        if n < 4 or n % 2:
            raise ValueError(f"n must be an even integer >= 4, got {n}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        if len(controls) == 0:
            raise ValueError("the control set must not be empty")

        self.n = n
        self.controls: tuple[Control, ...] = tuple(controls)
        self.tau = float(tau)
        self.kappa = float(kappa)
        self.substeps = int(substeps)
        self.dt = self.tau / self.substeps

        laplacian_symbol = -4 * np.pi**2 * _squared_wavenumbers(n)
        self._diffuse_half = np.exp(self.kappa * laplacian_symbol * self.dt / 2)
        self._diffuse_full = np.exp(self.kappa * laplacian_symbol * self.dt)
        self._diffuse_interval = np.exp(self.kappa * laplacian_symbol * self.tau)
        self._phases = tuple(self._phase_factor(control) for control in self.controls)

    def __repr__(self) -> str:
        return (
            f"SpectralSolver(n={self.n}, actions={self.num_actions}, tau={self.tau:g}, "
            f"kappa={self.kappa:g}, substeps={self.substeps})"
        )

    @property
    def num_actions(self) -> int:
        return len(self.controls)

    # -- public API ----------------------------------------------------------
    def sample(self, initial_condition: InitialCondition) -> Array:
        """Evaluate ``initial_condition(x, y)`` at the cell centres."""
        x, y = grid(self.n)
        return np.broadcast_to(np.asarray(initial_condition(x, y), dtype=float), x.shape).copy()

    def step(self, rho: Array, action: int) -> Array:
        """Apply ``S_h``: advance ``rho`` by one control interval under ``controls[action]``."""
        rho = self._validated_state(rho)
        self._validate_action(action)
        control, phase = self.controls[action], self._phases[action]

        if control is None:
            # Without stirring the whole interval is pure diffusion, which is exact in one multiply.
            return self._diffuse(rho, self._diffuse_interval)

        axis = -2 if control.direction == "x" else -1
        # Strang splitting (D_half T D_half)^substeps, with adjacent half-steps merged.
        rho = self._diffuse(rho, self._diffuse_half)
        for i in range(self.substeps):
            rho = self._translate(rho, phase, axis)
            last = i == self.substeps - 1
            rho = self._diffuse(rho, self._diffuse_half if last else self._diffuse_full)
        return rho

    def rollout(self, rho0: Array, actions: Sequence[int]) -> Array:
        """Apply ``S_h`` for each action in turn; returns ``[rho_0, ..., rho_K]`` stacked."""
        states = [self._validated_state(rho0)]
        for action in actions:
            states.append(self.step(states[-1], action))
        return np.stack(states)

    def matrix(self, action: int, *, chunk: int = 1024) -> Array:
        """Assemble ``S_h`` for one action as an explicit ``n^2 x n^2`` matrix.

        Column ``c`` is ``S_h`` applied to the ``c``-th grid basis vector, in the order of
        ``rho.ravel()`` (``c = i * n + j``). Intended for small grids: ``n = 32`` gives a
        1024 x 1024 matrix, ``n = 64`` a 4096 x 4096 matrix (about 130 MB).
        """
        self._validate_action(action)
        size = self.n * self.n
        result = np.empty((size, size))
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            basis = np.zeros((stop - start, self.n, self.n))
            basis.reshape(stop - start, size)[np.arange(stop - start), np.arange(start, stop)] = 1.0
            result[:, start:stop] = self.step(basis, action).reshape(stop - start, size).T
        return result

    # -- internals -----------------------------------------------------------
    def _phase_factor(self, control: Control) -> Optional[NDArray[np.complex128]]:
        """``exp(-2 pi i k s)`` for the translation over one substep, laid out like the state."""
        if control is None:
            return None
        k = wavenumbers(self.n)
        s = control.displacement(cell_centres(self.n), self.dt)
        if control.direction == "x":  # FFT along x (axis -2); the shift depends on y (axis -1)
            return np.exp(-2j * np.pi * k[:, None] * s[None, :])
        return np.exp(-2j * np.pi * s[:, None] * k[None, :])  # FFT along y; shift depends on x

    def _diffuse(self, rho: Array, multiplier: Array) -> Array:
        if self.kappa == 0:
            return rho
        return np.real(np.fft.ifft2(np.fft.fft2(rho, axes=_ARRAY_AXES) * multiplier, axes=_ARRAY_AXES))

    @staticmethod
    def _translate(rho: Array, phase: NDArray[np.complex128], axis: int) -> Array:
        # Taking the real part leaves every resolved mode untouched. Only the Nyquist mode
        # k = -n/2, whose translation is ambiguous on a grid, is damped; it carries energy
        # only when the solution is under-resolved, which spectral_tail detects.
        return np.real(np.fft.ifft(np.fft.fft(rho, axis=axis) * phase, axis=axis))

    def _validated_state(self, rho: Array) -> Array:
        rho = np.asarray(rho, dtype=float)
        if rho.ndim < 2 or rho.shape[-2:] != (self.n, self.n):
            raise ValueError(f"state must have shape (..., {self.n}, {self.n}), got {rho.shape}")
        return rho

    def _validate_action(self, action: int) -> None:
        if not (isinstance(action, (int, np.integer)) and 0 <= action < self.num_actions):
            raise ValueError(f"action must be an integer in [0, {self.num_actions - 1}], got {action!r}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _inverse_laplacian_weights(n: int) -> Array:
    """``1 / (4 pi^2 |m|^2)`` for ``m != 0`` and ``0`` for ``m = 0``."""
    m2 = _squared_wavenumbers(n)
    weights = np.zeros_like(m2)
    weights[m2 > 0] = 1.0 / (4 * np.pi**2 * m2[m2 > 0])
    weights.setflags(write=False)
    return weights


def mixing_cost(rho: Array) -> Array:
    """Mixing functional ``Phi = 1/2 ||rho - mean||^2`` in the homogeneous ``H^{-1}`` norm.

    ``Phi = 1/2 sum_{m != 0} |f^(m)|^2 / (4 pi^2 |m|^2)``, evaluated over the last two axes.
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[-1]
    coefficients = np.fft.fft2(rho, axes=_ARRAY_AXES) / n**2
    return 0.5 * np.sum(np.abs(coefficients) ** 2 * _inverse_laplacian_weights(n), axis=_ARRAY_AXES)


def total_mass(rho: Array) -> Array:
    """Integral of ``rho`` over the unit torus (the mean of the grid values)."""
    return np.mean(rho, axis=_ARRAY_AXES)


def variance(rho: Array) -> Array:
    """``||rho - mean||^2`` in ``L^2`` of the unit torus."""
    return np.var(rho, axis=_ARRAY_AXES)


def spectral_tail(rho: Array, band: float = 0.25) -> Array:
    """Fraction of the fluctuation energy in the outermost frequency band.

    The band is ``max(|k|, |l|) > (1 - band) * n / 2``. A non-negligible value means the
    grid is too coarse for the solution.
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[-1]
    energy = np.abs(np.fft.fft2(rho, axes=_ARRAY_AXES)) ** 2
    energy[..., 0, 0] = 0.0
    k = np.abs(wavenumbers(n))
    outer = np.maximum.outer(k, k) > (1 - band) * n / 2
    total = energy.sum(axis=_ARRAY_AXES)
    return np.divide(energy[..., outer].sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)


# ---------------------------------------------------------------------------
# Exact solutions used for validation
# ---------------------------------------------------------------------------
def transport_solution(
    initial_condition: InitialCondition,
    controls: Sequence[Control],
    actions: Sequence[int],
    x: Array,
    y: Array,
    tau: float,
) -> Array:
    """Exact solution without diffusion: ``rho_K = rho_0 o X_0^{-1} o ... o X_{K-1}^{-1}``.

    Each point is mapped back through the inverse flow of the last action first, and the
    initial condition is evaluated at the resulting departure point.
    """
    px, py = np.array(x, dtype=float), np.array(y, dtype=float)
    for action in reversed(list(actions)):
        control = controls[action]
        if control is None:
            continue
        if control.direction == "x":
            px = px - control.displacement(py, tau)
        else:
            py = py - control.displacement(px, tau)
    return initial_condition(px % 1.0, py % 1.0)


def heat_mode_solution(x: Array, y: Array, t: float, kappa: float, k: int = 1, l: int = 0) -> Array:
    """Exact solution of the heat equation from ``1 + 1/2 sin(2 pi (k x + l y))``."""
    decay = np.exp(-4 * np.pi**2 * kappa * (k**2 + l**2) * t)
    return 1 + 0.5 * decay * np.sin(2 * np.pi * (k * x + l * y))


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
DEFAULT_ACTIONS: tuple[int, ...] = (1, 3, 1, 3, 2, 4, 2, 4, 1, 1, 3, 3, 0, 1, 3, 2, 4, 1, 3, 1)


def cosine_initial_condition(x: Array, y: Array) -> Array:
    """The default initial dye distribution ``1 + 1/2 cos(2 pi x)``."""
    return 1 + 0.5 * np.cos(2 * np.pi * x)


@dataclass(frozen=True)
class ValidationConfig:
    """Settings for ``run_validation``. The physical parameters are placeholders."""

    grids: tuple[int, ...] = (32, 64)
    substeps: tuple[int, ...] = (5, 10)
    n_steps: int = 20
    tau: float = 0.25
    kappa: float = 1e-3
    amplitude: float = 1.0
    reference_grid: int = 256
    reference_substeps: int = 40
    tolerance: float = 1e-2  # largest acceptable relative error of the Phi curve
    exact_tolerance: float = 1e-12  # for checks against closed-form solutions
    matrix_grid: Optional[int] = 32  # None skips the matrix check
    seed: int = 0


@dataclass
class _Report:
    passed: bool = True

    def check(self, condition: bool, message: str) -> None:
        self.passed &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")


def _smooth_test_field(x: Array, y: Array) -> Array:
    """A field that every default control moves: ``1 + cos(2 pi x)/2 + 0.3 sin(2 pi y)``."""
    return 1 + 0.5 * np.cos(2 * np.pi * x) + 0.3 * np.sin(2 * np.pi * y)


def _test_sequences(
    actions: Sequence[int], num_actions: int, n_steps: int, seed: int
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    return {
        "given": list(actions),
        f"random (seed {seed})": [int(a) for a in rng.integers(0, num_actions, n_steps)],
        "always action 1": [1 % num_actions] * n_steps,
        "cycle all actions": [i % num_actions for i in range(n_steps)],
    }


def run_validation(
    controls: Optional[Sequence[Control]] = None,
    actions: Optional[Sequence[int]] = None,
    initial_condition: InitialCondition = cosine_initial_condition,
    config: ValidationConfig = ValidationConfig(),
    figures_dir: Optional[Path] = None,
) -> bool:
    """Check ``S_h`` against exact solutions and a high-resolution reference.

    Returns ``True`` if every check passes. Pass ``figures_dir`` to also save plots.
    """
    controls = default_controls(config.amplitude) if controls is None else list(controls)
    if actions is None:
        actions = list(itertools.islice(itertools.cycle(DEFAULT_ACTIONS), config.n_steps))
    if len(actions) != config.n_steps:
        raise ValueError(f"expected {config.n_steps} actions, got {len(actions)}")

    report = _Report()
    print("Controls:")
    for index, control in enumerate(controls):
        print(f"  {index}: {control_label(control)}")
    print(f"Actions: {list(actions)}")
    print(f"tau = {config.tau:g}, kappa = {config.kappa:g}, steps = {config.n_steps}")

    # 1. Each operator on its own against a closed-form solution.
    print("\n1. Single operators against exact solutions (n = 32)")
    x, y = grid(32)
    report.check(
        abs(mixing_cost(cosine_initial_condition(x, y)) - 1 / (64 * np.pi**2)) < 1e-15,
        "Phi(1 + cos(2 pi x)/2) = 1/(64 pi^2)",
    )
    transport_only = SpectralSolver(32, controls, tau=config.tau, kappa=0.0, substeps=5)
    probe = transport_only.sample(_smooth_test_field)
    shear_error = max(
        np.max(np.abs(transport_only.step(probe, action)
                      - transport_solution(_smooth_test_field, controls, [action], x, y, config.tau)))
        for action in range(len(controls))
    )
    report.check(shear_error < config.exact_tolerance,
                 f"one shear step without diffusion matches the exact flow map (error {shear_error:.1e})")
    diffusion_only = SpectralSolver(32, [None], tau=config.tau, kappa=config.kappa, substeps=5)
    heat = diffusion_only.step(heat_mode_solution(x, y, t=0.0, kappa=config.kappa), action=0)
    heat_error = np.max(np.abs(heat - heat_mode_solution(x, y, t=config.tau, kappa=config.kappa)))
    report.check(heat_error < config.exact_tolerance,
                 f"one step of pure diffusion matches the heat equation (error {heat_error:.1e})")

    # 2. S_h as an explicit matrix.
    if config.matrix_grid:
        n = config.matrix_grid
        solver = SpectralSolver(n, controls, tau=config.tau, kappa=config.kappa, substeps=config.substeps[0])
        print(f"\n2. S_h as an explicit {n * n} x {n * n} matrix (n = {n}, substeps = {solver.substeps})")
        print("     action            |Mv - S_h v|   |row sum - 1|   |col sum - 1|   ||M||_2")
        probe = solver.sample(_smooth_test_field)
        matrices_ok = True
        for action, control in enumerate(controls):
            matrix = solver.matrix(action)
            apply_error = np.max(np.abs(matrix @ probe.ravel() - solver.step(probe, action).ravel()))
            row_error = np.max(np.abs(matrix.sum(axis=1) - 1))
            col_error = np.max(np.abs(matrix.sum(axis=0) - 1))
            norm = np.linalg.norm(matrix, 2)
            print(f"     {control_label(control):<16s}  {apply_error:.1e}        {row_error:.1e}         "
                  f"{col_error:.1e}         {norm:.6f}")
            matrices_ok &= bool(max(apply_error, row_error, col_error) < 1e-12 and norm <= 1 + 1e-12)
        report.check(matrices_ok,
                     "each matrix agrees with S_h, preserves constants and mass, and has norm <= 1")

    # 3. Whole episodes against a high-resolution reference.
    sequences = _test_sequences(actions, len(controls), config.n_steps, config.seed)
    reference_solver = SpectralSolver(config.reference_grid, controls, tau=config.tau,
                                      kappa=config.kappa, substeps=config.reference_substeps)
    references = {
        name: reference_solver.rollout(reference_solver.sample(initial_condition), seq)
        for name, seq in sequences.items()
    }
    reference_phi = {name: mixing_cost(traj) for name, traj in references.items()}
    print(f"\n3. {config.n_steps}-step episodes against a reference "
          f"(n = {config.reference_grid}, substeps = {config.reference_substeps})")
    for name, phi in reference_phi.items():
        print(f"     {name:<20s} final Phi/Phi_0 = {phi[-1] / phi[0]:.4f}, "
              f"spectral tail {spectral_tail(references[name][-1]):.0e}")
    header = "".join(f"{name:>20s}" for name in sequences)
    print(f"\n       n  substeps {header}   mass drift  tail      ms/step")

    trajectories: dict[tuple[int, int, str], Array] = {}
    worst, conservation_ok = 0.0, True
    for n in config.grids:
        for substeps in config.substeps:
            solver = SpectralSolver(n, controls, tau=config.tau, kappa=config.kappa, substeps=substeps)
            errors, drift, tail, monotone = [], 0.0, 0.0, True
            start = time.perf_counter()
            for name, seq in sequences.items():
                traj = solver.rollout(solver.sample(initial_condition), seq)
                trajectories[(n, substeps, name)] = traj
                phi = mixing_cost(traj)
                errors.append(float(np.max(np.abs(phi - reference_phi[name]) / reference_phi[name])))
                mass = total_mass(traj)
                drift = max(drift, float(np.max(np.abs(mass - mass[0]))))
                tail = max(tail, float(spectral_tail(traj[-1])))
                if config.kappa > 0:
                    monotone &= bool(np.all(np.diff(variance(traj)) <= 1e-15))
            ms_per_step = 1e3 * (time.perf_counter() - start) / (len(sequences) * config.n_steps)
            worst = max(worst, max(errors))
            row = "".join(f"{error:20.1e}" for error in errors)
            print(f"     {n:3d}  {substeps:8d} {row}   {drift:.0e}       {tail:.0e}   {ms_per_step:6.1f}")
            conservation_ok &= drift < 1e-12 and monotone
    report.check(
        worst < config.tolerance,
        f"largest relative error of the Phi curves is {worst:.1e} (tolerance {config.tolerance:.0e})",
    )
    variance_note = " and the variance decreases at every step" if config.kappa > 0 else ""
    report.check(conservation_ok, "mass is conserved" + variance_note)

    if figures_dir is not None:
        paths = _plot(config, sequences, reference_phi, references, trajectories, Path(figures_dir))
        print("\nFigures: " + ", ".join(str(p) for p in paths))

    print(f"\n{'All checks passed.' if report.passed else 'Some checks FAILED.'}")
    return report.passed


def _plot(
    config: ValidationConfig,
    sequences: dict[str, list[int]],
    reference_phi: dict[str, Array],
    references: dict[str, Array],
    trajectories: dict[tuple[int, int, str], Array],
    directory: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    steps = np.arange(config.n_steps + 1)
    colours = ("#D85A30", "#534AB7", "#1D9E75", "#BA7517")

    fig, axes = plt.subplots(1, len(sequences), figsize=(4.2 * len(sequences), 3.6), squeeze=False)
    for ax, name in zip(axes[0], sequences):
        ref = reference_phi[name]
        ax.semilogy(steps, ref / ref[0], color="k", lw=3, alpha=0.3, label="reference")
        for colour, n in zip(colours, config.grids):
            for style, substeps in zip(("-", ":", "--", "-."), config.substeps):
                phi = mixing_cost(trajectories[(n, substeps, name)])
                ax.semilogy(steps, phi / phi[0], color=colour, ls=style, lw=1.4,
                            label=f"n={n}, substeps={substeps}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("control step k")
    axes[0, 0].set_ylabel(r"$\Phi(\rho_k)\,/\,\Phi(\rho_0)$")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    curves = directory / "phi_curves.png"
    fig.savefig(curves, dpi=110)
    plt.close(fig)

    name, finest = "given", config.substeps[-1]
    panels = [("initial", trajectories[(config.grids[-1], finest, name)][0])]
    panels += [(f"k = {config.n_steps}, n = {n}", trajectories[(n, finest, name)][-1]) for n in config.grids]
    panels += [(f"k = {config.n_steps}, reference", references[name][-1])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), squeeze=False)
    for ax, (title, field_values) in zip(axes[0], panels):
        image = ax.imshow(field_values.T, origin="lower", extent=(0, 1, 0, 1),
                          cmap="viridis", vmin=0.5, vmax=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.015, pad=0.02)
    fields_path = directory / "final_fields.png"
    fig.savefig(fields_path, dpi=110)
    plt.close(fig)
    return [curves, fields_path]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    defaults = ValidationConfig()
    parser = argparse.ArgumentParser(
        description="Validate the spectral approximation S_h of the torus mixing dynamics.")
    parser.add_argument("--grids", type=int, nargs="+", default=list(defaults.grids), help="grid sizes n")
    parser.add_argument("--substeps", type=int, nargs="+", default=list(defaults.substeps),
                        help="Strang substeps per control interval")
    parser.add_argument("--steps", type=int, default=defaults.n_steps, help="control steps per episode")
    parser.add_argument("--tau", type=float, default=defaults.tau, help="control interval length")
    parser.add_argument("--kappa", type=float, default=defaults.kappa, help="diffusivity")
    parser.add_argument("--amplitude", type=float, default=defaults.amplitude, help="shear amplitude")
    parser.add_argument("--tolerance", type=float, default=defaults.tolerance,
                        help="largest acceptable relative error of the Phi curve")
    parser.add_argument("--no-matrix", action="store_true", help="skip the explicit matrix check")
    parser.add_argument("--figures", type=Path, default=Path("figures"), help="directory for plots")
    parser.add_argument("--no-figures", action="store_true", help="do not save plots")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = ValidationConfig(
        grids=tuple(args.grids),
        substeps=tuple(args.substeps),
        n_steps=args.steps,
        tau=args.tau,
        kappa=args.kappa,
        amplitude=args.amplitude,
        tolerance=args.tolerance,
        matrix_grid=None if args.no_matrix else ValidationConfig().matrix_grid,
    )
    passed = run_validation(config=config, figures_dir=None if args.no_figures else args.figures)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
