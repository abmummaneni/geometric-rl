"""Tests for torus_spectral. Run with ``pytest test_torus_spectral.py``."""

import numpy as np
import pytest

from torus_spectral import (
    Shear,
    SpectralSolver,
    cosine_initial_condition,
    default_controls,
    grid,
    heat_mode_solution,
    mixing_cost,
    shear_x,
    shear_y,
    spectral_tail,
    total_mass,
    transport_solution,
    variance,
)

TAU = 0.25
KAPPA = 1e-3


def smooth_field(x, y):
    return 1 + 0.5 * np.cos(2 * np.pi * x) + 0.3 * np.sin(2 * np.pi * y)


@pytest.fixture(scope="module")
def controls():
    """The five default controls plus two user-defined shears."""
    return default_controls(1.0) + [
        shear_x(1.0, lambda s: np.cos(2 * np.pi * s), "+cos-x"),
        shear_y(-0.5, lambda s: np.sin(4 * np.pi * s), "-0.5 sin4pi-y"),
    ]


# -- input validation ----------------------------------------------------------
def test_shear_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        Shear("z", 1.0, np.sin)


def test_shear_rejects_non_periodic_profile():
    with pytest.raises(ValueError, match="1-periodic"):
        shear_x(1.0, lambda s: s)


def test_shear_accepts_constant_profile():
    uniform = shear_x(1.0, lambda s: 1.0)
    assert np.allclose(uniform.displacement(np.array([0.1, 0.7]), 0.5), 0.5)


@pytest.mark.parametrize("kwargs", [dict(n=31), dict(n=2), dict(tau=0.0), dict(kappa=-1.0), dict(substeps=0)])
def test_solver_rejects_invalid_parameters(kwargs):
    arguments = dict(n=32, tau=TAU, kappa=KAPPA, substeps=5) | kwargs
    n = arguments.pop("n")
    with pytest.raises(ValueError):
        SpectralSolver(n, default_controls(), **arguments)


def test_step_rejects_bad_action_and_shape():
    solver = SpectralSolver(16, default_controls())
    rho = solver.sample(smooth_field)
    with pytest.raises(ValueError, match="action"):
        solver.step(rho, 5)
    with pytest.raises(ValueError, match="shape"):
        solver.step(np.zeros((8, 8)), 0)


# -- the mixing functional -------------------------------------------------------
def test_mixing_cost_of_cosine():
    x, _ = grid(32)
    assert mixing_cost(cosine_initial_condition(x, x)) == pytest.approx(1 / (64 * np.pi**2), rel=1e-12)


@pytest.mark.parametrize("m", [1, 2, 5, 15])
def test_mixing_cost_of_resolved_sine_modes(m):
    x, _ = grid(32)
    assert mixing_cost(np.sin(2 * np.pi * m * x)) == pytest.approx(1 / (16 * np.pi**2 * m**2), rel=1e-12)


def test_mixing_cost_ignores_the_mean():
    x, y = grid(16)
    field = smooth_field(x, y)
    assert mixing_cost(field + 3.0) == pytest.approx(mixing_cost(field), rel=1e-12)


# -- each operator against its exact solution -------------------------------------
def test_shear_step_is_exact_without_diffusion(controls):
    solver = SpectralSolver(32, controls, tau=TAU, kappa=0.0, substeps=5)
    x, y = grid(32)
    rho = solver.sample(smooth_field)
    for action in range(solver.num_actions):
        exact = transport_solution(smooth_field, controls, [action], x, y, TAU)
        assert np.max(np.abs(solver.step(rho, action) - exact)) < 1e-12


def test_diffusion_step_matches_heat_equation():
    solver = SpectralSolver(32, [None], tau=TAU, kappa=KAPPA, substeps=5)
    x, y = grid(32)
    for k, l in [(1, 0), (2, 3)]:
        stepped = solver.step(heat_mode_solution(x, y, 0.0, KAPPA, k, l), 0)
        assert np.max(np.abs(stepped - heat_mode_solution(x, y, TAU, KAPPA, k, l))) < 1e-14


def test_transport_is_reversible():
    solver = SpectralSolver(32, default_controls(), tau=TAU, kappa=0.0, substeps=5)
    rho = solver.sample(smooth_field)
    for forward, backward in [(1, 2), (3, 4)]:
        assert np.max(np.abs(solver.step(solver.step(rho, forward), backward) - rho)) < 1e-12


# -- structural properties -----------------------------------------------------------
def test_rollout_conserves_mass_and_dissipates_variance(controls):
    solver = SpectralSolver(32, controls, tau=TAU, kappa=KAPPA, substeps=5)
    actions = np.random.default_rng(0).integers(0, solver.num_actions, 20)
    trajectory = solver.rollout(solver.sample(smooth_field), actions)
    assert trajectory.shape == (21, 32, 32)
    assert np.max(np.abs(total_mass(trajectory) - total_mass(trajectory[0]))) < 1e-13
    assert np.all(np.diff(variance(trajectory)) <= 1e-15)


def test_transport_conserves_variance_when_resolved():
    solver = SpectralSolver(64, default_controls(), tau=TAU, kappa=0.0, substeps=5)
    trajectory = solver.rollout(solver.sample(cosine_initial_condition), [1] * 5)
    assert np.allclose(variance(trajectory), variance(trajectory[0]), rtol=1e-12)


def test_step_acts_independently_on_a_stack():
    solver = SpectralSolver(16, default_controls(), tau=TAU, kappa=KAPPA, substeps=5)
    a = solver.sample(smooth_field)
    b = solver.sample(cosine_initial_condition)
    stacked = solver.step(np.stack([a, b]), 3)
    assert np.allclose(stacked[0], solver.step(a, 3)) and np.allclose(stacked[1], solver.step(b, 3))


def test_assembled_matrix(controls):
    solver = SpectralSolver(16, controls, tau=TAU, kappa=KAPPA, substeps=5)
    probe = solver.sample(smooth_field)
    for action in range(solver.num_actions):
        matrix = solver.matrix(action, chunk=100)
        assert matrix.shape == (256, 256)
        assert np.allclose(matrix @ probe.ravel(), solver.step(probe, action).ravel(), atol=1e-13)
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-13)  # constants are preserved
        assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-13)  # mass is conserved
        assert np.linalg.norm(matrix, 2) <= 1 + 1e-12  # S_h never amplifies


# -- accuracy of whole episodes ------------------------------------------------------
@pytest.mark.parametrize("substeps, tolerance", [(5, 3e-3), (10, 1e-3)])
def test_episode_converges_to_reference(controls, substeps, tolerance):
    actions = list(np.random.default_rng(1).integers(0, len(controls), 20))
    reference = SpectralSolver(128, controls, tau=TAU, kappa=KAPPA, substeps=40)
    phi_reference = mixing_cost(reference.rollout(reference.sample(smooth_field), actions))
    solver = SpectralSolver(32, controls, tau=TAU, kappa=KAPPA, substeps=substeps)
    trajectory = solver.rollout(solver.sample(smooth_field), actions)
    assert np.max(np.abs(mixing_cost(trajectory) / phi_reference - 1)) < tolerance
    assert spectral_tail(trajectory[-1]) < 1e-2
