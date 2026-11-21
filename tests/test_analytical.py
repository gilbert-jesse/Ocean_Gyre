import numpy as np

from analytical import run_task_c, stommel_exact_solution
from diagnostics import gaussian_eta_bump
from parameters import L


def test_analytical_fields_are_finite_on_rectangular_grid():
    result = run_task_c(Nx=8, Ny=6)
    # Task C deliberately uses meshgrid(indexing="ij"), unlike the solvers.
    for key in ("u_st", "v_st", "eta_st"):
        assert result[key].shape == (8, 6)
        assert np.isfinite(result[key]).all()


def test_analytical_no_normal_flow_at_basin_walls():
    along = np.linspace(0.0, L, 23)
    for wall in (0.0, L):
        u, _, _ = stommel_exact_solution(np.full_like(along, wall), along)
        _, v, _ = stommel_exact_solution(along, np.full_like(along, wall))
        np.testing.assert_allclose(u, 0.0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(v, 0.0, atol=1e-14, rtol=0)


def test_reference_elevation_changes_only_eta_gauge():
    x, y = np.meshgrid(np.linspace(0.0, L, 9), np.linspace(0.0, L, 7))
    u0, v0, eta0 = stommel_exact_solution(x, y, eta0=0.0)
    u1, v1, eta1 = stommel_exact_solution(x, y, eta0=0.125)
    np.testing.assert_array_equal(u0, u1)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_allclose(eta1 - eta0, 0.125, atol=1e-15, rtol=0)


def test_analytical_velocity_is_divergence_free_interior():
    x, y = np.meshgrid(np.linspace(0.1 * L, 0.9 * L, 9),
                       np.linspace(0.1 * L, 0.9 * L, 7))
    h = 1e-5 * L
    du_dx = (stommel_exact_solution(x + h, y)[0]
             - stommel_exact_solution(x - h, y)[0]) / (2 * h)
    dv_dy = (stommel_exact_solution(x, y + h)[1]
             - stommel_exact_solution(x, y - h)[1]) / (2 * h)
    scale = max(np.linalg.norm(du_dx), np.linalg.norm(dv_dy))
    # Finite-difference truncation and floating-point error, not an exact identity.
    assert np.linalg.norm(du_dx + dv_dy) < 1e-6 * scale


def test_gaussian_initial_perturbation_has_zero_mean():
    eta = gaussian_eta_bump(8, 6, amp=0.01)
    assert eta.shape == (6, 8)
    assert np.isfinite(eta).all()
    assert abs(eta.mean()) < 1e-15
