import numpy as np

from semi_lagrangian import (
    apply_cgrid_boundaries, cgrid_divergence, eta_grad_to_u_v, interp2,
    step_semilag_cgrid_trapezoid,
)
from parameters import L, H, g, f0, beta, gamma, rho


def test_bilinear_interpolation_reproduces_linear_field_and_clips():
    x = np.arange(5, dtype=float)
    y = np.arange(4, dtype=float)
    xx, yy = np.meshgrid(x, y)
    field = 2 * xx - 3 * yy + 7
    xp = np.array([0.25, 2.3, -1.0, 5.0])
    yp = np.array([0.75, 1.2, 2.0, 4.0])
    expected = 2 * np.clip(xp, x[0], x[-1]) - 3 * np.clip(yp, y[0], y[-1]) + 7
    np.testing.assert_allclose(interp2(field, x, y, xp, yp), expected, atol=1e-14)


def test_pressure_gradient_of_linear_elevation():
    dx, dy = 2.0, 3.0
    x, y = np.meshgrid((np.arange(8) + 0.5) * dx,
                       (np.arange(6) + 0.5) * dy)
    gx, gy = eta_grad_to_u_v(2 * x - 3 * y, dx, dy)
    np.testing.assert_allclose(gx[:, 1:-1], 2.0)
    np.testing.assert_allclose(gy[1:-1, :], -3.0)
    np.testing.assert_array_equal(gx[:, [0, -1]], 0.0)
    np.testing.assert_array_equal(gy[[0, -1], :], 0.0)


def test_closed_boundary_divergence_has_zero_domain_integral():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(6, 9))
    v = rng.normal(size=(7, 8))
    u, v = apply_cgrid_boundaries(u, v)
    divergence = cgrid_divergence(u, v, 2.0, 3.0)
    tolerance = 64 * np.finfo(float).eps * np.abs(divergence).sum()
    assert abs(divergence.sum()) <= tolerance


def test_semi_lagrangian_unforced_uniform_state_is_stationary():
    eta = np.full((6, 8), 0.125)
    u = np.zeros((6, 9))
    v = np.zeros((7, 8))
    params = dict(L=L, H=H, g=g, f0=f0, beta=beta, gamma=gamma,
                  rho=rho, tau0=0.0, dt=10.0)
    unew, vnew, enew = step_semilag_cgrid_trapezoid(u, v, eta, params)
    np.testing.assert_array_equal(unew, u)
    np.testing.assert_array_equal(vnew, v)
    np.testing.assert_allclose(enew, eta, atol=1e-12, rtol=0)
