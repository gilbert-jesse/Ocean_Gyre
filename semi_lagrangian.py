import numpy as np

def interp2(field, xg, yg, xp, yp):
    """
    Bilinear interpolation for field defined on an grid xg, yg.
    field shape = (len(yg), len(xg)). In cases where some coordinates fall out of the
    the specified range, I use clipping. This is very crude but it prevents blowups.
    Ghost cells can also be used.
    """

    xp = np.clip(xp, xg[0], xg[-1])
    yp = np.clip(yp, yg[0], yg[-1])

    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    xi = (xp - xg[0]) / dx
    yi = (yp - yg[0]) / dy

    i0 = np.floor(xi).astype(int)
    j0 = np.floor(yi).astype(int)

    i0 = np.clip(i0, 0, len(xg) - 2)
    j0 = np.clip(j0, 0, len(yg) - 2)

    i1 = i0 + 1
    j1 = j0 + 1

    wx = xi - i0
    wy = yi - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (
        (1 - wx) * (1 - wy) * f00
        + wx * (1 - wy) * f10
        + (1 - wx) * wy * f01
        + wx * wy * f11
    )


def eta_grad_to_u_v(eta, dx, dy):
    """
    Pressure gradients on C-grid velocity points.

    eta: (Ny, Nx)
    deta_dx_u: (Ny, Nx+1)
    deta_dy_v: (Ny+1, Nx)
    """

    Ny, Nx = eta.shape

    deta_dx_u = np.zeros((Ny, Nx + 1))
    deta_dy_v = np.zeros((Ny + 1, Nx))

    deta_dx_u[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / dx
    deta_dy_v[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / dy

    return deta_dx_u, deta_dy_v


def cgrid_divergence(u, v, dx, dy):
    """
    Divergence at eta points.
    """

    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


def apply_cgrid_boundaries(u, v):
    """
    Closed basin no-normal-flow boundary condition.
    """

    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v


def cgrid_coordinates(eta, u, v, L):
    """
    Construct C-grid coordinates.
    """

    Ny, Nx = eta.shape
    dx = L / Nx
    dy = L / Ny

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy

    x_u = np.arange(Nx + 1) * dx
    y_u = y_eta.copy()

    x_v = x_eta.copy()
    y_v = np.arange(Ny + 1) * dy

    Xeta, Yeta = np.meshgrid(x_eta, y_eta)
    Xu, Yu = np.meshgrid(x_u, y_u)
    Xv, Yv = np.meshgrid(x_v, y_v)

    return dx, dy, x_eta, y_eta, x_u, y_u, x_v, y_v, Xeta, Yeta, Xu, Yu, Xv, Yv


def sources_cgrid(u, v, eta, p):
    """
    Compute source terms S_u, S_v, S_h on the C-grid.

    S_u at u points.
    S_v at v points.
    S_h at eta points.
    """

    L = p["L"]
    H = p["H"]
    g = p["g"]
    f0 = p["f0"]
    beta = p["beta"]
    gamma = p["gamma"]
    rho = p["rho"]
    tau0 = p["tau0"]

    (
        dx, dy,
        x_eta, y_eta,
        x_u, y_u,
        x_v, y_v,
        Xeta, Yeta,
        Xu, Yu,
        Xv, Yv,
    ) = cgrid_coordinates(eta, u, v, L)

    h = H + eta
    h = np.maximum(h, 1.0)

    # interpolate opposite velocity components
    v_on_u = interp2(v, x_v, y_v, Xu, Yu)
    u_on_v = interp2(u, x_u, y_u, Xv, Yv)

    # interpolate h to velocity grids
    h_on_u = interp2(h, x_eta, y_eta, Xu, Yu)
    h_on_v = interp2(h, x_eta, y_eta, Xv, Yv)

    deta_dx_u, deta_dy_v = eta_grad_to_u_v(eta, dx, dy)
    div_eta = cgrid_divergence(u, v, dx, dy)

    f_u = f0 + beta * Yu
    f_v = f0 + beta * Yv

    tau_x_u = -tau0 * np.cos(np.pi * Yu / L)
    tau_y_v = np.zeros_like(v)

    S_u = (
        f_u * v_on_u
        - g * deta_dx_u
        - gamma * u
        + tau_x_u / (rho * h_on_u)
    )

    S_v = (
        -f_v * u_on_v
        - g * deta_dy_v
        - gamma * v
        + tau_y_v / (rho * h_on_v)
    )

    S_h = -h * div_eta

    return S_u, S_v, S_h


def step_semilag_cgrid_trapezoid(u, v, eta, p):

    L = p["L"]
    H = p["H"]
    dt = p["dt"]

    (
        dx, dy,
        x_eta, y_eta,
        x_u, y_u,
        x_v, y_v,
        Xeta, Yeta,
        Xu, Yu,
        Xv, Yv,
    ) = cgrid_coordinates(eta, u, v, L)

    h = H + eta

    # velocity at eta points
    u_eta = 0.5 * (u[:, 1:] + u[:, :-1])
    v_eta = 0.5 * (v[1:, :] + v[:-1, :])

    # velocity at opposite C-grid points
    v_on_u = interp2(v, x_v, y_v, Xu, Yu)
    u_on_v = interp2(u, x_u, y_u, Xv, Yv)

    # departure points
    xdep_eta = Xeta - u_eta * dt
    ydep_eta = Yeta - v_eta * dt

    xdep_u = Xu - u * dt
    ydep_u = Yu - v_on_u * dt

    xdep_v = Xv - u_on_v * dt
    ydep_v = Yv - v * dt

    # fields at departure points
    eta_d = interp2(eta, x_eta, y_eta, xdep_eta, ydep_eta)
    h_d = H + eta_d

    u_d = interp2(u, x_u, y_u, xdep_u, ydep_u)
    v_d = interp2(v, x_v, y_v, xdep_v, ydep_v)

    # old source terms on arrival grid
    S_u_old, S_v_old, S_h_old = sources_cgrid(u, v, eta, p)

    # old source terms interpolated to departure points
    S_u_d = interp2(S_u_old, x_u, y_u, xdep_u, ydep_u)
    S_v_d = interp2(S_v_old, x_v, y_v, xdep_v, ydep_v)
    S_h_d = interp2(S_h_old, x_eta, y_eta, xdep_eta, ydep_eta)

    # predictor
    u_star = u_d + dt * S_u_d
    v_star = v_d + dt * S_v_d
    h_star = h_d + dt * S_h_d
    eta_star = h_star - H

    u_star, v_star = apply_cgrid_boundaries(u_star, v_star)

    # source terms at future arrival points, estimated from predictor
    S_u_star, S_v_star, S_h_star = sources_cgrid(u_star, v_star, eta_star, p)

    # corrector: the crude trapezoidal source average
    u_new = u_d + 0.5 * dt * (S_u_d + S_u_star)
    v_new = v_d + 0.5 * dt * (S_v_d + S_v_star)
    h_new = h_d + 0.5 * dt * (S_h_d + S_h_star)

    h_new = np.maximum(h_new, 1.0)
    eta_new = h_new - H

    u_new, v_new = apply_cgrid_boundaries(u_new, v_new)

    return u_new, v_new, eta_new