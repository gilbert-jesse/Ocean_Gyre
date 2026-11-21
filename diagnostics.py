import numpy as np

from parameters import L, H, g, gamma, rho, f0, beta, tau0
from analytical import stommel_exact_solution


# Numerical model

def run_gyre_model(Nx=51, Ny=51, days=200, cfl_factor=0.5, lower_dt=None):
    dx = L / Nx
    dy = L / Ny

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy

    x_u = np.arange(Nx + 1) * dx
    y_u = y_eta.copy()

    x_v = x_eta.copy()
    y_v = np.arange(Ny + 1) * dy

    X_eta, Y_eta = np.meshgrid(x_eta, y_eta, indexing="xy")
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing="xy")
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing="xy")

    eta = np.zeros((Ny, Nx))
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    tau_x_u = -tau0 * np.cos(np.pi * Y_u / L)
    tau_y_v = np.zeros_like(Y_v)

    f_u = f0 + beta * Y_u
    f_v = f0 + beta * Y_v

    c = np.sqrt(g * H)
    dt_cfl = 1.0 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))

    if lower_dt is None:
      dt = cfl_factor * dt_cfl
    else:
      dt = lower_dt
    nt = int(days * 24 * 3600 / (2 * dt))

    def apply_bc(u, v):
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        return u, v

    def divergence(u, v):
        return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy

    def grad_eta_u(eta):
        out = np.zeros((Ny, Nx + 1))
        out[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / dx
        return out

    def grad_eta_v(eta):
        out = np.zeros((Ny + 1, Nx))
        out[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / dy
        return out

    def interp_v_to_u(v):
        out = np.zeros((Ny, Nx + 1))
        out[:, 1:-1] = 0.25 * (
            v[:-1, :-1] + v[:-1, 1:] +
            v[1:, :-1] + v[1:, 1:]
        )
        return out

    def interp_u_to_v(u):
        out = np.zeros((Ny + 1, Nx))
        out[1:-1, :] = 0.25 * (
            u[:-1, :-1] + u[:-1, 1:] +
            u[1:, :-1] + u[1:, 1:]
        )
        return out

    def total_energy(u, v, eta):
        eta_anom = eta - np.mean(eta)
        E_u = 0.5 * rho * H * np.sum(u**2) * dx * dy
        E_v = 0.5 * rho * H * np.sum(v**2) * dx * dy
        E_eta = 0.5 * rho * g * np.sum(eta_anom**2) * dx * dy
        return E_u + E_v + E_eta

    energy = np.empty(nt)
    time_days = np.empty(nt)

    for n in range(nt):
        # u-first
        u, v = apply_bc(u, v)
        eta_new = eta - H * dt * divergence(u, v)

        u_new = (
            u
            + f_u * dt * interp_v_to_u(v)
            - g * dt * grad_eta_u(eta_new)
            - gamma * dt * u
            + (tau_x_u / (rho * H)) * dt
        )

        u_new, v = apply_bc(u_new, v)

        v_new = (
            v
            - f_v * dt * interp_u_to_v(u_new)
            - g * dt * grad_eta_v(eta_new)
            - gamma * dt * v
            + (tau_y_v / (rho * H)) * dt
        )

        eta, u, v = eta_new, u_new, v_new
        u, v = apply_bc(u, v)

        # v-first
        eta_new = eta - H * dt * divergence(u, v)

        v_new = (
            v
            - f_v * dt * interp_u_to_v(u)
            - g * dt * grad_eta_v(eta_new)
            - gamma * dt * v
            + (tau_y_v / (rho * H)) * dt
        )

        u, v_new = apply_bc(u, v_new)

        u_new = (
            u
            + f_u * dt * interp_v_to_u(v_new)
            - g * dt * grad_eta_u(eta_new)
            - gamma * dt * u
            + (tau_x_u / (rho * H)) * dt
        )

        eta, u, v = eta_new, u_new, v_new
        u, v = apply_bc(u, v)

        energy[n] = total_energy(u, v, eta)
        time_days[n] = (n + 1) * 2.0 * dt / 86400.0

    return {
        "Nx": Nx, "Ny": Ny,
        "dx": dx, "dy": dy,
        "dt": dt,
        "dt_cfl": dt_cfl,
        "x_eta": x_eta, "y_eta": y_eta,
        "x_u": x_u, "y_u": y_u,
        "x_v": x_v, "y_v": y_v,
        "X_eta": X_eta, "Y_eta": Y_eta,
        "X_u": X_u, "Y_u": Y_u,
        "X_v": X_v, "Y_v": Y_v,
        "eta": eta,
        "u": u,
        "v": v,
        "energy": energy,
        "time_days": time_days,
    }

def gaussian_eta_bump(Nx, Ny, amp=0.01, x0=0.5, y0=0.5, sigma_frac=0.12):
    dx = L / Nx
    dy = L / Ny

    x_eta = (np.arange(Nx) + 0.5) * dx
    y_eta = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x_eta, y_eta, indexing="xy")

    sigma = sigma_frac * L

    eta0 = amp * np.exp(
        -((X - x0 * L)**2 + (Y - y0 * L)**2) / (2.0 * sigma**2)
    )

    eta0 -= np.mean(eta0)
    return eta0


# Error energy against analytical solution

def compute_error_energy(result):
    dx = result["dx"]
    dy = result["dy"]

    eta = result["eta"]
    u = result["u"]
    v = result["v"]

    # use fixed reference level
    eta0 = 0.0

    u_st, _, _ = stommel_exact_solution(result["X_u"], result["Y_u"], eta0=eta0)
    _, v_st, _ = stommel_exact_solution(result["X_v"], result["Y_v"], eta0=eta0)
    _, _, eta_st = stommel_exact_solution(result["X_eta"], result["Y_eta"], eta0=eta0)

    u_prime = u - u_st
    v_prime = v - v_st
    eta_prime = eta - eta_st
    eta_prime = eta_prime - np.mean(eta_prime)

    # centre velocity errors on eta grid before energy integral
    u_prime_c = 0.5 * (u_prime[:, :-1] + u_prime[:, 1:])
    v_prime_c = 0.5 * (v_prime[:-1, :] + v_prime[1:, :])

    E_u = 0.5 * rho * H * np.sum(u_prime_c**2) * dx * dy
    E_v = 0.5 * rho * H * np.sum(v_prime_c**2) * dx * dy
    E_eta = 0.5 * rho * g * np.sum(eta_prime**2) * dx * dy

    return {
        "E_error": E_u + E_v + E_eta,
        "u_prime": u_prime,
        "v_prime": v_prime,
        "eta_prime": eta_prime,
        "u_prime_c": u_prime_c,
        "v_prime_c": v_prime_c,
        "u_st": u_st,
        "v_st": v_st,
        "eta_st": eta_st,
        "eta0": eta0,
    }

def task_d3_difference_fields(result):
    eta = result["eta"]
    u = result["u"]
    v = result["v"]

    j_mid = np.argmin(np.abs(result["y_eta"] - L / 2))
    eta0 = eta[j_mid, 0]

    u_st, _, _ = stommel_exact_solution(result["X_u"], result["Y_u"], eta0=eta0)
    _, v_st, _ = stommel_exact_solution(result["X_v"], result["Y_v"], eta0=eta0)
    _, _, eta_st = stommel_exact_solution(result["X_eta"], result["Y_eta"], eta0=eta0)

    u_prime = u - u_st
    v_prime = v - v_st
    eta_prime = eta - eta_st
    eta_prime = eta_prime - np.mean(eta_prime)

    return {
        "u_prime": u_prime,
        "v_prime": v_prime,
        "eta_prime": eta_prime,
        "u_st": u_st,
        "v_st": v_st,
        "eta_st": eta_st,
        "eta0": eta0,
    }

def task_d3_difference_energy(result, d3):
    dx = result["dx"]
    dy = result["dy"]

    u_prime = d3["u_prime"]
    v_prime = d3["v_prime"]
    eta_prime = d3["eta_prime"]

    u_prime_c = 0.5 * (u_prime[:, :-1] + u_prime[:, 1:])
    v_prime_c = 0.5 * (v_prime[:-1, :] + v_prime[1:, :])

    E_u = 0.5 * rho * H * np.sum(u_prime_c**2) * dx * dy
    E_v = 0.5 * rho * H * np.sum(v_prime_c**2) * dx * dy
    E_eta = 0.5 * rho * g * np.sum(eta_prime**2) * dx * dy

    return E_u + E_v + E_eta